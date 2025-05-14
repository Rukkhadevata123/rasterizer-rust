use crate::camera::Camera;
use crate::color_utils::get_face_color;
use crate::lighting::{Light, SimpleMaterial, calculate_blinn_phong};
use crate::model_types::{Material, ModelData};
use crate::rasterizer::{RasterizerConfig, TriangleData, rasterize_triangle};
use crate::scene_object::SceneObject;
use crate::transform::{
    compute_normal_matrix, ndc_to_pixel, transform_normals, world_to_ndc, world_to_view,
}; // Added world_to_view and world_to_ndc
use atomic_float::AtomicF32;
use nalgebra::{Point2, Point3, Vector3};
use rayon::prelude::*;
use std::sync::Mutex;
use std::sync::atomic::Ordering;
use std::time::Instant;

pub struct FrameBuffer {
    pub width: usize,
    pub height: usize,
    /// Stores positive depth values, smaller is closer. Atomic for parallel writes.
    pub depth_buffer: Vec<AtomicF32>,
    /// Stores RGB color values [0, 255] as u8. Mutex for parallel writes.
    pub color_buffer: Mutex<Vec<u8>>,
}

impl FrameBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        let num_pixels = width * height;
        let depth_buffer = (0..num_pixels)
            .map(|_| AtomicF32::new(f32::INFINITY))
            .collect();
        let color_buffer = Mutex::new(vec![0u8; num_pixels * 3]); // Initialize black
        FrameBuffer {
            width,
            height,
            depth_buffer,
            color_buffer,
        }
    }

    pub fn clear(&self) {
        // Reset depth buffer using parallel iteration for potential speedup
        self.depth_buffer.par_iter().for_each(|atomic_depth| {
            atomic_depth.store(f32::INFINITY, Ordering::Relaxed);
        });

        // Reset color buffer
        let mut color_guard = self.color_buffer.lock().unwrap();
        // Consider parallelizing this if locking becomes a bottleneck for clearing large buffers
        color_guard.fill(0);
    }

    pub fn get_color_buffer_bytes(&self) -> Vec<u8> {
        self.color_buffer.lock().unwrap().clone()
    }

    pub fn get_depth_buffer_f32(&self) -> Vec<f32> {
        self.depth_buffer
            .iter()
            .map(|atomic_depth| atomic_depth.load(Ordering::Relaxed))
            .collect()
    }
}

pub struct RenderSettings {
    pub projection_type: String,
    pub use_zbuffer: bool,
    pub use_face_colors: bool,
    pub use_texture: bool,
    pub light: Light,      // Store the configured light source
    pub use_phong: bool,   // 是否使用 Phong 着色（逐像素光照）
    pub apply_gamma: bool, // 是否应用gamma矫正
    pub use_pbr: bool,     // 是否使用基于物理的渲染
                           // Add more settings like backface culling, wireframe etc.
}

impl RenderSettings {
    // 将 RenderSettings 转换为 RasterizerConfig
    pub fn to_rasterizer_config(&self) -> RasterizerConfig {
        RasterizerConfig {
            use_zbuffer: self.use_zbuffer,
            use_lighting: true, // 总是启用光照
            use_perspective: self.projection_type == "perspective",
            use_phong: self.use_phong,
            use_pbr: self.use_pbr, // 添加PBR设置
            use_texture: self.use_texture,
            apply_gamma_correction: self.apply_gamma,
        }
    }

    // 获取光照模型的名称（用于调试和日志记录）
    pub fn get_lighting_description(&self) -> String {
        if self.use_pbr {
            "基于物理的渲染(PBR)".to_string()
        } else if self.use_phong {
            "Phong着色模型".to_string()
        } else {
            "平面着色模型".to_string()
        }
    }
}

pub struct Renderer {
    pub frame_buffer: FrameBuffer,
}

impl Renderer {
    pub fn new(width: usize, height: usize) -> Self {
        Renderer {
            frame_buffer: FrameBuffer::new(width, height),
        }
    }

    /// 渲染一个场景对象
    /// 这个方法接受模型数据、场景对象和相机，应用对象的变换矩阵
    pub fn render(
        &self,
        model_data: &ModelData,
        scene_object: &SceneObject,
        camera: &Camera,
        settings: &RenderSettings,
    ) {
        let start_time = Instant::now();

        println!("Rendering scene object...");

        // 获取对象的模型矩阵（将顶点从模型空间变换到世界空间）
        let model_matrix = scene_object.transform;

        // 检查是否使用自定义材质
        let material_override = if let Some(material_id) = scene_object.material_id {
            if material_id < model_data.materials.len() {
                Some(&model_data.materials[material_id])
            } else {
                println!("警告: 场景对象指定的材质ID {} 无效", material_id);
                None
            }
        } else {
            None
        };

        println!("Transforming vertices...");
        let transform_start_time = Instant::now();

        // 收集所有顶点和法线以进行批量变换
        let mut all_vertices_model: Vec<Point3<f32>> = Vec::new();
        let mut all_normals_model: Vec<Vector3<f32>> = Vec::new();
        let mut mesh_vertex_offsets: Vec<usize> = vec![0];

        for mesh in &model_data.meshes {
            all_vertices_model.extend(mesh.vertices.iter().map(|v| v.position));
            all_normals_model.extend(mesh.vertices.iter().map(|v| v.normal));
            mesh_vertex_offsets.push(all_vertices_model.len());
        }

        // 获取相机变换矩阵
        let view_matrix = camera.get_view_matrix();
        let projection_matrix = camera.get_projection_matrix(&settings.projection_type);

        // 计算组合矩阵
        let view_projection_matrix = projection_matrix * view_matrix; // Precompute for world_to_ndc

        // 计算法线变换矩阵（使用模型-视图矩阵的逆转置）
        // For normal transformation, model_view_matrix is appropriate.
        let model_view_for_normals = view_matrix * model_matrix;
        let normal_matrix = compute_normal_matrix(&model_view_for_normals);

        // 首先将模型顶点变换到世界空间
        let all_vertices_world: Vec<Point3<f32>> = all_vertices_model
            .iter()
            .map(|model_v| {
                let world_h = model_matrix * model_v.to_homogeneous(); // Multiply by model_matrix to get homogeneous world coordinates
                Point3::from_homogeneous(world_h).unwrap_or_else(Point3::origin) // Convert back to Point3 by perspective division
            })
            .collect();

        // 计算从世界空间到视图空间的顶点坐标
        let all_view_coords = world_to_view(&all_vertices_world, view_matrix);

        // 计算从模型空间到视图空间的法线向量
        // Normals are transformed from model space to view space using the normal_matrix derived from model_view_matrix
        let all_view_normals = transform_normals(&all_normals_model, &normal_matrix);

        // 计算从世界空间到NDC空间的顶点坐标
        let all_ndc_coords = world_to_ndc(&all_vertices_world, &view_projection_matrix);

        // NDC坐标 -> 像素坐标
        let all_pixel_coords = ndc_to_pixel(
            &all_ndc_coords,
            self.frame_buffer.width as f32,
            self.frame_buffer.height as f32,
        );

        let transform_duration = transform_start_time.elapsed();

        // 打印视图空间Z范围
        if !all_view_coords.is_empty() {
            println!(
                "View space Z range: [{:.3}, {:.3}]",
                all_view_coords
                    .iter()
                    .map(|p| p.z)
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(0.0),
                all_view_coords
                    .iter()
                    .map(|p| p.z)
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(0.0)
            );
        }

        println!("Rasterizing meshes...");
        let raster_start_time = Instant::now();

        // 创建 RasterizerConfig
        let rasterizer_config = settings.to_rasterizer_config();

        // --- 使用Rayon进行并行光栅化 ---
        let all_pixel_coords_ref = &all_pixel_coords;
        let all_view_coords_ref = &all_view_coords;
        let all_view_normals_ref = &all_view_normals;
        let mesh_vertex_offsets_ref = &mesh_vertex_offsets;
        let model_materials_ref = &model_data.materials;

        let triangles_to_render: Vec<_> = model_data
            .meshes
            .par_iter() // 并行处理网格
            .enumerate()
            .flat_map(|(mesh_idx, mesh)| {
                let vertex_offset = mesh_vertex_offsets_ref[mesh_idx];

                // 检查是否使用自定义材质
                let material_opt: Option<&Material> =
                    if let Some(material_reference) = material_override {
                        // 使用场景对象覆盖的材质
                        Some(material_reference)
                    } else {
                        // 使用网格默认材质
                        mesh.material_id.and_then(|id| model_materials_ref.get(id))
                    };

                // 使用From特性进行SimpleMaterial转换
                let simple_material =
                    material_opt.map_or_else(SimpleMaterial::default, SimpleMaterial::from);
                let default_color = simple_material.diffuse; // 使用材质的漫反射颜色作为默认颜色

                // 不需要克隆实现了Copy特性的类型
                let simple_material_clone = simple_material;
                let light_clone = settings.light;

                // 使用RasterizerConfig中的设置
                let use_texture = rasterizer_config.use_texture;
                let texture = if use_texture {
                    material_opt.and_then(|m| m.diffuse_texture.as_ref())
                } else {
                    None
                };

                // 获取PBR材质（如果有）
                let pbr_material = if rasterizer_config.use_pbr {
                    material_opt.and_then(|m| m.pbr_material.as_ref())
                } else {
                    None
                };

                // 处理三角形索引（每3个一组）
                mesh.indices
                    .chunks_exact(3)
                    .enumerate()
                    .filter_map(move |(face_idx_in_mesh, indices)| {
                        let i0 = indices[0] as usize;
                        let i1 = indices[1] as usize;
                        let i2 = indices[2] as usize;

                        let v0 = &mesh.vertices[i0];
                        let v1 = &mesh.vertices[i1];
                        let v2 = &mesh.vertices[i2];

                        let global_i0 = vertex_offset + i0;
                        let global_i1 = vertex_offset + i1;
                        let global_i2 = vertex_offset + i2;

                        // 确保索引在有效范围内
                        if global_i0 >= all_pixel_coords_ref.len()
                            || global_i1 >= all_pixel_coords_ref.len()
                            || global_i2 >= all_pixel_coords_ref.len()
                        {
                            println!("Warning: Invalid vertex index in mesh {}!", mesh_idx);
                            return None;
                        }

                        let pix0 = all_pixel_coords_ref[global_i0];
                        let pix1 = all_pixel_coords_ref[global_i1];
                        let pix2 = all_pixel_coords_ref[global_i2];

                        let view_pos0 = all_view_coords_ref[global_i0];
                        let view_pos1 = all_view_coords_ref[global_i1];
                        let view_pos2 = all_view_coords_ref[global_i2];

                        // --- 背面剔除 ---
                        // 暂时注释掉背面剔除代码，以便在调试阶段可以看到所有面
                        // 实际应用中应该取消注释或根据需要进行条件剔除
                        // let edge1 = view_pos1 - view_pos0;
                        // let edge2 = view_pos2 - view_pos0;
                        // let face_normal_view = edge1.cross(&edge2).normalize();
                        // let view_dir_to_face = (view_pos0 - Point3::origin()).normalize();
                        // if face_normal_view.dot(&view_dir_to_face) > -1e-6 {
                        //     return None; // 背面剔除
                        // }

                        // --- 光照计算 ---
                        let avg_normal_view = (all_view_normals_ref[global_i0]
                            + all_view_normals_ref[global_i1]
                            + all_view_normals_ref[global_i2])
                            .normalize();
                        let face_center_view = Point3::from(
                            (view_pos0.coords + view_pos1.coords + view_pos2.coords) / 3.0,
                        );
                        let view_dir_from_face = (-face_center_view.coords).normalize();
                        let lit_color = calculate_blinn_phong(
                            face_center_view,
                            avg_normal_view,
                            view_dir_from_face,
                            &light_clone,
                            &simple_material_clone,
                        );

                        // --- 确定基础颜色 ---
                        let base_color = if settings.use_face_colors {
                            get_face_color(mesh_idx * 1000 + face_idx_in_mesh, true)
                        } else {
                            default_color // 使用材质漫反射颜色或默认灰色
                        };

                        // --- 准备TriangleData ---
                        Some(TriangleData {
                            v1_pix: Point2::new(pix0.x, pix0.y),
                            v2_pix: Point2::new(pix1.x, pix1.y),
                            v3_pix: Point2::new(pix2.x, pix2.y),
                            z1_view: view_pos0.z,
                            z2_view: view_pos1.z,
                            z3_view: view_pos2.z,
                            base_color, // 传递基础颜色
                            lit_color,  // 传递预计算的光照颜色（用于非Phong着色）
                            tc1: texture.map(|_| v0.texcoord),
                            tc2: texture.map(|_| v1.texcoord),
                            tc3: texture.map(|_| v2.texcoord),
                            texture,
                            is_perspective: rasterizer_config.use_perspective,
                            // Phong着色所需的额外数据
                            n1_view: Some(all_view_normals_ref[global_i0]),
                            n2_view: Some(all_view_normals_ref[global_i1]),
                            n3_view: Some(all_view_normals_ref[global_i2]),
                            v1_view: Some(view_pos0),
                            v2_view: Some(view_pos1),
                            v3_view: Some(view_pos2),
                            material: Some(simple_material_clone), // 使用克隆的材质的值，而不是引用
                            light: Some(light_clone),              // 使用克隆的光源的值，而不是引用
                            use_phong: rasterizer_config.use_phong,
                            pbr_material, // 使用简化的字段名赋值
                        })
                    })
                    .collect::<Vec<_>>() // 在展平前先收集这个网格的所有三角形
            })
            .collect();

        // 并行处理所有三角形
        triangles_to_render.par_iter().for_each(|triangle_data| {
            rasterize_triangle(
                triangle_data,
                self.frame_buffer.width,
                self.frame_buffer.height,
                &self.frame_buffer.depth_buffer,
                &self.frame_buffer.color_buffer,
                &rasterizer_config, // 传递整个RasterizerConfig对象的引用
            );
        });

        let raster_duration = raster_start_time.elapsed();
        let total_duration = start_time.elapsed();

        println!(
            "Rendering finished. Transform: {:?}, Raster: {:?}, Total: {:?}",
            transform_duration, raster_duration, total_duration
        );
        println!("Rendered {} triangles.", triangles_to_render.len());
    }

    /// 渲染一个场景，包含多个模型和对象
    pub fn render_scene(&self, scene: &crate::scene::Scene, settings: &RenderSettings) {
        // 清除帧缓冲区
        self.frame_buffer.clear();

        // 逐个渲染场景中的每个对象
        for object in &scene.objects {
            // 获取该对象引用的模型数据
            if object.model_id < scene.models.len() {
                let model = &scene.models[object.model_id];
                self.render(model, object, &scene.active_camera, settings);
            } else {
                // 在实际渲染循环中打印警告，而不是在 render_scene 中
                // println!("警告：对象引用了无效的模型 ID {}，在 render_scene 中跳过", object.model_id);
            }
        }
    }
}
