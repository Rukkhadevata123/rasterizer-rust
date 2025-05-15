use crate::camera::Camera;
use crate::color_utils::get_face_color;
use crate::material_system::MaterialView;
use crate::model_types::{Material, MaterialMode, ModelData};
use crate::rasterizer::{TriangleData, rasterize_triangle};
use crate::scene_object::SceneObject;
use crate::transform::{
    compute_normal_matrix, ndc_to_pixel, transform_normals, world_to_ndc, world_to_view,
};
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

/// 统一的渲染配置结构体，整合了所有渲染相关设置
#[derive(Debug, Clone)]
pub struct RenderConfig {
    // 投影相关设置
    /// 投影类型："perspective" 或 "orthographic"
    pub projection_type: String,

    // 缓冲区控制
    /// 是否启用深度缓冲和深度测试
    pub use_zbuffer: bool,

    // 着色和光照
    /// 是否应用光照计算
    pub use_lighting: bool,
    /// 是否使用面颜色而非材质颜色
    pub use_face_colors: bool,
    /// 是否使用Phong着色（逐像素光照计算）
    pub use_phong: bool,
    /// 是否使用基于物理的渲染 (PBR)
    pub use_pbr: bool,

    // 纹理和后处理
    /// 是否使用纹理映射
    pub use_texture: bool,
    /// 是否应用gamma校正（sRGB空间转换）
    pub apply_gamma_correction: bool,

    // 光照信息
    /// 默认光源配置
    pub light: crate::material_system::Light,

    // 几何处理
    /// 是否启用背面剔除
    pub use_backface_culling: bool,
    /// 是否以线框模式渲染
    pub use_wireframe: bool,

    // 性能优化设置
    /// 是否启用多线程渲染
    pub use_multithreading: bool,
    /// 是否对小三角形进行剔除
    pub cull_small_triangles: bool,
    /// 用于剔除的最小三角形面积
    pub min_triangle_area: f32,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            projection_type: "perspective".to_string(),
            use_zbuffer: true,
            use_lighting: true,
            use_face_colors: false,
            use_phong: false,
            use_pbr: false,
            use_texture: true,
            apply_gamma_correction: true,
            light: crate::material_system::Light::directional(
                nalgebra::Vector3::new(0.0, -1.0, -1.0).normalize(),
                nalgebra::Vector3::new(1.0, 1.0, 1.0),
            ),
            use_backface_culling: false,
            use_wireframe: false,
            use_multithreading: true,
            cull_small_triangles: false,
            min_triangle_area: 1e-3,
        }
    }
}

impl RenderConfig {
    /// 获取光照模型的描述字符串
    pub fn get_lighting_description(&self) -> String {
        if self.use_pbr {
            "基于物理的渲染(PBR)".to_string()
        } else if self.use_phong {
            "Phong着色模型".to_string()
        } else {
            "平面着色模型".to_string()
        }
    }

    // 构建器方法，便于链式配置
    pub fn with_projection(mut self, projection_type: &str) -> Self {
        self.projection_type = projection_type.to_string();
        self
    }

    pub fn with_zbuffer(mut self, use_zbuffer: bool) -> Self {
        self.use_zbuffer = use_zbuffer;
        self
    }

    pub fn with_lighting(mut self, use_lighting: bool) -> Self {
        self.use_lighting = use_lighting;
        self
    }

    pub fn with_face_colors(mut self, use_face_colors: bool) -> Self {
        self.use_face_colors = use_face_colors;
        self
    }

    pub fn with_phong(mut self, use_phong: bool) -> Self {
        self.use_phong = use_phong;
        self
    }

    pub fn with_pbr(mut self, use_pbr: bool) -> Self {
        self.use_pbr = use_pbr;
        self
    }

    pub fn with_texture(mut self, use_texture: bool) -> Self {
        self.use_texture = use_texture;
        self
    }

    pub fn with_gamma_correction(mut self, apply_gamma_correction: bool) -> Self {
        self.apply_gamma_correction = apply_gamma_correction;
        self
    }

    pub fn with_light(mut self, light: crate::material_system::Light) -> Self {
        self.light = light;
        self
    }

    pub fn with_backface_culling(mut self, use_backface_culling: bool) -> Self {
        self.use_backface_culling = use_backface_culling;
        self
    }

    pub fn with_wireframe(mut self, use_wireframe: bool) -> Self {
        self.use_wireframe = use_wireframe;
        self
    }

    pub fn with_multithreading(mut self, use_multithreading: bool) -> Self {
        self.use_multithreading = use_multithreading;
        self
    }

    pub fn with_small_triangle_culling(mut self, enable: bool, min_area: f32) -> Self {
        self.cull_small_triangles = enable;
        self.min_triangle_area = min_area;
        self
    }

    /// 转换为光栅化器配置
    pub fn to_rasterizer_config(&self) -> crate::rasterizer::RasterizerConfig {
        crate::rasterizer::RasterizerConfig {
            use_zbuffer: self.use_zbuffer,
            use_lighting: self.use_lighting,
            use_perspective: self.projection_type == "perspective",
            use_phong: self.use_phong,
            use_pbr: self.use_pbr,
            use_texture: self.use_texture,
            apply_gamma_correction: self.apply_gamma_correction,
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
    /// 这个方法接受模型数据、场景对象、相机和渲染配置
    pub fn render(
        &self,
        model_data: &ModelData,
        scene_object: &SceneObject,
        camera: &Camera,
        config: &RenderConfig,
    ) {
        let start_time = Instant::now();

        println!("渲染场景对象...");

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

        println!("变换顶点...");
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
        let projection_matrix = camera.get_projection_matrix(&config.projection_type);

        // 计算组合矩阵
        let view_projection_matrix = projection_matrix * view_matrix; // 预计算用于world_to_ndc

        // 计算法线变换矩阵（使用模型-视图矩阵的逆转置）
        let model_view_for_normals = view_matrix * model_matrix;
        let normal_matrix = compute_normal_matrix(&model_view_for_normals);

        // 首先将模型顶点变换到世界空间
        let all_vertices_world: Vec<Point3<f32>> = all_vertices_model
            .iter()
            .map(|model_v| {
                let world_h = model_matrix * model_v.to_homogeneous(); // 乘以模型矩阵得到齐次世界坐标
                Point3::from_homogeneous(world_h).unwrap_or_else(Point3::origin) // 透视除法转回Point3
            })
            .collect();

        // 计算从世界空间到视图空间的顶点坐标
        let all_view_coords = world_to_view(&all_vertices_world, view_matrix);

        // 计算从模型空间到视图空间的法线向量
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
                "视图空间Z范围: [{:.3}, {:.3}]",
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

        println!("光栅化网格...");
        let raster_start_time = Instant::now();

        // 创建 RasterizerConfig
        let rasterizer_config = config.to_rasterizer_config();

        // --- 使用Rayon进行并行光栅化 ---
        let all_pixel_coords_ref = &all_pixel_coords;
        let all_view_coords_ref = &all_view_coords;
        let all_view_normals_ref = &all_view_normals;
        let mesh_vertex_offsets_ref = &mesh_vertex_offsets;
        let model_materials_ref = &model_data.materials;

        // 创建要渲染的三角形列表
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

                let _default_color = material_opt
                    .and_then(|m| m.diffuse_texture.as_ref())
                    .map_or_else(
                        || nalgebra::Vector3::new(1.0, 1.0, 1.0), // 默认白色
                        |_tex| {
                            // 这里可以添加纹理的加载和采样代码
                            // 目前返回一个占位的白色向量
                            nalgebra::Vector3::new(1.0, 1.0, 1.0)
                        },
                    );

                // 不需要克隆实现了Copy特性的类型
                let light_clone = config.light;

                // 使用RasterizerConfig中的设置
                let use_texture = rasterizer_config.use_texture;
                let texture = if use_texture {
                    material_opt.and_then(|m| m.diffuse_texture.as_ref())
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
                            println!("警告: 网格 {} 中的顶点索引无效!", mesh_idx);
                            return None;
                        }

                        let pix0 = all_pixel_coords_ref[global_i0];
                        let pix1 = all_pixel_coords_ref[global_i1];
                        let pix2 = all_pixel_coords_ref[global_i2];

                        let view_pos0 = all_view_coords_ref[global_i0];
                        let view_pos1 = all_view_coords_ref[global_i1];
                        let view_pos2 = all_view_coords_ref[global_i2];

                        // --- 背面剔除 ---
                        if config.use_backface_culling {
                            let edge1 = view_pos1 - view_pos0;
                            let edge2 = view_pos2 - view_pos0;
                            let face_normal_view = edge1.cross(&edge2).normalize();
                            let view_dir_to_face = (view_pos0 - Point3::origin()).normalize();
                            if face_normal_view.dot(&view_dir_to_face) > -1e-6 {
                                return None; // 背面剔除
                            }
                        }

                        // --- 小三角形剔除 ---
                        if config.cull_small_triangles {
                            let area = ((pix1.x - pix0.x) * (pix2.y - pix0.y)
                                - (pix2.x - pix0.x) * (pix1.y - pix0.y))
                                .abs()
                                * 0.5;
                            if area < config.min_triangle_area {
                                return None; // 剔除面积小的三角形
                            }
                        }

                        // --- 光照计算 ---
                        let _avg_normal_view = (all_view_normals_ref[global_i0]
                            + all_view_normals_ref[global_i1]
                            + all_view_normals_ref[global_i2])
                            .normalize();
                        let face_center_view = Point3::from(
                            (view_pos0.coords + view_pos1.coords + view_pos2.coords) / 3.0,
                        );
                        let _view_dir_from_face = (-face_center_view.coords).normalize();

                        // --- 确定基础颜色 ---
                        let base_color = if config.use_face_colors {
                            get_face_color(mesh_idx * 1000 + face_idx_in_mesh, true)
                        } else {
                            material_opt.map_or_else(
                                || Vector3::new(0.7, 0.7, 0.7), // 默认灰色
                                |m| m.diffuse,                  // 使用材质的漫反射颜色
                            )
                        };

                        // 创建材质视图
                        let material_view = if rasterizer_config.use_pbr {
                            material_opt.map(|m| MaterialView::from_material(m, MaterialMode::PBR))
                        } else {
                            material_opt
                                .map(|m| MaterialView::from_material(m, MaterialMode::BlinnPhong))
                        };

                        // --- 准备TriangleData ---
                        Some(TriangleData {
                            v1_pix: Point2::new(pix0.x, pix0.y),
                            v2_pix: Point2::new(pix1.x, pix1.y),
                            v3_pix: Point2::new(pix2.x, pix2.y),
                            z1_view: view_pos0.z,
                            z2_view: view_pos1.z,
                            z3_view: view_pos2.z,
                            base_color,            // 传递基础颜色
                            lit_color: base_color, // 传递预计算的光照颜色（用于非Phong着色）
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
                            material: material_opt,   // 传递材质引用
                            light: Some(light_clone), // 使用克隆的光源的值，而不是引用
                            use_phong: rasterizer_config.use_phong,
                            material_view, // 使用新的材质视图
                        })
                    })
                    .collect::<Vec<_>>() // 在展平前先收集这个网格的所有三角形
            })
            .collect();

        // 光栅化三角形
        if config.use_multithreading {
            // 并行处理所有三角形
            triangles_to_render.par_iter().for_each(|triangle_data| {
                rasterize_triangle(
                    triangle_data,
                    self.frame_buffer.width,
                    self.frame_buffer.height,
                    &self.frame_buffer.depth_buffer,
                    &self.frame_buffer.color_buffer,
                    &rasterizer_config,
                );
            });
        } else {
            // 串行处理所有三角形
            triangles_to_render.iter().for_each(|triangle_data| {
                rasterize_triangle(
                    triangle_data,
                    self.frame_buffer.width,
                    self.frame_buffer.height,
                    &self.frame_buffer.depth_buffer,
                    &self.frame_buffer.color_buffer,
                    &rasterizer_config,
                );
            });
        }

        let raster_duration = raster_start_time.elapsed();
        let total_duration = start_time.elapsed();

        println!(
            "渲染完成. 变换: {:?}, 光栅化: {:?}, 总时间: {:?}",
            transform_duration, raster_duration, total_duration
        );
        println!("渲染了 {} 个三角形。", triangles_to_render.len());
    }

    /// 渲染一个场景，包含多个模型和对象
    pub fn render_scene(&self, scene: &crate::scene::Scene, config: &RenderConfig) {
        // 清除帧缓冲区
        self.frame_buffer.clear();

        // 逐个渲染场景中的每个对象
        for object in &scene.objects {
            // 获取该对象引用的模型数据
            if object.model_id < scene.models.len() {
                let model = &scene.models[object.model_id];
                self.render(model, object, &scene.active_camera, config);
            } else {
                println!("警告：对象引用了无效的模型 ID {}", object.model_id);
            }
        }
    }
}
