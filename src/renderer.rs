use crate::camera::Camera;
use crate::color_utils::get_face_color;
use crate::lighting::{Light, SimpleMaterial, calculate_blinn_phong}; // Import lighting components
// Updated use statement for model types
use crate::model_types::{Material, ModelData};
use crate::rasterizer::{RasterizerConfig, TriangleData, rasterize_triangle}; // 导入 RasterizerConfig
// 更新 transform 的导入，引入所有坐标变换函数
use crate::transform::{
    compute_normal_matrix, ndc_to_pixel, transform_normals, 
    world_to_ndc, world_to_view
};
use atomic_float::AtomicF32;
use nalgebra::{Point2, Point3, Vector3}; // 移除 Matrix3 导入
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
    pub light: Light,    // Store the configured light source
    pub use_phong: bool, // 是否使用 Phong 着色（逐像素光照）
    pub apply_gamma: bool, // 是否应用gamma矫正
                         // Add more settings like backface culling, wireframe etc.
}

impl RenderSettings {
    // 新增：将 RenderSettings 转换为 RasterizerConfig
    pub fn to_rasterizer_config(&self) -> RasterizerConfig {
        RasterizerConfig {
            use_zbuffer: self.use_zbuffer,
            use_lighting: true, // 总是启用光照
            use_perspective: self.projection_type == "perspective",
            use_phong: self.use_phong,
            use_texture: self.use_texture,
            apply_gamma_correction: self.apply_gamma,
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

    // Change model_data to be an immutable reference
    pub fn render(&self, model_data: &ModelData, camera: &Camera, settings: &RenderSettings) {
        let start_time = Instant::now();

        println!("Clearing buffers...");
        self.frame_buffer.clear();
        let clear_duration = start_time.elapsed();

        println!("Transforming vertices...");
        let transform_start_time = Instant::now();

        // Collect all vertices from all meshes for batch transformation
        let mut all_vertices_world: Vec<Point3<f32>> = Vec::new();
        let mut all_normals_world: Vec<Vector3<f32>> = Vec::new();
        let mut mesh_vertex_offsets: Vec<usize> = vec![0];
        for mesh in &model_data.meshes {
            all_vertices_world.extend(mesh.vertices.iter().map(|v| v.position));
            all_normals_world.extend(mesh.vertices.iter().map(|v| v.normal));
            mesh_vertex_offsets.push(all_vertices_world.len());
        }

        // 获取变换矩阵
        let view_matrix = camera.get_view_matrix();
        let projection_matrix = camera.get_projection_matrix(&settings.projection_type);
        let view_projection_matrix = projection_matrix * view_matrix;
        
        // 计算法线变换矩阵
        let normal_matrix = compute_normal_matrix(&view_matrix);

        // 使用 transform.rs 中的函数进行坐标变换
        // 世界坐标 -> 视图坐标
        let all_view_coords = world_to_view(&all_vertices_world, &view_matrix);
        
        // 变换法线向量（世界坐标 -> 视图坐标）
        let all_view_normals = transform_normals(&all_normals_world, &normal_matrix);
        
        // 世界坐标 -> NDC坐标
        let all_ndc_coords = world_to_ndc(&all_vertices_world, &view_projection_matrix);

        // NDC坐标 -> 像素坐标
        let all_pixel_coords = ndc_to_pixel(
            &all_ndc_coords,
            self.frame_buffer.width as f32,
            self.frame_buffer.height as f32,
        );

        let transform_duration = transform_start_time.elapsed();
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

        println!("Rasterizing meshes...");
        let raster_start_time = Instant::now();

        // 创建 RasterizerConfig
        let rasterizer_config = settings.to_rasterizer_config();

        // --- Parallel Rasterization using Rayon ---
        let all_pixel_coords_ref = &all_pixel_coords;
        let all_view_coords_ref = &all_view_coords;
        let all_view_normals_ref = &all_view_normals; // Reference to view-space normals
        let mesh_vertex_offsets_ref = &mesh_vertex_offsets;
        let model_materials_ref = &model_data.materials;

        let triangles_to_render: Vec<_> = model_data
            .meshes
            .par_iter() // Parallelize mesh iteration
            .enumerate()
            .flat_map(|(mesh_idx, mesh)| {
                let vertex_offset = mesh_vertex_offsets_ref[mesh_idx];
                let material_opt: Option<&Material> =
                    mesh.material_id.and_then(|id| model_materials_ref.get(id));

                // Use From trait for SimpleMaterial conversion
                // Use Material::default() if material_opt is None
                let simple_material =
                    material_opt.map_or_else(SimpleMaterial::default, SimpleMaterial::from);
                let default_color = simple_material.diffuse; // Use material diffuse as default

                // 克隆一份 SimpleMaterial 和 Light，避免使用引用
                let simple_material_clone = simple_material.clone();
                let light_clone = settings.light.clone();

                // 使用 RasterizerConfig 中的设置
                let use_texture = rasterizer_config.use_texture;
                let texture = if use_texture {
                    material_opt.and_then(|m| m.diffuse_texture.as_ref())
                } else {
                    None
                };

                // Process indices in parallel chunks for potential further optimization if needed
                // For now, keep chunk processing within the flat_map
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

                        let pix0 = all_pixel_coords_ref[global_i0];
                        let pix1 = all_pixel_coords_ref[global_i1];
                        let pix2 = all_pixel_coords_ref[global_i2];

                        let view_pos0 = all_view_coords_ref[global_i0];
                        let view_pos1 = all_view_coords_ref[global_i1];
                        let view_pos2 = all_view_coords_ref[global_i2];

                        // --- Backface Culling --- (remains the same)
                        let edge1 = view_pos1 - view_pos0;
                        let edge2 = view_pos2 - view_pos0;
                        let face_normal_view = edge1.cross(&edge2).normalize();
                        let view_dir_to_face = (view_pos0 - Point3::origin()).normalize();
                        if face_normal_view.dot(&view_dir_to_face) > -1e-6 {
                            return None; // Backface culling
                        }

                        // --- Lighting Calculation --- (remains the same, using average normal)
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

                        // --- Base Color Determination ---
                        let base_color = if settings.use_face_colors {
                            get_face_color(mesh_idx * 1000 + face_idx_in_mesh, true)
                        } else {
                            default_color // Use material diffuse or default grey
                        };

                        // --- Prepare TriangleData ---
                        // 构建三角形数据，包括 Phong 着色所需的额外数据
                        Some(TriangleData {
                            v1_pix: Point2::new(pix0.x, pix0.y),
                            v2_pix: Point2::new(pix1.x, pix1.y),
                            v3_pix: Point2::new(pix2.x, pix2.y),
                            z1_view: view_pos0.z,
                            z2_view: view_pos1.z,
                            z3_view: view_pos2.z,
                            base_color, // 传递基础颜色
                            lit_color,  // 传递预计算的光照颜色（用于非 Phong 着色）
                            tc1: texture.map(|_| v0.texcoord),
                            tc2: texture.map(|_| v1.texcoord),
                            tc3: texture.map(|_| v2.texcoord),
                            texture,
                            is_perspective: rasterizer_config.use_perspective,
                            // 删除这行: use_zbuffer: rasterizer_config.use_zbuffer,
                            // Phong 着色所需的额外数据
                            n1_view: Some(all_view_normals_ref[global_i0]),
                            n2_view: Some(all_view_normals_ref[global_i1]),
                            n3_view: Some(all_view_normals_ref[global_i2]),
                            v1_view: Some(view_pos0),
                            v2_view: Some(view_pos1),
                            v3_view: Some(view_pos2),
                            material: Some(simple_material_clone), // 使用克隆的材质的值，而不是引用
                            light: Some(light_clone),              // 使用克隆的光源的值，而不是引用
                            use_phong: rasterizer_config.use_phong,
                        })
                    })
                    .collect::<Vec<_>>() // Collect triangles for this mesh before flattening
            })
            .collect();

        // Parallel loop over triangles
        triangles_to_render.par_iter().for_each(|triangle_data| {
            rasterize_triangle(
                triangle_data,
                self.frame_buffer.width,
                self.frame_buffer.height,
                &self.frame_buffer.depth_buffer,
                &self.frame_buffer.color_buffer,
                &rasterizer_config, // 修改：传递整个 RasterizerConfig 对象的引用，而不是单独的布尔值
            );
        });

        let raster_duration = raster_start_time.elapsed();
        let total_duration = start_time.elapsed();

        println!(
            "Rendering finished. Clear: {:?}, Transform: {:?}, Raster: {:?}, Total: {:?}",
            clear_duration, transform_duration, raster_duration, total_duration
        );
        println!("Rendered {} triangles.", triangles_to_render.len());
    }
}
