use crate::core::rasterizer::{TextureSource, TriangleData, VertexRenderData, rasterize_triangle};
use crate::scene::scene_object::SceneObject;
use crate::geometry::camera::Camera;
use crate::geometry::culling::{is_backface, should_cull_small_triangle}; // 导入剔除函数
use crate::geometry::transform::{compute_normal_matrix, transform_normals};
use crate::materials::material_system::MaterialView;
use crate::materials::model_types::{Material, ModelData};
use atomic_float::AtomicF32;
use nalgebra::{Point2, Point3, Vector3};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Instant;

/// 帧缓冲区实现，存储渲染结果
pub struct FrameBuffer {
    pub width: usize,
    pub height: usize,
    /// 存储正深度值，数值越小表示越近。使用原子类型以支持并行写入。
    pub depth_buffer: Vec<AtomicF32>,
    /// 存储RGB颜色值 [0, 255]，类型为u8。使用原子类型以支持并行写入。
    pub color_buffer: Vec<AtomicU8>,
}

impl FrameBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        let num_pixels = width * height;

        // 为深度缓冲区创建原子浮点数向量
        let depth_buffer = (0..num_pixels)
            .map(|_| AtomicF32::new(f32::INFINITY))
            .collect();

        // 使用迭代器创建颜色缓冲区，避免使用vec!宏
        let color_buffer = (0..num_pixels * 3).map(|_| AtomicU8::new(0)).collect();

        FrameBuffer {
            width,
            height,
            depth_buffer,
            color_buffer,
        }
    }

    /// 清除所有缓冲区
    pub fn clear(&self) {
        // 重置深度缓冲区，为了避免数据竞争，使用原子操作
        self.depth_buffer.par_iter().for_each(|atomic_depth| {
            atomic_depth.store(f32::INFINITY, Ordering::Relaxed);
        });

        // 重置颜色缓冲区
        self.color_buffer
            .par_iter()
            .for_each(|atomic_color| atomic_color.store(0, Ordering::Relaxed));
    }

    /// 获取颜色缓冲区的字节数据，用于保存图像
    pub fn get_color_buffer_bytes(&self) -> Vec<u8> {
        self.color_buffer
            .iter()
            .map(|atomic_color| atomic_color.load(Ordering::Relaxed))
            .collect()
    }

    /// 获取深度缓冲区的浮点数据，用于保存深度图
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
    pub light: crate::materials::material_system::Light,

    // 环境光信息（作为场景的基础属性）
    /// 环境光强度 - 控制场景整体亮度 [0.0, 1.0]
    pub ambient_intensity: f32,
    /// 环境光颜色 - 控制场景基础色调 (RGB)
    pub ambient_color: nalgebra::Vector3<f32>,

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
            light: crate::materials::material_system::Light::directional(
                nalgebra::Vector3::new(0.0, -1.0, -1.0).normalize(),
                nalgebra::Vector3::new(1.0, 1.0, 1.0),
            ),
            ambient_intensity: 0.1, // 默认环境光强度
            ambient_color: nalgebra::Vector3::new(1.0, 1.0, 1.0), // 默认环境光颜色（白色）
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

    pub fn with_light(mut self, light: crate::materials::material_system::Light) -> Self {
        self.light = light;
        self
    }

    pub fn with_ambient_intensity(mut self, intensity: f32) -> Self {
        self.ambient_intensity = intensity;
        self
    }

    pub fn with_ambient_color(mut self, color: nalgebra::Vector3<f32>) -> Self {
        self.ambient_color = color;
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

    /// 判断是否使用透视投影
    pub fn is_perspective(&self) -> bool {
        self.projection_type == "perspective"
    }
}

/// 渲染器结构体 - 负责高层次渲染流程
pub struct Renderer {
    pub frame_buffer: FrameBuffer,
}

impl Renderer {
    /// 创建一个新的渲染器实例
    pub fn new(width: usize, height: usize) -> Self {
        Renderer {
            frame_buffer: FrameBuffer::new(width, height),
        }
    }

    /// 渲染一个场景，包含多个模型和对象
    pub fn render_scene(&self, scene: &crate::scene::scene_utils::Scene, config: &mut RenderConfig) {
        // 从场景中获取环境光设置
        config.ambient_intensity = scene.ambient_intensity;
        config.ambient_color = scene.ambient_color;

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

    /// 渲染一个场景对象
    /// 这个方法处理几何变换、可见性剔除和材质准备
    pub fn render(
        &self,
        model_data: &ModelData,
        scene_object: &SceneObject,
        camera: &Camera,
        config: &RenderConfig,
    ) {
        let start_time = Instant::now();

        println!("渲染场景对象...");

        // --- 材质准备 ---
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

        // --- 几何变换 ---
        println!("变换顶点...");
        let transform_start_time = Instant::now();

        // 使用优化的几何变换函数
        let (all_pixel_coords, all_view_coords, all_view_normals, mesh_vertex_offsets) =
            self.transform_geometry(model_data, scene_object, camera, config);

        let transform_duration = transform_start_time.elapsed();

        // 打印视图空间Z范围（调试用）
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

        // --- 三角形准备 ---
        println!("准备三角形数据...");
        let triangles_to_render = self.prepare_triangles(
            model_data,
            &all_pixel_coords,
            &all_view_coords,
            &all_view_normals,
            &mesh_vertex_offsets,
            material_override,
            config,
        );

        // --- 光栅化 ---
        println!("光栅化网格...");
        let raster_start_time = Instant::now();

        // 光栅化三角形 - 使用配置的多线程设置
        if config.use_multithreading {
            // 并行处理所有三角形
            triangles_to_render.par_iter().for_each(|triangle_data| {
                rasterize_triangle(
                    triangle_data,
                    self.frame_buffer.width,
                    self.frame_buffer.height,
                    &self.frame_buffer.depth_buffer,
                    &self.frame_buffer.color_buffer,
                    config,
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
                    config,
                );
            });
        }

        // --- 性能统计 ---
        let raster_duration = raster_start_time.elapsed();
        let total_duration = start_time.elapsed();

        println!(
            "渲染完成. 变换: {:?}, 光栅化: {:?}, 总时间: {:?}",
            transform_duration, raster_duration, total_duration
        );
        println!("渲染了 {} 个三角形。", triangles_to_render.len());
    }

    /// 准备三角形数据 - 将几何和材质数据组织为光栅化器可以使用的形式
    /// 这个阶段处理：
    /// 1. 视锥剔除（在transform_geometry中完成）
    /// 2. 背面剔除
    /// 3. 小三角形剔除
    /// 4. 纹理和材质决策
    #[allow(clippy::too_many_arguments)]
    fn prepare_triangles<'a>(
        &self,
        model_data: &'a ModelData,
        all_pixel_coords: &[Point2<f32>],
        all_view_coords: &[Point3<f32>],
        all_view_normals: &[Vector3<f32>],
        mesh_vertex_offsets: &[usize],
        material_override: Option<&'a Material>,
        config: &'a RenderConfig,
    ) -> Vec<TriangleData<'a>> {
        // 创建要渲染的三角形列表
        model_data
            .meshes
            .par_iter() // 并行处理网格
            .enumerate()
            .flat_map(|(mesh_idx, mesh)| {
                let vertex_offset = mesh_vertex_offsets[mesh_idx];
                let model_materials = &model_data.materials;

                // 检查是否使用自定义材质
                let material_opt: Option<&Material> =
                    if let Some(material_reference) = material_override {
                        // 使用场景对象覆盖的材质
                        Some(material_reference)
                    } else {
                        // 使用网格默认材质
                        mesh.material_id.and_then(|id| model_materials.get(id))
                    };

                // 处理三角形索引（每3个一组）
                mesh.indices
                    .chunks_exact(3)
                    .enumerate() // 添加枚举以获取面索引
                    .filter_map(move |(face_idx, indices)| {
                        // --- 提取顶点索引 ---
                        let i0 = indices[0] as usize;
                        let i1 = indices[1] as usize;
                        let i2 = indices[2] as usize;

                        // --- 提取顶点数据 ---
                        let v0 = &mesh.vertices[i0];
                        let v1 = &mesh.vertices[i1];
                        let v2 = &mesh.vertices[i2];

                        // --- 计算全局索引 ---
                        let global_i0 = vertex_offset + i0;
                        let global_i1 = vertex_offset + i1;
                        let global_i2 = vertex_offset + i2;

                        // --- 生成面的唯一ID ---
                        // 使用mesh_idx * 1000 + face_idx作为基础，确保不同网格的面索引不会冲突
                        let global_face_index = (mesh_idx * 1000 + face_idx) as u64;

                        // --- 检查索引有效性 ---
                        if global_i0 >= all_pixel_coords.len()
                            || global_i1 >= all_pixel_coords.len()
                            || global_i2 >= all_pixel_coords.len()
                        {
                            println!("警告: 网格 {} 中的顶点索引无效!", mesh_idx);
                            return None;
                        }

                        // --- 获取坐标和法线 ---
                        let pix0 = all_pixel_coords[global_i0];
                        let pix1 = all_pixel_coords[global_i1];
                        let pix2 = all_pixel_coords[global_i2];

                        let view_pos0 = all_view_coords[global_i0];
                        let view_pos1 = all_view_coords[global_i1];
                        let view_pos2 = all_view_coords[global_i2];

                        // --- 背面剔除 ---
                        if config.use_backface_culling
                            && is_backface(&view_pos0, &view_pos1, &view_pos2)
                        {
                            return None; // 剔除背面
                        }

                        // --- 小三角形剔除 ---
                        if config.cull_small_triangles
                            && should_cull_small_triangle(
                                &pix0,
                                &pix1,
                                &pix2,
                                config.min_triangle_area,
                            )
                        {
                            return None; // 剔除小三角形
                        }

                        // --- 确定纹理源 ---
                        let texture_source =
                            self.determine_texture_source(config, material_opt, global_face_index);

                        // --- 确定基础颜色 ---
                        let base_color =
                            self.determine_base_color(config, &texture_source, material_opt);

                        // --- 创建材质视图 ---
                        let material_view = material_opt.map(|m| {
                            if config.use_pbr {
                                MaterialView::PBR(m)
                            } else {
                                MaterialView::BlinnPhong(m)
                            }
                        });

                        // --- 创建顶点渲染数据 ---
                        let vertex_data = [
                            self.create_vertex_render_data(
                                &pix0,
                                view_pos0,
                                v0,
                                global_i0,
                                &texture_source,
                                all_view_normals,
                            ),
                            self.create_vertex_render_data(
                                &pix1,
                                view_pos1,
                                v1,
                                global_i1,
                                &texture_source,
                                all_view_normals,
                            ),
                            self.create_vertex_render_data(
                                &pix2,
                                view_pos2,
                                v2,
                                global_i2,
                                &texture_source,
                                all_view_normals,
                            ),
                        ];

                        // --- 准备光照引用 ---
                        let light_ref = if config.use_lighting {
                            Some(&config.light)
                        } else {
                            None
                        };

                        // --- 创建TriangleData ---
                        Some(TriangleData {
                            vertices: vertex_data,
                            base_color,
                            texture_source,
                            material_view,
                            light: light_ref,
                            is_perspective: config.is_perspective(),
                        })
                    })
                    .collect::<Vec<_>>() // 在展平前先收集这个网格的所有三角形
            })
            .collect()
    }

    /// 确定纹理来源
    fn determine_texture_source<'a>(
        &self,
        config: &RenderConfig,
        material_opt: Option<&'a Material>,
        global_face_index: u64,
    ) -> TextureSource<'a> {
        // 首先判断是否启用纹理功能
        if !config.use_texture {
            // 未启用纹理功能时：
            // 检查是否启用面颜色模式，即使未启用纹理也可以应用面颜色
            if config.use_face_colors {
                return TextureSource::FaceColor(global_face_index);
            }
            return TextureSource::None;
        }

        // 已启用纹理功能时，遵循优先级：PNG材质 > 面随机颜色 > SolidColor
        
        // 1. 优先使用PNG材质（如果存在）
        if let Some(tex) = material_opt.and_then(|m| m.texture.as_ref()) {
            return TextureSource::Image(tex);
        }
        
        // 2. 其次检查是否应用面随机颜色
        if config.use_face_colors {
            return TextureSource::FaceColor(global_face_index);
        }
        
        // 3. 最后使用材质颜色作为固体纹理
        let color = material_opt.map_or_else(
            || Vector3::new(0.7, 0.7, 0.7), // 默认灰色
            |m| m.diffuse(),                // 使用材质的漫反射颜色
        );
        TextureSource::SolidColor(color)
    }

    /// 确定基础颜色
    fn determine_base_color(
        &self,
        _config: &RenderConfig,         // 添加下划线避免未使用警告
        texture_source: &TextureSource, // 改为借用而非所有权转移
        material_opt: Option<&Material>,
    ) -> Vector3<f32> {
        match texture_source {
            TextureSource::FaceColor(_) => {
                // 对于面随机颜色，使用白色作为基础，实际颜色会在光栅化器中生成
                Vector3::new(1.0, 1.0, 1.0)
            }
            TextureSource::None | TextureSource::Image(_) | TextureSource::SolidColor(_) => {
                // 使用材质的漫反射颜色
                material_opt.map_or_else(
                    || Vector3::new(0.7, 0.7, 0.7), // 默认灰色
                    |m| m.diffuse(),                // 使用材质的漫反射颜色
                )
            }
        }
    }

    /// 创建顶点渲染数据
    fn create_vertex_render_data(
        &self,
        pix: &Point2<f32>,
        view_pos: Point3<f32>,
        vertex: &crate::materials::model_types::Vertex,
        global_index: usize,
        texture_source: &TextureSource,
        all_view_normals: &[Vector3<f32>],
    ) -> VertexRenderData {
        VertexRenderData {
            pix: Point2::new(pix.x, pix.y),
            z_view: view_pos.z,
            texcoord: if matches!(texture_source, TextureSource::Image(_)) {
                Some(vertex.texcoord)
            } else {
                None
            },
            normal_view: Some(all_view_normals[global_index]),
            position_view: Some(view_pos),
        }
    }

    /// 获取ModelData中的总顶点数
    fn estimate_vertex_count(model_data: &ModelData) -> usize {
        model_data
            .meshes
            .iter()
            .map(|mesh| mesh.vertices.len())
            .sum()
    }

    /// 优化的顶点和法线变换函数
    fn transform_geometry(
        &self,
        model_data: &ModelData,
        scene_object: &SceneObject,
        camera: &Camera,
        config: &RenderConfig,
    ) -> GeometryTransformResult {
        // 获取对象的模型矩阵（将顶点从模型空间变换到世界空间）
        let model_matrix = scene_object.transform;

        // 收集所有顶点和法线以进行批量变换
        let vertex_count = Self::estimate_vertex_count(model_data);
        let mut all_vertices_model: Vec<Point3<f32>> = Vec::with_capacity(vertex_count);
        let mut all_normals_model: Vec<Vector3<f32>> = Vec::with_capacity(vertex_count);
        let mut mesh_vertex_offsets: Vec<usize> = vec![0];

        for mesh in &model_data.meshes {
            all_vertices_model.extend(mesh.vertices.iter().map(|v| v.position));
            all_normals_model.extend(mesh.vertices.iter().map(|v| v.normal));
            mesh_vertex_offsets.push(all_vertices_model.len());
        }

        // 获取相机变换矩阵
        let view_matrix = camera.get_view_matrix();
        let projection_matrix = camera.get_projection_matrix(&config.projection_type);

        // 计算组合矩阵 - 预计算提高效率
        let model_view_matrix = view_matrix * model_matrix;
        let mvp_matrix = projection_matrix * model_view_matrix;

        // 计算法线变换矩阵（使用模型-视图矩阵的逆转置）
        let normal_matrix = compute_normal_matrix(&model_view_matrix);

        // 变换顶点到视图空间 - 单次循环同时计算所有需要的坐标
        let mut all_pixel_coords = Vec::with_capacity(all_vertices_model.len());
        let mut all_view_coords = Vec::with_capacity(all_vertices_model.len());

        for vertex in &all_vertices_model {
            // 模型空间 -> 剪裁空间 (MVP变换)
            let clip_space = mvp_matrix * vertex.to_homogeneous();

            // 剪裁空间 -> NDC空间 (透视除法)
            let w = clip_space.w;
            let ndc = if w.abs() > 1e-6 {
                Point3::new(clip_space.x / w, clip_space.y / w, clip_space.z / w)
            } else {
                Point3::new(0.0, 0.0, 0.0)
            };

            // NDC空间 -> 屏幕空间
            let pixel = Point2::new(
                (ndc.x * 0.5 + 0.5) * self.frame_buffer.width as f32,
                (1.0 - (ndc.y * 0.5 + 0.5)) * self.frame_buffer.height as f32,
            );
            all_pixel_coords.push(pixel);

            // 计算视图空间坐标 (仅用于深度和光照计算)
            let view_space = model_view_matrix * vertex.to_homogeneous();
            all_view_coords.push(Point3::new(view_space.x, view_space.y, view_space.z));
        }

        // 变换法线到视图空间
        let all_view_normals = transform_normals(&all_normals_model, &normal_matrix);

        (
            all_pixel_coords,
            all_view_coords,
            all_view_normals,
            mesh_vertex_offsets,
        )
    }
}

/// 优化的几何变换结果类型
pub type GeometryTransformResult = (
    Vec<Point2<f32>>,  // 屏幕坐标
    Vec<Point3<f32>>,  // 视图空间坐标
    Vec<Vector3<f32>>, // 视图空间法线
    Vec<usize>,        // 网格顶点偏移量
);
