use crate::core::shadow_map::ShadowMap;
use crate::geometry::camera::{Camera, ProjectionType};
use crate::io::render_settings::RenderSettings;
use crate::material_system::{color, texture::Texture};
use atomic_float::AtomicF32;
use log::{debug, warn};
use nalgebra::{Matrix4, Point3, Vector3};
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

/// 缓存的背景状态
#[derive(Debug, Clone)]
struct BackgroundCache {
    pixels: Arc<Vec<Vector3<f32>>>,
    width: usize,
    height: usize,
}

/// 地面本体缓存（不含阴影）
#[derive(Debug, Clone)]
pub struct GroundBaseCache {
    ground_factors: Arc<Vec<f32>>,
    ground_colors: Arc<Vec<Vector3<f32>>>,
    width: usize,
    height: usize,
}

/// 地面阴影缓存（仅阴影因子）
#[derive(Debug, Clone)]
pub struct GroundShadowCache {
    shadow_factors: Arc<Vec<f32>>,
    width: usize,
    height: usize,
}

/// 帧缓冲区实现，存储渲染结果
pub struct FrameBuffer {
    pub width: usize,
    pub height: usize,
    pub depth_buffer: Vec<AtomicF32>,
    pub color_buffer: Vec<AtomicU8>,
    cached_background: Option<Texture>,
    cached_path: Option<String>,
    background_cache: Option<BackgroundCache>,
    ground_base_cache: Option<GroundBaseCache>,
    ground_shadow_cache: Option<GroundShadowCache>,
}

impl FrameBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        let num_pixels = width * height;
        let depth_buffer = (0..num_pixels)
            .map(|_| AtomicF32::new(f32::INFINITY))
            .collect();
        let color_buffer = (0..num_pixels * 3).map(|_| AtomicU8::new(0)).collect();
        FrameBuffer {
            width,
            height,
            depth_buffer,
            color_buffer,
            cached_background: None,
            cached_path: None,
            background_cache: None,
            ground_base_cache: None,
            ground_shadow_cache: None,
        }
    }

    /// 清空并准备帧缓冲区
    pub fn clear(
        &mut self,
        settings: &RenderSettings,
        camera: &Camera,
        shadow_map: Option<&ShadowMap>,
    ) {
        // 重置深度缓冲区
        self.depth_buffer.par_iter().for_each(|atomic_depth| {
            atomic_depth.store(f32::INFINITY, Ordering::Relaxed);
        });

        let width = self.width;
        let height = self.height;

        // 1. 背景缓存逻辑
        let background_pixels_ref = self.compute_background_cache(settings, width, height);

        // 2. 地面本体缓存（不含阴影）
        let (ground_factors_ref, ground_colors_ref) = if settings.enable_ground_plane {
            self.compute_ground_base_cache(settings, camera, width, height)
        } else {
            (
                Arc::new(vec![0.0; width * height]),
                Arc::new(vec![Vector3::zeros(); width * height]),
            )
        };

        // 3. 地面阴影缓存（仅阴影因子）
        let shadow_factors_ref = if settings.enable_ground_plane && settings.enable_shadow_mapping {
            self.compute_ground_shadow_cache(settings, camera, shadow_map, width, height)
        } else {
            Arc::new(vec![1.0; width * height])
        };

        // 4. 并行合成最终颜色
        self.compose_final_colors(
            settings,
            &background_pixels_ref,
            &ground_factors_ref,
            &ground_colors_ref,
            &shadow_factors_ref,
        );
    }

    fn compute_background_cache(
        &mut self,
        settings: &RenderSettings,
        width: usize,
        height: usize,
    ) -> Arc<Vec<Vector3<f32>>> {
        let cache_valid = self.background_cache.is_some()
            && self.background_cache.as_ref().unwrap().width == width
            && self.background_cache.as_ref().unwrap().height == height;

        if !cache_valid {
            debug!("计算背景缓存");

            let background_texture =
                if settings.use_background_image && settings.background_image_path.is_some() {
                    self.get_background_image(settings).cloned()
                } else {
                    None
                };

            let mut background_pixels = vec![Vector3::zeros(); width * height];

            background_pixels
                .par_iter_mut()
                .enumerate()
                .for_each(|(buffer_index, pixel)| {
                    let y = buffer_index / width;
                    let x = buffer_index % width;
                    let t_y = y as f32 / (height - 1) as f32;
                    let t_x = x as f32 / (width - 1) as f32;

                    *pixel = compute_background(settings, background_texture.as_ref(), t_x, t_y);
                });

            self.background_cache = Some(BackgroundCache {
                pixels: Arc::new(background_pixels),
                width,
                height,
            });

            debug!("背景缓存计算完成 ({width}x{height})");
        }

        self.background_cache.as_ref().unwrap().pixels.clone()
    }

    fn compute_ground_base_cache(
        &mut self,
        settings: &RenderSettings,
        camera: &Camera,
        width: usize,
        height: usize,
    ) -> (Arc<Vec<f32>>, Arc<Vec<Vector3<f32>>>) {
        let cache_valid = self.ground_base_cache.is_some()
            && self.ground_base_cache.as_ref().unwrap().width == width
            && self.ground_base_cache.as_ref().unwrap().height == height;

        if !cache_valid {
            debug!("重新计算地面本体缓存");

            let mut ground_factors = vec![0.0; width * height];
            let mut ground_colors = vec![Vector3::zeros(); width * height];

            ground_factors
                .par_iter_mut()
                .zip(ground_colors.par_iter_mut())
                .enumerate()
                .for_each(|(buffer_index, (factor, color))| {
                    let y = buffer_index / width;
                    let x = buffer_index % width;
                    let t_y = y as f32 / (height - 1) as f32;
                    let t_x = x as f32 / (width - 1) as f32;

                    let (ground_factor, ground_color) =
                        compute_ground_base(settings, camera, t_x, t_y);
                    *factor = ground_factor;
                    *color = ground_color;
                });

            self.ground_base_cache = Some(GroundBaseCache {
                ground_factors: Arc::new(ground_factors),
                ground_colors: Arc::new(ground_colors),
                width,
                height,
            });

            debug!("地面本体缓存计算完成");
        }

        let cache = self.ground_base_cache.as_ref().unwrap();
        (cache.ground_factors.clone(), cache.ground_colors.clone())
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_ground_shadow_cache(
        &mut self,
        settings: &RenderSettings,
        camera: &Camera,
        shadow_map: Option<&ShadowMap>,
        width: usize,
        height: usize,
    ) -> Arc<Vec<f32>> {
        let cache_valid = self.ground_shadow_cache.is_some()
            && self.ground_shadow_cache.as_ref().unwrap().width == width
            && self.ground_shadow_cache.as_ref().unwrap().height == height;

        if !cache_valid {
            debug!("重新计算地面阴影缓存");

            let mut shadow_factors = vec![1.0; width * height];

            shadow_factors
                .par_iter_mut()
                .enumerate()
                .for_each(|(buffer_index, shadow_factor)| {
                    let y = buffer_index / width;
                    let x = buffer_index % width;
                    let t_y = y as f32 / (height - 1) as f32;
                    let t_x = x as f32 / (width - 1) as f32;

                    *shadow_factor = compute_ground_shadow(settings, camera, t_x, t_y, shadow_map);
                });

            self.ground_shadow_cache = Some(GroundShadowCache {
                shadow_factors: Arc::new(shadow_factors),
                width,
                height,
            });

            debug!("地面阴影缓存计算完成");
        }

        self.ground_shadow_cache
            .as_ref()
            .unwrap()
            .shadow_factors
            .clone()
    }

    fn compose_final_colors(
        &self,
        settings: &RenderSettings,
        background_pixels_ref: &[Vector3<f32>],
        ground_factors_ref: &[f32],
        ground_colors_ref: &[Vector3<f32>],
        shadow_factors_ref: &[f32],
    ) {
        let width = self.width;
        let height = self.height;

        (0..height).into_par_iter().for_each(|y| {
            for x in 0..width {
                let buffer_index = y * width + x;
                let color_index = buffer_index * 3;

                let mut final_color = background_pixels_ref[buffer_index];

                if settings.enable_ground_plane {
                    let ground_factor = ground_factors_ref[buffer_index];
                    if ground_factor > 0.0 {
                        let mut ground_color = ground_colors_ref[buffer_index];
                        let shadow_factor = shadow_factors_ref[buffer_index];

                        // 让阴影影响地面颜色（包括网格线）
                        ground_color *= shadow_factor;

                        let enhanced_ground_factor = ground_factor.powf(0.65) * 2.0;
                        let final_ground_factor = enhanced_ground_factor.min(0.95);
                        let darkened_background =
                            final_color * (0.8 - final_ground_factor * 0.5).max(0.1);
                        final_color = darkened_background * (1.0 - final_ground_factor)
                            + ground_color * final_ground_factor;
                    }
                }

                let color_u8 = color::linear_rgb_to_u8(&final_color, settings.use_gamma);
                self.color_buffer[color_index].store(color_u8[0], Ordering::Relaxed);
                self.color_buffer[color_index + 1].store(color_u8[1], Ordering::Relaxed);
                self.color_buffer[color_index + 2].store(color_u8[2], Ordering::Relaxed);
            }
        });
    }

    fn get_background_image(&mut self, settings: &RenderSettings) -> Option<&Texture> {
        if !settings.use_background_image {
            return None;
        }

        let current_path = settings.background_image_path.as_ref()?;

        if let Some(cached_path) = &self.cached_path {
            if cached_path == current_path && self.cached_background.is_some() {
                return self.cached_background.as_ref();
            }
        }

        match Texture::from_file(current_path) {
            Some(texture) => {
                debug!("背景图片加载成功: {}x{}", texture.width, texture.height);
                self.cached_background = Some(texture);
                self.cached_path = Some(current_path.clone());
                self.cached_background.as_ref()
            }
            None => {
                warn!("无法加载背景图片 '{current_path}'");
                None
            }
        }
    }

    pub fn invalidate_caches(&mut self) {
        self.background_cache = None;
        self.ground_base_cache = None;
        self.ground_shadow_cache = None;
        debug!("已清除所有缓存");
    }

    pub fn invalidate_ground_shadow_cache(&mut self) {
        self.ground_shadow_cache = None;
        debug!("已清除地面阴影缓存");
    }

    pub fn invalidate_ground_base_cache(&mut self) {
        self.ground_base_cache = None;
        debug!("已清除地面本体缓存");
    }

    pub fn invalidate_background_cache(&mut self) {
        self.background_cache = None;
        debug!("已清除背景缓存");
    }

    pub fn get_color_buffer_bytes(&self) -> Vec<u8> {
        self.color_buffer
            .iter()
            .map(|atomic_color| atomic_color.load(Ordering::Relaxed))
            .collect()
    }

    pub fn get_depth_buffer_f32(&self) -> Vec<f32> {
        self.depth_buffer
            .iter()
            .map(|atomic_depth| atomic_depth.load(Ordering::Relaxed))
            .collect()
    }

    pub fn get_pixel_color(&self, x: usize, y: usize) -> Option<Vector3<f32>> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let buffer_index = y * self.width + x;
        let color_index = buffer_index * 3;
        if color_index + 2 < self.color_buffer.len() {
            let r = self.color_buffer[color_index].load(Ordering::Relaxed) as f32 / 255.0;
            let g = self.color_buffer[color_index + 1].load(Ordering::Relaxed) as f32 / 255.0;
            let b = self.color_buffer[color_index + 2].load(Ordering::Relaxed) as f32 / 255.0;
            Some(Vector3::new(r, g, b))
        } else {
            None
        }
    }

    pub fn get_pixel_color_as_color(&self, x: usize, y: usize) -> color::Color {
        if let Some(color_vec) = self.get_pixel_color(x, y) {
            color::Color::new(color_vec.x, color_vec.y, color_vec.z)
        } else {
            color::Color::new(0.1, 0.1, 0.1)
        }
    }
}

// ===== 背景和地面计算函数 =====

/// 纯背景颜色计算（不包括地面）
pub fn compute_background(
    settings: &RenderSettings,
    background_texture: Option<&Texture>,
    t_x: f32,
    t_y: f32,
) -> Vector3<f32> {
    // 1. 背景图片或渐变（基础层）
    let mut final_color = if let Some(background) = background_texture {
        let tex_x = t_x;
        let tex_y = 1.0 - t_y; // 翻转Y轴
        background.sample(tex_x, tex_y).into()
    } else if settings.enable_gradient_background {
        let top_color = settings.get_gradient_top_color_vec();
        let bottom_color = settings.get_gradient_bottom_color_vec();
        top_color * (1.0 - t_y) + bottom_color * t_y
    } else {
        Vector3::new(0.0, 0.0, 0.0)
    };

    // 2. 渐变叠加（如果有背景图片且启用渐变）
    if background_texture.is_some() && settings.enable_gradient_background {
        let top_color = settings.get_gradient_top_color_vec();
        let bottom_color = settings.get_gradient_bottom_color_vec();
        let gradient_color = top_color * (1.0 - t_y) + bottom_color * t_y;
        final_color = final_color * 0.3 + gradient_color * 0.7;
    }

    final_color
}

/// 地面本体（不含阴影）计算
pub fn compute_ground_base(
    settings: &RenderSettings,
    camera: &Camera,
    t_x: f32,
    t_y: f32,
) -> (f32, Vector3<f32>) {
    let (intersection, valid) = screen_to_ground_intersection(settings, camera, t_x, t_y);
    if !valid {
        return (0.0, Vector3::zeros());
    }

    let camera_position = camera.position();
    let far_plane = camera.far();

    // 网格线判定
    let grid_size = 1.0;
    let grid_x = ((intersection.x - camera_position.x * 0.1) / grid_size).abs() % 1.0;
    let grid_z = ((intersection.z - camera_position.z * 0.1) / grid_size).abs() % 1.0;
    let distance_from_camera = (intersection - camera_position).magnitude();
    let adaptive_line_width = (0.02 + distance_from_camera * 0.001).min(0.1);
    let is_grid_line = grid_x < adaptive_line_width
        || grid_x > (1.0 - adaptive_line_width)
        || grid_z < adaptive_line_width
        || grid_z > (1.0 - adaptive_line_width);
    let grid_factor = if is_grid_line { 0.8 } else { 0.0 };

    // 距离衰减
    let effective_far = far_plane * 0.8;
    let distance_factor = (distance_from_camera / effective_far).min(1.0);

    // 基础地面强度
    let camera_height = camera_position.y - settings.ground_plane_height;
    let height_factor = (camera_height / 8.0).clamp(0.3, 1.5);

    // 角度增强
    let ground_normal = Vector3::y();
    let ray_dir = (intersection - camera_position).normalize();
    let ray_to_ground_angle = ray_dir.dot(&ground_normal).abs();
    let angle_enhanced = ray_to_ground_angle.powf(0.8) * height_factor * 1.2;

    // 世界空间边缘淡出
    let world_center_dist =
        ((intersection.x / 20.0).powi(2) + (intersection.z / 20.0).powi(2)).sqrt();
    let world_edge_factor = (1.0 - (world_center_dist / 5.0).min(1.0)).max(0.0);

    // 聚光灯效果
    let view_matrix = camera.view_matrix();
    let view_forward = view_matrix.column(2).xyz().normalize();
    let center_alignment = ray_dir.dot(&view_forward).max(0.0);
    let spotlight_factor = center_alignment.powf(2.0);

    let combined_edge_factor = world_edge_factor * 0.4 + spotlight_factor * 0.6;

    let combined_factor = (1.0 - distance_factor).powf(0.35)
        * angle_enhanced
        * (1.0 - grid_factor * 0.75)
        * combined_edge_factor;

    let ground_factor = (combined_factor * 1.1).max(0.0);

    // 地面颜色
    let mut ground_color = settings.get_ground_plane_color_vec() * 1.6;

    // 增强饱和度
    let luminance = ground_color.x * 0.299 + ground_color.y * 0.587 + ground_color.z * 0.114;
    ground_color = ground_color * 0.8 + Vector3::new(luminance, luminance, luminance) * 0.2;
    ground_color *= 1.1;

    // 色调变化 - 增强对比度
    let t_x_centered = (t_x - 0.5) * 2.0;
    let camera_influence = (camera.position().x * 0.05).sin() * 0.05;
    ground_color.x *= 1.0 + t_x_centered * 0.1 + camera_influence;
    ground_color.y *= 1.0 - t_x_centered.abs() * 0.04 + camera_influence * 0.5;
    ground_color.z *= 1.0 - t_x_centered.abs() * 0.05;

    // 减轻大气透视影响
    let distance_from_center = ((t_x - 0.5).powi(2) + (t_y - 0.75).powi(2)).sqrt();
    let camera_height = camera.position().y;
    let height_factor = (camera_height / 8.0).clamp(0.3, 1.5);
    let atmospheric_factor = distance_from_center * 0.1 * height_factor;

    ground_color = ground_color * (1.0 - atmospheric_factor)
        + Vector3::new(0.7, 0.8, 0.9) * atmospheric_factor;

    // 减少天空反射影响，加强地面本身颜色
    let sky_reflection_strength = (camera_height / 15.0).clamp(0.02, 0.08);
    let sky_reflection = settings.get_gradient_top_color_vec() * sky_reflection_strength;
    ground_color += sky_reflection * (1.0 - (t_y - 0.5) * 1.5).max(0.0);

    // 确保地面颜色不会过暗
    ground_color = ground_color.map(|x| x.max(0.15));

    (ground_factor, ground_color)
}

/// 地面阴影因子计算
pub fn compute_ground_shadow(
    settings: &RenderSettings,
    camera: &Camera,
    t_x: f32,
    t_y: f32,
    shadow_map: Option<&ShadowMap>,
) -> f32 {
    let (intersection, valid) = screen_to_ground_intersection(settings, camera, t_x, t_y);
    if !valid {
        return 1.0;
    }

    if settings.enable_shadow_mapping {
        if let Some(shadow_map) = shadow_map {
            if shadow_map.is_valid {
                shadow_map.compute_shadow_factor(
                    &intersection,
                    &Matrix4::identity(),
                    settings.shadow_bias,
                    settings.enable_pcf,
                    &settings.pcf_type,
                    settings.pcf_kernel,
                    settings.pcf_sigma,
                )
            } else {
                1.0
            }
        } else {
            1.0
        }
    } else {
        1.0
    }
}

/// 屏幕坐标到地面交点
pub fn screen_to_ground_intersection(
    settings: &RenderSettings,
    camera: &Camera,
    t_x: f32,
    t_y: f32,
) -> (Point3<f32>, bool) {
    let (origin, dir) = screen_to_world_ray(camera.clone(), t_x, t_y);
    let ground_y = settings.ground_plane_height;
    let ground_normal = Vector3::y();
    let plane_point = Point3::new(0.0, ground_y, 0.0);

    let denominator = ground_normal.dot(&dir);
    if denominator.abs() <= 1e-4 {
        return (Point3::origin(), false);
    }
    let t = (plane_point - origin).dot(&ground_normal) / denominator;
    if t < camera.near() || t > camera.far() * 1.5 {
        return (Point3::origin(), false);
    }
    let intersection = origin + t * dir;
    let max_render_distance = 100.0;
    let horizontal_distance =
        ((intersection.x - origin.x).powi(2) + (intersection.z - origin.z).powi(2)).sqrt();
    if horizontal_distance > max_render_distance {
        return (Point3::origin(), false);
    }
    (intersection, true)
}

fn screen_to_world_ray(camera: Camera, t_x: f32, t_y: f32) -> (Point3<f32>, Vector3<f32>) {
    let fov_y_rad = match &camera.params.projection {
        ProjectionType::Perspective { fov_y_degrees, .. } => fov_y_degrees.to_radians(),
        ProjectionType::Orthographic { .. } => 45.0_f32.to_radians(),
    };
    let aspect_ratio = camera.aspect_ratio();
    let camera_position = camera.position();
    let view_matrix = camera.view_matrix();

    let ndc_x = t_x * 2.0 - 1.0;
    let ndc_y = 1.0 - t_y * 2.0;

    let view_x = ndc_x * aspect_ratio * (fov_y_rad / 2.0).tan();
    let view_y = ndc_y * (fov_y_rad / 2.0).tan();
    let view_dir = Vector3::new(view_x, view_y, -1.0).normalize();

    let view_to_world = view_matrix.try_inverse().unwrap_or_else(Matrix4::identity);
    let world_ray_dir = view_to_world.transform_vector(&view_dir).normalize();

    (camera_position, world_ray_dir)
}
pub mod frame_buffer;
pub mod rasterizer;
pub mod renderer;
pub mod shadow_map;
use crate::core::frame_buffer::FrameBuffer;
use crate::core::rasterizer::Rasterizer;
use crate::core::shadow_map::ShadowMap;
use crate::geometry::camera::Camera;
use crate::geometry::transform::{
    TransformFactory, clip_to_screen, compute_normal_matrix, point_to_clip, transform_normal,
    transform_point,
};
use crate::io::render_settings::RenderSettings;
use crate::material_system::light::Light;
use crate::scene::scene_object::SceneObject;
use crate::scene::scene_utils::Scene;
use log::debug;
use nalgebra::{Point2, Point3, Vector3};
use rayon::prelude::*;
use std::time::Instant;

pub struct TransformedGeometry {
    pub screen_coords: Vec<Point2<f32>>,
    pub view_coords: Vec<Point3<f32>>,
    pub view_normals: Vec<Vector3<f32>>,
    pub mesh_offsets: Vec<usize>,
}

pub fn transform_geometry(
    scene_object: &SceneObject,
    camera: &mut Camera,
    frame_width: usize,
    frame_height: usize,
) -> TransformedGeometry {
    camera.update_matrices();

    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut mesh_offsets = vec![0];
    for mesh in &scene_object.model.meshes {
        vertices.extend(mesh.vertices.iter().map(|v| v.position));
        normals.extend(mesh.vertices.iter().map(|v| v.normal));
        mesh_offsets.push(vertices.len());
    }

    let model_matrix = &scene_object.transform;
    let view_matrix = &camera.view_matrix();
    let projection_matrix = &camera.projection_matrix();

    let model_view = TransformFactory::model_view(model_matrix, view_matrix);
    let mvp = TransformFactory::model_view_projection(model_matrix, view_matrix, projection_matrix);
    let normal_matrix = compute_normal_matrix(&model_view);

    let view_positions = vertices
        .par_iter()
        .map(|v| transform_point(v, &model_view))
        .collect();
    let screen_coords = vertices
        .par_iter()
        .map(|v| {
            let clip = point_to_clip(v, &mvp);
            clip_to_screen(&clip, frame_width as f32, frame_height as f32)
        })
        .collect();
    let view_normals = normals
        .par_iter()
        .map(|n| transform_normal(n, &normal_matrix))
        .collect();

    TransformedGeometry {
        screen_coords,
        view_coords: view_positions,
        view_normals,
        mesh_offsets,
    }
}

pub struct Renderer {
    pub frame_buffer: FrameBuffer,
    shadow_map: Option<ShadowMap>,
    last_frame_time: Option<std::time::Duration>,
}

impl Renderer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            frame_buffer: FrameBuffer::new(width, height),
            shadow_map: None,
            last_frame_time: None,
        }
    }

    pub fn resize(&mut self, width: usize, height: usize) {
        if self.frame_buffer.width != width || self.frame_buffer.height != height {
            debug!(
                "渲染器尺寸变化: {}x{} -> {}x{}",
                self.frame_buffer.width, self.frame_buffer.height, width, height
            );
            self.frame_buffer = FrameBuffer::new(width, height);
        }
    }

    pub fn render_scene(&mut self, scene: &mut Scene, settings: &RenderSettings) {
        let frame_start = Instant::now();

        self.resize(settings.width, settings.height);

        if settings.enable_shadow_mapping {
            self.generate_shadow_map(scene, settings);
        }

        self.frame_buffer
            .clear(settings, &scene.active_camera, self.shadow_map.as_ref());

        let geometry = transform_geometry(
            &scene.object,
            &mut scene.active_camera,
            self.frame_buffer.width,
            self.frame_buffer.height,
        );

        let triangles = Rasterizer::prepare_triangles(
            &scene.object.model,
            &geometry,
            None,
            settings,
            &scene.lights,
            scene.ambient_intensity,
            scene.ambient_color,
        );

        Rasterizer::rasterize_triangles(
            &triangles,
            self.frame_buffer.width,
            self.frame_buffer.height,
            &self.frame_buffer.depth_buffer,
            &self.frame_buffer.color_buffer,
            settings,
            &self.frame_buffer,
        );

        self.last_frame_time = Some(frame_start.elapsed());
        debug!(
            "渲染完成 '{}': {} 三角形, 耗时: {:?}",
            scene.object.model.name,
            triangles.len(),
            self.last_frame_time.unwrap()
        );
    }

    fn generate_shadow_map(&mut self, scene: &Scene, settings: &RenderSettings) {
        if self.shadow_map.is_none()
            || self.shadow_map.as_ref().unwrap().size != settings.shadow_map_size
        {
            self.shadow_map = Some(ShadowMap::new(settings.shadow_map_size));
            debug!(
                "创建新阴影贴图: {}x{}",
                settings.shadow_map_size, settings.shadow_map_size
            );
        }

        let shadow_map = self.shadow_map.as_mut().unwrap();

        if let Some(directional_light) = scene
            .lights
            .iter()
            .find(|light| matches!(light, Light::Directional { enabled: true, .. }))
        {
            let scene_bounds = Self::compute_scene_bounds(scene, settings);
            shadow_map.generate(&scene.object, directional_light, scene_bounds);

            if shadow_map.is_valid {
                debug!("阴影贴图已更新");
            } else {
                debug!("阴影贴图生成失败");
            }
        } else {
            shadow_map.is_valid = false;
            debug!("未找到可用的方向光源，跳过阴影贴图生成");
        }
    }

    fn compute_scene_bounds(
        scene: &Scene,
        settings: &RenderSettings,
    ) -> (nalgebra::Point3<f32>, f32) {
        let mut min_pos = nalgebra::Point3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut max_pos =
            nalgebra::Point3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
        let mut has_vertices = false;

        for mesh in &scene.object.model.meshes {
            for vertex in &mesh.vertices {
                let world_pos = scene.object.transform.transform_point(&vertex.position);

                min_pos.x = min_pos.x.min(world_pos.x);
                min_pos.y = min_pos.y.min(world_pos.y);
                min_pos.z = min_pos.z.min(world_pos.z);

                max_pos.x = max_pos.x.max(world_pos.x);
                max_pos.y = max_pos.y.max(world_pos.y);
                max_pos.z = max_pos.z.max(world_pos.z);

                has_vertices = true;
            }
        }

        if !has_vertices {
            debug!("场景无顶点数据，使用默认阴影边界");
            return (
                nalgebra::Point3::new(0.0, 0.0, 0.0),
                settings.shadow_distance,
            );
        }

        let center = nalgebra::Point3::new(
            (min_pos.x + max_pos.x) * 0.5,
            (min_pos.y + max_pos.y) * 0.5,
            (min_pos.z + max_pos.z) * 0.5,
        );

        let size = max_pos - min_pos;
        let radius = (size.x.max(size.y).max(size.z) * 0.6).max(settings.shadow_distance * 0.5);

        let ground_extended_radius = if settings.enable_ground_plane {
            let ground_distance = (center.y - settings.ground_plane_height).abs() + radius;
            radius.max(ground_distance)
        } else {
            radius
        };

        debug!(
            "场景边界: 中心({:.2}, {:.2}, {:.2}), 半径: {:.2}",
            center.x, center.y, center.z, ground_extended_radius
        );

        (center, ground_extended_radius)
    }
}

impl Default for Renderer {
    fn default() -> Self {
        Self::new(800, 600)
    }
}
use crate::geometry::interpolation::barycentric_coordinates;
use crate::geometry::transform::{TransformFactory, transform_point};
use crate::material_system::light::Light;
use crate::scene::scene_object::SceneObject;
use log::debug;
use nalgebra::{Matrix4, Point2, Point3, Vector3};

/// 简化阴影贴图
#[derive(Debug, Clone)]
pub struct ShadowMap {
    pub depth_buffer: Vec<f32>,
    pub size: usize,
    pub light_view_proj_matrix: Matrix4<f32>,
    pub is_valid: bool,
}

impl ShadowMap {
    pub fn new(size: usize) -> Self {
        Self {
            depth_buffer: vec![f32::INFINITY; size * size],
            size,
            light_view_proj_matrix: Matrix4::identity(),
            is_valid: false,
        }
    }

    pub fn generate(
        &mut self,
        scene_object: &SceneObject,
        directional_light: &Light,
        scene_bounds: (Point3<f32>, f32),
    ) -> bool {
        if let Light::Directional {
            direction, enabled, ..
        } = directional_light
        {
            if !enabled {
                return false;
            }

            self.setup_light_matrices(direction, scene_bounds);
            self.render_shadow_casters(scene_object);
            self.is_valid = true;
            debug!("阴影贴图生成完成: {}x{}", self.size, self.size);
            true
        } else {
            false
        }
    }

    fn setup_light_matrices(
        &mut self,
        light_direction: &Vector3<f32>,
        (scene_center, scene_radius): (Point3<f32>, f32),
    ) {
        let light_pos = scene_center - light_direction * scene_radius * 2.0;

        let up = if light_direction.y.abs() > 0.9 {
            Vector3::x()
        } else {
            Vector3::y()
        };

        let light_view = TransformFactory::view(&light_pos, &scene_center, &up);
        let ortho_size = scene_radius * 1.2;
        let light_proj = TransformFactory::orthographic(
            -ortho_size,
            ortho_size,
            -ortho_size,
            ortho_size,
            0.1,
            scene_radius * 4.0,
        );

        self.light_view_proj_matrix = light_proj * light_view;
    }

    fn render_shadow_casters(&mut self, scene_object: &SceneObject) {
        self.depth_buffer.fill(f32::INFINITY);
        let full_transform = self.light_view_proj_matrix * scene_object.transform;

        let mut triangles_processed = 0;
        let mut triangles_culled = 0;

        for mesh in &scene_object.model.meshes {
            for indices in mesh.indices.chunks_exact(3) {
                let vertices = [
                    mesh.vertices[indices[0] as usize].position,
                    mesh.vertices[indices[1] as usize].position,
                    mesh.vertices[indices[2] as usize].position,
                ];

                let transformed_vertices = [
                    transform_point(&vertices[0], &full_transform),
                    transform_point(&vertices[1], &full_transform),
                    transform_point(&vertices[2], &full_transform),
                ];

                if self.is_triangle_outside_frustum(&transformed_vertices) {
                    triangles_culled += 1;
                    continue;
                }

                let screen_coords = [
                    self.ndc_to_shadow_coord(transformed_vertices[0].x, transformed_vertices[0].y),
                    self.ndc_to_shadow_coord(transformed_vertices[1].x, transformed_vertices[1].y),
                    self.ndc_to_shadow_coord(transformed_vertices[2].x, transformed_vertices[2].y),
                ];

                self.rasterize_triangle(&transformed_vertices, &screen_coords);
                triangles_processed += 1;
            }
        }

        debug!("阴影三角形: 处理 {triangles_processed}, 剔除 {triangles_culled}");
    }

    #[inline]
    fn is_triangle_outside_frustum(&self, vertices: &[Point3<f32>; 3]) -> bool {
        let outside_bounds = |get_axis: fn(&Point3<f32>) -> f32| {
            vertices.iter().all(|v| get_axis(v) < -1.0)
                || vertices.iter().all(|v| get_axis(v) > 1.0)
        };

        outside_bounds(|v| v.x) || outside_bounds(|v| v.y) || outside_bounds(|v| v.z)
    }

    fn rasterize_triangle(&mut self, vertices: &[Point3<f32>; 3], screen_coords: &[(f32, f32); 3]) {
        let (min_x, max_x) = screen_coords
            .iter()
            .map(|(x, _)| *x as i32)
            .fold((i32::MAX, i32::MIN), |(min_x, max_x), x| {
                (min_x.min(x), max_x.max(x))
            });

        let (min_y, max_y) = screen_coords
            .iter()
            .map(|(_, y)| *y as i32)
            .fold((i32::MAX, i32::MIN), |(min_y, max_y), y| {
                (min_y.min(y), max_y.max(y))
            });

        let min_x = min_x.max(0);
        let max_x = max_x.min(self.size as i32 - 1);
        let min_y = min_y.max(0);
        let max_y = max_y.min(self.size as i32 - 1);

        if max_x <= min_x || max_y <= min_y {
            return;
        }

        let triangle_points = [
            Point2::new(screen_coords[0].0, screen_coords[0].1),
            Point2::new(screen_coords[1].0, screen_coords[1].1),
            Point2::new(screen_coords[2].0, screen_coords[2].1),
        ];

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let pixel_point = Point2::new(x as f32 + 0.5, y as f32 + 0.5); // ← 推荐这样
                if let Some(bary) = barycentric_coordinates(
                    pixel_point,
                    triangle_points[0],
                    triangle_points[1],
                    triangle_points[2],
                ) {
                    if bary.x >= 0.0 && bary.y >= 0.0 && bary.z >= 0.0 {
                        let depth = bary.x * vertices[0].z
                            + bary.y * vertices[1].z
                            + bary.z * vertices[2].z;
                        let shadow_index = (y as usize) * self.size + (x as usize);

                        if shadow_index < self.depth_buffer.len()
                            && depth < self.depth_buffer[shadow_index]
                        {
                            self.depth_buffer[shadow_index] = depth;
                        }
                    }
                }
            }
        }
    }

    #[inline]
    fn ndc_to_shadow_coord(&self, ndc_x: f32, ndc_y: f32) -> (f32, f32) {
        let coord = |ndc: f32| (ndc + 1.0) * 0.5 * self.size as f32;
        (coord(ndc_x), coord(ndc_y))
    }

    pub fn sample_depth(&self, u: f32, v: f32) -> f32 {
        if !self.is_valid || !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
            return f32::INFINITY;
        }

        let x = (u * (self.size - 1) as f32) as usize;
        let y = (v * (self.size - 1) as f32) as usize;
        let index = y * self.size + x;

        self.depth_buffer
            .get(index)
            .copied()
            .unwrap_or(f32::INFINITY)
    }

    pub fn compute_shadow_factor(
        &self,
        world_pos: &Point3<f32>,
        model_matrix: &Matrix4<f32>,
        bias: f32,
        enable_pcf: bool,
        pcf_type: &str,
        pcf_kernel: usize,
        pcf_sigma: f32,
    ) -> f32 {
        if !self.is_valid {
            return 1.0;
        }

        let full_transform = self.light_view_proj_matrix * model_matrix;
        let light_space_pos = transform_point(world_pos, &full_transform);

        let shadow_coords = (
            (light_space_pos.x + 1.0) * 0.5,
            (light_space_pos.y + 1.0) * 0.5,
        );

        if !(0.0..=1.0).contains(&shadow_coords.0) || !(0.0..=1.0).contains(&shadow_coords.1) {
            return 1.0;
        }

        let current_depth = light_space_pos.z;

        if enable_pcf {
            let kernel = pcf_kernel as i32;
            let sigma = pcf_sigma;

            let mut shadow = 0.0;
            let mut total_weight = 0.0;

            for dx in -kernel..=kernel {
                for dy in -kernel..=kernel {
                    let u = shadow_coords.0 + dx as f32 / self.size as f32;
                    let v = shadow_coords.1 + dy as f32 / self.size as f32;

                    let weight = if pcf_type == "Gauss" {
                        (-((dx * dx + dy * dy) as f32) / (2.0 * sigma * sigma)).exp()
                    } else {
                        1.0 // Box
                    };

                    let pcf_depth = self.sample_depth(u, v);
                    if current_depth - bias > pcf_depth {
                        shadow += weight;
                    }
                    total_weight += weight;
                }
            }
            shadow /= total_weight;
            1.0 - shadow
        } else {
            // 普通硬阴影
            let pcf_depth = self.sample_depth(shadow_coords.0, shadow_coords.1);
            if current_depth - bias > pcf_depth {
                0.2
            } else {
                1.0
            }
        }
    }
}
use crate::core::frame_buffer::FrameBuffer;
use crate::core::renderer::TransformedGeometry;
use crate::geometry::culling::{
    is_backface, is_on_triangle_edge, is_valid_triangle, should_cull_small_triangle,
};
use crate::geometry::interpolation::{
    barycentric_coordinates, interpolate_depth, interpolate_normal, interpolate_position,
    interpolate_texcoords, is_inside_triangle,
};
use crate::io::render_settings::RenderSettings;
use crate::material_system::color::{apply_aces_tonemap, get_random_color, linear_rgb_to_u8};
use crate::material_system::light::Light;
use crate::material_system::materials::{Material, Model, Vertex, compute_material_response};
use crate::material_system::texture::Texture;
use atomic_float::AtomicF32;
use nalgebra::{Point2, Point3, Vector2, Vector3};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Debug, Clone)]
pub struct RasterVertex {
    pub pix: Point2<f32>,
    pub z_view: f32,
    pub texcoord: Option<Vector2<f32>>,
    pub normal_view: Option<Vector3<f32>>,
    pub position_view: Option<Point3<f32>>,
}

pub struct RasterTriangle<'a> {
    pub vertices: [RasterVertex; 3],
    pub base_color: Vector3<f32>,
    pub texture: Option<&'a Texture>,
    pub material: Option<&'a Material>,
    pub lights: &'a [Light],
    pub ambient_intensity: f32,
    pub ambient_color: Vector3<f32>,
    pub is_perspective: bool,
    pub face_seed: Option<u64>,
}

impl<'a> RasterTriangle<'a> {
    pub fn is_valid(&self) -> bool {
        is_valid_triangle(
            &self.vertices[0].pix,
            &self.vertices[1].pix,
            &self.vertices[2].pix,
        )
    }
}

pub struct Rasterizer;

impl Rasterizer {
    pub fn prepare_triangles<'a>(
        model: &'a Model,
        geometry: &TransformedGeometry,
        material_override: Option<&'a Material>,
        settings: &'a RenderSettings,
        lights: &'a [Light],
        ambient_intensity: f32,
        ambient_color: Vector3<f32>,
    ) -> Vec<RasterTriangle<'a>> {
        model
            .meshes
            .par_iter()
            .enumerate()
            .flat_map(|(mesh_idx, mesh)| {
                let vertex_offset = geometry.mesh_offsets[mesh_idx];
                let material_opt =
                    material_override.or_else(|| model.materials.get(mesh.material_id));
                mesh.indices
                    .chunks_exact(3)
                    .enumerate()
                    .filter_map(move |(face_idx, indices)| {
                        let global_face_index = (mesh_idx * 1000 + face_idx) as u64;
                        Self::process_triangle(
                            indices,
                            &mesh.vertices,
                            vertex_offset,
                            global_face_index,
                            geometry,
                            material_opt,
                            settings,
                            lights,
                            ambient_intensity,
                            ambient_color,
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn process_triangle<'a>(
        indices: &[u32],
        vertices: &[Vertex],
        vertex_offset: usize,
        global_face_index: u64,
        geometry: &TransformedGeometry,
        material_opt: Option<&'a Material>,
        settings: &'a RenderSettings,
        lights: &'a [Light],
        ambient_intensity: f32,
        ambient_color: Vector3<f32>,
    ) -> Option<RasterTriangle<'a>> {
        let i0 = indices[0] as usize;
        let i1 = indices[1] as usize;
        let i2 = indices[2] as usize;
        let global_i0 = vertex_offset + i0;
        let global_i1 = vertex_offset + i1;
        let global_i2 = vertex_offset + i2;

        if global_i0 >= geometry.screen_coords.len()
            || global_i1 >= geometry.screen_coords.len()
            || global_i2 >= geometry.screen_coords.len()
        {
            return None;
        }

        let pix0 = geometry.screen_coords[global_i0];
        let pix1 = geometry.screen_coords[global_i1];
        let pix2 = geometry.screen_coords[global_i2];
        let view_pos0 = geometry.view_coords[global_i0];
        let view_pos1 = geometry.view_coords[global_i1];
        let view_pos2 = geometry.view_coords[global_i2];

        if settings.backface_culling && is_backface(&view_pos0, &view_pos1, &view_pos2) {
            return None;
        }
        if settings.cull_small_triangles
            && should_cull_small_triangle(&pix0, &pix1, &pix2, settings.min_triangle_area)
        {
            return None;
        }

        let (texture, base_color, face_seed) = if let Some(mat) = material_opt {
            if let Some(tex) = &mat.texture {
                (Some(tex), mat.base_color, None)
            } else if settings.colorize {
                (None, Vector3::new(1.0, 1.0, 1.0), Some(global_face_index))
            } else {
                (None, mat.base_color, None)
            }
        } else if settings.colorize {
            (None, Vector3::new(1.0, 1.0, 1.0), Some(global_face_index))
        } else {
            (None, Vector3::new(0.7, 0.7, 0.7), None)
        };

        let vertex_data = [
            Self::create_vertex(
                &pix0,
                view_pos0,
                &vertices[i0],
                global_i0,
                texture,
                geometry,
            ),
            Self::create_vertex(
                &pix1,
                view_pos1,
                &vertices[i1],
                global_i1,
                texture,
                geometry,
            ),
            Self::create_vertex(
                &pix2,
                view_pos2,
                &vertices[i2],
                global_i2,
                texture,
                geometry,
            ),
        ];

        Some(RasterTriangle {
            vertices: vertex_data,
            base_color,
            texture,
            material: material_opt,
            lights,
            ambient_intensity,
            ambient_color,
            is_perspective: settings.is_perspective(),
            face_seed,
        })
    }

    fn create_vertex(
        pix: &Point2<f32>,
        view_pos: Point3<f32>,
        vertex: &Vertex,
        global_index: usize,
        texture: Option<&Texture>,
        geometry: &TransformedGeometry,
    ) -> RasterVertex {
        RasterVertex {
            pix: *pix,
            z_view: view_pos.z,
            texcoord: if texture.is_some() {
                Some(vertex.texcoord)
            } else {
                None
            },
            normal_view: Some(geometry.view_normals[global_index]),
            position_view: Some(view_pos),
        }
    }

    pub fn rasterize_triangles(
        triangles: &[RasterTriangle],
        width: usize,
        height: usize,
        depth_buffer: &[AtomicF32],
        color_buffer: &[AtomicU8],
        settings: &RenderSettings,
        frame_buffer: &FrameBuffer,
    ) {
        if triangles.is_empty() {
            return;
        }
        triangles.par_iter().for_each(|tri| {
            Self::rasterize_triangle(
                tri,
                width,
                height,
                depth_buffer,
                color_buffer,
                settings,
                frame_buffer,
            )
        });
    }

    pub fn rasterize_triangle(
        triangle: &RasterTriangle,
        width: usize,
        height: usize,
        depth_buffer: &[AtomicF32],
        color_buffer: &[AtomicU8],
        settings: &RenderSettings,
        frame_buffer: &FrameBuffer,
    ) {
        if !triangle.is_valid() {
            return;
        }
        let (min_x, min_y, max_x, max_y) = Self::compute_bounding_box(triangle, width, height);
        if max_x <= min_x || max_y <= min_y {
            return;
        }
        let ambient_contribution = Self::calculate_ambient(triangle);

        for y in min_y..max_y {
            for x in min_x..max_x {
                let pixel_index = y * width + x;
                let pixel_center = Point2::new(x as f32 + 0.5, y as f32 + 0.5);
                Self::process_pixel(
                    triangle,
                    pixel_center,
                    pixel_index,
                    x,
                    y,
                    settings.use_lighting,
                    &ambient_contribution,
                    depth_buffer,
                    color_buffer,
                    settings,
                    frame_buffer,
                );
            }
        }
    }

    fn compute_bounding_box(
        triangle: &RasterTriangle,
        width: usize,
        height: usize,
    ) -> (usize, usize, usize, usize) {
        let v0 = &triangle.vertices[0].pix;
        let v1 = &triangle.vertices[1].pix;
        let v2 = &triangle.vertices[2].pix;
        let min_x = v0.x.min(v1.x).min(v2.x).floor().max(0.0) as usize;
        let min_y = v0.y.min(v1.y).min(v2.y).floor().max(0.0) as usize;
        let max_x = v0.x.max(v1.x).max(v2.x).ceil().min(width as f32) as usize;
        let max_y = v0.y.max(v1.y).max(v2.y).ceil().min(height as f32) as usize;
        (min_x, min_y, max_x, max_y)
    }

    #[allow(clippy::too_many_arguments)]
    fn process_pixel(
        triangle: &RasterTriangle,
        pixel_center: Point2<f32>,
        pixel_index: usize,
        pixel_x: usize,
        pixel_y: usize,
        use_lighting: bool,
        ambient_contribution: &Vector3<f32>,
        depth_buffer: &[AtomicF32],
        color_buffer: &[AtomicU8],
        settings: &RenderSettings,
        frame_buffer: &FrameBuffer,
    ) {
        let v0 = &triangle.vertices[0].pix;
        let v1 = &triangle.vertices[1].pix;
        let v2 = &triangle.vertices[2].pix;
        let bary = match barycentric_coordinates(pixel_center, *v0, *v1, *v2) {
            Some(b) => b,
            None => return,
        };
        if !is_inside_triangle(bary) {
            return;
        }
        let final_alpha = Self::get_alpha(triangle, settings);
        if final_alpha <= 0.01 {
            return;
        }
        if settings.wireframe && !is_on_triangle_edge(pixel_center, *v0, *v1, *v2, 1.0) {
            return;
        }
        let interpolated_depth = interpolate_depth(
            bary,
            triangle.vertices[0].z_view,
            triangle.vertices[1].z_view,
            triangle.vertices[2].z_view,
            settings.is_perspective() && triangle.is_perspective,
        );
        if !interpolated_depth.is_finite() || interpolated_depth >= f32::INFINITY {
            return;
        }
        if settings.use_zbuffer {
            let current_depth_atomic = &depth_buffer[pixel_index];
            let old_depth = current_depth_atomic.fetch_min(interpolated_depth, Ordering::Relaxed);
            if old_depth <= interpolated_depth {
                return;
            }
        }
        let material_color =
            Self::calculate_color(triangle, bary, settings, use_lighting, ambient_contribution);
        let final_color = Self::apply_alpha_blending(
            &material_color,
            final_alpha,
            pixel_x,
            pixel_y,
            frame_buffer,
        );
        Self::write_pixel_color(pixel_index, &final_color, color_buffer, settings);
    }

    fn calculate_color(
        triangle: &RasterTriangle,
        bary: Vector3<f32>,
        settings: &RenderSettings,
        use_lighting: bool,
        ambient_contribution: &Vector3<f32>,
    ) -> Vector3<f32> {
        let surface_color = if let Some(tex) = triangle.texture {
            if let (Some(tc1), Some(tc2), Some(tc3)) = (
                triangle.vertices[0].texcoord,
                triangle.vertices[1].texcoord,
                triangle.vertices[2].texcoord,
            ) {
                let tc = interpolate_texcoords(
                    bary,
                    tc1,
                    tc2,
                    tc3,
                    triangle.vertices[0].z_view,
                    triangle.vertices[1].z_view,
                    triangle.vertices[2].z_view,
                    triangle.is_perspective,
                );
                let arr = tex.sample(tc.x, tc.y);
                Vector3::new(arr[0], arr[1], arr[2])
            } else {
                Vector3::new(1.0, 1.0, 1.0)
            }
        } else if let Some(seed) = triangle.face_seed {
            get_random_color(seed, true)
        } else {
            triangle.base_color
        };

        if use_lighting
            && triangle.material.is_some()
            && triangle.vertices[0].normal_view.is_some()
            && triangle.vertices[0].position_view.is_some()
            && !triangle.lights.is_empty()
        {
            let interp_normal = interpolate_normal(
                bary,
                triangle.vertices[0].normal_view.unwrap(),
                triangle.vertices[1].normal_view.unwrap(),
                triangle.vertices[2].normal_view.unwrap(),
                triangle.is_perspective,
                triangle.vertices[0].z_view,
                triangle.vertices[1].z_view,
                triangle.vertices[2].z_view,
            );
            let interp_position = interpolate_position(
                bary,
                triangle.vertices[0].position_view.unwrap(),
                triangle.vertices[1].position_view.unwrap(),
                triangle.vertices[2].position_view.unwrap(),
                triangle.is_perspective,
                triangle.vertices[0].z_view,
                triangle.vertices[1].z_view,
                triangle.vertices[2].z_view,
            );
            let view_dir = (-interp_position.coords).normalize();
            let mut total_direct_light = Vector3::zeros();
            for light in triangle.lights {
                let light_dir = light.get_direction(&interp_position);
                let light_intensity = light.get_intensity(&interp_position);
                let response = compute_material_response(
                    triangle.material.unwrap(),
                    &light_dir,
                    &view_dir,
                    &interp_normal,
                );
                total_direct_light += Vector3::new(
                    response.x * light_intensity.x,
                    response.y * light_intensity.y,
                    response.z * light_intensity.z,
                );
            }
            surface_color.component_mul(&(total_direct_light + *ambient_contribution))
        } else if settings.use_lighting {
            surface_color.component_mul(ambient_contribution)
        } else {
            surface_color
        }
    }

    fn calculate_ambient(triangle: &RasterTriangle) -> Vector3<f32> {
        let ambient_color = triangle.ambient_color;
        let ambient_intensity = triangle.ambient_intensity;
        let ambient = Vector3::new(
            ambient_color.x * ambient_intensity,
            ambient_color.y * ambient_intensity,
            ambient_color.z * ambient_intensity,
        );
        if let Some(material) = triangle.material {
            return Vector3::new(
                material.ambient_factor.x * ambient.x,
                material.ambient_factor.y * ambient.y,
                material.ambient_factor.z * ambient.z,
            );
        }
        ambient
    }

    fn get_alpha(triangle: &RasterTriangle, settings: &RenderSettings) -> f32 {
        let material_alpha = triangle.material.map_or(1.0, |m| m.alpha);
        (material_alpha * settings.alpha).clamp(0.0, 1.0)
    }

    fn apply_alpha_blending(
        material_color: &Vector3<f32>,
        alpha: f32,
        pixel_x: usize,
        pixel_y: usize,
        frame_buffer: &FrameBuffer,
    ) -> Vector3<f32> {
        if alpha >= 1.0 {
            return *material_color;
        }
        if alpha <= 0.0 {
            return if let Some(bg_color) = frame_buffer.get_pixel_color(pixel_x, pixel_y) {
                bg_color
            } else {
                Vector3::new(0.0, 0.0, 0.0)
            };
        }
        let background_color = frame_buffer.get_pixel_color_as_color(pixel_x, pixel_y);
        Vector3::new(
            material_color.x * alpha + background_color.x * (1.0 - alpha),
            material_color.y * alpha + background_color.y * (1.0 - alpha),
            material_color.z * alpha + background_color.z * (1.0 - alpha),
        )
    }

    #[inline]
    fn write_pixel_color(
        pixel_index: usize,
        color: &Vector3<f32>,
        color_buffer: &[AtomicU8],
        settings: &RenderSettings,
    ) {
        let final_color = if settings.enable_aces {
            apply_aces_tonemap(color)
        } else {
            *color
        };
        let buffer_start_index = pixel_index * 3;
        if buffer_start_index + 2 < color_buffer.len() {
            let [r, g, b] = linear_rgb_to_u8(&final_color, settings.use_gamma);
            color_buffer[buffer_start_index].store(r, Ordering::Relaxed);
            color_buffer[buffer_start_index + 1].store(g, Ordering::Relaxed);
            color_buffer[buffer_start_index + 2].store(b, Ordering::Relaxed);
        }
    }
}
use crate::geometry::transform::TransformFactory;
use nalgebra::{Matrix4, Point3, Vector3};

/// 投影类型枚举，提供类型安全的投影方式选择
#[derive(Debug, Clone, PartialEq)]
pub enum ProjectionType {
    Perspective {
        fov_y_degrees: f32,
        aspect_ratio: f32,
    },
    Orthographic {
        width: f32,
        height: f32,
    },
}

impl ProjectionType {
    /// 获取宽高比
    pub fn aspect_ratio(&self) -> f32 {
        match self {
            ProjectionType::Perspective { aspect_ratio, .. } => *aspect_ratio,
            ProjectionType::Orthographic { width, height } => width / height,
        }
    }
}

/// 相机参数结构体，包含所有相机配置信息
#[derive(Debug, Clone)]
pub struct CameraParams {
    pub position: Point3<f32>,
    pub target: Point3<f32>,
    pub up: Vector3<f32>,
    pub projection: ProjectionType,
    pub near: f32,
    pub far: f32,
}

impl Default for CameraParams {
    fn default() -> Self {
        Self {
            position: Point3::new(0.0, 0.0, 3.0),
            target: Point3::new(0.0, 0.0, 0.0),
            up: Vector3::new(0.0, 1.0, 0.0),
            projection: ProjectionType::Perspective {
                fov_y_degrees: 45.0,
                aspect_ratio: 1.0,
            },
            near: 0.1,
            far: 100.0,
        }
    }
}

/// 简化的相机类
#[derive(Debug, Clone)]
pub struct Camera {
    pub params: CameraParams,

    // 预计算的矩阵 - 每次创建时计算一次
    view_matrix: Matrix4<f32>,
    projection_matrix: Matrix4<f32>,
}

impl Camera {
    /// 使用参数结构体创建相机
    pub fn new(params: CameraParams) -> Self {
        let mut camera = Camera {
            params,
            view_matrix: Matrix4::identity(),
            projection_matrix: Matrix4::identity(),
        };
        camera.update_matrices();
        camera
    }

    /// 创建透视投影相机的便捷方法
    pub fn perspective(
        position: Point3<f32>,
        target: Point3<f32>,
        up: Vector3<f32>,
        fov_y_degrees: f32,
        aspect_ratio: f32,
        near: f32,
        far: f32,
    ) -> Self {
        let params = CameraParams {
            position,
            target,
            up: up.normalize(),
            projection: ProjectionType::Perspective {
                fov_y_degrees,
                aspect_ratio,
            },
            near,
            far,
        };
        Self::new(params)
    }

    /// 创建正交投影相机的便捷方法
    pub fn orthographic(
        position: Point3<f32>,
        target: Point3<f32>,
        up: Vector3<f32>,
        width: f32,
        height: f32,
        near: f32,
        far: f32,
    ) -> Self {
        let params = CameraParams {
            position,
            target,
            up: up.normalize(),
            projection: ProjectionType::Orthographic { width, height },
            near,
            far,
        };
        Self::new(params)
    }

    // ============ 基本访问方法 ============

    /// 获取相机位置
    pub fn position(&self) -> Point3<f32> {
        self.params.position
    }

    /// 获取宽高比
    pub fn aspect_ratio(&self) -> f32 {
        self.params.projection.aspect_ratio()
    }

    /// 获取远裁剪面
    pub fn far(&self) -> f32 {
        self.params.far
    }

    /// 获取近裁剪面  
    pub fn near(&self) -> f32 {
        self.params.near
    }

    // ============ 矩阵访问方法 ============

    /// 获取视图矩阵
    pub fn view_matrix(&self) -> Matrix4<f32> {
        self.view_matrix
    }

    /// 获取投影矩阵
    pub fn projection_matrix(&self) -> Matrix4<f32> {
        self.projection_matrix
    }

    // ============ 相机运动方法（用于动画）============

    /// 围绕目标点进行任意轴旋转（用于轨道动画）
    pub fn orbit(&mut self, axis: &Vector3<f32>, angle_rad: f32) {
        let camera_to_target = self.params.position - self.params.target;
        let rotation_matrix = TransformFactory::rotation(axis, angle_rad);
        let rotated_vector = rotation_matrix.transform_vector(&camera_to_target);
        self.params.position = self.params.target + rotated_vector;
        self.update_matrices();
    }

    /// 相机沿视线方向移动（保留用于动画）
    pub fn dolly(&mut self, amount: f32) {
        let direction = (self.params.target - self.params.position).normalize();
        let translation = direction * amount;
        self.params.position += translation;
        self.update_matrices();
    }

    // ============ GUI 交互方法 ============

    /// 屏幕拖拽转换为世界坐标平移（GUI专用）
    /// 返回值：是否需要清除地面缓存
    pub fn pan_from_screen_delta(
        &mut self,
        screen_delta: egui::Vec2,
        screen_size: egui::Vec2,
        sensitivity: f32,
    ) -> bool {
        // 计算世界坐标增量
        let distance_to_target = (self.params.position - self.params.target).magnitude();

        let world_scale = match &self.params.projection {
            ProjectionType::Perspective { fov_y_degrees, .. } => {
                let fov_rad = fov_y_degrees.to_radians();
                distance_to_target * (fov_rad / 2.0).tan() * 2.0 / screen_size.y
            }
            ProjectionType::Orthographic { height, .. } => height / screen_size.y,
        };

        // 应用敏感度
        let adjusted_scale = world_scale * sensitivity;
        let world_delta_x = -screen_delta.x * adjusted_scale;
        let world_delta_y = screen_delta.y * adjusted_scale;

        // 计算相机的右向量和上向量
        let forward = (self.params.target - self.params.position).normalize();
        let right = forward.cross(&self.params.up).normalize();
        let up = right.cross(&forward).normalize();

        // 计算世界空间中的平移向量
        let translation = right * world_delta_x + up * world_delta_y;

        // 同时移动相机位置和目标点
        self.params.position += translation;
        self.params.target += translation;

        self.update_matrices();

        // 相机位置变化，需要清除地面缓存
        true
    }

    /// 滚轮缩放转换为相机推拉（GUI专用）
    /// 返回值：是否需要清除地面缓存
    pub fn dolly_from_scroll(&mut self, scroll_delta: f32, sensitivity: f32) -> bool {
        let distance_to_target = (self.params.position - self.params.target).magnitude();

        // 基础敏感度：距离的 10%
        let base_sensitivity = distance_to_target * 0.1;

        // 应用用户敏感度设置
        let dolly_amount = scroll_delta * base_sensitivity * sensitivity;

        // 确保不会推得太近（最小距离 0.1）
        let min_distance = 0.1;
        if distance_to_target - dolly_amount > min_distance {
            self.dolly(dolly_amount);
        } else {
            // 如果会太近，就移动到最小距离位置
            let direction = (self.params.position - self.params.target).normalize();
            self.params.position = self.params.target + direction * min_distance;
            self.update_matrices();
        }

        // 相机位置变化，需要清除地面缓存
        true
    }

    /// 屏幕拖拽转换为轨道旋转（GUI专用）
    /// 返回值：是否需要清除地面缓存
    pub fn orbit_from_screen_delta(&mut self, screen_delta: egui::Vec2, sensitivity: f32) -> bool {
        // 基础旋转敏感度
        let base_rotation_sensitivity = 0.01;
        let adjusted_sensitivity = base_rotation_sensitivity * sensitivity;

        let angle_x = -screen_delta.y * adjusted_sensitivity;
        let angle_y = -screen_delta.x * adjusted_sensitivity;

        // Y轴旋转（水平拖拽）
        if angle_y.abs() > 1e-6 {
            self.orbit(&Vector3::y(), angle_y);
        }

        // X轴旋转（垂直拖拽） - 围绕相机的右向量
        if angle_x.abs() > 1e-6 {
            let forward = (self.params.target - self.params.position).normalize();
            let right = forward.cross(&self.params.up).normalize();

            // 限制垂直旋转角度，避免翻转
            let camera_to_target = self.params.position - self.params.target;
            let current_elevation = camera_to_target
                .y
                .atan2((camera_to_target.x.powi(2) + camera_to_target.z.powi(2)).sqrt());

            // 限制在 -85° 到 85° 之间
            let max_elevation = 85.0_f32.to_radians();
            let new_elevation = current_elevation + angle_x;

            if new_elevation.abs() < max_elevation {
                self.orbit(&right, angle_x);
            }
        }

        // 相机位置变化，需要清除地面缓存
        true
    }

    /// 重置相机到默认视角（GUI专用）
    /// 返回值：是否需要清除地面缓存
    pub fn reset_to_default_view(&mut self) -> bool {
        self.params.position = Point3::new(0.0, 0.0, 3.0);
        self.params.target = Point3::new(0.0, 0.0, 0.0);
        self.params.up = Vector3::new(0.0, 1.0, 0.0);
        self.update_matrices();

        // 相机位置变化，需要清除地面缓存
        true
    }

    /// 聚焦到物体（自动调整距离）（GUI专用）
    /// 返回值：是否需要清除地面缓存
    pub fn focus_on_object(&mut self, object_center: Point3<f32>, object_radius: f32) -> bool {
        // 计算合适的距离（确保物体完全可见）
        let optimal_distance = match &self.params.projection {
            ProjectionType::Perspective { fov_y_degrees, .. } => {
                let fov_rad = fov_y_degrees.to_radians();
                object_radius / (fov_rad / 2.0).tan() * 1.5 // 1.5倍确保有边距
            }
            ProjectionType::Orthographic { .. } => {
                object_radius * 3.0 // 正交投影下的合适距离
            }
        };

        // 保持当前的观察方向，但调整距离
        let current_direction = (self.params.position - self.params.target).normalize();

        self.params.target = object_center;
        self.params.position = object_center + current_direction * optimal_distance;

        self.update_matrices();

        // 相机位置变化，需要清除地面缓存
        true
    }

    // ============ 内部实现方法 ============

    /// 更新所有矩阵（创建时和修改后调用）
    pub fn update_matrices(&mut self) {
        self.update_view_matrix();
        self.update_projection_matrix();
    }

    /// 更新视图矩阵
    fn update_view_matrix(&mut self) {
        self.view_matrix =
            TransformFactory::view(&self.params.position, &self.params.target, &self.params.up);
    }

    /// 更新投影矩阵
    fn update_projection_matrix(&mut self) {
        self.projection_matrix = match &self.params.projection {
            ProjectionType::Perspective {
                fov_y_degrees,
                aspect_ratio,
            } => TransformFactory::perspective(
                *aspect_ratio,
                fov_y_degrees.to_radians(),
                self.params.near,
                self.params.far,
            ),
            ProjectionType::Orthographic { width, height } => TransformFactory::orthographic(
                -width / 2.0,
                width / 2.0,
                -height / 2.0,
                height / 2.0,
                self.params.near,
                self.params.far,
            ),
        };
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self::new(CameraParams::default())
    }
}
use nalgebra::{Point2, Point3};

/// 计算二维三角形面积
///
/// # 参数
/// * `v0`, `v1`, `v2` - 三角形的三个顶点（屏幕坐标）
///
/// # 返回值
/// 三角形面积
#[inline]
pub fn calculate_triangle_area(v0: &Point2<f32>, v1: &Point2<f32>, v2: &Point2<f32>) -> f32 {
    ((v1.x - v0.x) * (v2.y - v0.y) - (v2.x - v0.x) * (v1.y - v0.y)).abs() * 0.5
}

/// 检查面积是否合理
/// # 参数
/// * `v0`, `v1`, `v2` - 三角形的三个顶点（屏幕坐标）
///
/// # 返回值
/// 如果三角形面积大于一个很小的阈值（1e-6），返回true
///
#[inline]
pub fn is_valid_triangle(v0: &Point2<f32>, v1: &Point2<f32>, v2: &Point2<f32>) -> bool {
    let area = calculate_triangle_area(v0, v1, v2);
    area > 1e-6
}

/// 检查三角形是否应该被剔除（面积过小）
///
/// # 参数
/// * `v0`, `v1`, `v2` - 三角形的三个顶点（屏幕坐标）
/// * `min_area` - 最小面积阈值
///
/// # 返回值
/// 如果三角形应被剔除，返回true
#[inline]
pub fn should_cull_small_triangle(
    v0: &Point2<f32>,
    v1: &Point2<f32>,
    v2: &Point2<f32>,
    min_area: f32,
) -> bool {
    calculate_triangle_area(v0, v1, v2) < min_area
}

/// 进行背面剔除判断
///
/// # 参数
/// * `v0`, `v1`, `v2` - 三角形的三个顶点（视图空间坐标）
///
/// # 返回值
/// 如果三角形是背面（应被剔除），返回true
#[inline]
pub fn is_backface(v0: &Point3<f32>, v1: &Point3<f32>, v2: &Point3<f32>) -> bool {
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let face_normal = edge1.cross(&edge2).normalize();
    let view_dir = (v0 - Point3::origin()).normalize();

    // 如果法线与视线方向夹角大于90度，则是背面
    face_normal.dot(&view_dir) > -1e-6
}

/// 检查像素是否在三角形边缘附近（用于线框渲染模式）
///
/// # 参数
/// * `pixel_point` - 像素中心点坐标
/// * `v0`, `v1`, `v2` - 三角形的三个顶点（屏幕坐标）
/// * `edge_threshold` - 边缘检测阈值（像素距离边缘的最大距离）
///
/// # 返回值
/// 如果像素在三角形任意边缘附近，返回true
#[inline]
pub fn is_on_triangle_edge(
    pixel_point: Point2<f32>,
    v0: Point2<f32>,
    v1: Point2<f32>,
    v2: Point2<f32>,
    edge_threshold: f32,
) -> bool {
    // 计算点到线段的距离
    let dist_to_edge = |p: Point2<f32>, edge_start: Point2<f32>, edge_end: Point2<f32>| -> f32 {
        let edge_vec = edge_end - edge_start.coords;
        let edge_length_sq = edge_vec.coords.norm_squared();

        // 如果边长为0，直接返回点到起点的距离
        if edge_length_sq < 1e-6 {
            return (p - edge_start.coords).coords.norm();
        }

        // 计算投影比例
        let t =
            ((p - edge_start.coords).coords.dot(&edge_vec.coords) / edge_length_sq).clamp(0.0, 1.0);

        // 计算投影点
        let projection = Point2::new(edge_start.x + t * edge_vec.x, edge_start.y + t * edge_vec.y);

        // 返回点到投影点的距离
        (p - projection.coords).coords.norm()
    };

    // 检查点到三条边的距离是否小于阈值
    dist_to_edge(pixel_point, v0, v1) <= edge_threshold
        || dist_to_edge(pixel_point, v1, v2) <= edge_threshold
        || dist_to_edge(pixel_point, v2, v0) <= edge_threshold
}
use nalgebra::{Point2, Point3, Vector2, Vector3};
use std::ops::{Add, Mul};

const EPSILON: f32 = 1e-5; // 浮点比较的小值

/// 计算点p相对于三角形(v1, v2, v3)的重心坐标(alpha, beta, gamma)
/// 如果三角形退化则返回None
/// Alpha对应v1, Beta对应v2, Gamma对应v3
pub fn barycentric_coordinates(
    p: Point2<f32>,
    v1: Point2<f32>,
    v2: Point2<f32>,
    v3: Point2<f32>,
) -> Option<Vector3<f32>> {
    let e1 = v2 - v1;
    let e2 = v3 - v1;
    let p_v1 = p - v1;

    // 主三角形面积(乘2)，使用2D叉积行列式
    let total_area_x2 = e1.x * e2.y - e1.y * e2.x;

    if total_area_x2.abs() < EPSILON {
        return None; // 退化三角形
    }

    let inv_total_area_x2 = 1.0 / total_area_x2;

    // 子三角形(p, v3, v1)面积/总面积 -> v2的重心坐标(beta)
    let area2_x2 = p_v1.x * e2.y - p_v1.y * e2.x;
    let beta = area2_x2 * inv_total_area_x2;

    // 子三角形(p, v1, v2)面积/总面积 -> v3的重心坐标(gamma)
    let area3_x2 = e1.x * p_v1.y - e1.y * p_v1.x;
    let gamma = area3_x2 * inv_total_area_x2;

    // v1的重心坐标(alpha)
    let alpha = 1.0 - beta - gamma;

    Some(Vector3::new(alpha, beta, gamma))
}

/// 检查重心坐标是否表示点在三角形内部
#[inline(always)]
pub fn is_inside_triangle(bary: Vector3<f32>) -> bool {
    bary.x >= -EPSILON && bary.y >= -EPSILON && bary.z >= -EPSILON
}

/// 通用的透视校正插值函数，适用于任意可线性组合的类型
#[allow(clippy::too_many_arguments)]
fn perspective_correct_interpolate<T>(
    bary: Vector3<f32>,
    v1: T,
    v2: T,
    v3: T,
    z1_view: f32,
    z2_view: f32,
    z3_view: f32,
    is_perspective: bool,
) -> T
where
    T: Copy + Add<Output = T> + Mul<f32, Output = T>,
{
    if !is_perspective {
        // 正交投影：线性插值
        v1 * bary.x + v2 * bary.y + v3 * bary.z
    } else {
        // 透视投影：插值 attribute/z
        let inv_z1 = if z1_view.abs() > EPSILON {
            1.0 / z1_view
        } else {
            0.0
        };
        let inv_z2 = if z2_view.abs() > EPSILON {
            1.0 / z2_view
        } else {
            0.0
        };
        let inv_z3 = if z3_view.abs() > EPSILON {
            1.0 / z3_view
        } else {
            0.0
        };

        let interpolated_inv_z = bary.x * inv_z1 + bary.y * inv_z2 + bary.z * inv_z3;

        if interpolated_inv_z.abs() > EPSILON {
            let inv_z = 1.0 / interpolated_inv_z;
            // 插值 attr/z 并乘以插值后的 z
            (v1 * (bary.x * inv_z1) + v2 * (bary.y * inv_z2) + v3 * (bary.z * inv_z3)) * inv_z
        } else {
            // 回退到线性插值
            v1 * bary.x + v2 * bary.y + v3 * bary.z
        }
    }
}

/// 使用重心坐标插值深度(z)，带透视校正
/// 采用视空间Z值(通常为负值)
/// 返回正值深度用于缓冲区比较，无效则返回f32::INFINITY
pub fn interpolate_depth(
    bary: Vector3<f32>,
    z1_view: f32,
    z2_view: f32,
    z3_view: f32,
    is_perspective: bool,
) -> f32 {
    if !is_inside_triangle(bary) {
        return f32::INFINITY;
    }

    let interpolated_z = perspective_correct_interpolate(
        bary,
        z1_view,
        z2_view,
        z3_view,
        z1_view,
        z2_view,
        z3_view,
        is_perspective,
    );

    // 返回正值深度用于缓冲区(较小值表示更近)
    if interpolated_z > -EPSILON {
        // 检查是否在近平面后方或非常接近
        f32::INFINITY
    } else {
        -interpolated_z
    }
}

/// 使用重心坐标插值纹理坐标(UV)，带透视校正
/// 采用视空间Z值进行校正
#[allow(clippy::too_many_arguments)]
pub fn interpolate_texcoords(
    bary: Vector3<f32>,
    tc1: Vector2<f32>,
    tc2: Vector2<f32>,
    tc3: Vector2<f32>,
    z1_view: f32,
    z2_view: f32,
    z3_view: f32,
    is_perspective: bool,
) -> Vector2<f32> {
    perspective_correct_interpolate(
        bary,
        tc1,
        tc2,
        tc3,
        z1_view,
        z2_view,
        z3_view,
        is_perspective,
    )
}

/// 使用重心坐标插值法线向量，带透视校正
/// 采用视空间Z值进行校正
#[allow(clippy::too_many_arguments)]
pub fn interpolate_normal(
    bary: Vector3<f32>,
    n1: Vector3<f32>,
    n2: Vector3<f32>,
    n3: Vector3<f32>,
    is_perspective: bool,
    z1_view: f32,
    z2_view: f32,
    z3_view: f32,
) -> Vector3<f32> {
    let result = perspective_correct_interpolate(
        bary,
        n1,
        n2,
        n3,
        z1_view,
        z2_view,
        z3_view,
        is_perspective,
    );

    // 归一化结果
    if result.norm_squared() > EPSILON {
        result.normalize()
    } else {
        Vector3::z() // 使用默认Z方向作为备用
    }
}

/// 使用重心坐标插值视空间位置，带透视校正
/// 采用视空间Z值进行校正
#[allow(clippy::too_many_arguments)]
pub fn interpolate_position(
    bary: Vector3<f32>,
    p1: Point3<f32>,
    p2: Point3<f32>,
    p3: Point3<f32>,
    is_perspective: bool,
    z1_view: f32,
    z2_view: f32,
    z3_view: f32,
) -> Point3<f32> {
    // 通过向量计算插值，然后转回点
    let coords1 = p1.coords;
    let coords2 = p2.coords;
    let coords3 = p3.coords;

    let result = perspective_correct_interpolate(
        bary,
        coords1,
        coords2,
        coords3,
        z1_view,
        z2_view,
        z3_view,
        is_perspective,
    );

    Point3::from(result)
}
pub mod camera;
pub mod culling;
pub mod interpolation;
pub mod transform;
use log::warn;
use nalgebra::{Matrix3, Matrix4, Point2, Point3, Vector3, Vector4};

//=================================
// 变换矩阵创建工厂 (手动实现)
//=================================

/// 变换矩阵工厂，提供创建各种变换矩阵的静态方法
pub struct TransformFactory;

#[rustfmt::skip]
impl TransformFactory {
    /// 创建绕任意轴旋转的变换矩阵 (使用 Rodrigues' rotation formula)
    pub fn rotation(axis: &Vector3<f32>, angle_rad: f32) -> Matrix4<f32> {
        let axis_unit = axis.normalize();
        let x = axis_unit.x;
        let y = axis_unit.y;
        let z = axis_unit.z;
        let c = angle_rad.cos();
        let s = angle_rad.sin();
        let t = 1.0 - c;

        Matrix4::new(
            t * x * x + c,     t * x * y - z * s, t * x * z + y * s, 0.0,
            t * x * y + z * s, t * y * y + c,     t * y * z - x * s, 0.0,
            t * x * z - y * s, t * y * z + x * s, t * z * z + c,     0.0,
            0.0,               0.0,               0.0,               1.0,
        )
    }

    /// 创建绕X轴旋转的变换矩阵
    pub fn rotation_x(angle_rad: f32) -> Matrix4<f32> {
        let c = angle_rad.cos();
        let s = angle_rad.sin();
        Matrix4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, c,  -s,   0.0,
            0.0, s,   c,   0.0,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    /// 创建绕Y轴旋转的变换矩阵
    pub fn rotation_y(angle_rad: f32) -> Matrix4<f32> {
        let c = angle_rad.cos();
        let s = angle_rad.sin();
        Matrix4::new(
            c,   0.0, s,   0.0,
            0.0, 1.0, 0.0, 0.0,
           -s,   0.0, c,   0.0,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    /// 创建绕Z轴旋转的变换矩阵
    pub fn rotation_z(angle_rad: f32) -> Matrix4<f32> {
        let c = angle_rad.cos();
        let s = angle_rad.sin();
        Matrix4::new(
            c,  -s,   0.0, 0.0,
            s,   c,   0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    /// 创建平移矩阵
    pub fn translation(translation: &Vector3<f32>) -> Matrix4<f32> {
        Matrix4::new(
            1.0, 0.0, 0.0, translation.x,
            0.0, 1.0, 0.0, translation.y,
            0.0, 0.0, 1.0, translation.z,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    /// 创建非均匀缩放矩阵
    pub fn scaling_nonuniform(scale: &Vector3<f32>) -> Matrix4<f32> {
        Matrix4::new(
            scale.x, 0.0,     0.0,     0.0,
            0.0,     scale.y, 0.0,     0.0,
            0.0,     0.0,     scale.z, 0.0,
            0.0,     0.0,     0.0,     1.0,
        )
    }

    /// 创建视图矩阵 (Look-At, Right-Handed)
    pub fn view(eye: &Point3<f32>, target: &Point3<f32>, up: &Vector3<f32>) -> Matrix4<f32> {
        let z_axis = (eye - target).normalize(); // 在右手坐标系中，摄像机看向自己的-Z方向
        let x_axis = up.cross(&z_axis).normalize();
        let y_axis = z_axis.cross(&x_axis);

        // 创建从世界空间到视图空间的旋转矩阵
        let rotation = Matrix4::new(
            x_axis.x, x_axis.y, x_axis.z, 0.0,
            y_axis.x, y_axis.y, y_axis.z, 0.0,
            z_axis.x, z_axis.y, z_axis.z, 0.0,
            0.0,      0.0,      0.0,      1.0,
        );

        // 创建平移矩阵，将摄像机位置移到原点
        let translation = Self::translation(&-eye.coords);

        // 视图矩阵 = 旋转矩阵 * 平移矩阵
        rotation * translation
    }

    /// 创建透视投影矩阵 (Right-Handed)
    pub fn perspective(aspect_ratio: f32, fov_y_rad: f32, near: f32, far: f32) -> Matrix4<f32> {
        let f = 1.0 / (fov_y_rad / 2.0).tan();
        let nf = 1.0 / (near - far);

        Matrix4::new(
            f / aspect_ratio, 0.0, 0.0,                          0.0,
            0.0,              f,   0.0,                          0.0,
            0.0,              0.0, (far + near) * nf,            2.0 * far * near * nf,
            0.0,              0.0, -1.0,                         0.0,
        )
    }

    /// 创建正交投影矩阵 (Right-Handed)
    pub fn orthographic(
        left: f32,
        right: f32,
        bottom: f32,
        top: f32,
        near: f32,
        far: f32,
    ) -> Matrix4<f32> {
        let rl = 1.0 / (right - left);
        let tb = 1.0 / (top - bottom);
        let nf = 1.0 / (near - far);

        Matrix4::new(
            2.0 * rl,      0.0,           0.0,          -(right + left) * rl,
            0.0,           2.0 * tb,      0.0,          -(top + bottom) * tb,
            0.0,           0.0,           2.0 * nf,     (far + near) * nf, // 注意这里是 `(far + near) * nf`
            0.0,           0.0,           0.0,          1.0,
        )
    }

    /// 创建MVP矩阵（Model-View-Projection）
    pub fn model_view_projection(
        model: &Matrix4<f32>,
        view: &Matrix4<f32>,
        projection: &Matrix4<f32>,
    ) -> Matrix4<f32> {
        projection * view * model
    }

    /// 创建MV矩阵（Model-View）
    pub fn model_view(model: &Matrix4<f32>, view: &Matrix4<f32>) -> Matrix4<f32> {
        view * model
    }
}

//=================================
// 核心变换函数
//=================================

/// 计算法线变换矩阵（模型-视图矩阵的逆转置）
///
/// 法线向量需要特殊处理：使用变换矩阵的逆转置来保持垂直性
#[inline]
pub fn compute_normal_matrix(model_view_matrix: &Matrix4<f32>) -> Matrix3<f32> {
    model_view_matrix.try_inverse().map_or_else(
        || {
            warn!("模型-视图矩阵不可逆，使用单位矩阵代替法线矩阵");
            Matrix3::identity()
        },
        |inv| inv.transpose().fixed_view::<3, 3>(0, 0).into_owned(),
    )
}

/// 3D点变换：将点从一个坐标空间变换到另一个坐标空间
///
/// 使用齐次坐标进行变换，最后执行齐次除法得到3D点
#[inline]
pub fn transform_point(point: &Point3<f32>, matrix: &Matrix4<f32>) -> Point3<f32> {
    let homogeneous_point = point.to_homogeneous();
    let transformed_homogeneous = matrix * homogeneous_point;

    // 执行齐次除法，处理w分量
    if transformed_homogeneous.w.abs() < 1e-9 {
        Point3::new(
            transformed_homogeneous.x,
            transformed_homogeneous.y,
            transformed_homogeneous.z,
        )
    } else {
        Point3::from(transformed_homogeneous.xyz() / transformed_homogeneous.w)
    }
}

/// 法线向量变换：使用法线矩阵变换法线向量并归一化
#[inline]
pub fn transform_normal(normal: &Vector3<f32>, normal_matrix: &Matrix3<f32>) -> Vector3<f32> {
    (normal_matrix * normal).normalize()
}

/// 透视除法：将裁剪空间坐标转换为NDC坐标
///
/// 裁剪空间 → NDC（标准化设备坐标）：除以w分量
#[inline]
pub fn apply_perspective_division(clip: &Vector4<f32>) -> Point3<f32> {
    let w = clip.w;
    if w.abs() > 1e-6 {
        Point3::new(clip.x / w, clip.y / w, clip.z / w)
    } else {
        Point3::origin() // 避免除以零
    }
}

/// NDC到屏幕坐标转换（视口变换）
///
/// NDC范围[-1,1] → 屏幕像素坐标[0,width/height]
/// 注意Y轴翻转：NDC的+Y向上，屏幕坐标的+Y向下
#[inline]
pub fn ndc_to_screen(ndc_x: f32, ndc_y: f32, width: f32, height: f32) -> Point2<f32> {
    Point2::new(
        (ndc_x + 1.0) * 0.5 * width,
        (1.0 - (ndc_y + 1.0) * 0.5) * height, // 使用更标准的NDC -> Screen映射
    )
}

/// 裁剪空间到屏幕坐标的完整转换
///
/// 组合透视除法和视口变换：裁剪空间 → NDC → 屏幕坐标
#[inline]
pub fn clip_to_screen(clip: &Vector4<f32>, width: f32, height: f32) -> Point2<f32> {
    let ndc = apply_perspective_division(clip);
    ndc_to_screen(ndc.x, ndc.y, width, height)
}

/// 点到裁剪坐标的转换
///
/// 将3D点转换为齐次裁剪坐标（用于后续透视除法）
#[inline]
pub fn point_to_clip(point: &Point3<f32>, matrix: &Matrix4<f32>) -> Vector4<f32> {
    matrix * point.to_homogeneous()
}
use crate::io::render_settings::{
    AnimationType, RenderSettings, RotationAxis, parse_point3, parse_vec3,
};
use crate::material_system::light::Light;
use log::warn;
use std::path::Path;
use toml::Value;

/// TOML配置管理器
pub struct TomlConfigLoader;

impl TomlConfigLoader {
    /// 从TOML文件加载完整配置
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<RenderSettings, String> {
        let content =
            std::fs::read_to_string(path.as_ref()).map_err(|e| format!("读取配置文件失败: {e}"))?;

        Self::load_from_content(&content)
    }

    /// 从TOML内容字符串加载配置
    pub fn load_from_content(content: &str) -> Result<RenderSettings, String> {
        let toml_value: Value =
            toml::from_str(content).map_err(|e| format!("解析TOML失败: {e}"))?;

        Self::parse_toml_to_settings(toml_value)
    }

    /// 保存配置到TOML文件
    pub fn save_to_file<P: AsRef<Path>>(settings: &RenderSettings, path: P) -> Result<(), String> {
        let toml_content = Self::settings_to_toml(settings)?;
        std::fs::write(path, toml_content).map_err(|e| format!("写入配置文件失败: {e}"))
    }

    /// 直接生成示例配置文件
    pub fn create_example_config<P: AsRef<Path>>(path: P) -> Result<(), String> {
        let settings = RenderSettings {
            obj: Some("obj/simple/bunny.obj".to_string()),
            texture: None,
            background_image_path: None,
            ..Default::default()
        };

        // 保存配置
        Self::save_to_file(&settings, path).map_err(|e| format!("创建示例配置失败: {e}"))
    }

    // ===== TOML -> RenderSettings 转换 =====

    fn parse_toml_to_settings(toml: Value) -> Result<RenderSettings, String> {
        let mut settings = RenderSettings::default();

        // [files] 部分
        if let Some(files) = toml.get("files").and_then(|v| v.as_table()) {
            Self::parse_files_section(&mut settings, files)?;
        }

        // [render] 部分
        if let Some(render) = toml.get("render").and_then(|v| v.as_table()) {
            Self::parse_render_section(&mut settings, render)?;
        }

        // [camera] 部分
        if let Some(camera) = toml.get("camera").and_then(|v| v.as_table()) {
            Self::parse_camera_section(&mut settings, camera)?;
        }

        // [object] 部分
        if let Some(object) = toml.get("object").and_then(|v| v.as_table()) {
            Self::parse_object_section(&mut settings, object)?;
        }

        // [lighting] 部分
        if let Some(lighting) = toml.get("lighting").and_then(|v| v.as_table()) {
            Self::parse_lighting_section(&mut settings, lighting)?;
        }

        // [[light]] 数组 - 多光源支持
        settings.lights = Self::parse_lights_array(&toml)?;

        // [material] 部分
        if let Some(material) = toml.get("material").and_then(|v| v.as_table()) {
            Self::parse_material_section(&mut settings, material)?;
        }

        // [background] 部分
        if let Some(background) = toml.get("background").and_then(|v| v.as_table()) {
            Self::parse_background_section(&mut settings, background)?;
        }

        // [animation] 部分
        if let Some(animation) = toml.get("animation").and_then(|v| v.as_table()) {
            Self::parse_animation_section(&mut settings, animation)?;
        }

        // [shadow] 部分
        if let Some(shadow) = toml.get("shadow").and_then(|v| v.as_table()) {
            Self::parse_shadow_section(&mut settings, shadow)?;
        }

        Ok(settings)
    }

    // ===== 各个section的解析方法 =====

    fn parse_files_section(
        settings: &mut RenderSettings,
        files: &toml::Table,
    ) -> Result<(), String> {
        if let Some(obj) = files.get("obj").and_then(|v| v.as_str()) {
            settings.obj = Some(obj.to_string());
        }
        if let Some(output) = files.get("output").and_then(|v| v.as_str()) {
            settings.output = output.to_string();
        }
        if let Some(output_dir) = files.get("output_dir").and_then(|v| v.as_str()) {
            settings.output_dir = output_dir.to_string();
        }
        if let Some(texture) = files.get("texture").and_then(|v| v.as_str()) {
            settings.texture = Some(texture.to_string());
        }
        if let Some(bg_image) = files.get("background_image").and_then(|v| v.as_str()) {
            settings.background_image_path = Some(bg_image.to_string());
        }
        Ok(())
    }

    fn parse_render_section(
        settings: &mut RenderSettings,
        render: &toml::Table,
    ) -> Result<(), String> {
        if let Some(width) = render.get("width").and_then(|v| v.as_integer()) {
            settings.width = width as usize;
        }
        if let Some(height) = render.get("height").and_then(|v| v.as_integer()) {
            settings.height = height as usize;
        }
        if let Some(projection) = render.get("projection").and_then(|v| v.as_str()) {
            settings.projection = projection.to_string();
        }
        if let Some(use_zbuffer) = render.get("use_zbuffer").and_then(|v| v.as_bool()) {
            settings.use_zbuffer = use_zbuffer;
        }
        if let Some(colorize) = render.get("colorize").and_then(|v| v.as_bool()) {
            settings.colorize = colorize;
        }
        if let Some(use_texture) = render.get("use_texture").and_then(|v| v.as_bool()) {
            settings.use_texture = use_texture;
        }
        if let Some(use_gamma) = render.get("use_gamma").and_then(|v| v.as_bool()) {
            settings.use_gamma = use_gamma;
        }
        if let Some(enable_aces) = render.get("enable_aces").and_then(|v| v.as_bool()) {
            settings.enable_aces = enable_aces;
        }
        if let Some(backface_culling) = render.get("backface_culling").and_then(|v| v.as_bool()) {
            settings.backface_culling = backface_culling;
        }
        if let Some(wireframe) = render.get("wireframe").and_then(|v| v.as_bool()) {
            settings.wireframe = wireframe;
        }
        if let Some(cull_small_triangles) =
            render.get("cull_small_triangles").and_then(|v| v.as_bool())
        {
            settings.cull_small_triangles = cull_small_triangles;
        }
        if let Some(min_triangle_area) = render.get("min_triangle_area").and_then(|v| v.as_float())
        {
            settings.min_triangle_area = min_triangle_area as f32;
        }
        if let Some(save_depth) = render.get("save_depth").and_then(|v| v.as_bool()) {
            settings.save_depth = save_depth;
        }
        Ok(())
    }

    fn parse_camera_section(
        settings: &mut RenderSettings,
        camera: &toml::Table,
    ) -> Result<(), String> {
        if let Some(from) = camera.get("from").and_then(|v| v.as_str()) {
            settings.camera_from = from.to_string();
        }
        if let Some(at) = camera.get("at").and_then(|v| v.as_str()) {
            settings.camera_at = at.to_string();
        }
        if let Some(up) = camera.get("up").and_then(|v| v.as_str()) {
            settings.camera_up = up.to_string();
        }
        if let Some(fov) = camera.get("fov").and_then(|v| v.as_float()) {
            settings.camera_fov = fov as f32;
        }
        Ok(())
    }

    fn parse_object_section(
        settings: &mut RenderSettings,
        object: &toml::Table,
    ) -> Result<(), String> {
        if let Some(position) = object.get("position").and_then(|v| v.as_str()) {
            settings.object_position = position.to_string();
        }
        if let Some(rotation) = object.get("rotation").and_then(|v| v.as_str()) {
            settings.object_rotation = rotation.to_string();
        }
        if let Some(scale_xyz) = object.get("scale_xyz").and_then(|v| v.as_str()) {
            settings.object_scale_xyz = scale_xyz.to_string();
        }
        if let Some(scale) = object.get("scale").and_then(|v| v.as_float()) {
            settings.object_scale = scale as f32;
        }
        Ok(())
    }

    fn parse_lighting_section(
        settings: &mut RenderSettings,
        lighting: &toml::Table,
    ) -> Result<(), String> {
        if let Some(use_lighting) = lighting.get("use_lighting").and_then(|v| v.as_bool()) {
            settings.use_lighting = use_lighting;
        }
        if let Some(ambient) = lighting.get("ambient").and_then(|v| v.as_float()) {
            settings.ambient = ambient as f32;
        }
        if let Some(ambient_color) = lighting.get("ambient_color").and_then(|v| v.as_str()) {
            settings.ambient_color = ambient_color.to_string();
        }
        Ok(())
    }

    /// 多光源解析 - 支持 [[light]] 数组语法
    fn parse_lights_array(toml: &Value) -> Result<Vec<Light>, String> {
        let mut lights = Vec::new();

        if let Some(lights_array) = toml.get("light").and_then(|v| v.as_array()) {
            for (i, light_value) in lights_array.iter().enumerate() {
                if let Some(light_table) = light_value.as_table() {
                    let light = Self::parse_single_light(light_table)
                        .map_err(|e| format!("第{}个光源解析失败: {}", i + 1, e))?;
                    lights.push(light);
                }
            }
        }

        Ok(lights)
    }

    fn parse_single_light(light_table: &toml::Table) -> Result<Light, String> {
        let light_type = light_table
            .get("type")
            .and_then(|v| v.as_str())
            .ok_or("光源缺少type字段")?;

        let enabled = light_table
            .get("enabled")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let intensity = light_table
            .get("intensity")
            .and_then(|v| v.as_float())
            .unwrap_or(1.0) as f32;

        let color_str = light_table
            .get("color")
            .and_then(|v| v.as_str())
            .unwrap_or("1,1,1");

        let color_vec = parse_vec3(color_str).map_err(|e| format!("解析光源颜色失败: {e}"))?;

        match light_type {
            "directional" => {
                let direction_str = light_table
                    .get("direction")
                    .and_then(|v| v.as_str())
                    .ok_or("方向光缺少direction字段")?;

                let direction_vec =
                    parse_vec3(direction_str).map_err(|e| format!("解析方向光方向失败: {e}"))?;

                let mut light = Light::directional(direction_vec, color_vec, intensity);
                if let Light::Directional {
                    enabled: ref mut light_enabled,
                    ..
                } = light
                {
                    *light_enabled = enabled;
                }
                Ok(light)
            }
            "point" => {
                let position_str = light_table
                    .get("position")
                    .and_then(|v| v.as_str())
                    .ok_or("点光源缺少position字段")?;

                let position_point =
                    parse_point3(position_str).map_err(|e| format!("解析点光源位置失败: {e}"))?;

                let constant = light_table
                    .get("constant_attenuation")
                    .and_then(|v| v.as_float())
                    .unwrap_or(1.0) as f32;
                let linear = light_table
                    .get("linear_attenuation")
                    .and_then(|v| v.as_float())
                    .unwrap_or(0.09) as f32;
                let quadratic = light_table
                    .get("quadratic_attenuation")
                    .and_then(|v| v.as_float())
                    .unwrap_or(0.032) as f32;

                let mut light = Light::point(
                    position_point,
                    color_vec,
                    intensity,
                    Some((constant, linear, quadratic)),
                );
                if let Light::Point {
                    enabled: ref mut light_enabled,
                    ..
                } = light
                {
                    *light_enabled = enabled;
                }
                Ok(light)
            }
            _ => Err(format!("未知的光源类型: {light_type}")),
        }
    }

    fn parse_material_section(
        settings: &mut RenderSettings,
        material: &toml::Table,
    ) -> Result<(), String> {
        if let Some(use_phong) = material.get("use_phong").and_then(|v| v.as_bool()) {
            settings.use_phong = use_phong;
        }
        if let Some(use_pbr) = material.get("use_pbr").and_then(|v| v.as_bool()) {
            settings.use_pbr = use_pbr;
        }
        if let Some(diffuse_color) = material.get("diffuse_color").and_then(|v| v.as_str()) {
            settings.diffuse_color = diffuse_color.to_string();
        }
        if let Some(alpha) = material.get("alpha").and_then(|v| v.as_float()) {
            settings.alpha = alpha as f32;
        }

        if let Some(diffuse_intensity) =
            material.get("diffuse_intensity").and_then(|v| v.as_float())
        {
            settings.diffuse_intensity = diffuse_intensity as f32;
        }

        if let Some(specular_color) = material.get("specular_color").and_then(|v| v.as_str()) {
            settings.specular_color = specular_color.to_string();
        }
        if let Some(specular_intensity) = material
            .get("specular_intensity")
            .and_then(|v| v.as_float())
        {
            settings.specular_intensity = specular_intensity as f32;
        }

        if let Some(shininess) = material.get("shininess").and_then(|v| v.as_float()) {
            settings.shininess = shininess as f32;
        }
        if let Some(base_color) = material.get("base_color").and_then(|v| v.as_str()) {
            settings.base_color = base_color.to_string();
        }
        if let Some(metallic) = material.get("metallic").and_then(|v| v.as_float()) {
            settings.metallic = metallic as f32;
        }
        if let Some(roughness) = material.get("roughness").and_then(|v| v.as_float()) {
            settings.roughness = roughness as f32;
        }
        if let Some(ambient_occlusion) =
            material.get("ambient_occlusion").and_then(|v| v.as_float())
        {
            settings.ambient_occlusion = ambient_occlusion as f32;
        }
        if let Some(emissive) = material.get("emissive").and_then(|v| v.as_str()) {
            settings.emissive = emissive.to_string();
        }
        Ok(())
    }

    fn parse_background_section(
        settings: &mut RenderSettings,
        background: &toml::Table,
    ) -> Result<(), String> {
        if let Some(use_background_image) = background
            .get("use_background_image")
            .and_then(|v| v.as_bool())
        {
            settings.use_background_image = use_background_image;
        }
        if let Some(enable_gradient_background) = background
            .get("enable_gradient_background")
            .and_then(|v| v.as_bool())
        {
            settings.enable_gradient_background = enable_gradient_background;
        }
        if let Some(gradient_top_color) = background
            .get("gradient_top_color")
            .and_then(|v| v.as_str())
        {
            settings.gradient_top_color = gradient_top_color.to_string();
        }
        if let Some(gradient_bottom_color) = background
            .get("gradient_bottom_color")
            .and_then(|v| v.as_str())
        {
            settings.gradient_bottom_color = gradient_bottom_color.to_string();
        }
        if let Some(enable_ground_plane) = background
            .get("enable_ground_plane")
            .and_then(|v| v.as_bool())
        {
            settings.enable_ground_plane = enable_ground_plane;
        }
        if let Some(ground_plane_color) = background
            .get("ground_plane_color")
            .and_then(|v| v.as_str())
        {
            settings.ground_plane_color = ground_plane_color.to_string();
        }
        if let Some(ground_plane_height) = background
            .get("ground_plane_height")
            .and_then(|v| v.as_float())
        {
            settings.ground_plane_height = ground_plane_height as f32;
        }
        Ok(())
    }

    fn parse_animation_section(
        settings: &mut RenderSettings,
        animation: &toml::Table,
    ) -> Result<(), String> {
        if let Some(animate) = animation.get("animate").and_then(|v| v.as_bool()) {
            settings.animate = animate;
        }
        if let Some(fps) = animation.get("fps").and_then(|v| v.as_integer()) {
            settings.fps = fps as usize;
        }
        if let Some(rotation_speed) = animation.get("rotation_speed").and_then(|v| v.as_float()) {
            settings.rotation_speed = rotation_speed as f32;
        }
        if let Some(rotation_cycles) = animation.get("rotation_cycles").and_then(|v| v.as_float()) {
            settings.rotation_cycles = rotation_cycles as f32;
        }
        if let Some(animation_type) = animation.get("animation_type").and_then(|v| v.as_str()) {
            settings.animation_type = match animation_type {
                "CameraOrbit" => AnimationType::CameraOrbit,
                "ObjectLocalRotation" => AnimationType::ObjectLocalRotation,
                "None" => AnimationType::None,
                _ => return Err(format!("未知的动画类型: {animation_type}")),
            };
        }
        if let Some(rotation_axis) = animation.get("rotation_axis").and_then(|v| v.as_str()) {
            settings.rotation_axis = match rotation_axis {
                "X" => RotationAxis::X,
                "Y" => RotationAxis::Y,
                "Z" => RotationAxis::Z,
                "Custom" => RotationAxis::Custom,
                _ => return Err(format!("未知的旋转轴: {rotation_axis}")),
            };
        }
        if let Some(custom_rotation_axis) = animation
            .get("custom_rotation_axis")
            .and_then(|v| v.as_str())
        {
            settings.custom_rotation_axis = custom_rotation_axis.to_string();
        }
        Ok(())
    }

    /// 阴影配置解析
    fn parse_shadow_section(
        settings: &mut RenderSettings,
        shadow: &toml::Table,
    ) -> Result<(), String> {
        // === 阴影映射 ===
        if let Some(enable_shadow_mapping) = shadow
            .get("enable_shadow_mapping")
            .and_then(|v| v.as_bool())
        {
            settings.enable_shadow_mapping = enable_shadow_mapping;
        }
        if let Some(shadow_map_size) = shadow.get("shadow_map_size").and_then(|v| v.as_integer()) {
            let size = shadow_map_size as usize;
            if (64..=4096).contains(&size) && size.is_power_of_two() {
                settings.shadow_map_size = size;
            } else {
                warn!("无效的阴影贴图尺寸 {size}, 必须是64-4096之间的2的幂，使用默认值256");
            }
        }
        if let Some(shadow_bias) = shadow.get("shadow_bias").and_then(|v| v.as_float()) {
            settings.shadow_bias = (shadow_bias as f32).clamp(0.0001, 0.1);
        }
        if let Some(shadow_distance) = shadow.get("shadow_distance").and_then(|v| v.as_float()) {
            settings.shadow_distance = (shadow_distance as f32).clamp(1.0, 100.0);
        }
        if let Some(enable_pcf) = shadow.get("enable_pcf").and_then(|v| v.as_bool()) {
            settings.enable_pcf = enable_pcf;
        }
        if let Some(pcf_type) = shadow.get("pcf_type").and_then(|v| v.as_str()) {
            settings.pcf_type = pcf_type.to_string();
        }
        if let Some(pcf_kernel) = shadow.get("pcf_kernel").and_then(|v| v.as_integer()) {
            settings.pcf_kernel = pcf_kernel as usize;
        }
        if let Some(pcf_sigma) = shadow.get("pcf_sigma").and_then(|v| v.as_float()) {
            settings.pcf_sigma = pcf_sigma as f32;
        }
        Ok(())
    }

    // ===== RenderSettings -> TOML 转换 =====

    fn settings_to_toml(settings: &RenderSettings) -> Result<String, String> {
        let mut content = String::new();

        // 文件头注释
        content.push_str("# 光栅化渲染器配置文件\n");
        content.push_str("# 由 GUI 界面生成并保存的配置\n\n");

        // [files] 部分
        content.push_str("[files]\n");
        if let Some(obj) = &settings.obj {
            content.push_str(&format!("obj = \"{obj}\"\n"));
        } else {
            content.push_str("# obj = \"path/to/your/model.obj\"  # OBJ文件路径\n");
        }
        content.push_str(&format!("output = \"{}\"\n", settings.output));
        content.push_str(&format!("output_dir = \"{}\"\n", settings.output_dir));
        if let Some(texture) = &settings.texture {
            content.push_str(&format!("texture = \"{texture}\"\n"));
        } else {
            content.push_str("# texture = \"path/to/texture.jpg\"  # 可选：覆盖MTL纹理\n");
        }
        if let Some(bg_image) = &settings.background_image_path {
            content.push_str(&format!("background_image = \"{bg_image}\"\n"));
        } else {
            content.push_str("# background_image = \"path/to/background.jpg\"  # 可选：背景图片\n");
        }
        content.push('\n');

        // [render] 部分
        content.push_str("[render]\n");
        content.push_str(&format!("width = {}\n", settings.width));
        content.push_str(&format!("height = {}\n", settings.height));
        content.push_str(&format!("projection = \"{}\"\n", settings.projection));
        content.push_str(&format!("use_zbuffer = {}\n", settings.use_zbuffer));
        content.push_str(&format!("colorize = {}\n", settings.colorize));
        content.push_str(&format!("use_texture = {}\n", settings.use_texture));
        content.push_str(&format!("use_gamma = {}\n", settings.use_gamma));
        content.push_str(&format!(
            "backface_culling = {}\n",
            settings.backface_culling
        ));
        content.push_str(&format!("enable_aces = {}\n", settings.enable_aces));
        content.push_str(&format!("wireframe = {}\n", settings.wireframe));
        content.push_str(&format!(
            "cull_small_triangles = {}\n",
            settings.cull_small_triangles
        ));
        content.push_str(&format!(
            "min_triangle_area = {}\n",
            settings.min_triangle_area
        ));
        content.push_str(&format!("save_depth = {}\n", settings.save_depth));
        content.push('\n');

        // [camera] 部分
        content.push_str("[camera]\n");
        content.push_str(&format!("from = \"{}\"\n", settings.camera_from));
        content.push_str(&format!("at = \"{}\"\n", settings.camera_at));
        content.push_str(&format!("up = \"{}\"\n", settings.camera_up));
        content.push_str(&format!("fov = {}\n", settings.camera_fov));
        content.push('\n');

        // [object] 部分
        content.push_str("[object]\n");
        content.push_str(&format!("position = \"{}\"\n", settings.object_position));
        content.push_str(&format!("rotation = \"{}\"\n", settings.object_rotation));
        content.push_str(&format!("scale_xyz = \"{}\"\n", settings.object_scale_xyz));
        content.push_str(&format!("scale = {}\n", settings.object_scale));
        content.push('\n');

        // [lighting] 部分
        content.push_str("[lighting]\n");
        content.push_str(&format!("use_lighting = {}\n", settings.use_lighting));
        content.push_str(&format!("ambient = {}\n", settings.ambient));
        content.push_str(&format!("ambient_color = \"{}\"\n", settings.ambient_color));
        content.push('\n');

        // [[light]] 数组
        if !settings.lights.is_empty() {
            content.push_str("# 光源配置\n");
            for light in &settings.lights {
                content.push_str("[[light]]\n");
                match light {
                    Light::Directional {
                        enabled,
                        direction_str,
                        color_str,
                        intensity,
                        ..
                    } => {
                        content.push_str("type = \"directional\"\n");
                        content.push_str(&format!("enabled = {enabled}\n"));
                        content.push_str(&format!("direction = \"{direction_str}\"\n"));
                        content.push_str(&format!("color = \"{color_str}\"\n"));
                        content.push_str(&format!("intensity = {intensity}\n"));
                    }
                    Light::Point {
                        enabled,
                        position_str,
                        color_str,
                        intensity,
                        constant_attenuation,
                        linear_attenuation,
                        quadratic_attenuation,
                        ..
                    } => {
                        content.push_str("type = \"point\"\n");
                        content.push_str(&format!("enabled = {enabled}\n"));
                        content.push_str(&format!("position = \"{position_str}\"\n"));
                        content.push_str(&format!("color = \"{color_str}\"\n"));
                        content.push_str(&format!("intensity = {intensity}\n"));
                        content
                            .push_str(&format!("constant_attenuation = {constant_attenuation}\n"));
                        content.push_str(&format!("linear_attenuation = {linear_attenuation}\n"));
                        content.push_str(&format!(
                            "quadratic_attenuation = {quadratic_attenuation}\n"
                        ));
                    }
                }
                content.push('\n');
            }
        }

        // [material] 部分
        content.push_str("[material]\n");
        content.push_str(&format!("use_phong = {}\n", settings.use_phong));
        content.push_str(&format!("use_pbr = {}\n", settings.use_pbr));
        content.push_str(&format!("diffuse_color = \"{}\"\n", settings.diffuse_color));
        content.push_str(&format!(
            "diffuse_intensity = {}\n",
            settings.diffuse_intensity
        ));
        content.push_str(&format!("alpha = {}\n", settings.alpha));
        content.push_str(&format!(
            "specular_color = \"{}\"\n",
            settings.specular_color
        ));
        content.push_str(&format!(
            "specular_intensity = {}\n",
            settings.specular_intensity
        ));
        content.push_str(&format!("shininess = {}\n", settings.shininess));
        content.push_str(&format!("base_color = \"{}\"\n", settings.base_color));
        content.push_str(&format!("metallic = {}\n", settings.metallic));
        content.push_str(&format!("roughness = {}\n", settings.roughness));
        content.push_str(&format!(
            "ambient_occlusion = {}\n",
            settings.ambient_occlusion
        ));
        content.push_str(&format!("emissive = \"{}\"\n", settings.emissive));
        content.push('\n');

        // [background] 部分
        content.push_str("[background]\n");
        content.push_str(&format!(
            "use_background_image = {}\n",
            settings.use_background_image
        ));
        content.push_str(&format!(
            "enable_gradient_background = {}\n",
            settings.enable_gradient_background
        ));
        content.push_str(&format!(
            "gradient_top_color = \"{}\"\n",
            settings.gradient_top_color
        ));
        content.push_str(&format!(
            "gradient_bottom_color = \"{}\"\n",
            settings.gradient_bottom_color
        ));
        content.push_str(&format!(
            "enable_ground_plane = {}\n",
            settings.enable_ground_plane
        ));
        content.push_str(&format!(
            "ground_plane_color = \"{}\"\n",
            settings.ground_plane_color
        ));
        content.push_str(&format!(
            "ground_plane_height = {}\n",
            settings.ground_plane_height
        ));
        content.push('\n');

        // [animation] 部分
        content.push_str("[animation]\n");
        content.push_str(&format!("animate = {}\n", settings.animate));
        content.push_str(&format!("fps = {}\n", settings.fps));
        content.push_str(&format!("rotation_speed = {}\n", settings.rotation_speed));
        content.push_str(&format!("rotation_cycles = {}\n", settings.rotation_cycles));
        content.push_str(&format!(
            "animation_type = \"{:?}\"\n",
            settings.animation_type
        ));
        content.push_str(&format!(
            "rotation_axis = \"{:?}\"\n",
            settings.rotation_axis
        ));
        content.push_str(&format!(
            "custom_rotation_axis = \"{}\"\n",
            settings.custom_rotation_axis
        ));
        content.push('\n');

        // [shadow] 部分
        content.push_str("# 阴影配置\n");
        content.push_str("[shadow]\n");
        content.push_str("enable_shadow_mapping = ");
        content.push_str(&settings.enable_shadow_mapping.to_string());
        content.push('\n');
        content.push_str("shadow_map_size = ");
        content.push_str(&settings.shadow_map_size.to_string());
        content.push('\n');
        content.push_str("shadow_bias = ");
        content.push_str(&settings.shadow_bias.to_string());
        content.push('\n');
        content.push_str("shadow_distance = ");
        content.push_str(&settings.shadow_distance.to_string());
        content.push('\n');
        content.push_str("enable_pcf = ");
        content.push_str(&settings.enable_pcf.to_string());
        content.push('\n');
        content.push_str("pcf_type = \"");
        content.push_str(&settings.pcf_type);
        content.push_str("\"\n");
        content.push_str("pcf_kernel = ");
        content.push_str(&settings.pcf_kernel.to_string());
        content.push('\n');
        content.push_str("pcf_sigma = ");
        content.push_str(&settings.pcf_sigma.to_string());
        content.push('\n');

        Ok(content)
    }
}
pub mod config_loader;
pub mod model_loader;
pub mod obj_loader;
pub mod render_settings;
pub mod simple_cli;
use crate::io::obj_loader::load_obj_model;
use crate::io::render_settings::RenderSettings;
use crate::material_system::materials::Model;
use crate::scene::scene_utils::Scene;
use crate::utils::model_utils::normalize_and_center_model;
use log::{debug, info};
use std::path::Path;
use std::time::Instant;

/// 模型加载器
pub struct ModelLoader;

impl ModelLoader {
    /// 主要功能：加载OBJ模型并创建场景
    pub fn load_and_create_scene(
        obj_path: &str,
        settings: &RenderSettings,
    ) -> Result<(Scene, Model), String> {
        info!("加载模型：{obj_path}");
        let load_start = Instant::now();

        // 检查文件存在
        if !Path::new(obj_path).exists() {
            return Err(format!("输入的 OBJ 文件未找到：{obj_path}"));
        }

        // 加载模型数据
        let mut model = load_obj_model(obj_path, settings)?;
        debug!("模型加载耗时 {:?}", load_start.elapsed());

        // 归一化模型
        debug!("归一化模型...");
        let norm_start_time = Instant::now();
        let (original_center, scale_factor) = normalize_and_center_model(&mut model);
        debug!(
            "模型归一化耗时 {:?}，原始中心：{:.3?}，缩放系数：{:.3}",
            norm_start_time.elapsed(),
            original_center,
            scale_factor
        );

        // 创建场景
        debug!("创建场景...");
        let scene = Scene::new(model.clone(), settings)?;

        Ok((scene, model))
    }

    /// 验证资源
    pub fn validate_resources(settings: &RenderSettings) -> Result<(), String> {
        // 验证 OBJ 文件
        if let Some(obj_path) = &settings.obj {
            if !Path::new(obj_path).exists() {
                return Err(format!("OBJ 文件不存在: {obj_path}"));
            }
        }

        // 验证背景图片（如果启用）
        if settings.use_background_image {
            if let Some(bg_path) = &settings.background_image_path {
                if !Path::new(bg_path).exists() {
                    return Err(format!("背景图片文件不存在: {bg_path}"));
                }
            } else {
                return Err("启用了背景图片但未指定路径".to_string());
            }
        }

        // 验证纹理文件（如果指定）
        if let Some(texture_path) = &settings.texture {
            if !Path::new(texture_path).exists() {
                return Err(format!("纹理文件不存在: {texture_path}"));
            }
        }

        info!("所有资源验证通过");
        Ok(())
    }
}
use crate::io::config_loader::TomlConfigLoader;
use crate::io::render_settings::RenderSettings;
use clap::Parser;
use log::info;

/// 极简CLI
#[derive(Parser, Debug)]
#[command(name = "rasterizer")]
#[command(about = "TOML驱动的光栅化渲染器")]
pub struct SimpleCli {
    /// 配置文件路径（TOML格式）
    #[arg(short, long, value_name = "FILE")]
    pub config: Option<String>,

    /// 无头模式（不启动GUI）
    #[arg(long)]
    pub headless: bool,

    /// 使用示例配置（临时创建并加载）
    #[arg(long)]
    pub use_example_config: bool,
}

impl SimpleCli {
    /// 处理CLI参数并返回RenderSettings和是否启动GUI
    pub fn process() -> Result<(RenderSettings, bool), String> {
        let cli = Self::parse();

        // 处理示例配置
        if cli.use_example_config {
            let temp_config_path = "temp_example_config.toml";

            TomlConfigLoader::create_example_config(temp_config_path)
                .map_err(|e| format!("创建示例配置失败: {e}"))?;

            info!("已创建临时示例配置: {temp_config_path}");

            let settings = TomlConfigLoader::load_from_file(temp_config_path)
                .map_err(|e| format!("加载示例配置失败: {e}"))?;

            let should_start_gui = !cli.headless;
            return Ok((settings, should_start_gui));
        }

        // 加载配置文件或使用默认设置
        let settings = if let Some(config_path) = &cli.config {
            info!("加载配置文件: {config_path}");
            TomlConfigLoader::load_from_file(config_path)
                .map_err(|e| format!("配置文件加载失败: {e}"))?
        } else {
            info!("使用默认设置");
            RenderSettings::default()
        };

        let should_start_gui = !cli.headless;
        Ok((settings, should_start_gui))
    }
}
use crate::io::render_settings::RenderSettings;
use crate::material_system::materials::{Material, MaterialType, Mesh, Model, Vertex};
use crate::material_system::texture::Texture;
use image::DynamicImage;
use log::{debug, info, warn};
use nalgebra::{Point3, Vector2, Vector3};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// 生成平滑的顶点法线，通过平均面法线实现
fn generate_smooth_vertex_normals(
    vertices: &[Point3<f32>],
    indices: &[u32],
) -> Result<Vec<Vector3<f32>>, String> {
    if indices.len() % 3 != 0 {
        return Err("三角形索引数量必须是3的倍数".to_string());
    }
    if vertices.is_empty() {
        return Ok(Vec::new());
    }

    let num_vertices = vertices.len();
    let num_faces = indices.len() / 3;
    let mut vertex_normals = vec![Vector3::zeros(); num_vertices];

    for i in 0..num_faces {
        let idx0 = indices[i * 3] as usize;
        let idx1 = indices[i * 3 + 1] as usize;
        let idx2 = indices[i * 3 + 2] as usize;

        if idx0 >= num_vertices || idx1 >= num_vertices || idx2 >= num_vertices {
            warn!("面 {i} 包含越界的顶点索引，跳过");
            continue;
        }

        let v0 = vertices[idx0];
        let v1 = vertices[idx1];
        let v2 = vertices[idx2];

        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let face_normal = edge1.cross(&edge2);

        vertex_normals[idx0] += face_normal;
        vertex_normals[idx1] += face_normal;
        vertex_normals[idx2] += face_normal;
    }

    let mut zero_norm_count = 0;
    for normal in vertex_normals.iter_mut() {
        let norm_squared = normal.norm_squared();
        if norm_squared > 1e-12 {
            normal.normalize_mut();
        } else {
            *normal = Vector3::y();
            zero_norm_count += 1;
        }
    }

    if zero_norm_count > 0 {
        warn!("{zero_norm_count} 个顶点的法线为零，设置为默认值 [0, 1, 0]");
    }

    Ok(vertex_normals)
}

fn get_basename_from_path(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unknown".to_string())
}

/// 主要功能：加载并处理 OBJ 模型文件
pub fn load_obj_model<P: AsRef<Path>>(
    obj_path: P,
    settings: &RenderSettings,
) -> Result<Model, String> {
    let obj_path_ref = obj_path.as_ref();
    info!("加载 OBJ 文件: {obj_path_ref:?}");

    let obj_basename = get_basename_from_path(obj_path_ref);
    let base_path = obj_path_ref.parent().unwrap_or_else(|| Path::new("."));

    let cli_texture: Option<Texture> = if let Some(tex_path_str) = &settings.texture {
        let tex_path = Path::new(tex_path_str);
        debug!("使用命令行指定的纹理: {tex_path:?}");
        Some(Texture::from_file(tex_path).unwrap_or_else(|| {
            warn!("无法加载命令行指定的纹理，使用默认颜色");
            Texture {
                image: Arc::new(DynamicImage::new_rgb8(1, 1)),
                width: 1,
                height: 1,
            }
        }))
    } else {
        None
    };

    let load_options = tobj::LoadOptions {
        triangulate: true,
        single_index: false,
        ignore_points: true,
        ignore_lines: true,
    };

    let (models, materials_result) =
        tobj::load_obj(obj_path_ref, &load_options).map_err(|e| format!("加载 OBJ 失败: {e}"))?;

    let mut loaded_materials: Vec<Material> = match materials_result {
        Ok(mats) => {
            if !mats.is_empty() {
                info!("从 MTL 加载了 {} 个材质", mats.len());
                mats.into_iter()
                    .map(|mat| {
                        // 只加载图片纹理
                        let texture = if let Some(cli_tex) = &cli_texture {
                            Some(cli_tex.clone())
                        } else if let Some(tex_name) = mat.diffuse_texture {
                            let texture_path = base_path.join(&tex_name);
                            Some(Texture::from_file(&texture_path).unwrap_or_else(|| {
                                warn!("无法加载纹理 '{texture_path:?}'，使用默认颜色");
                                Texture {
                                    image: Arc::new(DynamicImage::new_rgb8(1, 1)),
                                    width: 1,
                                    height: 1,
                                }
                            }))
                        } else {
                            None
                        };

                        Material {
                            material_type: if settings.use_pbr {
                                MaterialType::PBR
                            } else {
                                MaterialType::Phong
                            },
                            base_color: Vector3::from(mat.diffuse.unwrap_or([0.8, 0.8, 0.8])),
                            alpha: 1.0,
                            texture,
                            metallic: 0.0,
                            roughness: 0.5,
                            ambient_occlusion: 1.0,
                            specular: Vector3::from(mat.specular.unwrap_or([0.5, 0.5, 0.5])),
                            shininess: mat.shininess.unwrap_or(32.0),
                            diffuse_intensity: 1.0,
                            specular_intensity: 1.0,
                            emissive: Vector3::zeros(),
                            ambient_factor: Vector3::from(mat.diffuse.unwrap_or([0.8, 0.8, 0.8]))
                                * 0.3,
                        }
                    })
                    .collect()
            } else {
                info!("MTL 文件中没有材质");
                Vec::new()
            }
        }
        Err(e) => {
            warn!("加载材质失败: {e}");
            Vec::new()
        }
    };

    if loaded_materials.is_empty() {
        let default_type = if settings.use_pbr {
            MaterialType::PBR
        } else {
            MaterialType::Phong
        };
        let mut default_mat = Material::default(default_type);

        if let Some(texture) = cli_texture {
            default_mat.texture = Some(texture);
        } else {
            default_mat.texture = None;
        }
        loaded_materials.push(default_mat);
    }

    let mut loaded_meshes: Vec<Mesh> = Vec::with_capacity(models.len());

    for model in models.iter() {
        let mesh = &model.mesh;
        let num_vertices_in_obj = mesh.positions.len() / 3;

        let mesh_name = if model.name.is_empty() || model.name == "unnamed_object" {
            obj_basename.clone()
        } else {
            model.name.clone()
        };

        if mesh.indices.is_empty() {
            debug!("跳过没有索引的网格 '{mesh_name}'");
            continue;
        }

        let has_normals = !mesh.normals.is_empty();
        let has_texcoords = !mesh.texcoords.is_empty();

        let generated_normals: Option<Vec<Vector3<f32>>> = if !has_normals {
            warn!("网格 '{mesh_name}' 缺少法线，计算平滑顶点法线");

            let positions: Vec<Point3<f32>> = mesh
                .positions
                .chunks_exact(3)
                .map(|p| Point3::new(p[0], p[1], p[2]))
                .collect();

            match generate_smooth_vertex_normals(&positions, &mesh.indices) {
                Ok(normals) => Some(normals),
                Err(e) => {
                    warn!("生成平滑法线错误: {e}，使用默认法线 [0,1,0]");
                    Some(vec![Vector3::y(); num_vertices_in_obj])
                }
            }
        } else {
            None
        };

        if !has_texcoords {
            debug!("网格 '{mesh_name}' 缺少纹理坐标，纹理映射可能不正确");
        }

        let mut vertices: Vec<Vertex> = Vec::new();
        let mut index_map: HashMap<(u32, Option<u32>, Option<u32>), u32> = HashMap::new();
        let mut final_indices: Vec<u32> = Vec::with_capacity(mesh.indices.len());

        for i in 0..mesh.indices.len() {
            let pos_idx = mesh.indices[i];
            let norm_idx_opt = mesh.normal_indices.get(i).copied();
            let tc_idx_opt = mesh.texcoord_indices.get(i).copied();

            let key = (pos_idx, norm_idx_opt, tc_idx_opt);

            if let Some(&final_idx) = index_map.get(&key) {
                final_indices.push(final_idx);
            } else {
                let p_start = pos_idx as usize * 3;
                let position = if p_start + 2 < mesh.positions.len() {
                    Point3::new(
                        mesh.positions[p_start],
                        mesh.positions[p_start + 1],
                        mesh.positions[p_start + 2],
                    )
                } else {
                    warn!("遇到无效的 OBJ 位置索引 {pos_idx}");
                    Point3::origin()
                };

                let normal = match norm_idx_opt {
                    Some(normal_source_idx) => {
                        if let Some(ref gen_normals) = generated_normals {
                            gen_normals
                                .get(pos_idx as usize)
                                .copied()
                                .unwrap_or_else(|| {
                                    warn!("生成的法线索引 {pos_idx} 越界");
                                    Vector3::y()
                                })
                        } else {
                            let n_start = normal_source_idx as usize * 3;
                            if n_start + 2 < mesh.normals.len() {
                                Vector3::new(
                                    mesh.normals[n_start],
                                    mesh.normals[n_start + 1],
                                    mesh.normals[n_start + 2],
                                )
                                .normalize()
                            } else {
                                warn!("遇到无效的 OBJ 法线索引 {normal_source_idx}");
                                Vector3::y()
                            }
                        }
                    }
                    None => {
                        if let Some(ref gen_normals) = generated_normals {
                            gen_normals
                                .get(pos_idx as usize)
                                .copied()
                                .unwrap_or_else(|| {
                                    warn!("生成的法线索引 {pos_idx} 越界（回退）");
                                    Vector3::y()
                                })
                        } else {
                            warn!("缺少顶点 {pos_idx} 的法线索引和生成法线");
                            Vector3::y()
                        }
                    }
                };

                let texcoord = if let Some(idx) = tc_idx_opt {
                    let t_start = idx as usize * 2;
                    if t_start + 1 < mesh.texcoords.len() {
                        Vector2::new(mesh.texcoords[t_start], mesh.texcoords[t_start + 1])
                    } else {
                        warn!("遇到无效的 OBJ 纹理坐标索引 {idx}");
                        Vector2::zeros()
                    }
                } else {
                    Vector2::zeros()
                };

                let new_vertex = Vertex {
                    position,
                    normal,
                    texcoord,
                };
                let new_final_idx = vertices.len() as u32;
                vertices.push(new_vertex);
                index_map.insert(key, new_final_idx);
                final_indices.push(new_final_idx);
            }
        }

        let material_id = mesh.material_id.unwrap_or(0);
        let final_material_id = if material_id < loaded_materials.len() {
            material_id
        } else if !loaded_materials.is_empty() {
            warn!("网格 '{mesh_name}' 有无效的材质 ID {material_id}。分配默认材质 ID 0");
            0
        } else {
            0
        };

        loaded_meshes.push(Mesh {
            vertices,
            indices: final_indices,
            material_id: final_material_id,
            name: mesh_name.clone(),
        });

        debug!(
            "处理网格 '{}': {} 个唯一顶点, {} 个三角形, 材质 ID: {}",
            loaded_meshes.last().unwrap().name,
            loaded_meshes.last().unwrap().vertices.len(),
            loaded_meshes.last().unwrap().indices.len() / 3,
            final_material_id
        );
    }

    if loaded_meshes.is_empty() {
        return Err("OBJ 文件中没有可处理的网格".to_string());
    }

    let model = Model {
        meshes: loaded_meshes,
        materials: loaded_materials,
        name: obj_basename,
    };

    info!("创建模型 '{}' 成功", model.name);
    Ok(model)
}
use crate::material_system::light::Light;
use log::warn;
use nalgebra::{Point3, Vector3};

/// 动画类型枚举
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum AnimationType {
    #[default]
    CameraOrbit,
    ObjectLocalRotation,
    None,
}

/// 旋转轴枚举
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum RotationAxis {
    X,
    #[default]
    Y,
    Z,
    Custom,
}

/// 纯数据结构
#[derive(Debug, Clone)]
pub struct RenderSettings {
    // ===== 文件路径设置 =====
    /// 输入OBJ文件的路径
    pub obj: Option<String>,
    /// 输出文件的基础名称
    pub output: String,
    /// 输出图像的目录
    pub output_dir: String,
    /// 显式指定要使用的纹理文件，覆盖MTL设置
    pub texture: Option<String>,
    /// 背景图片路径
    pub background_image_path: Option<String>,

    // ===== 渲染基础设置 =====
    /// 输出图像的宽度
    pub width: usize,
    /// 输出图像的高度
    pub height: usize,
    /// 投影类型："perspective"或"orthographic"
    pub projection: String,
    /// 启用Z缓冲（深度测试）
    pub use_zbuffer: bool,
    /// 使用伪随机面颜色而非材质颜色
    pub colorize: bool,
    /// 启用纹理加载和使用
    pub use_texture: bool,
    /// 启用gamma矫正
    pub use_gamma: bool,
    /// 启用ACES色彩管理
    pub enable_aces: bool,
    /// 启用背面剔除
    pub backface_culling: bool,
    /// 以线框模式渲染
    pub wireframe: bool,
    /// 启用小三角形剔除
    pub cull_small_triangles: bool,
    /// 小三角形剔除的最小面积阈值
    pub min_triangle_area: f32,
    /// 启用渲染和保存深度图
    pub save_depth: bool,

    // ===== 物体变换控制（字符串格式，用于TOML序列化） =====
    /// 物体位置 (x,y,z)
    pub object_position: String,
    /// 物体旋转 (欧拉角，度)
    pub object_rotation: String,
    /// 物体缩放 (x,y,z)
    pub object_scale_xyz: String,
    /// 物体的全局均匀缩放因子
    pub object_scale: f32,

    // ===== 相机参数 =====
    /// 相机位置（视点），格式为"x,y,z"
    pub camera_from: String,
    /// 相机目标（观察点），格式为"x,y,z"
    pub camera_at: String,
    /// 相机世界坐标系上方向，格式为"x,y,z"
    pub camera_up: String,
    /// 相机垂直视场角（度，用于透视投影）
    pub camera_fov: f32,

    // ===== 光照基础参数 =====
    /// 启用光照计算
    pub use_lighting: bool,
    /// 环境光强度因子
    pub ambient: f32,
    /// 环境光强度RGB值，格式为"r,g,b"
    pub ambient_color: String,

    // ===== 着色模型选择 =====
    /// 使用Phong着色（逐像素光照）
    pub use_phong: bool,
    /// 使用基于物理的渲染(PBR)
    pub use_pbr: bool,

    // ===== Phong着色模型参数 =====
    /// 漫反射颜色，格式为"r,g,b"
    pub diffuse_color: String,
    /// 漫反射强度(0.0-2.0)
    pub diffuse_intensity: f32,
    /// 镜面反射颜色，格式为"r,g,b" (之前是单一数值)
    pub specular_color: String,
    /// 镜面反射强度(0.0-2.0)
    pub specular_intensity: f32,
    /// 材质的光泽度(硬度)参数
    pub shininess: f32,

    // ===== PBR材质参数 =====
    /// 材质的基础颜色，格式为"r,g,b"
    pub base_color: String,
    /// 材质的金属度(0.0-1.0)
    pub metallic: f32,
    /// 材质的粗糙度(0.0-1.0)
    pub roughness: f32,
    /// 环境光遮蔽系数(0.0-1.0)
    pub ambient_occlusion: f32,
    /// 材质透明度(0.0-1.0)，1.0为完全不透明
    pub alpha: f32,
    /// 材质的自发光颜色，格式为"r,g,b"
    pub emissive: String,

    // ===== 阴影设置 =====
    /// 启用简单阴影映射（仅地面）
    pub enable_shadow_mapping: bool,
    /// 阴影贴图尺寸
    pub shadow_map_size: usize,
    /// 阴影偏移，避免阴影痤疮
    pub shadow_bias: f32,
    /// 阴影渲染距离
    pub shadow_distance: f32,
    /// 是否启用PCF软阴影
    pub enable_pcf: bool,
    /// PCF类型
    pub pcf_type: String,
    /// PCF采样窗口半径
    pub pcf_kernel: usize,
    /// PCF高斯模糊的sigma
    pub pcf_sigma: f32,

    // ===== 背景与环境设置 =====
    /// 启用渐变背景
    pub enable_gradient_background: bool,
    /// 渐变背景顶部颜色，格式为"r,g,b"
    pub gradient_top_color: String,
    /// 渐变背景底部颜色，格式为"r,g,b"
    pub gradient_bottom_color: String,
    /// 启用地面平面
    pub enable_ground_plane: bool,
    /// 地面平面颜色，格式为"r,g,b"
    pub ground_plane_color: String,
    /// 地面平面在Y轴上的高度
    pub ground_plane_height: f32,
    /// 使用背景图片
    pub use_background_image: bool,

    // ===== 动画设置 =====
    /// 运行完整动画循环而非单帧渲染
    pub animate: bool,
    /// 动画帧率 (fps)，用于视频生成和预渲染
    pub fps: usize,
    /// 旋转速度系数，控制动画旋转的速度
    pub rotation_speed: f32,
    /// 完整旋转圈数，用于视频生成(默认1圈)
    pub rotation_cycles: f32,
    /// 动画类型 (用于 animate 模式或实时渲染)
    pub animation_type: AnimationType,
    /// 动画旋转轴 (用于 CameraOrbit 和 ObjectLocalRotation)
    pub rotation_axis: RotationAxis,
    /// 自定义旋转轴 (当 rotation_axis 为 Custom 时使用)，格式 "x,y,z"
    pub custom_rotation_axis: String,

    // ===== 光源数组（运行时字段） =====
    /// 场景中的所有光源
    pub lights: Vec<Light>,
}

/// 辅助函数用于解析逗号分隔的浮点数
pub fn parse_vec3(s: &str) -> Result<Vector3<f32>, String> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 3 {
        return Err("需要3个逗号分隔的值".to_string());
    }
    let x = parts[0]
        .trim()
        .parse::<f32>()
        .map_err(|e| format!("无效数字 '{}': {}", parts[0], e))?;
    let y = parts[1]
        .trim()
        .parse::<f32>()
        .map_err(|e| format!("无效数字 '{}': {}", parts[1], e))?;
    let z = parts[2]
        .trim()
        .parse::<f32>()
        .map_err(|e| format!("无效数字 '{}': {}", parts[2], e))?;
    Ok(Vector3::new(x, y, z))
}

pub fn parse_point3(s: &str) -> Result<Point3<f32>, String> {
    parse_vec3(s).map(Point3::from)
}

/// 将 RenderSettings 中的旋转轴配置转换为 Vector3<f32>
pub fn get_animation_axis_vector(settings: &RenderSettings) -> Vector3<f32> {
    match settings.rotation_axis {
        RotationAxis::X => Vector3::x_axis().into_inner(),
        RotationAxis::Y => Vector3::y_axis().into_inner(),
        RotationAxis::Z => Vector3::z_axis().into_inner(),
        RotationAxis::Custom => parse_vec3(&settings.custom_rotation_axis)
            .unwrap_or_else(|_| {
                warn!(
                    "无效的自定义旋转轴 '{}', 使用默认Y轴",
                    settings.custom_rotation_axis
                );
                Vector3::y_axis().into_inner()
            })
            .normalize(),
    }
}

impl Default for RenderSettings {
    fn default() -> Self {
        let mut settings = Self {
            // ===== 文件路径设置 =====
            obj: None,
            output: "output".to_string(),
            output_dir: "output_rust".to_string(),
            texture: None,
            background_image_path: None,

            // ===== 渲染基础设置 =====
            width: 1024,
            height: 1024,
            projection: "perspective".to_string(),
            use_zbuffer: true,
            colorize: false,
            use_texture: true,
            use_gamma: true,
            enable_aces: false,
            backface_culling: false,
            wireframe: false,
            cull_small_triangles: false,
            min_triangle_area: 1e-3,
            save_depth: true,

            // ===== 物体变换控制 =====
            object_position: "0,0,0".to_string(),
            object_rotation: "0,0,0".to_string(),
            object_scale_xyz: "1,1,1".to_string(),
            object_scale: 1.0,

            // ===== 相机参数 =====
            camera_from: "0,0,3".to_string(),
            camera_at: "0,0,0".to_string(),
            camera_up: "0,1,0".to_string(),
            camera_fov: 45.0,

            // ===== 光照基础参数 =====
            use_lighting: true,
            ambient: 0.3,
            ambient_color: "0.3,0.4,0.5".to_string(),

            // ===== 着色模型选择 =====
            use_phong: true,
            use_pbr: false,

            // ===== Phong着色模型参数 =====
            diffuse_color: "0.8,0.8,0.8".to_string(),
            diffuse_intensity: 1.0,
            specular_color: "0.5,0.5,0.5".to_string(),
            specular_intensity: 1.0,
            shininess: 32.0,

            // ===== PBR材质参数 =====
            base_color: "0.8,0.8,0.8".to_string(),
            metallic: 0.0,
            roughness: 0.5,
            ambient_occlusion: 1.0,
            alpha: 1.0, // 默认完全不透明
            emissive: "0.0,0.0,0.0".to_string(),

            // ===== 阴影设置 =====

            // 简单阴影映射配置
            enable_shadow_mapping: false, // 启用地面阴影映射
            shadow_map_size: 256,         // 阴影贴图尺寸（较小，只用于地面）
            shadow_bias: 0.001,           // 阴影偏移
            shadow_distance: 20.0,

            // 新增：PCF相关参数
            enable_pcf: false,           // 是否启用PCF软阴影
            pcf_type: "Box".to_string(), // PCF类型：Box或 Gauss
            pcf_kernel: 2,               // PCF采样窗口半径
            pcf_sigma: 1.0,              // Gauss类型的sigma

            // ===== 背景与环境设置 =====
            enable_gradient_background: false,
            gradient_top_color: "0.5,0.7,1.0".to_string(),
            gradient_bottom_color: "0.1,0.2,0.4".to_string(),
            enable_ground_plane: false,
            ground_plane_color: "0.3,0.5,0.2".to_string(),
            ground_plane_height: -1.0,
            use_background_image: false,

            // ===== 动画设置 =====
            animate: false,
            fps: 30,
            rotation_speed: 1.0,
            rotation_cycles: 1.0,
            animation_type: AnimationType::CameraOrbit,
            rotation_axis: RotationAxis::Y,
            custom_rotation_axis: "0,1,0".to_string(),

            // ===== 光源数组 =====
            lights: Vec::new(),
        };

        // 如果启用了光照且没有光源，创建默认方向光
        settings.initialize_lights();

        settings
    }
}

impl RenderSettings {
    /// 初始化默认光源
    pub fn initialize_lights(&mut self) {
        if self.use_lighting && self.lights.is_empty() {
            self.lights = vec![Light::directional(
                Vector3::new(0.0, -1.0, -1.0),
                Vector3::new(1.0, 1.0, 1.0),
                0.8,
            )];
        }
    }

    // ===== 按需计算方法 =====

    /// 获取环境光颜色向量（按需计算）
    pub fn get_ambient_color_vec(&self) -> Vector3<f32> {
        parse_vec3(&self.ambient_color).unwrap_or_else(|_| Vector3::new(0.1, 0.1, 0.1))
    }

    /// 获取渐变顶部颜色向量（按需计算）
    pub fn get_gradient_top_color_vec(&self) -> Vector3<f32> {
        parse_vec3(&self.gradient_top_color).unwrap_or_else(|_| Vector3::new(0.5, 0.7, 1.0))
    }

    /// 获取渐变底部颜色向量（按需计算）
    pub fn get_gradient_bottom_color_vec(&self) -> Vector3<f32> {
        parse_vec3(&self.gradient_bottom_color).unwrap_or_else(|_| Vector3::new(0.1, 0.2, 0.4))
    }

    /// 获取地面平面颜色向量（按需计算）
    pub fn get_ground_plane_color_vec(&self) -> Vector3<f32> {
        parse_vec3(&self.ground_plane_color).unwrap_or_else(|_| Vector3::new(0.3, 0.5, 0.2))
    }

    /// 解析物体变换参数为向量（统一接口）
    pub fn get_object_transform_components(&self) -> (Vector3<f32>, Vector3<f32>, Vector3<f32>) {
        // 解析位置
        let position =
            parse_vec3(&self.object_position).unwrap_or_else(|_| Vector3::new(0.0, 0.0, 0.0));

        // 解析旋转（度转弧度）
        let rotation_deg =
            parse_vec3(&self.object_rotation).unwrap_or_else(|_| Vector3::new(0.0, 0.0, 0.0));
        let rotation_rad = Vector3::new(
            rotation_deg.x.to_radians(),
            rotation_deg.y.to_radians(),
            rotation_deg.z.to_radians(),
        );

        // 解析缩放
        let scale =
            parse_vec3(&self.object_scale_xyz).unwrap_or_else(|_| Vector3::new(1.0, 1.0, 1.0));

        (position, rotation_rad, scale)
    }

    /// 判断是否使用透视投影
    pub fn is_perspective(&self) -> bool {
        self.projection == "perspective"
    }

    /// 获取着色模型的描述字符串
    pub fn get_lighting_description(&self) -> String {
        if self.use_pbr {
            "基于物理的渲染(PBR)".to_string()
        } else if self.use_phong {
            "Phong着色模型".to_string()
        } else {
            "平面着色模型".to_string()
        }
    }

    /// 验证渲染参数
    pub fn validate(&self) -> Result<(), String> {
        if self.width == 0 || self.height == 0 {
            return Err("错误: 图像宽度和高度必须大于0".to_string());
        }

        if let Some(obj_path) = &self.obj {
            if !std::path::Path::new(obj_path).exists() {
                return Err(format!("错误: 找不到OBJ文件 '{obj_path}'"));
            }
        } else {
            return Err("错误: 未指定OBJ文件路径".to_string());
        }

        if self.output_dir.trim().is_empty() {
            return Err("错误: 输出目录不能为空".to_string());
        }

        if self.output.trim().is_empty() {
            return Err("错误: 输出文件名不能为空".to_string());
        }

        // 验证相机参数
        if parse_vec3(&self.camera_from).is_err() {
            return Err("错误: 相机位置格式不正确，应为 x,y,z 格式".to_string());
        }

        if parse_vec3(&self.camera_at).is_err() {
            return Err("错误: 相机目标格式不正确，应为 x,y,z 格式".to_string());
        }

        if parse_vec3(&self.camera_up).is_err() {
            return Err("错误: 相机上方向格式不正确，应为 x,y,z 格式".to_string());
        }

        // 验证物体变换参数
        if parse_vec3(&self.object_position).is_err() {
            return Err("错误: 物体位置格式不正确，应为 x,y,z 格式".to_string());
        }

        if parse_vec3(&self.object_rotation).is_err() {
            return Err("错误: 物体旋转格式不正确，应为 x,y,z 格式".to_string());
        }

        if parse_vec3(&self.object_scale_xyz).is_err() {
            return Err("错误: 物体缩放格式不正确，应为 x,y,z 格式".to_string());
        }

        Ok(())
    }
}
use nalgebra::Vector3;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// 表示具有浮点分量[0.0, 1.0]的RGB颜色。
pub type Color = Vector3<f32>;

/// 应用gamma矫正，将线性RGB值转换为sRGB空间
///
/// # 参数
/// * `linear_color` - 线性空间的RGB颜色值 [0.0-1.0]
///
/// # 返回值
/// 应用了gamma矫正的RGB颜色值 [0.0-1.0]
pub fn apply_gamma_correction(linear_color: &Color) -> Color {
    // 使用标准的gamma值2.2
    let gamma = 2.2;
    let inv_gamma = 1.0 / gamma;

    // 对每个颜色通道应用幂函数
    Color::new(
        linear_color.x.powf(inv_gamma),
        linear_color.y.powf(inv_gamma),
        linear_color.z.powf(inv_gamma),
    )
}

/// 从sRGB空间转换回线性RGB值（解码）
///
/// # 参数
/// * `srgb_color` - sRGB空间的RGB颜色值 [0.0-1.0]
///
/// # 返回值
/// 线性空间的RGB颜色值 [0.0-1.0]
pub fn srgb_to_linear(srgb_color: &Color) -> Color {
    // 使用标准的gamma值2.2
    let gamma = 2.2;

    // 应用逆变换
    Color::new(
        srgb_color.x.powf(gamma),
        srgb_color.y.powf(gamma),
        srgb_color.z.powf(gamma),
    )
}

/// 应用ACES色调映射，将高动态范围颜色压缩到显示范围
///
/// # 参数
/// * `color` - 线性RGB颜色值 [0.0-1.0]
/// # 返回值
/// 压缩后的RGB颜色值 [0.0-1.0]
pub fn apply_aces_tonemap(color: &Vector3<f32>) -> Vector3<f32> {
    // ACES Filmic Curve参数
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    Vector3::new(
        ((color.x * (a * color.x + b)) / (color.x * (c * color.x + d) + e)).clamp(0.0, 1.0),
        ((color.y * (a * color.y + b)) / (color.y * (c * color.y + d) + e)).clamp(0.0, 1.0),
        ((color.z * (a * color.z + b)) / (color.z * (c * color.z + d) + e)).clamp(0.0, 1.0),
    )
}

/// 将线性RGB值转换为u8数组，应用gamma矫正
///
/// # 参数
/// * `linear_color` - 线性空间的RGB颜色值 [0.0-1.0]
/// * `apply_gamma` - 是否应用gamma矫正
///
/// # 返回值
/// 一个包含三个u8值的数组，表示颜色的RGB通道
pub fn linear_rgb_to_u8(linear_color: &Color, apply_gamma: bool) -> [u8; 3] {
    let display_color = if apply_gamma {
        apply_gamma_correction(linear_color)
    } else {
        *linear_color
    };

    [
        (display_color.x * 255.0).clamp(0.0, 255.0) as u8,
        (display_color.y * 255.0).clamp(0.0, 255.0) as u8,
        (display_color.z * 255.0).clamp(0.0, 255.0) as u8,
    ]
}

/// 获取基于种子的随机颜色。
///
/// 如果`colorize`为false，返回默认的灰色。
/// 如果`colorize`为true，根据种子生成伪随机颜色
/// （对于相同的种子，结果是确定性的）。
///
/// # 参数
/// * `seed` - 随机数种子
/// * `colorize` - 是否生成彩色（否则返回默认灰色）
pub fn get_random_color(seed: u64, colorize: bool) -> Color {
    if !colorize {
        // 默认灰色
        Color::new(0.7, 0.7, 0.7)
    } else {
        // 使用种子生成确定性随机颜色
        let mut rng = StdRng::seed_from_u64(seed);
        Color::new(
            0.3 + rng.random::<f32>() * 0.4, // R 在 [0.3, 0.7) 范围内
            0.3 + rng.random::<f32>() * 0.4, // G 在 [0.3, 0.7) 范围内
            0.3 + rng.random::<f32>() * 0.4, // B 在 [0.3, 0.7) 范围内
        )
    }
}

/// 将归一化的深度图（值范围0.0-1.0）转换为使用JET色彩映射的RGB彩色图像。
///
/// 无效的深度值（NaN、无穷大）将显示为黑色像素。
///
/// # 参数
/// * `normalized_depth` - 扁平化的深度值切片（行优先）。
/// * `width` - 深度图的宽度。
/// * `height` - 深度图的高度。
/// * `apply_gamma` - 是否应用gamma矫正
///
/// # 返回值
/// 包含扁平化RGB图像数据的`Vec<u8>`（每个通道0-255）。
pub fn apply_colormap_jet(
    normalized_depth: &[f32],
    width: usize,
    height: usize,
    apply_gamma: bool,
) -> Vec<u8> {
    let num_pixels = width * height;
    if normalized_depth.len() != num_pixels {
        // 或返回一个错误Result
        panic!("Depth buffer size does not match width * height");
    }

    let mut result = vec![0u8; num_pixels * 3]; // 初始化为黑色

    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            let depth = normalized_depth[index];

            if depth.is_finite() {
                let value = depth.clamp(0.0, 1.0); // 确保值在[0, 1]范围内

                let mut r = 0.0;
                let g;
                let mut b = 0.0;

                // 应用JET色彩映射逻辑
                if value <= 0.25 {
                    // 从蓝色到青色
                    b = 1.0;
                    g = value * 4.0;
                } else if value <= 0.5 {
                    // 从青色到绿色
                    g = 1.0;
                    b = 1.0 - (value - 0.25) * 4.0;
                } else if value <= 0.75 {
                    // 从绿色到黄色
                    g = 1.0;
                    r = (value - 0.5) * 4.0;
                } else {
                    // 从黄色到红色
                    r = 1.0;
                    g = 1.0 - (value - 0.75) * 4.0;
                }

                let color = Color::new(r, g, b);
                let [r_u8, g_u8, b_u8] = linear_rgb_to_u8(&color, apply_gamma);

                // 写入结果缓冲区
                let base_index = index * 3;
                result[base_index] = r_u8;
                result[base_index + 1] = g_u8;
                result[base_index + 2] = b_u8;
            }
            // 如果深度值不是有限的，像素保持黑色（初始化为0）
        }
    }

    result
}
pub mod color;
pub mod light;
pub mod materials;
pub mod texture;
use image::{DynamicImage, GenericImageView};
use log::warn;
use std::path::Path;
use std::sync::Arc;

use crate::material_system::color::{Color, srgb_to_linear};

#[derive(Debug, Clone)]
pub struct Texture {
    pub image: Arc<DynamicImage>,
    pub width: u32,
    pub height: u32,
}

impl Texture {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Option<Self> {
        match image::open(path) {
            Ok(img) => Some(Texture {
                width: img.width(),
                height: img.height(),
                image: Arc::new(img),
            }),
            Err(e) => {
                warn!("无法加载纹理: {e}");
                None
            }
        }
    }

    pub fn sample(&self, u: f32, v: f32) -> [f32; 3] {
        let u = u.fract().abs();
        let v = v.fract().abs();
        let x = (u * self.width as f32) as u32 % self.width;
        let y = ((1.0 - v) * self.height as f32) as u32 % self.height;

        let pixel = self.image.get_pixel(x, y);
        let srgb_color = Color::new(
            pixel[0] as f32 / 255.0,
            pixel[1] as f32 / 255.0,
            pixel[2] as f32 / 255.0,
        );
        let linear_color = srgb_to_linear(&srgb_color);
        [linear_color.x, linear_color.y, linear_color.z]
    }
}
use crate::io::render_settings::{parse_point3, parse_vec3};
use nalgebra::{Point3, Vector3};

/// 统一的光源结构
#[derive(Debug, Clone)]
pub enum Light {
    Directional {
        // 配置字段 (用于GUI控制)
        enabled: bool,
        direction_str: String, // "x,y,z" 格式，用于GUI编辑
        color_str: String,     // "r,g,b" 格式，用于GUI编辑
        intensity: f32,

        // 运行时字段 (用于渲染计算，从配置字段解析)
        direction: Vector3<f32>, // 解析后的方向向量
        color: Vector3<f32>,     // 解析后的颜色向量
    },
    Point {
        // 配置字段 (用于GUI控制)
        enabled: bool,
        position_str: String, // "x,y,z" 格式，用于GUI编辑
        color_str: String,    // "r,g,b" 格式，用于GUI编辑
        intensity: f32,
        constant_attenuation: f32,
        linear_attenuation: f32,
        quadratic_attenuation: f32,

        // 运行时字段 (用于渲染计算，从配置字段解析)
        position: Point3<f32>, // 解析后的位置
        color: Vector3<f32>,   // 解析后的颜色向量
    },
}

impl Light {
    /// 创建方向光
    pub fn directional(direction: Vector3<f32>, color: Vector3<f32>, intensity: f32) -> Self {
        let direction_normalized = direction.normalize();
        Self::Directional {
            enabled: true,
            direction_str: format!(
                "{},{},{}",
                direction_normalized.x, direction_normalized.y, direction_normalized.z
            ),
            color_str: format!("{},{},{}", color.x, color.y, color.z),
            intensity,
            direction: direction_normalized,
            color,
        }
    }

    /// 创建点光源
    pub fn point(
        position: Point3<f32>,
        color: Vector3<f32>,
        intensity: f32,
        attenuation: Option<(f32, f32, f32)>,
    ) -> Self {
        let (constant, linear, quadratic) = attenuation.unwrap_or((1.0, 0.09, 0.032));
        Self::Point {
            enabled: true,
            position_str: format!("{},{},{}", position.x, position.y, position.z),
            color_str: format!("{},{},{}", color.x, color.y, color.z),
            intensity,
            constant_attenuation: constant,
            linear_attenuation: linear,
            quadratic_attenuation: quadratic,
            position,
            color,
        }
    }

    /// 更新运行时字段
    pub fn update_runtime_fields(&mut self) -> Result<(), String> {
        match self {
            Self::Directional {
                direction_str,
                color_str,
                direction,
                color,
                ..
            } => {
                *direction = parse_vec3(direction_str)?.normalize();
                *color = parse_vec3(color_str)?;
            }
            Self::Point {
                position_str,
                color_str,
                position,
                color,
                ..
            } => {
                *position = parse_point3(position_str)?;
                *color = parse_vec3(color_str)?;
            }
        }
        Ok(())
    }

    /// 获取光源方向（用于渲染）
    pub fn get_direction(&self, point: &Point3<f32>) -> Vector3<f32> {
        match self {
            Self::Directional { direction, .. } => -direction,
            Self::Point { position, .. } => (position - point).normalize(),
        }
    }

    /// 获取光源强度（用于渲染）
    pub fn get_intensity(&self, point: &Point3<f32>) -> Vector3<f32> {
        match self {
            Self::Directional {
                color,
                intensity,
                enabled,
                ..
            } => {
                if *enabled {
                    color * *intensity
                } else {
                    Vector3::zeros()
                }
            }
            Self::Point {
                position,
                color,
                intensity,
                constant_attenuation,
                linear_attenuation,
                quadratic_attenuation,
                enabled,
                ..
            } => {
                if *enabled {
                    let distance = (position - point).magnitude();
                    let attenuation_factor = 1.0
                        / (constant_attenuation
                            + linear_attenuation * distance
                            + quadratic_attenuation * distance * distance);
                    color * *intensity * attenuation_factor
                } else {
                    Vector3::zeros()
                }
            }
        }
    }
}
use crate::io::render_settings::{RenderSettings, parse_vec3};
use crate::material_system::texture::Texture;
use log::warn;
use nalgebra::{Point3, Vector2, Vector3};
use std::fmt::Debug;

#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    pub position: Point3<f32>,
    pub normal: Vector3<f32>,
    pub texcoord: Vector2<f32>,
}

/// 材质类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterialType {
    Phong,
    #[allow(clippy::upper_case_acronyms)]
    PBR,
}

/// 材质结构体，统一包含所有参数
#[derive(Debug, Clone)]
pub struct Material {
    pub material_type: MaterialType, // 材质类型
    pub base_color: Vector3<f32>,    // 基础色（PBR/Phong通用）
    pub alpha: f32,                  // 透明度
    pub texture: Option<Texture>,    // 纹理资源

    // ===== PBR参数 =====
    pub metallic: f32,
    pub roughness: f32,
    pub ambient_occlusion: f32,

    // ===== Phong参数 =====
    pub specular: Vector3<f32>,
    pub shininess: f32,
    pub diffuse_intensity: f32,
    pub specular_intensity: f32,

    // ===== 通用参数 =====
    pub emissive: Vector3<f32>,
    pub ambient_factor: Vector3<f32>,
}

impl Material {
    pub fn default(material_type: MaterialType) -> Self {
        Material {
            material_type,
            base_color: Vector3::new(0.8, 0.8, 0.8),
            alpha: 1.0,
            texture: None,
            metallic: 0.0,
            roughness: 0.5,
            ambient_occlusion: 1.0,
            specular: Vector3::new(0.5, 0.5, 0.5),
            shininess: 32.0,
            diffuse_intensity: 1.0,
            specular_intensity: 1.0,
            emissive: Vector3::zeros(),
            ambient_factor: Vector3::new(1.0, 1.0, 1.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub material_id: usize,
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct Model {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
    pub name: String,
}

/// 材质响应计算（统一接口）
pub fn compute_material_response(
    material: &Material,
    light_dir: &Vector3<f32>,
    view_dir: &Vector3<f32>,
    surface_normal: &Vector3<f32>,
) -> Vector3<f32> {
    match material.material_type {
        MaterialType::Phong => {
            let n_dot_l = surface_normal.dot(light_dir).max(0.0);
            if n_dot_l <= 0.0 {
                return material.emissive;
            }
            let diffuse = material.base_color * material.diffuse_intensity * n_dot_l;
            let halfway_dir = (light_dir + view_dir).normalize();
            let n_dot_h = surface_normal.dot(&halfway_dir).max(0.0);
            let spec_intensity = n_dot_h.powf(material.shininess);
            let specular = material.specular * material.specular_intensity * spec_intensity;
            diffuse + specular + material.emissive
        }
        MaterialType::PBR => {
            let base_color = material.base_color;
            let metallic = material.metallic;
            let roughness = material.roughness;
            let ao = material.ambient_occlusion;

            let l = *light_dir;
            let v = *view_dir;
            let h = (l + v).normalize();

            let n_dot_l = surface_normal.dot(&l).max(0.0);
            let n_dot_v = surface_normal.dot(&v).max(0.0);
            let n_dot_h = surface_normal.dot(&h).max(0.0);
            let h_dot_v = h.dot(&v).max(0.0);

            if n_dot_l <= 0.0 {
                return material.emissive;
            }

            // 标准PBR F0计算
            let f0_dielectric = Vector3::new(0.04, 0.04, 0.04);
            let f0 = f0_dielectric.lerp(&base_color, metallic);

            let d = pbr::distribution_ggx(n_dot_h, roughness);
            let g = pbr::geometry_smith(n_dot_v, n_dot_l, roughness);
            let f = pbr::fresnel_schlick(h_dot_v, f0);

            let numerator = d * g * f;
            let denominator = 4.0 * n_dot_v * n_dot_l;
            let specular = numerator / denominator.max(0.001);

            let k_s = f;
            let k_d = (Vector3::new(1.0, 1.0, 1.0) - k_s) * (1.0 - metallic);
            let diffuse = k_d.component_mul(&base_color) / std::f32::consts::PI;

            // 标准Cook-Torrance BRDF
            let brdf_result = (diffuse + specular) * n_dot_l * ao;
            brdf_result + material.emissive
        }
    }
}

pub mod pbr {
    use nalgebra::Vector3;

    pub fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
        let alpha = roughness * roughness;
        let alpha2 = alpha * alpha;
        let n_dot_h2 = n_dot_h * n_dot_h;
        let numerator = alpha2;
        let denominator = n_dot_h2 * (alpha2 - 1.0) + 1.0;
        let denominator = std::f32::consts::PI * denominator * denominator;
        numerator / denominator.max(0.0001)
    }

    pub fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
        let r = roughness + 1.0;
        let k = (r * r) / 8.0;
        n_dot_v / (n_dot_v * (1.0 - k) + k)
    }

    pub fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
        let ggx1 = geometry_schlick_ggx(n_dot_v, roughness);
        let ggx2 = geometry_schlick_ggx(n_dot_l, roughness);
        ggx1 * ggx2
    }

    pub fn fresnel_schlick(cos_theta: f32, f0: Vector3<f32>) -> Vector3<f32> {
        let cos_theta = cos_theta.clamp(0.0, 1.0);
        let one_minus_cos_theta = 1.0 - cos_theta;
        let one_minus_cos_theta5 = one_minus_cos_theta.powi(5);
        f0 + (Vector3::new(1.0, 1.0, 1.0) - f0) * one_minus_cos_theta5
    }
}

/// 材质参数应用（统一接口）
pub fn apply_material_parameters(model: &mut Model, args: &RenderSettings) {
    for material in &mut model.materials {
        match material.material_type {
            MaterialType::PBR => {
                material.metallic = args.metallic.clamp(0.0, 1.0);
                material.roughness = args.roughness.clamp(0.0, 1.0);
                material.ambient_occlusion = args.ambient_occlusion.clamp(0.0, 1.0);
                material.alpha = args.alpha.clamp(0.0, 1.0);

                if let Ok(base_color) = parse_vec3(&args.base_color) {
                    material.base_color = base_color;
                } else {
                    warn!("无法解析基础颜色, 使用默认值: {:?}", material.base_color);
                }

                if let Ok(emissive) = parse_vec3(&args.emissive) {
                    material.emissive = emissive;
                }

                let ambient_response = material.ambient_occlusion * (1.0 - material.metallic);
                material.ambient_factor =
                    Vector3::new(ambient_response, ambient_response, ambient_response);
            }
            MaterialType::Phong => {
                if let Ok(specular_color) = parse_vec3(&args.specular_color) {
                    material.specular = specular_color;
                } else {
                    warn!("无法解析镜面反射颜色, 使用默认值: {:?}", material.specular);
                }

                material.shininess = args.shininess.max(1.0);
                material.diffuse_intensity = args.diffuse_intensity.clamp(0.0, 2.0);
                material.specular_intensity = args.specular_intensity.clamp(0.0, 2.0);
                material.alpha = args.alpha.clamp(0.0, 1.0);

                if let Ok(diffuse_color) = parse_vec3(&args.diffuse_color) {
                    material.base_color = diffuse_color;
                } else {
                    warn!("无法解析漫反射颜色, 使用默认值: {:?}", material.base_color);
                }

                if let Ok(emissive) = parse_vec3(&args.emissive) {
                    material.emissive = emissive;
                }

                material.ambient_factor = material.base_color * 0.3;
            }
        }
    }
}
pub mod scene_object;
pub mod scene_utils;
use crate::geometry::transform::TransformFactory;
use crate::material_system::materials::Model;
use nalgebra::{Matrix4, Vector3};

/// 表示场景中的单个对象实例
///
/// 包含几何数据（模型）和变换信息，是渲染器的基本单位
#[derive(Debug, Clone)]
pub struct SceneObject {
    /// 对象的几何数据（网格、材质等）
    pub model: Model,

    /// 对象在世界空间中的变换矩阵
    pub transform: Matrix4<f32>,
}

impl SceneObject {
    /// 从模型数据创建新的场景对象
    pub fn from_model_data(model: Model) -> Self {
        Self {
            model,
            transform: Matrix4::identity(),
        }
    }

    /// 创建空的场景对象（用于测试或占位）
    pub fn empty(name: &str) -> Self {
        Self {
            model: Model {
                meshes: Vec::new(),
                materials: Vec::new(),
                name: name.to_string(),
            },
            transform: Matrix4::identity(),
        }
    }

    /// 设置完整变换（从组件构建变换矩阵）
    pub fn set_transform_from_components(
        &mut self,
        position: Vector3<f32>,
        rotation_rad: Vector3<f32>,
        scale: Vector3<f32>,
    ) {
        // 按正确顺序组合变换：缩放 -> 旋转 -> 平移
        let scale_matrix = TransformFactory::scaling_nonuniform(&scale);
        let rotation_x_matrix = TransformFactory::rotation_x(rotation_rad.x);
        let rotation_y_matrix = TransformFactory::rotation_y(rotation_rad.y);
        let rotation_z_matrix = TransformFactory::rotation_z(rotation_rad.z);
        let translation_matrix = TransformFactory::translation(&position);

        // 组合变换矩阵：T * Rz * Ry * Rx * S
        self.transform = translation_matrix
            * rotation_z_matrix
            * rotation_y_matrix
            * rotation_x_matrix
            * scale_matrix;
    }

    /// 应用增量旋转（用于动画）
    pub fn rotate(&mut self, axis: &Vector3<f32>, angle_rad: f32) {
        let rotation_matrix = TransformFactory::rotation(axis, angle_rad);
        self.transform = rotation_matrix * self.transform;
    }
}

impl Default for SceneObject {
    fn default() -> Self {
        Self::empty("Default")
    }
}
use crate::geometry::camera::Camera;
use crate::io::render_settings::{RenderSettings, parse_point3, parse_vec3};
use crate::material_system::light::Light;
use crate::material_system::materials::Model;
use crate::material_system::materials::apply_material_parameters;
use crate::scene::scene_object::SceneObject;
use nalgebra::Vector3;

/// 表示一个 3D 场景，包含对象、光源和相机
#[derive(Debug, Clone)]
pub struct Scene {
    /// 场景中的主要对象（简化为单个对象）
    pub object: SceneObject,

    /// 场景中的光源
    pub lights: Vec<Light>,

    /// 当前活动相机
    pub active_camera: Camera,

    /// 环境光强度
    pub ambient_intensity: f32,

    /// 环境光颜色
    pub ambient_color: Vector3<f32>,
}

impl Scene {
    /// 链式创建场景，自动应用所有设置
    pub fn new(model_data: Model, settings: &RenderSettings) -> Result<Self, String> {
        let mut model_data = model_data.clone();
        // 应用材质参数
        apply_material_parameters(&mut model_data, settings);

        // 创建对象
        let mut object = SceneObject::from_model_data(model_data);

        // 应用对象变换
        let (position, rotation_rad, scale) = settings.get_object_transform_components();
        let final_scale = if settings.object_scale != 1.0 {
            scale * settings.object_scale
        } else {
            scale
        };
        object.set_transform_from_components(position, rotation_rad, final_scale);

        // 相机
        let aspect_ratio = settings.width as f32 / settings.height as f32;
        let camera_from =
            parse_point3(&settings.camera_from).map_err(|e| format!("无效的相机位置格式: {e}"))?;
        let camera_at =
            parse_point3(&settings.camera_at).map_err(|e| format!("无效的相机目标格式: {e}"))?;
        let camera_up =
            parse_vec3(&settings.camera_up).map_err(|e| format!("无效的相机上方向格式: {e}"))?;
        let camera = match settings.projection.as_str() {
            "perspective" => Camera::perspective(
                camera_from,
                camera_at,
                camera_up,
                settings.camera_fov,
                aspect_ratio,
                0.1,
                100.0,
            ),
            "orthographic" => {
                let height = 4.0;
                let width = height * aspect_ratio;
                Camera::orthographic(camera_from, camera_at, camera_up, width, height, 0.1, 100.0)
            }
            _ => return Err(format!("不支持的投影类型: {}", settings.projection)),
        };

        // 光源
        let lights = settings.lights.clone();

        // 环境光
        let ambient_intensity = settings.ambient;
        let ambient_color = settings.get_ambient_color_vec();

        Ok(Scene {
            object,
            lights,
            active_camera: camera,
            ambient_intensity,
            ambient_color,
        })
    }

    /// 链式设置对象变换
    pub fn set_object_transform(
        &mut self,
        position: Vector3<f32>,
        rotation_rad: Vector3<f32>,
        scale: Vector3<f32>,
    ) -> &mut Self {
        self.object
            .set_transform_from_components(position, rotation_rad, scale);
        self
    }

    /// 链式设置光源
    pub fn set_lights(&mut self, lights: Vec<Light>) -> &mut Self {
        self.lights = lights;
        self
    }

    /// 链式设置相机
    pub fn set_camera(&mut self, camera: Camera) -> &mut Self {
        self.active_camera = camera;
        self
    }

    /// 链式设置环境光
    pub fn set_ambient(&mut self, intensity: f32, color: Vector3<f32>) -> &mut Self {
        self.ambient_intensity = intensity;
        self.ambient_color = color;
        self
    }

    /// 获取场景统计信息
    pub fn get_scene_stats(&self) -> SceneStats {
        let mut vertex_count = 0;
        let mut triangle_count = 0;
        let material_count = self.object.model.materials.len();
        let mesh_count = self.object.model.meshes.len();

        for mesh in &self.object.model.meshes {
            vertex_count += mesh.vertices.len();
            triangle_count += mesh.indices.len() / 3;
        }

        SceneStats {
            vertex_count,
            triangle_count,
            material_count,
            mesh_count,
            light_count: self.lights.len(),
        }
    }
}

/// 场景统计信息
#[derive(Debug, Clone)]
pub struct SceneStats {
    pub vertex_count: usize,
    pub triangle_count: usize,
    pub material_count: usize,
    pub mesh_count: usize,
    pub light_count: usize,
}
use crate::ModelLoader;
use crate::core::renderer::Renderer;
use crate::geometry::camera::ProjectionType;
use crate::io::render_settings::{RenderSettings, parse_point3, parse_vec3};
use crate::material_system::materials::apply_material_parameters;
use crate::ui::app::RasterizerApp;
use crate::utils::render_utils::calculate_rotation_parameters;
use crate::utils::save_utils::save_render_with_settings;
use egui::{Color32, Context};
use log::{debug, error, warn};
use std::fs;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use super::app::InterfaceInteraction;

/// 核心业务逻辑方法
///
/// 该trait包含应用的核心功能：
/// - 渲染和加载逻辑
/// - 状态转换与管理
/// - 错误处理
/// - 性能统计
/// - 资源管理
pub trait CoreMethods {
    // === 核心渲染和加载 ===

    /// 渲染当前场景 - 统一渲染入口
    fn render(&mut self, ctx: &Context);

    /// 在UI中显示渲染结果
    fn display_render_result(&mut self, ctx: &Context);

    /// 如果任何事情发生变化，执行重新渲染
    fn render_if_anything_changed(&mut self, ctx: &Context);

    /// 保存当前渲染结果为截图
    fn take_screenshot(&mut self) -> Result<String, String>;

    /// 智能计算地面平面的最佳高度
    fn calculate_optimal_ground_height(&self) -> Option<f32>;

    // === 状态管理 ===

    /// 设置错误信息
    fn set_error(&mut self, message: String);

    /// 将应用状态重置为默认值
    fn reset_to_defaults(&mut self);

    /// 切换预渲染模式开启/关闭状态
    fn toggle_pre_render_mode(&mut self);

    /// 清空预渲染的动画帧缓冲区
    fn clear_pre_rendered_frames(&mut self);

    // === 状态查询 ===

    /// 检查是否可以清除预渲染缓冲区
    fn can_clear_buffer(&self) -> bool;

    /// 检查是否可以切换预渲染模式
    fn can_toggle_pre_render(&self) -> bool;

    /// 检查是否可以开始或停止动画渲染
    fn can_render_animation(&self) -> bool;

    /// 检查是否可以生成视频
    fn can_generate_video(&self) -> bool;

    // === 动画状态管理 ===

    /// 开始实时渲染动画
    fn start_animation_rendering(&mut self) -> Result<(), String>;

    /// 停止实时渲染动画
    fn stop_animation_rendering(&mut self);

    // === 性能统计 ===

    /// 更新帧率统计信息
    fn update_fps_stats(&mut self, frame_time: Duration);

    /// 获取格式化的帧率显示文本和颜色
    fn get_fps_display(&self) -> (String, Color32);

    // === 资源管理 ===

    /// 执行资源清理操作
    fn cleanup_resources(&mut self);
}

impl CoreMethods for RasterizerApp {
    // === 核心渲染和加载实现 ===

    /// 渲染当前场景
    fn render(&mut self, ctx: &Context) {
        // 验证参数
        if let Err(e) = self.settings.validate() {
            self.set_error(e);
            return;
        }

        // 获取OBJ路径
        let obj_path = match &self.settings.obj {
            Some(path) => path.clone(),
            None => {
                self.set_error("错误: 未指定OBJ文件路径".to_string());
                return;
            }
        };

        self.status_message = format!("正在加载 {obj_path}...");
        ctx.request_repaint(); // 立即更新状态消息

        // 加载模型
        match ModelLoader::load_and_create_scene(&obj_path, &self.settings) {
            Ok((scene, model_data)) => {
                debug!(
                    "场景创建完成: 光源数量={}, 使用光照={}, 环境光强度={}",
                    scene.lights.len(),
                    self.settings.use_lighting,
                    self.settings.ambient
                );

                // 直接设置场景和模型数据
                self.scene = Some(scene);
                self.model_data = Some(model_data);

                self.status_message = "模型加载成功，开始渲染...".to_string();
            }
            Err(e) => {
                self.set_error(format!("加载模型失败: {e}"));
                return;
            }
        }

        self.status_message = "模型加载成功，开始渲染...".to_string();
        ctx.request_repaint();

        // 确保输出目录存在
        let output_dir = self.settings.output_dir.clone();
        if let Err(e) = fs::create_dir_all(&output_dir) {
            self.set_error(format!("创建输出目录失败: {e}"));
            return;
        }

        // 渲染
        let start_time = Instant::now();

        if let Some(scene) = &mut self.scene {
            // 渲染到帧缓冲区
            self.renderer.render_scene(scene, &self.settings);

            // 保存输出文件
            if let Err(e) = save_render_with_settings(&self.renderer, &self.settings, None) {
                warn!("保存渲染结果时发生错误: {e}");
            }

            // 更新状态
            self.last_render_time = Some(start_time.elapsed());
            let output_dir = self.settings.output_dir.clone();
            let output_name = self.settings.output.clone();
            let elapsed = self.last_render_time.unwrap();
            self.status_message =
                format!("渲染完成，耗时 {elapsed:.2?}，已保存到 {output_dir}/{output_name}");

            // 在UI中显示渲染结果
            self.display_render_result(ctx);
        }
    }

    /// 在UI中显示渲染结果
    fn display_render_result(&mut self, ctx: &Context) {
        // 从渲染器获取图像数据
        let color_data = self.renderer.frame_buffer.get_color_buffer_bytes();

        // 确保分辨率与渲染器匹配
        let width = self.renderer.frame_buffer.width;
        let height = self.renderer.frame_buffer.height;

        // 创建或更新纹理
        let rendered_texture = self.rendered_image.get_or_insert_with(|| {
            // 创建一个全黑的空白图像
            let color = Color32::BLACK;
            ctx.load_texture(
                "rendered_image",
                egui::ColorImage::new([width, height], vec![color; width * height]),
                egui::TextureOptions::default(),
            )
        });

        // 将RGB数据转换为RGBA格式
        let mut rgba_data = Vec::with_capacity(color_data.len() / 3 * 4);
        for i in (0..color_data.len()).step_by(3) {
            if i + 2 < color_data.len() {
                rgba_data.push(color_data[i]); // R
                rgba_data.push(color_data[i + 1]); // G
                rgba_data.push(color_data[i + 2]); // B
                rgba_data.push(255); // A (完全不透明)
            }
        }

        // 更新纹理，使用渲染器的实际大小
        rendered_texture.set(
            egui::ColorImage::from_rgba_unmultiplied([width, height], &rgba_data),
            egui::TextureOptions::default(),
        );
    }

    /// 统一同步入口
    fn render_if_anything_changed(&mut self, ctx: &Context) {
        if self.interface_interaction.anything_changed && self.scene.is_some() {
            if let Some(scene) = &mut self.scene {
                // 检测渲染尺寸变化
                if self.renderer.frame_buffer.width != self.settings.width
                    || self.renderer.frame_buffer.height != self.settings.height
                {
                    self.renderer.frame_buffer.invalidate_caches();
                }

                // 强制清除地面本体和阴影缓存
                self.renderer.frame_buffer.invalidate_ground_base_cache();
                self.renderer.frame_buffer.invalidate_ground_shadow_cache();

                // 统一同步所有状态

                // 1. 光源同步
                scene.set_lights(self.settings.lights.clone());

                // 2. 相机同步
                if let Ok(from) = parse_point3(&self.settings.camera_from) {
                    scene.active_camera.params.position = from;
                }
                if let Ok(at) = parse_point3(&self.settings.camera_at) {
                    scene.active_camera.params.target = at;
                }
                if let Ok(up) = parse_vec3(&self.settings.camera_up) {
                    scene.active_camera.params.up = up.normalize();
                }
                if let ProjectionType::Perspective { fov_y_degrees, .. } =
                    &mut scene.active_camera.params.projection
                {
                    *fov_y_degrees = self.settings.camera_fov;
                }
                scene.active_camera.update_matrices();

                // 3. 物体变换同步
                let (position, rotation_rad, scale) =
                    self.settings.get_object_transform_components();
                let final_scale = if self.settings.object_scale != 1.0 {
                    scale * self.settings.object_scale
                } else {
                    scale
                };
                scene.set_object_transform(position, rotation_rad, final_scale);

                // 4. 材质参数同步
                apply_material_parameters(&mut scene.object.model, &self.settings);

                // 5. 环境光同步
                scene.set_ambient(self.settings.ambient, self.settings.get_ambient_color_vec());

                // 6. 执行渲染
                self.renderer.render_scene(scene, &self.settings);
            }

            self.display_render_result(ctx);
            self.interface_interaction.anything_changed = false;
        }
    }

    /// 保存当前渲染结果为截图
    fn take_screenshot(&mut self) -> Result<String, String> {
        // 确保输出目录存在
        if let Err(e) = fs::create_dir_all(&self.settings.output_dir) {
            return Err(format!("创建输出目录失败: {e}"));
        }

        // 生成唯一的文件名（基于时间戳）
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| format!("获取时间戳失败: {e}"))?
            .as_secs();

        let snapshot_name = format!("{}_snapshot_{}", self.settings.output, timestamp);

        // 检查是否有可用的渲染结果
        if self.rendered_image.is_none() {
            return Err("没有可用的渲染结果".to_string());
        }

        // 使用共享的渲染工具函数保存截图
        save_render_with_settings(&self.renderer, &self.settings, Some(&snapshot_name))?;

        // 返回颜色图像的路径
        let color_path =
            Path::new(&self.settings.output_dir).join(format!("{snapshot_name}_color.png"));
        Ok(color_path.to_string_lossy().to_string())
    }

    fn calculate_optimal_ground_height(&self) -> Option<f32> {
        let scene = self.scene.as_ref()?;
        let model_data = self.model_data.as_ref()?;

        let mut min_y = f32::INFINITY;
        let mut has_vertices = false;

        // 计算模型在当前变换下的最低点
        for mesh in &model_data.meshes {
            for vertex in &mesh.vertices {
                let world_pos = scene.object.transform.transform_point(&vertex.position);
                min_y = min_y.min(world_pos.y);
                has_vertices = true;
            }
        }

        if !has_vertices {
            return None;
        }

        // 智能策略：让物体贴地，避免浮空
        let ground_height = if self.settings.enable_shadow_mapping {
            // 阴影映射时：让物体稍微"嵌入"地面，确保阴影可见但物体不浮空
            min_y - 0.01 // 非常小的负偏移，让物体微微贴地
        } else {
            // 无阴影时：让物体完全贴地
            min_y
        };

        Some(ground_height)
    }

    // === 状态管理实现 ===

    /// 设置错误信息
    fn set_error(&mut self, message: String) {
        error!("{message}");
        self.status_message = format!("错误: {message}");
    }

    /// 重置应用状态到默认值
    fn reset_to_defaults(&mut self) {
        // 保留当前的文件路径设置
        let obj_path = self.settings.obj.clone();
        let output_dir = self.settings.output_dir.clone();
        let output_name = self.settings.output.clone();

        let new_settings = RenderSettings {
            obj: obj_path,
            output_dir,
            output: output_name,
            ..Default::default()
        };

        // 如果渲染尺寸变化，重新创建渲染器
        if self.renderer.frame_buffer.width != new_settings.width
            || self.renderer.frame_buffer.height != new_settings.height
        {
            self.renderer = Renderer::new(new_settings.width, new_settings.height);
            self.rendered_image = None;
        }

        self.settings = new_settings;

        // 重置GUI状态
        self.camera_pan_sensitivity = 1.0;
        self.camera_orbit_sensitivity = 1.0;
        self.camera_dolly_sensitivity = 1.0;
        self.interface_interaction = InterfaceInteraction::default();

        // 重置其他状态
        self.is_realtime_rendering = false;
        self.is_pre_rendering = false;
        self.is_generating_video = false;
        self.pre_render_mode = false;
        self.animation_time = 0.0;
        self.current_frame_index = 0;
        self.last_frame_time = None;

        // 清空预渲染缓冲区
        if let Ok(mut frames) = self.pre_rendered_frames.lock() {
            frames.clear();
        }

        self.pre_render_progress.store(0, Ordering::SeqCst);
        self.video_progress.store(0, Ordering::SeqCst);

        // 重置 FPS 统计
        self.current_fps = 0.0;
        self.fps_history.clear();
        self.avg_fps = 0.0;

        self.status_message = "已重置应用状态，光源已恢复默认设置".to_string();
    }

    /// 切换预渲染模式
    fn toggle_pre_render_mode(&mut self) {
        // 统一的状态检查
        if self.is_pre_rendering || self.is_generating_video || self.is_realtime_rendering {
            self.status_message = "无法更改渲染模式: 请先停止正在进行的操作".to_string();
            return;
        }

        // 切换模式
        self.pre_render_mode = !self.pre_render_mode;

        if self.pre_render_mode {
            // 确保旋转速度合理
            if self.settings.rotation_speed.abs() < 0.01 {
                self.settings.rotation_speed = 1.0;
            }
            self.status_message = "已启用预渲染模式，开始动画渲染时将预先计算所有帧".to_string();
        } else {
            self.status_message = "已禁用预渲染模式，缓冲区中的预渲染帧仍可使用".to_string();
        }
    }

    /// 清空预渲染帧缓冲区
    fn clear_pre_rendered_frames(&mut self) {
        // 统一的状态检查逻辑
        if self.is_realtime_rendering || self.is_pre_rendering {
            self.status_message = "无法清除缓冲区: 请先停止动画渲染或等待预渲染完成".to_string();
            return;
        }

        // 执行清除操作
        let had_frames = !self.pre_rendered_frames.lock().unwrap().is_empty();
        if had_frames {
            self.pre_rendered_frames.lock().unwrap().clear();
            self.current_frame_index = 0;
            self.pre_render_progress.store(0, Ordering::SeqCst);

            if self.is_generating_video {
                let (_, _, frames_per_rotation) =
                    calculate_rotation_parameters(self.settings.rotation_speed, self.settings.fps);
                let total_frames =
                    (frames_per_rotation as f32 * self.settings.rotation_cycles) as usize;
                let progress = self.video_progress.load(Ordering::SeqCst);
                let percent = (progress as f32 / total_frames as f32 * 100.0).round();

                self.status_message =
                    format!("生成视频中... ({progress}/{total_frames}，{percent:.0}%)");
            } else {
                self.status_message = "已清空预渲染缓冲区".to_string();
            }
        } else {
            self.status_message = "缓冲区已为空".to_string();
        }
    }

    // === 状态查询实现 ===

    fn can_clear_buffer(&self) -> bool {
        !self.pre_rendered_frames.lock().unwrap().is_empty()
            && !self.is_realtime_rendering
            && !self.is_pre_rendering
    }

    fn can_toggle_pre_render(&self) -> bool {
        !self.is_pre_rendering && !self.is_generating_video && !self.is_realtime_rendering
    }

    fn can_render_animation(&self) -> bool {
        !self.is_generating_video
    }

    fn can_generate_video(&self) -> bool {
        !self.is_realtime_rendering && !self.is_generating_video && self.ffmpeg_available
    }

    // === 动画状态管理实现 ===

    fn start_animation_rendering(&mut self) -> Result<(), String> {
        if self.is_generating_video {
            return Err("无法开始动画: 视频正在生成中".to_string());
        }

        self.is_realtime_rendering = true;
        self.last_frame_time = None;
        self.current_fps = 0.0;
        self.fps_history.clear();
        self.avg_fps = 0.0;
        self.status_message = "开始动画渲染...".to_string();

        Ok(())
    }

    fn stop_animation_rendering(&mut self) {
        self.is_realtime_rendering = false;
        self.status_message = "已停止动画渲染".to_string();
    }

    // === 性能统计实现 ===

    fn update_fps_stats(&mut self, frame_time: Duration) {
        const FPS_HISTORY_SIZE: usize = 30;
        let current_fps = 1.0 / frame_time.as_secs_f32();
        self.current_fps = current_fps;

        // 更新 FPS 历史
        self.fps_history.push(current_fps);
        if self.fps_history.len() > FPS_HISTORY_SIZE {
            self.fps_history.remove(0); // 移除最早的记录
        }

        // 计算平均 FPS
        if !self.fps_history.is_empty() {
            let sum: f32 = self.fps_history.iter().sum();
            self.avg_fps = sum / self.fps_history.len() as f32;
        }
    }

    fn get_fps_display(&self) -> (String, Color32) {
        // 根据 FPS 水平选择颜色
        let fps_color = if self.avg_fps >= 30.0 {
            Color32::from_rgb(50, 220, 50) // 绿色
        } else if self.avg_fps >= 15.0 {
            Color32::from_rgb(220, 180, 50) // 黄色
        } else {
            Color32::from_rgb(220, 50, 50) // 红色
        };

        (format!("FPS: {:.1}", self.avg_fps), fps_color)
    }

    // === 资源管理实现 ===

    fn cleanup_resources(&mut self) {
        // 实际的资源清理逻辑

        // 1. 限制FPS历史记录大小，防止内存泄漏
        if self.fps_history.len() > 60 {
            self.fps_history.drain(0..30); // 保留最近30帧的数据
        }

        // 2. 清理已完成的视频生成线程
        if let Some(handle) = &self.video_generation_thread {
            if handle.is_finished() {
                // 线程已完成，标记需要在主循环中处理
                debug!("检测到已完成的视频生成线程，等待主循环处理");
            }
        }

        // 3. 在空闲状态下进行额外清理
        if !self.is_realtime_rendering && !self.is_generating_video && !self.is_pre_rendering {
            // 清理可能的临时资源
            if self.rendered_image.is_some() && self.last_render_time.is_none() {
                // 如果有渲染结果但没有最近的渲染时间，说明可能是陈旧的结果
                // 这里可以添加更多清理逻辑
            }

            // 清理预渲染进度计数器（如果没有预渲染帧）
            if self.pre_rendered_frames.lock().unwrap().is_empty() {
                self.pre_render_progress.store(0, Ordering::SeqCst);
            }
        }
    }
}
pub mod animation;
pub mod app;
pub mod core;
pub mod render_ui;
pub mod widgets;
use crate::Renderer;
use crate::io::config_loader::TomlConfigLoader;
use crate::io::model_loader::ModelLoader;
use crate::io::render_settings::RenderSettings;
use crate::ui::app::RasterizerApp;
use log::debug;
use native_dialog::FileDialogBuilder;

/// 渲染UI交互方法的特质
///
/// 该trait专门处理与文件选择和UI交互相关的功能：
/// - 文件选择对话框
/// - 背景图片处理
/// - 输出目录选择
/// - 配置文件管理
pub trait RenderUIMethods {
    /// 选择OBJ文件
    fn select_obj_file(&mut self);

    /// 选择纹理文件
    fn select_texture_file(&mut self);

    /// 选择背景图片
    fn select_background_image(&mut self);

    /// 选择输出目录
    fn select_output_dir(&mut self);

    /// 加载配置文件
    fn load_config_file(&mut self);

    /// 保存配置文件
    fn save_config_file(&mut self);

    /// 应用加载的配置到GUI
    fn apply_loaded_config(&mut self, settings: RenderSettings);
}

impl RenderUIMethods for RasterizerApp {
    /// 选择OBJ文件
    fn select_obj_file(&mut self) {
        let result = FileDialogBuilder::default()
            .set_title("选择OBJ模型文件")
            .add_filter("OBJ模型", ["obj"])
            .open_single_file()
            .show();

        match result {
            Ok(Some(path)) => {
                if let Some(path_str) = path.to_str() {
                    self.settings.obj = Some(path_str.to_string());
                    self.status_message = format!("已选择模型: {path_str}");

                    // OBJ文件变化需要重新加载场景和重新渲染
                    self.interface_interaction.anything_changed = true;
                    self.scene = None; // 清除现有场景，强制重新加载
                    self.rendered_image = None; // 清除渲染结果
                }
            }
            Ok(None) => {
                self.status_message = "文件选择被取消".to_string();
            }
            Err(e) => {
                self.set_error(format!("文件选择器错误: {e}"));
            }
        }
    }

    /// 选择纹理文件
    fn select_texture_file(&mut self) {
        let result = FileDialogBuilder::default()
            .set_title("选择纹理文件")
            .add_filter("图像文件", ["png", "jpg", "jpeg", "bmp", "tga"])
            .open_single_file()
            .show();

        match result {
            Ok(Some(path)) => {
                if let Some(path_str) = path.to_str() {
                    self.settings.texture = Some(path_str.to_string());
                    self.status_message = format!("已选择纹理: {path_str}");

                    // 纹理变化需要重新渲染
                    self.interface_interaction.anything_changed = true;
                }
            }
            Ok(None) => {
                self.status_message = "纹理选择被取消".to_string();
            }
            Err(e) => {
                self.set_error(format!("纹理选择错误: {e}"));
            }
        }
    }

    /// 选择背景图片
    fn select_background_image(&mut self) {
        let result = FileDialogBuilder::default()
            .set_title("选择背景图片")
            .add_filter("图片文件", ["png", "jpg", "jpeg", "bmp"])
            .open_single_file()
            .show();

        match result {
            Ok(Some(path)) => {
                if let Some(path_str) = path.to_str() {
                    // 只设置背景图片路径，不再直接加载到 settings
                    self.settings.background_image_path = Some(path_str.to_string());
                    self.settings.use_background_image = true;

                    // 使用 ModelLoader 验证背景图片是否有效
                    match ModelLoader::validate_resources(&self.settings) {
                        Ok(_) => {
                            self.status_message = format!("背景图片配置成功: {path_str}");

                            // 清除已渲染的图像，强制重新渲染以应用新背景
                            self.rendered_image = None;

                            debug!("背景图片路径已设置: {path_str}");
                            debug!("背景图片将在下次渲染时由 FrameBuffer 自动加载");
                        }
                        Err(e) => {
                            // 验证失败，重置背景设置
                            self.set_error(format!("背景图片验证失败: {e}"));
                            self.settings.background_image_path = None;
                            self.settings.use_background_image = false;
                        }
                    }
                }
            }
            Ok(None) => {
                self.status_message = "图片选择被取消".to_string();
            }
            Err(e) => {
                self.set_error(format!("文件选择器错误: {e}"));
            }
        }
    }

    /// 选择输出目录
    fn select_output_dir(&mut self) {
        let result = FileDialogBuilder::default()
            .set_title("选择输出目录")
            .open_single_dir()
            .show();

        match result {
            Ok(Some(path)) => {
                if let Some(path_str) = path.to_str() {
                    self.settings.output_dir = path_str.to_string();
                    self.status_message = format!("已选择输出目录: {path_str}");
                }
            }
            Ok(None) => {
                self.status_message = "目录选择被取消".to_string();
            }
            Err(e) => {
                self.set_error(format!("目录选择器错误: {e}"));
            }
        }
    }

    /// 加载配置文件
    fn load_config_file(&mut self) {
        let result = FileDialogBuilder::default()
            .set_title("加载配置文件")
            .add_filter("TOML配置文件", ["toml"])
            .open_single_file()
            .show();

        match result {
            Ok(Some(path)) => {
                if let Some(path_str) = path.to_str() {
                    match TomlConfigLoader::load_from_file(path_str) {
                        Ok(loaded_settings) => {
                            self.apply_loaded_config(loaded_settings);
                            self.status_message = format!("配置已加载: {path_str}");
                        }
                        Err(e) => {
                            self.set_error(format!("配置加载失败: {e}"));
                        }
                    }
                }
            }
            Ok(None) => {
                self.status_message = "配置加载被取消".to_string();
            }
            Err(e) => {
                self.set_error(format!("文件选择器错误: {e}"));
            }
        }
    }

    /// 保存配置文件
    fn save_config_file(&mut self) {
        let result = FileDialogBuilder::default()
            .set_title("保存配置文件")
            .add_filter("TOML配置文件", ["toml"])
            .save_single_file()
            .show();

        match result {
            Ok(Some(path)) => {
                let mut save_path = path;

                // 自动添加.toml扩展名（如果没有）
                if save_path.extension().is_none() {
                    save_path.set_extension("toml");
                }

                if let Some(path_str) = save_path.to_str() {
                    match TomlConfigLoader::save_to_file(&self.settings, path_str) {
                        Ok(_) => {
                            self.status_message = format!("配置已保存: {path_str}");
                        }
                        Err(e) => {
                            self.set_error(format!("配置保存失败: {e}"));
                        }
                    }
                }
            }
            Ok(None) => {
                self.status_message = "配置保存被取消".to_string();
            }
            Err(e) => {
                self.set_error(format!("文件选择器错误: {e}"));
            }
        }
    }

    /// 应用加载的配置到GUI
    fn apply_loaded_config(&mut self, loaded_settings: RenderSettings) {
        // 直接替换settings，无需同步GUI专用向量字段
        self.settings = loaded_settings;

        // 如果分辨率变化，重新创建渲染器
        if self.renderer.frame_buffer.width != self.settings.width
            || self.renderer.frame_buffer.height != self.settings.height
        {
            self.renderer = Renderer::new(self.settings.width, self.settings.height);
        }

        // 清除现有场景和渲染结果，强制重新加载
        self.scene = None;
        self.rendered_image = None;
        self.interface_interaction.anything_changed = true;

        debug!("配置已应用到GUI界面");
    }
}
use crate::ModelLoader;
use crate::core::renderer::Renderer;
use crate::io::render_settings::{AnimationType, RenderSettings, get_animation_axis_vector};
use crate::scene::scene_utils::Scene;
use crate::utils::render_utils::{
    animate_scene_step, calculate_rotation_delta, calculate_rotation_parameters,
};
use crate::utils::save_utils::save_image;
use egui::{ColorImage, Context, TextureOptions};
use log::debug;
use std::fs;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use super::app::RasterizerApp;
use super::core::CoreMethods;

/// 将ColorImage转换为PNG数据
pub fn frame_to_png_data(image: &ColorImage) -> Vec<u8> {
    // ColorImage是RGBA格式，我们需要转换为RGB格式
    let mut rgb_data = Vec::with_capacity(image.width() * image.height() * 3);
    for pixel in &image.pixels {
        rgb_data.push(pixel.r());
        rgb_data.push(pixel.g());
        rgb_data.push(pixel.b());
    }
    rgb_data
}

/// 渲染一圈的动画帧
///
/// # 参数
/// * `scene_copy` - 场景的克隆
/// * `settings` - 渲染参数
/// * `progress_arc` - 进度计数器
/// * `ctx_clone` - UI上下文，用于更新界面
/// * `width` - 渲染宽度
/// * `height` - 渲染高度
/// * `on_frame_rendered` - 帧渲染完成后的回调函数，参数为(帧序号, RGB颜色数据)
///
/// # 返回值
/// 渲染的总帧数
fn render_one_rotation_cycle<F>(
    mut scene_copy: Scene,
    settings: &RenderSettings,
    progress_arc: &Arc<AtomicUsize>,
    ctx_clone: &Context,
    width: usize,
    height: usize,
    mut on_frame_rendered: F,
) -> usize
where
    F: FnMut(usize, Vec<u8>),
{
    let mut thread_renderer = Renderer::new(width, height);
    let (effective_rotation_speed_dps, _, frames_to_render) =
        calculate_rotation_parameters(settings.rotation_speed, settings.fps);

    let rotation_axis_vec = get_animation_axis_vector(settings);
    let rotation_increment_rad_per_frame =
        (360.0 / frames_to_render as f32).to_radians() * effective_rotation_speed_dps.signum();

    for frame_num in 0..frames_to_render {
        progress_arc.store(frame_num, Ordering::SeqCst);

        if frame_num > 0 {
            animate_scene_step(
                &mut scene_copy,
                &settings.animation_type,
                &rotation_axis_vec,
                rotation_increment_rad_per_frame,
            );
        }

        // === 缓存失效策略 ===
        match settings.animation_type {
            AnimationType::CameraOrbit => {
                // 相机轨道动画：地面本体和阴影都依赖相机，必须全部失效
                thread_renderer.frame_buffer.invalidate_ground_base_cache();
                thread_renderer
                    .frame_buffer
                    .invalidate_ground_shadow_cache();
            }
            AnimationType::ObjectLocalRotation => {
                if settings.enable_shadow_mapping {
                    // 物体动画+阴影：只需失效阴影缓存，地面本体可复用
                    thread_renderer
                        .frame_buffer
                        .invalidate_ground_shadow_cache();
                }
                // 未开阴影时，地面缓存可复用，无需清理
            }
            AnimationType::None => {}
        }

        thread_renderer.render_scene(&mut scene_copy, settings);
        let color_data_rgb = thread_renderer.frame_buffer.get_color_buffer_bytes();
        on_frame_rendered(frame_num, color_data_rgb);

        if frame_num % (frames_to_render.max(1) / 20).max(1) == 0 {
            ctx_clone.request_repaint();
        }
    }

    progress_arc.store(frames_to_render, Ordering::SeqCst);
    ctx_clone.request_repaint();

    frames_to_render
}

/// 动画与视频生成相关方法的特质
pub trait AnimationMethods {
    /// 执行实时渲染循环
    fn perform_realtime_rendering(&mut self, ctx: &Context);

    /// 在后台生成视频
    fn start_video_generation(&mut self, ctx: &Context);

    /// 启动预渲染过程
    fn start_pre_rendering(&mut self, ctx: &Context);

    /// 处理预渲染帧
    fn handle_pre_rendering_tasks(&mut self, ctx: &Context);

    /// 播放预渲染帧
    fn play_pre_rendered_frames(&mut self, ctx: &Context);
}

impl AnimationMethods for RasterizerApp {
    /// 执行实时渲染循环
    fn perform_realtime_rendering(&mut self, ctx: &Context) {
        // 如果启用了预渲染模式且没有预渲染帧，才进入预渲染
        if self.pre_render_mode
            && !self.is_pre_rendering
            && self.pre_rendered_frames.lock().unwrap().is_empty()
        {
            // 检查模型是否已加载
            if self.scene.is_none() {
                let obj_path = match &self.settings.obj {
                    Some(path) => path.clone(),
                    None => {
                        self.set_error("错误: 未指定OBJ文件路径".to_string());
                        self.stop_animation_rendering();
                        return;
                    }
                };
                match ModelLoader::load_and_create_scene(&obj_path, &self.settings) {
                    Ok((scene, model_data)) => {
                        self.scene = Some(scene);
                        self.model_data = Some(model_data);
                        self.start_pre_rendering(ctx);
                        return;
                    }
                    Err(e) => {
                        self.set_error(format!("加载模型失败: {e}"));
                        self.stop_animation_rendering();
                        return;
                    }
                }
            } else {
                self.start_pre_rendering(ctx);
                return;
            }
        }

        // 如果正在预渲染，处理预渲染任务
        if self.is_pre_rendering {
            self.handle_pre_rendering_tasks(ctx);
            return;
        }

        // 如果启用预渲染模式且有预渲染帧，播放预渲染帧
        if self.pre_render_mode && !self.pre_rendered_frames.lock().unwrap().is_empty() {
            self.play_pre_rendered_frames(ctx);
            return;
        }

        // === 常规实时渲染（未启用预渲染模式或预渲染复选框未勾选） ===

        // 确保场景已加载
        if self.scene.is_none() {
            let obj_path = match &self.settings.obj {
                Some(path) => path.clone(),
                None => {
                    self.set_error("错误: 未指定OBJ文件路径".to_string());
                    self.stop_animation_rendering();
                    return;
                }
            };
            match ModelLoader::load_and_create_scene(&obj_path, &self.settings) {
                Ok((scene, model_data)) => {
                    self.scene = Some(scene);
                    self.model_data = Some(model_data);
                    // 注意：这里不再自动跳转到预渲染，而是继续执行实时渲染
                    self.status_message = "模型加载成功，开始实时渲染...".to_string();
                }
                Err(e) => {
                    self.set_error(format!("加载模型失败: {e}"));
                    self.stop_animation_rendering();
                    return;
                }
            }
        }

        // 检查渲染器尺寸，但避免不必要的缓存清除
        if self.renderer.frame_buffer.width != self.settings.width
            || self.renderer.frame_buffer.height != self.settings.height
        {
            self.renderer
                .resize(self.settings.width, self.settings.height);
            self.rendered_image = None;
            debug!(
                "重新创建渲染器，尺寸: {}x{}",
                self.settings.width, self.settings.height
            );
        }

        let now = Instant::now();
        let dt = if let Some(last_time) = self.last_frame_time {
            now.duration_since(last_time).as_secs_f32()
        } else {
            1.0 / 60.0 // 默认 dt
        };
        if let Some(last_time) = self.last_frame_time {
            let frame_time = now.duration_since(last_time);
            self.update_fps_stats(frame_time);
        }
        self.last_frame_time = Some(now);

        if self.is_realtime_rendering && self.settings.rotation_speed.abs() < 0.01 {
            self.settings.rotation_speed = 1.0; // 确保实时渲染时有旋转速度
        }

        self.animation_time += dt;

        if let Some(scene) = &mut self.scene {
            // 动画过程中不清除缓存
            // 物体动画不影响背景和地面（相机不动），所以缓存仍然有效

            // 使用通用函数计算旋转增量
            let rotation_delta_rad = calculate_rotation_delta(self.settings.rotation_speed, dt);
            let rotation_axis_vec = get_animation_axis_vector(&self.settings);

            // 使用通用函数执行动画步骤
            animate_scene_step(
                scene,
                &self.settings.animation_type,
                &rotation_axis_vec,
                rotation_delta_rad,
            );

            debug!(
                "实时渲染中: FPS={:.1}, 动画类型={:?}, 轴={:?}, 旋转速度={}, 角度增量={:.3}rad, Phong={}",
                self.avg_fps,
                self.settings.animation_type,
                self.settings.rotation_axis,
                self.settings.rotation_speed,
                rotation_delta_rad,
                self.settings.use_phong
            );

            match self.settings.animation_type {
                AnimationType::CameraOrbit => {
                    // 相机轨道动画：地面本体和阴影都依赖相机，必须全部失效
                    self.renderer.frame_buffer.invalidate_ground_base_cache();
                    self.renderer.frame_buffer.invalidate_ground_shadow_cache();
                }
                AnimationType::ObjectLocalRotation => {
                    if self.settings.enable_shadow_mapping {
                        // 物体动画+阴影：只需失效阴影缓存，地面本体可复用
                        self.renderer.frame_buffer.invalidate_ground_shadow_cache();
                    }
                    // 未开阴影时，地面缓存可复用，无需清理
                }
                AnimationType::None => {}
            }

            self.renderer.render_scene(scene, &self.settings);
            self.display_render_result(ctx);
            ctx.request_repaint();
        }
    }

    fn start_video_generation(&mut self, ctx: &Context) {
        if !self.ffmpeg_available {
            self.set_error("无法生成视频：未检测到ffmpeg。请安装ffmpeg后重试。".to_string());
            return;
        }
        if self.is_generating_video {
            self.status_message = "视频已在生成中，请等待完成...".to_string();
            return;
        }

        // 使用 CoreMethods 验证参数
        match self.settings.validate() {
            Ok(_) => {
                let output_dir = self.settings.output_dir.clone();
                if let Err(e) = fs::create_dir_all(&output_dir) {
                    self.set_error(format!("创建输出目录失败: {e}"));
                    return;
                }
                let frames_dir = format!(
                    "{}/temp_frames_{}",
                    output_dir,
                    chrono::Utc::now().timestamp_millis()
                );
                if let Err(e) = fs::create_dir_all(&frames_dir) {
                    self.set_error(format!("创建帧目录失败: {e}"));
                    return;
                }

                // 计算旋转参数，获取视频帧数
                let (_, _, frames_per_rotation) =
                    calculate_rotation_parameters(self.settings.rotation_speed, self.settings.fps);

                let total_frames =
                    (frames_per_rotation as f32 * self.settings.rotation_cycles) as usize;

                // 如果场景未加载，尝试加载
                if self.scene.is_none() {
                    let obj_path = match &self.settings.obj {
                        Some(path) => path.clone(),
                        None => {
                            self.set_error("错误: 未指定OBJ文件路径".to_string());
                            return;
                        }
                    };
                    match ModelLoader::load_and_create_scene(&obj_path, &self.settings) {
                        Ok((scene, model_data)) => {
                            self.scene = Some(scene);
                            self.model_data = Some(model_data);
                            self.status_message = "模型加载成功，开始生成视频...".to_string();
                        }
                        Err(e) => {
                            self.set_error(format!("加载模型失败，无法生成视频: {e}"));
                            return;
                        }
                    }
                }

                let settings_for_thread = self.settings.clone();
                let video_progress_arc = self.video_progress.clone();
                let fps = self.settings.fps;
                let scene_clone = self.scene.as_ref().expect("场景已检查").clone();

                // 检查是否有预渲染帧
                let has_pre_rendered_frames = {
                    let frames_guard = self.pre_rendered_frames.lock().unwrap();
                    !frames_guard.is_empty()
                };

                // 如果没有预渲染帧，那么我们需要同时为预渲染缓冲区生成帧
                let frames_for_pre_render = if !has_pre_rendered_frames {
                    Some(self.pre_rendered_frames.clone())
                } else {
                    None
                };

                // 设置渲染状态
                self.is_generating_video = true;
                video_progress_arc.store(0, Ordering::SeqCst);

                // 更新状态消息
                self.status_message = format!(
                    "开始生成视频 (0/{} 帧，{:.1} 秒时长)...",
                    total_frames,
                    total_frames as f32 / fps as f32
                );

                ctx.request_repaint();
                let ctx_clone = ctx.clone();
                let video_filename = format!("{}.mp4", settings_for_thread.output);
                let video_output_path = format!("{output_dir}/{video_filename}");
                let frames_dir_clone = frames_dir.clone();

                // 如果有预渲染帧，复制到线程中
                let pre_rendered_frames_clone = if has_pre_rendered_frames {
                    let frames_guard = self.pre_rendered_frames.lock().unwrap();
                    Some(frames_guard.clone())
                } else {
                    None
                };

                let thread_handle = thread::spawn(move || {
                    let width = settings_for_thread.width;
                    let height = settings_for_thread.height;
                    let mut rendered_frames = Vec::new();

                    // 使用预渲染帧或重新渲染
                    if let Some(frames) = pre_rendered_frames_clone {
                        // 使用预渲染帧
                        let pre_rendered_count = frames.len();

                        for frame_num in 0..total_frames {
                            video_progress_arc.store(frame_num, Ordering::SeqCst);

                            // 计算当前帧在哪个圈和圈内的位置
                            let cycle_position = frame_num % frames_per_rotation;

                            // 将圈内位置映射到预渲染帧索引
                            // 这处理了预渲染帧数量可能与理论帧数不匹配的情况
                            let pre_render_idx =
                                (cycle_position * pre_rendered_count) / frames_per_rotation;

                            let frame = &frames[pre_render_idx.min(pre_rendered_count - 1)]; // 避免越界访问

                            // 将ColorImage转换为PNG并保存
                            let frame_path = format!("{frames_dir_clone}/frame_{frame_num:04}.png");
                            let color_data = frame_to_png_data(frame);
                            save_image(&frame_path, &color_data, width as u32, height as u32);

                            if frame_num % (total_frames.max(1) / 20).max(1) == 0 {
                                ctx_clone.request_repaint();
                            }
                        }
                    } else {
                        // 使用通用渲染函数渲染一圈或部分圈
                        let frames_arc = frames_for_pre_render.clone();

                        let rendered_frame_count = render_one_rotation_cycle(
                            scene_clone,
                            &settings_for_thread,
                            &video_progress_arc,
                            &ctx_clone,
                            width,
                            height,
                            |frame_num, color_data_rgb| {
                                // 保存RGB数据用于后续复用
                                rendered_frames.push(color_data_rgb.clone());

                                // 同时为视频保存PNG文件
                                let frame_path =
                                    format!("{frames_dir_clone}/frame_{frame_num:04}.png");
                                save_image(
                                    &frame_path,
                                    &color_data_rgb,
                                    width as u32,
                                    height as u32,
                                );

                                // 如果需要同时保存到预渲染缓冲区
                                if let Some(ref frames_arc) = frames_arc {
                                    // 转换为RGBA格式以用于预渲染帧
                                    let mut rgba_data = Vec::with_capacity(width * height * 4);
                                    for chunk in color_data_rgb.chunks_exact(3) {
                                        rgba_data.extend_from_slice(chunk);
                                        rgba_data.push(255); // Alpha
                                    }
                                    let color_image = ColorImage::from_rgba_unmultiplied(
                                        [width, height],
                                        &rgba_data,
                                    );
                                    frames_arc.lock().unwrap().push(color_image);
                                }
                            },
                        );

                        // 如果需要多于一圈，使用前面渲染的帧复用
                        if rendered_frame_count < total_frames {
                            for frame_num in rendered_frame_count..total_frames {
                                video_progress_arc.store(frame_num, Ordering::SeqCst);

                                // 复用之前渲染的帧
                                let source_frame_idx = frame_num % rendered_frame_count;
                                let source_data = &rendered_frames[source_frame_idx];

                                // 保存为图片文件
                                let frame_path =
                                    format!("{frames_dir_clone}/frame_{frame_num:04}.png");
                                save_image(&frame_path, source_data, width as u32, height as u32);

                                if frame_num % (total_frames.max(1) / 20).max(1) == 0 {
                                    ctx_clone.request_repaint();
                                }
                            }
                        }
                    }

                    video_progress_arc.store(total_frames, Ordering::SeqCst);
                    ctx_clone.request_repaint();

                    // 使用ffmpeg将帧序列合成为视频，并解决阻塞问题
                    let frames_pattern = format!("{frames_dir_clone}/frame_%04d.png");
                    let ffmpeg_status = std::process::Command::new("ffmpeg")
                        .args([
                            "-y",
                            "-framerate",
                            &fps.to_string(),
                            "-i",
                            &frames_pattern,
                            "-c:v",
                            "libx264",
                            "-pix_fmt",
                            "yuv420p",
                            "-crf",
                            "23",
                            &video_output_path,
                        ])
                        .status();

                    let success = ffmpeg_status.is_ok_and(|s| s.success());

                    // 视频生成后清理临时文件
                    let _ = fs::remove_dir_all(&frames_dir_clone);

                    (success, video_output_path)
                });

                self.video_generation_thread = Some(thread_handle);
            }
            Err(e) => self.set_error(e),
        }
    }

    fn start_pre_rendering(&mut self, ctx: &Context) {
        if self.is_pre_rendering {
            return;
        }

        // 使用 CoreMethods 验证参数
        match self.settings.validate() {
            Ok(_) => {
                if self.scene.is_none() {
                    let obj_path = match &self.settings.obj {
                        Some(path) => path.clone(),
                        None => {
                            self.set_error("错误: 未指定OBJ文件路径".to_string());
                            self.stop_animation_rendering();
                            return;
                        }
                    };
                    match ModelLoader::load_and_create_scene(&obj_path, &self.settings) {
                        Ok((scene, model_data)) => {
                            self.scene = Some(scene);
                            self.model_data = Some(model_data);
                            self.status_message = "模型加载成功，开始预渲染...".to_string();
                        }
                        Err(e) => {
                            self.set_error(format!("加载模型失败，无法预渲染: {e}"));
                            return;
                        }
                    }
                }

                // 使用通用函数计算旋转参数
                let (_, seconds_per_rotation, frames_to_render) =
                    calculate_rotation_parameters(self.settings.rotation_speed, self.settings.fps);

                self.total_frames_for_pre_render_cycle = frames_to_render;

                self.is_pre_rendering = true;
                self.pre_rendered_frames.lock().unwrap().clear();
                self.pre_render_progress.store(0, Ordering::SeqCst);
                self.current_frame_index = 0;

                let settings_for_thread = self.settings.clone();
                let progress_arc = self.pre_render_progress.clone();
                let frames_arc = self.pre_rendered_frames.clone();
                let width = settings_for_thread.width;
                let height = settings_for_thread.height;
                let scene_clone = self.scene.as_ref().expect("场景已检查存在").clone();

                self.status_message = format!(
                    "开始预渲染动画 (0/{frames_to_render} 帧，转一圈需 {seconds_per_rotation:.1} 秒)..."
                );
                ctx.request_repaint();
                let ctx_clone = ctx.clone();

                thread::spawn(move || {
                    // 使用通用渲染函数
                    render_one_rotation_cycle(
                        scene_clone,
                        &settings_for_thread,
                        &progress_arc,
                        &ctx_clone,
                        width,
                        height,
                        |_, color_data_rgb| {
                            // 将RGB数据转换为RGBA并存储为ColorImage
                            let mut rgba_data = Vec::with_capacity(width * height * 4);
                            for chunk in color_data_rgb.chunks_exact(3) {
                                rgba_data.extend_from_slice(chunk);
                                rgba_data.push(255); // Alpha
                            }
                            let color_image =
                                ColorImage::from_rgba_unmultiplied([width, height], &rgba_data);
                            frames_arc.lock().unwrap().push(color_image);
                        },
                    );
                });
            }
            Err(e) => {
                self.set_error(e);
                self.is_pre_rendering = false;
            }
        }
    }

    fn handle_pre_rendering_tasks(&mut self, ctx: &Context) {
        let progress = self.pre_render_progress.load(Ordering::SeqCst);
        let expected_total_frames = self.total_frames_for_pre_render_cycle;

        // 使用通用函数计算参数
        let (_, seconds_per_rotation, _) =
            calculate_rotation_parameters(self.settings.rotation_speed, self.settings.fps);

        self.status_message = format!(
            "预渲染动画中... ({}/{} 帧，{:.1}%，转一圈约需 {:.1} 秒)",
            progress,
            expected_total_frames,
            if expected_total_frames > 0 {
                progress as f32 / expected_total_frames as f32 * 100.0
            } else {
                0.0
            },
            seconds_per_rotation
        );

        if progress >= expected_total_frames && expected_total_frames > 0 {
            self.is_pre_rendering = false;
            let final_frame_count = self.pre_rendered_frames.lock().unwrap().len();
            self.status_message = format!(
                "预渲染完成！已缓存 {} 帧动画 (目标 {} FPS, 转一圈 {:.1} 秒)",
                final_frame_count, self.settings.fps, seconds_per_rotation
            );
            if self.is_realtime_rendering || self.pre_render_mode {
                self.current_frame_index = 0;
                self.last_frame_time = None;
                ctx.request_repaint();
            }
        } else {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
    }

    fn play_pre_rendered_frames(&mut self, ctx: &Context) {
        let frame_to_display_idx;
        let frame_image;
        let frames_len;
        {
            let frames_guard = self.pre_rendered_frames.lock().unwrap();
            frames_len = frames_guard.len();
            if frames_len == 0 {
                self.pre_render_mode = false;
                self.status_message = "预渲染帧丢失或未生成，退出预渲染模式。".to_string();
                ctx.request_repaint();
                return;
            }
            frame_to_display_idx = self.current_frame_index % frames_len;
            frame_image = frames_guard[frame_to_display_idx].clone();
        }

        let now = Instant::now();
        let target_frame_duration = Duration::from_secs_f32(1.0 / self.settings.fps.max(1) as f32);

        if let Some(last_frame_display_time) = self.last_frame_time {
            let time_since_last_display = now.duration_since(last_frame_display_time);
            if time_since_last_display < target_frame_duration {
                let time_to_wait = target_frame_duration - time_since_last_display;
                ctx.request_repaint_after(time_to_wait);
                return;
            }
            self.update_fps_stats(time_since_last_display);
        } else {
            self.update_fps_stats(target_frame_duration);
        }
        self.last_frame_time = Some(now);

        let texture_name = format!("pre_rendered_tex_{frame_to_display_idx}");
        self.rendered_image =
            Some(ctx.load_texture(texture_name, frame_image, TextureOptions::LINEAR));
        self.current_frame_index = (self.current_frame_index + 1) % frames_len;

        // 使用通用函数计算参数
        let (_, seconds_per_rotation, _) =
            calculate_rotation_parameters(self.settings.rotation_speed, self.settings.fps);

        self.status_message = format!(
            "播放预渲染: 帧 {}/{} (目标 {} FPS, 平均 {:.1} FPS, 1圈 {:.1}秒)",
            frame_to_display_idx + 1,
            frames_len,
            self.settings.fps,
            self.avg_fps,
            seconds_per_rotation
        );
        ctx.request_repaint();
    }
}
use super::animation::AnimationMethods;
use super::core::CoreMethods;
use super::widgets::WidgetMethods;
use crate::core::renderer::Renderer;
use crate::io::render_settings::RenderSettings;
use crate::material_system::materials::Model;
use crate::scene::scene_utils::Scene;
use crate::utils::render_utils::calculate_rotation_parameters;
use egui::{Color32, ColorImage, RichText, Vec2};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// GUI应用状态
pub struct RasterizerApp {
    // TOML可配置参数
    pub settings: RenderSettings,

    // 渲染运行时状态
    pub renderer: Renderer,
    pub scene: Option<Scene>,
    pub model_data: Option<Model>,

    // GUI界面状态
    pub rendered_image: Option<egui::TextureHandle>,
    pub last_render_time: Option<std::time::Duration>,
    pub status_message: String,
    pub show_error_dialog: bool,
    pub error_message: String,
    pub is_dark_theme: bool,

    // 实时渲染状态
    pub current_fps: f32,
    pub fps_history: Vec<f32>,
    pub avg_fps: f32,
    pub is_realtime_rendering: bool,
    pub last_frame_time: Option<std::time::Instant>,

    // 预渲染状态
    pub pre_render_mode: bool,
    pub is_pre_rendering: bool,
    pub pre_rendered_frames: Arc<Mutex<Vec<ColorImage>>>,
    pub current_frame_index: usize,
    pub pre_render_progress: Arc<AtomicUsize>,
    pub animation_time: f32,
    pub total_frames_for_pre_render_cycle: usize,

    // 视频生成状态
    pub is_generating_video: bool,
    pub video_generation_thread: Option<std::thread::JoinHandle<(bool, String)>>,
    pub video_progress: Arc<AtomicUsize>,

    // 相机交互设置
    pub camera_pan_sensitivity: f32,
    pub camera_orbit_sensitivity: f32,
    pub camera_dolly_sensitivity: f32,

    // 相机交互状态
    pub interface_interaction: InterfaceInteraction,

    // 系统状态
    pub ffmpeg_available: bool,
}

/// 相机交互状态
#[derive(Default)]
pub struct InterfaceInteraction {
    pub camera_is_dragging: bool,
    pub camera_is_orbiting: bool,
    pub last_mouse_pos: Option<egui::Pos2>,
    pub anything_changed: bool, // 标记相机等是否发生变化，需要重新渲染
}

impl RasterizerApp {
    /// 创建新的GUI应用实例
    pub fn new(settings: RenderSettings, cc: &eframe::CreationContext<'_>) -> Self {
        // 配置字体，添加中文支持
        let mut fonts = egui::FontDefinitions::default();

        fonts.font_data.insert(
            "chinese_font".to_owned(),
            egui::FontData::from_static(include_bytes!(
                "../../assets/Noto_Sans_SC/static/NotoSansSC-Regular.ttf"
            ))
            .into(),
        );

        for (_text_style, font_ids) in fonts.families.iter_mut() {
            font_ids.push("chinese_font".to_owned());
        }

        cc.egui_ctx.set_fonts(fonts);

        // 浅色主题
        // cc.egui_ctx.set_visuals(egui::Visuals::light());

        // 深色主题
        cc.egui_ctx.set_visuals(egui::Visuals::dark());

        // 创建渲染器
        let renderer = Renderer::new(settings.width, settings.height);

        // 检查ffmpeg是否可用
        let ffmpeg_available = Self::check_ffmpeg_available();

        Self {
            // ===== TOML可配置参数 =====
            settings,

            // ===== 渲染运行时状态 =====
            renderer,
            scene: None,
            model_data: None,

            // ===== GUI界面状态 =====
            rendered_image: None,
            last_render_time: None,
            status_message: String::new(),
            show_error_dialog: false,
            error_message: String::new(),
            is_dark_theme: true, // 默认使用深色主题

            // ===== 实时渲染状态 =====
            current_fps: 0.0,
            fps_history: Vec::new(),
            avg_fps: 0.0,
            is_realtime_rendering: false,
            last_frame_time: None,

            // ===== 预渲染状态 =====
            pre_render_mode: false,
            is_pre_rendering: false,
            pre_rendered_frames: Arc::new(Mutex::new(Vec::new())),
            current_frame_index: 0,
            pre_render_progress: Arc::new(AtomicUsize::new(0)),
            animation_time: 0.0,
            total_frames_for_pre_render_cycle: 0,

            // ===== 视频生成状态 =====
            is_generating_video: false,
            video_generation_thread: None,
            video_progress: Arc::new(AtomicUsize::new(0)),

            // ===== 相机交互设置 =====
            camera_pan_sensitivity: 1.0,
            camera_orbit_sensitivity: 1.0,
            camera_dolly_sensitivity: 1.0,

            // ===== 相机交互状态 =====
            interface_interaction: InterfaceInteraction::default(),

            // ===== 系统状态 =====
            ffmpeg_available,
        }
    }

    /// 检查ffmpeg是否可用
    fn check_ffmpeg_available() -> bool {
        std::process::Command::new("ffmpeg")
            .arg("-version")
            .output()
            .is_ok()
    }

    /// 设置错误信息并显示错误对话框
    pub fn set_error(&mut self, message: String) {
        CoreMethods::set_error(self, message.clone());
        self.error_message = message;
        self.show_error_dialog = true;
    }

    fn handle_camera_interaction(&mut self, image_response: &egui::Response, ctx: &egui::Context) {
        if let Some(scene) = &mut self.scene {
            let mut camera_changed = false;
            let mut need_clear_ground_cache = false;

            let screen_size = Vec2::new(
                self.renderer.frame_buffer.width as f32,
                self.renderer.frame_buffer.height as f32,
            );

            // 处理鼠标拖拽
            if image_response.dragged() {
                if let Some(last_pos) = self.interface_interaction.last_mouse_pos {
                    let current_pos = image_response.interact_pointer_pos().unwrap_or_default();
                    let delta = current_pos - last_pos;

                    // 设置最小移动阈值，避免微小抖动触发重新渲染
                    if delta.length() < 1.0 {
                        return;
                    }

                    let is_shift_pressed = ctx.input(|i| i.modifiers.shift);

                    if is_shift_pressed && !self.interface_interaction.camera_is_orbiting {
                        self.interface_interaction.camera_is_orbiting = true;
                        self.interface_interaction.camera_is_dragging = false;
                    } else if !is_shift_pressed && !self.interface_interaction.camera_is_dragging {
                        self.interface_interaction.camera_is_dragging = true;
                        self.interface_interaction.camera_is_orbiting = false;
                    }

                    if self.interface_interaction.camera_is_orbiting && is_shift_pressed {
                        need_clear_ground_cache = scene
                            .active_camera
                            .orbit_from_screen_delta(delta, self.camera_orbit_sensitivity);
                        camera_changed = true;
                    } else if self.interface_interaction.camera_is_dragging && !is_shift_pressed {
                        need_clear_ground_cache = scene.active_camera.pan_from_screen_delta(
                            delta,
                            screen_size,
                            self.camera_pan_sensitivity,
                        );
                        camera_changed = true;
                    }
                }

                self.interface_interaction.last_mouse_pos = image_response.interact_pointer_pos();
            } else {
                self.interface_interaction.camera_is_dragging = false;
                self.interface_interaction.camera_is_orbiting = false;
                self.interface_interaction.last_mouse_pos = None;
            }

            // 处理鼠标滚轮缩放
            if image_response.hovered() {
                let scroll_delta = ctx.input(|i| i.smooth_scroll_delta.y);
                if scroll_delta.abs() > 0.1 {
                    let zoom_delta = scroll_delta * 0.01;
                    need_clear_ground_cache = scene
                        .active_camera
                        .dolly_from_scroll(zoom_delta, self.camera_dolly_sensitivity);
                    camera_changed = true;
                }
            }

            // 处理快捷键
            ctx.input(|i| {
                if i.key_pressed(egui::Key::R) {
                    need_clear_ground_cache = scene.active_camera.reset_to_default_view();
                    camera_changed = true;
                }

                if i.key_pressed(egui::Key::F) {
                    let object_center = nalgebra::Point3::new(0.0, 0.0, 0.0);
                    let object_radius = 2.0;
                    need_clear_ground_cache = scene
                        .active_camera
                        .focus_on_object(object_center, object_radius);
                    camera_changed = true;
                }
            });

            // 如果相机发生变化，直接更新settings并标记
            if camera_changed {
                // 如果相机变化，清除地面缓存（但保留背景缓存）
                if need_clear_ground_cache {
                    // 只清除地面本体和阴影缓存
                    self.renderer.frame_buffer.invalidate_ground_base_cache();
                    self.renderer.frame_buffer.invalidate_ground_shadow_cache();
                }

                // 直接更新settings字符串
                let pos = scene.active_camera.position();
                let target = scene.active_camera.params.target;
                let up = scene.active_camera.params.up;

                self.settings.camera_from = format!("{},{},{}", pos.x, pos.y, pos.z);
                self.settings.camera_at = format!("{},{},{}", target.x, target.y, target.z);
                self.settings.camera_up = format!("{},{},{}", up.x, up.y, up.z);

                // 统一标记
                self.interface_interaction.anything_changed = true;

                // 在非实时模式下请求重绘
                if !self.is_realtime_rendering {
                    ctx.request_repaint();
                }
            }
        }
    }

    /// 统一的资源清理方法
    fn cleanup_resources(&mut self) {
        CoreMethods::cleanup_resources(self);
    }
}

impl eframe::App for RasterizerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // 显示错误对话框（如果有）
        self.show_error_dialog_ui(ctx);

        // 检查快捷键
        if ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::R)) {
            CoreMethods::render(self, ctx);
        }

        // 执行实时渲染循环
        if self.is_realtime_rendering {
            self.perform_realtime_rendering(ctx);
        }

        // 检查视频生成进度
        if self.is_generating_video {
            if let Some(handle) = &self.video_generation_thread {
                if handle.is_finished() {
                    let result = self
                        .video_generation_thread
                        .take()
                        .unwrap()
                        .join()
                        .unwrap_or_else(|_| (false, "线程崩溃".to_string()));

                    self.is_generating_video = false;

                    if result.0 {
                        self.status_message = format!("视频生成成功: {}", result.1);
                    } else {
                        self.set_error(format!("视频生成失败: {}", result.1));
                    }

                    self.video_progress.store(0, Ordering::SeqCst);
                } else {
                    let progress = self.video_progress.load(Ordering::SeqCst);

                    let (_, _, frames_per_rotation) = calculate_rotation_parameters(
                        self.settings.rotation_speed,
                        self.settings.fps,
                    );
                    let total_frames =
                        (frames_per_rotation as f32 * self.settings.rotation_cycles) as usize;

                    let percent = (progress as f32 / total_frames as f32 * 100.0).round();

                    self.status_message =
                        format!("生成视频中... ({progress}/{total_frames}，{percent:.0}%)");

                    ctx.request_repaint_after(std::time::Duration::from_millis(500));
                }
            }
        }

        // UI布局
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("光栅化渲染器");
                ui.separator();
                ui.label(&self.status_message);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if self.is_realtime_rendering {
                        let (fps_text, fps_color) = CoreMethods::get_fps_display(self);
                        ui.label(RichText::new(&fps_text).color(fps_color));
                        ui.separator();
                    }
                    ui.label("Ctrl+R: 快速渲染");
                });
            });
        });

        egui::SidePanel::left("left_panel")
            .min_width(350.0)
            .resizable(false)
            .show(ctx, |ui| {
                self.draw_side_panel(ctx, ui);
            });

        // 中央面板 - 显示渲染结果和处理相机交互
        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(texture) = &self.rendered_image {
                let available_size = ui.available_size();
                let square_size = available_size.x.min(available_size.y) * 0.95;

                let image_aspect = self.renderer.frame_buffer.width as f32
                    / self.renderer.frame_buffer.height as f32;

                let (width, height) = if image_aspect > 1.0 {
                    (square_size, square_size / image_aspect)
                } else {
                    (square_size * image_aspect, square_size)
                };

                let image_response = ui
                    .horizontal(|ui| {
                        ui.add(
                            egui::Image::new(texture)
                                .fit_to_exact_size(Vec2::new(width, height))
                                .sense(egui::Sense::click_and_drag()),
                        )
                    })
                    .inner;

                self.handle_camera_interaction(&image_response, ctx);

                // 显示交互提示
                let overlay_rect = egui::Rect::from_min_size(
                    ui.max_rect().right_bottom() - Vec2::new(220.0, 20.0),
                    Vec2::new(220.0, 20.0),
                );

                ui.scope_builder(
                    egui::UiBuilder::new()
                        .max_rect(overlay_rect)
                        .layout(egui::Layout::right_to_left(egui::Align::BOTTOM)),
                    |ui| {
                        ui.group(|ui| {
                            ui.label(RichText::new("相机交互").size(14.0).strong());
                            ui.separator();
                            ui.small("• 拖拽 - 平移相机");
                            ui.small("• Shift+拖拽 - 轨道旋转");
                            ui.small("• 滚轮 - 推拉缩放");
                            ui.small("• R键 - 重置视角");
                            ui.small("• F键 - 聚焦物体");
                            ui.separator();
                            ui.small(format!("平移敏感度: {:.1}x", self.camera_pan_sensitivity));
                            ui.small(format!("旋转敏感度: {:.1}x", self.camera_orbit_sensitivity));
                            ui.small(format!("缩放敏感度: {:.1}x", self.camera_dolly_sensitivity));
                            ui.separator();
                            ui.small(RichText::new("交互已启用").color(Color32::GREEN));
                        });
                    },
                );
            } else {
                ui.vertical_centered(|ui| {
                    ui.add_space(100.0);
                    ui.label(RichText::new("无渲染结果").size(24.0).color(Color32::GRAY));
                    ui.label(RichText::new("点击「开始渲染」按钮或按Ctrl+R").color(Color32::GRAY));
                    ui.add_space(20.0);
                    ui.label(
                        RichText::new("加载模型后可在此区域进行相机交互")
                            .color(Color32::from_rgb(100, 150, 255)),
                    );
                });
            }
        });

        // 统一处理所有变化引起的重新渲染
        CoreMethods::render_if_anything_changed(self, ctx);

        // 在每帧更新结束时清理不需要的资源
        self.cleanup_resources();
    }
}

/// 启动GUI应用
pub fn start_gui(settings: RenderSettings) -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_min_inner_size([800.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Rust 光栅化渲染器",
        options,
        Box::new(|cc| Ok(Box::new(RasterizerApp::new(settings, cc)))),
    )
}
use egui::{Color32, Context, RichText, Vec2};
use std::sync::atomic::Ordering;

use super::animation::AnimationMethods;
use super::app::RasterizerApp;
use super::core::CoreMethods;
use super::render_ui::RenderUIMethods;
use crate::core::renderer::Renderer;
use crate::geometry::camera::ProjectionType;
use crate::io::config_loader::TomlConfigLoader;
use crate::io::render_settings::{AnimationType, RotationAxis, parse_point3, parse_vec3};
use crate::material_system::light::Light;
use crate::utils::render_utils::calculate_rotation_parameters;

/// UI组件和工具提示相关方法的特质
pub trait WidgetMethods {
    /// 绘制UI的侧边栏
    fn draw_side_panel(&mut self, ctx: &Context, ui: &mut egui::Ui);

    /// 显示错误对话框
    fn show_error_dialog_ui(&mut self, ctx: &Context);

    /// 显示工具提示
    fn add_tooltip(response: egui::Response, ctx: &Context, text: &str) -> egui::Response;

    // === 面板函数接口 ===

    /// 绘制文件与输出设置面板
    fn ui_file_output_panel(app: &mut RasterizerApp, ui: &mut egui::Ui, ctx: &Context);

    /// 绘制渲染属性设置面板
    fn ui_render_properties_panel(app: &mut RasterizerApp, ui: &mut egui::Ui, ctx: &Context);

    /// 绘制物体变换控制面板
    fn ui_object_transform_panel(app: &mut RasterizerApp, ui: &mut egui::Ui, ctx: &Context);

    /// 绘制背景与环境设置面板
    fn ui_background_settings(app: &mut RasterizerApp, ui: &mut egui::Ui);

    /// 绘制相机设置面板
    fn ui_camera_settings_panel(app: &mut RasterizerApp, ui: &mut egui::Ui, ctx: &Context);

    /// 绘制光照设置面板
    fn ui_lighting_panel(app: &mut RasterizerApp, ui: &mut egui::Ui, ctx: &Context);

    /// 绘制PBR材质设置面板
    fn ui_pbr_material_panel(app: &mut RasterizerApp, ui: &mut egui::Ui, ctx: &Context);

    /// 绘制Phong材质设置面板
    fn ui_phong_material_panel(app: &mut RasterizerApp, ui: &mut egui::Ui, ctx: &Context);

    /// 绘制动画设置面板
    fn ui_animation_panel(app: &mut RasterizerApp, ui: &mut egui::Ui, ctx: &Context);

    /// 绘制按钮控制面板
    fn ui_button_controls_panel(app: &mut RasterizerApp, ui: &mut egui::Ui, ctx: &Context);

    /// 绘制渲染信息面板
    fn ui_render_info_panel(app: &mut RasterizerApp, ui: &mut egui::Ui);
}

impl WidgetMethods for RasterizerApp {
    /// 重构后的侧边栏
    fn draw_side_panel(&mut self, ctx: &Context, ui: &mut egui::Ui) {
        // 主题切换控件（放在侧边栏顶部）
        ui.horizontal(|ui| {
            ui.label("主题：");
            egui::ComboBox::from_id_salt("theme_switch")
                .selected_text(if self.is_dark_theme {
                    "深色"
                } else {
                    "浅色"
                })
                .show_ui(ui, |ui| {
                    if ui
                        .selectable_value(&mut self.is_dark_theme, true, "深色")
                        .clicked()
                    {
                        ctx.set_visuals(egui::Visuals::dark());
                    }
                    if ui
                        .selectable_value(&mut self.is_dark_theme, false, "浅色")
                        .clicked()
                    {
                        ctx.set_visuals(egui::Visuals::light());
                    }
                });
        });
        ui.separator();

        egui::ScrollArea::vertical().show(ui, |ui| {
            // === 核心设置组 ===
            ui.collapsing("📁 文件与输出", |ui| {
                Self::ui_file_output_panel(self, ui, ctx);
            });

            ui.collapsing("🎨 场景与视觉", |ui| {
                // 合并渲染属性和背景设置
                ui.group(|ui| {
                    ui.label(RichText::new("渲染设置").size(14.0).strong());
                    Self::ui_render_properties_panel(self, ui, ctx);
                });

                ui.separator();

                ui.group(|ui| {
                    ui.label(RichText::new("背景设置").size(14.0).strong());
                    Self::ui_background_settings(self, ui);
                });
            });

            // === 3D变换组 ===
            ui.collapsing("🔄 3D变换与相机", |ui| {
                ui.group(|ui| {
                    ui.label(RichText::new("物体变换").size(14.0).strong());
                    Self::ui_object_transform_panel(self, ui, ctx);
                });

                ui.separator();

                ui.group(|ui| {
                    ui.label(RichText::new("相机控制").size(14.0).strong());
                    Self::ui_camera_settings_panel(self, ui, ctx);
                });
            });

            // === 材质与光照组 ===
            ui.collapsing("💡 光照与材质", |ui| {
                // 先显示光照和通用材质属性
                Self::ui_lighting_panel(self, ui, ctx);

                ui.separator();

                // 然后根据着色模型显示专用设置
                if self.settings.use_pbr {
                    ui.group(|ui| {
                        ui.label(RichText::new("✨ PBR专用参数").size(14.0).strong());
                        Self::ui_pbr_material_panel(self, ui, ctx);
                    });
                }

                if self.settings.use_phong {
                    ui.group(|ui| {
                        ui.label(RichText::new("✨ Phong专用参数").size(14.0).strong());
                        Self::ui_phong_material_panel(self, ui, ctx);
                    });
                }
            });

            // === 动画与渲染组 ===
            ui.collapsing("🎬 动画与渲染", |ui| {
                ui.group(|ui| {
                    ui.label(RichText::new("动画设置").size(14.0).strong());
                    Self::ui_animation_panel(self, ui, ctx);
                });

                ui.separator();

                ui.group(|ui| {
                    ui.label(RichText::new("渲染控制").size(14.0).strong());
                    Self::ui_button_controls_panel(self, ui, ctx);
                });
            });

            // === 信息显示组 ===
            ui.collapsing("📊 渲染信息", |ui| {
                Self::ui_render_info_panel(self, ui);
            });
        });
    }

    /// 显示错误对话框
    fn show_error_dialog_ui(&mut self, ctx: &Context) {
        if self.show_error_dialog {
            egui::Window::new("错误")
                .fixed_size([400.0, 150.0])
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.vertical_centered(|ui| {
                        ui.add_space(10.0);
                        ui.label(
                            RichText::new(&self.error_message)
                                .color(Color32::from_rgb(230, 50, 50))
                                .size(16.0),
                        );
                        ui.add_space(20.0);
                        if ui.button(RichText::new("确定").size(16.0)).clicked() {
                            self.show_error_dialog = false;
                        }
                    });
                });
        }
    }

    /// 显示工具提示
    fn add_tooltip(response: egui::Response, _ctx: &Context, text: &str) -> egui::Response {
        response.on_hover_ui(|ui| {
            ui.add(egui::Label::new(
                RichText::new(text).size(14.0).color(Color32::DARK_GRAY),
            ));
        })
    }

    /// 文件与输出设置面板
    fn ui_file_output_panel(app: &mut RasterizerApp, ui: &mut egui::Ui, ctx: &Context) {
        ui.horizontal(|ui| {
            ui.label("OBJ文件：");
            let mut obj_text = app.settings.obj.clone().unwrap_or_default();
            let response = ui.text_edit_singleline(&mut obj_text);
            if response.changed() {
                if obj_text.is_empty() {
                    app.settings.obj = None;
                } else {
                    app.settings.obj = Some(obj_text);
                }

                // OBJ路径变化需要重新加载场景
                app.interface_interaction.anything_changed = true;
                app.scene = None; // 清除现有场景，强制重新加载
                app.rendered_image = None; // 清除渲染结果
            }
            Self::add_tooltip(response, ctx, "选择要渲染的3D模型文件（.obj格式）");
            if ui.button("浏览").clicked() {
                app.select_obj_file();
            }
        });

        // 配置文件管理
        ui.separator();
        ui.horizontal(|ui| {
            ui.label("配置文件：");
            if ui.button("📁 加载配置").clicked() {
                app.load_config_file();
            }
            if ui.button("💾 保存配置").clicked() {
                app.save_config_file();
            }
            if ui.button("📋 示例配置").clicked() {
                // 创建示例配置并应用
                match TomlConfigLoader::create_example_config("temp_example_for_gui.toml") {
                    Ok(_) => {
                        match TomlConfigLoader::load_from_file("temp_example_for_gui.toml") {
                            Ok(example_settings) => {
                                app.apply_loaded_config(example_settings);
                                app.status_message = "示例配置已应用".to_string();
                                // 删除临时文件
                                let _ = std::fs::remove_file("temp_example_for_gui.toml");
                            }
                            Err(e) => {
                                app.set_error(format!("加载示例配置失败: {e}"));
                            }
                        }
                    }
                    Err(e) => {
                        app.set_error(format!("创建示例配置失败: {e}"));
                    }
                }
            }
        });
        ui.small("💡 提示：加载配置会覆盖当前所有设置");

        ui.separator();

        ui.horizontal(|ui| {
            ui.label("输出目录：");
            let response = ui.text_edit_singleline(&mut app.settings.output_dir);
            Self::add_tooltip(response, ctx, "选择渲染结果保存的目录");
            if ui.button("浏览").clicked() {
                app.select_output_dir();
            }
        });

        ui.horizontal(|ui| {
            ui.label("输出文件名：");
            let response = ui.text_edit_singleline(&mut app.settings.output);
            Self::add_tooltip(response, ctx, "渲染结果的文件名（不含扩展名）");
        });

        ui.separator();

        ui.horizontal(|ui| {
            ui.label("宽度：");
            let old_width = app.settings.width;
            let response = ui.add(
                egui::DragValue::new(&mut app.settings.width)
                    .speed(1)
                    .range(1..=4096),
            );
            if app.settings.width != old_width {
                // 分辨率变化需要重新创建渲染器
                app.renderer = Renderer::new(app.settings.width, app.settings.height);
                app.rendered_image = None;
                app.interface_interaction.anything_changed = true;
            }
            Self::add_tooltip(response, ctx, "渲染图像的宽度（像素）");
        });

        ui.horizontal(|ui| {
            ui.label("高度：");
            let old_height = app.settings.height;
            let response = ui.add(
                egui::DragValue::new(&mut app.settings.height)
                    .speed(1)
                    .range(1..=4096),
            );
            if app.settings.height != old_height {
                app.renderer = Renderer::new(app.settings.width, app.settings.height);
                app.rendered_image = None;
                app.interface_interaction.anything_changed = true;
            }
            Self::add_tooltip(response, ctx, "渲染图像的高度（像素）");
        });

        let response = ui.checkbox(&mut app.settings.save_depth, "保存深度图");
        Self::add_tooltip(response, ctx, "同时保存深度图（深度信息可视化）");
    }

    /// 渲染属性设置面板
    fn ui_render_properties_panel(app: &mut RasterizerApp, ui: &mut egui::Ui, ctx: &Context) {
        ui.horizontal(|ui| {
            ui.label("投影类型：");
            let old_projection = app.settings.projection.clone();
            let resp1 = ui.radio_value(
                &mut app.settings.projection,
                "perspective".to_string(),
                "透视",
            );
            let resp2 = ui.radio_value(
                &mut app.settings.projection,
                "orthographic".to_string(),
                "正交",
            );
            if app.settings.projection != old_projection {
                app.interface_interaction.anything_changed = true;
            }
            Self::add_tooltip(resp1, ctx, "使用透视投影（符合人眼观察方式）");
            Self::add_tooltip(resp2, ctx, "使用正交投影（无透视变形）");
        });

        ui.separator();

        // 深度缓冲
        let old_zbuffer = app.settings.use_zbuffer;
        let resp1 = ui.checkbox(&mut app.settings.use_zbuffer, "深度缓冲");
        if app.settings.use_zbuffer != old_zbuffer {
            app.interface_interaction.anything_changed = true;
        }
        Self::add_tooltip(resp1, ctx, "启用Z缓冲进行深度测试，处理物体遮挡关系");

        // 表面颜色设置
        ui.horizontal(|ui| {
            ui.label("表面颜色：");

            let old_texture = app.settings.use_texture;
            let old_colorize = app.settings.colorize;

            let texture_response = ui.radio_value(&mut app.settings.use_texture, true, "使用纹理");
            if texture_response.clicked() && app.settings.use_texture {
                app.settings.colorize = false;
            }

            let face_color_response =
                ui.radio_value(&mut app.settings.colorize, true, "使用面颜色");
            if face_color_response.clicked() && app.settings.colorize {
                app.settings.use_texture = false;
            }

            let material_color_response = ui.radio(
                !app.settings.use_texture && !app.settings.colorize,
                "使用材质颜色",
            );
            if material_color_response.clicked() {
                app.settings.use_texture = false;
                app.settings.colorize = false;
            }

            if app.settings.use_texture != old_texture || app.settings.colorize != old_colorize {
                app.interface_interaction.anything_changed = true;
            }

            Self::add_tooltip(
                texture_response,
                ctx,
                "使用模型的纹理贴图（如果有）\n优先级最高，会覆盖面颜色设置",
            );
            Self::add_tooltip(
                face_color_response,
                ctx,
                "为每个面分配随机颜色\n仅在没有纹理或纹理被禁用时生效",
            );
            Self::add_tooltip(
                material_color_response,
                ctx,
                "使用材质的基本颜色（如.mtl文件中定义）\n在没有纹理且不使用面颜色时生效",
            );
        });

        // 着色模型设置
        ui.horizontal(|ui| {
            ui.label("着色模型：");
            let old_phong = app.settings.use_phong;
            let old_pbr = app.settings.use_pbr;

            let phong_response = ui.radio_value(&mut app.settings.use_phong, true, "Phong着色");
            if phong_response.clicked() && app.settings.use_phong {
                app.settings.use_pbr = false;
            }

            let pbr_response = ui.radio_value(&mut app.settings.use_pbr, true, "PBR渲染");
            if pbr_response.clicked() && app.settings.use_pbr {
                app.settings.use_phong = false;
            }

            if app.settings.use_phong != old_phong || app.settings.use_pbr != old_pbr {
                app.interface_interaction.anything_changed = true;
            }

            Self::add_tooltip(phong_response, ctx, "使用 Phong 着色（逐像素着色）和 Blinn-Phong 光照模型\n提供高质量的光照效果，适合大多数场景");
            Self::add_tooltip(pbr_response, ctx, "使用基于物理的渲染（PBR）\n提供更真实的材质效果，但需要更多的参数调整");
        });

        ui.separator();

        // 修改原有的增强光照效果组，添加阴影映射
        ui.group(|ui| {

            // 阴影映射设置
            let old_shadow_mapping = app.settings.enable_shadow_mapping;
            let resp = ui.checkbox(&mut app.settings.enable_shadow_mapping, "地面阴影映射");
            if app.settings.enable_shadow_mapping != old_shadow_mapping {
                app.interface_interaction.anything_changed = true;
            }
            Self::add_tooltip(
                resp,
                ctx,
                "启用简单阴影映射，在地面显示物体阴影\n需要至少一个方向光源\n相比软阴影更真实但需要更多计算"
            );

            if app.settings.enable_shadow_mapping {
                ui.group(|ui| {
                    ui.label(RichText::new("阴影映射参数").size(12.0).strong());

                    ui.horizontal(|ui| {
                        ui.label("阴影贴图尺寸:");
                        let old_size = app.settings.shadow_map_size;
                        let resp = ui.add(
                            egui::DragValue::new(&mut app.settings.shadow_map_size)
                                .speed(128)
                                .range(128..=10240)
                        );
                        if app.settings.shadow_map_size != old_size {
                            app.interface_interaction.anything_changed = true;
                        }
                        Self::add_tooltip(resp, ctx, "输入阴影贴图分辨率（如4096），越大越清晰但越慢");
                    });

                    ui.horizontal(|ui| {
                        ui.label("阴影偏移:");
                        let old_bias = app.settings.shadow_bias;
                        let resp = ui.add(
                            egui::Slider::new(&mut app.settings.shadow_bias, 0.0001..=0.01)
                                .step_by(0.0001)
                                .custom_formatter(|n, _| format!("{n:.4}"))
                        );
                        if (app.settings.shadow_bias - old_bias).abs() > f32::EPSILON {
                            app.interface_interaction.anything_changed = true;
                        }
                        Self::add_tooltip(resp, ctx, "防止阴影痤疮的偏移值\n值太小会出现自阴影，值太大会使阴影分离");
                    });

                    ui.horizontal(|ui| {
                        ui.label("阴影距离:");
                        let old_distance = app.settings.shadow_distance;
                        let resp = ui.add(
                            egui::Slider::new(&mut app.settings.shadow_distance, 1.0..=100.0)
                                .suffix(" 单位")
                        );
                        if (app.settings.shadow_distance - old_distance).abs() > f32::EPSILON {
                            app.interface_interaction.anything_changed = true;
                        }
                        Self::add_tooltip(resp, ctx, "阴影渲染的最大距离\n距离越大覆盖范围越广，但阴影精度可能降低");
                    });

                    // 是否启用PCF
                    let old_enable_pcf = app.settings.enable_pcf;
                    let resp = ui.checkbox(&mut app.settings.enable_pcf, "启用PCF软阴影");
                    if app.settings.enable_pcf != old_enable_pcf {
                        app.interface_interaction.anything_changed = true;
                    }
                    Self::add_tooltip(resp, ctx, "开启后阴影边缘会变软，抗锯齿但性能消耗增加");

                    if app.settings.enable_pcf {
                        // PCF类型选择
                        let old_pcf_type = app.settings.pcf_type.clone();
                        egui::ComboBox::from_id_salt("pcf_type_combo")
                            .selected_text(&app.settings.pcf_type)
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut app.settings.pcf_type, "Box".to_string(), "Box");
                                ui.selectable_value(&mut app.settings.pcf_type, "Gauss".to_string(), "Gauss");
                            });
                        if app.settings.pcf_type != old_pcf_type {
                            app.interface_interaction.anything_changed = true;
                        }

                        // kernel参数
                        let old_kernel = app.settings.pcf_kernel;
                        let resp = ui.add(
                            egui::Slider::new(&mut app.settings.pcf_kernel, 1..=10)
                                .text("PCF窗口(kernel)")
                        );
                        if app.settings.pcf_kernel != old_kernel {
                            app.interface_interaction.anything_changed = true;
                        }
                        Self::add_tooltip(resp, ctx, "采样窗口半径，越大越软，性能消耗也越高");

                        // Gauss类型时显示sigma
                        if app.settings.pcf_type == "Gauss" {
                            let old_sigma = app.settings.pcf_sigma;
                            let resp = ui.add(
                                egui::Slider::new(&mut app.settings.pcf_sigma, 0.1..=10.0)
                                    .text("高斯σ")
                            );
                            if (app.settings.pcf_sigma - old_sigma).abs() > f32::EPSILON {
                                app.interface_interaction.anything_changed = true;
                            }
                            Self::add_tooltip(resp, ctx, "高斯采样的σ参数，影响软化范围");
                        }
                    }
                });

                // 阴影映射状态提示
                if app.settings.lights.iter().any(|light| matches!(light, Light::Directional { enabled: true, .. })) {
                    ui.label(RichText::new("✅ 检测到方向光源，阴影映射可用").color(Color32::LIGHT_GREEN).size(12.0));
                } else {
                    ui.label(RichText::new("⚠️ 需要至少一个启用的方向光源").color(Color32::DARK_GRAY).size(12.0));
                }
            }
        });

        ui.separator();
        let old_gamma = app.settings.use_gamma;
        let resp7 = ui.checkbox(&mut app.settings.use_gamma, "Gamma校正");
        if app.settings.use_gamma != old_gamma {
            app.interface_interaction.anything_changed = true;
        }
        Self::add_tooltip(resp7, ctx, "应用伽马校正，使亮度显示更准确");

        // ACES色调映射开关
        let old_aces = app.settings.enable_aces;
        let resp = ui.checkbox(&mut app.settings.enable_aces, "启用ACES色调映射");
        if app.settings.enable_aces != old_aces {
            app.interface_interaction.anything_changed = true;
        }
        Self::add_tooltip(
            resp,
            ctx,
            "让高动态范围颜色更自然，避免过曝和死黑，推荐开启",
        );

        let old_backface = app.settings.backface_culling;
        let resp8 = ui.checkbox(&mut app.settings.backface_culling, "背面剔除");
        if app.settings.backface_culling != old_backface {
            app.interface_interaction.anything_changed = true;
        }
        Self::add_tooltip(resp8, ctx, "剔除背向相机的三角形面，提高渲染效率");

        let old_wireframe = app.settings.wireframe;
        let resp9 = ui.checkbox(&mut app.settings.wireframe, "线框模式");
        if app.settings.wireframe != old_wireframe {
            app.interface_interaction.anything_changed = true;
        }
        Self::add_tooltip(resp9, ctx, "仅渲染三角形边缘，显示为线框");

        // 小三角形剔除设置
        ui.horizontal(|ui| {
            let old_cull = app.settings.cull_small_triangles;
            let resp = ui.checkbox(&mut app.settings.cull_small_triangles, "剔除小三角形");
            if app.settings.cull_small_triangles != old_cull {
                app.interface_interaction.anything_changed = true;
            }
            Self::add_tooltip(resp, ctx, "忽略投影后面积很小的三角形，提高性能");

            if app.settings.cull_small_triangles {
                let old_area = app.settings.min_triangle_area;
                let resp = ui.add(
                    egui::DragValue::new(&mut app.settings.min_triangle_area)
                        .speed(0.0001)
                        .range(0.0..=1.0)
                        .prefix("面积阈值："),
                );
                if (app.settings.min_triangle_area - old_area).abs() > f32::EPSILON {
                    app.interface_interaction.anything_changed = true;
                }
                Self::add_tooltip(resp, ctx, "小于此面积的三角形将被剔除（范围0.0-1.0）");
            }
        });

        ui.separator();

        // 纹理设置
        ui.horizontal(|ui| {
            ui.label("纹理文件 (覆盖MTL)：");
            let mut texture_path_str = app.settings.texture.clone().unwrap_or_default();
            let resp = ui.text_edit_singleline(&mut texture_path_str);
            Self::add_tooltip(resp.clone(), ctx, "选择自定义纹理，将覆盖MTL中的定义");

            if resp.changed() {
                if texture_path_str.is_empty() {
                    app.settings.texture = None;
                } else {
                    app.settings.texture = Some(texture_path_str);
                }

                // 纹理变化应该立即触发重绘
                app.interface_interaction.anything_changed = true;
            }

            if ui.button("浏览").clicked() {
                app.select_texture_file(); // 调用 render_ui.rs 中的方法
            }
        });
    }

    /// 物体变换控制面板
    fn ui_object_transform_panel(app: &mut RasterizerApp, ui: &mut egui::Ui, ctx: &Context) {
        // 位置控制
        ui.group(|ui| {
            ui.label("物体位置 (x,y,z)：");
            let old = app.settings.object_position.clone();
            let resp = ui.text_edit_singleline(&mut app.settings.object_position);
            if app.settings.object_position != old {
                app.interface_interaction.anything_changed = true;
            }
            Self::add_tooltip(resp, ctx, "输入物体的世界坐标，例如 0,0,0");
        });

        // 旋转控制（度）
        ui.group(|ui| {
            ui.label("物体旋转 (x,y,z，度)：");
            let old = app.settings.object_rotation.clone();
            let resp = ui.text_edit_singleline(&mut app.settings.object_rotation);
            if app.settings.object_rotation != old {
                app.interface_interaction.anything_changed = true;
            }
            Self::add_tooltip(resp, ctx, "输入旋转角度（度），例如 0,45,0");
        });

        // 缩放控制
        ui.group(|ui| {
            ui.label("物体缩放 (x,y,z)：");
            let old = app.settings.object_scale_xyz.clone();
            let resp = ui.text_edit_singleline(&mut app.settings.object_scale_xyz);
            if app.settings.object_scale_xyz != old {
                app.interface_interaction.anything_changed = true;
            }
            Self::add_tooltip(resp, ctx, "输入缩放比例，例如 1,1,1");
            ui.horizontal(|ui| {
                ui.label("全局缩放:");
                let old_scale = app.settings.object_scale;
                let resp = ui.add(
                    egui::Slider::new(&mut app.settings.object_scale, 0.1..=5.0)
                        .logarithmic(true)
                        .text("倍率"),
                );
                if app.settings.object_scale != old_scale {
                    app.interface_interaction.anything_changed = true;
                }
                Self::add_tooltip(resp, ctx, "整体缩放倍率，影响所有轴");
            });
        });
    }

    /// 背景与环境设置面板
    fn ui_background_settings(app: &mut RasterizerApp, ui: &mut egui::Ui) {
        // 背景图片选项
        let old_bg_image = app.settings.use_background_image;
        ui.checkbox(&mut app.settings.use_background_image, "使用背景图片");
        if app.settings.use_background_image != old_bg_image {
            app.interface_interaction.anything_changed = true;
            app.renderer.frame_buffer.invalidate_background_cache(); // 失效背景缓存
        }

        if app.settings.use_background_image {
            ui.horizontal(|ui| {
                let mut path_text = app
                    .settings
                    .background_image_path
                    .clone()
                    .unwrap_or_default();
                ui.label("背景图片:");
                let response = ui.text_edit_singleline(&mut path_text);

                if response.changed() {
                    if path_text.is_empty() {
                        app.settings.background_image_path = None;
                    } else {
                        app.settings.background_image_path = Some(path_text.clone());
                        app.status_message = format!("背景图片路径已设置: {path_text}");
                    }

                    app.interface_interaction.anything_changed = true;
                    app.renderer.frame_buffer.invalidate_background_cache(); // 失效背景缓存
                }

                if ui.button("浏览...").clicked() {
                    app.select_background_image();
                }
            });
        }

        // 渐变背景设置
        let old_gradient = app.settings.enable_gradient_background;
        ui.checkbox(&mut app.settings.enable_gradient_background, "使用渐变背景");
        if app.settings.enable_gradient_background != old_gradient {
            app.interface_interaction.anything_changed = true;
            app.renderer.frame_buffer.invalidate_background_cache(); // 失效背景缓存
        }

        if app.settings.enable_gradient_background {
            if app.settings.use_background_image && app.settings.background_image_path.is_some() {
                ui.label(
                    egui::RichText::new("注意：渐变背景将覆盖在背景图片上")
                        .color(Color32::DARK_GRAY),
                );
            }

            // 使用按需计算的颜色值
            let top_color = app.settings.get_gradient_top_color_vec();
            let mut top_color_array = [top_color.x, top_color.y, top_color.z];
            if ui.color_edit_button_rgb(&mut top_color_array).changed() {
                app.settings.gradient_top_color = format!(
                    "{},{},{}",
                    top_color_array[0], top_color_array[1], top_color_array[2]
                );

                app.interface_interaction.anything_changed = true;
                app.renderer.frame_buffer.invalidate_background_cache(); // 失效背景缓存
            }
            ui.label("渐变顶部颜色");

            let bottom_color = app.settings.get_gradient_bottom_color_vec();
            let mut bottom_color_array = [bottom_color.x, bottom_color.y, bottom_color.z];
            if ui.color_edit_button_rgb(&mut bottom_color_array).changed() {
                app.settings.gradient_bottom_color = format!(
                    "{},{},{}",
                    bottom_color_array[0], bottom_color_array[1], bottom_color_array[2]
                );

                app.interface_interaction.anything_changed = true;
                app.renderer.frame_buffer.invalidate_background_cache(); // 失效背景缓存
            }
            ui.label("渐变底部颜色");
        }

        // 地面平面设置
        let old_ground = app.settings.enable_ground_plane;
        ui.checkbox(&mut app.settings.enable_ground_plane, "显示地面平面");
        if app.settings.enable_ground_plane != old_ground {
            app.interface_interaction.anything_changed = true;
        }

        if app.settings.enable_ground_plane {
            if app.settings.use_background_image && app.settings.background_image_path.is_some() {
                ui.label(
                    RichText::new("注意：地面平面将覆盖在背景图片上").color(Color32::DARK_GRAY),
                );
            }

            // 使用按需计算的地面颜色
            let ground_color = app.settings.get_ground_plane_color_vec();
            let mut ground_color_array = [ground_color.x, ground_color.y, ground_color.z];
            if ui.color_edit_button_rgb(&mut ground_color_array).changed() {
                app.settings.ground_plane_color = format!(
                    "{},{},{}",
                    ground_color_array[0], ground_color_array[1], ground_color_array[2]
                );

                app.interface_interaction.anything_changed = true;
            }
            ui.label("地面颜色");

            ui.horizontal(|ui| {
                if ui
                    .add(
                        egui::Slider::new(&mut app.settings.ground_plane_height, -10.0..=5.0)
                            .text("地面高度")
                            .step_by(0.1),
                    )
                    .changed()
                {
                    app.interface_interaction.anything_changed = true;
                }

                // 自动适配按钮
                if ui.button("自动适配").clicked() {
                    if let Some(optimal_height) = app.calculate_optimal_ground_height() {
                        app.settings.ground_plane_height = optimal_height;

                        app.interface_interaction.anything_changed = true;
                        app.status_message = format!("地面高度已自动调整为 {optimal_height:.2}");
                    } else {
                        app.status_message = "无法计算地面高度：请先加载模型".to_string();
                    }
                }
            });
        }
    }

    fn ui_camera_settings_panel(app: &mut RasterizerApp, ui: &mut egui::Ui, ctx: &Context) {
        ui.horizontal(|ui| {
            ui.label("相机位置 (x,y,z)：");
            let old = app.settings.camera_from.clone();
            let resp = ui.text_edit_singleline(&mut app.settings.camera_from);
            if app.settings.camera_from != old {
                if let Some(scene) = &mut app.scene {
                    if let Ok(from) = parse_point3(&app.settings.camera_from) {
                        scene.active_camera.params.position = from;
                        scene.active_camera.update_matrices();
                        app.interface_interaction.anything_changed = true;
                    }
                }
            }
            Self::add_tooltip(resp, ctx, "相机的位置坐标，格式为x,y,z");
        });

        ui.horizontal(|ui| {
            ui.label("相机目标 (x,y,z)：");
            let old = app.settings.camera_at.clone();
            let resp = ui.text_edit_singleline(&mut app.settings.camera_at);
            if app.settings.camera_at != old {
                if let Some(scene) = &mut app.scene {
                    if let Ok(at) = parse_point3(&app.settings.camera_at) {
                        scene.active_camera.params.target = at;
                        scene.active_camera.update_matrices();
                        app.interface_interaction.anything_changed = true;
                    }
                }
            }
            Self::add_tooltip(resp, ctx, "相机看向的目标点坐标，格式为x,y,z");
        });

        ui.horizontal(|ui| {
            ui.label("相机上方向 (x,y,z)：");
            let old = app.settings.camera_up.clone();
            let resp = ui.text_edit_singleline(&mut app.settings.camera_up);
            if app.settings.camera_up != old {
                if let Some(scene) = &mut app.scene {
                    if let Ok(up) = parse_vec3(&app.settings.camera_up) {
                        scene.active_camera.params.up = up.normalize();
                        scene.active_camera.update_matrices();
                        app.interface_interaction.anything_changed = true;
                    }
                }
            }
            Self::add_tooltip(resp, ctx, "相机的上方向向量，格式为x,y,z");
        });

        ui.horizontal(|ui| {
            ui.label("视场角 (度)：");
            let old_fov = app.settings.camera_fov;
            let resp = ui.add(egui::Slider::new(
                &mut app.settings.camera_fov,
                10.0..=120.0,
            ));
            if (app.settings.camera_fov - old_fov).abs() > 0.1 {
                if let Some(scene) = &mut app.scene {
                    if let ProjectionType::Perspective { fov_y_degrees, .. } =
                        &mut scene.active_camera.params.projection
                    {
                        *fov_y_degrees = app.settings.camera_fov;
                        scene.active_camera.update_matrices();
                        app.interface_interaction.anything_changed = true;
                    }
                }
            }
            Self::add_tooltip(resp, ctx, "相机视场角，值越大视野范围越广（鱼眼效果）");
        });
        ui.separator();

        // 相机交互控制设置（敏感度设置不需要立即响应，它们只影响交互行为）
        ui.group(|ui| {
            ui.label(RichText::new("相机交互控制").size(16.0).strong());
            ui.separator();

            ui.horizontal(|ui| {
                ui.label("平移敏感度:");
                let resp = ui.add(
                    egui::Slider::new(&mut app.camera_pan_sensitivity, 0.1..=5.0)
                        .step_by(0.1)
                        .text("倍率"),
                );
                Self::add_tooltip(
                    resp,
                    ctx,
                    "鼠标拖拽时的平移敏感度\n数值越大，鼠标移动相同距离时相机移动越快",
                );
            });

            ui.horizontal(|ui| {
                ui.label("旋转敏感度:");
                let resp = ui.add(
                    egui::Slider::new(&mut app.camera_orbit_sensitivity, 0.1..=5.0)
                        .step_by(0.1)
                        .text("倍率"),
                );
                Self::add_tooltip(
                    resp,
                    ctx,
                    "Shift+拖拽时的轨道旋转敏感度\n数值越大，鼠标移动相同距离时相机旋转角度越大",
                );
            });

            ui.horizontal(|ui| {
                ui.label("缩放敏感度:");
                let resp = ui.add(
                    egui::Slider::new(&mut app.camera_dolly_sensitivity, 0.1..=5.0)
                        .step_by(0.1)
                        .text("倍率"),
                );
                Self::add_tooltip(
                    resp,
                    ctx,
                    "鼠标滚轮的推拉缩放敏感度\n数值越大，滚轮滚动相同距离时相机前后移动越快",
                );
            });

            // 重置按钮
            ui.horizontal(|ui| {
                if ui.button("重置交互敏感度").clicked() {
                    app.camera_pan_sensitivity = 1.0;
                    app.camera_orbit_sensitivity = 1.0;
                    app.camera_dolly_sensitivity = 1.0;
                }

                // 预设敏感度按钮
                if ui.button("精确模式").clicked() {
                    app.camera_pan_sensitivity = 0.3;
                    app.camera_orbit_sensitivity = 0.3;
                    app.camera_dolly_sensitivity = 0.3;
                }

                if ui.button("快速模式").clicked() {
                    app.camera_pan_sensitivity = 2.0;
                    app.camera_orbit_sensitivity = 2.0;
                    app.camera_dolly_sensitivity = 2.0;
                }
            });

            // 交互说明
            ui.group(|ui| {
                ui.label(RichText::new("交互说明:").size(14.0).strong());
                ui.label("• 拖拽 - 平移相机视角");
                ui.label("• Shift + 拖拽 - 围绕目标旋转");
                ui.label("• 鼠标滚轮 - 推拉缩放");
                ui.label(
                    RichText::new("注意: 需要在中央渲染区域操作")
                        .size(12.0)
                        .color(Color32::DARK_GRAY),
                );
            });
        });
    }

    /// 光照设置面板
    fn ui_lighting_panel(app: &mut RasterizerApp, ui: &mut egui::Ui, ctx: &Context) {
        // 总光照开关
        let resp = ui
            .checkbox(&mut app.settings.use_lighting, "启用光照")
            .on_hover_text("总光照开关，关闭则仅使用环境光");
        if resp.changed() {
            app.interface_interaction.anything_changed = true;
        }

        ui.separator();

        // 环境光设置
        ui.horizontal(|ui| {
            ui.label("环境光颜色:");
            let ambient_color_vec = app.settings.get_ambient_color_vec();
            let mut ambient_color_rgb = [
                ambient_color_vec.x,
                ambient_color_vec.y,
                ambient_color_vec.z,
            ];
            let resp = ui.color_edit_button_rgb(&mut ambient_color_rgb);
            if resp.changed() {
                app.settings.ambient_color = format!(
                    "{},{},{}",
                    ambient_color_rgb[0], ambient_color_rgb[1], ambient_color_rgb[2]
                );
                app.interface_interaction.anything_changed = true;
            }
        });

        ui.horizontal(|ui| {
            ui.label("环境光强度:");
            let resp = ui.add(egui::Slider::new(&mut app.settings.ambient, 0.0..=1.0));
            if resp.changed() {
                app.interface_interaction.anything_changed = true;
            }
        });
        ui.separator();

        // 统一的材质通用属性控制
        ui.group(|ui| {
    ui.label(RichText::new("🎨 材质通用属性").size(16.0).strong());
    ui.separator();

    // 基础颜色（通用于PBR和Phong）
    ui.horizontal(|ui| {
        ui.label("基础颜色 (Base Color / Diffuse):");
        let base_color_vec = if app.settings.use_pbr {
            parse_vec3(&app.settings.base_color)
        } else {
            parse_vec3(&app.settings.diffuse_color)
        }.unwrap_or_else(|_| nalgebra::Vector3::new(0.8, 0.8, 0.8));

        let mut base_color_rgb = [base_color_vec.x, base_color_vec.y, base_color_vec.z];
        let resp = ui.color_edit_button_rgb(&mut base_color_rgb);
        if resp.changed() {
            let color_str = format!(
                "{:.3},{:.3},{:.3}",
                base_color_rgb[0], base_color_rgb[1], base_color_rgb[2]
            );

            // 同时更新PBR和Phong的颜色设置
            if app.settings.use_pbr {
                app.settings.base_color = color_str;
            } else {
                app.settings.diffuse_color = color_str;
            }
            app.interface_interaction.anything_changed = true;
        }
        Self::add_tooltip(
            resp,
            ctx,
            "材质的基础颜色\nPBR模式下为Base Color，Phong模式下为Diffuse Color",
        );
    });

    // 透明度控制（通用于PBR和Phong）
    ui.horizontal(|ui| {
        ui.label("透明度 (Alpha)：");
        let resp = ui.add(egui::Slider::new(&mut app.settings.alpha, 0.0..=1.0));
        if resp.changed() {
            app.interface_interaction.anything_changed = true;
        }
        Self::add_tooltip(
            resp,
            ctx,
            "材质透明度，0为完全透明，1为完全不透明\n适用于PBR和Phong着色模型\n调整此值可立即看到透明效果",
        );
    });

    // 自发光控制（通用于PBR和Phong）
    ui.horizontal(|ui| {
        ui.label("自发光颜色 (Emissive):");
        let emissive_color_vec = parse_vec3(&app.settings.emissive)
            .unwrap_or_else(|_| nalgebra::Vector3::new(0.0, 0.0, 0.0));
        let mut emissive_color_rgb = [
            emissive_color_vec.x,
            emissive_color_vec.y,
            emissive_color_vec.z,
        ];
        let resp = ui.color_edit_button_rgb(&mut emissive_color_rgb);
        if resp.changed() {
            app.settings.emissive = format!(
                "{:.3},{:.3},{:.3}",
                emissive_color_rgb[0], emissive_color_rgb[1], emissive_color_rgb[2]
            );
            app.interface_interaction.anything_changed = true;
        }
        Self::add_tooltip(
            resp,
            ctx,
            "材质的自发光颜色，表示材质本身发出的光\n不受光照影响，适用于发光物体",
        );
    });
});

        ui.separator();

        // 直接光源管理
        if app.settings.use_lighting {
            ui.horizontal(|ui| {
                if ui.button("➕ 添加方向光").clicked() {
                    app.settings.lights.push(Light::directional(
                        nalgebra::Vector3::new(0.0, -1.0, -1.0),
                        nalgebra::Vector3::new(1.0, 1.0, 1.0),
                        0.8, // 直接使用合理的默认强度
                    ));
                    app.interface_interaction.anything_changed = true;
                }

                if ui.button("➕ 添加点光源").clicked() {
                    app.settings.lights.push(Light::point(
                        nalgebra::Point3::new(0.0, 2.0, 0.0),
                        nalgebra::Vector3::new(1.0, 1.0, 1.0),
                        1.0, // 直接使用合理的默认强度
                        Some((1.0, 0.09, 0.032)),
                    ));
                    app.interface_interaction.anything_changed = true;
                }

                ui.separator();
                ui.label(format!("光源总数: {}", app.settings.lights.len()));
            });

            ui.separator();

            // 可编辑的光源列表
            let mut to_remove = Vec::new();
            for (i, light) in app.settings.lights.iter_mut().enumerate() {
                let mut light_changed = false;

                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        // 删除按钮
                        if ui.button("🗑").on_hover_text("删除此光源").clicked() {
                            to_remove.push(i);
                            app.interface_interaction.anything_changed = true;
                        }

                        // 光源类型和编号
                        match light {
                            Light::Directional { .. } => {
                                ui.label(format!("🔦 方向光 #{}", i + 1));
                            }
                            Light::Point { .. } => {
                                ui.label(format!("💡 点光源 #{}", i + 1));
                            }
                        }
                    });

                    // 光源参数编辑
                    match light {
                        Light::Directional {
                            enabled,
                            direction_str,
                            color_str,
                            intensity,
                            ..
                        } => {
                            ui.horizontal(|ui| {
                                let resp = ui.checkbox(enabled, "启用");
                                if resp.changed() {
                                    light_changed = true;
                                }

                                if *enabled {
                                    // 独立的强度控制
                                    let resp = ui.add(
                                        egui::Slider::new(intensity, 0.0..=3.0)
                                            .text("强度")
                                            .step_by(0.1),
                                    );
                                    if resp.changed() {
                                        light_changed = true;
                                    }
                                }
                            });

                            if *enabled {
                                ui.horizontal(|ui| {
                                    ui.label("方向 (x,y,z):");
                                    let resp = ui.text_edit_singleline(direction_str);
                                    if resp.changed() {
                                        light_changed = true;
                                    }
                                });

                                ui.horizontal(|ui| {
                                    ui.label("颜色:");
                                    let color_vec = parse_vec3(color_str)
                                        .unwrap_or_else(|_| nalgebra::Vector3::new(1.0, 1.0, 1.0));
                                    let mut color_rgb = [color_vec.x, color_vec.y, color_vec.z];
                                    let resp = ui.color_edit_button_rgb(&mut color_rgb);
                                    if resp.changed() {
                                        *color_str = format!(
                                            "{},{},{}",
                                            color_rgb[0], color_rgb[1], color_rgb[2]
                                        );
                                        light_changed = true;
                                    }
                                });
                            }
                        }
                        Light::Point {
                            enabled,
                            position_str,
                            color_str,
                            intensity,
                            constant_attenuation,
                            linear_attenuation,
                            quadratic_attenuation,
                            ..
                        } => {
                            ui.horizontal(|ui| {
                                let resp = ui.checkbox(enabled, "启用");
                                if resp.changed() {
                                    light_changed = true;
                                }

                                if *enabled {
                                    // 独立的强度控制
                                    let resp = ui.add(
                                        egui::Slider::new(intensity, 0.0..=10.0)
                                            .text("强度")
                                            .step_by(0.1),
                                    );
                                    if resp.changed() {
                                        light_changed = true;
                                    }
                                }
                            });

                            if *enabled {
                                ui.horizontal(|ui| {
                                    ui.label("位置 (x,y,z):");
                                    let resp = ui.text_edit_singleline(position_str);
                                    if resp.changed() {
                                        light_changed = true;
                                    }
                                });

                                ui.horizontal(|ui| {
                                    ui.label("颜色:");
                                    let color_vec = parse_vec3(color_str)
                                        .unwrap_or_else(|_| nalgebra::Vector3::new(1.0, 1.0, 1.0));
                                    let mut color_rgb = [color_vec.x, color_vec.y, color_vec.z];
                                    let resp = ui.color_edit_button_rgb(&mut color_rgb);
                                    if resp.changed() {
                                        *color_str = format!(
                                            "{},{},{}",
                                            color_rgb[0], color_rgb[1], color_rgb[2]
                                        );
                                        light_changed = true;
                                    }
                                });

                                // 衰减设置
                                ui.collapsing("衰减参数", |ui| {
                                    ui.horizontal(|ui| {
                                        ui.label("常数:");
                                        let resp = ui.add(
                                            egui::DragValue::new(constant_attenuation)
                                                .speed(0.05)
                                                .range(0.0..=10.0),
                                        );
                                        if resp.changed() {
                                            light_changed = true;
                                        }
                                    });
                                    ui.horizontal(|ui| {
                                        ui.label("线性:");
                                        let resp = ui.add(
                                            egui::DragValue::new(linear_attenuation)
                                                .speed(0.01)
                                                .range(0.0..=1.0),
                                        );
                                        if resp.changed() {
                                            light_changed = true;
                                        }
                                    });
                                    ui.horizontal(|ui| {
                                        ui.label("二次:");
                                        let resp = ui.add(
                                            egui::DragValue::new(quadratic_attenuation)
                                                .speed(0.001)
                                                .range(0.0..=0.5),
                                        );
                                        if resp.changed() {
                                            light_changed = true;
                                        }
                                    });
                                    ui.small("💡 推荐值: 常数=1.0, 线性=0.09, 二次=0.032");
                                });
                            }
                        }
                    }
                });

                if light_changed {
                    let _ = light.update_runtime_fields();
                    app.interface_interaction.anything_changed = true;
                }
            }

            // 删除标记的光源
            for &index in to_remove.iter().rev() {
                app.settings.lights.remove(index);
            }

            // 如果没有光源，显示提示
            if app.settings.lights.is_empty() {
                ui.group(|ui| {
                    ui.label("💡 提示：当前没有光源");
                    ui.label("点击上方的「➕ 添加」按钮来添加光源");
                });
            }
        }
    }

    /// PBR材质设置面板
    fn ui_pbr_material_panel(app: &mut RasterizerApp, ui: &mut egui::Ui, ctx: &Context) {
        ui.horizontal(|ui| {
            ui.label("金属度 (Metallic)：");
            let resp = ui.add(egui::Slider::new(&mut app.settings.metallic, 0.0..=1.0));
            if resp.changed() {
                app.interface_interaction.anything_changed = true;
            }
            Self::add_tooltip(resp, ctx, "材质的金属特性，0为非金属，1为纯金属");
        });

        ui.horizontal(|ui| {
            ui.label("粗糙度 (Roughness)：");
            let resp = ui.add(egui::Slider::new(&mut app.settings.roughness, 0.0..=1.0));
            if resp.changed() {
                app.interface_interaction.anything_changed = true;
            }
            Self::add_tooltip(resp, ctx, "材质的粗糙程度，影响高光的散射");
        });

        ui.horizontal(|ui| {
            ui.label("环境光遮蔽 (AO)：");
            let resp = ui.add(egui::Slider::new(
                &mut app.settings.ambient_occlusion,
                0.0..=1.0,
            ));
            if resp.changed() {
                app.interface_interaction.anything_changed = true;
            }
            Self::add_tooltip(resp, ctx, "环境光遮蔽程度，模拟凹陷处的阴影");
        });
    }

    /// 简化后的Phong材质设置面板
    fn ui_phong_material_panel(app: &mut RasterizerApp, ui: &mut egui::Ui, ctx: &Context) {
        ui.horizontal(|ui| {
            ui.label("镜面反射颜色：");
            let specular_color_vec = parse_vec3(&app.settings.specular_color)
                .unwrap_or_else(|_| nalgebra::Vector3::new(0.5, 0.5, 0.5));
            let mut specular_color_rgb = [
                specular_color_vec.x,
                specular_color_vec.y,
                specular_color_vec.z,
            ];
            let resp = ui.color_edit_button_rgb(&mut specular_color_rgb);
            if resp.changed() {
                app.settings.specular_color = format!(
                    "{:.3},{:.3},{:.3}",
                    specular_color_rgb[0], specular_color_rgb[1], specular_color_rgb[2]
                );
                app.interface_interaction.anything_changed = true;
            }
            Self::add_tooltip(resp, ctx, "高光的颜色");
        });

        ui.horizontal(|ui| {
            ui.label("漫反射强度：");
            let resp = ui.add(egui::Slider::new(
                &mut app.settings.diffuse_intensity,
                0.0..=2.0,
            ));
            if resp.changed() {
                app.interface_interaction.anything_changed = true;
            }
            Self::add_tooltip(resp, ctx, "漫反射光的强度倍数");
        });

        ui.horizontal(|ui| {
            ui.label("镜面反射强度：");
            let resp = ui.add(egui::Slider::new(
                &mut app.settings.specular_intensity,
                0.0..=2.0,
            ));
            if resp.changed() {
                app.interface_interaction.anything_changed = true;
            }
            Self::add_tooltip(resp, ctx, "高光的强度倍数");
        });

        ui.horizontal(|ui| {
            ui.label("光泽度：");
            let resp = ui.add(egui::Slider::new(&mut app.settings.shininess, 1.0..=100.0));
            if resp.changed() {
                app.interface_interaction.anything_changed = true;
            }
            Self::add_tooltip(resp, ctx, "高光的锐利程度，值越大越集中");
        });
    }

    /// 动画设置面板
    fn ui_animation_panel(app: &mut RasterizerApp, ui: &mut egui::Ui, ctx: &Context) {
        ui.horizontal(|ui| {
            ui.label("旋转圈数:");
            let resp = ui.add(
                egui::DragValue::new(&mut app.settings.rotation_cycles)
                    .speed(0.1)
                    .range(0.1..=10.0),
            );
            Self::add_tooltip(resp, ctx, "动画完成的旋转圈数，影响生成的总帧数");
        });

        ui.horizontal(|ui| {
            ui.label("视频生成及预渲染帧率 (FPS):");
            let resp = ui.add(
                egui::DragValue::new(&mut app.settings.fps)
                    .speed(1)
                    .range(1..=60),
            );
            Self::add_tooltip(resp, ctx, "生成视频的每秒帧数");
        });

        let (_, seconds_per_rotation, frames_per_rotation) =
            calculate_rotation_parameters(app.settings.rotation_speed, app.settings.fps);
        let total_frames = (frames_per_rotation as f32 * app.settings.rotation_cycles) as usize;
        let total_seconds = seconds_per_rotation * app.settings.rotation_cycles;

        ui.label(format!(
            "估计总帧数: {total_frames} (视频长度: {total_seconds:.1}秒)"
        ));

        // 动画类型选择
        ui.horizontal(|ui| {
            ui.label("动画类型:");
            let current_animation_type = app.settings.animation_type.clone();
            egui::ComboBox::from_id_salt("animation_type_combo")
                .selected_text(match current_animation_type {
                    AnimationType::CameraOrbit => "相机轨道旋转",
                    AnimationType::ObjectLocalRotation => "物体局部旋转",
                    AnimationType::None => "无动画",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut app.settings.animation_type,
                        AnimationType::CameraOrbit,
                        "相机轨道旋转",
                    );
                    ui.selectable_value(
                        &mut app.settings.animation_type,
                        AnimationType::ObjectLocalRotation,
                        "物体局部旋转",
                    );
                    ui.selectable_value(
                        &mut app.settings.animation_type,
                        AnimationType::None,
                        "无动画",
                    );
                });
        });

        // 旋转轴选择 (仅当动画类型不是 None 时显示)
        if app.settings.animation_type != AnimationType::None {
            ui.horizontal(|ui| {
                ui.label("旋转轴:");
                let current_rotation_axis = app.settings.rotation_axis.clone();
                egui::ComboBox::from_id_salt("rotation_axis_combo")
                    .selected_text(match current_rotation_axis {
                        RotationAxis::X => "X 轴",
                        RotationAxis::Y => "Y 轴",
                        RotationAxis::Z => "Z 轴",
                        RotationAxis::Custom => "自定义轴",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut app.settings.rotation_axis,
                            RotationAxis::X,
                            "X 轴",
                        );
                        ui.selectable_value(
                            &mut app.settings.rotation_axis,
                            RotationAxis::Y,
                            "Y 轴",
                        );
                        ui.selectable_value(
                            &mut app.settings.rotation_axis,
                            RotationAxis::Z,
                            "Z 轴",
                        );
                        ui.selectable_value(
                            &mut app.settings.rotation_axis,
                            RotationAxis::Custom,
                            "自定义轴",
                        );
                    });
            });

            if app.settings.rotation_axis == RotationAxis::Custom {
                ui.horizontal(|ui| {
                    ui.label("自定义轴 (x,y,z):");
                    let resp = ui.text_edit_singleline(&mut app.settings.custom_rotation_axis);
                    Self::add_tooltip(resp, ctx, "输入自定义旋转轴，例如 1,0,0 或 0.707,0.707,0");
                });
            }
        }
        Self::add_tooltip(
            ui.label(""),
            ctx,
            "选择实时渲染和视频生成时的动画效果和旋转轴",
        );

        // 简化预渲染模式复选框逻辑
        let pre_render_enabled = app.can_toggle_pre_render();
        let mut pre_render_value = app.pre_render_mode;

        let pre_render_resp = ui.add_enabled(
            pre_render_enabled,
            egui::Checkbox::new(&mut pre_render_value, "启用预渲染模式"),
        );

        if pre_render_resp.changed() && pre_render_value != app.pre_render_mode {
            app.toggle_pre_render_mode();
        }
        Self::add_tooltip(
            pre_render_resp,
            ctx,
            "启用后，首次开始实时渲染时会预先计算所有帧，\n然后以选定帧率无卡顿播放。\n要求更多内存，但播放更流畅。",
        );

        ui.horizontal(|ui| {
            ui.label("旋转速度 (实时渲染):");
            let resp = ui.add(egui::Slider::new(
                &mut app.settings.rotation_speed,
                0.1..=5.0,
            ));
            Self::add_tooltip(resp, ctx, "实时渲染中的旋转速度倍率");
        });
    }

    /// 按钮控制面板
    fn ui_button_controls_panel(app: &mut RasterizerApp, ui: &mut egui::Ui, ctx: &Context) {
        ui.add_space(20.0);

        // 计算按钮的统一宽度
        let available_width = ui.available_width();
        let spacing = ui.spacing().item_spacing.x;

        // 第一行：2个按钮等宽
        let button_width_row1 = (available_width - spacing) / 2.0;

        // 第二行：2个按钮等宽
        let button_width_row2 = (available_width - spacing) / 2.0;

        // 第三行：2个按钮等宽
        let button_width_row3 = (available_width - spacing) / 2.0;

        let button_height = 40.0;

        // === 第一行：恢复默认值 + 开始渲染 ===
        ui.horizontal(|ui| {
            // 恢复默认值按钮
            let reset_button = ui.add_sized(
                [button_width_row1, button_height],
                egui::Button::new(RichText::new("恢复默认值").size(15.0)),
            );

            if reset_button.clicked() {
                app.reset_to_defaults();
            }

            Self::add_tooltip(
                reset_button,
                ctx,
                "重置所有渲染参数为默认值，保留文件路径设置",
            );

            // 渲染按钮
            let render_button = ui.add_sized(
                [button_width_row1, button_height],
                egui::Button::new(RichText::new("开始渲染").size(18.0).strong()),
            );

            if render_button.clicked() {
                app.render(ctx);
            }

            Self::add_tooltip(render_button, ctx, "快捷键: Ctrl+R");
        });

        ui.add_space(10.0);

        // === 第二行：动画渲染 + 截图 ===
        ui.horizontal(|ui| {
            // 动画渲染按钮
            let realtime_button_text = if app.is_realtime_rendering {
                "停止动画渲染"
            } else if app.pre_render_mode {
                "开始动画渲染 (预渲染模式)"
            } else {
                "开始动画渲染 (实时模式)"
            };

            let realtime_button = ui.add_enabled(
                app.can_render_animation(),
                egui::Button::new(RichText::new(realtime_button_text).size(15.0))
                    .min_size(Vec2::new(button_width_row2, button_height)),
            );

            if realtime_button.clicked() {
                // 如果当前在播放预渲染帧，点击时只是停止播放
                if app.is_realtime_rendering && app.pre_render_mode {
                    app.is_realtime_rendering = false;
                    app.status_message = "已停止动画渲染".to_string();
                }
                // 否则切换实时渲染状态
                else if !app.is_realtime_rendering {
                    // 使用CoreMethods中的开始动画渲染方法
                    if let Err(e) = app.start_animation_rendering() {
                        app.set_error(e);
                    }
                } else {
                    // 使用CoreMethods中的停止动画渲染方法
                    app.stop_animation_rendering();
                }
            }

            // 更新工具提示文本
            let tooltip_text = if app.pre_render_mode {
                "启动动画渲染（预渲染模式）\n• 首次启动会预先计算所有帧\n• 然后以目标帧率流畅播放\n• 需要更多内存但播放更流畅"
            } else {
                "启动动画渲染（实时模式）\n• 每帧实时计算和渲染\n• 帧率取决于硬件性能\n• 内存占用较少"
            };

            Self::add_tooltip(realtime_button, ctx, tooltip_text);

            // 截图按钮
            let screenshot_button = ui.add_enabled(
                app.rendered_image.is_some(),
                egui::Button::new(RichText::new("截图").size(15.0))
                    .min_size(Vec2::new(button_width_row2, button_height)),
            );

            if screenshot_button.clicked() {
                match app.take_screenshot() {
                    Ok(path) => {
                        app.status_message = format!("截图已保存至 {path}");
                    }
                    Err(e) => {
                        app.set_error(format!("截图失败: {e}"));
                    }
                }
            }

            Self::add_tooltip(screenshot_button, ctx, "保存当前渲染结果为图片文件");
        });

        ui.add_space(10.0);

        // === 第三行：生成视频 + 清空缓冲区 ===
        ui.horizontal(|ui| {
            let video_button_text = if app.is_generating_video {
                let progress = app.video_progress.load(Ordering::SeqCst);

                // 使用通用函数计算实际帧数
                let (_, _, frames_per_rotation) =
                    calculate_rotation_parameters(app.settings.rotation_speed, app.settings.fps);
                let total_frames =
                    (frames_per_rotation as f32 * app.settings.rotation_cycles) as usize;

                let percent = (progress as f32 / total_frames as f32 * 100.0).round();
                format!("生成视频中... {percent}%")
            } else if app.ffmpeg_available {
                "生成视频".to_string()
            } else {
                "生成视频 (需ffmpeg)".to_string()
            };

            let is_video_button_enabled = app.can_generate_video();

            // 视频生成按钮
            let video_button_response = ui.add_enabled(
                is_video_button_enabled,
                egui::Button::new(RichText::new(video_button_text).size(15.0))
                    .min_size(Vec2::new(button_width_row3, button_height)),
            );

            if video_button_response.clicked() {
                app.start_video_generation(ctx);
            }
            Self::add_tooltip(
                video_button_response,
                ctx,
                "在后台渲染多帧并生成MP4视频。\n需要系统安装ffmpeg。\n生成过程不会影响UI使用。",
            );

            // 清空缓冲区按钮
            let is_clear_buffer_enabled = app.can_clear_buffer();

            let clear_buffer_response = ui.add_enabled(
                is_clear_buffer_enabled,
                egui::Button::new(RichText::new("清空缓冲区").size(15.0))
                    .min_size(Vec2::new(button_width_row3, button_height)),
            );

            if clear_buffer_response.clicked() {
                // 使用CoreMethods实现
                app.clear_pre_rendered_frames();
            }
            Self::add_tooltip(
                clear_buffer_response,
                ctx,
                "清除已预渲染的动画帧，释放内存。\n请先停止动画渲染再清除缓冲区。",
            );
        });
    }

    /// 渲染信息面板
    fn ui_render_info_panel(app: &mut RasterizerApp, ui: &mut egui::Ui) {
        // 渲染信息
        if let Some(time) = app.last_render_time {
            ui.separator();
            ui.label(format!("渲染耗时: {time:.2?}"));

            // 显示场景统计信息（直接使用SceneStats）
            if let Some(scene) = &app.scene {
                let stats = scene.get_scene_stats();
                ui.label(format!("网格数量: {}", stats.mesh_count));
                ui.label(format!("三角形数量: {}", stats.triangle_count));
                ui.label(format!("顶点数量: {}", stats.vertex_count));
                ui.label(format!("材质数量: {}", stats.material_count));
                ui.label(format!("光源数量: {}", stats.light_count));
            }
        }

        // FPS显示
        if app.is_realtime_rendering {
            let (fps_text, fps_color) = app.get_fps_display();
            ui.separator();
            ui.label(RichText::new(fps_text).color(fps_color).size(16.0));
        }
    }
}
pub mod model_utils;
pub mod render_utils;
pub mod save_utils;
use crate::material_system::materials::Model;
use nalgebra::{Point3, Vector3};

/// 归一化和中心化模型顶点
pub fn normalize_and_center_model(model_data: &mut Model) -> (Vector3<f32>, f32) {
    if model_data.meshes.is_empty() {
        return (Vector3::zeros(), 1.0);
    }

    // 计算所有顶点的边界框或质心
    let mut min_coord = Point3::new(f32::MAX, f32::MAX, f32::MAX);
    let mut max_coord = Point3::new(f32::MIN, f32::MIN, f32::MIN);
    let mut vertex_sum = Vector3::zeros();
    let mut vertex_count = 0;

    for mesh in &model_data.meshes {
        for vertex in &mesh.vertices {
            min_coord = min_coord.inf(&vertex.position);
            max_coord = max_coord.sup(&vertex.position);
            vertex_sum += vertex.position.coords;
            vertex_count += 1;
        }
    }

    if vertex_count == 0 {
        return (Vector3::zeros(), 1.0);
    }

    let center = vertex_sum / (vertex_count as f32);
    let extent = max_coord - min_coord;
    let max_extent = extent.x.max(extent.y).max(extent.z);

    let scale_factor = if max_extent > 1e-6 {
        1.6 / max_extent // 缩放以大致适合[-0.8, 0.8]立方体（类似于Python的0.8因子）
    } else {
        1.0
    };

    // 对所有顶点应用变换
    for mesh in &mut model_data.meshes {
        for vertex in &mut mesh.vertices {
            vertex.position = Point3::from((vertex.position.coords - center) * scale_factor);
        }
    }

    (center, scale_factor)
}
use crate::core::renderer::Renderer;
use crate::io::render_settings::{
    AnimationType, RenderSettings, RotationAxis, get_animation_axis_vector,
};
use crate::scene::scene_utils::Scene;
use crate::utils::save_utils::save_render_with_settings;
use log::{debug, info};
use nalgebra::Vector3;
use std::time::Instant;

const BASE_SPEED: f32 = 60.0; // 1s旋转60度

/// 渲染单帧并保存结果
pub fn render_single_frame(
    scene: &mut Scene,
    renderer: &mut Renderer,
    settings: &RenderSettings,
    output_name: &str,
) -> Result<(), String> {
    let frame_start_time = Instant::now();
    debug!("渲染帧: {output_name}");

    // 直接渲染场景，无需额外同步
    renderer.render_scene(scene, settings);

    // 保存输出图像
    debug!("保存 {output_name} 的输出图像...");
    save_render_with_settings(renderer, settings, Some(output_name))?;

    debug!(
        "帧 {} 渲染完成，耗时 {:?}",
        output_name,
        frame_start_time.elapsed()
    );
    Ok(())
}

/// 执行单个步骤的场景动画
pub fn animate_scene_step(
    scene: &mut Scene,
    animation_type: &AnimationType,
    rotation_axis: &Vector3<f32>,
    rotation_delta_rad: f32,
) {
    match animation_type {
        AnimationType::CameraOrbit => {
            let mut camera = scene.active_camera.clone();
            camera.orbit(rotation_axis, rotation_delta_rad);
            scene.set_camera(camera);
        }
        AnimationType::ObjectLocalRotation => {
            scene.object.rotate(rotation_axis, rotation_delta_rad);
        }
        AnimationType::None => { /* 无动画 */ }
    }
}

/// 计算旋转增量的辅助函数
pub fn calculate_rotation_delta(rotation_speed: f32, dt: f32) -> f32 {
    (rotation_speed * dt * BASE_SPEED).to_radians()
}

/// 计算有效旋转速度及旋转周期
pub fn calculate_rotation_parameters(rotation_speed: f32, fps: usize) -> (f32, f32, usize) {
    // 计算有效旋转速度 (度/秒)
    let mut effective_rotation_speed_dps = rotation_speed * BASE_SPEED;

    // 确保旋转速度不会太小
    if effective_rotation_speed_dps.abs() < 0.001 {
        effective_rotation_speed_dps = 0.1_f32.copysign(rotation_speed.signum());
        if effective_rotation_speed_dps == 0.0 {
            effective_rotation_speed_dps = 0.1;
        }
    }

    // 计算完成一圈需要的秒数
    let seconds_per_rotation = 360.0 / effective_rotation_speed_dps.abs();

    // 计算一圈需要的帧数
    let frames_for_one_rotation = (seconds_per_rotation * fps as f32).ceil() as usize;

    (
        effective_rotation_speed_dps,
        seconds_per_rotation,
        frames_for_one_rotation,
    )
}

/// 执行完整的动画渲染循环
pub fn run_animation_loop(
    scene: &mut Scene,
    renderer: &mut Renderer,
    settings: &RenderSettings,
) -> Result<(), String> {
    // 使用通用函数计算旋转参数
    let (effective_rotation_speed_dps, _, frames_to_render) =
        calculate_rotation_parameters(settings.rotation_speed, settings.fps);

    // 根据用户要求的旋转圈数计算实际帧数
    let total_frames = (frames_to_render as f32 * settings.rotation_cycles) as usize;

    info!(
        "开始动画渲染 ({} 帧, {:.2} 秒)...",
        total_frames,
        total_frames as f32 / settings.fps as f32
    );
    info!(
        "动画类型: {:?}, 旋转轴类型: {:?}, 速度: {:.1}度/秒",
        settings.animation_type, settings.rotation_axis, effective_rotation_speed_dps
    );

    // 计算旋转方向
    let rotation_axis_vec = get_animation_axis_vector(settings);
    if settings.rotation_axis == RotationAxis::Custom {
        debug!("自定义旋转轴: {rotation_axis_vec:?}");
    }

    // 计算每帧的旋转角度
    let rotation_per_frame_rad =
        (360.0 / frames_to_render as f32).to_radians() * settings.rotation_speed.signum();

    // 渲染所有帧
    for frame_num in 0..total_frames {
        let frame_start_time = Instant::now();
        debug!("--- 准备帧 {} / {} ---", frame_num + 1, total_frames);

        // 第一帧通常不旋转，保留原始状态
        if frame_num > 0 {
            animate_scene_step(
                scene,
                &settings.animation_type,
                &rotation_axis_vec,
                rotation_per_frame_rad,
            );
        }

        // 渲染和保存当前帧
        let frame_output_name = format!("frame_{frame_num:03}");
        render_single_frame(scene, renderer, settings, &frame_output_name)?;

        debug!(
            "帧 {} 渲染完成，耗时 {:?}",
            frame_output_name,
            frame_start_time.elapsed()
        );
    }

    info!(
        "动画渲染完成。总时长：{:.2}秒",
        total_frames as f32 / settings.fps as f32
    );
    Ok(())
}
use crate::core::renderer::Renderer;
use crate::io::render_settings::RenderSettings;
use crate::material_system::color::apply_colormap_jet;
use image::ColorType;
use log::{debug, info, warn};
use std::path::Path;

/// 保存RGB图像数据到PNG文件
pub fn save_image(path: &str, data: &[u8], width: u32, height: u32) {
    match image::save_buffer(path, data, width, height, ColorType::Rgb8) {
        Ok(_) => info!("图像已保存到 {path}"),
        Err(e) => warn!("保存图像到 {path} 时出错: {e}"),
    }
}

/// 将深度缓冲数据归一化到指定的百分位数范围
pub fn normalize_depth(depth_buffer: &[f32], min_percentile: f32, max_percentile: f32) -> Vec<f32> {
    // 1. 收集所有有限的深度值
    let mut finite_depths: Vec<f32> = depth_buffer
        .iter()
        .filter(|&&d| d.is_finite())
        .cloned()
        .collect();

    let mut min_clip: f32;
    let mut max_clip: f32;

    // 2. 使用百分位数确定归一化范围
    if finite_depths.len() >= 2 {
        finite_depths.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        let min_idx = ((min_percentile / 100.0 * (finite_depths.len() - 1) as f32).round()
            as usize)
            .clamp(0, finite_depths.len() - 1);
        let max_idx = ((max_percentile / 100.0 * (finite_depths.len() - 1) as f32).round()
            as usize)
            .clamp(0, finite_depths.len() - 1);

        min_clip = finite_depths[min_idx];
        max_clip = finite_depths[max_idx];

        if (max_clip - min_clip).abs() < 1e-6 {
            min_clip = *finite_depths.first().unwrap();
            max_clip = *finite_depths.last().unwrap();
            if (max_clip - min_clip).abs() < 1e-6 {
                max_clip = min_clip + 1.0;
            }
        }
        debug!(
            "使用百分位数归一化深度: [{min_percentile:.1}%, {max_percentile:.1}%] -> [{min_clip:.3}, {max_clip:.3}]"
        );
    } else {
        warn!("没有足够的有限深度值进行百分位裁剪。使用默认范围 [0.1, 10.0]");
        min_clip = 0.1;
        max_clip = 10.0;
    }

    let range = max_clip - min_clip;
    let inv_range = if range > 1e-6 { 1.0 / range } else { 0.0 };

    depth_buffer
        .iter()
        .map(|&depth| {
            if depth.is_finite() {
                ((depth.clamp(min_clip, max_clip) - min_clip) * inv_range).clamp(0.0, 1.0)
            } else {
                1.0
            }
        })
        .collect()
}

/// 保存渲染结果（彩色图像和可选的深度图）
#[allow(clippy::too_many_arguments)]
pub fn save_render_result(
    color_data: &[u8],
    depth_data: Option<&[f32]>,
    width: usize,
    height: usize,
    output_dir: &str,
    output_name: &str,
    settings: &RenderSettings,
    save_depth: bool,
) -> Result<(), String> {
    // 保存彩色图像
    let color_path = Path::new(output_dir)
        .join(format!("{output_name}_color.png"))
        .to_str()
        .ok_or_else(|| "创建彩色输出路径字符串失败".to_string())?
        .to_string();

    save_image(&color_path, color_data, width as u32, height as u32);

    // 保存深度图（如果启用）
    if settings.use_zbuffer && save_depth {
        if let Some(depth_data_raw) = depth_data {
            let depth_normalized = normalize_depth(depth_data_raw, 1.0, 99.0);
            let depth_colored = apply_colormap_jet(
                &depth_normalized
                    .iter()
                    .map(|&d| 1.0 - d) // 反转：越近 = 越热
                    .collect::<Vec<_>>(),
                width,
                height,
                settings.use_gamma,
            );

            let depth_path = Path::new(output_dir)
                .join(format!("{output_name}_depth.png"))
                .to_str()
                .ok_or_else(|| "创建深度输出路径字符串失败".to_string())?
                .to_string();

            save_image(&depth_path, &depth_colored, width as u32, height as u32);
        }
    }

    Ok(())
}

/// 从渲染器中获取数据并保存渲染结果
pub fn save_render_with_settings(
    renderer: &Renderer,
    settings: &RenderSettings,
    output_name: Option<&str>,
) -> Result<(), String> {
    let color_data = renderer.frame_buffer.get_color_buffer_bytes();
    let depth_data = if settings.save_depth {
        Some(renderer.frame_buffer.get_depth_buffer_f32())
    } else {
        None
    };

    let width = renderer.frame_buffer.width;
    let height = renderer.frame_buffer.height;
    let output_name = output_name.unwrap_or(&settings.output);

    save_render_result(
        &color_data,
        depth_data.as_deref(),
        width,
        height,
        &settings.output_dir,
        output_name,
        settings,
        settings.save_depth,
    )
}
use log::{error, info, warn};
use std::fs;
use std::time::Instant;

mod core;
mod geometry;
mod io;
mod material_system;
mod scene;
mod ui;
mod utils;

use crate::ui::app::start_gui;
use core::renderer::Renderer;
use io::model_loader::ModelLoader;
use io::simple_cli::SimpleCli;
use utils::render_utils::{render_single_frame, run_animation_loop};

fn main() -> Result<(), String> {
    // 初始化日志系统
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Debug)
        .filter_module("eframe", log::LevelFilter::Warn) // 只显示 eframe 的警告和错误
        .filter_module("egui_glow", log::LevelFilter::Warn) // 只显示 egui_glow 的警告和错误
        .filter_module("egui_winit", log::LevelFilter::Warn) // 只显示 egui_winit 的警告和错误
        .filter_module("winit", log::LevelFilter::Warn) // 只显示 winit 的警告和错误
        .filter_module("wgpu", log::LevelFilter::Warn) // 只显示 wgpu 的警告和错误
        .filter_module("glutin", log::LevelFilter::Warn) // 只显示 glutin 的警告和错误
        .filter_module("sctk", log::LevelFilter::Warn) // 只显示 sctk 的警告和错误
        .format_timestamp(None)
        .format_level(true)
        .init();

    info!("🎨 光栅化渲染器启动");

    let (settings, should_start_gui) = SimpleCli::process()?;

    // 判断是否应该启动GUI模式
    if should_start_gui {
        info!("启动GUI模式...");
        if let Err(err) = start_gui(settings) {
            error!("GUI启动失败: {err}");
            return Err("GUI启动失败".to_string());
        }
        return Ok(());
    }

    // 无头渲染模式 - 需要OBJ文件
    if settings.obj.is_none() {
        error!("无头模式需要指定OBJ文件路径");
        return Err("缺少OBJ文件路径".to_string());
    }

    let start_time = Instant::now();
    let obj_path = settings.obj.as_ref().unwrap();

    // 确保输出目录存在
    fs::create_dir_all(&settings.output_dir).map_err(|e| {
        error!("创建输出目录 '{}' 失败：{}", settings.output_dir, e);
        "创建输出目录失败".to_string()
    })?;

    // 验证资源
    info!("验证资源...");
    if let Err(e) = ModelLoader::validate_resources(&settings) {
        warn!("{e}");
    }

    // 加载模型和创建场景
    let (mut scene, _model_data) = ModelLoader::load_and_create_scene(obj_path, &settings)
        .map_err(|e| {
            error!("模型加载失败: {e}");
            "模型加载失败".to_string()
        })?;

    // 创建渲染器
    let mut renderer = Renderer::new(settings.width, settings.height);

    // 渲染动画或单帧
    if settings.animate {
        run_animation_loop(&mut scene, &mut renderer, &settings).map_err(|e| {
            error!("动画渲染失败: {e}");
            "动画渲染失败".to_string()
        })?;
    } else {
        info!("--- 开始单帧渲染 ---");
        info!("分辨率: {}x{}", settings.width, settings.height);
        info!("投影类型: {}", settings.projection);
        info!(
            "光照: {} ({} 个光源)",
            if settings.use_lighting {
                "启用"
            } else {
                "禁用"
            },
            settings.lights.len()
        );
        info!("材质: {}", settings.get_lighting_description());

        if settings.use_background_image {
            if let Some(bg_path) = &settings.background_image_path {
                info!("背景图片: {bg_path}");
            }
        }
        if settings.enable_gradient_background {
            info!("渐变背景: 启用");
        }
        if settings.enable_ground_plane {
            info!("地面平面: 启用");
        }

        render_single_frame(&mut scene, &mut renderer, &settings, &settings.output).map_err(
            |e| {
                error!("单帧渲染失败: {e}");
                "单帧渲染失败".to_string()
            },
        )?;
    }

    info!("总执行时间：{:?}", start_time.elapsed());
    Ok(())
}
