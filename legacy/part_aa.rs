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

