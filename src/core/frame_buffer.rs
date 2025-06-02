use crate::core::simple_shadow_map::SimpleShadowMap;
use crate::geometry::camera::Camera;
use crate::io::render_settings::RenderSettings;
use crate::material_system::{color, texture::Texture};
use atomic_float::AtomicF32;
use log::{debug, warn};
use nalgebra::{Matrix4, Point3, Vector3};
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

/// 缓存的背景状态 - 使用Arc避免克隆
#[derive(Debug, Clone)]
struct BackgroundCache {
    /// 缓存的背景像素数据 (RGB) - 使用Arc共享
    pixels: Arc<Vec<Vector3<f32>>>,
    /// 缓存时的设置哈希值，用于检测变化
    settings_hash: u64,
    /// 渲染尺寸
    width: usize,
    height: usize,
}

/// 缓存的地面状态
#[derive(Debug, Clone)]
pub struct GroundCache {
    /// 缓存的地面因子数据 - 使用Arc共享
    ground_factors: Arc<Vec<f32>>,
    /// 缓存的地面颜色数据 - 使用Arc共享
    ground_colors: Arc<Vec<Vector3<f32>>>,
    /// 缓存的阴影因子数据
    shadow_factors: Arc<Vec<f32>>,
    /// 相机状态哈希值
    camera_hash: u64,
    /// 地面设置哈希值
    ground_settings_hash: u64,
    /// 渲染尺寸
    width: usize,
    height: usize,
}

/// 帧缓冲区实现，存储渲染结果
pub struct FrameBuffer {
    pub width: usize,
    pub height: usize,
    pub depth_buffer: Vec<AtomicF32>,
    pub color_buffer: Vec<AtomicU8>,

    // 简化：只保留缓存的背景纹理
    cached_background: Option<Texture>,
    cached_path: Option<String>,

    // 新增：背景和地面缓存
    background_cache: Option<BackgroundCache>,
    pub ground_cache: Option<GroundCache>,
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
            ground_cache: None,
        }
    }

    /// 修改：支持阴影的清除方法 - 增加物体变换哈希
    pub fn clear_with_shadow_map(
        &mut self,
        settings: &RenderSettings,
        camera: &Camera,
        shadow_map: Option<&SimpleShadowMap>,
    ) {
        // 重置深度缓冲区
        self.depth_buffer.par_iter().for_each(|atomic_depth| {
            atomic_depth.store(f32::INFINITY, Ordering::Relaxed);
        });

        let width = self.width;
        let height = self.height;

        // 1. 背景缓存逻辑
        let background_pixels_ref = self.get_or_compute_background_cache(settings, width, height);

        // 2. 修改：地面缓存逻辑 - 增加物体变换哈希
        let (ground_factors_ref, ground_colors_ref, shadow_factors_ref) =
            if settings.enable_ground_plane {
                self.get_or_compute_ground_cache(settings, camera, shadow_map, width, height)
            } else {
                // 为未启用地面的情况提供默认值
                (
                    Arc::new(vec![0.0; width * height]),
                    Arc::new(vec![Vector3::zeros(); width * height]),
                    Arc::new(vec![1.0; width * height]),
                )
            };

        // 3. 并行合成最终颜色
        self.compose_final_colors(
            settings,
            &background_pixels_ref,
            &ground_factors_ref,
            &ground_colors_ref,
            &shadow_factors_ref,
        );
    }

    /// 提取：获取或计算背景缓存
    fn get_or_compute_background_cache(
        &mut self,
        settings: &RenderSettings,
        width: usize,
        height: usize,
    ) -> Arc<Vec<Vector3<f32>>> {
        let current_hash = self.compute_background_settings_hash(settings);

        let cache_valid = if let Some(ref cache) = self.background_cache {
            cache.settings_hash == current_hash && cache.width == width && cache.height == height
        } else {
            false
        };

        if !cache_valid {
            debug!("计算背景缓存...");

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

                    *pixel =
                        compute_background_only(settings, background_texture.as_ref(), t_x, t_y);
                });

            self.background_cache = Some(BackgroundCache {
                pixels: Arc::new(background_pixels),
                settings_hash: current_hash,
                width,
                height,
            });

            debug!("背景缓存计算完成 ({}x{})", width, height);
        }

        self.background_cache.as_ref().unwrap().pixels.clone()
    }

    /// 提取：获取或计算地面缓存
    #[allow(clippy::type_complexity)]
    fn get_or_compute_ground_cache(
        &mut self,
        settings: &RenderSettings,
        camera: &Camera,
        shadow_map: Option<&SimpleShadowMap>,
        width: usize,
        height: usize,
    ) -> (Arc<Vec<f32>>, Arc<Vec<Vector3<f32>>>, Arc<Vec<f32>>) {
        let camera_hash = self.compute_camera_hash_stable(camera);
        let ground_hash = self.compute_ground_settings_hash_stable(settings);

        let cache_valid = if let Some(ref cache) = self.ground_cache {
            cache.camera_hash == camera_hash
                && cache.ground_settings_hash == ground_hash
                && cache.width == width
                && cache.height == height
        } else {
            false
        };

        if !cache_valid {
            debug!("重新计算地面+阴影缓存...");

            let mut ground_factors = vec![0.0; width * height];
            let mut ground_colors = vec![Vector3::zeros(); width * height];
            let mut shadow_factors = vec![1.0; width * height];

            ground_factors
                .par_iter_mut()
                .zip(ground_colors.par_iter_mut())
                .zip(shadow_factors.par_iter_mut())
                .enumerate()
                .for_each(|(buffer_index, ((factor, color), shadow_factor))| {
                    let y = buffer_index / width;
                    let x = buffer_index % width;
                    let t_y = y as f32 / (height - 1) as f32;
                    let t_x = x as f32 / (width - 1) as f32;

                    *factor = compute_ground_factor(settings, camera, t_x, t_y);
                    if *factor > 0.0 {
                        *color = compute_ground_color(settings, camera, t_x, t_y);

                        // 计算阴影因子
                        *shadow_factor =
                            compute_ground_shadow_factor(settings, camera, t_x, t_y, shadow_map);
                    }
                });

            self.ground_cache = Some(GroundCache {
                ground_factors: Arc::new(ground_factors),
                ground_colors: Arc::new(ground_colors),
                shadow_factors: Arc::new(shadow_factors),
                camera_hash,
                ground_settings_hash: ground_hash,
                width,
                height,
            });

            debug!("地面+阴影缓存计算完成");
        }

        let cache = self.ground_cache.as_ref().unwrap();
        (
            cache.ground_factors.clone(),
            cache.ground_colors.clone(),
            cache.shadow_factors.clone(),
        )
    }

    /// 提取：合成最终颜色
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
                        let ground_color = ground_colors_ref[buffer_index];
                        let shadow_factor = shadow_factors_ref[buffer_index];

                        let shadowed_ground_color = ground_color * shadow_factor;

                        let enhanced_ground_factor = ground_factor.powf(0.65) * 2.0;
                        let final_ground_factor = enhanced_ground_factor.min(0.95);
                        let darkened_background =
                            final_color * (0.8 - final_ground_factor * 0.5).max(0.1);
                        final_color = darkened_background * (1.0 - final_ground_factor)
                            + shadowed_ground_color * final_ground_factor;
                    }
                }

                let color_u8 = color::linear_rgb_to_u8(&final_color, settings.use_gamma);
                self.color_buffer[color_index].store(color_u8[0], Ordering::Relaxed);
                self.color_buffer[color_index + 1].store(color_u8[1], Ordering::Relaxed);
                self.color_buffer[color_index + 2].store(color_u8[2], Ordering::Relaxed);
            }
        });
    }

    /// 直接加载背景图片，去除多层包装
    fn get_background_image(&mut self, settings: &RenderSettings) -> Option<&Texture> {
        if !settings.use_background_image {
            return None;
        }

        let current_path = settings.background_image_path.as_ref()?;

        // 检查缓存
        if let Some(cached_path) = &self.cached_path {
            if cached_path == current_path && self.cached_background.is_some() {
                return self.cached_background.as_ref();
            }
        }

        // 直接加载，无中间层
        match Texture::from_file(current_path) {
            Some(texture) => {
                debug!("背景图片加载成功: {}x{}", texture.width, texture.height);
                self.cached_background = Some(texture);
                self.cached_path = Some(current_path.clone());
                self.cached_background.as_ref()
            }
            None => {
                warn!("无法加载背景图片 '{}'", current_path);
                None
            }
        }
    }

    /// 优化的哈希函数 - 稳定的浮点数处理
    fn hash_f32_stable(value: f32, hasher: &mut std::collections::hash_map::DefaultHasher) {
        use std::hash::Hash;
        // 使用更粗粒度的量化，减少微小变化导致的哈希抖动
        let quantized = (value * 100.0).round() as i32;
        quantized.hash(hasher);
    }

    /// 计算背景设置的哈希值
    fn compute_background_settings_hash(&self, settings: &RenderSettings) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // 背景相关的设置
        settings.use_background_image.hash(&mut hasher);
        settings.background_image_path.hash(&mut hasher);
        settings.enable_gradient_background.hash(&mut hasher);
        settings.gradient_top_color.hash(&mut hasher);
        settings.gradient_bottom_color.hash(&mut hasher);

        hasher.finish()
    }

    /// 稳定的相机哈希计算
    fn compute_camera_hash_stable(&self, camera: &Camera) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        let mut hasher = DefaultHasher::new();

        // 相机位置和方向（影响地面渲染）
        let pos = camera.position();
        let params = &camera.params;

        // 使用稳定的浮点数哈希
        Self::hash_f32_stable(pos.x, &mut hasher);
        Self::hash_f32_stable(pos.y, &mut hasher);
        Self::hash_f32_stable(pos.z, &mut hasher);

        Self::hash_f32_stable(params.target.x, &mut hasher);
        Self::hash_f32_stable(params.target.y, &mut hasher);
        Self::hash_f32_stable(params.target.z, &mut hasher);

        Self::hash_f32_stable(params.up.x, &mut hasher);
        Self::hash_f32_stable(params.up.y, &mut hasher);
        Self::hash_f32_stable(params.up.z, &mut hasher);

        // 投影参数
        match &params.projection {
            crate::geometry::camera::ProjectionType::Perspective {
                fov_y_degrees,
                aspect_ratio,
            } => {
                use std::hash::Hash;
                0u8.hash(&mut hasher);
                Self::hash_f32_stable(*fov_y_degrees, &mut hasher);
                Self::hash_f32_stable(*aspect_ratio, &mut hasher);
            }
            crate::geometry::camera::ProjectionType::Orthographic { width, height } => {
                use std::hash::Hash;
                1u8.hash(&mut hasher);
                Self::hash_f32_stable(*width, &mut hasher);
                Self::hash_f32_stable(*height, &mut hasher);
            }
        }

        Self::hash_f32_stable(params.near, &mut hasher);
        Self::hash_f32_stable(params.far, &mut hasher);

        hasher.finish()
    }

    /// 稳定的地面设置哈希计算
    fn compute_ground_settings_hash_stable(&self, settings: &RenderSettings) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // 地面相关的设置
        settings.enable_ground_plane.hash(&mut hasher);
        Self::hash_f32_stable(settings.ground_plane_height, &mut hasher);
        settings.ground_plane_color.hash(&mut hasher);

        hasher.finish()
    }

    /// 强制清除所有缓存（当渲染尺寸改变时调用）
    pub fn invalidate_caches(&mut self) {
        self.background_cache = None;
        self.ground_cache = None;
        debug!("已清除所有缓存");
    }

    /// 获取颜色缓冲区的字节数据
    pub fn get_color_buffer_bytes(&self) -> Vec<u8> {
        self.color_buffer
            .iter()
            .map(|atomic_color| atomic_color.load(Ordering::Relaxed))
            .collect()
    }

    /// 获取深度缓冲区的浮点数据
    pub fn get_depth_buffer_f32(&self) -> Vec<f32> {
        self.depth_buffer
            .iter()
            .map(|atomic_depth| atomic_depth.load(Ordering::Relaxed))
            .collect()
    }

    /// 获取指定像素的背景颜色（线性空间）
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

    /// 获取指定像素的背景颜色（返回Color类型，用于着色器）
    pub fn get_pixel_color_as_color(
        &self,
        x: usize,
        y: usize,
    ) -> crate::material_system::color::Color {
        if let Some(color_vec) = self.get_pixel_color(x, y) {
            crate::material_system::color::Color::new(color_vec.x, color_vec.y, color_vec.z)
        } else {
            crate::material_system::color::Color::new(0.1, 0.1, 0.1)
        }
    }
}

// ===== 背景和地面计算函数 =====

/// 纯背景颜色计算（不包括地面）
pub fn compute_background_only(
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

/// 计算地面因子（原 GroundRenderer 的方法改为函数）
pub fn compute_ground_factor(
    settings: &RenderSettings,
    camera: &Camera,
    t_x: f32,
    t_y: f32,
) -> f32 {
    // 获取相机的视场角
    let fov_y_rad = match &camera.params.projection {
        crate::geometry::camera::ProjectionType::Perspective { fov_y_degrees, .. } => {
            fov_y_degrees.to_radians()
        }
        crate::geometry::camera::ProjectionType::Orthographic { .. } => 45.0_f32.to_radians(),
    };

    // 使用相机数据
    let aspect_ratio = camera.aspect_ratio();
    let camera_position = camera.position();
    let view_matrix = camera.view_matrix();
    let far_plane = camera.far();
    let near_plane = camera.near();

    // 将屏幕坐标转换为NDC坐标
    let ndc_x = t_x * 2.0 - 1.0;
    let ndc_y = 1.0 - t_y * 2.0;

    // 计算视图空间射线方向
    let view_x = ndc_x * aspect_ratio * (fov_y_rad / 2.0).tan();
    let view_y = ndc_y * (fov_y_rad / 2.0).tan();
    let view_dir = Vector3::new(view_x, view_y, -1.0).normalize();

    // 获取相机的逆视图矩阵
    let view_to_world = view_matrix.try_inverse().unwrap_or_else(Matrix4::identity);
    let world_ray_dir = view_to_world.transform_vector(&view_dir).normalize();
    let world_ray_origin = camera_position;

    // 计算与地面平面的交点
    let ground_y = settings.ground_plane_height;
    let ground_normal = Vector3::y();
    let plane_point = Point3::new(0.0, ground_y, 0.0);

    let denominator = ground_normal.dot(&world_ray_dir);

    // 平行检测
    if denominator.abs() <= 1e-4 {
        return 0.0;
    }

    let t = (plane_point - world_ray_origin).dot(&ground_normal) / denominator;

    // 后方检测和距离限制
    if t < near_plane || t > far_plane * 1.5 {
        return 0.0;
    }

    // 计算交点
    let intersection = world_ray_origin + t * world_ray_dir;

    // 限制地面的渲染范围
    let max_render_distance = 100.0;
    let horizontal_distance = ((intersection.x - camera_position.x).powi(2)
        + (intersection.z - camera_position.z).powi(2))
    .sqrt();

    if horizontal_distance > max_render_distance {
        return 0.0;
    }

    // 网格计算 - 使用相机位置偏移
    let grid_size = 1.0;
    let grid_x = ((intersection.x - camera_position.x * 0.1) / grid_size).abs() % 1.0;
    let grid_z = ((intersection.z - camera_position.z * 0.1) / grid_size).abs() % 1.0;

    // 动态调整网格线宽度
    let distance_from_camera = (intersection - camera_position).magnitude();
    let adaptive_line_width = (0.02 + distance_from_camera * 0.001).min(0.1);

    let is_grid_line = grid_x < adaptive_line_width
        || grid_x > (1.0 - adaptive_line_width)
        || grid_z < adaptive_line_width
        || grid_z > (1.0 - adaptive_line_width);

    // 增强网格线对比度
    let grid_factor = if is_grid_line { 0.8 } else { 0.0 };

    // 距离衰减
    let effective_far = far_plane * 0.8;
    let distance_factor = (distance_from_camera / effective_far).min(1.0);

    // 基础地面强度计算
    let camera_height = camera_position.y - ground_y;
    let height_factor = (camera_height / 8.0).clamp(0.3, 1.5);

    // 使用射线与地面的角度来计算强度
    let ray_to_ground_angle = world_ray_dir.dot(&ground_normal).abs();
    let angle_enhanced = ray_to_ground_angle.powf(0.8) * height_factor * 1.2;

    // 世界空间中的边缘淡出效果
    let world_center_dist =
        ((intersection.x / 20.0).powi(2) + (intersection.z / 20.0).powi(2)).sqrt();
    let world_edge_factor = (1.0 - (world_center_dist / 5.0).min(1.0)).max(0.0);

    // 聚光灯效果
    let view_forward = view_matrix.column(2).xyz().normalize();
    let center_alignment = world_ray_dir.dot(&view_forward).max(0.0);
    let spotlight_factor = center_alignment.powf(2.0);

    // 混合边缘效果
    let combined_edge_factor = world_edge_factor * 0.4 + spotlight_factor * 0.6;

    // 组合所有因子
    let combined_factor = (1.0 - distance_factor).powf(0.35)
        * angle_enhanced
        * (1.0 - grid_factor * 0.75)
        * combined_edge_factor;

    (combined_factor * 1.1).max(0.0)
}

/// 计算地面颜色（使用按需计算）
pub fn compute_ground_color(
    settings: &RenderSettings,
    camera: &Camera,
    t_x: f32,
    t_y: f32,
) -> Vector3<f32> {
    // 按需计算地面颜色，不存储
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
    ground_color.map(|x| x.max(0.15))
}

/// 计算地面阴影因子
pub fn compute_ground_shadow_factor(
    settings: &RenderSettings,
    camera: &Camera,
    t_x: f32,
    t_y: f32,
    shadow_map: Option<&SimpleShadowMap>,
) -> f32 {
    if !settings.enable_shadow_mapping {
        return 1.0;
    }

    let shadow_map = match shadow_map {
        Some(sm) if sm.is_valid => sm,
        _ => return 1.0,
    };

    // 重复使用地面交点计算逻辑
    if let Some(ground_intersection) = compute_ground_intersection(settings, camera, t_x, t_y) {
        // 🔧 关键修复：直接使用地面交点进行阴影测试，不使用物体变换矩阵
        // 阴影贴图本身已经是在正确的光源空间中生成的，包含了物体的变换信息
        shadow_map.compute_shadow_factor(
            &ground_intersection,
            &Matrix4::identity(), // 🔧 使用单位矩阵，因为地面交点已经在世界空间中
            settings.shadow_bias,
        )
    } else {
        1.0
    }
}

/// 提取：计算地面交点（复用射线求交逻辑）
fn compute_ground_intersection(
    settings: &RenderSettings,
    camera: &Camera,
    t_x: f32,
    t_y: f32,
) -> Option<Point3<f32>> {
    // 复用 compute_ground_factor 中的射线与地面求交逻辑
    let fov_y_rad = match &camera.params.projection {
        crate::geometry::camera::ProjectionType::Perspective { fov_y_degrees, .. } => {
            fov_y_degrees.to_radians()
        }
        crate::geometry::camera::ProjectionType::Orthographic { .. } => 45.0_f32.to_radians(),
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

    let ground_y = settings.ground_plane_height;
    let ground_normal = Vector3::y();
    let plane_point = Point3::new(0.0, ground_y, 0.0);

    let denominator = ground_normal.dot(&world_ray_dir);
    if denominator.abs() <= 1e-4 {
        return None;
    }

    let t = (plane_point - camera_position).dot(&ground_normal) / denominator;
    if t < camera.near() {
        return None;
    }

    Some(camera_position + t * world_ray_dir)
}
