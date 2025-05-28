use crate::geometry::camera::Camera;
use crate::io::render_settings::RenderSettings;
use crate::material_system::{color, texture::Texture};
use atomic_float::AtomicF32;
use log::{debug, warn};
use nalgebra::{Matrix4, Point3, Vector3};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU8, Ordering};

/// 帧缓冲区实现，存储渲染结果
pub struct FrameBuffer {
    pub width: usize,
    pub height: usize,
    pub depth_buffer: Vec<AtomicF32>,
    pub color_buffer: Vec<AtomicU8>,
    // 简化：只保留缓存的背景纹理
    cached_background: Option<Texture>,
    cached_path: Option<String>,
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
        }
    }

    /// 清除所有缓冲区，并根据配置绘制背景和地面
    pub fn clear(&mut self, settings: &RenderSettings, camera: &Camera) {
        // 重置深度缓冲区
        self.depth_buffer.par_iter().for_each(|atomic_depth| {
            atomic_depth.store(f32::INFINITY, Ordering::Relaxed);
        });

        // 预加载背景纹理用于并行处理
        let background_texture =
            if settings.use_background_image && settings.background_image_path.is_some() {
                self.get_background_image(settings).cloned()
            } else {
                None
            };

        // 并行绘制背景 - 使用统一方法
        (0..self.height).into_par_iter().for_each(|y| {
            for x in 0..self.width {
                let buffer_index = y * self.width + x;
                let color_index = buffer_index * 3;
                let t_y = y as f32 / (self.height - 1) as f32;
                let t_x = x as f32 / (self.width - 1) as f32;

                // 使用统一的背景颜色计算方法
                let final_color = compute_background_color_unified(
                    settings,
                    camera,
                    background_texture.as_ref(),
                    t_x,
                    t_y,
                );

                // 转换为u8颜色并保存到缓冲区
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
}

// ===== 背景和地面计算函数（原来的结构体改为函数）=====

/// 统一的背景颜色计算方法 - 支持并行和串行调用
pub fn compute_background_color_unified(
    settings: &RenderSettings,
    camera: &Camera,
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

    // 3. 地面平面（最高层）
    if settings.enable_ground_plane {
        let ground_factor = compute_ground_factor(settings, camera, t_x, t_y);
        if ground_factor > 0.0 {
            let ground_color = compute_ground_color(settings, camera, t_x, t_y);

            // 地面混合计算
            let enhanced_ground_factor = ground_factor.powf(0.65) * 2.0;
            let final_ground_factor = enhanced_ground_factor.min(0.95);
            let darkened_background = final_color * (0.8 - final_ground_factor * 0.5).max(0.1);
            final_color = darkened_background * (1.0 - final_ground_factor)
                + ground_color * final_ground_factor;
        }
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
    // 🔥 按需计算地面颜色，不存储
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
