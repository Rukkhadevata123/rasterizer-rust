use crate::geometry::camera::Camera;
use crate::io::render_settings::RenderSettings;
use crate::material_system::color;
use atomic_float::AtomicF32;
use nalgebra::{Matrix4, Point3, Vector3};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU8, Ordering};

/// 帧缓冲区实现，存储渲染结果
pub struct FrameBuffer {
    pub width: usize,
    pub height: usize,
    /// 存储正深度值，数值越小表示越近。使用原子类型以支持并行写入。
    pub depth_buffer: Vec<AtomicF32>,
    /// 存储RGB颜色值 [0, 255]，类型为u8。使用原子类型以支持并行写入。
    pub color_buffer: Vec<AtomicU8>,
}

/// 相机数据快照，用于线程安全的并行处理
#[derive(Debug, Clone)]
struct CameraSnapshot {
    position: Point3<f32>,
    view_matrix: Matrix4<f32>,
    aspect_ratio: f32,
    fov_y_rad: f32,
    far_plane: f32,
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

    /// 清除所有缓冲区，并根据配置绘制背景和地面
    /// 现在接受相机参数以进行准确的地面渲染
    pub fn clear(&self, settings: &RenderSettings, camera: &Camera) {
        // 重置深度缓冲区
        self.depth_buffer.par_iter().for_each(|atomic_depth| {
            atomic_depth.store(f32::INFINITY, Ordering::Relaxed);
        });

        // 提前提取相机数据，避免在并行上下文中访问非Sync类型
        let camera_snapshot = self.extract_camera_data(camera);

        // 绘制背景
        (0..self.height).into_par_iter().for_each(|y| {
            for x in 0..self.width {
                let buffer_index = y * self.width + x;
                let color_index = buffer_index * 3;

                let t_y = y as f32 / (self.height - 1) as f32;
                let t_x = x as f32 / (self.width - 1) as f32;

                let final_color =
                    self.compute_background_color(settings, &camera_snapshot, t_x, t_y);

                // 转换为u8颜色并保存到缓冲区
                let color_u8 = color::linear_rgb_to_u8(&final_color, settings.use_gamma);
                self.color_buffer[color_index].store(color_u8[0], Ordering::Relaxed);
                self.color_buffer[color_index + 1].store(color_u8[1], Ordering::Relaxed);
                self.color_buffer[color_index + 2].store(color_u8[2], Ordering::Relaxed);
            }
        });
    }

    /// 提取相机数据到线程安全的快照
    fn extract_camera_data(&self, camera: &Camera) -> CameraSnapshot {
        // 获取相机的视场角
        let fov_y_rad = match &camera.params().projection {
            crate::geometry::camera::ProjectionType::Perspective { fov_y_degrees, .. } => {
                fov_y_degrees.to_radians()
            }
            crate::geometry::camera::ProjectionType::Orthographic { .. } => {
                // 对于正交投影，使用固定的"视场角"来计算射线方向
                45.0_f32.to_radians()
            }
        };

        CameraSnapshot {
            position: camera.position(),
            view_matrix: camera.view_matrix(),
            aspect_ratio: camera.aspect_ratio(),
            fov_y_rad,
            far_plane: camera.far(),
        }
    }

    fn compute_background_color(
        &self,
        settings: &RenderSettings,
        camera_snapshot: &CameraSnapshot,
        t_x: f32,
        t_y: f32,
    ) -> Vector3<f32> {
        // 1. 背景图片（最底层）
        let mut final_color = if settings.use_background_image
            && settings.background_image.is_some()
        {
            let background = settings.background_image.as_ref().unwrap();
            let tex_x = t_x;
            let tex_y = 1.0 - t_y; // 翻转Y轴
            background.sample(tex_x, tex_y).into()
        } else if settings.enable_gradient_background {
            settings.gradient_top_color_vec * (1.0 - t_y) + settings.gradient_bottom_color_vec * t_y
        } else {
            Vector3::new(0.0, 0.0, 0.0)
        };

        // 2. 渐变叠加
        if settings.use_background_image
            && settings.background_image.is_some()
            && settings.enable_gradient_background
        {
            let gradient_color = settings.gradient_top_color_vec * (1.0 - t_y)
                + settings.gradient_bottom_color_vec * t_y;
            final_color = final_color * 0.3 + gradient_color * 0.7;
        }

        // 3. 地面平面（最高层） - 使用增强的混合方式使地面更明显
        if settings.enable_ground_plane {
            let ground_factor = self.compute_ground_factor(settings, camera_snapshot, t_x, t_y);
            if ground_factor > 0.0 {
                let ground_color = self.compute_ground_color(settings, camera_snapshot, t_x, t_y);

                // 更强的地面混合权重
                let enhanced_ground_factor = ground_factor.powf(0.65) * 2.0; // 增强权重从1.5到2.0
                let final_ground_factor = enhanced_ground_factor.min(0.95); // 提高最大限制从0.9到0.95

                // 使用更强的对比度混合模式
                let darkened_background = final_color * (0.8 - final_ground_factor * 0.5).max(0.1); // 适当压暗背景
                final_color = darkened_background * (1.0 - final_ground_factor)
                    + ground_color * final_ground_factor;
            }
        }

        final_color
    }

    fn compute_ground_factor(
        &self,
        settings: &RenderSettings,
        camera_snapshot: &CameraSnapshot,
        t_x: f32,
        t_y: f32,
    ) -> f32 {
        // 限制地面只在屏幕下半部分显示
        if t_y <= 0.5 {
            return 0.0;
        }

        // 使用相机快照中的数据
        let aspect_ratio = camera_snapshot.aspect_ratio;
        let fov_y_rad = camera_snapshot.fov_y_rad;

        // 将屏幕坐标转换为NDC坐标
        let ndc_x = t_x * 2.0 - 1.0;
        let ndc_y = 1.0 - t_y * 2.0;

        // 计算视图空间射线方向
        let view_x = ndc_x * aspect_ratio * (fov_y_rad / 2.0).tan();
        let view_y = ndc_y * (fov_y_rad / 2.0).tan();
        let view_dir = Vector3::new(view_x, view_y, -1.0).normalize();

        // 使用相机快照中的数据
        let camera_position = camera_snapshot.position;

        // 获取相机的逆视图矩阵（世界到视图的逆变换）
        let view_to_world = camera_snapshot
            .view_matrix
            .try_inverse()
            .unwrap_or_else(Matrix4::identity);

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

        // 后方检测和最大距离限制
        if t < 0.0 || t > camera_snapshot.far_plane * 1.5 {
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
        let grid_factor = if is_grid_line { 0.8 } else { 0.0 }; // 从0.6提高到0.8

        // 距离衰减 - 减轻衰减
        let max_distance = camera_snapshot.far_plane * 0.8; // 增大可见范围
        let distance_factor = (distance_from_camera / max_distance).min(1.0);

        // 基础地面强度增强
        let camera_height = camera_position.y - ground_y;
        let height_factor = (camera_height / 8.0).min(1.5).max(0.3); // 增强最小高度影响

        let ground_influence = (t_y - 0.5) * 2.0;
        let depth_enhanced = ground_influence.powf(1.05) * height_factor * 1.5; // 增强系数

        // 世界空间中的边缘淡出效果
        let world_center_dist =
            ((intersection.x / 20.0).powi(2) + (intersection.z / 20.0).powi(2)).sqrt();
        let world_edge_factor = (1.0 - (world_center_dist / 5.0).min(1.0)).max(0.0);

        // 聚光灯效果 - 增强中心亮度
        let screen_center_x = 0.5;
        let dx = t_x - screen_center_x;
        let screen_distance = ((dx * 1.1).powi(2) + (t_y - 0.85).powi(2) * 0.6).sqrt() * 1.0; // 减小衰减
        let edge_softness = 0.2; // 从0.25减小到0.2，使边缘更清晰
        let spotlight_factor = (1.0 - screen_distance.min(1.0)).powf(1.0 / edge_softness);

        // 混合边缘效果，增加聚光灯权重
        let combined_edge_factor = world_edge_factor * 0.4 + spotlight_factor * 0.6; // 增加聚光灯权重

        // 组合所有因子
        let combined_factor = (1.0 - distance_factor).powf(0.35) // 减轻距离衰减
            * depth_enhanced
            * (1.0 - grid_factor * 0.75) // 降低网格线凹陷程度，使地面更平滑
            * combined_edge_factor;

        // 返回最终因子，稍微提升整体强度
        (combined_factor * 1.1).max(0.0)
    }

    fn compute_ground_color(
        &self,
        settings: &RenderSettings,
        camera_snapshot: &CameraSnapshot,
        t_x: f32,
        t_y: f32,
    ) -> Vector3<f32> {
        // 增强基础地面颜色，提高亮度
        let mut ground_color = settings.ground_plane_color_vec * 1.6; // 增强基础亮度从1.3到1.6

        // 增强饱和度
        let luminance = ground_color.x * 0.299 + ground_color.y * 0.587 + ground_color.z * 0.114;
        ground_color = ground_color * 0.8 + Vector3::new(luminance, luminance, luminance) * 0.2;
        ground_color = ground_color * 1.1; // 整体亮度再提升10%

        // 色调变化 - 增强对比度
        let t_x_centered = (t_x - 0.5) * 2.0;
        let camera_influence = (camera_snapshot.position.x * 0.05).sin() * 0.05; // 增强相机影响
        ground_color.x *= 1.0 + t_x_centered * 0.1 + camera_influence; // 增强色调变化
        ground_color.y *= 1.0 - t_x_centered.abs() * 0.04 + camera_influence * 0.5;
        // 对蓝色分量单独处理，增加对比度
        ground_color.z *= 1.0 - t_x_centered.abs() * 0.05;

        // 减轻大气透视影响
        let distance_from_center = ((t_x - 0.5).powi(2) + (t_y - 0.75).powi(2)).sqrt();
        let camera_height = camera_snapshot.position.y;
        let atmospheric_factor = distance_from_center * 0.1 * (camera_height / 8.0).min(1.0); // 减少大气影响

        ground_color = ground_color * (1.0 - atmospheric_factor)
            + Vector3::new(0.7, 0.8, 0.9) * atmospheric_factor; // 调整大气颜色，略减少蓝色偏差

        // 减少天空反射影响，加强地面本身颜色
        let sky_reflection_strength = (camera_height / 15.0).min(0.08).max(0.02); // 减少天空反射
        let sky_reflection = settings.gradient_top_color_vec * sky_reflection_strength;
        ground_color = ground_color + sky_reflection * (1.0 - (t_y - 0.5) * 1.5).max(0.0);

        // 确保地面颜色不会过暗，增加最小亮度值
        ground_color = ground_color.map(|x| x.max(0.15)); // 提高最小亮度

        ground_color
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
