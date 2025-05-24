pub use frame_buffer::FrameBuffer;
use geometry_processor::GeometryProcessor;
use triangle_processor::{TriangleData, TriangleProcessor};

use crate::core::rasterizer::rasterize_triangle;
use crate::io::render_settings::RenderSettings;
use crate::scene::scene_utils::Scene;
use rayon::prelude::*;
use std::time::Instant;

use super::{frame_buffer, geometry_processor, triangle_processor};

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

    /// 渲染完整场景 - 唯一的公共渲染接口
    pub fn render_scene(&self, scene: &mut Scene, settings: &RenderSettings) {
        let start_time = Instant::now();

        // 清空帧缓冲区
        self.frame_buffer.clear(settings, &scene.active_camera);

        // 🔥 **简化日志 - 只在debug模式下输出详细信息**
        if cfg!(debug_assertions) {
            println!("渲染场景对象: '{}'...", scene.object.model_data.name);
            self.log_lighting_info(scene, settings);
        }

        // 几何变换阶段 - 使用设置中的多线程选项
        let transform_start = Instant::now();
        let (all_pixel_coords, all_view_coords, all_view_normals, mesh_vertex_offsets) =
            GeometryProcessor::transform_geometry(
                &scene.object,
                &mut scene.active_camera,
                self.frame_buffer.width,
                self.frame_buffer.height,
                settings, // 传递设置以使用多线程选项
            );
        let transform_duration = transform_start.elapsed();

        // 调试信息
        if cfg!(debug_assertions) {
            self.log_geometry_info(&all_view_coords, settings);
        }

        // 三角形准备阶段 - 直接传递场景光源数据
        let triangles_to_render = TriangleProcessor::prepare_triangles(
            &scene.object.model_data,
            &all_pixel_coords,
            &all_view_coords,
            &all_view_normals,
            &mesh_vertex_offsets,
            None, // 材质覆盖
            settings,
            &scene.lights,
            scene.ambient_intensity,
            scene.ambient_color,
        );

        // 光栅化阶段
        let raster_start = Instant::now();
        self.rasterize_triangles(&triangles_to_render, settings);
        let raster_duration = raster_start.elapsed();

        // 性能统计
        let total_duration = start_time.elapsed();
        if cfg!(debug_assertions) {
            self.log_performance_stats(
                &scene.object.model_data.name,
                triangles_to_render.len(),
                transform_duration,
                raster_duration,
                total_duration,
                settings,
            );
        }
    }

    /// 🔥 **新增：记录光照信息** - 适配新的Light结构
    fn log_lighting_info(&self, scene: &Scene, settings: &RenderSettings) {
        println!("🔦 场景光源数量: {}", scene.lights.len());
        println!("🔦 Settings光源数量: {}", settings.lights.len());
        println!(
            "🌍 环境光: 强度={}, 颜色={:?}",
            scene.ambient_intensity, scene.ambient_color
        );

        // 🔥 **适配新的Light枚举结构**
        for (i, light) in scene.lights.iter().enumerate() {
            match light {
                crate::material_system::light::Light::Directional {
                    enabled,
                    direction,
                    color,
                    intensity,
                    direction_str,
                    color_str,
                    ..
                } => {
                    if *enabled {
                        println!(
                            "  方向光 #{}: 方向={:?}, 颜色={:?}, 强度={} [配置: 方向='{}', 颜色='{}']",
                            i, direction, color, intensity, direction_str, color_str
                        );
                    } else {
                        println!("  方向光 #{}: 已禁用", i);
                    }
                }
                crate::material_system::light::Light::Point {
                    enabled,
                    position,
                    color,
                    intensity,
                    position_str,
                    color_str,
                    constant_attenuation,
                    linear_attenuation,
                    quadratic_attenuation,
                    ..
                } => {
                    if *enabled {
                        println!(
                            "  点光源 #{}: 位置={:?}, 颜色={:?}, 强度={}, 衰减=({:.2},{:.3},{:.3}) [配置: 位置='{}', 颜色='{}']",
                            i,
                            position,
                            color,
                            intensity,
                            constant_attenuation,
                            linear_attenuation,
                            quadratic_attenuation,
                            position_str,
                            color_str
                        );
                    } else {
                        println!("  点光源 #{}: 已禁用", i);
                    }
                }
            }
        }
    }

    /// 光栅化三角形列表
    fn rasterize_triangles(&self, triangles: &[TriangleData], settings: &RenderSettings) {
        if settings.use_multithreading {
            triangles.par_iter().for_each(|triangle_data| {
                rasterize_triangle(
                    triangle_data,
                    self.frame_buffer.width,
                    self.frame_buffer.height,
                    &self.frame_buffer.depth_buffer,
                    &self.frame_buffer.color_buffer,
                    settings,
                );
            });
        } else {
            triangles.iter().for_each(|triangle_data| {
                rasterize_triangle(
                    triangle_data,
                    self.frame_buffer.width,
                    self.frame_buffer.height,
                    &self.frame_buffer.depth_buffer,
                    &self.frame_buffer.color_buffer,
                    settings,
                );
            });
        }
    }

    /// 记录几何信息
    fn log_geometry_info(&self, view_coords: &[nalgebra::Point3<f32>], settings: &RenderSettings) {
        if !view_coords.is_empty() {
            let z_min = view_coords
                .iter()
                .map(|p| p.z)
                .fold(f32::INFINITY, f32::min);
            let z_max = view_coords
                .iter()
                .map(|p| p.z)
                .fold(f32::NEG_INFINITY, f32::max);

            println!("视图空间Z范围: [{:.3}, {:.3}]", z_min, z_max);

            let thread_mode = if settings.use_multithreading {
                "并行"
            } else {
                "串行"
            };
            println!("几何变换模式: {}", thread_mode);
        }
    }

    /// 记录性能统计信息
    fn log_performance_stats(
        &self,
        object_name: &str,
        triangle_count: usize,
        transform_duration: std::time::Duration,
        raster_duration: std::time::Duration,
        total_duration: std::time::Duration,
        settings: &RenderSettings,
    ) {
        let thread_mode = if settings.use_multithreading {
            "并行"
        } else {
            "串行"
        };

        println!(
            "对象 '{}' 渲染完成: {} 三角形 ({}模式)",
            object_name, triangle_count, thread_mode
        );
        println!(
            "性能统计 - 变换: {:?}, 光栅化: {:?}, 总时间: {:?}",
            transform_duration, raster_duration, total_duration
        );
    }
}
