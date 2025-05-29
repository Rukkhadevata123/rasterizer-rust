use super::{
    frame_buffer::FrameBuffer, geometry_processor::GeometryProcessor,
    parallel_rasterizer::ParallelRasterizer, triangle_processor::TriangleProcessor,
};
use crate::io::render_settings::RenderSettings;
use crate::scene::scene_utils::Scene;
use log::debug;
use std::time::Instant;

/// 渲染器 - 专注于核心渲染流程
pub struct Renderer {
    pub frame_buffer: FrameBuffer,
    // 简化性能追踪，只保留基本计时
    last_frame_time: Option<std::time::Duration>,
}

impl Renderer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            frame_buffer: FrameBuffer::new(width, height),
            last_frame_time: None,
        }
    }

    /// 核心渲染接口 - 简化流程
    pub fn render_scene(&mut self, scene: &mut Scene, settings: &RenderSettings) {
        let frame_start = Instant::now();

        // 1. 清空帧缓冲区
        self.frame_buffer.clear(settings, &scene.active_camera);

        // 2. 几何变换阶段
        let geometry_result = GeometryProcessor::transform_geometry(
            &scene.object,
            &mut scene.active_camera,
            self.frame_buffer.width,
            self.frame_buffer.height,
        );

        // 3. 三角形准备阶段
        let triangles = TriangleProcessor::prepare_triangles(
            &scene.object.model_data,
            &geometry_result,
            None,
            settings,
            &scene.lights,
            scene.ambient_intensity,
            scene.ambient_color,
        );

        // 4. 光栅化阶段
        ParallelRasterizer::rasterize_triangles(
            &triangles,
            self.frame_buffer.width,
            self.frame_buffer.height,
            &self.frame_buffer.depth_buffer,
            &self.frame_buffer.color_buffer,
            settings,
            &self.frame_buffer,
        );

        // 简化的性能统计
        let frame_time = frame_start.elapsed();
        self.last_frame_time = Some(frame_time);

        if log::log_enabled!(log::Level::Debug) {
            debug!(
                "渲染完成 '{}': {} 三角形, 耗时: {:?}",
                scene.object.model_data.name,
                triangles.len(),
                frame_time
            );
        }
    }
}
