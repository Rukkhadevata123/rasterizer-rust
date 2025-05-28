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
    performance_tracker: PerformanceTracker,
}

impl Renderer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            frame_buffer: FrameBuffer::new(width, height),
            performance_tracker: PerformanceTracker::new(),
        }
    }

    /// 核心渲染接口 - 修复借用检查问题
    pub fn render_scene(&mut self, scene: &mut Scene, settings: &RenderSettings) {
        // 开始帧计时
        self.performance_tracker
            .start_frame_timing(&scene.object.model_data.name);

        // 1. 清空帧缓冲区
        self.frame_buffer.clear(settings, &scene.active_camera);

        // 2. 几何变换阶段
        let geometry_result = {
            self.performance_tracker.start_stage_timing("geometry");
            let result = GeometryProcessor::transform_geometry(
                &scene.object,
                &mut scene.active_camera,
                self.frame_buffer.width,
                self.frame_buffer.height,
            );
            self.performance_tracker.end_stage_timing("geometry");
            result
        };

        // 3. 三角形准备阶段
        let triangles = {
            self.performance_tracker.start_stage_timing("triangles");
            let result = TriangleProcessor::prepare_triangles(
                &scene.object.model_data,
                &geometry_result,
                None,
                settings,
                &scene.lights,
                scene.ambient_intensity,
                scene.ambient_color,
            );
            self.performance_tracker.end_stage_timing("triangles");
            result
        };

        // 4. 光栅化阶段
        {
            self.performance_tracker.start_stage_timing("rasterization");
            ParallelRasterizer::rasterize_triangles(
                &triangles,
                self.frame_buffer.width,
                self.frame_buffer.height,
                &self.frame_buffer.depth_buffer,
                &self.frame_buffer.color_buffer,
                settings,
            );
            self.performance_tracker.end_stage_timing("rasterization");
        }

        // 记录统计信息
        self.performance_tracker.log_stats(triangles.len());
        self.performance_tracker.end_frame_timing();
    }
}

/// 性能追踪器 - 简化设计，避免借用问题
struct PerformanceTracker {
    frame_start: Option<Instant>,
    current_stage_start: Option<Instant>,
    object_name: String,
    stage_times: Vec<(&'static str, std::time::Duration)>,
}

impl PerformanceTracker {
    fn new() -> Self {
        Self {
            frame_start: None,
            current_stage_start: None,
            object_name: String::new(),
            stage_times: Vec::new(),
        }
    }

    /// 开始帧计时
    fn start_frame_timing(&mut self, object_name: &str) {
        self.frame_start = Some(Instant::now());
        self.object_name = object_name.to_string();
        self.stage_times.clear();
    }

    /// 结束帧计时
    fn end_frame_timing(&mut self) {
        // 可以在这里添加额外的帧级统计
    }

    /// 开始阶段计时
    fn start_stage_timing(&mut self, _stage_name: &'static str) {
        self.current_stage_start = Some(Instant::now());
    }

    /// 结束阶段计时
    fn end_stage_timing(&mut self, stage_name: &'static str) {
        if let Some(start) = self.current_stage_start.take() {
            self.stage_times.push((stage_name, start.elapsed()));
        }
    }

    /// 记录统计信息
    fn log_stats(&self, triangle_count: usize) {
        if !log::log_enabled!(log::Level::Debug) {
            return;
        }

        if let Some(start) = self.frame_start {
            let total = start.elapsed();
            debug!(
                "渲染完成 '{}': {} 三角形, 总时间: {:?}",
                self.object_name, triangle_count, total
            );

            for (stage, duration) in &self.stage_times {
                debug!("  {}: {:?}", stage, duration);
            }
        }
    }
}
