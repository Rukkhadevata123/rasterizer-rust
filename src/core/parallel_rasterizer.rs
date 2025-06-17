use crate::core::triangle_processor::TriangleData;
use crate::io::render_settings::RenderSettings;
use atomic_float::AtomicF32;
use nalgebra::Point2;
use rayon::prelude::*;
use std::sync::atomic::AtomicU8;

/// 智能并行光栅化器 - 自动选择最优策略
pub struct ParallelRasterizer;

impl ParallelRasterizer {
    /// 智能并行光栅化 - 根据三角形特征自动选择策略
    pub fn rasterize_triangles(
        triangles: &[TriangleData],
        width: usize,
        height: usize,
        depth_buffer: &[AtomicF32],
        color_buffer: &[AtomicU8],
        settings: &RenderSettings,
        frame_buffer: &crate::core::frame_buffer::FrameBuffer, // 帧缓冲区引用
    ) {
        if triangles.is_empty() {
            return;
        }

        // 快速策略选择
        let strategy = Self::choose_strategy(triangles, width, height);

        match strategy {
            RenderStrategy::LargeTrianglePixelParallel => {
                Self::rasterize_large_triangles_pixel_parallel(
                    triangles,
                    width,
                    height,
                    depth_buffer,
                    color_buffer,
                    settings,
                    frame_buffer,
                );
            }
            RenderStrategy::SmallTriangleParallel => {
                Self::rasterize_small_triangles_parallel(
                    triangles,
                    width,
                    height,
                    depth_buffer,
                    color_buffer,
                    settings,
                    frame_buffer,
                );
            }
            RenderStrategy::Mixed => {
                Self::rasterize_mixed_strategy(
                    triangles,
                    width,
                    height,
                    depth_buffer,
                    color_buffer,
                    settings,
                    frame_buffer,
                );
            }
        }
    }

    /// 快速策略选择
    fn choose_strategy(triangles: &[TriangleData], width: usize, height: usize) -> RenderStrategy {
        let screen_area = (width * height) as f32;
        let triangle_count = triangles.len();

        // 估算平均三角形大小
        let avg_triangle_size = if triangle_count > 0 {
            let total_area: f32 = triangles
                .iter()
                .take(triangle_count.min(50)) // 只采样前50个三角形
                .map(|tri| Self::estimate_triangle_area(tri))
                .sum();
            total_area / triangle_count.min(50) as f32
        } else {
            0.0
        };

        // 策略决策
        if triangle_count > 2000 && avg_triangle_size > screen_area * 0.0005 {
            RenderStrategy::Mixed
        } else if avg_triangle_size > 500.0 || triangle_count < 100 {
            RenderStrategy::LargeTrianglePixelParallel
        } else {
            RenderStrategy::SmallTriangleParallel
        }
    }

    /// 估算三角形面积
    fn estimate_triangle_area(triangle: &TriangleData) -> f32 {
        let v0 = &triangle.vertices[0].pix;
        let v1 = &triangle.vertices[1].pix;
        let v2 = &triangle.vertices[2].pix;

        0.5 * ((v1.x - v0.x) * (v2.y - v0.y) - (v2.x - v0.x) * (v1.y - v0.y)).abs()
    }

    /// 大三角形像素级并行策略
    fn rasterize_large_triangles_pixel_parallel(
        triangles: &[TriangleData],
        width: usize,
        height: usize,
        depth_buffer: &[AtomicF32],
        color_buffer: &[AtomicU8],
        settings: &RenderSettings,
        frame_buffer: &crate::core::frame_buffer::FrameBuffer,
    ) {
        triangles.par_iter().for_each(|triangle| {
            Self::rasterize_triangle_pixel_parallel(
                triangle,
                width,
                height,
                depth_buffer,
                color_buffer,
                settings,
                frame_buffer,
            );
        });
    }

    /// 小三角形传统并行策略
    fn rasterize_small_triangles_parallel(
        triangles: &[TriangleData],
        width: usize,
        height: usize,
        depth_buffer: &[AtomicF32],
        color_buffer: &[AtomicU8],
        settings: &RenderSettings,
        frame_buffer: &crate::core::frame_buffer::FrameBuffer,
    ) {
        triangles.par_iter().for_each(|triangle| {
            crate::core::rasterizer::pixel_processor::rasterize_triangle(
                triangle,
                width,
                height,
                depth_buffer,
                color_buffer,
                settings,
                frame_buffer,
            );
        });
    }

    /// 混合策略 - 分别处理大小三角形
    fn rasterize_mixed_strategy(
        triangles: &[TriangleData],
        width: usize,
        height: usize,
        depth_buffer: &[AtomicF32],
        color_buffer: &[AtomicU8],
        settings: &RenderSettings,
        frame_buffer: &crate::core::frame_buffer::FrameBuffer,
    ) {
        let screen_area = (width * height) as f32;
        let large_threshold = screen_area * 0.001;

        let (large_indices, small_indices): (Vec<_>, Vec<_>) = triangles
            .iter()
            .enumerate()
            .partition(|(_, tri)| Self::estimate_triangle_area(tri) > large_threshold);

        // 并行处理两组，使用索引访问原始切片
        rayon::join(
            || {
                large_indices.par_iter().for_each(|(idx, _)| {
                    Self::rasterize_triangle_pixel_parallel(
                        &triangles[*idx],
                        width,
                        height,
                        depth_buffer,
                        color_buffer,
                        settings,
                        frame_buffer,
                    );
                });
            },
            || {
                small_indices.par_iter().for_each(|(idx, _)| {
                    crate::core::rasterizer::pixel_processor::rasterize_triangle(
                        &triangles[*idx],
                        width,
                        height,
                        depth_buffer,
                        color_buffer,
                        settings,
                        frame_buffer,
                    );
                });
            },
        );
    }

    /// 单个大三角形的像素级并行处理
    fn rasterize_triangle_pixel_parallel(
        triangle: &TriangleData,
        width: usize,
        height: usize,
        depth_buffer: &[AtomicF32],
        color_buffer: &[AtomicU8],
        settings: &RenderSettings,
        frame_buffer: &crate::core::frame_buffer::FrameBuffer,
    ) {
        let v0 = &triangle.vertices[0].pix;
        let v1 = &triangle.vertices[1].pix;
        let v2 = &triangle.vertices[2].pix;

        let min_x = v0.x.min(v1.x).min(v2.x).floor().max(0.0) as usize;
        let min_y = v0.y.min(v1.y).min(v2.y).floor().max(0.0) as usize;
        let max_x = v0.x.max(v1.x).max(v2.x).ceil().min(width as f32) as usize;
        let max_y = v0.y.max(v1.y).max(v2.y).ceil().min(height as f32) as usize;

        if max_x <= min_x || max_y <= min_y {
            return;
        }

        // 像素级并行处理
        (min_y..max_y).into_par_iter().for_each(|y| {
            for x in min_x..max_x {
                let pixel_center = Point2::new(x as f32 + 0.5, y as f32 + 0.5);
                let pixel_index = y * width + x;

                crate::core::rasterizer::pixel_processor::rasterize_pixel(
                    triangle,
                    pixel_center,
                    pixel_index,
                    x,
                    y,
                    depth_buffer,
                    color_buffer,
                    settings,
                    frame_buffer,
                );
            }
        });
    }
}

#[derive(Debug, Clone, Copy)]
enum RenderStrategy {
    LargeTrianglePixelParallel,
    SmallTriangleParallel,
    Mixed,
}
