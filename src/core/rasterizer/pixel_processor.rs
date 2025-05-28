use super::color_calculator::calculate_pixel_color;
use super::lighting_effects::calculate_ambient_contribution;
use super::triangle_data::{BoundingBox, TriangleData};
use crate::geometry::culling::is_on_triangle_edge;
use crate::geometry::interpolation::{
    barycentric_coordinates, interpolate_depth, is_inside_triangle,
};
use crate::io::render_settings::RenderSettings;
use crate::material_system::color::linear_rgb_to_u8;
use atomic_float::AtomicF32;
use nalgebra::Point2;
use std::sync::atomic::{AtomicU8, Ordering};

/// 渲染上下文 - 缓存预计算值
struct RenderContext {
    use_phong_or_pbr: bool,
    use_texture: bool,
    ambient_contribution: nalgebra::Vector3<f32>,
}

impl RenderContext {
    fn new(triangle: &TriangleData, settings: &RenderSettings) -> Self {
        let use_phong_or_pbr = (settings.use_pbr || settings.use_phong)
            && triangle.vertices[0].normal_view.is_some()
            && triangle.vertices[0].position_view.is_some()
            && !triangle.lights.is_empty();

        let use_texture = !matches!(
            triangle.texture_source,
            super::triangle_data::TextureSource::None
        );
        let ambient_contribution = calculate_ambient_contribution(triangle);

        Self {
            use_phong_or_pbr,
            use_texture,
            ambient_contribution,
        }
    }
}

/// 光栅化单个三角形
pub fn rasterize_triangle(
    triangle: &TriangleData,
    width: usize,
    height: usize,
    depth_buffer: &[AtomicF32],
    color_buffer: &[AtomicU8],
    settings: &RenderSettings,
) {
    if !triangle.is_valid() {
        return;
    }

    let bbox = match BoundingBox::from_triangle(triangle, width, height) {
        Some(bbox) => bbox,
        None => return,
    };

    let render_context = RenderContext::new(triangle, settings);

    bbox.for_each_pixel(|x, y| {
        let pixel_center = Point2::new(x as f32 + 0.5, y as f32 + 0.5);
        let pixel_index = y * width + x;

        process_pixel(
            triangle,
            pixel_center,
            pixel_index,
            &render_context,
            depth_buffer,
            color_buffer,
            settings,
        );
    });
}

/// 处理单个像素
pub fn rasterize_pixel(
    triangle: &TriangleData,
    pixel_center: Point2<f32>,
    pixel_index: usize,
    depth_buffer: &[AtomicF32],
    color_buffer: &[AtomicU8],
    settings: &RenderSettings,
) {
    let render_context = RenderContext::new(triangle, settings);

    process_pixel(
        triangle,
        pixel_center,
        pixel_index,
        &render_context,
        depth_buffer,
        color_buffer,
        settings,
    );
}

/// 核心像素处理
fn process_pixel(
    triangle: &TriangleData,
    pixel_center: Point2<f32>,
    pixel_index: usize,
    render_context: &RenderContext,
    depth_buffer: &[AtomicF32],
    color_buffer: &[AtomicU8],
    settings: &RenderSettings,
) {
    let v0 = &triangle.vertices[0].pix;
    let v1 = &triangle.vertices[1].pix;
    let v2 = &triangle.vertices[2].pix;

    let bary = match barycentric_coordinates(pixel_center, *v0, *v1, *v2) {
        Some(bary) => bary,
        None => return,
    };

    if !is_inside_triangle(bary) {
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

    let final_color = calculate_pixel_color(
        triangle,
        bary,
        settings,
        render_context.use_phong_or_pbr,
        render_context.use_texture,
        &render_context.ambient_contribution,
    );

    write_pixel_color(pixel_index, &final_color, color_buffer, settings.use_gamma);
}

#[inline]
fn write_pixel_color(
    pixel_index: usize,
    color: &nalgebra::Vector3<f32>,
    color_buffer: &[AtomicU8],
    apply_gamma: bool,
) {
    let buffer_start_index = pixel_index * 3;
    if buffer_start_index + 2 < color_buffer.len() {
        let [r, g, b] = linear_rgb_to_u8(color, apply_gamma);
        color_buffer[buffer_start_index].store(r, Ordering::Relaxed);
        color_buffer[buffer_start_index + 1].store(g, Ordering::Relaxed);
        color_buffer[buffer_start_index + 2].store(b, Ordering::Relaxed);
    }
}
