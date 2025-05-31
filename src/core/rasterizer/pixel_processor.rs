use super::shading::{calculate_ambient_contribution, calculate_pixel_color};
use super::triangle_data::{BoundingBox, TriangleData};
use crate::geometry::culling::is_on_triangle_edge;
use crate::geometry::interpolation::{
    barycentric_coordinates, interpolate_depth, is_inside_triangle,
};
use crate::io::render_settings::RenderSettings;
use crate::material_system::color::{Color, linear_rgb_to_u8};
use atomic_float::AtomicF32;
use nalgebra::{Point2, Vector3};
use std::sync::atomic::{AtomicU8, Ordering};

/// 光栅化单个三角形 - 需要传递帧缓冲区用于alpha混合
pub fn rasterize_triangle(
    triangle: &TriangleData,
    width: usize,
    height: usize,
    depth_buffer: &[AtomicF32],
    color_buffer: &[AtomicU8],
    settings: &RenderSettings,
    frame_buffer: &crate::core::frame_buffer::FrameBuffer, // 新增：帧缓冲区引用
) {
    if !triangle.is_valid() {
        return;
    }

    let bbox = match BoundingBox::from_triangle(triangle, width, height) {
        Some(bbox) => bbox,
        None => return,
    };

    // 预计算一次，避免重复计算
    let use_phong_or_pbr = (settings.use_pbr || settings.use_phong)
        && triangle.vertices[0].normal_view.is_some()
        && triangle.vertices[0].position_view.is_some()
        && !triangle.lights.is_empty();

    let use_texture = !matches!(
        triangle.texture_source,
        super::triangle_data::TextureSource::None
    );

    let ambient_contribution = calculate_ambient_contribution(triangle);

    bbox.for_each_pixel(|x, y| {
        let pixel_center = Point2::new(x as f32 + 0.5, y as f32 + 0.5);
        let pixel_index = y * width + x;

        process_pixel(
            triangle,
            pixel_center,
            pixel_index,
            x, // 新增：传递x坐标
            y, // 新增：传递y坐标
            use_phong_or_pbr,
            use_texture,
            &ambient_contribution,
            depth_buffer,
            color_buffer,
            settings,
            frame_buffer, // 新增：传递帧缓冲区
        );
    });
}

/// 处理单个像素 - 需要传递帧缓冲区用于alpha混合
#[allow(clippy::too_many_arguments)]
pub fn rasterize_pixel(
    triangle: &TriangleData,
    pixel_center: Point2<f32>,
    pixel_index: usize,
    pixel_x: usize, // 新增：像素X坐标
    pixel_y: usize, // 新增：像素Y坐标
    depth_buffer: &[AtomicF32],
    color_buffer: &[AtomicU8],
    settings: &RenderSettings,
    frame_buffer: &crate::core::frame_buffer::FrameBuffer, // 新增：帧缓冲区引用
) {
    let use_phong_or_pbr = (settings.use_pbr || settings.use_phong)
        && triangle.vertices[0].normal_view.is_some()
        && triangle.vertices[0].position_view.is_some()
        && !triangle.lights.is_empty();

    let use_texture = !matches!(
        triangle.texture_source,
        super::triangle_data::TextureSource::None
    );

    let ambient_contribution = calculate_ambient_contribution(triangle);

    process_pixel(
        triangle,
        pixel_center,
        pixel_index,
        pixel_x, // 新增：传递x坐标
        pixel_y, // 新增：传递y坐标
        use_phong_or_pbr,
        use_texture,
        &ambient_contribution,
        depth_buffer,
        color_buffer,
        settings,
        frame_buffer, // 新增：传递帧缓冲区
    );
}

/// 核心像素处理 - 修改参数以支持alpha混合
#[allow(clippy::too_many_arguments)]
fn process_pixel(
    triangle: &TriangleData,
    pixel_center: Point2<f32>,
    pixel_index: usize,
    pixel_x: usize, // 新增：像素X坐标
    pixel_y: usize, // 新增：像素Y坐标
    use_phong_or_pbr: bool,
    use_texture: bool,
    ambient_contribution: &Color,
    depth_buffer: &[AtomicF32],
    color_buffer: &[AtomicU8],
    settings: &RenderSettings,
    frame_buffer: &crate::core::frame_buffer::FrameBuffer, // 新增：帧缓冲区引用
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

    if settings.alpha <= 0.01 {
        return; // 完全不处理这个像素
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

    // 修改：传递所有必需的参数给calculate_pixel_color
    let final_color = calculate_pixel_color(
        triangle,
        bary,
        settings,
        use_phong_or_pbr,
        use_texture,
        ambient_contribution,
        pixel_x,      // 新增：传递像素X坐标
        pixel_y,      // 新增：传递像素Y坐标
        frame_buffer, // 新增：传递帧缓冲区引用
    );

    write_pixel_color(pixel_index, &final_color, color_buffer, settings.use_gamma);
}

#[inline]
fn write_pixel_color(
    pixel_index: usize,
    color: &Vector3<f32>,
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
