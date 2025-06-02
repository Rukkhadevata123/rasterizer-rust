use super::msaa::{MSAAPattern, MSAASample, generate_sample_points, resolve_msaa_samples}; // 新增导入
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

/// 光栅化单个三角形
/// 光栅化单个三角形 - 添加MSAA支持
pub fn rasterize_triangle(
    triangle: &TriangleData,
    width: usize,
    height: usize,
    depth_buffer: &[AtomicF32],
    color_buffer: &[AtomicU8],
    settings: &RenderSettings,
    frame_buffer: &crate::core::frame_buffer::FrameBuffer,
) {
    if !triangle.is_valid() {
        return;
    }

    let bbox = match BoundingBox::from_triangle(triangle, width, height) {
        Some(bbox) => bbox,
        None => return,
    };

    let use_phong_or_pbr = (settings.use_pbr || settings.use_phong)
        && triangle.vertices[0].normal_view.is_some()
        && triangle.vertices[0].position_view.is_some()
        && !triangle.lights.is_empty();

    let use_texture = !matches!(
        triangle.texture_source,
        super::triangle_data::TextureSource::None
    );

    let ambient_contribution = calculate_ambient_contribution(triangle);

    // 新增：MSAA模式检测
    let msaa_pattern = MSAAPattern::get_pattern(settings.msaa_samples);
    let use_msaa = settings.msaa_samples > 1;

    bbox.for_each_pixel(|x, y| {
        let pixel_index = y * width + x;

        if use_msaa {
            // MSAA路径
            process_pixel_msaa(
                triangle,
                x,
                y,
                pixel_index,
                &msaa_pattern,
                use_phong_or_pbr,
                use_texture,
                &ambient_contribution,
                depth_buffer,
                color_buffer,
                settings,
                frame_buffer,
            );
        } else {
            // 原有路径 - 无变化
            let pixel_center = Point2::new(x as f32 + 0.5, y as f32 + 0.5);
            process_pixel(
                triangle,
                pixel_center,
                pixel_index,
                x,
                y,
                use_phong_or_pbr,
                use_texture,
                &ambient_contribution,
                depth_buffer,
                color_buffer,
                settings,
                frame_buffer,
            );
        }
    });
}

/// 改进：MSAA像素处理 - 修复深度测试逻辑
#[allow(clippy::too_many_arguments)]
fn process_pixel_msaa(
    triangle: &TriangleData,
    pixel_x: usize,
    pixel_y: usize,
    pixel_index: usize,
    msaa_pattern: &MSAAPattern,
    use_phong_or_pbr: bool,
    use_texture: bool,
    ambient_contribution: &Color,
    depth_buffer: &[AtomicF32],
    color_buffer: &[AtomicU8],
    settings: &RenderSettings,
    frame_buffer: &crate::core::frame_buffer::FrameBuffer,
) {
    let sample_points = generate_sample_points(pixel_x, pixel_y, msaa_pattern);
    let mut samples = vec![MSAASample::default(); sample_points.len()];
    let mut any_hit = false;

    // 对每个采样点进行渲染
    for (i, &sample_point) in sample_points.iter().enumerate() {
        let sample_result = process_sample_point(
            triangle,
            sample_point,
            pixel_x,
            pixel_y,
            use_phong_or_pbr,
            use_texture,
            ambient_contribution,
            settings,
            frame_buffer,
        );

        if let Some((color, depth)) = sample_result {
            samples[i] = MSAASample {
                color,
                depth,
                hit: true,
            };
            any_hit = true;
        }
    }

    // 如果没有任何采样点命中，直接返回
    if !any_hit {
        return;
    }

    // 解析MSAA结果
    let (final_color, final_depth) = resolve_msaa_samples(&samples, msaa_pattern);

    // 修复：深度测试应该使用解析后的深度
    if settings.use_zbuffer && final_depth < f32::INFINITY {
        let current_depth_atomic = &depth_buffer[pixel_index];
        let old_depth = current_depth_atomic.fetch_min(final_depth, Ordering::Relaxed);
        if old_depth <= final_depth {
            return; // 深度测试失败
        }
    }

    // 写入最终颜色
    write_pixel_color(pixel_index, &final_color, color_buffer, settings.use_gamma);
}

/// 新增：处理单个采样点
fn process_sample_point(
    triangle: &TriangleData,
    sample_point: Point2<f32>,
    pixel_x: usize,
    pixel_y: usize,
    use_phong_or_pbr: bool,
    use_texture: bool,
    ambient_contribution: &Color,
    settings: &RenderSettings,
    frame_buffer: &crate::core::frame_buffer::FrameBuffer,
) -> Option<(Vector3<f32>, f32)> {
    let v0 = &triangle.vertices[0].pix;
    let v1 = &triangle.vertices[1].pix;
    let v2 = &triangle.vertices[2].pix;

    let bary = barycentric_coordinates(sample_point, *v0, *v1, *v2)?;

    if !is_inside_triangle(bary) {
        return None;
    }

    // Alpha检查
    let final_alpha = get_effective_alpha(triangle, settings);
    if final_alpha <= 0.01 {
        return None;
    }

    // 线框模式检查
    if settings.wireframe && !is_on_triangle_edge(sample_point, *v0, *v1, *v2, 1.0) {
        return None;
    }

    // 深度插值
    let interpolated_depth = interpolate_depth(
        bary,
        triangle.vertices[0].z_view,
        triangle.vertices[1].z_view,
        triangle.vertices[2].z_view,
        settings.is_perspective() && triangle.is_perspective,
    );

    if !interpolated_depth.is_finite() || interpolated_depth >= f32::INFINITY {
        return None;
    }

    // 颜色计算
    let material_color = calculate_pixel_color(
        triangle,
        bary,
        settings,
        use_phong_or_pbr,
        use_texture,
        ambient_contribution,
    );

    // Alpha混合
    let final_color =
        apply_alpha_blending(&material_color, final_alpha, pixel_x, pixel_y, frame_buffer);

    Some((final_color, interpolated_depth))
}

/// 处理单个像素
#[allow(clippy::too_many_arguments)]
pub fn rasterize_pixel(
    triangle: &TriangleData,
    pixel_center: Point2<f32>,
    pixel_index: usize,
    pixel_x: usize,
    pixel_y: usize,
    depth_buffer: &[AtomicF32],
    color_buffer: &[AtomicU8],
    settings: &RenderSettings,
    frame_buffer: &crate::core::frame_buffer::FrameBuffer,
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
        pixel_x,
        pixel_y,
        use_phong_or_pbr,
        use_texture,
        &ambient_contribution,
        depth_buffer,
        color_buffer,
        settings,
        frame_buffer,
    );
}

/// 核心像素处理 - 优化Alpha处理
#[allow(clippy::too_many_arguments)]
fn process_pixel(
    triangle: &TriangleData,
    pixel_center: Point2<f32>,
    pixel_index: usize,
    pixel_x: usize,
    pixel_y: usize,
    use_phong_or_pbr: bool,
    use_texture: bool,
    ambient_contribution: &Color,
    depth_buffer: &[AtomicF32],
    color_buffer: &[AtomicU8],
    settings: &RenderSettings,
    frame_buffer: &crate::core::frame_buffer::FrameBuffer,
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

    // 统一的Alpha检查 - 合并全局和材质Alpha
    let final_alpha = get_effective_alpha(triangle, settings);
    if final_alpha <= 0.01 {
        return; // 完全透明，跳过处理
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

    let material_color = calculate_pixel_color(
        triangle,
        bary,
        settings,
        use_phong_or_pbr,
        use_texture,
        ambient_contribution,
    );

    // Alpha混合处理 - 你的实现是正确的！
    let final_color =
        apply_alpha_blending(&material_color, final_alpha, pixel_x, pixel_y, frame_buffer);

    write_pixel_color(pixel_index, &final_color, color_buffer, settings.use_gamma);
}

/// 统一的Alpha获取函数 - 消除代码重复
fn get_effective_alpha(triangle: &TriangleData, settings: &RenderSettings) -> f32 {
    let material_alpha = if let Some(material_view) = &triangle.material_view {
        match material_view {
            crate::material_system::materials::MaterialView::BlinnPhong(material) => material.alpha,
            crate::material_system::materials::MaterialView::PBR(material) => material.alpha,
        }
    } else {
        1.0
    };

    // 合并全局Alpha和材质Alpha
    (material_alpha * settings.alpha).clamp(0.0, 1.0)
}

/// 优化的Alpha混合 - 简化参数
fn apply_alpha_blending(
    material_color: &Color,
    alpha: f32,
    pixel_x: usize,
    pixel_y: usize,
    frame_buffer: &crate::core::frame_buffer::FrameBuffer,
) -> Vector3<f32> {
    if alpha >= 1.0 {
        return Vector3::new(material_color.x, material_color.y, material_color.z);
    }

    if alpha <= 0.0 {
        if let Some(bg_color) = frame_buffer.get_pixel_color(pixel_x, pixel_y) {
            return bg_color;
        } else {
            return Vector3::new(0.0, 0.0, 0.0);
        }
    }

    // 标准Alpha混合：result = source * alpha + background * (1 - alpha)
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
