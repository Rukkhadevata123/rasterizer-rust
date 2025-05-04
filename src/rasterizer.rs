use crate::color_utils::Color;
use crate::interpolation::{
    barycentric_coordinates,
    interpolate_depth,
    interpolate_normal,
    interpolate_position, // 添加新的插值函数
    interpolate_texcoords,
    is_inside_triangle,
};
use crate::lighting::{Light, SimpleMaterial, calculate_blinn_phong}; // 添加光照计算
use crate::texture_utils::Texture;
use atomic_float::AtomicF32;
use nalgebra::{Point2, Point3, Vector2, Vector3}; // 添加 Point3, Vector3
use std::sync::Mutex;
use std::sync::atomic::Ordering; // Using Mutex for color buffer for simplicity first

/// Input data for rasterizing a single triangle.
pub struct TriangleData<'a> {
    // Pixel coordinates (2D) + View space Z (for interpolation)
    pub v1_pix: Point2<f32>,
    pub v2_pix: Point2<f32>,
    pub v3_pix: Point2<f32>,
    pub z1_view: f32,
    pub z2_view: f32,
    pub z3_view: f32,
    // Attributes
    pub base_color: Color, // Base color (material diffuse or face color)
    pub lit_color: Color,  // Calculated light contribution
    pub tc1: Option<Vector2<f32>>, // Texture coordinates
    pub tc2: Option<Vector2<f32>>,
    pub tc3: Option<Vector2<f32>>,
    pub texture: Option<&'a Texture>,
    // Settings
    pub is_perspective: bool,
    pub use_zbuffer: bool,

    // Phong 着色所需的额外数据
    pub n1_view: Option<Vector3<f32>>, // 视图空间法线
    pub n2_view: Option<Vector3<f32>>,
    pub n3_view: Option<Vector3<f32>>,
    pub v1_view: Option<Point3<f32>>, // 视图空间位置
    pub v2_view: Option<Point3<f32>>,
    pub v3_view: Option<Point3<f32>>,
    pub material: Option<SimpleMaterial>, // 材质信息 - 直接持有值，不是引用
    pub light: Option<Light>,             // 光源信息 - 直接持有值，不是引用
    pub use_phong: bool,                  // 是否使用 Phong 着色
}

/// Rasterizes a single triangle onto the frame buffers.
/// Uses atomic operations for depth buffer and mutex for color buffer.
pub fn rasterize_triangle(
    triangle: &TriangleData,
    width: usize,
    height: usize,
    depth_buffer: &[AtomicF32],    // Use slice of atomics
    color_buffer: &Mutex<Vec<u8>>, // Use Mutex for simplicity first
) {
    // 1. Calculate bounding box
    let min_x = (triangle
        .v1_pix
        .x
        .min(triangle.v2_pix.x)
        .min(triangle.v3_pix.x))
    .floor()
    .max(0.0) as usize;
    let min_y = (triangle
        .v1_pix
        .y
        .min(triangle.v2_pix.y)
        .min(triangle.v3_pix.y))
    .floor()
    .max(0.0) as usize;
    let max_x = (triangle
        .v1_pix
        .x
        .max(triangle.v2_pix.x)
        .max(triangle.v3_pix.x))
    .ceil()
    .min(width as f32) as usize;
    let max_y = (triangle
        .v1_pix
        .y
        .max(triangle.v2_pix.y)
        .max(triangle.v3_pix.y))
    .ceil()
    .min(height as f32) as usize;

    // Check for invalid bounding box
    if max_x <= min_x || max_y <= min_y {
        return;
    }

    // Optional: Early exit for degenerate triangles (check area)
    // let area_x2 = ((triangle.v2_pix.x - triangle.v1_pix.x) * (triangle.v3_pix.y - triangle.v1_pix.y)
    //             - (triangle.v3_pix.x - triangle.v1_pix.x) * (triangle.v2_pix.y - triangle.v1_pix.y)).abs();
    // if area_x2 < 1e-3 { return; }

    // 2. Iterate over pixels in the bounding box
    for y in min_y..max_y {
        for x in min_x..max_x {
            let pixel_center = Point2::new(x as f32 + 0.5, y as f32 + 0.5);
            let pixel_index = y * width + x;

            // 3. Calculate barycentric coordinates
            if let Some(bary) = barycentric_coordinates(
                pixel_center,
                triangle.v1_pix,
                triangle.v2_pix,
                triangle.v3_pix,
            ) {
                // 4. Check if pixel is inside the triangle
                if is_inside_triangle(bary) {
                    // 5. Interpolate depth
                    let interpolated_depth = interpolate_depth(
                        bary,
                        triangle.z1_view,
                        triangle.z2_view,
                        triangle.z3_view,
                        triangle.is_perspective,
                    );

                    // Check if depth is valid (not behind camera / too far)
                    if interpolated_depth.is_finite() && interpolated_depth < f32::INFINITY {
                        // 6. Depth Test (Atomic)
                        let current_depth_atomic = &depth_buffer[pixel_index];
                        let previous_depth = current_depth_atomic.load(Ordering::Relaxed);

                        if !triangle.use_zbuffer || interpolated_depth < previous_depth {
                            // Attempt to update depth atomically
                            // fetch_min returns the *previous* value before the potential update
                            let old_depth_before_update = current_depth_atomic
                                .fetch_min(interpolated_depth, Ordering::Relaxed);

                            // Only write color if *this thread* successfully updated the depth
                            if !triangle.use_zbuffer || old_depth_before_update > interpolated_depth
                            {
                                // 7. Calculate final color
                                let final_color: Color = if triangle.use_phong
                                    && triangle.n1_view.is_some()
                                    && triangle.material.is_some()
                                    && triangle.light.is_some()
                                    && triangle.v1_view.is_some()
                                {
                                    // --- Phong 着色（逐像素光照）---

                                    // 插值法线
                                    let interp_normal = interpolate_normal(
                                        bary,
                                        triangle.n1_view.unwrap(),
                                        triangle.n2_view.unwrap(),
                                        triangle.n3_view.unwrap(),
                                        triangle.is_perspective,
                                        triangle.z1_view,
                                        triangle.z2_view,
                                        triangle.z3_view,
                                    );

                                    // 插值视图空间位置
                                    let interp_position = interpolate_position(
                                        bary,
                                        triangle.v1_view.unwrap(),
                                        triangle.v2_view.unwrap(),
                                        triangle.v3_view.unwrap(),
                                        triangle.is_perspective,
                                        triangle.z1_view,
                                        triangle.z2_view,
                                        triangle.z3_view,
                                    );

                                    // 计算视线方向
                                    let view_dir = (-interp_position.coords).normalize();

                                    // 计算 Phong/Blinn-Phong 光照
                                    let light = triangle.light.as_ref().unwrap();
                                    let material = triangle.material.as_ref().unwrap();
                                    let pixel_lit_color = calculate_blinn_phong(
                                        interp_position,
                                        interp_normal,
                                        view_dir,
                                        light,
                                        material,
                                    );

                                    // 应用纹理（如果有）
                                    if let (Some(tex), Some(tc1), Some(tc2), Some(tc3)) =
                                        (triangle.texture, triangle.tc1, triangle.tc2, triangle.tc3)
                                    {
                                        let interp_tc = interpolate_texcoords(
                                            bary,
                                            tc1,
                                            tc2,
                                            tc3,
                                            triangle.z1_view,
                                            triangle.z2_view,
                                            triangle.z3_view,
                                            triangle.is_perspective,
                                        );
                                        let texel = tex.sample(interp_tc.x, interp_tc.y);
                                        let texel_color = Color::new(texel[0], texel[1], texel[2]);
                                        texel_color.component_mul(&pixel_lit_color)
                                    } else {
                                        // 没有纹理，使用基础颜色
                                        triangle.base_color.component_mul(&pixel_lit_color)
                                    }
                                } else {
                                    // --- 使用预计算的面光照（Flat/Gouraud 着色） ---
                                    if let (Some(tex), Some(tc1), Some(tc2), Some(tc3)) =
                                        (triangle.texture, triangle.tc1, triangle.tc2, triangle.tc3)
                                    {
                                        // 插值纹理坐标
                                        let interp_tc = interpolate_texcoords(
                                            bary,
                                            tc1,
                                            tc2,
                                            tc3,
                                            triangle.z1_view,
                                            triangle.z2_view,
                                            triangle.z3_view,
                                            triangle.is_perspective,
                                        );
                                        // 采样纹理
                                        let texel = tex.sample(interp_tc.x, interp_tc.y);
                                        let texel_color = Color::new(texel[0], texel[1], texel[2]);
                                        // 应用光照到纹理颜色
                                        texel_color.component_mul(&triangle.lit_color)
                                    } else {
                                        // 应用光照到基础颜色
                                        triangle.base_color.component_mul(&triangle.lit_color)
                                    }
                                };

                                // 8. Write color to buffer (using Mutex)
                                {
                                    // Scope for MutexGuard
                                    let mut cbuf_guard = color_buffer.lock().unwrap();
                                    let buffer_start_index = pixel_index * 3;
                                    if buffer_start_index + 2 < cbuf_guard.len() {
                                        cbuf_guard[buffer_start_index] =
                                            (final_color.x * 255.0).clamp(0.0, 255.0) as u8;
                                        cbuf_guard[buffer_start_index + 1] =
                                            (final_color.y * 255.0).clamp(0.0, 255.0) as u8;
                                        cbuf_guard[buffer_start_index + 2] =
                                            (final_color.z * 255.0).clamp(0.0, 255.0) as u8;
                                    }
                                } // MutexGuard dropped here
                            }
                        }
                    }
                }
            }
        }
    }
}
