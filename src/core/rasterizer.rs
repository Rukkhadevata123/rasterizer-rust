use crate::core::framebuffer::FrameBuffer;
use crate::core::math::interpolation::{
    barycentric_coordinates, is_inside_triangle, perspective_correct_interpolate,
};
use crate::core::math::transform::{apply_perspective_division, ndc_to_screen};
use crate::core::pipeline::Shader;
use crate::scene::material::Material;
use nalgebra::{Point2, Vector4};

/// The Rasterizer is responsible for drawing geometric primitives onto the FrameBuffer.
pub struct Rasterizer;

impl Default for Rasterizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Rasterizer {
    pub fn new() -> Self {
        Self {}
    }

    pub fn rasterize_triangle<S: Shader>(
        &self,
        framebuffer: &FrameBuffer,
        shader: &S,
        clip_coords: &[Vector4<f32>; 3],
        varyings: &[S::Varying; 3],
        material: Option<&Material>,
    ) {
        // Use the actual buffer dimensions for rasterization
        let width = framebuffer.buffer_width as f32;
        let height = framebuffer.buffer_height as f32;

        let v0 = &clip_coords[0];
        let v1 = &clip_coords[1];
        let v2 = &clip_coords[2];

        // Check X axis (Left/Right)
        // If all x > w, it's to the right of the screen.
        // If all x < -w, it's to the left of the screen.
        if (v0.x > v0.w && v1.x > v1.w && v2.x > v2.w)
            || (v0.x < -v0.w && v1.x < -v1.w && v2.x < -v2.w)
        {
            return;
        }

        // Check Y axis (Top/Bottom)
        if (v0.y > v0.w && v1.y > v1.w && v2.y > v2.w)
            || (v0.y < -v0.w && v1.y < -v1.w && v2.y < -v2.w)
        {
            return;
        }

        // Check Z axis (Near/Far)
        // Note: Depending on projection matrix, Z range is usually [-w, w] or [0, w].
        // Assuming standard OpenGL-style [-w, w] here.
        // TODO: may check
        if (v0.z > v0.w && v1.z > v1.w && v2.z > v2.w)
            || (v0.z < -v0.w && v1.z < -v1.w && v2.z < -v2.w)
        {
            return;
        }

        // If any vertex is behind the camera (w <= epsilon), discard the triangle.
        // This is a crude "clipping" replacement to prevent math errors.
        if v0.w <= 1e-6 || v1.w <= 1e-6 || v2.w <= 1e-6 {
            return;
        }

        // 1. Perspective Division & Viewport Transform
        let mut screen_coords = [Point2::origin(); 3];
        let mut w_values = [0.0; 3];

        for i in 0..3 {
            let ndc = apply_perspective_division(&clip_coords[i]);
            w_values[i] = clip_coords[i].w;
            screen_coords[i] = ndc_to_screen(ndc.x, ndc.y, width, height);
        }

        // 2. Backface Culling
        let v0 = screen_coords[0];
        let v1 = screen_coords[1];
        let v2 = screen_coords[2];
        let edge1 = v1 - v0;
        let edge2 = v2 - v1;
        let signed_area = edge1.x * edge2.y - edge1.y * edge2.x;

        // Assuming CCW winding order. If area is positive, it's facing away (or towards, depending on Y-axis).
        // In our screen space (+Y down), CCW produces negative area for front-facing?
        // Let's stick to the logic that worked for you: if signed_area >= 0.0, cull.
        // TODO: Need to be an optional choice
        if signed_area >= 0.0 {
            return;
        }

        // Precompute NDC z values for depth interpolation
        let z0_ndc = clip_coords[0].z / clip_coords[0].w;
        let z1_ndc = clip_coords[1].z / clip_coords[1].w;
        let z2_ndc = clip_coords[2].z / clip_coords[2].w;

        // 3. Compute Bounding Box
        let (min_x, min_y, max_x, max_y) = self.compute_bounding_box(&screen_coords);

        // Clip against screen bounds (Scissor Test)
        if max_x < 0
            || max_y < 0
            || min_x >= framebuffer.buffer_width as i32
            || min_y >= framebuffer.buffer_height as i32
        {
            return;
        }

        let start_x = min_x.max(0) as usize;
        let end_x = (max_x.min(framebuffer.buffer_width as i32 - 1)) as usize;
        let start_y = min_y.max(0) as usize;
        let end_y = (max_y.min(framebuffer.buffer_height as i32 - 1)) as usize;

        // 4. Pixel Loop
        // TODO: Parallelize this loop for performance
        for y in start_y..=end_y {
            for x in start_x..=end_x {
                let pixel_center = Point2::new(x as f32 + 0.5, y as f32 + 0.5);

                if let Some(bary) = barycentric_coordinates(
                    pixel_center,
                    screen_coords[0],
                    screen_coords[1],
                    screen_coords[2],
                ) && is_inside_triangle(bary)
                {
                    let z_ndc = z0_ndc * bary.x + z1_ndc * bary.y + z2_ndc * bary.z;
                    let depth = z_ndc * 0.5 + 0.5;

                    // Use atomic depth test. Only if it passes do we calculate color and write.
                    if framebuffer.depth_test_and_update(x, y, depth) {
                        let interpolated_varying = perspective_correct_interpolate(
                            bary,
                            varyings[0],
                            varyings[1],
                            varyings[2],
                            w_values[0],
                            w_values[1],
                            w_values[2],
                        );

                        let color = shader.fragment(interpolated_varying, material);

                        // Use thread-safe pixel setter (uses locks internally)
                        framebuffer.set_pixel_safe(x, y, color);
                    }
                }
            }
        }
    }

    fn compute_bounding_box(&self, points: &[Point2<f32>; 3]) -> (i32, i32, i32, i32) {
        let min_x = points[0].x.min(points[1].x).min(points[2].x).floor() as i32;
        let min_y = points[0].y.min(points[1].y).min(points[2].y).floor() as i32;
        let max_x = points[0].x.max(points[1].x).max(points[2].x).ceil() as i32;
        let max_y = points[0].y.max(points[1].y).max(points[2].y).ceil() as i32;
        (min_x, min_y, max_x, max_y)
    }
}
