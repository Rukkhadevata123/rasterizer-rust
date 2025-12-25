use crate::core::framebuffer::FrameBuffer;
use crate::core::math::interpolation::{
    barycentric_coordinates, is_inside_triangle, perspective_correct_interpolate,
};
use crate::core::math::transform::{apply_perspective_division, ndc_to_screen};
use crate::core::pipeline::Shader;
use nalgebra::{Point2, Vector4};

/// The Rasterizer is responsible for drawing geometric primitives onto the FrameBuffer.
/// It acts as the fixed-function stage of the pipeline, invoking programmable Shaders.
pub struct Rasterizer {
    pub width: usize,
    pub height: usize,
}

impl Rasterizer {
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }

    /// Rasterizes a single triangle.
    ///
    /// # Arguments
    /// * `framebuffer` - The target buffer to write pixels to.
    /// * `shader` - The shader program defining vertex and fragment logic.
    /// * `clip_coords` - The 3 vertices in Homogeneous Clip Space (output from Vertex Shader).
    /// * `varyings` - The 3 sets of data to be interpolated (output from Vertex Shader).
    pub fn rasterize_triangle<S: Shader>(
        &self,
        framebuffer: &mut FrameBuffer,
        shader: &S,
        clip_coords: &[Vector4<f32>; 3],
        varyings: &[S::Varying; 3],
    ) {
        // 1. Perspective Division & Viewport Transform
        // Convert Clip Space -> NDC -> Screen Space
        let mut screen_coords = [Point2::origin(); 3];
        let mut w_values = [0.0; 3]; // We need 'w' (depth in view space) for perspective correction

        for i in 0..3 {
            let ndc = apply_perspective_division(&clip_coords[i]);
            // Store 1/w or w depending on convention. Here we store w from clip space (which is usually -z_view).
            // For interpolation, we usually need 1/w.
            w_values[i] = clip_coords[i].w;

            screen_coords[i] = ndc_to_screen(ndc.x, ndc.y, self.width as f32, self.height as f32);
        }

        // Precompute NDC z values for depth interpolation
        let z0_ndc = clip_coords[0].z / clip_coords[0].w;
        let z1_ndc = clip_coords[1].z / clip_coords[1].w;
        let z2_ndc = clip_coords[2].z / clip_coords[2].w;

        // 2. Backface Culling (Optional, simple 2D cross product check)
        // If the triangle area is negative (or positive depending on winding order), skip it.
        // For now, we skip this to ensure we see something (double-sided).
        // TODO: Implement backface culling if needed.

        // 3. Compute Bounding Box
        let (min_x, min_y, max_x, max_y) = self.compute_bounding_box(&screen_coords);

        // Clip against screen bounds
        if max_x < 0 || max_y < 0 || min_x >= self.width as i32 || min_y >= self.height as i32 {
            return;
        }

        let start_x = min_x.max(0) as usize;
        let end_x = (max_x.min(self.width as i32 - 1)) as usize;
        let start_y = min_y.max(0) as usize;
        let end_y = (max_y.min(self.height as i32 - 1)) as usize;

        // 4. Pixel Loop
        for y in start_y..=end_y {
            for x in start_x..=end_x {
                // TODO: Anti-Aliasing Support
                // Instead of sampling just the center (x+0.5, y+0.5),
                // sample multiple sub-pixels (e.g., 4 samples) and average the results.
                let pixel_center = Point2::new(x as f32 + 0.5, y as f32 + 0.5);

                // a. Calculate Barycentric Coordinates
                if let Some(bary) = barycentric_coordinates(
                    pixel_center,
                    screen_coords[0],
                    screen_coords[1],
                    screen_coords[2],
                ) {
                    // b. Check if pixel is inside triangle
                    if is_inside_triangle(bary) {
                        // c. Interpolate Depth (Z-Buffering)
                        // Calculate NDC Z (z/w) for each vertex.
                        // TODO: Optimize

                        // Linearly interpolate NDC depth.
                        // Mathematically, while View-Space Z is NOT linear in screen space,
                        // NDC Z (which is roughly A + B/Z_view) IS linear in screen space.
                        // So linear barycentric interpolation is the correct way to calculate depth for the Z-buffer.
                        let z_ndc = z0_ndc * bary.x + z1_ndc * bary.y + z2_ndc * bary.z;

                        // Map NDC z [-1, 1] to [0, 1] for depth buffer
                        let depth = z_ndc * 0.5 + 0.5;

                        // d. Depth Test
                        if framebuffer.depth_test(x, y, depth) {
                            // e. Perspective Correct Interpolation of Varyings
                            let interpolated_varying = perspective_correct_interpolate(
                                bary,
                                varyings[0],
                                varyings[1],
                                varyings[2],
                                w_values[0],
                                w_values[1],
                                w_values[2],
                            );

                            // f. Fragment Shader
                            let color = shader.fragment(interpolated_varying);

                            // g. Write to FrameBuffer
                            framebuffer.set_pixel(x, y, color);
                            framebuffer.set_depth(x, y, depth);
                        }
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
