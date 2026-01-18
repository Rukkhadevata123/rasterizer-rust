use crate::core::framebuffer::FrameBuffer;
use crate::core::math::interpolation::{
    barycentric_coordinates, is_inside_triangle, perspective_correct_barycentric,
};
use crate::core::math::transform::{apply_perspective_division, ndc_to_screen};
use crate::core::pipeline::{Interpolatable, Shader};
use crate::scene::material::{AlphaMode, Material};
use nalgebra::{Point2, Vector4};
use rayon::prelude::*;

/// The Rasterizer is responsible for drawing geometric primitives onto the FrameBuffer.
pub struct Rasterizer {
    pub cull_mode: CullMode,
    pub wireframe: bool,
    pub blend_mode: BlendMode,
}

#[derive(PartialEq, Copy, Clone, Debug)]
pub enum CullMode {
    Back,
    Front,
    None,
}

#[derive(PartialEq, Copy, Clone, Debug)]
pub enum BlendMode {
    /// Opaque rendering: Writes to depth buffer, replaces color.
    Opaque,
    /// Alpha blending: Reads depth buffer (no write), blends color: src * a + dst * (1-a).
    Alpha,
}

impl Default for Rasterizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Rasterizer {
    pub fn new() -> Self {
        Self {
            cull_mode: CullMode::Back,
            wireframe: false,
            blend_mode: BlendMode::Opaque,
        }
    }

    pub fn set_cull_mode(&mut self, mode: CullMode) {
        self.cull_mode = mode;
    }

    /// Rasterize a single triangle given clip-space coordinates and corresponding varyings.
    ///
    /// This function performs **Sutherland–Hodgman clipping** against the canonical
    /// view frustum (W-normalization planes) in Homogeneous Clip Space.
    ///
    /// # Optimization
    /// Uses a double-buffering strategy for the vertex lists to minimize heap allocations
    /// during the multi-stage clipping process.
    pub fn rasterize_triangle<S: Shader>(
        &self,
        framebuffer: &FrameBuffer,
        shader: &S,
        clip_coords: &[Vector4<f32>; 3],
        varyings: &[S::Varying; 3],
        material: Option<&Material>,
    ) where
        S::Varying: Interpolatable + Copy,
    {
        // 1. Initialize buffers
        // We allocate capacity for 16 vertices, which is sufficient for almost all
        // clipped triangles (a triangle clipped by a cube can have at most 7-9 vertices).
        // Allocating here once is much faster than allocating inside the loop.
        let mut current_poly: Vec<(Vector4<f32>, S::Varying)> = Vec::with_capacity(16);
        let mut clip_buffer: Vec<(Vector4<f32>, S::Varying)> = Vec::with_capacity(16);

        // Fill initial polygon
        for i in 0..3 {
            current_poly.push((clip_coords[i], varyings[i]));
        }

        // 2. Define Clip Planes
        // Format: (Axis Index, Sign).
        // Plane Eq: Sign * P[Axis] <= P.w
        // 0=X, 1=Y, 2=Z
        let planes = [
            (0, 1.0),  // Right:  +X <= W
            (0, -1.0), // Left:   -X <= W
            (1, 1.0),  // Top:    +Y <= W
            (1, -1.0), // Bottom: -Y <= W
            (2, 1.0),  // Far:    +Z <= W
            (2, -1.0), // Near:   -Z <= W
        ];

        // 3. Perform Clipping
        for &(axis, sign) in &planes {
            // If the polygon is already fully clipped, stop early
            if current_poly.is_empty() {
                return;
            }

            // Clip current_poly against the plane, writing results to clip_buffer
            self.clip_polygon_against_plane::<S>(&current_poly, &mut clip_buffer, axis, sign);

            // Swap buffers: clip_buffer becomes the input for the next stage
            // current_poly is cleared inside the clip function, so we just swap structure content
            std::mem::swap(&mut current_poly, &mut clip_buffer);
        }

        // 4. Triangulate and Rasterize
        // The result is a convex polygon. We assume it's a fan centered at v0.
        if current_poly.len() < 3 {
            return;
        }

        let v0 = current_poly[0];
        for i in 1..(current_poly.len() - 1) {
            let v1 = current_poly[i];
            let v2 = current_poly[i + 1];

            self.rasterize_screen_triangle(
                framebuffer,
                shader,
                &[v0.0, v1.0, v2.0],
                &[v0.1, v1.1, v2.1],
                material,
            );
        }
    }

    /// Clips a polygon against a specific plane.
    ///
    /// - `input`: Source vertices.
    /// - `output`: Destination buffer (will be cleared before writing).
    /// - `axis`: 0 (X), 1 (Y), or 2 (Z).
    /// - `sign`: +1.0 or -1.0.
    fn clip_polygon_against_plane<S: Shader>(
        &self,
        input: &[(Vector4<f32>, S::Varying)],
        output: &mut Vec<(Vector4<f32>, S::Varying)>,
        axis: usize,
        sign: f32,
    ) where
        S::Varying: Interpolatable + Copy,
    {
        output.clear();

        if input.is_empty() {
            return;
        }

        let mut prev = input[input.len() - 1];
        // Point is inside if: sign * p[axis] <= p.w
        // We use a small EPS for robustness against floating point errors.
        let is_inside = |p: &Vector4<f32>| sign * p[axis] <= p.w + 1e-6;

        let mut prev_inside = is_inside(&prev.0);

        for curr in input {
            let curr_inside = is_inside(&curr.0);

            if curr_inside {
                if !prev_inside {
                    // OUT -> IN: Intersection point + Current point
                    if let Some(inter) = Self::intersect_edge_plane::<S>(prev, *curr, axis, sign) {
                        output.push(inter);
                    }
                }
                // IN -> IN: Current point
                output.push(*curr);
            } else if prev_inside {
                // IN -> OUT: Intersection point only
                if let Some(inter) = Self::intersect_edge_plane::<S>(prev, *curr, axis, sign) {
                    output.push(inter);
                }
            }
            // OUT -> OUT: Do nothing

            prev = *curr;
            prev_inside = curr_inside;
        }
    }

    /// Computes the intersection of a line segment and a clip plane.
    /// Linearly interpolates both Position and Varying attributes.
    #[inline(always)]
    fn intersect_edge_plane<S: Shader>(
        a: (Vector4<f32>, S::Varying),
        b: (Vector4<f32>, S::Varying),
        axis: usize,
        sign: f32,
    ) -> Option<(Vector4<f32>, S::Varying)>
    where
        S::Varying: Interpolatable + Copy,
    {
        // Plane equation: sign * P[axis] = P.w
        // Parameter t = (a.w - sign * a[axis]) / ((sign * b[axis] - sign * a[axis]) - (b.w - a.w))
        // Simplifies to: numerator / denominator

        let ac = a.0[axis];
        let bc = b.0[axis];
        let aw = a.0.w;
        let bw = b.0.w;

        // Denominator = sign*(bc - ac) - (bw - aw)
        // This represents the "signed distance difference" relative to the W plane.
        let denom = sign * (bc - ac) - (bw - aw);

        // Check for parallel lines or very small intersections to avoid NaN
        if denom.abs() < 1e-9 {
            return None;
        }

        let t = (aw - sign * ac) / denom;

        // Valid intersection should be within [0, 1], but we allow slight tolerance
        // for floating point inaccuracies.
        if !t.is_finite() {
            return None;
        }

        // Interpolate Position
        let pos = a.0 + (b.0 - a.0) * t;

        // Interpolate Varying
        let vary = a.1 * (1.0 - t) + b.1 * t;

        Some((pos, vary))
    }

    /// Internal function to rasterize a triangle that is guaranteed to be inside the frustum.
    /// Performs perspective division, viewport transform, and pixel shading.
    fn rasterize_screen_triangle<S: Shader>(
        &self,
        framebuffer: &FrameBuffer,
        shader: &S,
        clip_coords: &[Vector4<f32>; 3],
        varyings: &[S::Varying; 3],
        material: Option<&Material>,
    ) where
        S::Varying: Interpolatable
            + Copy
            + std::ops::Add<Output = S::Varying>
            + std::ops::Mul<f32, Output = S::Varying>,
    {
        // Use the actual buffer dimensions for rasterization
        let width = framebuffer.buffer_width as f32;
        let height = framebuffer.buffer_height as f32;

        // 1. Perspective Division & Viewport Transform
        let mut screen_coords = [Point2::origin(); 3];
        let mut w_values = [0.0; 3];

        for i in 0..3 {
            // Note: We safeguard against w near 0, though clipping should effectively prevent this.
            if clip_coords[i].w.abs() < 1e-6 {
                return;
            }

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

        match self.cull_mode {
            CullMode::Back if signed_area >= 0.0 => return,
            CullMode::Front if signed_area <= 0.0 => return,
            _ => {}
        }

        // Note: Depth must be perspective-correct interpolated per-pixel (not linearly in NDC).
        // We'll compute the corrected z_ndc inside the pixel loop using `perspective_correct_interpolate`.

        // Compute simple triangle-level UV density estimator used for mipmap LOD selection.
        // area_screen = 0.5 * |(x1-x0)*(y2-y0) - (x2-x0)*(y1-y0)|
        let area_screen =
            0.5 * ((v1.x - v0.x) * (v2.y - v0.y) - (v2.x - v0.x) * (v1.y - v0.y)).abs();
        let uv_density = if area_screen > 1e-6 {
            // Ask the varying (via trait) whether it exposes UVs and compute
            // triangle-level UV density: sqrt(Area_uv / Area_screen).
            if let (Some(uv0), Some(uv1), Some(uv2)) = (
                varyings[0].get_uv(),
                varyings[1].get_uv(),
                varyings[2].get_uv(),
            ) {
                // area_uv = 0.5 * |(u1-u0)*(v2-v0) - (u2-u0)*(v1-v0)|
                let area_uv = 0.5
                    * ((uv1.x - uv0.x) * (uv2.y - uv0.y) - (uv2.x - uv0.x) * (uv1.y - uv0.y)).abs();
                (area_uv / area_screen).sqrt()
            } else {
                0.0
            }
        } else {
            0.0
        };

        // 3. Compute Bounding Box
        let (min_x, min_y, max_x, max_y) = self.compute_bounding_box(&screen_coords);

        // Scissor Test
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

        // Retrieve Alpha Mode parameters for pixel discard logic
        let (alpha_mode, alpha_cutoff) = if let Some(Material::Pbr(mat)) = material {
            match mat.alpha_mode {
                AlphaMode::Mask(cutoff) => (AlphaMode::Mask(cutoff), cutoff),
                mode => (mode, 0.5),
            }
        } else {
            (AlphaMode::Opaque, 0.5)
        };

        // 4. Pixel Loop
        // Rayon parallel iterator: work-stealing is effective here as row workloads vary.
        (start_y..=end_y).into_par_iter().for_each(|y| {
            for x in start_x..=end_x {
                let pixel_center = Point2::new(x as f32 + 0.5, y as f32 + 0.5);

                if let Some(bary) = barycentric_coordinates(
                    pixel_center,
                    screen_coords[0],
                    screen_coords[1],
                    screen_coords[2],
                ) {
                    if !is_inside_triangle(bary) {
                        continue;
                    }

                    // --- Wireframe Mode ---
                    if self.wireframe {
                        // Simple wireframe threshold: discard pixels not near the edges
                        let threshold = 0.02;
                        if bary.x > threshold && bary.y > threshold && bary.z > threshold {
                            continue;
                        }
                    }

                    // Perspective Correct Interpolation for Z
                    // Z in NDC [-1, 1], we map to [0, 1] for depth buffer
                    // Note: In standard OpenGL, gl_FragCoord.z is (z_ndc + 1) / 2
                    // We use 1/w interpolation for correctness.
                    let z_ndc = bary.x * clip_coords[0].z / clip_coords[0].w
                        + bary.y * clip_coords[1].z / clip_coords[1].w
                        + bary.z * clip_coords[2].z / clip_coords[2].w;

                    // Depth value to be stored [0.0, 1.0]
                    // If w < 0 (behind camera), we shouldn't act, but strict clipping handles this.
                    // Simple Z-buffer approximation:
                    let depth = z_ndc * 0.5 + 0.5;

                    // Early Depth Test
                    // For Alpha blending (read-only depth), we must check before shading to save performance.
                    if self.blend_mode == BlendMode::Alpha && !framebuffer.test_depth(x, y, depth) {
                        continue;
                    }

                    // Interpolate Varyings
                    let varying_interp = if let Some(v) =
                        perspective_correct_barycentric(bary, w_values[0], w_values[1], w_values[2])
                    {
                        varyings[0] * v.x + varyings[1] * v.y + varyings[2] * v.z
                    } else {
                        // Numeric instability fallback
                        varyings[0] * bary.x + varyings[1] * bary.y + varyings[2] * bary.z
                    };

                    // Execute Fraction Shader
                    let color = shader.fragment(varying_interp, material, uv_density);

                    // --- Alpha Test (Discard) ---
                    if let AlphaMode::Mask(_) = alpha_mode
                        && color.w < alpha_cutoff
                    {
                        continue; // Discard pixel
                    }

                    // --- Output Merger ---
                    match self.blend_mode {
                        BlendMode::Opaque => {
                            // Opaque Pass: Read-Modify-Write Depth
                            if framebuffer.test_and_update_depth(x, y, depth) {
                                framebuffer.set_pixel_safe(x, y, color.xyz());
                            }
                        }
                        BlendMode::Alpha => {
                            // Transparent Pass: Read Depth, Blend Color.
                            // Note: We already did an early depth test, but checking again is strictly
                            // more correct if the depth buffer is being updated concurrently (though unlikely in sorted pass).
                            // We'll skip the second check for performance as per original implementation logic.
                            if color.w > 0.001 {
                                framebuffer.blend_pixel_safe(x, y, color.xyz(), color.w);
                            }
                        }
                    }
                }
            }
        });
    }

    fn compute_bounding_box(&self, points: &[Point2<f32>; 3]) -> (i32, i32, i32, i32) {
        let min_x = points[0].x.min(points[1].x).min(points[2].x).floor() as i32;
        let min_y = points[0].y.min(points[1].y).min(points[2].y).floor() as i32;
        let max_x = points[0].x.max(points[1].x).max(points[2].x).ceil() as i32;
        let max_y = points[0].y.max(points[1].y).max(points[2].y).ceil() as i32;
        (min_x, min_y, max_x, max_y)
    }
}
