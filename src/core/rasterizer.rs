use crate::core::framebuffer::{FrameBuffer, Sample};
use crate::core::geometry::SUPPORTED_TEXCOORD_SETS;
use crate::core::math::interpolation::{
    barycentric_coordinates, is_inside_triangle, perspective_correct_barycentric,
};
use crate::core::math::transform::{apply_perspective_division, ndc_to_screen};
use crate::core::pipeline::{FragmentInput, FragmentOutput, Interpolatable, Shader};
use nalgebra::{Point2, Vector4};
use rayon::prelude::*;
use std::ops::RangeInclusive;

const RASTER_BAND_HEIGHT: usize = 16;

pub struct Rasterizer;

#[derive(PartialEq, Copy, Clone, Debug)]
pub enum CullMode {
    Back,
    Front,
    None,
}

#[derive(PartialEq, Copy, Clone, Debug)]
pub enum BlendMode {
    Opaque,
    Alpha,
}

#[derive(PartialEq, Copy, Clone, Debug)]
pub enum DepthCompare {
    Never,
    Less,
    LessEqual,
    Equal,
    NotEqual,
    GreaterEqual,
    Greater,
    Always,
}

impl DepthCompare {
    fn test(self, incoming: f32, stored: f32) -> bool {
        match self {
            Self::Never => false,
            Self::Less => incoming < stored,
            Self::LessEqual => incoming <= stored,
            Self::Equal => incoming == stored,
            Self::NotEqual => incoming != stored,
            Self::GreaterEqual => incoming >= stored,
            Self::Greater => incoming > stored,
            Self::Always => true,
        }
    }
}

#[derive(PartialEq, Copy, Clone, Debug)]
pub struct RenderState {
    pub cull_mode: CullMode,
    pub depth_test: bool,
    pub depth_compare: DepthCompare,
    pub depth_write: bool,
    pub blend_mode: BlendMode,
    pub wireframe: bool,
}

impl Default for RenderState {
    fn default() -> Self {
        Self {
            cull_mode: CullMode::Back,
            depth_test: true,
            depth_compare: DepthCompare::Less,
            depth_write: true,
            blend_mode: BlendMode::Opaque,
            wireframe: false,
        }
    }
}

pub(crate) struct PreparedTriangle<'a, V, S, C> {
    screen_coords: [Point2<f32>; 3],
    clip_z: [f32; 3],
    w_values: [f32; 3],
    varyings: [V; 3],
    shader: &'a S,
    state: RenderState,
    fragment_context: C,
    front_facing: bool,
    uv_densities: [f32; SUPPORTED_TEXCOORD_SETS],
    start_x: usize,
    end_x: usize,
    start_y: usize,
    end_y: usize,
}

impl Default for Rasterizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Rasterizer {
    pub fn new() -> Self {
        Self
    }

    pub(crate) fn prepare_triangle<'a, S, C>(
        &self,
        framebuffer_size: (usize, usize),
        clip_coords: &[Vector4<f32>; 3],
        varyings: &[S::Varying; 3],
        shader: &'a S,
        state: RenderState,
        fragment_context: C,
    ) -> Vec<PreparedTriangle<'a, S::Varying, S, C>>
    where
        S: Shader<C>,
        S::Varying: Interpolatable + Copy,
        C: Copy + Send + Sync,
    {
        let mut current_poly = Vec::with_capacity(16);
        let mut clip_buffer = Vec::with_capacity(16);

        for index in 0..3 {
            current_poly.push((clip_coords[index], varyings[index]));
        }

        let planes = [
            (0, 1.0),
            (0, -1.0),
            (1, 1.0),
            (1, -1.0),
            (2, 1.0),
            (2, -1.0),
        ];

        for &(axis, sign) in &planes {
            if current_poly.is_empty() {
                return Vec::new();
            }

            Self::clip_polygon_against_plane::<S, C>(&current_poly, &mut clip_buffer, axis, sign);
            std::mem::swap(&mut current_poly, &mut clip_buffer);
        }

        if current_poly.len() < 3 {
            return Vec::new();
        }

        let first = current_poly[0];
        (1..current_poly.len() - 1)
            .filter_map(|index| {
                let second = current_poly[index];
                let third = current_poly[index + 1];
                self.prepare_screen_triangle(
                    framebuffer_size,
                    &[first.0, second.0, third.0],
                    &[first.1, second.1, third.1],
                    shader,
                    state,
                    fragment_context,
                )
            })
            .collect()
    }

    pub(crate) fn rasterize_prepared<S, C>(
        &self,
        framebuffer: &mut FrameBuffer,
        triangles: &[PreparedTriangle<'_, S::Varying, S, C>],
    ) where
        S: Shader<C>,
        S::Varying: Interpolatable + Copy,
        C: Copy + Send + Sync,
    {
        if triangles.is_empty() {
            return;
        }

        let width = framebuffer.buffer_width;
        let height = framebuffer.buffer_height;
        let band_count = height.div_ceil(RASTER_BAND_HEIGHT);
        let mut band_bins = vec![Vec::new(); band_count];

        for (triangle_index, triangle) in triangles.iter().enumerate() {
            let first_band = triangle.start_y / RASTER_BAND_HEIGHT;
            let last_band = triangle.end_y / RASTER_BAND_HEIGHT;
            for band in &mut band_bins[first_band..=last_band] {
                band.push(triangle_index);
            }
        }

        framebuffer
            .samples_mut()
            .par_chunks_mut(width * RASTER_BAND_HEIGHT)
            .enumerate()
            .for_each(|(band_index, samples)| {
                let band_start_y = band_index * RASTER_BAND_HEIGHT;
                let band_end_y = (band_start_y + samples.len() / width - 1).min(height - 1);

                for &triangle_index in &band_bins[band_index] {
                    self.rasterize_triangle_band(
                        samples,
                        width,
                        band_start_y..=band_end_y,
                        &triangles[triangle_index],
                    );
                }
            });
    }

    fn clip_polygon_against_plane<S, C>(
        input: &[(Vector4<f32>, S::Varying)],
        output: &mut Vec<(Vector4<f32>, S::Varying)>,
        axis: usize,
        sign: f32,
    ) where
        S: Shader<C>,
        S::Varying: Interpolatable + Copy,
        C: Copy + Send + Sync,
    {
        output.clear();
        if input.is_empty() {
            return;
        }

        let is_inside = |position: &Vector4<f32>| sign * position[axis] <= position.w + 1e-6;
        let mut previous = input[input.len() - 1];
        let mut previous_inside = is_inside(&previous.0);

        for current in input {
            let current_inside = is_inside(&current.0);

            if current_inside {
                if !previous_inside
                    && let Some(intersection) =
                        Self::intersect_edge_plane::<S, C>(previous, *current, axis, sign)
                {
                    output.push(intersection);
                }
                output.push(*current);
            } else if previous_inside
                && let Some(intersection) =
                    Self::intersect_edge_plane::<S, C>(previous, *current, axis, sign)
            {
                output.push(intersection);
            }

            previous = *current;
            previous_inside = current_inside;
        }
    }

    #[inline(always)]
    fn intersect_edge_plane<S, C>(
        a: (Vector4<f32>, S::Varying),
        b: (Vector4<f32>, S::Varying),
        axis: usize,
        sign: f32,
    ) -> Option<(Vector4<f32>, S::Varying)>
    where
        S: Shader<C>,
        S::Varying: Interpolatable + Copy,
        C: Copy + Send + Sync,
    {
        let denominator = sign * (b.0[axis] - a.0[axis]) - (b.0.w - a.0.w);
        if denominator.abs() < 1e-9 {
            return None;
        }

        let t = (a.0.w - sign * a.0[axis]) / denominator;
        if !t.is_finite() {
            return None;
        }

        Some((a.0 + (b.0 - a.0) * t, a.1 * (1.0 - t) + b.1 * t))
    }

    fn prepare_screen_triangle<'a, V, S, C>(
        &self,
        framebuffer_size: (usize, usize),
        clip_coords: &[Vector4<f32>; 3],
        varyings: &[V; 3],
        shader: &'a S,
        state: RenderState,
        fragment_context: C,
    ) -> Option<PreparedTriangle<'a, V, S, C>>
    where
        V: Interpolatable + Copy,
        C: Copy,
    {
        let (framebuffer_width, framebuffer_height) = framebuffer_size;
        let width = framebuffer_width as f32;
        let height = framebuffer_height as f32;
        let mut screen_coords = [Point2::origin(); 3];
        let mut w_values = [0.0; 3];
        let mut clip_z = [0.0; 3];

        for index in 0..3 {
            if clip_coords[index].w.abs() < 1e-6 {
                return None;
            }

            let ndc = apply_perspective_division(&clip_coords[index]);
            if !ndc.coords.iter().all(|component| component.is_finite()) {
                return None;
            }

            w_values[index] = clip_coords[index].w;
            clip_z[index] = clip_coords[index].z;
            screen_coords[index] = ndc_to_screen(ndc.x, ndc.y, width, height);
        }

        let edge1 = screen_coords[1] - screen_coords[0];
        let edge2 = screen_coords[2] - screen_coords[1];
        let signed_area = edge1.x * edge2.y - edge1.y * edge2.x;
        if signed_area.abs() < 1e-6 {
            return None;
        }

        match state.cull_mode {
            CullMode::Back if signed_area >= 0.0 => return None,
            CullMode::Front if signed_area <= 0.0 => return None,
            _ => {}
        }

        let area_screen = signed_area.abs() * 0.5;
        let uv_densities = std::array::from_fn(|set| {
            match (
                varyings[0].get_uv(set),
                varyings[1].get_uv(set),
                varyings[2].get_uv(set),
            ) {
                (Some(uv0), Some(uv1), Some(uv2)) => {
                    let area_uv = 0.5
                        * ((uv1.x - uv0.x) * (uv2.y - uv0.y) - (uv2.x - uv0.x) * (uv1.y - uv0.y))
                            .abs();
                    (area_uv / area_screen).sqrt()
                }
                _ => 0.0,
            }
        });

        let min_x = screen_coords[0]
            .x
            .min(screen_coords[1].x)
            .min(screen_coords[2].x)
            .floor() as i32;
        let min_y = screen_coords[0]
            .y
            .min(screen_coords[1].y)
            .min(screen_coords[2].y)
            .floor() as i32;
        let max_x = screen_coords[0]
            .x
            .max(screen_coords[1].x)
            .max(screen_coords[2].x)
            .ceil() as i32;
        let max_y = screen_coords[0]
            .y
            .max(screen_coords[1].y)
            .max(screen_coords[2].y)
            .ceil() as i32;

        if max_x < 0
            || max_y < 0
            || min_x >= framebuffer_width as i32
            || min_y >= framebuffer_height as i32
        {
            return None;
        }

        Some(PreparedTriangle {
            screen_coords,
            clip_z,
            w_values,
            varyings: *varyings,
            shader,
            state,
            fragment_context,
            front_facing: signed_area < 0.0,
            uv_densities,
            start_x: min_x.max(0) as usize,
            end_x: max_x.min(framebuffer_width as i32 - 1) as usize,
            start_y: min_y.max(0) as usize,
            end_y: max_y.min(framebuffer_height as i32 - 1) as usize,
        })
    }

    fn rasterize_triangle_band<S, C>(
        &self,
        samples: &mut [Sample],
        framebuffer_width: usize,
        band_rows: RangeInclusive<usize>,
        triangle: &PreparedTriangle<'_, S::Varying, S, C>,
    ) where
        S: Shader<C>,
        S::Varying: Interpolatable + Copy,
        C: Copy + Send + Sync,
    {
        let band_start_y = *band_rows.start();
        let band_end_y = *band_rows.end();
        let start_y = triangle.start_y.max(band_start_y);
        let end_y = triangle.end_y.min(band_end_y);
        if start_y > end_y {
            return;
        }

        for y in start_y..=end_y {
            let row_offset = (y - band_start_y) * framebuffer_width;
            for x in triangle.start_x..=triangle.end_x {
                let pixel_center = Point2::new(x as f32 + 0.5, y as f32 + 0.5);
                let Some(barycentric) = barycentric_coordinates(
                    pixel_center,
                    triangle.screen_coords[0],
                    triangle.screen_coords[1],
                    triangle.screen_coords[2],
                ) else {
                    continue;
                };

                if !is_inside_triangle(barycentric) {
                    continue;
                }

                if triangle.state.wireframe
                    && barycentric.x > 0.02
                    && barycentric.y > 0.02
                    && barycentric.z > 0.02
                {
                    continue;
                }

                let z_ndc = barycentric.x * triangle.clip_z[0] / triangle.w_values[0]
                    + barycentric.y * triangle.clip_z[1] / triangle.w_values[1]
                    + barycentric.z * triangle.clip_z[2] / triangle.w_values[2];
                let depth = z_ndc * 0.5 + 0.5;
                if !depth.is_finite() {
                    continue;
                }

                let sample = &mut samples[row_offset + x];
                if triangle.state.depth_test
                    && !triangle.state.depth_compare.test(depth, sample.depth)
                {
                    continue;
                }

                let varying = perspective_correct_barycentric(
                    barycentric,
                    triangle.w_values[0],
                    triangle.w_values[1],
                    triangle.w_values[2],
                )
                .map(|weights| {
                    triangle.varyings[0] * weights.x
                        + triangle.varyings[1] * weights.y
                        + triangle.varyings[2] * weights.z
                })
                .unwrap_or_else(|| {
                    triangle.varyings[0] * barycentric.x
                        + triangle.varyings[1] * barycentric.y
                        + triangle.varyings[2] * barycentric.z
                });

                let input = FragmentInput {
                    varying,
                    front_facing: triangle.front_facing,
                    uv_densities: triangle.uv_densities,
                };
                let FragmentOutput::Color(color) =
                    triangle.shader.fragment(input, triangle.fragment_context)
                else {
                    continue;
                };

                match triangle.state.blend_mode {
                    BlendMode::Opaque => {
                        if triangle.state.depth_write {
                            sample.depth = depth;
                        }
                        sample.color = color.xyz();
                    }
                    BlendMode::Alpha if color.w > 0.001 => {
                        if triangle.state.depth_write {
                            sample.depth = depth;
                        }
                        sample.color = color.xyz() * color.w + sample.color * (1.0 - color.w);
                    }
                    BlendMode::Alpha => {}
                }
            }
        }
    }
}
