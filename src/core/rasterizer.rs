use crate::core::framebuffer::{FrameBuffer, Sample};
use crate::core::geometry::SUPPORTED_TEXCOORD_SETS;
use crate::core::math::interpolation::{barycentric_coordinates, perspective_correct_barycentric};
use crate::core::math::transform::{apply_perspective_division, ndc_to_screen};
use crate::core::pipeline::{FragmentInput, FragmentOutput, Interpolatable, Shader};
use nalgebra::{Point2, Vector4};
use rayon::prelude::*;
use std::ops::RangeInclusive;

const RASTER_BAND_HEIGHT: usize = 8;
const WIREFRAME_HALF_WIDTH: f32 = 1.0;
const MAX_CLIPPED_VERTICES: usize = 9;
pub(crate) const MAX_PREPARED_TRIANGLES: usize = MAX_CLIPPED_VERTICES - 2;

pub struct Rasterizer {
    band_bins: Vec<Vec<usize>>,
}

struct ClippedPolygon<V: Copy> {
    vertices: [Option<(Vector4<f32>, V)>; MAX_CLIPPED_VERTICES],
    len: usize,
}

impl<V: Copy> ClippedPolygon<V> {
    fn new() -> Self {
        Self {
            vertices: [None; MAX_CLIPPED_VERTICES],
            len: 0,
        }
    }

    fn clear(&mut self) {
        self.len = 0;
    }

    fn push(&mut self, vertex: (Vector4<f32>, V)) {
        debug_assert!(self.len < MAX_CLIPPED_VERTICES);
        self.vertices[self.len] = Some(vertex);
        self.len += 1;
    }

    fn get(&self, index: usize) -> (Vector4<f32>, V) {
        self.vertices[index].expect("clipped vertex index is in bounds")
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }
}

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
    pub front_face_inverted: bool,
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
            front_face_inverted: false,
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
    edge_is_top_left: [bool; 3],
    edge_inverse_lengths: [f32; 3],
    orientation: f32,
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
        Self {
            band_bins: Vec::new(),
        }
    }

    pub(crate) fn prepare_triangle<'a, S, C>(
        &self,
        framebuffer_size: (usize, usize),
        clip_coords: &[Vector4<f32>; 3],
        varyings: &[S::Varying; 3],
        shader: &'a S,
        state: RenderState,
        fragment_context: C,
    ) -> [Option<PreparedTriangle<'a, S::Varying, S, C>>; MAX_PREPARED_TRIANGLES]
    where
        S: Shader<C>,
        S::Varying: Interpolatable + Copy,
        C: Copy + Send + Sync,
    {
        if clip_coords
            .iter()
            .any(|position| !position.iter().all(|component| component.is_finite()))
        {
            return std::array::from_fn(|_| None);
        }

        let mut current_poly = ClippedPolygon::new();
        let mut clip_buffer = ClippedPolygon::new();

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
                return std::array::from_fn(|_| None);
            }

            Self::clip_polygon_against_plane(&current_poly, &mut clip_buffer, axis, sign);
            std::mem::swap(&mut current_poly, &mut clip_buffer);
        }

        if current_poly.len < 3 {
            return std::array::from_fn(|_| None);
        }

        let first = current_poly.get(0);
        let mut prepared = std::array::from_fn(|_| None);
        let mut prepared_len = 0;
        for index in 1..current_poly.len - 1 {
            let second = current_poly.get(index);
            let third = current_poly.get(index + 1);
            if let Some(triangle) = self.prepare_screen_triangle(
                framebuffer_size,
                &[first.0, second.0, third.0],
                &[first.1, second.1, third.1],
                shader,
                state,
                fragment_context,
            ) {
                prepared[prepared_len] = Some(triangle);
                prepared_len += 1;
            }
        }
        prepared
    }

    pub(crate) fn rasterize_prepared<S, C>(
        &mut self,
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
        if self.band_bins.len() < band_count {
            self.band_bins.resize_with(band_count, Vec::new);
        }
        for band in &mut self.band_bins[..band_count] {
            band.clear();
        }

        for (triangle_index, triangle) in triangles.iter().enumerate() {
            let first_band = triangle.start_y / RASTER_BAND_HEIGHT;
            let last_band = triangle.end_y / RASTER_BAND_HEIGHT;
            for band in &mut self.band_bins[first_band..=last_band] {
                band.push(triangle_index);
            }
        }

        let band_bins = &self.band_bins[..band_count];

        framebuffer
            .samples_mut()
            .par_chunks_mut(width * RASTER_BAND_HEIGHT)
            .enumerate()
            .for_each(|(band_index, samples)| {
                let band_start_y = band_index * RASTER_BAND_HEIGHT;
                let band_end_y = (band_start_y + samples.len() / width - 1).min(height - 1);

                for &triangle_index in &band_bins[band_index] {
                    Self::rasterize_triangle_band(
                        samples,
                        width,
                        band_start_y..=band_end_y,
                        &triangles[triangle_index],
                    );
                }
            });
    }

    fn clip_polygon_against_plane<V>(
        input: &ClippedPolygon<V>,
        output: &mut ClippedPolygon<V>,
        axis: usize,
        sign: f32,
    ) where
        V: Interpolatable + Copy,
    {
        output.clear();
        if input.is_empty() {
            return;
        }

        let is_inside = |position: &Vector4<f32>| sign * position[axis] <= position.w + 1e-6;
        let mut previous = input.get(input.len - 1);
        let mut previous_inside = is_inside(&previous.0);

        for index in 0..input.len {
            let current = input.get(index);
            let current_inside = is_inside(&current.0);

            if current_inside {
                if !previous_inside
                    && let Some(intersection) =
                        Self::intersect_edge_plane(previous, current, axis, sign)
                {
                    output.push(intersection);
                }
                output.push(current);
            } else if previous_inside
                && let Some(intersection) =
                    Self::intersect_edge_plane(previous, current, axis, sign)
            {
                output.push(intersection);
            }

            previous = current;
            previous_inside = current_inside;
        }
    }

    #[inline(always)]
    fn intersect_edge_plane<V>(
        a: (Vector4<f32>, V),
        b: (Vector4<f32>, V),
        axis: usize,
        sign: f32,
    ) -> Option<(Vector4<f32>, V)>
    where
        V: Interpolatable + Copy,
    {
        let denominator = sign * (b.0[axis] - a.0[axis]) - (b.0.w - a.0.w);
        if denominator.abs() < 1e-9 {
            return None;
        }

        let t = (a.0.w - sign * a.0[axis]) / denominator;
        if !t.is_finite() {
            return None;
        }

        let position = a.0 + (b.0 - a.0) * t;
        if !position.iter().all(|component| component.is_finite()) {
            return None;
        }

        Some((position, a.1 * (1.0 - t) + b.1 * t))
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
            if !clip_coords[index]
                .iter()
                .all(|component| component.is_finite())
                || clip_coords[index].w.abs() < 1e-6
            {
                return None;
            }

            let ndc = apply_perspective_division(&clip_coords[index]);
            if !ndc.coords.iter().all(|component| component.is_finite()) {
                return None;
            }

            w_values[index] = clip_coords[index].w;
            clip_z[index] = clip_coords[index].z;
            screen_coords[index] = ndc_to_screen(ndc.x, ndc.y, width, height);
            if !screen_coords[index]
                .coords
                .iter()
                .all(|component| component.is_finite())
            {
                return None;
            }
        }

        let edge1 = screen_coords[1] - screen_coords[0];
        let edge2 = screen_coords[2] - screen_coords[1];
        let signed_area = edge1.x * edge2.y - edge1.y * edge2.x;
        if !signed_area.is_finite() || signed_area.abs() < 1e-6 {
            return None;
        }

        let orientation = signed_area.signum();
        let edges = [
            (screen_coords[1], screen_coords[2]),
            (screen_coords[2], screen_coords[0]),
            (screen_coords[0], screen_coords[1]),
        ];
        let edge_is_top_left = edges.map(|(start, end)| {
            let (start, end) = if orientation > 0.0 {
                (start, end)
            } else {
                (end, start)
            };
            Self::is_top_left_edge(start, end)
        });
        let edge_inverse_lengths = edges.map(|(start, end)| 1.0 / (end - start).norm());
        if edge_inverse_lengths
            .iter()
            .any(|inverse_length| !inverse_length.is_finite())
        {
            return None;
        }

        let front_facing = (signed_area < 0.0) != state.front_face_inverted;

        match state.cull_mode {
            CullMode::Back if !front_facing => return None,
            CullMode::Front if front_facing => return None,
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
            front_facing,
            uv_densities,
            edge_is_top_left,
            edge_inverse_lengths,
            orientation,
            start_x: min_x.max(0) as usize,
            end_x: max_x.min(framebuffer_width as i32 - 1) as usize,
            start_y: min_y.max(0) as usize,
            end_y: max_y.min(framebuffer_height as i32 - 1) as usize,
        })
    }

    #[inline(always)]
    fn edge_function(start: Point2<f32>, end: Point2<f32>, point: Point2<f32>) -> f32 {
        let edge = end - start;
        let offset = point - start;
        edge.x * offset.y - edge.y * offset.x
    }

    #[inline(always)]
    fn is_top_left_edge(start: Point2<f32>, end: Point2<f32>) -> bool {
        let edge = end - start;
        edge.y < 0.0 || (edge.y == 0.0 && edge.x > 0.0)
    }

    fn rasterize_triangle_band<S, C>(
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
                let edge_values = [
                    Self::edge_function(
                        triangle.screen_coords[1],
                        triangle.screen_coords[2],
                        pixel_center,
                    ),
                    Self::edge_function(
                        triangle.screen_coords[2],
                        triangle.screen_coords[0],
                        pixel_center,
                    ),
                    Self::edge_function(
                        triangle.screen_coords[0],
                        triangle.screen_coords[1],
                        pixel_center,
                    ),
                ];
                let covered = edge_values.iter().enumerate().all(|(index, value)| {
                    let oriented_value = value * triangle.orientation;
                    oriented_value > 0.0
                        || (oriented_value == 0.0 && triangle.edge_is_top_left[index])
                });
                if !covered {
                    continue;
                }

                let Some(barycentric) = barycentric_coordinates(
                    pixel_center,
                    triangle.screen_coords[0],
                    triangle.screen_coords[1],
                    triangle.screen_coords[2],
                ) else {
                    continue;
                };

                if triangle.state.wireframe
                    && edge_values.iter().zip(triangle.edge_inverse_lengths).all(
                        |(edge, inverse_length)| edge.abs() * inverse_length > WIREFRAME_HALF_WIDTH,
                    )
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
