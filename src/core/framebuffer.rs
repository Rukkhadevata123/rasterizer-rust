use nalgebra::Vector3;
use rayon::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub color: Vector3<f32>,
    pub depth: f32,
}

impl Sample {
    fn cleared(color: Vector3<f32>, depth: f32) -> Self {
        Self { color, depth }
    }
}

pub struct FrameBuffer {
    pub width: usize,
    pub height: usize,
    pub supersample_scale: usize,
    pub buffer_width: usize,
    pub buffer_height: usize,
    samples: Vec<Sample>,
}

impl FrameBuffer {
    pub fn new(width: usize, height: usize, supersample_scale: usize) -> Self {
        let buffer_width = width * supersample_scale;
        let buffer_height = height * supersample_scale;
        let size = buffer_width * buffer_height;

        Self {
            width,
            height,
            supersample_scale,
            buffer_width,
            buffer_height,
            samples: vec![Sample::cleared(Vector3::zeros(), f32::INFINITY); size],
        }
    }

    #[inline(always)]
    pub fn in_bounds(&self, x: usize, y: usize) -> bool {
        x < self.buffer_width && y < self.buffer_height
    }

    #[inline(always)]
    fn index(&self, x: usize, y: usize) -> usize {
        y * self.buffer_width + x
    }

    pub fn sample(&self, x: usize, y: usize) -> Option<&Sample> {
        self.in_bounds(x, y)
            .then(|| &self.samples[self.index(x, y)])
    }

    pub(crate) fn samples_mut(&mut self) -> &mut [Sample] {
        &mut self.samples
    }

    pub fn depth_values(&self) -> Vec<f32> {
        self.samples.iter().map(|sample| sample.depth).collect()
    }

    pub fn clear_with<F>(&mut self, depth: f32, color_at: F)
    where
        F: Fn(usize, usize) -> Vector3<f32> + Sync,
    {
        let width = self.buffer_width;
        self.samples
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(y, row)| {
                for (x, sample) in row.iter_mut().enumerate() {
                    *sample = Sample::cleared(color_at(x, y), depth);
                }
            });
    }

    pub fn get_pixel(&self, x: usize, y: usize) -> Option<Vector3<f32>> {
        if x >= self.width || y >= self.height {
            return None;
        }

        if self.supersample_scale == 1 {
            return Some(self.samples[self.index(x, y)].color);
        }

        let mut sum_color = Vector3::zeros();
        let start_x = x * self.supersample_scale;
        let start_y = y * self.supersample_scale;

        for dy in 0..self.supersample_scale {
            for dx in 0..self.supersample_scale {
                sum_color += self.samples[self.index(start_x + dx, start_y + dy)].color;
            }
        }

        let sample_total = (self.supersample_scale * self.supersample_scale) as f32;
        Some(sum_color / sample_total)
    }
}
