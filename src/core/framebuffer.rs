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
    pub sample_count: usize,
    pub buffer_width: usize,
    pub buffer_height: usize,
    samples: Vec<Sample>,
}

impl FrameBuffer {
    pub fn new(width: usize, height: usize, sample_count: usize) -> Self {
        let buffer_width = width * sample_count;
        let buffer_height = height * sample_count;
        let size = buffer_width * buffer_height;

        Self {
            width,
            height,
            sample_count,
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

        if self.sample_count == 1 {
            return Some(self.samples[self.index(x, y)].color);
        }

        let mut sum_color = Vector3::zeros();
        let start_x = x * self.sample_count;
        let start_y = y * self.sample_count;

        for dy in 0..self.sample_count {
            for dx in 0..self.sample_count {
                sum_color += self.samples[self.index(start_x + dx, start_y + dy)].color;
            }
        }

        let samples = (self.sample_count * self.sample_count) as f32;
        Some(sum_color / samples)
    }
}
