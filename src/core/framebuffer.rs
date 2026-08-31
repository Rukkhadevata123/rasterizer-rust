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
    pub fn new(width: usize, height: usize, supersample_scale: usize) -> Result<Self, String> {
        let (buffer_width, buffer_height, size) =
            Self::checked_dimensions(width, height, supersample_scale)?;

        Ok(Self {
            width,
            height,
            supersample_scale,
            buffer_width,
            buffer_height,
            samples: vec![Sample::cleared(Vector3::zeros(), f32::INFINITY); size],
        })
    }

    pub fn validate_dimensions(
        width: usize,
        height: usize,
        supersample_scale: usize,
    ) -> Result<(), String> {
        Self::checked_dimensions(width, height, supersample_scale).map(|_| ())
    }

    pub(crate) fn validate_layout(&self) -> Result<(), String> {
        let (expected_width, expected_height, expected_samples) =
            Self::checked_dimensions(self.width, self.height, self.supersample_scale)?;
        if (self.buffer_width, self.buffer_height) != (expected_width, expected_height) {
            return Err(format!(
                "buffer dimensions {}x{} do not match the expected {}x{}",
                self.buffer_width, self.buffer_height, expected_width, expected_height
            ));
        }
        if self.samples.len() != expected_samples {
            return Err(format!(
                "sample count {} does not match the expected {expected_samples}",
                self.samples.len()
            ));
        }

        Ok(())
    }

    fn checked_dimensions(
        width: usize,
        height: usize,
        supersample_scale: usize,
    ) -> Result<(usize, usize, usize), String> {
        if width == 0 || height == 0 {
            return Err("render dimensions must be greater than zero".to_string());
        }
        if supersample_scale == 0 {
            return Err("supersample_scale must be greater than zero".to_string());
        }

        let buffer_width = width
            .checked_mul(supersample_scale)
            .ok_or_else(|| "supersampled framebuffer width overflows usize".to_string())?;
        let buffer_height = height
            .checked_mul(supersample_scale)
            .ok_or_else(|| "supersampled framebuffer height overflows usize".to_string())?;
        let sample_count = buffer_width
            .checked_mul(buffer_height)
            .ok_or_else(|| "framebuffer sample count overflows usize".to_string())?;
        sample_count
            .checked_mul(std::mem::size_of::<Sample>())
            .ok_or_else(|| "framebuffer allocation size overflows usize".to_string())?;

        Ok((buffer_width, buffer_height, sample_count))
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

    pub(crate) fn copy_depth_values_into(&self, output: &mut Vec<f32>) {
        output.clear();
        output.reserve(self.samples.len());
        output.extend(self.samples.iter().map(|sample| sample.depth));
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

    pub(crate) fn clear_color(&mut self, color: Vector3<f32>) {
        let width = self.buffer_width;
        self.samples.par_chunks_mut(width).for_each(|row| {
            for sample in row {
                sample.color = color;
            }
        });
    }

    pub(crate) fn fill_color_with<F>(&mut self, color_at: F)
    where
        F: Fn(usize, usize) -> Vector3<f32> + Sync,
    {
        let width = self.buffer_width;
        self.samples
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(y, row)| {
                for (x, sample) in row.iter_mut().enumerate() {
                    sample.color = color_at(x, y);
                }
            });
    }

    pub(crate) fn clear_depth(&mut self, depth: f32) {
        let width = self.buffer_width;
        self.samples.par_chunks_mut(width).for_each(|row| {
            for sample in row {
                sample.depth = depth;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructor_rejects_invalid_dimensions() {
        assert!(FrameBuffer::new(0, 1, 1).is_err());
        assert!(FrameBuffer::new(1, 1, 0).is_err());
        assert!(FrameBuffer::new(usize::MAX, 1, 2).is_err());
        assert!(FrameBuffer::new(usize::MAX / 2 + 1, 2, 1).is_err());
        assert!(FrameBuffer::new(usize::MAX / std::mem::size_of::<Sample>() + 1, 1, 1).is_err());
    }
}
