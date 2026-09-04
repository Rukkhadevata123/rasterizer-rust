use nalgebra::Vector3;
use rayon::prelude::*;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum RenderTargetError {
    #[error("render dimensions must be greater than zero, got {width}x{height}")]
    ZeroDimensions { width: usize, height: usize },
    #[error("supersample scale must be greater than zero")]
    ZeroSupersampleScale,
    #[error("supersampled framebuffer width overflows usize")]
    SupersampledWidthOverflow,
    #[error("supersampled framebuffer height overflows usize")]
    SupersampledHeightOverflow,
    #[error("framebuffer sample count overflows usize")]
    SampleCountOverflow,
    #[error("framebuffer allocation size overflows usize")]
    AllocationSizeOverflow,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub(crate) enum FrameBufferLayoutError {
    #[error(transparent)]
    InvalidDimensions(#[from] RenderTargetError),
    #[error(
        "buffer dimensions {actual_width}x{actual_height} do not match the expected {expected_width}x{expected_height}"
    )]
    LayoutDimensionsMismatch {
        actual_width: usize,
        actual_height: usize,
        expected_width: usize,
        expected_height: usize,
    },
    #[error("sample count {actual} does not match the expected {expected}")]
    LayoutSampleCountMismatch { actual: usize, expected: usize },
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Sample {
    pub(crate) color: Vector3<f32>,
    pub(crate) depth: f32,
}

impl Sample {
    fn cleared(color: Vector3<f32>, depth: f32) -> Self {
        Self { color, depth }
    }
}

pub(crate) struct FrameBuffer {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) supersample_scale: usize,
    pub(crate) buffer_width: usize,
    pub(crate) buffer_height: usize,
    samples: Vec<Sample>,
}

impl FrameBuffer {
    pub(crate) fn new(
        width: usize,
        height: usize,
        supersample_scale: usize,
    ) -> Result<Self, RenderTargetError> {
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

    pub(crate) fn validate_dimensions(
        width: usize,
        height: usize,
        supersample_scale: usize,
    ) -> Result<(), RenderTargetError> {
        Self::checked_dimensions(width, height, supersample_scale).map(|_| ())
    }

    pub(crate) fn validate_layout(&self) -> Result<(), FrameBufferLayoutError> {
        let (expected_width, expected_height, expected_samples) =
            Self::checked_dimensions(self.width, self.height, self.supersample_scale)?;
        if (self.buffer_width, self.buffer_height) != (expected_width, expected_height) {
            return Err(FrameBufferLayoutError::LayoutDimensionsMismatch {
                actual_width: self.buffer_width,
                actual_height: self.buffer_height,
                expected_width,
                expected_height,
            });
        }
        if self.samples.len() != expected_samples {
            return Err(FrameBufferLayoutError::LayoutSampleCountMismatch {
                actual: self.samples.len(),
                expected: expected_samples,
            });
        }

        Ok(())
    }

    fn checked_dimensions(
        width: usize,
        height: usize,
        supersample_scale: usize,
    ) -> Result<(usize, usize, usize), RenderTargetError> {
        if width == 0 || height == 0 {
            return Err(RenderTargetError::ZeroDimensions { width, height });
        }
        if supersample_scale == 0 {
            return Err(RenderTargetError::ZeroSupersampleScale);
        }

        let buffer_width = width
            .checked_mul(supersample_scale)
            .ok_or(RenderTargetError::SupersampledWidthOverflow)?;
        let buffer_height = height
            .checked_mul(supersample_scale)
            .ok_or(RenderTargetError::SupersampledHeightOverflow)?;
        let sample_count = buffer_width
            .checked_mul(buffer_height)
            .ok_or(RenderTargetError::SampleCountOverflow)?;
        sample_count
            .checked_mul(std::mem::size_of::<Sample>())
            .ok_or(RenderTargetError::AllocationSizeOverflow)?;

        Ok((buffer_width, buffer_height, sample_count))
    }

    #[inline(always)]
    pub(crate) fn in_bounds(&self, x: usize, y: usize) -> bool {
        x < self.buffer_width && y < self.buffer_height
    }

    #[inline(always)]
    fn index(&self, x: usize, y: usize) -> usize {
        y * self.buffer_width + x
    }

    pub(crate) fn sample(&self, x: usize, y: usize) -> Option<&Sample> {
        self.in_bounds(x, y)
            .then(|| &self.samples[self.index(x, y)])
    }

    pub(crate) fn samples_mut(&mut self) -> &mut [Sample] {
        &mut self.samples
    }

    pub(crate) fn depth_values(&self) -> Vec<f32> {
        self.samples.iter().map(|sample| sample.depth).collect()
    }

    pub(crate) fn copy_depth_values_into(&self, output: &mut Vec<f32>) {
        output.clear();
        output.reserve(self.samples.len());
        output.extend(self.samples.iter().map(|sample| sample.depth));
    }

    pub(crate) fn clear_with<F>(&mut self, depth: f32, color_at: F)
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

    pub(crate) fn get_pixel(&self, x: usize, y: usize) -> Option<Vector3<f32>> {
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
        assert!(matches!(
            FrameBuffer::new(0, 1, 1),
            Err(RenderTargetError::ZeroDimensions {
                width: 0,
                height: 1,
            })
        ));
        assert!(matches!(
            FrameBuffer::new(1, 1, 0),
            Err(RenderTargetError::ZeroSupersampleScale)
        ));
        assert!(matches!(
            FrameBuffer::new(usize::MAX, 1, 2),
            Err(RenderTargetError::SupersampledWidthOverflow)
        ));
        assert!(matches!(
            FrameBuffer::new(usize::MAX / 2 + 1, 2, 1),
            Err(RenderTargetError::SampleCountOverflow)
        ));
        assert!(matches!(
            FrameBuffer::new(usize::MAX / std::mem::size_of::<Sample>() + 1, 1, 1),
            Err(RenderTargetError::AllocationSizeOverflow)
        ));
    }

    #[test]
    fn resolves_supersampled_pixels() {
        let mut framebuffer = FrameBuffer::new(1, 1, 2).expect("test dimensions should be valid");
        framebuffer.clear_with(f32::INFINITY, |x, y| match (x, y) {
            (0, 0) => Vector3::new(1.0, 0.0, 0.0),
            (1, 0) => Vector3::new(0.0, 1.0, 0.0),
            (0, 1) => Vector3::new(0.0, 0.0, 1.0),
            (1, 1) => Vector3::new(1.0, 1.0, 1.0),
            _ => unreachable!(),
        });

        let color = framebuffer
            .get_pixel(0, 0)
            .expect("resolved pixel should be in bounds");
        assert!((color - Vector3::new(0.5, 0.5, 0.5)).norm() < 1e-4);
    }
}
