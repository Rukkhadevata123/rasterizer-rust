use nalgebra::Vector3;

/// Represents a 2D buffer containing color and depth information.
/// Coordinate system: (0,0) is top-left.
pub struct FrameBuffer {
    /// The logical width of the output image.
    pub width: usize,
    /// The logical height of the output image.
    pub height: usize,

    /// Super-sampling factor (e.g., 2 for 2x2 SSAA).
    pub sample_count: usize,

    /// The actual width of the internal buffers (width * sample_count).
    pub buffer_width: usize,
    /// The actual height of the internal buffers (height * sample_count).
    pub buffer_height: usize,

    /// Color buffer stored as linear RGB float vectors [0.0, 1.0].
    pub color_buffer: Vec<Vector3<f32>>,
    /// Depth buffer (Z-buffer).
    pub depth_buffer: Vec<f32>,
}

impl FrameBuffer {
    /// Creates a new FrameBuffer with specified dimensions and SSAA factor.
    /// sample_count: 1 = No AA, 2 = 2x2 SSAA (4x pixels), etc.
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
            color_buffer: vec![Vector3::zeros(); size],
            depth_buffer: vec![f32::INFINITY; size],
        }
    }

    /// Resizes the framebuffer. Clears data.
    pub fn resize(&mut self, width: usize, height: usize, sample_count: usize) {
        if self.width == width && self.height == height && self.sample_count == sample_count {
            return;
        }
        self.width = width;
        self.height = height;
        self.sample_count = sample_count;
        self.buffer_width = width * sample_count;
        self.buffer_height = height * sample_count;

        let size = self.buffer_width * self.buffer_height;
        self.color_buffer = vec![Vector3::zeros(); size];
        self.depth_buffer = vec![f32::INFINITY; size];
    }

    /// Clears the buffers with specified values.
    pub fn clear(&mut self, clear_color: Vector3<f32>, clear_depth: f32) {
        self.color_buffer.fill(clear_color);
        self.depth_buffer.fill(clear_depth);
    }

    /// Checks if a pixel coordinate (in buffer space) is within bounds.
    #[inline(always)]
    pub fn in_bounds(&self, x: usize, y: usize) -> bool {
        x < self.buffer_width && y < self.buffer_height
    }

    /// Gets the index in the linear vector for (x, y) in buffer space.
    #[inline(always)]
    fn index(&self, x: usize, y: usize) -> usize {
        y * self.buffer_width + x
    }

    /// Writes a pixel color to the internal high-res buffer.
    #[inline]
    pub fn set_pixel(&mut self, x: usize, y: usize, color: Vector3<f32>) {
        if self.in_bounds(x, y) {
            let idx = self.index(x, y);
            self.color_buffer[idx] = color;
        }
    }

    /// Writes a depth value to the internal high-res buffer.
    #[inline]
    pub fn set_depth(&mut self, x: usize, y: usize, depth: f32) {
        if self.in_bounds(x, y) {
            let idx = self.index(x, y);
            self.depth_buffer[idx] = depth;
        }
    }

    /// Performs a depth test in buffer space.
    #[inline]
    pub fn depth_test(&self, x: usize, y: usize, new_depth: f32) -> bool {
        if !self.in_bounds(x, y) {
            return false;
        }
        let idx = self.index(x, y);
        new_depth < self.depth_buffer[idx]
    }

    /// Gets the resolved color at logical coordinates (x, y).
    /// Performs downsampling (averaging) if SSAA is enabled.
    pub fn get_pixel(&self, x: usize, y: usize) -> Option<Vector3<f32>> {
        if x >= self.width || y >= self.height {
            return None;
        }

        if self.sample_count == 1 {
            // Fast path for no AA
            return Some(self.color_buffer[self.index(x, y)]);
        }

        // Downsample: Average the colors in the sample_count x sample_count block
        let mut sum_color = Vector3::zeros();
        let start_x = x * self.sample_count;
        let start_y = y * self.sample_count;

        for dy in 0..self.sample_count {
            for dx in 0..self.sample_count {
                let idx = self.index(start_x + dx, start_y + dy);
                sum_color += self.color_buffer[idx];
            }
        }

        let samples = (self.sample_count * self.sample_count) as f32;
        Some(sum_color / samples)
    }
}
