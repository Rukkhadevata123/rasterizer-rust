use nalgebra::Vector3;

/// Represents a 2D buffer containing color and depth information.
/// Coordinate system: (0,0) is top-left.
pub struct FrameBuffer {
    pub width: usize,
    pub height: usize,
    /// Color buffer stored as linear RGB float vectors [0.0, 1.0].
    /// Layout: Row-major (y * width + x).
    pub color_buffer: Vec<Vector3<f32>>,
    /// Depth buffer (Z-buffer).
    /// Stores depth values (usually 0.0 to 1.0, or view-space Z).
    pub depth_buffer: Vec<f32>,
    // TODO: Implement Anti-Aliasing (SSAA/MSAA).
    // Idea: Store a buffer that is 2x or 4x larger than the output resolution.
    // When saving/displaying, downsample (average) the pixels to smooth edges.
}

impl FrameBuffer {
    /// Creates a new FrameBuffer with specified dimensions.
    pub fn new(width: usize, height: usize) -> Self {
        let size = width * height;
        Self {
            width,
            height,
            color_buffer: vec![Vector3::zeros(); size],
            depth_buffer: vec![f32::INFINITY; size],
        }
    }

    /// Resizes the framebuffer. Clears data.
    pub fn resize(&mut self, width: usize, height: usize) {
        if self.width == width && self.height == height {
            return;
        }
        self.width = width;
        self.height = height;
        let size = width * height;
        self.color_buffer = vec![Vector3::zeros(); size];
        self.depth_buffer = vec![f32::INFINITY; size];
    }

    /// Clears the buffers with specified values.
    pub fn clear(&mut self, clear_color: Vector3<f32>, clear_depth: f32) {
        // Using fill is faster than iterating
        self.color_buffer.fill(clear_color);
        self.depth_buffer.fill(clear_depth);
    }

    /// Checks if a pixel coordinate is within bounds.
    #[inline(always)]
    pub fn in_bounds(&self, x: usize, y: usize) -> bool {
        x < self.width && y < self.height
    }

    /// Gets the index in the linear vector for (x, y).
    #[inline(always)]
    fn index(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    /// Writes a pixel color. Does NOT perform blending or depth testing.
    #[inline]
    pub fn set_pixel(&mut self, x: usize, y: usize, color: Vector3<f32>) {
        if self.in_bounds(x, y) {
            let idx = self.index(x, y);
            self.color_buffer[idx] = color;
        }
    }

    /// Writes a depth value.
    #[inline]
    pub fn set_depth(&mut self, x: usize, y: usize, depth: f32) {
        if self.in_bounds(x, y) {
            let idx = self.index(x, y);
            self.depth_buffer[idx] = depth;
        }
    }

    /// Performs a depth test.
    /// Returns true if the new depth is closer (less than) the existing depth.
    #[inline]
    pub fn depth_test(&self, x: usize, y: usize, new_depth: f32) -> bool {
        if !self.in_bounds(x, y) {
            return false;
        }
        let idx = self.index(x, y);
        new_depth < self.depth_buffer[idx]
    }

    /// Gets the color at (x, y).
    pub fn get_pixel(&self, x: usize, y: usize) -> Option<Vector3<f32>> {
        if self.in_bounds(x, y) {
            Some(self.color_buffer[self.index(x, y)])
        } else {
            None
        }
    }
}
