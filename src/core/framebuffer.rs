use nalgebra::Vector3;
use std::cell::UnsafeCell;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};

/// Represents a 2D buffer containing color and depth information.
/// Thread-safe for parallel rendering using atomic depth and striped locking for color.
pub struct FrameBuffer {
    pub width: usize,
    pub height: usize,
    pub sample_count: usize,
    pub buffer_width: usize,
    pub buffer_height: usize,

    /// Color buffer wrapped in UnsafeCell to allow interior mutability.
    /// Safety is guaranteed by `locks` and depth testing.
    pub color_buffer: UnsafeCell<Vec<Vector3<f32>>>,

    /// Depth buffer stored as atomic bits of f32.
    pub depth_buffer: Vec<AtomicU32>,

    /// Striped locks to protect color writes.
    /// We map pixel coordinates to a lock index to reduce contention.
    locks: Vec<Mutex<()>>,
}

// We implement Sync because we manage thread safety manually via Atomics and Locks.
unsafe impl Sync for FrameBuffer {}

impl FrameBuffer {
    pub fn new(width: usize, height: usize, sample_count: usize) -> Self {
        let buffer_width = width * sample_count;
        let buffer_height = height * sample_count;
        let size = buffer_width * buffer_height;

        // Initialize depth with f32::INFINITY bits
        let inf_bits = f32::INFINITY.to_bits();
        let mut depth_buffer = Vec::with_capacity(size);
        for _ in 0..size {
            depth_buffer.push(AtomicU32::new(inf_bits));
        }

        // Create a pool of locks (e.g., 1024 locks) to reduce memory overhead
        // compared to one lock per pixel.
        let lock_count = 1024;
        let mut locks = Vec::with_capacity(lock_count);
        for _ in 0..lock_count {
            locks.push(Mutex::new(()));
        }

        Self {
            width,
            height,
            sample_count,
            buffer_width,
            buffer_height,
            color_buffer: UnsafeCell::new(vec![Vector3::zeros(); size]),
            depth_buffer,
            locks,
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

    /// Thread-safe depth test and update.
    /// Returns true if the new depth is closer than the existing value.
    /// If true, it updates the depth buffer atomically.
    #[inline]
    pub fn depth_test_and_update(&self, x: usize, y: usize, new_depth: f32) -> bool {
        if !self.in_bounds(x, y) {
            return false;
        }
        let idx = self.index(x, y);
        let new_bits = new_depth.to_bits();
        let depth_atomic = &self.depth_buffer[idx];

        // CAS Loop (Compare and Swap)
        let mut current_bits = depth_atomic.load(Ordering::Relaxed);
        loop {
            let current_depth = f32::from_bits(current_bits);
            if new_depth >= current_depth {
                return false; // Failed test
            }

            // Try to swap
            match depth_atomic.compare_exchange_weak(
                current_bits,
                new_bits,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,                             // Success
                Err(updated_bits) => current_bits = updated_bits, // Retry with new value
            }
        }
    }

    /// Thread-safe pixel write.
    /// Should only be called AFTER depth_test_and_update returns true.
    #[inline]
    pub fn set_pixel_safe(&self, x: usize, y: usize, color: Vector3<f32>) {
        if self.in_bounds(x, y) {
            let idx = self.index(x, y);

            // Map pixel index to a lock index (simple hashing)
            let lock_idx = idx % self.locks.len();
            let _guard = self.locks[lock_idx].lock().unwrap();

            // Unsafe access is safe because we hold the lock for this "stripe" of pixels
            unsafe {
                let buffer = &mut *self.color_buffer.get();
                buffer[idx] = color;
            }
        }
    }

    // Legacy helper for single-threaded context or read-only
    pub fn get_pixel(&self, x: usize, y: usize) -> Option<Vector3<f32>> {
        if x >= self.width || y >= self.height {
            return None;
        }

        // Reading doesn't strictly need locks if we accept tearing during rendering,
        // but for outputting the final image (when rendering is done), it's safe.
        let buffer = unsafe { &*self.color_buffer.get() };

        if self.sample_count == 1 {
            return Some(buffer[self.index(x, y)]);
        }

        let mut sum_color = Vector3::zeros();
        let start_x = x * self.sample_count;
        let start_y = y * self.sample_count;

        for dy in 0..self.sample_count {
            for dx in 0..self.sample_count {
                let idx = self.index(start_x + dx, start_y + dy);
                sum_color += buffer[idx];
            }
        }

        let samples = (self.sample_count * self.sample_count) as f32;
        Some(sum_color / samples)
    }
}
