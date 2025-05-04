use nalgebra::Vector3;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Represents an RGB color with float components [0.0, 1.0].
pub type Color = Vector3<f32>;

/// Gets the base color for a face.
///
/// If `colorize` is false, returns a default gray.
/// If `colorize` is true, generates a pseudo-random color based on the face index
/// (deterministic for the same index).
pub fn get_face_color(face_index: usize, colorize: bool) -> Color {
    if !colorize {
        // Default gray
        Color::new(0.7, 0.7, 0.7)
    } else {
        // Generate pseudo-random color based on face index
        // Seed the RNG with the face index for deterministic results
        let mut rng = StdRng::seed_from_u64(face_index as u64);
        Color::new(
            0.3 + rng.random::<f32>() * 0.4, // R in [0.3, 0.7)
            0.3 + rng.random::<f32>() * 0.4, // G in [0.3, 0.7)
            0.3 + rng.random::<f32>() * 0.4, // B in [0.3, 0.7)
        )
    }
}

/// Converts a normalized depth map (values 0.0-1.0) into an RGB color image
/// using the JET colormap.
///
/// Invalid depth values (NaN, Infinity) will result in black pixels.
///
/// # Arguments
/// * `normalized_depth` - Flattened slice of depth values (row-major).
/// * `width` - Width of the depth map.
/// * `height` - Height of the depth map.
///
/// # Returns
/// A `Vec<u8>` containing the flattened RGB image data (0-255 per channel).
// #[allow(unused_assignments)]  // 抑制 g 变量的未使用赋值警告
pub fn apply_colormap_jet(normalized_depth: &[f32], width: usize, height: usize) -> Vec<u8> {
    let num_pixels = width * height;
    if normalized_depth.len() != num_pixels {
        // Or return an error Result
        panic!("Depth buffer size does not match width * height");
    }

    let mut result = vec![0u8; num_pixels * 3]; // Initialize with black

    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            let depth = normalized_depth[index];

            if depth.is_finite() {
                let value = depth.clamp(0.0, 1.0); // Ensure value is in [0, 1]

                let mut r = 0.0;
                let g;
                let mut b = 0.0;

                // Apply JET colormap logic
                if value <= 0.25 {
                    // Blue to Cyan
                    b = 1.0;
                    g = value * 4.0;
                } else if value <= 0.5 {
                    // Cyan to Green
                    g = 1.0;
                    b = 1.0 - (value - 0.25) * 4.0;
                } else if value <= 0.75 {
                    // Green to Yellow
                    g = 1.0;
                    r = (value - 0.5) * 4.0;
                } else {
                    // Yellow to Red
                    r = 1.0;
                    g = 1.0 - (value - 0.75) * 4.0;
                }

                // Convert [0,1] float to [0,255] u8 and write to result buffer
                let base_index = index * 3;
                result[base_index] = (r * 255.0).clamp(0.0, 255.0) as u8;
                result[base_index + 1] = (g * 255.0).clamp(0.0, 255.0) as u8;
                result[base_index + 2] = (b * 255.0).clamp(0.0, 255.0) as u8;
            }
            // If depth is not finite, pixel remains black (initialized to 0)
        }
    }

    result
}
