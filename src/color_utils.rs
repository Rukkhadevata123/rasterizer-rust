use nalgebra::Vector3;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Represents an RGB color with float components [0.0, 1.0].
pub type Color = Vector3<f32>;

/// 应用gamma矫正，将线性RGB值转换为sRGB空间
///
/// # Arguments
/// * `linear_color` - 线性空间的RGB颜色值 [0.0-1.0]
///
/// # Returns
/// 应用了gamma矫正的RGB颜色值 [0.0-1.0]
pub fn apply_gamma_correction(linear_color: &Color) -> Color {
    // 使用标准的gamma值2.2
    let gamma = 2.2;
    let inv_gamma = 1.0 / gamma;

    // 对每个颜色通道应用幂函数
    Color::new(
        linear_color.x.powf(inv_gamma),
        linear_color.y.powf(inv_gamma),
        linear_color.z.powf(inv_gamma),
    )
}

/// 从sRGB空间转换回线性RGB值（解码）
///
/// # Arguments
/// * `srgb_color` - sRGB空间的RGB颜色值 [0.0-1.0]
///
/// # Returns
/// 线性空间的RGB颜色值 [0.0-1.0]
pub fn srgb_to_linear(srgb_color: &Color) -> Color {
    // 使用标准的gamma值2.2
    let gamma = 2.2;

    // 应用逆变换
    Color::new(
        srgb_color.x.powf(gamma),
        srgb_color.y.powf(gamma),
        srgb_color.z.powf(gamma),
    )
}

/// 将线性RGB值转换为u8数组，应用gamma矫正
///
/// # Arguments
/// * `linear_color` - 线性空间的RGB颜色值 [0.0-1.0]
/// * `apply_gamma` - 是否应用gamma矫正
///
/// # Returns
/// 一个包含三个u8值的数组，表示颜色的RGB通道
pub fn linear_rgb_to_u8(linear_color: &Color, apply_gamma: bool) -> [u8; 3] {
    let display_color = if apply_gamma {
        apply_gamma_correction(linear_color)
    } else {
        *linear_color
    };

    [
        (display_color.x * 255.0).clamp(0.0, 255.0) as u8,
        (display_color.y * 255.0).clamp(0.0, 255.0) as u8,
        (display_color.z * 255.0).clamp(0.0, 255.0) as u8,
    ]
}

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
/// * `apply_gamma` - 是否应用gamma矫正
///
/// # Returns
/// A `Vec<u8>` containing the flattened RGB image data (0-255 per channel).
pub fn apply_colormap_jet(
    normalized_depth: &[f32],
    width: usize,
    height: usize,
    apply_gamma: bool,
) -> Vec<u8> {
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

                let color = Color::new(r, g, b);
                let [r_u8, g_u8, b_u8] = linear_rgb_to_u8(&color, apply_gamma);

                // Write to result buffer
                let base_index = index * 3;
                result[base_index] = r_u8;
                result[base_index + 1] = g_u8;
                result[base_index + 2] = b_u8;
            }
            // If depth is not finite, pixel remains black (initialized to 0)
        }
    }

    result
}
