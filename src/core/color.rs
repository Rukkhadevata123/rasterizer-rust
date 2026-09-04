use nalgebra::Vector3;

fn linear_channel_to_srgb(value: f32) -> f32 {
    if !value.is_finite() || value <= 0.0 {
        0.0
    } else if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

fn srgb_channel_to_linear(value: f32) -> f32 {
    if !value.is_finite() || value <= 0.0 {
        0.0
    } else if value <= 0.040_45 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn sanitize_linear_channel(value: f32) -> f32 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        0.0
    }
}

/// ACES (Academy Color Encoding System) filmic tone mapping curve.
/// Maps high dynamic range (HDR) values to [0, 1] range with a film-like look.
pub(crate) fn aces_tone_mapping(color: Vector3<f32>) -> Vector3<f32> {
    let color = color.map(sanitize_linear_channel);
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;

    let r = (color.x * (a * color.x + b)) / (color.x * (c * color.x + d) + e);
    let g = (color.y * (a * color.y + b)) / (color.y * (c * color.y + d) + e);
    let b = (color.z * (a * color.z + b)) / (color.z * (c * color.z + d) + e);

    Vector3::new(r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0))
}

/// Converts linear RGB to sRGB using the standard piecewise transfer function.
pub(crate) fn linear_to_srgb(color: Vector3<f32>) -> Vector3<f32> {
    color.map(linear_channel_to_srgb)
}

/// Converts sRGB to linear RGB using the standard piecewise transfer function.
pub(crate) fn srgb_to_linear(color: Vector3<f32>) -> Vector3<f32> {
    color.map(srgb_channel_to_linear)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_vec3_approx(actual: Vector3<f32>, expected: Vector3<f32>) {
        assert!(
            (actual - expected).norm() < 1.0e-6,
            "expected {expected:?}, got {actual:?}"
        );
    }

    #[test]
    fn srgb_transfer_functions_match_standard_reference_values() {
        assert_vec3_approx(
            srgb_to_linear(Vector3::new(0.040_45, 0.5, 1.0)),
            Vector3::new(0.003_130_805, 0.214_041_14, 1.0),
        );
        assert_vec3_approx(
            linear_to_srgb(Vector3::new(0.003_130_8, 0.214_041_14, 1.0)),
            Vector3::new(0.040_449_936, 0.5, 1.0),
        );
    }

    #[test]
    fn output_transforms_sanitize_negative_and_non_finite_channels() {
        let invalid = Vector3::new(-1.0, f32::NAN, f32::INFINITY);
        assert_eq!(aces_tone_mapping(invalid), Vector3::zeros());
        assert_eq!(linear_to_srgb(invalid), Vector3::zeros());
    }
}
