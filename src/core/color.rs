use nalgebra::Vector3;

/// ACES (Academy Color Encoding System) filmic tone mapping curve.
/// Maps high dynamic range (HDR) values to [0, 1] range with a film-like look.
pub fn aces_tone_mapping(color: Vector3<f32>) -> Vector3<f32> {
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

/// Converts linear RGB to sRGB (Gamma Correction).
/// Usually applied after tone mapping.
pub fn linear_to_srgb(color: Vector3<f32>) -> Vector3<f32> {
    let gamma = 1.0 / 2.2;
    Vector3::new(
        color.x.powf(gamma),
        color.y.powf(gamma),
        color.z.powf(gamma),
    )
}
