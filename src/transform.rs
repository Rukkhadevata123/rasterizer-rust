use nalgebra::Point3;

/// Transforms Normalized Device Coordinates (NDC) to screen pixel coordinates.
/// NDC range is assumed to be [-1, 1] for x and y.
/// Pixel coordinates origin (0,0) is typically top-left.
///
/// # Arguments
/// * `ndc_coords` - A slice of points in NDC space (x, y, z). Z is usually ignored but kept.
/// * `width` - The width of the target screen/image in pixels.
/// * `height` - The height of the target screen/image in pixels.
///
/// # Returns
/// A `Vec<Point3<f32>>` containing coordinates in pixel space.
/// X maps from [-1, 1] to [0, width].
/// Y maps from [-1, 1] to [height, 0] (flips Y axis).
/// Z is typically passed through unchanged.
pub fn ndc_to_pixel(ndc_coords: &[Point3<f32>], width: f32, height: f32) -> Vec<Point3<f32>> {
    ndc_coords
        .iter()
        .map(|ndc| {
            let pixel_x = (ndc.x + 1.0) * width / 2.0;
            // Flip Y axis: NDC +1 is top, Pixel 0 is top
            let pixel_y = height - (ndc.y + 1.0) * height / 2.0;
            // let pixel_y = (ndc.y + 1.0) * height / 2.0; // Use if Y is not flipped

            Point3::new(pixel_x, pixel_y, ndc.z) // Pass Z through
        })
        .collect()
}
