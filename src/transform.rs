use nalgebra::{Matrix4, Point3};

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

/// Rotates vertices around the world Y axis.
///
/// # Arguments
/// * `vertices` - A slice of points representing model vertices in world or model space.
/// * `angle_degrees` - The rotation angle in degrees.
///
/// # Returns
/// A `Vec<Point3<f32>>` containing the rotated vertices.
pub fn rotate_model_y(vertices: &[Point3<f32>], angle_degrees: f32) -> Vec<Point3<f32>> {
    let angle_rad = angle_degrees.to_radians();
    let cos_theta = angle_rad.cos();
    let sin_theta = angle_rad.sin();

    // Rotation matrix around Y axis
    let rotation_matrix = Matrix4::new(
        cos_theta, 0.0, sin_theta, 0.0, // Col 1
        0.0, 1.0, 0.0, 0.0, // Col 2
        -sin_theta, 0.0, cos_theta, 0.0, // Col 3
        0.0, 0.0, 0.0, 1.0, // Col 4
    );

    vertices
        .iter()
        .map(|vertex| {
            // Apply rotation (Point3 needs conversion to homogeneous for matrix multiplication)
            let rotated_h = rotation_matrix * vertex.to_homogeneous();
            Point3::from_homogeneous(rotated_h).unwrap_or(*vertex) // Convert back, fallback to original if needed
        })
        .collect()
}

