use crate::model_types::ModelData;
use nalgebra::{Point3, Vector3}; // Ensure ModelData is accessible

// Helper function to save image data to a file
pub fn save_image(path: &str, data: &[u8], width: u32, height: u32) {
    match image::save_buffer(path, data, width, height, image::ColorType::Rgb8) {
        Ok(_) => println!("Image saved to {}", path),
        Err(e) => eprintln!("Error saving image to {}: {}", path, e),
    }
}

/// Normalizes depth buffer values for visualization using percentile clipping.
// Allow dead code because it's only used when !no_depth
pub fn normalize_depth(
    depth_buffer: &[f32],
    min_percentile: f32, // e.g., 1.0 for 1st percentile
    max_percentile: f32, // e.g., 99.0 for 99th percentile
) -> Vec<f32> {
    // 1. Collect finite depth values
    let mut finite_depths: Vec<f32> = depth_buffer
        .iter()
        .filter(|&&d| d.is_finite())
        .cloned()
        .collect();

    // Declare min_clip and max_clip as mutable
    let mut min_clip: f32;
    let mut max_clip: f32;

    // 2. Determine normalization range using percentiles
    if finite_depths.len() >= 2 {
        // Need at least two points to define a range
        // Sort the finite depths
        finite_depths.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap()); // Use unstable sort for performance

        // Calculate indices for percentiles
        let min_idx = ((min_percentile / 100.0 * (finite_depths.len() - 1) as f32).round()
            as usize)
            .clamp(0, finite_depths.len() - 1);
        let max_idx = ((max_percentile / 100.0 * (finite_depths.len() - 1) as f32).round()
            as usize)
            .clamp(0, finite_depths.len() - 1);

        min_clip = finite_depths[min_idx]; // Initial assignment
        max_clip = finite_depths[max_idx]; // Initial assignment

        // Ensure min_clip < max_clip
        if (max_clip - min_clip).abs() < 1e-6 {
            // If range is too small, expand it slightly or use a default
            // For simplicity, let's just use the absolute min/max in this edge case
            min_clip = *finite_depths.first().unwrap(); // Re-assignment is now allowed
            max_clip = *finite_depths.last().unwrap(); // Re-assignment is now allowed
            // Ensure max > min even if all values were identical
            if (max_clip - min_clip).abs() < 1e-6 {
                max_clip = min_clip + 1.0; // Re-assignment is now allowed
            }
        }
        println!(
            "Normalizing depth using percentiles: [{:.1}%, {:.1}%] -> [{:.3}, {:.3}]",
            min_percentile, max_percentile, min_clip, max_clip
        );
    } else {
        // Fallback if not enough finite values
        println!(
            "Warning: Not enough finite depth values for percentile clipping. Using default range [0.1, 10.0]."
        );
        min_clip = 0.1; // Default near // Assignment
        max_clip = 10.0; // Default far (adjust as needed) // Assignment
    }

    let range = max_clip - min_clip;
    let inv_range = if range > 1e-6 { 1.0 / range } else { 0.0 }; // Avoid division by zero

    // 3. Normalize the original buffer using the calculated range
    depth_buffer
        .iter()
        .map(|&depth| {
            if depth.is_finite() {
                // Clamp depth to the calculated range and normalize
                ((depth.clamp(min_clip, max_clip) - min_clip) * inv_range).clamp(0.0, 1.0)
            } else {
                // Map non-finite values (infinity) to 1.0 (far)
                1.0
            }
        })
        .collect()
}

/// Normalizes and centers the model's vertices in place.
/// Returns the original center and scaling factor.
pub fn normalize_and_center_model(model_data: &mut ModelData) -> (Vector3<f32>, f32) {
    if model_data.meshes.is_empty() {
        return (Vector3::zeros(), 1.0);
    }

    // Calculate bounding box or centroid of all vertices
    let mut min_coord = Point3::new(f32::MAX, f32::MAX, f32::MAX);
    let mut max_coord = Point3::new(f32::MIN, f32::MIN, f32::MIN);
    let mut vertex_sum = Vector3::zeros();
    let mut vertex_count = 0;

    for mesh in &model_data.meshes {
        for vertex in &mesh.vertices {
            min_coord = min_coord.inf(&vertex.position);
            max_coord = max_coord.sup(&vertex.position);
            vertex_sum += vertex.position.coords;
            vertex_count += 1;
        }
    }

    if vertex_count == 0 {
        return (Vector3::zeros(), 1.0);
    }

    let center = vertex_sum / (vertex_count as f32);
    let extent = max_coord - min_coord;
    let max_extent = extent.x.max(extent.y).max(extent.z);

    let scale_factor = if max_extent > 1e-6 {
        1.6 / max_extent // Scale to fit roughly in [-0.8, 0.8] cube (like Python's 0.8 factor)
    } else {
        1.0
    };

    // Apply transformation to all vertices
    for mesh in &mut model_data.meshes {
        for vertex in &mut mesh.vertices {
            vertex.position = Point3::from((vertex.position.coords - center) * scale_factor);
        }
    }

    (center, scale_factor)
}
