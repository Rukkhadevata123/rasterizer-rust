use crate::scene::model::Model;
use nalgebra::Point3;

/// Analyzes the model's bounding box and transforms all vertices
/// so that the model is centered at (0,0,0) and fits within a unit sphere [-1, 1].
///
/// Returns the original center and the scaling factor used.
pub fn normalize_and_center_model(model: &mut Model) -> (Point3<f32>, f32) {
    if model.meshes.is_empty() {
        return (Point3::origin(), 1.0);
    }

    // 1. Calculate Bounding Box
    let mut min_bound = Point3::new(f32::MAX, f32::MAX, f32::MAX);
    let mut max_bound = Point3::new(f32::MIN, f32::MIN, f32::MIN);
    let mut has_verts = false;

    for mesh in &model.meshes {
        for vertex in &mesh.vertices {
            min_bound.x = min_bound.x.min(vertex.position.x);
            min_bound.y = min_bound.y.min(vertex.position.y);
            min_bound.z = min_bound.z.min(vertex.position.z);

            max_bound.x = max_bound.x.max(vertex.position.x);
            max_bound.y = max_bound.y.max(vertex.position.y);
            max_bound.z = max_bound.z.max(vertex.position.z);
            has_verts = true;
        }
    }

    if !has_verts {
        return (Point3::origin(), 1.0);
    }

    // 2. Calculate Center and Size
    let center = nalgebra::center(&min_bound, &max_bound);
    let extent = max_bound - min_bound;
    let max_dimension = extent.x.max(extent.y).max(extent.z);

    // Scale to fit in [-1, 1] (size 2.0), with a little padding (1.8)
    let scale_factor = if max_dimension > 1e-6 {
        1.8 / max_dimension
    } else {
        1.0
    };

    // 3. Apply Transform to all vertices
    for mesh in &mut model.meshes {
        for vertex in &mut mesh.vertices {
            // Translate to origin then scale
            let centered = vertex.position - center;
            vertex.position = Point3::from(centered * scale_factor);
        }
    }

    (center, scale_factor)
}
