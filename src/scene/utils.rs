use crate::scene::model::Model;
use nalgebra::Point3;

/// Centers the model at the origin without changing its dimensions.
pub fn center_model(model: &mut Model) -> Point3<f32> {
    let Some((min_bound, max_bound)) = model_bounds(model) else {
        return Point3::origin();
    };
    let center = nalgebra::center(&min_bound, &max_bound);
    transform_vertices(model, center, 1.0);
    center
}

/// Analyzes the model's bounding box and transforms all vertices
/// so that the model is centered at (0,0,0) and fits within a unit sphere [-1, 1].
///
/// Returns the original center and the scaling factor used.
pub fn normalize_and_center_model(model: &mut Model) -> (Point3<f32>, f32) {
    let Some((min_bound, max_bound)) = model_bounds(model) else {
        return (Point3::origin(), 1.0);
    };
    let center = nalgebra::center(&min_bound, &max_bound);
    let extent = max_bound - min_bound;
    let max_dimension = extent.x.max(extent.y).max(extent.z);
    let scale_factor = if max_dimension > 1e-6 {
        1.8 / max_dimension
    } else {
        1.0
    };
    transform_vertices(model, center, scale_factor);
    (center, scale_factor)
}

fn model_bounds(model: &Model) -> Option<(Point3<f32>, Point3<f32>)> {
    let mut min_bound = Point3::new(f32::MAX, f32::MAX, f32::MAX);
    let mut max_bound = Point3::new(f32::MIN, f32::MIN, f32::MIN);
    let mut has_vertices = false;

    for mesh in &model.meshes {
        for vertex in &mesh.vertices {
            min_bound.x = min_bound.x.min(vertex.position.x);
            min_bound.y = min_bound.y.min(vertex.position.y);
            min_bound.z = min_bound.z.min(vertex.position.z);

            max_bound.x = max_bound.x.max(vertex.position.x);
            max_bound.y = max_bound.y.max(vertex.position.y);
            max_bound.z = max_bound.z.max(vertex.position.z);
            has_vertices = true;
        }
    }

    if has_vertices {
        Some((min_bound, max_bound))
    } else {
        None
    }
}

fn transform_vertices(model: &mut Model, center: Point3<f32>, scale: f32) {
    for mesh in &mut model.meshes {
        for vertex in &mut mesh.vertices {
            vertex.position = Point3::from((vertex.position - center) * scale);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::geometry::Vertex;
    use crate::scene::mesh::Mesh;
    use nalgebra::{Vector2, Vector3};

    fn model_with_extent(origin: Point3<f32>, extent: Vector3<f32>) -> Model {
        let vertices = vec![
            Vertex::new(origin, Vector3::z(), Vector2::zeros()),
            Vertex::new(origin + extent, Vector3::z(), Vector2::zeros()),
        ];
        Model::new(vec![Mesh::new(vertices, vec![], 0)], vec![])
    }

    fn bounds(model: &Model) -> (Point3<f32>, Point3<f32>) {
        model_bounds(model).expect("test model should have vertices")
    }

    #[test]
    fn center_preserves_dimensions_across_asset_placements() {
        for (origin, extent) in [
            (Point3::new(10.0, -4.0, 2.0), Vector3::new(2.0, 4.0, 6.0)),
            (Point3::new(-3.0, 8.0, 1.0), Vector3::new(8.0, 1.0, 2.0)),
        ] {
            let mut model = model_with_extent(origin, extent);
            center_model(&mut model);
            let (min_bound, max_bound) = bounds(&model);

            assert_eq!(max_bound - min_bound, extent);
            assert_eq!(nalgebra::center(&min_bound, &max_bound), Point3::origin());
        }
    }

    #[test]
    fn normalize_centers_assets_and_sets_largest_dimension() {
        for (origin, extent) in [
            (Point3::new(10.0, -4.0, 2.0), Vector3::new(2.0, 4.0, 6.0)),
            (Point3::new(-3.0, 8.0, 1.0), Vector3::new(8.0, 1.0, 2.0)),
        ] {
            let mut model = model_with_extent(origin, extent);
            normalize_and_center_model(&mut model);
            let (min_bound, max_bound) = bounds(&model);
            let normalized_extent = max_bound - min_bound;

            assert_eq!(nalgebra::center(&min_bound, &max_bound), Point3::origin());
            assert!((normalized_extent.max() - 1.8).abs() < 1.0e-6);
        }
    }
}
