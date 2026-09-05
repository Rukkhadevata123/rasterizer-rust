use crate::core::geometry::Vertex;
use nalgebra::{Point3, Vector2, Vector3};

/// Indexed triangle geometry with a material reference.
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    /// List of indices defining triangles (3 indices per triangle).
    pub indices: Vec<u32>,
    /// Index into the Model's material list.
    pub material_id: usize,
    reuses_vertices: bool,
}

impl Mesh {
    pub fn new(vertices: Vec<Vertex>, indices: Vec<u32>, material_id: usize) -> Self {
        let reuses_vertices = indices.len() > vertices.len() || {
            let mut referenced = vec![false; vertices.len()];
            indices.iter().any(|&index| {
                referenced
                    .get_mut(index as usize)
                    .is_some_and(|was_referenced| {
                        let reused = *was_referenced;
                        *was_referenced = true;
                        reused
                    })
            })
        };
        Self {
            vertices,
            indices,
            material_id,
            reuses_vertices,
        }
    }

    pub fn reuses_vertices(&self) -> bool {
        self.reuses_vertices
    }

    pub fn create_plane(size: f32, material_id: usize) -> Self {
        let half_size = size / 2.0;
        let y = 0.0;

        // Ground UVs tile the texture ten times across each axis.
        let vertices = vec![
            Vertex::new(
                Point3::new(-half_size, y, -half_size),
                Vector3::new(0.0, 1.0, 0.0),
                Vector2::new(0.0, 0.0),
            ),
            Vertex::new(
                Point3::new(-half_size, y, half_size),
                Vector3::new(0.0, 1.0, 0.0),
                Vector2::new(0.0, 10.0),
            ),
            Vertex::new(
                Point3::new(half_size, y, half_size),
                Vector3::new(0.0, 1.0, 0.0),
                Vector2::new(10.0, 10.0),
            ),
            Vertex::new(
                Point3::new(half_size, y, -half_size),
                Vector3::new(0.0, 1.0, 0.0),
                Vector2::new(10.0, 0.0),
            ),
        ];

        // Counter-clockwise when viewed from above the ground plane.
        let indices = vec![0, 1, 2, 0, 2, 3];

        Self::new(vertices, indices, material_id)
    }

    #[cfg(test)]
    pub(super) fn create_test_triangle(material_id: usize) -> Self {
        let vertices = vec![
            Vertex::new(
                Point3::new(-0.5, -0.5, 0.0),
                Vector3::new(0.0, 0.0, 1.0),
                Vector2::new(0.0, 0.0),
            ),
            Vertex::new(
                Point3::new(0.5, -0.5, 0.0),
                Vector3::new(0.0, 0.0, 1.0),
                Vector2::new(1.0, 0.0),
            ),
            Vertex::new(
                Point3::new(0.0, 0.5, 0.0),
                Vector3::new(0.0, 0.0, 1.0),
                Vector2::new(0.5, 1.0),
            ),
        ];
        let indices = vec![0, 1, 2];
        Self::new(vertices, indices, material_id)
    }
}
