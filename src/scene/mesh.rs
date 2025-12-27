use crate::core::geometry::Vertex;
use nalgebra::{Point3, Vector2, Vector3};

/// A collection of vertices and indices representing a 3D object.
pub struct Mesh {
    /// List of vertices.
    pub vertices: Vec<Vertex>,
    /// List of indices defining triangles (3 indices per triangle).
    pub indices: Vec<u32>,
    /// Index into the Model's material list.
    pub material_id: usize,
}

impl Mesh {
    pub fn new(vertices: Vec<Vertex>, indices: Vec<u32>, material_id: usize) -> Self {
        Self {
            vertices,
            indices,
            material_id,
        }
    }

    pub fn create_plane(size: f32, material_id: usize) -> Self {
        let half_size = size / 2.0;
        let y = 0.0;

        // 4 Vertices
        let vertices = vec![
            // 0: Bottom-Left (Min X, Min Z)
            Vertex::new(
                Point3::new(-half_size, y, -half_size),
                Vector3::new(0.0, 1.0, 0.0), // Up
                Vector2::new(0.0, 0.0),
            ),
            // 1: Top-Left (Min X, Max Z)
            Vertex::new(
                Point3::new(-half_size, y, half_size),
                Vector3::new(0.0, 1.0, 0.0),
                Vector2::new(0.0, 10.0), // Repeat texture 10 times
            ),
            // 2: Top-Right (Max X, Max Z)
            Vertex::new(
                Point3::new(half_size, y, half_size),
                Vector3::new(0.0, 1.0, 0.0),
                Vector2::new(10.0, 10.0),
            ),
            // 3: Bottom-Right (Max X, Min Z)
            Vertex::new(
                Point3::new(half_size, y, -half_size),
                Vector3::new(0.0, 1.0, 0.0),
                Vector2::new(10.0, 0.0),
            ),
        ];

        // 2 Triangles (CCW)
        let indices = vec![
            0, 1, 2, // First triangle
            0, 2, 3, // Second triangle
        ];

        Self::new(vertices, indices, material_id)
    }

    pub fn create_test_triangle(material_id: usize) -> Self {
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
