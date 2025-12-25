use crate::core::geometry::Vertex;
use nalgebra::{Point3, Vector2, Vector3};

/// A collection of vertices and indices representing a 3D object.
pub struct Mesh {
    /// List of vertices.
    pub vertices: Vec<Vertex>,
    /// List of indices defining triangles (3 indices per triangle).
    pub indices: Vec<u32>,
}

impl Mesh {
    pub fn new(vertices: Vec<Vertex>, indices: Vec<u32>) -> Self {
        Self { vertices, indices }
    }

    /// Creates a simple triangle mesh for testing purposes.
    ///
    /// Vertices are arranged in Counter-Clockwise (CCW) order.
    pub fn create_test_triangle() -> Self {
        let vertices = vec![
            Vertex::new(
                Point3::new(0.0, 0.5, 0.0),  // Top
                Vector3::new(0.0, 0.0, 1.0), // Normal facing Z+
                Vector2::new(0.5, 1.0),      // UV
            ),
            Vertex::new(
                Point3::new(-0.5, -0.5, 0.0), // Bottom Left
                Vector3::new(0.0, 0.0, 1.0),
                Vector2::new(0.0, 0.0),
            ),
            Vertex::new(
                Point3::new(0.5, -0.5, 0.0), // Bottom Right
                Vector3::new(0.0, 0.0, 1.0),
                Vector2::new(1.0, 0.0),
            ),
        ];

        let indices = vec![0, 1, 2];

        Self::new(vertices, indices)
    }
}
