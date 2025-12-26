use nalgebra::{Point3, Vector2, Vector3};

/// Represents a single vertex in 3D space.
/// Contains position, normal vector, and texture coordinates.
#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    /// Position in local object space.
    pub position: Point3<f32>,
    /// Normal vector for lighting calculations.
    pub normal: Vector3<f32>,
    /// Texture coordinates (UV) for texture mapping.
    pub texcoord: Vector2<f32>,
}

impl Vertex {
    /// Creates a new vertex.
    pub fn new(position: Point3<f32>, normal: Vector3<f32>, texcoord: Vector2<f32>) -> Self {
        Self {
            position,
            normal,
            texcoord,
        }
    }
}
