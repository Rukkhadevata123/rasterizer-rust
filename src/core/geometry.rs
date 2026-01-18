use nalgebra::{Point3, Vector2, Vector3, Vector4};

/// Represents a single vertex in 3D space.
#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    /// Position in local object space.
    pub position: Point3<f32>,
    /// Normal vector for lighting calculations.
    pub normal: Vector3<f32>,
    /// Texture coordinates (UV).
    pub texcoord: Vector2<f32>,
    /// Changed from Vector3 to Vector4 to store the Sign (w component) required by glTF/PBR.
    pub tangent: Vector4<f32>,
}

impl Vertex {
    pub fn new(position: Point3<f32>, normal: Vector3<f32>, texcoord: Vector2<f32>) -> Self {
        Self {
            position,
            normal,
            texcoord,
            tangent: Vector4::zeros(),
        }
    }
}
