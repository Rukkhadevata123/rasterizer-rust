use nalgebra::{Point3, Vector2, Vector3};

/// Represents a single vertex in 3D space.
#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    /// Position in local object space.
    pub position: Point3<f32>,
    /// Normal vector for lighting calculations.
    pub normal: Vector3<f32>,
    /// Texture coordinates (UV).
    pub texcoord: Vector2<f32>,
    /// Tangent vector (xyz) for Normal Mapping.
    /// We use Vector3 to keep interpolation simple (ignoring handedness for now).
    pub tangent: Vector3<f32>,
}

impl Vertex {
    pub fn new(position: Point3<f32>, normal: Vector3<f32>, texcoord: Vector2<f32>) -> Self {
        Self {
            position,
            normal,
            texcoord,
            tangent: Vector3::zeros(),
        }
    }
}
