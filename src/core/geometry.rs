use nalgebra::{Point3, Vector2, Vector3, Vector4};

pub const SUPPORTED_TEXCOORD_SETS: usize = 2;

/// Vertex attributes consumed by programmable vertex shaders.
#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    /// Position in local object space.
    pub position: Point3<f32>,
    /// Normal in local object space.
    pub normal: Vector3<f32>,
    /// Texture coordinates for `TEXCOORD_0` and `TEXCOORD_1`.
    pub texcoords: [Vector2<f32>; SUPPORTED_TEXCOORD_SETS],
    /// Tangent direction in XYZ and glTF bitangent handedness in W.
    pub tangent: Vector4<f32>,
}

impl Vertex {
    pub fn new(position: Point3<f32>, normal: Vector3<f32>, texcoord: Vector2<f32>) -> Self {
        Self {
            position,
            normal,
            texcoords: [texcoord, Vector2::zeros()],
            tangent: Vector4::zeros(),
        }
    }
}
