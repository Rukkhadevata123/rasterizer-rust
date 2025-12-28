use crate::core::geometry::Vertex;
use crate::core::pipeline::Interpolatable;
use crate::core::pipeline::Shader;
use crate::scene::material::Material;
use nalgebra::{Matrix4, Vector3, Vector4};
use std::ops::{Add, Mul};

#[derive(Clone, Copy, Debug)]
pub struct ShadowVarying; // We don't need to interpolate anything for depth-only pass

impl Add for ShadowVarying {
    type Output = Self;
    fn add(self, _other: Self) -> Self {
        Self
    }
}

impl Mul<f32> for ShadowVarying {
    type Output = Self;
    fn mul(self, _scalar: f32) -> Self {
        Self
    }
}

// Explicitly implement Interpolatable for the depth-only varying type.
impl Interpolatable for ShadowVarying {}

pub struct ShadowShader {
    pub mvp_matrix: Matrix4<f32>,
}

impl ShadowShader {
    pub fn new(model: Matrix4<f32>, view: Matrix4<f32>, projection: Matrix4<f32>) -> Self {
        Self {
            mvp_matrix: projection * view * model,
        }
    }
}

impl Shader for ShadowShader {
    type Varying = ShadowVarying;

    fn vertex(&self, vertex: &Vertex) -> (Vector4<f32>, Self::Varying) {
        let clip_pos = self.mvp_matrix * vertex.position.to_homogeneous();
        (clip_pos, ShadowVarying)
    }

    fn fragment(
        &self,
        _varying: Self::Varying,
        _material: Option<&Material>,
        _uv_density: f32,
    ) -> Vector3<f32> {
        // Color doesn't matter for the depth-only pass; the rasterizer writes depth.
        Vector3::zeros()
    }
}
