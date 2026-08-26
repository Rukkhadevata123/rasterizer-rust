use crate::core::geometry::Vertex;
use crate::core::pipeline::{Interpolatable, Shader};
use crate::scene::material::Material;
use nalgebra::{Matrix4, Vector2, Vector4};
use std::ops::{Add, Mul};

#[derive(Clone, Copy, Debug)]
pub struct ShadowVarying {
    uv: Vector2<f32>,
}

impl Add for ShadowVarying {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            uv: self.uv + other.uv,
        }
    }
}

impl Mul<f32> for ShadowVarying {
    type Output = Self;

    fn mul(self, scalar: f32) -> Self {
        Self {
            uv: self.uv * scalar,
        }
    }
}

impl Interpolatable for ShadowVarying {
    fn get_uv(&self) -> Option<Vector2<f32>> {
        Some(self.uv)
    }
}

pub struct ShadowShader {
    mvp_matrix: Matrix4<f32>,
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
        (
            self.mvp_matrix * vertex.position.to_homogeneous(),
            ShadowVarying {
                uv: vertex.texcoord,
            },
        )
    }

    fn fragment(
        &self,
        varying: Self::Varying,
        material: Option<&Material>,
        uv_density: f32,
    ) -> Vector4<f32> {
        let alpha = material
            .map(|material| match material {
                Material::Pbr(material) => material.alpha,
            })
            .unwrap_or(1.0);
        let texture_alpha = material
            .and_then(|material| match material {
                Material::Pbr(material) => material.albedo_texture.as_ref(),
            })
            .map(|texture| {
                texture
                    .sample_color_with_density(varying.uv.x, varying.uv.y, uv_density)
                    .w
            })
            .unwrap_or(1.0);

        Vector4::new(0.0, 0.0, 0.0, alpha * texture_alpha)
    }
}
