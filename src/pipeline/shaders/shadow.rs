use crate::core::geometry::Vertex;
use crate::core::pipeline::{FragmentInput, FragmentOutput, Interpolatable, Shader};
use crate::scene::material::{AlphaMode, Material};
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

impl<'a> Shader<Option<&'a Material>> for ShadowShader {
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
        input: FragmentInput<Self::Varying>,
        material: Option<&'a Material>,
    ) -> FragmentOutput {
        let varying = input.varying;
        let pbr_material = material.map(|material| match material {
            Material::Pbr(material) => material,
        });
        let alpha = pbr_material.map_or(1.0, |material| material.alpha);
        let texture_alpha = pbr_material
            .and_then(|material| material.albedo_texture.as_ref())
            .map(|texture| {
                texture
                    .sample_color_with_density(varying.uv.x, varying.uv.y, input.uv_density)
                    .w
            })
            .unwrap_or(1.0);
        let alpha = alpha * texture_alpha;

        if matches!(pbr_material.map(|material| material.alpha_mode), Some(AlphaMode::Mask(cutoff)) if alpha < cutoff)
        {
            return FragmentOutput::Discard;
        }

        FragmentOutput::Color(Vector4::new(0.0, 0.0, 0.0, alpha))
    }
}
