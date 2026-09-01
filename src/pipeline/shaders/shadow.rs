use crate::core::geometry::{SUPPORTED_TEXCOORD_SETS, Vertex};
use crate::core::shader::{FragmentInput, FragmentOutput, Interpolatable, Shader};
use crate::scene::material::{AlphaMode, Material};
use nalgebra::{Matrix4, Vector2, Vector4};
use std::ops::{Add, Mul};

#[derive(Clone, Copy, Debug)]
pub struct ShadowVarying {
    uvs: [Vector2<f32>; SUPPORTED_TEXCOORD_SETS],
}

impl Add for ShadowVarying {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            uvs: std::array::from_fn(|set| self.uvs[set] + other.uvs[set]),
        }
    }
}

impl Mul<f32> for ShadowVarying {
    type Output = Self;

    fn mul(self, scalar: f32) -> Self {
        Self {
            uvs: std::array::from_fn(|set| self.uvs[set] * scalar),
        }
    }
}

impl Interpolatable for ShadowVarying {
    fn get_uv(&self, set: usize) -> Option<Vector2<f32>> {
        self.uvs.get(set).copied()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ShadowFrameBindings {
    light_space_matrix: Matrix4<f32>,
}

impl ShadowFrameBindings {
    pub fn new(view: Matrix4<f32>, projection: Matrix4<f32>) -> Self {
        Self {
            light_space_matrix: projection * view,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ShadowObjectBindings {
    model_matrix: Matrix4<f32>,
}

impl ShadowObjectBindings {
    pub fn new(model_matrix: Matrix4<f32>) -> Self {
        Self { model_matrix }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ShadowMaterialBindings<'a> {
    material: Option<&'a Material>,
}

impl<'a> ShadowMaterialBindings<'a> {
    pub fn new(material: Option<&'a Material>) -> Self {
        Self { material }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ShadowDrawContext<'a> {
    frame: &'a ShadowFrameBindings,
    object: &'a ShadowObjectBindings,
    material: ShadowMaterialBindings<'a>,
}

impl<'a> ShadowDrawContext<'a> {
    pub fn new(
        frame: &'a ShadowFrameBindings,
        object: &'a ShadowObjectBindings,
        material: ShadowMaterialBindings<'a>,
    ) -> Self {
        Self {
            frame,
            object,
            material,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ShadowShader;

impl Shader<ShadowDrawContext<'_>> for ShadowShader {
    type Varying = ShadowVarying;

    fn vertex(
        &self,
        vertex: &Vertex,
        context: ShadowDrawContext<'_>,
    ) -> (Vector4<f32>, Self::Varying) {
        (
            context.frame.light_space_matrix
                * context.object.model_matrix
                * vertex.position.to_homogeneous(),
            ShadowVarying {
                uvs: vertex.texcoords,
            },
        )
    }

    fn fragment(
        &self,
        input: FragmentInput<Self::Varying>,
        context: ShadowDrawContext<'_>,
    ) -> FragmentOutput {
        let varying = input.varying;
        let pbr_material = context.material.material.map(|material| match material {
            Material::Pbr(material) => material,
        });
        let alpha = pbr_material.map_or(1.0, |material| material.alpha);
        let texture_alpha = pbr_material
            .and_then(|material| material.albedo_texture.as_ref())
            .map(|texture| {
                let set = texture.tex_coord.index();
                let uv = varying.uvs[set];
                texture
                    .sample_with_density(uv.x, uv.y, input.uv_density(set))
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
