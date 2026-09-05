use crate::core::geometry::{SUPPORTED_TEXCOORD_SETS, Vertex};
use crate::core::math::transform::TangentFrameTransform;
use crate::core::shader::{FragmentInput, FragmentOutput, Interpolatable, Shader};
use crate::scene::light::Light;
use crate::scene::material::{AlphaMode, Material, PbrMaterial};
use crate::scene::texture::TextureBinding;
use nalgebra::{Matrix3, Matrix4, Point3, Vector2, Vector3, Vector4};
use std::f32::consts::PI;
use std::ops::{Add, Mul};
use thiserror::Error;

fn base_color(material: &PbrMaterial, texel: Option<Vector4<f32>>) -> (Vector3<f32>, f32) {
    let texel = texel.unwrap_or_else(|| Vector4::repeat(1.0));
    (
        texel.xyz().component_mul(&material.albedo),
        texel.w * material.alpha,
    )
}

fn metallic_roughness(material: &PbrMaterial, texel: Option<Vector4<f32>>) -> (f32, f32) {
    let (roughness_sample, metallic_sample) = texel.map_or((1.0, 1.0), |texel| {
        // glTF stores roughness in G and metallic in B.
        (texel.y, texel.z)
    });
    (
        roughness_sample * material.roughness,
        metallic_sample * material.metallic,
    )
}

fn occlusion(material: &PbrMaterial, texel: Option<Vector4<f32>>) -> f32 {
    texel.map_or(1.0, |texel| {
        1.0 + material.occlusion_strength * (texel.x - 1.0)
    })
}

fn emissive(material: &PbrMaterial, texel: Option<Vector4<f32>>) -> Vector3<f32> {
    texel.map_or(material.emissive, |texel| {
        texel.xyz().component_mul(&material.emissive)
    })
}

fn tangent_space_normal(texel: Vector4<f32>, scale: f32) -> Vector3<f32> {
    Vector3::new(
        (texel.x * 2.0 - 1.0) * scale,
        (texel.y * 2.0 - 1.0) * scale,
        texel.z * 2.0 - 1.0,
    )
}

/// Interpolated world-space PBR vertex output.
#[derive(Clone, Copy, Debug)]
pub struct PbrVarying {
    pub world_pos: Point3<f32>,
    pub normal: Vector3<f32>,
    pub uvs: [Vector2<f32>; SUPPORTED_TEXCOORD_SETS],
    pub tangent: Vector4<f32>,
}

impl Add for PbrVarying {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            world_pos: Point3::from(self.world_pos.coords + other.world_pos.coords),
            normal: self.normal + other.normal,
            uvs: std::array::from_fn(|set| self.uvs[set] + other.uvs[set]),
            // Preserve the interpolated handedness sign in W; only XYZ is normalized later.
            tangent: self.tangent + other.tangent,
        }
    }
}

impl Mul<f32> for PbrVarying {
    type Output = Self;
    fn mul(self, scalar: f32) -> Self {
        Self {
            world_pos: Point3::from(self.world_pos.coords * scalar),
            normal: self.normal * scalar,
            uvs: std::array::from_fn(|set| self.uvs[set] * scalar),
            tangent: self.tangent * scalar,
        }
    }
}

impl Interpolatable for PbrVarying {
    fn get_uv(&self, set: usize) -> Option<Vector2<f32>> {
        self.uvs.get(set).copied()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PbrShadowBindingsDescriptor<'a> {
    pub depth: &'a [f32],
    pub size: usize,
    pub light_index: usize,
    pub light_space_matrix: Matrix4<f32>,
    pub constant_bias: f32,
    pub slope_bias: f32,
    pub use_pcf: bool,
    pub pcf_kernel_size: i32,
}

#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum PbrShadowBindingsError {
    #[error("shadow map size {size} does not match its {actual_len}-element depth buffer")]
    InvalidMapDimensions { size: usize, actual_len: usize },
    #[error("shadow light-space matrix must contain only finite values")]
    NonFiniteLightSpaceMatrix,
    #[error("shadow {field} must be finite and non-negative, got {value}")]
    InvalidBias { field: &'static str, value: f32 },
    #[error("shadow PCF kernel size must be non-negative, got {value}")]
    InvalidPcfKernelSize { value: i32 },
}

#[derive(Clone, Copy, Debug)]
pub struct PbrShadowBindings<'a> {
    depth: &'a [f32],
    size: usize,
    light_index: usize,
    light_space_matrix: Matrix4<f32>,
    constant_bias: f32,
    slope_bias: f32,
    use_pcf: bool,
    pcf_kernel_size: i32,
}

impl<'a> PbrShadowBindings<'a> {
    pub fn new(
        descriptor: PbrShadowBindingsDescriptor<'a>,
    ) -> Result<Self, PbrShadowBindingsError> {
        let expected_len = descriptor.size.checked_mul(descriptor.size);
        if descriptor.size == 0 || expected_len != Some(descriptor.depth.len()) {
            return Err(PbrShadowBindingsError::InvalidMapDimensions {
                size: descriptor.size,
                actual_len: descriptor.depth.len(),
            });
        }
        if !descriptor
            .light_space_matrix
            .iter()
            .all(|value| value.is_finite())
        {
            return Err(PbrShadowBindingsError::NonFiniteLightSpaceMatrix);
        }
        for (field, value) in [
            ("constant bias", descriptor.constant_bias),
            ("slope bias", descriptor.slope_bias),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(PbrShadowBindingsError::InvalidBias { field, value });
            }
        }
        if descriptor.pcf_kernel_size < 0 {
            return Err(PbrShadowBindingsError::InvalidPcfKernelSize {
                value: descriptor.pcf_kernel_size,
            });
        }

        Ok(Self {
            depth: descriptor.depth,
            size: descriptor.size,
            light_index: descriptor.light_index,
            light_space_matrix: descriptor.light_space_matrix,
            constant_bias: descriptor.constant_bias,
            slope_bias: descriptor.slope_bias,
            use_pcf: descriptor.use_pcf,
            pcf_kernel_size: descriptor.pcf_kernel_size,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PbrFrameBindings<'a> {
    pub view_matrix: Matrix4<f32>,
    pub projection_matrix: Matrix4<f32>,
    pub camera_pos: Point3<f32>,
    pub lights: &'a [Light],
    pub ambient_light: Vector3<f32>,
    pub shadow: Option<PbrShadowBindings<'a>>,
}

impl<'a> PbrFrameBindings<'a> {
    pub fn new(
        view_matrix: Matrix4<f32>,
        projection_matrix: Matrix4<f32>,
        camera_pos: Point3<f32>,
    ) -> Self {
        Self {
            view_matrix,
            projection_matrix,
            camera_pos,
            lights: &[],
            ambient_light: Vector3::new(0.03, 0.03, 0.03),
            shadow: None,
        }
    }
}

#[derive(Clone, Copy)]
pub struct PbrObjectBindings {
    model_matrix: Matrix4<f32>,
    tangent_frame_transform: TangentFrameTransform,
}

impl PbrObjectBindings {
    pub fn new(model_matrix: Matrix4<f32>) -> Self {
        let model_3x3 = model_matrix.fixed_view::<3, 3>(0, 0).into_owned();
        Self::new_with_tangent_frame_transform(model_matrix, TangentFrameTransform::new(model_3x3))
    }

    /// Returns the immutable object transform used to derive this binding's tangent frame.
    pub fn model_matrix(&self) -> Matrix4<f32> {
        self.model_matrix
    }

    pub(crate) fn new_with_tangent_frame_transform(
        model_matrix: Matrix4<f32>,
        tangent_frame_transform: TangentFrameTransform,
    ) -> Self {
        Self {
            model_matrix,
            tangent_frame_transform,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PbrMaterialBindings<'a> {
    material: &'a PbrMaterial,
}

impl<'a> PbrMaterialBindings<'a> {
    pub fn new(material: Option<&'a Material>, fallback: &'a PbrMaterial) -> Self {
        Self {
            material: material.map_or(fallback, |material| match material {
                Material::Pbr(material) => material,
            }),
        }
    }

    pub fn from_pbr(material: &'a PbrMaterial) -> Self {
        Self { material }
    }
}

#[derive(Clone, Copy)]
pub struct PbrDrawContext<'a> {
    frame: &'a PbrFrameBindings<'a>,
    object: &'a PbrObjectBindings,
    material: PbrMaterialBindings<'a>,
}

impl<'a> PbrDrawContext<'a> {
    pub fn new(
        frame: &'a PbrFrameBindings<'a>,
        object: &'a PbrObjectBindings,
        material: PbrMaterialBindings<'a>,
    ) -> Self {
        Self {
            frame,
            object,
            material,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PbrShader;

impl PbrShader {
    fn calculate_shadow(
        frame: &PbrFrameBindings<'_>,
        world_pos: &Point3<f32>,
        n_dot_l: f32,
    ) -> f32 {
        let Some(shadow) = frame.shadow else {
            return 1.0;
        };

        let light_space_pos = shadow.light_space_matrix * world_pos.to_homogeneous();
        let proj_coords = light_space_pos.xyz() / light_space_pos.w;
        let u = proj_coords.x * 0.5 + 0.5;
        let v = 1.0 - (proj_coords.y * 0.5 + 0.5);
        // Clip-space Z uses [-1, 1], while the shadow depth buffer uses [0, 1].
        let current_depth = proj_coords.z * 0.5 + 0.5;

        if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) || current_depth > 1.0 {
            return 1.0;
        }

        let bias = Self::shadow_bias(shadow, n_dot_l);

        if !shadow.use_pcf {
            let map_x =
                (u * (shadow.size - 1) as f32).clamp(0.0, (shadow.size - 1) as f32) as usize;
            let map_y =
                (v * (shadow.size - 1) as f32).clamp(0.0, (shadow.size - 1) as f32) as usize;
            let index = map_y * shadow.size + map_x;
            return if current_depth - bias > shadow.depth[index] {
                0.0
            } else {
                1.0
            };
        }

        let mut visibility = 0.0;
        let texel_size = 1.0 / shadow.size as f32;
        let kernel_size = shadow.pcf_kernel_size;

        for x in -kernel_size..=kernel_size {
            for y in -kernel_size..=kernel_size {
                let pcf_u = u + x as f32 * texel_size;
                let pcf_v = v + y as f32 * texel_size;

                if !(0.0..=1.0).contains(&pcf_u) || !(0.0..=1.0).contains(&pcf_v) {
                    visibility += 1.0;
                    continue;
                }

                let map_x = (pcf_u * (shadow.size - 1) as f32) as usize;
                let map_y = (pcf_v * (shadow.size - 1) as f32) as usize;
                let index = map_y * shadow.size + map_x;

                let pcf_depth = shadow.depth[index];
                visibility += if current_depth - bias > pcf_depth {
                    0.0
                } else {
                    1.0
                };
            }
        }

        visibility / ((kernel_size * 2 + 1_i32).pow(2) as f32)
    }

    fn shadow_bias(shadow: PbrShadowBindings<'_>, n_dot_l: f32) -> f32 {
        shadow.constant_bias + shadow.slope_bias * (1.0 - n_dot_l.clamp(0.0, 1.0))
    }
    fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
        let a = roughness * roughness;
        let a2 = a * a;
        let n_dot_h2 = n_dot_h * n_dot_h;

        let num = a2;
        let denom = n_dot_h2 * (a2 - 1.0) + 1.0;
        let denom = PI * denom * denom;

        num / denom.max(0.0001)
    }

    fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
        let r = roughness + 1.0;
        let k = (r * r) / 8.0;

        let num = n_dot_v;
        let denom = n_dot_v * (1.0 - k) + k;

        num / denom.max(0.0001)
    }

    fn geometry_smith(n: &Vector3<f32>, v: &Vector3<f32>, l: &Vector3<f32>, roughness: f32) -> f32 {
        let n_dot_v = n.dot(v).max(0.0);
        let n_dot_l = n.dot(l).max(0.0);
        let ggx2 = PbrShader::geometry_schlick_ggx(n_dot_v, roughness);
        let ggx1 = PbrShader::geometry_schlick_ggx(n_dot_l, roughness);

        ggx1 * ggx2
    }

    fn fresnel_schlick(cos_theta: f32, f0: Vector3<f32>) -> Vector3<f32> {
        let val = (1.0 - cos_theta).clamp(0.0, 1.0).powi(5);
        f0 + (Vector3::new(1.0, 1.0, 1.0) - f0) * val
    }
}

impl Shader<PbrDrawContext<'_>> for PbrShader {
    type Varying = PbrVarying;

    fn vertex(
        &self,
        vertex: &Vertex,
        context: PbrDrawContext<'_>,
    ) -> (Vector4<f32>, Self::Varying) {
        let world_pos = Point3::from_homogeneous(
            context.object.model_matrix * vertex.position.to_homogeneous(),
        )
        .unwrap();
        let (world_normal, world_tangent) = context
            .object
            .tangent_frame_transform
            .transform(vertex.normal, vertex.tangent);
        let clip_pos = context.frame.projection_matrix
            * context.frame.view_matrix
            * context.object.model_matrix
            * vertex.position.to_homogeneous();

        (
            clip_pos,
            PbrVarying {
                world_pos,
                normal: world_normal,
                uvs: vertex.texcoords,
                tangent: world_tangent,
            },
        )
    }

    fn fragment(
        &self,
        input: FragmentInput<Self::Varying>,
        context: PbrDrawContext<'_>,
    ) -> FragmentOutput {
        let varying = input.varying;
        let mat = context.material.material;
        let sample_texture = |binding: &TextureBinding| {
            let set = binding.tex_coord.index();
            let uv = varying.uvs[set];
            binding.sample_with_density(uv.x, uv.y, input.uv_density(set))
        };

        let (albedo, alpha) = base_color(mat, mat.albedo_texture.as_ref().map(sample_texture));

        if matches!(mat.alpha_mode, AlphaMode::Mask(cutoff) if alpha < cutoff) {
            return FragmentOutput::Discard;
        }

        let (roughness, metallic) = metallic_roughness(
            mat,
            mat.metallic_roughness_texture.as_ref().map(sample_texture),
        );

        let ao = occlusion(mat, mat.ao_texture.as_ref().map(sample_texture));

        let emissive_color = emissive(mat, mat.emissive_texture.as_ref().map(sample_texture));

        let geom_normal = if input.front_facing {
            varying.normal.normalize()
        } else {
            -varying.normal.normalize()
        };

        let n = if let Some(normal_map) = &mat.normal_texture {
            if varying.tangent.xyz().norm_squared() > 1e-6 {
                let geom_tangent = varying.tangent.xyz().normalize();
                let tangent_sign = varying.tangent.w;
                let t = (geom_tangent - geom_normal * geom_normal.dot(&geom_tangent)).normalize();
                // glTF stores the bitangent handedness in tangent.w.
                let b = geom_normal.cross(&t) * tangent_sign;

                let tbn = Matrix3::from_columns(&[t, b, geom_normal]);

                let local_normal =
                    tangent_space_normal(sample_texture(normal_map), mat.normal_scale);

                (tbn * local_normal).normalize()
            } else {
                geom_normal
            }
        } else {
            geom_normal
        };

        let v = (context.frame.camera_pos - varying.world_pos).normalize();

        // Dielectrics use F0=0.04; metals derive F0 from their base color.
        let f0 = Vector3::new(0.04, 0.04, 0.04).lerp(&albedo, metallic);
        let mut lo = Vector3::zeros();

        for (i, light) in context.frame.lights.iter().enumerate() {
            let l = light.get_direction_to_light(&varying.world_pos);
            let h = (v + l).normalize();

            let radiance = light.get_intensity(&varying.world_pos);
            let n_dot_v = n.dot(&v).max(0.0);
            let n_dot_l = n.dot(&l).max(0.0);
            let n_dot_h = n.dot(&h).max(0.0);
            let h_dot_v = h.dot(&v).max(0.0);
            let shadow = if context
                .frame
                .shadow
                .is_some_and(|shadow| shadow.light_index == i)
            {
                Self::calculate_shadow(context.frame, &varying.world_pos, n_dot_l)
            } else {
                1.0
            };

            let d = Self::distribution_ggx(n_dot_h, roughness);
            let g = Self::geometry_smith(&n, &v, &l, roughness);
            let f = Self::fresnel_schlick(h_dot_v, f0);

            let numerator = f * d * g;
            let denominator = 4.0 * n_dot_v * n_dot_l + 0.0001;
            let specular = numerator / denominator;

            let k_s = f;
            // Metals have no diffuse term; dielectrics retain non-reflected energy.
            let k_d = (Vector3::new(1.0, 1.0, 1.0) - k_s) * (1.0 - metallic);
            let diffuse = k_d.component_mul(&albedo) / PI;
            let brdf = diffuse + specular;
            let light_contribution = brdf.component_mul(&radiance) * n_dot_l * shadow;

            lo += light_contribution;
        }

        let ambient = context.frame.ambient_light.component_mul(&albedo) * ao;

        let final_color = ambient + lo + emissive_color;
        FragmentOutput::Color(Vector4::new(
            final_color.x,
            final_color.y,
            final_color.z,
            alpha,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_vec3_approx(actual: Vector3<f32>, expected: Vector3<f32>) {
        assert!(
            (actual - expected).abs().max() < 1.0e-6,
            "expected {expected:?}, got {actual:?}"
        );
    }

    #[test]
    fn texture_channels_are_multiplied_by_material_factors() {
        let material = PbrMaterial {
            albedo: Vector3::new(0.5, 0.25, 0.75),
            alpha: 0.4,
            metallic: 0.8,
            roughness: 0.6,
            emissive: Vector3::new(0.2, 0.4, 0.6),
            ..Default::default()
        };

        let (albedo, alpha) = base_color(&material, Some(Vector4::new(0.8, 0.4, 0.2, 0.5)));
        let (roughness, metallic) =
            metallic_roughness(&material, Some(Vector4::new(0.0, 0.5, 0.25, 1.0)));
        let emissive = emissive(&material, Some(Vector4::new(0.5, 0.25, 1.0, 1.0)));

        assert_vec3_approx(albedo, Vector3::new(0.4, 0.1, 0.15));
        assert!((alpha - 0.2).abs() < 1.0e-6);
        assert!((roughness - 0.3).abs() < 1.0e-6);
        assert!((metallic - 0.2).abs() < 1.0e-6);
        assert_vec3_approx(emissive, Vector3::new(0.1, 0.1, 0.6));
    }

    #[test]
    fn occlusion_strength_blends_between_unoccluded_and_sampled_values() {
        let material = PbrMaterial {
            occlusion_strength: 0.25,
            ..Default::default()
        };

        assert!(
            (occlusion(&material, Some(Vector4::new(0.2, 0.0, 0.0, 1.0))) - 0.8).abs() < 1.0e-6
        );
        assert_eq!(occlusion(&material, None), 1.0);
    }

    #[test]
    fn normal_scale_affects_only_tangent_space_xy() {
        let texel = Vector4::new(0.75, 0.25, 1.0, 1.0);

        assert_vec3_approx(
            tangent_space_normal(texel, 0.5),
            Vector3::new(0.25, -0.25, 1.0),
        );
    }

    #[test]
    fn shadow_bias_separates_constant_and_slope_terms() {
        let shadow_map = [0.0];
        let shadow = PbrShadowBindings::new(PbrShadowBindingsDescriptor {
            depth: &shadow_map,
            size: 1,
            light_index: 0,
            light_space_matrix: Matrix4::identity(),
            constant_bias: 0.001,
            slope_bias: 0.01,
            use_pcf: false,
            pcf_kernel_size: 0,
        })
        .expect("valid shadow bindings");

        assert!((PbrShader::shadow_bias(shadow, 1.0) - 0.001).abs() < 1.0e-6);
        assert!((PbrShader::shadow_bias(shadow, 0.5) - 0.006).abs() < 1.0e-6);
        assert!((PbrShader::shadow_bias(shadow, 0.0) - 0.011).abs() < 1.0e-6);
    }

    #[test]
    fn pcf_uses_lit_border_for_out_of_bounds_taps() {
        let shadow_map = vec![0.0; 9];
        let mut frame =
            PbrFrameBindings::new(Matrix4::identity(), Matrix4::identity(), Point3::origin());
        frame.shadow = Some(
            PbrShadowBindings::new(PbrShadowBindingsDescriptor {
                depth: &shadow_map,
                size: 3,
                light_index: 0,
                light_space_matrix: Matrix4::identity(),
                constant_bias: 0.0,
                slope_bias: 0.0,
                use_pcf: true,
                pcf_kernel_size: 1,
            })
            .expect("valid shadow bindings"),
        );

        let visibility = PbrShader::calculate_shadow(&frame, &Point3::new(-1.0, 1.0, 0.0), 1.0);
        assert!((visibility - 5.0 / 9.0).abs() < 1.0e-6);
    }

    #[test]
    fn shadow_bindings_reject_invalid_resources_and_parameters() {
        let shadow_map = [0.0; 3];
        let descriptor = PbrShadowBindingsDescriptor {
            depth: &shadow_map,
            size: 2,
            light_index: 0,
            light_space_matrix: Matrix4::identity(),
            constant_bias: 0.0,
            slope_bias: 0.0,
            use_pcf: false,
            pcf_kernel_size: 0,
        };

        assert!(matches!(
            PbrShadowBindings::new(descriptor),
            Err(PbrShadowBindingsError::InvalidMapDimensions {
                size: 2,
                actual_len: 3,
            })
        ));

        let valid_map = [0.0];
        let valid = PbrShadowBindingsDescriptor {
            depth: &valid_map,
            size: 1,
            ..descriptor
        };
        assert!(matches!(
            PbrShadowBindings::new(PbrShadowBindingsDescriptor {
                light_space_matrix: Matrix4::repeat(f32::NAN),
                ..valid
            }),
            Err(PbrShadowBindingsError::NonFiniteLightSpaceMatrix)
        ));
        assert!(matches!(
            PbrShadowBindings::new(PbrShadowBindingsDescriptor {
                constant_bias: -0.001,
                ..valid
            }),
            Err(PbrShadowBindingsError::InvalidBias {
                field: "constant bias",
                ..
            })
        ));
        assert!(matches!(
            PbrShadowBindings::new(PbrShadowBindingsDescriptor {
                pcf_kernel_size: -1,
                ..valid
            }),
            Err(PbrShadowBindingsError::InvalidPcfKernelSize { value: -1 })
        ));
    }
}
