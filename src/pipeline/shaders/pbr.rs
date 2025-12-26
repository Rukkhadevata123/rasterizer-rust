use crate::core::geometry::Vertex;
use crate::core::pipeline::Shader;
use crate::scene::light::Light;
use crate::scene::material::{Material, PbrMaterial};
use nalgebra::{Matrix3, Matrix4, Point3, Vector2, Vector3, Vector4};
use std::f32::consts::PI;
use std::ops::{Add, Mul};

// --- Varying Data ---
#[derive(Clone, Copy, Debug)]
pub struct PbrVarying {
    pub world_pos: Point3<f32>,
    pub normal: Vector3<f32>,
    pub uv: Vector2<f32>,
}

impl Add for PbrVarying {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            world_pos: Point3::from(self.world_pos.coords + other.world_pos.coords),
            normal: self.normal + other.normal,
            uv: self.uv + other.uv,
        }
    }
}

impl Mul<f32> for PbrVarying {
    type Output = Self;
    fn mul(self, scalar: f32) -> Self {
        Self {
            world_pos: Point3::from(self.world_pos.coords * scalar),
            normal: self.normal * scalar,
            uv: self.uv * scalar,
        }
    }
}

// --- PBR Shader ---
pub struct PbrShader {
    pub model_matrix: Matrix4<f32>,
    pub view_matrix: Matrix4<f32>,
    pub projection_matrix: Matrix4<f32>,
    pub normal_matrix: Matrix3<f32>,

    pub camera_pos: Point3<f32>,
    pub lights: Vec<Light>,

    // Fallback if material is missing or wrong type
    pub fallback_material: PbrMaterial,
}

impl PbrShader {
    pub fn new(
        model: Matrix4<f32>,
        view: Matrix4<f32>,
        projection: Matrix4<f32>,
        camera_pos: Point3<f32>,
    ) -> Self {
        let model_3x3 = model.fixed_view::<3, 3>(0, 0).into_owned();
        let normal_matrix = model_3x3.try_inverse().unwrap_or(model_3x3).transpose();

        Self {
            model_matrix: model,
            view_matrix: view,
            projection_matrix: projection,
            normal_matrix,
            camera_pos,
            lights: Vec::new(),
            fallback_material: PbrMaterial::default(),
        }
    }

    // --- PBR Helper Functions ---

    // Normal Distribution Function (GGX)
    fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
        let a = roughness * roughness;
        let a2 = a * a;
        let n_dot_h2 = n_dot_h * n_dot_h;

        let num = a2;
        let denom = n_dot_h2 * (a2 - 1.0) + 1.0;
        let denom = PI * denom * denom;

        num / denom.max(0.0001)
    }

    // Geometry Function (Smith's Schlick-GGX)
    fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
        let r = roughness + 1.0;
        let k = (r * r) / 8.0; // Direct light

        let num = n_dot_v;
        let denom = n_dot_v * (1.0 - k) + k;

        num / denom.max(0.0001)
    }

    fn geometry_smith(n: &Vector3<f32>, v: &Vector3<f32>, l: &Vector3<f32>, roughness: f32) -> f32 {
        let n_dot_v = n.dot(v).max(0.0);
        let n_dot_l = n.dot(l).max(0.0);
        let ggx2 = self::PbrShader::geometry_schlick_ggx(n_dot_v, roughness);
        let ggx1 = self::PbrShader::geometry_schlick_ggx(n_dot_l, roughness);

        ggx1 * ggx2
    }

    // Fresnel Equation (Fresnel-Schlick)
    fn fresnel_schlick(cos_theta: f32, f0: Vector3<f32>) -> Vector3<f32> {
        let val = (1.0 - cos_theta).clamp(0.0, 1.0).powi(5);
        f0 + (Vector3::new(1.0, 1.0, 1.0) - f0) * val
    }
}

impl Shader for PbrShader {
    type Varying = PbrVarying;

    fn vertex(&self, vertex: &Vertex) -> (Vector4<f32>, Self::Varying) {
        let world_pos =
            Point3::from_homogeneous(self.model_matrix * vertex.position.to_homogeneous()).unwrap();
        let world_normal = (self.normal_matrix * vertex.normal).normalize();
        let clip_pos = self.projection_matrix
            * self.view_matrix
            * self.model_matrix
            * vertex.position.to_homogeneous();

        (
            clip_pos,
            PbrVarying {
                world_pos,
                normal: world_normal,
                uv: vertex.texcoord,
            },
        )
    }

    fn fragment(&self, varying: Self::Varying, material: Option<&Material>) -> Vector3<f32> {
        // 1. Retrieve Material Properties
        let mat = if let Some(Material::Pbr(m)) = material {
            m
        } else {
            &self.fallback_material
        };

        let albedo = if let Some(tex) = &mat.albedo_texture {
            tex.sample(varying.uv.x, varying.uv.y)
        } else {
            mat.albedo
        };

        // TODO: Future: Sample roughness/metallic from textures if available
        let roughness = mat.roughness;
        let metallic = mat.metallic;
        let ao = mat.ao;

        let n = varying.normal.normalize();
        let v = (self.camera_pos - varying.world_pos).normalize();

        // F0: Surface reflection at zero incidence
        // 0.04 for dielectrics, albedo for metals
        let f0 = Vector3::new(0.04, 0.04, 0.04).lerp(&albedo, metallic);

        // 2. Lighting Loop
        let mut lo = Vector3::zeros();

        for light in &self.lights {
            let l = light.get_direction_to_light(&varying.world_pos);
            let h = (v + l).normalize();

            // Radiance (Light Color * Intensity)
            let radiance = light.get_intensity(&varying.world_pos);

            // Cook-Torrance BRDF
            let n_dot_v = n.dot(&v).max(0.0);
            let n_dot_l = n.dot(&l).max(0.0);
            let n_dot_h = n.dot(&h).max(0.0);
            let h_dot_v = h.dot(&v).max(0.0);

            let d = Self::distribution_ggx(n_dot_h, roughness);
            let g = Self::geometry_smith(&n, &v, &l, roughness);
            let f = Self::fresnel_schlick(h_dot_v, f0);

            let numerator = f * d * g; // f is Vector3, d and g are f32
            let denominator = 4.0 * n_dot_v * n_dot_l + 0.0001;
            let specular = numerator / denominator;

            // kS is Fresnel (F)
            let k_s = f;
            // kD is the remaining energy (1 - kS), multiplied by (1 - metallic)
            // because metals absorb all refracted light.
            let k_d = (Vector3::new(1.0, 1.0, 1.0) - k_s) * (1.0 - metallic);

            // Lambertian Diffuse
            let diffuse = k_d.component_mul(&albedo) / PI;

            // Add to outgoing radiance Lo
            // Lo += (kD * albedo / PI + specular) * radiance * NdotL
            let brdf = diffuse + specular;
            let light_contribution = brdf.component_mul(&radiance) * n_dot_l;

            lo += light_contribution;
        }

        // 3. Ambient (Simplified IBL placeholder)
        // TODO: IBL ?
        // In a real PBR engine, this would come from an irradiance map
        let ambient = Vector3::new(0.03, 0.03, 0.03).component_mul(&albedo) * ao;

        let color = ambient + lo + mat.emissive;

        // Note: Tone mapping is done in post-processing (main.rs), not here.
        // We return linear HDR color.
        color
    }
}
