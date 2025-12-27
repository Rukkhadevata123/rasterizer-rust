use crate::core::geometry::Vertex;
use crate::core::pipeline::Shader;
use crate::scene::light::Light;
use crate::scene::material::{Material, PbrMaterial};
use nalgebra::{Matrix3, Matrix4, Point3, Vector2, Vector3, Vector4};
use std::f32::consts::PI;
use std::ops::{Add, Mul};
use std::sync::Arc;

/// Data passed from Vertex Shader to Fragment Shader.
#[derive(Clone, Copy, Debug)]
pub struct PbrVarying {
    pub world_pos: Point3<f32>,
    pub normal: Vector3<f32>,
    pub uv: Vector2<f32>,
    pub tangent: Vector3<f32>,
}

impl Add for PbrVarying {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            world_pos: Point3::from(self.world_pos.coords + other.world_pos.coords),
            normal: self.normal + other.normal,
            uv: self.uv + other.uv,
            tangent: self.tangent + other.tangent, // Standard linear interpolation
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
            tangent: self.tangent * scalar,
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

    // Added: Ambient Light Control
    pub ambient_light: Vector3<f32>,

    // Shadow Mapping Fields
    pub shadow_map: Option<Arc<Vec<f32>>>,
    pub shadow_map_size: usize,
    pub light_space_matrix: Matrix4<f32>,
    pub shadow_bias: f32,
    pub use_pcf: bool,
    pub pcf_kernel_size: i32,

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
            ambient_light: Vector3::new(0.03, 0.03, 0.03), // Default low ambient
            shadow_map: None,
            shadow_map_size: 0,
            light_space_matrix: Matrix4::identity(),
            shadow_bias: 0.005,
            use_pcf: true,
            pcf_kernel_size: 1,
            fallback_material: PbrMaterial::default(),
        }
    }

    // --- Shadow Calculation ---
    fn calculate_shadow(&self, world_pos: &Point3<f32>, n_dot_l: f32) -> f32 {
        if self.shadow_map.is_none() {
            return 1.0;
        }
        let shadow_map = self.shadow_map.as_ref().unwrap();

        // 1. Transform world position to light space
        let light_space_pos = self.light_space_matrix * world_pos.to_homogeneous();

        // 2. Perspective divide
        let proj_coords = light_space_pos.xyz() / light_space_pos.w;

        // 3. Transform to [0, 1] range
        let u = proj_coords.x * 0.5 + 0.5;
        let v = 1.0 - (proj_coords.y * 0.5 + 0.5); // Flip Y

        // FIX: Remap Z from [-1, 1] to [0, 1] to match depth buffer
        let current_depth = proj_coords.z * 0.5 + 0.5;

        // Check if outside shadow map
        if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) || current_depth > 1.0 {
            return 1.0;
        }

        // Adaptive bias based on surface angle
        let bias = self.shadow_bias.max(0.05 * (1.0 - n_dot_l));

        if !self.use_pcf {
            let map_x = (u * (self.shadow_map_size - 1) as f32)
                .clamp(0.0, (self.shadow_map_size - 1) as f32) as usize;
            let map_y = (v * (self.shadow_map_size - 1) as f32)
                .clamp(0.0, (self.shadow_map_size - 1) as f32) as usize;
            let index = map_y * self.shadow_map_size + map_x;
            return if current_depth - bias > shadow_map[index] {
                0.0
            } else {
                1.0
            };
        }

        // PCF (Percentage Closer Filtering) for soft shadows
        let mut shadow = 0.0;
        let texel_size = 1.0 / self.shadow_map_size as f32;
        let kernel_size = self.pcf_kernel_size;

        for x in -kernel_size..=kernel_size {
            for y in -kernel_size..=kernel_size {
                let pcf_u = u + x as f32 * texel_size;
                let pcf_v = v + y as f32 * texel_size;

                // Clamp coordinates
                let map_x = (pcf_u * (self.shadow_map_size - 1) as f32)
                    .clamp(0.0, (self.shadow_map_size - 1) as f32)
                    as usize;
                let map_y = (pcf_v * (self.shadow_map_size - 1) as f32)
                    .clamp(0.0, (self.shadow_map_size - 1) as f32)
                    as usize;
                let index = map_y * self.shadow_map_size + map_x;

                let pcf_depth = shadow_map[index];
                // Use the remapped current_depth here
                shadow += if current_depth - bias > pcf_depth {
                    0.0
                } else {
                    1.0
                };
            }
        }

        shadow / ((kernel_size * 2 + 1_i32).pow(2) as f32)
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
        let ggx2 = PbrShader::geometry_schlick_ggx(n_dot_v, roughness);
        let ggx1 = PbrShader::geometry_schlick_ggx(n_dot_l, roughness);

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
        let world_tangent = (self.normal_matrix * vertex.tangent).normalize(); // Transform tangent to world space
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
                tangent: world_tangent,
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
            tex.sample_color(varying.uv.x, varying.uv.y)
        } else {
            mat.albedo
        };

        // Metallic/Roughness uses sample_data (no Gamma correction)
        // Standard glTF packing: Green = Roughness, Blue = Metallic
        let (roughness, metallic) = if let Some(tex) = &mat.metallic_roughness_texture {
            let sample = tex.sample_data(varying.uv.x, varying.uv.y);
            // Multiply by the uniform factor
            (sample.y * mat.roughness, sample.z * mat.metallic)
        } else {
            (mat.roughness, mat.metallic)
        };

        let ao = mat.ao;

        // 2. Calculate Normal (Normal Mapping)
        let geom_normal = varying.normal.normalize();

        // Use normal map if available, otherwise fallback to geometry normal
        let n = if let Some(normal_map) = &mat.normal_texture {
            // Check for valid tangent (avoid NaN if tangent is zero, e.g. no UVs)
            if varying.tangent.norm_squared() > 1e-6 {
                let geom_tangent = varying.tangent.normalize();

                // 2.1 Re-orthogonalize Tangent (Gram-Schmidt)
                // Interpolation can denormalize vectors. This ensures T is perpendicular to N.
                // T = T - N * (N . T)
                let t = (geom_tangent - geom_normal * geom_normal.dot(&geom_tangent)).normalize();

                // 2.2 Calculate Bitangent (B = N x T)
                // Assumes right-handed coordinates.
                let b = geom_normal.cross(&t).normalize();

                // 2.3 Construct TBN Matrix
                // Transforms from Tangent Space to World Space
                let tbn = Matrix3::from_columns(&[t, b, geom_normal]);

                // 2.4 Sample Normal Map
                // MUST use sample_data (Linear) because normals are vectors, not colors.
                let packed_normal = normal_map.sample_data(varying.uv.x, varying.uv.y);

                // 2.5 Decode [0, 1] range to [-1, 1] range
                let local_normal = Vector3::new(
                    packed_normal.x * 2.0 - 1.0,
                    -packed_normal.y * 2.0 + 1.0, // Flip Y
                    packed_normal.z * 2.0 - 1.0,
                );

                // 2.6 Transform to World Space
                (tbn * local_normal).normalize()
            } else {
                geom_normal
            }
        } else {
            geom_normal
        };

        let v = (self.camera_pos - varying.world_pos).normalize();

        // F0: Surface reflection at zero incidence
        // 0.04 for dielectrics, albedo for metals
        let f0 = Vector3::new(0.04, 0.04, 0.04).lerp(&albedo, metallic);

        // 3. Lighting Loop
        let mut lo = Vector3::zeros();

        for (i, light) in self.lights.iter().enumerate() {
            let l = light.get_direction_to_light(&varying.world_pos);
            let h = (v + l).normalize();

            // Radiance (Light Color * Intensity)
            let radiance = light.get_intensity(&varying.world_pos);

            // Cook-Torrance BRDF
            let n_dot_v = n.dot(&v).max(0.0);
            let n_dot_l = n.dot(&l).max(0.0);
            let n_dot_h = n.dot(&h).max(0.0);
            let h_dot_v = h.dot(&v).max(0.0);

            // --- SHADOW CALCULATION ---
            // Only apply shadow for the first light (assuming it's the main directional light)
            let shadow = if i == 0 {
                self.calculate_shadow(&varying.world_pos, n_dot_l)
            } else {
                1.0
            };

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

            // Add to outgoing radiance Lo,
            // Lo += (kD * albedo / PI + specular) * radiance * NdotL
            let brdf = diffuse + specular;
            let light_contribution = brdf.component_mul(&radiance) * n_dot_l * shadow;

            lo += light_contribution;
        }

        // 4. Ambient (Using configurable ambient_light)
        let ambient = self.ambient_light.component_mul(&albedo) * ao;

        ambient + lo + mat.emissive
    }
}
