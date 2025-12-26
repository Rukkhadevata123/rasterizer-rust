use crate::core::geometry::Vertex;
use crate::core::pipeline::Shader;
use crate::scene::light::Light;
use crate::scene::material::{Material, PhongMaterial};
use nalgebra::{Matrix3, Matrix4, Point3, Vector2, Vector3, Vector4};
use std::ops::{Add, Mul};
use std::sync::Arc;

/// Data that needs to be interpolated across the triangle surface.
#[derive(Clone, Copy, Debug)]
pub struct PhongVarying {
    pub normal: Vector3<f32>,
    pub world_pos: Point3<f32>,
    pub uv: Vector2<f32>,
}

impl Add for PhongVarying {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            normal: self.normal + other.normal,
            world_pos: Point3::from(self.world_pos.coords + other.world_pos.coords),
            uv: self.uv + other.uv,
        }
    }
}

impl Mul<f32> for PhongVarying {
    type Output = Self;
    fn mul(self, scalar: f32) -> Self {
        Self {
            normal: self.normal * scalar,
            world_pos: Point3::from(self.world_pos.coords * scalar),
            uv: self.uv * scalar,
        }
    }
}

/// A Blinn-Phong lighting shader supporting multiple lights.
pub struct PhongShader {
    // Matrices
    pub model_matrix: Matrix4<f32>,
    pub view_matrix: Matrix4<f32>,
    pub projection_matrix: Matrix4<f32>,
    pub normal_matrix: Matrix3<f32>,

    // Lighting
    pub lights: Vec<Light>,
    pub ambient_light: Vector3<f32>, // Global ambient color * intensity

    // Camera
    pub camera_pos: Point3<f32>,

    // Shadow Mapping Fields
    pub shadow_map: Option<Arc<Vec<f32>>>,
    pub shadow_map_size: usize,
    pub light_space_matrix: Matrix4<f32>,
    pub shadow_bias: f32,

    // Fallback Material
    pub fallback_material: PhongMaterial,
}

impl PhongShader {
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
            lights: Vec::new(), // Initialize empty
            ambient_light: Vector3::new(0.1, 0.1, 0.1),
            shadow_map: None,
            shadow_map_size: 0,
            light_space_matrix: Matrix4::identity(),
            shadow_bias: 0.005,
            fallback_material: PhongMaterial::default(),
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

        // FIX: Remap Z from [-1, 1] to [0, 1]
        let current_depth = proj_coords.z * 0.5 + 0.5;

        // Check if outside shadow map
        if u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0 || current_depth > 1.0 {
            return 1.0;
        }

        // Adaptive bias based on surface angle
        let bias = self.shadow_bias.max(0.05 * (1.0 - n_dot_l));

        // PCF (Percentage Closer Filtering) for soft shadows
        let mut shadow = 0.0;
        let texel_size = 1.0 / self.shadow_map_size as f32;
        let kernel_size = 1; // 3x3 kernel

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

        shadow / ((kernel_size * 2 + 1 as i32).pow(2) as f32)
    }
}

impl Shader for PhongShader {
    type Varying = PhongVarying;

    fn vertex(&self, vertex: &Vertex) -> (Vector4<f32>, Self::Varying) {
        // 1. Transform Position to World Space
        let world_pos_homo = self.model_matrix * vertex.position.to_homogeneous();
        let world_pos = Point3::from_homogeneous(world_pos_homo).unwrap();

        // 2. Transform Normal to World Space
        // Use the precomputed Normal Matrix for correct non-uniform scaling handling
        let world_normal = (self.normal_matrix * vertex.normal).normalize();

        // 3. Transform Position to Clip Space (MVP)
        let mvp = self.projection_matrix * self.view_matrix * self.model_matrix;
        let clip_pos = mvp * vertex.position.to_homogeneous();

        (
            clip_pos,
            PhongVarying {
                normal: world_normal,
                world_pos,
                uv: vertex.texcoord,
            },
        )
    }

    fn fragment(&self, varying: Self::Varying, material: Option<&Material>) -> Vector3<f32> {
        // 1. Determine Material Properties
        let mat_props = if let Some(Material::Phong(m)) = material {
            m
        } else {
            &self.fallback_material
        };

        // 2. Sample Texture or use Color
        let diffuse_color = if let Some(texture) = &mat_props.diffuse_texture {
            texture.sample(varying.uv.x, varying.uv.y)
        } else {
            mat_props.diffuse_color
        };

        // 3. Prepare Vectors
        let normal = varying.normal.normalize();
        let view_dir = (self.camera_pos - varying.world_pos).normalize();

        // 4. Calculate Lighting (Accumulate from all lights)
        let mut total_diffuse = Vector3::zeros();
        let mut total_specular = Vector3::zeros();

        for (i, light) in self.lights.iter().enumerate() {
            // Get direction and intensity (radiance) from the light source
            let light_dir = light.get_direction_to_light(&varying.world_pos);
            let light_intensity = light.get_intensity(&varying.world_pos);

            let n_dot_l = normal.dot(&light_dir).max(0.0);

            // --- SHADOW CALCULATION ---
            // Only apply shadow for the first light (assuming it's the main directional light)
            let shadow = if i == 0 {
                self.calculate_shadow(&varying.world_pos, n_dot_l)
            } else {
                1.0
            };

            // Diffuse (Lambertian)
            let diffuse_term = light_intensity.component_mul(&diffuse_color) * n_dot_l * shadow;
            total_diffuse += diffuse_term;

            // Specular (Blinn-Phong)
            if n_dot_l > 0.0 {
                let halfway_dir = (light_dir + view_dir).normalize();
                let n_dot_h = normal.dot(&halfway_dir).max(0.0);
                let spec_factor = n_dot_h.powf(mat_props.shininess);

                let specular_term =
                    light_intensity.component_mul(&mat_props.specular_color) * spec_factor * shadow;
                total_specular += specular_term;
            }
        }

        // 5. Ambient
        let ambient = self.ambient_light.component_mul(&mat_props.ambient_color);

        // 6. Combine
        let result = ambient + total_diffuse + total_specular;

        // Simple Tone Mapping / Clamping (Usually done in post-process, but Phong often clamps here)
        // Since we use ACES in main.rs, we can return HDR values here too.
        // TODO: Check this
        result
    }
}
