use crate::core::geometry::Vertex;
use crate::core::pipeline::Shader;
use nalgebra::{Matrix4, Point3, Vector2, Vector3, Vector4};
use std::ops::{Add, Mul};

/// Data that needs to be interpolated across the triangle surface.
/// Passed from Vertex Shader -> Rasterizer -> Fragment Shader.
#[derive(Clone, Copy, Debug)]
pub struct PhongVarying {
    /// Normal vector in World Space.
    pub normal: Vector3<f32>,
    /// Position in World Space (needed for calculating View Vector and Light Vector).
    pub world_pos: Point3<f32>,
    /// Texture coordinates (UV).
    pub uv: Vector2<f32>,
}

// Implement math operations required for barycentric interpolation.
// Note: nalgebra's Point3 doesn't support addition with Point3 directly,
// so we handle it via coordinates.
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

/// A standard Phong lighting shader.
/// Calculates Ambient + Diffuse + Specular components.
pub struct PhongShader {
    // --- Transformation Matrices ---
    pub model_matrix: Matrix4<f32>,
    pub view_matrix: Matrix4<f32>,
    pub projection_matrix: Matrix4<f32>,

    // --- Lighting Parameters ---
    /// Direction TO the light source (normalized).
    pub light_dir: Vector3<f32>,
    /// Color/Intensity of the light.
    pub light_color: Vector3<f32>,
    /// Ambient light color/intensity.
    pub ambient_intensity: Vector3<f32>,

    // --- Camera ---
    /// Camera position in World Space (needed for Specular calculation).
    pub camera_pos: Point3<f32>,

    // --- Material Properties ---
    /// Base color of the object.
    pub diffuse_color: Vector3<f32>,
    /// Color of the specular highlight.
    pub specular_color: Vector3<f32>,
    /// Shininess factor (higher = smaller, sharper highlight).
    pub shininess: f32,
}

impl PhongShader {
    /// Creates a new PhongShader with default material and lighting values.
    pub fn new(
        model: Matrix4<f32>,
        view: Matrix4<f32>,
        projection: Matrix4<f32>,
        camera_pos: Point3<f32>,
    ) -> Self {
        Self {
            model_matrix: model,
            view_matrix: view,
            projection_matrix: projection,
            camera_pos,
            // Default lighting (Directional light from top-right)
            light_dir: Vector3::new(1.0, 1.0, 1.0).normalize(),
            light_color: Vector3::new(1.0, 1.0, 1.0),
            ambient_intensity: Vector3::new(0.1, 0.1, 0.1),
            // Default material (Greyish)
            diffuse_color: Vector3::new(0.8, 0.8, 0.8),
            specular_color: Vector3::new(1.0, 1.0, 1.0),
            shininess: 32.0,
        }
    }
}

impl Shader for PhongShader {
    type Varying = PhongVarying;

    fn vertex(&self, vertex: &Vertex) -> (Vector4<f32>, Self::Varying) {
        // 1. Transform Position to World Space
        let world_pos_homo = self.model_matrix * vertex.position.to_homogeneous();
        let world_pos = Point3::from_homogeneous(world_pos_homo).unwrap();

        // 2. Transform Normal to World Space
        // TODO: For non-uniform scaling, we should use the Inverse Transpose of the Model Matrix.
        // For uniform scaling and rotation, the upper-left 3x3 of Model Matrix is sufficient.
        let normal_matrix = self.model_matrix.fixed_view::<3, 3>(0, 0);
        let world_normal = (normal_matrix * vertex.normal).normalize();

        // 3. Transform Position to Clip Space (MVP)
        let mvp = self.projection_matrix * self.view_matrix * self.model_matrix;
        let clip_pos = mvp * vertex.position.to_homogeneous();

        // 4. Pass data to Fragment Shader
        let varying = PhongVarying {
            normal: world_normal,
            world_pos,
            uv: vertex.texcoord,
        };

        (clip_pos, varying)
    }

    fn fragment(&self, varying: Self::Varying) -> Vector3<f32> {
        // 1. Re-normalize interpolated normal (interpolation can shorten vectors)
        let normal = varying.normal.normalize();
        let view_dir = (self.camera_pos - varying.world_pos).normalize();
        let light_dir = self.light_dir.normalize();

        // 2. Ambient Component
        // Ambient = Ka * Ia
        let ambient = self.ambient_intensity.component_mul(&self.diffuse_color);

        // 3. Diffuse Component (Lambertian)
        // Diffuse = Kd * I * max(0, N dot L)
        let diff = normal.dot(&light_dir).max(0.0);
        let diffuse = self.light_color.component_mul(&self.diffuse_color) * diff;

        // 4. Specular Component (Phong Reflection Model)
        // Reflect direction: R = 2 * (N dot L) * N - L
        // Note: nalgebra's reflect might differ, so we implement manually for clarity.
        // We want the reflection of the light vector around the normal.
        // Since light_dir points TO the light, we reverse it for reflection calculation if using standard formula,
        // OR we use: R = 2(N.L)N - L
        // TODO: Check if correct
        let reflect_dir = (normal * (2.0 * normal.dot(&light_dir)) - light_dir).normalize();

        // Specular = Ks * I * max(0, V dot R)^shininess
        let spec = view_dir.dot(&reflect_dir).max(0.0).powf(self.shininess);
        let specular = self.light_color.component_mul(&self.specular_color) * spec;

        // 5. Combine results
        let result = ambient + diffuse + specular;

        // 6. Simple Tone Mapping / Clamping (ensure values are in [0, 1])
        Vector3::new(result.x.min(1.0), result.y.min(1.0), result.z.min(1.0))
    }
}
