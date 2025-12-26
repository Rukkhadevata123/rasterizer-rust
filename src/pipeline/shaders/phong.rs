use crate::core::geometry::Vertex;
use crate::core::pipeline::Shader;
use crate::scene::material::{Material, PhongMaterial};
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
    // Matrices
    pub model_matrix: Matrix4<f32>,
    pub view_matrix: Matrix4<f32>,
    pub projection_matrix: Matrix4<f32>,

    // Lighting (Global)
    pub light_dir: Vector3<f32>,
    pub light_color: Vector3<f32>,
    pub ambient_intensity: Vector3<f32>,

    // Camera
    pub camera_pos: Point3<f32>,

    // Fallback Material (used if no material is passed to fragment)
    pub fallback_material: PhongMaterial,
}

impl PhongShader {
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
            light_dir: Vector3::new(1.0, 1.0, 1.0).normalize(),
            light_color: Vector3::new(1.0, 1.0, 1.0),
            ambient_intensity: Vector3::new(0.1, 0.1, 0.1),
            fallback_material: PhongMaterial::default(),
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

        // 3. Lighting Calculation
        let normal = varying.normal.normalize();
        let view_dir = (self.camera_pos - varying.world_pos).normalize();
        let light_dir = self.light_dir.normalize();

        // Ambient
        let ambient = self
            .ambient_intensity
            .component_mul(&mat_props.ambient_color);

        // Diffuse
        let diff = normal.dot(&light_dir).max(0.0);
        let diffuse = self.light_color.component_mul(&diffuse_color) * diff;

        // Specular
        let reflect_dir = (normal * (2.0 * normal.dot(&light_dir)) - light_dir).normalize();
        let spec = view_dir
            .dot(&reflect_dir)
            .max(0.0)
            .powf(mat_props.shininess);
        let specular = self.light_color.component_mul(&mat_props.specular_color) * spec;

        // Combine
        let result = ambient + diffuse + specular;
        Vector3::new(result.x.min(1.0), result.y.min(1.0), result.z.min(1.0))
    }
}
