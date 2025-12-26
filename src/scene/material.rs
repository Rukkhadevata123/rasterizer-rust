use crate::scene::texture::Texture;
use nalgebra::Vector3;
use std::sync::Arc;

/// Defines how an object interacts with light.
#[derive(Debug, Clone)]
pub enum Material {
    Phong(PhongMaterial),
    Pbr(PbrMaterial),
}

impl Default for Material {
    fn default() -> Self {
        Material::Phong(PhongMaterial::default())
    }
}

/// Parameters for the Phong lighting model.
#[derive(Debug, Clone)]
pub struct PhongMaterial {
    pub diffuse_color: Vector3<f32>,
    pub specular_color: Vector3<f32>,
    pub ambient_color: Vector3<f32>,
    pub shininess: f32,
    pub diffuse_texture: Option<Arc<Texture>>,
}

impl Default for PhongMaterial {
    fn default() -> Self {
        Self {
            diffuse_color: Vector3::new(0.8, 0.8, 0.8),
            specular_color: Vector3::new(1.0, 1.0, 1.0),
            ambient_color: Vector3::new(0.1, 0.1, 0.1),
            shininess: 32.0,
            diffuse_texture: None,
        }
    }
}

/// Parameters for Physically Based Rendering (Metallic-Roughness workflow).
#[derive(Debug, Clone)]
pub struct PbrMaterial {
    /// Albedo (Base Color).
    pub albedo: Vector3<f32>,
    /// Metallic (0.0 = dielectric, 1.0 = metal).
    pub metallic: f32,
    /// Roughness (0.0 = smooth, 1.0 = rough).
    pub roughness: f32,
    /// Ambient Occlusion factor.
    pub ao: f32,
    /// Emissive color (light emitted by the surface).
    pub emissive: Vector3<f32>,

    // Textures (Optional)
    pub albedo_texture: Option<Arc<Texture>>,
    pub metallic_roughness_texture: Option<Arc<Texture>>, // Usually packed: G=Roughness, B=Metallic
    pub normal_texture: Option<Arc<Texture>>,
}

impl Default for PbrMaterial {
    fn default() -> Self {
        Self {
            albedo: Vector3::new(1.0, 1.0, 1.0),
            metallic: 0.0,  // Non-metal
            roughness: 0.5, // Medium roughness
            ao: 1.0,
            emissive: Vector3::zeros(),
            albedo_texture: None,
            metallic_roughness_texture: None,
            normal_texture: None,
        }
    }
}
