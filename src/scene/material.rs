use crate::scene::texture::Texture;
use nalgebra::Vector3;
use std::sync::Arc;

/// Defines how an object interacts with light.
/// Unified to PBR workflow.
#[derive(Debug, Clone)]
pub enum Material {
    Pbr(PbrMaterial),
}

impl Default for Material {
    fn default() -> Self {
        Material::Pbr(PbrMaterial::default())
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
    pub metallic_roughness_texture: Option<Arc<Texture>>,
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
