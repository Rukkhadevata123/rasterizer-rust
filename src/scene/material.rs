use crate::scene::texture::Texture;
use nalgebra::Vector3;
use std::sync::Arc;

/// Defines how the material handles alpha transparency.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AlphaMode {
    /// Fully opaque. Alpha value is ignored.
    Opaque,
    /// Pixels are discarded if alpha < cutoff.
    Mask(f32),
    /// Alpha blending: src * alpha + dst * (1 - alpha).
    Blend,
}

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
    /// Base Alpha factor (0.0 = transparent, 1.0 = opaque).
    pub alpha: f32,
    /// Metallic (0.0 = dielectric, 1.0 = metal).
    pub metallic: f32,
    /// Roughness (0.0 = smooth, 1.0 = rough).
    pub roughness: f32,
    /// Ambient Occlusion factor.
    pub ao: f32,
    /// Emissive color (light emitted by the surface).
    pub emissive: Vector3<f32>,
    /// Alpha Mode
    pub alpha_mode: AlphaMode,
    /// Render both sides of each primitive.
    pub double_sided: bool,

    // Textures (Optional)
    pub albedo_texture: Option<Arc<Texture>>,
    pub metallic_roughness_texture: Option<Arc<Texture>>,
    pub normal_texture: Option<Arc<Texture>>,

    // GLTF compliant optional textures
    pub ao_texture: Option<Arc<Texture>>,
    pub emissive_texture: Option<Arc<Texture>>,
}

impl Default for PbrMaterial {
    fn default() -> Self {
        Self {
            albedo: Vector3::new(1.0, 1.0, 1.0),
            alpha: 1.0,
            metallic: 0.0,  // Non-metal
            roughness: 0.5, // Medium roughness
            ao: 1.0,
            emissive: Vector3::zeros(),
            alpha_mode: AlphaMode::Opaque,
            double_sided: false,
            albedo_texture: None,
            metallic_roughness_texture: None,
            normal_texture: None,
            ao_texture: None,
            emissive_texture: None,
        }
    }
}
