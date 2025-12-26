use crate::scene::texture::Texture;
use nalgebra::Vector3;
use std::sync::Arc;

/// Defines how an object interacts with light.
#[derive(Debug, Clone)]
pub enum Material {
    Phong(PhongMaterial),
    // TODO: Future: PBR(PbrMaterial)
}

impl Default for Material {
    fn default() -> Self {
        Material::Phong(PhongMaterial::default())
    }
}

/// Parameters for the Phong lighting model.
#[derive(Debug, Clone)]
pub struct PhongMaterial {
    /// Base color of the surface.
    pub diffuse_color: Vector3<f32>,
    /// Color of the specular highlight.
    pub specular_color: Vector3<f32>,
    /// Ambient color factor.
    pub ambient_color: Vector3<f32>,
    /// Shininess exponent (higher = smaller, sharper highlight).
    pub shininess: f32,

    /// Optional diffuse texture map.
    /// If present, it overrides `diffuse_color`.
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
