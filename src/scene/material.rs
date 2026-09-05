use crate::scene::texture::TextureBinding;
use nalgebra::Vector3;

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

/// PBR surface material supported by the renderer.
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
    /// Linear base-color factor.
    pub albedo: Vector3<f32>,
    /// Base Alpha factor (0.0 = transparent, 1.0 = opaque).
    pub alpha: f32,
    /// Metallic (0.0 = dielectric, 1.0 = metal).
    pub metallic: f32,
    /// Roughness (0.0 = smooth, 1.0 = rough).
    pub roughness: f32,
    /// Scalar applied to the X/Y components decoded from a tangent-space normal map.
    pub normal_scale: f32,
    /// Strength used to blend a sampled occlusion value with fully lit ambient light.
    pub occlusion_strength: f32,
    /// Emissive color (light emitted by the surface).
    pub emissive: Vector3<f32>,
    /// Controls opaque, masked, or blended rendering.
    pub alpha_mode: AlphaMode,
    /// Render both sides of each primitive.
    pub double_sided: bool,

    // Each binding retains its image, sampler, UV set, and color/data usage.
    pub albedo_texture: Option<TextureBinding>,
    pub metallic_roughness_texture: Option<TextureBinding>,
    pub normal_texture: Option<TextureBinding>,

    pub ao_texture: Option<TextureBinding>,
    pub emissive_texture: Option<TextureBinding>,
}

impl Default for PbrMaterial {
    fn default() -> Self {
        Self {
            albedo: Vector3::new(1.0, 1.0, 1.0),
            alpha: 1.0,
            metallic: 0.0,
            roughness: 0.5,
            normal_scale: 1.0,
            occlusion_strength: 1.0,
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

impl PbrMaterial {
    /// Replaces non-finite factor values with defaults and clamps factors whose glTF
    /// core-material ranges are bounded. Normal scale and alpha cutoff remain unbounded.
    pub fn sanitize_factors(&mut self) {
        self.albedo = self.albedo.map(|value| sanitize_unit(value, 1.0));
        self.alpha = sanitize_unit(self.alpha, 1.0);
        self.metallic = sanitize_unit(self.metallic, 0.0);
        self.roughness = sanitize_unit(self.roughness, 0.5);
        self.normal_scale = sanitize_finite(self.normal_scale, 1.0);
        self.occlusion_strength = sanitize_unit(self.occlusion_strength, 1.0);
        self.emissive = self.emissive.map(|value| sanitize_unit(value, 0.0));
        if let AlphaMode::Mask(cutoff) = &mut self.alpha_mode {
            *cutoff = sanitize_finite(*cutoff, 0.5);
        }
    }
}

fn sanitize_unit(value: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        fallback
    }
}

fn sanitize_finite(value: f32, fallback: f32) -> f32 {
    if value.is_finite() { value } else { fallback }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn material_factors_are_finite_and_clamped_to_supported_ranges() {
        let mut material = PbrMaterial {
            albedo: Vector3::new(-1.0, f32::NAN, 2.0),
            alpha: f32::INFINITY,
            metallic: -0.5,
            roughness: 1.5,
            normal_scale: f32::NEG_INFINITY,
            occlusion_strength: 2.0,
            emissive: Vector3::new(f32::NAN, -1.0, 3.0),
            alpha_mode: AlphaMode::Mask(f32::NAN),
            ..Default::default()
        };

        material.sanitize_factors();

        assert_eq!(material.albedo, Vector3::new(0.0, 1.0, 1.0));
        assert_eq!(material.alpha, 1.0);
        assert_eq!(material.metallic, 0.0);
        assert_eq!(material.roughness, 1.0);
        assert_eq!(material.normal_scale, 1.0);
        assert_eq!(material.occlusion_strength, 1.0);
        assert_eq!(material.emissive, Vector3::new(0.0, 0.0, 1.0));
        assert_eq!(material.alpha_mode, AlphaMode::Mask(0.5));
    }

    #[test]
    fn unbounded_material_factors_keep_values_outside_unit_interval() {
        let mut material = PbrMaterial {
            normal_scale: -2.5,
            alpha_mode: AlphaMode::Mask(1.5),
            ..Default::default()
        };

        material.sanitize_factors();

        assert_eq!(material.normal_scale, -2.5);
        assert_eq!(material.alpha_mode, AlphaMode::Mask(1.5));
    }
}
