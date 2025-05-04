use nalgebra::{Point3, Vector3};

/// Represents different types of light sources.
#[derive(Debug, Clone, Copy)]
pub enum Light {
    /// Directional light with a direction *towards* the light source.
    Directional {
        direction: Vector3<f32>,
        intensity: Vector3<f32>,
    },
    /// Point light with position and attenuation factors.
    Point {
        position: Point3<f32>,
        intensity: Vector3<f32>,
        /// Attenuation factors: (constant, linear, quadratic)
        attenuation: (f32, f32, f32),
    },
    /// Ambient light only (or disabled lighting).
    Ambient(Vector3<f32>), // Represents the ambient intensity
}

/// Basic material properties relevant for simple lighting.
#[derive(Debug, Clone, Copy)]
pub struct SimpleMaterial {
    pub ambient: Vector3<f32>,
    pub diffuse: Vector3<f32>,
    pub specular: Vector3<f32>,
    pub shininess: f32,
}

impl Default for SimpleMaterial {
    fn default() -> Self {
        SimpleMaterial {
            ambient: Vector3::new(0.1, 0.1, 0.1),  // Low ambient default
            diffuse: Vector3::new(0.8, 0.8, 0.8),  // Default diffuse color (grey)
            specular: Vector3::new(1.0, 1.0, 1.0), // White specular highlight
            shininess: 32.0,
        }
    }
}

/// Calculates the color of a point based on Blinn-Phong lighting model.
///
/// # Arguments
/// * `surface_point_view`: Position of the point on the surface in view space.
/// * `normal_view`: Surface normal at the point in view space (must be normalized).
/// * `view_dir_view`: Direction from the surface point towards the camera in view space (must be normalized).
/// * `light`: The light source definition.
/// * `material`: Material properties of the surface.
///
/// # Returns
/// The calculated RGB color (f32, clamped [0, 1]).
pub fn calculate_blinn_phong(
    surface_point_view: Point3<f32>,
    normal_view: Vector3<f32>,
    view_dir_view: Vector3<f32>,
    light: &Light,
    material: &SimpleMaterial,
) -> Vector3<f32> {
    let ambient_color = material.ambient.component_mul(&match light {
        // Use light intensity for ambient if available, otherwise just material ambient
        Light::Ambient(intensity) => *intensity,
        Light::Directional { intensity, .. } => *intensity,
        Light::Point { intensity, .. } => *intensity,
    });

    let (light_dir_view, light_intensity, attenuation) = match light {
        Light::Directional {
            direction,
            intensity,
        } => {
            // Direction is *towards* the light source
            (*direction, *intensity, 1.0)
        }
        Light::Point {
            position,
            intensity,
            attenuation,
        } => {
            let light_vec = position - surface_point_view;
            let distance = light_vec.norm();
            if distance < 1e-6 {
                // At the light source, avoid division by zero, return max intensity? Or handle differently?
                // For simplicity, treat as directional from Z+ if too close
                (Vector3::z(), *intensity, 1.0)
            } else {
                let dir = light_vec / distance;
                let att = 1.0
                    / (attenuation.0
                        + attenuation.1 * distance
                        + attenuation.2 * distance * distance);
                (dir, *intensity, att.max(0.0)) // Ensure non-negative attenuation
            }
        }
        Light::Ambient(_) => {
            // Only ambient component, return early
            return ambient_color.map(|c| c.clamp(0.0, 1.0));
        }
    };

    // Diffuse term (Lambertian)
    let n_dot_l = normal_view.dot(&light_dir_view).max(0.0);
    let diffuse_color = material.diffuse.component_mul(&light_intensity) * n_dot_l;

    // Specular term (Blinn-Phong)
    let specular_color = if n_dot_l > 0.0 {
        let halfway_dir = (light_dir_view + view_dir_view).normalize();
        let n_dot_h = normal_view.dot(&halfway_dir).max(0.0);
        let spec_intensity = n_dot_h.powf(material.shininess);
        // Use material's specular color multiplied by light intensity
        material.specular.component_mul(&light_intensity) * spec_intensity
    } else {
        Vector3::zeros()
    };

    // Combine components
    let final_color = ambient_color + attenuation * (diffuse_color + specular_color);

    // Clamp final color to [0, 1] range
    final_color.map(|c| c.clamp(0.0, 1.0))
}
