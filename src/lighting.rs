use crate::material_system::ILight;
use crate::model_types::Material;
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

// Add From trait implementation
impl From<&Material> for SimpleMaterial {
    fn from(material: &Material) -> Self {
        SimpleMaterial {
            ambient: material.ambient,
            diffuse: material.diffuse,
            specular: material.specular,
            shininess: material.shininess,
        }
    }
}

/// Calculates the color of a point based on Blinn-Phong lighting model.
/// If the light type is Ambient, returns white (1.0, 1.0, 1.0) to signify no lighting effect.
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
    // If light is Ambient, return white immediately.
    if let Light::Ambient(_) = light {
        return Vector3::new(1.0, 1.0, 1.0);
    }

    // Calculate ambient component based on material only (light intensity is handled below)
    // Note: This ambient calculation is technically part of Blinn-Phong,
    // but since we return early for Light::Ambient, this only applies
    // when Directional or Point light is active. We can consider
    // using a separate ambient term if needed outside this function.
    // For now, let's keep it simple and assume ambient is part of the lit color.
    let ambient_color = material.ambient; // Use material's ambient property

    let (light_dir_view, light_intensity, attenuation) = match light {
        Light::Directional {
            direction,
            intensity,
        } => (*direction, *intensity, 1.0),
        Light::Point {
            position,
            intensity,
            attenuation,
        } => {
            let light_vec = position - surface_point_view;
            let distance = light_vec.norm();
            if distance < 1e-6 {
                (Vector3::z(), *intensity, 1.0) // Avoid division by zero
            } else {
                let dir = light_vec / distance;
                let att = 1.0
                    / (attenuation.0
                        + attenuation.1 * distance
                        + attenuation.2 * distance * distance);
                (dir, *intensity, att.max(0.0))
            }
        }
        Light::Ambient(_) => unreachable!(), // Already handled above
    };

    // Diffuse term
    let n_dot_l = normal_view.dot(&light_dir_view).max(0.0);
    let diffuse_color = material.diffuse.component_mul(&light_intensity) * n_dot_l;

    // Specular term
    let specular_color = if n_dot_l > 0.0 {
        let halfway_dir = (light_dir_view + view_dir_view).normalize();
        let n_dot_h = normal_view.dot(&halfway_dir).max(0.0);
        let spec_intensity = n_dot_h.powf(material.shininess);
        material.specular.component_mul(&light_intensity) * spec_intensity
    } else {
        Vector3::zeros()
    };

    // Combine components: Ambient + Attenuated(Diffuse + Specular)
    // We use material.ambient directly here.
    let final_color = ambient_color + attenuation * (diffuse_color + specular_color);

    final_color.map(|c| c.clamp(0.0, 1.0))
}

// 实现 ILight 特质对于 Light 枚举
impl ILight for Light {
    fn get_direction(&self, point: &Point3<f32>) -> Vector3<f32> {
        match self {
            Light::Directional { direction, .. } => *direction,
            Light::Point { position, .. } => {
                let to_light = position - point;
                let distance = to_light.norm();
                if distance < 1e-6 {
                    Vector3::z() // 默认方向，避免除零
                } else {
                    to_light / distance
                }
            }
            Light::Ambient(_) => Vector3::z(), // 环境光没有方向，返回默认方向
        }
    }

    fn get_intensity(&self, point: &Point3<f32>) -> Vector3<f32> {
        match self {
            Light::Directional { intensity, .. } => *intensity,
            Light::Point {
                position,
                intensity,
                attenuation,
            } => {
                let distance = (position - point).norm();
                let att_factor = 1.0
                    / (attenuation.0
                        + attenuation.1 * distance
                        + attenuation.2 * distance * distance);
                intensity * att_factor.max(0.0)
            }
            Light::Ambient(intensity) => *intensity,
        }
    }

    fn get_type_name(&self) -> &'static str {
        match self {
            Light::Directional { .. } => "Directional",
            Light::Point { .. } => "Point",
            Light::Ambient(_) => "Ambient",
        }
    }

    fn to_light_enum(&self) -> crate::lighting::Light {
        *self // Since Light itself is the concrete enum, we just return a copy
    }
}

// 为 SimpleMaterial 实现 IMaterial 特性
impl crate::material_system::IMaterial for SimpleMaterial {
    fn compute_response(
        &self,
        light_dir: &Vector3<f32>,
        view_dir: &Vector3<f32>,
        normal: &Vector3<f32>,
    ) -> Vector3<f32> {
        // 简化版的 Blinn-Phong 响应计算
        let n_dot_l = normal.dot(light_dir).max(0.0);
        if n_dot_l <= 0.0 {
            return Vector3::zeros(); // 如果光线方向与法线夹角大于90度，返回黑色
        }

        // 漫反射项
        let diffuse = self.diffuse * n_dot_l;

        // 镜面反射项（Blinn-Phong）
        let half_dir = (light_dir + view_dir).normalize();
        let n_dot_h = normal.dot(&half_dir).max(0.0);
        let specular = self.specular * n_dot_h.powf(self.shininess);

        // 返回总响应（不考虑环境光，因为环境光通常在光照模型中单独处理）
        diffuse + specular
    }

    fn get_diffuse_color(&self) -> Vector3<f32> {
        self.diffuse
    }

    fn get_ambient_color(&self) -> Vector3<f32> {
        self.ambient
    }
}
