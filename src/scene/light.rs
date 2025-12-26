use nalgebra::{Point3, Vector3};

/// Represents a light source in the scene.
#[derive(Debug, Clone)]
pub enum Light {
    /// A light source that is infinitely far away (e.g., Sun).
    /// Rays are parallel.
    Directional {
        direction: Vector3<f32>,
        color: Vector3<f32>,
        intensity: f32,
    },
    /// A light source at a specific position that radiates in all directions.
    Point {
        position: Point3<f32>,
        color: Vector3<f32>,
        intensity: f32,
        /// Attenuation coefficients: (constant, linear, quadratic)
        attenuation: (f32, f32, f32),
    },
}

impl Light {
    /// Creates a simple directional light.
    pub fn new_directional(direction: Vector3<f32>, color: Vector3<f32>, intensity: f32) -> Self {
        Self::Directional {
            direction: direction.normalize(),
            color,
            intensity,
        }
    }

    /// Creates a simple point light.
    pub fn new_point(position: Point3<f32>, color: Vector3<f32>, intensity: f32) -> Self {
        Self::Point {
            position,
            color,
            intensity,
            attenuation: (1.0, 0.09, 0.032), // Default attenuation values
        }
    }

    /// Calculates the direction vector FROM the surface point TO the light source.
    pub fn get_direction_to_light(&self, surface_point: &Point3<f32>) -> Vector3<f32> {
        match self {
            Light::Directional { direction, .. } => -direction, // Direction is usually defined as "direction the light travels"
            Light::Point { position, .. } => (position - surface_point).normalize(),
        }
    }

    /// Calculates the light intensity arriving at the surface point.
    /// Handles attenuation for point lights.
    pub fn get_intensity(&self, surface_point: &Point3<f32>) -> Vector3<f32> {
        match self {
            Light::Directional {
                color, intensity, ..
            } => color * *intensity,

            Light::Point {
                position,
                color,
                intensity,
                attenuation,
            } => {
                let distance = (position - surface_point).norm();
                let (c, l, q) = attenuation;
                let attenuation_factor = 1.0 / (c + l * distance + q * distance * distance);
                color * *intensity * attenuation_factor
            }
        }
    }
}
