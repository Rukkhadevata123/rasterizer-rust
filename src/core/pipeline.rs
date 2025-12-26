use crate::core::geometry::Vertex;
use crate::scene::material::Material;
use nalgebra::{Vector3, Vector4};
use std::ops::{Add, Mul};

/// Trait representing data that can be interpolated (e.g., Colors, Normals, UVs).
/// Requires Copy, Addition, and Multiplication by scalar (f32).
pub trait Interpolatable: Copy + Clone + Add<Output = Self> + Mul<f32, Output = Self> {}

// Implement Interpolatable for common types automatically
impl<T> Interpolatable for T where T: Copy + Add<Output = T> + Mul<f32, Output = T> + Send + Sync {}

/// The Shader trait represents the programmable stages of the graphics pipeline.
/// Implementing this trait allows defining custom rendering logic (Phong, PBR, etc.).
pub trait Shader: Send + Sync {
    /// The type of data passed from Vertex Shader to Fragment Shader.
    /// Must support interpolation (e.g., a struct containing Normal and UV).
    type Varying: Interpolatable;

    /// Vertex Shader Stage.
    /// Transforms a raw vertex into Clip Space position and generates varying data.
    ///
    /// # Arguments
    /// * `vertex` - The input vertex data.
    ///s
    /// # Returns
    /// * `Vector4<f32>` - Position in Homogeneous Clip Space.
    /// * `Self::Varying` - Data to be interpolated and passed to the fragment shader.
    fn vertex(&self, vertex: &Vertex) -> (Vector4<f32>, Self::Varying);

    /// Fragment Shader Stage.
    /// Computes the final color of a pixel based on interpolated data.
    ///
    /// # Arguments
    /// * `varying` - The interpolated data for the current pixel.
    /// * `material` - Optional material properties for shading calculations.
    ///
    /// # Returns
    /// * `Vector3<f32>` - The final RGB color (usually linear space, 0.0-1.0).
    ///   Return `None` (or handle discard logic internally) to discard the pixel (alpha masking).
    ///   For simplicity here, we return Vector3, assuming alpha blending is handled by the pipeline.
    fn fragment(&self, varying: Self::Varying, material: Option<&Material>) -> Vector3<f32>;
}
