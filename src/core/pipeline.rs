use crate::core::geometry::Vertex;
use crate::scene::material::Material;
use nalgebra::{Vector3, Vector4};
use std::ops::{Add, Mul};

/// Trait for types that can be linearly interpolated across a triangle's surface.
///
/// Requirements:
/// - Copy + Clone: cheaply duplicable values for per-vertex storage and interpolation.
/// - Add + Mul<f32>: support linear combination (a + b * t) used by barycentric interpolation.
/// - Send + Sync: safe to use from multiple threads during parallel rasterization.
pub trait Interpolatable:
    Copy + Clone + Add<Output = Self> + Mul<f32, Output = Self> + Send + Sync
{
}

/// Blanket implementation for any type satisfying the trait bounds above.
impl<T> Interpolatable for T where T: Copy + Add<Output = T> + Mul<f32, Output = T> + Send + Sync {}

/// Shader represents the programmable stages of the pipeline.
///
/// Implementations must be thread-safe (Send + Sync) because shading may be invoked
/// concurrently across fragments.
///
/// Associated types:
/// - Varying: per-vertex outputs from the vertex stage that will be interpolated
///   for each fragment. Varying must be Interpolatable to support barycentric interpolation.
pub trait Shader: Send + Sync {
    /// Per-vertex varying data to be interpolated and provided to the fragment shader.
    type Varying: Interpolatable;

    /// Vertex shader stage.
    ///
    /// Transforms the given vertex into homogeneous clip space (Vector4<f32>) used by
    /// clipping and perspective divide. Also returns the varying data associated with
    /// that vertex which will be interpolated across the primitive.
    ///
    /// # Arguments
    /// - `vertex`: input vertex attributes (position, normal, uv, etc.)
    ///
    /// # Returns
    /// - `(Vector4<f32>, Self::Varying)`: clip-space position and per-vertex varying.
    fn vertex(&self, vertex: &Vertex) -> (Vector4<f32>, Self::Varying);

    /// Fragment shader stage.
    ///
    /// Computes the final linear RGB color for the current fragment, using the
    /// interpolated varying and optional material state. The pipeline currently
    /// expects a Vector3<f32> color in linear 0.0..1.0 range; discard/alpha logic
    /// is not modeled here (implementations can choose to encode discard by
    /// returning a special color convention if needed).
    ///
    /// # Arguments
    /// - `varying`: interpolated per-fragment data.
    /// - `material`: optional material parameters available to the shader.
    ///
    /// # Returns
    /// - `Vector3<f32>`: final RGB color (linear space).
    fn fragment(&self, varying: Self::Varying, material: Option<&Material>) -> Vector3<f32>;
}
