use crate::core::geometry::{SUPPORTED_TEXCOORD_SETS, Vertex};
use nalgebra::Vector4;
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
    /// Optionally returns one UV set if the varying contains it.
    fn get_uv(&self, _set: usize) -> Option<nalgebra::Vector2<f32>> {
        None
    }
}

/// Shader represents the programmable stages of the pipeline.
///
/// Implementations must be thread-safe (Send + Sync) because shading may be invoked
/// concurrently across fragments.
///
/// Associated types:
/// - Varying: per-vertex outputs from the vertex stage that will be interpolated
///   for each fragment. Varying must be Interpolatable to support barycentric interpolation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FragmentOutput {
    Discard,
    Color(Vector4<f32>),
}

#[derive(Clone, Copy, Debug)]
pub struct FragmentInput<V> {
    pub varying: V,
    pub front_facing: bool,
    pub uv_densities: [f32; SUPPORTED_TEXCOORD_SETS],
}

impl<V> FragmentInput<V> {
    pub fn uv_density(&self, set: usize) -> f32 {
        self.uv_densities.get(set).copied().unwrap_or(0.0)
    }
}

pub trait Shader<C>: Send + Sync
where
    C: Copy + Send + Sync,
{
    /// Per-vertex varying data to be interpolated and provided to the fragment shader.
    type Varying: Interpolatable;

    /// Vertex shader stage.
    ///
    /// Transforms the given vertex into homogeneous clip space (Vector4<f32>) used by
    /// clipping and perspective divide. Also returns the varying data associated with
    /// that vertex which will be interpolated across the primitive.
    ///
    /// # Arguments
    /// - `vertex`: input vertex attributes (position, normal, uv, etc.).
    /// - `context`: caller-defined state associated with the draw.
    ///
    /// # Returns
    /// - `(Vector4<f32>, Self::Varying)`: clip-space position and per-vertex varying.
    fn vertex(&self, vertex: &Vertex, context: C) -> (Vector4<f32>, Self::Varying);

    /// Fragment shader stage.
    ///
    /// Computes either a discarded fragment or its final linear RGBA color using
    /// the interpolated varying and caller-provided draw context.
    ///
    /// Additionally, `uv_densities` provides a triangle-level estimate for each supported UV set
    /// (sqrt(Area_uv / Area_screen)). Shaders use the density selected by each texture binding.
    ///
    /// # Arguments
    /// - `input`: interpolated varying and non-interpolated fragment metadata.
    /// - `context`: the same caller-defined state provided to the vertex stage.
    ///
    /// # Returns
    /// - `FragmentOutput`: explicit discard or final RGBA color (linear space).
    fn fragment(&self, input: FragmentInput<Self::Varying>, context: C) -> FragmentOutput;
}
