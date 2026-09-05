use crate::core::geometry::{SUPPORTED_TEXCOORD_SETS, Vertex};
use nalgebra::Vector4;
use std::ops::{Add, Mul};

/// Varying data that supports barycentric interpolation during parallel rasterization.
pub trait Interpolatable:
    Copy + Clone + Add<Output = Self> + Mul<f32, Output = Self> + Send + Sync
{
    /// Optionally returns one UV set if the varying contains it.
    fn get_uv(&self, _set: usize) -> Option<nalgebra::Vector2<f32>> {
        None
    }
}

/// Result of shading one covered fragment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FragmentOutput {
    Discard,
    Color(Vector4<f32>),
}

/// Interpolated varying and triangle metadata for one covered fragment.
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

/// Programmable vertex and fragment stages used concurrently by the rasterizer.
pub trait Shader<C>: Send + Sync
where
    C: Copy + Send + Sync,
{
    /// Per-vertex varying data to be interpolated and provided to the fragment shader.
    type Varying: Interpolatable;

    /// Produces a homogeneous clip-space position and interpolated varying for one vertex.
    fn vertex(&self, vertex: &Vertex, context: C) -> (Vector4<f32>, Self::Varying);

    /// Returns discard or linear RGBA for one fragment. Each UV density is the triangle-level
    /// `sqrt(area_uv / area_screen)` estimate used for texture mip selection.
    fn fragment(&self, input: FragmentInput<Self::Varying>, context: C) -> FragmentOutput;
}
