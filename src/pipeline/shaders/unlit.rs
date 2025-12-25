use crate::core::geometry::Vertex;
use crate::core::pipeline::Shader;
use nalgebra::{Matrix4, Vector3, Vector4};

/// A simple shader that visualizes normals as colors.
/// Useful for debugging geometry and the rasterization pipeline.
pub struct UnlitShader {
    /// Model-View-Projection matrix.
    pub mvp_matrix: Matrix4<f32>,
}

impl UnlitShader {
    pub fn new(mvp_matrix: Matrix4<f32>) -> Self {
        Self { mvp_matrix }
    }
}

impl Shader for UnlitShader {
    /// We pass the Normal vector to the fragment shader to visualize it.
    type Varying = Vector3<f32>;

    fn vertex(&self, vertex: &Vertex) -> (Vector4<f32>, Self::Varying) {
        // 1. Transform position to Clip Space
        let clip_pos = self.mvp_matrix * vertex.position.to_homogeneous();

        // 2. Pass normal as varying data.
        // Map normal from [-1, 1] to [0, 1] for visualization.
        let color = (vertex.normal + Vector3::new(1.0, 1.0, 1.0)) * 0.5;

        (clip_pos, color)
    }

    fn fragment(&self, varying: Self::Varying) -> Vector3<f32> {
        // Simply return the interpolated color (normal).
        varying
    }
}
