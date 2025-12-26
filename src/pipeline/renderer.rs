use crate::core::framebuffer::FrameBuffer;
use crate::core::pipeline::Shader;
use crate::core::rasterizer::Rasterizer;
use crate::scene::material::Material;
use crate::scene::mesh::Mesh;
use crate::scene::model::Model;

/// The high-level renderer that orchestrates the pipeline stages.
pub struct Renderer {
    pub rasterizer: Rasterizer,
    pub framebuffer: FrameBuffer,
}

impl Renderer {
    /// Creates a new renderer.
    /// sample_count: 1 for no AA, 2 for 2x2 SSAA, etc.
    pub fn new(width: usize, height: usize, sample_count: usize) -> Self {
        Self {
            // Rasterizer is stateless regarding size now, it relies on the framebuffer passed to it.
            rasterizer: Rasterizer::new(),
            framebuffer: FrameBuffer::new(width, height, sample_count),
        }
    }

    /// Clears the framebuffer.
    pub fn clear(&mut self, color: nalgebra::Vector3<f32>) {
        self.framebuffer.clear(color, f32::INFINITY);
    }

    /// Draws a complete model containing multiple meshes.
    pub fn draw_model<S: Shader>(&mut self, model: &Model, shader: &S) {
        for mesh in &model.meshes {
            // Retrieve the material for this mesh
            // If the ID is invalid, we pass None (Shader will use fallback)
            let material = if mesh.material_id < model.materials.len() {
                Some(&model.materials[mesh.material_id])
            } else {
                None
            };

            self.draw_mesh(mesh, shader, material);
        }
    }

    /// Draws a mesh using the provided shader and material.
    pub fn draw_mesh<S: Shader>(&mut self, mesh: &Mesh, shader: &S, material: Option<&Material>) {
        // 1. Vertex Processing & Primitive Assembly Loop
        // Iterate over indices in chunks of 3 (triangles)
        for chunk in mesh.indices.chunks(3) {
            if chunk.len() < 3 {
                break;
            }

            let i0 = chunk[0] as usize;
            let i1 = chunk[1] as usize;
            let i2 = chunk[2] as usize;

            // Fetch vertices
            let v0 = &mesh.vertices[i0];
            let v1 = &mesh.vertices[i1];
            let v2 = &mesh.vertices[i2];

            // Run Vertex Shader
            let (pos0, var0) = shader.vertex(v0);
            let (pos1, var1) = shader.vertex(v1);
            let (pos2, var2) = shader.vertex(v2);

            // Assemble primitive data
            let clip_coords = [pos0, pos1, pos2];
            let varyings = [var0, var1, var2];

            // 4. Rasterization
            self.rasterizer.rasterize_triangle(
                &mut self.framebuffer,
                shader,
                &clip_coords,
                &varyings,
                material, // Pass the material down to the rasterizer
            );
        }
    }
}
