use crate::core::framebuffer::FrameBuffer;
use crate::core::rasterizer::Rasterizer;
use crate::core::pipeline::Shader;
use crate::scene::mesh::Mesh;

/// The high-level renderer that orchestrates the pipeline stages.
pub struct Renderer {
    pub rasterizer: Rasterizer,
    pub framebuffer: FrameBuffer,
}

impl Renderer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            rasterizer: Rasterizer::new(width, height),
            framebuffer: FrameBuffer::new(width, height),
        }
    }

    /// Clears the framebuffer.
    pub fn clear(&mut self, color: nalgebra::Vector3<f32>) {
        self.framebuffer.clear(color, f32::INFINITY);
    }

    /// Draws a mesh using the provided shader.
    /// This function simulates the graphics pipeline:
    /// 1. Vertex Specification (Mesh)
    /// 2. Vertex Shader Execution
    /// 3. Primitive Assembly (Triangle setup)
    /// 4. Rasterization (via Rasterizer)
    pub fn draw_mesh<S: Shader>(&mut self, mesh: &Mesh, shader: &S) {
        // TODO: Optimization - Vertex Cache?
        // Currently we process vertices per triangle, which is inefficient for shared vertices.
        // A better approach is to transform all vertices first, then index them.
        
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

            // TODO: Clipping Stage (Sutherland-Hodgman) should happen here
            // before rasterization. For now, we rely on the rasterizer's bounding box check.

            // 4. Rasterization
            self.rasterizer.rasterize_triangle(
                &mut self.framebuffer,
                shader,
                &clip_coords,
                &varyings,
            );
        }
    }
}