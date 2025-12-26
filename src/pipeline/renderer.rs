use crate::core::framebuffer::FrameBuffer;
use crate::core::pipeline::Shader;
use crate::core::rasterizer::Rasterizer;
use crate::scene::material::Material;
use crate::scene::mesh::Mesh;
use crate::scene::model::Model;
use crate::scene::texture::Texture;
use nalgebra::Vector3;
use rayon::prelude::*;

/// Options for clearing the framebuffer.
pub struct ClearOptions<'a> {
    /// Fallback solid color if no gradient/texture is used.
    pub color: Vector3<f32>,
    /// Optional gradient (Top Color, Bottom Color).
    pub gradient: Option<(Vector3<f32>, Vector3<f32>)>,
    /// Optional background image. Overrides gradient if present.
    pub texture: Option<&'a Texture>,
    /// Depth value to clear to (usually f32::INFINITY).
    pub depth: f32,
}

impl Default for ClearOptions<'_> {
    fn default() -> Self {
        Self {
            color: Vector3::new(0.0, 0.0, 0.0),
            gradient: None,
            texture: None,
            depth: f32::INFINITY,
        }
    }
}

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

    /// Clears the framebuffer using advanced options (Gradient, Texture).
    pub fn clear_with_options(&mut self, options: ClearOptions) {
        // 1. Clear Depth Buffer
        let depth_bits = options.depth.to_bits();
        // Parallel clear for depth buffer (optional optimization)
        self.framebuffer.depth_buffer.par_iter().for_each(|d| {
            d.store(depth_bits, std::sync::atomic::Ordering::Relaxed);
        });

        // 2. Clear Color Buffer
        let width = self.framebuffer.buffer_width;
        let height = self.framebuffer.buffer_height;

        // Get mutable reference to the underlying vector
        let color_buffer = unsafe { &mut *self.framebuffer.color_buffer.get() };

        // Parallel clear for color buffer
        // We iterate over rows to make gradient calculation easier
        color_buffer
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(y, row)| {
                let v = y as f32 / height as f32;
                for (x, pixel) in row.iter_mut().enumerate() {
                    let u = x as f32 / width as f32;

                    let color = if let Some(tex) = options.texture {
                        tex.sample(u, v)
                    } else if let Some((top, bottom)) = options.gradient {
                        top.lerp(&bottom, v)
                    } else {
                        options.color
                    };
                    *pixel = color;
                }
            });
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
    pub fn draw_mesh<S: Shader + Sync>(
        &mut self,
        mesh: &Mesh,
        shader: &S,
        material: Option<&Material>,
    ) {
        // Use Rayon to process triangles in parallel
        // chunk size can be tuned. 64 indices = ~21 triangles per task.
        mesh.indices.par_chunks(3).for_each(|chunk| {
            if chunk.len() < 3 {
                return;
            }

            let i0 = chunk[0] as usize;
            let i1 = chunk[1] as usize;
            let i2 = chunk[2] as usize;

            let v0 = &mesh.vertices[i0];
            let v1 = &mesh.vertices[i1];
            let v2 = &mesh.vertices[i2];

            // Vertex Shader (Parallelized!)
            let (pos0, var0) = shader.vertex(v0);
            let (pos1, var1) = shader.vertex(v1);
            let (pos2, var2) = shader.vertex(v2);

            let clip_coords = [pos0, pos1, pos2];
            let varyings = [var0, var1, var2];

            // Rasterization (Parallelized!)
            // Note: We pass &self.framebuffer (shared reference)
            // The framebuffer handles synchronization internally.
            self.rasterizer.rasterize_triangle(
                &self.framebuffer,
                shader,
                &clip_coords,
                &varyings,
                material,
            );
        });
    }
}
