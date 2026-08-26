use crate::core::framebuffer::FrameBuffer;
use crate::core::geometry::Vertex;
use crate::core::pipeline::Shader;
use crate::core::rasterizer::{PreparedTriangle, Rasterizer};
use crate::scene::material::Material;
use crate::scene::mesh::Mesh;
use crate::scene::model::Model;
use crate::scene::texture::Texture;
use nalgebra::Vector3;
use rayon::prelude::*;

pub struct ClearOptions<'a> {
    pub color: Vector3<f32>,
    pub gradient: Option<(Vector3<f32>, Vector3<f32>)>,
    pub texture: Option<&'a Texture>,
    pub depth: f32,
}

impl Default for ClearOptions<'_> {
    fn default() -> Self {
        Self {
            color: Vector3::zeros(),
            gradient: None,
            texture: None,
            depth: f32::INFINITY,
        }
    }
}

pub struct Renderer {
    pub rasterizer: Rasterizer,
    pub framebuffer: FrameBuffer,
}

impl Renderer {
    pub fn new(width: usize, height: usize, sample_count: usize) -> Self {
        Self {
            rasterizer: Rasterizer::new(),
            framebuffer: FrameBuffer::new(width, height, sample_count),
        }
    }

    pub fn clear_with_options(&mut self, options: ClearOptions) {
        let width = self.framebuffer.buffer_width;
        let height = self.framebuffer.buffer_height;
        self.framebuffer.clear_with(options.depth, |x, y| {
            let u = x as f32 / width as f32;
            let v = y as f32 / height as f32;

            if let Some(texture) = options.texture {
                texture.sample_color(u, v).xyz()
            } else if let Some((top, bottom)) = options.gradient {
                top.lerp(&bottom, v)
            } else {
                options.color
            }
        });
    }

    pub fn draw_model<S: Shader>(&mut self, model: &Model, shader: &S) {
        for mesh in &model.meshes {
            let material = model.materials.get(mesh.material_id);
            self.draw_mesh(mesh, shader, material);
        }
    }

    pub fn draw_sorted_triangles<S: Shader>(
        &mut self,
        triangles: Vec<(&Vertex, &Vertex, &Vertex, &Material)>,
        shader: &S,
    ) {
        let width = self.framebuffer.buffer_width;
        let height = self.framebuffer.buffer_height;
        let prepared: Vec<PreparedTriangle<'_, S::Varying>> = triangles
            .into_iter()
            .flat_map(|(v0, v1, v2, material)| {
                let (pos0, var0) = shader.vertex(v0);
                let (pos1, var1) = shader.vertex(v1);
                let (pos2, var2) = shader.vertex(v2);
                self.rasterizer.prepare_triangle::<S>(
                    width,
                    height,
                    &[pos0, pos1, pos2],
                    &[var0, var1, var2],
                    Some(material),
                )
            })
            .collect();

        self.rasterizer
            .rasterize_prepared(&mut self.framebuffer, shader, &prepared);
    }

    pub fn draw_mesh<S: Shader>(&mut self, mesh: &Mesh, shader: &S, material: Option<&Material>) {
        let width = self.framebuffer.buffer_width;
        let height = self.framebuffer.buffer_height;
        let prepared: Vec<PreparedTriangle<'_, S::Varying>> = mesh
            .indices
            .par_chunks(3)
            .flat_map_iter(|indices| {
                if indices.len() < 3 {
                    return Vec::new().into_iter();
                }

                let v0 = &mesh.vertices[indices[0] as usize];
                let v1 = &mesh.vertices[indices[1] as usize];
                let v2 = &mesh.vertices[indices[2] as usize];
                let (pos0, var0) = shader.vertex(v0);
                let (pos1, var1) = shader.vertex(v1);
                let (pos2, var2) = shader.vertex(v2);

                self.rasterizer
                    .prepare_triangle::<S>(
                        width,
                        height,
                        &[pos0, pos1, pos2],
                        &[var0, var1, var2],
                        material,
                    )
                    .into_iter()
            })
            .collect();

        self.rasterizer
            .rasterize_prepared(&mut self.framebuffer, shader, &prepared);
    }
}
