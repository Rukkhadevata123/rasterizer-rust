use crate::core::framebuffer::FrameBuffer;
use crate::core::geometry::Vertex;
use crate::core::pipeline::Shader;
use crate::core::rasterizer::{PreparedTriangle, Rasterizer, RenderState};
use crate::scene::material::Material;
use crate::scene::mesh::Mesh;
use crate::scene::texture::Texture;
use nalgebra::Vector3;
use rayon::prelude::*;

pub enum RenderGeometry<'a> {
    Mesh(&'a Mesh),
    Triangle([Vertex; 3]),
}

pub struct RenderCommand<'a> {
    pub insertion_id: u64,
    pub shader_index: usize,
    pub geometry: RenderGeometry<'a>,
    pub material: Option<&'a Material>,
    pub state: RenderState,
    pub sort_depth: f32,
}

#[derive(Default)]
pub struct RenderQueue<'a> {
    commands: Vec<RenderCommand<'a>>,
    next_insertion_id: u64,
}

impl<'a> RenderQueue<'a> {
    pub fn push(
        &mut self,
        shader_index: usize,
        geometry: RenderGeometry<'a>,
        material: Option<&'a Material>,
        state: RenderState,
        sort_depth: f32,
    ) {
        let insertion_id = self.next_insertion_id;
        self.next_insertion_id += 1;
        self.commands.push(RenderCommand {
            insertion_id,
            shader_index,
            geometry,
            material,
            state,
            sort_depth,
        });
    }

    pub fn sort_transparent(&mut self) {
        self.commands.sort_by(|a, b| {
            a.sort_depth
                .total_cmp(&b.sort_depth)
                .then_with(|| a.insertion_id.cmp(&b.insertion_id))
        });
    }

    pub fn commands(&self) -> &[RenderCommand<'a>] {
        &self.commands
    }
}

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
    pub fn new(width: usize, height: usize, supersample_scale: usize) -> Result<Self, String> {
        Ok(Self {
            rasterizer: Rasterizer::new(),
            framebuffer: FrameBuffer::new(width, height, supersample_scale)?,
        })
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

    pub fn draw_queue<'a, S>(&mut self, queue: &RenderQueue<'a>, shaders: &'a [S])
    where
        S: Shader<Option<&'a Material>>,
    {
        self.draw_queues(&[queue], shaders);
    }

    pub fn draw_queues<'a, S>(&mut self, queues: &[&RenderQueue<'a>], shaders: &'a [S])
    where
        S: Shader<Option<&'a Material>>,
    {
        let width = self.framebuffer.buffer_width;
        let height = self.framebuffer.buffer_height;
        let prepared: Vec<PreparedTriangle<'_, S::Varying, S, Option<&Material>>> = queues
            .iter()
            .flat_map(|queue| queue.commands())
            .flat_map(|command| {
                let shader = &shaders[command.shader_index];
                match &command.geometry {
                    RenderGeometry::Mesh(mesh) => mesh
                        .indices
                        .par_chunks(3)
                        .flat_map_iter(|indices| {
                            if indices.len() < 3 {
                                return Vec::new().into_iter();
                            }
                            self.prepare_vertices(
                                width,
                                height,
                                [
                                    &mesh.vertices[indices[0] as usize],
                                    &mesh.vertices[indices[1] as usize],
                                    &mesh.vertices[indices[2] as usize],
                                ],
                                shader,
                                command.material,
                                command.state,
                            )
                            .into_iter()
                        })
                        .collect(),
                    RenderGeometry::Triangle(vertices) => self.prepare_vertices(
                        width,
                        height,
                        [&vertices[0], &vertices[1], &vertices[2]],
                        shader,
                        command.material,
                        command.state,
                    ),
                }
            })
            .collect();

        self.rasterizer
            .rasterize_prepared(&mut self.framebuffer, &prepared);
    }

    fn prepare_vertices<'a, S>(
        &self,
        width: usize,
        height: usize,
        vertices: [&Vertex; 3],
        shader: &'a S,
        material: Option<&'a Material>,
        state: RenderState,
    ) -> Vec<PreparedTriangle<'a, S::Varying, S, Option<&'a Material>>>
    where
        S: Shader<Option<&'a Material>>,
    {
        let (pos0, var0) = shader.vertex(vertices[0]);
        let (pos1, var1) = shader.vertex(vertices[1]);
        let (pos2, var2) = shader.vertex(vertices[2]);
        self.rasterizer.prepare_triangle::<S, _>(
            (width, height),
            &[pos0, pos1, pos2],
            &[var0, var1, var2],
            shader,
            state,
            material,
        )
    }
}
