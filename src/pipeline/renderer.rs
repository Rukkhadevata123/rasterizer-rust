use crate::core::framebuffer::FrameBuffer;
use crate::core::geometry::Vertex;
use crate::core::pipeline::Shader;
use crate::core::rasterizer::{MAX_PREPARED_TRIANGLES, PreparedTriangle, Rasterizer, RenderState};
use crate::scene::material::Material;
use crate::scene::mesh::Mesh;
use crate::scene::texture::{
    MinFilter, SamplerState, TexCoordSet, TextureBinding, TextureImage, TextureUsage,
};
use nalgebra::{Vector3, Vector4};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub enum RenderGeometry<'a> {
    Mesh(&'a Mesh),
    IndexedTriangle {
        vertices: &'a [Vertex],
        indices: [u32; 3],
        cache_vertices: bool,
    },
    Triangle([Vertex; 3]),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum VertexSourceKey {
    Mesh(usize),
    Vertices(usize),
}

struct CachedBackgroundTexture {
    path: PathBuf,
    use_mipmap: bool,
    binding: Arc<TextureBinding>,
}

type PreparedBatch<'a, V, S> =
    [Option<PreparedTriangle<'a, V, S, Option<&'a Material>>>; MAX_PREPARED_TRIANGLES];

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DrawTimings {
    pub preparation: Duration,
    pub rasterization: Duration,
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
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            commands: Vec::with_capacity(capacity),
            next_insertion_id: 0,
        }
    }

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
    pub texture: Option<&'a TextureBinding>,
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
    cached_background: Option<CachedBackgroundTexture>,
    shared_depth: Arc<Vec<f32>>,
}

impl Renderer {
    pub fn new(width: usize, height: usize, supersample_scale: usize) -> Result<Self, String> {
        Ok(Self {
            rasterizer: Rasterizer::new(),
            framebuffer: FrameBuffer::new(width, height, supersample_scale)?,
            cached_background: None,
            shared_depth: Arc::new(Vec::new()),
        })
    }

    pub(crate) fn background_texture(
        &mut self,
        path: &Path,
        use_mipmap: bool,
    ) -> Result<Arc<TextureBinding>, image::ImageError> {
        let cache_matches = self
            .cached_background
            .as_ref()
            .is_some_and(|cached| cached.path == path && cached.use_mipmap == use_mipmap);
        if !cache_matches {
            let image = TextureImage::load(path, use_mipmap)?;
            self.cached_background = Some(CachedBackgroundTexture {
                path: path.to_path_buf(),
                use_mipmap,
                binding: Arc::new(TextureBinding::new(
                    Arc::new(image),
                    SamplerState {
                        min_filter: MinFilter::LinearMipmapLinear,
                        ..Default::default()
                    },
                    TexCoordSet::TexCoord0,
                    TextureUsage::Color,
                )),
            });
        }
        Ok(Arc::clone(
            &self
                .cached_background
                .as_ref()
                .expect("background cache was populated")
                .binding,
        ))
    }

    pub(crate) fn shared_depth_values(&mut self) -> Arc<Vec<f32>> {
        self.framebuffer
            .copy_depth_values_into(Arc::make_mut(&mut self.shared_depth));
        Arc::clone(&self.shared_depth)
    }

    pub fn clear_with_options(&mut self, options: ClearOptions) {
        let width = self.framebuffer.buffer_width;
        let height = self.framebuffer.buffer_height;
        self.framebuffer.clear_with(options.depth, |x, y| {
            let u = x as f32 / width as f32;
            let v = y as f32 / height as f32;

            if let Some(texture) = options.texture {
                texture.sample(u, v).xyz()
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
        let _ = self.draw_queues_profiled(&[queue], shaders);
    }

    pub fn draw_queues<'a, S>(&mut self, queues: &[&RenderQueue<'a>], shaders: &'a [S])
    where
        S: Shader<Option<&'a Material>>,
    {
        let _ = self.draw_queues_profiled(queues, shaders);
    }

    pub fn draw_queue_profiled<'a, S>(
        &mut self,
        queue: &RenderQueue<'a>,
        shaders: &'a [S],
    ) -> DrawTimings
    where
        S: Shader<Option<&'a Material>>,
    {
        self.draw_queues_profiled(&[queue], shaders)
    }

    pub fn draw_queues_profiled<'a, S>(
        &mut self,
        queues: &[&RenderQueue<'a>],
        shaders: &'a [S],
    ) -> DrawTimings
    where
        S: Shader<Option<&'a Material>>,
    {
        let preparation_started = Instant::now();
        let width = self.framebuffer.buffer_width;
        let height = self.framebuffer.buffer_height;
        let mut vertex_sources = HashMap::new();
        for command in queues.iter().flat_map(|queue| queue.commands()) {
            let source = match &command.geometry {
                RenderGeometry::Mesh(mesh) if mesh.reuses_vertices() => Some((
                    VertexSourceKey::Mesh(*mesh as *const Mesh as usize),
                    &mesh.vertices[..],
                )),
                RenderGeometry::IndexedTriangle {
                    vertices,
                    cache_vertices: true,
                    ..
                } => Some((
                    VertexSourceKey::Vertices(vertices.as_ptr() as usize),
                    *vertices,
                )),
                RenderGeometry::Mesh(_)
                | RenderGeometry::IndexedTriangle {
                    cache_vertices: false,
                    ..
                }
                | RenderGeometry::Triangle(_) => None,
            };
            if let Some((source_key, vertices)) = source {
                let key = (command.shader_index, source_key);
                vertex_sources
                    .entry(key)
                    .or_insert((vertices, &shaders[command.shader_index]));
            }
        }
        let vertex_cache: HashMap<_, _> = vertex_sources
            .into_par_iter()
            .map(|(key, (vertices, shader))| {
                let transformed = vertices
                    .par_iter()
                    .map(|vertex| shader.vertex(vertex))
                    .collect::<Vec<_>>();
                (key, transformed)
            })
            .collect();

        let mut prepared: Vec<PreparedTriangle<'_, S::Varying, S, Option<&Material>>> = Vec::new();
        for command in queues.iter().flat_map(|queue| queue.commands()) {
            let shader = &shaders[command.shader_index];
            match &command.geometry {
                RenderGeometry::Mesh(mesh) => {
                    let command_triangles: Vec<_> = mesh
                        .indices
                        .par_chunks(3)
                        .flat_map_iter(|indices| {
                            let triangles = if indices.len() < 3 {
                                std::array::from_fn(|_| None)
                            } else if let Some(transformed) = vertex_cache.get(&(
                                command.shader_index,
                                VertexSourceKey::Mesh(*mesh as *const Mesh as usize),
                            )) {
                                self.prepare_shaded_vertices(
                                    width,
                                    height,
                                    [
                                        transformed[indices[0] as usize],
                                        transformed[indices[1] as usize],
                                        transformed[indices[2] as usize],
                                    ],
                                    shader,
                                    command.material,
                                    command.state,
                                )
                            } else {
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
                            };
                            triangles.into_iter().flatten()
                        })
                        .collect();
                    prepared.extend(command_triangles);
                }
                RenderGeometry::IndexedTriangle {
                    vertices,
                    indices,
                    cache_vertices,
                } => {
                    let triangles = if *cache_vertices {
                        let transformed = &vertex_cache[&(
                            command.shader_index,
                            VertexSourceKey::Vertices(vertices.as_ptr() as usize),
                        )];
                        self.prepare_shaded_vertices(
                            width,
                            height,
                            [
                                transformed[indices[0] as usize],
                                transformed[indices[1] as usize],
                                transformed[indices[2] as usize],
                            ],
                            shader,
                            command.material,
                            command.state,
                        )
                    } else {
                        self.prepare_vertices(
                            width,
                            height,
                            [
                                &vertices[indices[0] as usize],
                                &vertices[indices[1] as usize],
                                &vertices[indices[2] as usize],
                            ],
                            shader,
                            command.material,
                            command.state,
                        )
                    };
                    prepared.extend(triangles.into_iter().flatten());
                }
                RenderGeometry::Triangle(vertices) => {
                    prepared.extend(
                        self.prepare_vertices(
                            width,
                            height,
                            [&vertices[0], &vertices[1], &vertices[2]],
                            shader,
                            command.material,
                            command.state,
                        )
                        .into_iter()
                        .flatten(),
                    );
                }
            }
        }
        let preparation = preparation_started.elapsed();

        let rasterization_started = Instant::now();
        self.rasterizer
            .rasterize_prepared(&mut self.framebuffer, &prepared);
        DrawTimings {
            preparation,
            rasterization: rasterization_started.elapsed(),
        }
    }

    fn prepare_vertices<'a, S>(
        &self,
        width: usize,
        height: usize,
        vertices: [&Vertex; 3],
        shader: &'a S,
        material: Option<&'a Material>,
        state: RenderState,
    ) -> PreparedBatch<'a, S::Varying, S>
    where
        S: Shader<Option<&'a Material>>,
    {
        let (pos0, var0) = shader.vertex(vertices[0]);
        let (pos1, var1) = shader.vertex(vertices[1]);
        let (pos2, var2) = shader.vertex(vertices[2]);
        self.prepare_shaded_vertices(
            width,
            height,
            [(pos0, var0), (pos1, var1), (pos2, var2)],
            shader,
            material,
            state,
        )
    }

    fn prepare_shaded_vertices<'a, S>(
        &self,
        width: usize,
        height: usize,
        vertices: [(Vector4<f32>, S::Varying); 3],
        shader: &'a S,
        material: Option<&'a Material>,
        state: RenderState,
    ) -> PreparedBatch<'a, S::Varying, S>
    where
        S: Shader<Option<&'a Material>>,
    {
        self.rasterizer.prepare_triangle::<S, _>(
            (width, height),
            &[vertices[0].0, vertices[1].0, vertices[2].0],
            &[vertices[0].1, vertices[1].1, vertices[2].1],
            shader,
            state,
            material,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgba, RgbaImage};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn background_cache_reuses_matching_path_and_mip_policy() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should follow the Unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "rasterizer-background-cache-{}-{unique}.png",
            std::process::id()
        ));
        RgbaImage::from_pixel(1, 1, Rgba([1, 2, 3, 255]))
            .save(&path)
            .expect("test background should be writable");
        let mut renderer = Renderer::new(1, 1, 1).expect("test dimensions should be valid");

        let first = renderer
            .background_texture(&path, false)
            .expect("test background should load");
        let second = renderer
            .background_texture(&path, false)
            .expect("cached test background should load");
        assert!(Arc::ptr_eq(&first, &second));

        let mipmapped = renderer
            .background_texture(&path, true)
            .expect("test background should reload with mipmaps");
        assert!(!Arc::ptr_eq(&first, &mipmapped));

        std::fs::remove_file(path).expect("test background should be removable");
    }

    #[test]
    fn shadow_depth_storage_is_reused_after_consumers_release_it() {
        let mut renderer = Renderer::new(2, 2, 1).expect("test dimensions should be valid");
        let first = renderer.shared_depth_values();
        let allocation = Arc::as_ptr(&first);
        drop(first);

        let second = renderer.shared_depth_values();

        assert_eq!(Arc::as_ptr(&second), allocation);
        assert_eq!(second.len(), 4);
    }
}
