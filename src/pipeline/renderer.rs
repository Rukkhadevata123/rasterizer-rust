use crate::core::framebuffer::FrameBuffer;
use crate::core::geometry::Vertex;
use crate::core::pipeline_state::GraphicsPipelineState;
use crate::core::rasterizer::{MAX_PREPARED_TRIANGLES, PreparedTriangle, Rasterizer};
use crate::core::shader::Shader;
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
    pub backend_preparation: Duration,
    pub rasterization: Duration,
    /// Inclusive synchronous draw duration. Backend preparation and rasterization are nested
    /// within this value and must not be added to it when computing a total.
    pub submission_total: Duration,
}

pub struct DrawPacket<'a> {
    pub insertion_id: u64,
    pub shader_index: usize,
    pub geometry: RenderGeometry<'a>,
    pub material: Option<&'a Material>,
    pub state: GraphicsPipelineState,
    pub sort_depth: f32,
}

#[derive(Default)]
pub struct RenderPhase<'a> {
    commands: Vec<DrawPacket<'a>>,
    next_insertion_id: u64,
}

impl<'a> RenderPhase<'a> {
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
        state: GraphicsPipelineState,
        sort_depth: f32,
    ) {
        let insertion_id = self.next_insertion_id;
        self.next_insertion_id += 1;
        self.commands.push(DrawPacket {
            insertion_id,
            shader_index,
            geometry,
            material,
            state,
            sort_depth,
        });
    }

    /// Sorts transparent work back-to-front for the renderer's view-space convention.
    ///
    /// Visible view-space Z values are negative, so ascending Z visits farther draws first.
    /// Insertion IDs make equal-depth draws deterministic. Later preparation, clipping, and
    /// band binning must preserve this resulting order for alpha blending to remain correct.
    pub fn sort_transparent(&mut self) {
        self.commands.sort_by(|a, b| {
            a.sort_depth
                .total_cmp(&b.sort_depth)
                .then_with(|| a.insertion_id.cmp(&b.insertion_id))
        });
    }

    pub fn commands(&self) -> &[DrawPacket<'a>] {
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

pub struct RenderTarget {
    framebuffer: FrameBuffer,
}

impl RenderTarget {
    pub fn new(width: usize, height: usize, supersample_scale: usize) -> Result<Self, String> {
        Ok(Self {
            framebuffer: FrameBuffer::new(width, height, supersample_scale)?,
        })
    }

    pub fn framebuffer(&self) -> &FrameBuffer {
        &self.framebuffer
    }

    fn framebuffer_mut(&mut self) -> &mut FrameBuffer {
        &mut self.framebuffer
    }
}

pub struct FrameResources {
    cached_background: Option<CachedBackgroundTexture>,
    shadow_snapshot: Arc<Vec<f32>>,
}

impl Default for FrameResources {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameResources {
    pub fn new() -> Self {
        Self {
            cached_background: None,
            shadow_snapshot: Arc::new(Vec::new()),
        }
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

    pub(crate) fn shadow_depth_snapshot(&mut self, target: &RenderTarget) -> Arc<Vec<f32>> {
        target
            .framebuffer()
            .copy_depth_values_into(Arc::make_mut(&mut self.shadow_snapshot));
        Arc::clone(&self.shadow_snapshot)
    }
}

pub struct SoftwareRasterBackend {
    rasterizer: Rasterizer,
}

impl Default for SoftwareRasterBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl SoftwareRasterBackend {
    pub fn new() -> Self {
        Self {
            rasterizer: Rasterizer::new(),
        }
    }

    pub fn clear_with_options(&mut self, target: &mut RenderTarget, options: ClearOptions) {
        let width = target.framebuffer().buffer_width;
        let height = target.framebuffer().buffer_height;
        target.framebuffer_mut().clear_with(options.depth, |x, y| {
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

    pub fn draw_phase<'a, S>(
        &mut self,
        target: &mut RenderTarget,
        phase: &RenderPhase<'a>,
        shaders: &'a [S],
    ) where
        S: Shader<Option<&'a Material>>,
    {
        let _ = self.draw_phases_profiled(target, &[phase], shaders);
    }

    pub fn draw_phases<'a, S>(
        &mut self,
        target: &mut RenderTarget,
        phases: &[&RenderPhase<'a>],
        shaders: &'a [S],
    ) where
        S: Shader<Option<&'a Material>>,
    {
        let _ = self.draw_phases_profiled(target, phases, shaders);
    }

    pub fn draw_phase_profiled<'a, S>(
        &mut self,
        target: &mut RenderTarget,
        phase: &RenderPhase<'a>,
        shaders: &'a [S],
    ) -> DrawTimings
    where
        S: Shader<Option<&'a Material>>,
    {
        self.draw_phases_profiled(target, &[phase], shaders)
    }

    pub fn draw_phases_profiled<'a, S>(
        &mut self,
        target: &mut RenderTarget,
        phases: &[&RenderPhase<'a>],
        shaders: &'a [S],
    ) -> DrawTimings
    where
        S: Shader<Option<&'a Material>>,
    {
        let submission_started = Instant::now();
        let preparation_started = Instant::now();
        let width = target.framebuffer().buffer_width;
        let height = target.framebuffer().buffer_height;
        let commands: Vec<_> = phases.iter().flat_map(|phase| phase.commands()).collect();
        let mut vertex_sources = HashMap::new();
        for command in &commands {
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

        let prepare_draw_packet_triangle =
            |command: &DrawPacket<'a>, local_triangle_index: usize| {
                let shader = &shaders[command.shader_index];
                match &command.geometry {
                    RenderGeometry::Mesh(mesh) => {
                        let index_offset = local_triangle_index * 3;
                        let indices =
                            &mesh.indices[index_offset..(index_offset + 3).min(mesh.indices.len())];
                        if indices.len() < 3 {
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
                        }
                    }
                    RenderGeometry::IndexedTriangle {
                        vertices,
                        indices,
                        cache_vertices,
                    } => {
                        if *cache_vertices {
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
                        }
                    }
                    RenderGeometry::Triangle(vertices) => self.prepare_vertices(
                        width,
                        height,
                        [&vertices[0], &vertices[1], &vertices[2]],
                        shader,
                        command.material,
                        command.state,
                    ),
                }
            };
        let contains_mesh = commands
            .iter()
            .any(|command| matches!(command.geometry, RenderGeometry::Mesh(_)));
        let prepared: Vec<PreparedTriangle<'_, S::Varying, S, Option<&Material>>> = if contains_mesh
        {
            let mut triangle_ends = Vec::with_capacity(commands.len());
            let mut triangle_count = 0;
            for command in &commands {
                triangle_count += match &command.geometry {
                    RenderGeometry::Mesh(mesh) => mesh.indices.len().div_ceil(3),
                    RenderGeometry::IndexedTriangle { .. } | RenderGeometry::Triangle(_) => 1,
                };
                triangle_ends.push(triangle_count);
            }
            // This indexed parallel traversal and collection preserve source triangle order.
            // Each clipped fan is emitted in order as well, so transparent blending observes
            // the command order established before preparation regardless of worker count.
            (0..triangle_count)
                .into_par_iter()
                .flat_map_iter(|triangle_index| {
                    let command_index = triangle_ends.partition_point(|&end| end <= triangle_index);
                    let command_start = command_index
                        .checked_sub(1)
                        .map_or(0, |previous| triangle_ends[previous]);
                    prepare_draw_packet_triangle(
                        commands[command_index],
                        triangle_index - command_start,
                    )
                    .into_iter()
                    .flatten()
                })
                .collect()
        } else {
            commands
                .iter()
                .flat_map(|command| {
                    prepare_draw_packet_triangle(command, 0)
                        .into_iter()
                        .flatten()
                })
                .collect()
        };
        let backend_preparation = preparation_started.elapsed();

        let rasterization_started = Instant::now();
        self.rasterizer
            .rasterize_prepared(target.framebuffer_mut(), &prepared);
        let rasterization = rasterization_started.elapsed();
        DrawTimings {
            backend_preparation,
            rasterization,
            submission_total: submission_started.elapsed(),
        }
    }

    fn prepare_vertices<'a, S>(
        &self,
        width: usize,
        height: usize,
        vertices: [&Vertex; 3],
        shader: &'a S,
        material: Option<&'a Material>,
        state: GraphicsPipelineState,
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
        state: GraphicsPipelineState,
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
        let mut resources = FrameResources::new();

        let first = resources
            .background_texture(&path, false)
            .expect("test background should load");
        let second = resources
            .background_texture(&path, false)
            .expect("cached test background should load");
        assert!(Arc::ptr_eq(&first, &second));

        let mipmapped = resources
            .background_texture(&path, true)
            .expect("test background should reload with mipmaps");
        assert!(!Arc::ptr_eq(&first, &mipmapped));

        std::fs::remove_file(path).expect("test background should be removable");
    }

    #[test]
    fn frame_resources_reuse_shadow_storage_across_target_rebuilds() {
        let mut resources = FrameResources::new();
        let target = RenderTarget::new(2, 2, 1).expect("test dimensions should be valid");
        let first = resources.shadow_depth_snapshot(&target);
        let allocation = Arc::as_ptr(&first);
        drop(first);

        let target = RenderTarget::new(2, 2, 1).expect("rebuilt dimensions should be valid");
        let second = resources.shadow_depth_snapshot(&target);

        assert_eq!(Arc::as_ptr(&second), allocation);
        assert_eq!(second.len(), 4);
    }
}
