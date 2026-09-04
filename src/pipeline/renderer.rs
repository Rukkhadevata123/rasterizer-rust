use crate::core::framebuffer::FrameBuffer;
use crate::core::geometry::Vertex;
use crate::core::pipeline_state::{GraphicsPipeline, GraphicsPipelineState};
use crate::core::rasterizer::{MAX_PREPARED_TRIANGLES, PreparedTriangle, Rasterizer};
use crate::core::shader::Shader;
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
use thiserror::Error;

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
pub struct ObjectBindingId(usize);

impl ObjectBindingId {
    pub fn from_pass_index(index: usize) -> Self {
        Self(index)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum VertexSourceKey {
    Mesh(usize),
    Vertices(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct VertexCacheKey {
    vertex_program_id: crate::core::pipeline_state::VertexProgramId,
    object_binding_id: ObjectBindingId,
    source: VertexSourceKey,
}

struct CachedBackgroundTexture {
    path: PathBuf,
    use_mipmap: bool,
    binding: Arc<TextureBinding>,
}

type PreparedBatch<'shader, V, S, C> =
    [Option<PreparedTriangle<'shader, V, S, C>>; MAX_PREPARED_TRIANGLES];

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct BackendExecutionTimings {
    backend_preparation: Duration,
    rasterization: Duration,
    /// Inclusive synchronous backend execution duration. Preparation and rasterization are nested
    /// within this value and must not be added to it when computing a total.
    submission_total: Duration,
}

struct DrawPacket<'a, S, C> {
    insertion_id: u64,
    pipeline: &'a GraphicsPipeline<S>,
    geometry: RenderGeometry<'a>,
    draw_context: C,
    object_binding_id: ObjectBindingId,
    sort_depth: f32,
}

struct RenderPhase<'a, S, C> {
    commands: Vec<DrawPacket<'a, S, C>>,
    next_insertion_id: u64,
}

impl<'a, S, C> Default for RenderPhase<'a, S, C> {
    fn default() -> Self {
        Self::with_capacity(0)
    }
}

impl<'a, S, C> RenderPhase<'a, S, C> {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            commands: Vec::with_capacity(capacity),
            next_insertion_id: 0,
        }
    }

    fn push(
        &mut self,
        pipeline: &'a GraphicsPipeline<S>,
        geometry: RenderGeometry<'a>,
        draw_context: C,
        object_binding_id: ObjectBindingId,
        sort_depth: f32,
    ) {
        let insertion_id = self.next_insertion_id;
        self.next_insertion_id += 1;
        self.commands.push(DrawPacket {
            insertion_id,
            pipeline,
            geometry,
            draw_context,
            object_binding_id,
            sort_depth,
        });
    }

    fn reserve(&mut self, additional: usize) {
        self.commands.reserve(additional);
    }

    fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// Sorts transparent work back-to-front for the renderer's view-space convention.
    ///
    /// Visible view-space Z values are negative, so ascending Z visits farther draws first.
    /// Insertion IDs make equal-depth draws deterministic. Later preparation, clipping, and
    /// band binning must preserve this resulting order for alpha blending to remain correct.
    fn sort_transparent(&mut self) {
        self.commands.sort_by(|a, b| {
            a.sort_depth
                .total_cmp(&b.sort_depth)
                .then_with(|| a.insertion_id.cmp(&b.insertion_id))
        });
    }

    fn commands(&self) -> &[DrawPacket<'a, S, C>] {
        &self.commands
    }
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LoadOp<T> {
    Load,
    Clear(T),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Operations<T> {
    pub load: LoadOp<T>,
}

pub struct RenderPassDescriptor<'a> {
    pub label: Option<&'a str>,
    pub target: &'a mut RenderTarget,
    pub color_ops: Option<Operations<Vector3<f32>>>,
    pub depth_ops: Option<Operations<f32>>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub(crate) enum RenderPassError {
    #[error(
        "render pass '{label}' does not declare a color attachment, depth attachment, or background"
    )]
    EmptyPass { label: String },
    #[error("render pass '{label}' has an invalid target: {reason}")]
    InvalidTarget { label: String, reason: String },
}

impl RenderPassDescriptor<'_> {
    pub(crate) fn validate(&self, has_background: bool) -> Result<(), RenderPassError> {
        let label = || self.label.unwrap_or("<unnamed>").to_owned();
        self.target
            .framebuffer()
            .validate_layout()
            .map_err(|reason| RenderPassError::InvalidTarget {
                label: label(),
                reason,
            })?;
        if self.color_ops.is_none() && self.depth_ops.is_none() && !has_background {
            return Err(RenderPassError::EmptyPass { label: label() });
        }

        Ok(())
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum CommandError {
    #[error("command encoder '{encoder}' already has an active render pass")]
    PassAlreadyActive { encoder: String },
    #[error("command encoder '{encoder}' already contains a finished render pass")]
    PassAlreadyRecorded { encoder: String },
    #[error("command encoder '{encoder}' cannot finish while a render pass is active")]
    PassNotEnded { encoder: String },
    #[error("command encoder '{encoder}' contains no render pass")]
    MissingPass { encoder: String },
    #[error("render pass '{pass}' cannot draw without a selected pipeline")]
    MissingPipeline { pass: String },
    #[error("render pass '{pass}' cannot draw without typed draw bindings")]
    MissingBindings { pass: String },
    #[error("render pass '{pass}' is invalid: {reason}")]
    InvalidPass { pass: String, reason: String },
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum RenderError {
    #[error(transparent)]
    Command(#[from] CommandError),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhaseSubmissionReport {
    pub label: String,
    pub backend_preparation: Duration,
    pub rasterization: Duration,
    pub execution_total: Duration,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SubmissionReport {
    pub attachment_processing: Duration,
    pub backend_preparation: Duration,
    pub rasterization: Duration,
    /// Inclusive synchronous queue submission duration, including attachment processing and every
    /// recorded phase. Nested timing fields must not be added to this value.
    pub submission_total: Duration,
    pub phases: Vec<PhaseSubmissionReport>,
}

impl SubmissionReport {
    pub fn phase(&self, label: &str) -> Option<&PhaseSubmissionReport> {
        self.phases.iter().find(|phase| phase.label == label)
    }
}

#[derive(Default)]
pub struct RenderDevice;

impl RenderDevice {
    pub fn new() -> Self {
        Self
    }

    pub fn create_command_encoder<'a, S, C>(
        &self,
        label: impl Into<String>,
    ) -> CommandEncoder<'a, S, C> {
        CommandEncoder {
            label: label.into(),
            pass_active: false,
            pass: None,
        }
    }

    /// Creates a submission queue that owns its software execution backend.
    pub fn create_queue(&self) -> GraphicsQueue {
        GraphicsQueue {
            backend: SoftwareRasterBackend::new(),
        }
    }
}

pub struct CommandEncoder<'a, S, C> {
    label: String,
    pass_active: bool,
    pass: Option<EncodedRenderPass<'a, S, C>>,
}

impl<'a, S, C> CommandEncoder<'a, S, C> {
    pub fn begin_render_pass<'encoder>(
        &'encoder mut self,
        descriptor: RenderPassDescriptor<'a>,
        background: Option<BackgroundPass<'a>>,
    ) -> Result<RenderPassEncoder<'encoder, 'a, S, C>, CommandError> {
        if self.pass_active {
            return Err(CommandError::PassAlreadyActive {
                encoder: self.label.clone(),
            });
        }
        if self.pass.is_some() {
            return Err(CommandError::PassAlreadyRecorded {
                encoder: self.label.clone(),
            });
        }
        descriptor
            .validate(background.is_some())
            .map_err(|error| CommandError::InvalidPass {
                pass: descriptor.label.unwrap_or("<unnamed>").to_owned(),
                reason: error.to_string(),
            })?;
        let RenderPassDescriptor {
            label,
            target,
            color_ops,
            depth_ops,
        } = descriptor;
        self.pass_active = true;
        Ok(RenderPassEncoder {
            parent: self,
            label: label.unwrap_or("<unnamed>").to_owned(),
            target: Some(target),
            color_ops,
            depth_ops,
            background,
            phases: Vec::new(),
            phase: RenderPhase::default(),
            pipeline: None,
            bindings: None,
            ended: false,
        })
    }

    pub fn finish(mut self) -> Result<CommandBuffer<'a, S, C>, CommandError> {
        if self.pass_active {
            return Err(CommandError::PassNotEnded {
                encoder: self.label,
            });
        }
        let pass = self.pass.take().ok_or_else(|| CommandError::MissingPass {
            encoder: self.label.clone(),
        })?;
        Ok(CommandBuffer {
            label: self.label,
            pass,
        })
    }
}

pub struct RenderPassEncoder<'encoder, 'a, S, C> {
    parent: &'encoder mut CommandEncoder<'a, S, C>,
    label: String,
    target: Option<&'a mut RenderTarget>,
    color_ops: Option<Operations<Vector3<f32>>>,
    depth_ops: Option<Operations<f32>>,
    background: Option<BackgroundPass<'a>>,
    phases: Vec<EncodedPhase<'a, S, C>>,
    phase: RenderPhase<'a, S, C>,
    pipeline: Option<&'a GraphicsPipeline<S>>,
    bindings: Option<(C, ObjectBindingId)>,
    ended: bool,
}

impl<'encoder, 'a, S, C> RenderPassEncoder<'encoder, 'a, S, C>
where
    C: Copy,
{
    pub fn set_pipeline(&mut self, pipeline: &'a GraphicsPipeline<S>) {
        self.pipeline = Some(pipeline);
    }

    pub fn set_draw_bindings(&mut self, context: C, object_binding_id: ObjectBindingId) {
        self.bindings = Some((context, object_binding_id));
    }

    pub fn reserve_draws(&mut self, additional: usize) {
        self.phase.reserve(additional);
    }

    pub fn draw_mesh(&mut self, mesh: &'a Mesh, sort_depth: f32) -> Result<(), CommandError> {
        self.draw(RenderGeometry::Mesh(mesh), sort_depth)
    }

    pub fn draw(
        &mut self,
        geometry: RenderGeometry<'a>,
        sort_depth: f32,
    ) -> Result<(), CommandError> {
        let pipeline = self.pipeline.ok_or_else(|| CommandError::MissingPipeline {
            pass: self.label.clone(),
        })?;
        let (context, object_binding_id) =
            self.bindings.ok_or_else(|| CommandError::MissingBindings {
                pass: self.label.clone(),
            })?;
        self.phase
            .push(pipeline, geometry, context, object_binding_id, sort_depth);
        Ok(())
    }

    pub fn sort_transparent(&mut self) {
        self.phase.sort_transparent();
    }

    pub fn finish_phase(&mut self, label: impl Into<String>) {
        self.phases.push(EncodedPhase {
            label: label.into(),
            phase: std::mem::take(&mut self.phase),
        });
    }

    pub fn end(mut self) -> Result<(), CommandError> {
        if self.phases.is_empty() || !self.phase.is_empty() {
            self.phases.push(EncodedPhase {
                label: self.label.clone(),
                phase: std::mem::take(&mut self.phase),
            });
        }
        let pass = EncodedRenderPass {
            target: self
                .target
                .take()
                .expect("active render pass retains its target"),
            color_ops: self.color_ops,
            depth_ops: self.depth_ops,
            background: self.background.take(),
            phases: std::mem::take(&mut self.phases),
        };
        self.parent.pass = Some(pass);
        self.parent.pass_active = false;
        self.ended = true;
        Ok(())
    }
}

struct EncodedPhase<'a, S, C> {
    label: String,
    phase: RenderPhase<'a, S, C>,
}

struct EncodedRenderPass<'a, S, C> {
    target: &'a mut RenderTarget,
    color_ops: Option<Operations<Vector3<f32>>>,
    depth_ops: Option<Operations<f32>>,
    background: Option<BackgroundPass<'a>>,
    phases: Vec<EncodedPhase<'a, S, C>>,
}

pub struct CommandBuffer<'a, S, C> {
    label: String,
    pass: EncodedRenderPass<'a, S, C>,
}

impl<S, C> CommandBuffer<'_, S, C> {
    pub fn label(&self) -> &str {
        &self.label
    }
}

pub struct GraphicsQueue {
    backend: SoftwareRasterBackend,
}

impl GraphicsQueue {
    /// Executes all recorded work synchronously and returns only after rasterization completes.
    pub fn submit<'a, S, C>(
        &mut self,
        command_buffer: CommandBuffer<'a, S, C>,
    ) -> Result<SubmissionReport, RenderError>
    where
        S: Shader<C>,
        C: Copy + Send + Sync,
    {
        let submission_started = Instant::now();
        let EncodedRenderPass {
            target,
            color_ops,
            depth_ops,
            background,
            phases,
        } = command_buffer.pass;
        let attachment_started = Instant::now();
        self.backend
            .process_attachments(target, color_ops, depth_ops, background);
        let attachment_processing = attachment_started.elapsed();
        let mut phase_reports = Vec::with_capacity(phases.len());
        let mut backend_preparation = Duration::ZERO;
        let mut rasterization = Duration::ZERO;
        for EncodedPhase { label, phase } in phases {
            let timings = self.backend.execute_phase_profiled(target, &phase);
            backend_preparation += timings.backend_preparation;
            rasterization += timings.rasterization;
            phase_reports.push(PhaseSubmissionReport {
                label,
                backend_preparation: timings.backend_preparation,
                rasterization: timings.rasterization,
                execution_total: timings.submission_total,
            });
        }
        Ok(SubmissionReport {
            attachment_processing,
            backend_preparation,
            rasterization,
            submission_total: submission_started.elapsed(),
            phases: phase_reports,
        })
    }
}
pub enum BackgroundSource<'a> {
    Gradient {
        top: Vector3<f32>,
        bottom: Vector3<f32>,
    },
    Texture(&'a TextureBinding),
}

impl BackgroundSource<'_> {
    fn color_at(&self, x: usize, y: usize, width: usize, height: usize) -> Vector3<f32> {
        let u = x as f32 / width as f32;
        let v = y as f32 / height as f32;
        match self {
            Self::Gradient { top, bottom } => top.lerp(bottom, v),
            Self::Texture(texture) => texture.sample(u, v).xyz(),
        }
    }
}

pub struct BackgroundPass<'a> {
    pub source: BackgroundSource<'a>,
}

pub struct RenderTarget {
    framebuffer: FrameBuffer,
}

/// Read-only access to resolved pixels and individual render samples.
#[derive(Clone, Copy)]
pub struct RenderTargetReadback<'a> {
    framebuffer: &'a FrameBuffer,
}

impl RenderTargetReadback<'_> {
    /// Width of the resolved output in pixels.
    pub fn width(&self) -> usize {
        self.framebuffer.width
    }

    /// Height of the resolved output in pixels.
    pub fn height(&self) -> usize {
        self.framebuffer.height
    }

    /// Linear supersampling scale used by the render target.
    pub fn supersample_scale(&self) -> usize {
        self.framebuffer.supersample_scale
    }

    /// Width of the underlying render-sample grid.
    pub fn sample_width(&self) -> usize {
        self.framebuffer.buffer_width
    }

    /// Height of the underlying render-sample grid.
    pub fn sample_height(&self) -> usize {
        self.framebuffer.buffer_height
    }

    /// Returns the resolved HDR color at an output pixel.
    pub fn color(&self, x: usize, y: usize) -> Option<Vector3<f32>> {
        self.framebuffer.get_pixel(x, y)
    }

    /// Returns the HDR color at one location in the render-sample grid.
    pub fn sample_color(&self, x: usize, y: usize) -> Option<Vector3<f32>> {
        self.framebuffer.sample(x, y).map(|sample| sample.color)
    }

    /// Returns the depth at one location in the render-sample grid.
    pub fn sample_depth(&self, x: usize, y: usize) -> Option<f32> {
        self.framebuffer.sample(x, y).map(|sample| sample.depth)
    }

    /// Copies the complete depth attachment in row-major sample-grid order.
    pub fn depth_values(&self) -> Vec<f32> {
        self.framebuffer.depth_values()
    }
}

impl RenderTarget {
    pub fn new(width: usize, height: usize, supersample_scale: usize) -> Result<Self, String> {
        Ok(Self {
            framebuffer: FrameBuffer::new(width, height, supersample_scale)?,
        })
    }

    pub fn readback(&self) -> RenderTargetReadback<'_> {
        RenderTargetReadback {
            framebuffer: &self.framebuffer,
        }
    }

    pub(crate) fn framebuffer(&self) -> &FrameBuffer {
        &self.framebuffer
    }

    fn framebuffer_mut(&mut self) -> &mut FrameBuffer {
        &mut self.framebuffer
    }
}

pub struct MainHdrTarget {
    target: RenderTarget,
}

impl MainHdrTarget {
    pub fn new(width: usize, height: usize, supersample_scale: usize) -> Result<Self, String> {
        Ok(Self {
            target: RenderTarget::new(width, height, supersample_scale)?,
        })
    }

    pub fn readback(&self) -> RenderTargetReadback<'_> {
        self.target.readback()
    }

    #[cfg(test)]
    pub(crate) fn framebuffer_mut(&mut self) -> &mut FrameBuffer {
        self.target.framebuffer_mut()
    }

    pub fn render_target_mut(&mut self) -> &mut RenderTarget {
        &mut self.target
    }
}

pub struct PresentBuffer {
    width: usize,
    height: usize,
    pixels: Vec<u32>,
}

impl PresentBuffer {
    pub fn new(width: usize, height: usize) -> Result<Self, String> {
        if width == 0 || height == 0 {
            return Err("present dimensions must be greater than zero".to_string());
        }
        let pixel_count = width
            .checked_mul(height)
            .ok_or_else(|| "present buffer pixel count overflows usize".to_string())?;
        pixel_count
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| "present buffer allocation size overflows usize".to_string())?;

        Ok(Self {
            width,
            height,
            pixels: vec![0; pixel_count],
        })
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn pixels(&self) -> &[u32] {
        &self.pixels
    }

    pub(crate) fn pixels_mut(&mut self) -> &mut [u32] {
        &mut self.pixels
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

struct SoftwareRasterBackend {
    rasterizer: Rasterizer,
}

impl Default for SoftwareRasterBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl SoftwareRasterBackend {
    fn new() -> Self {
        Self {
            rasterizer: Rasterizer::new(),
        }
    }

    fn process_attachments(
        &mut self,
        target: &mut RenderTarget,
        color_ops: Option<Operations<Vector3<f32>>>,
        depth_ops: Option<Operations<f32>>,
        background: Option<BackgroundPass<'_>>,
    ) {
        let color_load = color_ops.map(|operations| operations.load);
        let depth_load = depth_ops.map(|operations| operations.load);

        if let Some(background) = background {
            let width = target.framebuffer().buffer_width;
            let height = target.framebuffer().buffer_height;
            if let Some(LoadOp::Clear(depth)) = depth_load {
                target.framebuffer_mut().clear_with(depth, |x, y| {
                    background.source.color_at(x, y, width, height)
                });
            } else {
                target
                    .framebuffer_mut()
                    .fill_color_with(|x, y| background.source.color_at(x, y, width, height));
            }
            return;
        }

        match (color_load, depth_load) {
            (Some(LoadOp::Clear(color)), Some(LoadOp::Clear(depth))) => {
                target.framebuffer_mut().clear_with(depth, |_, _| color);
            }
            (Some(LoadOp::Clear(color)), _) => {
                target.framebuffer_mut().clear_color(color);
            }
            (_, Some(LoadOp::Clear(depth))) => {
                target.framebuffer_mut().clear_depth(depth);
            }
            _ => {}
        }
    }

    fn execute_phase_profiled<'a, S, C>(
        &mut self,
        target: &mut RenderTarget,
        phase: &RenderPhase<'a, S, C>,
    ) -> BackendExecutionTimings
    where
        S: Shader<C>,
        C: Copy + Send + Sync,
    {
        let submission_started = Instant::now();
        let preparation_started = Instant::now();
        let width = target.framebuffer().buffer_width;
        let height = target.framebuffer().buffer_height;
        let commands = phase.commands();
        let mut vertex_sources = HashMap::new();
        for command in commands {
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
            if let Some((source, vertices)) = source {
                let key = VertexCacheKey {
                    vertex_program_id: command.pipeline.vertex_program_id(),
                    object_binding_id: command.object_binding_id,
                    source,
                };
                vertex_sources.entry(key).or_insert((
                    vertices,
                    command.pipeline.shader(),
                    command.draw_context,
                ));
            }
        }
        let vertex_cache: HashMap<_, _> = vertex_sources
            .into_par_iter()
            .map(|(key, (vertices, shader, draw_context))| {
                let transformed = vertices
                    .par_iter()
                    .map(|vertex| shader.vertex(vertex, draw_context))
                    .collect::<Vec<_>>();
                (key, transformed)
            })
            .collect();

        let prepare_draw_packet_triangle =
            |command: &DrawPacket<'a, S, C>, local_triangle_index: usize| {
                let shader = command.pipeline.shader();
                match &command.geometry {
                    RenderGeometry::Mesh(mesh) => {
                        let index_offset = local_triangle_index * 3;
                        let indices =
                            &mesh.indices[index_offset..(index_offset + 3).min(mesh.indices.len())];
                        if indices.len() < 3 {
                            std::array::from_fn(|_| None)
                        } else if let Some(transformed) = vertex_cache.get(&VertexCacheKey {
                            vertex_program_id: command.pipeline.vertex_program_id(),
                            object_binding_id: command.object_binding_id,
                            source: VertexSourceKey::Mesh(*mesh as *const Mesh as usize),
                        }) {
                            self.prepare_shaded_vertices(
                                width,
                                height,
                                [
                                    transformed[indices[0] as usize],
                                    transformed[indices[1] as usize],
                                    transformed[indices[2] as usize],
                                ],
                                shader,
                                command.draw_context,
                                command.pipeline.state(),
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
                                command.draw_context,
                                command.pipeline.state(),
                            )
                        }
                    }
                    RenderGeometry::IndexedTriangle {
                        vertices,
                        indices,
                        cache_vertices,
                    } => {
                        if *cache_vertices {
                            let transformed = &vertex_cache[&VertexCacheKey {
                                vertex_program_id: command.pipeline.vertex_program_id(),
                                object_binding_id: command.object_binding_id,
                                source: VertexSourceKey::Vertices(vertices.as_ptr() as usize),
                            }];
                            self.prepare_shaded_vertices(
                                width,
                                height,
                                [
                                    transformed[indices[0] as usize],
                                    transformed[indices[1] as usize],
                                    transformed[indices[2] as usize],
                                ],
                                shader,
                                command.draw_context,
                                command.pipeline.state(),
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
                                command.draw_context,
                                command.pipeline.state(),
                            )
                        }
                    }
                    RenderGeometry::Triangle(vertices) => self.prepare_vertices(
                        width,
                        height,
                        [&vertices[0], &vertices[1], &vertices[2]],
                        shader,
                        command.draw_context,
                        command.pipeline.state(),
                    ),
                }
            };
        let contains_mesh = commands
            .iter()
            .any(|command| matches!(command.geometry, RenderGeometry::Mesh(_)));
        let prepared: Vec<PreparedTriangle<'_, S::Varying, S, C>> = if contains_mesh {
            let mut triangle_ends = Vec::with_capacity(commands.len());
            let mut triangle_count = 0;
            for command in commands {
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
                        &commands[command_index],
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
        BackendExecutionTimings {
            backend_preparation,
            rasterization,
            submission_total: submission_started.elapsed(),
        }
    }

    fn prepare_vertices<'shader, S, C>(
        &self,
        width: usize,
        height: usize,
        vertices: [&Vertex; 3],
        shader: &'shader S,
        draw_context: C,
        state: GraphicsPipelineState,
    ) -> PreparedBatch<'shader, S::Varying, S, C>
    where
        S: Shader<C>,
        C: Copy + Send + Sync,
    {
        let (pos0, var0) = shader.vertex(vertices[0], draw_context);
        let (pos1, var1) = shader.vertex(vertices[1], draw_context);
        let (pos2, var2) = shader.vertex(vertices[2], draw_context);
        self.prepare_shaded_vertices(
            width,
            height,
            [(pos0, var0), (pos1, var1), (pos2, var2)],
            shader,
            draw_context,
            state,
        )
    }

    fn prepare_shaded_vertices<'shader, S, C>(
        &self,
        width: usize,
        height: usize,
        vertices: [(Vector4<f32>, S::Varying); 3],
        shader: &'shader S,
        draw_context: C,
        state: GraphicsPipelineState,
    ) -> PreparedBatch<'shader, S::Varying, S, C>
    where
        S: Shader<C>,
        C: Copy + Send + Sync,
    {
        self.rasterizer.prepare_triangle::<S, _>(
            (width, height),
            &[vertices[0].0, vertices[1].0, vertices[2].0],
            &[vertices[0].1, vertices[1].1, vertices[2].1],
            shader,
            state,
            draw_context,
        )
    }
}

#[cfg(test)]
#[path = "renderer/backend_tests.rs"]
mod backend_tests;

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

    #[test]
    fn depth_only_pass_clear_preserves_color() {
        let mut backend = SoftwareRasterBackend::new();
        let mut target = RenderTarget::new(2, 2, 1).expect("test dimensions should be valid");
        let original_color = Vector3::new(0.25, 0.5, 0.75);
        target
            .framebuffer_mut()
            .clear_with(0.125, |_, _| original_color);

        backend.process_attachments(
            &mut target,
            None,
            Some(Operations {
                load: LoadOp::Clear(0.875),
            }),
            None,
        );

        for y in 0..2 {
            for x in 0..2 {
                let sample = target
                    .framebuffer()
                    .sample(x, y)
                    .expect("sample should be in bounds");
                assert_eq!(sample.color, original_color);
                assert_eq!(sample.depth, 0.875);
            }
        }
    }

    #[test]
    fn color_only_pass_clear_preserves_depth() {
        let mut backend = SoftwareRasterBackend::new();
        let mut target = RenderTarget::new(2, 2, 1).expect("test dimensions should be valid");
        let clear_color = Vector3::new(0.75, 0.5, 0.25);
        target
            .framebuffer_mut()
            .clear_with(0.125, |_, _| Vector3::zeros());

        backend.process_attachments(
            &mut target,
            Some(Operations {
                load: LoadOp::Clear(clear_color),
            }),
            None,
            None,
        );

        for y in 0..2 {
            for x in 0..2 {
                let sample = target
                    .framebuffer()
                    .sample(x, y)
                    .expect("sample should be in bounds");
                assert_eq!(sample.color, clear_color);
                assert_eq!(sample.depth, 0.125);
            }
        }
    }

    #[test]
    fn load_operations_preserve_existing_attachments() {
        let mut backend = SoftwareRasterBackend::new();
        let mut target = RenderTarget::new(1, 1, 1).expect("test dimensions should be valid");
        let original_color = Vector3::new(0.2, 0.4, 0.6);
        target
            .framebuffer_mut()
            .clear_with(0.25, |_, _| original_color);

        backend.process_attachments(
            &mut target,
            Some(Operations { load: LoadOp::Load }),
            Some(Operations { load: LoadOp::Load }),
            None,
        );

        let sample = target
            .framebuffer()
            .sample(0, 0)
            .expect("sample should be in bounds");
        assert_eq!(sample.color, original_color);
        assert_eq!(sample.depth, 0.25);
    }

    #[test]
    fn background_pass_fuses_gradient_fill_with_depth_clear() {
        let mut backend = SoftwareRasterBackend::new();
        let mut target = RenderTarget::new(1, 2, 1).expect("test dimensions should be valid");
        let top = Vector3::new(1.0, 0.5, 0.25);
        let bottom = Vector3::new(0.0, 0.25, 0.75);

        backend.process_attachments(
            &mut target,
            None,
            Some(Operations {
                load: LoadOp::Clear(0.75),
            }),
            Some(BackgroundPass {
                source: BackgroundSource::Gradient { top, bottom },
            }),
        );

        let top_sample = target
            .framebuffer()
            .sample(0, 0)
            .expect("top sample should be in bounds");
        assert_eq!(top_sample.color, top);
        assert_eq!(top_sample.depth, 0.75);

        let middle_sample = target
            .framebuffer()
            .sample(0, 1)
            .expect("middle sample should be in bounds");
        assert_eq!(middle_sample.color, top.lerp(&bottom, 0.5));
        assert_eq!(middle_sample.depth, 0.75);
    }

    #[test]
    fn invalid_render_pass_descriptors_are_rejected_before_writes() {
        let mut target = RenderTarget::new(1, 1, 1).expect("test dimensions should be valid");

        let error = RenderPassDescriptor {
            label: Some("empty"),
            target: &mut target,
            color_ops: None,
            depth_ops: None,
        }
        .validate(false)
        .expect_err("an empty pass should be rejected");
        assert_eq!(
            error,
            RenderPassError::EmptyPass {
                label: "empty".to_owned()
            }
        );

        let original_color = Vector3::new(0.2, 0.4, 0.6);
        target
            .framebuffer_mut()
            .clear_with(0.25, |_, _| original_color);
        target.framebuffer.buffer_width = 2;
        let error = RenderPassDescriptor {
            label: Some("invalid-target"),
            target: &mut target,
            color_ops: Some(Operations {
                load: LoadOp::Clear(Vector3::zeros()),
            }),
            depth_ops: None,
        }
        .validate(false)
        .expect_err("an inconsistent target should be rejected");
        assert!(matches!(
            error,
            RenderPassError::InvalidTarget { label, reason }
                if label == "invalid-target" && reason.contains("buffer dimensions")
        ));
        let sample = target
            .framebuffer()
            .sample(0, 0)
            .expect("the first sample should remain addressable");
        assert_eq!(sample.color, original_color);
        assert_eq!(sample.depth, 0.25);
    }
    #[test]
    fn present_buffer_validates_dimensions_and_storage() {
        let present = PresentBuffer::new(3, 2).expect("present dimensions should be valid");
        assert_eq!(present.width(), 3);
        assert_eq!(present.height(), 2);
        assert_eq!(present.pixels().len(), 6);
        assert!(present.pixels().iter().all(|pixel| *pixel == 0));

        assert!(PresentBuffer::new(0, 1).is_err());
        assert!(PresentBuffer::new(1, 0).is_err());
        assert!(PresentBuffer::new(usize::MAX, 2).is_err());
    }
}
