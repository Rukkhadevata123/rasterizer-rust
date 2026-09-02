use super::*;
use crate::core::pipeline_state::{CullMode, PolygonMode, PrimitiveState, VertexProgramId};
use crate::core::shader::{FragmentInput, FragmentOutput, Interpolatable};
use nalgebra::{Matrix4, Point3, Vector2};
use std::ops::{Add, Mul};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone, Copy)]
struct ColorVarying {
    color: Vector4<f32>,
}

impl Add for ColorVarying {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            color: self.color + rhs.color,
        }
    }
}

impl Mul<f32> for ColorVarying {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            color: self.color * rhs,
        }
    }
}

impl Interpolatable for ColorVarying {}

#[derive(Clone, Copy)]
struct ClipSpaceShader;

impl Shader<()> for ClipSpaceShader {
    type Varying = ColorVarying;

    fn vertex(&self, vertex: &Vertex, _context: ()) -> (Vector4<f32>, Self::Varying) {
        (
            vertex.position.to_homogeneous(),
            ColorVarying {
                color: vertex.tangent,
            },
        )
    }

    fn fragment(&self, input: FragmentInput<Self::Varying>, _context: ()) -> FragmentOutput {
        FragmentOutput::Color(input.varying.color)
    }
}

#[derive(Clone, Copy)]
struct CountingShader<'a> {
    vertex_calls: &'a AtomicUsize,
}

impl Shader<()> for CountingShader<'_> {
    type Varying = ColorVarying;

    fn vertex(&self, vertex: &Vertex, _context: ()) -> (Vector4<f32>, Self::Varying) {
        self.vertex_calls.fetch_add(1, Ordering::Relaxed);
        (
            vertex.position.to_homogeneous(),
            ColorVarying {
                color: Vector4::new(1.0, 1.0, 1.0, 1.0),
            },
        )
    }

    fn fragment(&self, input: FragmentInput<Self::Varying>, _context: ()) -> FragmentOutput {
        FragmentOutput::Color(input.varying.color)
    }
}

#[derive(Clone, Copy)]
struct FrameObjectDrawContext<'a> {
    frame_transform: &'a Matrix4<f32>,
    object_transform: &'a Matrix4<f32>,
    color: Vector4<f32>,
    vertex_calls: &'a AtomicUsize,
}

#[derive(Clone, Copy)]
struct FrameObjectContextShader;

impl Shader<FrameObjectDrawContext<'_>> for FrameObjectContextShader {
    type Varying = ColorVarying;

    fn vertex(
        &self,
        vertex: &Vertex,
        context: FrameObjectDrawContext<'_>,
    ) -> (Vector4<f32>, Self::Varying) {
        context.vertex_calls.fetch_add(1, Ordering::Relaxed);
        (
            context.frame_transform * context.object_transform * vertex.position.to_homogeneous(),
            ColorVarying {
                color: context.color,
            },
        )
    }

    fn fragment(
        &self,
        input: FragmentInput<Self::Varying>,
        _context: FrameObjectDrawContext<'_>,
    ) -> FragmentOutput {
        FragmentOutput::Color(input.varying.color)
    }
}

#[derive(Clone, Copy)]
struct TangentDrawContext<'a> {
    clip_transform: &'a Matrix4<f32>,
    tangent_transform: &'a Matrix4<f32>,
    vertex_calls: &'a AtomicUsize,
}

#[derive(Clone, Copy)]
struct TangentContextShader;

impl Shader<TangentDrawContext<'_>> for TangentContextShader {
    type Varying = ColorVarying;

    fn vertex(
        &self,
        vertex: &Vertex,
        context: TangentDrawContext<'_>,
    ) -> (Vector4<f32>, Self::Varying) {
        context.vertex_calls.fetch_add(1, Ordering::Relaxed);
        let tangent = context.tangent_transform * vertex.tangent;
        (
            context.clip_transform * vertex.position.to_homogeneous(),
            ColorVarying { color: tangent },
        )
    }

    fn fragment(
        &self,
        input: FragmentInput<Self::Varying>,
        _context: TangentDrawContext<'_>,
    ) -> FragmentOutput {
        FragmentOutput::Color(input.varying.color)
    }
}

#[derive(Clone, Copy)]
struct TransformDrawContext<'a> {
    transform: &'a Matrix4<f32>,
    color: Vector4<f32>,
    vertex_calls: &'a AtomicUsize,
}

#[derive(Clone, Copy)]
struct TransformContextShader;

impl Shader<TransformDrawContext<'_>> for TransformContextShader {
    type Varying = ColorVarying;

    fn vertex(
        &self,
        vertex: &Vertex,
        context: TransformDrawContext<'_>,
    ) -> (Vector4<f32>, Self::Varying) {
        context.vertex_calls.fetch_add(1, Ordering::Relaxed);
        (
            context.transform * vertex.position.to_homogeneous(),
            ColorVarying {
                color: context.color,
            },
        )
    }

    fn fragment(
        &self,
        input: FragmentInput<Self::Varying>,
        _context: TransformDrawContext<'_>,
    ) -> FragmentOutput {
        FragmentOutput::Color(input.varying.color)
    }
}

#[derive(Clone, Copy)]
struct ProgramVariantShader<'a> {
    x_offset: f32,
    color: Vector4<f32>,
    vertex_calls: &'a AtomicUsize,
}

impl Shader<()> for ProgramVariantShader<'_> {
    type Varying = ColorVarying;

    fn vertex(&self, vertex: &Vertex, _context: ()) -> (Vector4<f32>, Self::Varying) {
        self.vertex_calls.fetch_add(1, Ordering::Relaxed);
        (
            Vector4::new(
                vertex.position.x + self.x_offset,
                vertex.position.y,
                vertex.position.z,
                1.0,
            ),
            ColorVarying { color: self.color },
        )
    }

    fn fragment(&self, input: FragmentInput<Self::Varying>, _context: ()) -> FragmentOutput {
        FragmentOutput::Color(input.varying.color)
    }
}

struct BackendTestHarness {
    backend: SoftwareRasterBackend,
    target: RenderTarget,
}

impl BackendTestHarness {
    fn new(width: usize, height: usize) -> Self {
        Self {
            backend: SoftwareRasterBackend::new(),
            target: RenderTarget::new(width, height, 1).expect("test dimensions should be valid"),
        }
    }

    fn framebuffer(&self) -> &FrameBuffer {
        self.target.framebuffer()
    }
}

fn triangle(z: f32, color: Vector4<f32>) -> Mesh {
    let mut vertices = vec![
        Vertex::new(Point3::new(-0.8, -0.8, z), Vector3::z(), Vector2::zeros()),
        Vertex::new(Point3::new(0.8, -0.8, z), Vector3::z(), Vector2::zeros()),
        Vertex::new(Point3::new(0.0, 0.8, z), Vector3::z(), Vector2::zeros()),
    ];
    for vertex in &mut vertices {
        vertex.tangent = color;
    }
    Mesh::new(vertices, vec![0, 1, 2], 0)
}

fn assert_vec3_approx(actual: Vector3<f32>, expected: Vector3<f32>) {
    assert!(
        (actual - expected).norm() < 1e-4,
        "expected {expected:?}, got {actual:?}"
    );
}

fn test_pipeline_state() -> GraphicsPipelineState {
    GraphicsPipelineState {
        primitive: PrimitiveState {
            cull_mode: CullMode::None,
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn indexed_mesh_shades_each_vertex_once_per_draw() {
    let calls = AtomicUsize::new(0);
    let shader = CountingShader {
        vertex_calls: &calls,
    };
    let vertices = vec![
        Vertex::new(Point3::new(-0.8, -0.8, 0.0), Vector3::z(), Vector2::zeros()),
        Vertex::new(Point3::new(0.8, -0.8, 0.0), Vector3::z(), Vector2::zeros()),
        Vertex::new(Point3::new(0.8, 0.8, 0.0), Vector3::z(), Vector2::zeros()),
        Vertex::new(Point3::new(-0.8, 0.8, 0.0), Vector3::z(), Vector2::zeros()),
    ];
    let mesh = Mesh::new(vertices, vec![0, 1, 2, 0, 2, 3], 0);
    let mut renderer = BackendTestHarness::new(32, 32);
    let pipeline = GraphicsPipeline::new(
        shader,
        test_pipeline_state(),
        VertexProgramId::from_pass_index(0),
    );
    let mut phase = RenderPhase::default();
    phase.push(
        &pipeline,
        RenderGeometry::Mesh(&mesh),
        (),
        ObjectBindingId::from_pass_index(0),
        0.0,
    );

    renderer.backend.execute_phase(&mut renderer.target, &phase);

    assert_eq!(calls.load(Ordering::Relaxed), mesh.vertices.len());
}

#[test]
fn vertex_cache_is_scoped_to_each_camera_submission() {
    let calls = AtomicUsize::new(0);
    let pipeline = GraphicsPipeline::new(
        FrameObjectContextShader,
        test_pipeline_state(),
        VertexProgramId::from_pass_index(0),
    );
    let mesh = Mesh::new(
        vec![
            Vertex::new(
                Point3::new(-0.25, -0.8, 0.0),
                Vector3::z(),
                Vector2::zeros(),
            ),
            Vertex::new(Point3::new(0.25, -0.8, 0.0), Vector3::z(), Vector2::zeros()),
            Vertex::new(Point3::new(0.0, 0.8, 0.0), Vector3::z(), Vector2::zeros()),
        ],
        vec![0, 1, 2, 0, 2, 1],
        0,
    );
    let object_transform = Matrix4::identity();
    let left_camera = Matrix4::new_translation(&Vector3::new(-0.5, 0.0, 0.0));
    let right_camera = Matrix4::new_translation(&Vector3::new(0.5, 0.0, 0.0));
    let mut renderer = BackendTestHarness::new(64, 32);

    for (camera, color) in [
        (&left_camera, Vector4::new(1.0, 0.0, 0.0, 1.0)),
        (&right_camera, Vector4::new(0.0, 1.0, 0.0, 1.0)),
    ] {
        let mut phase = RenderPhase::default();
        phase.push(
            &pipeline,
            RenderGeometry::Mesh(&mesh),
            FrameObjectDrawContext {
                frame_transform: camera,
                object_transform: &object_transform,
                color,
                vertex_calls: &calls,
            },
            ObjectBindingId::from_pass_index(0),
            0.0,
        );
        renderer.backend.execute_phase(&mut renderer.target, &phase);
    }

    assert_eq!(calls.load(Ordering::Relaxed), mesh.vertices.len() * 2);
    assert_vec3_approx(
        renderer.framebuffer().get_pixel(16, 16).unwrap(),
        Vector3::x(),
    );
    assert_vec3_approx(
        renderer.framebuffer().get_pixel(48, 16).unwrap(),
        Vector3::y(),
    );
}

#[test]
fn vertex_cache_isolates_distinct_tangent_frame_bindings() {
    let calls = AtomicUsize::new(0);
    let pipeline = GraphicsPipeline::new(
        TangentContextShader,
        test_pipeline_state(),
        VertexProgramId::from_pass_index(0),
    );
    let mut mesh = Mesh::new(
        vec![
            Vertex::new(
                Point3::new(-0.25, -0.8, 0.0),
                Vector3::z(),
                Vector2::zeros(),
            ),
            Vertex::new(Point3::new(0.25, -0.8, 0.0), Vector3::z(), Vector2::zeros()),
            Vertex::new(Point3::new(0.0, 0.8, 0.0), Vector3::z(), Vector2::zeros()),
        ],
        vec![0, 1, 2, 0, 2, 1],
        0,
    );
    for vertex in &mut mesh.vertices {
        vertex.tangent = Vector4::new(1.0, 0.0, 0.0, 1.0);
    }
    let left_clip = Matrix4::new_translation(&Vector3::new(-0.5, 0.0, 0.0));
    let right_clip = Matrix4::new_translation(&Vector3::new(0.5, 0.0, 0.0));
    let red_tangent = Matrix4::identity();
    let green_tangent = Matrix4::new(
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    );
    let contexts = [
        TangentDrawContext {
            clip_transform: &left_clip,
            tangent_transform: &red_tangent,
            vertex_calls: &calls,
        },
        TangentDrawContext {
            clip_transform: &right_clip,
            tangent_transform: &green_tangent,
            vertex_calls: &calls,
        },
    ];
    let mut phase = RenderPhase::with_capacity(2);
    for (index, context) in contexts.into_iter().enumerate() {
        phase.push(
            &pipeline,
            RenderGeometry::Mesh(&mesh),
            context,
            ObjectBindingId::from_pass_index(index),
            0.0,
        );
    }
    let mut renderer = BackendTestHarness::new(64, 32);

    renderer.backend.execute_phase(&mut renderer.target, &phase);

    assert_eq!(calls.load(Ordering::Relaxed), mesh.vertices.len() * 2);
    assert_vec3_approx(
        renderer.framebuffer().get_pixel(16, 16).unwrap(),
        Vector3::x(),
    );
    assert_vec3_approx(
        renderer.framebuffer().get_pixel(48, 16).unwrap(),
        Vector3::y(),
    );
}

#[test]
fn transparent_world_vertices_use_a_distinct_source_domain() {
    let calls = AtomicUsize::new(0);
    let pipeline = GraphicsPipeline::new(
        CountingShader {
            vertex_calls: &calls,
        },
        test_pipeline_state(),
        VertexProgramId::from_pass_index(0),
    );
    let mesh = Mesh::new(
        vec![
            Vertex::new(Point3::new(-0.8, -0.8, 0.0), Vector3::z(), Vector2::zeros()),
            Vertex::new(Point3::new(0.8, -0.8, 0.0), Vector3::z(), Vector2::zeros()),
            Vertex::new(Point3::new(0.8, 0.8, 0.0), Vector3::z(), Vector2::zeros()),
            Vertex::new(Point3::new(-0.8, 0.8, 0.0), Vector3::z(), Vector2::zeros()),
        ],
        vec![0, 1, 2, 0, 2, 3],
        0,
    );
    let world_vertices = mesh.vertices.clone();
    let object_binding_id = ObjectBindingId::from_pass_index(0);
    let mut phase = RenderPhase::with_capacity(2);
    phase.push(
        &pipeline,
        RenderGeometry::Mesh(&mesh),
        (),
        object_binding_id,
        0.0,
    );
    phase.push(
        &pipeline,
        RenderGeometry::IndexedTriangle {
            vertices: &world_vertices,
            indices: [0, 1, 2],
            cache_vertices: true,
        },
        (),
        object_binding_id,
        0.0,
    );
    let mut renderer = BackendTestHarness::new(32, 32);

    renderer.backend.execute_phase(&mut renderer.target, &phase);

    assert_eq!(
        calls.load(Ordering::Relaxed),
        mesh.vertices.len() + world_vertices.len()
    );
}

#[test]
fn vertex_cache_reuses_across_raster_only_pipeline_variants() {
    let calls = AtomicUsize::new(0);
    let shader = CountingShader {
        vertex_calls: &calls,
    };
    let mesh = Mesh::new(
        vec![
            Vertex::new(Point3::new(-0.8, -0.8, 0.0), Vector3::z(), Vector2::zeros()),
            Vertex::new(Point3::new(0.8, -0.8, 0.0), Vector3::z(), Vector2::zeros()),
            Vertex::new(Point3::new(0.8, 0.8, 0.0), Vector3::z(), Vector2::zeros()),
            Vertex::new(Point3::new(-0.8, 0.8, 0.0), Vector3::z(), Vector2::zeros()),
        ],
        vec![0, 1, 2, 0, 2, 3],
        0,
    );
    let vertex_program_id = VertexProgramId::from_pass_index(0);
    let fill_pipeline = GraphicsPipeline::new(shader, test_pipeline_state(), vertex_program_id);
    let line_pipeline = GraphicsPipeline::new(
        shader,
        GraphicsPipelineState {
            primitive: PrimitiveState {
                polygon_mode: PolygonMode::Line,
                ..test_pipeline_state().primitive
            },
            ..test_pipeline_state()
        },
        vertex_program_id,
    );
    let object_binding_id = ObjectBindingId::from_pass_index(0);
    let mut phase = RenderPhase::with_capacity(2);
    phase.push(
        &fill_pipeline,
        RenderGeometry::Mesh(&mesh),
        (),
        object_binding_id,
        0.0,
    );
    phase.push(
        &line_pipeline,
        RenderGeometry::Mesh(&mesh),
        (),
        object_binding_id,
        0.0,
    );
    let mut renderer = BackendTestHarness::new(32, 32);

    renderer.backend.execute_phase(&mut renderer.target, &phase);

    assert_eq!(calls.load(Ordering::Relaxed), mesh.vertices.len());
}

#[test]
fn vertex_cache_reuses_matching_context_and_isolates_distinct_contexts() {
    let calls = AtomicUsize::new(0);
    let pipeline = GraphicsPipeline::new(
        TransformContextShader,
        test_pipeline_state(),
        VertexProgramId::from_pass_index(0),
    );
    let vertices = vec![
        Vertex::new(Point3::new(-0.3, -0.8, 0.0), Vector3::z(), Vector2::zeros()),
        Vertex::new(Point3::new(0.3, -0.8, 0.0), Vector3::z(), Vector2::zeros()),
        Vertex::new(Point3::new(0.3, 0.8, 0.0), Vector3::z(), Vector2::zeros()),
        Vertex::new(Point3::new(-0.3, 0.8, 0.0), Vector3::z(), Vector2::zeros()),
    ];
    let mesh = Mesh::new(vertices, vec![0, 1, 2, 0, 2, 3], 0);
    let left_transform = Matrix4::new_translation(&Vector3::new(-0.5, 0.0, 0.0));
    let right_transform = Matrix4::new_translation(&Vector3::new(0.5, 0.0, 0.0));
    let left = TransformDrawContext {
        transform: &left_transform,
        color: Vector4::new(1.0, 0.0, 0.0, 1.0),
        vertex_calls: &calls,
    };
    let right = TransformDrawContext {
        transform: &right_transform,
        color: Vector4::new(0.0, 1.0, 0.0, 1.0),
        vertex_calls: &calls,
    };
    let mut phase = RenderPhase::with_capacity(3);
    phase.push(
        &pipeline,
        RenderGeometry::Mesh(&mesh),
        left,
        ObjectBindingId::from_pass_index(0),
        0.0,
    );
    phase.push(
        &pipeline,
        RenderGeometry::Mesh(&mesh),
        left,
        ObjectBindingId::from_pass_index(0),
        0.0,
    );
    phase.push(
        &pipeline,
        RenderGeometry::Mesh(&mesh),
        right,
        ObjectBindingId::from_pass_index(1),
        0.0,
    );
    let mut renderer = BackendTestHarness::new(64, 32);

    renderer.backend.execute_phase(&mut renderer.target, &phase);

    assert_eq!(calls.load(Ordering::Relaxed), mesh.vertices.len() * 2);
    assert_vec3_approx(
        renderer.framebuffer().get_pixel(16, 16).unwrap(),
        Vector3::x(),
    );
    assert_vec3_approx(
        renderer.framebuffer().get_pixel(48, 16).unwrap(),
        Vector3::y(),
    );
}

#[test]
fn vertex_cache_distinguishes_vertex_program_ids() {
    let calls = AtomicUsize::new(0);
    let mesh = Mesh::new(
        vec![
            Vertex::new(
                Point3::new(-0.25, -0.8, 0.0),
                Vector3::z(),
                Vector2::zeros(),
            ),
            Vertex::new(Point3::new(0.25, -0.8, 0.0), Vector3::z(), Vector2::zeros()),
            Vertex::new(Point3::new(0.0, 0.8, 0.0), Vector3::z(), Vector2::zeros()),
        ],
        vec![0, 1, 2, 0, 2, 1],
        0,
    );
    let left_pipeline = GraphicsPipeline::new(
        ProgramVariantShader {
            x_offset: -0.5,
            color: Vector4::new(1.0, 0.0, 0.0, 1.0),
            vertex_calls: &calls,
        },
        test_pipeline_state(),
        VertexProgramId::from_pass_index(0),
    );
    let right_pipeline = GraphicsPipeline::new(
        ProgramVariantShader {
            x_offset: 0.5,
            color: Vector4::new(0.0, 1.0, 0.0, 1.0),
            vertex_calls: &calls,
        },
        test_pipeline_state(),
        VertexProgramId::from_pass_index(1),
    );
    let mut phase = RenderPhase::with_capacity(2);
    phase.push(
        &left_pipeline,
        RenderGeometry::Mesh(&mesh),
        (),
        ObjectBindingId::from_pass_index(0),
        0.0,
    );
    phase.push(
        &right_pipeline,
        RenderGeometry::Mesh(&mesh),
        (),
        ObjectBindingId::from_pass_index(0),
        0.0,
    );
    let mut renderer = BackendTestHarness::new(64, 32);

    renderer.backend.execute_phase(&mut renderer.target, &phase);

    assert_eq!(calls.load(Ordering::Relaxed), mesh.vertices.len() * 2);
    assert_vec3_approx(
        renderer.framebuffer().get_pixel(16, 16).unwrap(),
        Vector3::x(),
    );
    assert_vec3_approx(
        renderer.framebuffer().get_pixel(48, 16).unwrap(),
        Vector3::y(),
    );
}

#[test]
fn one_backend_executes_different_sized_targets_sequentially() {
    let mesh = triangle(0.0, Vector4::new(1.0, 0.5, 0.25, 1.0));
    let pipeline = GraphicsPipeline::new(
        ClipSpaceShader,
        test_pipeline_state(),
        VertexProgramId::from_pass_index(0),
    );
    let mut phase = RenderPhase::default();
    phase.push(
        &pipeline,
        RenderGeometry::Mesh(&mesh),
        (),
        ObjectBindingId::from_pass_index(0),
        0.0,
    );
    let mut backend = SoftwareRasterBackend::new();
    let mut shadow_target = RenderTarget::new(16, 16, 1).expect("shadow target should be valid");
    let mut main_target = RenderTarget::new(48, 32, 1).expect("main target should be valid");

    backend.execute_phase(&mut shadow_target, &phase);
    backend.execute_phase(&mut main_target, &phase);

    let expected = Vector3::new(1.0, 0.5, 0.25);
    assert_vec3_approx(
        shadow_target.framebuffer().get_pixel(8, 8).unwrap(),
        expected,
    );
    assert_vec3_approx(
        main_target.framebuffer().get_pixel(24, 16).unwrap(),
        expected,
    );
}

#[test]
fn whole_pass_preparation_skips_empty_mesh_commands_across_phases() {
    let empty = Mesh::new(Vec::new(), Vec::new(), 0);
    let visible = triangle(0.0, Vector4::new(0.0, 1.0, 0.0, 1.0));
    let pipeline = GraphicsPipeline::new(
        ClipSpaceShader,
        test_pipeline_state(),
        VertexProgramId::from_pass_index(0),
    );
    let mut empty_phase = RenderPhase::default();
    empty_phase.push(
        &pipeline,
        RenderGeometry::Mesh(&empty),
        (),
        ObjectBindingId::from_pass_index(0),
        0.0,
    );
    let mut visible_phase = RenderPhase::default();
    visible_phase.push(
        &pipeline,
        RenderGeometry::Mesh(&visible),
        (),
        ObjectBindingId::from_pass_index(0),
        0.0,
    );
    let mut renderer = BackendTestHarness::new(32, 32);

    renderer
        .backend
        .execute_phases(&mut renderer.target, &[&empty_phase, &visible_phase]);

    assert_vec3_approx(
        renderer.framebuffer().sample(16, 16).unwrap().color,
        Vector3::y(),
    );
}
