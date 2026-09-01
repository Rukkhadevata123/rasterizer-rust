use image::{DynamicImage, RgbaImage};
use nalgebra::{Matrix4, Point3, Vector2, Vector3, Vector4};
use rasterizer_rust::core::framebuffer::FrameBuffer;
use rasterizer_rust::core::geometry::Vertex;
use rasterizer_rust::core::pipeline_state::{
    BlendState, ColorTargetState, CompareFunction, CullMode, DepthStencilState, FrontFace,
    GraphicsPipeline, GraphicsPipelineState, PolygonMode, PrimitiveState, VertexProgramId,
};
use rasterizer_rust::core::shader::{FragmentInput, FragmentOutput, Interpolatable, Shader};
use rasterizer_rust::pipeline::passes::{
    ResolveTonemapPassDescriptor, ShadowPassOutput, TonemapOperator, execute_resolve_tonemap_pass,
    render_main_pass, render_shadow_pass,
};
use rasterizer_rust::pipeline::renderer::{
    FrameResources, MainHdrTarget, ObjectBindingId, PresentBuffer, RenderGeometry, RenderPhase,
    RenderTarget, SoftwareRasterBackend,
};
use rasterizer_rust::pipeline::shaders::pbr::{
    PbrDrawContext, PbrFrameBindings, PbrMaterialBindings, PbrObjectBindings, PbrShader, PbrVarying,
};
use rasterizer_rust::pipeline::shaders::shadow::{
    ShadowDrawContext, ShadowFrameBindings, ShadowMaterialBindings, ShadowObjectBindings,
    ShadowShader,
};
use rasterizer_rust::scene::camera::Camera;
use rasterizer_rust::scene::context::{RenderScene, ShadowLight};
use rasterizer_rust::scene::light::Light;
use rasterizer_rust::scene::material::{AlphaMode, Material, PbrMaterial};
use rasterizer_rust::scene::mesh::Mesh;
use rasterizer_rust::scene::model::Model;
use rasterizer_rust::scene::scene_object::{SceneObject, SceneObjectKind};
use rasterizer_rust::scene::texture::{
    MagFilter, MinFilter, SamplerState, TexCoordSet, TextureBinding, TextureImage, TextureUsage,
    WrapMode,
};
use std::ops::{Add, Mul};
use std::sync::Arc;
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
struct DualUvVarying {
    uvs: [Vector2<f32>; 2],
}

impl Add for DualUvVarying {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            uvs: std::array::from_fn(|set| self.uvs[set] + rhs.uvs[set]),
        }
    }
}

impl Mul<f32> for DualUvVarying {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            uvs: std::array::from_fn(|set| self.uvs[set] * rhs),
        }
    }
}

impl Interpolatable for DualUvVarying {
    fn get_uv(&self, set: usize) -> Option<Vector2<f32>> {
        self.uvs.get(set).copied()
    }
}

#[derive(Clone, Copy)]
struct DualUvDensityShader;

impl<'a> Shader<Option<&'a Material>> for DualUvDensityShader {
    type Varying = DualUvVarying;

    fn vertex(
        &self,
        vertex: &Vertex,
        _material: Option<&'a Material>,
    ) -> (Vector4<f32>, Self::Varying) {
        (
            Vector4::new(vertex.position.x, vertex.position.y, vertex.position.z, 1.0),
            DualUvVarying {
                uvs: vertex.texcoords,
            },
        )
    }

    fn fragment(
        &self,
        input: FragmentInput<Self::Varying>,
        _material: Option<&'a Material>,
    ) -> FragmentOutput {
        FragmentOutput::Color(Vector4::new(
            input.uv_densities[0],
            input.uv_densities[1],
            0.0,
            1.0,
        ))
    }
}

#[derive(Clone, Copy)]
struct ClipSpaceShader;

impl<'a> Shader<Option<&'a Material>> for ClipSpaceShader {
    type Varying = ColorVarying;

    fn vertex(
        &self,
        vertex: &Vertex,
        _material: Option<&'a Material>,
    ) -> (Vector4<f32>, Self::Varying) {
        (
            Vector4::new(vertex.position.x, vertex.position.y, vertex.position.z, 1.0),
            ColorVarying {
                color: vertex.tangent,
            },
        )
    }

    fn fragment(
        &self,
        input: FragmentInput<Self::Varying>,
        material: Option<&'a Material>,
    ) -> FragmentOutput {
        let varying = input.varying;
        let alpha_mode = material.map(|material| match material {
            Material::Pbr(material) => material.alpha_mode,
        });
        if matches!(alpha_mode, Some(AlphaMode::Mask(cutoff)) if varying.color.w < cutoff) {
            FragmentOutput::Discard
        } else {
            FragmentOutput::Color(varying.color)
        }
    }
}

#[derive(Clone, Copy)]
struct FacingShader;

impl<'a> Shader<Option<&'a Material>> for FacingShader {
    type Varying = ColorVarying;

    fn vertex(
        &self,
        vertex: &Vertex,
        _material: Option<&'a Material>,
    ) -> (Vector4<f32>, Self::Varying) {
        (
            Vector4::new(vertex.position.x, vertex.position.y, vertex.position.z, 1.0),
            ColorVarying {
                color: vertex.tangent,
            },
        )
    }

    fn fragment(
        &self,
        input: FragmentInput<Self::Varying>,
        _material: Option<&'a Material>,
    ) -> FragmentOutput {
        let color = if input.front_facing {
            Vector4::new(0.0, 1.0, 0.0, 1.0)
        } else {
            Vector4::new(1.0, 0.0, 0.0, 1.0)
        };
        FragmentOutput::Color(color)
    }
}

#[derive(Clone, Copy)]
struct AdditiveCoverageShader;

impl<'a> Shader<Option<&'a Material>> for AdditiveCoverageShader {
    type Varying = ColorVarying;

    fn vertex(
        &self,
        vertex: &Vertex,
        _material: Option<&'a Material>,
    ) -> (Vector4<f32>, Self::Varying) {
        (
            Vector4::new(vertex.position.x, vertex.position.y, vertex.position.z, 1.0),
            ColorVarying {
                color: vertex.tangent,
            },
        )
    }

    fn fragment(
        &self,
        input: FragmentInput<Self::Varying>,
        _material: Option<&'a Material>,
    ) -> FragmentOutput {
        FragmentOutput::Color(input.varying.color)
    }
}

#[derive(Clone, Copy)]
struct NonFiniteClipShader {
    clip: Vector4<f32>,
}

#[derive(Clone, Copy)]
struct CountingShader<'a> {
    vertex_calls: &'a AtomicUsize,
}

impl<'a, 'material> Shader<Option<&'material Material>> for CountingShader<'a> {
    type Varying = ColorVarying;

    fn vertex(
        &self,
        vertex: &Vertex,
        _material: Option<&'material Material>,
    ) -> (Vector4<f32>, Self::Varying) {
        self.vertex_calls.fetch_add(1, Ordering::Relaxed);
        (
            vertex.position.to_homogeneous(),
            ColorVarying {
                color: Vector4::new(1.0, 1.0, 1.0, 1.0),
            },
        )
    }

    fn fragment(
        &self,
        input: FragmentInput<Self::Varying>,
        _material: Option<&'material Material>,
    ) -> FragmentOutput {
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
struct ProgramVariantShader<'a> {
    x_offset: f32,
    color: Vector4<f32>,
    vertex_calls: &'a AtomicUsize,
}

impl Shader<Option<&Material>> for ProgramVariantShader<'_> {
    type Varying = ColorVarying;

    fn vertex(
        &self,
        vertex: &Vertex,
        _context: Option<&Material>,
    ) -> (Vector4<f32>, Self::Varying) {
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

    fn fragment(
        &self,
        input: FragmentInput<Self::Varying>,
        _context: Option<&Material>,
    ) -> FragmentOutput {
        FragmentOutput::Color(input.varying.color)
    }
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

impl<'a> Shader<Option<&'a Material>> for NonFiniteClipShader {
    type Varying = ColorVarying;

    fn vertex(
        &self,
        vertex: &Vertex,
        _material: Option<&'a Material>,
    ) -> (Vector4<f32>, Self::Varying) {
        (
            self.clip,
            ColorVarying {
                color: vertex.tangent,
            },
        )
    }

    fn fragment(
        &self,
        input: FragmentInput<Self::Varying>,
        _material: Option<&'a Material>,
    ) -> FragmentOutput {
        FragmentOutput::Color(input.varying.color)
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

struct TestRenderHarness {
    backend: SoftwareRasterBackend,
    target: MainHdrTarget,
    resources: FrameResources,
}

impl TestRenderHarness {
    fn new(width: usize, height: usize, supersample_scale: usize) -> Self {
        Self {
            backend: SoftwareRasterBackend::new(),
            target: MainHdrTarget::new(width, height, supersample_scale)
                .expect("test dimensions should be valid"),
            resources: FrameResources::new(),
        }
    }

    fn framebuffer(&self) -> &FrameBuffer {
        self.target.framebuffer()
    }
}

fn draw_mesh<'a, S>(
    renderer: &mut TestRenderHarness,
    mesh: &'a Mesh,
    shader: &'a S,
    material: Option<&'a Material>,
    state: GraphicsPipelineState,
) where
    S: Shader<Option<&'a Material>> + Copy,
{
    let pipeline = GraphicsPipeline::new(*shader, state, VertexProgramId::from_pass_index(0));
    let mut phase = RenderPhase::default();
    phase.push(
        &pipeline,
        RenderGeometry::Mesh(mesh),
        material,
        ObjectBindingId::from_pass_index(0),
        0.0,
    );
    renderer
        .backend
        .execute_phase(renderer.target.render_target_mut(), &phase);
}

fn draw_pbr_mesh(
    renderer: &mut TestRenderHarness,
    mesh: &Mesh,
    material: Option<&Material>,
    model: Matrix4<f32>,
    state: GraphicsPipelineState,
) {
    let frame = PbrFrameBindings::new(
        Matrix4::identity(),
        Matrix4::identity(),
        Point3::new(0.0, 0.0, 2.0),
    );
    let object = PbrObjectBindings::new(model);
    let fallback = PbrMaterial::default();
    let context = PbrDrawContext::new(
        &frame,
        &object,
        PbrMaterialBindings::new(material, &fallback),
    );
    let pipeline = GraphicsPipeline::new(PbrShader, state, VertexProgramId::from_pass_index(0));
    let mut phase = RenderPhase::default();
    phase.push(
        &pipeline,
        RenderGeometry::Mesh(mesh),
        context,
        ObjectBindingId::from_pass_index(0),
        0.0,
    );
    renderer
        .backend
        .execute_phase(renderer.target.render_target_mut(), &phase);
}

fn draw_shadow_mesh(
    renderer: &mut TestRenderHarness,
    mesh: &Mesh,
    material: Option<&Material>,
    model: Matrix4<f32>,
    object_binding_id: ObjectBindingId,
) {
    let frame = ShadowFrameBindings::new(Matrix4::identity(), Matrix4::identity());
    let object = ShadowObjectBindings::new(model);
    let context = ShadowDrawContext::new(&frame, &object, ShadowMaterialBindings::new(material));
    let pipeline = GraphicsPipeline::new(
        ShadowShader,
        GraphicsPipelineState {
            color_target: None,
            ..test_pipeline_state()
        },
        VertexProgramId::from_pass_index(0),
    );
    let mut phase = RenderPhase::default();
    phase.push(
        &pipeline,
        RenderGeometry::Mesh(mesh),
        context,
        object_binding_id,
        0.0,
    );
    renderer
        .backend
        .execute_phase(renderer.target.render_target_mut(), &phase);
}
fn shadow_test_camera() -> Camera {
    Camera::new_orthographic(
        Point3::new(0.0, 0.0, 2.0),
        Point3::origin(),
        Vector3::y(),
        2.0,
        1.0,
        0.1,
        10.0,
    )
}

#[path = "rendering/materials.rs"]
mod materials;
#[path = "rendering/rasterization.rs"]
mod rasterization;
#[path = "rendering/shadows.rs"]
mod shadows;
