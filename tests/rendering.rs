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
    CommandEncoder, CommandError, FrameResources, GraphicsQueue, LoadOp, MainHdrTarget,
    ObjectBindingId, Operations, PresentBuffer, RenderDevice, RenderGeometry, RenderPassDescriptor,
    RenderPhase, RenderTarget, SoftwareRasterBackend,
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
    queue: GraphicsQueue,
    target: MainHdrTarget,
    resources: FrameResources,
}

impl TestRenderHarness {
    fn new(width: usize, height: usize, supersample_scale: usize) -> Self {
        Self {
            queue: RenderDevice::new().create_queue(),
            target: MainHdrTarget::new(width, height, supersample_scale)
                .expect("test dimensions should be valid"),
            resources: FrameResources::new(),
        }
    }

    fn framebuffer(&self) -> &FrameBuffer {
        self.target.framebuffer()
    }
}

struct BackendTestHarness {
    backend: SoftwareRasterBackend,
    target: MainHdrTarget,
}

impl BackendTestHarness {
    fn new(width: usize, height: usize, supersample_scale: usize) -> Self {
        Self {
            backend: SoftwareRasterBackend::new(),
            target: MainHdrTarget::new(width, height, supersample_scale)
                .expect("test dimensions should be valid"),
        }
    }

    fn framebuffer(&self) -> &FrameBuffer {
        self.target.framebuffer()
    }
}

fn submit_test_mesh<'a, S, C>(
    queue: &mut GraphicsQueue,
    target: &'a mut RenderTarget,
    pipeline: &'a GraphicsPipeline<S>,
    mesh: &'a Mesh,
    context: C,
    object_binding_id: ObjectBindingId,
) where
    S: Shader<C>,
    C: Copy + Send + Sync,
{
    let device = RenderDevice::new();
    let mut encoder = device.create_command_encoder("test-draw");
    {
        let mut pass = encoder
            .begin_render_pass(
                RenderPassDescriptor {
                    label: Some("test-draw"),
                    target,
                    color_ops: Some(Operations { load: LoadOp::Load }),
                    depth_ops: Some(Operations { load: LoadOp::Load }),
                },
                None,
            )
            .expect("the test render pass should be valid");
        pass.set_pipeline(pipeline);
        pass.set_draw_bindings(context, object_binding_id);
        pass.draw_mesh(mesh, 0.0)
            .expect("the test draw should record");
        pass.end().expect("the test render pass should end");
    }
    queue
        .submit(
            encoder
                .finish()
                .expect("the test command buffer should finish"),
        )
        .expect("the test command buffer should submit");
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
    submit_test_mesh(
        &mut renderer.queue,
        renderer.target.render_target_mut(),
        &pipeline,
        mesh,
        material,
        ObjectBindingId::from_pass_index(0),
    );
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
    submit_test_mesh(
        &mut renderer.queue,
        renderer.target.render_target_mut(),
        &pipeline,
        mesh,
        context,
        ObjectBindingId::from_pass_index(0),
    );
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
    submit_test_mesh(
        &mut renderer.queue,
        renderer.target.render_target_mut(),
        &pipeline,
        mesh,
        context,
        object_binding_id,
    );
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

#[path = "rendering/commands.rs"]
mod commands;
#[path = "rendering/materials.rs"]
mod materials;
#[path = "rendering/rasterization.rs"]
mod rasterization;
#[path = "rendering/shadows.rs"]
mod shadows;
