use image::{DynamicImage, RgbaImage};
use nalgebra::{Matrix4, Point3, Vector2, Vector3, Vector4};
use rasterizer_rust::core::framebuffer::FrameBuffer;
use rasterizer_rust::core::geometry::Vertex;
use rasterizer_rust::core::pipeline_state::{
    BlendState, ColorTargetState, CompareFunction, CullMode, DepthStencilState, FrontFace,
    GraphicsPipelineState, PolygonMode, PrimitiveState,
};
use rasterizer_rust::core::shader::{FragmentInput, FragmentOutput, Interpolatable, Shader};
use rasterizer_rust::pipeline::passes::{
    ResolveTonemapPassDescriptor, ShadowPassOutput, TonemapOperator, execute_resolve_tonemap_pass,
    render_main_pass, render_shadow_pass,
};
use rasterizer_rust::pipeline::renderer::{
    FrameResources, MainHdrTarget, PresentBuffer, RenderGeometry, RenderPhase, RenderTarget,
    SoftwareRasterBackend,
};
use rasterizer_rust::pipeline::shaders::pbr::{PbrShader, PbrVarying};
use rasterizer_rust::pipeline::shaders::shadow::ShadowShader;
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

struct NonFiniteClipShader {
    clip: Vector4<f32>,
}

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

struct MaterialContextShader<'a> {
    vertex_calls: &'a AtomicUsize,
}

impl<'shader, 'material> Shader<Option<&'material Material>> for MaterialContextShader<'shader> {
    type Varying = ColorVarying;

    fn vertex(
        &self,
        vertex: &Vertex,
        material: Option<&'material Material>,
    ) -> (Vector4<f32>, Self::Varying) {
        self.vertex_calls.fetch_add(1, Ordering::Relaxed);
        let material = material.expect("test draw should bind a material");
        let Material::Pbr(material) = material;
        let x_offset = if material.albedo.x > material.albedo.y {
            -0.5
        } else {
            0.5
        };
        (
            Vector4::new(
                vertex.position.x + x_offset,
                vertex.position.y,
                vertex.position.z,
                1.0,
            ),
            ColorVarying {
                color: material.albedo.push(1.0),
            },
        )
    }

    fn fragment(
        &self,
        input: FragmentInput<Self::Varying>,
        material: Option<&'material Material>,
    ) -> FragmentOutput {
        assert!(material.is_some());
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
    S: Shader<Option<&'a Material>>,
{
    let mut phase = RenderPhase::default();
    phase.push(0, RenderGeometry::Mesh(mesh), material, state, 0.0);
    renderer.backend.execute_phase(
        renderer.target.render_target_mut(),
        &phase,
        std::slice::from_ref(shader),
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

#[path = "rendering/materials.rs"]
mod materials;
#[path = "rendering/rasterization.rs"]
mod rasterization;
#[path = "rendering/shadows.rs"]
mod shadows;
