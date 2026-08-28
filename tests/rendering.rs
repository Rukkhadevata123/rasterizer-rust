use image::{DynamicImage, RgbaImage};
use nalgebra::{Matrix4, Point3, Vector2, Vector3, Vector4};
use rasterizer_rust::core::framebuffer::FrameBuffer;
use rasterizer_rust::core::geometry::Vertex;
use rasterizer_rust::core::pipeline::{FragmentInput, FragmentOutput, Interpolatable, Shader};
use rasterizer_rust::core::rasterizer::{
    BlendMode, CullMode, DepthCompare, Rasterizer, RenderState,
};
use rasterizer_rust::pipeline::passes::{
    ShadowPassOutput, post_process_to_buffer, render_main_pass, render_shadow_pass,
};
use rasterizer_rust::pipeline::renderer::{RenderGeometry, RenderQueue, Renderer};
use rasterizer_rust::pipeline::shaders::pbr::{PbrShader, PbrVarying};
use rasterizer_rust::pipeline::shaders::shadow::ShadowShader;
use rasterizer_rust::scene::camera::Camera;
use rasterizer_rust::scene::context::{RenderContext, ShadowLight};
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

struct DualUvDensityShader;

impl<'a> Shader<Option<&'a Material>> for DualUvDensityShader {
    type Varying = DualUvVarying;

    fn vertex(&self, vertex: &Vertex) -> (Vector4<f32>, Self::Varying) {
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

    fn vertex(&self, vertex: &Vertex) -> (Vector4<f32>, Self::Varying) {
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

    fn vertex(&self, vertex: &Vertex) -> (Vector4<f32>, Self::Varying) {
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

fn test_render_state() -> RenderState {
    RenderState {
        cull_mode: CullMode::None,
        ..Default::default()
    }
}

fn draw_mesh<'a, S>(
    renderer: &mut Renderer,
    mesh: &'a Mesh,
    shader: &'a S,
    material: Option<&'a Material>,
    state: RenderState,
) where
    S: Shader<Option<&'a Material>>,
{
    let mut queue = RenderQueue::default();
    queue.push(0, RenderGeometry::Mesh(mesh), material, state, 0.0);
    renderer.draw_queue(&queue, std::slice::from_ref(shader));
}

#[test]
fn nearer_triangle_wins_depth_test() {
    let shader = ClipSpaceShader;
    let mut renderer = Renderer::new(32, 32, 1).expect("test dimensions should be valid");

    let far = triangle(0.5, Vector4::new(1.0, 0.0, 0.0, 1.0));
    let near = triangle(-0.5, Vector4::new(0.0, 1.0, 0.0, 1.0));
    draw_mesh(&mut renderer, &far, &shader, None, test_render_state());
    draw_mesh(&mut renderer, &near, &shader, None, test_render_state());

    assert_vec3_approx(
        renderer.framebuffer.get_pixel(16, 16).unwrap(),
        Vector3::new(0.0, 1.0, 0.0),
    );
}

#[test]
fn depth_state_is_explicit_per_draw() {
    let shader = ClipSpaceShader;
    let mut renderer = Renderer::new(32, 32, 1).expect("test dimensions should be valid");
    let red = triangle(-0.5, Vector4::new(1.0, 0.0, 0.0, 1.0));
    let blue = triangle(0.5, Vector4::new(0.0, 0.0, 1.0, 1.0));

    draw_mesh(
        &mut renderer,
        &red,
        &shader,
        None,
        RenderState {
            depth_write: false,
            ..test_render_state()
        },
    );
    assert!(
        renderer
            .framebuffer
            .sample(16, 16)
            .unwrap()
            .depth
            .is_infinite()
    );

    draw_mesh(
        &mut renderer,
        &blue,
        &shader,
        None,
        RenderState {
            depth_compare: DepthCompare::Greater,
            ..test_render_state()
        },
    );
    assert_vec3_approx(
        renderer.framebuffer.get_pixel(16, 16).unwrap(),
        Vector3::new(1.0, 0.0, 0.0),
    );

    draw_mesh(
        &mut renderer,
        &blue,
        &shader,
        None,
        RenderState {
            depth_test: false,
            ..test_render_state()
        },
    );
    assert_vec3_approx(
        renderer.framebuffer.get_pixel(16, 16).unwrap(),
        Vector3::new(0.0, 0.0, 1.0),
    );
}

#[test]
fn triangle_crossing_near_plane_is_clipped_and_rendered() {
    let shader = ClipSpaceShader;
    let mut renderer = Renderer::new(32, 32, 1).expect("test dimensions should be valid");

    let mut mesh = triangle(0.0, Vector4::new(1.0, 0.0, 1.0, 1.0));
    mesh.vertices[0].position.z = -2.0;
    draw_mesh(&mut renderer, &mesh, &shader, None, test_render_state());

    let colored_pixels = (0..32)
        .flat_map(|y| (0..32).map(move |x| (x, y)))
        .filter(|&(x, y)| renderer.framebuffer.get_pixel(x, y).unwrap().norm_squared() > 0.0)
        .count();
    assert!(colored_pixels > 0);
}

#[test]
fn alpha_mask_discards_fragments_below_cutoff() {
    let shader = ClipSpaceShader;
    let mut renderer = Renderer::new(32, 32, 1).expect("test dimensions should be valid");
    let mesh = triangle(0.0, Vector4::new(1.0, 0.0, 0.0, 0.25));
    let material = Material::Pbr(PbrMaterial {
        alpha_mode: AlphaMode::Mask(0.5),
        ..Default::default()
    });

    draw_mesh(
        &mut renderer,
        &mesh,
        &shader,
        Some(&material),
        test_render_state(),
    );

    assert_eq!(
        renderer.framebuffer.get_pixel(16, 16).unwrap(),
        Vector3::zeros()
    );
    assert!(
        renderer
            .framebuffer
            .sample(16, 16)
            .unwrap()
            .depth
            .is_infinite()
    );
}

#[test]
fn framebuffer_resolves_supersampled_pixels() {
    let mut framebuffer = FrameBuffer::new(1, 1, 2).expect("test dimensions should be valid");
    framebuffer.clear_with(f32::INFINITY, |x, y| match (x, y) {
        (0, 0) => Vector3::new(1.0, 0.0, 0.0),
        (1, 0) => Vector3::new(0.0, 1.0, 0.0),
        (0, 1) => Vector3::new(0.0, 0.0, 1.0),
        (1, 1) => Vector3::new(1.0, 1.0, 1.0),
        _ => unreachable!(),
    });

    assert_vec3_approx(
        framebuffer.get_pixel(0, 0).unwrap(),
        Vector3::new(0.5, 0.5, 0.5),
    );
}

#[test]
fn headless_pbr_triangle_produces_visible_output() {
    let mut renderer = Renderer::new(32, 32, 1).expect("test dimensions should be valid");
    let shader = PbrShader::new(
        Matrix4::identity(),
        Matrix4::identity(),
        Matrix4::identity(),
        Point3::new(0.0, 0.0, 2.0),
    );
    let mesh = triangle(0.0, Vector4::zeros());
    let material = Material::Pbr(PbrMaterial {
        albedo: Vector3::new(0.8, 0.2, 0.1),
        emissive: Vector3::new(0.2, 0.1, 0.05),
        ..Default::default()
    });

    draw_mesh(
        &mut renderer,
        &mesh,
        &shader,
        Some(&material),
        test_render_state(),
    );

    let mut config = rasterizer_rust::io::config::Config::default();
    config.render.width = 32;
    config.render.height = 32;
    config.render.use_aces = false;
    let mut output = vec![0; 32 * 32];
    post_process_to_buffer(&renderer.framebuffer, &mut output, &config);

    assert_ne!(output[16 * 32 + 16] & 0x00ff_ffff, 0);
}

#[test]
fn masked_pbr_fragments_respect_material_alpha() {
    let mut renderer = Renderer::new(32, 32, 1).expect("test dimensions should be valid");
    let shader = PbrShader::new(
        Matrix4::identity(),
        Matrix4::identity(),
        Matrix4::identity(),
        Point3::new(0.0, 0.0, 2.0),
    );
    let mesh = triangle(0.0, Vector4::zeros());
    let discarded = Material::Pbr(PbrMaterial {
        alpha: 0.25,
        alpha_mode: AlphaMode::Mask(0.5),
        ..Default::default()
    });

    draw_mesh(
        &mut renderer,
        &mesh,
        &shader,
        Some(&discarded),
        test_render_state(),
    );
    assert!(
        renderer
            .framebuffer
            .sample(16, 16)
            .unwrap()
            .depth
            .is_infinite()
    );

    let visible = Material::Pbr(PbrMaterial {
        alpha: 0.75,
        alpha_mode: AlphaMode::Mask(0.5),
        ..Default::default()
    });
    draw_mesh(
        &mut renderer,
        &mesh,
        &shader,
        Some(&visible),
        test_render_state(),
    );
    assert!((renderer.framebuffer.sample(16, 16).unwrap().depth - 0.5).abs() < 1e-5);
}

#[test]
fn cull_mode_can_reject_one_winding() {
    let shader = ClipSpaceShader;
    let mesh = triangle(0.0, Vector4::new(1.0, 1.0, 1.0, 1.0));

    let render = |mode| {
        let mut renderer = Renderer::new(32, 32, 1).expect("test dimensions should be valid");
        renderer.rasterizer = Rasterizer::new();
        let state = RenderState {
            cull_mode: mode,
            ..Default::default()
        };
        draw_mesh(&mut renderer, &mesh, &shader, None, state);
        renderer.framebuffer.get_pixel(16, 16).unwrap()
    };

    let back = render(CullMode::Back);
    let front = render(CullMode::Front);
    assert_ne!(back.norm_squared() > 0.0, front.norm_squared() > 0.0);
}

#[test]
fn fragment_input_reports_triangle_facing() {
    let shader = FacingShader;
    let front_mesh = triangle(0.0, Vector4::zeros());
    let mut back_mesh = triangle(0.0, Vector4::zeros());
    back_mesh.indices = vec![0, 2, 1];
    let render = |mesh: &Mesh| {
        let mut renderer = Renderer::new(32, 32, 1).expect("test dimensions should be valid");
        draw_mesh(&mut renderer, mesh, &shader, None, test_render_state());
        renderer.framebuffer.get_pixel(16, 16).unwrap()
    };

    assert_vec3_approx(render(&front_mesh), Vector3::new(0.0, 1.0, 0.0));
    assert_vec3_approx(render(&back_mesh), Vector3::new(1.0, 0.0, 0.0));
}

#[test]
fn mirrored_render_state_inverts_culling_and_fragment_facing() {
    let mesh = triangle(0.0, Vector4::zeros());
    let render = |cull_mode, front_face_inverted| {
        let mut renderer = Renderer::new(32, 32, 1).expect("test dimensions should be valid");
        let state = RenderState {
            cull_mode,
            front_face_inverted,
            ..Default::default()
        };
        draw_mesh(&mut renderer, &mesh, &FacingShader, None, state);
        renderer.framebuffer.get_pixel(16, 16).unwrap()
    };

    assert_vec3_approx(render(CullMode::None, false), Vector3::new(0.0, 1.0, 0.0));
    assert_vec3_approx(render(CullMode::None, true), Vector3::new(1.0, 0.0, 0.0));
    assert!(render(CullMode::Back, false).norm_squared() > 0.0);
    assert_eq!(render(CullMode::Back, true), Vector3::zeros());
}

#[test]
fn double_sided_material_disables_culling_per_command() {
    let render = |double_sided| {
        let mut mesh = triangle(0.0, Vector4::zeros());
        mesh.indices = vec![0, 2, 1];
        let material = Material::Pbr(PbrMaterial {
            emissive: Vector3::new(1.0, 0.0, 0.0),
            double_sided,
            ..Default::default()
        });
        let context = RenderContext {
            camera: shadow_test_camera(),
            lights: Vec::new(),
            scene_objects: vec![SceneObject::new(
                SceneObjectKind::Model { config_index: 0 },
                Model::new(vec![mesh], vec![material]),
                Matrix4::identity(),
            )],
            shadow_light: None,
        };
        let config = rasterizer_rust::io::config::Config::default();
        let shadow = ShadowPassOutput {
            depth: None,
            size: 0,
            light_space_matrix: Matrix4::identity(),
            light_index: None,
        };
        let mut renderer = Renderer::new(32, 32, 1).expect("test dimensions should be valid");

        render_main_pass(
            &config,
            &context,
            &mut renderer,
            &shadow,
            RenderState::default(),
        )
        .expect("test scene should render");
        renderer.framebuffer.sample(16, 16).unwrap().depth
    };

    assert!(render(false).is_infinite());
    assert!(render(true).is_finite());
}
#[test]
fn triangle_rasterization_crosses_band_boundaries() {
    let shader = ClipSpaceShader;
    let color = Vector4::new(0.2, 0.4, 0.8, 1.0);
    let mut vertices = vec![
        Vertex::new(Point3::new(-1.0, -1.0, 0.0), Vector3::z(), Vector2::zeros()),
        Vertex::new(Point3::new(1.0, -1.0, 0.0), Vector3::z(), Vector2::zeros()),
        Vertex::new(Point3::new(1.0, 1.0, 0.0), Vector3::z(), Vector2::zeros()),
        Vertex::new(Point3::new(-1.0, 1.0, 0.0), Vector3::z(), Vector2::zeros()),
    ];
    for vertex in &mut vertices {
        vertex.tangent = color;
    }
    let mesh = Mesh::new(vertices, vec![0, 1, 2, 0, 2, 3], 0);
    let mut renderer = Renderer::new(48, 70, 1).expect("test dimensions should be valid");

    draw_mesh(&mut renderer, &mesh, &shader, None, test_render_state());

    for y in [0, 15, 16, 31, 32, 47, 48, 63, 64, 69] {
        assert_vec3_approx(renderer.framebuffer.get_pixel(24, y).unwrap(), color.xyz());
    }
}

#[test]
fn rasterizer_tracks_uv_density_per_texture_coordinate_set() {
    let mut mesh = triangle(0.0, Vector4::zeros());
    mesh.vertices[0].texcoords[1] = Vector2::new(0.0, 0.0);
    mesh.vertices[1].texcoords[1] = Vector2::new(1.0, 0.0);
    mesh.vertices[2].texcoords[1] = Vector2::new(0.0, 1.0);
    let mut renderer = Renderer::new(32, 32, 1).expect("test dimensions should be valid");

    draw_mesh(
        &mut renderer,
        &mesh,
        &DualUvDensityShader,
        None,
        test_render_state(),
    );

    let density = renderer.framebuffer.get_pixel(16, 16).unwrap();
    assert_eq!(density.x, 0.0);
    assert!(density.y > 0.0);
}

#[test]
fn overlapping_triangles_produce_deterministic_depth_and_color() {
    let shader = ClipSpaceShader;
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for triangle_index in 0..128_u32 {
        let is_nearest = triangle_index == 57;
        let z = if is_nearest {
            -0.9
        } else {
            -0.5 + triangle_index as f32 * 0.005
        };
        let color = if is_nearest {
            Vector4::new(0.0, 1.0, 0.0, 1.0)
        } else {
            Vector4::new(1.0, 0.0, 0.0, 1.0)
        };
        let base = vertices.len() as u32;
        let mesh = triangle(z, color);
        vertices.extend(mesh.vertices);
        indices.extend([base, base + 1, base + 2]);
    }
    let mesh = Mesh::new(vertices, indices, 0);

    for _ in 0..16 {
        let mut renderer = Renderer::new(64, 64, 1).expect("test dimensions should be valid");
        draw_mesh(&mut renderer, &mesh, &shader, None, test_render_state());

        let sample = renderer.framebuffer.sample(32, 32).unwrap();
        assert_vec3_approx(sample.color, Vector3::new(0.0, 1.0, 0.0));
        assert!((sample.depth - 0.05).abs() < 1e-5);
    }
}

#[test]
fn transparent_queue_sorts_back_to_front_and_preserves_band_order() {
    let shader = ClipSpaceShader;
    let far = triangle(0.5, Vector4::new(1.0, 0.0, 0.0, 0.5));
    let near = triangle(-0.5, Vector4::new(0.0, 0.0, 1.0, 0.5));
    let material = Material::Pbr(PbrMaterial {
        alpha_mode: AlphaMode::Blend,
        ..Default::default()
    });
    let mut renderer = Renderer::new(64, 64, 1).expect("test dimensions should be valid");
    let state = RenderState {
        blend_mode: BlendMode::Alpha,
        depth_write: false,
        ..test_render_state()
    };
    let mut queue = RenderQueue::default();
    queue.push(0, RenderGeometry::Mesh(&near), Some(&material), state, 0.5);
    queue.push(0, RenderGeometry::Mesh(&far), Some(&material), state, -0.5);
    queue.sort_transparent();
    renderer.draw_queue(&queue, std::slice::from_ref(&shader));

    for y in [8, 24, 40, 56] {
        assert_vec3_approx(
            renderer.framebuffer.get_pixel(32, y).unwrap(),
            Vector3::new(0.25, 0.0, 0.5),
        );
    }
}

#[test]
fn transparent_queue_uses_insertion_id_to_break_depth_ties() {
    let first = triangle(0.0, Vector4::zeros());
    let second = triangle(0.0, Vector4::zeros());
    let third = triangle(0.0, Vector4::zeros());
    let mut queue = RenderQueue::default();
    queue.push(
        0,
        RenderGeometry::Mesh(&first),
        None,
        test_render_state(),
        1.0,
    );
    queue.push(
        0,
        RenderGeometry::Mesh(&second),
        None,
        test_render_state(),
        -1.0,
    );
    queue.push(
        0,
        RenderGeometry::Mesh(&third),
        None,
        test_render_state(),
        1.0,
    );

    queue.sort_transparent();

    let ordering: Vec<(f32, u64)> = queue
        .commands()
        .iter()
        .map(|command| (command.sort_depth, command.insertion_id))
        .collect();
    assert_eq!(ordering, vec![(-1.0, 1), (1.0, 0), (1.0, 2)]);
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

#[test]
fn masked_shadow_fragments_respect_material_alpha() {
    let mut renderer = Renderer::new(32, 32, 1).expect("test dimensions should be valid");
    let shader = ShadowShader::new(
        Matrix4::identity(),
        Matrix4::identity(),
        Matrix4::identity(),
    );
    let mesh = triangle(0.0, Vector4::zeros());
    let discarded = Material::Pbr(PbrMaterial {
        alpha: 0.25,
        alpha_mode: AlphaMode::Mask(0.5),
        ..Default::default()
    });

    draw_mesh(
        &mut renderer,
        &mesh,
        &shader,
        Some(&discarded),
        test_render_state(),
    );
    assert!(
        renderer
            .framebuffer
            .sample(16, 16)
            .unwrap()
            .depth
            .is_infinite()
    );

    let visible = Material::Pbr(PbrMaterial {
        alpha: 0.75,
        alpha_mode: AlphaMode::Mask(0.5),
        ..Default::default()
    });
    draw_mesh(
        &mut renderer,
        &mesh,
        &shader,
        Some(&visible),
        test_render_state(),
    );
    assert!((renderer.framebuffer.sample(16, 16).unwrap().depth - 0.5).abs() < 1e-5);
}

#[test]
fn masked_shadow_fragments_sample_base_color_texture_alpha() {
    let image = TextureImage::from_image(
        DynamicImage::ImageRgba8(
            RgbaImage::from_vec(2, 1, vec![255, 255, 255, 255, 255, 255, 255, 0])
                .expect("test pixels should match image dimensions"),
        ),
        false,
    );
    let texture = TextureBinding::new(
        Arc::new(image),
        SamplerState {
            wrap_u: WrapMode::ClampToEdge,
            wrap_v: WrapMode::ClampToEdge,
            mag_filter: MagFilter::Nearest,
            min_filter: MinFilter::Nearest,
        },
        TexCoordSet::TexCoord1,
        TextureUsage::Color,
    );
    let material = Material::Pbr(PbrMaterial {
        alpha_mode: AlphaMode::Mask(0.5),
        albedo_texture: Some(texture),
        ..Default::default()
    });
    let mut renderer = Renderer::new(32, 32, 1).expect("test dimensions should be valid");
    let shader = ShadowShader::new(
        Matrix4::identity(),
        Matrix4::identity(),
        Matrix4::identity(),
    );
    let mut mesh = triangle(0.0, Vector4::zeros());
    for vertex in &mut mesh.vertices {
        vertex.texcoords[0] = Vector2::new(0.25, 0.5);
        vertex.texcoords[1] = Vector2::new(0.75, 0.5);
    }

    draw_mesh(
        &mut renderer,
        &mesh,
        &shader,
        Some(&material),
        test_render_state(),
    );

    assert!(
        renderer
            .framebuffer
            .sample(16, 16)
            .unwrap()
            .depth
            .is_infinite()
    );
}

#[test]
fn blended_materials_do_not_write_shadow_depth() {
    let material = Material::Pbr(PbrMaterial {
        alpha_mode: AlphaMode::Blend,
        ..Default::default()
    });
    let context = RenderContext {
        camera: shadow_test_camera(),
        lights: vec![Light::new_directional(
            Vector3::new(0.0, 0.0, -1.0),
            Vector3::new(1.0, 1.0, 1.0),
            1.0,
        )],
        scene_objects: vec![SceneObject::new(
            SceneObjectKind::Model { config_index: 0 },
            Model::new(vec![triangle(0.0, Vector4::zeros())], vec![material]),
            Matrix4::identity(),
        )],
        shadow_light: Some(ShadowLight {
            light_index: 0,
            position: Point3::new(0.0, 0.0, 2.0),
        }),
    };
    let mut config = rasterizer_rust::io::config::Config::default();
    config.render.shadow_map_size = 32;
    let mut renderer = Renderer::new(32, 32, 1).expect("test dimensions should be valid");

    let shadow = render_shadow_pass(&config, &context, &mut renderer);

    assert!(
        shadow
            .depth
            .unwrap()
            .iter()
            .all(|depth| depth.is_infinite())
    );
}

#[test]
fn double_sided_material_disables_shadow_culling_per_command() {
    let render = |double_sided| {
        let mut mesh = triangle(0.0, Vector4::zeros());
        mesh.indices = vec![0, 2, 1];
        let material = Material::Pbr(PbrMaterial {
            double_sided,
            ..Default::default()
        });
        let context = RenderContext {
            camera: shadow_test_camera(),
            lights: vec![Light::new_directional(
                Vector3::new(0.0, 0.0, -1.0),
                Vector3::new(1.0, 1.0, 1.0),
                1.0,
            )],
            scene_objects: vec![SceneObject::new(
                SceneObjectKind::Model { config_index: 0 },
                Model::new(vec![mesh], vec![material]),
                Matrix4::identity(),
            )],
            shadow_light: Some(ShadowLight {
                light_index: 0,
                position: Point3::new(0.0, 0.0, 2.0),
            }),
        };
        let mut config = rasterizer_rust::io::config::Config::default();
        config.render.shadow_map_size = 32;
        let mut renderer = Renderer::new(32, 32, 1).expect("test dimensions should be valid");

        render_shadow_pass(&config, &context, &mut renderer)
            .depth
            .unwrap()
    };

    assert!(render(false).iter().all(|depth| depth.is_infinite()));
    assert!(render(true).iter().any(|depth| depth.is_finite()));
}

#[test]
fn point_only_scene_disables_shadow_pass() {
    let context = RenderContext {
        camera: shadow_test_camera(),
        lights: vec![Light::new_point(
            Point3::new(0.0, 1.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0),
            1.0,
        )],
        scene_objects: Vec::new(),
        shadow_light: None,
    };
    let config = rasterizer_rust::io::config::Config::default();
    let mut renderer = Renderer::new(32, 32, 1).expect("test dimensions should be valid");

    let shadow = render_shadow_pass(&config, &context, &mut renderer);

    assert!(shadow.depth.is_none());
    assert_eq!(shadow.size, 0);
    assert!(shadow.light_index.is_none());
}

#[test]
fn pbr_material_texture_binding_selects_texcoord1() {
    let image = TextureImage::from_image(
        DynamicImage::ImageRgba8(
            RgbaImage::from_vec(2, 1, vec![255, 0, 0, 255, 0, 255, 0, 255])
                .expect("test pixels should match image dimensions"),
        ),
        false,
    );
    let material = Material::Pbr(PbrMaterial {
        albedo: Vector3::new(0.25, 0.5, 0.75),
        albedo_texture: Some(TextureBinding::new(
            Arc::new(image),
            SamplerState {
                wrap_u: WrapMode::ClampToEdge,
                wrap_v: WrapMode::ClampToEdge,
                mag_filter: MagFilter::Nearest,
                min_filter: MinFilter::Nearest,
            },
            TexCoordSet::TexCoord1,
            TextureUsage::Color,
        )),
        ..Default::default()
    });
    let varying = PbrVarying {
        world_pos: Point3::origin(),
        normal: Vector3::z(),
        uvs: [Vector2::new(0.25, 0.5), Vector2::new(0.75, 0.5)],
        tangent: Vector4::new(1.0, 0.0, 0.0, 1.0),
    };
    let mut shader = PbrShader::new(
        Matrix4::identity(),
        Matrix4::identity(),
        Matrix4::identity(),
        Point3::new(0.0, 0.0, 2.0),
    );
    shader.ambient_light = Vector3::repeat(1.0);

    let color = match shader.fragment(
        FragmentInput {
            varying,
            front_facing: true,
            uv_densities: [0.0; 2],
        },
        Some(&material),
    ) {
        FragmentOutput::Color(color) => color,
        FragmentOutput::Discard => panic!("opaque PBR fragment should produce color"),
    };

    assert_vec3_approx(color.xyz(), Vector3::new(0.0, 0.5, 0.0));
}

#[test]
fn pbr_vertex_preserves_tangent_frame_under_mirrored_non_uniform_scale() {
    let model = Matrix4::new_nonuniform_scaling(&Vector3::new(-2.0, 3.0, 0.5));
    let shader = PbrShader::new(
        model,
        Matrix4::identity(),
        Matrix4::identity(),
        Point3::new(0.0, 0.0, 2.0),
    );
    let mut vertex = Vertex::new(Point3::origin(), Vector3::z(), Vector2::zeros());
    vertex.tangent = Vector4::new(1.0, 1.0, 0.0, 1.0);

    let (_, varying) = shader.vertex(&vertex);

    assert_vec3_approx(varying.normal, Vector3::z());
    assert_vec3_approx(
        varying.tangent.xyz(),
        Vector3::new(-2.0, 3.0, 0.0).normalize(),
    );
    assert!(varying.normal.dot(&varying.tangent.xyz()).abs() < 1e-5);
    assert_eq!(varying.tangent.w, -1.0);
}

#[test]
fn pbr_shadow_uses_recorded_light_index() {
    let varying = PbrVarying {
        world_pos: Point3::origin(),
        normal: Vector3::z(),
        uvs: [Vector2::zeros(); 2],
        tangent: Vector4::new(1.0, 0.0, 0.0, 1.0),
    };
    let material = Material::Pbr(PbrMaterial {
        albedo: Vector3::new(1.0, 1.0, 1.0),
        roughness: 0.5,
        ..Default::default()
    });
    let render = |shadow_light_index| {
        let mut shader = PbrShader::new(
            Matrix4::identity(),
            Matrix4::identity(),
            Matrix4::identity(),
            Point3::new(0.0, 0.0, 2.0),
        );
        shader.lights = vec![
            Light::new_directional(
                Vector3::new(0.0, 0.0, -1.0),
                Vector3::new(1.0, 0.0, 0.0),
                1.0,
            ),
            Light::new_directional(
                Vector3::new(0.0, 0.0, -1.0),
                Vector3::new(0.0, 1.0, 0.0),
                1.0,
            ),
        ];
        shader.ambient_light = Vector3::zeros();
        shader.shadow_map = Some(Arc::new(vec![0.0]));
        shader.shadow_map_size = 1;
        shader.shadow_light_index = Some(shadow_light_index);
        shader.light_space_matrix = Matrix4::identity();
        shader.shadow_bias = 0.0;
        shader.use_pcf = false;
        match shader.fragment(
            FragmentInput {
                varying,
                front_facing: true,
                uv_densities: [0.0; 2],
            },
            Some(&material),
        ) {
            FragmentOutput::Color(color) => color.xyz(),
            FragmentOutput::Discard => panic!("opaque PBR fragment should produce color"),
        }
    };

    let second_light_shadowed = render(1);
    assert!(second_light_shadowed.x > 0.0);
    assert_eq!(second_light_shadowed.y, 0.0);

    let first_light_shadowed = render(0);
    assert_eq!(first_light_shadowed.x, 0.0);
    assert!(first_light_shadowed.y > 0.0);
}

#[test]
fn pbr_back_faces_flip_the_geometric_normal() {
    let varying = PbrVarying {
        world_pos: Point3::origin(),
        normal: Vector3::z(),
        uvs: [Vector2::zeros(); 2],
        tangent: Vector4::new(1.0, 0.0, 0.0, 1.0),
    };
    let material = Material::Pbr(PbrMaterial {
        albedo: Vector3::new(1.0, 1.0, 1.0),
        roughness: 0.5,
        double_sided: true,
        ..Default::default()
    });
    let mut shader = PbrShader::new(
        Matrix4::identity(),
        Matrix4::identity(),
        Matrix4::identity(),
        Point3::new(0.0, 0.0, -2.0),
    );
    shader.lights = vec![Light::new_directional(
        Vector3::z(),
        Vector3::new(1.0, 1.0, 1.0),
        1.0,
    )];
    shader.ambient_light = Vector3::zeros();
    let shade = |front_facing| match shader.fragment(
        FragmentInput {
            varying,
            front_facing,
            uv_densities: [0.0; 2],
        },
        Some(&material),
    ) {
        FragmentOutput::Color(color) => color.xyz(),
        FragmentOutput::Discard => panic!("opaque PBR fragment should produce color"),
    };

    assert_eq!(shade(true), Vector3::zeros());
    assert!(shade(false).norm_squared() > 0.0);
}

#[test]
fn shadow_output_reports_actual_buffer_size() {
    let context = RenderContext {
        camera: shadow_test_camera(),
        lights: vec![Light::new_directional(
            Vector3::new(0.0, 0.0, -1.0),
            Vector3::new(1.0, 1.0, 1.0),
            1.0,
        )],
        scene_objects: Vec::new(),
        shadow_light: Some(ShadowLight {
            light_index: 0,
            position: Point3::new(0.0, 0.0, 2.0),
        }),
    };
    let mut config = rasterizer_rust::io::config::Config::default();
    config.render.shadow_map_size = 64;
    let mut renderer = Renderer::new(16, 16, 1).expect("test dimensions should be valid");

    let shadow = render_shadow_pass(&config, &context, &mut renderer);

    assert_eq!(shadow.size, 16);
    assert_eq!(shadow.depth.unwrap().len(), 16 * 16);
    assert_eq!(shadow.light_index, Some(0));
}
