use nalgebra::{Matrix4, Point3, Vector2, Vector3, Vector4};
use rasterizer_rust::core::framebuffer::FrameBuffer;
use rasterizer_rust::core::geometry::Vertex;
use rasterizer_rust::core::pipeline::{Interpolatable, Shader};
use rasterizer_rust::core::rasterizer::{CullMode, Rasterizer};
use rasterizer_rust::pipeline::passes::post_process_to_buffer;
use rasterizer_rust::pipeline::renderer::Renderer;
use rasterizer_rust::pipeline::shaders::pbr::PbrShader;
use rasterizer_rust::scene::material::{AlphaMode, Material, PbrMaterial};
use rasterizer_rust::scene::mesh::Mesh;
use std::ops::{Add, Mul};

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

struct ClipSpaceShader;

impl Shader for ClipSpaceShader {
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
        varying: Self::Varying,
        _material: Option<&Material>,
        _uv_density: f32,
    ) -> Vector4<f32> {
        varying.color
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

#[test]
fn nearer_triangle_wins_depth_test() {
    let shader = ClipSpaceShader;
    let mut renderer = Renderer::new(32, 32, 1);
    renderer.rasterizer.set_cull_mode(CullMode::None);

    let far = triangle(0.5, Vector4::new(1.0, 0.0, 0.0, 1.0));
    let near = triangle(-0.5, Vector4::new(0.0, 1.0, 0.0, 1.0));
    renderer.draw_mesh(&far, &shader, None);
    renderer.draw_mesh(&near, &shader, None);

    assert_vec3_approx(
        renderer.framebuffer.get_pixel(16, 16).unwrap(),
        Vector3::new(0.0, 1.0, 0.0),
    );
}

#[test]
fn triangle_crossing_near_plane_is_clipped_and_rendered() {
    let shader = ClipSpaceShader;
    let mut renderer = Renderer::new(32, 32, 1);
    renderer.rasterizer.set_cull_mode(CullMode::None);

    let mut mesh = triangle(0.0, Vector4::new(1.0, 0.0, 1.0, 1.0));
    mesh.vertices[0].position.z = -2.0;
    renderer.draw_mesh(&mesh, &shader, None);

    let colored_pixels = (0..32)
        .flat_map(|y| (0..32).map(move |x| (x, y)))
        .filter(|&(x, y)| renderer.framebuffer.get_pixel(x, y).unwrap().norm_squared() > 0.0)
        .count();
    assert!(colored_pixels > 0);
}

#[test]
fn alpha_mask_discards_fragments_below_cutoff() {
    let shader = ClipSpaceShader;
    let mut renderer = Renderer::new(32, 32, 1);
    renderer.rasterizer.set_cull_mode(CullMode::None);
    let mesh = triangle(0.0, Vector4::new(1.0, 0.0, 0.0, 0.25));
    let material = Material::Pbr(PbrMaterial {
        alpha_mode: AlphaMode::Mask(0.5),
        ..Default::default()
    });

    renderer.draw_mesh(&mesh, &shader, Some(&material));

    assert_eq!(
        renderer.framebuffer.get_pixel(16, 16).unwrap(),
        Vector3::zeros()
    );
    assert!(renderer.framebuffer.test_depth(16, 16, 0.5));
}

#[test]
fn framebuffer_resolves_supersampled_pixels() {
    let framebuffer = FrameBuffer::new(1, 1, 2);
    framebuffer.set_pixel_safe(0, 0, Vector3::new(1.0, 0.0, 0.0));
    framebuffer.set_pixel_safe(1, 0, Vector3::new(0.0, 1.0, 0.0));
    framebuffer.set_pixel_safe(0, 1, Vector3::new(0.0, 0.0, 1.0));
    framebuffer.set_pixel_safe(1, 1, Vector3::new(1.0, 1.0, 1.0));

    assert_vec3_approx(
        framebuffer.get_pixel(0, 0).unwrap(),
        Vector3::new(0.5, 0.5, 0.5),
    );
}

#[test]
fn headless_pbr_triangle_produces_visible_output() {
    let mut renderer = Renderer::new(32, 32, 1);
    renderer.rasterizer.set_cull_mode(CullMode::None);
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

    renderer.draw_mesh(&mesh, &shader, Some(&material));

    let mut config = rasterizer_rust::io::config::Config::default();
    config.render.width = 32;
    config.render.height = 32;
    config.render.use_aces = false;
    let mut output = vec![0; 32 * 32];
    post_process_to_buffer(&renderer.framebuffer, &mut output, &config);

    assert_ne!(output[16 * 32 + 16] & 0x00ff_ffff, 0);
}

#[test]
fn cull_mode_can_reject_one_winding() {
    let shader = ClipSpaceShader;
    let mesh = triangle(0.0, Vector4::new(1.0, 1.0, 1.0, 1.0));

    let render = |mode| {
        let mut renderer = Renderer::new(32, 32, 1);
        renderer.rasterizer = Rasterizer::new();
        renderer.rasterizer.set_cull_mode(mode);
        renderer.draw_mesh(&mesh, &shader, None);
        renderer.framebuffer.get_pixel(16, 16).unwrap()
    };

    let back = render(CullMode::Back);
    let front = render(CullMode::Front);
    assert_ne!(back.norm_squared() > 0.0, front.norm_squared() > 0.0);
}
