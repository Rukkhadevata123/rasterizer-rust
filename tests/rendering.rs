use image::{DynamicImage, Rgba, RgbaImage};
use nalgebra::{Matrix4, Point3, Vector2, Vector3, Vector4};
use rasterizer_rust::core::framebuffer::FrameBuffer;
use rasterizer_rust::core::geometry::Vertex;
use rasterizer_rust::core::pipeline::{FragmentOutput, Interpolatable, Shader};
use rasterizer_rust::core::rasterizer::{CullMode, Rasterizer};
use rasterizer_rust::pipeline::passes::{post_process_to_buffer, render_shadow_pass};
use rasterizer_rust::pipeline::renderer::Renderer;
use rasterizer_rust::pipeline::shaders::pbr::{PbrShader, PbrVarying};
use rasterizer_rust::pipeline::shaders::shadow::ShadowShader;
use rasterizer_rust::scene::camera::Camera;
use rasterizer_rust::scene::context::{RenderContext, ShadowLight};
use rasterizer_rust::scene::light::Light;
use rasterizer_rust::scene::material::{AlphaMode, Material, PbrMaterial};
use rasterizer_rust::scene::mesh::Mesh;
use rasterizer_rust::scene::model::Model;
use rasterizer_rust::scene::scene_object::SceneObject;
use rasterizer_rust::scene::texture::Texture;
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
        varying: Self::Varying,
        material: Option<&'a Material>,
        _uv_density: f32,
    ) -> FragmentOutput {
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
    let mut renderer = Renderer::new(32, 32, 1).expect("test dimensions should be valid");
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
    let mut renderer = Renderer::new(32, 32, 1).expect("test dimensions should be valid");
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
    let mut renderer = Renderer::new(32, 32, 1).expect("test dimensions should be valid");
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
fn masked_pbr_fragments_respect_material_alpha() {
    let mut renderer = Renderer::new(32, 32, 1).expect("test dimensions should be valid");
    renderer.rasterizer.set_cull_mode(CullMode::None);
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

    renderer.draw_mesh(&mesh, &shader, Some(&discarded));
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
    renderer.draw_mesh(&mesh, &shader, Some(&visible));
    assert!((renderer.framebuffer.sample(16, 16).unwrap().depth - 0.5).abs() < 1e-5);
}

#[test]
fn cull_mode_can_reject_one_winding() {
    let shader = ClipSpaceShader;
    let mesh = triangle(0.0, Vector4::new(1.0, 1.0, 1.0, 1.0));

    let render = |mode| {
        let mut renderer = Renderer::new(32, 32, 1).expect("test dimensions should be valid");
        renderer.rasterizer = Rasterizer::new();
        renderer.rasterizer.set_cull_mode(mode);
        renderer.draw_mesh(&mesh, &shader, None);
        renderer.framebuffer.get_pixel(16, 16).unwrap()
    };

    let back = render(CullMode::Back);
    let front = render(CullMode::Front);
    assert_ne!(back.norm_squared() > 0.0, front.norm_squared() > 0.0);
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
    renderer.rasterizer.set_cull_mode(CullMode::None);

    renderer.draw_mesh(&mesh, &shader, None);

    for y in [0, 15, 16, 31, 32, 47, 48, 63, 64, 69] {
        assert_vec3_approx(renderer.framebuffer.get_pixel(24, y).unwrap(), color.xyz());
    }
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
        renderer.rasterizer.set_cull_mode(CullMode::None);
        renderer.draw_mesh(&mesh, &shader, None);

        let sample = renderer.framebuffer.sample(32, 32).unwrap();
        assert_vec3_approx(sample.color, Vector3::new(0.0, 1.0, 0.0));
        assert!((sample.depth - 0.05).abs() < 1e-5);
    }
}

#[test]
fn transparent_triangles_preserve_input_order_within_each_band() {
    let shader = ClipSpaceShader;
    let far = triangle(0.5, Vector4::new(1.0, 0.0, 0.0, 0.5));
    let near = triangle(-0.5, Vector4::new(0.0, 0.0, 1.0, 0.5));
    let material = Material::Pbr(PbrMaterial {
        alpha_mode: AlphaMode::Blend,
        ..Default::default()
    });
    let mut renderer = Renderer::new(64, 64, 1).expect("test dimensions should be valid");
    renderer.rasterizer.set_cull_mode(CullMode::None);
    renderer.rasterizer.blend_mode = rasterizer_rust::core::rasterizer::BlendMode::Alpha;
    renderer.draw_sorted_triangles(
        vec![
            (
                &far.vertices[0],
                &far.vertices[1],
                &far.vertices[2],
                &material,
            ),
            (
                &near.vertices[0],
                &near.vertices[1],
                &near.vertices[2],
                &material,
            ),
        ],
        &shader,
    );

    for y in [8, 24, 40, 56] {
        assert_vec3_approx(
            renderer.framebuffer.get_pixel(32, y).unwrap(),
            Vector3::new(0.25, 0.0, 0.5),
        );
    }
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
    renderer.rasterizer.set_cull_mode(CullMode::None);
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

    renderer.draw_mesh(&mesh, &shader, Some(&discarded));
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
    renderer.draw_mesh(&mesh, &shader, Some(&visible));
    assert!((renderer.framebuffer.sample(16, 16).unwrap().depth - 0.5).abs() < 1e-5);
}

#[test]
fn masked_shadow_fragments_sample_base_color_texture_alpha() {
    let texture = Texture::from_image(
        DynamicImage::ImageRgba8(RgbaImage::from_pixel(1, 1, Rgba([255, 255, 255, 0]))),
        false,
    );
    let material = Material::Pbr(PbrMaterial {
        alpha_mode: AlphaMode::Mask(0.5),
        albedo_texture: Some(Arc::new(texture)),
        ..Default::default()
    });
    let mut renderer = Renderer::new(32, 32, 1).expect("test dimensions should be valid");
    renderer.rasterizer.set_cull_mode(CullMode::None);
    let shader = ShadowShader::new(
        Matrix4::identity(),
        Matrix4::identity(),
        Matrix4::identity(),
    );

    renderer.draw_mesh(&triangle(0.0, Vector4::zeros()), &shader, Some(&material));

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
    renderer.rasterizer.set_cull_mode(CullMode::None);

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
fn pbr_shadow_uses_recorded_light_index() {
    let varying = PbrVarying {
        world_pos: Point3::origin(),
        normal: Vector3::z(),
        uv: Vector2::zeros(),
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
        match shader.fragment(varying, Some(&material), 0.0) {
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
