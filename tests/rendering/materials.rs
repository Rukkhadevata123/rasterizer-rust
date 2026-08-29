use super::*;
use rayon::ThreadPoolBuilder;

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
fn main_pass_combines_opaque_masked_and_transparent_phases() {
    let make_object = |config_index, mesh, emissive, alpha, alpha_mode| {
        SceneObject::new(
            SceneObjectKind::Model { config_index },
            Model::new(
                vec![mesh],
                vec![Material::Pbr(PbrMaterial {
                    albedo: Vector3::zeros(),
                    emissive,
                    alpha,
                    alpha_mode,
                    double_sided: true,
                    ..Default::default()
                })],
            ),
            Matrix4::identity(),
        )
    };

    let opaque = triangle(0.0, Vector4::zeros());
    let discarded_mask = triangle(0.6, Vector4::zeros());
    let mut visible_mask = triangle(0.4, Vector4::zeros());
    visible_mask.vertices[0].position.x = -0.65;
    visible_mask.vertices[1].position.x = -0.05;
    visible_mask.vertices[2].position.x = -0.35;
    visible_mask.vertices[0].position.y = -0.45;
    visible_mask.vertices[1].position.y = -0.45;
    visible_mask.vertices[2].position.y = 0.45;
    let transparent = triangle(0.8, Vector4::zeros());

    let camera = shadow_test_camera();
    let context = RenderContext {
        camera: camera.clone(),
        lights: Vec::new(),
        scene_objects: vec![
            make_object(0, opaque, Vector3::x(), 1.0, AlphaMode::Opaque),
            make_object(1, discarded_mask, Vector3::y(), 0.25, AlphaMode::Mask(0.5)),
            make_object(2, visible_mask, Vector3::y(), 0.75, AlphaMode::Mask(0.5)),
            make_object(3, transparent, Vector3::z(), 0.5, AlphaMode::Blend),
        ],
        shadow_light: None,
    };
    let mut config = rasterizer_rust::io::config::Config::default();
    config.render.width = 64;
    config.render.height = 64;
    config.render.supersample_scale = 1;
    config.render.ambient_light = [0.0, 0.0, 0.0];
    config.render.background_color = Some([0.0, 0.0, 0.0]);
    config.render.background_gradient_top = None;
    config.render.background_gradient_bottom = None;
    config.render.background_image = None;
    config.render.use_shadows = false;

    let shadow = ShadowPassOutput {
        depth: None,
        size: 0,
        light_space_matrix: Matrix4::identity(),
        light_index: None,
    };
    let mut renderer = Renderer::new(64, 64, 1).expect("test dimensions should be valid");
    render_main_pass(
        &config,
        &context,
        &mut renderer,
        &shadow,
        RenderState {
            cull_mode: CullMode::None,
            ..Default::default()
        },
    )
    .expect("mixed-phase scene should render");

    let center = renderer.framebuffer.sample(32, 32).unwrap();
    assert_vec3_approx(center.color, Vector3::new(0.5, 0.0, 0.5));

    let masked = renderer.framebuffer.sample(22, 32).unwrap();
    assert_vec3_approx(masked.color, Vector3::new(0.0, 0.5, 0.5));

    let expected_depth = |world_z| {
        let clip = camera.projection_matrix()
            * camera.view_matrix()
            * Point3::new(0.0, 0.0, world_z).to_homogeneous();
        clip.z / clip.w * 0.5 + 0.5
    };
    assert!((center.depth - expected_depth(0.0)).abs() < 1.0e-6);
    assert!((masked.depth - expected_depth(0.4)).abs() < 1.0e-6);
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

#[test]
fn transparent_rendering_is_deterministic_across_worker_counts() {
    let render = |thread_count| {
        ThreadPoolBuilder::new()
            .num_threads(thread_count)
            .build()
            .expect("test thread pool should build")
            .install(|| {
                let shader = ClipSpaceShader;
                let material = Material::Pbr(PbrMaterial {
                    alpha_mode: AlphaMode::Blend,
                    ..Default::default()
                });
                let state = RenderState {
                    blend_mode: BlendMode::Alpha,
                    depth_write: false,
                    ..test_render_state()
                };
                let mut layers = [
                    (triangle(0.6, Vector4::new(1.0, 0.0, 0.0, 0.35)), -0.8),
                    (triangle(0.2, Vector4::new(0.0, 1.0, 0.0, 0.45)), 0.0),
                    (triangle(-0.2, Vector4::new(0.0, 0.0, 1.0, 0.55)), 0.0),
                    (triangle(-0.6, Vector4::new(1.0, 1.0, 0.0, 0.25)), 0.6),
                ];
                for (mesh, _) in &mut layers {
                    mesh.vertices[0].position.x = -1.4;
                    mesh.vertices[1].position.x = 1.4;
                    mesh.vertices[0].position.y = -0.95;
                    mesh.vertices[1].position.y = -0.95;
                    mesh.vertices[2].position.y = 0.95;
                }

                let mut queue = RenderQueue::with_capacity(layers.len());
                for (mesh, sort_depth) in &layers {
                    queue.push(
                        0,
                        RenderGeometry::Mesh(mesh),
                        Some(&material),
                        state,
                        *sort_depth,
                    );
                }
                queue.sort_transparent();

                let (width, height) = (96, 80);
                let mut renderer =
                    Renderer::new(width, height, 1).expect("test dimensions should be valid");
                renderer.draw_queue(&queue, std::slice::from_ref(&shader));

                let mut config = rasterizer_rust::io::config::Config::default();
                config.render.width = width;
                config.render.height = height;
                config.render.exposure = 1.0;
                config.render.use_aces = false;
                let mut output = vec![0; width * height];
                post_process_to_buffer(&renderer.framebuffer, &mut output, &config);
                (output, renderer.framebuffer.depth_values())
            })
    };

    let single_worker = render(1);
    let four_workers = render(4);
    assert_eq!(single_worker.0, four_workers.0);
    assert_eq!(single_worker.1, four_workers.1);
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
        let lights = [
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
        let mut shader = PbrShader::new(
            Matrix4::identity(),
            Matrix4::identity(),
            Matrix4::identity(),
            Point3::new(0.0, 0.0, 2.0),
        );
        shader.lights = &lights;
        shader.ambient_light = Vector3::zeros();
        let shadow_map = vec![0.0];
        shader.shadow_map = Some(&shadow_map);
        shader.shadow_map_size = 1;
        shader.shadow_light_index = Some(shadow_light_index);
        shader.light_space_matrix = Matrix4::identity();
        shader.shadow_constant_bias = 0.0;
        shader.shadow_slope_bias = 0.0;
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
    let lights = [Light::new_directional(
        Vector3::z(),
        Vector3::new(1.0, 1.0, 1.0),
        1.0,
    )];
    let mut shader = PbrShader::new(
        Matrix4::identity(),
        Matrix4::identity(),
        Matrix4::identity(),
        Point3::new(0.0, 0.0, -2.0),
    );
    shader.lights = &lights;
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
