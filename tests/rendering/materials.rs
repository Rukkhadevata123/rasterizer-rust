use super::*;
use rayon::ThreadPoolBuilder;

#[test]
fn alpha_mask_discards_fragments_below_cutoff() {
    let shader = ClipSpaceShader;
    let mut renderer = TestRenderHarness::new(32, 32, 1);
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
        test_pipeline_state(),
    );

    assert_eq!(
        renderer.framebuffer().get_pixel(16, 16).unwrap(),
        Vector3::zeros()
    );
    assert!(
        renderer
            .framebuffer()
            .sample(16, 16)
            .unwrap()
            .depth
            .is_infinite()
    );
}

#[test]
fn headless_pbr_triangle_produces_visible_output() {
    let mut renderer = TestRenderHarness::new(32, 32, 1);
    let mesh = triangle(0.0, Vector4::zeros());
    let material = Material::Pbr(PbrMaterial {
        albedo: Vector3::new(0.8, 0.2, 0.1),
        emissive: Vector3::new(0.2, 0.1, 0.05),
        ..Default::default()
    });

    draw_pbr_mesh(
        &mut renderer,
        &mesh,
        Some(&material),
        Matrix4::identity(),
        test_pipeline_state(),
    );

    let mut config = rasterizer_rust::io::config::Config::default();
    config.render.width = 32;
    config.render.height = 32;
    config.render.use_aces = false;
    let mut present = PresentBuffer::new(32, 32).expect("present dimensions should be valid");
    execute_resolve_tonemap_pass(ResolveTonemapPassDescriptor {
        label: Some("test-present"),
        source: &renderer.target,
        destination: &mut present,
        exposure: config.render.exposure,
        tonemap: TonemapOperator::None,
    })
    .expect("resolve-tonemap pass should succeed");

    assert_ne!(present.pixels()[16 * 32 + 16] & 0x00ff_ffff, 0);
}

#[test]
fn pbr_draw_context_applies_distinct_object_and_material_bindings() {
    let vertices = vec![
        Vertex::new(
            Point3::new(-0.25, -0.8, 0.0),
            Vector3::z(),
            Vector2::zeros(),
        ),
        Vertex::new(Point3::new(0.25, -0.8, 0.0), Vector3::z(), Vector2::zeros()),
        Vertex::new(Point3::new(0.0, 0.8, 0.0), Vector3::z(), Vector2::zeros()),
    ];
    let mesh = Mesh::new(vertices, vec![0, 1, 2, 0, 2, 1], 0);
    let mut frame = PbrFrameBindings::new(
        Matrix4::identity(),
        Matrix4::identity(),
        Point3::new(0.0, 0.0, 2.0),
    );
    frame.ambient_light = Vector3::repeat(1.0);
    let object_bindings = [
        PbrObjectBindings::new(Matrix4::new_translation(&Vector3::new(-0.5, 0.0, 0.0))),
        PbrObjectBindings::new(Matrix4::new_translation(&Vector3::new(0.5, 0.0, 0.0))),
    ];
    let materials = [
        PbrMaterial {
            albedo: Vector3::x(),
            ..Default::default()
        },
        PbrMaterial {
            albedo: Vector3::y(),
            ..Default::default()
        },
    ];
    let pipeline = GraphicsPipeline::new(
        PbrShader,
        test_pipeline_state(),
        VertexProgramId::from_pass_index(0),
    );
    let mut phase = RenderPhase::with_capacity(2);
    for index in 0..2 {
        phase.push(
            &pipeline,
            RenderGeometry::Mesh(&mesh),
            PbrDrawContext::new(
                &frame,
                &object_bindings[index],
                PbrMaterialBindings::from_pbr(&materials[index]),
            ),
            ObjectBindingId::from_pass_index(index),
            0.0,
        );
    }
    let mut renderer = TestRenderHarness::new(64, 32, 1);

    renderer
        .backend
        .execute_phase(renderer.target.render_target_mut(), &phase);

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
    let context = RenderScene {
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
    let mut renderer = TestRenderHarness::new(64, 64, 1);
    render_main_pass(
        &config,
        &context,
        &mut renderer.backend,
        &mut renderer.target,
        &mut renderer.resources,
        &shadow,
        GraphicsPipelineState {
            primitive: PrimitiveState {
                cull_mode: CullMode::None,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .expect("mixed-phase scene should render");

    let center = renderer.framebuffer().sample(32, 32).unwrap();
    assert_vec3_approx(center.color, Vector3::new(0.5, 0.0, 0.5));

    let masked = renderer.framebuffer().sample(22, 32).unwrap();
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
    let mut renderer = TestRenderHarness::new(32, 32, 1);
    let mesh = triangle(0.0, Vector4::zeros());
    let discarded = Material::Pbr(PbrMaterial {
        alpha: 0.25,
        alpha_mode: AlphaMode::Mask(0.5),
        ..Default::default()
    });

    draw_pbr_mesh(
        &mut renderer,
        &mesh,
        Some(&discarded),
        Matrix4::identity(),
        test_pipeline_state(),
    );
    assert!(
        renderer
            .framebuffer()
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
    draw_pbr_mesh(
        &mut renderer,
        &mesh,
        Some(&visible),
        Matrix4::identity(),
        test_pipeline_state(),
    );
    assert!((renderer.framebuffer().sample(16, 16).unwrap().depth - 0.5).abs() < 1e-5);
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
        let context = RenderScene {
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
        let mut renderer = TestRenderHarness::new(32, 32, 1);

        render_main_pass(
            &config,
            &context,
            &mut renderer.backend,
            &mut renderer.target,
            &mut renderer.resources,
            &shadow,
            GraphicsPipelineState::default(),
        )
        .expect("test scene should render");
        renderer.framebuffer().sample(16, 16).unwrap().depth
    };

    assert!(render(false).is_infinite());
    assert!(render(true).is_finite());
}

#[test]
fn transparent_phase_sorts_back_to_front_and_preserves_band_order() {
    let shader = ClipSpaceShader;
    let far = triangle(0.5, Vector4::new(1.0, 0.0, 0.0, 0.5));
    let near = triangle(-0.5, Vector4::new(0.0, 0.0, 1.0, 0.5));
    let material = Material::Pbr(PbrMaterial {
        alpha_mode: AlphaMode::Blend,
        ..Default::default()
    });
    let mut renderer = TestRenderHarness::new(64, 64, 1);
    let state = GraphicsPipelineState {
        color_target: Some(ColorTargetState {
            blend: Some(BlendState::Alpha),
        }),
        depth_stencil: Some(DepthStencilState {
            depth_write_enabled: false,
            ..Default::default()
        }),
        ..test_pipeline_state()
    };
    let pipeline = GraphicsPipeline::new(shader, state, VertexProgramId::from_pass_index(0));
    let mut phase = RenderPhase::default();
    phase.push(
        &pipeline,
        RenderGeometry::Mesh(&near),
        Some(&material),
        ObjectBindingId::from_pass_index(0),
        0.5,
    );
    phase.push(
        &pipeline,
        RenderGeometry::Mesh(&far),
        Some(&material),
        ObjectBindingId::from_pass_index(0),
        -0.5,
    );
    phase.sort_transparent();
    renderer
        .backend
        .execute_phase(renderer.target.render_target_mut(), &phase);

    for y in [8, 24, 40, 56] {
        assert_vec3_approx(
            renderer.framebuffer().get_pixel(32, y).unwrap(),
            Vector3::new(0.25, 0.0, 0.5),
        );
    }
}
#[test]
fn transparent_phase_uses_insertion_id_to_break_depth_ties() {
    let first = triangle(0.0, Vector4::zeros());
    let second = triangle(0.0, Vector4::zeros());
    let third = triangle(0.0, Vector4::zeros());
    let pipeline = GraphicsPipeline::new(
        ClipSpaceShader,
        test_pipeline_state(),
        VertexProgramId::from_pass_index(0),
    );
    let mut phase: RenderPhase<'_, ClipSpaceShader, Option<&Material>> = RenderPhase::default();
    phase.push(
        &pipeline,
        RenderGeometry::Mesh(&first),
        None,
        ObjectBindingId::from_pass_index(0),
        1.0,
    );
    phase.push(
        &pipeline,
        RenderGeometry::Mesh(&second),
        None,
        ObjectBindingId::from_pass_index(0),
        -1.0,
    );
    phase.push(
        &pipeline,
        RenderGeometry::Mesh(&third),
        None,
        ObjectBindingId::from_pass_index(0),
        1.0,
    );

    phase.sort_transparent();

    let ordering: Vec<(f32, u64)> = phase
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
                let state = GraphicsPipelineState {
                    color_target: Some(ColorTargetState {
                        blend: Some(BlendState::Alpha),
                    }),
                    depth_stencil: Some(DepthStencilState {
                        depth_write_enabled: false,
                        ..Default::default()
                    }),
                    ..test_pipeline_state()
                };
                let pipeline =
                    GraphicsPipeline::new(shader, state, VertexProgramId::from_pass_index(0));
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

                let mut phase = RenderPhase::with_capacity(layers.len());
                for (mesh, sort_depth) in &layers {
                    phase.push(
                        &pipeline,
                        RenderGeometry::Mesh(mesh),
                        Some(&material),
                        ObjectBindingId::from_pass_index(0),
                        *sort_depth,
                    );
                }
                phase.sort_transparent();

                let (width, height) = (96, 80);
                let mut renderer = TestRenderHarness::new(width, height, 1);
                renderer
                    .backend
                    .execute_phase(renderer.target.render_target_mut(), &phase);

                let mut config = rasterizer_rust::io::config::Config::default();
                config.render.width = width;
                config.render.height = height;
                config.render.exposure = 1.0;
                config.render.use_aces = false;
                let mut present =
                    PresentBuffer::new(width, height).expect("present dimensions should be valid");
                execute_resolve_tonemap_pass(ResolveTonemapPassDescriptor {
                    label: Some("determinism-present"),
                    source: &renderer.target,
                    destination: &mut present,
                    exposure: config.render.exposure,
                    tonemap: TonemapOperator::None,
                })
                .expect("resolve-tonemap pass should succeed");
                (
                    present.pixels().to_vec(),
                    renderer.framebuffer().depth_values(),
                )
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
    let material = PbrMaterial {
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
    };
    let varying = PbrVarying {
        world_pos: Point3::origin(),
        normal: Vector3::z(),
        uvs: [Vector2::new(0.25, 0.5), Vector2::new(0.75, 0.5)],
        tangent: Vector4::new(1.0, 0.0, 0.0, 1.0),
    };
    let mut frame = PbrFrameBindings::new(
        Matrix4::identity(),
        Matrix4::identity(),
        Point3::new(0.0, 0.0, 2.0),
    );
    frame.ambient_light = Vector3::repeat(1.0);
    let object = PbrObjectBindings::new(Matrix4::identity());
    let context = PbrDrawContext::new(&frame, &object, PbrMaterialBindings::from_pbr(&material));

    let color = match PbrShader.fragment(
        FragmentInput {
            varying,
            front_facing: true,
            uv_densities: [0.0; 2],
        },
        context,
    ) {
        FragmentOutput::Color(color) => color,
        FragmentOutput::Discard => panic!("opaque PBR fragment should produce color"),
    };

    assert_vec3_approx(color.xyz(), Vector3::new(0.0, 0.5, 0.0));
}
#[test]
fn pbr_vertex_preserves_tangent_frame_under_mirrored_non_uniform_scale() {
    let model = Matrix4::new_nonuniform_scaling(&Vector3::new(-2.0, 3.0, 0.5));
    let frame = PbrFrameBindings::new(
        Matrix4::identity(),
        Matrix4::identity(),
        Point3::new(0.0, 0.0, 2.0),
    );
    let object = PbrObjectBindings::new(model);
    let material = PbrMaterial::default();
    let context = PbrDrawContext::new(&frame, &object, PbrMaterialBindings::from_pbr(&material));
    let mut vertex = Vertex::new(Point3::origin(), Vector3::z(), Vector2::zeros());
    vertex.tangent = Vector4::new(1.0, 1.0, 0.0, 1.0);

    let (_, varying) = PbrShader.vertex(&vertex, context);

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
    let material = PbrMaterial {
        albedo: Vector3::new(1.0, 1.0, 1.0),
        roughness: 0.5,
        ..Default::default()
    };
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
        let shadow_map = vec![0.0];
        let mut frame = PbrFrameBindings::new(
            Matrix4::identity(),
            Matrix4::identity(),
            Point3::new(0.0, 0.0, 2.0),
        );
        frame.lights = &lights;
        frame.ambient_light = Vector3::zeros();
        frame.shadow_map = Some(&shadow_map);
        frame.shadow_map_size = 1;
        frame.shadow_light_index = Some(shadow_light_index);
        frame.light_space_matrix = Matrix4::identity();
        frame.shadow_constant_bias = 0.0;
        frame.shadow_slope_bias = 0.0;
        frame.use_pcf = false;
        let object = PbrObjectBindings::new(Matrix4::identity());
        let context =
            PbrDrawContext::new(&frame, &object, PbrMaterialBindings::from_pbr(&material));
        match PbrShader.fragment(
            FragmentInput {
                varying,
                front_facing: true,
                uv_densities: [0.0; 2],
            },
            context,
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
    let material = PbrMaterial {
        albedo: Vector3::new(1.0, 1.0, 1.0),
        roughness: 0.5,
        double_sided: true,
        ..Default::default()
    };
    let lights = [Light::new_directional(
        Vector3::z(),
        Vector3::new(1.0, 1.0, 1.0),
        1.0,
    )];
    let mut frame = PbrFrameBindings::new(
        Matrix4::identity(),
        Matrix4::identity(),
        Point3::new(0.0, 0.0, -2.0),
    );
    frame.lights = &lights;
    frame.ambient_light = Vector3::zeros();
    let object = PbrObjectBindings::new(Matrix4::identity());
    let context = PbrDrawContext::new(&frame, &object, PbrMaterialBindings::from_pbr(&material));
    let shade = |front_facing| match PbrShader.fragment(
        FragmentInput {
            varying,
            front_facing,
            uv_densities: [0.0; 2],
        },
        context,
    ) {
        FragmentOutput::Color(color) => color.xyz(),
        FragmentOutput::Discard => panic!("opaque PBR fragment should produce color"),
    };

    assert_eq!(shade(true), Vector3::zeros());
    assert!(shade(false).norm_squared() > 0.0);
}
