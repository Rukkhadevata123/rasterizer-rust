use super::*;

#[test]
fn masked_shadow_fragments_respect_material_alpha() {
    let mut renderer = TestRenderHarness::new(32, 32, 1);
    let mesh = triangle(0.0, Vector4::zeros());
    let discarded = Material::Pbr(PbrMaterial {
        alpha: 0.25,
        alpha_mode: AlphaMode::Mask(0.5),
        ..Default::default()
    });

    draw_shadow_mesh(&mut renderer, &mesh, Some(&discarded), Matrix4::identity());
    assert!(
        renderer
            .readback()
            .sample_depth(16, 16)
            .unwrap()
            .is_infinite()
    );

    let visible = Material::Pbr(PbrMaterial {
        alpha: 0.75,
        alpha_mode: AlphaMode::Mask(0.5),
        ..Default::default()
    });
    draw_shadow_mesh(&mut renderer, &mesh, Some(&visible), Matrix4::identity());
    assert!((renderer.readback().sample_depth(16, 16).unwrap() - 0.5).abs() < 1e-5);
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
    let mut renderer = TestRenderHarness::new(32, 32, 1);
    let mut mesh = triangle(0.0, Vector4::zeros());
    for vertex in &mut mesh.vertices {
        vertex.texcoords[0] = Vector2::new(0.25, 0.5);
        vertex.texcoords[1] = Vector2::new(0.75, 0.5);
    }

    draw_shadow_mesh(&mut renderer, &mesh, Some(&material), Matrix4::identity());

    assert!(
        renderer
            .readback()
            .sample_depth(16, 16)
            .unwrap()
            .is_infinite()
    );
}

#[test]
fn shadow_draw_context_applies_distinct_object_bindings() {
    let mesh = repeated_index_triangle(0.25);
    let frame = ShadowFrameBindings::new(Matrix4::identity(), Matrix4::identity());
    let object_bindings = [
        ShadowObjectBindings::new(Matrix4::new_translation(&Vector3::new(-0.5, 0.0, 0.0))),
        ShadowObjectBindings::new(Matrix4::new_translation(&Vector3::new(0.5, 0.0, 0.0))),
    ];
    let pipeline = GraphicsPipeline::new(
        ShadowShader,
        GraphicsPipelineState {
            color_target: None,
            ..test_pipeline_state()
        },
        VertexProgramId::new(),
    );
    let draws = [
        (
            &mesh,
            ShadowDrawContext::new(
                &frame,
                &object_bindings[0],
                ShadowMaterialBindings::new(None),
            ),
            0.0,
        ),
        (
            &mesh,
            ShadowDrawContext::new(
                &frame,
                &object_bindings[1],
                ShadowMaterialBindings::new(None),
            ),
            0.0,
        ),
    ];
    let mut renderer = TestRenderHarness::new(32, 32, 1);

    submit_test_draws(
        &mut renderer.queue,
        renderer.target.render_target_mut(),
        &pipeline,
        &draws,
        false,
    );

    assert!(renderer.readback().sample_depth(8, 16).unwrap().is_finite());
    assert!(
        renderer
            .readback()
            .sample_depth(24, 16)
            .unwrap()
            .is_finite()
    );
}
#[test]
fn blended_materials_do_not_write_shadow_depth() {
    let material = Material::Pbr(PbrMaterial {
        alpha_mode: AlphaMode::Blend,
        ..Default::default()
    });
    let context = RenderScene {
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
    let mut renderer = TestRenderHarness::new(32, 32, 1);

    let shadow = render_shadow_pass(
        &config,
        &context,
        &mut renderer.queue,
        renderer.target.render_target_mut(),
        &mut renderer.resources,
    );

    assert!(
        shadow
            .depth()
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
        let context = RenderScene {
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
        let mut renderer = TestRenderHarness::new(32, 32, 1);

        render_shadow_pass(
            &config,
            &context,
            &mut renderer.queue,
            renderer.target.render_target_mut(),
            &mut renderer.resources,
        )
        .depth_snapshot()
        .unwrap()
    };

    assert!(render(false).iter().all(|depth| depth.is_infinite()));
    assert!(render(true).iter().any(|depth| depth.is_finite()));
}

#[test]
fn point_only_scene_disables_shadow_pass() {
    let context = RenderScene {
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
    let mut renderer = TestRenderHarness::new(32, 32, 1);

    let shadow = render_shadow_pass(
        &config,
        &context,
        &mut renderer.queue,
        renderer.target.render_target_mut(),
        &mut renderer.resources,
    );

    assert!(!shadow.is_enabled());
    assert!(shadow.depth().is_none());
    assert!(shadow.size().is_none());
    assert!(shadow.light_space_matrix().is_none());
    assert!(shadow.light_index().is_none());
}

#[test]
fn shadow_output_reports_actual_buffer_size() {
    let context = RenderScene {
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
    let mut renderer = TestRenderHarness::new(16, 16, 1);

    let shadow = render_shadow_pass(
        &config,
        &context,
        &mut renderer.queue,
        renderer.target.render_target_mut(),
        &mut renderer.resources,
    );

    assert!(shadow.is_enabled());
    assert_eq!(shadow.size(), Some(16));
    assert_eq!(shadow.depth().unwrap().len(), 16 * 16);
    assert!(shadow.light_space_matrix().is_some());
    assert_eq!(shadow.light_index(), Some(0));
}

#[test]
fn directional_shadow_bounds_follow_the_camera_frustum() {
    let render = |camera_x| {
        let context = RenderScene {
            camera: Camera::new_perspective(
                Point3::new(camera_x, 0.0, 3.0),
                Point3::new(camera_x, 0.0, 0.0),
                Vector3::y(),
                45.0_f32.to_radians(),
                1.0,
                0.1,
                100.0,
            ),
            lights: vec![Light::new_directional(
                Vector3::new(0.0, 0.0, -1.0),
                Vector3::repeat(1.0),
                1.0,
            )],
            scene_objects: Vec::new(),
            shadow_light: Some(ShadowLight {
                light_index: 0,
                position: Point3::new(0.0, 0.0, 10.0),
            }),
        };
        let mut config = rasterizer_rust::io::config::Config::default();
        config.render.shadow_map_size = 16;
        config.render.shadow_ortho_size = 10.0;
        let mut renderer = TestRenderHarness::new(16, 16, 1);
        render_shadow_pass(
            &config,
            &context,
            &mut renderer.queue,
            renderer.target.render_target_mut(),
            &mut renderer.resources,
        )
        .light_space_matrix()
        .expect("enabled shadow output has a light-space matrix")
    };

    let origin = render(0.0);
    let translated = render(20.0);
    assert!((origin - translated).abs().max() > 0.1);
    assert!(translated.iter().all(|component| component.is_finite()));
}

#[test]
fn directional_shadow_bounds_include_scene_geometry() {
    let render = |scene_objects| {
        let context = RenderScene {
            camera: shadow_test_camera(),
            lights: vec![Light::new_directional(
                Vector3::new(0.0, 0.0, -1.0),
                Vector3::repeat(1.0),
                1.0,
            )],
            scene_objects,
            shadow_light: Some(ShadowLight {
                light_index: 0,
                position: Point3::new(0.0, 0.0, 10.0),
            }),
        };
        let mut config = rasterizer_rust::io::config::Config::default();
        config.render.shadow_ortho_size = 2.0;
        let mut renderer = TestRenderHarness::new(16, 16, 1);
        render_shadow_pass(
            &config,
            &context,
            &mut renderer.queue,
            renderer.target.render_target_mut(),
            &mut renderer.resources,
        )
        .light_space_matrix()
        .expect("enabled shadow output has a light-space matrix")
    };
    let far_caster = SceneObject::new(
        SceneObjectKind::Model { config_index: 0 },
        Model::new(
            vec![triangle(0.0, Vector4::zeros())],
            vec![Material::Pbr(PbrMaterial::default())],
        ),
        Matrix4::new_translation(&Vector3::new(20.0, 0.0, 0.0)),
    );

    let camera_only = render(Vec::new());
    let with_caster = render(vec![far_caster]);

    assert!(with_caster[(0, 0)] < camera_only[(0, 0)] * 0.5);
    assert!(with_caster.iter().all(|component| component.is_finite()));
}
