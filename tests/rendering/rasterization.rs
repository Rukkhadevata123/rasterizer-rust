use super::*;

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
    let mut renderer = TestRenderHarness::new(32, 32, 1);
    let pipeline = GraphicsPipeline::new(
        shader,
        test_pipeline_state(),
        VertexProgramId::from_pass_index(0),
    );
    let mut phase = RenderPhase::default();
    phase.push(
        &pipeline,
        RenderGeometry::Mesh(&mesh),
        None,
        ObjectBindingId::from_pass_index(0),
        0.0,
    );

    renderer
        .backend
        .execute_phase(renderer.target.render_target_mut(), &phase);

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
    let mut renderer = TestRenderHarness::new(64, 32, 1);

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
        renderer
            .backend
            .execute_phase(renderer.target.render_target_mut(), &phase);
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
    let mut renderer = TestRenderHarness::new(64, 32, 1);

    renderer
        .backend
        .execute_phase(renderer.target.render_target_mut(), &phase);

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
        None,
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
        None,
        object_binding_id,
        0.0,
    );
    let mut renderer = TestRenderHarness::new(32, 32, 1);

    renderer
        .backend
        .execute_phase(renderer.target.render_target_mut(), &phase);

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
        None,
        object_binding_id,
        0.0,
    );
    phase.push(
        &line_pipeline,
        RenderGeometry::Mesh(&mesh),
        None,
        object_binding_id,
        0.0,
    );
    let mut renderer = TestRenderHarness::new(32, 32, 1);

    renderer
        .backend
        .execute_phase(renderer.target.render_target_mut(), &phase);

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
    let mut renderer = TestRenderHarness::new(64, 32, 1);

    renderer
        .backend
        .execute_phase(renderer.target.render_target_mut(), &phase);

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
        None,
        ObjectBindingId::from_pass_index(0),
        0.0,
    );
    phase.push(
        &right_pipeline,
        RenderGeometry::Mesh(&mesh),
        None,
        ObjectBindingId::from_pass_index(0),
        0.0,
    );
    let mut renderer = TestRenderHarness::new(64, 32, 1);

    renderer
        .backend
        .execute_phase(renderer.target.render_target_mut(), &phase);

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
        None,
        ObjectBindingId::from_pass_index(0),
        0.0,
    );
    let mut backend = SoftwareRasterBackend::new();
    let mut shadow_target = RenderTarget::new(16, 16, 1).expect("shadow target should be valid");
    let mut main_target = RenderTarget::new(48, 32, 1).expect("main target should be valid");

    backend.execute_phase(&mut shadow_target, &phase);
    backend.execute_phase(&mut main_target, &phase);

    assert_vec3_approx(
        shadow_target.framebuffer().get_pixel(8, 8).unwrap(),
        Vector3::new(1.0, 0.5, 0.25),
    );
    assert_vec3_approx(
        main_target.framebuffer().get_pixel(24, 16).unwrap(),
        Vector3::new(1.0, 0.5, 0.25),
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
        None,
        ObjectBindingId::from_pass_index(0),
        0.0,
    );
    let mut visible_phase = RenderPhase::default();
    visible_phase.push(
        &pipeline,
        RenderGeometry::Mesh(&visible),
        None,
        ObjectBindingId::from_pass_index(0),
        0.0,
    );
    let mut renderer = TestRenderHarness::new(32, 32, 1);

    renderer.backend.execute_phases(
        renderer.target.render_target_mut(),
        &[&empty_phase, &visible_phase],
    );

    assert_vec3_approx(
        renderer.framebuffer().sample(16, 16).unwrap().color,
        Vector3::y(),
    );
}

#[test]
fn nearer_triangle_wins_depth_test() {
    let shader = ClipSpaceShader;
    let mut renderer = TestRenderHarness::new(32, 32, 1);

    let far = triangle(0.5, Vector4::new(1.0, 0.0, 0.0, 1.0));
    let near = triangle(-0.5, Vector4::new(0.0, 1.0, 0.0, 1.0));
    draw_mesh(&mut renderer, &far, &shader, None, test_pipeline_state());
    draw_mesh(&mut renderer, &near, &shader, None, test_pipeline_state());

    assert_vec3_approx(
        renderer.framebuffer().get_pixel(16, 16).unwrap(),
        Vector3::new(0.0, 1.0, 0.0),
    );
}

#[test]
fn depth_state_is_explicit_per_draw() {
    let shader = ClipSpaceShader;
    let mut renderer = TestRenderHarness::new(32, 32, 1);
    let red = triangle(-0.5, Vector4::new(1.0, 0.0, 0.0, 1.0));
    let blue = triangle(0.5, Vector4::new(0.0, 0.0, 1.0, 1.0));

    draw_mesh(
        &mut renderer,
        &red,
        &shader,
        None,
        GraphicsPipelineState {
            depth_stencil: Some(DepthStencilState {
                depth_write_enabled: false,
                ..Default::default()
            }),
            ..test_pipeline_state()
        },
    );
    assert!(
        renderer
            .framebuffer()
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
        GraphicsPipelineState {
            depth_stencil: Some(DepthStencilState {
                depth_compare: CompareFunction::Greater,
                ..Default::default()
            }),
            ..test_pipeline_state()
        },
    );
    assert_vec3_approx(
        renderer.framebuffer().get_pixel(16, 16).unwrap(),
        Vector3::new(1.0, 0.0, 0.0),
    );

    draw_mesh(
        &mut renderer,
        &blue,
        &shader,
        None,
        GraphicsPipelineState {
            depth_stencil: Some(DepthStencilState {
                depth_compare: CompareFunction::Always,
                ..Default::default()
            }),
            ..test_pipeline_state()
        },
    );
    assert_vec3_approx(
        renderer.framebuffer().get_pixel(16, 16).unwrap(),
        Vector3::new(0.0, 0.0, 1.0),
    );

    let stored_depth = renderer.framebuffer().sample(16, 16).unwrap().depth;
    draw_mesh(
        &mut renderer,
        &red,
        &shader,
        None,
        GraphicsPipelineState {
            depth_stencil: None,
            ..test_pipeline_state()
        },
    );
    assert_vec3_approx(
        renderer.framebuffer().get_pixel(16, 16).unwrap(),
        Vector3::new(1.0, 0.0, 0.0),
    );
    assert_eq!(
        renderer.framebuffer().sample(16, 16).unwrap().depth,
        stored_depth
    );
}

#[test]
fn depth_only_color_target_runs_fragments_without_storing_color() {
    let shader = ClipSpaceShader;
    let mesh = triangle(0.0, Vector4::new(1.0, 0.0, 0.0, 0.25));
    let masked = Material::Pbr(PbrMaterial {
        alpha_mode: AlphaMode::Mask(0.5),
        ..Default::default()
    });
    let pipeline = GraphicsPipelineState {
        color_target: None,
        ..test_pipeline_state()
    };
    let mut renderer = TestRenderHarness::new(32, 32, 1);

    draw_mesh(&mut renderer, &mesh, &shader, Some(&masked), pipeline);
    let discarded = renderer.framebuffer().sample(16, 16).unwrap();
    assert!(discarded.depth.is_infinite());
    assert_eq!(discarded.color, Vector3::zeros());

    draw_mesh(&mut renderer, &mesh, &shader, None, pipeline);
    let stored = renderer.framebuffer().sample(16, 16).unwrap();
    assert!(stored.depth.is_finite());
    assert_eq!(stored.color, Vector3::zeros());
}

#[test]
fn triangle_crossing_near_plane_is_clipped_and_rendered() {
    let shader = ClipSpaceShader;
    let mut renderer = TestRenderHarness::new(32, 32, 1);

    let mut mesh = triangle(0.0, Vector4::new(1.0, 0.0, 1.0, 1.0));
    mesh.vertices[0].position.z = -2.0;
    draw_mesh(&mut renderer, &mesh, &shader, None, test_pipeline_state());

    let colored_pixels = (0..32)
        .flat_map(|y| (0..32).map(move |x| (x, y)))
        .filter(|&(x, y)| {
            renderer
                .framebuffer()
                .get_pixel(x, y)
                .unwrap()
                .norm_squared()
                > 0.0
        })
        .count();
    assert!(colored_pixels > 0);
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
fn cull_mode_can_reject_one_winding() {
    let shader = ClipSpaceShader;
    let mesh = triangle(0.0, Vector4::new(1.0, 1.0, 1.0, 1.0));

    let render = |mode| {
        let mut renderer = TestRenderHarness::new(32, 32, 1);
        let state = GraphicsPipelineState {
            primitive: PrimitiveState {
                cull_mode: mode,
                ..Default::default()
            },
            ..Default::default()
        };
        draw_mesh(&mut renderer, &mesh, &shader, None, state);
        renderer.framebuffer().get_pixel(16, 16).unwrap()
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
        let mut renderer = TestRenderHarness::new(32, 32, 1);
        draw_mesh(&mut renderer, mesh, &shader, None, test_pipeline_state());
        renderer.framebuffer().get_pixel(16, 16).unwrap()
    };

    assert_vec3_approx(render(&front_mesh), Vector3::new(0.0, 1.0, 0.0));
    assert_vec3_approx(render(&back_mesh), Vector3::new(1.0, 0.0, 0.0));
}

#[test]
fn mirrored_front_face_inverts_culling_and_fragment_facing() {
    let mesh = triangle(0.0, Vector4::zeros());
    let render = |cull_mode, front_face| {
        let mut renderer = TestRenderHarness::new(32, 32, 1);
        let state = GraphicsPipelineState {
            primitive: PrimitiveState {
                cull_mode,
                front_face,
                ..Default::default()
            },
            ..Default::default()
        };
        draw_mesh(&mut renderer, &mesh, &FacingShader, None, state);
        renderer.framebuffer().get_pixel(16, 16).unwrap()
    };

    assert_vec3_approx(
        render(CullMode::None, FrontFace::CounterClockwise),
        Vector3::new(0.0, 1.0, 0.0),
    );
    assert_vec3_approx(
        render(CullMode::None, FrontFace::Clockwise),
        Vector3::new(1.0, 0.0, 0.0),
    );
    assert!(render(CullMode::Back, FrontFace::CounterClockwise).norm_squared() > 0.0);
    assert_eq!(
        render(CullMode::Back, FrontFace::Clockwise),
        Vector3::zeros()
    );
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
    let mut renderer = TestRenderHarness::new(48, 70, 1);

    draw_mesh(&mut renderer, &mesh, &shader, None, test_pipeline_state());

    for y in [0, 15, 16, 31, 32, 47, 48, 63, 64, 69] {
        assert_vec3_approx(
            renderer.framebuffer().get_pixel(24, y).unwrap(),
            color.xyz(),
        );
    }
}

#[test]
fn top_left_rule_covers_shared_edge_once_without_cracks() {
    let shader = AdditiveCoverageShader;
    let color = Vector4::new(1.0, 0.0, 0.0, 0.5);
    let size = 257;
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
    let mut renderer = TestRenderHarness::new(size, size, 1);
    draw_mesh(
        &mut renderer,
        &mesh,
        &shader,
        None,
        GraphicsPipelineState {
            color_target: Some(ColorTargetState {
                blend: Some(BlendState::Alpha),
            }),
            depth_stencil: Some(DepthStencilState {
                depth_compare: CompareFunction::Always,
                depth_write_enabled: false,
            }),
            ..test_pipeline_state()
        },
    );

    for coordinate in 0..size {
        assert_vec3_approx(
            renderer
                .framebuffer()
                .get_pixel(coordinate, coordinate)
                .unwrap(),
            Vector3::new(0.5, 0.0, 0.0),
        );
    }
}

#[test]
fn wireframe_width_is_stable_in_pixel_space() {
    let shader = ClipSpaceShader;
    let color = Vector4::new(1.0, 1.0, 1.0, 1.0);
    let render = |width| {
        let mut renderer = TestRenderHarness::new(width, width, 1);
        draw_mesh(
            &mut renderer,
            &triangle(0.0, color),
            &shader,
            None,
            GraphicsPipelineState {
                primitive: PrimitiveState {
                    polygon_mode: PolygonMode::Line,
                    ..test_pipeline_state().primitive
                },
                ..test_pipeline_state()
            },
        );
        (0..width)
            .flat_map(|y| (0..width).map(move |x| (x, y)))
            .filter(|&(x, y)| {
                renderer
                    .framebuffer()
                    .get_pixel(x, y)
                    .unwrap()
                    .norm_squared()
                    > 0.0
            })
            .count()
    };

    let small = render(32) as f32;
    let large = render(64) as f32;
    assert!(
        large / small < 3.0,
        "wireframe area scaled as {small} -> {large}"
    );
    assert!(
        large / small > 1.5,
        "wireframe area scaled as {small} -> {large}"
    );
}

#[test]
fn non_finite_clip_coordinates_are_rejected() {
    let mesh = triangle(0.0, Vector4::new(1.0, 0.0, 0.0, 1.0));
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let shader = NonFiniteClipShader {
            clip: Vector4::new(invalid, 0.0, 0.0, 1.0),
        };
        let mut renderer = TestRenderHarness::new(32, 32, 1);
        draw_mesh(&mut renderer, &mesh, &shader, None, test_pipeline_state());
        for y in 0..32 {
            for x in 0..32 {
                let sample = renderer.framebuffer().sample(x, y).unwrap();
                assert!(sample.depth.is_infinite());
                assert_eq!(sample.color, Vector3::zeros());
            }
        }
    }
}

#[test]
fn rasterizer_tracks_uv_density_per_texture_coordinate_set() {
    let mut mesh = triangle(0.0, Vector4::zeros());
    mesh.vertices[0].texcoords[1] = Vector2::new(0.0, 0.0);
    mesh.vertices[1].texcoords[1] = Vector2::new(1.0, 0.0);
    mesh.vertices[2].texcoords[1] = Vector2::new(0.0, 1.0);
    let mut renderer = TestRenderHarness::new(32, 32, 1);

    draw_mesh(
        &mut renderer,
        &mesh,
        &DualUvDensityShader,
        None,
        test_pipeline_state(),
    );

    let density = renderer.framebuffer().get_pixel(16, 16).unwrap();
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
        let mut renderer = TestRenderHarness::new(64, 64, 1);
        draw_mesh(&mut renderer, &mesh, &shader, None, test_pipeline_state());

        let sample = renderer.framebuffer().sample(32, 32).unwrap();
        assert_vec3_approx(sample.color, Vector3::new(0.0, 1.0, 0.0));
        assert!((sample.depth - 0.05).abs() < 1e-5);
    }
}
