use super::*;

#[test]
fn nearer_triangle_wins_depth_test() {
    let shader = ClipSpaceShader;
    let mut renderer = TestRenderHarness::new(32, 32, 1);

    let far = triangle(0.5, Vector4::new(1.0, 0.0, 0.0, 1.0));
    let near = triangle(-0.5, Vector4::new(0.0, 1.0, 0.0, 1.0));
    draw_mesh(&mut renderer, &far, &shader, None, test_pipeline_state());
    draw_mesh(&mut renderer, &near, &shader, None, test_pipeline_state());

    assert_vec3_approx(
        renderer.readback().color(16, 16).unwrap(),
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
            .readback()
            .sample_depth(16, 16)
            .unwrap()
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
        renderer.readback().color(16, 16).unwrap(),
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
        renderer.readback().color(16, 16).unwrap(),
        Vector3::new(0.0, 0.0, 1.0),
    );

    let stored_depth = renderer.readback().sample_depth(16, 16).unwrap();
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
        renderer.readback().color(16, 16).unwrap(),
        Vector3::new(1.0, 0.0, 0.0),
    );
    assert_eq!(
        renderer.readback().sample_depth(16, 16).unwrap(),
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
    let discarded = renderer.readback();
    assert!(discarded.sample_depth(16, 16).unwrap().is_infinite());
    assert_eq!(discarded.sample_color(16, 16).unwrap(), Vector3::zeros());

    draw_mesh(&mut renderer, &mesh, &shader, None, pipeline);
    let stored = renderer.readback();
    assert!(stored.sample_depth(16, 16).unwrap().is_finite());
    assert_eq!(stored.sample_color(16, 16).unwrap(), Vector3::zeros());
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
        renderer.readback().color(16, 16).unwrap()
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
        renderer.readback().color(16, 16).unwrap()
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
        renderer.readback().color(16, 16).unwrap()
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
            .filter(|&(x, y)| renderer.readback().color(x, y).unwrap().norm_squared() > 0.0)
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
                let readback = renderer.readback();
                assert!(readback.sample_depth(x, y).unwrap().is_infinite());
                assert_eq!(readback.sample_color(x, y).unwrap(), Vector3::zeros());
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

    let density = renderer.readback().color(16, 16).unwrap();
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

        let readback = renderer.readback();
        assert_vec3_approx(
            readback.sample_color(32, 32).unwrap(),
            Vector3::new(0.0, 1.0, 0.0),
        );
        assert!((readback.sample_depth(32, 32).unwrap() - 0.05).abs() < 1e-5);
    }
}
