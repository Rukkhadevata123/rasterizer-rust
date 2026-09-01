use super::*;

fn command_descriptor(target: &mut RenderTarget) -> RenderPassDescriptor<'_> {
    RenderPassDescriptor {
        label: Some("command-test"),
        target,
        color_ops: Some(Operations {
            load: LoadOp::Clear(Vector3::zeros()),
        }),
        depth_ops: Some(Operations {
            load: LoadOp::Clear(f32::INFINITY),
        }),
    }
}

fn command_pipeline(shader: ClipSpaceShader) -> GraphicsPipeline<ClipSpaceShader> {
    GraphicsPipeline::new(
        shader,
        test_pipeline_state(),
        VertexProgramId::from_pass_index(0),
    )
}

#[test]
fn command_encoder_rejects_finish_without_a_pass() {
    let device = RenderDevice::new();
    let encoder: CommandEncoder<'_, ClipSpaceShader, Option<&Material>> =
        device.create_command_encoder("empty");

    assert!(matches!(
        encoder.finish(),
        Err(CommandError::MissingPass { encoder }) if encoder == "empty"
    ));
}

#[test]
fn render_pass_requires_pipeline_before_draw() {
    let device = RenderDevice::new();
    let mut target = RenderTarget::new(16, 16, 1).expect("target should be valid");
    let mesh = triangle(0.0, Vector4::new(1.0, 0.0, 0.0, 1.0));
    let mut encoder: CommandEncoder<'_, ClipSpaceShader, Option<&Material>> =
        device.create_command_encoder("missing-pipeline");
    let mut pass = encoder
        .begin_render_pass(command_descriptor(&mut target), None)
        .expect("descriptor should be valid");
    pass.set_draw_bindings(None, ObjectBindingId::from_pass_index(0));

    assert_eq!(
        pass.draw_mesh(&mesh, 0.0).unwrap_err(),
        CommandError::MissingPipeline {
            pass: "command-test".to_owned(),
        }
    );
}

#[test]
fn render_pass_requires_bindings_before_draw() {
    let device = RenderDevice::new();
    let mut target = RenderTarget::new(16, 16, 1).expect("target should be valid");
    let mesh = triangle(0.0, Vector4::new(1.0, 0.0, 0.0, 1.0));
    let pipeline = command_pipeline(ClipSpaceShader);
    let mut encoder: CommandEncoder<'_, ClipSpaceShader, Option<&Material>> =
        device.create_command_encoder("missing-bindings");
    let mut pass = encoder
        .begin_render_pass(command_descriptor(&mut target), None)
        .expect("descriptor should be valid");
    pass.set_pipeline(&pipeline);

    assert_eq!(
        pass.draw_mesh(&mesh, 0.0).unwrap_err(),
        CommandError::MissingBindings {
            pass: "command-test".to_owned(),
        }
    );
}

#[test]
fn dropped_render_pass_causes_finish_to_report_pass_not_ended() {
    let device = RenderDevice::new();
    let mut target = RenderTarget::new(16, 16, 1).expect("target should be valid");
    let mut encoder: CommandEncoder<'_, ClipSpaceShader, Option<&Material>> =
        device.create_command_encoder("unfinished");
    drop(
        encoder
            .begin_render_pass(command_descriptor(&mut target), None)
            .expect("descriptor should be valid"),
    );

    assert!(matches!(
        encoder.finish(),
        Err(CommandError::PassNotEnded { encoder }) if encoder == "unfinished"
    ));
}

#[test]
fn command_encoder_rejects_a_second_render_pass() {
    let device = RenderDevice::new();
    let mut first_target = RenderTarget::new(16, 16, 1).expect("target should be valid");
    let mut second_target = RenderTarget::new(16, 16, 1).expect("target should be valid");
    let mut encoder: CommandEncoder<'_, ClipSpaceShader, Option<&Material>> =
        device.create_command_encoder("single-pass");
    encoder
        .begin_render_pass(command_descriptor(&mut first_target), None)
        .expect("first descriptor should be valid")
        .end()
        .expect("first pass should end");

    assert!(matches!(
        encoder.begin_render_pass(command_descriptor(&mut second_target), None),
        Err(CommandError::PassAlreadyRecorded { encoder }) if encoder == "single-pass"
    ));
}

#[test]
fn invalid_render_pass_descriptor_returns_a_labeled_command_error() {
    let device = RenderDevice::new();
    let mut target = RenderTarget::new(16, 16, 1).expect("target should be valid");
    let mut encoder: CommandEncoder<'_, ClipSpaceShader, Option<&Material>> =
        device.create_command_encoder("invalid-pass");

    assert!(matches!(
        encoder.begin_render_pass(
            RenderPassDescriptor {
                label: Some("empty-command-pass"),
                target: &mut target,
                color_ops: None,
                depth_ops: None,
            },
            None,
        ),
        Err(CommandError::InvalidPass { pass, reason })
            if pass == "empty-command-pass" && reason.contains("does not declare")
    ));
}
#[test]
fn recording_does_not_execute_attachment_or_draw_work() {
    let device = RenderDevice::new();
    let mut queue = device.create_queue();
    let mut target = RenderTarget::new(16, 16, 1).expect("target should be valid");
    let initial_mesh = triangle(0.0, Vector4::new(1.0, 0.0, 0.0, 1.0));
    let initial_pipeline = command_pipeline(ClipSpaceShader);
    submit_test_mesh(
        &mut queue,
        &mut target,
        &initial_pipeline,
        &initial_mesh,
        None,
        ObjectBindingId::from_pass_index(0),
    );
    assert_vec3_approx(target.framebuffer().get_pixel(8, 8).unwrap(), Vector3::x());

    let recorded_mesh = triangle(0.0, Vector4::new(0.0, 1.0, 0.0, 1.0));
    let recorded_pipeline = command_pipeline(ClipSpaceShader);
    let mut encoder: CommandEncoder<'_, ClipSpaceShader, Option<&Material>> =
        device.create_command_encoder("record-only");
    {
        let mut pass = encoder
            .begin_render_pass(command_descriptor(&mut target), None)
            .expect("descriptor should be valid");
        pass.set_pipeline(&recorded_pipeline);
        pass.set_draw_bindings(None, ObjectBindingId::from_pass_index(0));
        pass.draw_mesh(&recorded_mesh, 0.0)
            .expect("draw should record");
        pass.end().expect("pass should end");
    }
    let command_buffer = encoder.finish().expect("command buffer should finish");
    drop(command_buffer);

    assert_vec3_approx(target.framebuffer().get_pixel(8, 8).unwrap(), Vector3::x());
}
#[test]
fn queue_submission_is_synchronous_and_preserves_ordered_phase_boundaries() {
    let device = RenderDevice::new();
    let mut target = RenderTarget::new(32, 32, 1).expect("target should be valid");
    let first = triangle(0.75, Vector4::new(1.0, 0.0, 0.0, 0.5));
    let second = triangle(0.5, Vector4::new(0.0, 1.0, 0.0, 0.5));
    let transparent_near = triangle(-0.5, Vector4::new(0.0, 0.0, 1.0, 0.5));
    let transparent_far = triangle(0.25, Vector4::new(1.0, 1.0, 0.0, 0.5));
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
        GraphicsPipeline::new(ClipSpaceShader, state, VertexProgramId::from_pass_index(0));
    let mut encoder = device.create_command_encoder("ordered");
    {
        let mut pass = encoder
            .begin_render_pass(command_descriptor(&mut target), None)
            .expect("descriptor should be valid");
        pass.set_pipeline(&pipeline);
        pass.set_draw_bindings(Some(&material), ObjectBindingId::from_pass_index(0));
        pass.draw_mesh(&first, 1.0)
            .expect("first draw should record");
        pass.draw_mesh(&second, -1.0)
            .expect("second draw should record");
        pass.finish_phase("recorded-order");
        pass.draw_mesh(&transparent_near, 0.5)
            .expect("near transparent draw should record");
        pass.draw_mesh(&transparent_far, -0.5)
            .expect("far transparent draw should record");
        pass.sort_transparent();
        pass.finish_phase("transparent");
        pass.end().expect("pass should end");
    }
    let command_buffer = encoder.finish().expect("command buffer should finish");
    assert_eq!(command_buffer.label(), "ordered");

    let mut queue = device.create_queue();
    let report = queue
        .submit(command_buffer)
        .expect("submission should succeed");

    assert_vec3_approx(
        target.framebuffer().get_pixel(16, 16).unwrap(),
        Vector3::new(0.3125, 0.375, 0.5),
    );
    assert_eq!(report.phases.len(), 2);
    assert_eq!(report.phases[0].label, "recorded-order");
    assert_eq!(report.phases[1].label, "transparent");
    assert!(report.submission_total >= report.backend_preparation);
    assert!(report.submission_total >= report.attachment_processing);
    assert!(report.submission_total >= report.rasterization);
    assert!(report.submission_total >= report.phases[0].execution_total);
    assert!(report.submission_total >= report.phases[1].execution_total);
}
