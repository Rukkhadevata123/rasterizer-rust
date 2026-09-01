use crate::core::framebuffer::FrameBuffer;
use crate::core::pipeline_state::{CullMode, GraphicsPipelineState, PolygonMode, PrimitiveState};
use crate::error::{ApplicationError, WindowError};
use crate::io::config::{Config, CullModeConfig, RenderConfig};
use crate::io::image::save_buffer_to_image;
use crate::pipeline::passes::{
    ResolveTonemapPassDescriptor, TonemapOperator, execute_resolve_tonemap_pass, render_main_pass,
    render_shadow_pass,
};
use crate::pipeline::renderer::{
    FrameResources, MainHdrTarget, PresentBuffer, RenderTarget, SoftwareRasterBackend,
};
use crate::scene::loader::{build_lights_from_config, init_scene_resources, update_scene_objects};
use crate::ui::input::CameraController;
use log::{debug, info, warn};
use minifb::{Key, MouseButton, Window, WindowOptions};
use std::time::Instant;

fn cull_mode_index(mode: CullModeConfig) -> usize {
    match mode {
        CullModeConfig::None => 0,
        CullModeConfig::Front => 1,
        CullModeConfig::Back => 2,
    }
}

fn cull_mode_from_index(index: usize) -> CullMode {
    match index {
        0 => CullMode::None,
        1 => CullMode::Front,
        _ => CullMode::Back,
    }
}

fn primitive_state(cull_mode: CullMode, wireframe: bool) -> PrimitiveState {
    PrimitiveState {
        cull_mode,
        polygon_mode: if wireframe {
            PolygonMode::Line
        } else {
            PolygonMode::Fill
        },
        ..Default::default()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HotReloadPlan {
    target_rebuild: bool,
    resource_reload: bool,
    window_restart: bool,
}

impl HotReloadPlan {
    fn is_live_update(self) -> bool {
        !self.target_rebuild && !self.resource_reload && !self.window_restart
    }
}

fn classify_hot_reload(current: &Config, next: &Config) -> HotReloadPlan {
    let window_restart =
        current.render.width != next.render.width || current.render.height != next.render.height;
    let target_rebuild = current.render.supersample_scale != next.render.supersample_scale
        || current.render.shadow_map_size != next.render.shadow_map_size;
    let objects_changed = current.objects.len() != next.objects.len()
        || current
            .objects
            .iter()
            .zip(&next.objects)
            .any(|(current_object, next_object)| {
                current.resolve_path(&current_object.path) != next.resolve_path(&next_object.path)
                    || current_object.normalization != next_object.normalization
            });
    let ground_geometry_changed =
        current.ground.enabled != next.ground.enabled || current.ground.size != next.ground.size;
    let resource_reload = objects_changed
        || ground_geometry_changed
        || current.render.use_mipmap != next.render.use_mipmap;

    HotReloadPlan {
        target_rebuild,
        resource_reload,
        window_restart,
    }
}

struct HotReloadRenderSettings {
    render: RenderConfig,
    resize_requested: bool,
    supersample_scale_rejected: bool,
    shadow_map_size_rejected: bool,
}

fn apply_hot_reload_render_settings(
    mut render: RenderConfig,
    window_width: usize,
    window_height: usize,
    target: &mut MainHdrTarget,
    shadow_target: &mut RenderTarget,
) -> HotReloadRenderSettings {
    let supersample_scale_rejected =
        FrameBuffer::validate_dimensions(window_width, window_height, render.supersample_scale)
            .is_err();
    if supersample_scale_rejected {
        render.supersample_scale = target.framebuffer().supersample_scale;
    } else if target.framebuffer().supersample_scale != render.supersample_scale {
        if let Ok(new_target) =
            MainHdrTarget::new(window_width, window_height, render.supersample_scale)
        {
            *target = new_target;
        } else {
            render.supersample_scale = target.framebuffer().supersample_scale;
        }
    }

    let shadow_map_size_rejected =
        FrameBuffer::validate_dimensions(render.shadow_map_size, render.shadow_map_size, 1)
            .is_err();
    if shadow_map_size_rejected {
        render.shadow_map_size = shadow_target.framebuffer().width;
    } else if shadow_target.framebuffer().width != render.shadow_map_size
        || shadow_target.framebuffer().height != render.shadow_map_size
    {
        if let Ok(new_target) = RenderTarget::new(render.shadow_map_size, render.shadow_map_size, 1)
        {
            *shadow_target = new_target;
        } else {
            render.shadow_map_size = shadow_target.framebuffer().width;
        }
    }

    let resize_requested = render.width != window_width || render.height != window_height;
    render.width = window_width;
    render.height = window_height;

    HotReloadRenderSettings {
        render,
        resize_requested,
        supersample_scale_rejected,
        shadow_map_size_rejected,
    }
}
/// Runs the application in GUI mode with real-time rendering and interactivity.
pub fn run_gui(mut config: Config, config_path: &str) -> Result<(), ApplicationError> {
    config
        .validate()
        .map_err(|reason| ApplicationError::InvalidConfiguration { reason })?;
    let width = config.render.width;
    let height = config.render.height;

    let mut frame_count = 0;
    let mut last_fps_update = Instant::now();

    info!("Starting GUI mode ({}x{})...", width, height);
    info!(
        "Controls: WASD=Move, Space/LeftShift=Up/Down, LeftClick=Look, Scroll=FOV, R=Reload Config"
    );

    let mut window = Window::new(
        "Rust PBR Rasterizer",
        width,
        height,
        WindowOptions {
            resize: false,
            ..WindowOptions::default()
        },
    )
    .map_err(|source| WindowError::Create { source })?;

    let mut context = init_scene_resources(&config)?;

    let mut backend = SoftwareRasterBackend::new();
    let mut target =
        MainHdrTarget::new(width, height, config.render.supersample_scale).map_err(|reason| {
            ApplicationError::RenderInitialization {
                target: "main framebuffer",
                reason,
            }
        })?;
    let mut shadow_target = RenderTarget::new(
        config.render.shadow_map_size,
        config.render.shadow_map_size,
        1,
    )
    .map_err(|reason| ApplicationError::RenderInitialization {
        target: "shadow framebuffer",
        reason,
    })?;
    let mut frame_resources = FrameResources::new();

    let mut cam_controller = CameraController::new(
        config.camera.speed,
        config.camera.sensitivity,
        config.camera.fov,
        config.camera.zoom_speed,
        &context.camera,
    );

    let mut last_frame_time = Instant::now();
    let mut last_right_click = false;
    let mut last_middle_click = false;
    let mut cull_mode_idx = cull_mode_index(config.render.cull_mode);
    let mut pipeline_state = GraphicsPipelineState {
        primitive: primitive_state(cull_mode_from_index(cull_mode_idx), config.render.wireframe),
        ..Default::default()
    };

    let mut present = PresentBuffer::new(width, height).map_err(|reason| {
        ApplicationError::RenderInitialization {
            target: "present buffer",
            reason,
        }
    })?;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let now = Instant::now();
        let dt = (now - last_frame_time).as_secs_f32();
        last_frame_time = now;

        if window.is_key_pressed(Key::R, minifb::KeyRepeat::No) {
            info!("Reloading configuration...");
            {
                let mut new_config = Config::load(config_path)?;
                let reload_plan = classify_hot_reload(&config, &new_config);

                let render_settings = apply_hot_reload_render_settings(
                    new_config.render.clone(),
                    width,
                    height,
                    &mut target,
                    &mut shadow_target,
                );
                if render_settings.resize_requested {
                    warn!(
                        "Ignoring hot-reloaded render size; restart the GUI to resize the window."
                    );
                }
                if render_settings.supersample_scale_rejected {
                    warn!("Ignoring invalid hot-reloaded supersampling scale.");
                }
                if render_settings.shadow_map_size_rejected {
                    warn!("Ignoring invalid hot-reloaded shadow-map size.");
                }
                new_config.render = render_settings.render;

                if reload_plan.resource_reload {
                    let camera = context.camera.clone();
                    context = init_scene_resources(&new_config)?;
                    context.camera = camera;
                    info!("Reloaded scene model and texture resources.");
                } else {
                    let (new_lights, new_shadow_light) = build_lights_from_config(&new_config);
                    context.lights = new_lights;
                    context.shadow_light = new_shadow_light;
                    update_scene_objects(&mut context.scene_objects, &new_config);
                }

                cam_controller.speed = new_config.camera.speed;
                cam_controller.sensitivity = new_config.camera.sensitivity;
                cam_controller.zoom_speed = new_config.camera.zoom_speed;

                pipeline_state.primitive.polygon_mode = if new_config.render.wireframe {
                    PolygonMode::Line
                } else {
                    PolygonMode::Fill
                };
                cull_mode_idx = cull_mode_index(new_config.render.cull_mode);
                pipeline_state.primitive.cull_mode = cull_mode_from_index(cull_mode_idx);
                config = new_config;

                if reload_plan.is_live_update() {
                    info!("Applied live configuration update.");
                } else {
                    info!("Hot reload successful!");
                }
            }
        }

        cam_controller.update(&window, &mut context.camera, dt);

        let right_click = window.get_mouse_down(MouseButton::Right);
        if right_click && !last_right_click {
            cull_mode_idx = (cull_mode_idx + 1) % 3;
            let new_mode = cull_mode_from_index(cull_mode_idx);
            pipeline_state.primitive.cull_mode = new_mode;
            info!("Cull mode changed to: {:?}", new_mode);
        }
        last_right_click = right_click;

        let middle_click = window.get_mouse_down(MouseButton::Middle);
        if middle_click && !last_middle_click {
            pipeline_state.primitive.polygon_mode = match pipeline_state.primitive.polygon_mode {
                PolygonMode::Fill => PolygonMode::Line,
                PolygonMode::Line => PolygonMode::Fill,
            };
            info!(
                "Wireframe mode: {}",
                pipeline_state.primitive.polygon_mode == PolygonMode::Line
            );
        }
        last_middle_click = middle_click;

        let shadow = render_shadow_pass(
            &config,
            &context,
            &mut backend,
            &mut shadow_target,
            &mut frame_resources,
        );
        render_main_pass(
            &config,
            &context,
            &mut backend,
            &mut target,
            &mut frame_resources,
            &shadow,
            pipeline_state,
        )?;

        execute_resolve_tonemap_pass(ResolveTonemapPassDescriptor {
            label: Some("present"),
            source: &target,
            destination: &mut present,
            exposure: config.render.exposure,
            tonemap: if config.render.use_aces {
                TonemapOperator::Aces
            } else {
                TonemapOperator::None
            },
        })?;
        window
            .update_with_buffer(present.pixels(), present.width(), present.height())
            .map_err(|source| WindowError::Present { source })?;

        window.set_title(&format!(
            "Rust PBR - {:.1} FPS - FOV: {:.1}",
            1.0 / dt,
            cam_controller.fov.to_degrees()
        ));

        frame_count += 1;
        if last_fps_update.elapsed().as_secs_f32() >= 2.0 {
            info!(
                "Average FPS: {:.1}",
                frame_count as f32 / last_fps_update.elapsed().as_secs_f32()
            );
            frame_count = 0;
            last_fps_update = Instant::now();
        }
    }

    Ok(())
}

/// Runs the application in CLI mode (headless) for a single high-quality render.
pub fn run_cli(config: Config) -> Result<(), ApplicationError> {
    config
        .validate()
        .map_err(|reason| ApplicationError::InvalidConfiguration { reason })?;
    info!("Starting CLI mode...");
    let context = init_scene_resources(&config)?;
    let start_time = Instant::now();

    let mut backend = SoftwareRasterBackend::new();
    let mut target = MainHdrTarget::new(
        config.render.width,
        config.render.height,
        config.render.supersample_scale,
    )
    .map_err(|reason| ApplicationError::RenderInitialization {
        target: "main framebuffer",
        reason,
    })?;
    let mut shadow_target = RenderTarget::new(
        config.render.shadow_map_size,
        config.render.shadow_map_size,
        1,
    )
    .map_err(|reason| ApplicationError::RenderInitialization {
        target: "shadow framebuffer",
        reason,
    })?;
    let mut frame_resources = FrameResources::new();

    let pipeline_state = GraphicsPipelineState {
        primitive: primitive_state(
            cull_mode_from_index(cull_mode_index(config.render.cull_mode)),
            config.render.wireframe,
        ),
        ..Default::default()
    };

    let shadow = render_shadow_pass(
        &config,
        &context,
        &mut backend,
        &mut shadow_target,
        &mut frame_resources,
    );
    if shadow.depth.is_some() {
        debug!("Shadow pass completed.");
    }
    render_main_pass(
        &config,
        &context,
        &mut backend,
        &mut target,
        &mut frame_resources,
        &shadow,
        pipeline_state,
    )?;

    info!("Render completed in {:.2?}", start_time.elapsed());

    let output_path = config.resolve_path(&config.render.output);
    info!("Saving output to '{}'...", output_path.display());
    let mut present =
        PresentBuffer::new(config.render.width, config.render.height).map_err(|reason| {
            ApplicationError::RenderInitialization {
                target: "present buffer",
                reason,
            }
        })?;
    execute_resolve_tonemap_pass(ResolveTonemapPassDescriptor {
        label: Some("png-output"),
        source: &target,
        destination: &mut present,
        exposure: config.render.exposure,
        tonemap: if config.render.use_aces {
            TonemapOperator::Aces
        } else {
            TonemapOperator::None
        },
    })?;
    save_buffer_to_image(
        present.pixels(),
        present.width(),
        present.height(),
        &output_path,
    )?;
    info!("Done.");
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hot_reload_classifies_live_target_and_window_changes() {
        let current = Config::default();
        let mut live = current.clone();
        live.render.exposure = 2.0;
        assert!(classify_hot_reload(&current, &live).is_live_update());

        let mut target = current.clone();
        target.render.supersample_scale += 1;
        assert_eq!(
            classify_hot_reload(&current, &target),
            HotReloadPlan {
                target_rebuild: true,
                resource_reload: false,
                window_restart: false,
            }
        );

        let mut window = current.clone();
        window.render.width += 1;
        assert_eq!(
            classify_hot_reload(&current, &window),
            HotReloadPlan {
                target_rebuild: false,
                resource_reload: false,
                window_restart: true,
            }
        );
    }

    #[test]
    fn hot_reload_classifies_all_scene_resource_changes() {
        let current = Config::default();
        let assert_resource_reload = |next: &Config| {
            let plan = classify_hot_reload(&current, next);
            assert!(plan.resource_reload);
            assert!(!plan.is_live_update());
        };

        let mut path = current.clone();
        path.objects[0].path = "replacement.glb".to_string();
        assert_resource_reload(&path);

        let mut base_dir = current.clone();
        base_dir.base_dir = std::path::PathBuf::from("replacement-scene");
        assert_resource_reload(&base_dir);

        let mut count = current.clone();
        count.objects.push(count.objects[0].clone());
        assert_resource_reload(&count);

        let mut normalization = current.clone();
        normalization.objects[0].normalization = crate::io::config::ModelNormalization::Preserve;
        assert_resource_reload(&normalization);

        let mut mip_policy = current.clone();
        mip_policy.render.use_mipmap = !mip_policy.render.use_mipmap;
        assert_resource_reload(&mip_policy);

        let mut ground_enabled = current.clone();
        ground_enabled.ground.enabled = !ground_enabled.ground.enabled;
        assert_resource_reload(&ground_enabled);

        let mut ground_size = current.clone();
        ground_size.ground.size += 1.0;
        assert_resource_reload(&ground_size);
    }

    #[test]
    fn hot_reload_rebuilds_main_and_shadow_targets() {
        let mut target = MainHdrTarget::new(80, 45, 1).expect("test dimensions should be valid");
        let mut shadow_target =
            RenderTarget::new(64, 64, 1).expect("test dimensions should be valid");
        let render = RenderConfig {
            width: 160,
            height: 90,
            supersample_scale: 2,
            shadow_map_size: 128,
            ..Default::default()
        };

        let settings =
            apply_hot_reload_render_settings(render, 80, 45, &mut target, &mut shadow_target);

        assert!(settings.resize_requested);
        assert!(!settings.supersample_scale_rejected);
        assert!(!settings.shadow_map_size_rejected);
        assert_eq!((settings.render.width, settings.render.height), (80, 45));
        assert_eq!(target.framebuffer().supersample_scale, 2);
        assert_eq!(target.framebuffer().buffer_width, 160);
        assert_eq!(target.framebuffer().buffer_height, 90);
        assert_eq!(shadow_target.framebuffer().width, 128);
        assert_eq!(shadow_target.framebuffer().height, 128);
    }

    #[test]
    fn hot_reload_keeps_window_size_when_no_resize_is_requested() {
        let mut target = MainHdrTarget::new(80, 45, 1).expect("test dimensions should be valid");
        let mut shadow_target =
            RenderTarget::new(64, 64, 1).expect("test dimensions should be valid");
        let render = RenderConfig {
            width: 80,
            height: 45,
            supersample_scale: 1,
            shadow_map_size: 64,
            ..Default::default()
        };

        let settings =
            apply_hot_reload_render_settings(render, 80, 45, &mut target, &mut shadow_target);

        assert!(!settings.resize_requested);
        assert!(!settings.supersample_scale_rejected);
        assert!(!settings.shadow_map_size_rejected);
        assert_eq!((settings.render.width, settings.render.height), (80, 45));
        assert_eq!(target.framebuffer().supersample_scale, 1);
        assert_eq!(shadow_target.framebuffer().width, 64);
    }
    #[test]
    fn hot_reload_rejects_zero_sized_render_resources() {
        let mut target = MainHdrTarget::new(80, 45, 2).expect("test dimensions should be valid");
        let mut shadow_target =
            RenderTarget::new(64, 64, 1).expect("test dimensions should be valid");
        let render = RenderConfig {
            width: 80,
            height: 45,
            supersample_scale: 0,
            shadow_map_size: 0,
            ..Default::default()
        };

        let settings =
            apply_hot_reload_render_settings(render, 80, 45, &mut target, &mut shadow_target);

        assert!(settings.supersample_scale_rejected);
        assert!(settings.shadow_map_size_rejected);
        assert_eq!(settings.render.supersample_scale, 2);
        assert_eq!(settings.render.shadow_map_size, 64);
        assert_eq!(target.framebuffer().supersample_scale, 2);
        assert_eq!(shadow_target.framebuffer().width, 64);
    }
}
