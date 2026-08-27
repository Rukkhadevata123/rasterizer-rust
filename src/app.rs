use crate::core::framebuffer::FrameBuffer;
use crate::core::rasterizer::CullMode;
use crate::io::config::{Config, CullModeConfig, RenderConfig};
use crate::io::image::save_buffer_to_image;
use crate::pipeline::passes::{post_process_to_buffer, render_main_pass, render_shadow_pass};
use crate::pipeline::renderer::Renderer;
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
    renderer: &mut Renderer,
    shadow_renderer: &mut Renderer,
) -> HotReloadRenderSettings {
    let supersample_scale_rejected =
        FrameBuffer::validate_dimensions(window_width, window_height, render.supersample_scale)
            .is_err();
    if supersample_scale_rejected {
        render.supersample_scale = renderer.framebuffer.supersample_scale;
    } else if renderer.framebuffer.supersample_scale != render.supersample_scale {
        *renderer = Renderer::new(window_width, window_height, render.supersample_scale)
            .expect("hot-reloaded framebuffer dimensions were checked");
    }

    let shadow_map_size_rejected =
        FrameBuffer::validate_dimensions(render.shadow_map_size, render.shadow_map_size, 1)
            .is_err();
    if shadow_map_size_rejected {
        render.shadow_map_size = shadow_renderer.framebuffer.width;
    } else if shadow_renderer.framebuffer.width != render.shadow_map_size
        || shadow_renderer.framebuffer.height != render.shadow_map_size
    {
        *shadow_renderer = Renderer::new(render.shadow_map_size, render.shadow_map_size, 1)
            .expect("hot-reloaded shadow-map dimensions were checked");
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
pub fn run_gui(mut config: Config, config_path: &str) {
    config
        .validate()
        .expect("configuration must be valid before running the GUI");
    let width = config.render.width;
    let height = config.render.height;

    let mut frame_count = 0;
    let mut last_fps_update = Instant::now();

    info!("Starting GUI mode ({}x{})...", width, height);
    info!(
        "Controls: WASD=Move, Space/LeftShift=Up/Down, LeftClick=Look, Scroll=FOV, R=Reload Config"
    );

    // 1. Initialize Window
    let mut window = Window::new(
        "Rust PBR Rasterizer",
        width,
        height,
        WindowOptions {
            resize: false,
            ..WindowOptions::default()
        },
    )
    .unwrap_or_else(|e| panic!("{}", e));

    // window.set_target_fps(60);

    // 2. Initialize Resources
    let mut context = init_scene_resources(&config);

    // Renderers
    let mut renderer = Renderer::new(width, height, config.render.supersample_scale)
        .expect("configuration must be validated before running the GUI");
    let mut shadow_renderer = Renderer::new(
        config.render.shadow_map_size,
        config.render.shadow_map_size,
        1,
    )
    .expect("configuration must be validated before running the GUI");

    // Camera Controller
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
    renderer
        .rasterizer
        .set_cull_mode(cull_mode_from_index(cull_mode_idx));
    // Also apply initial wireframe setting
    renderer.rasterizer.wireframe = config.render.wireframe;

    let mut buffer = vec![0u32; width * height];

    // 3. Main Loop
    while window.is_open() && !window.is_key_down(Key::Escape) {
        let now = Instant::now();
        let dt = (now - last_frame_time).as_secs_f32();
        last_frame_time = now;

        // --- Hot Reloading ---
        if window.is_key_pressed(Key::R, minifb::KeyRepeat::No) {
            info!("Reloading configuration...");
            match Config::load(config_path) {
                Ok(new_config) => {
                    let (new_lights, new_shadow_light) = build_lights_from_config(&new_config);
                    context.lights = new_lights;
                    context.shadow_light = new_shadow_light;
                    update_scene_objects(&mut context.scene_objects, &new_config);

                    cam_controller.speed = new_config.camera.speed;
                    cam_controller.sensitivity = new_config.camera.sensitivity;
                    cam_controller.zoom_speed = new_config.camera.zoom_speed;

                    let render_settings = apply_hot_reload_render_settings(
                        new_config.render,
                        width,
                        height,
                        &mut renderer,
                        &mut shadow_renderer,
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
                    config.render = render_settings.render;

                    renderer.rasterizer.wireframe = config.render.wireframe;
                    cull_mode_idx = cull_mode_index(config.render.cull_mode);
                    renderer
                        .rasterizer
                        .set_cull_mode(cull_mode_from_index(cull_mode_idx));

                    info!("Hot reload successful!");
                }
                Err(e) => warn!("Failed to reload config: {}", e),
            }
        }

        // --- Input ---
        cam_controller.update(&window, &mut context.camera, dt);

        let right_click = window.get_mouse_down(MouseButton::Right);
        if right_click && !last_right_click {
            cull_mode_idx = (cull_mode_idx + 1) % 3;
            let new_mode = cull_mode_from_index(cull_mode_idx);
            renderer.rasterizer.set_cull_mode(new_mode);
            info!("Cull mode changed to: {:?}", new_mode);
        }
        last_right_click = right_click;

        let middle_click = window.get_mouse_down(MouseButton::Middle);
        if middle_click && !last_middle_click {
            renderer.rasterizer.wireframe = !renderer.rasterizer.wireframe;
            info!("Wireframe mode: {}", renderer.rasterizer.wireframe);
        }
        last_middle_click = middle_click;

        // --- Render ---
        let shadow = render_shadow_pass(&config, &context, &mut shadow_renderer);
        render_main_pass(&config, &context, &mut renderer, &shadow);

        // --- Display ---
        post_process_to_buffer(&renderer.framebuffer, &mut buffer, &config);
        window.update_with_buffer(&buffer, width, height).unwrap();

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
}

/// Runs the application in CLI mode (headless) for a single high-quality render.
pub fn run_cli(config: Config) {
    config
        .validate()
        .expect("configuration must be valid before running the CLI");
    info!("Starting CLI mode...");
    let context = init_scene_resources(&config);
    let start_time = Instant::now();

    let mut renderer = Renderer::new(
        config.render.width,
        config.render.height,
        config.render.supersample_scale,
    )
    .expect("configuration must be validated before running the CLI");
    let mut shadow_renderer = Renderer::new(
        config.render.shadow_map_size,
        config.render.shadow_map_size,
        1,
    )
    .expect("configuration must be validated before running the CLI");

    let cull_mode = cull_mode_from_index(cull_mode_index(config.render.cull_mode));
    renderer.rasterizer.set_cull_mode(cull_mode);
    renderer.rasterizer.wireframe = config.render.wireframe;

    // Render
    let shadow = render_shadow_pass(&config, &context, &mut shadow_renderer);
    if shadow.depth.is_some() {
        debug!("Shadow pass completed.");
    }
    render_main_pass(&config, &context, &mut renderer, &shadow);

    info!("Render completed in {:.2?}", start_time.elapsed());

    // Save
    info!("Saving output to '{}'...", config.render.output);
    let mut buffer = vec![0u32; config.render.width * config.render.height];
    post_process_to_buffer(&renderer.framebuffer, &mut buffer, &config);
    save_buffer_to_image(
        &buffer,
        config.render.width,
        config.render.height,
        &config.render.output,
    );
    info!("Done.");
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hot_reload_rebuilds_sample_and_shadow_buffers() {
        let mut renderer = Renderer::new(80, 45, 1).expect("test dimensions should be valid");
        let mut shadow_renderer =
            Renderer::new(64, 64, 1).expect("test dimensions should be valid");
        let render = RenderConfig {
            width: 160,
            height: 90,
            supersample_scale: 2,
            shadow_map_size: 128,
            ..Default::default()
        };

        let settings =
            apply_hot_reload_render_settings(render, 80, 45, &mut renderer, &mut shadow_renderer);

        assert!(settings.resize_requested);
        assert!(!settings.supersample_scale_rejected);
        assert!(!settings.shadow_map_size_rejected);
        assert_eq!((settings.render.width, settings.render.height), (80, 45));
        assert_eq!(renderer.framebuffer.supersample_scale, 2);
        assert_eq!(renderer.framebuffer.buffer_width, 160);
        assert_eq!(renderer.framebuffer.buffer_height, 90);
        assert_eq!(shadow_renderer.framebuffer.width, 128);
        assert_eq!(shadow_renderer.framebuffer.height, 128);
    }

    #[test]
    fn hot_reload_keeps_window_size_when_no_resize_is_requested() {
        let mut renderer = Renderer::new(80, 45, 1).expect("test dimensions should be valid");
        let mut shadow_renderer =
            Renderer::new(64, 64, 1).expect("test dimensions should be valid");
        let render = RenderConfig {
            width: 80,
            height: 45,
            supersample_scale: 1,
            shadow_map_size: 64,
            ..Default::default()
        };

        let settings =
            apply_hot_reload_render_settings(render, 80, 45, &mut renderer, &mut shadow_renderer);

        assert!(!settings.resize_requested);
        assert!(!settings.supersample_scale_rejected);
        assert!(!settings.shadow_map_size_rejected);
        assert_eq!((settings.render.width, settings.render.height), (80, 45));
        assert_eq!(renderer.framebuffer.supersample_scale, 1);
        assert_eq!(shadow_renderer.framebuffer.width, 64);
    }
    #[test]
    fn hot_reload_rejects_zero_sized_render_resources() {
        let mut renderer = Renderer::new(80, 45, 2).expect("test dimensions should be valid");
        let mut shadow_renderer =
            Renderer::new(64, 64, 1).expect("test dimensions should be valid");
        let render = RenderConfig {
            width: 80,
            height: 45,
            supersample_scale: 0,
            shadow_map_size: 0,
            ..Default::default()
        };

        let settings =
            apply_hot_reload_render_settings(render, 80, 45, &mut renderer, &mut shadow_renderer);

        assert!(settings.supersample_scale_rejected);
        assert!(settings.shadow_map_size_rejected);
        assert_eq!(settings.render.supersample_scale, 2);
        assert_eq!(settings.render.shadow_map_size, 64);
        assert_eq!(renderer.framebuffer.supersample_scale, 2);
        assert_eq!(shadow_renderer.framebuffer.width, 64);
    }
}
