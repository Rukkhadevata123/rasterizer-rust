use crate::core::rasterizer::CullMode;
use crate::io::config::Config;
use crate::io::image::save_buffer_to_image;
use crate::pipeline::passes::{post_process_to_buffer, render_main_pass, render_shadow_pass};
use crate::pipeline::renderer::Renderer;
use crate::scene::loader::{build_lights_from_config, init_scene_resources, update_scene_objects};
use crate::ui::input::CameraController;
use log::{debug, info, warn};
use minifb::{Key, MouseButton, Window, WindowOptions};
use std::time::Instant;

/// Runs the application in GUI mode with real-time rendering and interactivity.
pub fn run_gui(mut config: Config, config_path: &str) {
    let width = config.render.width;
    let height = config.render.height;

    let mut frame_count = 0;
    let mut last_fps_update = Instant::now();

    info!("Starting GUI mode ({}x{})...", width, height);
    info!(
        "Controls: WASD=Move, Space/LeftShift=Up/Down, RightClick=Look, Scroll=FOV, R=Reload Config"
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

    window.set_target_fps(60);

    // 2. Initialize Resources
    let mut context = init_scene_resources(&config);

    // Renderers
    let mut renderer = Renderer::new(width, height, config.render.samples);
    let mut shadow_renderer = Renderer::new(
        config.render.shadow_map_size,
        config.render.shadow_map_size,
        1,
    );

    // Camera Controller
    let mut cam_controller = CameraController::new(
        config.camera.speed,
        config.camera.sensitivity,
        config.camera.fov,
        config.camera.zoom_speed,
        &context.camera,
    );

    let mut last_frame_time = Instant::now();
    let mut last_left_click = false;
    let mut last_middle_click = false;
    let mut cull_mode_idx = match config.render.cull_mode.as_str() {
        "none" => 0,
        "front" => 1,
        _ => 2,
    };

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
                    let (new_lights, new_shadow_pos) = build_lights_from_config(&new_config);
                    context.lights = new_lights;
                    context.shadow_light_pos = new_shadow_pos;
                    update_scene_objects(&mut context.scene_objects, &new_config);

                    cam_controller.speed = new_config.camera.speed;
                    cam_controller.sensitivity = new_config.camera.sensitivity;
                    cam_controller.zoom_speed = new_config.camera.zoom_speed;

                    config.render = new_config.render;
                    info!("Hot reload successful!");
                }
                Err(e) => warn!("Failed to reload config: {}", e),
            }
        }

        // --- Input ---
        cam_controller.update(&window, &mut context.camera, dt);

        let left_click = window.get_mouse_down(MouseButton::Left);
        if left_click && !last_left_click {
            cull_mode_idx = (cull_mode_idx + 1) % 3;
            let new_mode = match cull_mode_idx {
                0 => CullMode::None,
                1 => CullMode::Front,
                _ => CullMode::Back,
            };
            renderer.rasterizer.set_cull_mode(new_mode);
            info!("Cull mode changed to: {:?}", new_mode);
        }
        last_left_click = left_click;

        let middle_click = window.get_mouse_down(MouseButton::Middle);
        if middle_click && !last_middle_click {
            renderer.rasterizer.wireframe = !renderer.rasterizer.wireframe;
            info!("Wireframe mode: {}", renderer.rasterizer.wireframe);
        }
        last_middle_click = middle_click;

        // --- Render ---
        let (shadow_map, light_matrix) =
            render_shadow_pass(&config, &context, &mut shadow_renderer);
        render_main_pass(&config, &context, &mut renderer, shadow_map, light_matrix);

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
    info!("Starting CLI mode...");
    let context = init_scene_resources(&config);
    let start_time = Instant::now();

    let mut renderer = Renderer::new(
        config.render.width,
        config.render.height,
        config.render.samples,
    );
    let mut shadow_renderer = Renderer::new(
        config.render.shadow_map_size,
        config.render.shadow_map_size,
        1,
    );

    let cull_mode = match config.render.cull_mode.as_str() {
        "front" => CullMode::Front,
        "none" => CullMode::None,
        _ => CullMode::Back,
    };
    renderer.rasterizer.set_cull_mode(cull_mode);
    renderer.rasterizer.wireframe = config.render.wireframe;

    // Render
    let (shadow_map, light_matrix) = render_shadow_pass(&config, &context, &mut shadow_renderer);
    if shadow_map.is_some() {
        debug!("Shadow pass completed.");
    }
    render_main_pass(&config, &context, &mut renderer, shadow_map, light_matrix);

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
