mod core;
mod io;
mod pipeline;
mod scene;
mod ui;

use crate::core::rasterizer::CullMode;
use clap::Parser;
use core::color::{aces_tone_mapping, linear_to_srgb};
use core::framebuffer::FrameBuffer;
use core::math::transform::TransformFactory;
use io::config::Config;
use io::obj_loader::load_obj;
use log::{debug, error, info, warn};
use minifb::{Key, MouseButton, Window, WindowOptions};
use nalgebra::{Point3, Vector3};
use pipeline::renderer::{ClearOptions, Renderer};
use pipeline::shaders::pbr::PbrShader;
use pipeline::shaders::shadow::ShadowShader;
use rayon::prelude::*;
use scene::camera::Camera;
use scene::light::Light;
use scene::material::{Material, PbrMaterial};
use scene::mesh::Mesh;
use scene::model::Model;
use scene::scene_object::SceneObject;
use scene::texture::Texture;
use scene::utils::normalize_and_center_model;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;
use ui::input::CameraController;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "scene.toml")]
    config: String,

    /// Start in GUI mode with real-time rendering
    #[arg(long)]
    gui: bool,
}

fn main() {
    env_logger::init();
    let args = Args::parse();

    info!("Loading configuration from: {}", args.config);
    let config = match Config::load(&args.config) {
        Ok(c) => c,
        Err(e) => {
            warn!(
                "Failed to load config '{}': {}. Using defaults.",
                args.config, e
            );
            Config::default()
        }
    };

    if args.gui {
        run_gui(config);
    } else {
        run_cli(config);
    }
}

// ===============================================================================================
// GUI Mode (Real-time)
// ===============================================================================================
fn run_gui(config: Config) {
    let width = config.render.width;
    let height = config.render.height;

    let mut frame_count = 0;
    let mut last_fps_update = Instant::now();

    info!("Starting GUI mode ({}x{})...", width, height);
    info!(
        "Controls: WASD=Move, Space/LeftShift=Up/Down, Z=Freeze, RightClick=Look, Scroll=FOV, LeftClick=Cull Mode, MiddleClick=Wireframe"
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

    // Limit update rate to 240 FPS
    window.set_target_fps(240);

    // 2. Initialize Resources
    let (mut camera, lights, scene_objects, shadow_light_pos) = init_scene_resources(&config);

    // Respect samples from config (user choice)
    let mut renderer = Renderer::new(width, height, config.render.samples);

    let mut shadow_renderer = Renderer::new(
        config.render.shadow_map_size,
        config.render.shadow_map_size,
        1, // Shadow map usually doesn't need MSAA
    );

    // Camera Controller
    let mut cam_controller = CameraController::new(
        config.camera.speed,
        config.camera.sensitivity,
        config.camera.fov,
        config.camera.zoom_speed,
        &camera,
    );

    let mut last_frame_time = Instant::now();

    // Input state tracking for debouncing
    let mut last_left_click = false;
    let mut last_middle_click = false;

    // Track current cull mode index: 0=None, 1=Front, 2=Back
    let mut cull_mode_idx = match config.render.cull_mode.as_str() {
        "none" => 0,
        "front" => 1,
        _ => 2,
    };

    // [Memory Reuse] Allocate display buffer once outside the loop
    let mut buffer = vec![0u32; width * height];

    // 3. Main Loop
    while window.is_open() && !window.is_key_down(Key::Escape) {
        let now = Instant::now();
        let dt = (now - last_frame_time).as_secs_f32();
        last_frame_time = now;

        // --- Input Handling ---
        cam_controller.update(&window, &mut camera, dt);

        // Left Click: Cycle Cull Mode (None -> Front -> Back)
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

        // Middle Click: Toggle Wireframe
        let middle_click = window.get_mouse_down(MouseButton::Middle);
        if middle_click && !last_middle_click {
            renderer.rasterizer.wireframe = !renderer.rasterizer.wireframe;
            info!("Wireframe mode: {}", renderer.rasterizer.wireframe);
        }
        last_middle_click = middle_click;

        // --- Rendering ---

        // Pass 1: Shadow Map
        let mut shadow_map_arc = None;
        let mut light_space_matrix = nalgebra::Matrix4::identity();

        if config.render.use_shadows {
            let light_target = Point3::new(0.0, 0.0, 0.0);
            let light_up = Vector3::new(0.0, 1.0, 0.0);
            let light_view = TransformFactory::view(&shadow_light_pos, &light_target, &light_up);
            let ortho_size = config.render.shadow_ortho_size;
            let light_proj = TransformFactory::orthographic(
                -ortho_size,
                ortho_size,
                -ortho_size,
                ortho_size,
                0.1,
                30.0,
            );
            light_space_matrix = light_proj * light_view;

            shadow_renderer.clear_with_options(ClearOptions {
                depth: f32::INFINITY,
                ..Default::default()
            });

            for obj in &scene_objects {
                let shader = ShadowShader::new(obj.transform, light_view, light_proj);
                shadow_renderer.draw_model(&obj.model, &shader);
            }

            let shadow_depth_data: Vec<f32> = shadow_renderer
                .framebuffer
                .depth_buffer
                .iter()
                .map(|atomic| f32::from_bits(atomic.load(Ordering::Relaxed)))
                .collect();
            shadow_map_arc = Some(Arc::new(shadow_depth_data));
        }

        // Pass 2: Main Render
        let (gradient, color) = if let Some(c) = config.render.background_color {
            (None, Vector3::from(c))
        } else if let (Some(top), Some(bottom)) = (
            config.render.background_gradient_top,
            config.render.background_gradient_bottom,
        ) {
            (
                Some((Vector3::from(top), Vector3::from(bottom))),
                Vector3::zeros(),
            )
        } else {
            (None, Vector3::zeros())
        };

        renderer.clear_with_options(ClearOptions {
            color,
            gradient,
            texture: None,
            depth: f32::INFINITY,
        });

        let ambient_light = Vector3::from(config.render.ambient_light);

        for obj in &scene_objects {
            let mut shader = PbrShader::new(
                obj.transform,
                camera.view_matrix(),
                camera.projection_matrix(),
                camera.position,
            );
            shader.lights = lights.clone();
            shader.ambient_light = ambient_light;
            shader.shadow_map = shadow_map_arc.clone();
            shader.shadow_map_size = config.render.shadow_map_size;
            shader.light_space_matrix = light_space_matrix;
            shader.shadow_bias = config.render.shadow_bias;
            shader.use_pcf = config.render.use_pcf;
            shader.pcf_kernel_size = config.render.pcf_kernel_size;

            renderer.draw_model(&obj.model, &shader);
        }

        // --- Display ---
        // [Multi-threading] Use Rayon to parallelize pixel conversion
        // [Memory Reuse] Directly modify the buffer without re-allocation

        buffer
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(y, row)| {
                for (x, pixel) in row.iter_mut().enumerate() {
                    // get_pixel computation is expensive (SSAA), parallelization provides high benefit
                    if let Some(color) = renderer.framebuffer.get_pixel(x, y) {
                        let exposed = color * config.render.exposure;
                        let mapped = if config.render.use_aces {
                            aces_tone_mapping(exposed)
                        } else {
                            exposed
                        };
                        let srgb = linear_to_srgb(mapped);

                        let r = (srgb.x.clamp(0.0, 1.0) * 255.0) as u32;
                        let g = (srgb.y.clamp(0.0, 1.0) * 255.0) as u32;
                        let b = (srgb.z.clamp(0.0, 1.0) * 255.0) as u32;

                        *pixel = (255 << 24) | (r << 16) | (g << 8) | b;
                    } else {
                        *pixel = 0;
                    }
                }
            });

        // update_with_buffer is now very fast since it only does memory copy
        window.update_with_buffer(&buffer, width, height).unwrap();
        window.set_title(&format!(
            "Rust PBR - {:.1} FPS - FOV: {:.1}",
            1.0 / dt,
            cam_controller.fov.to_degrees()
        ));

        // FPS logging
        frame_count += 1;
        let elapsed = last_fps_update.elapsed();
        if elapsed.as_secs_f32() >= 2.0 {
            let fps = frame_count as f32 / elapsed.as_secs_f32();
            info!("Average FPS: {:.1}", fps);
            frame_count = 0;
            last_fps_update = Instant::now();
        }
    }
}

// ===============================================================================================
// CLI Mode (Single Frame)
// ===============================================================================================
fn run_cli(config: Config) {
    info!("Starting CLI mode...");
    let (camera, lights, scene_objects, shadow_light_pos) = init_scene_resources(&config);

    // --- Render Pipeline ---
    let total_start_time = Instant::now();

    // Pass 1: Shadow Map
    let mut shadow_map_arc = None;
    let mut light_space_matrix = nalgebra::Matrix4::identity();
    let shadow_map_size = config.render.shadow_map_size;

    if config.render.use_shadows {
        info!(
            "Pass 1: Shadow Map Generation ({}x{})...",
            shadow_map_size, shadow_map_size
        );
        let start_time = Instant::now();
        let mut shadow_renderer = Renderer::new(shadow_map_size, shadow_map_size, 1);
        let light_target = Point3::new(0.0, 0.0, 0.0);
        let light_up = Vector3::new(0.0, 1.0, 0.0);
        let light_view = TransformFactory::view(&shadow_light_pos, &light_target, &light_up);
        let ortho_size = config.render.shadow_ortho_size;
        let light_proj = TransformFactory::orthographic(
            -ortho_size,
            ortho_size,
            -ortho_size,
            ortho_size,
            0.1,
            30.0,
        );
        light_space_matrix = light_proj * light_view;

        shadow_renderer.clear_with_options(ClearOptions {
            depth: f32::INFINITY,
            ..Default::default()
        });
        for obj in &scene_objects {
            let shader = ShadowShader::new(obj.transform, light_view, light_proj);
            shadow_renderer.draw_model(&obj.model, &shader);
        }
        let shadow_depth_data: Vec<f32> = shadow_renderer
            .framebuffer
            .depth_buffer
            .iter()
            .map(|atomic| f32::from_bits(atomic.load(Ordering::Relaxed)))
            .collect();
        shadow_map_arc = Some(Arc::new(shadow_depth_data));
        debug!("Shadow pass completed in {:.2?}", start_time.elapsed());
    }

    // Pass 2: Main Render
    info!(
        "Pass 2: Main Render ({}x{}, {} samples)...",
        config.render.width, config.render.height, config.render.samples
    );
    let start_time = Instant::now();
    let mut renderer = Renderer::new(
        config.render.width,
        config.render.height,
        config.render.samples,
    );

    let cull_mode = match config.render.cull_mode.as_str() {
        "front" => CullMode::Front,
        "none" => CullMode::None,
        _ => CullMode::Back,
    };
    renderer.rasterizer.set_cull_mode(cull_mode);
    renderer.rasterizer.wireframe = config.render.wireframe;

    let bg_texture = if let Some(path) = &config.render.background_image {
        Texture::load(path).ok()
    } else {
        None
    };
    let (gradient, color) = if let Some(c) = config.render.background_color {
        (None, Vector3::from(c))
    } else if let (Some(top), Some(bottom)) = (
        config.render.background_gradient_top,
        config.render.background_gradient_bottom,
    ) {
        (
            Some((Vector3::from(top), Vector3::from(bottom))),
            Vector3::zeros(),
        )
    } else {
        (
            Some((Vector3::new(0.1, 0.1, 0.15), Vector3::new(0.02, 0.02, 0.02))),
            Vector3::zeros(),
        )
    };

    renderer.clear_with_options(ClearOptions {
        color,
        gradient: if bg_texture.is_some() { None } else { gradient },
        texture: bg_texture.as_ref(),
        depth: f32::INFINITY,
    });
    let ambient_light = Vector3::from(config.render.ambient_light);

    for obj in &scene_objects {
        let mut shader = PbrShader::new(
            obj.transform,
            camera.view_matrix(),
            camera.projection_matrix(),
            camera.position,
        );
        shader.lights = lights.clone();
        shader.ambient_light = ambient_light;
        shader.shadow_map = shadow_map_arc.clone();
        shader.shadow_map_size = shadow_map_size;
        shader.light_space_matrix = light_space_matrix;
        shader.shadow_bias = config.render.shadow_bias;
        shader.use_pcf = config.render.use_pcf;
        shader.pcf_kernel_size = config.render.pcf_kernel_size;
        renderer.draw_model(&obj.model, &shader);
    }

    info!("Main render pass completed in {:.2?}", start_time.elapsed());
    info!("Total rendering time: {:.2?}", total_start_time.elapsed());
    info!("Saving output to '{}'...", config.render.output);
    save_buffer_to_image(
        &renderer.framebuffer,
        &config.render.output,
        config.render.exposure,
        config.render.use_aces,
    );
    info!("Done.");
}

// Helper: Resource Initialization
fn init_scene_resources(config: &Config) -> (Camera, Vec<Light>, Vec<SceneObject>, Point3<f32>) {
    // 1. Camera
    let cam_pos = Point3::from(config.camera.position);
    let cam_target = Point3::from(config.camera.target);
    let cam_up = Vector3::from(config.camera.up);
    let aspect_ratio = config.render.width as f32 / config.render.height as f32;

    let camera = if config.camera.projection == "orthographic" {
        Camera::new_orthographic(
            cam_pos,
            cam_target,
            cam_up,
            config.camera.ortho_height,
            aspect_ratio,
            config.camera.near,
            config.camera.far,
        )
    } else {
        Camera::new_perspective(
            cam_pos,
            cam_target,
            cam_up,
            config.camera.fov.to_radians(),
            aspect_ratio,
            config.camera.near,
            config.camera.far,
        )
    };

    // 2. Lights
    let mut lights = Vec::new();
    let mut shadow_light_pos = Point3::new(0.0, 10.0, 0.0);
    let mut has_shadow_light = false;
    for l in &config.lights {
        let color = Vector3::from(l.color);
        match l.r#type.as_str() {
            "directional" => {
                if let Some(dir) = l.direction {
                    let dir_vec = Vector3::from(dir).normalize();
                    lights.push(Light::new_directional(dir_vec, color, l.intensity));
                    if !has_shadow_light {
                        shadow_light_pos = Point3::origin() - dir_vec * 10.0;
                        has_shadow_light = true;
                    }
                }
            }
            "point" => {
                if let Some(pos) = l.position {
                    let mut light = Light::new_point(Point3::from(pos), color, l.intensity);
                    if let Light::Point {
                        ref mut attenuation,
                        ..
                    } = light
                        && let Some(a) = l.attenuation
                    {
                        *attenuation = (a[0], a[1], a[2]);
                    }
                    lights.push(light);
                }
            }
            _ => {}
        }
    }
    info!("Scene initialized with {} lights.", lights.len());

    // 3. Objects
    let mut scene_objects: Vec<SceneObject> = Vec::new();
    if config.ground.enabled {
        let ground_mesh = Mesh::create_plane(config.ground.size, 0);
        let ground_mat = Material::Pbr(PbrMaterial {
            albedo: config
                .ground
                .albedo
                .map(Vector3::from)
                .unwrap_or(Vector3::new(0.9, 0.9, 0.9)),
            metallic: config.ground.metallic.unwrap_or(0.0),
            roughness: config.ground.roughness.unwrap_or(0.8),
            ao: 1.0,
            emissive: Vector3::zeros(),
            ..Default::default()
        });
        scene_objects.push(SceneObject::new(
            Model::new(vec![ground_mesh], vec![ground_mat]),
            TransformFactory::translation(&Vector3::new(0.0, -1.0, 0.0)),
        ));
    }

    for obj_conf in &config.objects {
        let mut model = match load_obj(&obj_conf.path) {
            Ok(mut m) => {
                normalize_and_center_model(&mut m);
                m
            }
            Err(e) => {
                error!(
                    "Error loading model '{}': {}. Using fallback mesh.",
                    obj_conf.path, e
                );
                let mesh = Mesh::create_test_triangle(0);
                let mat = PbrMaterial {
                    albedo: Vector3::new(1.0, 0.0, 1.0),
                    ..Default::default()
                };
                Model::new(vec![mesh], vec![Material::Pbr(mat)])
            }
        };
        if model.materials.is_empty() {
            model.materials.push(Material::default());
        }
        let Material::Pbr(ref mut mat) = model.materials[0];
        if let Some(c) = obj_conf.albedo {
            mat.albedo = Vector3::from(c);
        }
        if let Some(m) = obj_conf.metallic {
            mat.metallic = m;
        }
        if let Some(r) = obj_conf.roughness {
            mat.roughness = r;
        }
        if let Some(ao) = obj_conf.ao {
            mat.ao = ao;
        }
        if let Some(e) = obj_conf.emissive {
            mat.emissive = Vector3::from(e) * obj_conf.emissive_intensity;
        }
        if let Some(path) = &obj_conf.albedo_texture {
            if let Ok(tex) = Texture::load(path) {
                mat.albedo_texture = Some(Arc::new(tex));
            } else {
                warn!("Failed to load Albedo texture '{}'", path);
            }
        }
        if let Some(path) = &obj_conf.metallic_roughness_texture {
            if let Ok(tex) = Texture::load(path) {
                mat.metallic_roughness_texture = Some(Arc::new(tex));
            } else {
                warn!("Failed to load Metallic/Roughness texture '{}'", path);
            }
        }
        if let Some(path) = &obj_conf.normal_texture {
            if let Ok(tex) = Texture::load(path) {
                mat.normal_texture = Some(Arc::new(tex));
            } else {
                warn!("Failed to load Normal texture '{}'", path);
            }
        }

        let translation = TransformFactory::translation(&Vector3::from(obj_conf.position));
        let rotation = TransformFactory::rotation_x(obj_conf.rotation[0].to_radians())
            * TransformFactory::rotation_y(obj_conf.rotation[1].to_radians())
            * TransformFactory::rotation_z(obj_conf.rotation[2].to_radians());
        let scale = TransformFactory::scaling_nonuniform(&Vector3::from(obj_conf.scale));
        scene_objects.push(SceneObject::new(model, translation * rotation * scale));
    }
    info!("Scene initialized with {} objects.", scene_objects.len());
    (camera, lights, scene_objects, shadow_light_pos)
}

fn save_buffer_to_image(fb: &FrameBuffer, path: &str, exposure: f32, use_aces: bool) {
    let mut img_buf = image::ImageBuffer::new(fb.width as u32, fb.height as u32);
    for (x, y, pixel) in img_buf.enumerate_pixels_mut() {
        if let Some(hdr_color) = fb.get_pixel(x as usize, y as usize) {
            let exposed_color = hdr_color * exposure;
            let srgb = linear_to_srgb(if use_aces {
                aces_tone_mapping(exposed_color)
            } else {
                Vector3::new(
                    exposed_color.x.clamp(0.0, 1.0),
                    exposed_color.y.clamp(0.0, 1.0),
                    exposed_color.z.clamp(0.0, 1.0),
                )
            });
            let r = (srgb.x.clamp(0.0, 1.0) * 255.0) as u8;
            let g = (srgb.y.clamp(0.0, 1.0) * 255.0) as u8;
            let b = (srgb.z.clamp(0.0, 1.0) * 255.0) as u8;
            *pixel = image::Rgb([r, g, b]);
        }
    }
    if let Err(e) = img_buf.save(Path::new(path)) {
        error!("Failed to save image to '{}': {}", path, e);
    }
}
