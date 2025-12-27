mod core;
mod io;
mod pipeline;
mod scene;

use crate::core::rasterizer::CullMode;
use clap::Parser;
use core::color::{aces_tone_mapping, linear_to_srgb};
use core::math::transform::TransformFactory;
use io::config::Config;
use io::obj_loader::load_obj;
use nalgebra::{Point3, Vector3};
use pipeline::renderer::{ClearOptions, Renderer};
use pipeline::shaders::pbr::PbrShader;
use pipeline::shaders::shadow::ShadowShader;
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

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "scene.toml")]
    config: String,
}

fn main() {
    env_logger::init();
    let args = Args::parse();

    // --- Configuration Loading with Fallback ---
    println!("Loading configuration from: {}", args.config);
    let config = match Config::load(&args.config) {
        Ok(c) => c,
        Err(e) => {
            eprintln!(
                "Warning: Failed to load config '{}': {}. \nUsing built-in default scene.",
                args.config, e
            );
            Config::default()
        }
    };

    // --- 1. Setup Camera ---
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

    // --- 2. Setup Lights ---
    let mut lights = Vec::new();
    let mut shadow_light_dir = Vector3::new(0.0, -1.0, 0.0);
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
                        shadow_light_dir = dir_vec;
                        shadow_light_pos = Point3::origin() - shadow_light_dir * 10.0;
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

    // --- 3. Setup Objects ---
    let mut scene_objects: Vec<SceneObject> = Vec::new();

    // Ground
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
        let ground_transform = TransformFactory::translation(&Vector3::new(0.0, -1.0, 0.0));
        scene_objects.push(SceneObject::new(
            Model::new(vec![ground_mesh], vec![ground_mat]),
            ground_transform,
        ));
    }

    // Objects
    for obj_conf in &config.objects {
        // Try loading model, fallback to test triangle on failure
        let mut model = match load_obj(&obj_conf.path) {
            Ok(mut m) => {
                normalize_and_center_model(&mut m);
                m
            }
            Err(e) => {
                eprintln!(
                    "Error loading model '{}': {}. Using fallback mesh.",
                    obj_conf.path, e
                );
                // Create a magenta test triangle as fallback
                let mesh = Mesh::create_test_triangle(0);
                let mut mat = PbrMaterial::default();
                mat.albedo = Vector3::new(1.0, 0.0, 1.0); // Magenta for error
                Model::new(vec![mesh], vec![Material::Pbr(mat)])
            }
        };

        if model.materials.is_empty() {
            model.materials.push(Material::default());
        }

        // Apply Overrides (Model is guaranteed to be PBR now)
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

        // Graceful texture loading
        if let Some(path) = &obj_conf.albedo_texture {
            if let Ok(tex) = Texture::load(path) {
                mat.albedo_texture = Some(Arc::new(tex));
            } else {
                eprintln!("Warning: Failed to load texture '{}'", path);
            }
        }
        if let Some(path) = &obj_conf.metallic_roughness_texture {
            if let Ok(tex) = Texture::load(path) {
                mat.metallic_roughness_texture = Some(Arc::new(tex));
            } else {
                eprintln!("Warning: Failed to load texture '{}'", path);
            }
        }
        if let Some(path) = &obj_conf.normal_texture {
            if let Ok(tex) = Texture::load(path) {
                mat.normal_texture = Some(Arc::new(tex));
            } else {
                eprintln!("Warning: Failed to load texture '{}'", path);
            }
        }

        let translation = TransformFactory::translation(&Vector3::from(obj_conf.position));
        let rotation = TransformFactory::rotation_x(obj_conf.rotation[0].to_radians())
            * TransformFactory::rotation_y(obj_conf.rotation[1].to_radians())
            * TransformFactory::rotation_z(obj_conf.rotation[2].to_radians());
        let scale = TransformFactory::scaling_nonuniform(&Vector3::from(obj_conf.scale));

        scene_objects.push(SceneObject::new(model, translation * rotation * scale));
    }

    // --- 4. Render Pipeline ---

    // Pass 1: Shadow Map (Conditional)
    let mut shadow_map_arc = None;
    let mut light_space_matrix = nalgebra::Matrix4::identity();
    let shadow_map_size = config.render.shadow_map_size;

    if config.render.use_shadows {
        println!("Pass 1: Shadow Map...");
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
    }

    // Pass 2: Main Render
    println!(
        "Pass 2: Main Render ({}x{})...",
        config.render.width, config.render.height
    );
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

        // 1. Lights & Environment
        shader.lights = lights.clone();
        shader.ambient_light = ambient_light;

        // 2. Shadow Settings
        shader.shadow_map = shadow_map_arc.clone();
        shader.shadow_map_size = shadow_map_size;
        shader.light_space_matrix = light_space_matrix;
        shader.shadow_bias = config.render.shadow_bias;
        shader.use_pcf = config.render.use_pcf;
        shader.pcf_kernel_size = config.render.pcf_kernel_size;

        renderer.draw_model(&obj.model, &shader);
    }

    println!("Saving to {}", config.render.output);
    save_buffer_to_image(
        &renderer.framebuffer,
        &config.render.output,
        config.render.exposure,
    );
    println!("Done.");
}

fn save_buffer_to_image(fb: &core::framebuffer::FrameBuffer, path: &str, exposure: f32) {
    let mut img_buf = image::ImageBuffer::new(fb.width as u32, fb.height as u32);
    for (x, y, pixel) in img_buf.enumerate_pixels_mut() {
        if let Some(hdr_color) = fb.get_pixel(x as usize, y as usize) {
            let exposed_color = hdr_color * exposure;
            let mapped = aces_tone_mapping(exposed_color);
            let srgb = linear_to_srgb(mapped);
            let r = (srgb.x.clamp(0.0, 1.0) * 255.0) as u8;
            let g = (srgb.y.clamp(0.0, 1.0) * 255.0) as u8;
            let b = (srgb.z.clamp(0.0, 1.0) * 255.0) as u8;
            *pixel = image::Rgb([r, g, b]);
        }
    }
    if let Err(e) = img_buf.save(Path::new(path)) {
        eprintln!("Error: Failed to save image to '{}': {}", path, e);
    }
}
