mod core;
mod io;
mod pipeline;
mod scene;

use core::color::{aces_tone_mapping, linear_to_srgb};
use core::math::transform::TransformFactory;
use io::obj_loader::load_obj;
use nalgebra::{Point3, Vector3};
use pipeline::renderer::{ClearOptions, Renderer};
use pipeline::shaders::pbr::PbrShader;
use pipeline::shaders::phong::PhongShader;
use pipeline::shaders::shadow::ShadowShader;
use rayon::prelude::*;
use scene::camera::Camera;
use scene::light::Light;
use scene::material::{Material, PbrMaterial, PhongMaterial};
use scene::mesh::Mesh;
use scene::model::Model;
use scene::scene_object::SceneObject;
use scene::texture::Texture;
use scene::utils::normalize_and_center_model;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::Ordering;

fn main() {
    env_logger::init();

    // --- CONFIGURATION ---
    let width = 1920; // Reduced for video speed
    let height = 1080;
    let total_frames = 120; // 4 seconds at 30fps
    let output_video = "output_shadow_orbit.mp4";
    let temp_dir = "frames";

    // Create temp directory
    if !Path::new(temp_dir).exists() {
        fs::create_dir(temp_dir).expect("Failed to create temp directory");
    }

    let mut renderer = Renderer::new(width, height, 2); // 2x SSAA

    // --- SCENE SETUP ---
    let mut scene_objects: Vec<SceneObject> = Vec::new();

    // 1. Ground (PBR)
    let ground_mesh = Mesh::create_plane(10.0, 0);
    let ground_mat = Material::Pbr(PbrMaterial {
        albedo: Vector3::new(0.9, 0.9, 0.9),
        metallic: 0.0,
        roughness: 0.8,
        ao: 1.0,
        emissive: Vector3::zeros(),
        ..Default::default()
    });
    let ground_transform = TransformFactory::translation(&Vector3::new(0.0, -1.0, 0.0));
    scene_objects.push(SceneObject::new(
        Model::new(vec![ground_mesh], vec![ground_mat]),
        ground_transform,
    ));

    // 2. Spot (PBR - Gold)
    if let Ok(mut model) = load_obj("assets/spot/spot_triangulated.obj") {
        let (_, _) = normalize_and_center_model(&mut model);
        if !model.materials.is_empty() {
            model.materials[0] = Material::Pbr(PbrMaterial {
                albedo: Vector3::new(1.00, 0.76, 0.33),
                metallic: 1.0,
                roughness: 0.3,
                ao: 1.0,
                emissive: Vector3::zeros(),
                ..Default::default()
            });
        }
        let transform = TransformFactory::translation(&Vector3::new(-1.0, 0.0, 0.5))
            * TransformFactory::rotation_y(30.0_f32.to_radians());
        scene_objects.push(SceneObject::new(model, transform));
    }

    // 3. Sphere (Phong - Blue Plastic)
    if let Ok(mut model) = load_obj("assets/simple/sphere.obj") {
        let (_, _) = normalize_and_center_model(&mut model);
        if !model.materials.is_empty() {
            model.materials[0] = Material::Phong(PhongMaterial {
                diffuse_color: Vector3::new(0.1, 0.1, 0.8),
                specular_color: Vector3::new(1.0, 1.0, 1.0),
                ambient_color: Vector3::new(0.05, 0.05, 0.1),
                shininess: 64.0,
                diffuse_texture: None,
            });
        }
        let transform = TransformFactory::translation(&Vector3::new(1.5, 0.5, -1.0))
            * TransformFactory::scaling_nonuniform(&Vector3::new(1.5, 1.5, 1.5));
        scene_objects.push(SceneObject::new(model, transform));
    }

    // --- LIGHTS & SHADOW SETUP ---
    let light_pos = Point3::new(5.0, 10.0, 5.0);
    let light_target = Point3::new(0.0, 0.0, 0.0);
    let light_up = Vector3::new(0.0, 1.0, 0.0);
    let light_dir = (light_target - light_pos).normalize();

    let lights = vec![
        Light::new_directional(light_dir, Vector3::new(1.0, 0.95, 0.8), 3.5),
        Light::new_point(
            Point3::new(-3.0, 2.0, 2.0),
            Vector3::new(0.1, 0.1, 0.3),
            1.0,
        ),
    ];

    // Shadow Map Setup
    let shadow_map_size = 1024;
    let mut shadow_renderer = Renderer::new(shadow_map_size, shadow_map_size, 1);
    let light_view = TransformFactory::view(&light_pos, &light_target, &light_up);
    let ortho_size = 8.0;
    let light_proj =
        TransformFactory::orthographic(-ortho_size, ortho_size, -ortho_size, ortho_size, 0.1, 30.0);
    let light_space_matrix = light_proj * light_view;

    // --- PASS 1: RENDER SHADOW MAP (ONCE) ---
    // Since lights and objects don't move, we only need to render the shadow map once!
    println!("Pass 1: Rendering Shadow Map (Static)...");
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
    let shadow_map_arc = Arc::new(shadow_depth_data);

    // --- PASS 2: RENDER LOOP (Parallelized) ---
    println!(
        "Starting parallel render loop for {} frames...",
        total_frames
    );
    let bg_texture = Texture::load("assets/background.jpg").ok();
    let bg_texture_arc = Arc::new(bg_texture); // Wrap in Arc for sharing
    let gradient_top = Vector3::new(0.1, 0.1, 0.15);
    let gradient_bottom = Vector3::new(0.02, 0.02, 0.02);

    // Use into_par_iter to render frames in parallel
    (0..total_frames).into_par_iter().for_each(|frame| {
        // Each thread needs its own renderer because FrameBuffer is not thread-safe for writing
        let mut local_renderer = Renderer::new(width, height, 2);

        // 1. Update Camera (Orbit)
        let angle = (frame as f32 / total_frames as f32) * std::f32::consts::PI * 2.0;
        let radius = 8.0;
        let cam_x = angle.sin() * radius;
        let cam_z = angle.cos() * radius;

        let camera = Camera::new_perspective(
            Point3::new(cam_x, 4.0, cam_z),
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            45.0_f32.to_radians(),
            width as f32 / height as f32,
            0.1,
            100.0,
        );

        // 2. Clear Buffer
        local_renderer.clear_with_options(ClearOptions {
            color: Vector3::new(0.0, 0.0, 0.0),
            gradient: Some((gradient_top, gradient_bottom)),
            texture: bg_texture_arc.as_ref().as_ref(),
            depth: f32::INFINITY,
        });

        // 3. Draw Objects
        for obj in &scene_objects {
            let is_pbr = if let Some(mat) = obj.model.materials.first() {
                matches!(mat, Material::Pbr(_))
            } else {
                true
            };

            if is_pbr {
                let mut shader = PbrShader::new(
                    obj.transform,
                    camera.view_matrix(),
                    camera.projection_matrix(),
                    camera.position,
                );
                shader.lights = lights.clone();
                shader.ambient_light = Vector3::new(0.1, 0.1, 0.1);
                shader.shadow_map = Some(shadow_map_arc.clone());
                shader.shadow_map_size = shadow_map_size;
                shader.light_space_matrix = light_space_matrix;
                shader.shadow_bias = 0.01;
                local_renderer.draw_model(&obj.model, &shader);
            } else {
                let mut shader = PhongShader::new(
                    obj.transform,
                    camera.view_matrix(),
                    camera.projection_matrix(),
                    camera.position,
                );
                shader.lights = lights.clone();
                shader.ambient_light = Vector3::new(0.1, 0.1, 0.1);
                shader.shadow_map = Some(shadow_map_arc.clone());
                shader.shadow_map_size = shadow_map_size;
                shader.light_space_matrix = light_space_matrix;
                shader.shadow_bias = 0.01;
                local_renderer.draw_model(&obj.model, &shader);
            }
        }

        // 4. Save Frame
        let filename = format!("{}/frame_{:03}.png", temp_dir, frame);
        save_buffer_to_image(&local_renderer.framebuffer, &filename);

        println!("Finished rendering frame {}/{}", frame + 1, total_frames);
    });

    println!("\nRendering complete.");

    // --- VIDEO GENERATION ---
    println!("Generating video using ffmpeg...");
    let status = Command::new("ffmpeg")
        .arg("-y")
        .arg("-framerate")
        .arg("30")
        .arg("-i")
        .arg(format!("{}/frame_%03d.png", temp_dir))
        .arg("-c:v")
        .arg("libx264")
        .arg("-pix_fmt")
        .arg("yuv420p")
        .arg(output_video)
        .status();

    match status {
        Ok(s) if s.success() => {
            println!("Video generated successfully: {}", output_video);
            println!("Cleaning up temporary frames...");
            for frame in 1..total_frames {
                let filename = format!("{}/frame_{:03}.png", temp_dir, frame);
                let _ = fs::remove_file(filename);
            }
        }
        _ => eprintln!("Failed to run ffmpeg."),
    }
}

fn save_buffer_to_image(fb: &core::framebuffer::FrameBuffer, path: &str) {
    let mut img_buf = image::ImageBuffer::new(fb.width as u32, fb.height as u32);
    for (x, y, pixel) in img_buf.enumerate_pixels_mut() {
        if let Some(hdr_color) = fb.get_pixel(x as usize, y as usize) {
            let mapped = aces_tone_mapping(hdr_color);
            let srgb = linear_to_srgb(mapped);
            let r = (srgb.x.clamp(0.0, 1.0) * 255.0) as u8;
            let g = (srgb.y.clamp(0.0, 1.0) * 255.0) as u8;
            let b = (srgb.z.clamp(0.0, 1.0) * 255.0) as u8;
            *pixel = image::Rgb([r, g, b]);
        }
    }
    img_buf.save(Path::new(path)).expect("Failed to save image");
}
