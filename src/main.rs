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
use pipeline::shaders::phong::PhongShader; // Import PhongShader
use scene::camera::Camera;
use scene::light::Light;
use scene::material::{Material, PbrMaterial, PhongMaterial}; // Import PhongMaterial
use scene::mesh::Mesh;
use scene::model::Model;
use scene::scene_object::SceneObject;
use scene::texture::Texture;
use scene::utils::normalize_and_center_model;
use std::path::Path;

fn main() {
    env_logger::init();

    let width = 1280;
    let height = 720;
    let mut renderer = Renderer::new(width, height, 2);

    // --- SCENE SETUP ---
    let mut scene_objects: Vec<SceneObject> = Vec::new();

    // 1. Ground (PBR - Rough Concrete)
    let ground_mesh = Mesh::create_plane(8.0, 0);
    let ground_mat = Material::Pbr(PbrMaterial {
        albedo: Vector3::new(0.2, 0.2, 0.25),
        metallic: 0.0,
        roughness: 0.9,
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
                albedo: Vector3::new(1.00, 0.76, 0.33), // Gold
                metallic: 1.0,
                roughness: 0.15,
                ao: 1.0,
                emissive: Vector3::zeros(),
                ..Default::default()
            });
        }
        let transform = TransformFactory::translation(&Vector3::new(-1.0, 0.0, 0.5))
            * TransformFactory::rotation_y(30.0_f32.to_radians());
        scene_objects.push(SceneObject::new(model, transform));
    }

    // 3. Sphere (Phong - Shiny Blue Plastic)
    if let Ok(mut model) = load_obj("assets/simple/sphere.obj") {
        let (_, _) = normalize_and_center_model(&mut model);
        if !model.materials.is_empty() {
            // Explicitly use Phong Material here
            model.materials[0] = Material::Phong(PhongMaterial {
                diffuse_color: Vector3::new(0.1, 0.1, 0.8),  // Blue
                specular_color: Vector3::new(1.0, 1.0, 1.0), // White highlights
                ambient_color: Vector3::new(0.05, 0.05, 0.1),
                shininess: 64.0,
                diffuse_texture: None,
            });
        }
        let transform = TransformFactory::translation(&Vector3::new(1.5, 0.5, -1.0))
            * TransformFactory::scaling_nonuniform(&Vector3::new(1.5, 1.5, 1.5));
        scene_objects.push(SceneObject::new(model, transform));
    }

    // --- Camera ---
    let mut camera = Camera::new_perspective(
        Point3::new(0.0, 3.0, 6.0),
        Point3::new(0.0, 0.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
        45.0_f32.to_radians(),
        width as f32 / height as f32,
        0.1,
        100.0,
    );

    // --- Lights (More Colorful) ---
    let lights = vec![
        // Main Warm Sun
        Light::new_directional(
            Vector3::new(1.0, 1.5, 1.0),
            Vector3::new(1.0, 0.9, 0.8),
            2.0,
        ),
        // Cyan Fill Light (Left)
        Light::new_point(
            Point3::new(-4.0, 2.0, 2.0),
            Vector3::new(0.2, 0.8, 1.0),
            3.0,
        ),
        // Magenta Rim Light (Right Back)
        Light::new_point(
            Point3::new(4.0, 1.0, -2.0),
            Vector3::new(1.0, 0.2, 0.8),
            3.0,
        ),
    ];

    // --- Render ---
    let bg_texture = Texture::load("assets/background.jpg").ok();
    let gradient_top = Vector3::new(0.05, 0.05, 0.1);
    let gradient_bottom = Vector3::new(0.01, 0.01, 0.01);

    renderer.clear_with_options(ClearOptions {
        color: Vector3::new(0.0, 0.0, 0.0),
        gradient: Some((gradient_top, gradient_bottom)),
        texture: bg_texture.as_ref(),
        depth: f32::INFINITY,
    });

    println!("Rendering mixed PBR/Phong scene...");

    for (i, obj) in scene_objects.iter().enumerate() {
        // Check material type to decide which shader to use
        // Note: A model can have multiple materials, we check the first one for simplicity here.
        // In a robust engine, you'd group by material or have the mesh tell you.
        let is_pbr = if let Some(mat) = obj.model.materials.first() {
            matches!(mat, Material::Pbr(_))
        } else {
            true // Default to PBR
        };

        if is_pbr {
            println!("Drawing object {} with PBR Shader...", i);
            let mut shader = PbrShader::new(
                obj.transform,
                camera.view_matrix(),
                camera.projection_matrix(),
                camera.position,
            );
            shader.lights = lights.clone();
            renderer.draw_model(&obj.model, &shader);
        } else {
            println!("Drawing object {} with Phong Shader...", i);
            let mut shader = PhongShader::new(
                obj.transform,
                camera.view_matrix(),
                camera.projection_matrix(),
                camera.position,
            );
            shader.lights = lights.clone();
            // Phong shader might need manual ambient adjustment if not using global ambient
            shader.ambient_light = Vector3::new(0.05, 0.05, 0.05);
            renderer.draw_model(&obj.model, &shader);
        }
    }

    let output_path = "output_mixed.png";
    save_buffer_to_image(&renderer.framebuffer, output_path);
    println!("Render saved to {}", output_path);
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
