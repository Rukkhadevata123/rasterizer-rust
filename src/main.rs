mod core;
mod io;
mod pipeline;
mod scene;

use core::math::transform::TransformFactory;
use io::obj_loader::load_obj;
use nalgebra::{Point3, Vector3};
use pipeline::renderer::{ClearOptions, Renderer};
use pipeline::shaders::phong::PhongShader;
use scene::camera::Camera;
use scene::light::Light;
use scene::material::{Material, PhongMaterial};
use scene::mesh::Mesh;
use scene::model::Model;
use scene::scene_object::SceneObject;
use scene::texture::Texture;
use scene::utils::normalize_and_center_model;
use std::path::Path;

fn main() {
    env_logger::init();

    // 1. Setup Renderer
    let width = 1280;
    let height = 720;
    let mut renderer = Renderer::new(width, height, 2);

    // --- SCENE SETUP ---
    let mut scene_objects: Vec<SceneObject> = Vec::new();

    // --- 1. The Ground Platform ---
    // 缩小尺寸到 8.0，这样能看到边缘，像一个舞台
    let ground_mesh = Mesh::create_plane(8.0, 0);
    let ground_material = Material::Phong(PhongMaterial {
        diffuse_color: Vector3::new(0.4, 0.4, 0.45), // 中灰色，带一点点蓝
        specular_color: Vector3::new(0.3, 0.3, 0.3), // 较强的高光，模拟光滑地板
        shininess: 64.0,                             // 比较光滑
        ambient_color: Vector3::new(0.1, 0.1, 0.1),
        diffuse_texture: None,
    });
    let ground_transform = TransformFactory::translation(&Vector3::new(0.0, -1.0, 0.0));
    scene_objects.push(SceneObject::new(
        Model::new(vec![ground_mesh], vec![ground_material]),
        ground_transform,
    ));

    // List of models to load: (path, position, scale, rotation_y)
    let models_to_load = vec![
        (
            "assets/spot/spot_triangulated.obj",
            Vector3::new(0.0, 0.0, 0.0),
            1.0,
            30.0 as f32,
        ),
        (
            "assets/Crate/Crate1.obj",
            Vector3::new(-2.5, -0.25, 0.0),
            0.75,
            -15.0,
        ),
        (
            "assets/bunny/bunny.obj",
            Vector3::new(2.5, -0.2, 0.0),
            0.8,
            -45.0,
        ),
        (
            "assets/rock/rock.obj",
            Vector3::new(-1.0, -0.7, 1.5),
            0.3,
            90.0,
        ),
        (
            "assets/simple/sphere.obj",
            Vector3::new(1.5, 0.5, -2.0),
            1.5,
            0.0,
        ),
    ];

    for (path, pos, scale_factor, rotation_y) in models_to_load {
        match load_obj(path) {
            Ok(mut model) => {
                let (_, _) = normalize_and_center_model(&mut model);
                let transform = TransformFactory::translation(&pos)
                    * TransformFactory::rotation_y(rotation_y.to_radians())
                    * TransformFactory::scaling_nonuniform(&Vector3::new(
                        scale_factor,
                        scale_factor,
                        scale_factor,
                    ));

                scene_objects.push(SceneObject::new(model, transform));
                println!("Added model: {}", path);
            }
            Err(e) => println!("Failed to load {}: {}", path, e),
        }
    }

    // --- CAMERA & LIGHTS ---
    let mut camera = Camera::new_perspective(
        Point3::new(0.0, 3.0, 7.0), // 相机稍微抬高拉远一点，俯视看清地面边界
        Point3::new(0.0, 0.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
        45.0_f32.to_radians(),
        width as f32 / height as f32,
        0.1,
        100.0,
    );

    let lights = vec![
        Light::new_directional(
            Vector3::new(1.0, 1.5, 1.0),
            Vector3::new(1.0, 0.98, 0.9),
            1.0,
        ),
        Light::new_point(
            Point3::new(-4.0, 3.0, 4.0),
            Vector3::new(0.6, 0.7, 0.8),
            0.8,
        ),
        Light::new_point(
            Point3::new(3.0, 2.0, -3.0),
            Vector3::new(1.0, 0.8, 0.6),
            1.2,
        ),
    ];

    // --- RENDER ---
    println!("Starting render of {} objects...", scene_objects.len());

    let bg_texture = Texture::load("assets/background.jpg").ok();

    // 背景改为较亮的渐变，与深色地面形成对比
    let gradient_top = Vector3::new(0.2, 0.3, 0.5); // 天空蓝
    let gradient_bottom = Vector3::new(0.6, 0.6, 0.65); // 地平线灰白

    renderer.clear_with_options(ClearOptions {
        color: Vector3::new(0.0, 0.0, 0.0),
        gradient: Some((gradient_top, gradient_bottom)),
        texture: bg_texture.as_ref(),
        depth: f32::INFINITY,
    });

    for (i, obj) in scene_objects.iter().enumerate() {
        println!("Drawing object {}...", i);
        let mut shader = PhongShader::new(
            obj.transform,
            camera.view_matrix(),
            camera.projection_matrix(),
            camera.position,
        );
        shader.lights = lights.clone();
        shader.ambient_light = Vector3::new(0.05, 0.05, 0.05);

        renderer.draw_model(&obj.model, &shader);
    }

    let output_path = "output_multi_obj.png";
    save_buffer_to_image(&renderer.framebuffer, output_path);
    println!("Render saved to {}", output_path);
}

fn save_buffer_to_image(fb: &core::framebuffer::FrameBuffer, path: &str) {
    let mut img_buf = image::ImageBuffer::new(fb.width as u32, fb.height as u32);
    for (x, y, pixel) in img_buf.enumerate_pixels_mut() {
        if let Some(linear_color) = fb.get_pixel(x as usize, y as usize) {
            let gamma = 1.0 / 2.2;
            let r = (linear_color.x.powf(gamma).clamp(0.0, 1.0) * 255.0) as u8;
            let g = (linear_color.y.powf(gamma).clamp(0.0, 1.0) * 255.0) as u8;
            let b = (linear_color.z.powf(gamma).clamp(0.0, 1.0) * 255.0) as u8;
            *pixel = image::Rgb([r, g, b]);
        }
    }
    img_buf.save(Path::new(path)).expect("Failed to save image");
}
