use nalgebra::{Point3, Vector3};
use rasterizer_rust::core::math::transform::TransformFactory;
use rasterizer_rust::io::obj_loader::load_obj;
use rasterizer_rust::pipeline::renderer::{ClearOptions, Renderer};
use rasterizer_rust::pipeline::shaders::phong::PhongShader;
use rasterizer_rust::scene::camera::Camera;
use rasterizer_rust::scene::light::Light;
use rasterizer_rust::scene::material::Material;
use rasterizer_rust::scene::mesh::Mesh;
use rasterizer_rust::scene::texture::Texture;
use rasterizer_rust::scene::utils::normalize_and_center_model;
use std::path::Path;

fn main() {
    env_logger::init();

    // 1. Setup Renderer
    let width = 800;
    let height = 600;
    let mut renderer = Renderer::new(width, height, 2); // 2x SSAA

    // 2. Load Scene Data
    let obj_path = "assets/spot_triangulated.obj";
    let mut model = match load_obj(obj_path) {
        Ok(m) => {
            println!("Successfully loaded model: {}", obj_path);
            m
        }
        Err(e) => {
            println!("Failed to load model '{}': {}", obj_path, e);
            let mesh = Mesh::create_test_triangle();
            rasterizer_rust::scene::model::Model::new(vec![mesh], vec![Material::default()])
        }
    };

    // 3. Normalize Model
    let (center, scale) = normalize_and_center_model(&mut model);
    println!(
        "Model normalized. Center: {:?}, Scale: {:.4}",
        center, scale
    );

    // 4. Setup Camera (Using the Camera struct!)
    let mut camera = Camera::new_perspective(
        Point3::new(0.0, 0.5, 3.0),
        Point3::new(0.0, 0.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
        45.0_f32.to_radians(),
        width as f32 / height as f32,
        0.1,
        100.0,
    );

    // 或者，如果你想测试正交投影 (物体不会随距离变小)：
    /*
    let mut camera = Camera::new_orthographic(
        Point3::new(0.0, 0.5, 3.0),
        Point3::new(0.0, 0.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
        2.0, // 视野高度为 2.0 个单位
        width as f32 / height as f32,
        0.1,
        100.0,
    );
    */

    // 5. Setup Lights (Multi-light setup!)
    let lights = vec![
        // Light 1: Main Directional Light (Warm Sun from top-right)
        Light::new_directional(
            Vector3::new(1.0, 1.0, 1.0),  // Direction
            Vector3::new(1.0, 0.95, 0.8), // Warm White
            0.8,                          // Intensity
        ),
        // Light 2: Point Light (Red Fill from left)
        Light::new_point(
            Point3::new(-2.0, 1.0, 1.0),
            Vector3::new(1.0, 0.2, 0.2), // Red
            2.0,                         // Intensity (Point lights decay, so need higher intensity)
        ),
        // Light 3: Point Light (Blue Rim from back-right)
        Light::new_point(
            Point3::new(2.0, 0.5, -1.0),
            Vector3::new(0.2, 0.2, 1.0), // Blue
            2.0,
        ),
    ];

    // 6. Setup Shader
    let model_matrix = TransformFactory::rotation_y(30.0_f32.to_radians()); // Rotate cow slightly

    let mut shader = PhongShader::new(
        model_matrix,
        camera.view_matrix(),       // Get from Camera
        camera.projection_matrix(), // Get from Camera
        camera.position,            // Get from Camera
    );

    // Pass our lights to the shader
    shader.lights = lights;
    shader.ambient_light = Vector3::new(0.05, 0.05, 0.05); // Dim ambient

    // Option A: Load a background image (if you have one)
    let bg_texture = Texture::load("assets/background.jpg").ok(); // Returns Option<Texture>
    if bg_texture.is_some() {
        println!("Loaded background image.");
    }

    // Option B: Define a nice gradient (Sky Blue -> Ground Gray)
    let gradient_top = Vector3::new(0.5, 0.7, 1.0); // Sky Blue
    let gradient_bottom = Vector3::new(0.2, 0.2, 0.2); // Dark Gray

    // 7. Render
    println!("Starting render...");

    // Use the new clear method
    renderer.clear_with_options(ClearOptions {
        color: Vector3::new(0.1, 0.1, 0.1),              // Fallback
        gradient: Some((gradient_top, gradient_bottom)), // Use Gradient
        texture: bg_texture.as_ref(), // Use Texture if loaded (overrides gradient)
        depth: f32::INFINITY,
    });

    renderer.draw_model(&model, &shader);

    // 8. Save Result
    let output_path = "output_gradient.png";
    save_buffer_to_image(&renderer.framebuffer, output_path);
    println!("Render saved to {}", output_path);
}

fn save_buffer_to_image(fb: &rasterizer_rust::core::framebuffer::FrameBuffer, path: &str) {
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
