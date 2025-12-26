use nalgebra::{Point3, Vector3};
use rasterizer_rust::core::math::transform::TransformFactory;
use rasterizer_rust::io::obj_loader::load_obj;
use rasterizer_rust::pipeline::renderer::Renderer;
use rasterizer_rust::pipeline::shaders::phong::PhongShader;
use rasterizer_rust::scene::material::Material;
use rasterizer_rust::scene::mesh::Mesh;
use rasterizer_rust::scene::utils::normalize_and_center_model;
use std::path::Path;

fn main() {
    env_logger::init();

    // 1. Setup Renderer
    let width = 800;
    let height = 600;
    // Enable 2x SSAA for better quality
    let mut renderer = Renderer::new(width, height, 2);

    // 2. Load Scene Data
    // Ensure this path points to your model file
    let obj_path = "assets/spot_triangulated.obj";
    let mut model = match load_obj(obj_path) {
        Ok(m) => {
            println!("Successfully loaded model: {}", obj_path);
            m
        }
        Err(e) => {
            println!("Failed to load model '{}': {}", obj_path, e);
            println!("Falling back to built-in test triangle.");

            let mesh = Mesh::create_test_triangle();
            rasterizer_rust::scene::model::Model::new(vec![mesh], vec![Material::default()])
        }
    };

    // 3. Normalize Model
    let (center, scale) = normalize_and_center_model(&mut model);
    println!(
        "Model normalized. Original Center: {:?}, Scale: {:.4}",
        center, scale
    );

    // 4. Setup Camera
    let eye = Point3::new(0.0, 0.0, 3.0);
    let target = Point3::new(0.0, 0.0, 0.0);
    let up = Vector3::new(0.0, 1.0, 0.0);

    let model_matrix = TransformFactory::rotation_y(45.0_f32.to_radians());
    let view_matrix = TransformFactory::view(&eye, &target, &up);
    let projection_matrix = TransformFactory::perspective(
        width as f32 / height as f32,
        45.0_f32.to_radians(),
        0.1,
        100.0,
    );

    // 5. Setup Shader
    let mut shader = PhongShader::new(model_matrix, view_matrix, projection_matrix, eye);

    // Setup some nice lighting
    shader.light_dir = Vector3::new(1.0, 1.0, 1.0).normalize();
    shader.ambient_intensity = Vector3::new(0.1, 0.1, 0.1);
    // Note: shader.diffuse_color is no longer set here, it comes from the Material!

    // 6. Render
    println!("Starting render...");
    renderer.clear(Vector3::new(0.1, 0.1, 0.1));

    // Use the new draw_model method
    renderer.draw_model(&model, &shader);

    // 7. Save Result with Gamma Correction
    let output_path = "output_phong.png";
    save_buffer_to_image(&renderer.framebuffer, output_path);
    println!("Render saved to {}", output_path);
}

fn save_buffer_to_image(fb: &rasterizer_rust::core::framebuffer::FrameBuffer, path: &str) {
    let mut img_buf = image::ImageBuffer::new(fb.width as u32, fb.height as u32);

    for (x, y, pixel) in img_buf.enumerate_pixels_mut() {
        if let Some(linear_color) = fb.get_pixel(x as usize, y as usize) {
            // --- Gamma Correction (Linear -> sRGB) ---
            // Standard Gamma is 2.2. sRGB is roughly pow(color, 1.0/2.2)
            let gamma = 1.0 / 2.2;

            let r = (linear_color.x.powf(gamma).clamp(0.0, 1.0) * 255.0) as u8;
            let g = (linear_color.y.powf(gamma).clamp(0.0, 1.0) * 255.0) as u8;
            let b = (linear_color.z.powf(gamma).clamp(0.0, 1.0) * 255.0) as u8;

            *pixel = image::Rgb([r, g, b]);
        }
    }

    img_buf.save(Path::new(path)).expect("Failed to save image");
}
