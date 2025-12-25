use nalgebra::{Point3, Vector3};
use rasterizer_rust::core::math::transform::TransformFactory;
use rasterizer_rust::io::obj_loader::load_obj;
use rasterizer_rust::pipeline::renderer::Renderer;
use rasterizer_rust::pipeline::shaders::phong::PhongShader;
use rasterizer_rust::scene::mesh::Mesh;
use std::path::Path;

fn main() {
    // 1. Initialize Logger
    env_logger::init();

    // 2. Setup Renderer
    let width = 800;
    let height = 600;
    let mut renderer = Renderer::new(width, height);

    // 3. Load Scene Data
    // Try to load a model, fallback to a triangle if not found.
    // You can place a file named 'model.obj' in an 'assets' folder to test.
    let obj_path = "assets/model.obj";
    let mesh = match load_obj(obj_path) {
        Ok(m) => {
            println!("Successfully loaded model: {}", obj_path);
            m
        }
        Err(e) => {
            println!("Failed to load model '{}': {}", obj_path, e);
            println!("Falling back to built-in test triangle.");
            Mesh::create_test_triangle()
        }
    };

    // 4. Setup Camera & Matrices
    // Move camera up and back to see the object clearly
    let eye = Point3::new(0.0, 1.5, 4.0);
    let target = Point3::new(0.0, 0.0, 0.0);
    let up = Vector3::new(0.0, 1.0, 0.0);

    // Rotate the model slightly to show off the 3D shape and lighting
    let model_matrix = TransformFactory::rotation_y(45.0_f32.to_radians());

    let view_matrix = TransformFactory::view(&eye, &target, &up);
    let projection_matrix = TransformFactory::perspective(
        width as f32 / height as f32, // Aspect Ratio
        45.0_f32.to_radians(),        // FOV
        0.1,                          // Near
        100.0,                        // Far
    );

    // 5. Setup Phong Shader
    // Unlike UnlitShader, PhongShader needs individual matrices to calculate
    // world-space positions and normals for lighting.
    let mut shader = PhongShader::new(model_matrix, view_matrix, projection_matrix, eye);

    // Configure Material & Lighting
    shader.light_dir = Vector3::new(1.0, 1.0, 1.0).normalize(); // Light coming from top-right-front
    shader.light_color = Vector3::new(1.0, 1.0, 1.0); // White light
    shader.diffuse_color = Vector3::new(1.0, 0.5, 0.31); // Coral/Orange object color
    shader.ambient_intensity = Vector3::new(0.1, 0.1, 0.1); // Soft ambient light
    shader.specular_color = Vector3::new(1.0, 1.0, 1.0); // White highlights
    shader.shininess = 32.0;

    // 6. Render Loop
    println!("Starting render...");

    // Clear screen with a dark gray color
    renderer.clear(Vector3::new(0.1, 0.1, 0.1));

    // Draw the mesh using the Phong shader
    renderer.draw_mesh(&mesh, &shader);

    // 7. Save Result
    let output_path = "output_phong.png";
    save_buffer_to_image(&renderer.framebuffer, output_path);
    println!("Render saved to {}", output_path);
}

/// Helper function to save the framebuffer to a PNG file
fn save_buffer_to_image(fb: &rasterizer_rust::core::framebuffer::FrameBuffer, path: &str) {
    let mut img_buf = image::ImageBuffer::new(fb.width as u32, fb.height as u32);

    for (x, y, pixel) in img_buf.enumerate_pixels_mut() {
        // Get color from framebuffer (returns Option<Vector3>)
        if let Some(color) = fb.get_pixel(x as usize, y as usize) {
            // Convert float [0.0, 1.0] to u8 [0, 255]
            let r = (color.x.clamp(0.0, 1.0) * 255.0) as u8;
            let g = (color.y.clamp(0.0, 1.0) * 255.0) as u8;
            let b = (color.z.clamp(0.0, 1.0) * 255.0) as u8;
            *pixel = image::Rgb([r, g, b]);
        }
    }

    img_buf.save(Path::new(path)).expect("Failed to save image");
}
