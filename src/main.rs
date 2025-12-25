use nalgebra::{Matrix4, Point3, Vector3};
use rasterizer_rust::core::math::transform::TransformFactory;
use rasterizer_rust::pipeline::renderer::Renderer;
use rasterizer_rust::pipeline::shaders::unlit::UnlitShader;
use rasterizer_rust::scene::mesh::Mesh;
use std::path::Path;

fn main() {
    // 1. Initialize Logger
    // TODO: Inplement logs in other modules
    env_logger::init();

    // 2. Setup Renderer
    let width = 800;
    let height = 600;
    let mut renderer = Renderer::new(width, height);

    // 3. Create Scene Data (A simple triangle)
    let mesh = Mesh::create_test_triangle();

    // 4. Setup Camera & Matrices
    // Camera at (0, 0, 2), looking at (0, 0, 0), Up is (0, 1, 0)
    let eye = Point3::new(0.0, 0.0, 2.0);
    let target = Point3::new(0.0, 0.0, 0.0);
    let up = Vector3::new(0.0, 1.0, 0.0);

    let model_matrix = Matrix4::identity();
    let view_matrix = TransformFactory::view(&eye, &target, &up);
    let projection_matrix = TransformFactory::perspective(
        width as f32 / height as f32, // Aspect Ratio
        45.0_f32.to_radians(),        // FOV
        0.1,                          // Near
        100.0,                        // Far
    );

    // MVP = Projection * View * Model
    let mvp_matrix = projection_matrix * view_matrix * model_matrix;

    // 5. Setup Shader
    let shader = UnlitShader::new(mvp_matrix);

    // 6. Render Loop
    println!("Starting render...");

    // Clear screen with a dark gray color
    renderer.clear(Vector3::new(0.1, 0.1, 0.1));

    // Draw the mesh
    renderer.draw_mesh(&mesh, &shader);

    // 7. Save Result
    let output_path = "test_triangle.png";
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
