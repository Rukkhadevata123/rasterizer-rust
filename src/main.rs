use clap::Parser;
use nalgebra::{Point3, Vector3};
use std::fs;
use std::path::Path;
use std::time::Instant;

mod args;
mod camera;
mod color_utils;
mod interpolation;
mod lighting;
mod loaders;
mod rasterizer;
mod renderer;
mod texture_utils;
mod transform; // Declare lighting module

use args::{Args, parse_point3, parse_vec3};
use camera::Camera;
use color_utils::apply_colormap_jet;
use lighting::Light;
use loaders::load_obj_enhanced;
use renderer::{RenderSettings, Renderer}; // Use Light enum

// Helper function to save image data to a file
fn save_image(path: &str, data: &[u8], width: u32, height: u32) {
    match image::save_buffer(path, data, width, height, image::ColorType::Rgb8) {
        Ok(_) => println!("Image saved to {}", path),
        Err(e) => eprintln!("Error saving image to {}: {}", path, e),
    }
}

// Helper function to normalize depth buffer values for visualization
fn normalize_depth(depth_buffer: &[f32], near: f32, far: f32) -> Vec<f32> {
    // Find min/max finite depth values in the buffer
    let mut min_depth = f32::INFINITY;
    let mut max_depth = f32::NEG_INFINITY;
    let mut has_finite = false;
    for &depth in depth_buffer {
        if depth.is_finite() {
            min_depth = min_depth.min(depth);
            max_depth = max_depth.max(depth);
            has_finite = true;
        }
    }

    // Handle cases with no finite values or a single finite value
    if !has_finite {
        min_depth = near.max(0.0); // Use near/far as fallback range, ensure non-negative
        max_depth = far.max(min_depth + 1e-6);
    } else if (max_depth - min_depth).abs() < 1e-6 {
        // If all finite values are the same, create a small range for normalization
        min_depth = (min_depth - 0.5).max(0.0); // Adjust min slightly down
        max_depth = min_depth + 1.0; // Adjust max slightly up
    }

    let range = max_depth - min_depth;
    let inv_range = if range > 1e-6 { 1.0 / range } else { 0.0 }; // Avoid division by zero

    println!(
        "Normalizing depth buffer. Found range: [{:.3}, {:.3}]",
        min_depth, max_depth
    );

    depth_buffer
        .iter()
        .map(|&depth| {
            if depth.is_finite() {
                // Normalize finite values to [0, 1] based on the calculated range
                ((depth - min_depth) * inv_range).clamp(0.0, 1.0)
            } else {
                // Map non-finite values (infinity) to 1.0 (far)
                1.0
            }
        })
        .collect()
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    let start_time = Instant::now();

    // --- Validate Inputs ---
    if !Path::new(&args.obj).exists() {
        return Err(format!("Error: Input OBJ file not found: {}", args.obj));
    }
    // Ensure output directory exists
    fs::create_dir_all(&args.output_dir).map_err(|e| {
        format!(
            "Failed to create output directory '{}': {}",
            args.output_dir, e
        )
    })?;

    // --- Load Model ---
    println!("Loading model: {}", args.obj);
    let load_start = Instant::now();
    // Load the model, make it mutable for normalization
    let mut model_data = load_obj_enhanced(&args.obj)?;
    println!("Model loaded in {:?}", load_start.elapsed());

    // --- Setup Camera ---
    // Parse camera_from, camera_at, and camera_up strings using helper functions
    let camera_from =
        parse_point3(&args.camera_from).map_err(|e| format!("Invalid camera_from format: {}", e))?;
    let camera_at =
        parse_point3(&args.camera_at).map_err(|e| format!("Invalid camera_at format: {}", e))?;
    let camera_up =
        parse_vec3(&args.camera_up).map_err(|e| format!("Invalid camera_up format: {}", e))?;
    let aspect_ratio = args.width as f32 / args.height as f32;
    // Create the camera instance
    let camera = Camera::new(
        camera_from, // Use the parsed Point3
        camera_at,
        camera_up,
        args.camera_fov,
        aspect_ratio,
        0.1,   // near plane distance
        100.0, // far plane distance
    );

    // --- Setup Lighting ---
    // Determine the light source based on arguments
    let light = if args.no_lighting {
        // If lighting is disabled, use only ambient light
        println!("Lighting disabled. Using ambient only.");
        Light::Ambient(Vector3::new(args.ambient, args.ambient, args.ambient))
    } else {
        // Default light intensity (white light scaled by diffuse factor)
        let light_intensity = Vector3::new(1.0, 1.0, 1.0) * args.diffuse;

        match args.light_type.to_lowercase().as_str() {
            "point" => {
                // Setup point light
                let light_pos = parse_point3(&args.light_pos)
                    .map_err(|e| format!("Invalid light_pos format: {}", e))?;
                // Parse attenuation factors
                let atten_parts: Vec<Result<f32, _>> = args
                    .light_atten
                    .split(',')
                    .map(|s| s.trim().parse::<f32>())
                    .collect();
                if atten_parts.len() != 3 || atten_parts.iter().any(|r| r.is_err()) {
                    return Err(format!(
                        "Invalid light_atten format: '{}'. Expected 'constant,linear,quadratic'",
                        args.light_atten
                    ));
                }
                let attenuation = (
                    atten_parts[0].as_ref().unwrap().max(0.0), // constant
                    atten_parts[1].as_ref().unwrap().max(0.0), // linear
                    atten_parts[2].as_ref().unwrap().max(0.0), // quadratic
                );
                println!(
                    "Using Point Light at {:?}, Intensity Scale: {:.2}, Attenuation: {:?}",
                    light_pos, args.diffuse, attenuation
                );
                Light::Point {
                    position: light_pos,
                    intensity: light_intensity,
                    attenuation,
                }
            }
            "directional" | _ => {
                // Default to directional light
                // Setup directional light
                let mut light_dir = parse_vec3(&args.light_dir)
                    .map_err(|e| format!("Invalid light_dir format: {}", e))?;
                // Direction should be *towards* the light source (negate the direction *from* the source)
                light_dir = -light_dir.normalize();
                println!(
                    "Using Directional Light towards {:?}, Intensity Scale: {:.2}",
                    light_dir, args.diffuse
                );
                Light::Directional {
                    direction: light_dir,
                    intensity: light_intensity,
                }
            }
        }
    };

    // --- Setup Renderer ---
    // Create the renderer with specified dimensions
    let renderer = Renderer::new(args.width, args.height);
    // Configure render settings based on arguments
    let settings = RenderSettings {
        projection_type: args.projection.clone(),
        use_zbuffer: !args.no_zbuffer,
        use_face_colors: args.colorize,
        use_texture: !args.no_texture,
        light, // Pass the configured light source
    };

    // --- Render ---
    // Perform the rendering process
    println!("Starting render...");
    renderer.render(&mut model_data, &camera, &settings);

    // --- Save Output ---
    println!("Saving output images...");
    // Get the rendered color buffer data
    let color_data = renderer.frame_buffer.get_color_buffer_bytes();
    // Construct the output color image path
    let color_path = Path::new(&args.output_dir) // Uses --output-dir
        .join(format!("{}_color.png", args.output)) // Uses --output (e.g., "frame_000")
        .to_str()
        .ok_or("Failed to create color output path string")?
        .to_string();
    // Save the color image
    save_image(
        &color_path, // Should be like "output_rust_bunny_orbit_rust/frame_000_color.png"
        &color_data,
        args.width as u32,
        args.height as u32,
    );

    // Save depth map if Z-buffer was used and depth output is not disabled
    if settings.use_zbuffer && !args.no_depth {
        // Get the raw depth buffer data
        let depth_data_raw = renderer.frame_buffer.get_depth_buffer_f32();
        // Normalize depth values for visualization
        let depth_normalized = normalize_depth(&depth_data_raw, camera.near(), camera.far());
        // Apply JET colormap (inverting normalized depth so closer is hotter/red)
        let depth_colored = apply_colormap_jet(
            &depth_normalized
                .iter()
                .map(|&d| 1.0 - d)
                .collect::<Vec<_>>(),
            args.width,
            args.height,
        );
        // Construct the output depth image path
        let depth_path = Path::new(&args.output_dir)
            .join(format!("{}_depth.png", args.output))
            .to_str()
            .ok_or("Failed to create depth output path string")?
            .to_string();
        // Save the colored depth map
        save_image(
            &depth_path,
            &depth_colored,
            args.width as u32,
            args.height as u32,
        );
    }

    println!("Total execution time: {:?}", start_time.elapsed());
    println!("Done.");
    Ok(()) // Indicate successful execution
}
