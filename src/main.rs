use clap::Parser;
use nalgebra::{Point3, Vector3};
use std::fs;
use std::path::Path;
use std::time::Instant;

// Declare modules
mod args;
mod camera;
mod color_utils;
mod interpolation;
mod lighting;
mod loaders;
mod model_types; // Added module declaration
mod rasterizer;
mod renderer;
mod texture_utils;
mod transform;

// Use statements
use args::{Args, parse_point3, parse_vec3};
use camera::Camera;
use color_utils::apply_colormap_jet;
use lighting::Light;
use loaders::load_obj_enhanced; // load_obj_enhanced remains in loaders
use model_types::ModelData; // ModelData is now in model_types
use renderer::{RenderSettings, Renderer};

// Helper function to save image data to a file
fn save_image(path: &str, data: &[u8], width: u32, height: u32) {
    match image::save_buffer(path, data, width, height, image::ColorType::Rgb8) {
        Ok(_) => println!("Image saved to {}", path),
        Err(e) => eprintln!("Error saving image to {}: {}", path, e),
    }
}

/// Normalizes depth buffer values for visualization using percentile clipping.
#[allow(dead_code)] // Allow dead code because it's only used when !no_depth
fn normalize_depth(
    depth_buffer: &[f32],
    min_percentile: f32, // e.g., 1.0 for 1st percentile
    max_percentile: f32, // e.g., 99.0 for 99th percentile
) -> Vec<f32> {
    // 1. Collect finite depth values
    let mut finite_depths: Vec<f32> = depth_buffer
        .iter()
        .filter(|&&d| d.is_finite())
        .cloned()
        .collect();

    // Declare min_clip and max_clip as mutable
    let mut min_clip: f32;
    let mut max_clip: f32;

    // 2. Determine normalization range using percentiles
    if finite_depths.len() >= 2 {
        // Need at least two points to define a range
        // Sort the finite depths
        finite_depths.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap()); // Use unstable sort for performance

        // Calculate indices for percentiles
        let min_idx = ((min_percentile / 100.0 * (finite_depths.len() - 1) as f32).round()
            as usize)
            .clamp(0, finite_depths.len() - 1);
        let max_idx = ((max_percentile / 100.0 * (finite_depths.len() - 1) as f32).round()
            as usize)
            .clamp(0, finite_depths.len() - 1);

        min_clip = finite_depths[min_idx]; // Initial assignment
        max_clip = finite_depths[max_idx]; // Initial assignment

        // Ensure min_clip < max_clip
        if (max_clip - min_clip).abs() < 1e-6 {
            // If range is too small, expand it slightly or use a default
            // For simplicity, let's just use the absolute min/max in this edge case
            min_clip = *finite_depths.first().unwrap(); // Re-assignment is now allowed
            max_clip = *finite_depths.last().unwrap(); // Re-assignment is now allowed
            // Ensure max > min even if all values were identical
            if (max_clip - min_clip).abs() < 1e-6 {
                max_clip = min_clip + 1.0; // Re-assignment is now allowed
            }
        }
        println!(
            "Normalizing depth using percentiles: [{:.1}%, {:.1}%] -> [{:.3}, {:.3}]",
            min_percentile, max_percentile, min_clip, max_clip
        );
    } else {
        // Fallback if not enough finite values
        println!(
            "Warning: Not enough finite depth values for percentile clipping. Using default range [0.1, 10.0]."
        );
        min_clip = 0.1; // Default near // Assignment
        max_clip = 10.0; // Default far (adjust as needed) // Assignment
    }

    let range = max_clip - min_clip;
    let inv_range = if range > 1e-6 { 1.0 / range } else { 0.0 }; // Avoid division by zero

    // 3. Normalize the original buffer using the calculated range
    depth_buffer
        .iter()
        .map(|&depth| {
            if depth.is_finite() {
                // Clamp depth to the calculated range and normalize
                ((depth.clamp(min_clip, max_clip) - min_clip) * inv_range).clamp(0.0, 1.0)
            } else {
                // Map non-finite values (infinity) to 1.0 (far)
                1.0
            }
        })
        .collect()
}

/// Normalizes and centers the model's vertices in place.
/// Returns the original center and scaling factor.
/// Moved from Renderer
fn normalize_and_center_model(model_data: &mut ModelData) -> (Vector3<f32>, f32) {
    if model_data.meshes.is_empty() {
        return (Vector3::zeros(), 1.0);
    }

    // Calculate bounding box or centroid of all vertices
    let mut min_coord = Point3::new(f32::MAX, f32::MAX, f32::MAX);
    let mut max_coord = Point3::new(f32::MIN, f32::MIN, f32::MIN);
    let mut vertex_sum = Vector3::zeros();
    let mut vertex_count = 0;

    for mesh in &model_data.meshes {
        for vertex in &mesh.vertices {
            min_coord = min_coord.inf(&vertex.position);
            max_coord = max_coord.sup(&vertex.position);
            vertex_sum += vertex.position.coords;
            vertex_count += 1;
        }
    }

    if vertex_count == 0 {
        return (Vector3::zeros(), 1.0);
    }

    let center = vertex_sum / (vertex_count as f32);
    let extent = max_coord - min_coord;
    let max_extent = extent.x.max(extent.y).max(extent.z);

    let scale_factor = if max_extent > 1e-6 {
        1.6 / max_extent // Scale to fit roughly in [-0.8, 0.8] cube (like Python's 0.8 factor)
    } else {
        1.0
    };

    // Apply transformation to all vertices
    for mesh in &mut model_data.meshes {
        for vertex in &mut mesh.vertices {
            vertex.position = Point3::from((vertex.position.coords - center) * scale_factor);
        }
    }

    (center, scale_factor)
}

/// Sets up the camera, light, and render settings for a given frame.
fn setup_render_environment(
    args: &Args,
    frame_num: Option<usize>, // None for single frame, Some(i) for animation frame i
) -> Result<(Camera, Light, RenderSettings), String> {
    let aspect_ratio = args.width as f32 / args.height as f32;

    // --- Camera Setup ---
    let initial_camera_from = parse_point3(&args.camera_from)
        .map_err(|e| format!("Invalid camera_from format: {}", e))?;
    let camera_at =
        parse_point3(&args.camera_at).map_err(|e| format!("Invalid camera_at format: {}", e))?;
    let camera_up =
        parse_vec3(&args.camera_up).map_err(|e| format!("Invalid camera_up format: {}", e))?;

    let mut camera = Camera::new(
        initial_camera_from,
        camera_at,
        camera_up,
        args.camera_fov,
        aspect_ratio,
        0.1,   // near plane distance
        100.0, // far plane distance
    );

    if let Some(frame_idx) = frame_num {
        if frame_idx > 0 {
            let total_frames = 120; // TODO: Make this configurable or pass it in
            let rotation_per_frame = 360.0 / total_frames as f32;
            let current_angle = frame_idx as f32 * rotation_per_frame;
            camera.orbit_y(current_angle);
        }
    }

    // --- Light Setup ---
    let light = if args.no_lighting {
        if frame_num.is_none_or(|f| f == 0) {
            // Print only once or for single frame
            println!("Lighting disabled. Using ambient only.");
        }
        Light::Ambient(Vector3::new(args.ambient, args.ambient, args.ambient))
    } else {
        let light_intensity = Vector3::new(1.0, 1.0, 1.0) * args.diffuse;
        match args.light_type.to_lowercase().as_str() {
            "point" => {
                let light_pos = parse_point3(&args.light_pos)
                    .map_err(|e| format!("Invalid light_pos format: {}", e))?;
                let atten_parts: Vec<Result<f32, _>> = args
                    .light_atten
                    .split(',')
                    .map(|s| s.trim().parse::<f32>())
                    .collect();
                if atten_parts.len() != 3 || atten_parts.iter().any(|r| r.is_err()) {
                    return Err(format!(
                        "Invalid light_atten format: '{}'. Expected 'c,l,q'",
                        args.light_atten
                    ));
                }
                // Use map_or to handle potential errors during parsing, defaulting to 0.0
                let attenuation = (
                    atten_parts[0].as_ref().map_or(0.0, |v| *v).max(0.0),
                    atten_parts[1].as_ref().map_or(0.0, |v| *v).max(0.0),
                    atten_parts[2].as_ref().map_or(0.0, |v| *v).max(0.0),
                );
                if frame_num.is_none_or(|f| f == 0) {
                    // Print only once or for single frame
                    println!(
                        "Using Point Light at {:?}, Intensity Scale: {:.2}, Attenuation: {:?}",
                        light_pos, args.diffuse, attenuation
                    );
                }
                Light::Point {
                    position: light_pos,
                    intensity: light_intensity,
                    attenuation,
                }
            }
            "directional" => {
                let mut light_dir = parse_vec3(&args.light_dir)
                    .map_err(|e| format!("Invalid light_dir format: {}", e))?;
                light_dir = -light_dir.normalize(); // Direction *towards* light
                if frame_num.is_none_or(|f| f == 0) {
                    // Print only once or for single frame
                    println!(
                        "Using Directional Light towards {:?}, Intensity Scale: {:.2}",
                        light_dir, args.diffuse
                    );
                }
                Light::Directional {
                    direction: light_dir,
                    intensity: light_intensity,
                }
            }
            _ => {
                // 默认为定向光
                let mut light_dir = parse_vec3(&args.light_dir)
                    .map_err(|e| format!("Invalid light_dir format: {}", e))?;
                light_dir = -light_dir.normalize(); // Direction *towards* light
                if frame_num.is_none_or(|f| f == 0) {
                    // Print only once or for single frame
                    println!(
                        "Using Directional Light towards {:?}, Intensity Scale: {:.2}",
                        light_dir, args.diffuse
                    );
                }
                Light::Directional {
                    direction: light_dir,
                    intensity: light_intensity,
                }
            }
        }
    };

    // --- Render Settings Setup ---
    let settings = RenderSettings {
        projection_type: args.projection.clone(),
        use_zbuffer: !args.no_zbuffer,
        use_face_colors: args.colorize,
        use_texture: !args.no_texture,
        light,
        use_phong: args.use_phong, // 添加 Phong 着色设置
    };

    Ok((camera, settings.light, settings))
}

/// Renders a single frame with the given parameters.
fn render_single_frame(
    args: &Args,
    model_data: &ModelData,
    camera: &Camera,
    renderer: &Renderer,
    settings: &RenderSettings,
    output_name: &str, // Base name for output files (e.g., "output" or "frame_001")
) -> Result<(), String> {
    let frame_start_time = Instant::now();
    println!("Rendering frame: {}", output_name);

    // --- Render Current Frame ---
    renderer.render(model_data, camera, settings);

    // --- Save Output Frame ---
    println!("Saving output images for {}...", output_name);
    let color_data = renderer.frame_buffer.get_color_buffer_bytes();
    // Use args.output_dir for saving path
    let color_path = Path::new(&args.output_dir)
        .join(format!("{}_color.png", output_name))
        .to_str()
        .ok_or("Failed to create color output path string")?
        .to_string();
    save_image(
        &color_path,
        &color_data,
        args.width as u32,
        args.height as u32,
    );

    // Save depth map if enabled
    if settings.use_zbuffer && !args.no_depth {
        let depth_data_raw = renderer.frame_buffer.get_depth_buffer_f32();
        let depth_normalized = normalize_depth(&depth_data_raw, 1.0, 99.0);
        let depth_colored = apply_colormap_jet(
            &depth_normalized
                .iter()
                .map(|&d| 1.0 - d) // Invert: closer = hotter
                .collect::<Vec<_>>(),
            args.width,
            args.height,
        );
        // Use args.output_dir for saving path
        let depth_path = Path::new(&args.output_dir)
            .join(format!("{}_depth.png", output_name))
            .to_str()
            .ok_or("Failed to create depth output path string")?
            .to_string();
        save_image(
            &depth_path,
            &depth_colored,
            args.width as u32,
            args.height as u32,
        );
    }
    println!(
        "Frame {} finished in {:?}",
        output_name,
        frame_start_time.elapsed()
    );
    Ok(())
}

/// Runs the animation loop, rendering multiple frames.
fn run_animation_loop(
    args: &Args,
    model_data: &ModelData, // Model is already normalized
    renderer: &Renderer,
) -> Result<(), String> {
    let total_frames = 120; // Define total frames for the animation
    // Removed redundant aspect_ratio calculation

    println!("Starting animation render ({total_frames} frames)...");

    // Removed redundant initial camera/light parsing

    for frame_num in 0..total_frames {
        println!("--- Preparing Frame {} ---", frame_num);

        // --- Setup Environment for current frame ---
        let (camera, _light, settings) = setup_render_environment(args, Some(frame_num))?;

        // --- Render and Save Frame ---
        let frame_output_name = format!("frame_{:03}", frame_num);
        render_single_frame(
            args,
            model_data,
            &camera, // Pass the camera for this frame
            renderer,
            &settings, // Pass the settings for this frame
            &frame_output_name,
        )?;
    } // End of frame loop

    println!("Animation rendering complete.");
    Ok(())
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    let start_time = Instant::now();

    // --- Validate Inputs & Setup ---
    if !Path::new(&args.obj).exists() {
        return Err(format!("Error: Input OBJ file not found: {}", args.obj));
    }
    // Ensure the output directory exists (used by both modes)
    fs::create_dir_all(&args.output_dir).map_err(|e| {
        format!(
            "Failed to create output directory '{}': {}",
            args.output_dir, e
        )
    })?;

    // --- Load Model ---
    println!("Loading model: {}", args.obj);
    let load_start = Instant::now();
    // 修改这里，传递 &args 以支持命令行纹理
    let mut model_data = load_obj_enhanced(&args.obj, &args)?;
    println!("Model loaded in {:?}", load_start.elapsed());

    // --- Normalize Model (Once) ---
    println!("Normalizing model...");
    let norm_start_time = Instant::now();
    let (original_center, scale_factor) = normalize_and_center_model(&mut model_data);
    let norm_duration = norm_start_time.elapsed();
    println!(
        "Model normalized in {:?}. Original Center: {:.3?}, Scale Factor: {:.3}",
        norm_duration, original_center, scale_factor
    );

    // --- Create Renderer ---
    let renderer = Renderer::new(args.width, args.height);

    // --- Decide Mode: Animation or Single Frame ---
    if args.animate {
        // Animation mode uses args.output_dir directly
        run_animation_loop(&args, &model_data, &renderer)?;
    } else {
        // --- Setup for Single Frame Render ---
        println!("--- Preparing Single Frame Render ---");
        // Use the new setup function
        let (camera, _light, settings) = setup_render_environment(&args, None)?;

        // Single frame mode uses args.output as the base name
        render_single_frame(
            &args,
            &model_data,
            &camera,
            &renderer,
            &settings,
            &args.output, // Use the base output name for single frame
        )?;
    }

    println!("Total execution time: {:?}", start_time.elapsed());
    Ok(())
}
