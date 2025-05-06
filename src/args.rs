use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Path to the input OBJ file
    #[arg(short, long)]
    pub obj: String,

    /// Base name for output files (e.g., "render" -> "render_color.png", "render_depth.png")
    #[arg(short, long, default_value = "output")]
    pub output: String,

    /// Directory for output images
    #[arg(long, default_value = "output_rust")]
    pub output_dir: String,

    /// Width of the output image
    #[arg(long, default_value_t = 1024)]
    pub width: usize,

    /// Height of the output image
    #[arg(long, default_value_t = 1024)]
    pub height: usize,

    /// Projection type: "perspective" or "orthographic"
    #[arg(long, default_value = "perspective")]
    pub projection: String,

    /// Disable Z-buffer (depth testing)
    #[arg(long, default_value_t = false)]
    pub no_zbuffer: bool,

    /// Use pseudo-random face colors instead of material colors
    #[arg(long, default_value_t = false)]
    pub colorize: bool,

    /// Disable rendering and saving the depth map
    #[arg(long, default_value_t = false)]
    pub no_depth: bool,

    // --- Camera Arguments ---
    /// Camera position (eye) as "x,y,z"
    #[arg(long, default_value = "0,0,3", allow_negative_numbers = true)]
    pub camera_from: String,

    /// Camera target (look at) as "x,y,z"
    #[arg(long, default_value = "0,0,0", allow_negative_numbers = true)]
    pub camera_at: String,

    /// Camera world up direction as "x,y,z"
    #[arg(long, default_value = "0,1,0", allow_negative_numbers = true)]
    pub camera_up: String,

    /// Camera vertical field of view in degrees (for perspective)
    #[arg(long, default_value_t = 45.0)]
    pub camera_fov: f32,

    // --- Lighting Arguments ---
    /// Disable lighting calculations
    #[arg(long, default_value_t = false)]
    pub no_lighting: bool,

    /// Type of light source: "directional" or "point"
    #[arg(long, default_value = "directional")]
    pub light_type: String,

    /// Direction *from* the light source (for directional light) as "x,y,z"
    #[arg(long, default_value = "0,-1,-1", allow_negative_numbers = true)]
    pub light_dir: String,

    /// Position of the light source (for point light) as "x,y,z"
    #[arg(long, default_value = "0,5,5", allow_negative_numbers = true)]
    pub light_pos: String,

    /// Attenuation factors (constant, linear, quadratic) for point light as "c,l,q"
    #[arg(long, default_value = "1.0,0.09,0.032", allow_negative_numbers = true)]
    pub light_atten: String,

    /// Ambient light intensity factor
    #[arg(long, default_value_t = 0.1)]
    pub ambient: f32,

    /// Diffuse light intensity factor
    #[arg(long, default_value_t = 0.8)]
    pub diffuse: f32,

    /// 使用 Phong 着色（逐像素光照）而非默认的 Flat 着色
    #[arg(long, default_value_t = false)]
    pub use_phong: bool,

    // --- Texture Arguments ---
    /// Disable texture loading and usage
    #[arg(long, default_value_t = false)]
    pub no_texture: bool,

    /// Explicitly specify a texture file to use, overriding MTL settings.
    #[arg(long)]
    pub texture: Option<String>,

    /// 禁用gamma矫正（默认启用）
    #[arg(long, default_value_t = false)]
    pub no_gamma: bool,

    /// Run the full animation loop instead of a single frame render
    #[arg(long, default_value_t = false)]
    pub animate: bool,
}

// Helper function to parse comma-separated floats
// Standard parse should handle negatives, no special logic needed here
pub fn parse_vec3(s: &str) -> Result<nalgebra::Vector3<f32>, String> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 3 {
        return Err("Expected 3 comma-separated values".to_string());
    }
    let x = parts[0]
        .trim()
        .parse::<f32>()
        .map_err(|e| format!("Invalid number '{}': {}", parts[0], e))?;
    let y = parts[1]
        .trim()
        .parse::<f32>()
        .map_err(|e| format!("Invalid number '{}': {}", parts[1], e))?;
    let z = parts[2]
        .trim()
        .parse::<f32>()
        .map_err(|e| format!("Invalid number '{}': {}", parts[2], e))?;
    Ok(nalgebra::Vector3::new(x, y, z))
}

pub fn parse_point3(s: &str) -> Result<nalgebra::Point3<f32>, String> {
    parse_vec3(s).map(nalgebra::Point3::from)
}
