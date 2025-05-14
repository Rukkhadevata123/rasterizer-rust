use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Path to the input OBJ file
    #[arg(long)]
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

    /// 使用基于物理的渲染 (PBR) 而不是传统 Blinn-Phong
    #[arg(long, default_value_t = false)]
    pub use_pbr: bool,

    /// 材质的金属度 (0.0-1.0，仅在PBR模式下有效)
    #[arg(long, default_value_t = 0.0)]
    pub metallic: f32,

    /// 材质的粗糙度 (0.0-1.0，仅在PBR模式下有效)
    #[arg(long, default_value_t = 0.5)]
    pub roughness: f32,

    /// 材质的基础颜色，格式为"r,g,b"，每个分量在0.0-1.0范围内 (仅在PBR模式下有效)
    #[arg(long, default_value = "0.8,0.8,0.8")]
    pub base_color: String,

    /// 环境光遮蔽系数 (0.0-1.0，仅在PBR模式下有效)
    #[arg(long, default_value_t = 1.0)]
    pub ambient_occlusion: f32,

    /// 材质的自发光颜色，格式为"r,g,b"，每个分量在0.0-1.0范围内 (仅在PBR模式下有效)
    #[arg(long, default_value = "0.0,0.0,0.0")]
    pub emissive: String,

    /// Disable texture loading and usage
    #[arg(long, default_value_t = false)]
    pub no_texture: bool,

    /// Explicitly specify a texture file to use, overriding MTL settings.
    #[arg(long)]
    pub texture: Option<String>,

    /// 禁用gamma矫正（默认启用）
    #[arg(long, default_value_t = false)]
    pub no_gamma: bool,

    /// 场景中要创建的对象实例数量
    #[arg(long)]
    pub object_count: Option<String>,

    /// Run the full animation loop instead of a single frame render
    #[arg(long, default_value_t = false)]
    pub animate: bool,

    /// 动画的总帧数
    #[arg(long, default_value_t = 120)]
    pub total_frames: usize,

    /// 是否显示额外的调试信息
    #[arg(long, default_value_t = false)]
    pub show_debug_info: bool,

    /// 是否测试场景管理功能
    #[arg(long, default_value_t = false)]
    pub test_scene_management: bool,

    /// 是否测试场景清除功能
    #[arg(long, default_value_t = false)]
    pub test_clear_scene: bool,

    /// 是否测试材质操作功能
    #[arg(long, default_value_t = false)]
    pub test_materials: bool,
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
