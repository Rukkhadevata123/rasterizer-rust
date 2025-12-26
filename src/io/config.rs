use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub render: RenderConfig,
    pub camera: CameraConfig,
    #[serde(default)]
    pub ground: GroundConfig,
    #[serde(default)]
    pub lights: Vec<LightConfig>,
    #[serde(default)]
    pub objects: Vec<ObjectConfig>,
}

#[derive(Debug, Deserialize)]
pub struct RenderConfig {
    // --- Output & Quality ---
    pub width: usize,
    pub height: usize,
    pub output: String,
    #[serde(default = "default_samples")]
    pub samples: usize,
    #[serde(default = "default_exposure")]
    pub exposure: f32,

    // --- Environment & Background ---
    #[serde(default = "default_ambient")]
    pub ambient_light: [f32; 3],
    #[serde(default)]
    pub background_image: Option<String>,
    pub background_color: Option<[f32; 3]>,
    pub background_gradient_top: Option<[f32; 3]>,
    pub background_gradient_bottom: Option<[f32; 3]>,

    // --- Shadow System ---
    #[serde(default = "default_true")]
    pub use_shadows: bool,
    #[serde(default = "default_shadow_map_size")]
    pub shadow_map_size: usize,
    #[serde(default = "default_shadow_ortho_size")]
    pub shadow_ortho_size: f32,
    #[serde(default = "default_shadow_bias")]
    pub shadow_bias: f32,
    #[serde(default = "default_true")]
    pub use_pcf: bool,
    #[serde(default = "default_pcf_kernel")]
    pub pcf_kernel_size: i32,

    // --- Pipeline & Debug ---
    #[serde(default = "default_cull_mode")]
    pub cull_mode: String, // "back", "front", "none"
    #[serde(default = "default_false")]
    pub wireframe: bool,
}

fn default_shadow_map_size() -> usize {
    2048
}
fn default_samples() -> usize {
    2
}
fn default_ambient() -> [f32; 3] {
    [0.1, 0.1, 0.1]
}
fn default_shadow_bias() -> f32 {
    0.01
}
fn default_shadow_ortho_size() -> f32 {
    8.0
}
fn default_exposure() -> f32 {
    1.0
}
fn default_cull_mode() -> String {
    "back".to_string()
}
fn default_false() -> bool {
    false
}
fn default_pcf_kernel() -> i32 {
    1
}

#[derive(Debug, Deserialize)]
pub struct CameraConfig {
    pub position: [f32; 3],
    pub target: [f32; 3],
    pub up: [f32; 3],
    #[serde(default = "default_fov")]
    pub fov: f32,
    #[serde(default)]
    pub projection: String,
    #[serde(default = "default_ortho_height")]
    pub ortho_height: f32,
    #[serde(default = "default_near")]
    pub near: f32,
    #[serde(default = "default_far")]
    pub far: f32,
}

fn default_fov() -> f32 {
    45.0
}
fn default_ortho_height() -> f32 {
    10.0
}
fn default_near() -> f32 {
    0.1
}
fn default_far() -> f32 {
    100.0
}

#[derive(Debug, Deserialize, Default)]
pub struct GroundConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_ground_size")]
    pub size: f32,
    pub albedo: Option<[f32; 3]>,
    pub metallic: Option<f32>,
    pub roughness: Option<f32>,
}

fn default_true() -> bool {
    true
}
fn default_ground_size() -> f32 {
    10.0
}

#[derive(Debug, Deserialize)]
pub struct LightConfig {
    pub r#type: String,
    pub position: Option<[f32; 3]>,
    pub direction: Option<[f32; 3]>,
    pub color: [f32; 3],
    pub intensity: f32,
    pub attenuation: Option<[f32; 3]>,
}

#[derive(Debug, Deserialize)]
pub struct ObjectConfig {
    pub path: String,

    // --- Transform ---
    #[serde(default)]
    pub position: [f32; 3],
    #[serde(default)]
    pub rotation: [f32; 3],
    #[serde(default = "default_scale")]
    pub scale: [f32; 3],

    // --- Material Values ---
    pub albedo: Option<[f32; 3]>,
    pub metallic: Option<f32>,
    pub roughness: Option<f32>,
    pub ao: Option<f32>,
    pub emissive: Option<[f32; 3]>,
    #[serde(default = "default_emissive_intensity")]
    pub emissive_intensity: f32,

    // --- Material Textures ---
    pub albedo_texture: Option<String>,
    pub metallic_roughness_texture: Option<String>,
    pub normal_texture: Option<String>,
}

fn default_emissive_intensity() -> f32 {
    1.0
}
fn default_scale() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}

impl Config {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let content =
            fs::read_to_string(path).map_err(|e| format!("Failed to read config file: {}", e))?;
        toml::from_str(&content).map_err(|e| format!("Failed to parse TOML: {}", e))
    }
}
