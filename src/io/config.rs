use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub render: RenderConfig,
    #[serde(default)]
    pub camera: CameraConfig,
    #[serde(default)]
    pub ground: GroundConfig,
    #[serde(default)]
    pub lights: Vec<LightConfig>,
    #[serde(default)]
    pub objects: Vec<ObjectConfig>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            render: RenderConfig::default(),
            camera: CameraConfig::default(),
            ground: GroundConfig::default(),
            lights: vec![
                LightConfig {
                    r#type: "directional".to_string(),
                    direction: Some([-1.0, -2.0, -1.0]),
                    color: [1.0, 0.95, 0.8],
                    intensity: 3.5,
                    position: None,
                    attenuation: None,
                },
                LightConfig {
                    r#type: "point".to_string(),
                    position: Some([-3.0, 2.0, 2.0]),
                    color: [0.1, 0.1, 0.3],
                    intensity: 1.0,
                    attenuation: Some([1.0, 0.09, 0.032]),
                    direction: None,
                },
            ],
            objects: vec![
                ObjectConfig {
                    path: "assets/spot/spot_triangulated.obj".to_string(),
                    position: [-1.0, 0.0, 0.5],
                    rotation: [0.0, 30.0, 0.0],
                    scale: [1.0, 1.0, 1.0],
                    albedo: Some([1.00, 0.76, 0.33]),
                    metallic: Some(1.0),
                    roughness: Some(0.3),
                    ao: Some(0.99),
                    emissive: Some([0.01, 0.02, 0.03]),
                    emissive_intensity: 0.7,
                    albedo_texture: None,
                    metallic_roughness_texture: None,
                    normal_texture: None,
                },
                ObjectConfig {
                    path: "assets/simple/sphere.obj".to_string(),
                    position: [1.5, 0.5, -1.0],
                    scale: [1.5, 1.5, 1.5],
                    albedo: Some([0.1, 0.1, 0.8]),
                    metallic: Some(0.0),
                    roughness: Some(0.4),
                    rotation: [0.0, 0.0, 0.0],
                    ao: None,
                    emissive: None,
                    emissive_intensity: 1.0,
                    albedo_texture: None,
                    metallic_roughness_texture: None,
                    normal_texture: None,
                },
            ],
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct RenderConfig {
    // --- Output & Quality ---
    #[serde(default = "default_width")]
    pub width: usize,
    #[serde(default = "default_height")]
    pub height: usize,
    #[serde(default = "default_output")]
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

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            width: default_width(),
            height: default_height(),
            output: default_output(),
            samples: default_samples(),
            exposure: default_exposure(),
            ambient_light: default_ambient(),
            background_image: None,
            background_color: None,
            background_gradient_top: Some([0.2, 0.2, 0.3]),
            background_gradient_bottom: Some([0.05, 0.05, 0.1]),
            use_shadows: true,
            shadow_map_size: default_shadow_map_size(),
            shadow_ortho_size: default_shadow_ortho_size(),
            shadow_bias: default_shadow_bias(),
            use_pcf: true,
            pcf_kernel_size: default_pcf_kernel(),
            cull_mode: default_cull_mode(),
            wireframe: false,
        }
    }
}

// Defaults matching scene.toml
fn default_width() -> usize {
    3840
}
fn default_height() -> usize {
    2160
}
fn default_output() -> String {
    "output_default.png".to_string()
}
fn default_shadow_map_size() -> usize {
    4096
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
fn default_true() -> bool {
    true
}
fn default_pcf_kernel() -> i32 {
    2
}

#[derive(Debug, Deserialize)]
pub struct CameraConfig {
    #[serde(default)]
    pub position: [f32; 3],
    #[serde(default)]
    pub target: [f32; 3],
    #[serde(default)]
    pub up: [f32; 3],
    #[serde(default = "default_fov")]
    pub fov: f32,
    #[serde(default = "default_projection")]
    pub projection: String,
    #[serde(default = "default_ortho_height")]
    pub ortho_height: f32,
    #[serde(default = "default_near")]
    pub near: f32,
    #[serde(default = "default_far")]
    pub far: f32,
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            position: [0.0, 4.0, 5.0],
            target: [0.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            fov: default_fov(),
            projection: default_projection(),
            ortho_height: default_ortho_height(),
            near: default_near(),
            far: default_far(),
        }
    }
}

fn default_fov() -> f32 {
    45.0
}
fn default_projection() -> String {
    "perspective".to_string()
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

#[derive(Debug, Deserialize)]
pub struct GroundConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_ground_size")]
    pub size: f32,
    pub albedo: Option<[f32; 3]>,
    pub metallic: Option<f32>,
    pub roughness: Option<f32>,
}

impl Default for GroundConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            size: default_ground_size(),
            albedo: Some([0.8, 0.8, 0.8]),
            metallic: Some(0.0),
            roughness: Some(0.8),
        }
    }
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
