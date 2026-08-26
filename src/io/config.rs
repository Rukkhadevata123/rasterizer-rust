use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub render: RenderConfig,
    pub camera: CameraConfig,
    pub ground: GroundConfig,
    pub lights: Vec<LightConfig>,
    pub objects: Vec<ObjectConfig>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            render: RenderConfig::default(),
            camera: CameraConfig::default(),
            ground: GroundConfig::default(),
            lights: vec![LightConfig {
                kind: LightKind::Directional,
                direction: Some([-1.0, -2.0, -1.0]),
                color: [1.0, 0.95, 0.8],
                intensity: 3.5,
                position: None,
                attenuation: None,
            }],
            objects: vec![ObjectConfig {
                path: "assets/glbs/old_rusty_car.glb".to_string(),
                position: [0.0, 0.0, 0.0],
                rotation: [0.0, -45.0, 0.0],
                scale: [2.0, 2.0, 2.0],
            }],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum CullModeConfig {
    #[default]
    Back,
    Front,
    None,
}

#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RenderConfig {
    pub width: usize,
    pub height: usize,
    pub output: String,
    pub supersample_scale: usize,
    pub exposure: f32,
    pub ambient_light: [f32; 3],
    pub background_image: Option<String>,
    pub background_color: Option<[f32; 3]>,
    pub background_gradient_top: Option<[f32; 3]>,
    pub background_gradient_bottom: Option<[f32; 3]>,
    pub use_shadows: bool,
    pub shadow_map_size: usize,
    pub shadow_ortho_size: f32,
    pub shadow_bias: f32,
    pub use_pcf: bool,
    pub pcf_kernel_size: i32,
    pub use_aces: bool,
    pub cull_mode: CullModeConfig,
    pub wireframe: bool,
    pub use_mipmap: bool,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 720,
            output: "output_default.png".to_string(),
            supersample_scale: 1,
            exposure: 1.0,
            ambient_light: [0.1, 0.1, 0.1],
            background_image: None,
            background_color: None,
            background_gradient_top: Some([0.2, 0.2, 0.3]),
            background_gradient_bottom: Some([0.05, 0.05, 0.1]),
            use_shadows: true,
            shadow_map_size: 720,
            shadow_ortho_size: 8.0,
            shadow_bias: 0.01,
            use_pcf: true,
            pcf_kernel_size: 1,
            use_aces: true,
            cull_mode: CullModeConfig::Back,
            wireframe: false,
            use_mipmap: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ProjectionMode {
    #[default]
    Perspective,
    Orthographic,
}

#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CameraConfig {
    pub position: [f32; 3],
    pub target: [f32; 3],
    pub up: [f32; 3],
    pub fov: f32,
    pub projection: ProjectionMode,
    pub ortho_height: f32,
    pub near: f32,
    pub far: f32,
    pub speed: f32,
    pub sensitivity: f32,
    pub zoom_speed: f32,
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            position: [0.0, 4.0, 5.0],
            target: [0.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            fov: 45.0,
            projection: ProjectionMode::Perspective,
            ortho_height: 10.0,
            near: 0.1,
            far: 100.0,
            speed: 5.0,
            sensitivity: 0.005,
            zoom_speed: 0.02,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GroundConfig {
    pub enabled: bool,
    pub size: f32,
    pub albedo: Option<[f32; 3]>,
    pub metallic: Option<f32>,
    pub roughness: Option<f32>,
}

impl Default for GroundConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            size: 10.0,
            albedo: Some([0.8, 0.8, 0.8]),
            metallic: Some(0.0),
            roughness: Some(0.8),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LightKind {
    Directional,
    Point,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LightConfig {
    #[serde(rename = "type")]
    pub kind: LightKind,
    pub position: Option<[f32; 3]>,
    pub direction: Option<[f32; 3]>,
    pub color: [f32; 3],
    pub intensity: f32,
    pub attenuation: Option<[f32; 3]>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectConfig {
    pub path: String,
    #[serde(default)]
    pub position: [f32; 3],
    #[serde(default)]
    pub rotation: [f32; 3],
    #[serde(default = "default_scale")]
    pub scale: [f32; 3],
}

fn default_scale() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}

impl Config {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let content = fs::read_to_string(path)
            .map_err(|error| format!("Failed to read config file: {error}"))?;
        toml::from_str(&content).map_err(|error| format!("Failed to parse TOML: {error}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_document_uses_complete_repository_defaults() {
        let config: Config = toml::from_str("").expect("empty TOML should use defaults");

        assert_eq!(config.render.width, 1280);
        assert_eq!(config.render.height, 720);
        assert_eq!(config.render.supersample_scale, 1);
        assert_eq!(config.render.cull_mode, CullModeConfig::Back);
        assert_eq!(config.camera.position, [0.0, 4.0, 5.0]);
        assert_eq!(config.camera.projection, ProjectionMode::Perspective);
        assert!(config.ground.enabled);
        assert_eq!(config.lights.len(), 1);
        assert_eq!(config.lights[0].kind, LightKind::Directional);
        assert_eq!(config.objects.len(), 1);
        assert_eq!(config.objects[0].path, "assets/glbs/old_rusty_car.glb");
    }

    #[test]
    fn partial_tables_preserve_struct_defaults() {
        let config: Config = toml::from_str(
            r#"
                [render]
                width = 320

                [camera]
                fov = 60.0

                [ground]
                roughness = 0.25
            "#,
        )
        .expect("partial tables should inherit defaults");

        assert_eq!(config.render.width, 320);
        assert_eq!(config.render.height, 720);
        assert_eq!(config.render.background_gradient_top, Some([0.2, 0.2, 0.3]));
        assert_eq!(config.camera.position, [0.0, 4.0, 5.0]);
        assert_eq!(config.camera.target, [0.0, 0.0, 0.0]);
        assert_eq!(config.camera.up, [0.0, 1.0, 0.0]);
        assert_eq!(config.camera.fov, 60.0);
        assert_eq!(config.ground.albedo, Some([0.8, 0.8, 0.8]));
        assert_eq!(config.ground.roughness, Some(0.25));
        assert_eq!(config.lights.len(), 1);
        assert_eq!(config.objects.len(), 1);
    }

    #[test]
    fn parses_representative_scene_configuration() {
        let source = r#"
            [render]
            width = 320
            height = 180
            supersample_scale = 2
            cull_mode = "none"

            [camera]
            position = [1.0, 2.0, 3.0]
            target = [0.0, 0.0, 0.0]
            up = [0.0, 1.0, 0.0]
            fov = 60.0
            projection = "orthographic"

            [ground]
            enabled = false

            [[lights]]
            type = "point"
            position = [2.0, 3.0, 4.0]
            color = [1.0, 0.5, 0.25]
            intensity = 8.0

            [[objects]]
            path = "fixture.glb"
            rotation = [0.0, 90.0, 0.0]
        "#;

        let config: Config = toml::from_str(source).expect("representative config should parse");

        assert_eq!(config.render.width, 320);
        assert_eq!(config.render.height, 180);
        assert_eq!(config.render.supersample_scale, 2);
        assert_eq!(config.render.cull_mode, CullModeConfig::None);
        assert_eq!(config.camera.fov, 60.0);
        assert_eq!(config.camera.projection, ProjectionMode::Orthographic);
        assert!(!config.ground.enabled);
        assert_eq!(config.lights.len(), 1);
        assert_eq!(config.lights[0].kind, LightKind::Point);
        assert_eq!(config.objects[0].scale, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn rejects_unknown_typed_values() {
        for source in [
            "[render]\ncull_mode = \"sideways\"",
            "[camera]\nprojection = \"fisheye\"",
            "[[lights]]\ntype = \"spot\"\ncolor = [1.0, 1.0, 1.0]\nintensity = 1.0",
        ] {
            assert!(toml::from_str::<Config>(source).is_err());
        }
    }

    #[test]
    fn rejects_legacy_samples_field() {
        assert!(toml::from_str::<Config>("[render]\nsamples = 4").is_err());
    }

    #[test]
    fn rejects_unknown_fields() {
        assert!(toml::from_str::<Config>("unexpected = true").is_err());
        assert!(toml::from_str::<Config>("[camera]\nunknown = 1").is_err());
        assert!(
            toml::from_str::<Config>("[[objects]]\npath = \"fixture.glb\"\nunknown = 1").is_err()
        );
    }
}
