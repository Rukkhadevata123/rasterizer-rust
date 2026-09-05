use crate::core::framebuffer::{FrameBuffer, RenderTargetError};
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("failed to determine the process working directory: {source}")]
    CurrentDirectory {
        #[source]
        source: std::io::Error,
    },
    #[error("failed to read config '{}': {source}", path.display())]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse config '{}': {source}", path.display())]
    Parse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },
    #[error("invalid config '{}': {source}", path.display())]
    Invalid {
        path: PathBuf,
        #[source]
        source: ConfigValidationError,
    },
}

#[derive(Debug, Error, PartialEq)]
pub enum ConfigValidationError {
    #[error("invalid render target: {source}")]
    RenderTarget {
        #[source]
        source: RenderTargetError,
    },
    #[error("invalid shadow map dimensions: {source}")]
    ShadowTarget {
        #[source]
        source: RenderTargetError,
    },
    #[error("{field} must be finite")]
    NonFinite { field: String },
    #[error("{field} must contain only finite values")]
    NonFiniteArray { field: String },
    #[error("{field} must be greater than zero")]
    NonPositive { field: String },
    #[error("{field} must be non-negative")]
    Negative { field: String },
    #[error("{field} must have non-zero length")]
    ZeroVector { field: String },
    #[error("camera.up must not be parallel to the viewing direction")]
    ParallelCameraUp,
    #[error("camera.fov must be greater than 0 and less than 180 degrees, got {value}")]
    FovOutOfRange { value: f32 },
    #[error("camera.far ({far}) must be greater than camera.near ({near})")]
    FarNotBeyondNear { near: f32, far: f32 },
    #[error("{field} is required for a {light_kind} light")]
    MissingLightField {
        field: String,
        light_kind: &'static str,
    },
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub render: RenderConfig,
    pub camera: CameraConfig,
    pub ground: GroundConfig,
    pub lights: Vec<LightConfig>,
    pub objects: Vec<ObjectConfig>,
    #[serde(skip)]
    pub base_dir: PathBuf,
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
                normalization: ModelNormalization::Normalize,
            }],
            base_dir: PathBuf::new(),
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

#[derive(Debug, Clone, Deserialize)]
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
    pub shadow_constant_bias: f32,
    pub shadow_slope_bias: f32,
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
            shadow_constant_bias: 0.001,
            shadow_slope_bias: 0.01,
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

#[derive(Debug, Clone, Deserialize)]
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

#[derive(Debug, Clone, Deserialize)]
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

#[derive(Debug, Clone, Deserialize)]
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

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectConfig {
    pub path: String,
    #[serde(default)]
    pub position: [f32; 3],
    #[serde(default)]
    pub rotation: [f32; 3],
    #[serde(default = "default_scale")]
    pub scale: [f32; 3],
    #[serde(default)]
    pub normalization: ModelNormalization,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ModelNormalization {
    Preserve,
    Center,
    #[default]
    Normalize,
}

fn default_scale() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}

impl Config {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let content = fs::read_to_string(path).map_err(|source| ConfigError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        let mut config: Self = toml::from_str(&content).map_err(|source| ConfigError::Parse {
            path: path.to_path_buf(),
            source,
        })?;
        let absolute_path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir()
                .map_err(|source| ConfigError::CurrentDirectory { source })?
                .join(path)
        };
        config.base_dir = absolute_path
            .parent()
            .unwrap_or_else(|| Path::new(""))
            .to_path_buf();
        config.validate().map_err(|source| ConfigError::Invalid {
            path: path.to_path_buf(),
            source,
        })?;
        Ok(config)
    }

    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    pub fn resolve_path<P: AsRef<Path>>(&self, path: P) -> PathBuf {
        let path = path.as_ref();
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.base_dir.join(path)
        }
    }

    pub fn validate(&self) -> Result<(), ConfigValidationError> {
        FrameBuffer::validate_dimensions(
            self.render.width,
            self.render.height,
            self.render.supersample_scale,
        )
        .map_err(|source| ConfigValidationError::RenderTarget { source })?;
        FrameBuffer::validate_dimensions(
            self.render.shadow_map_size,
            self.render.shadow_map_size,
            1,
        )
        .map_err(|source| ConfigValidationError::ShadowTarget { source })?;

        validate_finite("render.exposure", self.render.exposure)?;
        if self.render.exposure < 0.0 {
            return Err(ConfigValidationError::Negative {
                field: "render.exposure".to_owned(),
            });
        }
        validate_finite_array("render.ambient_light", self.render.ambient_light)?;
        validate_optional_finite_array("render.background_color", self.render.background_color)?;
        validate_optional_finite_array(
            "render.background_gradient_top",
            self.render.background_gradient_top,
        )?;
        validate_optional_finite_array(
            "render.background_gradient_bottom",
            self.render.background_gradient_bottom,
        )?;
        validate_positive("render.shadow_ortho_size", self.render.shadow_ortho_size)?;
        validate_non_negative(
            "render.shadow_constant_bias",
            self.render.shadow_constant_bias,
        )?;
        validate_non_negative("render.shadow_slope_bias", self.render.shadow_slope_bias)?;
        if self.render.pcf_kernel_size < 0 {
            return Err(ConfigValidationError::Negative {
                field: "render.pcf_kernel_size".to_owned(),
            });
        }

        validate_finite_array("camera.position", self.camera.position)?;
        validate_finite_array("camera.target", self.camera.target)?;
        validate_nonzero_vector(
            "camera viewing direction",
            vector_between(self.camera.position, self.camera.target),
        )?;
        validate_nonzero_vector("camera.up", self.camera.up)?;
        let viewing_direction = vector_between(self.camera.position, self.camera.target);
        if squared_length(cross(viewing_direction, self.camera.up)) <= f32::EPSILON {
            return Err(ConfigValidationError::ParallelCameraUp);
        }
        validate_finite("camera.fov", self.camera.fov)?;
        if !(0.0..180.0).contains(&self.camera.fov) {
            return Err(ConfigValidationError::FovOutOfRange {
                value: self.camera.fov,
            });
        }
        validate_positive("camera.ortho_height", self.camera.ortho_height)?;
        validate_positive("camera.near", self.camera.near)?;
        validate_finite("camera.far", self.camera.far)?;
        if self.camera.far <= self.camera.near {
            return Err(ConfigValidationError::FarNotBeyondNear {
                near: self.camera.near,
                far: self.camera.far,
            });
        }
        validate_non_negative("camera.speed", self.camera.speed)?;
        validate_non_negative("camera.sensitivity", self.camera.sensitivity)?;
        validate_non_negative("camera.zoom_speed", self.camera.zoom_speed)?;

        validate_positive("ground.size", self.ground.size)?;
        validate_optional_finite_array("ground.albedo", self.ground.albedo)?;
        validate_optional_finite("ground.metallic", self.ground.metallic)?;
        validate_optional_finite("ground.roughness", self.ground.roughness)?;

        for (index, light) in self.lights.iter().enumerate() {
            let prefix = format!("lights[{index}]");
            validate_optional_finite_array(&format!("{prefix}.position"), light.position)?;
            validate_optional_finite_array(&format!("{prefix}.direction"), light.direction)?;
            validate_finite_array(&format!("{prefix}.color"), light.color)?;
            validate_non_negative(&format!("{prefix}.intensity"), light.intensity)?;
            if let Some(attenuation) = light.attenuation {
                validate_finite_array(&format!("{prefix}.attenuation"), attenuation)?;
            }

            match light.kind {
                LightKind::Directional => {
                    let direction = light.direction.ok_or_else(|| {
                        ConfigValidationError::MissingLightField {
                            field: format!("{prefix}.direction"),
                            light_kind: "directional",
                        }
                    })?;
                    validate_nonzero_vector(&format!("{prefix}.direction"), direction)?;
                }
                LightKind::Point => {
                    let position =
                        light
                            .position
                            .ok_or_else(|| ConfigValidationError::MissingLightField {
                                field: format!("{prefix}.position"),
                                light_kind: "point",
                            })?;
                    validate_finite_array(&format!("{prefix}.position"), position)?;
                }
            }
        }

        for (index, object) in self.objects.iter().enumerate() {
            validate_finite_array(&format!("objects[{index}].position"), object.position)?;
            validate_finite_array(&format!("objects[{index}].rotation"), object.rotation)?;
            validate_finite_array(&format!("objects[{index}].scale"), object.scale)?;
        }

        Ok(())
    }
}

fn validate_finite(name: &str, value: f32) -> Result<(), ConfigValidationError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ConfigValidationError::NonFinite {
            field: name.to_owned(),
        })
    }
}

fn validate_positive(name: &str, value: f32) -> Result<(), ConfigValidationError> {
    validate_finite(name, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(ConfigValidationError::NonPositive {
            field: name.to_owned(),
        })
    }
}

fn validate_non_negative(name: &str, value: f32) -> Result<(), ConfigValidationError> {
    validate_finite(name, value)?;
    if value >= 0.0 {
        Ok(())
    } else {
        Err(ConfigValidationError::Negative {
            field: name.to_owned(),
        })
    }
}

fn validate_optional_finite(name: &str, value: Option<f32>) -> Result<(), ConfigValidationError> {
    if let Some(value) = value {
        validate_finite(name, value)?;
    }
    Ok(())
}

fn validate_finite_array(name: &str, values: [f32; 3]) -> Result<(), ConfigValidationError> {
    if values.into_iter().all(f32::is_finite) {
        Ok(())
    } else {
        Err(ConfigValidationError::NonFiniteArray {
            field: name.to_owned(),
        })
    }
}

fn validate_optional_finite_array(
    name: &str,
    values: Option<[f32; 3]>,
) -> Result<(), ConfigValidationError> {
    if let Some(values) = values {
        validate_finite_array(name, values)?;
    }
    Ok(())
}

fn validate_nonzero_vector(name: &str, value: [f32; 3]) -> Result<(), ConfigValidationError> {
    validate_finite_array(name, value)?;
    if squared_length(value) > f32::EPSILON {
        Ok(())
    } else {
        Err(ConfigValidationError::ZeroVector {
            field: name.to_owned(),
        })
    }
}

fn vector_between(from: [f32; 3], to: [f32; 3]) -> [f32; 3] {
    [to[0] - from[0], to[1] - from[1], to[2] - from[2]]
}

fn squared_length(value: [f32; 3]) -> f32 {
    value[0] * value[0] + value[1] * value[1] + value[2] * value[2]
}

fn cross(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
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
        assert_eq!(
            config.objects[0].normalization,
            ModelNormalization::Normalize
        );
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
        assert_eq!(
            config.objects[0].normalization,
            ModelNormalization::Normalize
        );
    }

    #[test]
    fn parses_explicit_model_normalization_policies() {
        for (value, expected) in [
            ("preserve", ModelNormalization::Preserve),
            ("center", ModelNormalization::Center),
            ("normalize", ModelNormalization::Normalize),
        ] {
            let source =
                format!("[[objects]]\npath = \"fixture.glb\"\nnormalization = \"{value}\"");
            let config: Config = toml::from_str(&source).expect("normalization should parse");
            assert_eq!(config.objects[0].normalization, expected);
        }

        assert!(
            toml::from_str::<Config>(
                "[[objects]]\npath = \"fixture.glb\"\nnormalization = \"stretch\""
            )
            .is_err()
        );
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
    fn rejects_unknown_fields() {
        assert!(toml::from_str::<Config>("unexpected = true").is_err());
        assert!(toml::from_str::<Config>("base_dir = \"override\"").is_err());
        assert!(toml::from_str::<Config>("[render]\nsamples = 4").is_err());
        assert!(toml::from_str::<Config>("[camera]\nunknown = 1").is_err());
        assert!(
            toml::from_str::<Config>("[[objects]]\npath = \"fixture.glb\"\nunknown = 1").is_err()
        );
    }

    #[test]
    fn rejects_invalid_dimensions_before_allocation() {
        let mut config = Config::default();
        config.render.width = 0;
        assert_eq!(
            config.validate().unwrap_err(),
            ConfigValidationError::RenderTarget {
                source: RenderTargetError::ZeroDimensions {
                    width: 0,
                    height: config.render.height,
                },
            }
        );

        config.render.width = usize::MAX;
        config.render.supersample_scale = 2;
        assert_eq!(
            config.validate().unwrap_err(),
            ConfigValidationError::RenderTarget {
                source: RenderTargetError::SupersampledWidthOverflow,
            }
        );

        config.render.width = usize::MAX / 2 + 1;
        config.render.height = 2;
        config.render.supersample_scale = 1;
        assert_eq!(
            config.validate().unwrap_err(),
            ConfigValidationError::RenderTarget {
                source: RenderTargetError::SampleCountOverflow,
            }
        );
    }

    #[test]
    fn rejects_invalid_camera_geometry() {
        let mut config = Config::default();
        config.camera.target = config.camera.position;
        assert_eq!(
            config.validate().unwrap_err(),
            ConfigValidationError::ZeroVector {
                field: "camera viewing direction".to_owned(),
            }
        );

        config.camera.target = [0.0, 0.0, 0.0];
        config.camera.up = [0.0, 0.0, 0.0];
        assert_eq!(
            config.validate().unwrap_err(),
            ConfigValidationError::ZeroVector {
                field: "camera.up".to_owned(),
            }
        );

        config.camera.up = [0.0, 1.0, 0.0];
        config.camera.near = 10.0;
        config.camera.far = 1.0;
        assert_eq!(
            config.validate().unwrap_err(),
            ConfigValidationError::FarNotBeyondNear {
                near: 10.0,
                far: 1.0,
            }
        );
    }

    #[test]
    fn rejects_non_finite_transforms_and_invalid_light_vectors() {
        let mut config = Config::default();
        config.objects[0].scale[1] = f32::NAN;
        assert!(matches!(
            config.validate().unwrap_err(),
            ConfigValidationError::NonFiniteArray { field }
                if field == "objects[0].scale"
        ));

        config.objects[0].scale = [1.0, 1.0, 1.0];
        config.lights[0].direction = Some([0.0, 0.0, 0.0]);
        assert_eq!(
            config.validate().unwrap_err(),
            ConfigValidationError::ZeroVector {
                field: "lights[0].direction".to_owned(),
            }
        );

        config.lights[0].direction = Some([0.0, -1.0, 0.0]);
        config.lights[0].position = Some([f32::INFINITY, 0.0, 0.0]);
        assert!(matches!(
            config.validate().unwrap_err(),
            ConfigValidationError::NonFiniteArray { field }
                if field == "lights[0].position"
        ));
    }

    #[test]
    fn resolves_relative_paths_against_config_directory() {
        let config_path = std::env::temp_dir()
            .join("rasterizer-config-path-tests")
            .join("nested")
            .join("scene.toml");
        std::fs::create_dir_all(config_path.parent().unwrap())
            .expect("fixture directory should be created");
        std::fs::write(&config_path, "objects = []").expect("fixture config should be written");

        let config = Config::load(&config_path).expect("fixture config should load");
        assert_eq!(config.base_dir(), config_path.parent().unwrap());
        assert_eq!(
            config.resolve_path("assets/model.glb"),
            config_path.parent().unwrap().join("assets/model.glb")
        );

        let absolute = std::env::temp_dir().join("absolute-model.glb");
        assert_eq!(config.resolve_path(&absolute), absolute);
    }
}
