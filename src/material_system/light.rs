use crate::io::render_settings::parse_vec3;
use clap::ValueEnum;
use nalgebra::{Point3, Vector3};

/// 光照预设模式
#[derive(Debug, Clone, Default, PartialEq, Eq, ValueEnum)]
pub enum LightingPreset {
    #[default]
    SingleDirectional,
    ThreeDirectional,
    MixedComplete,
    None,
}

/// 🔥 **统一的光源结构** - 简化版本
#[derive(Debug, Clone)]
pub enum Light {
    Directional {
        // 配置字段 (用于GUI控制)
        enabled: bool,
        direction_str: String, // "x,y,z" 格式，用于GUI编辑
        color_str: String,     // "r,g,b" 格式，用于GUI编辑
        intensity: f32,

        // 运行时字段 (用于渲染计算，从配置字段解析)
        direction: Vector3<f32>, // 解析后的方向向量
        color: Vector3<f32>,     // 解析后的颜色向量
    },
    Point {
        // 配置字段 (用于GUI控制)
        enabled: bool,
        position_str: String, // "x,y,z" 格式，用于GUI编辑
        color_str: String,    // "r,g,b" 格式，用于GUI编辑
        intensity: f32,
        constant_attenuation: f32,
        linear_attenuation: f32,
        quadratic_attenuation: f32,

        // 运行时字段 (用于渲染计算，从配置字段解析)
        position: Point3<f32>, // 解析后的位置
        color: Vector3<f32>,   // 解析后的颜色向量
    },
}

impl Light {
    /// 🔥 **创建方向光** - 同时设置配置和运行时字段
    pub fn directional(direction: Vector3<f32>, color: Vector3<f32>, intensity: f32) -> Self {
        let direction_normalized = direction.normalize();
        Self::Directional {
            enabled: true,
            direction_str: format!(
                "{},{},{}",
                direction_normalized.x, direction_normalized.y, direction_normalized.z
            ),
            color_str: format!("{},{},{}", color.x, color.y, color.z),
            intensity,
            direction: direction_normalized,
            color,
        }
    }

    /// 🔥 **创建点光源** - 同时设置配置和运行时字段
    pub fn point(
        position: Point3<f32>,
        color: Vector3<f32>,
        intensity: f32,
        attenuation: Option<(f32, f32, f32)>,
    ) -> Self {
        let (constant, linear, quadratic) = attenuation.unwrap_or((1.0, 0.09, 0.032));
        Self::Point {
            enabled: true,
            position_str: format!("{},{},{}", position.x, position.y, position.z),
            color_str: format!("{},{},{}", color.x, color.y, color.z),
            intensity,
            constant_attenuation: constant,
            linear_attenuation: linear,
            quadratic_attenuation: quadratic,
            position,
            color,
        }
    }

    /// 🔥 **更新运行时字段** - 从字符串配置重新解析
    pub fn update_runtime_fields(&mut self) -> Result<(), String> {
        match self {
            Self::Directional {
                direction_str,
                color_str,
                direction,
                color,
                ..
            } => {
                *direction = parse_vec3(direction_str)?.normalize();
                *color = parse_vec3(color_str)?;
            }
            Self::Point {
                position_str,
                color_str,
                position,
                color,
                ..
            } => {
                *position = crate::io::render_settings::parse_point3(position_str)?;
                *color = parse_vec3(color_str)?;
            }
        }
        Ok(())
    }

    /// 获取光源方向（用于渲染）
    pub fn get_direction(&self, point: &Point3<f32>) -> Vector3<f32> {
        match self {
            Self::Directional { direction, .. } => -direction,
            Self::Point { position, .. } => (position - point).normalize(),
        }
    }

    /// 获取光源强度（用于渲染）
    pub fn get_intensity(&self, point: &Point3<f32>) -> Vector3<f32> {
        match self {
            Self::Directional {
                color,
                intensity,
                enabled,
                ..
            } => {
                if *enabled {
                    color * *intensity
                } else {
                    Vector3::zeros()
                }
            }
            Self::Point {
                position,
                color,
                intensity,
                constant_attenuation,
                linear_attenuation,
                quadratic_attenuation,
                enabled,
                ..
            } => {
                if *enabled {
                    let distance = (position - point).magnitude();
                    let attenuation_factor = 1.0
                        / (constant_attenuation
                            + linear_attenuation * distance
                            + quadratic_attenuation * distance * distance);
                    color * *intensity * attenuation_factor
                } else {
                    Vector3::zeros()
                }
            }
        }
    }
}

/// 🔥 **简化的光源管理器**
pub struct LightManager;

impl LightManager {
    /// 🔥 **创建预设光源** - 返回统一的Light数组
    pub fn create_preset_lights(preset: &LightingPreset, main_intensity: f32) -> Vec<Light> {
        match preset {
            LightingPreset::SingleDirectional => {
                vec![Light::directional(
                    Vector3::new(0.0, -1.0, -1.0),
                    Vector3::new(1.0, 1.0, 1.0),
                    main_intensity,
                )]
            }
            LightingPreset::ThreeDirectional => {
                vec![
                    Light::directional(
                        Vector3::new(0.0, -1.0, -1.0),
                        Vector3::new(1.0, 1.0, 1.0),
                        main_intensity * 0.7,
                    ),
                    Light::directional(
                        Vector3::new(-1.0, -0.5, 0.2),
                        Vector3::new(0.9, 0.9, 1.0),
                        main_intensity * 0.5,
                    ),
                    Light::directional(
                        Vector3::new(1.0, -0.5, 0.2),
                        Vector3::new(1.0, 0.9, 0.8),
                        main_intensity * 0.3,
                    ),
                ]
            }
            LightingPreset::MixedComplete => {
                let mut lights = vec![Light::directional(
                    Vector3::new(0.0, -1.0, -1.0),
                    Vector3::new(1.0, 1.0, 1.0),
                    main_intensity * 0.6,
                )];

                let point_configs = [
                    (Point3::new(2.0, 3.0, 2.0), Vector3::new(1.0, 0.8, 0.6)),
                    (Point3::new(-2.0, 3.0, 2.0), Vector3::new(0.6, 0.8, 1.0)),
                    (Point3::new(2.0, 3.0, -2.0), Vector3::new(0.8, 1.0, 0.8)),
                    (Point3::new(-2.0, 3.0, -2.0), Vector3::new(1.0, 0.8, 1.0)),
                ];

                for (pos, color) in &point_configs {
                    lights.push(Light::point(
                        *pos,
                        *color,
                        main_intensity * 0.5,
                        Some((1.0, 0.09, 0.032)),
                    ));
                }

                lights
            }
            LightingPreset::None => Vec::new(),
        }
    }

    /// 🔥 **确保有光源** - 如果为空则创建默认光源
    pub fn ensure_lights_exist(lights: &mut Vec<Light>, use_lighting: bool, main_intensity: f32) {
        if !use_lighting {
            lights.clear();
            return;
        }

        if lights.is_empty() {
            lights.push(Light::directional(
                Vector3::new(0.0, -1.0, -1.0),
                Vector3::new(1.0, 1.0, 1.0),
                main_intensity * 0.8,
            ));
        }
    }
}
