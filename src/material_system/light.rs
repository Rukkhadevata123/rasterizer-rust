use crate::io::render_settings::parse_vec3;
use nalgebra::{Point3, Vector3};

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
