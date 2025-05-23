use crate::io::args::{parse_point3, parse_vec3};
use clap::ValueEnum;
use nalgebra::{Point3, Vector3};

/// 光照预设模式
#[derive(Debug, Clone, Default, PartialEq, Eq, ValueEnum)]
pub enum LightingPreset {
    /// 单一方向光源（默认）
    #[default]
    SingleDirectional,
    /// 三面方向光源（更均匀的照明）
    ThreeDirectional,
    /// 一个方向光源加四个点光源（更生动的照明）
    MixedComplete,
    /// 无光照
    None,
}

/// 光源类型
#[derive(Debug, Clone)]
pub enum Light {
    /// 定向光，direction表示朝向光源的方向
    Directional {
        direction: Vector3<f32>,
        color: Vector3<f32>,
        intensity: f32,
    },
    /// 点光源，带位置和衰减因子
    Point {
        position: Point3<f32>,
        color: Vector3<f32>,
        intensity: f32,
        /// 衰减因子: (常数项, 一次项, 二次项)
        attenuation: (f32, f32, f32),
    },
}

impl Light {
    /// 创建定向光源
    pub fn directional(direction: Vector3<f32>, color: Vector3<f32>, intensity: f32) -> Self {
        Light::Directional {
            direction: direction.normalize(),
            color,
            intensity,
        }
    }

    /// 创建点光源
    pub fn point(
        position: Point3<f32>,
        color: Vector3<f32>,
        intensity: f32,
        attenuation: Option<(f32, f32, f32)>,
    ) -> Self {
        Light::Point {
            position,
            color,
            intensity,
            attenuation: attenuation.unwrap_or((1.0, 0.09, 0.032)),
        }
    }

    /// 从字符串创建方向光源
    pub fn directional_from_str(
        direction: &str,
        color: &str,
        intensity: f32,
    ) -> Result<Self, String> {
        let dir = parse_vec3(direction)?.normalize();
        let col = parse_vec3(color)?;
        Ok(Self::directional(dir, col, intensity))
    }

    /// 从字符串创建点光源
    pub fn point_from_str(
        position: &str,
        color: &str,
        intensity: f32,
        constant: f32,
        linear: f32,
        quadratic: f32,
    ) -> Result<Self, String> {
        let pos = parse_point3(position)?;
        let col = parse_vec3(color)?;
        Ok(Self::point(
            pos,
            col,
            intensity,
            Some((constant, linear, quadratic)),
        ))
    }

    /// 获取光源的方向（从表面点到光源）
    pub fn get_direction(&self, point: &Point3<f32>) -> Vector3<f32> {
        match self {
            Light::Directional { direction, .. } => -direction,
            Light::Point { position, .. } => (position - point).normalize(),
        }
    }

    /// 计算光源在给定点的强度（考虑衰减和颜色）
    pub fn get_intensity(&self, point: &Point3<f32>) -> Vector3<f32> {
        match self {
            Light::Directional {
                color, intensity, ..
            } => Vector3::new(
                color.x * intensity,
                color.y * intensity,
                color.z * intensity,
            ),
            Light::Point {
                position,
                color,
                intensity,
                attenuation,
            } => {
                let distance = (position - point).magnitude();
                let (constant, linear, quadratic) = *attenuation;
                let attenuation_factor =
                    1.0 / (constant + linear * distance + quadratic * distance * distance);

                Vector3::new(
                    color.x * intensity * attenuation_factor,
                    color.y * intensity * attenuation_factor,
                    color.z * intensity * attenuation_factor,
                )
            }
        }
    }
}

/// 单个方向光源的配置（仅用于UI和命令行交互）
#[derive(Debug, Clone)]
pub struct DirectionalLightConfig {
    pub enabled: bool,
    pub direction: String,
    pub color: String,
    pub intensity: f32,
}

impl Default for DirectionalLightConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            direction: "0,-1,-1".to_string(),
            color: "1.0,1.0,1.0".to_string(),
            intensity: 0.8,
        }
    }
}

impl DirectionalLightConfig {
    /// 转换为Light实例
    pub fn to_light(&self) -> Result<Light, String> {
        if !self.enabled {
            return Err("光源未启用".to_string());
        }
        Light::directional_from_str(&self.direction, &self.color, self.intensity)
    }
}

/// 单个点光源的配置（仅用于UI和命令行交互）
#[derive(Debug, Clone)]
pub struct PointLightConfig {
    pub enabled: bool,
    pub position: String,
    pub color: String,
    pub intensity: f32,
    pub constant_attenuation: f32,
    pub linear_attenuation: f32,
    pub quadratic_attenuation: f32,
}

impl Default for PointLightConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            position: "0,5,5".to_string(),
            color: "1.0,1.0,1.0".to_string(),
            intensity: 1.0,
            constant_attenuation: 1.0,
            linear_attenuation: 0.09,
            quadratic_attenuation: 0.032,
        }
    }
}

impl PointLightConfig {
    /// 转换为Light实例
    pub fn to_light(&self) -> Result<Light, String> {
        if !self.enabled {
            return Err("光源未启用".to_string());
        }
        Light::point_from_str(
            &self.position,
            &self.color,
            self.intensity,
            self.constant_attenuation,
            self.linear_attenuation,
            self.quadratic_attenuation,
        )
    }
}

/// 根据预设创建光源
pub fn create_lights_from_preset(preset: LightingPreset, main_intensity: f32) -> Vec<Light> {
    let mut lights = Vec::new();

    match preset {
        LightingPreset::SingleDirectional => {
            // 添加一个默认的方向光源
            if let Ok(light) = Light::directional_from_str("0,-1,-1", "1.0,1.0,1.0", main_intensity)
            {
                lights.push(light);
            }
        }
        LightingPreset::ThreeDirectional => {
            // 添加三个方向光源，从不同角度照亮场景
            if let Ok(light) =
                Light::directional_from_str("0,-1,-1", "1.0,1.0,1.0", main_intensity * 0.7)
            {
                lights.push(light);
            }

            if let Ok(light) =
                Light::directional_from_str("-1,-0.5,0.2", "0.9,0.9,1.0", main_intensity * 0.5)
            {
                lights.push(light);
            }

            if let Ok(light) =
                Light::directional_from_str("1,-0.5,0.2", "1.0,0.9,0.8", main_intensity * 0.3)
            {
                lights.push(light);
            }
        }
        LightingPreset::MixedComplete => {
            // 添加一个主方向光源
            if let Ok(light) =
                Light::directional_from_str("0,-1,-1", "1.0,1.0,1.0", main_intensity * 0.6)
            {
                lights.push(light);
            }

            // 添加四个点光源，创建更有趣的照明效果
            let point_configs = [
                ("2,3,2", "1.0,0.8,0.6"),   // 暖色调
                ("-2,3,2", "0.6,0.8,1.0"),  // 冷色调
                ("2,3,-2", "0.8,1.0,0.8"),  // 绿色调
                ("-2,3,-2", "1.0,0.8,1.0"), // 紫色调
            ];

            for (pos, color) in &point_configs {
                if let Ok(light) =
                    Light::point_from_str(pos, color, main_intensity * 0.5, 1.0, 0.09, 0.032)
                {
                    lights.push(light);
                }
            }
        }
        LightingPreset::None => {
            // 不添加任何光源
        }
    }

    lights
}

/// 从配置列表创建光源集合
pub fn create_lights_from_configs(
    directional_lights: &[DirectionalLightConfig],
    point_lights: &[PointLightConfig],
) -> Vec<Light> {
    let mut lights = Vec::new();

    // 添加方向光源
    for (i, light) in directional_lights.iter().enumerate() {
        if light.enabled {
            match light.to_light() {
                Ok(l) => lights.push(l),
                Err(e) => eprintln!("方向光 #{} 配置错误: {}", i + 1, e),
            }
        }
    }

    // 添加点光源
    for (i, light) in point_lights.iter().enumerate() {
        if light.enabled {
            match light.to_light() {
                Ok(l) => lights.push(l),
                Err(e) => eprintln!("点光源 #{} 配置错误: {}", i + 1, e),
            }
        }
    }

    lights
}
