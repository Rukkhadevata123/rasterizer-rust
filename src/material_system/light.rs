use nalgebra::{Point3, Vector3};

/// 光源类型
#[derive(Debug, Clone, Copy)]
pub enum Light {
    /// 定向光，direction表示朝向光源的方向
    Directional {
        direction: Vector3<f32>,
        intensity: Vector3<f32>,
    },
    /// 点光源，带位置和衰减因子
    Point {
        position: Point3<f32>,
        intensity: Vector3<f32>,
        /// 衰减因子: (常数项, 一次项, 二次项)
        attenuation: (f32, f32, f32),
    },
    // 移除 Ambient 变体，环境光将作为场景的基础属性
}

impl Light {
    /// 创建定向光源
    pub fn directional(direction: Vector3<f32>, intensity: Vector3<f32>) -> Self {
        Light::Directional {
            direction: direction.normalize(),
            intensity,
        }
    }

    /// 创建点光源
    pub fn point(
        position: Point3<f32>,
        intensity: Vector3<f32>,
        attenuation: Option<(f32, f32, f32)>,
    ) -> Self {
        Light::Point {
            position,
            intensity,
            attenuation: attenuation.unwrap_or((1.0, 0.1, 0.01)), // 默认衰减因子
        }
    }

    /// 获取光源的方向（从表面点到光源）
    pub fn get_direction(&self, point: &Point3<f32>) -> Vector3<f32> {
        match self {
            Light::Directional { direction, .. } => -direction,
            Light::Point { position, .. } => (position - point).normalize(),
        }
    }

    /// 计算光源在给定点的强度（考虑衰减）
    pub fn get_intensity(&self, point: &Point3<f32>) -> Vector3<f32> {
        match self {
            Light::Directional { intensity, .. } => *intensity,
            Light::Point {
                position,
                intensity,
                attenuation,
            } => {
                let distance = (position - point).magnitude();
                let (constant, linear, quadratic) = *attenuation;
                let attenuation_factor =
                    1.0 / (constant + linear * distance + quadratic * distance * distance);

                Vector3::new(
                    intensity.x * attenuation_factor,
                    intensity.y * attenuation_factor,
                    intensity.z * attenuation_factor,
                )
            }
        }
    }
}
