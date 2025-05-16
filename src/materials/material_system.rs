use crate::utils::model_types::{Material, MaterialMode};
use nalgebra::{Point3, Vector3};
use std::fmt::Debug;

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
    /// 环境光
    Ambient(Vector3<f32>), // 表示环境光强度
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
            attenuation: attenuation.unwrap_or((1.0, 0.09, 0.032)), // 默认衰减因子
        }
    }

    /// 创建环境光
    pub fn ambient(intensity: Vector3<f32>) -> Self {
        Light::Ambient(intensity)
    }

    /// 计算光源在给定点的方向（指向光源）
    pub fn get_direction(&self, point: &Point3<f32>) -> Vector3<f32> {
        match self {
            Light::Directional { direction, .. } => {
                // 对于定向光，返回朝向光源的方向（确保已归一化）
                *direction
            }
            Light::Point { position, .. } => {
                // 对于点光源，方向是从点到光源的向量
                let to_light = position - point;
                let distance = to_light.norm();
                if distance < 1e-6 {
                    Vector3::z() // 默认方向，避免除零
                } else {
                    to_light / distance // 归一化方向
                }
            }
            Light::Ambient(_) => {
                // 环境光没有明确方向，返回默认上方向
                Vector3::y()
            }
        }
    }

    /// 计算光源在给定点的强度（考虑衰减）
    pub fn get_intensity(&self, point: &Point3<f32>) -> Vector3<f32> {
        match self {
            Light::Directional { intensity, .. } => {
                // 定向光强度不随距离衰减
                *intensity
            }
            Light::Point {
                position,
                intensity,
                attenuation,
            } => {
                // 点光源强度随距离衰减
                let distance = (position - point).norm();
                let (constant, linear, quadratic) = *attenuation;
                let attenuation_factor =
                    1.0 / (constant + linear * distance + quadratic * distance * distance);

                // 应用衰减
                Vector3::new(
                    intensity.x * attenuation_factor,
                    intensity.y * attenuation_factor,
                    intensity.z * attenuation_factor,
                )
            }
            Light::Ambient(intensity) => {
                // 环境光强度不变
                *intensity
            }
        }
    }
}

/// 材质视图 - 作为统一材质数据的不同解释器
#[derive(Debug, Clone)]
pub enum MaterialView<'a> {
    /// Blinn-Phong着色模型视图
    BlinnPhong(&'a Material),
    /// 基于物理的渲染(PBR)视图
    PBR(&'a Material),
}

impl<'a> MaterialView<'a> {
    /// 通过材质和模式创建材质视图
    pub fn from_material(material: &'a Material, mode: MaterialMode) -> Self {
        match mode {
            MaterialMode::BlinnPhong => MaterialView::BlinnPhong(material),
            MaterialMode::PBR => MaterialView::PBR(material),
        }
    }

    /// 计算材质对光照的响应
    pub fn compute_response(
        &self,
        light_dir: &Vector3<f32>,
        view_dir: &Vector3<f32>,
        normal: &Vector3<f32>,
    ) -> Vector3<f32> {
        match self {
            MaterialView::BlinnPhong(material) => {
                // 使用Blinn-Phong着色模型

                // 漫反射部分
                let n_dot_l = normal.dot(light_dir).max(0.0);
                let diffuse = material.diffuse * n_dot_l;

                // 高光部分 (Blinn-Phong)
                let mut specular = Vector3::zeros();
                if n_dot_l > 0.0 {
                    let halfway_dir = (light_dir + view_dir).normalize();
                    let n_dot_h = normal.dot(&halfway_dir).max(0.0);
                    let spec_intensity = n_dot_h.powf(material.shininess);
                    specular = material.specular * spec_intensity;
                }

                // 返回漫反射和镜面反射的总和
                diffuse + specular
            }
            MaterialView::PBR(material) => {
                use crate::materials::material_system::pbr_functions::*;

                // 从材质获取基本属性
                let base_color = material.base_color;
                let metallic = material.metallic;
                let roughness = material.roughness;
                let ao = material.ambient_occlusion;

                // 使用原始法线
                let n = *normal;

                // 确保所有向量都是单位向量
                let l = light_dir;
                let v = view_dir;

                // 半程向量
                let h = (l + v).normalize();

                // 各种点积
                let n_dot_l = n.dot(l).max(0.0);
                let n_dot_v = n.dot(v).max(0.0);
                let n_dot_h = n.dot(&h).max(0.0);
                let h_dot_v = h.dot(v).max(0.0);

                if n_dot_l <= 0.0 {
                    // 添加发光贡献，即使表面不接收直接光照
                    return material.emissive;
                }

                // 从金属度计算基础反射率F0
                // 非金属F0一般为0.04，金属则使用基础色
                let f0_dielectric = Vector3::new(0.04, 0.04, 0.04);
                let f0 = f0_dielectric.lerp(&base_color, metallic);

                // Cook-Torrance BRDF
                let d = distribution_ggx(n_dot_h, roughness); // 法线分布项
                let f = fresnel_schlick(h_dot_v, f0); // 菲涅耳项
                let g = geometry_smith(n_dot_v, n_dot_l, roughness); // 几何项

                // Cook-Torrance 镜面反射项
                let numerator = d * g * f;
                let denominator = 4.0 * n_dot_v * n_dot_l;
                let specular = numerator / denominator.max(0.001f32);

                // 计算漫反射项（对金属无漫反射）
                let k_s = f; // 镜面反射比例由菲涅耳决定
                // 为每个分量分别计算
                let one = Vector3::new(1.0, 1.0, 1.0);
                let k_d_components = Vector3::new(
                    (one[0] - k_s[0]) * (1.0 - metallic),
                    (one[1] - k_s[1]) * (1.0 - metallic),
                    (one[2] - k_s[2]) * (1.0 - metallic),
                );

                // 使用组件乘法计算漫反射
                let diffuse = Vector3::new(
                    k_d_components[0] * base_color[0],
                    k_d_components[1] * base_color[1],
                    k_d_components[2] * base_color[2],
                ) / std::f32::consts::PI;

                // 合并漫反射、镜面反射和环境光遮蔽
                let brdf_result = Vector3::new(
                    (diffuse[0] + specular[0]) * n_dot_l * ao,
                    (diffuse[1] + specular[1]) * n_dot_l * ao,
                    (diffuse[2] + specular[2]) * n_dot_l * ao,
                );

                // 添加发光贡献
                brdf_result + material.emissive
            }
        }
    }

    /// 获取环境光颜色
    pub fn get_ambient_color(&self) -> Vector3<f32> {
        match self {
            MaterialView::BlinnPhong(material) => material.ambient,
            MaterialView::PBR(material) => {
                // 简化处理：基础色 * 环境光遮蔽作为环境色
                material.base_color * material.ambient_occlusion * (1.0 - material.metallic)
            }
        }
    }
}

/// PBR材质函数库
pub mod pbr_functions {
    use nalgebra::Vector3;

    /// 计算GGX法线分布函数
    pub fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
        let alpha = roughness * roughness;
        let alpha2 = alpha * alpha;
        let n_dot_h2 = n_dot_h * n_dot_h;

        let numerator = alpha2;
        let denominator = std::f32::consts::PI * (n_dot_h2 * (alpha2 - 1.0) + 1.0).powi(2);
        numerator / denominator.max(0.001) // 避免除零
    }

    /// 计算Schlick-GGX几何函数
    fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
        let k = (roughness + 1.0).powi(2) / 8.0;
        let denominator = n_dot_v * (1.0 - k) + k;
        n_dot_v / denominator.max(0.001) // 避免除零
    }

    /// 计算Smith联合几何函数
    pub fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
        let ggx1 = geometry_schlick_ggx(n_dot_v, roughness);
        let ggx2 = geometry_schlick_ggx(n_dot_l, roughness);
        ggx1 * ggx2
    }

    /// 计算Schlick菲涅耳近似
    pub fn fresnel_schlick(h_dot_v: f32, f0: Vector3<f32>) -> Vector3<f32> {
        let power = (-5.55473 * h_dot_v - 6.98316) * h_dot_v;
        f0 + (Vector3::new(1.0, 1.0, 1.0) - f0) * 2.0_f32.powf(power)
    }
}
