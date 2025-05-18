use crate::materials::model_types::Material;
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

/// 材质视图 - 作为统一材质数据的不同解释器
#[derive(Debug, Clone)]
pub enum MaterialView<'a> {
    /// Blinn-Phong着色模型视图
    BlinnPhong(&'a Material),
    /// 基于物理的渲染(PBR)视图
    #[allow(clippy::upper_case_acronyms)]
    PBR(&'a Material),
}

impl MaterialView<'_> {
    /// 计算材质对光照的响应
    pub fn compute_response(
        &self,
        light_dir: &Vector3<f32>,
        view_dir: &Vector3<f32>,
        normal: &Vector3<f32>,
    ) -> Vector3<f32> {
        match self {
            MaterialView::BlinnPhong(material) => {
                // --- Blinn-Phong 着色模型 ---

                // 计算光照方向与法线的夹角余弦
                let n_dot_l = normal.dot(light_dir).max(0.0);

                // 如果表面背向光源，只返回自发光
                if n_dot_l <= 0.0 {
                    return material.emissive;
                }

                // 漫反射部分 (Diffuse)
                // diffuse = kd * albedo * cos(θ)
                let diffuse = material.diffuse() * n_dot_l;

                // 高光部分 (Specular - Blinn-Phong)
                // 计算半程向量 (半角向量)
                let halfway_dir = (light_dir + view_dir).normalize();
                // 计算法线与半程向量的夹角余弦
                let n_dot_h = normal.dot(&halfway_dir).max(0.0);
                // 使用材质的光泽度(shininess)计算镜面反射强度
                let spec_intensity = n_dot_h.powf(material.shininess);
                let specular = material.specular * spec_intensity;

                // 返回漫反射、镜面反射和自发光的总和
                diffuse + specular + material.emissive
            }
            MaterialView::PBR(material) => {
                use crate::materials::material_system::pbr_functions::*;

                // --- 基于物理的渲染 (PBR) ---

                // 获取材质的基本属性
                let base_color = material.base_color();
                let metallic = material.metallic; // 金属度 (0=非金属, 1=金属)
                let roughness = material.roughness; // 粗糙度 (0=光滑, 1=粗糙)
                let ao = material.ambient_occlusion; // 环境光遮蔽

                // 使用原始法线和方向向量
                let n = *normal; // 表面法线
                let l = *light_dir; // 光照方向
                let v = *view_dir; // 视线方向
                let h = (l + v).normalize(); // 半程向量

                // 计算各种点积（用于后续计算）
                let n_dot_l = n.dot(&l).max(0.0); // 光照角度因子
                let n_dot_v = n.dot(&v).max(0.0); // 视角因子
                let n_dot_h = n.dot(&h).max(0.0); // 用于法线分布函数
                let h_dot_v = h.dot(&v).max(0.0); // 用于菲涅耳方程

                // 如果表面背向光源，只返回自发光
                if n_dot_l <= 0.0 {
                    return material.emissive;
                }

                // 计算基础反射率F0
                // 对于非金属材质，F0通常为0.04
                // 对于金属材质，F0等于基础颜色
                let f0_dielectric = Vector3::new(0.04, 0.04, 0.04);
                let f0 = f0_dielectric.lerp(&base_color, metallic);

                // --- Cook-Torrance BRDF 计算 ---
                // 法线分布函数 (D) - 微平面朝向分布
                let d = distribution_ggx(n_dot_h, roughness);

                // 菲涅耳项 (F) - 表面反射率随视角变化
                let f = fresnel_schlick(h_dot_v, f0);

                // 几何项 (G) - 微表面自遮挡
                let g = geometry_smith(n_dot_v, n_dot_l, roughness);

                // 计算镜面反射项 (Cook-Torrance BRDF)
                let numerator = d * g * f;
                let denominator = 4.0 * n_dot_v * n_dot_l;
                let specular = numerator / denominator.max(0.001f32);

                // 计算漫反射项 - 金属材质没有漫反射
                // kd = (1 - ks) * (1 - metallic)
                let k_s = f; // 镜面反射比例由菲涅耳项决定
                let one = Vector3::new(1.0, 1.0, 1.0);

                // 逐分量计算漫反射系数
                let k_d_components = Vector3::new(
                    (one[0] - k_s[0]) * (1.0 - metallic),
                    (one[1] - k_s[1]) * (1.0 - metallic),
                    (one[2] - k_s[2]) * (1.0 - metallic),
                );

                // 计算最终漫反射颜色
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

                // 添加自发光贡献
                brdf_result + material.emissive
            }
        }
    }
}

/// PBR材质函数库
pub mod pbr_functions {
    use nalgebra::Vector3;

    /// 正态分布函数 (GGX/Trowbridge-Reitz)
    /// 描述微平面的法线分布
    pub fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
        let alpha = roughness * roughness;
        let alpha2 = alpha * alpha;
        let n_dot_h2 = n_dot_h * n_dot_h;

        let numerator = alpha2;
        let denominator = n_dot_h2 * (alpha2 - 1.0) + 1.0;
        let denominator = std::f32::consts::PI * denominator * denominator;

        numerator / denominator.max(0.0001)
    }

    /// 几何函数 (Smith's Schlick-GGX)
    /// 描述微观几何遮挡
    pub fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
        let r = roughness + 1.0;
        let k = (r * r) / 8.0;
        let denominator = n_dot_v * (1.0 - k) + k;

        n_dot_v / denominator.max(0.0001)
    }

    /// 组合几何函数
    pub fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
        let ggx1 = geometry_schlick_ggx(n_dot_v, roughness);
        let ggx2 = geometry_schlick_ggx(n_dot_l, roughness);

        ggx1 * ggx2
    }

    /// 菲涅耳方程 (Schlick近似)
    /// 决定表面反射率随视角变化
    pub fn fresnel_schlick(cos_theta: f32, f0: Vector3<f32>) -> Vector3<f32> {
        let one_minus_cos_theta = 1.0 - cos_theta;
        let one_minus_cos_theta2 = one_minus_cos_theta * one_minus_cos_theta;
        let one_minus_cos_theta5 =
            one_minus_cos_theta2 * one_minus_cos_theta2 * one_minus_cos_theta;
        f0 + (Vector3::new(1.0, 1.0, 1.0) - f0) * one_minus_cos_theta5
    }
}
