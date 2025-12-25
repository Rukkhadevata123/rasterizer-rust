use crate::io::render_settings::{RenderSettings, parse_vec3};
use crate::material_system::texture::Texture;
use log::warn;
use nalgebra::{Point3, Vector2, Vector3};
use std::fmt::Debug;

#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    pub position: Point3<f32>,
    pub normal: Vector3<f32>,
    pub texcoord: Vector2<f32>,
}

/// 材质类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterialType {
    Phong,
    #[allow(clippy::upper_case_acronyms)]
    PBR,
}

/// 材质结构体，统一包含所有参数
#[derive(Debug, Clone)]
pub struct Material {
    pub material_type: MaterialType, // 材质类型
    pub base_color: Vector3<f32>,    // 基础色（PBR/Phong通用）
    pub alpha: f32,                  // 透明度
    pub texture: Option<Texture>,    // 纹理资源

    // ===== PBR参数 =====
    pub metallic: f32,
    pub roughness: f32,
    pub ambient_occlusion: f32,

    // ===== Phong参数 =====
    pub specular: Vector3<f32>,
    pub shininess: f32,
    pub diffuse_intensity: f32,
    pub specular_intensity: f32,

    // ===== 通用参数 =====
    pub emissive: Vector3<f32>,
    pub ambient_factor: Vector3<f32>,
}

impl Material {
    pub fn default(material_type: MaterialType) -> Self {
        Material {
            material_type,
            base_color: Vector3::new(0.8, 0.8, 0.8),
            alpha: 1.0,
            texture: None,
            metallic: 0.0,
            roughness: 0.5,
            ambient_occlusion: 1.0,
            specular: Vector3::new(0.5, 0.5, 0.5),
            shininess: 32.0,
            diffuse_intensity: 1.0,
            specular_intensity: 1.0,
            emissive: Vector3::zeros(),
            ambient_factor: Vector3::new(1.0, 1.0, 1.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub material_id: usize,
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct Model {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
    pub name: String,
}

/// 材质响应计算（统一接口）
pub fn compute_material_response(
    material: &Material,
    light_dir: &Vector3<f32>,
    view_dir: &Vector3<f32>,
    surface_normal: &Vector3<f32>,
) -> Vector3<f32> {
    match material.material_type {
        MaterialType::Phong => {
            let n_dot_l = surface_normal.dot(light_dir).max(0.0);
            if n_dot_l <= 0.0 {
                return material.emissive;
            }
            let diffuse = material.base_color * material.diffuse_intensity * n_dot_l;
            let halfway_dir = (light_dir + view_dir).normalize();
            let n_dot_h = surface_normal.dot(&halfway_dir).max(0.0);
            let spec_intensity = n_dot_h.powf(material.shininess);
            let specular = material.specular * material.specular_intensity * spec_intensity;
            diffuse + specular + material.emissive
        }
        MaterialType::PBR => {
            let base_color = material.base_color;
            let metallic = material.metallic;
            let roughness = material.roughness;
            let ao = material.ambient_occlusion;

            let l = *light_dir;
            let v = *view_dir;
            let h = (l + v).normalize();

            let n_dot_l = surface_normal.dot(&l).max(0.0);
            let n_dot_v = surface_normal.dot(&v).max(0.0);
            let n_dot_h = surface_normal.dot(&h).max(0.0);
            let h_dot_v = h.dot(&v).max(0.0);

            if n_dot_l <= 0.0 {
                return material.emissive;
            }

            // 标准PBR F0计算
            let f0_dielectric = Vector3::new(0.04, 0.04, 0.04);
            let f0 = f0_dielectric.lerp(&base_color, metallic);

            let d = pbr::distribution_ggx(n_dot_h, roughness);
            let g = pbr::geometry_smith(n_dot_v, n_dot_l, roughness);
            let f = pbr::fresnel_schlick(h_dot_v, f0);

            let numerator = d * g * f;
            let denominator = 4.0 * n_dot_v * n_dot_l;
            let specular = numerator / denominator.max(0.001);

            let k_s = f;
            let k_d = (Vector3::new(1.0, 1.0, 1.0) - k_s) * (1.0 - metallic);
            let diffuse = k_d.component_mul(&base_color) / std::f32::consts::PI;

            // 标准Cook-Torrance BRDF
            let brdf_result = (diffuse + specular) * n_dot_l * ao;
            brdf_result + material.emissive
        }
    }
}

pub mod pbr {
    use nalgebra::Vector3;

    pub fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
        let alpha = roughness * roughness;
        let alpha2 = alpha * alpha;
        let n_dot_h2 = n_dot_h * n_dot_h;
        let numerator = alpha2;
        let denominator = n_dot_h2 * (alpha2 - 1.0) + 1.0;
        let denominator = std::f32::consts::PI * denominator * denominator;
        numerator / denominator.max(0.0001)
    }

    pub fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
        let r = roughness + 1.0;
        let k = (r * r) / 8.0;
        n_dot_v / (n_dot_v * (1.0 - k) + k)
    }

    pub fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
        let ggx1 = geometry_schlick_ggx(n_dot_v, roughness);
        let ggx2 = geometry_schlick_ggx(n_dot_l, roughness);
        ggx1 * ggx2
    }

    pub fn fresnel_schlick(cos_theta: f32, f0: Vector3<f32>) -> Vector3<f32> {
        let cos_theta = cos_theta.clamp(0.0, 1.0);
        let one_minus_cos_theta = 1.0 - cos_theta;
        let one_minus_cos_theta5 = one_minus_cos_theta.powi(5);
        f0 + (Vector3::new(1.0, 1.0, 1.0) - f0) * one_minus_cos_theta5
    }
}

/// 材质参数应用（统一接口）
pub fn apply_material_parameters(model: &mut Model, args: &RenderSettings) {
    for material in &mut model.materials {
        match material.material_type {
            MaterialType::PBR => {
                material.metallic = args.metallic.clamp(0.0, 1.0);
                material.roughness = args.roughness.clamp(0.0, 1.0);
                material.ambient_occlusion = args.ambient_occlusion.clamp(0.0, 1.0);
                material.alpha = args.alpha.clamp(0.0, 1.0);

                if let Ok(base_color) = parse_vec3(&args.base_color) {
                    material.base_color = base_color;
                } else {
                    warn!("无法解析基础颜色, 使用默认值: {:?}", material.base_color);
                }

                if let Ok(emissive) = parse_vec3(&args.emissive) {
                    material.emissive = emissive;
                }

                let ambient_response = material.ambient_occlusion * (1.0 - material.metallic);
                material.ambient_factor =
                    Vector3::new(ambient_response, ambient_response, ambient_response);
            }
            MaterialType::Phong => {
                if let Ok(specular_color) = parse_vec3(&args.specular_color) {
                    material.specular = specular_color;
                } else {
                    warn!("无法解析镜面反射颜色, 使用默认值: {:?}", material.specular);
                }

                material.shininess = args.shininess.max(1.0);
                material.diffuse_intensity = args.diffuse_intensity.clamp(0.0, 2.0);
                material.specular_intensity = args.specular_intensity.clamp(0.0, 2.0);
                material.alpha = args.alpha.clamp(0.0, 1.0);

                if let Ok(diffuse_color) = parse_vec3(&args.diffuse_color) {
                    material.base_color = diffuse_color;
                } else {
                    warn!("无法解析漫反射颜色, 使用默认值: {:?}", material.base_color);
                }

                if let Ok(emissive) = parse_vec3(&args.emissive) {
                    material.emissive = emissive;
                }

                material.ambient_factor = material.base_color * 0.3;
            }
        }
    }
}
