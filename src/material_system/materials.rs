use crate::io::render_settings::{RenderSettings, parse_vec3};
use crate::material_system::texture::Texture;
use log::warn; // 移除未使用的debug导入
use nalgebra::{Matrix3, Point3, Vector2, Vector3};
use std::fmt::Debug;
use std::path::PathBuf;

/// 表示带有位置、法线和纹理坐标的顶点
#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    pub position: Point3<f32>,
    pub normal: Vector3<f32>,
    pub texcoord: Vector2<f32>,
    pub tangent: Vector3<f32>,
    pub bitangent: Vector3<f32>,
}

/// 表示材质属性，包含渲染所需的各种属性
#[derive(Debug, Clone)]
pub struct Material {
    // 通用属性
    pub texture: Option<Texture>,
    pub normal_map: Option<Texture>,
    pub emissive: Vector3<f32>,

    // 着色模型共享属性
    pub albedo: Vector3<f32>,
    pub ambient_factor: Vector3<f32>,

    // Blinn-Phong渲染专用属性
    pub specular: Vector3<f32>,
    pub shininess: f32,
    pub diffuse_intensity: f32,
    pub specular_intensity: f32,

    // PBR渲染专用属性
    pub metallic: f32,
    pub roughness: f32,
    pub ambient_occlusion: f32,
    pub subsurface: f32,
    pub anisotropy: f32,
    pub normal_intensity: f32,
}

impl Material {
    pub fn default() -> Self {
        Material {
            albedo: Vector3::new(0.8, 0.8, 0.8),
            specular: Vector3::new(0.5, 0.5, 0.5),
            shininess: 32.0,
            diffuse_intensity: 1.0,
            specular_intensity: 1.0,
            texture: None,
            normal_map: None,
            emissive: Vector3::zeros(),
            metallic: 0.0,
            roughness: 0.5,
            ambient_occlusion: 1.0,
            subsurface: 0.0,
            anisotropy: 0.0,
            normal_intensity: 1.0,
            ambient_factor: Vector3::new(1.0, 1.0, 1.0),
        }
    }

    pub fn configure_texture(
        &mut self,
        texture_type: &str,
        options: Option<TextureOptions>,
    ) -> &mut Self {
        match texture_type {
            "face_color" => {
                self.texture = Some(Texture::face_color());
            }
            "image" => {
                if let Some(options) = options {
                    if let Some(path) = options.path {
                        if let Some(texture) = Texture::from_file(path) {
                            self.texture = Some(texture);
                        } else {
                            warn!("无法加载纹理，保持当前纹理设置");
                        }
                    }
                }
            }
            "solid_color" => {
                if let Some(options) = options {
                    if let Some(color) = options.color {
                        self.texture = Some(Texture::solid_color(color));
                    }
                }
            }
            "normal_map" => {
                if let Some(options) = options {
                    if let Some(path) = options.path {
                        if let Some(normal_texture) = Texture::from_file(path) {
                            self.normal_map = Some(normal_texture);
                        } else {
                            warn!("无法加载法线贴图，保持当前设置");
                        }
                    }
                }
            }
            _ => warn!("未知的纹理类型: {}", texture_type),
        }
        self
    }

    pub fn diffuse(&self) -> Vector3<f32> {
        self.albedo
    }

    pub fn base_color(&self) -> Vector3<f32> {
        self.albedo
    }
}

#[derive(Debug, Clone)]
pub struct TextureOptions {
    pub path: Option<PathBuf>,
    pub color: Option<Vector3<f32>>,
}

#[derive(Debug, Clone)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub material_id: Option<usize>,
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct ModelData {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
    pub name: String,
}

#[derive(Debug, Clone)]
pub enum MaterialView<'a> {
    BlinnPhong(&'a Material),
    #[allow(clippy::upper_case_acronyms)]
    PBR(&'a Material),
}

impl MaterialView<'_> {
    /// 修复后的完整PBR实现，包含法线贴图和TBN支持
    pub fn compute_response(
        &self,
        light_dir: &Vector3<f32>,
        view_dir: &Vector3<f32>,
        surface_normal: &Vector3<f32>,
        surface_tangent: &Vector3<f32>,
        surface_bitangent: &Vector3<f32>,
        surface_uv: &Vector2<f32>,
    ) -> Vector3<f32> {
        match self {
            MaterialView::BlinnPhong(material) => {
                // Blinn-Phong可选法线贴图支持
                let effective_normal = if let Some(normal_map) = &material.normal_map {
                    // 构建TBN矩阵
                    let tbn = Matrix3::from_columns(&[
                        *surface_tangent,
                        *surface_bitangent,
                        *surface_normal,
                    ]);

                    // 采样法线贴图
                    let normal_map_sample = normal_map.sample_normal(surface_uv.x, surface_uv.y);
                    let tangent_normal = Vector3::new(
                        normal_map_sample[0],
                        normal_map_sample[1],
                        normal_map_sample[2],
                    );

                    // 应用法线强度
                    let adjusted_normal = if material.normal_intensity != 1.0 {
                        let mut tn = tangent_normal;
                        tn.x *= material.normal_intensity;
                        tn.y *= material.normal_intensity;
                        tn.normalize()
                    } else {
                        tangent_normal.normalize()
                    };

                    // 转换到世界空间
                    (tbn * adjusted_normal).normalize()
                } else {
                    *surface_normal
                };

                let n_dot_l = effective_normal.dot(light_dir).max(0.0);
                if n_dot_l <= 0.0 {
                    return material.emissive;
                }

                let diffuse = material.diffuse() * material.diffuse_intensity * n_dot_l;
                let halfway_dir = (light_dir + view_dir).normalize();
                let n_dot_h = effective_normal.dot(&halfway_dir).max(0.0);
                let spec_intensity = n_dot_h.powf(material.shininess);
                let specular = material.specular * material.specular_intensity * spec_intensity;

                diffuse + specular + material.emissive
            }
            MaterialView::PBR(material) => {
                use pbr_functions::*;

                let base_color = material.base_color();
                let metallic = material.metallic;
                let roughness = material.roughness;
                let ao = material.ambient_occlusion;
                let subsurface = material.subsurface;
                let anisotropy = material.anisotropy;

                // 1. 确定最终法线 (关键修复！)
                let (n_final, tbn_matrix_option) = if let Some(normal_map) = &material.normal_map {
                    // 构建标准化TBN矩阵
                    let n = surface_normal.normalize();
                    let t = (*surface_tangent - n * surface_tangent.dot(&n))
                        .try_normalize(1e-6)
                        .unwrap_or_else(|| compute_fallback_tangent(&n));
                    let b = n.cross(&t).normalize();

                    let tbn = Matrix3::from_columns(&[t, b, n]);

                    // 采样并处理法线贴图
                    let normal_sample = normal_map.sample_normal(surface_uv.x, surface_uv.y);
                    let mut tangent_space_normal = Vector3::new(
                        normal_sample[0],
                        normal_sample[1],
                        normal_sample[2].max(0.1), // 确保Z分量不为零
                    );

                    // 应用法线强度
                    if material.normal_intensity != 1.0 {
                        tangent_space_normal.x *= material.normal_intensity;
                        tangent_space_normal.y *= material.normal_intensity;
                    }

                    // 归一化并转换到世界空间
                    let normalized_tangent_normal = tangent_space_normal
                        .try_normalize(1e-6)
                        .unwrap_or_else(Vector3::z);

                    let world_normal = (tbn * normalized_tangent_normal).normalize();
                    (world_normal, Some(tbn))
                } else {
                    // 无法线贴图时的程序化法线强度
                    let processed_normal = if material.normal_intensity != 1.0 {
                        apply_procedural_normal_intensity(
                            *surface_normal,
                            material.normal_intensity,
                            *surface_uv, // 修复：直接传递Vector2<f32>而不是引用
                        )
                    } else {
                        *surface_normal
                    };

                    // 为各向异性构建近似TBN
                    let tbn_opt = if anisotropy.abs() > 0.001 {
                        let n = processed_normal.normalize();
                        let t = compute_fallback_tangent(&n);
                        let b = n.cross(&t).normalize();
                        Some(Matrix3::from_columns(&[t, b, n]))
                    } else {
                        None
                    };

                    (processed_normal, tbn_opt)
                };

                // 2. 光照计算
                let l = *light_dir;
                let v = *view_dir;
                let h = (l + v).normalize();

                let n_dot_l = n_final.dot(&l).max(0.0);
                let n_dot_v = n_final.dot(&v).max(0.0);
                let n_dot_h = n_final.dot(&h).max(0.0);
                let h_dot_v = h.dot(&v).max(0.0);

                if n_dot_l <= 0.0 {
                    return material.emissive;
                }

                // 3. F0计算 - 恢复经典lerp
                let f0_dielectric = Vector3::new(0.04, 0.04, 0.04);
                let f0 = f0_dielectric.lerp(&base_color, metallic);

                // 4. 分布函数选择
                let d = if anisotropy.abs() > 0.01 && tbn_matrix_option.is_some() {
                    let tbn = tbn_matrix_option.unwrap();
                    let t_surf = tbn.column(0).into_owned();
                    let b_surf = tbn.column(1).into_owned();

                    let h_dot_t = h.dot(&t_surf);
                    let h_dot_b = h.dot(&b_surf);

                    let (alpha_x, alpha_y) = apply_anisotropy(roughness, anisotropy);
                    distribution_ggx_anisotropic(n_dot_h, h_dot_t, h_dot_b, alpha_x, alpha_y)
                } else {
                    distribution_ggx(n_dot_h, roughness)
                };

                // 5. 几何函数
                let g = geometry_smith_standard(n_dot_v, n_dot_l, roughness);
                let f = fresnel_schlick(h_dot_v, f0);

                // 6. BRDF计算
                let numerator = d * g * f;
                let denominator = 4.0 * n_dot_v * n_dot_l;
                let specular = numerator / denominator.max(0.001);

                let k_s = f;
                let k_d = (Vector3::new(1.0, 1.0, 1.0) - k_s) * (1.0 - metallic);
                let diffuse = k_d.component_mul(&base_color) / std::f32::consts::PI;

                // 7. 次表面散射
                let subsurface_contrib = if subsurface > 0.0 && metallic < 0.5 {
                    calculate_subsurface_scattering(n_dot_l, n_dot_v, subsurface, base_color)
                } else {
                    Vector3::zeros()
                };

                // 8. 金属增强
                let metallic_enhancement = if metallic > 0.3 {
                    let enhancement_factor = (metallic - 0.3) / 0.7;
                    specular * enhancement_factor * 0.6
                } else {
                    Vector3::zeros()
                };

                let brdf_result =
                    (diffuse + specular + metallic_enhancement) * n_dot_l * ao + subsurface_contrib;
                brdf_result + material.emissive
            }
        }
    }
}

pub mod material_applicator {
    use super::{ModelData, RenderSettings, Vector3, parse_vec3};
    use log::warn;

    pub fn apply_pbr_parameters(model_data: &mut ModelData, args: &RenderSettings) {
        for material in &mut model_data.materials {
            material.metallic = args.metallic.clamp(0.0, 1.0);
            material.roughness = args.roughness.clamp(0.0, 1.0);
            material.ambient_occlusion = args.ambient_occlusion.clamp(0.0, 1.0);
            material.subsurface = args.subsurface.clamp(0.0, 1.0);
            material.anisotropy = args.anisotropy.clamp(-1.0, 1.0);
            material.normal_intensity = args.normal_intensity.clamp(0.0, 2.0);

            if let Ok(base_color) = parse_vec3(&args.base_color) {
                material.albedo = base_color;
            } else {
                warn!("无法解析基础颜色, 使用默认值: {:?}", material.albedo);
            }

            if let Ok(emissive) = parse_vec3(&args.emissive) {
                material.emissive = emissive;
            }

            // 恢复环境因子计算
            let ambient_response = material.ambient_occlusion * (1.0 - material.metallic);
            material.ambient_factor =
                Vector3::new(ambient_response, ambient_response, ambient_response);

            log::debug!(
                "应用PBR材质 - 基础色: {:?}, 金属度: {:.2}, 粗糙度: {:.2}, AO: {:.2}, 次表面: {:.2}, 法线强度: {:.2}",
                material.base_color(),
                material.metallic,
                material.roughness,
                material.ambient_occlusion,
                material.subsurface,
                material.normal_intensity
            );
        }
    }

    pub fn apply_phong_parameters(model_data: &mut ModelData, args: &RenderSettings) {
        for material in &mut model_data.materials {
            if let Ok(specular_color) = parse_vec3(&args.specular_color) {
                material.specular = specular_color;
            } else {
                warn!("无法解析镜面反射颜色, 使用默认值: {:?}", material.specular);
            }

            material.shininess = args.shininess.max(1.0);
            material.diffuse_intensity = args.diffuse_intensity.clamp(0.0, 2.0);
            material.specular_intensity = args.specular_intensity.clamp(0.0, 2.0);

            if let Ok(diffuse_color) = parse_vec3(&args.diffuse_color) {
                material.albedo = diffuse_color;
            } else {
                warn!("无法解析漫反射颜色, 使用默认值: {:?}", material.diffuse());
            }

            if let Ok(emissive) = parse_vec3(&args.emissive) {
                material.emissive = emissive;
            }

            material.ambient_factor = material.albedo * 0.3;

            log::debug!(
                "应用Phong材质 - 漫反射: {:?}, 镜面: {:?}, 光泽度: {:.2}",
                material.diffuse(),
                material.specular,
                material.shininess
            );
        }
    }
}

pub mod pbr_functions {
    use nalgebra::{Vector2, Vector3};

    // 标准GGX分布函数
    pub fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
        let alpha = roughness * roughness;
        let alpha2 = alpha * alpha;
        let n_dot_h2 = n_dot_h * n_dot_h;
        let numerator = alpha2;
        let denominator = n_dot_h2 * (alpha2 - 1.0) + 1.0;
        let denominator = std::f32::consts::PI * denominator * denominator;
        numerator / denominator.max(0.0001)
    }

    // 各向异性GGX分布函数
    pub fn distribution_ggx_anisotropic(
        n_dot_h: f32,
        h_dot_t: f32,
        h_dot_b: f32,
        alpha_x: f32,
        alpha_y: f32,
    ) -> f32 {
        let alpha_x2 = alpha_x * alpha_x;
        let alpha_y2 = alpha_y * alpha_y;
        let n_dot_h2 = n_dot_h * n_dot_h;

        let term1 = h_dot_t * h_dot_t / alpha_x2;
        let term2 = h_dot_b * h_dot_b / alpha_y2;
        let denominator = alpha_x * alpha_y * (term1 + term2 + n_dot_h2);

        1.0 / (std::f32::consts::PI * denominator * denominator).max(0.0001)
    }

    // 标准几何函数
    pub fn geometry_schlick_ggx_standard(n_dot_v: f32, roughness: f32) -> f32 {
        let r = roughness + 1.0;
        let k = (r * r) / 8.0;
        n_dot_v / (n_dot_v * (1.0 - k) + k)
    }

    pub fn geometry_smith_standard(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
        let ggx1 = geometry_schlick_ggx_standard(n_dot_v, roughness);
        let ggx2 = geometry_schlick_ggx_standard(n_dot_l, roughness);
        ggx1 * ggx2
    }

    // 菲涅耳方程
    pub fn fresnel_schlick(cos_theta: f32, f0: Vector3<f32>) -> Vector3<f32> {
        let cos_theta = cos_theta.clamp(0.0, 1.0);
        let one_minus_cos_theta = 1.0 - cos_theta;
        let one_minus_cos_theta5 = one_minus_cos_theta.powi(5);
        f0 + (Vector3::new(1.0, 1.0, 1.0) - f0) * one_minus_cos_theta5
    }

    // 各向异性参数计算
    pub fn apply_anisotropy(base_roughness: f32, anisotropy: f32) -> (f32, f32) {
        if anisotropy.abs() < 0.001 {
            return (base_roughness, base_roughness);
        }

        let aspect_ratio = (1.0 - anisotropy.abs() * 0.9).sqrt();
        let alpha_x = base_roughness / aspect_ratio;
        let alpha_y = base_roughness * aspect_ratio;

        if anisotropy > 0.0 {
            (alpha_x.clamp(0.01, 1.0), alpha_y.clamp(0.01, 1.0))
        } else {
            (alpha_y.clamp(0.01, 1.0), alpha_x.clamp(0.01, 1.0))
        }
    }

    // 次表面散射
    pub fn calculate_subsurface_scattering(
        n_dot_l: f32,
        n_dot_v: f32,
        strength: f32,
        base_color: Vector3<f32>,
    ) -> Vector3<f32> {
        if strength <= 0.0 {
            return Vector3::zeros();
        }

        let view_scatter = (1.0 - n_dot_v).powi(2);
        let light_scatter = (1.0 - n_dot_l).powi(2);
        let scatter = (view_scatter + light_scatter) * 0.5 * strength;

        let warmth_factor = Vector3::new(1.1, 0.9, 0.8);
        let subsurface_color = base_color.component_mul(&warmth_factor);

        subsurface_color * scatter * 0.7
    }

    // 工具函数：计算备用切线
    pub fn compute_fallback_tangent(normal: &Vector3<f32>) -> Vector3<f32> {
        let candidate = if normal.x.abs() < 0.9 {
            Vector3::x()
        } else {
            Vector3::y()
        };
        normal.cross(&candidate).normalize()
    }

    // 修复：程序化法线强度函数接受Vector2<f32>参数
    pub fn apply_procedural_normal_intensity(
        normal: Vector3<f32>,
        intensity: f32,
        uv: Vector2<f32>, // 修复：改为值传递，而不是引用
    ) -> Vector3<f32> {
        if (intensity - 1.0).abs() < 0.001 {
            return normal;
        }

        let intensity_factor = (intensity - 1.0).clamp(-1.0, 1.0);
        let perturbation_strength = intensity_factor * 0.2;

        let perturbation = Vector3::new(
            (uv.x * 12.0).sin() * perturbation_strength,
            (uv.y * 10.0).cos() * perturbation_strength,
            ((uv.x + uv.y) * 8.0).sin() * perturbation_strength * 0.3,
        );

        let tangent = compute_fallback_tangent(&normal);
        let bitangent = normal.cross(&tangent).normalize();

        let perturbed_normal = normal
            + tangent * perturbation.x
            + bitangent * perturbation.y
            + normal * perturbation.z;

        perturbed_normal.normalize()
    }
}

pub mod tbn_utils {
    use super::{Point3, Vector2, Vector3, Vertex};
    use log::warn;

    pub type TbnResult = Result<(Vec<Vector3<f32>>, Vec<Vector3<f32>>), String>;

    /// 改进的TBN计算 - 修复数值稳定性
    pub fn calculate_tangents_and_bitangents(
        positions: &[Point3<f32>],
        texcoords: &[Vector2<f32>],
        normals: Option<&[Vector3<f32>]>,
        indices: &[u32],
    ) -> TbnResult {
        if indices.len() % 3 != 0 {
            return Err("索引数量必须是3的倍数".to_string());
        }
        if positions.len() != texcoords.len()
            || (normals.is_some() && positions.len() != normals.unwrap().len())
        {
            return Err("位置、纹理坐标和法线数组长度必须相同".to_string());
        }

        if positions.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        let num_vertices = positions.len();
        let mut tangents = vec![Vector3::zeros(); num_vertices];
        let mut bitangents = vec![Vector3::zeros(); num_vertices];

        // 改进的TBN计算
        for i in (0..indices.len()).step_by(3) {
            let i0 = indices[i] as usize;
            let i1 = indices[i + 1] as usize;
            let i2 = indices[i + 2] as usize;

            if i0 >= num_vertices || i1 >= num_vertices || i2 >= num_vertices {
                warn!("TBN计算遇到越界索引，跳过三角形");
                continue;
            }

            let pos0 = positions[i0];
            let pos1 = positions[i1];
            let pos2 = positions[i2];

            let uv0 = texcoords[i0];
            let uv1 = texcoords[i1];
            let uv2 = texcoords[i2];

            let edge1 = pos1 - pos0;
            let edge2 = pos2 - pos0;
            let delta_uv1 = uv1 - uv0;
            let delta_uv2 = uv2 - uv0;

            // 改进的行列式计算，增强数值稳定性
            let det = delta_uv1.x * delta_uv2.y - delta_uv1.y * delta_uv2.x;

            if det.abs() < 1e-8 {
                // 退化的纹理坐标，使用几何信息生成备用切线
                if let Some(normals) = normals {
                    let avg_normal = (normals[i0] + normals[i1] + normals[i2]) / 3.0;
                    let fallback_tangent =
                        super::pbr_functions::compute_fallback_tangent(&avg_normal);
                    let fallback_bitangent = avg_normal.cross(&fallback_tangent).normalize();

                    tangents[i0] += fallback_tangent;
                    tangents[i1] += fallback_tangent;
                    tangents[i2] += fallback_tangent;

                    bitangents[i0] += fallback_bitangent;
                    bitangents[i1] += fallback_bitangent;
                    bitangents[i2] += fallback_bitangent;
                }
                continue;
            }

            let r = 1.0 / det;
            let tangent = (edge1 * delta_uv2.y - edge2 * delta_uv1.y) * r;
            let bitangent = (edge2 * delta_uv1.x - edge1 * delta_uv2.x) * r;

            // 累加到顶点
            tangents[i0] += tangent;
            tangents[i1] += tangent;
            tangents[i2] += tangent;

            bitangents[i0] += bitangent;
            bitangents[i1] += bitangent;
            bitangents[i2] += bitangent;
        }

        // 改进的正交化过程
        for i in 0..num_vertices {
            let n = if let Some(ns) = normals {
                ns[i].normalize()
            } else {
                Vector3::y()
            };

            // Gram-Schmidt正交化
            let t = tangents[i];
            let t_orthogonal = (t - n * t.dot(&n))
                .try_normalize(1e-6)
                .unwrap_or_else(|| super::pbr_functions::compute_fallback_tangent(&n));

            tangents[i] = t_orthogonal;

            // 重新计算副切线确保正交性
            bitangents[i] = n.cross(&t_orthogonal).normalize();
        }

        Ok((tangents, bitangents))
    }

    /// 改进的TBN验证和修复
    pub fn validate_and_fix_tbn(vertices: &mut [Vertex]) {
        for vertex in vertices.iter_mut() {
            // 归一化并验证法线
            if vertex.normal.norm_squared() < 1e-6 {
                vertex.normal = Vector3::y();
            } else {
                vertex.normal = vertex.normal.normalize();
            }

            // 处理切线
            if vertex.tangent.norm_squared() < 1e-6 {
                vertex.tangent = super::pbr_functions::compute_fallback_tangent(&vertex.normal);
            } else {
                // 确保切线与法线正交
                vertex.tangent = (vertex.tangent
                    - vertex.normal * vertex.tangent.dot(&vertex.normal))
                .try_normalize(1e-6)
                .unwrap_or_else(|| super::pbr_functions::compute_fallback_tangent(&vertex.normal));
            }

            // 重新计算副切线
            vertex.bitangent = vertex.normal.cross(&vertex.tangent).normalize();
        }
    }
}
