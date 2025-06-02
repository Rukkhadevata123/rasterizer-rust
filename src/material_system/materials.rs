use crate::io::render_settings::{RenderSettings, parse_vec3};
use crate::material_system::texture::Texture;
use log::warn;
use nalgebra::{Point3, Vector2, Vector3};
use std::fmt::Debug;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    pub position: Point3<f32>,
    pub normal: Vector3<f32>,
    pub texcoord: Vector2<f32>,
    pub tangent: Vector3<f32>,
    pub bitangent: Vector3<f32>,
}

#[derive(Debug, Clone)]
pub struct Material {
    // ===== 纹理资源 =====
    pub texture: Option<Texture>,
    pub normal_map: Option<Texture>,

    // ===== 通用材质属性 =====
    pub emissive: Vector3<f32>,
    pub alpha: f32,
    pub albedo: Vector3<f32>,
    pub ambient_factor: Vector3<f32>,

    // ===== Phong着色模型专用属性 =====
    pub specular: Vector3<f32>,
    pub shininess: f32,
    pub diffuse_intensity: f32,
    pub specular_intensity: f32,

    // ===== PBR渲染专用属性 =====
    pub metallic: f32,
    pub roughness: f32,
    pub ambient_occlusion: f32,

    // ===== PBR高级属性 =====
    pub subsurface: f32,
    pub anisotropy: f32,
    pub normal_intensity: f32,
}

impl Material {
    pub fn default() -> Self {
        Material {
            texture: None,
            normal_map: None,
            emissive: Vector3::zeros(),
            alpha: 1.0,
            albedo: Vector3::new(0.8, 0.8, 0.8),
            ambient_factor: Vector3::new(1.0, 1.0, 1.0),
            specular: Vector3::new(0.5, 0.5, 0.5),
            shininess: 32.0,
            diffuse_intensity: 1.0,
            specular_intensity: 1.0,
            metallic: 0.0,
            roughness: 0.5,
            ambient_occlusion: 1.0,
            subsurface: 0.0,
            anisotropy: 0.0,
            normal_intensity: 1.0,
        }
    }

    pub fn configure_texture(
        &mut self,
        texture_type: &str,
        options: Option<TextureOptions>,
    ) -> &mut Self {
        match texture_type {
            "face_color" => self.texture = Some(Texture::face_color()),
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
    /// 统一的材质响应计算 - 消除重复代码
    pub fn compute_response(
        &self,
        light_dir: &Vector3<f32>,
        view_dir: &Vector3<f32>,
        surface_normal: &Vector3<f32>,
        surface_tangent: &Vector3<f32>,
        surface_bitangent: &Vector3<f32>,
        surface_uv: &Vector2<f32>,
    ) -> Vector3<f32> {
        // 统一计算有效法线
        let effective_normal = self.compute_effective_normal(
            surface_normal,
            surface_tangent,
            surface_bitangent,
            surface_uv,
        );

        match self {
            MaterialView::BlinnPhong(material) => {
                // Blinn-Phong 保持不变...
                let n_dot_l = effective_normal.dot(light_dir).max(0.0);
                if n_dot_l <= 0.0 {
                    return material.emissive;
                }

                let diffuse = material.diffuse() * material.diffuse_intensity * n_dot_l;
                let halfway_dir = (light_dir + view_dir).normalize();
                let n_dot_h = effective_normal.dot(&halfway_dir).max(0.0);
                let spec_intensity = n_dot_h.powf(material.shininess);
                let specular = material.specular * material.specular_intensity * spec_intensity;

                let n_dot_v = effective_normal.dot(view_dir).max(0.0);
                let phong_subsurface = if material.subsurface > 0.0 {
                    pbr::calculate_subsurface_scattering(
                        n_dot_l,
                        n_dot_v,
                        material.subsurface * 0.6,
                        material.diffuse(),
                        material.alpha,
                    )
                } else {
                    Vector3::zeros()
                };

                diffuse + specular + phong_subsurface + material.emissive
            }
            MaterialView::PBR(material) => {
                let base_color = material.base_color();
                let metallic = material.metallic;
                let roughness = material.roughness;
                let ao = material.ambient_occlusion;
                let subsurface = material.subsurface;
                let anisotropy = material.anisotropy;
                let alpha = material.alpha;

                let l = *light_dir;
                let v = *view_dir;
                let h = (l + v).normalize();

                let n_dot_l = effective_normal.dot(&l).max(0.0);
                let n_dot_v = effective_normal.dot(&v).max(0.0);
                let n_dot_h = effective_normal.dot(&h).max(0.0);
                let h_dot_v = h.dot(&v).max(0.0);

                if n_dot_l <= 0.0 {
                    return material.emissive;
                }

                let f0_dielectric = Vector3::new(0.04, 0.04, 0.04);
                let transparency_factor = 1.0 - alpha;
                let adjusted_f0_dielectric = f0_dielectric * (1.0 - transparency_factor * 0.3);
                let f0 = adjusted_f0_dielectric.lerp(&base_color, metallic);

                // 修复：各向异性支持 - 使用原始表面法线构建TBN
                let d = if anisotropy.abs() > 0.01 {
                    // ✅ 修复：使用原始 surface_normal 而不是 effective_normal
                    let tbn = tbn::build_matrix(surface_normal, surface_tangent, surface_bitangent);
                    let t_surf = tbn.column(0).into_owned();
                    let b_surf = tbn.column(1).into_owned();
                    let h_dot_t = h.dot(&t_surf);
                    let h_dot_b = h.dot(&b_surf);
                    let (alpha_x, alpha_y) = pbr::apply_anisotropy(roughness, anisotropy);
                    pbr::distribution_ggx_anisotropic(n_dot_h, h_dot_t, h_dot_b, alpha_x, alpha_y)
                } else {
                    pbr::distribution_ggx(n_dot_h, roughness)
                };

                let g = pbr::geometry_smith(n_dot_v, n_dot_l, roughness);
                let f = pbr::fresnel_schlick(h_dot_v, f0);

                let numerator = d * g * f;
                let denominator = 4.0 * n_dot_v * n_dot_l;
                let specular = numerator / denominator.max(0.001);

                let k_s = f;
                let k_d = (Vector3::new(1.0, 1.0, 1.0) - k_s) * (1.0 - metallic);
                let diffuse = k_d.component_mul(&base_color) / std::f32::consts::PI;

                let subsurface_contrib = if subsurface > 0.0 && metallic < 0.5 {
                    pbr::calculate_subsurface_scattering(
                        n_dot_l, n_dot_v, subsurface, base_color, alpha,
                    )
                } else {
                    Vector3::zeros()
                };

                let metallic_enhancement = if metallic > 0.3 {
                    let enhancement_factor = (metallic - 0.3) / 0.7;
                    let transparency_mod = if alpha < 0.8 { 1.2 } else { 1.0 };
                    specular * enhancement_factor * 0.6 * transparency_mod
                } else {
                    Vector3::zeros()
                };

                let adjusted_ao = ao + transparency_factor * (1.0 - ao) * 0.4;

                let brdf_result =
                    (diffuse + specular + metallic_enhancement) * n_dot_l * adjusted_ao
                        + subsurface_contrib;
                brdf_result + material.emissive
            }
        }
    }

    /// 统一的有效法线计算 - 修复并简化
    fn compute_effective_normal(
        &self,
        surface_normal: &Vector3<f32>,
        surface_tangent: &Vector3<f32>,
        surface_bitangent: &Vector3<f32>,
        surface_uv: &Vector2<f32>,
    ) -> Vector3<f32> {
        let material = match self {
            MaterialView::BlinnPhong(m) => m,
            MaterialView::PBR(m) => m,
        };

        if let Some(normal_map) = &material.normal_map {
            let normal_sample = normal_map.sample_normal(surface_uv.x, surface_uv.y);

            let mut tangent_normal = Vector3::new(
                normal_sample[0],
                normal_sample[1],
                normal_sample[2], // 修复：不强制为正
            );

            // 应用法线强度
            if material.normal_intensity != 1.0 {
                tangent_normal.x *= material.normal_intensity;
                tangent_normal.y *= material.normal_intensity;

                // 重新计算Z以保持单位向量，但保持符号
                let xy_length_sq =
                    tangent_normal.x * tangent_normal.x + tangent_normal.y * tangent_normal.y;
                if xy_length_sq <= 1.0 {
                    let z_sign = if tangent_normal.z >= 0.0 { 1.0 } else { -1.0 };
                    tangent_normal.z = z_sign * (1.0 - xy_length_sq).sqrt().max(0.01);
                } else {
                    let xy_length = xy_length_sq.sqrt();
                    tangent_normal.x /= xy_length;
                    tangent_normal.y /= xy_length;
                    tangent_normal.z = if tangent_normal.z >= 0.0 { 0.01 } else { -0.01 };
                }
            }

            let tangent_normal = tangent_normal
                .try_normalize(1e-6)
                .unwrap_or_else(|| Vector3::new(0.0, 0.0, 1.0));

            // 使用统一的TBN矩阵构建
            let tbn = tbn::build_matrix(surface_normal, surface_tangent, surface_bitangent);
            (tbn * tangent_normal).normalize()
        } else {
            // 无法线贴图时的程序化法线强度
            if material.normal_intensity != 1.0 {
                pbr::apply_procedural_normal_intensity(
                    *surface_normal,
                    material.normal_intensity,
                    *surface_uv,
                )
            } else {
                *surface_normal
            }
        }
    }
}

pub mod pbr {
    use nalgebra::{Vector2, Vector3};

    pub fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
        let alpha = roughness * roughness;
        let alpha2 = alpha * alpha;
        let n_dot_h2 = n_dot_h * n_dot_h;
        let numerator = alpha2;
        let denominator = n_dot_h2 * (alpha2 - 1.0) + 1.0;
        let denominator = std::f32::consts::PI * denominator * denominator;
        numerator / denominator.max(0.0001)
    }

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

    pub fn calculate_subsurface_scattering(
        n_dot_l: f32,
        n_dot_v: f32,
        strength: f32,
        base_color: Vector3<f32>,
        alpha: f32,
    ) -> Vector3<f32> {
        if strength <= 0.0 {
            return Vector3::zeros();
        }

        let view_scatter = (1.0 - n_dot_v).powi(2);
        let light_scatter = (1.0 - n_dot_l).powi(2);
        let scatter = (view_scatter + light_scatter) * 0.5 * strength;

        let transparency_factor = 1.0 - alpha;
        let enhanced_scatter = scatter * (1.0 + transparency_factor * 0.8);
        let depth_factor = alpha + transparency_factor * 0.5;

        let base_warmth = Vector3::new(1.1, 0.9, 0.8);
        let transparency_warmth = Vector3::new(1.05, 0.95, 0.85);
        let warmth_factor = base_warmth.lerp(&transparency_warmth, transparency_factor);

        let subsurface_color = base_color.component_mul(&warmth_factor);

        let final_intensity = if alpha > 0.8 {
            0.7 * depth_factor
        } else if alpha > 0.3 {
            0.9 * depth_factor
        } else {
            0.5 * depth_factor
        };

        subsurface_color * enhanced_scatter * final_intensity
    }

    pub fn compute_fallback_tangent(normal: &Vector3<f32>) -> Vector3<f32> {
        let candidate = if normal.x.abs() < 0.9 {
            Vector3::x()
        } else {
            Vector3::y()
        };
        normal.cross(&candidate).normalize()
    }

    pub fn apply_procedural_normal_intensity(
        normal: Vector3<f32>,
        intensity: f32,
        uv: Vector2<f32>,
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

pub mod tbn {
    use super::{Point3, Vector2, Vector3, Vertex, pbr};
    use log::warn;
    use nalgebra::Matrix3;

    pub fn build_matrix(
        normal: &Vector3<f32>,
        tangent: &Vector3<f32>,
        bitangent: &Vector3<f32>,
    ) -> Matrix3<f32> {
        let n = normal.normalize();

        let t_raw = *tangent;
        let t_orthogonal = (t_raw - n * t_raw.dot(&n))
            .try_normalize(1e-8)
            .unwrap_or_else(|| pbr::compute_fallback_tangent(&n));

        let b_computed = n.cross(&t_orthogonal).normalize();
        let b_input = bitangent.try_normalize(1e-8).unwrap_or(b_computed);
        let handedness = b_input.dot(&b_computed);

        let final_bitangent = if handedness < 0.0 {
            -b_computed
        } else {
            b_computed
        };

        let tbn = Matrix3::from_columns(&[t_orthogonal, final_bitangent, n]);
        let det = tbn.determinant();

        if det.abs() < 0.1 {
            warn!("TBN矩阵行列式异常: {:.6}", det);
            let fallback_t = pbr::compute_fallback_tangent(&n);
            let fallback_b = n.cross(&fallback_t).normalize();
            Matrix3::from_columns(&[fallback_t, fallback_b, n])
        } else {
            tbn
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn calculate_tangents_and_bitangents(
        positions: &[Point3<f32>],
        texcoords: &[Vector2<f32>],
        normals: Option<&[Vector3<f32>]>,
        indices: &[u32],
    ) -> Result<(Vec<Vector3<f32>>, Vec<Vector3<f32>>), String> {
        if indices.len() % 3 != 0 {
            return Err("索引数量必须是3的倍数".to_string());
        }
        if positions.len() != texcoords.len() {
            return Err("位置和纹理坐标数组长度必须相同".to_string());
        }
        if positions.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        let num_vertices = positions.len();
        let mut tangents = vec![Vector3::zeros(); num_vertices];
        let mut bitangents = vec![Vector3::zeros(); num_vertices];
        let mut tangent_counts = vec![0u32; num_vertices];

        for triangle in indices.chunks_exact(3) {
            let [i0, i1, i2] = [
                triangle[0] as usize,
                triangle[1] as usize,
                triangle[2] as usize,
            ];

            if i0 >= num_vertices || i1 >= num_vertices || i2 >= num_vertices {
                warn!("TBN计算遇到越界索引，跳过三角形");
                continue;
            }

            let [pos0, pos1, pos2] = [positions[i0], positions[i1], positions[i2]];
            let [uv0, uv1, uv2] = [texcoords[i0], texcoords[i1], texcoords[i2]];

            let edge1 = pos1 - pos0;
            let edge2 = pos2 - pos0;
            let delta_uv1 = uv1 - uv0;
            let delta_uv2 = uv2 - uv0;

            let det = delta_uv1.x * delta_uv2.y - delta_uv1.y * delta_uv2.x;

            if det.abs() < 1e-8 {
                if let Some(normals) = normals {
                    let avg_normal = (normals[i0] + normals[i1] + normals[i2]) / 3.0;
                    if avg_normal.norm_squared() > 1e-15 {
                        let avg_normal = avg_normal.normalize();
                        let fallback_tangent = pbr::compute_fallback_tangent(&avg_normal);
                        let fallback_bitangent = avg_normal.cross(&fallback_tangent).normalize();

                        for &i in &[i0, i1, i2] {
                            tangents[i] += fallback_tangent;
                            bitangents[i] += fallback_bitangent;
                            tangent_counts[i] += 1;
                        }
                    }
                }
                continue;
            }

            let r = 1.0 / det;
            let tangent = (edge1 * delta_uv2.y - edge2 * delta_uv1.y) * r;
            let bitangent = (edge2 * delta_uv1.x - edge1 * delta_uv2.x) * r;

            if tangent.norm_squared() > 1e-15 && bitangent.norm_squared() > 1e-15 {
                for &i in &[i0, i1, i2] {
                    tangents[i] += tangent;
                    bitangents[i] += bitangent;
                    tangent_counts[i] += 1;
                }
            }
        }

        for i in 0..num_vertices {
            let count = tangent_counts[i];
            if count > 0 {
                tangents[i] /= count as f32;
                bitangents[i] /= count as f32;
            }

            let n = if let Some(ns) = normals {
                if ns[i].norm_squared() > 1e-15 {
                    ns[i].normalize()
                } else {
                    Vector3::y()
                }
            } else {
                Vector3::y()
            };

            let t = tangents[i];
            let t_orthogonal = if t.norm_squared() > 1e-15 {
                (t - n * t.dot(&n))
                    .try_normalize(1e-8)
                    .unwrap_or_else(|| pbr::compute_fallback_tangent(&n))
            } else {
                pbr::compute_fallback_tangent(&n)
            };

            tangents[i] = t_orthogonal;
            bitangents[i] = n.cross(&t_orthogonal).normalize();
        }

        Ok((tangents, bitangents))
    }

    pub fn validate_and_fix(vertices: &mut [Vertex]) {
        for vertex in vertices.iter_mut() {
            if vertex.normal.norm_squared() < 1e-12 {
                vertex.normal = Vector3::y();
            } else {
                vertex.normal = vertex.normal.normalize();
            }

            if vertex.tangent.norm_squared() < 1e-12 {
                vertex.tangent = pbr::compute_fallback_tangent(&vertex.normal);
            } else {
                let t_raw = vertex.tangent;
                let n = vertex.normal;
                vertex.tangent = (t_raw - n * t_raw.dot(&n))
                    .try_normalize(1e-10)
                    .unwrap_or_else(|| pbr::compute_fallback_tangent(&n));
            }

            vertex.bitangent = vertex.normal.cross(&vertex.tangent).normalize();
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
            material.alpha = args.alpha.clamp(0.0, 1.0);

            if let Ok(base_color) = parse_vec3(&args.base_color) {
                material.albedo = base_color;
            } else {
                warn!("无法解析基础颜色, 使用默认值: {:?}", material.albedo);
            }

            if let Ok(emissive) = parse_vec3(&args.emissive) {
                material.emissive = emissive;
            }

            let ambient_response = material.ambient_occlusion * (1.0 - material.metallic);
            material.ambient_factor =
                Vector3::new(ambient_response, ambient_response, ambient_response);
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
            material.alpha = args.alpha.clamp(0.0, 1.0);

            if let Ok(diffuse_color) = parse_vec3(&args.diffuse_color) {
                material.albedo = diffuse_color;
            } else {
                warn!("无法解析漫反射颜色, 使用默认值: {:?}", material.diffuse());
            }

            if let Ok(emissive) = parse_vec3(&args.emissive) {
                material.emissive = emissive;
            }

            material.ambient_factor = material.albedo * 0.3;
        }
    }
}
