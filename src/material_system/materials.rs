use crate::io::render_settings::{RenderSettings, parse_vec3};
use crate::material_system::texture::Texture;
use log::warn;
use nalgebra::{Matrix3, Point3, Vector2, Vector3};
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
    /// 主纹理（漫反射/基础颜色纹理）
    pub texture: Option<Texture>,
    /// 法线贴图纹理
    pub normal_map: Option<Texture>,

    // ===== 通用材质属性 =====
    /// 自发光颜色（不受光照影响）
    pub emissive: Vector3<f32>,
    /// 透明度 (0.0=完全透明, 1.0=完全不透明)
    pub alpha: f32,
    /// 基础颜色/反照率（PBR中的base color，Phong中的diffuse color）
    pub albedo: Vector3<f32>,
    /// 环境光因子
    pub ambient_factor: Vector3<f32>,

    // ===== Phong着色模型专用属性 =====
    /// 镜面反射颜色
    pub specular: Vector3<f32>,
    /// 光泽度/硬度（Phong指数）
    pub shininess: f32,
    /// 漫反射强度系数 (0.0-2.0)
    pub diffuse_intensity: f32,
    /// 镜面反射强度系数 (0.0-2.0)
    pub specular_intensity: f32,

    // ===== PBR渲染专用属性 =====
    /// 金属度 (0.0=非金属, 1.0=纯金属)
    pub metallic: f32,
    /// 粗糙度 (0.0=完全光滑, 1.0=完全粗糙)
    pub roughness: f32,
    /// 环境光遮蔽 (0.0=完全遮蔽, 1.0=无遮蔽)
    pub ambient_occlusion: f32,

    // ===== PBR高级属性 =====
    /// 次表面散射强度 (0.0-1.0)
    pub subsurface: f32,
    /// 各向异性 (-1.0到1.0)
    pub anisotropy: f32,
    /// 法线强度系数 (0.0-2.0)
    pub normal_intensity: f32,
}

impl Material {
    pub fn default() -> Self {
        Material {
            // ===== 纹理资源 =====
            texture: None,
            normal_map: None,

            // ===== 通用材质属性 =====
            emissive: Vector3::zeros(),
            alpha: 1.0, // 默认完全不透明
            albedo: Vector3::new(0.8, 0.8, 0.8),
            ambient_factor: Vector3::new(1.0, 1.0, 1.0),

            // ===== Phong着色模型专用属性 =====
            specular: Vector3::new(0.5, 0.5, 0.5),
            shininess: 32.0,
            diffuse_intensity: 1.0,
            specular_intensity: 1.0,

            // ===== PBR渲染专用属性 =====
            metallic: 0.0,
            roughness: 0.5,
            ambient_occlusion: 1.0,

            // ===== PBR高级属性 =====
            subsurface: 0.0,       // 默认无次表面散射
            anisotropy: 0.0,       // 默认各向同性
            normal_intensity: 1.0, // 默认法线强度
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
    PBR(&'a Material),
}

impl MaterialView<'_> {
    /// 完整的PBR和Blinn-Phong实现，确保所有参数都参与计算
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
                let effective_normal = self.compute_effective_normal(
                    material,
                    surface_normal,
                    surface_tangent,
                    surface_bitangent,
                    surface_uv,
                );

                let n_dot_l = effective_normal.dot(light_dir).max(0.0);
                if n_dot_l <= 0.0 {
                    return material.emissive;
                }

                // Phong着色计算
                let diffuse = material.diffuse() * material.diffuse_intensity * n_dot_l;
                let halfway_dir = (light_dir + view_dir).normalize();
                let n_dot_h = effective_normal.dot(&halfway_dir).max(0.0);
                let spec_intensity = n_dot_h.powf(material.shininess);
                let specular = material.specular * material.specular_intensity * spec_intensity;

                // 🔥 Phong模式下的简化次表面散射（考虑透明度）
                let n_dot_v = effective_normal.dot(view_dir).max(0.0);
                let phong_subsurface = if material.subsurface > 0.0 {
                    pbr::calculate_subsurface_scattering(
                        n_dot_l,
                        n_dot_v,
                        material.subsurface * 0.6, // Phong模式下减弱强度
                        material.diffuse(),
                        material.alpha, // 🌟 传入透明度
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
                let alpha = material.alpha; // 🌟 获取透明度

                // 1. 计算有效法线
                let (n_final, tbn_matrix_option) = if let Some(normal_map) = &material.normal_map {
                    let n = surface_normal.normalize();
                    let t = (*surface_tangent - n * surface_tangent.dot(&n))
                        .try_normalize(1e-6)
                        .unwrap_or_else(|| pbr::compute_fallback_tangent(&n));
                    let b = n.cross(&t).normalize();
                    let tbn = Matrix3::from_columns(&[t, b, n]);

                    let normal_sample = normal_map.sample_normal(surface_uv.x, surface_uv.y);
                    let mut tangent_space_normal = Vector3::new(
                        normal_sample[0],
                        normal_sample[1],
                        normal_sample[2].max(0.1),
                    );

                    if material.normal_intensity != 1.0 {
                        tangent_space_normal.x *= material.normal_intensity;
                        tangent_space_normal.y *= material.normal_intensity;
                        let xy_length_sq = tangent_space_normal.x * tangent_space_normal.x
                            + tangent_space_normal.y * tangent_space_normal.y;
                        tangent_space_normal.z = (1.0 - xy_length_sq.min(1.0)).sqrt().max(0.01);
                    }

                    let normalized_tangent_normal = tangent_space_normal
                        .try_normalize(1e-6)
                        .unwrap_or_else(|| Vector3::new(0.0, 0.0, 1.0));

                    let world_normal = (tbn * normalized_tangent_normal).normalize();
                    (world_normal, Some(tbn))
                } else {
                    let processed_normal = if material.normal_intensity != 1.0 {
                        pbr::apply_procedural_normal_intensity(
                            *surface_normal,
                            material.normal_intensity,
                            *surface_uv,
                        )
                    } else {
                        *surface_normal
                    };

                    let tbn_opt = if anisotropy.abs() > 0.001 {
                        let n = processed_normal.normalize();
                        let t = pbr::compute_fallback_tangent(&n);
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

                // 3. F0计算 - 🌟 透明度影响F0
                let f0_dielectric = Vector3::new(0.04, 0.04, 0.04);
                // 透明材质通常有稍低的F0值
                let transparency_factor = 1.0 - alpha;
                let adjusted_f0_dielectric = f0_dielectric * (1.0 - transparency_factor * 0.3);
                let f0 = adjusted_f0_dielectric.lerp(&base_color, metallic);

                // 4. 分布函数选择（各向异性支持）
                let d = if anisotropy.abs() > 0.01 && tbn_matrix_option.is_some() {
                    let tbn = tbn_matrix_option.unwrap();
                    let t_surf = tbn.column(0).into_owned();
                    let b_surf = tbn.column(1).into_owned();

                    let h_dot_t = h.dot(&t_surf);
                    let h_dot_b = h.dot(&b_surf);

                    let (alpha_x, alpha_y) = pbr::apply_anisotropy(roughness, anisotropy);
                    pbr::distribution_ggx_anisotropic(n_dot_h, h_dot_t, h_dot_b, alpha_x, alpha_y)
                } else {
                    pbr::distribution_ggx(n_dot_h, roughness)
                };

                // 5. 几何函数
                let g = pbr::geometry_smith(n_dot_v, n_dot_l, roughness);
                let f = pbr::fresnel_schlick(h_dot_v, f0);

                // 6. BRDF计算
                let numerator = d * g * f;
                let denominator = 4.0 * n_dot_v * n_dot_l;
                let specular = numerator / denominator.max(0.001);

                let k_s = f;
                let k_d = (Vector3::new(1.0, 1.0, 1.0) - k_s) * (1.0 - metallic);
                let diffuse = k_d.component_mul(&base_color) / std::f32::consts::PI;

                // 7. 🔥 次表面散射计算（集成透明度）
                let subsurface_contrib = if subsurface > 0.0 && metallic < 0.5 {
                    pbr::calculate_subsurface_scattering(
                        n_dot_l, n_dot_v, subsurface, base_color,
                        alpha, // 🌟 传入透明度参数
                    )
                } else {
                    Vector3::zeros()
                };

                // 8. 🌟 透明度影响的金属增强效果
                let metallic_enhancement = if metallic > 0.3 {
                    let enhancement_factor = (metallic - 0.3) / 0.7;
                    // 透明金属有不同的视觉效果
                    let transparency_mod = if alpha < 0.8 {
                        1.2 // 透明金属增强反射
                    } else {
                        1.0
                    };
                    specular * enhancement_factor * 0.6 * transparency_mod
                } else {
                    Vector3::zeros()
                };

                // 9. 🌟 透明度影响环境光遮蔽
                // 透明材质的AO效果应该减弱
                let transparency_factor = 1.0 - alpha;
                let adjusted_ao = ao + transparency_factor * (1.0 - ao) * 0.4;

                // 10. 最终组合
                let brdf_result =
                    (diffuse + specular + metallic_enhancement) * n_dot_l * adjusted_ao
                        + subsurface_contrib;

                brdf_result + material.emissive
            }
        }
    }

    /// 统一的有效法线计算
    fn compute_effective_normal(
        &self,
        material: &Material,
        surface_normal: &Vector3<f32>,
        surface_tangent: &Vector3<f32>,
        surface_bitangent: &Vector3<f32>,
        surface_uv: &Vector2<f32>,
    ) -> Vector3<f32> {
        if let Some(normal_map) = &material.normal_map {
            // 从法线贴图计算
            let normal_sample = normal_map.sample_normal(surface_uv.x, surface_uv.y);

            let mut tangent_normal = Vector3::new(
                normal_sample[0],
                normal_sample[1],
                normal_sample[2].max(0.01), // 确保Z分量为正
            );

            // 应用法线强度
            if material.normal_intensity != 1.0 {
                tangent_normal.x *= material.normal_intensity;
                tangent_normal.y *= material.normal_intensity;
                let xy_length_sq =
                    tangent_normal.x * tangent_normal.x + tangent_normal.y * tangent_normal.y;
                tangent_normal.z = (1.0 - xy_length_sq.min(1.0)).sqrt().max(0.01);
            }

            let tangent_normal = tangent_normal
                .try_normalize(1e-6)
                .unwrap_or_else(|| Vector3::new(0.0, 0.0, 1.0));

            // 构建TBN矩阵并转换到世界空间
            let tbn = tbn::build_matrix(surface_normal, surface_tangent, surface_bitangent);
            (tbn * tangent_normal).normalize()
        } else {
            *surface_normal
        }
    }
}

// 完整的PBR函数模块，确保所有参数都有实现
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

    /// 各向异性GGX分布函数
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

    /// 各向异性参数计算
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

    /// 次表面散射计算
    /// 次表面散射计算 - 集成Alpha透明度混合
    pub fn calculate_subsurface_scattering(
        n_dot_l: f32,
        n_dot_v: f32,
        strength: f32,
        base_color: Vector3<f32>,
        alpha: f32, // 新增：透明度参数
    ) -> Vector3<f32> {
        if strength <= 0.0 {
            return Vector3::zeros();
        }

        // 1. 基础散射计算
        let view_scatter = (1.0 - n_dot_v).powi(2);
        let light_scatter = (1.0 - n_dot_l).powi(2);
        let scatter = (view_scatter + light_scatter) * 0.5 * strength;

        // 2. 透明度对次表面散射的影响
        // 透明度越高，散射效果越强（光线更容易穿透）
        let transparency_factor = 1.0 - alpha; // alpha=0时完全透明，transparency_factor=1
        let enhanced_scatter = scatter * (1.0 + transparency_factor * 0.8); // 透明时增强散射

        // 3. 透明度影响散射颜色的传播深度
        // 透明材质允许更深的光线穿透，产生更丰富的散射色彩
        let depth_factor = alpha + transparency_factor * 0.5; // 混合不透明和透明的散射深度

        // 4. 暖色调处理 - 透明度影响色温
        let base_warmth = Vector3::new(1.1, 0.9, 0.8);
        let transparency_warmth = Vector3::new(1.05, 0.95, 0.85); // 透明时稍微冷色调
        let warmth_factor = base_warmth.lerp(&transparency_warmth, transparency_factor);

        // 5. 散射颜色计算
        let subsurface_color = base_color.component_mul(&warmth_factor);

        // 6. 透明度调制最终强度
        // 完全不透明时：正常散射强度
        // 半透明时：增强散射模拟光线穿透
        // 完全透明时：轻微散射避免过度效果
        let final_intensity = if alpha > 0.8 {
            // 高不透明度：标准散射
            0.7 * depth_factor
        } else if alpha > 0.3 {
            // 中等透明度：增强散射
            0.9 * depth_factor
        } else {
            // 高透明度：适度散射避免过亮
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

    /// 程序化法线强度
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

// TBN模块
pub mod tbn {
    use super::{Point3, Vector2, Vector3, Vertex, pbr};
    use log::warn;
    use nalgebra::Matrix3;

    /// 构建TBN矩阵
    pub fn build_matrix(
        normal: &Vector3<f32>,
        tangent: &Vector3<f32>,
        bitangent: &Vector3<f32>,
    ) -> Matrix3<f32> {
        let n = normal.normalize();

        // Gram-Schmidt正交化
        let t_raw = *tangent;
        let t_orthogonal = (t_raw - n * t_raw.dot(&n))
            .try_normalize(1e-8)
            .unwrap_or_else(|| pbr::compute_fallback_tangent(&n));

        // 确保右手坐标系
        let b_computed = n.cross(&t_orthogonal);
        let b_normalized = bitangent.try_normalize(1e-8).unwrap_or(b_computed);
        let handedness = b_normalized.dot(&b_computed).signum();
        let final_bitangent = if handedness >= 0.0 {
            b_computed
        } else {
            -b_computed
        };

        Matrix3::from_columns(&[t_orthogonal, final_bitangent, n])
    }

    /// 计算切线和副切线
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

        // 计算每个三角形的TBN
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
                // 退化处理
                if let Some(normals) = normals {
                    let avg_normal = (normals[i0] + normals[i1] + normals[i2]) / 3.0;
                    if avg_normal.norm_squared() > 1e-15 {
                        let avg_normal = avg_normal.normalize();
                        let fallback_tangent = pbr::compute_fallback_tangent(&avg_normal);
                        let fallback_bitangent = avg_normal.cross(&fallback_tangent).normalize();

                        for &i in &[i0, i1, i2] {
                            tangents[i] += fallback_tangent;
                            bitangents[i] += fallback_bitangent;
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
                }
            }
        }

        // 正交化过程
        for i in 0..num_vertices {
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

    /// TBN验证和修复
    pub fn validate_and_fix(vertices: &mut [Vertex]) {
        for vertex in vertices.iter_mut() {
            // 法线验证
            if vertex.normal.norm_squared() < 1e-12 {
                vertex.normal = Vector3::y();
            } else {
                vertex.normal = vertex.normal.normalize();
            }

            // 切线验证和正交化
            if vertex.tangent.norm_squared() < 1e-12 {
                vertex.tangent = pbr::compute_fallback_tangent(&vertex.normal);
            } else {
                let t_raw = vertex.tangent;
                let n = vertex.normal;
                vertex.tangent = (t_raw - n * t_raw.dot(&n))
                    .try_normalize(1e-10)
                    .unwrap_or_else(|| pbr::compute_fallback_tangent(&n));
            }

            // 确保副切线正交性
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
            material.alpha = args.alpha.clamp(0.0, 1.0); // 新增：应用透明度

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
            material.alpha = args.alpha.clamp(0.0, 1.0); // 新增：应用透明度

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
