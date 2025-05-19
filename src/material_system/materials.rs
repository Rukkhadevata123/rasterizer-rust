use crate::io::args::{Args, parse_vec3};
use crate::material_system::texture::Texture; // 假设 texture.rs 仍然存在于 materials 模块下
use nalgebra::{Point3, Vector2, Vector3};
use std::fmt::Debug;
use std::path::PathBuf; // 为 TextureOptions 添加

// --- 从原 material_types.rs 移入的类型定义 ---

/// 表示带有位置、法线和纹理坐标的顶点
#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    pub position: Point3<f32>,
    pub normal: Vector3<f32>,
    pub texcoord: Vector2<f32>,
}

/// 表示材质属性，包含渲染所需的各种属性
#[derive(Debug, Clone)]
pub struct Material {
    // --- 通用属性 ---
    pub name: String,
    /// 统一的纹理接口，可以是图像纹理、单色纹理或面颜色纹理
    pub texture: Option<Texture>,
    /// 自发光颜色，对两种渲染模式都适用
    pub emissive: Vector3<f32>,

    // --- 着色模型共享属性 ---
    /// 基础颜色，在Blinn-Phong为漫反射颜色，在PBR为基础颜色
    pub albedo: Vector3<f32>,
    /// 环境光响应系数，控制材质对环境光的反应程度
    pub ambient_factor: Vector3<f32>,

    // --- Blinn-Phong渲染专用属性 ---
    /// 镜面反射颜色
    pub specular: Vector3<f32>,
    /// 光泽度，值越大高光越小越集中
    pub shininess: f32,

    // --- PBR渲染专用属性 ---
    /// 金属度，0.0为非金属，1.0为金属
    pub metallic: f32,
    /// 粗糙度，0.0为完全光滑，1.0为完全粗糙
    pub roughness: f32,
    /// 环境光遮蔽，1.0为无遮蔽
    pub ambient_occlusion: f32,
}

impl Material {
    /// 创建默认材质
    pub fn default() -> Self {
        Material {
            name: "Default".to_string(),
            albedo: Vector3::new(0.8, 0.8, 0.8),
            specular: Vector3::new(0.5, 0.5, 0.5),
            shininess: 32.0,
            texture: None,
            emissive: Vector3::zeros(),
            metallic: 0.0,
            roughness: 0.5,
            ambient_occlusion: 1.0,
            ambient_factor: Vector3::new(1.0, 1.0, 1.0), // 默认环境光响应系数
        }
    }

    /// 配置材质的纹理
    /// 提供一个统一的接口来设置不同类型的纹理
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
                            println!("无法加载纹理，保持当前纹理设置");
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
            _ => println!("未知的纹理类型: {}", texture_type),
        }
        self
    }

    /// 获取材质名称
    pub fn get_name(&self) -> &str {
        &self.name
    }

    /// 获取漫反射颜色 (Blinn-Phong模式)
    pub fn diffuse(&self) -> Vector3<f32> {
        self.albedo
    }

    /// 获取基础颜色 (PBR模式)
    pub fn base_color(&self) -> Vector3<f32> {
        self.albedo
    }
}

/// 纹理配置选项
#[derive(Debug, Clone)]
pub struct TextureOptions {
    pub path: Option<PathBuf>,
    pub color: Option<Vector3<f32>>,
}

/// 表示一个网格，包含顶点、索引和材质ID
#[derive(Debug, Clone)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    /// 指向`vertices`数组的索引，形成三角形
    pub indices: Vec<u32>,
    /// 指向`ModelData`中`materials`向量的索引
    pub material_id: Option<usize>,
    /// 网格的名称
    pub name: String,
}

/// 保存所有加载的模型数据，包括网格和材质
#[derive(Debug, Clone)]
pub struct ModelData {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
    /// 模型的名称，通常来自文件名
    pub name: String,
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
                let n_dot_l = normal.dot(light_dir).max(0.0);
                if n_dot_l <= 0.0 {
                    return material.emissive;
                }
                let diffuse = material.diffuse() * n_dot_l;
                let halfway_dir = (light_dir + view_dir).normalize();
                let n_dot_h = normal.dot(&halfway_dir).max(0.0);
                let spec_intensity = n_dot_h.powf(material.shininess);
                let specular = material.specular * spec_intensity;
                diffuse + specular + material.emissive
            }
            MaterialView::PBR(material) => {
                use pbr_functions::*; // 使用当前文件内的模块

                // --- 基于物理的渲染 (PBR) ---
                let base_color = material.base_color();
                let metallic = material.metallic;
                let roughness = material.roughness;
                let ao = material.ambient_occlusion;

                let n = *normal;
                let l = *light_dir;
                let v = *view_dir;
                let h = (l + v).normalize();

                let n_dot_l = n.dot(&l).max(0.0);
                let n_dot_v = n.dot(&v).max(0.0);
                let n_dot_h = n.dot(&h).max(0.0);
                let h_dot_v = h.dot(&v).max(0.0);

                if n_dot_l <= 0.0 {
                    return material.emissive;
                }

                let f0_dielectric = Vector3::new(0.04, 0.04, 0.04);
                let f0 = f0_dielectric.lerp(&base_color, metallic);

                let d = distribution_ggx(n_dot_h, roughness);
                let f = fresnel_schlick(h_dot_v, f0);
                let g = geometry_smith(n_dot_v, n_dot_l, roughness);

                let numerator = d * g * f;
                let denominator = 4.0 * n_dot_v * n_dot_l;
                let specular = numerator / denominator.max(0.001f32);

                let k_s = f;
                let one = Vector3::new(1.0, 1.0, 1.0);

                let k_d_components = Vector3::new(
                    (one[0] - k_s[0]) * (1.0 - metallic),
                    (one[1] - k_s[1]) * (1.0 - metallic),
                    (one[2] - k_s[2]) * (1.0 - metallic),
                );

                let diffuse = Vector3::new(
                    k_d_components[0] * base_color[0],
                    k_d_components[1] * base_color[1],
                    k_d_components[2] * base_color[2],
                ) / std::f32::consts::PI;

                let brdf_result = Vector3::new(
                    (diffuse[0] + specular[0]) * n_dot_l * ao,
                    (diffuse[1] + specular[1]) * n_dot_l * ao,
                    (diffuse[2] + specular[2]) * n_dot_l * ao,
                );

                brdf_result + material.emissive
            }
        }
    }
}

/// 材质参数应用相关函数
pub mod material_applicator {
    use super::{Args, ModelData, Vector3, parse_vec3}; // 使用 super 访问同级模块的类型

    /// 应用PBR材质参数
    pub fn apply_pbr_parameters(model_data: &mut ModelData, args: &Args) {
        for material in &mut model_data.materials {
            material.metallic = args.metallic.clamp(0.0, 1.0);
            material.roughness = args.roughness.clamp(0.0, 1.0);
            material.ambient_occlusion = args.ambient_occlusion.clamp(0.0, 1.0);

            if let Ok(base_color) = parse_vec3(&args.base_color) {
                material.albedo = base_color;
                if material.metallic < 0.1 {
                    material.albedo = Vector3::new(
                        material.albedo.x.min(0.9),
                        material.albedo.y.min(0.9),
                        material.albedo.z.min(0.9),
                    );
                }
            } else {
                println!("警告: 无法解析基础颜色, 使用默认值: {:?}", material.albedo);
            }

            if let Ok(emissive) = parse_vec3(&args.emissive) {
                material.emissive = emissive;
            }

            let ambient_response = material.ambient_occlusion * (1.0 - material.metallic);
            material.ambient_factor =
                Vector3::new(ambient_response, ambient_response, ambient_response);

            println!(
                "应用PBR材质 - 基础色: {:?}, 金属度: {:.2}, 粗糙度: {:.2}, 环境光遮蔽: {:.2}, 自发光: {:?}",
                material.base_color(),
                material.metallic,
                material.roughness,
                material.ambient_occlusion,
                material.emissive
            );
        }
    }

    /// 应用Phong材质参数
    pub fn apply_phong_parameters(model_data: &mut ModelData, args: &Args) {
        for material in &mut model_data.materials {
            material.specular = Vector3::new(args.specular, args.specular, args.specular);
            material.shininess = args.shininess.max(1.0);

            if let Ok(diffuse_color) = parse_vec3(&args.diffuse_color) {
                material.albedo = diffuse_color;
            } else {
                println!(
                    "警告: 无法解析漫反射颜色, 使用默认值: {:?}",
                    material.diffuse()
                );
            }

            if let Ok(emissive) = parse_vec3(&args.emissive) {
                material.emissive = emissive;
            }
            material.ambient_factor = material.albedo * 0.3;

            println!(
                "应用Phong材质 - 漫反射: {:?}, 镜面: {:?}, 光泽度: {:.2}, 自发光: {:?}",
                material.diffuse(),
                material.specular,
                material.shininess,
                material.emissive
            );
        }
    }
}

/// PBR材质函数库
pub mod pbr_functions {
    use nalgebra::Vector3; // nalgebra::Vector3 已在文件顶部导入，但模块内显式导入更清晰

    /// 正态分布函数 (GGX/Trowbridge-Reitz)
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
    pub fn fresnel_schlick(cos_theta: f32, f0: Vector3<f32>) -> Vector3<f32> {
        let one_minus_cos_theta = 1.0 - cos_theta;
        let one_minus_cos_theta2 = one_minus_cos_theta * one_minus_cos_theta;
        let one_minus_cos_theta5 =
            one_minus_cos_theta2 * one_minus_cos_theta2 * one_minus_cos_theta;
        f0 + (Vector3::new(1.0, 1.0, 1.0) - f0) * one_minus_cos_theta5
    }
}
