use crate::materials::texture::Texture;
use nalgebra::{Point3, Vector2, Vector3};
use std::fmt::Debug;

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
    pub path: Option<std::path::PathBuf>,
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
