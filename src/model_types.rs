use crate::texture_utils::Texture;
use nalgebra::{Point3, Vector2, Vector3};
use std::fmt::Debug;

/// 表示带有位置、法线和纹理坐标的顶点
#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    pub position: Point3<f32>,
    pub normal: Vector3<f32>,
    pub texcoord: Vector2<f32>,
}

/// 材质渲染模式
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterialMode {
    /// 传统Blinn-Phong着色模型
    BlinnPhong,
    /// 基于物理的渲染
    PBR,
}

/// 表示材质属性，包含传统的Blinn-Phong属性和PBR属性
#[derive(Debug, Clone)]
pub struct Material {
    // --- 通用属性 ---
    pub name: String,
    pub dissolve: f32, // 透明度 (1.0 = 不透明) // TODO

    // --- 传统Blinn-Phong属性 ---
    pub ambient: Vector3<f32>,
    pub diffuse: Vector3<f32>,
    pub specular: Vector3<f32>,
    pub shininess: f32,
    pub diffuse_texture: Option<Texture>,

    // --- PBR属性 ---
    pub base_color: Vector3<f32>,
    pub metallic: f32,
    pub roughness: f32,
    pub ambient_occlusion: f32,
    pub emissive: Vector3<f32>,
}

impl Material {
    /// 创建默认材质
    pub fn default() -> Self {
        Material {
            name: "Default".to_string(),
            dissolve: 1.0,
            ambient: Vector3::new(0.2, 0.2, 0.2),
            diffuse: Vector3::new(0.8, 0.8, 0.8),
            specular: Vector3::new(0.5, 0.5, 0.5),
            shininess: 32.0,
            diffuse_texture: None,
            base_color: Vector3::new(0.8, 0.8, 0.8),
            metallic: 0.0,
            roughness: 0.5,
            ambient_occlusion: 1.0,
            emissive: Vector3::zeros(),
        }
    }

    /// 获取材质名称
    pub fn get_name(&self) -> &str {
        &self.name
    }
}

/// 表示一个网格，包含顶点、索引和材质ID
#[derive(Debug, Clone)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    /// 指向`vertices`数组的索引，形成三角形
    pub indices: Vec<u32>,
    /// 指向`ModelData`中`materials`向量的索引
    pub material_id: Option<usize>,
}

/// 保存所有加载的模型数据，包括网格和材质
#[derive(Debug, Clone)]
pub struct ModelData {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
}
