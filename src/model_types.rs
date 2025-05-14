use crate::texture_utils::Texture;
use nalgebra::{Point3, Vector2, Vector3};

/// Represents a vertex with position, normal, and texture coordinates.
#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    pub position: Point3<f32>,
    pub normal: Vector3<f32>,
    pub texcoord: Vector2<f32>,
}

/// Represents a material with its properties and optional texture map.
#[derive(Debug, Clone)]
pub struct Material {
    pub name: String,
    pub ambient: Vector3<f32>,  // Ka
    pub diffuse: Vector3<f32>,  // Kd
    pub specular: Vector3<f32>, // Ks
    pub shininess: f32,         // Ns
    pub dissolve: f32,          // Alpha / transparency
    pub diffuse_texture: Option<Texture>,
    // 添加 PBR 材质支持
    pub pbr_material: Option<crate::material_system::PBRMaterial>,
    // Add other properties like ambient_texture, specular_texture, bump_map etc. if needed
}

impl Material {
    /// Creates a default material.
    pub fn default() -> Self {
        // Made public
        Material {
            name: "Default".to_string(),
            ambient: Vector3::new(0.2, 0.2, 0.2),
            diffuse: Vector3::new(0.8, 0.8, 0.8),
            specular: Vector3::new(0.0, 0.0, 0.0),
            shininess: 10.0,
            dissolve: 1.0,
            diffuse_texture: None,
            pbr_material: None, // Initialize with None
        }
    }

    /// 返回材质名称
    pub fn get_name(&self) -> &str {
        &self.name
    }

    /// 设置材质名称
    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }

    /// 检查材质是否完全不透明
    pub fn is_opaque(&self) -> bool {
        self.dissolve >= 0.999
    }

    /// 获取材质的透明度值 (0.0 = 完全透明, 1.0 = 完全不透明)
    pub fn get_opacity(&self) -> f32 {
        self.dissolve
    }

    /// 设置材质的透明度值
    pub fn set_opacity(&mut self, opacity: f32) {
        self.dissolve = opacity.clamp(0.0, 1.0);
    }

    /// 创建一个PBR材质版本（如果尚未存在）
    pub fn ensure_pbr_material(&mut self) -> &mut crate::material_system::PBRMaterial {
        if self.pbr_material.is_none() {
            // 使用现有Blinn-Phong参数创建合理的PBR材质
            let base_color = self.diffuse;
            // 使用高光作为金属度的粗略估计 - 计算高光颜色的最大分量
            let metallic = self.specular.x.max(self.specular.y).max(self.specular.z);
            let roughness = 1.0 - (self.shininess / 128.0).clamp(0.0, 1.0); // 将光泽度转换为粗糙度

            let pbr_material =
                crate::material_system::PBRMaterial::new(base_color, metallic, roughness);

            self.pbr_material = Some(pbr_material);
        }

        self.pbr_material.as_mut().unwrap()
    }
}

/// Represents a mesh with vertices, indices, and material ID.
#[derive(Debug, Clone)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    /// Indices into the `vertices` array, forming triangles.
    pub indices: Vec<u32>,
    /// Index into the `materials` vector in `ModelData`.
    pub material_id: Option<usize>,
}

/// Holds all loaded model data, including meshes and materials.
#[derive(Debug, Clone)]
pub struct ModelData {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
}
