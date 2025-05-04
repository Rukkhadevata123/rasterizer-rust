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
    #[allow(dead_code)]
    pub name: String,
    pub ambient: Vector3<f32>,
    pub diffuse: Vector3<f32>,
    pub specular: Vector3<f32>,
    pub shininess: f32,
    #[allow(dead_code)]
    pub dissolve: f32, // Alpha / transparency
    pub diffuse_texture: Option<Texture>,
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
        }
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
