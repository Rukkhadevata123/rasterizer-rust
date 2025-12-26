use crate::scene::material::Material;
use crate::scene::mesh::Mesh;

/// A Model represents a complete 3D object.
/// It consists of one or more Meshes and a list of Materials.
pub struct Model {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
}

impl Model {
    pub fn new(meshes: Vec<Mesh>, materials: Vec<Material>) -> Self {
        Self { meshes, materials }
    }
}
