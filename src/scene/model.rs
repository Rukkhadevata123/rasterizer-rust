use crate::scene::material::Material;
use crate::scene::mesh::Mesh;

/// Meshes and their shared material table for one imported model.
pub struct Model {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
}

impl Model {
    pub fn new(meshes: Vec<Mesh>, materials: Vec<Material>) -> Self {
        Self { meshes, materials }
    }
}
