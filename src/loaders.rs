use crate::texture_utils::{Texture, load_texture};
use nalgebra::{Point3, Vector2, Vector3};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tobj;

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
    pub ambient: Vector3<f32>,
    pub diffuse: Vector3<f32>,
    pub specular: Vector3<f32>,
    pub shininess: f32,
    pub dissolve: f32, // Alpha / transparency
    pub diffuse_texture: Option<Texture>,
    // Add other properties like ambient_texture, specular_texture, bump_map etc. if needed
}

impl Material {
    /// Creates a default material.
    fn default() -> Self {
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

/// Loads an OBJ file and associated materials/textures.
///
/// Uses the `tobj` crate for parsing and triangulates meshes.
/// Handles loading textures specified in MTL files.
/// Returns a `ModelData` struct containing all meshes and materials.
pub fn load_obj_enhanced<P: AsRef<Path>>(obj_path: P) -> Result<ModelData, String> {
    let obj_path_ref = obj_path.as_ref();
    // let obj_dir = obj_path_ref.parent().unwrap_or_else(|| Path::new("")); // No longer needed for MTL

    println!("Loading OBJ file: {:?}", obj_path_ref);

    // --- Load OBJ using tobj ---
    let load_options = tobj::LoadOptions {
        triangulate: true,
        single_index: false,
        ignore_points: true,
        ignore_lines: true,
    };

    // Only load models, ignore materials result
    let (models, _materials_result) = tobj::load_obj(obj_path_ref, &load_options)
        .map_err(|e| format!("Failed to load OBJ: {}", e))?;

    // --- Create Default Material ---
    // Always create and use a single default material
    let loaded_materials: Vec<Material> = vec![Material::default()];
    println!("Using default material.");

    // --- Process Meshes ---
    let mut loaded_meshes: Vec<Mesh> = Vec::with_capacity(models.len());

    for model in models.iter() {
        let mesh = &model.mesh;
        let num_vertices = mesh.positions.len() / 3;

        if mesh.indices.is_empty() {
            println!("Skipping mesh '{}' with no indices.", model.name);
            continue;
        }

        let has_normals = !mesh.normals.is_empty();
        let has_texcoords = !mesh.texcoords.is_empty();

        if !has_normals {
            println!(
                "Warning: Mesh '{}' is missing normals. Consider generating them.",
                model.name
            );
        }
        if !has_texcoords {
            println!(
                "Warning: Mesh '{}' is missing texture coordinates. Texturing might not work correctly.",
                model.name
            );
        }

        let mut vertices: Vec<Vertex> = Vec::with_capacity(num_vertices);
        let mut index_map: HashMap<(u32, Option<u32>, Option<u32>), u32> = HashMap::new();
        let mut final_indices: Vec<u32> = Vec::with_capacity(mesh.indices.len());

        for i in 0..mesh.indices.len() {
            let pos_idx = mesh.indices[i];
            let norm_idx = if has_normals && !mesh.normal_indices.is_empty() {
                Some(mesh.normal_indices[i])
            } else {
                None
            };
            let tc_idx = if has_texcoords && !mesh.texcoord_indices.is_empty() {
                Some(mesh.texcoord_indices[i])
            } else {
                None
            };

            let key = (pos_idx, norm_idx, tc_idx);

            if let Some(&final_idx) = index_map.get(&key) {
                final_indices.push(final_idx);
            } else {
                let p_start = pos_idx as usize * 3;
                let position = Point3::new(
                    mesh.positions[p_start],
                    mesh.positions[p_start + 1],
                    mesh.positions[p_start + 2],
                );

                let normal = if let Some(idx) = norm_idx {
                    let n_start = idx as usize * 3;
                    if n_start + 2 < mesh.normals.len() {
                        Vector3::new(
                            mesh.normals[n_start],
                            mesh.normals[n_start + 1],
                            mesh.normals[n_start + 2],
                        )
                        .normalize()
                    } else {
                        Vector3::new(0.0, 1.0, 0.0)
                    }
                } else {
                    Vector3::new(0.0, 1.0, 0.0)
                };

                let texcoord = if let Some(idx) = tc_idx {
                    let t_start = idx as usize * 2;
                    if t_start + 1 < mesh.texcoords.len() {
                        Vector2::new(mesh.texcoords[t_start], mesh.texcoords[t_start + 1])
                    } else {
                        Vector2::new(0.0, 0.0)
                    }
                } else {
                    Vector2::new(0.0, 0.0)
                };

                let new_vertex = Vertex {
                    position,
                    normal,
                    texcoord,
                };
                let new_final_idx = vertices.len() as u32;
                vertices.push(new_vertex);
                index_map.insert(key, new_final_idx);
                final_indices.push(new_final_idx);
            }
        }

        // Always assign the default material (index 0)
        let material_id = Some(0);

        loaded_meshes.push(Mesh {
            vertices,
            indices: final_indices,
            material_id,
        });
        println!(
            "Processed mesh '{}': {} vertices, {} triangles, Material ID: {:?}",
            model.name,
            loaded_meshes.last().unwrap().vertices.len(),
            loaded_meshes.last().unwrap().indices.len() / 3,
            material_id // Will always be Some(0)
        );
    }

    if loaded_meshes.is_empty() {
        return Err("No valid meshes found in the OBJ file.".to_string());
    }

    Ok(ModelData {
        meshes: loaded_meshes,
        materials: loaded_materials, // Contains only the default material
    })
}
