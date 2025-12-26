use crate::core::geometry::Vertex;
use crate::scene::material::{Material, PhongMaterial};
use crate::scene::mesh::Mesh;
use crate::scene::model::Model;
use crate::scene::texture::Texture;
use log::{debug, info, warn};
use nalgebra::{Point3, Vector2, Vector3};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Loads an OBJ file and returns a Model containing Meshes and Materials.
///
/// This function handles:
/// 1. Parsing .obj and .mtl files using `tobj`.
/// 2. Loading referenced textures.
/// 3. Unifying vertex indices (OBJ allows separate indices for pos/uv/norm, we need one).
/// 4. Generating smooth normals if missing.
pub fn load_obj(path: &str) -> Result<Model, String> {
    let obj_path = Path::new(path);
    if !obj_path.exists() {
        return Err(format!("OBJ file not found: {}", path));
    }

    let base_path = obj_path.parent().unwrap_or_else(|| Path::new("."));
    info!("Loading OBJ file: {:?}", obj_path);

    // Configure tobj
    // We set single_index to FALSE because we want to handle the index remapping manually.
    // This is necessary because OBJ allows v/vt/vn triplets to share position indices
    // but have different normal indices (e.g. for sharp edges), which `single_index=true`
    // might handle incorrectly or duplicate inefficiently without control.
    let load_options = tobj::LoadOptions {
        triangulate: true,
        single_index: false,
        ignore_points: true,
        ignore_lines: true,
    };

    let (models, materials_result) = tobj::load_obj(obj_path, &load_options)
        .map_err(|e| format!("Failed to load OBJ: {}", e))?;

    // 1. Process Materials
    let materials = process_materials(materials_result, base_path);

    // 2. Process Meshes
    let mut meshes = Vec::new();

    for model in models {
        let mesh_name = if model.name.is_empty() {
            "unnamed_mesh".to_string()
        } else {
            model.name.clone()
        };

        // Process geometry and unify indices
        match process_mesh(&model.mesh, &mesh_name) {
            Ok(mesh) => meshes.push(mesh),
            Err(e) => warn!("Skipping mesh '{}': {}", mesh_name, e),
        }
    }

    if meshes.is_empty() {
        return Err("No valid meshes found in OBJ file".to_string());
    }

    info!(
        "Model loaded successfully. Meshes: {}, Materials: {}",
        meshes.len(),
        materials.len()
    );

    Ok(Model::new(meshes, materials))
}

/// Converts tobj materials into our internal Material format.
fn process_materials(
    materials_result: Result<Vec<tobj::Material>, tobj::LoadError>,
    base_path: &Path,
) -> Vec<Material> {
    let mut materials = Vec::new();

    match materials_result {
        Ok(tobj_materials) => {
            for mat in tobj_materials {
                // Load Diffuse Texture if present
                let diffuse_texture = if let Some(tex_name) = &mat.diffuse_texture {
                    // Remove comments starting with '#' and trim whitespace
                    let clean_name = tex_name.split('#').next().unwrap_or(tex_name).trim();
                    let tex_path = base_path.join(clean_name);
                    match Texture::load(&tex_path) {
                        Ok(tex) => {
                            debug!("Loaded texture: {:?}", tex_path);
                            Some(Arc::new(tex))
                        }
                        Err(e) => {
                            warn!("Failed to load texture {:?}: {}", tex_path, e);
                            None
                        }
                    }
                } else {
                    None
                };

                // Create Phong Material
                let phong = PhongMaterial {
                    diffuse_color: Vector3::from(mat.diffuse.unwrap_or([0.8, 0.8, 0.8])),
                    specular_color: Vector3::from(mat.specular.unwrap_or([0.0, 0.0, 0.0])),
                    ambient_color: Vector3::from(mat.ambient.unwrap_or([0.1, 0.1, 0.1])),
                    shininess: mat.shininess.unwrap_or(32.0),
                    diffuse_texture,
                };

                // TODO: Future: Handle other material types (e.g. PBR) here.

                materials.push(Material::Phong(phong));
            }
        }
        Err(e) => {
            warn!("Failed to load materials (MTL): {}", e);
        }
    }

    // Ensure there is at least one default material
    if materials.is_empty() {
        materials.push(Material::default());
    }

    materials
}

/// Processes a single tobj mesh: handles index unification and normal generation.
fn process_mesh(tobj_mesh: &tobj::Mesh, name: &str) -> Result<Mesh, String> {
    if tobj_mesh.indices.is_empty() {
        return Err("Mesh has no indices".to_string());
    }

    let has_normals = !tobj_mesh.normals.is_empty();
    let has_uvs = !tobj_mesh.texcoords.is_empty();

    // If normals are missing, we generate them based on face connectivity.
    let generated_normals = if !has_normals {
        debug!(
            "Mesh '{}' missing normals, generating smooth normals...",
            name
        );
        generate_smooth_normals(&tobj_mesh.positions, &tobj_mesh.indices)
    } else {
        Vec::new()
    };

    let mut vertices: Vec<Vertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    // Map (pos_idx, norm_idx, uv_idx) -> new_vertex_index
    let mut index_map: HashMap<(u32, u32, u32), u32> = HashMap::new();

    let num_indices = tobj_mesh.indices.len();

    for i in 0..num_indices {
        let pos_idx = tobj_mesh.indices[i];

        // tobj returns separate index arrays if single_index=false.
        // If the specific index array is empty, it usually implies 0 or same as pos_idx depending on context,
        // but here we handle the logic explicitly.
        let norm_idx = if has_normals {
            tobj_mesh.normal_indices[i]
        } else {
            pos_idx // Use position index for generated normals
        };

        let uv_idx = if has_uvs {
            tobj_mesh.texcoord_indices[i]
        } else {
            0 // Dummy index
        };

        let key = (pos_idx, norm_idx, uv_idx);

        if let Some(&existing_idx) = index_map.get(&key) {
            indices.push(existing_idx);
        } else {
            // Create new vertex

            // 1. Position
            let px = tobj_mesh.positions[pos_idx as usize * 3];
            let py = tobj_mesh.positions[pos_idx as usize * 3 + 1];
            let pz = tobj_mesh.positions[pos_idx as usize * 3 + 2];
            let position = Point3::new(px, py, pz);

            // 2. Normal
            let normal = if has_normals {
                Vector3::new(
                    tobj_mesh.normals[norm_idx as usize * 3],
                    tobj_mesh.normals[norm_idx as usize * 3 + 1],
                    tobj_mesh.normals[norm_idx as usize * 3 + 2],
                )
            } else if !generated_normals.is_empty() {
                generated_normals[norm_idx as usize]
            } else {
                Vector3::y() // Fallback
            };

            // 3. UV
            let texcoord = if has_uvs {
                Vector2::new(
                    tobj_mesh.texcoords[uv_idx as usize * 2],
                    tobj_mesh.texcoords[uv_idx as usize * 2 + 1],
                )
            } else {
                Vector2::zeros()
            };

            let vertex = Vertex {
                position,
                normal,
                texcoord,
            };

            let new_idx = vertices.len() as u32;
            vertices.push(vertex);
            index_map.insert(key, new_idx);
            indices.push(new_idx);
        }
    }

    // Resolve material ID
    // tobj uses Option<usize>, our Mesh uses usize.
    // If None, we assign 0 (the default material we ensured exists).
    let material_id = tobj_mesh.material_id.unwrap_or(0);

    Ok(Mesh::new(vertices, indices, material_id))
}

/// Generates smooth vertex normals by averaging face normals.
fn generate_smooth_normals(positions: &[f32], indices: &[u32]) -> Vec<Vector3<f32>> {
    let num_vertices = positions.len() / 3;
    let mut normals = vec![Vector3::zeros(); num_vertices];

    // Iterate over faces (triangles)
    for chunk in indices.chunks(3) {
        if chunk.len() < 3 {
            continue;
        }
        let i0 = chunk[0] as usize;
        let i1 = chunk[1] as usize;
        let i2 = chunk[2] as usize;

        let v0 = Point3::new(
            positions[i0 * 3],
            positions[i0 * 3 + 1],
            positions[i0 * 3 + 2],
        );
        let v1 = Point3::new(
            positions[i1 * 3],
            positions[i1 * 3 + 1],
            positions[i1 * 3 + 2],
        );
        let v2 = Point3::new(
            positions[i2 * 3],
            positions[i2 * 3 + 1],
            positions[i2 * 3 + 2],
        );

        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        // Face normal
        let normal = edge1.cross(&edge2).normalize();

        // Accumulate normals
        normals[i0] += normal;
        normals[i1] += normal;
        normals[i2] += normal;
    }

    // Normalize accumulated normals
    for n in &mut normals {
        if n.norm_squared() > 1e-6 {
            n.normalize_mut();
        } else {
            *n = Vector3::y(); // Default up if degenerate
        }
    }

    normals
}
