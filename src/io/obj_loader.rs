use crate::core::geometry::Vertex;
use crate::scene::material::{Material, PbrMaterial};
use crate::scene::mesh::Mesh;
use crate::scene::model::Model;
use crate::scene::texture::Texture;
use log::{debug, info, warn};
use nalgebra::{Point3, Vector2, Vector3};
use std::collections::HashMap;
use std::f32::consts::PI;
use std::path::Path;
use std::sync::Arc;

/// Loads an OBJ file and returns a Model containing Meshes and Materials.
///
/// This function handles:
/// 1. Parsing .obj and .mtl files using `tobj`.
/// 2. Loading referenced textures.
/// 3. Unifying vertex indices (OBJ allows separate indices for pos/uv/norm, we need one).
/// 4. Generating smooth normals if missing.
pub fn load_obj(path: &str, use_mipmap: bool) -> Result<Model, String> {
    let obj_path = Path::new(path);
    if !obj_path.exists() {
        return Err(format!("OBJ file not found: {}", path));
    }

    let base_path = obj_path.parent().unwrap_or_else(|| Path::new("."));
    info!("Loading OBJ file: {:?}", obj_path);

    let load_options = tobj::LoadOptions {
        triangulate: true,
        single_index: false,
        ignore_points: true,
        ignore_lines: true,
    };

    let (models, materials_result) = tobj::load_obj(obj_path, &load_options)
        .map_err(|e| format!("Failed to load OBJ: {}", e))?;

    // 1. Process Materials (Now returns PBR materials)
    let materials = process_materials(materials_result, base_path, use_mipmap);

    // 2. Process Meshes
    let mut meshes = Vec::new();
    for model in models {
        let mesh_name = if model.name.is_empty() {
            "unnamed_mesh".to_string()
        } else {
            model.name.clone()
        };
        match process_mesh(&model.mesh, &mesh_name) {
            Ok(mesh) => meshes.push(mesh),
            Err(e) => warn!("Skipping mesh '{}': {}", mesh_name, e),
        }
    }

    if meshes.is_empty() {
        return Err("No valid meshes found in OBJ file".to_string());
    }

    Ok(Model::new(meshes, materials))
}

/// Converts tobj materials directly into PBR Materials.
fn process_materials(
    materials_result: Result<Vec<tobj::Material>, tobj::LoadError>,
    base_path: &Path,
    use_mipmap: bool,
) -> Vec<Material> {
    let mut materials = Vec::new();

    match materials_result {
        Ok(tobj_materials) => {
            for mat in tobj_materials {
                // 1. Load Albedo Texture (map_Kd)
                let albedo_texture = if let Some(tex_name) = &mat.diffuse_texture {
                    let clean_name = tex_name.split('#').next().unwrap_or(tex_name).trim();
                    let tex_path = base_path.join(clean_name);
                    Texture::load(&tex_path, use_mipmap).ok().map(Arc::new)
                } else {
                    None
                };
                // 2. Map Parameters
                // Kd -> Albedo
                let albedo = Vector3::from(mat.diffuse.unwrap_or([0.8, 0.8, 0.8]));

                // Ns (Shininess 0-1000) -> Roughness (0-1)
                // Heuristic: High shininess = Low roughness
                let shininess = mat.shininess.unwrap_or(32.0);
                let roughness = (1.0 - (shininess / 100.0).clamp(0.0, 1.0)).max(0.05);

                // Standard OBJ materials are usually non-metals
                let metallic = 0.0;

                let pbr = PbrMaterial {
                    albedo,
                    metallic,
                    roughness,
                    ao: 1.0,
                    emissive: Vector3::zeros(),
                    albedo_texture,
                    metallic_roughness_texture: None, // OBJ usually doesn't have this
                    normal_texture: None,
                };

                materials.push(Material::Pbr(pbr));
            }
        }
        Err(e) => {
            warn!("Failed to load materials (MTL): {}", e);
        }
    }

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
    let mut index_map: HashMap<(u32, u32, u32), u32> = HashMap::new();

    let num_indices = tobj_mesh.indices.len();

    for i in 0..num_indices {
        let pos_idx = tobj_mesh.indices[i];
        let norm_idx = if has_normals {
            tobj_mesh.normal_indices[i]
        } else {
            pos_idx
        };
        // Note: Even if there are no UVs, we don't rely on uv_idx anymore
        let uv_idx = if has_uvs {
            tobj_mesh.texcoord_indices[i]
        } else {
            0
        };

        let key = (pos_idx, norm_idx, uv_idx);

        if let Some(&existing_idx) = index_map.get(&key) {
            indices.push(existing_idx);
        } else {
            let px = tobj_mesh.positions[pos_idx as usize * 3];
            let py = tobj_mesh.positions[pos_idx as usize * 3 + 1];
            let pz = tobj_mesh.positions[pos_idx as usize * 3 + 2];
            let position = Point3::new(px, py, pz);

            let normal = if has_normals {
                Vector3::new(
                    tobj_mesh.normals[norm_idx as usize * 3],
                    tobj_mesh.normals[norm_idx as usize * 3 + 1],
                    tobj_mesh.normals[norm_idx as usize * 3 + 2],
                )
            } else if !generated_normals.is_empty() {
                generated_normals[norm_idx as usize]
            } else {
                Vector3::y()
            };

            let texcoord = if has_uvs {
                // Case A: UVs exist in file, read normally
                Vector2::new(
                    tobj_mesh.texcoords[uv_idx as usize * 2],
                    tobj_mesh.texcoords[uv_idx as usize * 2 + 1],
                )
            } else {
                // Case B: No UVs in file, perform spherical projection
                // Assume model is centered at origin (normalize_and_center_model ensures this)
                // Algorithm: Convert normalized direction vector to latitude/longitude
                let dir = position.coords;
                let len = dir.norm();

                if len > 1e-6 {
                    // atan2(z, x) gives longitude (-PI to PI)
                    let phi = dir.z.atan2(dir.x);
                    // asin(y / len) gives latitude (-PI/2 to PI/2)
                    let theta = (dir.y / len).clamp(-1.0, 1.0).asin();

                    // Map to [0, 1]
                    let u = (phi + PI) / (2.0 * PI);
                    let v = 0.5 - (theta / PI); // 0.5 - ... to put poles at top/bottom and match typical texture direction

                    Vector2::new(u, v)
                } else {
                    Vector2::zeros()
                }
            };

            let vertex = Vertex {
                position,
                normal,
                texcoord,
                tangent: Vector3::zeros(), // Initialize tangent as zero vector
            };
            let new_idx = vertices.len() as u32;
            vertices.push(vertex);
            index_map.insert(key, new_idx);
            indices.push(new_idx);
        }
    }
    // Previously: if has_uvs { ... }
    // Now: Since we guarantee all vertices have UVs (original + auto-generated), we can always calculate
    // Only when triangles are extremely degenerate will calculate_tangents handle fallback internally
    if !vertices.is_empty() {
        if !has_uvs {
            debug!("Generated spherical UVs for mesh '{}'", name);
        }
        calculate_tangents(&mut vertices, &indices);
    }

    let material_id = tobj_mesh.material_id.unwrap_or(0);
    Ok(Mesh::new(vertices, indices, material_id))
}

fn generate_smooth_normals(positions: &[f32], indices: &[u32]) -> Vec<Vector3<f32>> {
    let num_vertices = positions.len() / 3;
    let mut normals = vec![Vector3::zeros(); num_vertices];

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
        let normal = edge1.cross(&edge2).normalize();

        normals[i0] += normal;
        normals[i1] += normal;
        normals[i2] += normal;
    }

    for n in &mut normals {
        if n.norm_squared() > 1e-6 {
            n.normalize_mut();
        } else {
            *n = Vector3::y();
        }
    }
    normals
}

/// Calculates tangent vectors using the matrix inversion method (Eric Lengyel).
///
/// Math derivation:
/// [ E1 ]   [ du1  dv1 ] [ T ]
/// [    ] = [          ] [   ]
/// [ E2 ]   [ du2  dv2 ] [ B ]
///
/// To solve for T, we multiply by the inverse of the UV matrix.
/// We only solve for T here to keep the Vertex struct simple (Vector3).
fn calculate_tangents(vertices: &mut [Vertex], indices: &[u32]) {
    // Accumulator for tangents per vertex
    let mut tan_sum = vec![Vector3::zeros(); vertices.len()];

    // 1. Accumulate tangents from triangles
    for chunk in indices.chunks(3) {
        if chunk.len() < 3 {
            continue;
        }

        let i0 = chunk[0] as usize;
        let i1 = chunk[1] as usize;
        let i2 = chunk[2] as usize;

        let v0 = &vertices[i0];
        let v1 = &vertices[i1];
        let v2 = &vertices[i2];

        // Edge vectors (E1, E2)
        let edge1 = v1.position - v0.position;
        let edge2 = v2.position - v0.position;

        // UV Delta vectors (Delta UV1, Delta UV2)
        let duv1 = v1.texcoord - v0.texcoord;
        let duv2 = v2.texcoord - v0.texcoord;

        // Calculate determinant of the UV matrix
        // det = du1 * dv2 - du2 * dv1
        let det = duv1.x * duv2.y - duv2.x * duv1.y;

        // Skip degenerate triangles (where UVs form a line or point)
        if det.abs() < 1e-8 {
            continue;
        }

        let inv_det = 1.0 / det;

        // Solve for Tangent (T)
        // T = (dv2 * E1 - dv1 * E2) * inv_det
        // Note: We use vector math directly here for precision and readability.
        let tangent = (edge1 * duv2.y - edge2 * duv1.y) * inv_det;

        // Accumulate
        tan_sum[i0] += tangent;
        tan_sum[i1] += tangent;
        tan_sum[i2] += tangent;
    }

    // 2. Orthogonalize (Gram-Schmidt)
    for (i, vert) in vertices.iter_mut().enumerate() {
        let n = vert.normal; // Assumed to be normalized already
        let t = tan_sum[i];

        // Gram-Schmidt orthogonalization:
        // T_ortho = T - N * (N . T)
        // This effectively projects T onto the plane defined by N.
        let ortho_t = t - n * n.dot(&t);

        // Normalize safe
        if ortho_t.norm_squared() > 1e-8 {
            vert.tangent = ortho_t.normalize();
        } else {
            // Fallback if tangent degenerates (e.g. extremely distorted UVs)
            // Any vector perpendicular to N works.
            vert.tangent = if n.x.abs() < 0.9 {
                Vector3::x().cross(&n).normalize()
            } else {
                Vector3::y().cross(&n).normalize()
            };
        }
    }
}
