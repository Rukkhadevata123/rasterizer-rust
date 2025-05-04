use crate::args::Args;
use crate::texture_utils::{Texture, load_texture};
use nalgebra::{Point3, Vector2, Vector3};
use std::collections::HashMap;
use std::path::Path;
use tobj; // 添加 Args 导入

// Moved Vertex, Material, Mesh, ModelData to model_types.rs
use crate::model_types::{Material, Mesh, ModelData, Vertex};

// Function remains in loaders.rs
/// Calculates smooth vertex normals by averaging face normals.
/// Mimics the Python function `generate_vertex_normals`.
///
/// # Arguments
/// * `vertices`: Slice of vertex positions.
/// * `indices`: Slice of triangle indices (groups of 3).
///
/// # Returns
/// A vector containing the calculated normal for each vertex, or an error string.
fn generate_smooth_vertex_normals(
    vertices: &[Point3<f32>],
    indices: &[u32],
) -> Result<Vec<Vector3<f32>>, String> {
    if indices.len() % 3 != 0 {
        return Err("Indices length must be a multiple of 3 for triangles.".to_string());
    }
    if vertices.is_empty() {
        return Ok(Vec::new()); // No vertices, no normals
    }

    let num_vertices = vertices.len();
    let num_faces = indices.len() / 3;
    let mut vertex_normals = vec![Vector3::zeros(); num_vertices];

    // 1. Calculate face normals and accumulate them onto vertices
    for i in 0..num_faces {
        let idx0 = indices[i * 3] as usize;
        let idx1 = indices[i * 3 + 1] as usize;
        let idx2 = indices[i * 3 + 2] as usize;

        // Basic bounds check
        if idx0 >= num_vertices || idx1 >= num_vertices || idx2 >= num_vertices {
            println!(
                "Warning: Face {} has out-of-bounds vertex index. Skipping.",
                i
            );
            continue;
        }

        let v0 = vertices[idx0];
        let v1 = vertices[idx1];
        let v2 = vertices[idx2];

        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let face_normal = edge1.cross(&edge2);

        // No need to normalize face normal before accumulation,
        // magnitude contributes to weighting (larger triangles have more influence).
        // Accumulate directly. This is safe in Rust's single-threaded context.
        vertex_normals[idx0] += face_normal;
        vertex_normals[idx1] += face_normal;
        vertex_normals[idx2] += face_normal;
    }

    // 2. Normalize vertex normals
    let mut zero_norm_count = 0;
    for normal in vertex_normals.iter_mut() {
        let norm_squared = normal.norm_squared();
        if norm_squared > 1e-12 {
            // Use squared norm to avoid sqrt
            normal.normalize_mut(); // Normalize in place
        } else {
            // Handle zero-length normals (e.g., vertex not used or part of degenerate faces)
            *normal = Vector3::y(); // Default to up vector
            zero_norm_count += 1;
        }
    }

    if zero_norm_count > 0 {
        println!(
            "Warning: {} vertices had zero normals, set to default [0, 1, 0].",
            zero_norm_count
        );
    }

    Ok(vertex_normals)
}

// Function remains in loaders.rs
pub fn load_obj_enhanced<P: AsRef<Path>>(obj_path: P, args: &Args) -> Result<ModelData, String> {
    let obj_path_ref = obj_path.as_ref();
    println!("Loading OBJ file: {:?}", obj_path_ref);

    // Determine the base path for loading materials and textures
    let base_path = obj_path_ref.parent().unwrap_or_else(|| Path::new("."));

    // --- 检查是否有命令行指定的纹理 ---
    let cli_texture: Option<Texture> = if let Some(tex_path_str) = &args.texture {
        let tex_path = Path::new(tex_path_str);
        println!("Using texture specified via command line: {:?}", tex_path);
        // 将材质的漫反射颜色用作默认颜色（如果纹理加载失败）
        let default_color = Vector3::new(0.8, 0.8, 0.8);
        Some(load_texture(tex_path, default_color))
    } else {
        None
    };

    let load_options = tobj::LoadOptions {
        triangulate: true,
        single_index: false, // Important: Keep false to handle separate indices
        ignore_points: true,
        ignore_lines: true,
    };

    // Load OBJ and associated MTL files
    let (models, materials_result) = tobj::load_obj(obj_path_ref, &load_options)
        .map_err(|e| format!("Failed to load OBJ: {}", e))?;

    // --- Load Materials --- (Simplified: Only loads diffuse texture for now)
    let mut loaded_materials: Vec<Material> = match materials_result {
        Ok(mats) => {
            if !mats.is_empty() {
                println!("Loaded {} materials from MTL.", mats.len());
                mats.into_iter()
                    .map(|mat| {
                        // 如果命令行指定了纹理，优先使用它，否则尝试加载 MTL 中指定的纹理
                        let diffuse_texture = if cli_texture.is_some() {
                            if mat.diffuse_texture.is_some() {
                                println!("Note: Command-line texture overrides texture '{}' from MTL for material '{}'.", 
                                    mat.diffuse_texture.unwrap(), mat.name);
                            }
                            cli_texture.clone()
                        } else {
                            mat.diffuse_texture.and_then(|tex_name| {
                                // Determine the default color (use material's diffuse if available, else a fallback)
                                let default_color = Vector3::from(mat.diffuse.unwrap_or([0.8, 0.8, 0.8]));
                                let texture_path = base_path.join(tex_name);
                                // Call load_texture with path and default color. load_texture handles errors internally.
                                Some(load_texture(&texture_path, default_color))
                            })
                        };

                        Material {
                            name: mat.name,
                            ambient: Vector3::from(mat.ambient.unwrap_or([0.2, 0.2, 0.2])),
                            diffuse: Vector3::from(mat.diffuse.unwrap_or([0.8, 0.8, 0.8])),
                            specular: Vector3::from(mat.specular.unwrap_or([0.0, 0.0, 0.0])),
                            shininess: mat.shininess.unwrap_or(10.0),
                            dissolve: mat.dissolve.unwrap_or(1.0),
                            diffuse_texture,
                        }
                    })
                    .collect()
            } else {
                println!("MTL file loaded but contained no materials.");
                Vec::new() // 空向量
            }
        }
        Err(e) => {
            println!("Warning: Failed to load materials: {}.", e);
            Vec::new() // 空向量
        }
    };

    // 处理无 MTL 材质或 MTL 文件加载失败的情况
    if loaded_materials.is_empty() {
        // 如果命令行指定了纹理，创建一个带该纹理的默认材质
        if let Some(texture) = cli_texture {
            println!(
                "No MTL materials found/loaded. Creating default material with command-line texture."
            );
            let mut default_mat = Material::default();
            default_mat.diffuse_texture = Some(texture);
            loaded_materials.push(default_mat);
        } else {
            // 没有材质且没有指定纹理，创建纯色默认材质
            println!("No MTL materials or command-line texture. Using plain default material.");
            loaded_materials.push(Material::default());
        }
    }

    let mut loaded_meshes: Vec<Mesh> = Vec::with_capacity(models.len());

    for model in models.iter() {
        let mesh = &model.mesh;
        let num_vertices_in_obj = mesh.positions.len() / 3;

        if mesh.indices.is_empty() {
            println!("Skipping mesh '{}' with no indices.", model.name);
            continue;
        }

        let has_normals = !mesh.normals.is_empty();
        let has_texcoords = !mesh.texcoords.is_empty();

        // --- Generate Smooth Normals if needed ---
        let generated_normals: Option<Vec<Vector3<f32>>> = if !has_normals {
            println!(
                "Warning: Mesh '{}' is missing normals. Calculating smooth vertex normals.",
                model.name
            );
            // Create a temporary Vec<Point3<f32>> for the function
            let positions: Vec<Point3<f32>> = mesh
                .positions
                .chunks_exact(3)
                .map(|p| Point3::new(p[0], p[1], p[2]))
                .collect();
            match generate_smooth_vertex_normals(&positions, &mesh.indices) {
                Ok(normals) => Some(normals),
                Err(e) => {
                    println!(
                        "Error generating smooth normals: {}. Using default [0,1,0].",
                        e
                    );
                    // Fallback: create a vector of default normals
                    Some(vec![Vector3::y(); num_vertices_in_obj])
                }
            }
        } else {
            // println!("Mesh '{}' has normals.", model.name);
            None // Use normals from OBJ
        };

        if !has_texcoords {
            println!(
                "Warning: Mesh '{}' is missing texture coordinates. Texturing might not work correctly.",
                model.name
            );
        }

        // --- Process Vertices and Indices (Vertex Deduplication) ---
        // Create unique vertices based on (position_idx, normal_idx, texcoord_idx)
        let mut vertices: Vec<Vertex> = Vec::new();
        // Map from original (pos_idx, norm_idx, tc_idx) tuple to the index in our `vertices` Vec
        let mut index_map: HashMap<(u32, Option<u32>, Option<u32>), u32> = HashMap::new();
        let mut final_indices: Vec<u32> = Vec::with_capacity(mesh.indices.len());

        // Iterate through the original face indices provided by tobj
        for i in 0..mesh.indices.len() {
            let pos_idx = mesh.indices[i];
            // tobj might not provide normal/texcoord indices if they don't exist
            let norm_idx_opt = mesh.normal_indices.get(i).copied();
            let tc_idx_opt = mesh.texcoord_indices.get(i).copied();

            let key = (pos_idx, norm_idx_opt, tc_idx_opt);

            if let Some(&final_idx) = index_map.get(&key) {
                // Vertex already exists, just add its index
                final_indices.push(final_idx);
            } else {
                // Create a new unique vertex
                let p_start = pos_idx as usize * 3;
                let position = if p_start + 2 < mesh.positions.len() {
                    Point3::new(
                        mesh.positions[p_start],
                        mesh.positions[p_start + 1],
                        mesh.positions[p_start + 2],
                    )
                } else {
                    println!(
                        "Warning: Invalid OBJ position index {} encountered.",
                        pos_idx
                    );
                    Point3::origin() // Fallback
                };

                let normal = match norm_idx_opt {
                    Some(normal_source_idx) => {
                        if let Some(ref gen_normals) = generated_normals {
                            // Use generated normal if available (index matches position index)
                            gen_normals
                                .get(pos_idx as usize)
                                .copied()
                                .unwrap_or_else(|| {
                                    println!(
                                        "Warning: Generated normal index {} out of bounds.",
                                        pos_idx
                                    );
                                    Vector3::y()
                                })
                        } else {
                            // Use normal from OBJ file
                            let n_start = normal_source_idx as usize * 3;
                            if n_start + 2 < mesh.normals.len() {
                                Vector3::new(
                                    mesh.normals[n_start],
                                    mesh.normals[n_start + 1],
                                    mesh.normals[n_start + 2],
                                )
                                .normalize() // Normalize OBJ normals too
                            } else {
                                println!(
                                    "Warning: Invalid OBJ normal index {} encountered.",
                                    normal_source_idx
                                );
                                Vector3::y() // Fallback
                            }
                        }
                    }
                    None => {
                        // No normal index, try using generated normal based on position index
                        if let Some(ref gen_normals) = generated_normals {
                            gen_normals.get(pos_idx as usize).copied().unwrap_or_else(|| {
                                println!(
                                    "Warning: Generated normal index {} out of bounds (fallback).",
                                    pos_idx
                                );
                                Vector3::y()
                            })
                        } else {
                            // Should not happen if generated_normals logic is correct
                            println!(
                                "Warning: Missing normal index and generated normals for vertex {}.",
                                pos_idx
                            );
                            Vector3::y()
                        }
                    }
                };

                let texcoord = if let Some(idx) = tc_idx_opt {
                    let t_start = idx as usize * 2;
                    if t_start + 1 < mesh.texcoords.len() {
                        // 不翻转 V 坐标，直接使用 OBJ 中的原始值
                        // 原来的代码：Vector2::new(mesh.texcoords[t_start], 1.0 - mesh.texcoords[t_start + 1])
                        Vector2::new(mesh.texcoords[t_start], mesh.texcoords[t_start + 1])
                    } else {
                        println!("Warning: Invalid OBJ texcoord index {} encountered.", idx);
                        Vector2::zeros() // Fallback
                    }
                } else {
                    Vector2::zeros() // Fallback if no texcoords
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

        // 确定最终的材质 ID
        let material_id = mesh.material_id;
        // 确保 material_id 有效或为 None
        let mut final_material_id = material_id.filter(|&id| id < loaded_materials.len());

        // 如果网格没有有效的材质 ID，但我们有加载/创建的材质（例如从命令行纹理创建的默认材质），
        // 则分配材质 ID 0
        if final_material_id.is_none() && !loaded_materials.is_empty() {
            final_material_id = Some(0);
            if material_id.is_some() {
                println!(
                    "Warning: Mesh '{}' had invalid material ID {}. Assigning default material ID 0.",
                    model.name,
                    material_id.unwrap()
                );
            }
        }

        loaded_meshes.push(Mesh {
            vertices,
            indices: final_indices,
            material_id: final_material_id,
        });
        println!(
            "Processed mesh '{}': {} unique vertices, {} triangles, Material ID: {:?}",
            model.name,
            loaded_meshes.last().unwrap().vertices.len(),
            loaded_meshes.last().unwrap().indices.len() / 3,
            final_material_id
        );
    }

    if loaded_meshes.is_empty() {
        return Err("No processable meshes found in the OBJ file.".to_string());
    }

    Ok(ModelData {
        meshes: loaded_meshes,
        materials: loaded_materials,
    })
}
