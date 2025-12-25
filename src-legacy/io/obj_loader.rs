use crate::io::render_settings::RenderSettings;
use crate::material_system::materials::{Material, MaterialType, Mesh, Model, Vertex};
use crate::material_system::texture::Texture;
use image::DynamicImage;
use log::{debug, info, warn};
use nalgebra::{Point3, Vector2, Vector3};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// 生成平滑的顶点法线，通过平均面法线实现
fn generate_smooth_vertex_normals(
    vertices: &[Point3<f32>],
    indices: &[u32],
) -> Result<Vec<Vector3<f32>>, String> {
    if indices.len() % 3 != 0 {
        return Err("三角形索引数量必须是3的倍数".to_string());
    }
    if vertices.is_empty() {
        return Ok(Vec::new());
    }

    let num_vertices = vertices.len();
    let num_faces = indices.len() / 3;
    let mut vertex_normals = vec![Vector3::zeros(); num_vertices];

    for i in 0..num_faces {
        let idx0 = indices[i * 3] as usize;
        let idx1 = indices[i * 3 + 1] as usize;
        let idx2 = indices[i * 3 + 2] as usize;

        if idx0 >= num_vertices || idx1 >= num_vertices || idx2 >= num_vertices {
            warn!("面 {i} 包含越界的顶点索引，跳过");
            continue;
        }

        let v0 = vertices[idx0];
        let v1 = vertices[idx1];
        let v2 = vertices[idx2];

        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let face_normal = edge1.cross(&edge2);

        vertex_normals[idx0] += face_normal;
        vertex_normals[idx1] += face_normal;
        vertex_normals[idx2] += face_normal;
    }

    let mut zero_norm_count = 0;
    for normal in vertex_normals.iter_mut() {
        let norm_squared = normal.norm_squared();
        if norm_squared > 1e-12 {
            normal.normalize_mut();
        } else {
            *normal = Vector3::y();
            zero_norm_count += 1;
        }
    }

    if zero_norm_count > 0 {
        warn!("{zero_norm_count} 个顶点的法线为零，设置为默认值 [0, 1, 0]");
    }

    Ok(vertex_normals)
}

fn get_basename_from_path(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unknown".to_string())
}

/// 主要功能：加载并处理 OBJ 模型文件
pub fn load_obj_model<P: AsRef<Path>>(
    obj_path: P,
    settings: &RenderSettings,
) -> Result<Model, String> {
    let obj_path_ref = obj_path.as_ref();
    info!("加载 OBJ 文件: {obj_path_ref:?}");

    let obj_basename = get_basename_from_path(obj_path_ref);
    let base_path = obj_path_ref.parent().unwrap_or_else(|| Path::new("."));

    let cli_texture: Option<Texture> = if let Some(tex_path_str) = &settings.texture {
        let tex_path = Path::new(tex_path_str);
        debug!("使用命令行指定的纹理: {tex_path:?}");
        Some(Texture::from_file(tex_path).unwrap_or_else(|| {
            warn!("无法加载命令行指定的纹理，使用默认颜色");
            Texture {
                image: Arc::new(DynamicImage::new_rgb8(1, 1)),
                width: 1,
                height: 1,
            }
        }))
    } else {
        None
    };

    let load_options = tobj::LoadOptions {
        triangulate: true,
        single_index: false,
        ignore_points: true,
        ignore_lines: true,
    };

    let (models, materials_result) =
        tobj::load_obj(obj_path_ref, &load_options).map_err(|e| format!("加载 OBJ 失败: {e}"))?;

    let mut loaded_materials: Vec<Material> = match materials_result {
        Ok(mats) => {
            if !mats.is_empty() {
                info!("从 MTL 加载了 {} 个材质", mats.len());
                mats.into_iter()
                    .map(|mat| {
                        // 只加载图片纹理
                        let texture = if let Some(cli_tex) = &cli_texture {
                            Some(cli_tex.clone())
                        } else if let Some(tex_name) = mat.diffuse_texture {
                            let texture_path = base_path.join(&tex_name);
                            Some(Texture::from_file(&texture_path).unwrap_or_else(|| {
                                warn!("无法加载纹理 '{texture_path:?}'，使用默认颜色");
                                Texture {
                                    image: Arc::new(DynamicImage::new_rgb8(1, 1)),
                                    width: 1,
                                    height: 1,
                                }
                            }))
                        } else {
                            None
                        };

                        Material {
                            material_type: if settings.use_pbr {
                                MaterialType::PBR
                            } else {
                                MaterialType::Phong
                            },
                            base_color: Vector3::from(mat.diffuse.unwrap_or([0.8, 0.8, 0.8])),
                            alpha: 1.0,
                            texture,
                            metallic: 0.0,
                            roughness: 0.5,
                            ambient_occlusion: 1.0,
                            specular: Vector3::from(mat.specular.unwrap_or([0.5, 0.5, 0.5])),
                            shininess: mat.shininess.unwrap_or(32.0),
                            diffuse_intensity: 1.0,
                            specular_intensity: 1.0,
                            emissive: Vector3::zeros(),
                            ambient_factor: Vector3::from(mat.diffuse.unwrap_or([0.8, 0.8, 0.8]))
                                * 0.3,
                        }
                    })
                    .collect()
            } else {
                info!("MTL 文件中没有材质");
                Vec::new()
            }
        }
        Err(e) => {
            warn!("加载材质失败: {e}");
            Vec::new()
        }
    };

    if loaded_materials.is_empty() {
        let default_type = if settings.use_pbr {
            MaterialType::PBR
        } else {
            MaterialType::Phong
        };
        let mut default_mat = Material::default(default_type);

        if let Some(texture) = cli_texture {
            default_mat.texture = Some(texture);
        } else {
            default_mat.texture = None;
        }
        loaded_materials.push(default_mat);
    }

    let mut loaded_meshes: Vec<Mesh> = Vec::with_capacity(models.len());

    for model in models.iter() {
        let mesh = &model.mesh;
        let num_vertices_in_obj = mesh.positions.len() / 3;

        let mesh_name = if model.name.is_empty() || model.name == "unnamed_object" {
            obj_basename.clone()
        } else {
            model.name.clone()
        };

        if mesh.indices.is_empty() {
            debug!("跳过没有索引的网格 '{mesh_name}'");
            continue;
        }

        let has_normals = !mesh.normals.is_empty();
        let has_texcoords = !mesh.texcoords.is_empty();

        let generated_normals: Option<Vec<Vector3<f32>>> = if !has_normals {
            warn!("网格 '{mesh_name}' 缺少法线，计算平滑顶点法线");

            let positions: Vec<Point3<f32>> = mesh
                .positions
                .chunks_exact(3)
                .map(|p| Point3::new(p[0], p[1], p[2]))
                .collect();

            match generate_smooth_vertex_normals(&positions, &mesh.indices) {
                Ok(normals) => Some(normals),
                Err(e) => {
                    warn!("生成平滑法线错误: {e}，使用默认法线 [0,1,0]");
                    Some(vec![Vector3::y(); num_vertices_in_obj])
                }
            }
        } else {
            None
        };

        if !has_texcoords {
            debug!("网格 '{mesh_name}' 缺少纹理坐标，纹理映射可能不正确");
        }

        let mut vertices: Vec<Vertex> = Vec::new();
        let mut index_map: HashMap<(u32, Option<u32>, Option<u32>), u32> = HashMap::new();
        let mut final_indices: Vec<u32> = Vec::with_capacity(mesh.indices.len());

        for i in 0..mesh.indices.len() {
            let pos_idx = mesh.indices[i];
            let norm_idx_opt = mesh.normal_indices.get(i).copied();
            let tc_idx_opt = mesh.texcoord_indices.get(i).copied();

            let key = (pos_idx, norm_idx_opt, tc_idx_opt);

            if let Some(&final_idx) = index_map.get(&key) {
                final_indices.push(final_idx);
            } else {
                let p_start = pos_idx as usize * 3;
                let position = if p_start + 2 < mesh.positions.len() {
                    Point3::new(
                        mesh.positions[p_start],
                        mesh.positions[p_start + 1],
                        mesh.positions[p_start + 2],
                    )
                } else {
                    warn!("遇到无效的 OBJ 位置索引 {pos_idx}");
                    Point3::origin()
                };

                let normal = match norm_idx_opt {
                    Some(normal_source_idx) => {
                        if let Some(ref gen_normals) = generated_normals {
                            gen_normals
                                .get(pos_idx as usize)
                                .copied()
                                .unwrap_or_else(|| {
                                    warn!("生成的法线索引 {pos_idx} 越界");
                                    Vector3::y()
                                })
                        } else {
                            let n_start = normal_source_idx as usize * 3;
                            if n_start + 2 < mesh.normals.len() {
                                Vector3::new(
                                    mesh.normals[n_start],
                                    mesh.normals[n_start + 1],
                                    mesh.normals[n_start + 2],
                                )
                                .normalize()
                            } else {
                                warn!("遇到无效的 OBJ 法线索引 {normal_source_idx}");
                                Vector3::y()
                            }
                        }
                    }
                    None => {
                        if let Some(ref gen_normals) = generated_normals {
                            gen_normals
                                .get(pos_idx as usize)
                                .copied()
                                .unwrap_or_else(|| {
                                    warn!("生成的法线索引 {pos_idx} 越界（回退）");
                                    Vector3::y()
                                })
                        } else {
                            warn!("缺少顶点 {pos_idx} 的法线索引和生成法线");
                            Vector3::y()
                        }
                    }
                };

                let texcoord = if let Some(idx) = tc_idx_opt {
                    let t_start = idx as usize * 2;
                    if t_start + 1 < mesh.texcoords.len() {
                        Vector2::new(mesh.texcoords[t_start], mesh.texcoords[t_start + 1])
                    } else {
                        warn!("遇到无效的 OBJ 纹理坐标索引 {idx}");
                        Vector2::zeros()
                    }
                } else {
                    Vector2::zeros()
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

        let material_id = mesh.material_id.unwrap_or(0);
        let final_material_id = if material_id < loaded_materials.len() {
            material_id
        } else if !loaded_materials.is_empty() {
            warn!("网格 '{mesh_name}' 有无效的材质 ID {material_id}。分配默认材质 ID 0");
            0
        } else {
            0
        };

        loaded_meshes.push(Mesh {
            vertices,
            indices: final_indices,
            material_id: final_material_id,
            name: mesh_name.clone(),
        });

        debug!(
            "处理网格 '{}': {} 个唯一顶点, {} 个三角形, 材质 ID: {}",
            loaded_meshes.last().unwrap().name,
            loaded_meshes.last().unwrap().vertices.len(),
            loaded_meshes.last().unwrap().indices.len() / 3,
            final_material_id
        );
    }

    if loaded_meshes.is_empty() {
        return Err("OBJ 文件中没有可处理的网格".to_string());
    }

    let model = Model {
        meshes: loaded_meshes,
        materials: loaded_materials,
        name: obj_basename,
    };

    info!("创建模型 '{}' 成功", model.name);
    Ok(model)
}
