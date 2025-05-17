use crate::io::args::Args;
use crate::materials::model_types::{Material, Mesh, ModelData, TextureOptions, Vertex};
use crate::materials::texture::{Texture, load_texture};
use nalgebra::{Point3, Vector2, Vector3};
use std::collections::HashMap;
use std::path::Path;

/// 生成平滑的顶点法线，通过平均面法线实现
fn generate_smooth_vertex_normals(
    vertices: &[Point3<f32>],
    indices: &[u32],
) -> Result<Vec<Vector3<f32>>, String> {
    // 验证输入数据
    if indices.len() % 3 != 0 {
        return Err("三角形索引数量必须是3的倍数".to_string());
    }
    if vertices.is_empty() {
        return Ok(Vec::new()); // 没有顶点，不计算法线
    }

    let num_vertices = vertices.len();
    let num_faces = indices.len() / 3;
    let mut vertex_normals = vec![Vector3::zeros(); num_vertices];

    // 1. 计算面法线并累加到顶点
    for i in 0..num_faces {
        let idx0 = indices[i * 3] as usize;
        let idx1 = indices[i * 3 + 1] as usize;
        let idx2 = indices[i * 3 + 2] as usize;

        // 边界检查
        if idx0 >= num_vertices || idx1 >= num_vertices || idx2 >= num_vertices {
            println!("警告: 面 {} 包含越界的顶点索引，跳过", i);
            continue;
        }

        let v0 = vertices[idx0];
        let v1 = vertices[idx1];
        let v2 = vertices[idx2];

        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let face_normal = edge1.cross(&edge2);

        // 累加法线（较大面积的三角形权重更大）
        vertex_normals[idx0] += face_normal;
        vertex_normals[idx1] += face_normal;
        vertex_normals[idx2] += face_normal;
    }

    // 2. 归一化顶点法线
    let mut zero_norm_count = 0;
    for normal in vertex_normals.iter_mut() {
        let norm_squared = normal.norm_squared();
        if norm_squared > 1e-12 {
            normal.normalize_mut();
        } else {
            // 处理零长度法线（顶点未使用或属于退化面）
            *normal = Vector3::y(); // 默认使用向上向量
            zero_norm_count += 1;
        }
    }

    if zero_norm_count > 0 {
        println!(
            "警告: {} 个顶点的法线为零，设置为默认值 [0, 1, 0]",
            zero_norm_count
        );
    }

    Ok(vertex_normals)
}

/// 从文件路径中提取基本文件名（不含扩展名）
fn get_basename_from_path(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unknown".to_string())
}

/// 加载并处理 OBJ 模型文件
pub fn load_obj_enhanced<P: AsRef<Path>>(obj_path: P, args: &Args) -> Result<ModelData, String> {
    let obj_path_ref = obj_path.as_ref();
    println!("加载 OBJ 文件: {:?}", obj_path_ref);

    // 提取 OBJ 文件的基本名称（不含扩展名）
    let obj_basename = get_basename_from_path(obj_path_ref);

    // 确定加载材质和纹理的基础路径
    let base_path = obj_path_ref.parent().unwrap_or_else(|| Path::new("."));

    // 检查命令行指定的纹理
    let cli_texture: Option<Texture> = if let Some(tex_path_str) = &args.texture {
        let tex_path = Path::new(tex_path_str);
        println!("使用命令行指定的纹理: {:?}", tex_path);
        let default_color = Vector3::new(0.8, 0.8, 0.8);
        Some(load_texture(tex_path, default_color))
    } else {
        None
    };

    let load_options = tobj::LoadOptions {
        triangulate: true,   // 将所有面转换为三角形
        single_index: false, // 保持独立的索引以处理分开的纹理/法线坐标
        ignore_points: true, // 忽略点元素
        ignore_lines: true,  // 忽略线元素
    };

    // 加载 OBJ 和关联的 MTL 文件
    let (models, materials_result) =
        tobj::load_obj(obj_path_ref, &load_options).map_err(|e| format!("加载 OBJ 失败: {}", e))?;

    // 加载材质
    let mut loaded_materials: Vec<Material> = match materials_result {
        Ok(mats) => {
            if !mats.is_empty() {
                println!("从 MTL 加载了 {} 个材质", mats.len());
                mats.into_iter()
                    .map(|mat| {
                        // 优先使用命令行指定的纹理
                        let texture = if cli_texture.is_some() {
                            if mat.diffuse_texture.is_some() {
                                println!(
                                    "注意: 命令行指定的纹理覆盖了材质 '{}' 中的纹理 '{}'",
                                    mat.name,
                                    mat.diffuse_texture.unwrap()
                                );
                            }
                            cli_texture.clone()
                        } else {
                            mat.diffuse_texture.map(|tex_name| {
                                let default_color =
                                    Vector3::from(mat.diffuse.unwrap_or([0.8, 0.8, 0.8]));
                                let texture_path = base_path.join(&tex_name);
                                let texture = load_texture(&texture_path, default_color);

                                // 获取并记录纹理信息 - 这会调用之前未使用的get_texture_info方法
                                let info = texture.get_texture_info();
                                println!(
                                    "  - 纹理 '{}': 类型={}, 尺寸={}x{}",
                                    tex_name, info.texture_type, info.width, info.height
                                );

                                texture
                            })
                        };

                        Material {
                            name: mat.name,
                            albedo: Vector3::from(mat.diffuse.unwrap_or([0.8, 0.8, 0.8])),
                            specular: Vector3::from(mat.specular.unwrap_or([0.0, 0.0, 0.0])),
                            shininess: mat.shininess.unwrap_or(10.0),
                            texture,

                            // 添加PBR参数
                            metallic: 0.0,          // 默认为非金属材质
                            roughness: 0.5,         // 中等粗糙度
                            ambient_occlusion: 1.0, // 默认无AO
                            emissive: Vector3::zeros(),
                        }
                    })
                    .collect()
            } else {
                println!("MTL 文件中没有材质");
                Vec::new()
            }
        }
        Err(e) => {
            println!("警告: 加载材质失败: {}", e);
            Vec::new()
        }
    };

    // 处理无材质的情况
    if loaded_materials.is_empty() {
        if let Some(texture) = cli_texture {
            println!("未找到 MTL 材质，创建带命令行纹理的默认材质");

            // 使用configure_texture方法配置默认材质
            let mut default_mat = Material::default();

            // 获取命令行纹理的信息
            let texture_info = texture.get_texture_info();
            println!(
                "应用命令行纹理: 类型={}, 尺寸={}x{}",
                texture_info.texture_type, texture_info.width, texture_info.height
            );

            // 使用configure_texture方法附加纹理
            let texture_type = if texture.is_face_color() {
                "face_color"
            } else {
                "image"
            };
            default_mat.configure_texture(
                texture_type,
                Some(TextureOptions {
                    path: None,
                    color: None,
                }),
            );
            default_mat.texture = Some(texture); // 直接设置纹理

            loaded_materials.push(default_mat);
        } else {
            println!("无 MTL 材质且无指定纹理，使用纯色默认材质");

            // 创建并配置纯色默认材质
            let mut default_mat = Material::default();
            let default_color = Vector3::new(0.8, 0.8, 0.8); // 默认灰色

            // 使用configure_texture方法配置纯色纹理
            default_mat.configure_texture(
                "solid_color",
                Some(TextureOptions {
                    path: None,
                    color: Some(default_color),
                }),
            );

            println!(
                "创建默认纯色纹理: RGB({:.2}, {:.2}, {:.2})",
                default_color.x, default_color.y, default_color.z
            );

            loaded_materials.push(default_mat);
        }
    }

    // 处理网格数据
    let mut loaded_meshes: Vec<Mesh> = Vec::with_capacity(models.len());

    for model in models.iter() {
        let mesh = &model.mesh;
        let num_vertices_in_obj = mesh.positions.len() / 3;

        // 使用模型名称或OBJ文件名
        let mesh_name = if model.name.is_empty() || model.name == "unnamed_object" {
            // 如果模型没有有效的名称，使用OBJ文件名
            obj_basename.clone()
        } else {
            model.name.clone()
        };

        if mesh.indices.is_empty() {
            println!("跳过没有索引的网格 '{}'", mesh_name);
            continue;
        }

        let has_normals = !mesh.normals.is_empty();
        let has_texcoords = !mesh.texcoords.is_empty();

        // 如果需要，生成平滑顶点法线
        let generated_normals: Option<Vec<Vector3<f32>>> = if !has_normals {
            println!("警告: 网格 '{}' 缺少法线，计算平滑顶点法线", mesh_name);

            let positions: Vec<Point3<f32>> = mesh
                .positions
                .chunks_exact(3)
                .map(|p| Point3::new(p[0], p[1], p[2]))
                .collect();

            match generate_smooth_vertex_normals(&positions, &mesh.indices) {
                Ok(normals) => Some(normals),
                Err(e) => {
                    println!("生成平滑法线错误: {}，使用默认法线 [0,1,0]", e);
                    Some(vec![Vector3::y(); num_vertices_in_obj])
                }
            }
        } else {
            None // 使用 OBJ 中的法线
        };

        if !has_texcoords {
            println!(
                "警告: 网格 '{}' 缺少纹理坐标，纹理映射可能不正确",
                mesh_name
            );
        }

        // 顶点去重和索引处理
        let mut vertices: Vec<Vertex> = Vec::new();
        let mut index_map: HashMap<(u32, Option<u32>, Option<u32>), u32> = HashMap::new();
        let mut final_indices: Vec<u32> = Vec::with_capacity(mesh.indices.len());

        // 遍历原始面索引
        for i in 0..mesh.indices.len() {
            let pos_idx = mesh.indices[i];
            let norm_idx_opt = mesh.normal_indices.get(i).copied();
            let tc_idx_opt = mesh.texcoord_indices.get(i).copied();

            let key = (pos_idx, norm_idx_opt, tc_idx_opt);

            if let Some(&final_idx) = index_map.get(&key) {
                // 顶点已存在，仅添加索引
                final_indices.push(final_idx);
            } else {
                // 创建新的唯一顶点
                let p_start = pos_idx as usize * 3;
                let position = if p_start + 2 < mesh.positions.len() {
                    Point3::new(
                        mesh.positions[p_start],
                        mesh.positions[p_start + 1],
                        mesh.positions[p_start + 2],
                    )
                } else {
                    println!("警告: 遇到无效的 OBJ 位置索引 {}", pos_idx);
                    Point3::origin() // 回退值
                };

                let normal = match norm_idx_opt {
                    Some(normal_source_idx) => {
                        if let Some(ref gen_normals) = generated_normals {
                            // 使用生成的法线
                            gen_normals
                                .get(pos_idx as usize)
                                .copied()
                                .unwrap_or_else(|| {
                                    println!("警告: 生成的法线索引 {} 越界", pos_idx);
                                    Vector3::y()
                                })
                        } else {
                            // 使用 OBJ 文件中的法线
                            let n_start = normal_source_idx as usize * 3;
                            if n_start + 2 < mesh.normals.len() {
                                Vector3::new(
                                    mesh.normals[n_start],
                                    mesh.normals[n_start + 1],
                                    mesh.normals[n_start + 2],
                                )
                                .normalize()
                            } else {
                                println!("警告: 遇到无效的 OBJ 法线索引 {}", normal_source_idx);
                                Vector3::y() // 回退值
                            }
                        }
                    }
                    None => {
                        // 无法线索引，尝试使用基于位置索引的生成法线
                        if let Some(ref gen_normals) = generated_normals {
                            gen_normals
                                .get(pos_idx as usize)
                                .copied()
                                .unwrap_or_else(|| {
                                    println!("警告: 生成的法线索引 {} 越界（回退）", pos_idx);
                                    Vector3::y()
                                })
                        } else {
                            println!("警告: 缺少顶点 {} 的法线索引和生成法线", pos_idx);
                            Vector3::y()
                        }
                    }
                };

                let texcoord = if let Some(idx) = tc_idx_opt {
                    let t_start = idx as usize * 2;
                    if t_start + 1 < mesh.texcoords.len() {
                        Vector2::new(mesh.texcoords[t_start], mesh.texcoords[t_start + 1])
                    } else {
                        println!("警告: 遇到无效的 OBJ 纹理坐标索引 {}", idx);
                        Vector2::zeros() // 回退值
                    }
                } else {
                    Vector2::zeros() // 无纹理坐标时的回退值
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
        let mut final_material_id = material_id.filter(|&id| id < loaded_materials.len());

        // 若网格没有有效的材质 ID，但我们有加载的材质，则分配材质 ID 0
        if final_material_id.is_none() && !loaded_materials.is_empty() {
            final_material_id = Some(0);
            if material_id.is_some() {
                println!(
                    "警告: 网格 '{}' 有无效的材质 ID {}。分配默认材质 ID 0",
                    mesh_name,
                    material_id.unwrap()
                );
            }
        }

        loaded_meshes.push(Mesh {
            vertices,
            indices: final_indices,
            material_id: final_material_id,
            name: mesh_name.clone(), // 添加网格名称字段
        });
        println!(
            "处理网格 '{}': {} 个唯一顶点, {} 个三角形, 材质 ID: {:?}",
            loaded_meshes.last().unwrap().name,
            loaded_meshes.last().unwrap().vertices.len(),
            loaded_meshes.last().unwrap().indices.len() / 3,
            final_material_id
        );
    }

    if loaded_meshes.is_empty() {
        return Err("OBJ 文件中没有可处理的网格".to_string());
    }

    // 创建并返回模型数据，设置模型名称为OBJ文件基本名称
    let model_data = ModelData {
        meshes: loaded_meshes,
        materials: loaded_materials,
        name: obj_basename, // 添加模型名称字段
    };

    println!("创建模型 '{}' 成功", model_data.name);
    Ok(model_data)
}
