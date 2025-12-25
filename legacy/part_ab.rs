        content.push_str(&format!("diffuse_color = \"{}\"\n", settings.diffuse_color));
        content.push_str(&format!(
            "diffuse_intensity = {}\n",
            settings.diffuse_intensity
        ));
        content.push_str(&format!("alpha = {}\n", settings.alpha));
        content.push_str(&format!(
            "specular_color = \"{}\"\n",
            settings.specular_color
        ));
        content.push_str(&format!(
            "specular_intensity = {}\n",
            settings.specular_intensity
        ));
        content.push_str(&format!("shininess = {}\n", settings.shininess));
        content.push_str(&format!("base_color = \"{}\"\n", settings.base_color));
        content.push_str(&format!("metallic = {}\n", settings.metallic));
        content.push_str(&format!("roughness = {}\n", settings.roughness));
        content.push_str(&format!(
            "ambient_occlusion = {}\n",
            settings.ambient_occlusion
        ));
        content.push_str(&format!("emissive = \"{}\"\n", settings.emissive));
        content.push('\n');

        // [background] 部分
        content.push_str("[background]\n");
        content.push_str(&format!(
            "use_background_image = {}\n",
            settings.use_background_image
        ));
        content.push_str(&format!(
            "enable_gradient_background = {}\n",
            settings.enable_gradient_background
        ));
        content.push_str(&format!(
            "gradient_top_color = \"{}\"\n",
            settings.gradient_top_color
        ));
        content.push_str(&format!(
            "gradient_bottom_color = \"{}\"\n",
            settings.gradient_bottom_color
        ));
        content.push_str(&format!(
            "enable_ground_plane = {}\n",
            settings.enable_ground_plane
        ));
        content.push_str(&format!(
            "ground_plane_color = \"{}\"\n",
            settings.ground_plane_color
        ));
        content.push_str(&format!(
            "ground_plane_height = {}\n",
            settings.ground_plane_height
        ));
        content.push('\n');

        // [animation] 部分
        content.push_str("[animation]\n");
        content.push_str(&format!("animate = {}\n", settings.animate));
        content.push_str(&format!("fps = {}\n", settings.fps));
        content.push_str(&format!("rotation_speed = {}\n", settings.rotation_speed));
        content.push_str(&format!("rotation_cycles = {}\n", settings.rotation_cycles));
        content.push_str(&format!(
            "animation_type = \"{:?}\"\n",
            settings.animation_type
        ));
        content.push_str(&format!(
            "rotation_axis = \"{:?}\"\n",
            settings.rotation_axis
        ));
        content.push_str(&format!(
            "custom_rotation_axis = \"{}\"\n",
            settings.custom_rotation_axis
        ));
        content.push('\n');

        // [shadow] 部分
        content.push_str("# 阴影配置\n");
        content.push_str("[shadow]\n");
        content.push_str("enable_shadow_mapping = ");
        content.push_str(&settings.enable_shadow_mapping.to_string());
        content.push('\n');
        content.push_str("shadow_map_size = ");
        content.push_str(&settings.shadow_map_size.to_string());
        content.push('\n');
        content.push_str("shadow_bias = ");
        content.push_str(&settings.shadow_bias.to_string());
        content.push('\n');
        content.push_str("shadow_distance = ");
        content.push_str(&settings.shadow_distance.to_string());
        content.push('\n');
        content.push_str("enable_pcf = ");
        content.push_str(&settings.enable_pcf.to_string());
        content.push('\n');
        content.push_str("pcf_type = \"");
        content.push_str(&settings.pcf_type);
        content.push_str("\"\n");
        content.push_str("pcf_kernel = ");
        content.push_str(&settings.pcf_kernel.to_string());
        content.push('\n');
        content.push_str("pcf_sigma = ");
        content.push_str(&settings.pcf_sigma.to_string());
        content.push('\n');

        Ok(content)
    }
}
pub mod config_loader;
pub mod model_loader;
pub mod obj_loader;
pub mod render_settings;
pub mod simple_cli;
use crate::io::obj_loader::load_obj_model;
use crate::io::render_settings::RenderSettings;
use crate::material_system::materials::Model;
use crate::scene::scene_utils::Scene;
use crate::utils::model_utils::normalize_and_center_model;
use log::{debug, info};
use std::path::Path;
use std::time::Instant;

/// 模型加载器
pub struct ModelLoader;

impl ModelLoader {
    /// 主要功能：加载OBJ模型并创建场景
    pub fn load_and_create_scene(
        obj_path: &str,
        settings: &RenderSettings,
    ) -> Result<(Scene, Model), String> {
        info!("加载模型：{obj_path}");
        let load_start = Instant::now();

        // 检查文件存在
        if !Path::new(obj_path).exists() {
            return Err(format!("输入的 OBJ 文件未找到：{obj_path}"));
        }

        // 加载模型数据
        let mut model = load_obj_model(obj_path, settings)?;
        debug!("模型加载耗时 {:?}", load_start.elapsed());

        // 归一化模型
        debug!("归一化模型...");
        let norm_start_time = Instant::now();
        let (original_center, scale_factor) = normalize_and_center_model(&mut model);
        debug!(
            "模型归一化耗时 {:?}，原始中心：{:.3?}，缩放系数：{:.3}",
            norm_start_time.elapsed(),
            original_center,
            scale_factor
        );

        // 创建场景
        debug!("创建场景...");
        let scene = Scene::new(model.clone(), settings)?;

        Ok((scene, model))
    }

    /// 验证资源
    pub fn validate_resources(settings: &RenderSettings) -> Result<(), String> {
        // 验证 OBJ 文件
        if let Some(obj_path) = &settings.obj {
            if !Path::new(obj_path).exists() {
                return Err(format!("OBJ 文件不存在: {obj_path}"));
            }
        }

        // 验证背景图片（如果启用）
        if settings.use_background_image {
            if let Some(bg_path) = &settings.background_image_path {
                if !Path::new(bg_path).exists() {
                    return Err(format!("背景图片文件不存在: {bg_path}"));
                }
            } else {
                return Err("启用了背景图片但未指定路径".to_string());
            }
        }

        // 验证纹理文件（如果指定）
        if let Some(texture_path) = &settings.texture {
            if !Path::new(texture_path).exists() {
                return Err(format!("纹理文件不存在: {texture_path}"));
            }
        }

        info!("所有资源验证通过");
        Ok(())
    }
}
use crate::io::config_loader::TomlConfigLoader;
use crate::io::render_settings::RenderSettings;
use clap::Parser;
use log::info;

/// 极简CLI
#[derive(Parser, Debug)]
#[command(name = "rasterizer")]
#[command(about = "TOML驱动的光栅化渲染器")]
pub struct SimpleCli {
    /// 配置文件路径（TOML格式）
    #[arg(short, long, value_name = "FILE")]
    pub config: Option<String>,

    /// 无头模式（不启动GUI）
    #[arg(long)]
    pub headless: bool,

    /// 使用示例配置（临时创建并加载）
    #[arg(long)]
    pub use_example_config: bool,
}

impl SimpleCli {
    /// 处理CLI参数并返回RenderSettings和是否启动GUI
    pub fn process() -> Result<(RenderSettings, bool), String> {
        let cli = Self::parse();

        // 处理示例配置
        if cli.use_example_config {
            let temp_config_path = "temp_example_config.toml";

            TomlConfigLoader::create_example_config(temp_config_path)
                .map_err(|e| format!("创建示例配置失败: {e}"))?;

            info!("已创建临时示例配置: {temp_config_path}");

            let settings = TomlConfigLoader::load_from_file(temp_config_path)
                .map_err(|e| format!("加载示例配置失败: {e}"))?;

            let should_start_gui = !cli.headless;
            return Ok((settings, should_start_gui));
        }

        // 加载配置文件或使用默认设置
        let settings = if let Some(config_path) = &cli.config {
            info!("加载配置文件: {config_path}");
            TomlConfigLoader::load_from_file(config_path)
                .map_err(|e| format!("配置文件加载失败: {e}"))?
        } else {
            info!("使用默认设置");
            RenderSettings::default()
        };

        let should_start_gui = !cli.headless;
        Ok((settings, should_start_gui))
    }
}
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
use crate::material_system::light::Light;
use log::warn;
use nalgebra::{Point3, Vector3};

/// 动画类型枚举
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum AnimationType {
    #[default]
    CameraOrbit,
    ObjectLocalRotation,
    None,
}

/// 旋转轴枚举
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum RotationAxis {
    X,
    #[default]
    Y,
    Z,
    Custom,
}

/// 纯数据结构
#[derive(Debug, Clone)]
pub struct RenderSettings {
    // ===== 文件路径设置 =====
    /// 输入OBJ文件的路径
    pub obj: Option<String>,
    /// 输出文件的基础名称
    pub output: String,
    /// 输出图像的目录
    pub output_dir: String,
    /// 显式指定要使用的纹理文件，覆盖MTL设置
    pub texture: Option<String>,
    /// 背景图片路径
    pub background_image_path: Option<String>,

    // ===== 渲染基础设置 =====
    /// 输出图像的宽度
    pub width: usize,
    /// 输出图像的高度
    pub height: usize,
    /// 投影类型："perspective"或"orthographic"
    pub projection: String,
    /// 启用Z缓冲（深度测试）
    pub use_zbuffer: bool,
    /// 使用伪随机面颜色而非材质颜色
    pub colorize: bool,
    /// 启用纹理加载和使用
    pub use_texture: bool,
    /// 启用gamma矫正
    pub use_gamma: bool,
    /// 启用ACES色彩管理
    pub enable_aces: bool,
    /// 启用背面剔除
    pub backface_culling: bool,
    /// 以线框模式渲染
    pub wireframe: bool,
    /// 启用小三角形剔除
    pub cull_small_triangles: bool,
    /// 小三角形剔除的最小面积阈值
    pub min_triangle_area: f32,
    /// 启用渲染和保存深度图
    pub save_depth: bool,

    // ===== 物体变换控制（字符串格式，用于TOML序列化） =====
    /// 物体位置 (x,y,z)
    pub object_position: String,
    /// 物体旋转 (欧拉角，度)
    pub object_rotation: String,
    /// 物体缩放 (x,y,z)
    pub object_scale_xyz: String,
    /// 物体的全局均匀缩放因子
    pub object_scale: f32,

    // ===== 相机参数 =====
    /// 相机位置（视点），格式为"x,y,z"
    pub camera_from: String,
    /// 相机目标（观察点），格式为"x,y,z"
    pub camera_at: String,
    /// 相机世界坐标系上方向，格式为"x,y,z"
    pub camera_up: String,
    /// 相机垂直视场角（度，用于透视投影）
    pub camera_fov: f32,

    // ===== 光照基础参数 =====
    /// 启用光照计算
    pub use_lighting: bool,
    /// 环境光强度因子
    pub ambient: f32,
    /// 环境光强度RGB值，格式为"r,g,b"
    pub ambient_color: String,

    // ===== 着色模型选择 =====
    /// 使用Phong着色（逐像素光照）
    pub use_phong: bool,
    /// 使用基于物理的渲染(PBR)
    pub use_pbr: bool,

    // ===== Phong着色模型参数 =====
    /// 漫反射颜色，格式为"r,g,b"
    pub diffuse_color: String,
    /// 漫反射强度(0.0-2.0)
    pub diffuse_intensity: f32,
    /// 镜面反射颜色，格式为"r,g,b" (之前是单一数值)
    pub specular_color: String,
    /// 镜面反射强度(0.0-2.0)
    pub specular_intensity: f32,
    /// 材质的光泽度(硬度)参数
    pub shininess: f32,

    // ===== PBR材质参数 =====
    /// 材质的基础颜色，格式为"r,g,b"
    pub base_color: String,
    /// 材质的金属度(0.0-1.0)
    pub metallic: f32,
    /// 材质的粗糙度(0.0-1.0)
    pub roughness: f32,
    /// 环境光遮蔽系数(0.0-1.0)
    pub ambient_occlusion: f32,
    /// 材质透明度(0.0-1.0)，1.0为完全不透明
    pub alpha: f32,
    /// 材质的自发光颜色，格式为"r,g,b"
    pub emissive: String,

    // ===== 阴影设置 =====
    /// 启用简单阴影映射（仅地面）
    pub enable_shadow_mapping: bool,
    /// 阴影贴图尺寸
    pub shadow_map_size: usize,
    /// 阴影偏移，避免阴影痤疮
    pub shadow_bias: f32,
    /// 阴影渲染距离
    pub shadow_distance: f32,
    /// 是否启用PCF软阴影
    pub enable_pcf: bool,
    /// PCF类型
    pub pcf_type: String,
    /// PCF采样窗口半径
    pub pcf_kernel: usize,
    /// PCF高斯模糊的sigma
    pub pcf_sigma: f32,

    // ===== 背景与环境设置 =====
    /// 启用渐变背景
    pub enable_gradient_background: bool,
    /// 渐变背景顶部颜色，格式为"r,g,b"
    pub gradient_top_color: String,
    /// 渐变背景底部颜色，格式为"r,g,b"
    pub gradient_bottom_color: String,
    /// 启用地面平面
    pub enable_ground_plane: bool,
    /// 地面平面颜色，格式为"r,g,b"
    pub ground_plane_color: String,
    /// 地面平面在Y轴上的高度
    pub ground_plane_height: f32,
    /// 使用背景图片
    pub use_background_image: bool,

    // ===== 动画设置 =====
    /// 运行完整动画循环而非单帧渲染
    pub animate: bool,
    /// 动画帧率 (fps)，用于视频生成和预渲染
    pub fps: usize,
    /// 旋转速度系数，控制动画旋转的速度
    pub rotation_speed: f32,
    /// 完整旋转圈数，用于视频生成(默认1圈)
    pub rotation_cycles: f32,
    /// 动画类型 (用于 animate 模式或实时渲染)
    pub animation_type: AnimationType,
    /// 动画旋转轴 (用于 CameraOrbit 和 ObjectLocalRotation)
    pub rotation_axis: RotationAxis,
    /// 自定义旋转轴 (当 rotation_axis 为 Custom 时使用)，格式 "x,y,z"
    pub custom_rotation_axis: String,

    // ===== 光源数组（运行时字段） =====
    /// 场景中的所有光源
    pub lights: Vec<Light>,
}

/// 辅助函数用于解析逗号分隔的浮点数
pub fn parse_vec3(s: &str) -> Result<Vector3<f32>, String> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 3 {
        return Err("需要3个逗号分隔的值".to_string());
    }
    let x = parts[0]
        .trim()
        .parse::<f32>()
        .map_err(|e| format!("无效数字 '{}': {}", parts[0], e))?;
    let y = parts[1]
        .trim()
        .parse::<f32>()
        .map_err(|e| format!("无效数字 '{}': {}", parts[1], e))?;
    let z = parts[2]
        .trim()
        .parse::<f32>()
        .map_err(|e| format!("无效数字 '{}': {}", parts[2], e))?;
    Ok(Vector3::new(x, y, z))
}

pub fn parse_point3(s: &str) -> Result<Point3<f32>, String> {
    parse_vec3(s).map(Point3::from)
}

/// 将 RenderSettings 中的旋转轴配置转换为 Vector3<f32>
pub fn get_animation_axis_vector(settings: &RenderSettings) -> Vector3<f32> {
    match settings.rotation_axis {
        RotationAxis::X => Vector3::x_axis().into_inner(),
        RotationAxis::Y => Vector3::y_axis().into_inner(),
        RotationAxis::Z => Vector3::z_axis().into_inner(),
        RotationAxis::Custom => parse_vec3(&settings.custom_rotation_axis)
            .unwrap_or_else(|_| {
                warn!(
                    "无效的自定义旋转轴 '{}', 使用默认Y轴",
                    settings.custom_rotation_axis
                );
                Vector3::y_axis().into_inner()
            })
            .normalize(),
    }
}

impl Default for RenderSettings {
    fn default() -> Self {
        let mut settings = Self {
            // ===== 文件路径设置 =====
            obj: None,
            output: "output".to_string(),
            output_dir: "output_rust".to_string(),
            texture: None,
            background_image_path: None,

            // ===== 渲染基础设置 =====
            width: 1024,
            height: 1024,
            projection: "perspective".to_string(),
            use_zbuffer: true,
            colorize: false,
            use_texture: true,
            use_gamma: true,
            enable_aces: false,
            backface_culling: false,
            wireframe: false,
            cull_small_triangles: false,
            min_triangle_area: 1e-3,
            save_depth: true,

            // ===== 物体变换控制 =====
            object_position: "0,0,0".to_string(),
            object_rotation: "0,0,0".to_string(),
            object_scale_xyz: "1,1,1".to_string(),
            object_scale: 1.0,

            // ===== 相机参数 =====
            camera_from: "0,0,3".to_string(),
            camera_at: "0,0,0".to_string(),
            camera_up: "0,1,0".to_string(),
            camera_fov: 45.0,

            // ===== 光照基础参数 =====
            use_lighting: true,
            ambient: 0.3,
            ambient_color: "0.3,0.4,0.5".to_string(),

            // ===== 着色模型选择 =====
            use_phong: true,
            use_pbr: false,

            // ===== Phong着色模型参数 =====
            diffuse_color: "0.8,0.8,0.8".to_string(),
            diffuse_intensity: 1.0,
            specular_color: "0.5,0.5,0.5".to_string(),
            specular_intensity: 1.0,
            shininess: 32.0,

            // ===== PBR材质参数 =====
            base_color: "0.8,0.8,0.8".to_string(),
            metallic: 0.0,
            roughness: 0.5,
            ambient_occlusion: 1.0,
            alpha: 1.0, // 默认完全不透明
            emissive: "0.0,0.0,0.0".to_string(),

            // ===== 阴影设置 =====

            // 简单阴影映射配置
            enable_shadow_mapping: false, // 启用地面阴影映射
            shadow_map_size: 256,         // 阴影贴图尺寸（较小，只用于地面）
            shadow_bias: 0.001,           // 阴影偏移
            shadow_distance: 20.0,

            // 新增：PCF相关参数
            enable_pcf: false,           // 是否启用PCF软阴影
            pcf_type: "Box".to_string(), // PCF类型：Box或 Gauss
            pcf_kernel: 2,               // PCF采样窗口半径
            pcf_sigma: 1.0,              // Gauss类型的sigma

            // ===== 背景与环境设置 =====
            enable_gradient_background: false,
            gradient_top_color: "0.5,0.7,1.0".to_string(),
            gradient_bottom_color: "0.1,0.2,0.4".to_string(),
            enable_ground_plane: false,
            ground_plane_color: "0.3,0.5,0.2".to_string(),
            ground_plane_height: -1.0,
            use_background_image: false,

            // ===== 动画设置 =====
            animate: false,
            fps: 30,
            rotation_speed: 1.0,
            rotation_cycles: 1.0,
            animation_type: AnimationType::CameraOrbit,
            rotation_axis: RotationAxis::Y,
            custom_rotation_axis: "0,1,0".to_string(),

            // ===== 光源数组 =====
            lights: Vec::new(),
        };

        // 如果启用了光照且没有光源，创建默认方向光
        settings.initialize_lights();

        settings
    }
}

impl RenderSettings {
    /// 初始化默认光源
    pub fn initialize_lights(&mut self) {
        if self.use_lighting && self.lights.is_empty() {
            self.lights = vec![Light::directional(
                Vector3::new(0.0, -1.0, -1.0),
                Vector3::new(1.0, 1.0, 1.0),
                0.8,
            )];
        }
    }

    // ===== 按需计算方法 =====

    /// 获取环境光颜色向量（按需计算）
    pub fn get_ambient_color_vec(&self) -> Vector3<f32> {
        parse_vec3(&self.ambient_color).unwrap_or_else(|_| Vector3::new(0.1, 0.1, 0.1))
    }

    /// 获取渐变顶部颜色向量（按需计算）
    pub fn get_gradient_top_color_vec(&self) -> Vector3<f32> {
        parse_vec3(&self.gradient_top_color).unwrap_or_else(|_| Vector3::new(0.5, 0.7, 1.0))
    }

    /// 获取渐变底部颜色向量（按需计算）
    pub fn get_gradient_bottom_color_vec(&self) -> Vector3<f32> {
        parse_vec3(&self.gradient_bottom_color).unwrap_or_else(|_| Vector3::new(0.1, 0.2, 0.4))
    }

    /// 获取地面平面颜色向量（按需计算）
    pub fn get_ground_plane_color_vec(&self) -> Vector3<f32> {
        parse_vec3(&self.ground_plane_color).unwrap_or_else(|_| Vector3::new(0.3, 0.5, 0.2))
    }

    /// 解析物体变换参数为向量（统一接口）
    pub fn get_object_transform_components(&self) -> (Vector3<f32>, Vector3<f32>, Vector3<f32>) {
        // 解析位置
        let position =
            parse_vec3(&self.object_position).unwrap_or_else(|_| Vector3::new(0.0, 0.0, 0.0));

        // 解析旋转（度转弧度）
        let rotation_deg =
            parse_vec3(&self.object_rotation).unwrap_or_else(|_| Vector3::new(0.0, 0.0, 0.0));
        let rotation_rad = Vector3::new(
            rotation_deg.x.to_radians(),
            rotation_deg.y.to_radians(),
            rotation_deg.z.to_radians(),
        );

        // 解析缩放
        let scale =
            parse_vec3(&self.object_scale_xyz).unwrap_or_else(|_| Vector3::new(1.0, 1.0, 1.0));

        (position, rotation_rad, scale)
    }

    /// 判断是否使用透视投影
    pub fn is_perspective(&self) -> bool {
        self.projection == "perspective"
    }

    /// 获取着色模型的描述字符串
    pub fn get_lighting_description(&self) -> String {
        if self.use_pbr {
            "基于物理的渲染(PBR)".to_string()
        } else if self.use_phong {
            "Phong着色模型".to_string()
        } else {
            "平面着色模型".to_string()
        }
    }

    /// 验证渲染参数
    pub fn validate(&self) -> Result<(), String> {
        if self.width == 0 || self.height == 0 {
            return Err("错误: 图像宽度和高度必须大于0".to_string());
        }

        if let Some(obj_path) = &self.obj {
            if !std::path::Path::new(obj_path).exists() {
                return Err(format!("错误: 找不到OBJ文件 '{obj_path}'"));
            }
        } else {
            return Err("错误: 未指定OBJ文件路径".to_string());
        }

        if self.output_dir.trim().is_empty() {
            return Err("错误: 输出目录不能为空".to_string());
        }

        if self.output.trim().is_empty() {
            return Err("错误: 输出文件名不能为空".to_string());
        }

        // 验证相机参数
        if parse_vec3(&self.camera_from).is_err() {
            return Err("错误: 相机位置格式不正确，应为 x,y,z 格式".to_string());
        }

        if parse_vec3(&self.camera_at).is_err() {
            return Err("错误: 相机目标格式不正确，应为 x,y,z 格式".to_string());
        }

        if parse_vec3(&self.camera_up).is_err() {
            return Err("错误: 相机上方向格式不正确，应为 x,y,z 格式".to_string());
        }

        // 验证物体变换参数
        if parse_vec3(&self.object_position).is_err() {
            return Err("错误: 物体位置格式不正确，应为 x,y,z 格式".to_string());
        }

        if parse_vec3(&self.object_rotation).is_err() {
            return Err("错误: 物体旋转格式不正确，应为 x,y,z 格式".to_string());
        }

        if parse_vec3(&self.object_scale_xyz).is_err() {
            return Err("错误: 物体缩放格式不正确，应为 x,y,z 格式".to_string());
        }

        Ok(())
    }
}
use nalgebra::Vector3;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// 表示具有浮点分量[0.0, 1.0]的RGB颜色。
pub type Color = Vector3<f32>;

/// 应用gamma矫正，将线性RGB值转换为sRGB空间
///
/// # 参数
/// * `linear_color` - 线性空间的RGB颜色值 [0.0-1.0]
///
/// # 返回值
/// 应用了gamma矫正的RGB颜色值 [0.0-1.0]
pub fn apply_gamma_correction(linear_color: &Color) -> Color {
    // 使用标准的gamma值2.2
    let gamma = 2.2;
    let inv_gamma = 1.0 / gamma;

    // 对每个颜色通道应用幂函数
    Color::new(
        linear_color.x.powf(inv_gamma),
        linear_color.y.powf(inv_gamma),
        linear_color.z.powf(inv_gamma),
    )
}

/// 从sRGB空间转换回线性RGB值（解码）
///
/// # 参数
/// * `srgb_color` - sRGB空间的RGB颜色值 [0.0-1.0]
///
/// # 返回值
/// 线性空间的RGB颜色值 [0.0-1.0]
pub fn srgb_to_linear(srgb_color: &Color) -> Color {
    // 使用标准的gamma值2.2
    let gamma = 2.2;

    // 应用逆变换
    Color::new(
        srgb_color.x.powf(gamma),
        srgb_color.y.powf(gamma),
        srgb_color.z.powf(gamma),
    )
}

/// 应用ACES色调映射，将高动态范围颜色压缩到显示范围
///
/// # 参数
/// * `color` - 线性RGB颜色值 [0.0-1.0]
/// # 返回值
/// 压缩后的RGB颜色值 [0.0-1.0]
pub fn apply_aces_tonemap(color: &Vector3<f32>) -> Vector3<f32> {
    // ACES Filmic Curve参数
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    Vector3::new(
        ((color.x * (a * color.x + b)) / (color.x * (c * color.x + d) + e)).clamp(0.0, 1.0),
        ((color.y * (a * color.y + b)) / (color.y * (c * color.y + d) + e)).clamp(0.0, 1.0),
        ((color.z * (a * color.z + b)) / (color.z * (c * color.z + d) + e)).clamp(0.0, 1.0),
    )
}

/// 将线性RGB值转换为u8数组，应用gamma矫正
///
/// # 参数
/// * `linear_color` - 线性空间的RGB颜色值 [0.0-1.0]
/// * `apply_gamma` - 是否应用gamma矫正
///
/// # 返回值
/// 一个包含三个u8值的数组，表示颜色的RGB通道
pub fn linear_rgb_to_u8(linear_color: &Color, apply_gamma: bool) -> [u8; 3] {
    let display_color = if apply_gamma {
        apply_gamma_correction(linear_color)
    } else {
        *linear_color
    };

    [
        (display_color.x * 255.0).clamp(0.0, 255.0) as u8,
        (display_color.y * 255.0).clamp(0.0, 255.0) as u8,
        (display_color.z * 255.0).clamp(0.0, 255.0) as u8,
    ]
}

/// 获取基于种子的随机颜色。
///
/// 如果`colorize`为false，返回默认的灰色。
/// 如果`colorize`为true，根据种子生成伪随机颜色
/// （对于相同的种子，结果是确定性的）。
///
/// # 参数
/// * `seed` - 随机数种子
/// * `colorize` - 是否生成彩色（否则返回默认灰色）
pub fn get_random_color(seed: u64, colorize: bool) -> Color {
    if !colorize {
        // 默认灰色
        Color::new(0.7, 0.7, 0.7)
    } else {
        // 使用种子生成确定性随机颜色
        let mut rng = StdRng::seed_from_u64(seed);
        Color::new(
            0.3 + rng.random::<f32>() * 0.4, // R 在 [0.3, 0.7) 范围内
            0.3 + rng.random::<f32>() * 0.4, // G 在 [0.3, 0.7) 范围内
            0.3 + rng.random::<f32>() * 0.4, // B 在 [0.3, 0.7) 范围内
        )
    }
}

/// 将归一化的深度图（值范围0.0-1.0）转换为使用JET色彩映射的RGB彩色图像。
///
/// 无效的深度值（NaN、无穷大）将显示为黑色像素。
///
/// # 参数
/// * `normalized_depth` - 扁平化的深度值切片（行优先）。
/// * `width` - 深度图的宽度。
/// * `height` - 深度图的高度。
/// * `apply_gamma` - 是否应用gamma矫正
///
/// # 返回值
/// 包含扁平化RGB图像数据的`Vec<u8>`（每个通道0-255）。
pub fn apply_colormap_jet(
    normalized_depth: &[f32],
    width: usize,
    height: usize,
    apply_gamma: bool,
) -> Vec<u8> {
    let num_pixels = width * height;
    if normalized_depth.len() != num_pixels {
        // 或返回一个错误Result
        panic!("Depth buffer size does not match width * height");
    }

    let mut result = vec![0u8; num_pixels * 3]; // 初始化为黑色

    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            let depth = normalized_depth[index];

            if depth.is_finite() {
                let value = depth.clamp(0.0, 1.0); // 确保值在[0, 1]范围内

                let mut r = 0.0;
                let g;
                let mut b = 0.0;

                // 应用JET色彩映射逻辑
                if value <= 0.25 {
                    // 从蓝色到青色
                    b = 1.0;
                    g = value * 4.0;
                } else if value <= 0.5 {
                    // 从青色到绿色
                    g = 1.0;
                    b = 1.0 - (value - 0.25) * 4.0;
                } else if value <= 0.75 {
                    // 从绿色到黄色
                    g = 1.0;
                    r = (value - 0.5) * 4.0;
                } else {
                    // 从黄色到红色
                    r = 1.0;
                    g = 1.0 - (value - 0.75) * 4.0;
                }

                let color = Color::new(r, g, b);
                let [r_u8, g_u8, b_u8] = linear_rgb_to_u8(&color, apply_gamma);

                // 写入结果缓冲区
                let base_index = index * 3;
                result[base_index] = r_u8;
                result[base_index + 1] = g_u8;
                result[base_index + 2] = b_u8;
            }
            // 如果深度值不是有限的，像素保持黑色（初始化为0）
        }
    }

    result
}
pub mod color;
pub mod light;
pub mod materials;
pub mod texture;
use image::{DynamicImage, GenericImageView};
use log::warn;
use std::path::Path;
use std::sync::Arc;

use crate::material_system::color::{Color, srgb_to_linear};

#[derive(Debug, Clone)]
pub struct Texture {
    pub image: Arc<DynamicImage>,
    pub width: u32,
    pub height: u32,
}

impl Texture {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Option<Self> {
        match image::open(path) {
            Ok(img) => Some(Texture {
                width: img.width(),
                height: img.height(),
                image: Arc::new(img),
            }),
            Err(e) => {
                warn!("无法加载纹理: {e}");
                None
            }
        }
    }

    pub fn sample(&self, u: f32, v: f32) -> [f32; 3] {
        let u = u.fract().abs();
        let v = v.fract().abs();
        let x = (u * self.width as f32) as u32 % self.width;
        let y = ((1.0 - v) * self.height as f32) as u32 % self.height;

        let pixel = self.image.get_pixel(x, y);
        let srgb_color = Color::new(
            pixel[0] as f32 / 255.0,
            pixel[1] as f32 / 255.0,
            pixel[2] as f32 / 255.0,
        );
        let linear_color = srgb_to_linear(&srgb_color);
        [linear_color.x, linear_color.y, linear_color.z]
    }
}
use crate::io::render_settings::{parse_point3, parse_vec3};
use nalgebra::{Point3, Vector3};

/// 统一的光源结构
#[derive(Debug, Clone)]
pub enum Light {
    Directional {
        // 配置字段 (用于GUI控制)
        enabled: bool,
        direction_str: String, // "x,y,z" 格式，用于GUI编辑
        color_str: String,     // "r,g,b" 格式，用于GUI编辑
        intensity: f32,

        // 运行时字段 (用于渲染计算，从配置字段解析)
        direction: Vector3<f32>, // 解析后的方向向量
        color: Vector3<f32>,     // 解析后的颜色向量
    },
    Point {
        // 配置字段 (用于GUI控制)
        enabled: bool,
        position_str: String, // "x,y,z" 格式，用于GUI编辑
        color_str: String,    // "r,g,b" 格式，用于GUI编辑
        intensity: f32,
        constant_attenuation: f32,
        linear_attenuation: f32,
        quadratic_attenuation: f32,

        // 运行时字段 (用于渲染计算，从配置字段解析)
        position: Point3<f32>, // 解析后的位置
        color: Vector3<f32>,   // 解析后的颜色向量
    },
}

impl Light {
    /// 创建方向光
    pub fn directional(direction: Vector3<f32>, color: Vector3<f32>, intensity: f32) -> Self {
        let direction_normalized = direction.normalize();
        Self::Directional {
            enabled: true,
            direction_str: format!(
                "{},{},{}",
                direction_normalized.x, direction_normalized.y, direction_normalized.z
            ),
            color_str: format!("{},{},{}", color.x, color.y, color.z),
            intensity,
            direction: direction_normalized,
            color,
        }
    }

    /// 创建点光源
    pub fn point(
        position: Point3<f32>,
        color: Vector3<f32>,
        intensity: f32,
        attenuation: Option<(f32, f32, f32)>,
    ) -> Self {
        let (constant, linear, quadratic) = attenuation.unwrap_or((1.0, 0.09, 0.032));
        Self::Point {
            enabled: true,
            position_str: format!("{},{},{}", position.x, position.y, position.z),
            color_str: format!("{},{},{}", color.x, color.y, color.z),
            intensity,
            constant_attenuation: constant,
            linear_attenuation: linear,
            quadratic_attenuation: quadratic,
            position,
            color,
        }
    }

    /// 更新运行时字段
    pub fn update_runtime_fields(&mut self) -> Result<(), String> {
        match self {
            Self::Directional {
                direction_str,
                color_str,
                direction,
                color,
                ..
            } => {
                *direction = parse_vec3(direction_str)?.normalize();
                *color = parse_vec3(color_str)?;
            }
            Self::Point {
                position_str,
                color_str,
                position,
                color,
                ..
            } => {
                *position = parse_point3(position_str)?;
                *color = parse_vec3(color_str)?;
            }
        }
        Ok(())
    }

    /// 获取光源方向（用于渲染）
    pub fn get_direction(&self, point: &Point3<f32>) -> Vector3<f32> {
        match self {
            Self::Directional { direction, .. } => -direction,
            Self::Point { position, .. } => (position - point).normalize(),
        }
    }

    /// 获取光源强度（用于渲染）
    pub fn get_intensity(&self, point: &Point3<f32>) -> Vector3<f32> {
        match self {
            Self::Directional {
                color,
                intensity,
                enabled,
                ..
            } => {
                if *enabled {
                    color * *intensity
                } else {
                    Vector3::zeros()
                }
            }
            Self::Point {
                position,
                color,
                intensity,
                constant_attenuation,
                linear_attenuation,
                quadratic_attenuation,
                enabled,
                ..
            } => {
                if *enabled {
                    let distance = (position - point).magnitude();
                    let attenuation_factor = 1.0
                        / (constant_attenuation
                            + linear_attenuation * distance
                            + quadratic_attenuation * distance * distance);
                    color * *intensity * attenuation_factor
                } else {
                    Vector3::zeros()
                }
            }
        }
    }
}
use crate::io::render_settings::{RenderSettings, parse_vec3};
use crate::material_system::texture::Texture;
use log::warn;
use nalgebra::{Point3, Vector2, Vector3};
use std::fmt::Debug;

#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    pub position: Point3<f32>,
    pub normal: Vector3<f32>,
    pub texcoord: Vector2<f32>,
}

/// 材质类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterialType {
    Phong,
    #[allow(clippy::upper_case_acronyms)]
    PBR,
}

/// 材质结构体，统一包含所有参数
#[derive(Debug, Clone)]
pub struct Material {
    pub material_type: MaterialType, // 材质类型
    pub base_color: Vector3<f32>,    // 基础色（PBR/Phong通用）
    pub alpha: f32,                  // 透明度
    pub texture: Option<Texture>,    // 纹理资源

    // ===== PBR参数 =====
    pub metallic: f32,
    pub roughness: f32,
    pub ambient_occlusion: f32,

    // ===== Phong参数 =====
    pub specular: Vector3<f32>,
    pub shininess: f32,
    pub diffuse_intensity: f32,
    pub specular_intensity: f32,

    // ===== 通用参数 =====
    pub emissive: Vector3<f32>,
    pub ambient_factor: Vector3<f32>,
}

impl Material {
    pub fn default(material_type: MaterialType) -> Self {
        Material {
            material_type,
            base_color: Vector3::new(0.8, 0.8, 0.8),
            alpha: 1.0,
            texture: None,
            metallic: 0.0,
            roughness: 0.5,
            ambient_occlusion: 1.0,
            specular: Vector3::new(0.5, 0.5, 0.5),
            shininess: 32.0,
            diffuse_intensity: 1.0,
            specular_intensity: 1.0,
            emissive: Vector3::zeros(),
            ambient_factor: Vector3::new(1.0, 1.0, 1.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub material_id: usize,
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct Model {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
    pub name: String,
}

/// 材质响应计算（统一接口）
pub fn compute_material_response(
    material: &Material,
    light_dir: &Vector3<f32>,
    view_dir: &Vector3<f32>,
    surface_normal: &Vector3<f32>,
) -> Vector3<f32> {
    match material.material_type {
        MaterialType::Phong => {
            let n_dot_l = surface_normal.dot(light_dir).max(0.0);
            if n_dot_l <= 0.0 {
                return material.emissive;
            }
            let diffuse = material.base_color * material.diffuse_intensity * n_dot_l;
            let halfway_dir = (light_dir + view_dir).normalize();
            let n_dot_h = surface_normal.dot(&halfway_dir).max(0.0);
            let spec_intensity = n_dot_h.powf(material.shininess);
            let specular = material.specular * material.specular_intensity * spec_intensity;
            diffuse + specular + material.emissive
        }
        MaterialType::PBR => {
            let base_color = material.base_color;
            let metallic = material.metallic;
            let roughness = material.roughness;
            let ao = material.ambient_occlusion;

            let l = *light_dir;
            let v = *view_dir;
            let h = (l + v).normalize();

            let n_dot_l = surface_normal.dot(&l).max(0.0);
            let n_dot_v = surface_normal.dot(&v).max(0.0);
            let n_dot_h = surface_normal.dot(&h).max(0.0);
            let h_dot_v = h.dot(&v).max(0.0);

            if n_dot_l <= 0.0 {
                return material.emissive;
            }

            // 标准PBR F0计算
            let f0_dielectric = Vector3::new(0.04, 0.04, 0.04);
            let f0 = f0_dielectric.lerp(&base_color, metallic);

            let d = pbr::distribution_ggx(n_dot_h, roughness);
            let g = pbr::geometry_smith(n_dot_v, n_dot_l, roughness);
            let f = pbr::fresnel_schlick(h_dot_v, f0);

            let numerator = d * g * f;
            let denominator = 4.0 * n_dot_v * n_dot_l;
            let specular = numerator / denominator.max(0.001);

            let k_s = f;
            let k_d = (Vector3::new(1.0, 1.0, 1.0) - k_s) * (1.0 - metallic);
            let diffuse = k_d.component_mul(&base_color) / std::f32::consts::PI;

            // 标准Cook-Torrance BRDF
            let brdf_result = (diffuse + specular) * n_dot_l * ao;
            brdf_result + material.emissive
        }
    }
}

pub mod pbr {
    use nalgebra::Vector3;

    pub fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
        let alpha = roughness * roughness;
        let alpha2 = alpha * alpha;
        let n_dot_h2 = n_dot_h * n_dot_h;
        let numerator = alpha2;
        let denominator = n_dot_h2 * (alpha2 - 1.0) + 1.0;
        let denominator = std::f32::consts::PI * denominator * denominator;
        numerator / denominator.max(0.0001)
    }

    pub fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
        let r = roughness + 1.0;
        let k = (r * r) / 8.0;
        n_dot_v / (n_dot_v * (1.0 - k) + k)
    }

    pub fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
        let ggx1 = geometry_schlick_ggx(n_dot_v, roughness);
        let ggx2 = geometry_schlick_ggx(n_dot_l, roughness);
        ggx1 * ggx2
    }

    pub fn fresnel_schlick(cos_theta: f32, f0: Vector3<f32>) -> Vector3<f32> {
        let cos_theta = cos_theta.clamp(0.0, 1.0);
        let one_minus_cos_theta = 1.0 - cos_theta;
        let one_minus_cos_theta5 = one_minus_cos_theta.powi(5);
        f0 + (Vector3::new(1.0, 1.0, 1.0) - f0) * one_minus_cos_theta5
    }
}

/// 材质参数应用（统一接口）
pub fn apply_material_parameters(model: &mut Model, args: &RenderSettings) {
    for material in &mut model.materials {
        match material.material_type {
            MaterialType::PBR => {
                material.metallic = args.metallic.clamp(0.0, 1.0);
                material.roughness = args.roughness.clamp(0.0, 1.0);
                material.ambient_occlusion = args.ambient_occlusion.clamp(0.0, 1.0);
                material.alpha = args.alpha.clamp(0.0, 1.0);

                if let Ok(base_color) = parse_vec3(&args.base_color) {
                    material.base_color = base_color;
                } else {
                    warn!("无法解析基础颜色, 使用默认值: {:?}", material.base_color);
                }

                if let Ok(emissive) = parse_vec3(&args.emissive) {
                    material.emissive = emissive;
                }

                let ambient_response = material.ambient_occlusion * (1.0 - material.metallic);
                material.ambient_factor =
                    Vector3::new(ambient_response, ambient_response, ambient_response);
            }
            MaterialType::Phong => {
                if let Ok(specular_color) = parse_vec3(&args.specular_color) {
                    material.specular = specular_color;
                } else {
                    warn!("无法解析镜面反射颜色, 使用默认值: {:?}", material.specular);
                }

                material.shininess = args.shininess.max(1.0);
                material.diffuse_intensity = args.diffuse_intensity.clamp(0.0, 2.0);
                material.specular_intensity = args.specular_intensity.clamp(0.0, 2.0);
                material.alpha = args.alpha.clamp(0.0, 1.0);

                if let Ok(diffuse_color) = parse_vec3(&args.diffuse_color) {
                    material.base_color = diffuse_color;
                } else {
                    warn!("无法解析漫反射颜色, 使用默认值: {:?}", material.base_color);
                }

                if let Ok(emissive) = parse_vec3(&args.emissive) {
                    material.emissive = emissive;
                }

                material.ambient_factor = material.base_color * 0.3;
            }
        }
    }
}
pub mod scene_object;
pub mod scene_utils;
use crate::geometry::transform::TransformFactory;
use crate::material_system::materials::Model;
use nalgebra::{Matrix4, Vector3};

/// 表示场景中的单个对象实例
///
/// 包含几何数据（模型）和变换信息，是渲染器的基本单位
#[derive(Debug, Clone)]
pub struct SceneObject {
    /// 对象的几何数据（网格、材质等）
    pub model: Model,

    /// 对象在世界空间中的变换矩阵
    pub transform: Matrix4<f32>,
}

impl SceneObject {
    /// 从模型数据创建新的场景对象
    pub fn from_model_data(model: Model) -> Self {
        Self {
            model,
            transform: Matrix4::identity(),
        }
    }

    /// 创建空的场景对象（用于测试或占位）
    pub fn empty(name: &str) -> Self {
        Self {
            model: Model {
                meshes: Vec::new(),
                materials: Vec::new(),
                name: name.to_string(),
            },
            transform: Matrix4::identity(),
        }
    }

    /// 设置完整变换（从组件构建变换矩阵）
    pub fn set_transform_from_components(
        &mut self,
        position: Vector3<f32>,
        rotation_rad: Vector3<f32>,
        scale: Vector3<f32>,
    ) {
        // 按正确顺序组合变换：缩放 -> 旋转 -> 平移
        let scale_matrix = TransformFactory::scaling_nonuniform(&scale);
        let rotation_x_matrix = TransformFactory::rotation_x(rotation_rad.x);
        let rotation_y_matrix = TransformFactory::rotation_y(rotation_rad.y);
        let rotation_z_matrix = TransformFactory::rotation_z(rotation_rad.z);
        let translation_matrix = TransformFactory::translation(&position);

        // 组合变换矩阵：T * Rz * Ry * Rx * S
        self.transform = translation_matrix
            * rotation_z_matrix
            * rotation_y_matrix
            * rotation_x_matrix
            * scale_matrix;
    }

    /// 应用增量旋转（用于动画）
    pub fn rotate(&mut self, axis: &Vector3<f32>, angle_rad: f32) {
        let rotation_matrix = TransformFactory::rotation(axis, angle_rad);
        self.transform = rotation_matrix * self.transform;
    }
}

impl Default for SceneObject {
    fn default() -> Self {
        Self::empty("Default")
    }
}
use crate::geometry::camera::Camera;
use crate::io::render_settings::{RenderSettings, parse_point3, parse_vec3};
use crate::material_system::light::Light;
use crate::material_system::materials::Model;
use crate::material_system::materials::apply_material_parameters;
use crate::scene::scene_object::SceneObject;
use nalgebra::Vector3;

/// 表示一个 3D 场景，包含对象、光源和相机
#[derive(Debug, Clone)]
pub struct Scene {
    /// 场景中的主要对象（简化为单个对象）
    pub object: SceneObject,

    /// 场景中的光源
    pub lights: Vec<Light>,

    /// 当前活动相机
    pub active_camera: Camera,

    /// 环境光强度
    pub ambient_intensity: f32,

    /// 环境光颜色
    pub ambient_color: Vector3<f32>,
}

impl Scene {
    /// 链式创建场景，自动应用所有设置
    pub fn new(model_data: Model, settings: &RenderSettings) -> Result<Self, String> {
        let mut model_data = model_data.clone();
        // 应用材质参数
        apply_material_parameters(&mut model_data, settings);

        // 创建对象
        let mut object = SceneObject::from_model_data(model_data);

        // 应用对象变换
        let (position, rotation_rad, scale) = settings.get_object_transform_components();
        let final_scale = if settings.object_scale != 1.0 {
            scale * settings.object_scale
        } else {
            scale
        };
        object.set_transform_from_components(position, rotation_rad, final_scale);

        // 相机
        let aspect_ratio = settings.width as f32 / settings.height as f32;
        let camera_from =
            parse_point3(&settings.camera_from).map_err(|e| format!("无效的相机位置格式: {e}"))?;
        let camera_at =
            parse_point3(&settings.camera_at).map_err(|e| format!("无效的相机目标格式: {e}"))?;
        let camera_up =
            parse_vec3(&settings.camera_up).map_err(|e| format!("无效的相机上方向格式: {e}"))?;
        let camera = match settings.projection.as_str() {
            "perspective" => Camera::perspective(
                camera_from,
                camera_at,
                camera_up,
                settings.camera_fov,
                aspect_ratio,
                0.1,
                100.0,
            ),
            "orthographic" => {
                let height = 4.0;
                let width = height * aspect_ratio;
                Camera::orthographic(camera_from, camera_at, camera_up, width, height, 0.1, 100.0)
            }
            _ => return Err(format!("不支持的投影类型: {}", settings.projection)),
        };

        // 光源
        let lights = settings.lights.clone();

        // 环境光
        let ambient_intensity = settings.ambient;
        let ambient_color = settings.get_ambient_color_vec();

        Ok(Scene {
            object,
            lights,
            active_camera: camera,
            ambient_intensity,
            ambient_color,
        })
    }

    /// 链式设置对象变换
    pub fn set_object_transform(
        &mut self,
        position: Vector3<f32>,
        rotation_rad: Vector3<f32>,
        scale: Vector3<f32>,
    ) -> &mut Self {
        self.object
            .set_transform_from_components(position, rotation_rad, scale);
        self
    }

    /// 链式设置光源
    pub fn set_lights(&mut self, lights: Vec<Light>) -> &mut Self {
        self.lights = lights;
        self
    }

    /// 链式设置相机
    pub fn set_camera(&mut self, camera: Camera) -> &mut Self {
        self.active_camera = camera;
        self
    }

    /// 链式设置环境光
    pub fn set_ambient(&mut self, intensity: f32, color: Vector3<f32>) -> &mut Self {
        self.ambient_intensity = intensity;
        self.ambient_color = color;
        self
    }

    /// 获取场景统计信息
    pub fn get_scene_stats(&self) -> SceneStats {
        let mut vertex_count = 0;
        let mut triangle_count = 0;
        let material_count = self.object.model.materials.len();
        let mesh_count = self.object.model.meshes.len();

        for mesh in &self.object.model.meshes {
            vertex_count += mesh.vertices.len();
            triangle_count += mesh.indices.len() / 3;
        }

        SceneStats {
            vertex_count,
            triangle_count,
            material_count,
            mesh_count,
            light_count: self.lights.len(),
        }
    }
}

/// 场景统计信息
#[derive(Debug, Clone)]
pub struct SceneStats {
    pub vertex_count: usize,
    pub triangle_count: usize,
    pub material_count: usize,
    pub mesh_count: usize,
    pub light_count: usize,
}
use crate::ModelLoader;
use crate::core::renderer::Renderer;
use crate::geometry::camera::ProjectionType;
use crate::io::render_settings::{RenderSettings, parse_point3, parse_vec3};
use crate::material_system::materials::apply_material_parameters;
use crate::ui::app::RasterizerApp;
use crate::utils::render_utils::calculate_rotation_parameters;
use crate::utils::save_utils::save_render_with_settings;
use egui::{Color32, Context};
use log::{debug, error, warn};
use std::fs;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use super::app::InterfaceInteraction;

/// 核心业务逻辑方法
///
/// 该trait包含应用的核心功能：
/// - 渲染和加载逻辑
/// - 状态转换与管理
/// - 错误处理
/// - 性能统计
/// - 资源管理
pub trait CoreMethods {
    // === 核心渲染和加载 ===

    /// 渲染当前场景 - 统一渲染入口
    fn render(&mut self, ctx: &Context);

    /// 在UI中显示渲染结果
    fn display_render_result(&mut self, ctx: &Context);

    /// 如果任何事情发生变化，执行重新渲染
    fn render_if_anything_changed(&mut self, ctx: &Context);

    /// 保存当前渲染结果为截图
    fn take_screenshot(&mut self) -> Result<String, String>;

    /// 智能计算地面平面的最佳高度
    fn calculate_optimal_ground_height(&self) -> Option<f32>;

    // === 状态管理 ===

    /// 设置错误信息
    fn set_error(&mut self, message: String);

    /// 将应用状态重置为默认值
    fn reset_to_defaults(&mut self);

    /// 切换预渲染模式开启/关闭状态
    fn toggle_pre_render_mode(&mut self);

    /// 清空预渲染的动画帧缓冲区
    fn clear_pre_rendered_frames(&mut self);

    // === 状态查询 ===

    /// 检查是否可以清除预渲染缓冲区
    fn can_clear_buffer(&self) -> bool;

    /// 检查是否可以切换预渲染模式
    fn can_toggle_pre_render(&self) -> bool;

    /// 检查是否可以开始或停止动画渲染
    fn can_render_animation(&self) -> bool;

    /// 检查是否可以生成视频
    fn can_generate_video(&self) -> bool;

    // === 动画状态管理 ===

    /// 开始实时渲染动画
    fn start_animation_rendering(&mut self) -> Result<(), String>;

    /// 停止实时渲染动画
    fn stop_animation_rendering(&mut self);

    // === 性能统计 ===

    /// 更新帧率统计信息
    fn update_fps_stats(&mut self, frame_time: Duration);

    /// 获取格式化的帧率显示文本和颜色
    fn get_fps_display(&self) -> (String, Color32);

    // === 资源管理 ===

    /// 执行资源清理操作
    fn cleanup_resources(&mut self);
}

impl CoreMethods for RasterizerApp {
    // === 核心渲染和加载实现 ===

    /// 渲染当前场景
    fn render(&mut self, ctx: &Context) {
        // 验证参数
        if let Err(e) = self.settings.validate() {
            self.set_error(e);
            return;
        }

        // 获取OBJ路径
        let obj_path = match &self.settings.obj {
            Some(path) => path.clone(),
            None => {
                self.set_error("错误: 未指定OBJ文件路径".to_string());
                return;
            }
        };

        self.status_message = format!("正在加载 {obj_path}...");
        ctx.request_repaint(); // 立即更新状态消息

        // 加载模型
        match ModelLoader::load_and_create_scene(&obj_path, &self.settings) {
            Ok((scene, model_data)) => {
                debug!(
                    "场景创建完成: 光源数量={}, 使用光照={}, 环境光强度={}",
                    scene.lights.len(),
                    self.settings.use_lighting,
                    self.settings.ambient
                );

                // 直接设置场景和模型数据
                self.scene = Some(scene);
                self.model_data = Some(model_data);

                self.status_message = "模型加载成功，开始渲染...".to_string();
            }
            Err(e) => {
                self.set_error(format!("加载模型失败: {e}"));
                return;
            }
        }

        self.status_message = "模型加载成功，开始渲染...".to_string();
        ctx.request_repaint();

        // 确保输出目录存在
        let output_dir = self.settings.output_dir.clone();
        if let Err(e) = fs::create_dir_all(&output_dir) {
            self.set_error(format!("创建输出目录失败: {e}"));
            return;
        }

        // 渲染
        let start_time = Instant::now();

        if let Some(scene) = &mut self.scene {
            // 渲染到帧缓冲区
            self.renderer.render_scene(scene, &self.settings);

            // 保存输出文件
            if let Err(e) = save_render_with_settings(&self.renderer, &self.settings, None) {
                warn!("保存渲染结果时发生错误: {e}");
            }

            // 更新状态
            self.last_render_time = Some(start_time.elapsed());
            let output_dir = self.settings.output_dir.clone();
            let output_name = self.settings.output.clone();
            let elapsed = self.last_render_time.unwrap();
            self.status_message =
                format!("渲染完成，耗时 {elapsed:.2?}，已保存到 {output_dir}/{output_name}");

            // 在UI中显示渲染结果
            self.display_render_result(ctx);
        }
    }

    /// 在UI中显示渲染结果
    fn display_render_result(&mut self, ctx: &Context) {
        // 从渲染器获取图像数据
        let color_data = self.renderer.frame_buffer.get_color_buffer_bytes();

        // 确保分辨率与渲染器匹配
        let width = self.renderer.frame_buffer.width;
        let height = self.renderer.frame_buffer.height;

        // 创建或更新纹理
        let rendered_texture = self.rendered_image.get_or_insert_with(|| {
            // 创建一个全黑的空白图像
            let color = Color32::BLACK;
            ctx.load_texture(
                "rendered_image",
                egui::ColorImage::new([width, height], vec![color; width * height]),
                egui::TextureOptions::default(),
            )
        });

        // 将RGB数据转换为RGBA格式
        let mut rgba_data = Vec::with_capacity(color_data.len() / 3 * 4);
        for i in (0..color_data.len()).step_by(3) {
            if i + 2 < color_data.len() {
                rgba_data.push(color_data[i]); // R
                rgba_data.push(color_data[i + 1]); // G
                rgba_data.push(color_data[i + 2]); // B
                rgba_data.push(255); // A (完全不透明)
            }
        }

        // 更新纹理，使用渲染器的实际大小
        rendered_texture.set(
            egui::ColorImage::from_rgba_unmultiplied([width, height], &rgba_data),
            egui::TextureOptions::default(),
        );
    }

    /// 统一同步入口
    fn render_if_anything_changed(&mut self, ctx: &Context) {
        if self.interface_interaction.anything_changed && self.scene.is_some() {
            if let Some(scene) = &mut self.scene {
                // 检测渲染尺寸变化
                if self.renderer.frame_buffer.width != self.settings.width
                    || self.renderer.frame_buffer.height != self.settings.height
                {
                    self.renderer.frame_buffer.invalidate_caches();
                }

                // 强制清除地面本体和阴影缓存
                self.renderer.frame_buffer.invalidate_ground_base_cache();
                self.renderer.frame_buffer.invalidate_ground_shadow_cache();

                // 统一同步所有状态

                // 1. 光源同步
                scene.set_lights(self.settings.lights.clone());

                // 2. 相机同步
                if let Ok(from) = parse_point3(&self.settings.camera_from) {
                    scene.active_camera.params.position = from;
                }
                if let Ok(at) = parse_point3(&self.settings.camera_at) {
                    scene.active_camera.params.target = at;
                }
                if let Ok(up) = parse_vec3(&self.settings.camera_up) {
                    scene.active_camera.params.up = up.normalize();
                }
                if let ProjectionType::Perspective { fov_y_degrees, .. } =
                    &mut scene.active_camera.params.projection
                {
                    *fov_y_degrees = self.settings.camera_fov;
                }
                scene.active_camera.update_matrices();

                // 3. 物体变换同步
                let (position, rotation_rad, scale) =
                    self.settings.get_object_transform_components();
                let final_scale = if self.settings.object_scale != 1.0 {
                    scale * self.settings.object_scale
                } else {
                    scale
                };
                scene.set_object_transform(position, rotation_rad, final_scale);

                // 4. 材质参数同步
                apply_material_parameters(&mut scene.object.model, &self.settings);

                // 5. 环境光同步
                scene.set_ambient(self.settings.ambient, self.settings.get_ambient_color_vec());

                // 6. 执行渲染
                self.renderer.render_scene(scene, &self.settings);
            }

            self.display_render_result(ctx);
            self.interface_interaction.anything_changed = false;
        }
    }

    /// 保存当前渲染结果为截图
    fn take_screenshot(&mut self) -> Result<String, String> {
        // 确保输出目录存在
        if let Err(e) = fs::create_dir_all(&self.settings.output_dir) {
            return Err(format!("创建输出目录失败: {e}"));
        }

        // 生成唯一的文件名（基于时间戳）
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| format!("获取时间戳失败: {e}"))?
            .as_secs();

        let snapshot_name = format!("{}_snapshot_{}", self.settings.output, timestamp);

        // 检查是否有可用的渲染结果
        if self.rendered_image.is_none() {
            return Err("没有可用的渲染结果".to_string());
        }

        // 使用共享的渲染工具函数保存截图
        save_render_with_settings(&self.renderer, &self.settings, Some(&snapshot_name))?;

        // 返回颜色图像的路径
        let color_path =
            Path::new(&self.settings.output_dir).join(format!("{snapshot_name}_color.png"));
        Ok(color_path.to_string_lossy().to_string())
    }

    fn calculate_optimal_ground_height(&self) -> Option<f32> {
        let scene = self.scene.as_ref()?;
        let model_data = self.model_data.as_ref()?;

        let mut min_y = f32::INFINITY;
        let mut has_vertices = false;

        // 计算模型在当前变换下的最低点
        for mesh in &model_data.meshes {
            for vertex in &mesh.vertices {
                let world_pos = scene.object.transform.transform_point(&vertex.position);
                min_y = min_y.min(world_pos.y);
                has_vertices = true;
            }
        }

        if !has_vertices {
            return None;
        }

        // 智能策略：让物体贴地，避免浮空
        let ground_height = if self.settings.enable_shadow_mapping {
            // 阴影映射时：让物体稍微"嵌入"地面，确保阴影可见但物体不浮空
            min_y - 0.01 // 非常小的负偏移，让物体微微贴地
        } else {
            // 无阴影时：让物体完全贴地
            min_y
        };

        Some(ground_height)
    }

    // === 状态管理实现 ===

    /// 设置错误信息
    fn set_error(&mut self, message: String) {
        error!("{message}");
        self.status_message = format!("错误: {message}");
    }

    /// 重置应用状态到默认值
    fn reset_to_defaults(&mut self) {
        // 保留当前的文件路径设置
        let obj_path = self.settings.obj.clone();
        let output_dir = self.settings.output_dir.clone();
        let output_name = self.settings.output.clone();

        let new_settings = RenderSettings {
            obj: obj_path,
            output_dir,
            output: output_name,
            ..Default::default()
        };

        // 如果渲染尺寸变化，重新创建渲染器
        if self.renderer.frame_buffer.width != new_settings.width
            || self.renderer.frame_buffer.height != new_settings.height
        {
            self.renderer = Renderer::new(new_settings.width, new_settings.height);
            self.rendered_image = None;
        }

        self.settings = new_settings;

        // 重置GUI状态
        self.camera_pan_sensitivity = 1.0;
        self.camera_orbit_sensitivity = 1.0;
        self.camera_dolly_sensitivity = 1.0;
        self.interface_interaction = InterfaceInteraction::default();

        // 重置其他状态
        self.is_realtime_rendering = false;
        self.is_pre_rendering = false;
        self.is_generating_video = false;
        self.pre_render_mode = false;
        self.animation_time = 0.0;
        self.current_frame_index = 0;
        self.last_frame_time = None;

        // 清空预渲染缓冲区
        if let Ok(mut frames) = self.pre_rendered_frames.lock() {
            frames.clear();
        }

        self.pre_render_progress.store(0, Ordering::SeqCst);
        self.video_progress.store(0, Ordering::SeqCst);

        // 重置 FPS 统计
        self.current_fps = 0.0;
        self.fps_history.clear();
        self.avg_fps = 0.0;

        self.status_message = "已重置应用状态，光源已恢复默认设置".to_string();
    }

    /// 切换预渲染模式
    fn toggle_pre_render_mode(&mut self) {
        // 统一的状态检查
        if self.is_pre_rendering || self.is_generating_video || self.is_realtime_rendering {
            self.status_message = "无法更改渲染模式: 请先停止正在进行的操作".to_string();
            return;
        }

        // 切换模式
        self.pre_render_mode = !self.pre_render_mode;

        if self.pre_render_mode {
            // 确保旋转速度合理
            if self.settings.rotation_speed.abs() < 0.01 {
                self.settings.rotation_speed = 1.0;
            }
            self.status_message = "已启用预渲染模式，开始动画渲染时将预先计算所有帧".to_string();
        } else {
            self.status_message = "已禁用预渲染模式，缓冲区中的预渲染帧仍可使用".to_string();
        }
    }

    /// 清空预渲染帧缓冲区
    fn clear_pre_rendered_frames(&mut self) {
        // 统一的状态检查逻辑
        if self.is_realtime_rendering || self.is_pre_rendering {
            self.status_message = "无法清除缓冲区: 请先停止动画渲染或等待预渲染完成".to_string();
            return;
        }

        // 执行清除操作
        let had_frames = !self.pre_rendered_frames.lock().unwrap().is_empty();
        if had_frames {
            self.pre_rendered_frames.lock().unwrap().clear();
            self.current_frame_index = 0;
            self.pre_render_progress.store(0, Ordering::SeqCst);

            if self.is_generating_video {
                let (_, _, frames_per_rotation) =
                    calculate_rotation_parameters(self.settings.rotation_speed, self.settings.fps);
                let total_frames =
                    (frames_per_rotation as f32 * self.settings.rotation_cycles) as usize;
                let progress = self.video_progress.load(Ordering::SeqCst);
                let percent = (progress as f32 / total_frames as f32 * 100.0).round();

                self.status_message =
                    format!("生成视频中... ({progress}/{total_frames}，{percent:.0}%)");
            } else {
                self.status_message = "已清空预渲染缓冲区".to_string();
            }
        } else {
            self.status_message = "缓冲区已为空".to_string();
        }
    }

    // === 状态查询实现 ===

    fn can_clear_buffer(&self) -> bool {
        !self.pre_rendered_frames.lock().unwrap().is_empty()
            && !self.is_realtime_rendering
            && !self.is_pre_rendering
    }

    fn can_toggle_pre_render(&self) -> bool {
        !self.is_pre_rendering && !self.is_generating_video && !self.is_realtime_rendering
    }

    fn can_render_animation(&self) -> bool {
        !self.is_generating_video
    }

    fn can_generate_video(&self) -> bool {
        !self.is_realtime_rendering && !self.is_generating_video && self.ffmpeg_available
    }

    // === 动画状态管理实现 ===

    fn start_animation_rendering(&mut self) -> Result<(), String> {
        if self.is_generating_video {
            return Err("无法开始动画: 视频正在生成中".to_string());
        }

        self.is_realtime_rendering = true;
        self.last_frame_time = None;
        self.current_fps = 0.0;
        self.fps_history.clear();
        self.avg_fps = 0.0;
        self.status_message = "开始动画渲染...".to_string();

        Ok(())
    }

    fn stop_animation_rendering(&mut self) {
        self.is_realtime_rendering = false;
        self.status_message = "已停止动画渲染".to_string();
    }

    // === 性能统计实现 ===

    fn update_fps_stats(&mut self, frame_time: Duration) {
        const FPS_HISTORY_SIZE: usize = 30;
        let current_fps = 1.0 / frame_time.as_secs_f32();
        self.current_fps = current_fps;

        // 更新 FPS 历史
        self.fps_history.push(current_fps);
        if self.fps_history.len() > FPS_HISTORY_SIZE {
            self.fps_history.remove(0); // 移除最早的记录
        }

        // 计算平均 FPS
        if !self.fps_history.is_empty() {
            let sum: f32 = self.fps_history.iter().sum();
            self.avg_fps = sum / self.fps_history.len() as f32;
        }
    }

    fn get_fps_display(&self) -> (String, Color32) {
        // 根据 FPS 水平选择颜色
        let fps_color = if self.avg_fps >= 30.0 {
            Color32::from_rgb(50, 220, 50) // 绿色
        } else if self.avg_fps >= 15.0 {
            Color32::from_rgb(220, 180, 50) // 黄色
        } else {
            Color32::from_rgb(220, 50, 50) // 红色
        };

        (format!("FPS: {:.1}", self.avg_fps), fps_color)
    }

    // === 资源管理实现 ===

    fn cleanup_resources(&mut self) {
        // 实际的资源清理逻辑

        // 1. 限制FPS历史记录大小，防止内存泄漏
        if self.fps_history.len() > 60 {
            self.fps_history.drain(0..30); // 保留最近30帧的数据
        }

        // 2. 清理已完成的视频生成线程
        if let Some(handle) = &self.video_generation_thread {
            if handle.is_finished() {
                // 线程已完成，标记需要在主循环中处理
                debug!("检测到已完成的视频生成线程，等待主循环处理");
            }
        }

        // 3. 在空闲状态下进行额外清理
        if !self.is_realtime_rendering && !self.is_generating_video && !self.is_pre_rendering {
            // 清理可能的临时资源
            if self.rendered_image.is_some() && self.last_render_time.is_none() {
                // 如果有渲染结果但没有最近的渲染时间，说明可能是陈旧的结果
                // 这里可以添加更多清理逻辑
            }

            // 清理预渲染进度计数器（如果没有预渲染帧）
            if self.pre_rendered_frames.lock().unwrap().is_empty() {
                self.pre_render_progress.store(0, Ordering::SeqCst);
            }
        }
    }
}
pub mod animation;
pub mod app;
pub mod core;
pub mod render_ui;
pub mod widgets;
use crate::Renderer;
use crate::io::config_loader::TomlConfigLoader;
use crate::io::model_loader::ModelLoader;
use crate::io::render_settings::RenderSettings;
use crate::ui::app::RasterizerApp;
use log::debug;
use native_dialog::FileDialogBuilder;

/// 渲染UI交互方法的特质
///
/// 该trait专门处理与文件选择和UI交互相关的功能：
/// - 文件选择对话框
/// - 背景图片处理
/// - 输出目录选择
/// - 配置文件管理
pub trait RenderUIMethods {
    /// 选择OBJ文件
    fn select_obj_file(&mut self);

    /// 选择纹理文件
    fn select_texture_file(&mut self);

    /// 选择背景图片
    fn select_background_image(&mut self);

    /// 选择输出目录
    fn select_output_dir(&mut self);

    /// 加载配置文件
    fn load_config_file(&mut self);

    /// 保存配置文件
    fn save_config_file(&mut self);

    /// 应用加载的配置到GUI
    fn apply_loaded_config(&mut self, settings: RenderSettings);
}

impl RenderUIMethods for RasterizerApp {
    /// 选择OBJ文件
    fn select_obj_file(&mut self) {
        let result = FileDialogBuilder::default()
            .set_title("选择OBJ模型文件")
            .add_filter("OBJ模型", ["obj"])
            .open_single_file()
            .show();

        match result {
            Ok(Some(path)) => {
                if let Some(path_str) = path.to_str() {
                    self.settings.obj = Some(path_str.to_string());
                    self.status_message = format!("已选择模型: {path_str}");

                    // OBJ文件变化需要重新加载场景和重新渲染
                    self.interface_interaction.anything_changed = true;
                    self.scene = None; // 清除现有场景，强制重新加载
                    self.rendered_image = None; // 清除渲染结果
                }
            }
            Ok(None) => {
                self.status_message = "文件选择被取消".to_string();
            }
            Err(e) => {
                self.set_error(format!("文件选择器错误: {e}"));
            }
        }
    }

    /// 选择纹理文件
    fn select_texture_file(&mut self) {
        let result = FileDialogBuilder::default()
            .set_title("选择纹理文件")
            .add_filter("图像文件", ["png", "jpg", "jpeg", "bmp", "tga"])
            .open_single_file()
            .show();

        match result {
            Ok(Some(path)) => {
                if let Some(path_str) = path.to_str() {
                    self.settings.texture = Some(path_str.to_string());
                    self.status_message = format!("已选择纹理: {path_str}");

                    // 纹理变化需要重新渲染
                    self.interface_interaction.anything_changed = true;
                }
            }
            Ok(None) => {
                self.status_message = "纹理选择被取消".to_string();
            }
            Err(e) => {
                self.set_error(format!("纹理选择错误: {e}"));
            }
        }
    }

    /// 选择背景图片
    fn select_background_image(&mut self) {
        let result = FileDialogBuilder::default()
            .set_title("选择背景图片")
            .add_filter("图片文件", ["png", "jpg", "jpeg", "bmp"])
            .open_single_file()
            .show();

        match result {
            Ok(Some(path)) => {
                if let Some(path_str) = path.to_str() {
                    // 只设置背景图片路径，不再直接加载到 settings
                    self.settings.background_image_path = Some(path_str.to_string());
                    self.settings.use_background_image = true;

                    // 使用 ModelLoader 验证背景图片是否有效
                    match ModelLoader::validate_resources(&self.settings) {
                        Ok(_) => {
                            self.status_message = format!("背景图片配置成功: {path_str}");

                            // 清除已渲染的图像，强制重新渲染以应用新背景
                            self.rendered_image = None;

                            debug!("背景图片路径已设置: {path_str}");
                            debug!("背景图片将在下次渲染时由 FrameBuffer 自动加载");
                        }
                        Err(e) => {
                            // 验证失败，重置背景设置
                            self.set_error(format!("背景图片验证失败: {e}"));
                            self.settings.background_image_path = None;
                            self.settings.use_background_image = false;
                        }
                    }
                }
            }
            Ok(None) => {
                self.status_message = "图片选择被取消".to_string();
            }
            Err(e) => {
                self.set_error(format!("文件选择器错误: {e}"));
            }
        }
    }

    /// 选择输出目录
    fn select_output_dir(&mut self) {
        let result = FileDialogBuilder::default()
            .set_title("选择输出目录")
            .open_single_dir()
            .show();

        match result {
            Ok(Some(path)) => {
                if let Some(path_str) = path.to_str() {
                    self.settings.output_dir = path_str.to_string();
                    self.status_message = format!("已选择输出目录: {path_str}");
                }
            }
            Ok(None) => {
                self.status_message = "目录选择被取消".to_string();
            }
            Err(e) => {
                self.set_error(format!("目录选择器错误: {e}"));
            }
        }
    }

    /// 加载配置文件
    fn load_config_file(&mut self) {
        let result = FileDialogBuilder::default()
            .set_title("加载配置文件")
            .add_filter("TOML配置文件", ["toml"])
            .open_single_file()
            .show();

        match result {
            Ok(Some(path)) => {
                if let Some(path_str) = path.to_str() {
                    match TomlConfigLoader::load_from_file(path_str) {
                        Ok(loaded_settings) => {
                            self.apply_loaded_config(loaded_settings);
                            self.status_message = format!("配置已加载: {path_str}");
                        }
                        Err(e) => {
                            self.set_error(format!("配置加载失败: {e}"));
                        }
                    }
                }
            }
            Ok(None) => {
                self.status_message = "配置加载被取消".to_string();
            }
            Err(e) => {
                self.set_error(format!("文件选择器错误: {e}"));
            }
        }
    }

    /// 保存配置文件
    fn save_config_file(&mut self) {
        let result = FileDialogBuilder::default()
            .set_title("保存配置文件")
            .add_filter("TOML配置文件", ["toml"])
            .save_single_file()
            .show();

        match result {
            Ok(Some(path)) => {
                let mut save_path = path;

                // 自动添加.toml扩展名（如果没有）
                if save_path.extension().is_none() {
                    save_path.set_extension("toml");
                }

                if let Some(path_str) = save_path.to_str() {
                    match TomlConfigLoader::save_to_file(&self.settings, path_str) {
                        Ok(_) => {
                            self.status_message = format!("配置已保存: {path_str}");
                        }
                        Err(e) => {
                            self.set_error(format!("配置保存失败: {e}"));
                        }
                    }
                }
            }
            Ok(None) => {
                self.status_message = "配置保存被取消".to_string();
            }
            Err(e) => {
                self.set_error(format!("文件选择器错误: {e}"));
            }
        }
    }

    /// 应用加载的配置到GUI
    fn apply_loaded_config(&mut self, loaded_settings: RenderSettings) {
        // 直接替换settings，无需同步GUI专用向量字段
        self.settings = loaded_settings;

        // 如果分辨率变化，重新创建渲染器
        if self.renderer.frame_buffer.width != self.settings.width
            || self.renderer.frame_buffer.height != self.settings.height
        {
            self.renderer = Renderer::new(self.settings.width, self.settings.height);
        }

        // 清除现有场景和渲染结果，强制重新加载
        self.scene = None;
        self.rendered_image = None;
        self.interface_interaction.anything_changed = true;

        debug!("配置已应用到GUI界面");
    }
}
use crate::ModelLoader;
use crate::core::renderer::Renderer;
use crate::io::render_settings::{AnimationType, RenderSettings, get_animation_axis_vector};
use crate::scene::scene_utils::Scene;
use crate::utils::render_utils::{
    animate_scene_step, calculate_rotation_delta, calculate_rotation_parameters,
};
use crate::utils::save_utils::save_image;
use egui::{ColorImage, Context, TextureOptions};
use log::debug;
use std::fs;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use super::app::RasterizerApp;
use super::core::CoreMethods;

/// 将ColorImage转换为PNG数据
pub fn frame_to_png_data(image: &ColorImage) -> Vec<u8> {
    // ColorImage是RGBA格式，我们需要转换为RGB格式
    let mut rgb_data = Vec::with_capacity(image.width() * image.height() * 3);
    for pixel in &image.pixels {
        rgb_data.push(pixel.r());
        rgb_data.push(pixel.g());
        rgb_data.push(pixel.b());
    }
    rgb_data
}

/// 渲染一圈的动画帧
///
/// # 参数
/// * `scene_copy` - 场景的克隆
/// * `settings` - 渲染参数
/// * `progress_arc` - 进度计数器
/// * `ctx_clone` - UI上下文，用于更新界面
/// * `width` - 渲染宽度
/// * `height` - 渲染高度
/// * `on_frame_rendered` - 帧渲染完成后的回调函数，参数为(帧序号, RGB颜色数据)
///
/// # 返回值
/// 渲染的总帧数
fn render_one_rotation_cycle<F>(
    mut scene_copy: Scene,
    settings: &RenderSettings,
    progress_arc: &Arc<AtomicUsize>,
    ctx_clone: &Context,
    width: usize,
    height: usize,
    mut on_frame_rendered: F,
) -> usize
where
    F: FnMut(usize, Vec<u8>),
{
    let mut thread_renderer = Renderer::new(width, height);
    let (effective_rotation_speed_dps, _, frames_to_render) =
        calculate_rotation_parameters(settings.rotation_speed, settings.fps);

    let rotation_axis_vec = get_animation_axis_vector(settings);
    let rotation_increment_rad_per_frame =
        (360.0 / frames_to_render as f32).to_radians() * effective_rotation_speed_dps.signum();

    for frame_num in 0..frames_to_render {
        progress_arc.store(frame_num, Ordering::SeqCst);

        if frame_num > 0 {
            animate_scene_step(
                &mut scene_copy,
                &settings.animation_type,
                &rotation_axis_vec,
                rotation_increment_rad_per_frame,
            );
        }

        // === 缓存失效策略 ===
        match settings.animation_type {
            AnimationType::CameraOrbit => {
                // 相机轨道动画：地面本体和阴影都依赖相机，必须全部失效
                thread_renderer.frame_buffer.invalidate_ground_base_cache();
                thread_renderer
                    .frame_buffer
                    .invalidate_ground_shadow_cache();
            }
            AnimationType::ObjectLocalRotation => {
                if settings.enable_shadow_mapping {
                    // 物体动画+阴影：只需失效阴影缓存，地面本体可复用
                    thread_renderer
                        .frame_buffer
                        .invalidate_ground_shadow_cache();
                }
                // 未开阴影时，地面缓存可复用，无需清理
            }
            AnimationType::None => {}
        }

        thread_renderer.render_scene(&mut scene_copy, settings);
        let color_data_rgb = thread_renderer.frame_buffer.get_color_buffer_bytes();
        on_frame_rendered(frame_num, color_data_rgb);

        if frame_num % (frames_to_render.max(1) / 20).max(1) == 0 {
            ctx_clone.request_repaint();
        }
    }

    progress_arc.store(frames_to_render, Ordering::SeqCst);
    ctx_clone.request_repaint();

    frames_to_render
}

/// 动画与视频生成相关方法的特质
pub trait AnimationMethods {
    /// 执行实时渲染循环
    fn perform_realtime_rendering(&mut self, ctx: &Context);

    /// 在后台生成视频
    fn start_video_generation(&mut self, ctx: &Context);

    /// 启动预渲染过程
    fn start_pre_rendering(&mut self, ctx: &Context);

    /// 处理预渲染帧
    fn handle_pre_rendering_tasks(&mut self, ctx: &Context);

    /// 播放预渲染帧
    fn play_pre_rendered_frames(&mut self, ctx: &Context);
}

impl AnimationMethods for RasterizerApp {
    /// 执行实时渲染循环
    fn perform_realtime_rendering(&mut self, ctx: &Context) {
        // 如果启用了预渲染模式且没有预渲染帧，才进入预渲染
        if self.pre_render_mode
            && !self.is_pre_rendering
            && self.pre_rendered_frames.lock().unwrap().is_empty()
        {
            // 检查模型是否已加载
            if self.scene.is_none() {
                let obj_path = match &self.settings.obj {
                    Some(path) => path.clone(),
                    None => {
                        self.set_error("错误: 未指定OBJ文件路径".to_string());
                        self.stop_animation_rendering();
                        return;
                    }
                };
                match ModelLoader::load_and_create_scene(&obj_path, &self.settings) {
                    Ok((scene, model_data)) => {
                        self.scene = Some(scene);
                        self.model_data = Some(model_data);
                        self.start_pre_rendering(ctx);
                        return;
                    }
                    Err(e) => {
                        self.set_error(format!("加载模型失败: {e}"));
                        self.stop_animation_rendering();
                        return;
                    }
                }
            } else {
                self.start_pre_rendering(ctx);
                return;
            }
        }

        // 如果正在预渲染，处理预渲染任务
        if self.is_pre_rendering {
            self.handle_pre_rendering_tasks(ctx);
            return;
        }

        // 如果启用预渲染模式且有预渲染帧，播放预渲染帧
        if self.pre_render_mode && !self.pre_rendered_frames.lock().unwrap().is_empty() {
            self.play_pre_rendered_frames(ctx);
            return;
        }

        // === 常规实时渲染（未启用预渲染模式或预渲染复选框未勾选） ===

        // 确保场景已加载
        if self.scene.is_none() {
            let obj_path = match &self.settings.obj {
                Some(path) => path.clone(),
                None => {
                    self.set_error("错误: 未指定OBJ文件路径".to_string());
                    self.stop_animation_rendering();
                    return;
                }
            };
            match ModelLoader::load_and_create_scene(&obj_path, &self.settings) {
                Ok((scene, model_data)) => {
                    self.scene = Some(scene);
                    self.model_data = Some(model_data);
                    // 注意：这里不再自动跳转到预渲染，而是继续执行实时渲染
                    self.status_message = "模型加载成功，开始实时渲染...".to_string();
                }
                Err(e) => {
                    self.set_error(format!("加载模型失败: {e}"));
                    self.stop_animation_rendering();
                    return;
                }
            }
        }

        // 检查渲染器尺寸，但避免不必要的缓存清除
        if self.renderer.frame_buffer.width != self.settings.width
            || self.renderer.frame_buffer.height != self.settings.height
        {
            self.renderer
                .resize(self.settings.width, self.settings.height);
            self.rendered_image = None;
            debug!(
                "重新创建渲染器，尺寸: {}x{}",
                self.settings.width, self.settings.height
            );
        }

        let now = Instant::now();
        let dt = if let Some(last_time) = self.last_frame_time {
            now.duration_since(last_time).as_secs_f32()
        } else {
            1.0 / 60.0 // 默认 dt
        };
        if let Some(last_time) = self.last_frame_time {
            let frame_time = now.duration_since(last_time);
            self.update_fps_stats(frame_time);
        }
        self.last_frame_time = Some(now);

        if self.is_realtime_rendering && self.settings.rotation_speed.abs() < 0.01 {
            self.settings.rotation_speed = 1.0; // 确保实时渲染时有旋转速度
        }

        self.animation_time += dt;

        if let Some(scene) = &mut self.scene {
            // 动画过程中不清除缓存
            // 物体动画不影响背景和地面（相机不动），所以缓存仍然有效

            // 使用通用函数计算旋转增量
            let rotation_delta_rad = calculate_rotation_delta(self.settings.rotation_speed, dt);
            let rotation_axis_vec = get_animation_axis_vector(&self.settings);

            // 使用通用函数执行动画步骤
            animate_scene_step(
                scene,
                &self.settings.animation_type,
                &rotation_axis_vec,
                rotation_delta_rad,
            );

            debug!(
                "实时渲染中: FPS={:.1}, 动画类型={:?}, 轴={:?}, 旋转速度={}, 角度增量={:.3}rad, Phong={}",
                self.avg_fps,
                self.settings.animation_type,
                self.settings.rotation_axis,
                self.settings.rotation_speed,
                rotation_delta_rad,
                self.settings.use_phong
            );

            match self.settings.animation_type {
                AnimationType::CameraOrbit => {
                    // 相机轨道动画：地面本体和阴影都依赖相机，必须全部失效
                    self.renderer.frame_buffer.invalidate_ground_base_cache();
                    self.renderer.frame_buffer.invalidate_ground_shadow_cache();
                }
                AnimationType::ObjectLocalRotation => {
                    if self.settings.enable_shadow_mapping {
                        // 物体动画+阴影：只需失效阴影缓存，地面本体可复用
                        self.renderer.frame_buffer.invalidate_ground_shadow_cache();
                    }
                    // 未开阴影时，地面缓存可复用，无需清理
                }
                AnimationType::None => {}
            }

            self.renderer.render_scene(scene, &self.settings);
            self.display_render_result(ctx);
            ctx.request_repaint();
        }
    }

    fn start_video_generation(&mut self, ctx: &Context) {
        if !self.ffmpeg_available {
            self.set_error("无法生成视频：未检测到ffmpeg。请安装ffmpeg后重试。".to_string());
            return;
        }
        if self.is_generating_video {
            self.status_message = "视频已在生成中，请等待完成...".to_string();
            return;
        }

        // 使用 CoreMethods 验证参数
        match self.settings.validate() {
            Ok(_) => {
                let output_dir = self.settings.output_dir.clone();
                if let Err(e) = fs::create_dir_all(&output_dir) {
                    self.set_error(format!("创建输出目录失败: {e}"));
                    return;
                }
                let frames_dir = format!(
                    "{}/temp_frames_{}",
                    output_dir,
                    chrono::Utc::now().timestamp_millis()
                );
                if let Err(e) = fs::create_dir_all(&frames_dir) {
                    self.set_error(format!("创建帧目录失败: {e}"));
                    return;
                }

                // 计算旋转参数，获取视频帧数
                let (_, _, frames_per_rotation) =
                    calculate_rotation_parameters(self.settings.rotation_speed, self.settings.fps);

                let total_frames =
                    (frames_per_rotation as f32 * self.settings.rotation_cycles) as usize;

                // 如果场景未加载，尝试加载
                if self.scene.is_none() {
                    let obj_path = match &self.settings.obj {
                        Some(path) => path.clone(),
                        None => {
                            self.set_error("错误: 未指定OBJ文件路径".to_string());
                            return;
                        }
                    };
                    match ModelLoader::load_and_create_scene(&obj_path, &self.settings) {
                        Ok((scene, model_data)) => {
                            self.scene = Some(scene);
                            self.model_data = Some(model_data);
                            self.status_message = "模型加载成功，开始生成视频...".to_string();
                        }
                        Err(e) => {
                            self.set_error(format!("加载模型失败，无法生成视频: {e}"));
                            return;
                        }
                    }
                }

                let settings_for_thread = self.settings.clone();
                let video_progress_arc = self.video_progress.clone();
                let fps = self.settings.fps;
                let scene_clone = self.scene.as_ref().expect("场景已检查").clone();

                // 检查是否有预渲染帧
                let has_pre_rendered_frames = {
                    let frames_guard = self.pre_rendered_frames.lock().unwrap();
                    !frames_guard.is_empty()
                };

                // 如果没有预渲染帧，那么我们需要同时为预渲染缓冲区生成帧
                let frames_for_pre_render = if !has_pre_rendered_frames {
                    Some(self.pre_rendered_frames.clone())
                } else {
                    None
                };

                // 设置渲染状态
                self.is_generating_video = true;
                video_progress_arc.store(0, Ordering::SeqCst);

                // 更新状态消息
                self.status_message = format!(
                    "开始生成视频 (0/{} 帧，{:.1} 秒时长)...",
                    total_frames,
                    total_frames as f32 / fps as f32
                );

                ctx.request_repaint();
                let ctx_clone = ctx.clone();
                let video_filename = format!("{}.mp4", settings_for_thread.output);
                let video_output_path = format!("{output_dir}/{video_filename}");
                let frames_dir_clone = frames_dir.clone();

                // 如果有预渲染帧，复制到线程中
                let pre_rendered_frames_clone = if has_pre_rendered_frames {
                    let frames_guard = self.pre_rendered_frames.lock().unwrap();
                    Some(frames_guard.clone())
                } else {
                    None
                };

                let thread_handle = thread::spawn(move || {
                    let width = settings_for_thread.width;
                    let height = settings_for_thread.height;
                    let mut rendered_frames = Vec::new();

                    // 使用预渲染帧或重新渲染
                    if let Some(frames) = pre_rendered_frames_clone {
                        // 使用预渲染帧
                        let pre_rendered_count = frames.len();

                        for frame_num in 0..total_frames {
                            video_progress_arc.store(frame_num, Ordering::SeqCst);

                            // 计算当前帧在哪个圈和圈内的位置
                            let cycle_position = frame_num % frames_per_rotation;

                            // 将圈内位置映射到预渲染帧索引
                            // 这处理了预渲染帧数量可能与理论帧数不匹配的情况
                            let pre_render_idx =
                                (cycle_position * pre_rendered_count) / frames_per_rotation;

                            let frame = &frames[pre_render_idx.min(pre_rendered_count - 1)]; // 避免越界访问

                            // 将ColorImage转换为PNG并保存
                            let frame_path = format!("{frames_dir_clone}/frame_{frame_num:04}.png");
                            let color_data = frame_to_png_data(frame);
                            save_image(&frame_path, &color_data, width as u32, height as u32);

                            if frame_num % (total_frames.max(1) / 20).max(1) == 0 {
                                ctx_clone.request_repaint();
                            }
                        }
                    } else {
                        // 使用通用渲染函数渲染一圈或部分圈
                        let frames_arc = frames_for_pre_render.clone();

                        let rendered_frame_count = render_one_rotation_cycle(
                            scene_clone,
                            &settings_for_thread,
                            &video_progress_arc,
                            &ctx_clone,
                            width,
                            height,
                            |frame_num, color_data_rgb| {
                                // 保存RGB数据用于后续复用
                                rendered_frames.push(color_data_rgb.clone());

                                // 同时为视频保存PNG文件
                                let frame_path =
                                    format!("{frames_dir_clone}/frame_{frame_num:04}.png");
                                save_image(
                                    &frame_path,
                                    &color_data_rgb,
                                    width as u32,
                                    height as u32,
                                );

                                // 如果需要同时保存到预渲染缓冲区
                                if let Some(ref frames_arc) = frames_arc {
                                    // 转换为RGBA格式以用于预渲染帧
                                    let mut rgba_data = Vec::with_capacity(width * height * 4);
                                    for chunk in color_data_rgb.chunks_exact(3) {
                                        rgba_data.extend_from_slice(chunk);
                                        rgba_data.push(255); // Alpha
                                    }
                                    let color_image = ColorImage::from_rgba_unmultiplied(
                                        [width, height],
                                        &rgba_data,
                                    );
                                    frames_arc.lock().unwrap().push(color_image);
                                }
                            },
                        );

                        // 如果需要多于一圈，使用前面渲染的帧复用
                        if rendered_frame_count < total_frames {
                            for frame_num in rendered_frame_count..total_frames {
                                video_progress_arc.store(frame_num, Ordering::SeqCst);

                                // 复用之前渲染的帧
                                let source_frame_idx = frame_num % rendered_frame_count;
                                let source_data = &rendered_frames[source_frame_idx];

                                // 保存为图片文件
                                let frame_path =
                                    format!("{frames_dir_clone}/frame_{frame_num:04}.png");
                                save_image(&frame_path, source_data, width as u32, height as u32);

                                if frame_num % (total_frames.max(1) / 20).max(1) == 0 {
                                    ctx_clone.request_repaint();
                                }
                            }
                        }
                    }

                    video_progress_arc.store(total_frames, Ordering::SeqCst);
                    ctx_clone.request_repaint();

                    // 使用ffmpeg将帧序列合成为视频，并解决阻塞问题
                    let frames_pattern = format!("{frames_dir_clone}/frame_%04d.png");
                    let ffmpeg_status = std::process::Command::new("ffmpeg")
                        .args([
                            "-y",
                            "-framerate",
                            &fps.to_string(),
                            "-i",
                            &frames_pattern,
                            "-c:v",
                            "libx264",
                            "-pix_fmt",
                            "yuv420p",
                            "-crf",
                            "23",
                            &video_output_path,
                        ])
                        .status();

