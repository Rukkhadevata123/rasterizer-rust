  
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



pub mod model_utils;
pub mod render_utils;
pub mod save_utils;
use crate::material_system::materials::Model;
use nalgebra::{Point3, Vector3};

/// 归一化和中心化模型顶点
pub fn normalize_and_center_model(model_data: &mut Model) -> (Vector3<f32>, f32) {
    if model_data.meshes.is_empty() {
        return (Vector3::zeros(), 1.0);
    }

    // 计算所有顶点的边界框或质心
    let mut min_coord = Point3::new(f32::MAX, f32::MAX, f32::MAX);
    let mut max_coord = Point3::new(f32::MIN, f32::MIN, f32::MIN);
    let mut vertex_sum = Vector3::zeros();
    let mut vertex_count = 0;

    for mesh in &model_data.meshes {
        for vertex in &mesh.vertices {
            min_coord = min_coord.inf(&vertex.position);
            max_coord = max_coord.sup(&vertex.position);
            vertex_sum += vertex.position.coords;
            vertex_count += 1;
        }
    }

    if vertex_count == 0 {
        return (Vector3::zeros(), 1.0);
    }

    let center = vertex_sum / (vertex_count as f32);
    let extent = max_coord - min_coord;
    let max_extent = extent.x.max(extent.y).max(extent.z);

    let scale_factor = if max_extent > 1e-6 {
        1.6 / max_extent // 缩放以大致适合[-0.8, 0.8]立方体（类似于Python的0.8因子）
    } else {
        1.0
    };

    // 对所有顶点应用变换
    for mesh in &mut model_data.meshes {
        for vertex in &mut mesh.vertices {
            vertex.position = Point3::from((vertex.position.coords - center) * scale_factor);
        }
    }

    (center, scale_factor)
}
use crate::core::renderer::Renderer;
use crate::io::render_settings::{
    AnimationType, RenderSettings, RotationAxis, get_animation_axis_vector,
};
use crate::scene::scene_utils::Scene;
use crate::utils::save_utils::save_render_with_settings;
use log::{debug, info};
use nalgebra::Vector3;
use std::time::Instant;

const BASE_SPEED: f32 = 60.0; // 1s旋转60度

/// 渲染单帧并保存结果
pub fn render_single_frame(
    scene: &mut Scene,
    renderer: &mut Renderer,
    settings: &RenderSettings,
    output_name: &str,
) -> Result<(), String> {
    let frame_start_time = Instant::now();
    debug!("渲染帧: {output_name}");

    // 直接渲染场景，无需额外同步
    renderer.render_scene(scene, settings);

    // 保存输出图像
    debug!("保存 {output_name} 的输出图像...");
    save_render_with_settings(renderer, settings, Some(output_name))?;

    debug!(
        "帧 {} 渲染完成，耗时 {:?}",
        output_name,
        frame_start_time.elapsed()
    );
    Ok(())
}

/// 执行单个步骤的场景动画
pub fn animate_scene_step(
    scene: &mut Scene,
    animation_type: &AnimationType,
    rotation_axis: &Vector3<f32>,
    rotation_delta_rad: f32,
) {
    match animation_type {
        AnimationType::CameraOrbit => {
            let mut camera = scene.active_camera.clone();
            camera.orbit(rotation_axis, rotation_delta_rad);
            scene.set_camera(camera);
        }
        AnimationType::ObjectLocalRotation => {
            scene.object.rotate(rotation_axis, rotation_delta_rad);
        }
        AnimationType::None => { /* 无动画 */ }
    }
}

/// 计算旋转增量的辅助函数
pub fn calculate_rotation_delta(rotation_speed: f32, dt: f32) -> f32 {
    (rotation_speed * dt * BASE_SPEED).to_radians()
}

/// 计算有效旋转速度及旋转周期
pub fn calculate_rotation_parameters(rotation_speed: f32, fps: usize) -> (f32, f32, usize) {
    // 计算有效旋转速度 (度/秒)
    let mut effective_rotation_speed_dps = rotation_speed * BASE_SPEED;

    // 确保旋转速度不会太小
    if effective_rotation_speed_dps.abs() < 0.001 {
        effective_rotation_speed_dps = 0.1_f32.copysign(rotation_speed.signum());
        if effective_rotation_speed_dps == 0.0 {
            effective_rotation_speed_dps = 0.1;
        }
    }

    // 计算完成一圈需要的秒数
    let seconds_per_rotation = 360.0 / effective_rotation_speed_dps.abs();

    // 计算一圈需要的帧数
    let frames_for_one_rotation = (seconds_per_rotation * fps as f32).ceil() as usize;

    (
        effective_rotation_speed_dps,
        seconds_per_rotation,
        frames_for_one_rotation,
    )
}

/// 执行完整的动画渲染循环
pub fn run_animation_loop(
    scene: &mut Scene,
    renderer: &mut Renderer,
    settings: &RenderSettings,
) -> Result<(), String> {
    // 使用通用函数计算旋转参数
    let (effective_rotation_speed_dps, _, frames_to_render) =
        calculate_rotation_parameters(settings.rotation_speed, settings.fps);

    // 根据用户要求的旋转圈数计算实际帧数
    let total_frames = (frames_to_render as f32 * settings.rotation_cycles) as usize;

    info!(
        "开始动画渲染 ({} 帧, {:.2} 秒)...",
        total_frames,
        total_frames as f32 / settings.fps as f32
    );
    info!(
        "动画类型: {:?}, 旋转轴类型: {:?}, 速度: {:.1}度/秒",
        settings.animation_type, settings.rotation_axis, effective_rotation_speed_dps
    );

    // 计算旋转方向
    let rotation_axis_vec = get_animation_axis_vector(settings);
    if settings.rotation_axis == RotationAxis::Custom {
        debug!("自定义旋转轴: {rotation_axis_vec:?}");
    }

    // 计算每帧的旋转角度
    let rotation_per_frame_rad =
        (360.0 / frames_to_render as f32).to_radians() * settings.rotation_speed.signum();

    // 渲染所有帧
    for frame_num in 0..total_frames {
        let frame_start_time = Instant::now();
        debug!("--- 准备帧 {} / {} ---", frame_num + 1, total_frames);

        // 第一帧通常不旋转，保留原始状态
        if frame_num > 0 {
            animate_scene_step(
                scene,
                &settings.animation_type,
                &rotation_axis_vec,
                rotation_per_frame_rad,
            );
        }

        // 渲染和保存当前帧
        let frame_output_name = format!("frame_{frame_num:03}");
        render_single_frame(scene, renderer, settings, &frame_output_name)?;

        debug!(
            "帧 {} 渲染完成，耗时 {:?}",
            frame_output_name,
            frame_start_time.elapsed()
        );
    }

    info!(
        "动画渲染完成。总时长：{:.2}秒",
        total_frames as f32 / settings.fps as f32
    );
    Ok(())
}
use crate::core::renderer::Renderer;
use crate::io::render_settings::RenderSettings;
use crate::material_system::color::apply_colormap_jet;
use image::ColorType;
use log::{debug, info, warn};
use std::path::Path;

/// 保存RGB图像数据到PNG文件
pub fn save_image(path: &str, data: &[u8], width: u32, height: u32) {
    match image::save_buffer(path, data, width, height, ColorType::Rgb8) {
        Ok(_) => info!("图像已保存到 {path}"),
        Err(e) => warn!("保存图像到 {path} 时出错: {e}"),
    }
}

/// 将深度缓冲数据归一化到指定的百分位数范围
pub fn normalize_depth(depth_buffer: &[f32], min_percentile: f32, max_percentile: f32) -> Vec<f32> {
    // 1. 收集所有有限的深度值
    let mut finite_depths: Vec<f32> = depth_buffer
        .iter()
        .filter(|&&d| d.is_finite())
        .cloned()
        .collect();

    let mut min_clip: f32;
    let mut max_clip: f32;

    // 2. 使用百分位数确定归一化范围
    if finite_depths.len() >= 2 {
        finite_depths.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        let min_idx = ((min_percentile / 100.0 * (finite_depths.len() - 1) as f32).round()
            as usize)
            .clamp(0, finite_depths.len() - 1);
        let max_idx = ((max_percentile / 100.0 * (finite_depths.len() - 1) as f32).round()
            as usize)
            .clamp(0, finite_depths.len() - 1);

        min_clip = finite_depths[min_idx];
        max_clip = finite_depths[max_idx];

        if (max_clip - min_clip).abs() < 1e-6 {
            min_clip = *finite_depths.first().unwrap();
            max_clip = *finite_depths.last().unwrap();
            if (max_clip - min_clip).abs() < 1e-6 {
                max_clip = min_clip + 1.0;
            }
        }
        debug!(
            "使用百分位数归一化深度: [{min_percentile:.1}%, {max_percentile:.1}%] -> [{min_clip:.3}, {max_clip:.3}]"
        );
    } else {
        warn!("没有足够的有限深度值进行百分位裁剪。使用默认范围 [0.1, 10.0]");
        min_clip = 0.1;
        max_clip = 10.0;
    }

    let range = max_clip - min_clip;
    let inv_range = if range > 1e-6 { 1.0 / range } else { 0.0 };

    depth_buffer
        .iter()
        .map(|&depth| {
            if depth.is_finite() {
                ((depth.clamp(min_clip, max_clip) - min_clip) * inv_range).clamp(0.0, 1.0)
            } else {
                1.0
            }
        })
        .collect()
}

/// 保存渲染结果（彩色图像和可选的深度图）
#[allow(clippy::too_many_arguments)]
pub fn save_render_result(
    color_data: &[u8],
    depth_data: Option<&[f32]>,
    width: usize,
    height: usize,
    output_dir: &str,
    output_name: &str,
    settings: &RenderSettings,
    save_depth: bool,
) -> Result<(), String> {
    // 保存彩色图像
    let color_path = Path::new(output_dir)
        .join(format!("{output_name}_color.png"))
        .to_str()
        .ok_or_else(|| "创建彩色输出路径字符串失败".to_string())?
        .to_string();

    save_image(&color_path, color_data, width as u32, height as u32);

    // 保存深度图（如果启用）
    if settings.use_zbuffer && save_depth {
        if let Some(depth_data_raw) = depth_data {
            let depth_normalized = normalize_depth(depth_data_raw, 1.0, 99.0);
            let depth_colored = apply_colormap_jet(
                &depth_normalized
                    .iter()
                    .map(|&d| 1.0 - d) // 反转：越近 = 越热
                    .collect::<Vec<_>>(),
                width,
                height,
                settings.use_gamma,
            );

            let depth_path = Path::new(output_dir)
                .join(format!("{output_name}_depth.png"))
                .to_str()
                .ok_or_else(|| "创建深度输出路径字符串失败".to_string())?
                .to_string();

            save_image(&depth_path, &depth_colored, width as u32, height as u32);
        }
    }

    Ok(())
}

/// 从渲染器中获取数据并保存渲染结果
pub fn save_render_with_settings(
    renderer: &Renderer,
    settings: &RenderSettings,
    output_name: Option<&str>,
) -> Result<(), String> {
    let color_data = renderer.frame_buffer.get_color_buffer_bytes();
    let depth_data = if settings.save_depth {
        Some(renderer.frame_buffer.get_depth_buffer_f32())
    } else {
        None
    };

    let width = renderer.frame_buffer.width;
    let height = renderer.frame_buffer.height;
    let output_name = output_name.unwrap_or(&settings.output);

    save_render_result(
        &color_data,
        depth_data.as_deref(),
        width,
        height,
        &settings.output_dir,
        output_name,
        settings,
        settings.save_depth,
    )
}
use log::{error, info, warn};
use std::fs;
use std::time::Instant;

mod core;
mod geometry;
mod io;
mod material_system;
mod scene;
mod ui;
mod utils;

use crate::ui::app::start_gui;
use core::renderer::Renderer;
use io::model_loader::ModelLoader;
use io::simple_cli::SimpleCli;
use utils::render_utils::{render_single_frame, run_animation_loop};

fn main() -> Result<(), String> {
    // 初始化日志系统
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Debug)
        .filter_module("eframe", log::LevelFilter::Warn) // 只显示 eframe 的警告和错误
        .filter_module("egui_glow", log::LevelFilter::Warn) // 只显示 egui_glow 的警告和错误
        .filter_module("egui_winit", log::LevelFilter::Warn) // 只显示 egui_winit 的警告和错误
        .filter_module("winit", log::LevelFilter::Warn) // 只显示 winit 的警告和错误
        .filter_module("wgpu", log::LevelFilter::Warn) // 只显示 wgpu 的警告和错误
        .filter_module("glutin", log::LevelFilter::Warn) // 只显示 glutin 的警告和错误
        .filter_module("sctk", log::LevelFilter::Warn) // 只显示 sctk 的警告和错误
        .format_timestamp(None)
        .format_level(true)
        .init();

    info!("🎨 光栅化渲染器启动");

    let (settings, should_start_gui) = SimpleCli::process()?;

    // 判断是否应该启动GUI模式
    if should_start_gui {
        info!("启动GUI模式...");
        if let Err(err) = start_gui(settings) {
            error!("GUI启动失败: {err}");
            return Err("GUI启动失败".to_string());
        }
        return Ok(());
    }

    // 无头渲染模式 - 需要OBJ文件
    if settings.obj.is_none() {
        error!("无头模式需要指定OBJ文件路径");
        return Err("缺少OBJ文件路径".to_string());
    }

    let start_time = Instant::now();
    let obj_path = settings.obj.as_ref().unwrap();

    // 确保输出目录存在
    fs::create_dir_all(&settings.output_dir).map_err(|e| {
        error!("创建输出目录 '{}' 失败：{}", settings.output_dir, e);
        "创建输出目录失败".to_string()
    })?;

    // 验证资源
    info!("验证资源...");
    if let Err(e) = ModelLoader::validate_resources(&settings) {
        warn!("{e}");
    }

    // 加载模型和创建场景
    let (mut scene, _model_data) = ModelLoader::load_and_create_scene(obj_path, &settings)
        .map_err(|e| {
            error!("模型加载失败: {e}");
            "模型加载失败".to_string()
        })?;

    // 创建渲染器
    let mut renderer = Renderer::new(settings.width, settings.height);

    // 渲染动画或单帧
    if settings.animate {
        run_animation_loop(&mut scene, &mut renderer, &settings).map_err(|e| {
            error!("动画渲染失败: {e}");
            "动画渲染失败".to_string()
        })?;
    } else {
        info!("--- 开始单帧渲染 ---");
        info!("分辨率: {}x{}", settings.width, settings.height);
        info!("投影类型: {}", settings.projection);
        info!(
            "光照: {} ({} 个光源)",
            if settings.use_lighting {
                "启用"
            } else {
                "禁用"
            },
            settings.lights.len()
        );
        info!("材质: {}", settings.get_lighting_description());

        if settings.use_background_image {
            if let Some(bg_path) = &settings.background_image_path {
                info!("背景图片: {bg_path}");
            }
        }
        if settings.enable_gradient_background {
            info!("渐变背景: 启用");
        }
        if settings.enable_ground_plane {
            info!("地面平面: 启用");
        }

        render_single_frame(&mut scene, &mut renderer, &settings, &settings.output).map_err(
            |e| {
                error!("单帧渲染失败: {e}");
                "单帧渲染失败".to_string()
            },
        )?;
    }

    info!("总执行时间：{:?}", start_time.elapsed());
    Ok(())
}