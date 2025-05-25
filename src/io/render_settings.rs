use crate::material_system::light::{Light, LightingPreset};
use clap::{Parser, ValueEnum};
use nalgebra::{Point3, Vector3};

/// 动画类型枚举
#[derive(ValueEnum, Debug, Clone, Default, PartialEq, Eq)]
pub enum AnimationType {
    #[default]
    CameraOrbit,
    ObjectLocalRotation,
    None,
}

/// 旋转轴枚举
#[derive(ValueEnum, Debug, Clone, PartialEq, Eq, Default)]
pub enum RotationAxis {
    X,
    #[default]
    Y,
    Z,
    Custom,
}

/// 统一的渲染设置结构体
///
/// 同时处理命令行参数解析和渲染配置功能
#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
pub struct RenderSettings {
    // ===== 基础设置 =====
    /// 输入OBJ文件的路径
    #[arg(long)]
    pub obj: Option<String>,

    /// 运行完整动画循环而非单帧渲染
    #[arg(long, default_value_t = false)]
    pub animate: bool,

    /// 动画帧率 (fps)，用于视频生成和预渲染
    #[arg(long, default_value_t = 30)]
    pub fps: usize,

    /// 旋转速度系数，控制动画旋转的速度
    #[arg(long, default_value_t = 1.0)]
    pub rotation_speed: f32,

    /// 完整旋转圈数，用于视频生成(默认1圈)
    #[arg(long, default_value_t = 1.0)]
    pub rotation_cycles: f32,

    /// 动画类型 (用于 animate 模式或实时渲染)
    #[arg(long, value_enum, default_value_t = AnimationType::CameraOrbit)]
    pub animation_type: AnimationType,

    /// 动画旋转轴 (用于 CameraOrbit 和 ObjectLocalRotation)
    #[arg(long, value_enum, default_value_t = RotationAxis::Y)]
    pub rotation_axis: RotationAxis,

    /// 自定义旋转轴 (当 rotation_axis 为 Custom 时使用)，格式 "x,y,z"
    #[arg(long, default_value = "0,1,0")]
    pub custom_rotation_axis: String,

    // ===== 输出设置 =====
    /// 输出文件的基础名称
    #[arg(short, long, default_value = "output")]
    pub output: String,

    /// 输出图像的目录
    #[arg(long, default_value = "output_rust")]
    pub output_dir: String,

    /// 输出图像的宽度
    #[arg(long, default_value_t = 1024)]
    pub width: usize,

    /// 输出图像的高度
    #[arg(long, default_value_t = 1024)]
    pub height: usize,

    /// 启用渲染和保存深度图
    #[arg(long, default_value_t = true)]
    pub save_depth: bool,

    // ===== 渲染基础设置 =====
    /// 投影类型："perspective"或"orthographic"
    #[arg(long, default_value = "perspective")]
    pub projection: String,

    /// 启用Z缓冲（深度测试）
    #[arg(long, default_value_t = true)]
    pub use_zbuffer: bool,

    /// 使用伪随机面颜色而非材质颜色
    #[arg(long, default_value_t = false)]
    pub colorize: bool,

    /// 启用纹理加载和使用
    #[arg(long, default_value_t = true)]
    pub use_texture: bool,

    /// 显式指定要使用的纹理文件，覆盖MTL设置
    #[arg(long)]
    pub texture: Option<String>,

    /// 启用gamma矫正
    #[arg(long, default_value_t = true)]
    pub use_gamma: bool,

    /// 启用背面剔除
    #[arg(long, default_value_t = false)]
    pub backface_culling: bool,

    /// 以线框模式渲染
    #[arg(long, default_value_t = false)]
    pub wireframe: bool,

    /// 启用多线程渲染
    #[arg(long, default_value_t = true)]
    pub use_multithreading: bool,

    /// 启用小三角形剔除
    #[arg(long, default_value_t = false)]
    pub cull_small_triangles: bool,

    /// 小三角形剔除的最小面积阈值
    #[arg(long, default_value_t = 1e-3)]
    pub min_triangle_area: f32,

    /// 物体的全局均匀缩放因子
    #[arg(long, default_value_t = 1.0)]
    pub object_scale: f32,

    // ===== 物体变换控制（字符串格式，用于CLI和序列化） =====
    /// 物体位置 (x,y,z)
    #[arg(long, default_value = "0,0,0")]
    pub object_position: String,

    /// 物体旋转 (欧拉角，度)
    #[arg(long, default_value = "0,0,0")]
    pub object_rotation: String,

    /// 物体缩放 (x,y,z)
    #[arg(long, default_value = "1,1,1")]
    pub object_scale_xyz: String,

    // ===== 相机参数 =====
    /// 相机位置（视点），格式为"x,y,z"
    #[arg(long, default_value = "0,0,3", allow_negative_numbers = true)]
    pub camera_from: String,

    /// 相机目标（观察点），格式为"x,y,z"
    #[arg(long, default_value = "0,0,0", allow_negative_numbers = true)]
    pub camera_at: String,

    /// 相机世界坐标系上方向，格式为"x,y,z"
    #[arg(long, default_value = "0,1,0", allow_negative_numbers = true)]
    pub camera_up: String,

    /// 相机垂直视场角（度，用于透视投影）
    #[arg(long, default_value_t = 45.0)]
    pub camera_fov: f32,

    // ===== 光照基础参数 =====
    /// 启用光照计算
    #[arg(long, default_value_t = true)]
    pub use_lighting: bool,

    /// 环境光强度因子
    #[arg(long, default_value_t = 0.3)]
    pub ambient: f32,

    /// 环境光强度RGB值，格式为"r,g,b"
    #[arg(long, default_value = "0.3,0.4,0.5")]
    pub ambient_color: String,

    /// 光照预设模式
    #[arg(long, value_enum, default_value_t = LightingPreset::SingleDirectional)]
    pub lighting_preset: LightingPreset,

    /// 主光源强度 (0.0-1.0)
    #[arg(long, default_value_t = 0.8)]
    pub main_light_intensity: f32,

    // ===== 着色模型选择 =====
    /// 使用Phong着色（逐像素光照）
    #[arg(long, default_value_t = true)]
    pub use_phong: bool,

    /// 使用基于物理的渲染(PBR)
    #[arg(long, default_value_t = false)]
    pub use_pbr: bool,

    // ===== Phong着色模型参数 =====
    /// 漫反射颜色，格式为"r,g,b"
    #[arg(long, default_value = "0.8,0.8,0.8")]
    pub diffuse_color: String,

    /// 镜面反射强度(0.0-1.0)
    #[arg(long, default_value_t = 0.5)]
    pub specular: f32,

    /// 材质的光泽度(硬度)参数
    #[arg(long, default_value_t = 32.0)]
    pub shininess: f32,

    // ===== PBR材质参数 =====
    /// 材质的基础颜色，格式为"r,g,b"
    #[arg(long, default_value = "0.8,0.8,0.8")]
    pub base_color: String,

    /// 材质的金属度(0.0-1.0)
    #[arg(long, default_value_t = 0.0)]
    pub metallic: f32,

    /// 材质的粗糙度(0.0-1.0)
    #[arg(long, default_value_t = 0.5)]
    pub roughness: f32,

    /// 环境光遮蔽系数(0.0-1.0)
    #[arg(long, default_value_t = 1.0)]
    pub ambient_occlusion: f32,

    /// 材质的自发光颜色，格式为"r,g,b"
    #[arg(long, default_value = "0.0,0.0,0.0")]
    pub emissive: String,

    // ==== 阴影设置 ====
    /// 启用增强环境光遮蔽
    #[arg(long, default_value_t = true)]
    pub enhanced_ao: bool,

    /// 环境光遮蔽强度 (0.0-1.0)
    #[arg(long, default_value_t = 0.5)]
    pub ao_strength: f32,

    /// 启用软阴影
    #[arg(long, default_value_t = true)]
    pub soft_shadows: bool,

    /// 软阴影强度 (0.0-1.0)
    #[arg(long, default_value_t = 0.7)]
    pub shadow_strength: f32,

    // ===== 背景与环境设置 =====
    /// 启用渐变背景
    #[arg(long, default_value_t = false)]
    pub enable_gradient_background: bool,

    /// 渐变背景顶部颜色，格式为"r,g,b"
    #[arg(long, default_value = "0.5,0.7,1.0")]
    pub gradient_top_color: String,

    /// 渐变背景底部颜色，格式为"r,g,b"
    #[arg(long, default_value = "0.1,0.2,0.4")]
    pub gradient_bottom_color: String,

    /// 启用地面平面
    #[arg(long, default_value_t = false)]
    pub enable_ground_plane: bool,

    /// 地面平面颜色，格式为"r,g,b"
    #[arg(long, default_value = "0.3,0.5,0.2")]
    pub ground_plane_color: String,

    /// 地面平面在Y轴上的高度
    #[arg(long, default_value_t = -1.0, allow_negative_numbers = true)]
    pub ground_plane_height: f32,

    /// 使用背景图片
    #[arg(long, default_value_t = false)]
    pub use_background_image: bool,

    /// 背景图片路径
    #[arg(long)]
    pub background_image_path: Option<String>,

    // ===== 🔥 **运行时字段（不是命令行参数）** =====
    #[arg(skip)]
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
                eprintln!(
                    "警告: 无效的自定义旋转轴 '{}', 使用默认Y轴。",
                    settings.custom_rotation_axis
                );
                Vector3::y_axis().into_inner()
            })
            .normalize(),
    }
}

impl Default for RenderSettings {
    fn default() -> Self {
        // 🔥 **智能选择：检查是否有命令行参数**
        let args: Vec<String> = std::env::args().collect();

        let mut settings = if args.len() > 1
            && args
                .iter()
                .any(|arg| arg.starts_with("--") || arg.ends_with(".obj"))
        {
            // 有有效命令行参数，解析它们
            Self::parse()
        } else {
            // 无有效命令行参数，使用clap默认值
            Self::parse_from(std::iter::empty::<String>())
        };

        // 🔥 **关键修复：无论哪种情况都确保有光源**
        if settings.use_lighting {
            settings.lights = crate::material_system::light::LightManager::create_preset_lights(
                &settings.lighting_preset,
                settings.use_lighting,
                settings.main_light_intensity,
            );
        } else {
            settings.lights = Vec::new();
        }

        settings
    }
}

impl RenderSettings {
    // ===== 🔥 **新增：按需计算方法（替代重复存储）** =====

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

    // ===== 🔥 **删除了 update_color_vectors 方法** =====
    // 不再需要同步方法！

    // ===== **保留原有的方法** =====

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

    /// 检查是否应该启动GUI模式
    pub fn should_start_gui(&self) -> bool {
        if self.obj.is_none() {
            return true;
        }

        if std::env::args().count() <= 1 {
            return true;
        }

        false
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
                return Err(format!("错误: 找不到OBJ文件 '{}'", obj_path));
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
