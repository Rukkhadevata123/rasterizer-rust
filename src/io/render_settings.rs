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

/// 🔥 **纯数据结构** - 所有可通过TOML配置的渲染参数
/// 移除了clap逻辑和预设系统，专注于数据存储
#[derive(Debug, Clone)]
pub struct RenderSettings {
    // ===== 🔥 **文件路径设置** =====
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

    // ===== 🔥 **渲染基础设置** =====
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
    /// 启用背面剔除
    pub backface_culling: bool,
    /// 以线框模式渲染
    pub wireframe: bool,
    /// 启用多线程渲染
    pub use_multithreading: bool,
    /// 启用小三角形剔除
    pub cull_small_triangles: bool,
    /// 小三角形剔除的最小面积阈值
    pub min_triangle_area: f32,
    /// 启用渲染和保存深度图
    pub save_depth: bool,

    // ===== 🔥 **物体变换控制（字符串格式，用于TOML序列化）** =====
    /// 物体位置 (x,y,z)
    pub object_position: String,
    /// 物体旋转 (欧拉角，度)
    pub object_rotation: String,
    /// 物体缩放 (x,y,z)
    pub object_scale_xyz: String,
    /// 物体的全局均匀缩放因子
    pub object_scale: f32,

    // ===== 🔥 **相机参数** =====
    /// 相机位置（视点），格式为"x,y,z"
    pub camera_from: String,
    /// 相机目标（观察点），格式为"x,y,z"
    pub camera_at: String,
    /// 相机世界坐标系上方向，格式为"x,y,z"
    pub camera_up: String,
    /// 相机垂直视场角（度，用于透视投影）
    pub camera_fov: f32,

    // ===== 🔥 **光照基础参数** =====
    /// 启用光照计算
    pub use_lighting: bool,
    /// 环境光强度因子
    pub ambient: f32,
    /// 环境光强度RGB值，格式为"r,g,b"
    pub ambient_color: String,

    // ===== 🔥 **着色模型选择** =====
    /// 使用Phong着色（逐像素光照）
    pub use_phong: bool,
    /// 使用基于物理的渲染(PBR)
    pub use_pbr: bool,

    // ===== 🔥 **Phong着色模型参数** =====
    /// 漫反射颜色，格式为"r,g,b"
    pub diffuse_color: String,
    /// 镜面反射强度(0.0-1.0)
    pub specular: f32,
    /// 材质的光泽度(硬度)参数
    pub shininess: f32,

    // ===== 🔥 **PBR材质参数** =====
    /// 材质的基础颜色，格式为"r,g,b"
    pub base_color: String,
    /// 材质的金属度(0.0-1.0)
    pub metallic: f32,
    /// 材质的粗糙度(0.0-1.0)
    pub roughness: f32,
    /// 环境光遮蔽系数(0.0-1.0)
    pub ambient_occlusion: f32,
    /// 材质的自发光颜色，格式为"r,g,b"
    pub emissive: String,

    // ===== 🔥 **阴影设置** =====
    /// 启用增强环境光遮蔽
    pub enhanced_ao: bool,
    /// 环境光遮蔽强度 (0.0-1.0)
    pub ao_strength: f32,
    /// 启用软阴影
    pub soft_shadows: bool,
    /// 软阴影强度 (0.0-1.0)
    pub shadow_strength: f32,

    // ===== 🔥 **背景与环境设置** =====
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

    // ===== 🔥 **动画设置** =====
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

    // ===== 🔥 **光源数组（运行时字段）** =====
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
            backface_culling: false,
            wireframe: false,
            use_multithreading: true,
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
            specular: 0.5,
            shininess: 32.0,

            // ===== PBR材质参数 =====
            base_color: "0.8,0.8,0.8".to_string(),
            metallic: 0.0,
            roughness: 0.5,
            ambient_occlusion: 1.0,
            emissive: "0.0,0.0,0.0".to_string(),

            // ===== 阴影设置 =====
            enhanced_ao: true,
            ao_strength: 0.5,
            soft_shadows: true,
            shadow_strength: 0.7,

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

        // 🔥 **如果启用了光照且没有光源，创建默认方向光**
        settings.initialize_lights();

        settings
    }
}

impl RenderSettings {
    /// 🔥 **初始化默认光源** - 确保启用光照时有可用光源
    pub fn initialize_lights(&mut self) {
        if self.use_lighting && self.lights.is_empty() {
            self.lights = vec![Light::directional(
                Vector3::new(0.0, -1.0, -1.0),
                Vector3::new(1.0, 1.0, 1.0),
                0.8,
            )];
        }
    }

    // ===== 🔥 **按需计算方法（替代重复存储）** =====

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
