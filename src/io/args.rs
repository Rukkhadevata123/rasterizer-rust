use clap::Parser;
use nalgebra::Vector3;
// 导入LightingPreset和光源配置
use crate::material_system::light::{DirectionalLightConfig, LightingPreset, PointLightConfig};

#[derive(clap::ValueEnum, Debug, Clone, Default, PartialEq, Eq)]
pub enum AnimationType {
    #[default]
    CameraOrbit,
    ObjectLocalRotation,
    None,
}

#[derive(clap::ValueEnum, Debug, Clone, PartialEq, Eq, Default)]
pub enum RotationAxis {
    X,
    #[default]
    Y,
    Z,
    Custom,
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
pub struct Args {
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
    /// 输出文件的基础名称（例如: "render" -> "render_color.png", "render_depth.png"）
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
    /// 投影类型："perspective"（透视投影）或"orthographic"（正交投影）
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

    /// 场景中要创建的对象实例数量
    #[arg(long)]
    pub object_count: Option<String>,

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
    #[arg(long, default_value_t = 0.1)]
    pub ambient: f32,

    /// 环境光强度RGB值，格式为"r,g,b"
    #[arg(long, default_value = "0.1,0.1,0.1")]
    pub ambient_color: String,

    /// 光照预设模式
    #[arg(long, value_enum, default_value_t = LightingPreset::SingleDirectional)]
    pub lighting_preset: LightingPreset,

    /// 主光源强度 (0.0-1.0)
    #[arg(long, default_value_t = 0.8)]
    pub main_light_intensity: f32,

    // 这些字段不是直接CLI参数，只在内部使用
    #[arg(skip)]
    pub directional_lights: Vec<DirectionalLightConfig>,
    #[arg(skip)]
    pub point_lights: Vec<PointLightConfig>,

    // ===== 着色模型选择 =====
    /// 使用Phong着色（逐像素光照）而非默认的Flat着色
    #[arg(long, default_value_t = false)]
    pub use_phong: bool,

    /// 使用基于物理的渲染(PBR)而不是传统Blinn-Phong
    #[arg(long, default_value_t = false)]
    pub use_pbr: bool,

    // ===== Phong着色模型参数 =====
    /// 漫反射颜色，格式为"r,g,b"，每个分量在0.0-1.0范围内(仅在Phong模式下有效)
    #[arg(long, default_value = "0.8,0.8,0.8")]
    pub diffuse_color: String,

    /// 镜面反射强度(0.0-1.0，仅在Phong模式下有效)
    #[arg(long, default_value_t = 0.5)]
    pub specular: f32,

    /// 材质的光泽度(硬度)参数(仅在Phong模式下有效)
    #[arg(long, default_value_t = 32.0)]
    pub shininess: f32,

    // ===== PBR材质参数 =====
    /// 材质的基础颜色，格式为"r,g,b"，每个分量在0.0-1.0范围内(仅在PBR模式下有效)
    #[arg(long, default_value = "0.8,0.8,0.8")]
    pub base_color: String,

    /// 材质的金属度(0.0-1.0，仅在PBR模式下有效)
    #[arg(long, default_value_t = 0.0)]
    pub metallic: f32,

    /// 材质的粗糙度(0.0-1.0，仅在PBR模式下有效)
    #[arg(long, default_value_t = 0.5)]
    pub roughness: f32,

    /// 环境光遮蔽系数(0.0-1.0，仅在PBR模式下有效)
    #[arg(long, default_value_t = 1.0)]
    pub ambient_occlusion: f32,

    /// 材质的自发光颜色，格式为"r,g,b"，每个分量在0.0-1.0范围内(在Phong和PBR中都有效)
    #[arg(long, default_value = "0.0,0.0,0.0")]
    pub emissive: String,

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

    /// 地面平面在Y轴上的高度 (世界坐标系)
    #[arg(long, default_value_t = -1.0, allow_negative_numbers = true)]
    pub ground_plane_height: f32,
}

impl Default for Args {
    fn default() -> Self {
        let mut args = Self {
            obj: None,
            animate: false,
            fps: 30,
            rotation_speed: 1.0,
            rotation_cycles: 1.0,
            animation_type: AnimationType::CameraOrbit,
            rotation_axis: RotationAxis::Y,
            custom_rotation_axis: "0,1,0".to_string(),
            output: "output".to_string(),
            output_dir: "output_rust".to_string(),
            width: 1024,
            height: 1024,
            save_depth: true,
            projection: "perspective".to_string(),
            use_zbuffer: true,
            colorize: false,
            use_texture: true,
            texture: None,
            use_gamma: true,
            backface_culling: false,
            wireframe: false,
            use_multithreading: true,
            cull_small_triangles: false,
            min_triangle_area: 1e-3,
            object_count: None,
            camera_from: "0,0,3".to_string(),
            camera_at: "0,0,0".to_string(),
            camera_up: "0,1,0".to_string(),
            camera_fov: 45.0,
            use_lighting: true,
            ambient: 0.1,
            ambient_color: "0.1,0.1,0.1".to_string(),
            lighting_preset: LightingPreset::SingleDirectional,
            main_light_intensity: 0.8,
            directional_lights: Vec::new(),
            point_lights: Vec::new(),
            use_phong: true,
            use_pbr: false,
            diffuse_color: "0.8,0.8,0.8".to_string(),
            specular: 0.5,
            shininess: 32.0,
            base_color: "0.8,0.8,0.8".to_string(),
            metallic: 0.0,
            roughness: 0.5,
            ambient_occlusion: 1.0,
            emissive: "0.0,0.0,0.0".to_string(),
            enable_gradient_background: false,
            gradient_top_color: "0.5,0.7,1.0".to_string(),
            gradient_bottom_color: "0.1,0.2,0.4".to_string(),
            enable_ground_plane: false,
            ground_plane_color: "0.3,0.5,0.2".to_string(),
            ground_plane_height: -1.0,
        };

        // 根据预设初始化光源配置
        args.setup_light_sources();

        args
    }
}

// 辅助函数用于解析逗号分隔的浮点数
pub fn parse_vec3(s: &str) -> Result<nalgebra::Vector3<f32>, String> {
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
    Ok(nalgebra::Vector3::new(x, y, z))
}

pub fn parse_point3(s: &str) -> Result<nalgebra::Point3<f32>, String> {
    parse_vec3(s).map(nalgebra::Point3::from)
}

/// 将 Args 中的旋转轴配置转换为 Vector3<f32>
pub fn get_animation_axis_vector(args: &Args) -> Vector3<f32> {
    match args.rotation_axis {
        RotationAxis::X => Vector3::x_axis().into_inner(),
        RotationAxis::Y => Vector3::y_axis().into_inner(),
        RotationAxis::Z => Vector3::z_axis().into_inner(),
        RotationAxis::Custom => parse_vec3(&args.custom_rotation_axis)
            .unwrap_or_else(|_| {
                eprintln!(
                    "警告: 无效的自定义旋转轴 '{}', 使用默认Y轴。",
                    args.custom_rotation_axis
                );
                Vector3::y_axis().into_inner()
            })
            .normalize(),
    }
}

impl Args {
    /// 检查是否应该启动GUI模式
    pub fn should_start_gui(&self) -> bool {
        // 如果没有提供OBJ文件路径，则启动GUI
        if self.obj.is_none() {
            return true;
        }

        // 检查是否通过双击EXE启动（通常Windows下命令行参数为空）
        if std::env::args().count() <= 1 {
            return true;
        }

        false
    }

    /// 确保光源配置数组有正确的长度
    pub fn ensure_light_arrays(&mut self) {
        const MAX_DIRECTIONAL_LIGHTS: usize = 4;
        const MAX_POINT_LIGHTS: usize = 8;

        // 确保方向光源数组长度
        while self.directional_lights.len() < MAX_DIRECTIONAL_LIGHTS {
            let mut light = DirectionalLightConfig::default();
            light.enabled = false;
            self.directional_lights.push(light);
        }
        self.directional_lights.truncate(MAX_DIRECTIONAL_LIGHTS);

        // 确保点光源数组长度
        while self.point_lights.len() < MAX_POINT_LIGHTS {
            let mut light = PointLightConfig::default();
            light.enabled = false;
            self.point_lights.push(light);
        }
        self.point_lights.truncate(MAX_POINT_LIGHTS);
    }

    /// 初始化光源配置数组
    pub fn setup_light_sources(&mut self) {
        // 清除现有的光源配置
        self.directional_lights.clear();
        self.point_lights.clear();

        // 根据预设创建光源
        match self.lighting_preset {
            LightingPreset::SingleDirectional => {
                // 添加一个默认的方向光源
                self.directional_lights.push(DirectionalLightConfig {
                    enabled: true,
                    direction: "0,-1,-1".to_string(),
                    color: "1.0,1.0,1.0".to_string(),
                    intensity: self.main_light_intensity,
                });
            }
            LightingPreset::ThreeDirectional => {
                // 添加三个方向光源，从不同角度照亮场景
                self.directional_lights.push(DirectionalLightConfig {
                    enabled: true,
                    direction: "0,-1,-1".to_string(),
                    color: "1.0,1.0,1.0".to_string(),
                    intensity: self.main_light_intensity * 0.7,
                });
                self.directional_lights.push(DirectionalLightConfig {
                    enabled: true,
                    direction: "-1,-0.5,0.2".to_string(),
                    color: "0.9,0.9,1.0".to_string(),
                    intensity: self.main_light_intensity * 0.5,
                });
                self.directional_lights.push(DirectionalLightConfig {
                    enabled: true,
                    direction: "1,-0.5,0.2".to_string(),
                    color: "1.0,0.9,0.8".to_string(),
                    intensity: self.main_light_intensity * 0.3,
                });
            }
            LightingPreset::MixedComplete => {
                // 添加一个主方向光源
                self.directional_lights.push(DirectionalLightConfig {
                    enabled: true,
                    direction: "0,-1,-1".to_string(),
                    color: "1.0,1.0,1.0".to_string(),
                    intensity: self.main_light_intensity * 0.6,
                });

                // 添加四个点光源
                let point_configs = [
                    ("2,3,2", "1.0,0.8,0.6"),   // 暖色调
                    ("-2,3,2", "0.6,0.8,1.0"),  // 冷色调
                    ("2,3,-2", "0.8,1.0,0.8"),  // 绿色调
                    ("-2,3,-2", "1.0,0.8,1.0"), // 紫色调
                ];

                for (pos, color) in &point_configs {
                    self.point_lights.push(PointLightConfig {
                        enabled: true,
                        position: pos.to_string(),
                        color: color.to_string(),
                        intensity: self.main_light_intensity * 0.5,
                        constant_attenuation: 1.0,
                        linear_attenuation: 0.09,
                        quadratic_attenuation: 0.032,
                    });
                }
            }
            LightingPreset::None => {
                // 不添加任何光源
            }
        }

        // 确保光源数组长度正确
        self.ensure_light_arrays();
    }
}
