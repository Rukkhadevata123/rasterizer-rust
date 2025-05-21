use clap::Parser;
use nalgebra::Vector3;

#[derive(clap::ValueEnum, Debug, Clone, Default, PartialEq, Eq)]
pub enum AnimationType {
    #[default]
    CameraOrbit, // 重命名
    ObjectLocalRotation, // 重命名
    None,
}

#[derive(clap::ValueEnum, Debug, Clone, PartialEq, Eq, Default)]
pub enum RotationAxis {
    X,
    #[default]
    Y,
    Z,
    Custom, // 允许用户指定自定义轴
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    // ===== 基础设置 =====
    /// 启用GUI界面模式
    #[arg(long)]
    pub gui: bool,

    /// 输入OBJ文件的路径
    #[arg(
        long,
        required_unless_present = "gui",
        default_value = "obj/simple/bunny.obj"
    )]
    pub obj: String,

    /// 运行完整动画循环而非单帧渲染
    #[arg(long, default_value_t = false)]
    pub animate: bool,

    /// 动画帧率 (fps)，用于视频生成和预渲染
    #[arg(long, default_value_t = 30)]
    pub fps: usize,

    /// 旋转速度系数，控制动画旋转的速度
    #[arg(long, default_value_t = 1.0)]
    pub rotation_speed: f32,

    /// 动画的总帧数
    #[arg(long, default_value_t = 120)]
    pub total_frames: usize,

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

    /// 光源类型："directional"（定向光）或"point"（点光源）
    #[arg(long, default_value = "directional")]
    pub light_type: String,

    /// 光源方向（来自光源的方向，用于定向光），格式为"x,y,z"
    #[arg(long, default_value = "0,-1,-1", allow_negative_numbers = true)]
    pub light_dir: String,

    /// 光源位置（用于点光源），格式为"x,y,z"
    #[arg(long, default_value = "0,5,5", allow_negative_numbers = true)]
    pub light_pos: String,

    /// 点光源的衰减因子（常数项,线性项,二次项），格式为"c,l,q"
    #[arg(long, default_value = "1.0,0.09,0.032", allow_negative_numbers = true)]
    pub light_atten: String,

    /// 环境光强度因子
    #[arg(long, default_value_t = 0.1)]
    pub ambient: f32,

    /// 环境光强度RGB值，格式为"r,g,b"，优先级高于ambient参数
    #[arg(long, default_value = "0.1,0.1,0.1")]
    pub ambient_color: String,

    /// 漫反射光强度因子
    #[arg(long, default_value_t = 0.8)]
    pub diffuse: f32,

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

// 辅助函数用于解析逗号分隔的浮点数
// 标准解析应该能处理负数，这里不需要特殊逻辑
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

/// 解析点光源衰减参数，格式为 "constant,linear,quadratic"
pub fn parse_attenuation(s: &str) -> Result<(f32, f32, f32), String> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 3 {
        return Err("衰减参数需要3个逗号分隔的值".to_string());
    }

    let constant = parts[0]
        .trim()
        .parse::<f32>()
        .map_err(|e| format!("无效的衰减常数 '{}': {}", parts[0], e))?;

    let linear = parts[1]
        .trim()
        .parse::<f32>()
        .map_err(|e| format!("无效的衰减线性系数 '{}': {}", parts[1], e))?;

    let quadratic = parts[2]
        .trim()
        .parse::<f32>()
        .map_err(|e| format!("无效的衰减二次系数 '{}': {}", parts[2], e))?;

    Ok((constant, linear, quadratic))
}

/// 将 Args 中的旋转轴配置转换为 Vector3<f32>
pub fn get_animation_axis_vector(args: &Args) -> Vector3<f32> {
    match args.rotation_axis {
        RotationAxis::X => Vector3::x_axis().into_inner(),
        RotationAxis::Y => Vector3::y_axis().into_inner(),
        RotationAxis::Z => Vector3::z_axis().into_inner(),
        RotationAxis::Custom => {
            parse_vec3(&args.custom_rotation_axis)
                .unwrap_or_else(|_| {
                    eprintln!(
                        "警告: 无效的自定义旋转轴 '{}', 使用默认Y轴。",
                        args.custom_rotation_axis
                    );
                    Vector3::y_axis().into_inner()
                })
                .normalize() // 确保自定义轴是单位向量
        }
    }
}
