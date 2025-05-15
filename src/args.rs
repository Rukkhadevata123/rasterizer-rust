use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// 输入OBJ文件的路径
    #[arg(long)]
    pub obj: String,

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

    /// 投影类型："perspective"（透视投影）或"orthographic"（正交投影）
    #[arg(long, default_value = "perspective")]
    pub projection: String,

    /// 启用Z缓冲（深度测试）
    #[arg(long, default_value_t = true)]
    pub use_zbuffer: bool,

    /// 使用伪随机面颜色而非材质颜色
    #[arg(long, default_value_t = false)]
    pub colorize: bool,

    /// 启用渲染和保存深度图
    #[arg(long, default_value_t = true)]
    pub save_depth: bool,

    // --- 相机参数 ---
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

    // --- 光照参数 ---
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

    /// 使用Phong着色（逐像素光照）而非默认的Flat着色
    #[arg(long, default_value_t = false)]
    pub use_phong: bool,

    /// 使用基于物理的渲染(PBR)而不是传统Blinn-Phong
    #[arg(long, default_value_t = false)]
    pub use_pbr: bool,

    /// 材质的金属度(0.0-1.0，仅在PBR模式下有效)
    #[arg(long, default_value_t = 0.0)]
    pub metallic: f32,

    /// 材质的粗糙度(0.0-1.0，仅在PBR模式下有效)
    #[arg(long, default_value_t = 0.5)]
    pub roughness: f32,

    /// 材质的基础颜色，格式为"r,g,b"，每个分量在0.0-1.0范围内(仅在PBR模式下有效)
    #[arg(long, default_value = "0.8,0.8,0.8")]
    pub base_color: String,

    /// 环境光遮蔽系数(0.0-1.0，仅在PBR模式下有效)
    #[arg(long, default_value_t = 1.0)]
    pub ambient_occlusion: f32,

    /// 材质的自发光颜色，格式为"r,g,b"，每个分量在0.0-1.0范围内(仅在PBR模式下有效)
    #[arg(long, default_value = "0.0,0.0,0.0")]
    pub emissive: String,

    /// 启用纹理加载和使用
    #[arg(long, default_value_t = true)]
    pub use_texture: bool,

    /// 显式指定要使用的纹理文件，覆盖MTL设置
    #[arg(long)]
    pub texture: Option<String>,

    /// 启用gamma矫正
    #[arg(long, default_value_t = true)]
    pub use_gamma: bool,

    /// 场景中要创建的对象实例数量
    #[arg(long)]
    pub object_count: Option<String>,

    /// 运行完整动画循环而非单帧渲染
    #[arg(long, default_value_t = false)]
    pub animate: bool,

    /// 动画的总帧数
    #[arg(long, default_value_t = 120)]
    pub total_frames: usize,

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

    /// 测试API模式 - 仅用于测试未使用的变换API
    #[arg(long, default_value_t = false)]
    pub test_api: bool,
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
