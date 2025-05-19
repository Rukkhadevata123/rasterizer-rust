use crate::io::args::{Args, parse_vec3};
use crate::material_system::light::Light;
use crate::scene::scene_utils::Scene;
use nalgebra::Vector3;

/// 统一的渲染配置结构体，整合了所有渲染相关设置
#[derive(Debug, Clone)]
pub struct RenderConfig {
    // 投影相关设置
    /// 投影类型："perspective" 或 "orthographic"
    pub projection_type: String,

    // 缓冲区控制
    /// 是否启用深度缓冲和深度测试
    pub use_zbuffer: bool,

    // 着色和光照
    /// 是否应用光照计算
    pub use_lighting: bool,
    /// 是否使用面颜色而非材质颜色
    pub use_face_colors: bool,
    /// 是否使用Phong着色（逐像素光照计算）
    pub use_phong: bool,
    /// 是否使用基于物理的渲染 (PBR)
    pub use_pbr: bool,

    // 纹理和后处理
    /// 是否使用纹理映射
    pub use_texture: bool,
    /// 是否应用gamma校正（sRGB空间转换）
    pub apply_gamma_correction: bool,

    // 光照信息
    /// 默认光源配置
    pub light: Light,

    // 环境光信息（作为场景的基础属性）
    /// 环境光强度 - 控制场景整体亮度 [0.0, 1.0]
    pub ambient_intensity: f32,
    /// 环境光颜色 - 控制场景基础色调 (RGB)
    pub ambient_color: nalgebra::Vector3<f32>,

    // 几何处理
    /// 是否启用背面剔除
    pub use_backface_culling: bool,
    /// 是否以线框模式渲染
    pub use_wireframe: bool,

    // 性能优化设置
    /// 是否启用多线程渲染
    pub use_multithreading: bool,
    /// 是否对小三角形进行剔除
    pub cull_small_triangles: bool,
    /// 用于剔除的最小三角形面积
    pub min_triangle_area: f32,

    // 背景与环境设置
    /// 启用渐变背景
    pub enable_gradient_background: bool,
    /// 渐变背景顶部颜色
    pub gradient_top_color: Vector3<f32>,
    /// 渐变背景底部颜色
    pub gradient_bottom_color: Vector3<f32>,

    /// 启用地面平面
    pub enable_ground_plane: bool,
    /// 地面平面颜色
    pub ground_plane_color: Vector3<f32>,
    /// 地面平面在Y轴上的高度 (世界坐标系)
    pub ground_plane_height: f32,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            projection_type: "perspective".to_string(),
            use_zbuffer: true,
            use_lighting: true,
            use_face_colors: false,
            use_phong: false,
            use_pbr: false,
            use_texture: true,
            apply_gamma_correction: true,
            light: Light::directional(
                nalgebra::Vector3::new(0.0, -1.0, -1.0).normalize(),
                nalgebra::Vector3::new(1.0, 1.0, 1.0),
            ),
            ambient_intensity: 0.1, // 默认环境光强度
            ambient_color: nalgebra::Vector3::new(1.0, 1.0, 1.0), // 默认环境光颜色（白色）
            use_backface_culling: false,
            use_wireframe: false,
            use_multithreading: true,
            cull_small_triangles: false,
            min_triangle_area: 1e-3,
            enable_gradient_background: false,
            gradient_top_color: Vector3::new(0.5, 0.7, 1.0),
            gradient_bottom_color: Vector3::new(0.1, 0.2, 0.4),
            enable_ground_plane: false,
            ground_plane_color: Vector3::new(0.3, 0.5, 0.2),
            ground_plane_height: -1.0,
        }
    }
}

impl RenderConfig {
    /// 获取光照模型的描述字符串
    pub fn get_lighting_description(&self) -> String {
        if self.use_pbr {
            "基于物理的渲染(PBR)".to_string()
        } else if self.use_phong {
            "Phong着色模型".to_string()
        } else {
            "平面着色模型".to_string()
        }
    }

    // 构建器方法，便于链式配置
    pub fn with_projection(mut self, projection_type: &str) -> Self {
        self.projection_type = projection_type.to_string();
        self
    }

    pub fn with_zbuffer(mut self, use_zbuffer: bool) -> Self {
        self.use_zbuffer = use_zbuffer;
        self
    }

    pub fn with_lighting(mut self, use_lighting: bool) -> Self {
        self.use_lighting = use_lighting;
        self
    }

    pub fn with_face_colors(mut self, use_face_colors: bool) -> Self {
        self.use_face_colors = use_face_colors;
        self
    }

    pub fn with_phong(mut self, use_phong: bool) -> Self {
        self.use_phong = use_phong;
        self
    }

    pub fn with_pbr(mut self, use_pbr: bool) -> Self {
        self.use_pbr = use_pbr;
        self
    }

    pub fn with_texture(mut self, use_texture: bool) -> Self {
        self.use_texture = use_texture;
        self
    }

    pub fn with_gamma_correction(mut self, apply_gamma_correction: bool) -> Self {
        self.apply_gamma_correction = apply_gamma_correction;
        self
    }

    pub fn with_light(mut self, light: Light) -> Self {
        self.light = light;
        self
    }

    pub fn with_ambient_intensity(mut self, intensity: f32) -> Self {
        self.ambient_intensity = intensity;
        self
    }

    pub fn with_ambient_color(mut self, color: nalgebra::Vector3<f32>) -> Self {
        self.ambient_color = color;
        self
    }

    pub fn with_backface_culling(mut self, use_backface_culling: bool) -> Self {
        self.use_backface_culling = use_backface_culling;
        self
    }

    pub fn with_wireframe(mut self, use_wireframe: bool) -> Self {
        self.use_wireframe = use_wireframe;
        self
    }

    pub fn with_multithreading(mut self, use_multithreading: bool) -> Self {
        self.use_multithreading = use_multithreading;
        self
    }

    pub fn with_small_triangle_culling(mut self, enable: bool, min_area: f32) -> Self {
        self.cull_small_triangles = enable;
        self.min_triangle_area = min_area;
        self
    }

    pub fn with_gradient_background(
        mut self,
        enabled: bool,
        top_color: Vector3<f32>,
        bottom_color: Vector3<f32>,
    ) -> Self {
        self.enable_gradient_background = enabled;
        self.gradient_top_color = top_color;
        self.gradient_bottom_color = bottom_color;
        self
    }

    pub fn with_ground_plane(mut self, enabled: bool, color: Vector3<f32>, height: f32) -> Self {
        self.enable_ground_plane = enabled;
        self.ground_plane_color = color;
        self.ground_plane_height = height.min(-0.00001f32);
        self
    }

    /// 判断是否使用透视投影
    pub fn is_perspective(&self) -> bool {
        self.projection_type == "perspective"
    }
}

/// 创建渲染配置
///
/// 基于场景和命令行参数创建渲染配置
///
/// # 参数
/// * `scene` - 场景引用，用于获取光源信息
/// * `args` - 命令行参数引用
///
/// # 返回值
/// 配置好的RenderConfig对象
pub fn create_render_config(scene: &Scene, args: &Args) -> RenderConfig {
    // --- 光源处理 ---
    // 优先使用场景中的光源
    let main_light = if !scene.lights.is_empty() {
        scene.lights[0] // 使用场景中的第一个光源作为主光源
    } else {
        // 如果场景中没有光源，创建一个临时场景并设置光照
        let mut temp_scene = Scene::new(scene.active_camera.clone());
        // 使用统一的光照设置方法
        let _ = temp_scene.setup_lighting(
            Some(args),
            &args.light_type,
            args.diffuse,
            args.ambient,
            None,
        );

        // 获取创建的光源(如果有)，否则使用默认光源
        temp_scene.lights.first().cloned().unwrap_or_else(|| {
            Light::directional(
                Vector3::new(-1.0, -1.0, -1.0).normalize(),
                Vector3::new(1.0, 1.0, 1.0),
            )
        })
    };

    // --- 环境光处理 ---
    // 直接使用场景中的环境光设置
    let ambient_intensity = scene.ambient_intensity;
    let ambient_color = scene.ambient_color;

    // --- 创建渲染配置 ---
    let config = RenderConfig::default()
        // --- 投影和缓冲设置 ---
        .with_projection(&args.projection)
        .with_zbuffer(args.use_zbuffer)
        // --- 材质和着色设置 ---
        .with_face_colors(args.colorize)
        .with_texture(args.use_texture)
        .with_phong(args.use_phong)
        .with_pbr(args.use_pbr)
        // --- 光照设置 ---
        .with_lighting(args.use_lighting)
        .with_light(main_light)
        .with_ambient_intensity(ambient_intensity)
        .with_ambient_color(ambient_color)
        // --- 后处理设置 ---
        .with_gamma_correction(args.use_gamma)
        // --- 几何处理设置 ---
        .with_backface_culling(args.backface_culling)
        .with_wireframe(args.wireframe)
        // --- 性能优化设置 ---
        .with_multithreading(args.use_multithreading)
        .with_small_triangle_culling(args.cull_small_triangles, args.min_triangle_area)
        // --- 背景与环境设置 ---
        .with_gradient_background(
            args.enable_gradient_background,
            parse_vec3(&args.gradient_top_color).unwrap_or_else(|_| Vector3::new(0.5, 0.7, 1.0)),
            parse_vec3(&args.gradient_bottom_color).unwrap_or_else(|_| Vector3::new(0.1, 0.2, 0.4)),
        )
        .with_ground_plane(
            args.enable_ground_plane,
            parse_vec3(&args.ground_plane_color).unwrap_or_else(|_| Vector3::new(0.3, 0.5, 0.2)),
            args.ground_plane_height,
        );

    // --- 打印渲染设置摘要 ---
    print_render_config_summary(&config, args);

    config
}

/// 打印渲染配置摘要
pub fn print_render_config_summary(config: &RenderConfig, args: &Args) {
    // --- 着色模型 ---
    if args.use_pbr {
        println!("着色模型: 基于物理的渲染(PBR)");
    } else if args.use_phong {
        println!("着色模型: Phong着色模型");
    } else {
        println!("着色模型: 平面着色模型");
    }

    // --- 光照设置 ---
    println!(
        "光照: {}",
        if args.use_lighting {
            "启用"
        } else {
            "禁用"
        }
    );
    if args.use_lighting {
        println!("光源类型: {}", args.light_type);
        println!("主光源: {:?}", config.light);
        println!(
            "环境光: 强度={:.2}, 颜色={:?}",
            config.ambient_intensity, config.ambient_color
        );
    }

    // --- 材质设置 ---
    println!(
        "材质: 纹理={}, 面颜色={}, Gamma校正={}",
        if args.use_texture { "启用" } else { "禁用" },
        if args.colorize { "启用" } else { "禁用" },
        if args.use_gamma { "启用" } else { "禁用" }
    );

    // --- 几何处理 ---
    println!(
        "几何处理: 背面剔除={}, 线框模式={}",
        if args.backface_culling {
            "启用"
        } else {
            "禁用"
        },
        if args.wireframe { "启用" } else { "禁用" }
    );

    // --- 性能设置 ---
    println!(
        "性能设置: 多线程渲染={}, 小三角形剔除={}{}",
        if args.use_multithreading {
            "启用"
        } else {
            "禁用"
        },
        if args.cull_small_triangles {
            "启用"
        } else {
            "禁用"
        },
        if args.cull_small_triangles {
            format!(" (阈值: {:.5})", args.min_triangle_area)
        } else {
            String::new()
        }
    );
}
