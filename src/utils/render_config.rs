use crate::core::renderer::RenderConfig;
use crate::io::args::{Args, parse_vec3};
use crate::materials::material_system::Light;
use crate::scene::scene_utils::Scene;
use nalgebra::{Point3, Vector3};

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
    // 优先使用场景中的光源，如果没有则创建默认光源
    let main_light = scene
        .lights
        .iter()
        .copied()
        .next()
        .unwrap_or_else(|| create_light_from_args(args));

    // --- 环境光处理 ---
    // 解析环境光颜色
    let ambient_color = parse_ambient_color(args);

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
        .with_ambient_intensity(args.ambient)
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

/// 从命令行参数创建光源
pub fn create_light_from_args(args: &Args) -> Light {
    if args.light_type == "directional" {
        // 创建定向光源
        if let Ok(dir) = parse_vec3(&args.light_dir) {
            Light::directional(
                dir.normalize(),
                Vector3::new(1.0, 1.0, 1.0), // 默认白色光
            )
        } else {
            // 解析失败时使用默认方向
            Light::directional(
                Vector3::new(0.0, -1.0, -1.0).normalize(),
                Vector3::new(1.0, 1.0, 1.0),
            )
        }
    } else {
        // 创建点光源
        if let Ok(pos) = parse_vec3(&args.light_pos) {
            // 解析衰减参数
            let atten_parts: Vec<&str> = args.light_atten.split(',').collect();
            let constant = atten_parts
                .first()
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or(1.0);
            let linear = atten_parts
                .get(1)
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or(0.09);
            let quadratic = atten_parts
                .get(2)
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or(0.032);

            Light::point(
                Point3::from(pos),                   // 将Vector3转换为Point3
                Vector3::new(1.0, 1.0, 1.0),         // 默认白色光
                Some((constant, linear, quadratic)), // 使用Some包装衰减参数元组
            )
        } else {
            // 解析失败时使用默认位置
            Light::point(
                Point3::new(0.0, 5.0, 5.0),
                Vector3::new(1.0, 1.0, 1.0),
                Some((1.0, 0.09, 0.032)), // 默认衰减值，使用Some包装
            )
        }
    }
}

/// 解析环境光颜色
pub fn parse_ambient_color(args: &Args) -> Vector3<f32> {
    if let Ok(color) = parse_vec3(&args.ambient_color) {
        color
    } else {
        Vector3::new(args.ambient, args.ambient, args.ambient) // 使用单一环境光强度
    }
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
