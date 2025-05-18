use crate::core::renderer::RenderConfig;
use crate::scene::scene_utils::Scene;
use crate::io::args::{Args, parse_vec3};
use crate::materials::color::apply_colormap_jet;
use crate::materials::material_system::Light;
use crate::utils::image_utils::{normalize_depth, save_image};
use nalgebra::{Point3, Vector3};
use std::path::Path;
use std::time::Instant;

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
        .with_small_triangle_culling(args.cull_small_triangles, args.min_triangle_area);

    // --- 打印渲染设置摘要 ---
    print_render_config_summary(&config, args);

    config
}

/// 从命令行参数创建光源
fn create_light_from_args(args: &Args) -> Light {
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
fn parse_ambient_color(args: &Args) -> Vector3<f32> {
    if let Ok(color) = parse_vec3(&args.ambient_color) {
        color
    } else {
        Vector3::new(args.ambient, args.ambient, args.ambient) // 使用单一环境光强度
    }
}

/// 打印渲染配置摘要
fn print_render_config_summary(config: &RenderConfig, args: &Args) {
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

/// 保存渲染结果（彩色图像和可选的深度图）
///
/// # 参数
/// * `color_data` - 渲染的颜色数据（RGB格式的u8数组）
/// * `depth_data` - 可选的深度数据（f32数组）
/// * `width` - 图像宽度
/// * `height` - 图像高度
/// * `output_dir` - 输出目录路径
/// * `output_name` - 输出文件名（不含扩展名和后缀）
/// * `config` - 渲染配置
/// * `save_depth` - 是否保存深度图
///
/// # 返回值
/// Result，成功为()，失败为包含错误信息的字符串
#[allow(clippy::too_many_arguments)]
pub fn save_render_result(
    color_data: &[u8],
    depth_data: Option<&[f32]>,
    width: usize,
    height: usize,
    output_dir: &str,
    output_name: &str,
    config: &RenderConfig,
    save_depth: bool,
) -> Result<(), String> {
    // 保存彩色图像
    let color_path = Path::new(output_dir)
        .join(format!("{}_color.png", output_name))
        .to_str()
        .ok_or_else(|| "创建彩色输出路径字符串失败".to_string())?
        .to_string();

    // 使用image_utils.rs中的save_image函数
    save_image(&color_path, color_data, width as u32, height as u32);

    // 保存深度图（如果启用）
    if config.use_zbuffer && save_depth {
        if let Some(depth_data_raw) = depth_data {
            let depth_normalized = normalize_depth(depth_data_raw, 1.0, 99.0);
            let depth_colored = apply_colormap_jet(
                &depth_normalized
                    .iter()
                    .map(|&d| 1.0 - d) // 反转：越近 = 越热
                    .collect::<Vec<_>>(),
                width,
                height,
                config.apply_gamma_correction,
            );

            let depth_path = Path::new(output_dir)
                .join(format!("{}_depth.png", output_name))
                .to_str()
                .ok_or_else(|| "创建深度输出路径字符串失败".to_string())?
                .to_string();

            // 使用image_utils.rs中的save_image函数
            save_image(&depth_path, &depth_colored, width as u32, height as u32);
        }
    }

    Ok(())
}

/// 从渲染器中获取数据并保存渲染结果
///
/// 这是对 save_render_result 的便捷包装函数，适用于所有调用场景
///
/// # 参数
/// * `renderer` - 渲染器引用，用于获取渲染数据
/// * `args` - 命令行参数引用，包含输出路径信息
/// * `config` - 渲染配置引用
/// * `output_name` - 自定义输出名称（如果为None，则使用args.output）
///
/// # 返回值
/// Result，成功为()，失败为包含错误信息的字符串
pub fn save_render_with_args(
    renderer: &crate::core::renderer::Renderer,
    args: &Args,
    config: &RenderConfig,
    output_name: Option<&str>,
) -> Result<(), String> {
    let color_data = renderer.frame_buffer.get_color_buffer_bytes();
    let depth_data = if args.save_depth {
        Some(renderer.frame_buffer.get_depth_buffer_f32())
    } else {
        None
    };

    let width = renderer.frame_buffer.width;
    let height = renderer.frame_buffer.height;
    let output_name = output_name.unwrap_or(&args.output);

    save_render_result(
        &color_data,
        depth_data.as_deref(),
        width,
        height,
        &args.output_dir,
        output_name,
        config,
        args.save_depth,
    )
}

/// 渲染单帧并保存结果
///
/// 完整处理单帧渲染过程：渲染场景、保存输出、打印信息
///
/// # 参数
/// * `args` - 命令行参数引用
/// * `scene` - 场景引用
/// * `renderer` - 渲染器引用
/// * `config` - 渲染配置引用
/// * `output_name` - 输出文件名
///
/// # 返回值
/// Result，成功为()，失败为包含错误信息的字符串
pub fn render_single_frame(
    args: &Args,
    scene: &Scene,
    renderer: &crate::core::renderer::Renderer,
    config: &RenderConfig,
    output_name: &str,
) -> Result<(), String> {
    let frame_start_time = Instant::now();
    println!("渲染帧: {}", output_name);

    // 渲染场景 - 克隆配置以避免可变引用问题
    let mut config_clone = config.clone();
    renderer.render_scene(scene, &mut config_clone);

    // 保存输出图像
    println!("保存 {} 的输出图像...", output_name);
    save_render_with_args(renderer, args, config, Some(output_name))?;

    // 打印材质信息（调试用）
    if let Some(model) = scene.models.first() {
        for (i, material) in model.materials.iter().enumerate() {
            if i == 0 {
                println!("材质 #{}: {}", i, material.get_name());
                println!("  漫反射颜色: {:?}", material.diffuse());
            }
        }
    }

    println!(
        "帧 {} 渲染完成，耗时 {:?}",
        output_name,
        frame_start_time.elapsed()
    );
    Ok(())
}
