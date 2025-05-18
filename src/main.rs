use clap::Parser;
use std::fs;
use std::path::Path;
use std::time::Instant;

// 声明模块
mod core;
mod geometry;
mod io;
mod materials;
mod scene;
mod ui;
mod utils;

// 导入语句
use core::renderer::Renderer;
use geometry::camera::Camera;
use io::args::{Args, parse_point3, parse_vec3};
use io::loaders::load_obj_enhanced;
use materials::model_types::ModelData;
use scene::scene_utils::Scene;
use utils::animation_utils::run_animation_loop;
use utils::material_utils::{apply_pbr_parameters, apply_phong_parameters};
use utils::model_utils::normalize_and_center_model;
use utils::render_utils::{create_render_config, render_single_frame};
use utils::test_utils::test_transformation_api;

// 创建场景相机
fn create_camera(args: &Args) -> Result<Camera, String> {
    let aspect_ratio = args.width as f32 / args.height as f32;
    let camera_from =
        parse_point3(&args.camera_from).map_err(|e| format!("无效的相机位置格式: {}", e))?;
    let camera_at =
        parse_point3(&args.camera_at).map_err(|e| format!("无效的相机目标格式: {}", e))?;
    let camera_up =
        parse_vec3(&args.camera_up).map_err(|e| format!("无效的相机上方向格式: {}", e))?;

    Ok(Camera::new(
        camera_from,
        camera_at,
        camera_up,
        args.camera_fov,
        aspect_ratio,
        0.1,   // 近平面距离
        100.0, // 远平面距离
    ))
}

/// 设置场景光源
fn setup_lights(scene: &mut Scene, args: &Args) -> Result<(), String> {
    // 使用Scene中的统一方法设置光照系统
    scene.setup_lighting(
        args.use_lighting,
        &args.light_type,
        &args.light_dir,
        &args.light_pos,
        &args.light_atten,
        args.diffuse,
        args.ambient,
        &args.ambient_color,
    )?;

    // 打印光照信息
    if args.use_lighting {
        if args.light_type == "point" {
            println!(
                "使用点光源，位置: {}, 强度系数: {:.2}",
                args.light_pos, args.diffuse
            );
        } else {
            println!(
                "使用定向光，方向: {}, 强度系数: {:.2}",
                args.light_dir, args.diffuse
            );
        }
    } else if let Ok(ambient_color) = parse_vec3(&args.ambient_color) {
        println!("使用RGB环境光强度: {:?}", ambient_color);
    } else {
        println!("使用环境光强度: {:.2}", args.ambient);
    }

    Ok(())
}

/// 创建并设置场景
fn setup_scene(model_data: ModelData, args: &Args) -> Result<Scene, String> {
    // 创建相机
    let camera = create_camera(args)?;

    // 创建场景并设置相机
    let mut scene = Scene::new(camera);

    // 应用PBR材质参数(如果需要)和Phong材质参数(如果需要)
    let mut modified_model_data = model_data.clone();
    apply_pbr_parameters(&mut modified_model_data, args);
    apply_phong_parameters(&mut modified_model_data, args);

    // 设置场景对象
    let object_count = args
        .object_count
        .as_ref()
        .and_then(|count_str| count_str.parse::<usize>().ok());

    scene.setup_from_model_data(modified_model_data, object_count);

    // 如果有多个对象，打印信息
    if let Some(count) = object_count {
        if count > 1 {
            println!("创建了环形排列的 {} 个附加对象", count - 1);
        }
    }

    // 设置光照
    setup_lights(&mut scene, args)?;

    Ok(scene)
}

// 在main函数中添加对test_transformation_api的条件调用
fn main() -> Result<(), String> {
    let args = Args::parse();

    // 如果指定了GUI模式，则启动GUI界面
    if args.gui {
        println!("启动GUI模式...");
        if let Err(err) = ui::start_gui(args) {
            return Err(format!("GUI启动失败: {}", err));
        }
        return Ok(());
    }

    // 如果指定了test_api参数，则运行API测试
    if args.test_api {
        return test_transformation_api(&args.obj);
    }

    let start_time = Instant::now();

    // --- 验证输入和设置 ---
    if !Path::new(&args.obj).exists() {
        return Err(format!("错误：输入的 OBJ 文件未找到：{}", args.obj));
    }

    // 确保输出目录存在
    fs::create_dir_all(&args.output_dir)
        .map_err(|e| format!("创建输出目录 '{}' 失败：{}", args.output_dir, e))?;

    // --- 加载模型 ---
    println!("加载模型：{}", args.obj);
    let load_start = Instant::now();
    let mut model_data = load_obj_enhanced(&args.obj, &args)?;
    println!("模型加载耗时 {:?}", load_start.elapsed());

    // --- 归一化模型 ---
    println!("归一化模型...");
    let norm_start_time = Instant::now();
    let (original_center, scale_factor) = normalize_and_center_model(&mut model_data);
    println!(
        "模型归一化耗时 {:?}。原始中心：{:.3?}，缩放系数：{:.3}",
        norm_start_time.elapsed(),
        original_center,
        scale_factor
    );

    // --- 创建并设置场景 ---
    println!("创建场景...");
    let mut scene = setup_scene(model_data, &args)?;
    println!(
        "创建了包含 {} 个对象、{} 个光源的场景",
        scene.object_count(),
        scene.light_count()
    );

    // --- 创建渲染器 ---
    let renderer = Renderer::new(args.width, args.height);

    // --- 渲染动画或单帧 ---
    if args.animate {
        run_animation_loop(&args, &mut scene, &renderer)?;
    } else {
        println!("--- 准备单帧渲染 ---");
        // 使用从demos模块导入的create_render_config函数
        let config = create_render_config(&scene, &args);
        println!("使用{}渲染", config.get_lighting_description());
        render_single_frame(&args, &scene, &renderer, &config, &args.output)?;
    }

    println!("总执行时间：{:?}", start_time.elapsed());
    Ok(())
}
