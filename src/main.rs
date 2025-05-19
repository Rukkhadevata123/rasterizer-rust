use clap::Parser;
use std::fs;
use std::path::Path;
use std::time::Instant;

// 声明模块
mod core;
mod geometry;
mod io;
mod material_system;
mod scene;
mod ui;
mod utils;

// 导入语句
use core::render_config::create_render_config;
use core::renderer::Renderer;
use io::args::Args;
use io::loaders::load_obj_enhanced;
use scene::scene_utils::Scene;
use utils::model_utils::normalize_and_center_model;
use utils::render_utils::render_single_frame;
use utils::render_utils::run_animation_loop;

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
    let mut scene = Scene::create_from_model_and_args(model_data, &args)?;
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
        // 使用从utils模块导入的create_render_config函数
        let config = create_render_config(&scene, &args);
        println!("使用{}渲染", config.get_lighting_description());
        render_single_frame(&args, &scene, &renderer, &config, &args.output)?;
    }

    println!("总执行时间：{:?}", start_time.elapsed());
    Ok(())
}
