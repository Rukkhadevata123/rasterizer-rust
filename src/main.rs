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
use core::renderer::Renderer;
use io::loaders::load_obj_enhanced;
use io::render_settings::RenderSettings; // 替换原来的Args导入
use scene::scene_utils::Scene;
use utils::model_utils::normalize_and_center_model;
use utils::render_utils::render_single_frame;
use utils::render_utils::run_animation_loop;

fn main() -> Result<(), String> {
    // 解析命令行参数
    let mut settings = RenderSettings::parse();

    // 确保根据预设初始化光源配置
    settings.setup_light_sources();

    // 判断是否应该启动GUI模式
    if settings.should_start_gui() {
        println!("启动GUI模式...");
        if let Err(err) = ui::start_gui(settings) {
            return Err(format!("GUI启动失败: {}", err));
        }
        return Ok(());
    }

    // 如果代码执行到这里，说明有OBJ文件路径，进入命令行渲染模式
    let start_time = Instant::now();

    // 获取OBJ文件路径（此时我们确定obj是Some，所以可以安全unwrap）
    let obj_path = settings.obj.as_ref().unwrap();

    // --- 验证输入和设置 ---
    if !Path::new(obj_path).exists() {
        return Err(format!("错误：输入的 OBJ 文件未找到：{}", obj_path));
    }

    // 确保输出目录存在
    fs::create_dir_all(&settings.output_dir)
        .map_err(|e| format!("创建输出目录 '{}' 失败：{}", settings.output_dir, e))?;

    // --- 加载模型 ---
    println!("加载模型：{}", obj_path);
    let load_start = Instant::now();
    let mut model_data = load_obj_enhanced(obj_path, &settings)?;
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
    let mut scene = Scene::create_from_model_and_settings(model_data, &settings)?;
    println!(
        "创建了包含 {} 个对象、{} 个光源的场景",
        scene.object_count(),
        scene.light_count()
    );

    // --- 创建渲染器 ---
    let renderer = Renderer::new(settings.width, settings.height);

    // --- 渲染动画或单帧 ---
    if settings.animate {
        run_animation_loop(&settings, &mut scene, &renderer)?;
    } else {
        println!("--- 准备单帧渲染 ---");
        // 创建配置副本并从场景更新它
        let mut render_settings = settings.clone();
        render_settings.update_from_scene(&scene);

        // 打印配置摘要
        println!("--- 渲染配置摘要 ---");
        render_settings.print_summary();
        println!("-------------------");

        println!("使用{}渲染", render_settings.get_lighting_description());
        render_single_frame(
            &settings,
            &scene,
            &renderer,
            &render_settings,
            &settings.output,
        )?;
    }

    println!("总执行时间：{:?}", start_time.elapsed());
    Ok(())
}
