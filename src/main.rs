use log::{error, info, warn};
use std::fs;
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
use io::model_loader::ModelLoader;
use io::simple_cli::SimpleCli;
use utils::render_utils::{render_single_frame, run_animation_loop};

fn main() -> Result<(), String> {
    // 初始化日志系统 - 默认DEBUG级别
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Debug)
        .filter_module("eframe", log::LevelFilter::Warn) // 只显示 eframe 的警告和错误
        .filter_module("egui_glow", log::LevelFilter::Warn) // 只显示 egui_glow 的警告和错误
        .filter_module("egui_winit", log::LevelFilter::Warn) // 只显示 egui_winit 的警告和错误
        .filter_module("winit", log::LevelFilter::Warn) // 只显示 winit 的警告和错误
        .filter_module("wgpu", log::LevelFilter::Warn) // 只显示 wgpu 的警告和错误
        .filter_module("glutin", log::LevelFilter::Warn) // 只显示 glutin 的警告和错误
        .filter_module("sctk", log::LevelFilter::Warn) // 只显示 sctk 的警告和错误
        .format_timestamp(None)
        .format_level(true)
        .init();

    info!("🎨 光栅化渲染器启动");

    let (settings, should_start_gui) = SimpleCli::process()?;

    // 判断是否应该启动GUI模式
    if should_start_gui {
        info!("启动GUI模式...");
        if let Err(err) = ui::start_gui(settings) {
            error!("GUI启动失败: {}", err);
            return Err("GUI启动失败".to_string());
        }
        return Ok(());
    }

    // 无头渲染模式 - 需要OBJ文件
    if settings.obj.is_none() {
        error!("无头模式需要指定OBJ文件路径");
        return Err("缺少OBJ文件路径".to_string());
    }

    let start_time = Instant::now();
    let obj_path = settings.obj.as_ref().unwrap();

    // 确保输出目录存在
    fs::create_dir_all(&settings.output_dir).map_err(|e| {
        error!("创建输出目录 '{}' 失败：{}", settings.output_dir, e);
        "创建输出目录失败".to_string()
    })?;

    // 验证资源
    info!("验证资源...");
    if let Err(e) = ModelLoader::validate_resources(&settings) {
        warn!("{}", e);
    }

    // 加载模型和创建场景
    let (mut scene, _model_data) = ModelLoader::load_and_create_scene(obj_path, &settings)
        .map_err(|e| {
            error!("模型加载失败: {}", e);
            "模型加载失败".to_string()
        })?;

    // 创建渲染器
    let mut renderer = Renderer::new(settings.width, settings.height);

    // 渲染动画或单帧
    if settings.animate {
        run_animation_loop(&mut scene, &mut renderer, &settings).map_err(|e| {
            error!("动画渲染失败: {}", e);
            "动画渲染失败".to_string()
        })?;
    } else {
        info!("--- 开始单帧渲染 ---");
        info!("分辨率: {}x{}", settings.width, settings.height);
        info!("投影类型: {}", settings.projection);
        info!(
            "光照: {} ({} 个光源)",
            if settings.use_lighting {
                "启用"
            } else {
                "禁用"
            },
            settings.lights.len()
        );
        info!("材质: {}", settings.get_lighting_description());

        if settings.use_background_image {
            if let Some(bg_path) = &settings.background_image_path {
                info!("背景图片: {}", bg_path);
            }
        }
        if settings.enable_gradient_background {
            info!("渐变背景: 启用");
        }
        if settings.enable_ground_plane {
            info!("地面平面: 启用");
        }

        render_single_frame(&mut scene, &mut renderer, &settings, &settings.output).map_err(
            |e| {
                error!("单帧渲染失败: {}", e);
                "单帧渲染失败".to_string()
            },
        )?;
    }

    info!("总执行时间：{:?}", start_time.elapsed());
    Ok(())
}
