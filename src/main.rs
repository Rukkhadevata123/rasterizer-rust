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
    // 🔥 **使用新的CLI处理** - 获取设置和GUI标志
    let (settings, should_start_gui) = SimpleCli::process()?;

    // 判断是否应该启动GUI模式
    if should_start_gui {
        println!("启动GUI模式...");
        if let Err(err) = ui::start_gui(settings) {
            return Err(format!("GUI启动失败: {}", err));
        }
        return Ok(());
    }

    // 无头渲染模式 - 需要OBJ文件
    if settings.obj.is_none() {
        return Err("错误: 无头模式需要指定OBJ文件路径（通过配置文件或示例配置）".to_string());
    }

    let start_time = Instant::now();

    // 获取OBJ文件路径
    let obj_path = settings.obj.as_ref().unwrap();

    // 确保输出目录存在
    fs::create_dir_all(&settings.output_dir)
        .map_err(|e| format!("创建输出目录 '{}' 失败：{}", settings.output_dir, e))?;

    // 🔥 **验证资源** - 在config_loader中统一处理错误
    println!("验证资源...");
    if let Err(e) = ModelLoader::validate_resources(&settings) {
        println!("警告: {}", e);
        // 继续执行，允许部分资源缺失
    }

    // 🔥 **加载模型和创建场景**
    let (mut scene, _model_data) = ModelLoader::load_and_create_scene(obj_path, &settings)?;

    // 创建渲染器
    let mut renderer = Renderer::new(settings.width, settings.height);

    // 渲染动画或单帧
    if settings.animate {
        run_animation_loop(&mut scene, &mut renderer, &settings)?;
    } else {
        println!("--- 开始单帧渲染 ---");

        // 打印配置摘要
        println!("分辨率: {}x{}", settings.width, settings.height);
        println!("投影类型: {}", settings.projection);
        println!(
            "光照: {} ({} 个光源)",
            if settings.use_lighting {
                "启用"
            } else {
                "禁用"
            },
            settings.lights.len()
        );
        println!("材质: {}", settings.get_lighting_description());

        if settings.use_background_image {
            if let Some(bg_path) = &settings.background_image_path {
                println!("背景图片: {}", bg_path);
            }
        }
        if settings.enable_gradient_background {
            println!("渐变背景: 启用");
        }
        if settings.enable_ground_plane {
            println!("地面平面: 启用");
        }

        println!("-------------------");

        render_single_frame(&mut scene, &mut renderer, &settings, &settings.output)?;
    }

    println!("总执行时间：{:?}", start_time.elapsed());
    Ok(())
}
