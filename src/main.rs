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
use io::render_settings::RenderSettings;
use utils::render_utils::{render_single_frame, run_animation_loop};

fn main() -> Result<(), String> {
    // 🔥 **统一使用 default() - 自动处理命令行参数和光源**
    let settings = RenderSettings::default();

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

    // 确保输出目录存在
    fs::create_dir_all(&settings.output_dir)
        .map_err(|e| format!("创建输出目录 '{}' 失败：{}", settings.output_dir, e))?;

    // 🔥 **使用新的资源验证方法 - 预验证所有资源**
    println!("验证所有资源...");
    if let Err(e) = ModelLoader::validate_resources(&settings) {
        println!("资源验证问题: {}", e);
        // 继续执行，不中断渲染过程（允许部分资源缺失）
    }

    // 🔥 **使用新的 ModelLoader 加载模型和创建场景**
    let (mut scene, _model_data) = ModelLoader::load_and_create_scene(obj_path, &settings)?;

    // --- 创建渲染器 ---
    let mut renderer = Renderer::new(settings.width, settings.height);

    // --- 渲染动画或单帧 ---
    if settings.animate {
        run_animation_loop(&mut scene, &mut renderer, &settings)?;
    } else {
        println!("--- 准备单帧渲染 ---");

        // 打印配置摘要
        println!("--- 渲染配置摘要 ---");
        println!("分辨率: {}x{}", settings.width, settings.height);
        println!("投影类型: {}", settings.projection);
        println!("使用光照: {}", settings.use_lighting);
        println!("光源数量: {}", settings.lights.len()); // 🔥 **现在总是>=1**

        // 🔥 **添加背景配置信息**
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

        println!("使用{}", settings.get_lighting_description());
        render_single_frame(&mut scene, &mut renderer, &settings, &settings.output)?;
    }

    println!("总执行时间：{:?}", start_time.elapsed());
    Ok(())
}
