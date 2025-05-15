use clap::Parser;
use nalgebra::{Matrix4, Vector3}; // 保留Point3因为在函数参数中使用
use std::fs;
use std::path::Path;
use std::time::Instant;

// 声明模块
mod args;
mod camera;
mod color_utils;
mod interpolation;
mod loaders;
mod material_system;
mod model_types;
mod rasterizer;
mod renderer;
mod scene;
mod scene_object;
mod texture_utils;
mod transform;
mod utils;

// 导入语句
use args::{Args, parse_point3, parse_vec3};
use camera::Camera;
use color_utils::apply_colormap_jet;
use loaders::load_obj_enhanced;
use material_system::Light;
use model_types::ModelData;
use renderer::{RenderConfig, Renderer};
use scene::Scene;
use scene_object::{SceneObject, TransformOperations, Transformable};
use utils::{normalize_and_center_model, normalize_depth, save_image};

/// 创建场景相机
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

/// 应用PBR材质参数
fn apply_pbr_parameters(model_data: &mut ModelData, args: &Args) {
    if !args.use_pbr {
        return;
    }

    for material in &mut model_data.materials {
        // material.sync_with_mode(model_types::MaterialMode::PBR); // 已同步，无需ensure_pbr_material
        material.metallic = args.metallic;
        material.roughness = args.roughness;
        material.ambient_occlusion = args.ambient_occlusion;

        // 解析并设置基础颜色
        if let Ok(base_color) = parse_vec3(&args.base_color) {
            material.base_color = base_color;
        } else {
            println!(
                "警告: 无法解析基础颜色, 使用默认值: {:?}",
                material.base_color
            );
        }

        // 解析并设置自发光颜色
        if let Ok(emissive) = parse_vec3(&args.emissive) {
            material.emissive = emissive;
        } else {
            println!(
                "警告: 无法解析自发光颜色, 使用默认值: {:?}",
                material.emissive
            );
        }

        println!(
            "应用PBR材质 - 基础色: {:?}, 金属度: {:.2}, 粗糙度: {:.2}, 环境光遮蔽: {:.2}, 自发光: {:?}",
            material.base_color,
            material.metallic,
            material.roughness,
            material.ambient_occlusion,
            material.emissive
        );
    }
}

/// 设置场景光源
fn setup_lights(scene: &mut Scene, args: &Args) -> Result<(), String> {
    if args.no_lighting {
        // 使用环境光
        let ambient_intensity = if let Ok(ambient_color) = parse_vec3(&args.ambient_color) {
            println!("使用RGB环境光强度: {:?}", ambient_color);
            ambient_color
        } else {
            Vector3::new(args.ambient, args.ambient, args.ambient)
        };

        scene.create_ambient_light(ambient_intensity);
        return Ok(());
    }

    // 使用漫反射光源
    let light_intensity = Vector3::new(1.0, 1.0, 1.0) * args.diffuse;

    match args.light_type.to_lowercase().as_str() {
        "point" => {
            let light_pos =
                parse_point3(&args.light_pos).map_err(|e| format!("无效的光源位置格式: {}", e))?;

            let atten_parts: Vec<Result<f32, _>> = args
                .light_atten
                .split(',')
                .map(|s| s.trim().parse::<f32>())
                .collect();

            if atten_parts.len() != 3 || atten_parts.iter().any(|r| r.is_err()) {
                return Err(format!(
                    "无效的光衰减格式: '{}'. 应为 'c,l,q'",
                    args.light_atten
                ));
            }

            let attenuation = (
                atten_parts[0].as_ref().map_or(0.0, |v| *v).max(0.0),
                atten_parts[1].as_ref().map_or(0.0, |v| *v).max(0.0),
                atten_parts[2].as_ref().map_or(0.0, |v| *v).max(0.0),
            );

            println!(
                "使用点光源，位置: {:?}, 强度系数: {:.2}, 衰减: {:?}",
                light_pos, args.diffuse, attenuation
            );
            scene.create_point_light(light_pos, light_intensity, attenuation);
        }
        _ => {
            // 默认为定向光
            let mut light_dir =
                parse_vec3(&args.light_dir).map_err(|e| format!("无效的光源方向格式: {}", e))?;
            light_dir = -light_dir.normalize(); // 朝向光源的方向

            println!(
                "使用定向光，方向: {:?}, 强度系数: {:.2}",
                light_dir, args.diffuse
            );
            scene.create_directional_light(light_dir, light_intensity);
        }
    }

    Ok(())
}

/// 创建并设置场景
fn setup_scene(mut model_data: ModelData, args: &Args) -> Result<Scene, String> {
    // 创建相机
    let camera = create_camera(args)?;

    // 创建场景并设置相机
    let mut scene = Scene::new(camera);

    // 应用PBR材质参数(如果需要)
    apply_pbr_parameters(&mut model_data, args);

    // 添加模型和主对象
    let model_id = scene.add_model(model_data);
    let main_object = SceneObject::new_default(model_id);
    scene.add_object(main_object, Some("main"));

    // 添加多个对象实例(如果需要)
    if let Some(count_str) = &args.object_count {
        if let Ok(count) = count_str.parse::<usize>() {
            if count > 1 {
                // 创建环形对象阵列
                let radius = 2.0;
                scene.create_object_ring(model_id, count - 1, radius, Some("satellite"));
                println!("创建了环形排列的 {} 个附加对象", count - 1);
            }
        }
    }

    // 设置光照
    setup_lights(&mut scene, args)?;

    Ok(scene)
}

/// 创建渲染配置
fn create_render_config(scene: &Scene, args: &Args) -> RenderConfig {
    let light = scene
        .lights
        .first()
        .cloned() // 使用clone而不是to_light_enum
        .unwrap_or_else(|| Light::Ambient(Vector3::new(args.ambient, args.ambient, args.ambient)));

    RenderConfig::default()
        .with_projection(&args.projection)
        .with_zbuffer(!args.no_zbuffer)
        .with_face_colors(args.colorize)
        .with_texture(!args.no_texture)
        .with_light(light)
        .with_lighting(!args.no_lighting)
        .with_phong(args.use_phong)
        .with_gamma_correction(!args.no_gamma)
        .with_pbr(args.use_pbr)
        .with_backface_culling(args.backface_culling)
        .with_wireframe(args.wireframe)
        .with_multithreading(!args.no_multithreading)
        .with_small_triangle_culling(args.cull_small_triangles, args.min_triangle_area)
}

/// 保存渲染结果
fn save_render_result(
    renderer: &Renderer,
    args: &Args,
    config: &RenderConfig,
    output_name: &str,
) -> Result<(), String> {
    // 保存彩色图像
    let color_data = renderer.frame_buffer.get_color_buffer_bytes();
    let color_path = Path::new(&args.output_dir)
        .join(format!("{}_color.png", output_name))
        .to_str()
        .ok_or("创建彩色输出路径字符串失败")?
        .to_string();

    save_image(
        &color_path,
        &color_data,
        args.width as u32,
        args.height as u32,
    );

    // 保存深度图（如果启用）
    if config.use_zbuffer && !args.no_depth {
        let depth_data_raw = renderer.frame_buffer.get_depth_buffer_f32();
        let depth_normalized = normalize_depth(&depth_data_raw, 1.0, 99.0);
        let depth_colored = apply_colormap_jet(
            &depth_normalized
                .iter()
                .map(|&d| 1.0 - d) // 反转：越近 = 越热
                .collect::<Vec<_>>(),
            args.width,
            args.height,
            config.apply_gamma_correction,
        );

        let depth_path = Path::new(&args.output_dir)
            .join(format!("{}_depth.png", output_name))
            .to_str()
            .ok_or("创建深度输出路径字符串失败")?
            .to_string();

        save_image(
            &depth_path,
            &depth_colored,
            args.width as u32,
            args.height as u32,
        );
    }

    Ok(())
}

/// 渲染单帧
fn render_single_frame(
    args: &Args,
    scene: &Scene,
    renderer: &Renderer,
    config: &RenderConfig,
    output_name: &str,
) -> Result<(), String> {
    let frame_start_time = Instant::now();
    println!("渲染帧: {}", output_name);

    // 渲染场景
    renderer.render_scene(scene, config);

    // 保存输出图像
    println!("保存 {} 的输出图像...", output_name);
    save_render_result(renderer, args, config, output_name)?;

    // 打印材质信息（调试用）
    if let Some(model) = scene.models.first() {
        for (i, material) in model.materials.iter().enumerate() {
            if i == 0 || !material.is_opaque() {
                println!("材质 #{}: {}", i, material.get_name());
                println!("  漫反射颜色: {:?}", material.diffuse);

                if !material.is_opaque() {
                    println!("  透明度: {:.2}", material.get_opacity());
                }
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

/// 更新场景对象动画
fn update_scene_objects(scene: &mut Scene, frame_num: usize, rotation_increment: f32) {
    // 只对非主对象应用动画效果
    for (i, object) in scene.objects.iter_mut().enumerate() {
        if i > 0 {
            // 使用全局Y轴旋转
            let object_rotation_increment = rotation_increment * 0.5; // 调整速度
            object.rotate_global_y(object_rotation_increment.to_radians());

            // 使用一致的周期作为缩放变化的基础
            let normalized_phase = (frame_num as f32 * rotation_increment).to_radians();
            let scale_factor = 0.9 + 0.1 * normalized_phase.sin().abs();

            // 重置变换矩阵以避免累积效应
            object.set_transform(Matrix4::identity());
            object.scale_global(&Vector3::new(scale_factor, scale_factor, scale_factor));

            // 小幅上下移动，与旋转周期协调
            let y_offset = 0.03 * normalized_phase.sin();
            object.translate(&Vector3::new(0.0, y_offset, 0.0));
        }
    }
}

/// 运行动画循环
fn run_animation_loop(args: &Args, scene: &mut Scene, renderer: &Renderer) -> Result<(), String> {
    let total_frames = args.total_frames;
    println!("开始动画渲染 ({} 帧)...", total_frames);

    // 计算每帧旋转增量（角度）
    let rotation_increment = 360.0 / total_frames as f32;

    for frame_num in 0..total_frames {
        let frame_start_time = Instant::now();
        println!("--- 准备帧 {} ---", frame_num);

        // 更新场景状态
        if frame_num > 0 {
            // 更新相机位置 - 使用相机内置的orbit_y方法（接受角度参数）
            let mut camera = scene.active_camera.clone();
            camera.orbit_y(rotation_increment); // 直接使用角度值
            scene.set_camera(camera);

            // 更新场景对象
            update_scene_objects(scene, frame_num, rotation_increment);
        }

        // 创建渲染配置
        let config = create_render_config(scene, args);

        // 渲染并保存当前帧
        let frame_output_name = format!("frame_{:03}", frame_num);
        render_single_frame(args, scene, renderer, &config, &frame_output_name)?;

        println!(
            "帧 {} 渲染完成，耗时 {:?}",
            frame_output_name,
            frame_start_time.elapsed()
        );
    }

    println!("动画渲染完成。");
    Ok(())
}

/// 测试变换API和未被正式使用的方法
/// 这个函数不会被正常的程序流调用，仅用于验证API完整性
#[allow(dead_code)]
fn test_transformation_api(obj_path: &str) -> Result<(), String> {
    println!("测试变换API和未使用的方法...");
    let test_start = Instant::now();
    
    // --- 加载模型 ---
    let args = Args::parse_from(vec![
        "rasterizer", 
        "--obj", obj_path,
        "--width", "800",
        "--height", "600"
    ]);
    
    let mut model_data = load_obj_enhanced(obj_path, &args)?;
    normalize_and_center_model(&mut model_data);
    
    // --- 测试相机操作 ---
    println!("测试相机变换方法...");
    let mut camera = Camera::new(
        nalgebra::Point3::new(0.0, 0.0, 3.0),
        nalgebra::Point3::new(0.0, 0.0, 0.0),
        nalgebra::Vector3::new(0.0, 1.0, 0.0),
        45.0,
        1.333,
        0.1,
        100.0
    );
    
    // 测试Camera中未使用的方法
    camera.pan(0.1, 0.2);  // 测试相机平移
    camera.dolly(-0.5);    // 测试相机前后移动
    camera.set_fov(60.0);  // 测试设置视场角
    camera.set_aspect_ratio(1.5); // 测试设置宽高比
    
    // --- 测试场景对象变换 ---
    println!("测试场景对象变换方法...");
    let mut scene = Scene::new(camera.clone());
    let model_id = scene.add_model(model_data);
    
    // 创建一个使用with_transform的对象
    let transform = transform::TransformFactory::translation(&nalgebra::Vector3::new(1.0, 0.0, 0.0));
    let mut obj1 = SceneObject::with_transform(model_id, transform, None);
    
    // 使用SceneObject的set_position方法
    obj1.set_position(nalgebra::Point3::new(0.5, 0.5, 0.0));
    
    // 使用Transformable的get_transform方法
    let current_transform = obj1.get_transform();
    println!("当前变换矩阵: {:?}", current_transform);
    
    // 使用Transformable的apply_local方法
    let local_transform = transform::TransformFactory::rotation_x(45.0_f32.to_radians());
    obj1.apply_local(local_transform);
    
    // 测试TransformOperations中未使用的方法
    obj1.translate_local(&nalgebra::Vector3::new(0.1, 0.2, 0.3));
    obj1.rotate_local(&nalgebra::Vector3::new(0.0, 1.0, 0.0), 30.0_f32.to_radians());
    obj1.rotate_global(&nalgebra::Vector3::new(1.0, 0.0, 0.0), 45.0_f32.to_radians());
    obj1.rotate_local_x(15.0_f32.to_radians());
    obj1.rotate_local_y(15.0_f32.to_radians());
    obj1.rotate_local_z(15.0_f32.to_radians());
    obj1.rotate_global_x(15.0_f32.to_radians());
    obj1.rotate_global_z(15.0_f32.to_radians());
    obj1.scale_local(&nalgebra::Vector3::new(1.2, 1.2, 1.2));
    obj1.scale_local_uniform(1.5);
    obj1.scale_global_uniform(0.8);
    
    // 添加对象到场景
    scene.add_object(obj1, Some("test_object"));
    
    // --- 测试TransformFactory中未使用的方法 ---
    println!("测试TransformFactory未使用的方法...");
    let _rotation_x = transform::TransformFactory::rotation_x(30.0_f32.to_radians());
    let _rotation_z = transform::TransformFactory::rotation_z(30.0_f32.to_radians());
    let _scaling = transform::TransformFactory::scaling(1.5);
    
    // --- 测试坐标变换函数 ---
    println!("测试坐标变换函数...");
    let test_points = vec![
        nalgebra::Point3::new(0.0, 0.0, 0.0),
        nalgebra::Point3::new(1.0, 0.0, 0.0),
        nalgebra::Point3::new(0.0, 1.0, 0.0),
    ];
    
    // 测试world_to_screen函数 - 修复：使用view_projection_matrix作为字段而非方法
    let view_proj = &camera.view_projection_matrix;
    let screen_points = transform::world_to_screen(&test_points, view_proj, 800.0, 600.0);
    println!("世界坐标点转换为屏幕坐标: {:?}", screen_points);
    
    println!("变换API测试完成，耗时 {:?}", test_start.elapsed());
    println!("所有方法均正常工作");
    
    Ok(())
}

// 在main函数中添加对test_transformation_api的条件调用
fn main() -> Result<(), String> {
    let args = Args::parse();
    
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
        let config = create_render_config(&scene, &args);
        println!("使用{}渲染", config.get_lighting_description());
        render_single_frame(&args, &scene, &renderer, &config, &args.output)?;
    }

    println!("总执行时间：{:?}", start_time.elapsed());
    Ok(())
}
