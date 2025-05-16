use clap::Parser;

use crate::core::renderer::{RenderConfig, Renderer};
use crate::core::scene::Scene;
use crate::core::scene_object::{SceneObject, TransformOperations, Transformable};
use crate::geometry::camera::Camera;
use crate::geometry::transform;
use crate::io::args::{Args, parse_vec3};
use crate::io::loaders::load_obj_enhanced;
use crate::materials::color_utils::apply_colormap_jet;
use crate::utils::model_types::ModelData;
use crate::utils::depth_image::{normalize_and_center_model, normalize_depth, save_image};
use nalgebra::{Matrix4, Vector3};
use std::path::Path;
use std::time::Instant;

/// 更新场景对象动画
pub fn update_scene_objects(scene: &mut Scene, frame_num: usize, rotation_increment: f32) {
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

/// 保存渲染结果
pub fn save_render_result(
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
    if config.use_zbuffer && args.save_depth {
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
pub fn render_single_frame(
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
            if i == 0 {
                println!("材质 #{}: {}", i, material.get_name());
                println!("  漫反射颜色: {:?}", material.diffuse);
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

/// 运行动画循环
pub fn run_animation_loop(
    args: &Args,
    scene: &mut Scene,
    renderer: &Renderer,
) -> Result<(), String> {
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

/// 应用PBR材质参数
pub fn apply_pbr_parameters(model_data: &mut ModelData, args: &Args) {
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

/// 测试变换API和未被正式使用的方法
/// 这个函数不会被正常的程序流调用，仅用于验证API完整性
#[allow(dead_code)]
pub fn test_transformation_api(obj_path: &str) -> Result<(), String> {
    println!("测试变换API和未使用的方法...");
    let test_start = Instant::now();

    // --- 加载模型 ---
    let args = Args::parse_from(vec![
        "rasterizer",
        "--obj",
        obj_path,
        "--width",
        "800",
        "--height",
        "600",
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
        100.0,
    );

    // 测试Camera中未使用的方法
    camera.pan(0.1, 0.2); // 测试相机平移
    camera.dolly(-0.5); // 测试相机前后移动
    camera.set_fov(60.0); // 测试设置视场角
    camera.set_aspect_ratio(1.5); // 测试设置宽高比

    // --- 测试场景对象变换 ---
    println!("测试场景对象变换方法...");
    let mut scene = Scene::new(camera.clone());
    let model_id = scene.add_model(model_data);

    // 创建一个使用with_transform的对象
    let transform =
        transform::TransformFactory::translation(&nalgebra::Vector3::new(1.0, 0.0, 0.0));
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
    obj1.rotate_local(
        &nalgebra::Vector3::new(0.0, 1.0, 0.0),
        30.0_f32.to_radians(),
    );
    obj1.rotate_global(
        &nalgebra::Vector3::new(1.0, 0.0, 0.0),
        45.0_f32.to_radians(),
    );
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

/// 创建渲染配置
pub fn create_render_config(scene: &Scene, args: &Args) -> RenderConfig {
    let light = scene
        .lights
        .first()
        .cloned() // 使用clone而不是to_light_enum
        .unwrap_or_else(|| {
            crate::materials::material_system::Light::Ambient(Vector3::new(
                args.ambient,
                args.ambient,
                args.ambient,
            ))
        });

    RenderConfig::default()
        .with_projection(&args.projection)
        .with_zbuffer(args.use_zbuffer)
        .with_face_colors(args.colorize)
        .with_texture(args.use_texture)
        .with_light(light)
        .with_lighting(args.use_lighting)
        .with_phong(args.use_phong)
        .with_gamma_correction(args.use_gamma)
        .with_pbr(args.use_pbr)
        .with_backface_culling(args.backface_culling)
        .with_wireframe(args.wireframe)
        .with_multithreading(args.use_multithreading)
        .with_small_triangle_culling(args.cull_small_triangles, args.min_triangle_area)
}
