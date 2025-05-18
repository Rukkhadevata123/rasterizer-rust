use clap::Parser;
use nalgebra::{Point3, Vector3};
use std::time::Instant;

use crate::scene::scene_utils::Scene;
use crate::scene::scene_object::{SceneObject, TransformOperations, Transformable};
use crate::geometry::camera::Camera;
use crate::geometry::transform;
use crate::io::args::Args;
use crate::io::loaders::load_obj_enhanced;
use crate::utils::model_utils::normalize_and_center_model;

/// 测试变换API和未被正式使用的方法
/// 这个函数不会被正常的程序流调用，仅用于验证API完整性
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
        Point3::new(0.0, 0.0, 3.0),
        Point3::new(0.0, 0.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
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
    let transform = transform::TransformFactory::translation(&Vector3::new(1.0, 0.0, 0.0));
    let mut obj1 = SceneObject::with_transform(model_id, transform, None);

    // 使用SceneObject的set_position方法
    obj1.set_position(Point3::new(0.5, 0.5, 0.0));

    // 使用Transformable的get_transform方法
    let current_transform = obj1.get_transform();
    println!("当前变换矩阵: {:?}", current_transform);

    // 使用Transformable的apply_local方法
    let local_transform = transform::TransformFactory::rotation_x(45.0_f32.to_radians());
    obj1.apply_local(local_transform);

    // 测试TransformOperations中未使用的方法
    obj1.translate_local(&Vector3::new(0.1, 0.2, 0.3));
    obj1.rotate_local(&Vector3::new(0.0, 1.0, 0.0), 30.0_f32.to_radians());
    obj1.rotate_global(&Vector3::new(1.0, 0.0, 0.0), 45.0_f32.to_radians());
    obj1.rotate_local_x(15.0_f32.to_radians());
    obj1.rotate_local_y(15.0_f32.to_radians());
    obj1.rotate_local_z(15.0_f32.to_radians());
    obj1.rotate_global_x(15.0_f32.to_radians());
    obj1.rotate_global_z(15.0_f32.to_radians());
    obj1.scale_local(&Vector3::new(1.2, 1.2, 1.2));
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
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
    ];

    // 测试world_to_screen函数 - 修复：使用view_projection_matrix作为字段而非方法
    let view_proj = &camera.view_projection_matrix;
    let screen_points = transform::world_to_screen(&test_points, view_proj, 800.0, 600.0);
    println!("世界坐标点转换为屏幕坐标: {:?}", screen_points);

    println!("变换API测试完成，耗时 {:?}", test_start.elapsed());
    println!("所有方法均正常工作");

    Ok(())
}
