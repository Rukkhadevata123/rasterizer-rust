use crate::core::render_config::{RenderConfig, create_render_config};
use crate::core::renderer::Renderer;
use crate::io::args::{AnimationType, Args, get_animation_axis_vector}; // 导入 AnimationType 和 get_animation_axis_vector
use crate::scene::scene_object::Transformable;
use crate::scene::scene_utils::Scene;
use crate::utils::save_utils::save_render_with_args;
use nalgebra::Vector3;
use std::time::Instant;

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
    renderer: &Renderer,
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

/// 使场景中的所有对象围绕其局部自定义轴旋转。
fn animate_objects_local_rotation(
    scene: &mut Scene,
    rotation_axis: &Vector3<f32>,
    rotation_increment_rad: f32,
) {
    for object in scene.objects.iter_mut() {
        // SceneObject 实现了 Transformable, Transformable 提供了 rotate 方法
        object.rotate(rotation_axis, rotation_increment_rad);
    }
}

/// 使相机围绕场景的世界自定义轴旋转（轨道动画）。
fn animate_camera_orbit(
    scene: &mut Scene,
    rotation_axis: &Vector3<f32>,
    rotation_increment_rad: f32,
) {
    let mut camera = scene.active_camera.clone();
    // 直接使用 Camera 结构中已有的 orbit 方法
    camera.orbit(rotation_axis, rotation_increment_rad);
    scene.set_camera(camera);
}

pub fn run_animation_loop(
    args: &Args,
    scene: &mut Scene,
    renderer: &Renderer,
) -> Result<(), String> {
    let total_frames = args.total_frames;
    println!("开始动画渲染 ({} 帧)...", total_frames);
    println!(
        "动画类型: {:?}, 旋转轴类型: {:?}",
        args.animation_type, args.rotation_axis
    );

    let rotation_axis_vec = get_animation_axis_vector(args);
    if args.rotation_axis == crate::io::args::RotationAxis::Custom {
        println!("自定义旋转轴: {:?}", rotation_axis_vec);
    }

    let rotation_per_frame_deg = 360.0 / total_frames as f32;
    let rotation_per_frame_rad = rotation_per_frame_deg.to_radians();

    for frame_num in 0..total_frames {
        let frame_start_time = Instant::now();
        println!("--- 准备帧 {} ---", frame_num);

        if frame_num > 0 {
            // 第一帧是初始状态
            match args.animation_type {
                AnimationType::CameraOrbit => {
                    animate_camera_orbit(scene, &rotation_axis_vec, rotation_per_frame_rad);
                }
                AnimationType::ObjectLocalRotation => {
                    animate_objects_local_rotation(
                        scene,
                        &rotation_axis_vec,
                        rotation_per_frame_rad,
                    );
                }
                AnimationType::None => {
                    // 无动画
                }
            }
        }

        let config = create_render_config(scene, args);
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
