use crate::core::renderer::Renderer;
use crate::scene::scene_utils::Scene;
use crate::scene::scene_object::{TransformOperations, Transformable};
use crate::io::args::Args;
use crate::utils::render_utils::{create_render_config, render_single_frame};
use nalgebra::{Matrix4, Vector3};
use std::time::Instant;

/// 更新场景对象动画
///
/// 为场景中的非主对象应用动画效果，包括旋转、缩放和位移
///
/// # 参数
/// * `scene` - 要更新的场景
/// * `frame_num` - 当前动画帧号
/// * `rotation_increment` - 旋转增量（角度）
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

/// 运行动画循环
///
/// 负责逐帧更新场景、相机位置并渲染、保存每一帧
///
/// # 参数
/// * `args` - 命令行参数引用
/// * `scene` - 要渲染的场景引用
/// * `renderer` - 渲染器引用
///
/// # 返回值
/// Result，成功为()，失败为包含错误信息的字符串
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
