use crate::core::renderer::Renderer;
use crate::io::render_settings::{
    AnimationType, RenderSettings, RotationAxis, get_animation_axis_vector,
};
use crate::scene::scene_utils::Scene;
use crate::utils::save_utils::save_render_with_settings;
use nalgebra::Vector3;
use std::time::Instant;

const BASE_SPEED: f32 = 60.0; // 1s旋转60度

/// 渲染单帧并保存结果（精简版本）
///
/// # 参数
/// * `scene` - 场景引用
/// * `renderer` - 渲染器可变引用 🔥 **修复：改为可变引用**
/// * `settings` - 渲染设置引用
/// * `output_name` - 输出文件名
///
/// # 返回值
/// Result，成功为()，失败为包含错误信息的字符串
pub fn render_single_frame(
    scene: &mut Scene,
    renderer: &mut Renderer, // 🔥 **修复：改为 &mut Renderer**
    settings: &RenderSettings,
    output_name: &str,
) -> Result<(), String> {
    let frame_start_time = Instant::now();
    println!("渲染帧: {}", output_name);

    // 直接渲染场景，无需额外同步
    renderer.render_scene(scene, settings);

    // 保存输出图像
    println!("保存 {} 的输出图像...", output_name);
    save_render_with_settings(renderer, settings, Some(output_name))?;

    println!(
        "帧 {} 渲染完成，耗时 {:?}",
        output_name,
        frame_start_time.elapsed()
    );
    Ok(())
}

/// 执行单个步骤的场景动画（精简版本）
///
/// 根据指定的动画类型、旋转轴和角度增量更新场景
///
/// # 参数
/// * `scene` - 要更新的场景
/// * `animation_type` - 动画类型
/// * `rotation_axis` - 旋转轴向量
/// * `rotation_delta_rad` - 旋转角度增量（弧度）
pub fn animate_scene_step(
    scene: &mut Scene,
    animation_type: &AnimationType,
    rotation_axis: &Vector3<f32>,
    rotation_delta_rad: f32,
) {
    match animation_type {
        AnimationType::CameraOrbit => {
            let mut camera = scene.active_camera.clone();
            camera.orbit(rotation_axis, rotation_delta_rad);
            scene.set_camera(camera);
        }
        AnimationType::ObjectLocalRotation => {
            scene.object.rotate(rotation_axis, rotation_delta_rad);
        }
        AnimationType::None => { /* 无动画 */ }
    }
}

/// 计算旋转增量的辅助函数
///
/// 根据速度系数和时间增量计算旋转角度
///
/// # 参数
/// * `rotation_speed` - 旋转速度系数
/// * `dt` - 时间增量（秒）
///
/// # 返回值
/// 旋转角度增量（弧度）
pub fn calculate_rotation_delta(rotation_speed: f32, dt: f32) -> f32 {
    (rotation_speed * dt * BASE_SPEED).to_radians()
}

/// 计算有效旋转速度及旋转周期
///
/// 确保旋转速度不会太小，并计算完成一圈所需的时间和帧数
///
/// # 参数
/// * `rotation_speed` - 原始旋转速度系数
/// * `fps` - 每秒帧数
///
/// # 返回值
/// (有效旋转速度（度/秒），每圈秒数，每圈帧数)
pub fn calculate_rotation_parameters(rotation_speed: f32, fps: usize) -> (f32, f32, usize) {
    // 计算有效旋转速度 (度/秒)
    let mut effective_rotation_speed_dps = rotation_speed * BASE_SPEED;

    // 确保旋转速度不会太小
    if effective_rotation_speed_dps.abs() < 0.001 {
        effective_rotation_speed_dps = 0.1_f32.copysign(rotation_speed.signum());
        if effective_rotation_speed_dps == 0.0 {
            effective_rotation_speed_dps = 0.1;
        }
    }

    // 计算完成一圈需要的秒数
    let seconds_per_rotation = 360.0 / effective_rotation_speed_dps.abs();

    // 计算一圈需要的帧数
    let frames_for_one_rotation = (seconds_per_rotation * fps as f32).ceil() as usize;

    (
        effective_rotation_speed_dps,
        seconds_per_rotation,
        frames_for_one_rotation,
    )
}

/// 执行完整的动画渲染循环（精简版本）
///
/// # 参数
/// * `scene` - 场景引用
/// * `renderer` - 渲染器可变引用 🔥 **修复：改为可变引用**
/// * `settings` - 渲染设置引用
///
/// # 返回值
/// Result，成功为()，失败为包含错误信息的字符串
pub fn run_animation_loop(
    scene: &mut Scene,
    renderer: &mut Renderer, // 🔥 **修复：改为 &mut Renderer**
    settings: &RenderSettings,
) -> Result<(), String> {
    // 使用通用函数计算旋转参数
    let (effective_rotation_speed_dps, _, frames_to_render) =
        calculate_rotation_parameters(settings.rotation_speed, settings.fps);

    // 根据用户要求的旋转圈数计算实际帧数
    let total_frames = (frames_to_render as f32 * settings.rotation_cycles) as usize;

    println!(
        "开始动画渲染 ({} 帧, {:.2} 秒)...",
        total_frames,
        total_frames as f32 / settings.fps as f32
    );
    println!(
        "动画类型: {:?}, 旋转轴类型: {:?}, 速度: {:.1}度/秒",
        settings.animation_type, settings.rotation_axis, effective_rotation_speed_dps
    );

    // 计算旋转方向
    let rotation_axis_vec = get_animation_axis_vector(settings);
    if settings.rotation_axis == RotationAxis::Custom {
        println!("自定义旋转轴: {:?}", rotation_axis_vec);
    }

    // 计算每帧的旋转角度
    let rotation_per_frame_rad =
        (360.0 / frames_to_render as f32).to_radians() * settings.rotation_speed.signum();

    // 渲染所有帧
    for frame_num in 0..total_frames {
        let frame_start_time = Instant::now();
        println!("--- 准备帧 {} / {} ---", frame_num + 1, total_frames);

        // 第一帧通常不旋转，保留原始状态
        if frame_num > 0 {
            animate_scene_step(
                scene,
                &settings.animation_type,
                &rotation_axis_vec,
                rotation_per_frame_rad,
            );
        }

        // 渲染和保存当前帧
        let frame_output_name = format!("frame_{:03}", frame_num);
        render_single_frame(scene, renderer, settings, &frame_output_name)?;

        println!(
            "帧 {} 渲染完成，耗时 {:?}",
            frame_output_name,
            frame_start_time.elapsed()
        );
    }

    println!(
        "动画渲染完成。总时长：{:.2}秒",
        total_frames as f32 / settings.fps as f32
    );
    Ok(())
}
