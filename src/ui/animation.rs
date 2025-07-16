use crate::core::renderer::Renderer;
use crate::io::render_settings::{AnimationType, RenderSettings, get_animation_axis_vector};
use crate::scene::scene_utils::Scene;
use crate::utils::render_utils::{
    animate_scene_step, calculate_rotation_delta, calculate_rotation_parameters,
};
use crate::utils::save_utils::save_image;
use egui::{ColorImage, Context, TextureOptions};
use log::debug;
use std::fs;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use super::app::RasterizerApp;
use super::core::CoreMethods;

/// 将ColorImage转换为PNG数据
pub fn frame_to_png_data(image: &ColorImage) -> Vec<u8> {
    // ColorImage是RGBA格式，我们需要转换为RGB格式
    let mut rgb_data = Vec::with_capacity(image.width() * image.height() * 3);
    for pixel in &image.pixels {
        rgb_data.push(pixel.r());
        rgb_data.push(pixel.g());
        rgb_data.push(pixel.b());
    }
    rgb_data
}

/// 渲染一圈的动画帧
///
/// # 参数
/// * `scene_copy` - 场景的克隆
/// * `settings` - 渲染参数
/// * `progress_arc` - 进度计数器
/// * `ctx_clone` - UI上下文，用于更新界面
/// * `width` - 渲染宽度
/// * `height` - 渲染高度
/// * `on_frame_rendered` - 帧渲染完成后的回调函数，参数为(帧序号, RGB颜色数据)
///
/// # 返回值
/// 渲染的总帧数
fn render_one_rotation_cycle<F>(
    mut scene_copy: Scene,
    settings: &RenderSettings,
    progress_arc: &Arc<AtomicUsize>,
    ctx_clone: &Context,
    width: usize,
    height: usize,
    mut on_frame_rendered: F,
) -> usize
where
    F: FnMut(usize, Vec<u8>),
{
    let mut thread_renderer = Renderer::new(width, height);
    let (effective_rotation_speed_dps, _, frames_to_render) =
        calculate_rotation_parameters(settings.rotation_speed, settings.fps);

    let rotation_axis_vec = get_animation_axis_vector(settings);
    let rotation_increment_rad_per_frame =
        (360.0 / frames_to_render as f32).to_radians() * effective_rotation_speed_dps.signum();

    for frame_num in 0..frames_to_render {
        progress_arc.store(frame_num, Ordering::SeqCst);

        if frame_num > 0 {
            animate_scene_step(
                &mut scene_copy,
                &settings.animation_type,
                &rotation_axis_vec,
                rotation_increment_rad_per_frame,
            );
        }

        // === 缓存失效策略 ===
        match settings.animation_type {
            AnimationType::CameraOrbit => {
                // 相机轨道动画：地面本体和阴影都依赖相机，必须全部失效
                thread_renderer.frame_buffer.invalidate_ground_base_cache();
                thread_renderer
                    .frame_buffer
                    .invalidate_ground_shadow_cache();
            }
            AnimationType::ObjectLocalRotation => {
                if settings.enable_shadow_mapping {
                    // 物体动画+阴影：只需失效阴影缓存，地面本体可复用
                    thread_renderer
                        .frame_buffer
                        .invalidate_ground_shadow_cache();
                }
                // 未开阴影时，地面缓存可复用，无需清理
            }
            AnimationType::None => {}
        }

        thread_renderer.render_scene(&mut scene_copy, settings);
        let color_data_rgb = thread_renderer.frame_buffer.get_color_buffer_bytes();
        on_frame_rendered(frame_num, color_data_rgb);

        if frame_num % (frames_to_render.max(1) / 20).max(1) == 0 {
            ctx_clone.request_repaint();
        }
    }

    progress_arc.store(frames_to_render, Ordering::SeqCst);
    ctx_clone.request_repaint();

    frames_to_render
}

/// 动画与视频生成相关方法的特质
pub trait AnimationMethods {
    /// 执行实时渲染循环
    fn perform_realtime_rendering(&mut self, ctx: &Context);

    /// 在后台生成视频
    fn start_video_generation(&mut self, ctx: &Context);

    /// 启动预渲染过程
    fn start_pre_rendering(&mut self, ctx: &Context);

    /// 处理预渲染帧
    fn handle_pre_rendering_tasks(&mut self, ctx: &Context);

    /// 播放预渲染帧
    fn play_pre_rendered_frames(&mut self, ctx: &Context);
}

impl AnimationMethods for RasterizerApp {
    /// 执行实时渲染循环
    fn perform_realtime_rendering(&mut self, ctx: &Context) {
        // 如果启用了预渲染模式且没有预渲染帧，才进入预渲染
        if self.pre_render_mode
            && !self.is_pre_rendering
            && self.pre_rendered_frames.lock().unwrap().is_empty()
        {
            // 检查模型是否已加载
            if self.scene.is_none() {
                let obj_path = match &self.settings.obj {
                    Some(path) => path.clone(),
                    None => {
                        self.set_error("错误: 未指定OBJ文件路径".to_string());
                        self.stop_animation_rendering();
                        return;
                    }
                };
                match crate::io::model_loader::ModelLoader::load_and_create_scene(
                    &obj_path,
                    &self.settings,
                ) {
                    Ok((scene, model_data)) => {
                        self.scene = Some(scene);
                        self.model_data = Some(model_data);
                        self.start_pre_rendering(ctx);
                        return;
                    }
                    Err(e) => {
                        self.set_error(format!("加载模型失败: {e}"));
                        self.stop_animation_rendering();
                        return;
                    }
                }
            } else {
                self.start_pre_rendering(ctx);
                return;
            }
        }

        // 如果正在预渲染，处理预渲染任务
        if self.is_pre_rendering {
            self.handle_pre_rendering_tasks(ctx);
            return;
        }

        // 如果启用预渲染模式且有预渲染帧，播放预渲染帧
        if self.pre_render_mode && !self.pre_rendered_frames.lock().unwrap().is_empty() {
            self.play_pre_rendered_frames(ctx);
            return;
        }

        // === 常规实时渲染（未启用预渲染模式或预渲染复选框未勾选） ===

        // 确保场景已加载
        if self.scene.is_none() {
            let obj_path = match &self.settings.obj {
                Some(path) => path.clone(),
                None => {
                    self.set_error("错误: 未指定OBJ文件路径".to_string());
                    self.stop_animation_rendering();
                    return;
                }
            };
            match crate::io::model_loader::ModelLoader::load_and_create_scene(
                &obj_path,
                &self.settings,
            ) {
                Ok((scene, model_data)) => {
                    self.scene = Some(scene);
                    self.model_data = Some(model_data);
                    // 注意：这里不再自动跳转到预渲染，而是继续执行实时渲染
                    self.status_message = "模型加载成功，开始实时渲染...".to_string();
                }
                Err(e) => {
                    self.set_error(format!("加载模型失败: {e}"));
                    self.stop_animation_rendering();
                    return;
                }
            }
        }

        // 检查渲染器尺寸，但避免不必要的缓存清除
        if self.renderer.frame_buffer.width != self.settings.width
            || self.renderer.frame_buffer.height != self.settings.height
        {
            self.renderer
                .resize(self.settings.width, self.settings.height);
            self.rendered_image = None;
            debug!(
                "重新创建渲染器，尺寸: {}x{}",
                self.settings.width, self.settings.height
            );
        }

        let now = Instant::now();
        let dt = if let Some(last_time) = self.last_frame_time {
            now.duration_since(last_time).as_secs_f32()
        } else {
            1.0 / 60.0 // 默认 dt
        };
        if let Some(last_time) = self.last_frame_time {
            let frame_time = now.duration_since(last_time);
            self.update_fps_stats(frame_time);
        }
        self.last_frame_time = Some(now);

        if self.is_realtime_rendering && self.settings.rotation_speed.abs() < 0.01 {
            self.settings.rotation_speed = 1.0; // 确保实时渲染时有旋转速度
        }

        self.animation_time += dt;

        if let Some(scene) = &mut self.scene {
            // 动画过程中不清除缓存
            // 物体动画不影响背景和地面（相机不动），所以缓存仍然有效

            // 使用通用函数计算旋转增量
            let rotation_delta_rad = calculate_rotation_delta(self.settings.rotation_speed, dt);
            let rotation_axis_vec = get_animation_axis_vector(&self.settings);

            // 使用通用函数执行动画步骤
            animate_scene_step(
                scene,
                &self.settings.animation_type,
                &rotation_axis_vec,
                rotation_delta_rad,
            );

            debug!(
                "实时渲染中: FPS={:.1}, 动画类型={:?}, 轴={:?}, 旋转速度={}, 角度增量={:.3}rad, Phong={}",
                self.avg_fps,
                self.settings.animation_type,
                self.settings.rotation_axis,
                self.settings.rotation_speed,
                rotation_delta_rad,
                self.settings.use_phong
            );

            match self.settings.animation_type {
                AnimationType::CameraOrbit => {
                    // 相机轨道动画：地面本体和阴影都依赖相机，必须全部失效
                    self.renderer.frame_buffer.invalidate_ground_base_cache();
                    self.renderer.frame_buffer.invalidate_ground_shadow_cache();
                }
                AnimationType::ObjectLocalRotation => {
                    if self.settings.enable_shadow_mapping {
                        // 物体动画+阴影：只需失效阴影缓存，地面本体可复用
                        self.renderer.frame_buffer.invalidate_ground_shadow_cache();
                    }
                    // 未开阴影时，地面缓存可复用，无需清理
                }
                AnimationType::None => {}
            }

            self.renderer.render_scene(scene, &self.settings);
            self.display_render_result(ctx);
            ctx.request_repaint();
        }
    }

    fn start_video_generation(&mut self, ctx: &Context) {
        if !self.ffmpeg_available {
            self.set_error("无法生成视频：未检测到ffmpeg。请安装ffmpeg后重试。".to_string());
            return;
        }
        if self.is_generating_video {
            self.status_message = "视频已在生成中，请等待完成...".to_string();
            return;
        }

        // 使用 CoreMethods 验证参数
        match self.settings.validate() {
            Ok(_) => {
                let output_dir = self.settings.output_dir.clone();
                if let Err(e) = fs::create_dir_all(&output_dir) {
                    self.set_error(format!("创建输出目录失败: {e}"));
                    return;
                }
                let frames_dir = format!(
                    "{}/temp_frames_{}",
                    output_dir,
                    chrono::Utc::now().timestamp_millis()
                );
                if let Err(e) = fs::create_dir_all(&frames_dir) {
                    self.set_error(format!("创建帧目录失败: {e}"));
                    return;
                }

                // 计算旋转参数，获取视频帧数
                let (_, _, frames_per_rotation) =
                    calculate_rotation_parameters(self.settings.rotation_speed, self.settings.fps);

                let total_frames =
                    (frames_per_rotation as f32 * self.settings.rotation_cycles) as usize;

                // 如果场景未加载，尝试加载
                if self.scene.is_none() {
                    let obj_path = match &self.settings.obj {
                        Some(path) => path.clone(),
                        None => {
                            self.set_error("错误: 未指定OBJ文件路径".to_string());
                            return;
                        }
                    };
                    match crate::io::model_loader::ModelLoader::load_and_create_scene(
                        &obj_path,
                        &self.settings,
                    ) {
                        Ok((scene, model_data)) => {
                            self.scene = Some(scene);
                            self.model_data = Some(model_data);
                            self.status_message = "模型加载成功，开始生成视频...".to_string();
                        }
                        Err(e) => {
                            self.set_error(format!("加载模型失败，无法生成视频: {e}"));
                            return;
                        }
                    }
                }

                let settings_for_thread = self.settings.clone();
                let video_progress_arc = self.video_progress.clone();
                let fps = self.settings.fps;
                let scene_clone = self.scene.as_ref().expect("场景已检查").clone();

                // 检查是否有预渲染帧
                let has_pre_rendered_frames = {
                    let frames_guard = self.pre_rendered_frames.lock().unwrap();
                    !frames_guard.is_empty()
                };

                // 如果没有预渲染帧，那么我们需要同时为预渲染缓冲区生成帧
                let frames_for_pre_render = if !has_pre_rendered_frames {
                    Some(self.pre_rendered_frames.clone())
                } else {
                    None
                };

                // 设置渲染状态
                self.is_generating_video = true;
                video_progress_arc.store(0, Ordering::SeqCst);

                // 更新状态消息
                self.status_message = format!(
                    "开始生成视频 (0/{} 帧，{:.1} 秒时长)...",
                    total_frames,
                    total_frames as f32 / fps as f32
                );

                ctx.request_repaint();
                let ctx_clone = ctx.clone();
                let video_filename = format!("{}.mp4", settings_for_thread.output);
                let video_output_path = format!("{output_dir}/{video_filename}");
                let frames_dir_clone = frames_dir.clone();

                // 如果有预渲染帧，复制到线程中
                let pre_rendered_frames_clone = if has_pre_rendered_frames {
                    let frames_guard = self.pre_rendered_frames.lock().unwrap();
                    Some(frames_guard.clone())
                } else {
                    None
                };

                let thread_handle = thread::spawn(move || {
                    let width = settings_for_thread.width;
                    let height = settings_for_thread.height;
                    let mut rendered_frames = Vec::new();

                    // 使用预渲染帧或重新渲染
                    if let Some(frames) = pre_rendered_frames_clone {
                        // 使用预渲染帧
                        let pre_rendered_count = frames.len();

                        for frame_num in 0..total_frames {
                            video_progress_arc.store(frame_num, Ordering::SeqCst);

                            // 计算当前帧在哪个圈和圈内的位置
                            let cycle_position = frame_num % frames_per_rotation;

                            // 将圈内位置映射到预渲染帧索引
                            // 这处理了预渲染帧数量可能与理论帧数不匹配的情况
                            let pre_render_idx =
                                (cycle_position * pre_rendered_count) / frames_per_rotation;

                            let frame = &frames[pre_render_idx.min(pre_rendered_count - 1)]; // 避免越界访问

                            // 将ColorImage转换为PNG并保存
                            let frame_path = format!("{frames_dir_clone}/frame_{frame_num:04}.png");
                            let color_data = frame_to_png_data(frame);
                            save_image(&frame_path, &color_data, width as u32, height as u32);

                            if frame_num % (total_frames.max(1) / 20).max(1) == 0 {
                                ctx_clone.request_repaint();
                            }
                        }
                    } else {
                        // 使用通用渲染函数渲染一圈或部分圈
                        let frames_arc = frames_for_pre_render.clone();

                        let rendered_frame_count = render_one_rotation_cycle(
                            scene_clone,
                            &settings_for_thread,
                            &video_progress_arc,
                            &ctx_clone,
                            width,
                            height,
                            |frame_num, color_data_rgb| {
                                // 保存RGB数据用于后续复用
                                rendered_frames.push(color_data_rgb.clone());

                                // 同时为视频保存PNG文件
                                let frame_path =
                                    format!("{frames_dir_clone}/frame_{frame_num:04}.png");
                                save_image(
                                    &frame_path,
                                    &color_data_rgb,
                                    width as u32,
                                    height as u32,
                                );

                                // 如果需要同时保存到预渲染缓冲区
                                if let Some(ref frames_arc) = frames_arc {
                                    // 转换为RGBA格式以用于预渲染帧
                                    let mut rgba_data = Vec::with_capacity(width * height * 4);
                                    for chunk in color_data_rgb.chunks_exact(3) {
                                        rgba_data.extend_from_slice(chunk);
                                        rgba_data.push(255); // Alpha
                                    }
                                    let color_image = ColorImage::from_rgba_unmultiplied(
                                        [width, height],
                                        &rgba_data,
                                    );
                                    frames_arc.lock().unwrap().push(color_image);
                                }
                            },
                        );

                        // 如果需要多于一圈，使用前面渲染的帧复用
                        if rendered_frame_count < total_frames {
                            for frame_num in rendered_frame_count..total_frames {
                                video_progress_arc.store(frame_num, Ordering::SeqCst);

                                // 复用之前渲染的帧
                                let source_frame_idx = frame_num % rendered_frame_count;
                                let source_data = &rendered_frames[source_frame_idx];

                                // 保存为图片文件
                                let frame_path =
                                    format!("{frames_dir_clone}/frame_{frame_num:04}.png");
                                save_image(&frame_path, source_data, width as u32, height as u32);

                                if frame_num % (total_frames.max(1) / 20).max(1) == 0 {
                                    ctx_clone.request_repaint();
                                }
                            }
                        }
                    }

                    video_progress_arc.store(total_frames, Ordering::SeqCst);
                    ctx_clone.request_repaint();

                    // 使用ffmpeg将帧序列合成为视频，并解决阻塞问题
                    let frames_pattern = format!("{frames_dir_clone}/frame_%04d.png");
                    let ffmpeg_status = std::process::Command::new("ffmpeg")
                        .args([
                            "-y",
                            "-framerate",
                            &fps.to_string(),
                            "-i",
                            &frames_pattern,
                            "-c:v",
                            "libx264",
                            "-pix_fmt",
                            "yuv420p",
                            "-crf",
                            "23",
                            &video_output_path,
                        ])
                        .status();

                    let success = ffmpeg_status.is_ok_and(|s| s.success());

                    // 视频生成后清理临时文件
                    let _ = fs::remove_dir_all(&frames_dir_clone);

                    (success, video_output_path)
                });

                self.video_generation_thread = Some(thread_handle);
            }
            Err(e) => self.set_error(e),
        }
    }

    fn start_pre_rendering(&mut self, ctx: &Context) {
        if self.is_pre_rendering {
            return;
        }

        // 使用 CoreMethods 验证参数
        match self.settings.validate() {
            Ok(_) => {
                if self.scene.is_none() {
                    let obj_path = match &self.settings.obj {
                        Some(path) => path.clone(),
                        None => {
                            self.set_error("错误: 未指定OBJ文件路径".to_string());
                            self.stop_animation_rendering();
                            return;
                        }
                    };
                    match crate::io::model_loader::ModelLoader::load_and_create_scene(
                        &obj_path,
                        &self.settings,
                    ) {
                        Ok((scene, model_data)) => {
                            self.scene = Some(scene);
                            self.model_data = Some(model_data);
                            self.status_message = "模型加载成功，开始预渲染...".to_string();
                        }
                        Err(e) => {
                            self.set_error(format!("加载模型失败，无法预渲染: {e}"));
                            return;
                        }
                    }
                }

                // 使用通用函数计算旋转参数
                let (_, seconds_per_rotation, frames_to_render) =
                    calculate_rotation_parameters(self.settings.rotation_speed, self.settings.fps);

                self.total_frames_for_pre_render_cycle = frames_to_render;

                self.is_pre_rendering = true;
                self.pre_rendered_frames.lock().unwrap().clear();
                self.pre_render_progress.store(0, Ordering::SeqCst);
                self.current_frame_index = 0;

                let settings_for_thread = self.settings.clone();
                let progress_arc = self.pre_render_progress.clone();
                let frames_arc = self.pre_rendered_frames.clone();
                let width = settings_for_thread.width;
                let height = settings_for_thread.height;
                let scene_clone = self.scene.as_ref().expect("场景已检查存在").clone();

                self.status_message = format!(
                    "开始预渲染动画 (0/{frames_to_render} 帧，转一圈需 {seconds_per_rotation:.1} 秒)..."
                );
                ctx.request_repaint();
                let ctx_clone = ctx.clone();

                thread::spawn(move || {
                    // 使用通用渲染函数
                    render_one_rotation_cycle(
                        scene_clone,
                        &settings_for_thread,
                        &progress_arc,
                        &ctx_clone,
                        width,
                        height,
                        |_, color_data_rgb| {
                            // 将RGB数据转换为RGBA并存储为ColorImage
                            let mut rgba_data = Vec::with_capacity(width * height * 4);
                            for chunk in color_data_rgb.chunks_exact(3) {
                                rgba_data.extend_from_slice(chunk);
                                rgba_data.push(255); // Alpha
                            }
                            let color_image =
                                ColorImage::from_rgba_unmultiplied([width, height], &rgba_data);
                            frames_arc.lock().unwrap().push(color_image);
                        },
                    );
                });
            }
            Err(e) => {
                self.set_error(e);
                self.is_pre_rendering = false;
            }
        }
    }

    fn handle_pre_rendering_tasks(&mut self, ctx: &Context) {
        let progress = self.pre_render_progress.load(Ordering::SeqCst);
        let expected_total_frames = self.total_frames_for_pre_render_cycle;

        // 使用通用函数计算参数
        let (_, seconds_per_rotation, _) =
            calculate_rotation_parameters(self.settings.rotation_speed, self.settings.fps);

        self.status_message = format!(
            "预渲染动画中... ({}/{} 帧，{:.1}%，转一圈约需 {:.1} 秒)",
            progress,
            expected_total_frames,
            if expected_total_frames > 0 {
                progress as f32 / expected_total_frames as f32 * 100.0
            } else {
                0.0
            },
            seconds_per_rotation
        );

        if progress >= expected_total_frames && expected_total_frames > 0 {
            self.is_pre_rendering = false;
            let final_frame_count = self.pre_rendered_frames.lock().unwrap().len();
            self.status_message = format!(
                "预渲染完成！已缓存 {} 帧动画 (目标 {} FPS, 转一圈 {:.1} 秒)",
                final_frame_count, self.settings.fps, seconds_per_rotation
            );
            if self.is_realtime_rendering || self.pre_render_mode {
                self.current_frame_index = 0;
                self.last_frame_time = None;
                ctx.request_repaint();
            }
        } else {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
    }

    fn play_pre_rendered_frames(&mut self, ctx: &Context) {
        let frame_to_display_idx;
        let frame_image;
        let frames_len;
        {
            let frames_guard = self.pre_rendered_frames.lock().unwrap();
            frames_len = frames_guard.len();
            if frames_len == 0 {
                self.pre_render_mode = false;
                self.status_message = "预渲染帧丢失或未生成，退出预渲染模式。".to_string();
                ctx.request_repaint();
                return;
            }
            frame_to_display_idx = self.current_frame_index % frames_len;
            frame_image = frames_guard[frame_to_display_idx].clone();
        }

        let now = Instant::now();
        let target_frame_duration = Duration::from_secs_f32(1.0 / self.settings.fps.max(1) as f32);

        if let Some(last_frame_display_time) = self.last_frame_time {
            let time_since_last_display = now.duration_since(last_frame_display_time);
            if time_since_last_display < target_frame_duration {
                let time_to_wait = target_frame_duration - time_since_last_display;
                ctx.request_repaint_after(time_to_wait);
                return;
            }
            self.update_fps_stats(time_since_last_display);
        } else {
            self.update_fps_stats(target_frame_duration);
        }
        self.last_frame_time = Some(now);

        let texture_name = format!("pre_rendered_tex_{frame_to_display_idx}");
        self.rendered_image =
            Some(ctx.load_texture(texture_name, frame_image, TextureOptions::LINEAR));
        self.current_frame_index = (self.current_frame_index + 1) % frames_len;

        // 使用通用函数计算参数
        let (_, seconds_per_rotation, _) =
            calculate_rotation_parameters(self.settings.rotation_speed, self.settings.fps);

        self.status_message = format!(
            "播放预渲染: 帧 {}/{} (目标 {} FPS, 平均 {:.1} FPS, 1圈 {:.1}秒)",
            frame_to_display_idx + 1,
            frames_len,
            self.settings.fps,
            self.avg_fps,
            seconds_per_rotation
        );
        ctx.request_repaint();
    }
}
