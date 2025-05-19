use crate::core::render_config::create_render_config;
use crate::core::renderer::Renderer;
// update_scene_objects 已被删除
use crate::io::args::{AnimationType, RotationAxis, get_animation_axis_vector}; // 移除了未使用的 Args
use crate::scene::scene_object::Transformable; // 确保 Transformable trait 在作用域中
use crate::utils::save_utils::save_image;
use egui::{ColorImage, Context, TextureOptions};
use std::fs;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::{Duration, Instant};

use super::app::RasterizerApp;
use super::render::RenderMethods;

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

    /// 检查参数是否变化，需要重新预渲染
    fn check_animation_params_changed(&mut self, ctx: &Context) -> bool;
}

impl AnimationMethods for RasterizerApp {
    /// 执行实时渲染循环
    fn perform_realtime_rendering(&mut self, ctx: &egui::Context) {
        // 首先检查参数是否有变化，需要重新预渲染
        if self.pre_render_mode
            && !self.pre_rendered_frames.lock().unwrap().is_empty()
            && !self.is_pre_rendering
            && self.check_animation_params_changed(ctx)
        {
            self.start_pre_rendering(ctx);
            return;
        }

        if self.pre_render_mode
            && !self.is_pre_rendering
            && self.pre_rendered_frames.lock().unwrap().is_empty()
        {
            if self.rotation_speed.abs() < 0.001 {
                self.rotation_speed = 0.1; // 确保有最小旋转速度以启动预渲染
            }
            self.start_pre_rendering(ctx);
            return;
        }

        if self.is_pre_rendering {
            self.handle_pre_rendering_tasks(ctx);
            return;
        }

        if self.pre_render_mode && !self.pre_rendered_frames.lock().unwrap().is_empty() {
            self.play_pre_rendered_frames(ctx);
            return;
        }

        // --- 常规实时渲染 ---
        if self.scene.is_none() {
            let obj_path = self.args.obj.clone(); // self.args 用于获取路径
            match self.load_model(&obj_path) {
                Ok(_) => {
                    self.status_message = "模型加载成功，开始实时渲染...".to_string();
                }
                Err(e) => {
                    self.set_error(format!("加载模型失败: {}", e));
                    self.is_realtime_rendering = false;
                    return;
                }
            }
        }

        if self.renderer.frame_buffer.width != self.args.width // self.args 用于检查尺寸
            || self.renderer.frame_buffer.height != self.args.height
        {
            self.renderer = Renderer::new(self.args.width, self.args.height);
            self.rendered_image = None;
            println!(
                "重新创建渲染器，尺寸: {}x{}",
                self.args.width, self.args.height
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

        if self.is_realtime_rendering && self.rotation_speed.abs() < 0.01 {
            self.rotation_speed = 1.0; // 确保实时渲染时有旋转速度
        }

        self.animation_time += dt;

        if let Some(scene) = &mut self.scene {
            let rotation_delta_rad = (self.rotation_speed * dt * 50.0).to_radians();
            // 直接使用 self.args 来获取旋转轴和动画类型，因为这是在主线程中
            let rotation_axis_vec = get_animation_axis_vector(&self.args);

            match self.args.animation_type {
                AnimationType::CameraOrbit => {
                    let mut camera = scene.active_camera.clone();
                    camera.orbit(&rotation_axis_vec, rotation_delta_rad);
                    scene.set_camera(camera);
                }
                AnimationType::ObjectLocalRotation => {
                    for object in scene.objects.iter_mut() {
                        object.rotate(&rotation_axis_vec, rotation_delta_rad);
                    }
                }
                AnimationType::None => { /* 无动画 */ }
            }

            let config = create_render_config(scene, &self.args); // 使用 self.args

            if cfg!(debug_assertions) {
                println!(
                    "实时渲染中: FPS={:.1}, 动画类型={:?}, 轴={:?}, 旋转速度={}, 角度增量={:.3}rad, Phong={}",
                    self.avg_fps,
                    self.args.animation_type,
                    self.args.rotation_axis,
                    self.rotation_speed,
                    rotation_delta_rad,
                    self.args.use_phong
                );
            }

            let mut config_clone = config.clone(); // 克隆 config 用于渲染
            self.renderer.render_scene(scene, &mut config_clone);
            self.display_render_result(ctx);
            ctx.request_repaint();
        }
    }

    fn check_animation_params_changed(&mut self, ctx: &Context) -> bool {
        let mut changed = false;
        let mut messages = Vec::new();

        if self.last_pre_render_fps != Some(self.fps) {
            messages.push("帧率");
            changed = true;
        }
        if self
            .last_pre_render_speed
            .is_none_or(|last| (last - self.rotation_speed).abs() > 0.01)
        {
            messages.push("旋转速度");
            changed = true;
        }
        // 使用 self.args 进行比较
        if self
            .last_pre_render_animation_type
            .as_ref()
            .is_none_or(|last| last != &self.args.animation_type)
        {
            messages.push("动画类型");
            changed = true;
        }
        if self
            .last_pre_render_rotation_axis
            .as_ref()
            .is_none_or(|last| *last != self.args.rotation_axis)
        {
            messages.push("旋转轴");
            changed = true;
        }
        if self.args.rotation_axis == RotationAxis::Custom
            && self.last_pre_render_custom_axis.as_ref() != Some(&self.args.custom_rotation_axis)
        {
            messages.push("自定义旋转轴");
            changed = true;
        }

        if changed {
            self.status_message = format!("{}已改变，需要重新预渲染...", messages.join("、"));
            self.last_pre_render_fps = Some(self.fps);
            self.last_pre_render_speed = Some(self.rotation_speed);
            self.last_pre_render_animation_type = Some(self.args.animation_type.clone());
            self.last_pre_render_rotation_axis = Some(self.args.rotation_axis.clone());
            if self.args.rotation_axis == RotationAxis::Custom {
                self.last_pre_render_custom_axis = Some(self.args.custom_rotation_axis.clone());
            }
            ctx.request_repaint();
        }
        changed
    }

    fn start_video_generation(&mut self, ctx: &egui::Context) {
        if !self.ffmpeg_available {
            self.set_error("无法生成视频：未检测到ffmpeg。请安装ffmpeg后重试。".to_string());
            return;
        }
        if self.is_generating_video {
            self.status_message = "视频已在生成中，请等待完成...".to_string();
            return;
        }

        match self.validate_parameters() {
            Ok(_) => {
                let output_dir = self.args.output_dir.clone(); // 从 self.args 克隆
                if let Err(e) = fs::create_dir_all(&output_dir) {
                    self.set_error(format!("创建输出目录失败: {}", e));
                    return;
                }
                let frames_dir = format!(
                    "{}/temp_frames_{}",
                    output_dir,
                    chrono::Utc::now().timestamp_millis()
                );
                if let Err(e) = fs::create_dir_all(&frames_dir) {
                    self.set_error(format!("创建帧目录失败: {}", e));
                    return;
                }

                if self.scene.is_none() {
                    let obj_path = self.args.obj.clone(); // 从 self.args 克隆
                    match self.load_model(&obj_path) {
                        Ok(_) => self.status_message = "模型加载成功，开始生成视频...".to_string(),
                        Err(e) => {
                            self.set_error(format!("加载模型失败，无法生成视频: {}", e));
                            return;
                        }
                    }
                }

                let args_for_thread = self.args.clone(); // 克隆 self.args 以传递给线程
                let video_progress_arc = self.video_progress.clone();
                let total_frames = self.total_frames;
                let fps = self.fps;
                let scene_clone = self.scene.as_ref().expect("场景已检查").clone();

                self.is_generating_video = true;
                video_progress_arc.store(0, Ordering::SeqCst);
                self.status_message = format!("开始后台生成视频 (0/{})...", total_frames);
                ctx.request_repaint();
                let ctx_clone = ctx.clone();
                let video_filename = format!("{}.mp4", args_for_thread.output); // 使用克隆的 args
                let video_output_path = format!("{}/{}", output_dir, video_filename);

                let thread_handle = thread::spawn(move || {
                    let mut scene_copy = scene_clone;
                    // 在线程内部，克隆的 args_for_thread 现在被称为 args
                    let thread_renderer =
                        Renderer::new(args_for_thread.width, args_for_thread.height);
                    let rotation_per_frame_rad = (360.0 / total_frames.max(1) as f32).to_radians();
                    let rotation_axis_vec = get_animation_axis_vector(&args_for_thread);

                    for frame_num in 0..total_frames {
                        video_progress_arc.store(frame_num, Ordering::SeqCst);
                        if frame_num > 0 {
                            // 第一帧通常不旋转
                            match args_for_thread.animation_type {
                                AnimationType::CameraOrbit => {
                                    let mut camera = scene_copy.active_camera.clone();
                                    camera.orbit(&rotation_axis_vec, rotation_per_frame_rad);
                                    scene_copy.set_camera(camera);
                                }
                                AnimationType::ObjectLocalRotation => {
                                    for object in scene_copy.objects.iter_mut() {
                                        object.rotate(&rotation_axis_vec, rotation_per_frame_rad);
                                    }
                                }
                                AnimationType::None => {}
                            }
                        }

                        let config = create_render_config(&scene_copy, &args_for_thread);
                        let mut config_for_render = config.clone();
                        thread_renderer.render_scene(&scene_copy, &mut config_for_render);
                        let frame_path = format!("{}/frame_{:04}.png", frames_dir, frame_num);
                        let color_data = thread_renderer.frame_buffer.get_color_buffer_bytes();
                        save_image(
                            &frame_path,
                            &color_data,
                            args_for_thread.width as u32,
                            args_for_thread.height as u32,
                        );

                        if frame_num % (total_frames.max(1) / 20).max(1) == 0 {
                            // 每渲染约5%的帧就请求重绘UI
                            ctx_clone.request_repaint();
                        }
                    }
                    video_progress_arc.store(total_frames, Ordering::SeqCst); // 标记完成
                    ctx_clone.request_repaint(); // 更新UI显示完成状态

                    let frames_pattern = format!("{}/frame_%04d.png", frames_dir);
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
                            "23", // 合理的质量/大小折中
                            &video_output_path,
                        ])
                        .status();
                    let success = ffmpeg_status.is_ok_and(|s| s.success());
                    if success {
                        println!("视频生成成功：{}", video_output_path);
                    } else {
                        println!("ffmpeg调用失败或视频生成未成功。");
                    }
                    let _ = std::fs::remove_dir_all(&frames_dir); // 清理临时帧文件
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

        // 更新上次预渲染参数，使用 self.args
        self.last_pre_render_fps = Some(self.fps);
        self.last_pre_render_speed = Some(self.rotation_speed);
        self.last_pre_render_animation_type = Some(self.args.animation_type.clone());
        self.last_pre_render_rotation_axis = Some(self.args.rotation_axis.clone());
        if self.args.rotation_axis == RotationAxis::Custom {
            self.last_pre_render_custom_axis = Some(self.args.custom_rotation_axis.clone());
        }

        match self.validate_parameters() {
            Ok(_) => {
                if self.scene.is_none() {
                    let obj_path = self.args.obj.clone(); // 从 self.args 克隆
                    match self.load_model(&obj_path) {
                        Ok(_) => self.status_message = "模型加载成功，开始预渲染...".to_string(),
                        Err(e) => {
                            self.set_error(format!("加载模型失败，无法预渲染: {}", e));
                            return;
                        }
                    }
                }

                let mut effective_rotation_speed_dps = self.rotation_speed * 50.0;
                if effective_rotation_speed_dps.abs() < 0.001 {
                    effective_rotation_speed_dps =
                        0.1_f32.copysign(effective_rotation_speed_dps.signum());
                    if effective_rotation_speed_dps == 0.0 {
                        effective_rotation_speed_dps = 0.1;
                    }
                }

                let seconds_per_rotation = 360.0 / effective_rotation_speed_dps.abs();
                let frames_for_one_rotation =
                    (seconds_per_rotation * self.fps as f32).ceil() as usize;
                let frames_to_render = frames_for_one_rotation.max(1);
                self.total_frames_for_pre_render_cycle = frames_to_render;

                self.is_pre_rendering = true;
                self.pre_rendered_frames.lock().unwrap().clear();
                self.pre_render_progress.store(0, Ordering::SeqCst);
                self.current_frame_index = 0; // 重置播放索引

                let args_for_thread = self.args.clone(); // 克隆 self.args 以传递给线程
                let progress_arc = self.pre_render_progress.clone();
                let frames_arc = self.pre_rendered_frames.clone();
                let width = args_for_thread.width; // 使用克隆的 args
                let height = args_for_thread.height; // 使用克隆的 args
                let scene_clone = self.scene.as_ref().expect("场景已检查存在").clone();
                let rotation_axis_vec = get_animation_axis_vector(&args_for_thread);

                self.status_message = format!(
                    "开始预渲染动画 (0/{} 帧，转一圈需 {:.1} 秒)...",
                    frames_to_render, seconds_per_rotation
                );
                ctx.request_repaint();
                let ctx_clone = ctx.clone();

                thread::spawn(move || {
                    let mut scene_copy = scene_clone;
                    // 在线程内部，克隆的 args_for_thread 现在被称为 args_clone (或任何你选择的名称)
                    let thread_renderer = Renderer::new(width, height);
                    let rotation_increment_rad_per_frame = (360.0 / frames_to_render as f32)
                        .to_radians()
                        * effective_rotation_speed_dps.signum();

                    for frame_num in 0..frames_to_render {
                        progress_arc.store(frame_num, Ordering::SeqCst);
                        if frame_num > 0 {
                            // 第一帧通常不旋转
                            match args_for_thread.animation_type {
                                AnimationType::CameraOrbit => {
                                    let mut camera = scene_copy.active_camera.clone();
                                    camera.orbit(
                                        &rotation_axis_vec,
                                        rotation_increment_rad_per_frame,
                                    );
                                    scene_copy.set_camera(camera);
                                }
                                AnimationType::ObjectLocalRotation => {
                                    for object in scene_copy.objects.iter_mut() {
                                        object.rotate(
                                            &rotation_axis_vec,
                                            rotation_increment_rad_per_frame,
                                        );
                                    }
                                }
                                AnimationType::None => {}
                            }
                        }

                        let config = create_render_config(&scene_copy, &args_for_thread);
                        let mut config_for_render = config.clone();
                        thread_renderer.render_scene(&scene_copy, &mut config_for_render);
                        let color_data_rgb = thread_renderer.frame_buffer.get_color_buffer_bytes();
                        let mut rgba_data = Vec::with_capacity(width * height * 4);
                        for chunk in color_data_rgb.chunks_exact(3) {
                            rgba_data.extend_from_slice(chunk);
                            rgba_data.push(255); // Alpha
                        }
                        let color_image =
                            ColorImage::from_rgba_unmultiplied([width, height], &rgba_data);
                        frames_arc.lock().unwrap().push(color_image);

                        if frame_num % (frames_to_render.max(1) / 20).max(1) == 0 {
                            ctx_clone.request_repaint();
                        }
                    }
                    progress_arc.store(frames_to_render, Ordering::SeqCst); // 标记完成
                    ctx_clone.request_repaint(); // 更新UI显示完成状态
                });
            }
            Err(e) => {
                self.set_error(e);
                self.is_pre_rendering = false; // 确保出错时重置状态
            }
        }
    }

    fn handle_pre_rendering_tasks(&mut self, ctx: &Context) {
        let progress = self.pre_render_progress.load(Ordering::SeqCst);
        let expected_total_frames = self.total_frames_for_pre_render_cycle;
        let mut effective_rotation_speed_dps = self.rotation_speed * 50.0;
        if effective_rotation_speed_dps.abs() < 0.001 {
            effective_rotation_speed_dps = 0.1_f32.copysign(effective_rotation_speed_dps.signum());
            if effective_rotation_speed_dps == 0.0 {
                effective_rotation_speed_dps = 0.1;
            }
        }
        let seconds_per_rotation = 360.0 / effective_rotation_speed_dps.abs();

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
                final_frame_count, self.fps, seconds_per_rotation
            );
            if self.is_realtime_rendering || self.pre_render_mode {
                // 如果仍在预渲染模式或实时渲染模式，则准备播放
                self.current_frame_index = 0; // 重置播放索引
                self.last_frame_time = None; // 重置计时器以平滑播放第一帧
                ctx.request_repaint();
            }
        } else {
            ctx.request_repaint_after(Duration::from_millis(100)); // 轮询进度
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
                self.pre_render_mode = false; // 没有帧可播，退出预渲染模式
                self.status_message = "预渲染帧丢失或未生成，退出预渲染模式。".to_string();
                ctx.request_repaint();
                return;
            }
            frame_to_display_idx = self.current_frame_index % frames_len;
            frame_image = frames_guard[frame_to_display_idx].clone();
        }

        let now = Instant::now();
        let target_frame_duration = Duration::from_secs_f32(1.0 / self.fps.max(1) as f32);

        if let Some(last_frame_display_time) = self.last_frame_time {
            let time_since_last_display = now.duration_since(last_frame_display_time);
            if time_since_last_display < target_frame_duration {
                let time_to_wait = target_frame_duration - time_since_last_display;
                ctx.request_repaint_after(time_to_wait); // 等待直到下一帧的时间
                return;
            }
            self.update_fps_stats(time_since_last_display);
        } else {
            // 第一次播放或从暂停/停止状态恢复
            self.update_fps_stats(target_frame_duration); // 假设理想帧时间
        }
        self.last_frame_time = Some(now);

        let texture_name = format!("pre_rendered_tex_{}", frame_to_display_idx); // 唯一的纹理名称
        self.rendered_image =
            Some(ctx.load_texture(texture_name, frame_image, TextureOptions::LINEAR));
        self.current_frame_index = (self.current_frame_index + 1) % frames_len; // 循环播放

        let mut effective_rotation_speed_dps = self.rotation_speed * 50.0;
        if effective_rotation_speed_dps.abs() < 0.001 {
            effective_rotation_speed_dps = 0.1_f32.copysign(effective_rotation_speed_dps.signum());
            if effective_rotation_speed_dps == 0.0 {
                effective_rotation_speed_dps = 0.1;
            }
        }
        let seconds_per_rotation = 360.0 / effective_rotation_speed_dps.abs();

        self.status_message = format!(
            "播放预渲染: 帧 {}/{} (目标 {} FPS, 平均 {:.1} FPS, 1圈 {:.1}秒)",
            frame_to_display_idx + 1,
            frames_len,
            self.fps,
            self.avg_fps,
            seconds_per_rotation
        );
        ctx.request_repaint(); // 请求下一帧的重绘
    }
}
