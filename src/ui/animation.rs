use crate::core::render_config::create_render_config;
use crate::core::renderer::Renderer;
use crate::io::args::get_animation_axis_vector;
use crate::utils::render_utils::{
    animate_scene_step, calculate_frame_rotation, calculate_rotation_delta,
    calculate_rotation_parameters,
};
use crate::utils::save_utils::save_image;
use egui::{ColorImage, Context, TextureOptions};
use std::fs;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::{Duration, Instant};

use super::app::RasterizerApp;
use super::core::CoreMethods;
use super::render_ui::RenderMethods;

/// 动画与视频生成相关方法的特质
pub trait AnimationMethods {
    /// 执行实时渲染循环
    fn perform_realtime_rendering(&mut self, ctx: &Context);

    /// 在后台生成视频
    fn start_video_generation(&mut self, ctx: &Context);

    /// 切换预渲染模式
    fn toggle_pre_render_mode(&mut self);

    /// 启动预渲染过程
    fn start_pre_rendering(&mut self, ctx: &Context);

    /// 处理预渲染帧
    fn handle_pre_rendering_tasks(&mut self, ctx: &Context);

    /// 播放预渲染帧
    fn play_pre_rendered_frames(&mut self, ctx: &Context);
}

impl AnimationMethods for RasterizerApp {
    /// 执行实时渲染循环
    fn perform_realtime_rendering(&mut self, ctx: &egui::Context) {
        // 确保安全地进入预渲染模式
        if self.pre_render_mode
            && !self.is_pre_rendering
            && self.pre_rendered_frames.lock().unwrap().is_empty()
        {
            // 检查模型是否已加载
            if self.scene.is_none() {
                let obj_path = self.args.obj.clone();
                match self.load_model(&obj_path) {
                    Ok(_) => {
                        self.start_pre_rendering(ctx);
                        return;
                    }
                    Err(e) => {
                        self.set_error(format!("加载模型失败: {}", e));
                        self.is_realtime_rendering = false;
                        self.pre_render_mode = false; // 关闭预渲染模式以避免卡住
                        return;
                    }
                }
            } else {
                self.start_pre_rendering(ctx);
                return;
            }
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
            let obj_path = self.args.obj.clone();
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

        if self.renderer.frame_buffer.width != self.args.width
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
            CoreMethods::update_fps_stats(self, frame_time);
        }
        self.last_frame_time = Some(now);

        if self.is_realtime_rendering && self.args.rotation_speed.abs() < 0.01 {
            self.args.rotation_speed = 1.0; // 确保实时渲染时有旋转速度
        }

        self.animation_time += dt;

        if let Some(scene) = &mut self.scene {
            // 使用通用函数计算旋转增量
            let rotation_delta_rad = calculate_rotation_delta(self.args.rotation_speed, dt);
            let rotation_axis_vec = get_animation_axis_vector(&self.args);

            // 使用通用函数执行动画步骤
            animate_scene_step(
                scene,
                &self.args.animation_type,
                &rotation_axis_vec,
                rotation_delta_rad,
            );

            let config = create_render_config(scene, &self.args);

            if cfg!(debug_assertions) {
                println!(
                    "实时渲染中: FPS={:.1}, 动画类型={:?}, 轴={:?}, 旋转速度={}, 角度增量={:.3}rad, Phong={}",
                    self.avg_fps,
                    self.args.animation_type,
                    self.args.rotation_axis,
                    self.args.rotation_speed,
                    rotation_delta_rad,
                    self.args.use_phong
                );
            }

            let mut config_clone = config.clone();
            self.renderer.render_scene(scene, &mut config_clone);
            self.display_render_result(ctx);
            ctx.request_repaint();
        }
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
                let output_dir = self.args.output_dir.clone();
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
                    let obj_path = self.args.obj.clone();
                    match self.load_model(&obj_path) {
                        Ok(_) => self.status_message = "模型加载成功，开始生成视频...".to_string(),
                        Err(e) => {
                            self.set_error(format!("加载模型失败，无法生成视频: {}", e));
                            return;
                        }
                    }
                }

                let args_for_thread = self.args.clone();
                let video_progress_arc = self.video_progress.clone();
                let total_frames = self.args.total_frames;
                let fps = self.args.fps;
                let scene_clone = self.scene.as_ref().expect("场景已检查").clone();

                self.is_generating_video = true;
                video_progress_arc.store(0, Ordering::SeqCst);
                self.status_message = format!("开始后台生成视频 (0/{})...", total_frames);
                ctx.request_repaint();
                let ctx_clone = ctx.clone();
                let video_filename = format!("{}.mp4", args_for_thread.output);
                let video_output_path = format!("{}/{}", output_dir, video_filename);
                let frames_dir_clone = frames_dir.clone();

                let thread_handle = thread::spawn(move || {
                    let mut scene_copy = scene_clone;
                    let thread_renderer =
                        Renderer::new(args_for_thread.width, args_for_thread.height);

                    // 使用通用函数计算旋转角度
                    let rotation_per_frame_rad = calculate_frame_rotation(
                        total_frames,
                        args_for_thread.rotation_speed.signum(),
                    );
                    let rotation_axis_vec = get_animation_axis_vector(&args_for_thread);

                    for frame_num in 0..total_frames {
                        video_progress_arc.store(frame_num, Ordering::SeqCst);

                        if frame_num > 0 {
                            // 使用通用函数执行动画步骤
                            animate_scene_step(
                                &mut scene_copy,
                                &args_for_thread.animation_type,
                                &rotation_axis_vec,
                                rotation_per_frame_rad,
                            );
                        }

                        let config = create_render_config(&scene_copy, &args_for_thread);
                        let mut config_for_render = config.clone();
                        thread_renderer.render_scene(&scene_copy, &mut config_for_render);
                        let frame_path = format!("{}/frame_{:04}.png", frames_dir_clone, frame_num);
                        let color_data = thread_renderer.frame_buffer.get_color_buffer_bytes();
                        save_image(
                            &frame_path,
                            &color_data,
                            args_for_thread.width as u32,
                            args_for_thread.height as u32,
                        );

                        if frame_num % (total_frames.max(1) / 20).max(1) == 0 {
                            ctx_clone.request_repaint();
                        }
                    }
                    video_progress_arc.store(total_frames, Ordering::SeqCst);
                    ctx_clone.request_repaint();

                    let frames_pattern = format!("{}/frame_%04d.png", frames_dir_clone);
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
                    if success {
                        println!("视频生成成功：{}", video_output_path);
                    } else {
                        println!("ffmpeg调用失败或视频生成未成功。");
                    }
                    let _ = std::fs::remove_dir_all(&frames_dir_clone);
                    (success, video_output_path)
                });
                self.video_generation_thread = Some(thread_handle);
            }
            Err(e) => self.set_error(e),
        }
    }

    /// 切换预渲染模式 (使用CoreMethods实现)
    fn toggle_pre_render_mode(&mut self) {
        // 直接调用CoreMethods中的实现
        CoreMethods::toggle_pre_render_mode(self);
    }

    fn start_pre_rendering(&mut self, ctx: &Context) {
        if self.is_pre_rendering {
            return;
        }

        match self.validate_parameters() {
            Ok(_) => {
                if self.scene.is_none() {
                    let obj_path = self.args.obj.clone();
                    match self.load_model(&obj_path) {
                        Ok(_) => self.status_message = "模型加载成功，开始预渲染...".to_string(),
                        Err(e) => {
                            self.set_error(format!("加载模型失败，无法预渲染: {}", e));
                            return;
                        }
                    }
                }

                // 使用通用函数计算旋转参数
                let (effective_rotation_speed_dps, seconds_per_rotation, frames_to_render) =
                    calculate_rotation_parameters(self.args.rotation_speed, self.args.fps);

                self.total_frames_for_pre_render_cycle = frames_to_render;

                self.is_pre_rendering = true;
                self.pre_rendered_frames.lock().unwrap().clear();
                self.pre_render_progress.store(0, Ordering::SeqCst);
                self.current_frame_index = 0;

                let args_for_thread = self.args.clone();
                let progress_arc = self.pre_render_progress.clone();
                let frames_arc = self.pre_rendered_frames.clone();
                let width = args_for_thread.width;
                let height = args_for_thread.height;
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
                    let thread_renderer = Renderer::new(width, height);

                    // 计算每帧旋转增量
                    let rotation_increment_rad_per_frame = (360.0 / frames_to_render as f32)
                        .to_radians()
                        * effective_rotation_speed_dps.signum();

                    for frame_num in 0..frames_to_render {
                        progress_arc.store(frame_num, Ordering::SeqCst);

                        if frame_num > 0 {
                            // 使用通用函数执行动画步骤
                            animate_scene_step(
                                &mut scene_copy,
                                &args_for_thread.animation_type,
                                &rotation_axis_vec,
                                rotation_increment_rad_per_frame,
                            );
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

                    progress_arc.store(frames_to_render, Ordering::SeqCst);
                    ctx_clone.request_repaint();
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
            calculate_rotation_parameters(self.args.rotation_speed, self.args.fps);

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
                final_frame_count, self.args.fps, seconds_per_rotation
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
        let target_frame_duration = Duration::from_secs_f32(1.0 / self.args.fps.max(1) as f32);

        if let Some(last_frame_display_time) = self.last_frame_time {
            let time_since_last_display = now.duration_since(last_frame_display_time);
            if time_since_last_display < target_frame_duration {
                let time_to_wait = target_frame_duration - time_since_last_display;
                ctx.request_repaint_after(time_to_wait);
                return;
            }
            CoreMethods::update_fps_stats(self, time_since_last_display);
        } else {
            CoreMethods::update_fps_stats(self, target_frame_duration);
        }
        self.last_frame_time = Some(now);

        let texture_name = format!("pre_rendered_tex_{}", frame_to_display_idx);
        self.rendered_image =
            Some(ctx.load_texture(texture_name, frame_image, TextureOptions::LINEAR));
        self.current_frame_index = (self.current_frame_index + 1) % frames_len;

        // 使用通用函数计算参数
        let (_, seconds_per_rotation, _) =
            calculate_rotation_parameters(self.args.rotation_speed, self.args.fps);

        self.status_message = format!(
            "播放预渲染: 帧 {}/{} (目标 {} FPS, 平均 {:.1} FPS, 1圈 {:.1}秒)",
            frame_to_display_idx + 1,
            frames_len,
            self.args.fps,
            self.avg_fps,
            seconds_per_rotation
        );
        ctx.request_repaint();
    }
}
