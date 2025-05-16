use crate::core::renderer::Renderer;
use crate::demos::demos::*;
use crate::utils::depth_image;
use egui::Context;
use std::fs;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::Instant;

use super::app::RasterizerApp;
use super::render::RenderMethods;

/// 动画与视频生成相关方法的特质
pub trait AnimationMethods {
    /// 执行实时渲染循环
    fn perform_realtime_rendering(&mut self, ctx: &Context);

    /// 在后台生成视频
    fn start_video_generation(&mut self, ctx: &Context);
}

impl AnimationMethods for RasterizerApp {
    /// 执行实时渲染循环
    fn perform_realtime_rendering(&mut self, ctx: &egui::Context) {
        // 检查是否已有场景和模型
        if self.scene.is_none() {
            // 尝试加载模型
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

        // 检查渲染器的尺寸是否与args中的一致，如不一致则重新创建
        if self.renderer.frame_buffer.width != self.args.width
            || self.renderer.frame_buffer.height != self.args.height
        {
            self.renderer = Renderer::new(self.args.width, self.args.height);
            // 清除已渲染的图像
            self.rendered_image = None;
            println!(
                "重新创建渲染器，尺寸: {}x{}",
                self.args.width, self.args.height
            );
        }

        // 计算自上次渲染以来的时间增量
        let now = Instant::now();
        let dt = if let Some(last_time) = self.last_frame_time {
            now.duration_since(last_time).as_secs_f32()
        } else {
            1.0 / 60.0 // 首帧使用默认值
        };

        // 使用平滑帧率计算
        if let Some(last_time) = self.last_frame_time {
            let frame_time = now.duration_since(last_time);
            self.update_fps_stats(frame_time);
        }

        self.last_frame_time = Some(now);

        // 确保旋转速度不为零，这样模型就会旋转
        if self.rotation_speed < 0.1 {
            self.rotation_speed = 1.0; // 设置一个默认的旋转速度
        }

        // 更新旋转角度
        self.rotation_angle += dt * self.rotation_speed * 50.0; // 乘以常数调整转速

        // 创建args的一个克隆，避免在场景操作期间再次借用self
        let args_clone = self.args.clone();

        // 更新场景
        if let Some(scene) = &mut self.scene {
            // 更新相机，使其围绕Y轴旋转
            let mut camera = scene.active_camera.clone();
            let rotation_delta = self.rotation_speed * dt * 50.0;
            camera.orbit_y(rotation_delta);
            scene.set_camera(camera);

            // 计算当前帧号，仅用于动画效果计算
            let frame_num = (self.rotation_angle / 360.0 * self.total_frames as f32) as usize
                % self.total_frames;

            // 更新场景中的所有对象
            update_scene_objects(scene, frame_num, rotation_delta);

            // 创建渲染配置，使用克隆的args
            let config = create_render_config(scene, &args_clone);

            // 输出调试信息，帮助诊断问题
            if cfg!(debug_assertions) {
                println!(
                    "实时渲染中: FPS={:.1}, 旋转速度={}, 角度={:.2}, 帧号={}, 旋转增量={:.3}度, 使用Phong={}",
                    self.avg_fps, // 使用平均帧率代替即时帧率
                    self.rotation_speed,
                    self.rotation_angle % 360.0,
                    frame_num,
                    rotation_delta,
                    args_clone.use_phong
                );
            }

            // 渲染到帧缓冲区
            self.renderer.render_scene(scene, &config);

            // 更新UI中的图像
            self.display_render_result(ctx);

            // 请求连续重绘
            ctx.request_repaint();
        }
    }

    /// 在后台生成视频
    fn start_video_generation(&mut self, ctx: &egui::Context) {
        // 检查ffmpeg是否可用
        if !self.ffmpeg_available {
            self.set_error("无法生成视频：未检测到ffmpeg。请安装ffmpeg后重试。".to_string());
            return;
        }

        // 如果已经在生成视频，不要重复启动
        if self.is_generating_video {
            self.status_message = "视频已在生成中，请等待完成...".to_string();
            return;
        }

        // 验证参数
        match self.validate_parameters() {
            Ok(_) => {
                // 确保输出目录存在
                let output_dir = self.args.output_dir.clone();
                if let Err(e) = fs::create_dir_all(&output_dir) {
                    self.set_error(format!("创建输出目录失败: {}", e));
                    return;
                }

                // 创建临时帧目录
                let frames_dir = format!("{}/temp_frames", output_dir);
                if let Err(e) = fs::create_dir_all(&frames_dir) {
                    self.set_error(format!("创建帧目录失败: {}", e));
                    return;
                }

                // 在进行视频生成前，确保有可加载的模型
                if self.scene.is_none() {
                    let obj_path = self.args.obj.clone();
                    match self.load_model(&obj_path) {
                        Ok(_) => {
                            self.status_message = "模型加载成功，开始生成视频...".to_string();
                        }
                        Err(e) => {
                            self.set_error(format!("加载模型失败，无法生成视频: {}", e));
                            return;
                        }
                    }
                }

                // 克隆线程所需的数据（所有权转移到线程中）
                let args = self.args.clone();
                let video_progress = self.video_progress.clone();
                let total_frames = self.total_frames;
                let fps = self.fps;

                // 创建一个场景的完整克隆用于后台线程
                let scene = if let Some(ref scene) = self.scene {
                    scene.clone()
                } else {
                    self.set_error("场景未初始化，无法生成视频".to_string());
                    return;
                };

                // 设置渲染状态
                self.is_generating_video = true;
                video_progress.store(0, Ordering::SeqCst);

                // 更新状态消息
                self.status_message = format!("开始后台生成视频 (0/{})...", total_frames);
                ctx.request_repaint();

                // 创建一个新的渲染器用于后台线程
                let renderer = Renderer::new(args.width, args.height);

                // 计算视频输出路径
                let video_filename = format!("{}.mp4", args.output);
                let video_output_path = format!("{}/{}", output_dir, video_filename);

                // 启动后台线程
                let thread_handle = thread::spawn(move || {
                    // 创建一个可变的场景副本
                    let mut scene_copy = scene;

                    // 计算每帧旋转增量（角度）
                    let rotation_increment = 360.0 / total_frames as f32;

                    // 渲染每一帧
                    for frame_num in 0..total_frames {
                        // 更新进度
                        video_progress.store(frame_num, Ordering::SeqCst);

                        // 如果不是第一帧，更新场景
                        if frame_num > 0 {
                            // 更新相机，使其围绕Y轴旋转
                            let mut camera = scene_copy.active_camera.clone();
                            camera.orbit_y(rotation_increment);
                            scene_copy.set_camera(camera);

                            // 更新场景对象
                            update_scene_objects(&mut scene_copy, frame_num, rotation_increment);
                        }

                        // 创建渲染配置
                        let config = create_render_config(&scene_copy, &args);

                        // 渲染场景
                        renderer.render_scene(&scene_copy, &config);

                        // 保存帧
                        let frame_path = format!("{}/frame_{:04}.png", frames_dir, frame_num);
                        let color_data = renderer.frame_buffer.get_color_buffer_bytes();

                        // 保存彩色图像
                        depth_image::save_image(
                            &frame_path,
                            &color_data,
                            args.width as u32,
                            args.height as u32,
                        );
                    }

                    // 更新进度为100%
                    video_progress.store(total_frames, Ordering::SeqCst);

                    // 使用ffmpeg合成视频
                    let frames_path = format!("{}/frame_%04d.png", frames_dir);

                    let ffmpeg_result = std::process::Command::new("ffmpeg")
                        .args([
                            "-y", // 覆盖现有文件
                            "-framerate",
                            &fps.to_string(),
                            "-i",
                            &frames_path,
                            "-c:v",
                            "libx264",
                            "-pix_fmt",
                            "yuv420p",
                            "-crf",
                            "23", // 质量设置，较低的值 = 更高的质量
                            &video_output_path,
                        ])
                        .output();

                    // 处理ffmpeg输出结果（只记录，不会显示给用户）
                    match ffmpeg_result {
                        Ok(_) => {
                            println!("视频生成成功：{}", video_output_path);
                        }
                        Err(e) => {
                            println!("ffmpeg调用失败：{}", e);
                        }
                    }

                    // 清理临时帧目录
                    let _ = std::fs::remove_dir_all(frames_dir);
                });

                // 保存线程句柄
                self.video_generation_thread = Some(thread_handle);
            }
            Err(e) => {
                // 参数验证失败，显示错误
                self.set_error(e);
            }
        }
    }
}
