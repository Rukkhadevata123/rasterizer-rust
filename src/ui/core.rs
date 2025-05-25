use crate::ui::app::RasterizerApp;
use crate::utils::save_utils::save_render_with_settings;
use egui::{Color32, Context};
use log::{debug, error, warn};
use std::fs;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use super::app::InterfaceInteraction;

/// 核心业务逻辑方法
///
/// 该trait包含应用的核心功能：
/// - 渲染和加载逻辑
/// - 状态转换与管理
/// - 错误处理
/// - 性能统计
/// - 资源管理
pub trait CoreMethods {
    // === 🎯 **核心渲染和加载** ===

    /// 🔥 **渲染当前场景** - 统一渲染入口
    fn render(&mut self, ctx: &Context);

    /// 在UI中显示渲染结果
    fn display_render_result(&mut self, ctx: &Context);

    /// 如果相机发生变化，执行重新渲染
    fn render_if_anything_changed(&mut self, ctx: &Context);

    /// 保存当前渲染结果为截图
    fn take_screenshot(&mut self) -> Result<String, String>;

    // === 🎯 **状态管理** ===

    /// 设置错误信息
    fn set_error(&mut self, message: String);

    /// 将应用状态重置为默认值
    fn reset_to_defaults(&mut self);

    /// 切换预渲染模式开启/关闭状态
    fn toggle_pre_render_mode(&mut self);

    /// 清空预渲染的动画帧缓冲区
    fn clear_pre_rendered_frames(&mut self);

    // === 🎯 **状态查询** ===

    /// 检查是否可以清除预渲染缓冲区
    fn can_clear_buffer(&self) -> bool;

    /// 检查是否可以切换预渲染模式
    fn can_toggle_pre_render(&self) -> bool;

    /// 检查是否可以开始或停止动画渲染
    fn can_render_animation(&self) -> bool;

    /// 检查是否可以生成视频
    fn can_generate_video(&self) -> bool;

    // === 🎯 **动画状态管理** ===

    /// 开始实时渲染动画
    fn start_animation_rendering(&mut self) -> Result<(), String>;

    /// 停止实时渲染动画
    fn stop_animation_rendering(&mut self);

    // === 🎯 **性能统计** ===

    /// 更新帧率统计信息
    fn update_fps_stats(&mut self, frame_time: Duration);

    /// 获取格式化的帧率显示文本和颜色
    fn get_fps_display(&self) -> (String, Color32);

    // === 🎯 **资源管理** ===

    /// 执行资源清理操作
    fn cleanup_resources(&mut self);
}

impl CoreMethods for RasterizerApp {
    // === 🔥 **核心渲染和加载实现** ===

    /// 🔥 **渲染当前场景** - 统一渲染逻辑
    fn render(&mut self, ctx: &Context) {
        // 验证参数
        if let Err(e) = self.settings.validate() {
            self.set_error(e);
            return;
        }

        // 获取OBJ路径
        let obj_path = match &self.settings.obj {
            Some(path) => path.clone(),
            None => {
                self.set_error("错误: 未指定OBJ文件路径".to_string());
                return;
            }
        };

        self.status_message = format!("正在加载 {}...", obj_path);
        ctx.request_repaint(); // 立即更新状态消息

        // 加载模型
        match crate::io::model_loader::ModelLoader::load_and_create_scene(&obj_path, &self.settings)
        {
            Ok((scene, model_data)) => {
                debug!(
                    "场景创建完成: 光源数量={}, 使用光照={}, 环境光强度={}",
                    scene.lights.len(),
                    self.settings.use_lighting,
                    self.settings.ambient
                );

                // 直接设置场景和模型数据
                self.scene = Some(scene);
                self.model_data = Some(model_data);

                self.status_message = "模型加载成功，开始渲染...".to_string();
            }
            Err(e) => {
                self.set_error(format!("加载模型失败: {}", e));
                return;
            }
        }

        self.status_message = "模型加载成功，开始渲染...".to_string();
        ctx.request_repaint();

        // 确保输出目录存在
        let output_dir = self.settings.output_dir.clone();
        if let Err(e) = fs::create_dir_all(&output_dir) {
            self.set_error(format!("创建输出目录失败: {}", e));
            return;
        }

        // 渲染
        let start_time = Instant::now();

        if let Some(scene) = &mut self.scene {
            // 渲染到帧缓冲区
            self.renderer.render_scene(scene, &self.settings);

            // 保存输出文件
            if let Err(e) = save_render_with_settings(&self.renderer, &self.settings, None) {
                warn!("保存渲染结果时发生错误: {}", e);
            }

            // 更新状态
            self.last_render_time = Some(start_time.elapsed());
            let output_dir = self.settings.output_dir.clone();
            let output_name = self.settings.output.clone();
            let elapsed = self.last_render_time.unwrap();
            self.status_message = format!(
                "渲染完成，耗时 {:.2?}，已保存到 {}/{}",
                elapsed, output_dir, output_name
            );

            // 在UI中显示渲染结果
            self.display_render_result(ctx);
        }
    }

    /// 在UI中显示渲染结果
    fn display_render_result(&mut self, ctx: &Context) {
        // 从渲染器获取图像数据
        let color_data = self.renderer.frame_buffer.get_color_buffer_bytes();

        // 确保分辨率与渲染器匹配
        let width = self.renderer.frame_buffer.width;
        let height = self.renderer.frame_buffer.height;

        // 创建或更新纹理
        let rendered_texture = self.rendered_image.get_or_insert_with(|| {
            // 创建一个全黑的空白图像
            let color = Color32::BLACK;
            ctx.load_texture(
                "rendered_image",
                egui::ColorImage::new([width, height], color),
                egui::TextureOptions::default(),
            )
        });

        // 将RGB数据转换为RGBA格式
        let mut rgba_data = Vec::with_capacity(color_data.len() / 3 * 4);
        for i in (0..color_data.len()).step_by(3) {
            if i + 2 < color_data.len() {
                rgba_data.push(color_data[i]); // R
                rgba_data.push(color_data[i + 1]); // G
                rgba_data.push(color_data[i + 2]); // B
                rgba_data.push(255); // A (完全不透明)
            }
        }

        // 更新纹理，使用渲染器的实际大小
        rendered_texture.set(
            egui::ColorImage::from_rgba_unmultiplied([width, height], &rgba_data),
            egui::TextureOptions::default(),
        );
    }

    /// 🔥 **统一同步入口 - 所有变化都在这里处理**
    fn render_if_anything_changed(&mut self, ctx: &Context) {
        if self.interface_interaction.anything_changed && self.scene.is_some() {
            if let Some(scene) = &mut self.scene {
                // 🔥 **统一同步所有状态 - 消除不对称性**

                // 1. 光源同步
                scene.lights = self.settings.lights.clone();
                scene.set_ambient_light(
                    self.settings.ambient,
                    self.settings.get_ambient_color_vec(),
                );

                // 2. 相机同步
                if let Ok(from) =
                    crate::io::render_settings::parse_point3(&self.settings.camera_from)
                {
                    scene.active_camera.params.position = from;
                }
                if let Ok(at) = crate::io::render_settings::parse_point3(&self.settings.camera_at) {
                    scene.active_camera.params.target = at;
                }
                if let Ok(up) = crate::io::render_settings::parse_vec3(&self.settings.camera_up) {
                    scene.active_camera.params.up = up.normalize();
                }
                if let crate::geometry::camera::ProjectionType::Perspective {
                    fov_y_degrees, ..
                } = &mut scene.active_camera.params.projection
                {
                    *fov_y_degrees = self.settings.camera_fov;
                }
                scene.active_camera.update_matrices();

                // 3. 物体变换同步
                scene.update_object_transform(&self.settings);

                // 🔥 **4. 材质参数同步 - 新增的关键逻辑！**
                if let Some(model_data) = &mut self.model_data {
                    // 同步PBR材质参数
                    if self.settings.use_pbr {
                        // 使用现有的 apply_pbr_parameters 函数
                        crate::material_system::materials::material_applicator::apply_pbr_parameters(
                            model_data,
                            &self.settings
                        );
                    }

                    // 同步Phong材质参数
                    if self.settings.use_phong {
                        // 使用现有的 apply_phong_parameters 函数
                        crate::material_system::materials::material_applicator::apply_phong_parameters(
                            model_data,
                            &self.settings
                        );
                    }

                    // 🔥 **重要：将更新后的材质同步到场景对象**
                    scene.object.model_data = model_data.clone();
                }

                // 5. 执行渲染
                self.renderer.render_scene(scene, &self.settings);
            }

            self.display_render_result(ctx);
            self.interface_interaction.anything_changed = false;
        }
    }

    /// 保存当前渲染结果为截图
    fn take_screenshot(&mut self) -> Result<String, String> {
        // 确保输出目录存在
        if let Err(e) = fs::create_dir_all(&self.settings.output_dir) {
            return Err(format!("创建输出目录失败: {}", e));
        }

        // 生成唯一的文件名（基于时间戳）
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| format!("获取时间戳失败: {}", e))?
            .as_secs();

        let snapshot_name = format!("{}_snapshot_{}", self.settings.output, timestamp);

        // 检查是否有可用的渲染结果
        if self.rendered_image.is_none() {
            return Err("没有可用的渲染结果".to_string());
        }

        // 使用共享的渲染工具函数保存截图
        save_render_with_settings(&self.renderer, &self.settings, Some(&snapshot_name))?;

        // 返回颜色图像的路径
        let color_path =
            Path::new(&self.settings.output_dir).join(format!("{}_color.png", snapshot_name));
        Ok(color_path.to_string_lossy().to_string())
    }

    // === 🔥 **状态管理实现** ===

    /// 设置错误信息
    fn set_error(&mut self, message: String) {
        error!("{}", message);
        self.status_message = format!("错误: {}", message);
    }

    /// 重置应用状态到默认值
    fn reset_to_defaults(&mut self) {
        // 保留当前的文件路径设置
        let obj_path = self.settings.obj.clone();
        let output_dir = self.settings.output_dir.clone();
        let output_name = self.settings.output.clone();

        // 🔥 **修复 Clippy 警告：使用结构体初始化语法**
        let new_settings = crate::io::render_settings::RenderSettings {
            obj: obj_path,
            output_dir,
            output: output_name,
            ..Default::default()
        };

        // 如果渲染尺寸变化，重新创建渲染器
        if self.renderer.frame_buffer.width != new_settings.width
            || self.renderer.frame_buffer.height != new_settings.height
        {
            self.renderer =
                crate::core::renderer::Renderer::new(new_settings.width, new_settings.height);
            self.rendered_image = None;
        }

        self.settings = new_settings;

        // 🔥 **直接内联：从settings初始化GUI变换字段**
        if let Ok(pos) = crate::io::render_settings::parse_vec3(&self.settings.object_position) {
            self.object_position_vec = pos;
        } else {
            self.object_position_vec = nalgebra::Vector3::new(0.0, 0.0, 0.0);
        }

        if let Ok(rot) = crate::io::render_settings::parse_vec3(&self.settings.object_rotation) {
            self.object_rotation_vec =
                nalgebra::Vector3::new(rot.x.to_radians(), rot.y.to_radians(), rot.z.to_radians());
        } else {
            self.object_rotation_vec = nalgebra::Vector3::new(0.0, 0.0, 0.0);
        }

        if let Ok(scale) = crate::io::render_settings::parse_vec3(&self.settings.object_scale_xyz) {
            self.object_scale_vec = scale;
        } else {
            self.object_scale_vec = nalgebra::Vector3::new(1.0, 1.0, 1.0);
        }

        // 重置GUI状态
        self.camera_pan_sensitivity = 1.0;
        self.camera_orbit_sensitivity = 1.0;
        self.camera_dolly_sensitivity = 1.0;
        self.interface_interaction = InterfaceInteraction::default();

        // 重置其他状态
        self.is_realtime_rendering = false;
        self.is_pre_rendering = false;
        self.is_generating_video = false;
        self.pre_render_mode = false;
        self.animation_time = 0.0;
        self.current_frame_index = 0;
        self.last_frame_time = None;

        // 清空预渲染缓冲区
        if let Ok(mut frames) = self.pre_rendered_frames.lock() {
            frames.clear();
        }

        self.pre_render_progress.store(0, Ordering::SeqCst);
        self.video_progress.store(0, Ordering::SeqCst);

        // 重置 FPS 统计
        self.current_fps = 0.0;
        self.fps_history.clear();
        self.avg_fps = 0.0;

        self.status_message = "已重置应用状态，光源已恢复默认设置".to_string();
    }

    /// 切换预渲染模式
    fn toggle_pre_render_mode(&mut self) {
        // 统一的状态检查
        if self.is_pre_rendering || self.is_generating_video || self.is_realtime_rendering {
            self.status_message = "无法更改渲染模式: 请先停止正在进行的操作".to_string();
            return;
        }

        // 切换模式
        self.pre_render_mode = !self.pre_render_mode;

        if self.pre_render_mode {
            // 确保旋转速度合理
            if self.settings.rotation_speed.abs() < 0.01 {
                self.settings.rotation_speed = 1.0;
            }
            self.status_message = "已启用预渲染模式，开始动画渲染时将预先计算所有帧".to_string();
        } else {
            self.status_message = "已禁用预渲染模式，缓冲区中的预渲染帧仍可使用".to_string();
        }
    }

    /// 清空预渲染帧缓冲区
    fn clear_pre_rendered_frames(&mut self) {
        // 统一的状态检查逻辑
        if self.is_realtime_rendering || self.is_pre_rendering {
            self.status_message = "无法清除缓冲区: 请先停止动画渲染或等待预渲染完成".to_string();
            return;
        }

        // 执行清除操作
        let had_frames = !self.pre_rendered_frames.lock().unwrap().is_empty();
        if had_frames {
            self.pre_rendered_frames.lock().unwrap().clear();
            self.current_frame_index = 0;
            self.pre_render_progress.store(0, Ordering::SeqCst);

            if self.is_generating_video {
                let (_, _, frames_per_rotation) =
                    crate::utils::render_utils::calculate_rotation_parameters(
                        self.settings.rotation_speed,
                        self.settings.fps,
                    );
                let total_frames =
                    (frames_per_rotation as f32 * self.settings.rotation_cycles) as usize;
                let progress = self.video_progress.load(Ordering::SeqCst);
                let percent = (progress as f32 / total_frames as f32 * 100.0).round();

                self.status_message = format!(
                    "生成视频中... ({}/{}，{:.0}%)",
                    progress, total_frames, percent
                );
            } else {
                self.status_message = "已清空预渲染缓冲区".to_string();
            }
        } else {
            self.status_message = "缓冲区已为空".to_string();
        }
    }

    // === 🔥 **状态查询实现** ===

    fn can_clear_buffer(&self) -> bool {
        !self.pre_rendered_frames.lock().unwrap().is_empty()
            && !self.is_realtime_rendering
            && !self.is_pre_rendering
    }

    fn can_toggle_pre_render(&self) -> bool {
        !self.is_pre_rendering && !self.is_generating_video && !self.is_realtime_rendering
    }

    fn can_render_animation(&self) -> bool {
        !self.is_generating_video
    }

    fn can_generate_video(&self) -> bool {
        !self.is_realtime_rendering && !self.is_generating_video && self.ffmpeg_available
    }

    // === 🔥 **动画状态管理实现** ===

    fn start_animation_rendering(&mut self) -> Result<(), String> {
        if self.is_generating_video {
            return Err("无法开始动画: 视频正在生成中".to_string());
        }

        self.is_realtime_rendering = true;
        self.last_frame_time = None;
        self.current_fps = 0.0;
        self.fps_history.clear();
        self.avg_fps = 0.0;
        self.status_message = "开始动画渲染...".to_string();

        Ok(())
    }

    fn stop_animation_rendering(&mut self) {
        self.is_realtime_rendering = false;
        self.status_message = "已停止动画渲染".to_string();
    }

    // === 🔥 **性能统计实现** ===

    fn update_fps_stats(&mut self, frame_time: Duration) {
        const FPS_HISTORY_SIZE: usize = 30;
        let current_fps = 1.0 / frame_time.as_secs_f32();
        self.current_fps = current_fps;

        // 更新 FPS 历史
        self.fps_history.push(current_fps);
        if self.fps_history.len() > FPS_HISTORY_SIZE {
            self.fps_history.remove(0); // 移除最早的记录
        }

        // 计算平均 FPS
        if !self.fps_history.is_empty() {
            let sum: f32 = self.fps_history.iter().sum();
            self.avg_fps = sum / self.fps_history.len() as f32;
        }
    }

    fn get_fps_display(&self) -> (String, Color32) {
        // 根据 FPS 水平选择颜色
        let fps_color = if self.avg_fps >= 30.0 {
            Color32::from_rgb(50, 220, 50) // 绿色
        } else if self.avg_fps >= 15.0 {
            Color32::from_rgb(220, 180, 50) // 黄色
        } else {
            Color32::from_rgb(220, 50, 50) // 红色
        };

        (format!("FPS: {:.1}", self.avg_fps), fps_color)
    }

    // === 🔥 **资源管理实现** ===

    fn cleanup_resources(&mut self) {
        // 🔥 **实际的资源清理逻辑**

        // 1. 限制FPS历史记录大小，防止内存泄漏
        if self.fps_history.len() > 60 {
            self.fps_history.drain(0..30); // 保留最近30帧的数据
        }

        // 2. 清理已完成的视频生成线程
        if let Some(handle) = &self.video_generation_thread {
            if handle.is_finished() {
                // 线程已完成，标记需要在主循环中处理
                debug!("检测到已完成的视频生成线程，等待主循环处理");
            }
        }

        // 3. 在空闲状态下进行额外清理
        if !self.is_realtime_rendering && !self.is_generating_video && !self.is_pre_rendering {
            // 清理可能的临时资源
            if self.rendered_image.is_some() && self.last_render_time.is_none() {
                // 如果有渲染结果但没有最近的渲染时间，说明可能是陈旧的结果
                // 这里可以添加更多清理逻辑
            }

            // 清理预渲染进度计数器（如果没有预渲染帧）
            if self.pre_rendered_frames.lock().unwrap().is_empty() {
                self.pre_render_progress.store(0, Ordering::SeqCst);
            }
        }
    }
}
