use crate::ui::app::RasterizerApp;
use egui::{Color32, ColorImage};
use std::sync::atomic::Ordering;
use std::time::Duration;

/// 核心UI状态转换与管理相关方法
///
/// 该trait包含应用状态的核心管理功能，包括：
/// - 错误处理
/// - 状态重置与转换
/// - 状态查询
/// - 动画渲染控制
/// - 性能统计
/// - 资源管理
pub trait CoreMethods {
    // 状态转换与错误处理

    /// 设置错误信息
    ///
    /// # 副作用
    /// - 更新状态信息
    /// - 打印错误日志
    fn set_error(&mut self, message: String);

    /// 将应用状态重置为默认值
    ///
    /// # 副作用
    /// - 停止所有正在进行的渲染任务
    /// - 清空预渲染缓冲区
    /// - 重置性能统计信息
    fn reset_to_defaults(&mut self);

    /// 切换预渲染模式开启/关闭状态
    ///
    /// # 副作用
    /// - 如果开启预渲染模式且旋转速度过低，会自动设置合理的旋转速度
    fn toggle_pre_render_mode(&mut self);

    /// 清空预渲染的动画帧缓冲区
    ///
    /// # 副作用
    /// - 释放预渲染帧占用的内存
    /// - 重置帧索引和进度
    fn clear_pre_rendered_frames(&mut self);

    // 状态查询辅助函数

    /// 检查是否可以清除预渲染缓冲区
    ///
    /// 当满足以下条件时返回true:
    /// - 缓冲区中有预渲染帧
    /// - 不在实时渲染状态
    /// - 不在预渲染过程中
    fn can_clear_buffer(&self) -> bool;

    /// 检查是否可以切换预渲染模式
    ///
    /// 当满足以下条件时返回true:
    /// - 不在预渲染过程中
    /// - 不在视频生成过程中
    /// - 不在实时渲染过程中
    fn can_toggle_pre_render(&self) -> bool;

    /// 检查是否可以开始或停止动画渲染
    ///
    /// 当不在视频生成过程中时返回true
    fn can_render_animation(&self) -> bool;

    /// 检查是否可以生成视频
    ///
    /// 当满足以下条件时返回true:
    /// - 不在实时渲染过程中
    /// - 不在视频生成过程中
    /// - ffmpeg可用
    fn can_generate_video(&self) -> bool;

    // 动画状态管理

    /// 开始实时渲染动画
    ///
    /// # 副作用
    /// - 设置渲染状态标志
    /// - 重置性能统计信息
    ///
    /// # 错误
    /// - 如果正在生成视频，返回错误
    fn start_animation_rendering(&mut self) -> Result<(), String>;

    /// 停止实时渲染动画
    ///
    /// # 副作用
    /// - 清除渲染状态标志
    /// - 更新状态信息
    fn stop_animation_rendering(&mut self);

    // 性能统计

    /// 更新帧率统计信息
    ///
    /// # 参数
    /// - `frame_time`: 渲染单帧所用的时间
    ///
    /// # 副作用
    /// - 更新FPS历史记录
    /// - 更新平均帧率
    fn update_fps_stats(&mut self, frame_time: Duration);

    /// 获取格式化的帧率显示文本和颜色
    ///
    /// 返回元组 (文本, 颜色)，根据帧率的高低使用不同的颜色：
    /// - 高帧率（>=30FPS）: 绿色
    /// - 中等帧率（>=15FPS）: 黄色
    /// - 低帧率（<15FPS）: 红色
    fn get_fps_display(&self) -> (String, Color32);

    // 资源管理

    /// 执行资源清理操作
    ///
    /// 当应用处于空闲状态时，清理不需要的资源，释放内存。
    /// 应该定期调用此方法，如在应用的update循环中。
    ///
    /// # 副作用
    /// - 在适当条件下清空预渲染缓冲区
    fn cleanup_resources(&mut self);
}

impl CoreMethods for RasterizerApp {
    /// 设置错误信息
    fn set_error(&mut self, message: String) {
        eprintln!("错误: {}", message);
        self.status_message = format!("错误: {}", message);
        // 显示错误对话框在app.rs中单独处理
    }
    /// 重置应用状态到默认值
    fn reset_to_defaults(&mut self) {
        self.is_realtime_rendering = false;
        self.is_pre_rendering = false;
        self.is_generating_video = false;
        self.pre_render_mode = false;
        // 错误状态在app.rs中单独处理
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

        self.status_message = "已重置应用状态".to_string();
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

            // 检查是否正在生成视频，如果是，更新状态消息
            if self.is_generating_video {
                // 计算旋转参数，获取视频帧数
                let (_, _, frames_per_rotation) =
                    crate::utils::render_utils::calculate_rotation_parameters(
                        self.settings.rotation_speed,
                        self.settings.fps,
                    );
                let total_frames =
                    (frames_per_rotation as f32 * self.settings.rotation_cycles) as usize;
                let progress = self.video_progress.load(Ordering::SeqCst);
                let percent = (progress as f32 / total_frames as f32 * 100.0).round();

                // 更新状态消息，不再区分是否使用预渲染帧
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

    /// 检查是否可以清除预渲染缓冲区
    fn can_clear_buffer(&self) -> bool {
        !self.pre_rendered_frames.lock().unwrap().is_empty()
            && !self.is_realtime_rendering
            && !self.is_pre_rendering
    }

    /// 检查是否可以切换预渲染模式
    fn can_toggle_pre_render(&self) -> bool {
        !self.is_pre_rendering && !self.is_generating_video && !self.is_realtime_rendering
    }

    /// 检查是否可以渲染动画
    fn can_render_animation(&self) -> bool {
        !self.is_generating_video
    }

    /// 检查是否可以生成视频
    fn can_generate_video(&self) -> bool {
        !self.is_realtime_rendering && !self.is_generating_video && self.ffmpeg_available
    }

    /// 开始实时渲染动画
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

    /// 停止实时渲染动画
    fn stop_animation_rendering(&mut self) {
        self.is_realtime_rendering = false;
        self.status_message = "已停止动画渲染".to_string();
    }
    /// 更新 FPS 统计信息
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

    /// 获取 FPS 显示文本和颜色
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
    /// 执行资源清理操作
    ///
    /// 定期清理不再需要的资源以优化内存使用：
    /// - 可以在这里添加其他资源清理逻辑
    fn cleanup_resources(&mut self) {
        // 仅在应用空闲状态下执行清理，避免影响活跃的渲染过程
        if !self.is_realtime_rendering && !self.is_generating_video && !self.is_pre_rendering {
            // 我们不再自动清理预渲染帧，让用户自己决定何时清理
            // 可以在这里添加其他资源清理逻辑，如:
            // - 清理未使用的纹理
            // - 释放不再需要的大型模型数据
            // - 优化内存分配
        }
    }
}

/// 辅助函数：将ColorImage转换为PNG格式的字节数组
///
/// # 参数
/// * `image` - egui的ColorImage对象引用
///
/// # 返回值
/// 转换为RGB格式的字节数组，可用于保存PNG图像
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
