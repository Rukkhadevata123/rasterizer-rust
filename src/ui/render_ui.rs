use crate::ResourceLoader;
use crate::utils::save_utils::save_render_with_settings;
use egui::{Color32, Context};
use native_dialog::FileDialogBuilder;
use std::fs;
use std::path::Path;
use std::time::Instant;

use super::app::RasterizerApp;

/// 渲染相关方法的特质
pub trait RenderMethods {
    /// 验证渲染参数
    fn validate_parameters(&self) -> Result<(), String>;

    /// 渲染当前场景
    fn render(&mut self, ctx: &Context);

    /// 加载模型并设置场景
    fn load_model(&mut self, obj_path: &str) -> Result<(), String>;

    /// 在UI中显示渲染结果
    fn display_render_result(&mut self, ctx: &Context);

    /// 保存当前渲染结果为截图
    fn take_screenshot(&mut self) -> Result<String, String>;

    /// 选择OBJ文件
    fn select_obj_file(&mut self);

    /// 选择背景图片
    fn select_background_image(&mut self);

    /// 选择输出目录
    fn select_output_dir(&mut self);
}

impl RenderMethods for RasterizerApp {
    /// 验证渲染参数 - 直接调用RenderSettings的validate方法
    fn validate_parameters(&self) -> Result<(), String> {
        // 直接调用RenderSettings中的validate方法，消除代码重复
        self.settings.validate()
    }

    /// 渲染当前场景
    fn render(&mut self, ctx: &egui::Context) {
        // 验证参数
        match self.validate_parameters() {
            Ok(_) => {
                // 参数验证通过，继续渲染流程
                // 获取OBJ路径
                let obj_path = match &self.settings.obj {
                    Some(path) => path.clone(),
                    None => {
                        self.set_error("错误: 未指定OBJ文件路径".to_string());
                        return;
                    }
                };

                self.status_message = format!("正在加载 {}...", obj_path);
                // 请求重绘
                ctx.request_repaint(); // 立即更新状态消息

                // 加载模型
                match self.load_model(&obj_path) {
                    Ok(_) => {
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

                        if let Some(scene) = &self.scene {
                            // 更新渲染设置的颜色向量和光源
                            self.settings.update_color_vectors();
                            self.settings.update_from_scene(scene);

                            println!(
                                "环境光: 强度={}, 颜色={:?}",
                                self.settings.ambient, self.settings.ambient_color_vec
                            );

                            // 渲染到帧缓冲区 - 直接使用场景和设置
                            self.renderer.render_scene(scene, &self.settings);

                            // 保存输出文件
                            if let Err(e) =
                                save_render_with_settings(&self.renderer, &self.settings, None)
                            {
                                println!("警告：保存渲染结果时发生错误: {}", e);
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

                            // 可选: 在UI中显示渲染结果
                            self.display_render_result(ctx);
                        }
                    }
                    Err(e) => {
                        self.set_error(format!("加载模型失败: {}", e));
                    }
                }
            }
            Err(e) => {
                // 参数验证失败，显示错误
                self.set_error(e);
            }
        }
    }

    /// 加载模型并设置场景
    fn load_model(&mut self, obj_path: &str) -> Result<(), String> {
        // 使用ResourceLoader加载模型和创建场景
        let (scene, model_data) =
            ResourceLoader::load_model_and_create_scene(obj_path, &self.settings)?;

        // 保存场景和模型数据
        self.scene = Some(scene);
        self.model_data = Some(model_data);

        // 使用ResourceLoader加载背景图片
        if self.settings.use_background_image {
            if let Err(e) = ResourceLoader::load_background_image_if_enabled(&mut self.settings) {
                println!("背景图片加载问题: {}", e);
                // 继续执行，不中断加载过程
            }
        }

        Ok(())
    }

    /// 选择背景图片
    fn select_background_image(&mut self) {
        let result = FileDialogBuilder::default()
            .set_title("选择背景图片")
            .add_filter("图片文件", &["png", "jpg", "jpeg", "bmp"])
            .open_single_file()
            .show();

        match result {
            Ok(Some(path)) => {
                if let Some(path_str) = path.to_str() {
                    // 设置背景图片路径
                    self.settings.background_image_path = Some(path_str.to_string());
                    self.status_message = format!("已选择背景图片: {}", path_str);

                    // 使用ResourceLoader加载背景图片
                    match ResourceLoader::load_background_image_from_path(path_str) {
                        Ok(texture) => {
                            self.settings.background_image = Some(texture);
                            self.settings.use_background_image = true;
                            self.status_message = format!("背景图片加载成功: {}", path_str);
                        }
                        Err(e) => {
                            self.set_error(format!("背景图片加载失败: {}", e));
                            self.settings.background_image_path = None;
                            self.settings.background_image = None;
                        }
                    }
                }
            }
            Ok(None) => {
                self.status_message = "图片选择被取消".to_string();
            }
            Err(e) => {
                self.set_error(format!("文件选择器错误: {}", e));
            }
        }
    }

    /// 在UI中显示渲染结果
    fn display_render_result(&mut self, ctx: &egui::Context) {
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

        // 确保场景信息已更新到设置中
        if let Some(scene) = &self.scene {
            self.settings.update_from_scene(scene);
        }

        // 使用共享的渲染工具函数保存截图
        save_render_with_settings(&self.renderer, &self.settings, Some(&snapshot_name))?;

        // 返回颜色图像的路径
        let color_path =
            Path::new(&self.settings.output_dir).join(format!("{}_color.png", snapshot_name));
        Ok(color_path.to_string_lossy().to_string())
    }

    /// 选择OBJ文件
    fn select_obj_file(&mut self) {
        let result = FileDialogBuilder::default()
            .set_title("选择OBJ模型文件")
            .add_filter("OBJ模型", ["obj"])
            .open_single_file()
            .show();

        match result {
            Ok(Some(path)) => {
                if let Some(path_str) = path.to_str() {
                    self.settings.obj = Some(path_str.to_string());
                    self.status_message = format!("已选择模型: {}", path_str);
                }
            }
            Ok(None) => {
                self.status_message = "文件选择被取消".to_string();
            }
            Err(e) => {
                self.set_error(format!("文件选择器错误: {}", e));
            }
        }
    }

    /// 选择输出目录
    fn select_output_dir(&mut self) {
        let result = FileDialogBuilder::default()
            .set_title("选择输出目录")
            .open_single_dir()
            .show();

        match result {
            Ok(Some(path)) => {
                if let Some(path_str) = path.to_str() {
                    self.settings.output_dir = path_str.to_string();
                    self.status_message = format!("已选择输出目录: {}", path_str);
                }
            }
            Ok(None) => {
                self.status_message = "目录选择被取消".to_string();
            }
            Err(e) => {
                self.set_error(format!("目录选择器错误: {}", e));
            }
        }
    }
}
