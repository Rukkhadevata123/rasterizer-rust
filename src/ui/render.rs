use crate::core::renderer::RenderConfig;
use crate::core::scene::Scene;
use crate::io::args::{parse_point3, parse_vec3};
use crate::utils::material_utils::{apply_pbr_parameters, apply_phong_parameters};
use crate::utils::model_utils::normalize_and_center_model;
use crate::utils::render_utils::{create_render_config, save_render_with_args};
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

    /// 保存渲染结果
    fn save_render_result(&self, config: &RenderConfig);

    /// 在UI中显示渲染结果
    fn display_render_result(&mut self, ctx: &Context);

    /// 保存当前渲染结果为截图
    fn take_screenshot(&mut self) -> Result<String, String>;

    /// 选择OBJ文件
    fn select_obj_file(&mut self);

    /// 选择输出目录
    fn select_output_dir(&mut self);
}

impl RenderMethods for RasterizerApp {
    /// 验证渲染参数
    fn validate_parameters(&self) -> Result<(), String> {
        // 检查基本参数
        if self.args.width == 0 || self.args.height == 0 {
            return Err("错误: 图像宽度和高度必须大于0".to_string());
        }

        // 检查OBJ文件是否存在
        if !Path::new(&self.args.obj).exists() {
            return Err(format!("错误: 找不到OBJ文件 '{}'", self.args.obj));
        }

        // 检查输出目录和文件名
        if self.args.output_dir.trim().is_empty() {
            return Err("错误: 输出目录不能为空".to_string());
        }

        if self.args.output.trim().is_empty() {
            return Err("错误: 输出文件名不能为空".to_string());
        }

        // 验证相机参数
        if parse_vec3(&self.args.camera_from).is_err() {
            return Err("错误: 相机位置格式不正确，应为 x,y,z 格式".to_string());
        }

        if parse_vec3(&self.args.camera_at).is_err() {
            return Err("错误: 相机目标格式不正确，应为 x,y,z 格式".to_string());
        }

        if parse_vec3(&self.args.camera_up).is_err() {
            return Err("错误: 相机上方向格式不正确，应为 x,y,z 格式".to_string());
        }

        // 验证光照参数
        if self.args.use_lighting {
            if self.args.light_type == "directional" {
                if parse_vec3(&self.args.light_dir).is_err() {
                    return Err("错误: 光源方向格式不正确，应为 x,y,z 格式".to_string());
                }
            } else if self.args.light_type == "point" {
                if parse_vec3(&self.args.light_pos).is_err() {
                    return Err("错误: 光源位置格式不正确，应为 x,y,z 格式".to_string());
                }

                // 验证点光源衰减
                let atten_parts: Vec<&str> = self.args.light_atten.split(',').collect();
                if atten_parts.len() != 3 {
                    return Err(
                        "错误: 光源衰减格式不正确，应为 常数项,线性项,二次项 格式".to_string()
                    );
                }
                for part in atten_parts {
                    if part.trim().parse::<f32>().is_err() {
                        return Err("错误: 光源衰减参数必须是浮点数".to_string());
                    }
                }
            }
        }

        // 验证环境光颜色
        if !self.args.ambient_color.is_empty() && parse_vec3(&self.args.ambient_color).is_err() {
            return Err("错误: 环境光颜色格式不正确，应为 r,g,b 格式".to_string());
        }

        // 验证PBR参数
        if self.args.use_pbr {
            if self.args.metallic < 0.0 || self.args.metallic > 1.0 {
                return Err("错误: 金属度必须在0.0到1.0之间".to_string());
            }
            if self.args.roughness < 0.0 || self.args.roughness > 1.0 {
                return Err("错误: 粗糙度必须在0.0到1.0之间".to_string());
            }
            if !self.args.base_color.is_empty() && parse_vec3(&self.args.base_color).is_err() {
                return Err("错误: 基础颜色格式不正确，应为 r,g,b 格式".to_string());
            }
            if !self.args.emissive.is_empty() && parse_vec3(&self.args.emissive).is_err() {
                return Err("错误: 自发光颜色格式不正确，应为 r,g,b 格式".to_string());
            }
        }

        Ok(())
    }

    /// 渲染当前场景
    fn render(&mut self, ctx: &egui::Context) {
        // 验证参数
        match self.validate_parameters() {
            Ok(_) => {
                // 参数验证通过，继续渲染流程
                // 保存状态消息到临时变量，避免借用冲突
                let obj_path = self.args.obj.clone();
                self.status_message = format!("正在加载 {}...", obj_path);
                // 请求重绘
                ctx.request_repaint(); // 立即更新状态消息

                // 加载模型
                match self.load_model(&obj_path) {
                    Ok(_) => {
                        self.status_message = "模型加载成功，开始渲染...".to_string();
                        ctx.request_repaint();

                        // 确保输出目录存在
                        let output_dir = self.args.output_dir.clone();
                        if let Err(e) = fs::create_dir_all(&output_dir) {
                            self.set_error(format!("创建输出目录失败: {}", e));
                            return;
                        }

                        // 渲染
                        let start_time = Instant::now();

                        if let Some(scene) = &self.scene {
                            // 创建渲染配置
                            let config = create_render_config(scene, &self.args);

                            // 在GUI模式下输出光源信息（用于调试）
                            println!("GUI模式使用光源: {:?}", config.light);
                            println!(
                                "环境光: 强度={}, 颜色={:?}",
                                config.ambient_intensity, config.ambient_color
                            );

                            // 渲染到帧缓冲区 - 确保使用不可变引用避免修改配置
                            // 之前的代码:
                            // let mut config_clone = config.clone();
                            // self.renderer.render_scene(scene, &mut config_clone);

                            // 修改后的代码:
                            self.renderer.render_scene(scene, &mut config.clone());

                            // 保存输出文件
                            self.save_render_result(&config);

                            // 更新状态
                            self.last_render_time = Some(start_time.elapsed());
                            let output_dir = self.args.output_dir.clone();
                            let output_name = self.args.output.clone();
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
        use crate::io::loaders::load_obj_enhanced;

        // 加载模型数据
        let mut model_data = load_obj_enhanced(obj_path, &self.args)?;

        // 应用PBR材质参数
        if self.args.use_pbr {
            println!(
                "GUI模式: 应用PBR材质参数 - 金属度: {}, 粗糙度: {}",
                self.args.metallic, self.args.roughness
            );
            apply_pbr_parameters(&mut model_data, &self.args);
        }

        // 应用Phong材质参数
        if self.args.use_phong {
            println!(
                "GUI模式: 应用Phong材质参数 - 高光系数: {}, 光泽度: {}",
                self.args.specular, self.args.shininess
            );
            apply_phong_parameters(&mut model_data, &self.args);
        }

        // 归一化模型
        let (_center, _scale) = normalize_and_center_model(&mut model_data);

        // 创建相机
        let camera_from =
            parse_point3(&self.args.camera_from).map_err(|e| format!("相机位置格式错误: {}", e))?;
        let camera_at =
            parse_point3(&self.args.camera_at).map_err(|e| format!("相机目标格式错误: {}", e))?;
        let camera_up =
            parse_vec3(&self.args.camera_up).map_err(|e| format!("相机上方向格式错误: {}", e))?;

        let aspect_ratio = self.args.width as f32 / self.args.height as f32;
        let camera = crate::geometry::camera::Camera::new(
            camera_from,
            camera_at,
            camera_up,
            self.args.camera_fov,
            aspect_ratio,
            0.1,
            100.0,
        );

        // 创建场景
        let mut scene = Scene::new(camera);

        // 使用统一方法设置场景对象
        let object_count = if let Some(count_str) = &self.args.object_count {
            count_str.parse::<usize>().ok()
        } else {
            None
        };
        scene.setup_from_model_data(model_data.clone(), object_count);

        // 使用统一方法设置光照系统
        scene.setup_lighting(
            self.args.use_lighting,
            &self.args.light_type,
            &self.args.light_dir,
            &self.args.light_pos,
            &self.args.light_atten,
            self.args.diffuse,
            self.args.ambient,
            &self.args.ambient_color,
        )?;

        self.scene = Some(scene);
        self.model_data = Some(model_data);

        Ok(())
    }

    /// 保存渲染结果
    fn save_render_result(&self, config: &RenderConfig) {
        // 使用共享的渲染工具函数保存渲染结果
        if let Err(e) = save_render_with_args(&self.renderer, &self.args, config, None) {
            println!("警告：保存渲染结果时发生错误: {}", e);
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
        if let Err(e) = fs::create_dir_all(&self.args.output_dir) {
            return Err(format!("创建输出目录失败: {}", e));
        }

        // 生成唯一的文件名（基于时间戳）
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| format!("获取时间戳失败: {}", e))?
            .as_secs();

        let snapshot_name = format!("{}_snapshot_{}", self.args.output, timestamp);

        // 检查是否有可用的渲染结果
        if self.rendered_image.is_none() {
            return Err("没有可用的渲染结果".to_string());
        }

        // 获取当前渲染配置
        let config = if let Some(scene) = &self.scene {
            create_render_config(scene, &self.args)
        } else {
            return Err("无法创建渲染配置".to_string());
        };

        // 使用共享的渲染工具函数保存截图
        save_render_with_args(&self.renderer, &self.args, &config, Some(&snapshot_name))?;

        // 返回颜色图像的路径
        let color_path =
            Path::new(&self.args.output_dir).join(format!("{}_color.png", snapshot_name));
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
                    self.args.obj = path_str.to_string();
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
                    self.args.output_dir = path_str.to_string();
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
