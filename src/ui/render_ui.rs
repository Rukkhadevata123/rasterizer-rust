use crate::io::config_loader::TomlConfigLoader;
use crate::io::model_loader::ModelLoader;
use crate::ui::app::RasterizerApp;
use native_dialog::FileDialogBuilder;

/// 渲染UI交互方法的特质
///
/// 该trait专门处理与文件选择和UI交互相关的功能：
/// - 文件选择对话框
/// - 背景图片处理
/// - 输出目录选择
/// - 配置文件管理
pub trait RenderUIMethods {
    /// 选择OBJ文件
    fn select_obj_file(&mut self);

    /// 选择纹理文件
    fn select_texture_file(&mut self);

    /// 选择背景图片
    fn select_background_image(&mut self);

    /// 选择输出目录
    fn select_output_dir(&mut self);

    /// 🔥 **加载配置文件**
    fn load_config_file(&mut self);

    /// 🔥 **保存配置文件**
    fn save_config_file(&mut self);

    /// 🔥 **应用加载的配置到GUI**
    fn apply_loaded_config(&mut self, settings: crate::io::render_settings::RenderSettings);
}

impl RenderUIMethods for RasterizerApp {
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

                    // 🔥 **新增：OBJ文件变化需要重新加载场景和重新渲染**
                    self.interface_interaction.anything_changed = true;
                    self.scene = None; // 清除现有场景，强制重新加载
                    self.rendered_image = None; // 清除渲染结果
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

    /// 选择纹理文件
    fn select_texture_file(&mut self) {
        let result = FileDialogBuilder::default()
            .set_title("选择纹理文件")
            .add_filter("图像文件", ["png", "jpg", "jpeg", "bmp", "tga"])
            .open_single_file()
            .show();

        match result {
            Ok(Some(path)) => {
                if let Some(path_str) = path.to_str() {
                    self.settings.texture = Some(path_str.to_string());
                    self.status_message = format!("已选择纹理: {}", path_str);

                    // 🔥 **纹理变化需要重新渲染**
                    self.interface_interaction.anything_changed = true;
                }
            }
            Ok(None) => {
                self.status_message = "纹理选择被取消".to_string();
            }
            Err(e) => {
                self.set_error(format!("纹理选择错误: {}", e));
            }
        }
    }

    /// 🔥 **修复：选择背景图片** - 适配新的背景管理架构
    fn select_background_image(&mut self) {
        let result = FileDialogBuilder::default()
            .set_title("选择背景图片")
            .add_filter("图片文件", ["png", "jpg", "jpeg", "bmp"])
            .open_single_file()
            .show();

        match result {
            Ok(Some(path)) => {
                if let Some(path_str) = path.to_str() {
                    // 🔥 **只设置背景图片路径，不再直接加载到 settings**
                    self.settings.background_image_path = Some(path_str.to_string());
                    self.settings.use_background_image = true;

                    // 🔥 **使用 ModelLoader 验证背景图片是否有效**
                    match ModelLoader::validate_resources(&self.settings) {
                        Ok(_) => {
                            self.status_message = format!("背景图片配置成功: {}", path_str);

                            // 🔥 **清除已渲染的图像，强制重新渲染以应用新背景**
                            self.rendered_image = None;

                            println!("背景图片路径已设置: {}", path_str);
                            println!("背景图片将在下次渲染时由 FrameBuffer 自动加载");
                        }
                        Err(e) => {
                            // 🔥 **验证失败，重置背景设置**
                            self.set_error(format!("背景图片验证失败: {}", e));
                            self.settings.background_image_path = None;
                            self.settings.use_background_image = false;
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

    /// 🔥 **加载配置文件**
    fn load_config_file(&mut self) {
        let result = FileDialogBuilder::default()
            .set_title("加载配置文件")
            .add_filter("TOML配置文件", ["toml"])
            .open_single_file()
            .show();

        match result {
            Ok(Some(path)) => {
                if let Some(path_str) = path.to_str() {
                    match TomlConfigLoader::load_from_file(path_str) {
                        Ok(loaded_settings) => {
                            self.apply_loaded_config(loaded_settings);
                            self.status_message = format!("配置已加载: {}", path_str);
                        }
                        Err(e) => {
                            self.set_error(format!("配置加载失败: {}", e));
                        }
                    }
                }
            }
            Ok(None) => {
                self.status_message = "配置加载被取消".to_string();
            }
            Err(e) => {
                self.set_error(format!("文件选择器错误: {}", e));
            }
        }
    }

    /// 🔥 **保存配置文件**
    fn save_config_file(&mut self) {
        let result = FileDialogBuilder::default()
            .set_title("保存配置文件")
            .add_filter("TOML配置文件", ["toml"])
            .save_single_file()
            .show();

        match result {
            Ok(Some(path)) => {
                let mut save_path = path;

                // 自动添加.toml扩展名（如果没有）
                if save_path.extension().is_none() {
                    save_path.set_extension("toml");
                }

                if let Some(path_str) = save_path.to_str() {
                    match TomlConfigLoader::save_to_file(&self.settings, path_str) {
                        Ok(_) => {
                            self.status_message = format!("配置已保存: {}", path_str);
                        }
                        Err(e) => {
                            self.set_error(format!("配置保存失败: {}", e));
                        }
                    }
                }
            }
            Ok(None) => {
                self.status_message = "配置保存被取消".to_string();
            }
            Err(e) => {
                self.set_error(format!("文件选择器错误: {}", e));
            }
        }
    }

    /// 🔥 **应用加载的配置到GUI**
    fn apply_loaded_config(&mut self, loaded_settings: crate::io::render_settings::RenderSettings) {
        // 保存旧的settings
        self.settings = loaded_settings;

        // 🔥 **重新同步GUI专用向量字段**
        self.object_position_vec = if let Ok(pos) =
            crate::io::render_settings::parse_vec3(&self.settings.object_position)
        {
            pos
        } else {
            nalgebra::Vector3::new(0.0, 0.0, 0.0)
        };

        self.object_rotation_vec = if let Ok(rot) =
            crate::io::render_settings::parse_vec3(&self.settings.object_rotation)
        {
            nalgebra::Vector3::new(rot.x.to_radians(), rot.y.to_radians(), rot.z.to_radians())
        } else {
            nalgebra::Vector3::new(0.0, 0.0, 0.0)
        };

        self.object_scale_vec = if let Ok(scale) =
            crate::io::render_settings::parse_vec3(&self.settings.object_scale_xyz)
        {
            scale
        } else {
            nalgebra::Vector3::new(1.0, 1.0, 1.0)
        };

        // 🔥 **如果分辨率变化，重新创建渲染器**
        if self.renderer.frame_buffer.width != self.settings.width
            || self.renderer.frame_buffer.height != self.settings.height
        {
            self.renderer =
                crate::core::renderer::Renderer::new(self.settings.width, self.settings.height);
        }

        // 🔥 **清除现有场景和渲染结果，强制重新加载**
        self.scene = None;
        self.rendered_image = None;
        self.interface_interaction.anything_changed = true;

        println!("配置已应用到GUI界面");
    }
}
