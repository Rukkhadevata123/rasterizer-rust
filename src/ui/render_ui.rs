use crate::ResourceLoader;
use crate::ui::app::RasterizerApp;
use native_dialog::FileDialogBuilder;

/// 渲染UI交互方法的特质
///
/// 该trait专门处理与文件选择和UI交互相关的功能：
/// - 文件选择对话框
/// - 背景图片处理
/// - 输出目录选择
pub trait RenderUIMethods {
    /// 选择OBJ文件
    fn select_obj_file(&mut self);

    /// 选择背景图片
    fn select_background_image(&mut self);

    /// 选择输出目录
    fn select_output_dir(&mut self);
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

    /// 选择背景图片
    fn select_background_image(&mut self) {
        let result = FileDialogBuilder::default()
            .set_title("选择背景图片")
            .add_filter("图片文件", ["png", "jpg", "jpeg", "bmp"])
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
