use crate::io::loaders::{load_background_image, load_obj_enhanced};
use crate::io::render_settings::RenderSettings;
use crate::material_system::texture::Texture;
use crate::scene::scene_utils::Scene;
use crate::utils::model_utils::normalize_and_center_model;
use std::path::Path;
use std::time::Instant;

/// 资源加载器结构体
/// 用于统一处理模型、纹理和背景图片等资源的加载
pub struct ResourceLoader;

impl ResourceLoader {
    /// 加载OBJ模型并创建场景
    ///
    /// 整合了加载模型、归一化和创建场景的完整流程
    pub fn load_model_and_create_scene(
        obj_path: &str,
        settings: &RenderSettings,
    ) -> Result<(Scene, crate::material_system::materials::ModelData), String> {
        println!("加载模型：{}", obj_path);
        let load_start = Instant::now();

        // 检查文件是否存在
        if !Path::new(obj_path).exists() {
            return Err(format!("错误：输入的 OBJ 文件未找到：{}", obj_path));
        }

        // 加载模型数据
        let mut model_data = load_obj_enhanced(obj_path, settings)?;
        println!("模型加载耗时 {:?}", load_start.elapsed());

        // 归一化模型
        println!("归一化模型...");
        let norm_start_time = Instant::now();
        let (original_center, scale_factor) = normalize_and_center_model(&mut model_data);
        println!(
            "模型归一化耗时 {:?}。原始中心：{:.3?}，缩放系数：{:.3}",
            norm_start_time.elapsed(),
            original_center,
            scale_factor
        );

        // 创建场景
        println!("创建场景...");
        let scene = Scene::create_from_model_and_settings(model_data.clone(), settings)?;

        Ok((scene, model_data))
    }

    /// 加载背景图片
    ///
    /// 如果成功加载，更新settings中的background_image字段
    pub fn load_background_image_if_enabled(settings: &mut RenderSettings) -> Result<(), String> {
        if settings.use_background_image {
            if let Some(bg_path) = &settings.background_image_path {
                match load_background_image(bg_path) {
                    Ok(texture) => {
                        settings.background_image = Some(texture);
                        println!("背景图片加载成功: {}", bg_path);
                        Ok(())
                    }
                    Err(e) => {
                        let error_msg = format!("警告: 背景图片加载失败: {}", e);
                        println!("{}", error_msg);
                        // 返回错误但不中断流程
                        Err(error_msg)
                    }
                }
            } else {
                let error_msg = "警告: 启用了背景图片但未指定背景图片路径".to_string();
                println!("{}", error_msg);
                Err(error_msg)
            }
        } else {
            // 背景图片未启用，直接返回成功
            Ok(())
        }
    }

    /// 从路径加载单个背景图片
    ///
    /// 成功时返回加载的纹理
    pub fn load_background_image_from_path(path: &str) -> Result<Texture, String> {
        load_background_image(path)
    }
}
