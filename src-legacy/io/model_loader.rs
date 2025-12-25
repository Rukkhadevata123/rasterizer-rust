use crate::io::obj_loader::load_obj_model;
use crate::io::render_settings::RenderSettings;
use crate::material_system::materials::Model;
use crate::scene::scene_utils::Scene;
use crate::utils::model_utils::normalize_and_center_model;
use log::{debug, info};
use std::path::Path;
use std::time::Instant;

/// 模型加载器
pub struct ModelLoader;

impl ModelLoader {
    /// 主要功能：加载OBJ模型并创建场景
    pub fn load_and_create_scene(
        obj_path: &str,
        settings: &RenderSettings,
    ) -> Result<(Scene, Model), String> {
        info!("加载模型：{obj_path}");
        let load_start = Instant::now();

        // 检查文件存在
        if !Path::new(obj_path).exists() {
            return Err(format!("输入的 OBJ 文件未找到：{obj_path}"));
        }

        // 加载模型数据
        let mut model = load_obj_model(obj_path, settings)?;
        debug!("模型加载耗时 {:?}", load_start.elapsed());

        // 归一化模型
        debug!("归一化模型...");
        let norm_start_time = Instant::now();
        let (original_center, scale_factor) = normalize_and_center_model(&mut model);
        debug!(
            "模型归一化耗时 {:?}，原始中心：{:.3?}，缩放系数：{:.3}",
            norm_start_time.elapsed(),
            original_center,
            scale_factor
        );

        // 创建场景
        debug!("创建场景...");
        let scene = Scene::new(model.clone(), settings)?;

        Ok((scene, model))
    }

    /// 验证资源
    pub fn validate_resources(settings: &RenderSettings) -> Result<(), String> {
        // 验证 OBJ 文件
        if let Some(obj_path) = &settings.obj {
            if !Path::new(obj_path).exists() {
                return Err(format!("OBJ 文件不存在: {obj_path}"));
            }
        }

        // 验证背景图片（如果启用）
        if settings.use_background_image {
            if let Some(bg_path) = &settings.background_image_path {
                if !Path::new(bg_path).exists() {
                    return Err(format!("背景图片文件不存在: {bg_path}"));
                }
            } else {
                return Err("启用了背景图片但未指定路径".to_string());
            }
        }

        // 验证纹理文件（如果指定）
        if let Some(texture_path) = &settings.texture {
            if !Path::new(texture_path).exists() {
                return Err(format!("纹理文件不存在: {texture_path}"));
            }
        }

        info!("所有资源验证通过");
        Ok(())
    }
}
