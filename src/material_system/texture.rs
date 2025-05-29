// texture_utils.rs
// 统一的纹理抽象，支持从文件加载的纹理和面颜色纹理

use image::{DynamicImage, GenericImageView};
use log::warn;
use nalgebra::Vector3;
use std::path::Path;
use std::sync::Arc;

/// 统一的纹理类型枚举，支持多种纹理来源
#[derive(Debug, Clone)]
pub enum TextureData {
    /// 从文件加载的图像纹理
    Image(Arc<DynamicImage>),
    /// 面颜色纹理（每个面一个颜色，使用面索引作为种子）
    FaceColor(u64),
    /// 简单的单色纹理
    SolidColor(Vector3<f32>),
}

/// 纹理抽象，统一封装不同类型的纹理
#[derive(Debug, Clone)]
pub struct Texture {
    /// 纹理数据（可能是图像、面颜色或单色）
    pub data: TextureData,
    /// 纹理宽度（像素）
    pub width: u32,
    /// 纹理高度（像素）
    pub height: u32,
}

impl Texture {
    /// 从文件路径创建纹理
    pub fn from_file<P: AsRef<Path>>(path: P) -> Option<Self> {
        match image::open(path) {
            Ok(img) => {
                let width = img.width();
                let height = img.height();
                Some(Texture {
                    data: TextureData::Image(Arc::new(img)),
                    width,
                    height,
                })
            }
            Err(e) => {
                warn!("无法加载纹理: {}", e);
                None
            }
        }
    }

    /// 创建面颜色纹理
    pub fn face_color() -> Self {
        // 使用当前时间作为随机种子
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Texture {
            data: TextureData::FaceColor(seed),
            width: 1, // 面颜色纹理不需要真实的宽高
            height: 1,
        }
    }

    /// 创建单色纹理
    pub fn solid_color(color: Vector3<f32>) -> Self {
        Texture {
            data: TextureData::SolidColor(color),
            width: 1,
            height: 1,
        }
    }

    /// 获取纹理类型的描述字符串
    pub fn get_type_description(&self) -> &'static str {
        match &self.data {
            TextureData::Image(_) => "图像纹理",
            TextureData::FaceColor(_) => "面颜色纹理",
            TextureData::SolidColor(_) => "单色纹理",
        }
    }

    /// 采样纹理，返回颜色
    /// 统一的采样方法，支持所有纹理类型
    /// 对于面颜色纹理，如果提供了面索引参数，则使用该索引生成颜色
    pub fn sample(&self, u: f32, v: f32) -> [f32; 3] {
        match &self.data {
            TextureData::Image(img) => {
                // 规范化UV坐标到[0,1]范围
                let u = u.fract().abs();
                let v = v.fract().abs();

                // 计算图像坐标
                let x = (u * self.width as f32) as u32;
                // 翻转Y坐标 - 将v从纹理坐标系(下为0)转换为图像坐标系(上为0)
                let y = ((1.0 - v) * self.height as f32) as u32;

                // 处理环绕（防止越界）
                let x = x % self.width;
                let y = y % self.height;

                // 采样像素颜色（这是sRGB空间的颜色）
                let pixel = img.get_pixel(x, y);
                let srgb_color = crate::material_system::color::Color::new(
                    pixel[0] as f32 / 255.0,
                    pixel[1] as f32 / 255.0,
                    pixel[2] as f32 / 255.0,
                );

                // 将sRGB转换为线性RGB空间（用于正确的光照计算）
                let linear_color = crate::material_system::color::srgb_to_linear(&srgb_color);
                [linear_color.x, linear_color.y, linear_color.z]
            }
            TextureData::SolidColor(color) => [color.x, color.y, color.z],
            TextureData::FaceColor(seed) => {
                // 直接使用种子（可能是面索引）生成颜色
                let color = crate::material_system::color::get_random_color(*seed, true);
                [color.x, color.y, color.z]
            }
        }
    }

    /// 采样法线贴图，返回法线向量
    /// 专门用于法线贴图采样，将RGB值转换为法线向量
    pub fn sample_normal(&self, u: f32, v: f32) -> [f32; 3] {
        match &self.data {
            TextureData::Image(img) => {
                // 规范化UV坐标到[0,1]范围
                let u = u.fract().abs();
                let v = v.fract().abs();

                // 计算图像坐标
                let x = (u * self.width as f32) as u32;
                // 翻转Y坐标 - 将v从纹理坐标系(下为0)转换为图像坐标系(上为0)
                let y = ((1.0 - v) * self.height as f32) as u32;

                // 处理环绕（防止越界）
                let x = x % self.width;
                let y = y % self.height;

                // 采样像素颜色，直接使用原始RGB值（不进行sRGB转换）
                let pixel = img.get_pixel(x, y);

                // 将[0,255]映射到[-1,1]
                let normal_x = (pixel[0] as f32 / 255.0) * 2.0 - 1.0;
                let normal_y = (pixel[1] as f32 / 255.0) * 2.0 - 1.0;
                let normal_z = (pixel[2] as f32 / 255.0) * 2.0 - 1.0;

                [normal_x, normal_y, normal_z]
            }
            TextureData::SolidColor(_) => {
                // 单色纹理作为法线贴图时返回默认法线(0,0,1)
                [0.0, 0.0, 1.0]
            }
            TextureData::FaceColor(_) => {
                // 面颜色纹理作为法线贴图时返回默认法线(0,0,1)
                [0.0, 0.0, 1.0]
            }
        }
    }

    /// 检查是否为面颜色纹理
    pub fn is_face_color(&self) -> bool {
        matches!(&self.data, TextureData::FaceColor(_))
    }
}

/// 辅助函数 - 从文件加载或创建默认纹理
pub fn load_texture<P: AsRef<Path>>(path: P, default_color: Vector3<f32>) -> Texture {
    if let Some(texture) = Texture::from_file(path) {
        texture
    } else {
        warn!("无法加载纹理，使用默认颜色 {:?}", default_color);
        Texture::solid_color(default_color)
    }
}
