use image::{DynamicImage, GenericImageView};
use log::warn;
use nalgebra::Vector3;
use std::path::Path;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum TextureData {
    Image(Arc<DynamicImage>),
    FaceColor(u64),
    SolidColor(Vector3<f32>),
}

#[derive(Debug, Clone)]
pub struct Texture {
    pub data: TextureData,
    pub width: u32,
    pub height: u32,
}

impl Texture {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Option<Self> {
        match image::open(path) {
            Ok(img) => Some(Texture {
                width: img.width(),
                height: img.height(),
                data: TextureData::Image(Arc::new(img)),
            }),
            Err(e) => {
                warn!("无法加载纹理: {}", e);
                None
            }
        }
    }

    pub fn face_color() -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Texture {
            data: TextureData::FaceColor(seed),
            width: 1,
            height: 1,
        }
    }

    pub fn solid_color(color: Vector3<f32>) -> Self {
        Texture {
            data: TextureData::SolidColor(color),
            width: 1,
            height: 1,
        }
    }

    pub fn get_type_description(&self) -> &'static str {
        match &self.data {
            TextureData::Image(_) => "图像纹理",
            TextureData::FaceColor(_) => "面颜色纹理",
            TextureData::SolidColor(_) => "单色纹理",
        }
    }

    pub fn sample(&self, u: f32, v: f32) -> [f32; 3] {
        match &self.data {
            TextureData::Image(img) => {
                let u = u.fract().abs();
                let v = v.fract().abs();
                let x = (u * self.width as f32) as u32 % self.width;
                let y = ((1.0 - v) * self.height as f32) as u32 % self.height;

                let pixel = img.get_pixel(x, y);
                let srgb_color = crate::material_system::color::Color::new(
                    pixel[0] as f32 / 255.0,
                    pixel[1] as f32 / 255.0,
                    pixel[2] as f32 / 255.0,
                );

                let linear_color = crate::material_system::color::srgb_to_linear(&srgb_color);
                [linear_color.x, linear_color.y, linear_color.z]
            }
            TextureData::SolidColor(color) => [color.x, color.y, color.z],
            TextureData::FaceColor(seed) => {
                let color = crate::material_system::color::get_random_color(*seed, true);
                [color.x, color.y, color.z]
            }
        }
    }

    /// 修复：简化的法线贴图采样
    pub fn sample_normal(&self, u: f32, v: f32) -> [f32; 3] {
        match &self.data {
            TextureData::Image(img) => {
                let u = u.fract().abs();
                let v = v.fract().abs();
                let x = (u * self.width as f32) as u32 % self.width;
                let y = ((1.0 - v) * self.height as f32) as u32 % self.height;

                let pixel = img.get_pixel(x, y);

                // 修复：正确解码法线贴图
                let normal_x = (pixel[0] as f32 / 255.0) * 2.0 - 1.0;
                let normal_y = (pixel[1] as f32 / 255.0) * 2.0 - 1.0;
                let normal_z = (pixel[2] as f32 / 255.0) * 2.0 - 1.0; // 修复：也要解码

                // 确保法线向量有效，但允许所有方向
                let length_sq = normal_x * normal_x + normal_y * normal_y + normal_z * normal_z;
                if length_sq < 0.01 {
                    // 处理压缩法线贴图（只有XY通道）
                    let xy_length_sq = normal_x * normal_x + normal_y * normal_y;
                    if xy_length_sq <= 1.0 {
                        let z = (1.0 - xy_length_sq).sqrt().max(0.01);
                        [normal_x, normal_y, z]
                    } else {
                        // 归一化XY，保持Z为正
                        let xy_length = xy_length_sq.sqrt();
                        [normal_x / xy_length, normal_y / xy_length, 0.01]
                    }
                } else {
                    // 标准法线贴图，归一化但保持原始方向
                    let length = length_sq.sqrt().max(0.001);
                    [normal_x / length, normal_y / length, normal_z / length]
                }
            }
            _ => [0.0, 0.0, 1.0], // 默认切线空间法线
        }
    }

    pub fn is_face_color(&self) -> bool {
        matches!(&self.data, TextureData::FaceColor(_))
    }
}

pub fn load_texture<P: AsRef<Path>>(path: P, default_color: Vector3<f32>) -> Texture {
    Texture::from_file(path).unwrap_or_else(|| {
        warn!("无法加载纹理，使用默认颜色 {:?}", default_color);
        Texture::solid_color(default_color)
    })
}
