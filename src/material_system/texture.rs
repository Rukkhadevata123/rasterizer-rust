use crate::material_system::color::{Color, get_random_color, srgb_to_linear};
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
                let srgb_color = Color::new(
                    pixel[0] as f32 / 255.0,
                    pixel[1] as f32 / 255.0,
                    pixel[2] as f32 / 255.0,
                );

                let linear_color = srgb_to_linear(&srgb_color);
                [linear_color.x, linear_color.y, linear_color.z]
            }
            TextureData::SolidColor(color) => [color.x, color.y, color.z],
            TextureData::FaceColor(seed) => {
                let color = get_random_color(*seed, true);
                [color.x, color.y, color.z]
            }
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
