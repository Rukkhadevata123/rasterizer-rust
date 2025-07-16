use image::{DynamicImage, GenericImageView};
use log::warn;
use std::path::Path;
use std::sync::Arc;

use crate::material_system::color::{Color, srgb_to_linear};

#[derive(Debug, Clone)]
pub struct Texture {
    pub image: Arc<DynamicImage>,
    pub width: u32,
    pub height: u32,
}

impl Texture {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Option<Self> {
        match image::open(path) {
            Ok(img) => Some(Texture {
                width: img.width(),
                height: img.height(),
                image: Arc::new(img),
            }),
            Err(e) => {
                warn!("无法加载纹理: {e}");
                None
            }
        }
    }

    pub fn sample(&self, u: f32, v: f32) -> [f32; 3] {
        let u = u.fract().abs();
        let v = v.fract().abs();
        let x = (u * self.width as f32) as u32 % self.width;
        let y = ((1.0 - v) * self.height as f32) as u32 % self.height;

        let pixel = self.image.get_pixel(x, y);
        let srgb_color = Color::new(
            pixel[0] as f32 / 255.0,
            pixel[1] as f32 / 255.0,
            pixel[2] as f32 / 255.0,
        );
        let linear_color = srgb_to_linear(&srgb_color);
        [linear_color.x, linear_color.y, linear_color.z]
    }
}
