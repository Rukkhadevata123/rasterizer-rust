use image::{DynamicImage, GenericImageView};
use log::info;
use nalgebra::Vector3;
use std::path::Path;
use std::sync::Arc;

/// Represents a 2D texture map.
#[derive(Debug, Clone)]
pub struct Texture {
    pub image: Arc<DynamicImage>,
    pub width: u32,
    pub height: u32,
}

impl Texture {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let path_ref = path.as_ref();
        let img = image::open(path_ref).map_err(|e| format!("Failed to load texture: {}", e))?;

        let width = img.width();
        let height = img.height();

        info!("Loaded texture: {:?} ({}x{})", path_ref, width, height);

        Ok(Self {
            width,
            height,
            image: Arc::new(img),
        })
    }

    /// Samples the texture using Bilinear Interpolation.
    /// UV coordinates are in [0.0, 1.0].
    pub fn sample(&self, u: f32, v: f32) -> Vector3<f32> {
        // 1. Handle wrapping (Repeat mode) using fract()
        // This handles u=1.5 -> 0.5, u=-0.5 -> 0.5 correctly
        let u = u.fract();
        let v = v.fract();
        let u = if u < 0.0 { 1.0 + u } else { u };
        let v = if v < 0.0 { 1.0 + v } else { v };

        // 2. Map to pixel coordinates
        // -0.5 because pixel centers are at 0.5
        let x = u * self.width as f32 - 0.5;
        let y = (1.0 - v) * self.height as f32 - 0.5; // Flip V for standard UV

        // 3. Identify the 2x2 pixel block
        let x0 = x.floor() as i32;
        let y0 = y.floor() as i32;
        let x1 = x0 + 1;
        let y1 = y0 + 1;

        // 4. Calculate weights
        let wx = x - x.floor();
        let wy = y - y.floor();

        // 5. Fetch colors (Using WRAP mode, not clamp)
        let c00 = self.get_pixel_wrapped(x0, y0);
        let c10 = self.get_pixel_wrapped(x1, y0);
        let c01 = self.get_pixel_wrapped(x0, y1);
        let c11 = self.get_pixel_wrapped(x1, y1);

        // 6. Interpolate
        // Lerp X
        let top = c00 * (1.0 - wx) + c10 * wx;
        let bottom = c01 * (1.0 - wx) + c11 * wx;

        // Lerp Y
        let final_color = top * (1.0 - wy) + bottom * wy;

        // 7. sRGB to Linear conversion
        // Textures are usually sRGB. We must convert to Linear before lighting math.
        // Simple approximation: pow(2.2)
        Vector3::new(
            final_color.x.powf(2.2),
            final_color.y.powf(2.2),
            final_color.z.powf(2.2),
        )
    }

    /// Helper to get pixel with WRAPPING (Repeat) logic
    fn get_pixel_wrapped(&self, x: i32, y: i32) -> Vector3<f32> {
        let w = self.width as i32;
        let h = self.height as i32;

        // Euclidean modulo to handle negative numbers correctly
        // e.g., -1 % 100 = -1 in Rust, but we want 99.
        let x_wrapped = ((x % w) + w) % w;
        let y_wrapped = ((y % h) + h) % h;

        let pixel = self.image.get_pixel(x_wrapped as u32, y_wrapped as u32);

        // Return raw [0.0, 1.0] value
        Vector3::new(
            pixel[0] as f32 / 255.0,
            pixel[1] as f32 / 255.0,
            pixel[2] as f32 / 255.0,
        )
    }
}
