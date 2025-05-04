use image::{DynamicImage, GenericImageView, ImageError, Rgba, RgbaImage};
use nalgebra::Vector3;
use std::path::Path;

/// Represents a texture with dimensions and RGBA pixel data (f32, 0.0-1.0).
#[derive(Debug, Clone)]
pub struct Texture {
    pub width: u32,
    pub height: u32,
    /// Flattened RGBA pixel data (f32, 0.0-1.0 range). Layout: [R, G, B, A, R, G, B, A, ...]
    pub data: Vec<f32>,
}

impl Texture {
    /// Creates a default 1x1 texture with the given color.
    fn default_texture(color: Vector3<f32>) -> Self {
        Texture {
            width: 1,
            height: 1,
            data: vec![color.x, color.y, color.z, 1.0], // RGBA
        }
    }

    /// Samples the texture using UV coordinates (nearest neighbor).
    /// UV coordinates should be in the range [0.0, 1.0].
    /// Handles wrapping (repeating texture).
    pub fn sample(&self, u: f32, v: f32) -> Rgba<f32> {
        if self.width == 0 || self.height == 0 || self.data.is_empty() {
            return Rgba([0.0, 0.0, 0.0, 1.0]); // Should not happen with default texture
        }

        // Wrap UV coordinates
        let u_wrapped = u.fract(); // Keep fractional part for wrapping
        let v_wrapped = v.fract();
        let u_final = if u_wrapped < 0.0 {
            u_wrapped + 1.0
        } else {
            u_wrapped
        };
        let v_final = if v_wrapped < 0.0 {
            v_wrapped + 1.0
        } else {
            v_wrapped
        };

        // Calculate pixel coordinates (nearest neighbor)
        // Flip V coordinate because image origin is top-left, texture V origin is often bottom-left
        let x = (u_final * self.width as f32) as u32 % self.width;
        let y = ((1.0 - v_final) * self.height as f32) as u32 % self.height; // Flip V here

        let index = (y * self.width + x) as usize * 4; // 4 components (RGBA)

        if index + 3 < self.data.len() {
            Rgba([
                self.data[index],
                self.data[index + 1],
                self.data[index + 2],
                self.data[index + 3],
            ])
        } else {
            // Should ideally not happen if indices are calculated correctly
            Rgba([1.0, 0.0, 1.0, 1.0]) // Magenta error color
        }
    }
}

/// Loads a texture image file from the given path.
///
/// If loading fails or the path is invalid, returns a 1x1 default texture.
/// Normalizes pixel values to f32 [0.0, 1.0].
pub fn load_texture<P: AsRef<Path>>(texture_path: P, default_color: Vector3<f32>) -> Texture {
    let path_ref = texture_path.as_ref();

    match image::open(path_ref) {
        Ok(img) => {
            // No need to flip vertically here, handle V coordinate in sample() or during UV loading
            let rgba_img: RgbaImage = img.into_rgba8(); // Convert to RGBA u8
            let (width, height) = rgba_img.dimensions();

            // Convert u8 [0, 255] to f32 [0.0, 1.0]
            let data_f32: Vec<f32> = rgba_img
                .into_raw()
                .into_iter()
                .map(|byte| f32::from(byte) / 255.0)
                .collect();

            println!(
                "Successfully loaded texture: {:?} ({}x{})",
                path_ref, width, height
            );
            Texture {
                width,
                height,
                data: data_f32,
            }
        }
        Err(e) => {
            match e {
                ImageError::IoError(_) => {
                    println!("Texture file not found or cannot be read: {:?}", path_ref);
                }
                _ => {
                    println!("Error loading texture '{:?}': {}", path_ref, e);
                }
            }
            println!("Using 1x1 default color texture: {:?}", default_color);
            Texture::default_texture(default_color)
        }
    }
}

// Procedural texture generation can be added here later if needed.
// fn generate_procedural_texture(...) -> Texture { ... }
