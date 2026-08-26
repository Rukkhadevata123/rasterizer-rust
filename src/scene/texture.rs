use image::{DynamicImage, GenericImageView};
use log::info;
use nalgebra::Vector4;
use std::path::Path;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Texture {
    pub mips: Vec<Arc<DynamicImage>>,
    pub width: u32,
    pub height: u32,
}

impl Texture {
    /// Load from file path, now only used in background texture.
    pub fn load<P: AsRef<Path>>(path: P, use_mipmap: bool) -> Result<Self, String> {
        let path_ref = path.as_ref();
        let img = image::open(path_ref).map_err(|e| format!("Failed to load texture: {}", e))?;
        info!(
            "Loaded texture: {:?} ({}x{})",
            path_ref,
            img.width(),
            img.height()
        );

        Ok(Self::from_image(img, use_mipmap))
    }

    /// Create directly from an in-memory image (Crucial for GLTF/GLB)
    pub fn from_image(img: DynamicImage, use_mipmap: bool) -> Self {
        let width = img.width();
        let height = img.height();
        let mut mips = Vec::new();

        // Level 0
        mips.push(Arc::new(img.clone()));

        // Generate Mips
        if use_mipmap {
            let mut current = img;
            while current.width() > 1 && current.height() > 1 {
                current = current.resize(
                    (current.width() / 2).max(1),
                    (current.height() / 2).max(1),
                    image::imageops::FilterType::Triangle,
                );
                mips.push(Arc::new(current.clone()));
            }
        }

        Self {
            mips,
            width,
            height,
        }
    }

    /// Bilinear sample on a specific mip level.
    fn sample_bilinear_level(&self, u: f32, v: f32, level: usize) -> Vector4<f32> {
        let level = level.min(self.mips.len() - 1);
        let img = &self.mips[level];
        let width = img.width();
        let height = img.height();

        // 1. Handle wrapping (Repeat mode)
        let u = u.fract();
        let v = v.fract();
        let u = if u < 0.0 { 1.0 + u } else { u };
        let v = if v < 0.0 { 1.0 + v } else { v };

        // 2. Map to pixel coordinates
        let x = u * width as f32 - 0.5;
        let y = v * height as f32 - 0.5;

        // 3. Identify the 2x2 pixel block
        let x0 = x.floor() as i32;
        let y0 = y.floor() as i32;
        let x1 = x0 + 1;
        let y1 = y0 + 1;

        // 4. Calculate weights
        let wx = x - x.floor();
        let wy = y - y.floor();

        // 5. Fetch colors
        let c00 = Self::get_pixel_wrapped(img, x0, y0);
        let c10 = Self::get_pixel_wrapped(img, x1, y0);
        let c01 = Self::get_pixel_wrapped(img, x0, y1);
        let c11 = Self::get_pixel_wrapped(img, x1, y1);

        // 6. Interpolate
        let top = c00 * (1.0 - wx) + c10 * wx;
        let bottom = c01 * (1.0 - wx) + c11 * wx;

        top * (1.0 - wy) + bottom * wy
    }

    /// Helper: fetch pixel from a specific image with wrapping.
    fn get_pixel_wrapped(img: &DynamicImage, x: i32, y: i32) -> Vector4<f32> {
        let width = img.width() as i32;
        let height = img.height() as i32;
        let x_wrapped = ((x % width) + width) % width;
        let y_wrapped = ((y % height) + height) % height;
        let p = img.get_pixel(x_wrapped as u32, y_wrapped as u32);
        Vector4::new(
            p[0] as f32 / 255.0,
            p[1] as f32 / 255.0,
            p[2] as f32 / 255.0,
            p[3] as f32 / 255.0,
        )
    }

    /// Trilinear sampling given a triangle-level uv_density.
    /// uv_density: sqrt(Area_uv / Area_screen). 0.0 means "no special LOD" and selects level 0.
    pub fn sample_data_with_density(&self, u: f32, v: f32, uv_density: f32) -> Vector4<f32> {
        if self.mips.len() == 1 || uv_density <= 0.0 {
            return self.sample_bilinear_level(u, v, 0);
        }

        // Use the larger texture dimension to estimate texel coverage.
        let size = (self.width.max(self.height)) as f32;
        let lod = (uv_density * size).log2().max(0.0);
        let l0 = lod.floor() as usize;
        let l1 = (l0 + 1).min(self.mips.len() - 1);
        let w = lod - lod.floor();

        let c0 = self.sample_bilinear_level(u, v, l0);
        let c1 = self.sample_bilinear_level(u, v, l1);
        c0 * (1.0 - w) + c1 * w
    }

    /// Color sampling with gamma correction (linear output expected by pipeline).
    pub fn sample_color_with_density(&self, u: f32, v: f32, uv_density: f32) -> Vector4<f32> {
        let linear = self.sample_data_with_density(u, v, uv_density);
        Vector4::new(
            linear.x.powf(2.2),
            linear.y.powf(2.2),
            linear.z.powf(2.2),
            linear.w,
        )
    }

    /// For color map (Albedo, Emissive, Background)
    /// Performs sRGB -> Linear conversion (Gamma 2.2)
    pub fn sample_color(&self, u: f32, v: f32) -> Vector4<f32> {
        self.sample_color_with_density(u, v, 0.0)
    }

    /// For data map (Metallic, Roughness, Normal, AO)
    /// Returns linear values directly without Gamma correction
    pub fn sample_data(&self, u: f32, v: f32) -> Vector4<f32> {
        self.sample_data_with_density(u, v, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgba, RgbaImage};
    use std::sync::Arc;

    fn test_image() -> DynamicImage {
        let mut image = RgbaImage::new(2, 2);
        image.put_pixel(0, 0, Rgba([255, 0, 0, 255]));
        image.put_pixel(1, 0, Rgba([0, 255, 0, 255]));
        image.put_pixel(0, 1, Rgba([0, 0, 255, 255]));
        image.put_pixel(1, 1, Rgba([255, 255, 255, 255]));
        DynamicImage::ImageRgba8(image)
    }

    fn assert_vec4_approx(actual: Vector4<f32>, expected: Vector4<f32>) {
        assert!(
            (actual - expected).norm() < 1e-5,
            "expected {expected:?}, got {actual:?}"
        );
    }

    #[test]
    fn bilinear_sampling_interpolates_four_texels() {
        let texture = Texture::from_image(test_image(), false);
        assert_vec4_approx(
            texture.sample_data(0.5, 0.5),
            Vector4::new(0.5, 0.5, 0.5, 1.0),
        );
    }

    #[test]
    fn sampling_repeats_outside_unit_interval() {
        let texture = Texture::from_image(test_image(), false);
        assert_vec4_approx(
            texture.sample_data(0.25, 0.25),
            texture.sample_data(1.25, -0.75),
        );
    }

    #[test]
    fn square_texture_generates_complete_mip_chain() {
        let texture = Texture::from_image(DynamicImage::new_rgba8(4, 4), true);
        let dimensions: Vec<_> = texture
            .mips
            .iter()
            .map(|image| (image.width(), image.height()))
            .collect();

        assert_eq!(dimensions, vec![(4, 4), (2, 2), (1, 1)]);
    }

    #[test]
    fn density_selects_expected_mip_level() {
        let solid = |width, height, color| {
            Arc::new(DynamicImage::ImageRgba8(RgbaImage::from_pixel(
                width,
                height,
                Rgba(color),
            )))
        };
        let texture = Texture {
            mips: vec![
                solid(4, 4, [255, 0, 0, 255]),
                solid(2, 2, [0, 255, 0, 255]),
                solid(1, 1, [0, 0, 255, 255]),
            ],
            width: 4,
            height: 4,
        };

        assert_vec4_approx(
            texture.sample_data_with_density(0.5, 0.5, 0.25),
            Vector4::new(1.0, 0.0, 0.0, 1.0),
        );
        assert_vec4_approx(
            texture.sample_data_with_density(0.5, 0.5, 0.5),
            Vector4::new(0.0, 1.0, 0.0, 1.0),
        );
        assert_vec4_approx(
            texture.sample_data_with_density(0.5, 0.5, 1.0),
            Vector4::new(0.0, 0.0, 1.0, 1.0),
        );
    }

    #[test]
    fn color_sampling_decodes_srgb_but_preserves_alpha() {
        let texture = Texture::from_image(test_image(), false);
        let data = texture.sample_data(0.5, 0.5);
        let color = texture.sample_color(0.5, 0.5);

        assert!(color.x < data.x);
        assert!(color.y < data.y);
        assert!(color.z < data.z);
        assert_eq!(color.w, data.w);
    }
}
