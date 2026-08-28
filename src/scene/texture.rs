use image::{DynamicImage, GenericImageView};
use log::info;
use nalgebra::Vector4;
use std::path::Path;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct TextureImage {
    pub mips: Vec<Arc<DynamicImage>>,
    pub width: u32,
    pub height: u32,
}

impl TextureImage {
    pub fn load<P: AsRef<Path>>(path: P, use_mipmap: bool) -> Result<Self, image::ImageError> {
        let path_ref = path.as_ref();
        let image = image::open(path_ref)?;
        info!(
            "Loaded texture image: {:?} ({}x{})",
            path_ref,
            image.width(),
            image.height()
        );

        Ok(Self::from_image(image, use_mipmap))
    }

    pub fn from_image(image: DynamicImage, use_mipmap: bool) -> Self {
        let width = image.width();
        let height = image.height();
        let mut mips = vec![Arc::new(image.clone())];

        if use_mipmap {
            let mut current = image;
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WrapMode {
    Repeat,
    ClampToEdge,
    MirroredRepeat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MagFilter {
    Nearest,
    Linear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MinFilter {
    Nearest,
    Linear,
    NearestMipmapNearest,
    LinearMipmapNearest,
    NearestMipmapLinear,
    LinearMipmapLinear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamplerState {
    pub wrap_u: WrapMode,
    pub wrap_v: WrapMode,
    pub mag_filter: MagFilter,
    pub min_filter: MinFilter,
}

impl Default for SamplerState {
    fn default() -> Self {
        Self {
            wrap_u: WrapMode::Repeat,
            wrap_v: WrapMode::Repeat,
            mag_filter: MagFilter::Linear,
            min_filter: MinFilter::Linear,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureUsage {
    Color,
    Data,
}

#[derive(Debug, Clone)]
pub struct TextureBinding {
    pub image: Arc<TextureImage>,
    pub sampler: SamplerState,
    pub tex_coord: u32,
    pub usage: TextureUsage,
}

impl TextureBinding {
    pub fn new(
        image: Arc<TextureImage>,
        sampler: SamplerState,
        tex_coord: u32,
        usage: TextureUsage,
    ) -> Self {
        Self {
            image,
            sampler,
            tex_coord,
            usage,
        }
    }

    /// Samples with the renderer's existing repeat, bilinear, and trilinear behavior.
    /// `SamplerState` is retained independently so exact glTF choices can be applied by the
    /// sampler implementation without changing image ownership or material bindings.
    pub fn sample_with_density(&self, u: f32, v: f32, uv_density: f32) -> Vector4<f32> {
        let sample = self.sample_data_with_density(u, v, uv_density);
        match self.usage {
            TextureUsage::Color => Vector4::new(
                sample.x.powf(2.2),
                sample.y.powf(2.2),
                sample.z.powf(2.2),
                sample.w,
            ),
            TextureUsage::Data => sample,
        }
    }

    pub fn sample(&self, u: f32, v: f32) -> Vector4<f32> {
        self.sample_with_density(u, v, 0.0)
    }

    fn sample_data_with_density(&self, u: f32, v: f32, uv_density: f32) -> Vector4<f32> {
        if self.image.mips.len() == 1 || uv_density <= 0.0 {
            return self.sample_bilinear_level(u, v, 0);
        }

        let size = self.image.width.max(self.image.height) as f32;
        let lod = (uv_density * size).log2().max(0.0);
        let lower_level = lod.floor() as usize;
        let upper_level = (lower_level + 1).min(self.image.mips.len() - 1);
        let weight = lod - lod.floor();

        let lower = self.sample_bilinear_level(u, v, lower_level);
        let upper = self.sample_bilinear_level(u, v, upper_level);
        lower * (1.0 - weight) + upper * weight
    }

    fn sample_bilinear_level(&self, u: f32, v: f32, level: usize) -> Vector4<f32> {
        let level = level.min(self.image.mips.len() - 1);
        let image = &self.image.mips[level];
        let width = image.width();
        let height = image.height();

        let u = u.fract();
        let v = v.fract();
        let u = if u < 0.0 { 1.0 + u } else { u };
        let v = if v < 0.0 { 1.0 + v } else { v };

        let x = u * width as f32 - 0.5;
        let y = v * height as f32 - 0.5;
        let x0 = x.floor() as i32;
        let y0 = y.floor() as i32;
        let x1 = x0 + 1;
        let y1 = y0 + 1;
        let x_weight = x - x.floor();
        let y_weight = y - y.floor();

        let top_left = Self::get_pixel_wrapped(image, x0, y0);
        let top_right = Self::get_pixel_wrapped(image, x1, y0);
        let bottom_left = Self::get_pixel_wrapped(image, x0, y1);
        let bottom_right = Self::get_pixel_wrapped(image, x1, y1);
        let top = top_left * (1.0 - x_weight) + top_right * x_weight;
        let bottom = bottom_left * (1.0 - x_weight) + bottom_right * x_weight;

        top * (1.0 - y_weight) + bottom * y_weight
    }

    fn get_pixel_wrapped(image: &DynamicImage, x: i32, y: i32) -> Vector4<f32> {
        let width = image.width() as i32;
        let height = image.height() as i32;
        let wrapped_x = ((x % width) + width) % width;
        let wrapped_y = ((y % height) + height) % height;
        let pixel = image.get_pixel(wrapped_x as u32, wrapped_y as u32);
        Vector4::new(
            pixel[0] as f32 / 255.0,
            pixel[1] as f32 / 255.0,
            pixel[2] as f32 / 255.0,
            pixel[3] as f32 / 255.0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgba, RgbaImage};

    fn test_image() -> DynamicImage {
        let mut image = RgbaImage::new(2, 2);
        image.put_pixel(0, 0, Rgba([255, 0, 0, 255]));
        image.put_pixel(1, 0, Rgba([0, 255, 0, 255]));
        image.put_pixel(0, 1, Rgba([0, 0, 255, 255]));
        image.put_pixel(1, 1, Rgba([255, 255, 255, 255]));
        DynamicImage::ImageRgba8(image)
    }

    fn binding(image: TextureImage, usage: TextureUsage) -> TextureBinding {
        TextureBinding::new(Arc::new(image), SamplerState::default(), 0, usage)
    }

    fn assert_vec4_approx(actual: Vector4<f32>, expected: Vector4<f32>) {
        assert!(
            (actual - expected).norm() < 1e-5,
            "expected {expected:?}, got {actual:?}"
        );
    }

    #[test]
    fn bilinear_sampling_interpolates_four_texels() {
        let texture = binding(
            TextureImage::from_image(test_image(), false),
            TextureUsage::Data,
        );
        assert_vec4_approx(texture.sample(0.5, 0.5), Vector4::new(0.5, 0.5, 0.5, 1.0));
    }

    #[test]
    fn sampling_repeats_outside_unit_interval() {
        let texture = binding(
            TextureImage::from_image(test_image(), false),
            TextureUsage::Data,
        );
        assert_vec4_approx(texture.sample(0.25, 0.25), texture.sample(1.25, -0.75));
    }

    #[test]
    fn square_texture_generates_complete_mip_chain() {
        let texture = TextureImage::from_image(DynamicImage::new_rgba8(4, 4), true);
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
        let texture = binding(
            TextureImage {
                mips: vec![
                    solid(4, 4, [255, 0, 0, 255]),
                    solid(2, 2, [0, 255, 0, 255]),
                    solid(1, 1, [0, 0, 255, 255]),
                ],
                width: 4,
                height: 4,
            },
            TextureUsage::Data,
        );

        assert_vec4_approx(
            texture.sample_with_density(0.5, 0.5, 0.25),
            Vector4::new(1.0, 0.0, 0.0, 1.0),
        );
        assert_vec4_approx(
            texture.sample_with_density(0.5, 0.5, 0.5),
            Vector4::new(0.0, 1.0, 0.0, 1.0),
        );
        assert_vec4_approx(
            texture.sample_with_density(0.5, 0.5, 1.0),
            Vector4::new(0.0, 0.0, 1.0, 1.0),
        );
    }

    #[test]
    fn color_binding_decodes_srgb_but_preserves_alpha() {
        let image = Arc::new(TextureImage::from_image(test_image(), false));
        let data = TextureBinding::new(
            image.clone(),
            SamplerState::default(),
            0,
            TextureUsage::Data,
        )
        .sample(0.5, 0.5);
        let color = TextureBinding::new(image, SamplerState::default(), 0, TextureUsage::Color)
            .sample(0.5, 0.5);

        assert!(color.x < data.x);
        assert!(color.y < data.y);
        assert!(color.z < data.z);
        assert_eq!(color.w, data.w);
    }
}
