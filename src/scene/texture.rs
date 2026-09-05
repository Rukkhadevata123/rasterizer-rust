use crate::core::color::srgb_to_linear;
use image::{DynamicImage, GenericImageView};
use log::info;
use nalgebra::{Vector3, Vector4};
use std::path::Path;
use std::sync::{Arc, OnceLock};

#[derive(Debug)]
struct MipLevel {
    width: u32,
    height: u32,
    pixels: Vec<Vector4<f32>>,
}

impl MipLevel {
    fn pixel(&self, x: u32, y: u32) -> Vector4<f32> {
        self.pixels[(y * self.width + x) as usize]
    }
}

#[derive(Debug, Clone)]
pub struct TextureImage {
    source: Arc<DynamicImage>,
    use_mipmap: bool,
    color_mips: Arc<OnceLock<Arc<Vec<MipLevel>>>>,
    data_mips: Arc<OnceLock<Arc<Vec<MipLevel>>>>,
    normal_mips: Arc<OnceLock<Arc<Vec<MipLevel>>>>,
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

        Self {
            source: Arc::new(image),
            use_mipmap,
            color_mips: Arc::new(OnceLock::new()),
            data_mips: Arc::new(OnceLock::new()),
            normal_mips: Arc::new(OnceLock::new()),
            width,
            height,
        }
    }

    fn mips(&self, usage: TextureUsage) -> &Arc<Vec<MipLevel>> {
        let cache = match usage {
            TextureUsage::Color => &self.color_mips,
            TextureUsage::Data => &self.data_mips,
            TextureUsage::Normal => &self.normal_mips,
        };
        cache.get_or_init(|| Arc::new(self.generate_mips(usage)))
    }

    fn generate_mips(&self, usage: TextureUsage) -> Vec<MipLevel> {
        let mut levels = vec![self.base_level(usage)];
        if self.use_mipmap {
            while levels
                .last()
                .is_some_and(|level| level.width > 1 || level.height > 1)
            {
                levels.push(Self::downsample(
                    levels.last().expect("base mip level always exists"),
                    usage,
                ));
            }
        }
        levels
    }

    fn base_level(&self, usage: TextureUsage) -> MipLevel {
        let pixels = self
            .source
            .pixels()
            .map(|(_, _, pixel)| {
                let sample = Vector4::new(
                    pixel[0] as f32 / 255.0,
                    pixel[1] as f32 / 255.0,
                    pixel[2] as f32 / 255.0,
                    pixel[3] as f32 / 255.0,
                );
                if usage == TextureUsage::Color {
                    let linear = srgb_to_linear(sample.xyz());
                    Vector4::new(linear.x, linear.y, linear.z, sample.w)
                } else {
                    sample
                }
            })
            .collect();
        MipLevel {
            width: self.width,
            height: self.height,
            pixels,
        }
    }

    fn downsample(source: &MipLevel, usage: TextureUsage) -> MipLevel {
        let width = source.width.div_ceil(2);
        let height = source.height.div_ceil(2);
        let mut pixels = Vec::with_capacity((width * height) as usize);

        for y in 0..height {
            for x in 0..width {
                let mut sample = Vector4::zeros();
                let mut count = 0.0;
                for source_y in y * 2..(y * 2 + 2).min(source.height) {
                    for source_x in x * 2..(x * 2 + 2).min(source.width) {
                        sample += source.pixel(source_x, source_y);
                        count += 1.0;
                    }
                }
                sample /= count;
                if usage == TextureUsage::Normal {
                    let normal = sample.xyz() * 2.0 - Vector3::repeat(1.0);
                    let normal = normal
                        .try_normalize(f32::EPSILON)
                        .unwrap_or_else(Vector3::z);
                    sample = Vector4::new(
                        normal.x * 0.5 + 0.5,
                        normal.y * 0.5 + 0.5,
                        normal.z * 0.5 + 0.5,
                        sample.w,
                    );
                }
                pixels.push(sample);
            }
        }

        MipLevel {
            width,
            height,
            pixels,
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
    Normal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TexCoordSet {
    TexCoord0,
    TexCoord1,
}

impl TexCoordSet {
    pub fn index(self) -> usize {
        match self {
            Self::TexCoord0 => 0,
            Self::TexCoord1 => 1,
        }
    }
}

impl TryFrom<u32> for TexCoordSet {
    type Error = u32;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::TexCoord0),
            1 => Ok(Self::TexCoord1),
            unsupported => Err(unsupported),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TextureBinding {
    pub image: Arc<TextureImage>,
    pub sampler: SamplerState,
    pub tex_coord: TexCoordSet,
    pub usage: TextureUsage,
}

impl TextureBinding {
    pub fn new(
        image: Arc<TextureImage>,
        sampler: SamplerState,
        tex_coord: TexCoordSet,
        usage: TextureUsage,
    ) -> Self {
        Self {
            image,
            sampler,
            tex_coord,
            usage,
        }
    }

    pub fn sample_with_density(&self, u: f32, v: f32, uv_density: f32) -> Vector4<f32> {
        self.sample_data_with_density(u, v, uv_density)
    }

    pub fn sample(&self, u: f32, v: f32) -> Vector4<f32> {
        self.sample_with_density(u, v, 0.0)
    }

    fn sample_data_with_density(&self, u: f32, v: f32, uv_density: f32) -> Vector4<f32> {
        let mips = self.image.mips(self.usage);
        let size = self.image.width.max(self.image.height) as f32;
        let texels_per_pixel = uv_density * size;
        if texels_per_pixel <= 1.0 {
            return match self.sampler.mag_filter {
                MagFilter::Nearest => self.sample_nearest_level(u, v, 0),
                MagFilter::Linear => self.sample_bilinear_level(u, v, 0),
            };
        }

        let lod = texels_per_pixel.log2().clamp(0.0, (mips.len() - 1) as f32);
        match self.sampler.min_filter {
            MinFilter::Nearest => self.sample_nearest_level(u, v, 0),
            MinFilter::Linear => self.sample_bilinear_level(u, v, 0),
            MinFilter::NearestMipmapNearest => {
                self.sample_nearest_level(u, v, lod.round() as usize)
            }
            MinFilter::LinearMipmapNearest => {
                self.sample_bilinear_level(u, v, lod.round() as usize)
            }
            MinFilter::NearestMipmapLinear => {
                self.sample_mipmap_linear(u, v, lod, Self::sample_nearest_level)
            }
            MinFilter::LinearMipmapLinear => {
                self.sample_mipmap_linear(u, v, lod, Self::sample_bilinear_level)
            }
        }
    }

    fn sample_mipmap_linear(
        &self,
        u: f32,
        v: f32,
        lod: f32,
        sample_level: fn(&Self, f32, f32, usize) -> Vector4<f32>,
    ) -> Vector4<f32> {
        let lower_level = lod.floor() as usize;
        let upper_level = (lower_level + 1).min(self.image.mips(self.usage).len() - 1);
        let weight = lod - lower_level as f32;

        let lower = sample_level(self, u, v, lower_level);
        let upper = sample_level(self, u, v, upper_level);
        lower * (1.0 - weight) + upper * weight
    }

    fn sample_nearest_level(&self, u: f32, v: f32, level: usize) -> Vector4<f32> {
        let mips = self.image.mips(self.usage);
        let image = &mips[level.min(mips.len() - 1)];
        let x = (u * image.width as f32).floor() as i32;
        let y = (v * image.height as f32).floor() as i32;
        self.get_pixel(image, x, y)
    }

    fn sample_bilinear_level(&self, u: f32, v: f32, level: usize) -> Vector4<f32> {
        let mips = self.image.mips(self.usage);
        let image = &mips[level.min(mips.len() - 1)];
        let width = image.width;
        let height = image.height;

        let x = u * width as f32 - 0.5;
        let y = v * height as f32 - 0.5;
        let x0 = x.floor() as i32;
        let y0 = y.floor() as i32;
        let x1 = x0 + 1;
        let y1 = y0 + 1;
        let x_weight = x - x.floor();
        let y_weight = y - y.floor();

        let top_left = self.get_pixel(image, x0, y0);
        let top_right = self.get_pixel(image, x1, y0);
        let bottom_left = self.get_pixel(image, x0, y1);
        let bottom_right = self.get_pixel(image, x1, y1);
        let top = top_left * (1.0 - x_weight) + top_right * x_weight;
        let bottom = bottom_left * (1.0 - x_weight) + bottom_right * x_weight;

        top * (1.0 - y_weight) + bottom * y_weight
    }

    fn get_pixel(&self, image: &MipLevel, x: i32, y: i32) -> Vector4<f32> {
        let width = image.width as i32;
        let height = image.height as i32;
        let x = Self::address_index(x, width, self.sampler.wrap_u);
        let y = Self::address_index(y, height, self.sampler.wrap_v);
        image.pixel(x as u32, y as u32)
    }

    fn address_index(index: i32, size: i32, wrap: WrapMode) -> i32 {
        match wrap {
            WrapMode::Repeat => index.rem_euclid(size),
            WrapMode::ClampToEdge => index.clamp(0, size - 1),
            WrapMode::MirroredRepeat => {
                let period = size * 2;
                let mirrored = index.rem_euclid(period);
                if mirrored < size {
                    mirrored
                } else {
                    period - 1 - mirrored
                }
            }
        }
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
        TextureBinding::new(
            Arc::new(image),
            SamplerState::default(),
            TexCoordSet::TexCoord0,
            usage,
        )
    }

    fn binding_with_sampler(image: TextureImage, sampler: SamplerState) -> TextureBinding {
        TextureBinding::new(
            Arc::new(image),
            sampler,
            TexCoordSet::TexCoord0,
            TextureUsage::Data,
        )
    }

    fn two_texel_image() -> TextureImage {
        TextureImage::from_image(
            DynamicImage::ImageRgba8(
                RgbaImage::from_vec(2, 1, vec![255, 0, 0, 255, 0, 255, 0, 255])
                    .expect("test pixels should match image dimensions"),
            ),
            false,
        )
    }

    fn mip_selection_image() -> TextureImage {
        let values = [
            0, 64, 128, 128, 64, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
        ];
        let pixels = values
            .into_iter()
            .flat_map(|value| [value, value, value, 255])
            .collect();
        TextureImage::from_image(
            DynamicImage::ImageRgba8(
                RgbaImage::from_vec(4, 4, pixels)
                    .expect("mip-selection fixture should match its dimensions"),
            ),
            true,
        )
    }

    fn data_gray(value: u8) -> Vector4<f32> {
        let value = value as f32 / 255.0;
        Vector4::new(value, value, value, 1.0)
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
    fn wrap_modes_address_outside_coordinates() {
        let sample = |wrap_u, u| {
            binding_with_sampler(
                two_texel_image(),
                SamplerState {
                    wrap_u,
                    wrap_v: WrapMode::ClampToEdge,
                    mag_filter: MagFilter::Nearest,
                    min_filter: MinFilter::Nearest,
                },
            )
            .sample(u, 0.5)
        };
        let red = Vector4::new(1.0, 0.0, 0.0, 1.0);
        let green = Vector4::new(0.0, 1.0, 0.0, 1.0);

        assert_vec4_approx(sample(WrapMode::Repeat, -0.25), green);
        assert_vec4_approx(sample(WrapMode::Repeat, 1.25), red);
        assert_vec4_approx(sample(WrapMode::ClampToEdge, -0.25), red);
        assert_vec4_approx(sample(WrapMode::ClampToEdge, 1.25), green);
        assert_vec4_approx(sample(WrapMode::MirroredRepeat, -0.25), red);
        assert_vec4_approx(sample(WrapMode::MirroredRepeat, 1.25), green);
        assert_vec4_approx(sample(WrapMode::MirroredRepeat, 1.75), red);
    }

    #[test]
    fn magnification_filter_selects_nearest_or_linear_sampling() {
        let sample = |mag_filter| {
            binding_with_sampler(
                two_texel_image(),
                SamplerState {
                    wrap_u: WrapMode::ClampToEdge,
                    wrap_v: WrapMode::ClampToEdge,
                    mag_filter,
                    min_filter: MinFilter::Linear,
                },
            )
            .sample(0.5, 0.5)
        };

        assert_vec4_approx(sample(MagFilter::Nearest), Vector4::new(0.0, 1.0, 0.0, 1.0));
        assert_vec4_approx(sample(MagFilter::Linear), Vector4::new(0.5, 0.5, 0.0, 1.0));
    }

    #[test]
    fn density_selects_generated_mip_levels() {
        let texture = binding_with_sampler(
            mip_selection_image(),
            SamplerState {
                wrap_u: WrapMode::ClampToEdge,
                wrap_v: WrapMode::ClampToEdge,
                min_filter: MinFilter::LinearMipmapLinear,
                ..Default::default()
            },
        );

        assert_vec4_approx(
            texture.sample_with_density(0.125, 0.125, 0.25),
            data_gray(0),
        );
        assert_vec4_approx(
            texture.sample_with_density(0.125, 0.125, 0.5),
            data_gray(64),
        );
        assert_vec4_approx(
            texture.sample_with_density(0.125, 0.125, 1.0),
            data_gray(112),
        );
    }

    #[test]
    fn non_mip_minification_filters_stay_on_base_level() {
        for (min_filter, expected) in [
            (MinFilter::Nearest, data_gray(128)),
            (MinFilter::Linear, data_gray(64)),
        ] {
            let sampler = SamplerState {
                wrap_u: WrapMode::ClampToEdge,
                wrap_v: WrapMode::ClampToEdge,
                mag_filter: match min_filter {
                    MinFilter::Nearest => MagFilter::Nearest,
                    MinFilter::Linear => MagFilter::Linear,
                    _ => unreachable!(),
                },
                min_filter,
            };
            let texture = binding_with_sampler(mip_selection_image(), sampler);
            assert_vec4_approx(texture.sample_with_density(0.25, 0.25, 1.0), expected);
        }
    }

    #[test]
    fn mip_minification_filters_select_and_blend_levels() {
        let sample = |min_filter, density| {
            binding_with_sampler(
                mip_selection_image(),
                SamplerState {
                    wrap_u: WrapMode::ClampToEdge,
                    wrap_v: WrapMode::ClampToEdge,
                    min_filter,
                    ..Default::default()
                },
            )
            .sample_with_density(0.125, 0.125, density)
        };
        let halfway_density = 2.0_f32.sqrt() / 4.0;
        for (result, expected) in [
            (sample(MinFilter::NearestMipmapNearest, 0.5), data_gray(64)),
            (sample(MinFilter::LinearMipmapNearest, 0.5), data_gray(64)),
            (
                sample(MinFilter::NearestMipmapLinear, halfway_density),
                data_gray(32),
            ),
            (
                sample(MinFilter::LinearMipmapLinear, halfway_density),
                data_gray(32),
            ),
        ] {
            assert_vec4_approx(result, expected);
        }
    }

    #[test]
    fn color_binding_decodes_texels_before_filtering_and_preserves_alpha() {
        let image = Arc::new(two_texel_image());
        let sampler = SamplerState {
            wrap_u: WrapMode::ClampToEdge,
            wrap_v: WrapMode::ClampToEdge,
            mag_filter: MagFilter::Linear,
            min_filter: MinFilter::Linear,
        };
        let data = TextureBinding::new(
            image.clone(),
            sampler,
            TexCoordSet::TexCoord0,
            TextureUsage::Data,
        )
        .sample(0.5, 0.5);
        let color =
            TextureBinding::new(image, sampler, TexCoordSet::TexCoord0, TextureUsage::Color)
                .sample(0.5, 0.5);

        assert_vec4_approx(color, Vector4::new(0.5, 0.5, 0.0, 1.0));
        assert_vec4_approx(color, data);
        assert_eq!(color.w, data.w);
    }

    #[test]
    fn color_and_data_bindings_interpret_mid_gray_differently() {
        let image = Arc::new(TextureImage::from_image(
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(1, 1, Rgba([128, 128, 128, 128]))),
            false,
        ));
        let sample = |usage| {
            TextureBinding::new(
                image.clone(),
                SamplerState::default(),
                TexCoordSet::TexCoord0,
                usage,
            )
            .sample(0.5, 0.5)
        };

        let encoded = 128.0 / 255.0;
        let linear = srgb_to_linear(Vector3::repeat(encoded)).x;
        assert_vec4_approx(
            sample(TextureUsage::Color),
            Vector4::new(linear, linear, linear, encoded),
        );
        assert_vec4_approx(sample(TextureUsage::Data), Vector4::repeat(encoded));
    }

    #[test]
    fn trilinear_color_filtering_blends_decoded_mip_texels() {
        let texture = TextureBinding::new(
            Arc::new(TextureImage::from_image(
                DynamicImage::ImageRgba8(
                    RgbaImage::from_vec(2, 1, vec![128, 128, 128, 255, 255, 255, 255, 255])
                        .expect("test pixels should match image dimensions"),
                ),
                true,
            )),
            SamplerState {
                min_filter: MinFilter::LinearMipmapLinear,
                ..Default::default()
            },
            TexCoordSet::TexCoord0,
            TextureUsage::Color,
        );
        let density_for_halfway_lod = 2.0_f32.sqrt() / 2.0;
        let sample = texture.sample_with_density(0.5, 0.5, density_for_halfway_lod);
        let gray_linear = srgb_to_linear(Vector3::repeat(128.0 / 255.0)).x;
        let expected = (gray_linear + 1.0) * 0.5;

        assert_vec4_approx(sample, Vector4::new(expected, expected, expected, 1.0));
    }

    #[test]
    fn non_square_and_single_axis_images_generate_complete_mip_chains() {
        for (width, height, expected) in [
            (4, 4, vec![(4, 4), (2, 2), (1, 1)]),
            (4, 2, vec![(4, 2), (2, 1), (1, 1)]),
            (3, 2, vec![(3, 2), (2, 1), (1, 1)]),
            (1, 4, vec![(1, 4), (1, 2), (1, 1)]),
            (4, 1, vec![(4, 1), (2, 1), (1, 1)]),
        ] {
            let texture = TextureImage::from_image(DynamicImage::new_rgba8(width, height), true);
            let dimensions: Vec<_> = texture
                .mips(TextureUsage::Data)
                .iter()
                .map(|level| (level.width, level.height))
                .collect();
            assert_eq!(dimensions, expected);
        }
    }

    #[test]
    fn color_mips_average_linear_texels_while_data_mips_average_raw_values() {
        let image = Arc::new(TextureImage::from_image(
            DynamicImage::ImageRgba8(
                RgbaImage::from_vec(2, 1, vec![128, 128, 128, 255, 255, 255, 255, 255])
                    .expect("test pixels should match image dimensions"),
            ),
            true,
        ));
        let sampler = SamplerState {
            min_filter: MinFilter::NearestMipmapNearest,
            ..Default::default()
        };
        let sample = |usage| {
            TextureBinding::new(image.clone(), sampler, TexCoordSet::TexCoord0, usage)
                .sample_with_density(0.5, 0.5, 1.0)
        };

        let encoded_gray = 128.0 / 255.0;
        let linear_gray = srgb_to_linear(Vector3::repeat(encoded_gray)).x;
        let color_average = (linear_gray + 1.0) * 0.5;
        let data_average = (encoded_gray + 1.0) * 0.5;
        assert_vec4_approx(
            sample(TextureUsage::Color),
            Vector4::new(color_average, color_average, color_average, 1.0),
        );
        assert_vec4_approx(
            sample(TextureUsage::Data),
            Vector4::new(data_average, data_average, data_average, 1.0),
        );
    }

    #[test]
    fn normal_mips_renormalize_averaged_vectors() {
        let image = TextureImage::from_image(
            DynamicImage::ImageRgba8(
                RgbaImage::from_vec(2, 1, vec![255, 128, 128, 255, 128, 255, 128, 255])
                    .expect("test pixels should match image dimensions"),
            ),
            true,
        );
        let mip = &image.mips(TextureUsage::Normal)[1];
        let encoded = mip.pixel(0, 0).xyz();
        let normal = encoded * 2.0 - Vector3::repeat(1.0);

        assert!((normal.norm() - 1.0).abs() < 1.0e-5);
        assert!(normal.x > 0.7 && normal.y > 0.7);
    }
}
