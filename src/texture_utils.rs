use image::{Rgba, RgbaImage};
use nalgebra::Vector3;
use std::path::Path;

/// 表示具有维度和RGBA像素数据的纹理（f32，0.0-1.0范围）
#[derive(Debug, Clone, Default)]
pub struct Texture {
    pub width: u32,
    pub height: u32,
    /// 扁平化的RGBA像素数据（f32，0.0-1.0范围）。布局：[R, G, B, A, R, G, B, A, ...]
    pub data: Vec<f32>,
}

impl Texture {
    /// 创建具有给定颜色的默认1x1纹理
    fn default_texture(color: Vector3<f32>) -> Self {
        Texture {
            width: 1,
            height: 1,
            data: vec![color.x, color.y, color.z, 1.0], // RGBA
        }
    }

    /// 从文件加载纹理
    pub fn load_from_file<P: AsRef<Path>>(texture_path: P) -> Result<Self, std::io::Error> {
        let path_ref = texture_path.as_ref();

        match image::open(path_ref) {
            Ok(img) => {
                let rgba_img: RgbaImage = img.into_rgba8();
                let (width, height) = rgba_img.dimensions();

                let data_f32: Vec<f32> = rgba_img
                    .into_raw()
                    .into_iter()
                    .map(|byte| f32::from(byte) / 255.0)
                    .collect();

                println!("成功加载纹理: {:?} ({}x{})", path_ref, width, height);
                Ok(Texture {
                    width,
                    height,
                    data: data_f32,
                })
            }
            Err(e) => {
                println!("无法加载纹理 '{:?}': {}", path_ref, e);
                Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("纹理加载失败: {}", e),
                ))
            }
        }
    }

    /// 使用UV坐标对纹理进行采样（最近邻插值）
    /// UV坐标应在[0.0, 1.0]范围内
    /// 处理环绕（重复纹理）
    pub fn sample(&self, u: f32, v: f32) -> Rgba<f32> {
        if self.width == 0 || self.height == 0 || self.data.is_empty() {
            return Rgba([0.0, 0.0, 0.0, 1.0]); // 不应该发生，因为有默认纹理
        }

        // 环绕UV坐标
        let u_wrapped = u.fract(); // 保留小数部分以实现环绕
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

        // 计算像素坐标（最近邻）
        // 翻转V坐标，因为图像原点在左上角，而纹理V原点通常在左下角
        let x = (u_final * self.width as f32) as u32 % self.width;
        let y = ((1.0 - v_final) * self.height as f32) as u32 % self.height; // 在这里翻转V

        let index = (y * self.width + x) as usize * 4; // 4个分量(RGBA)

        if index + 3 < self.data.len() {
            Rgba([
                self.data[index],
                self.data[index + 1],
                self.data[index + 2],
                self.data[index + 3],
            ])
        } else {
            // 如果索引计算正确，理论上不应该发生
            Rgba([1.0, 0.0, 1.0, 1.0]) // 洋红色错误颜色
        }
    }
}

/// 从给定路径加载纹理图像文件
///
/// 如果加载失败或路径无效，则返回1x1默认纹理
/// 将像素值归一化为f32 [0.0, 1.0]
pub fn load_texture<P: AsRef<Path>>(texture_path: P, default_color: Vector3<f32>) -> Texture {
    match Texture::load_from_file(&texture_path) {
        Ok(texture) => texture,
        Err(e) => {
            println!("加载纹理'{:?}'时出错: {}", texture_path.as_ref(), e);
            println!("使用1x1默认颜色纹理: {:?}", default_color);
            Texture::default_texture(default_color)
        }
    }
}

// 稍后可以在此处添加程序化纹理生成（如需要）
// fn generate_procedural_texture(...) -> Texture { ... }
