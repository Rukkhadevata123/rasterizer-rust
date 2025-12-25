use nalgebra::Vector3;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// 表示具有浮点分量[0.0, 1.0]的RGB颜色。
pub type Color = Vector3<f32>;

/// 应用gamma矫正，将线性RGB值转换为sRGB空间
///
/// # 参数
/// * `linear_color` - 线性空间的RGB颜色值 [0.0-1.0]
///
/// # 返回值
/// 应用了gamma矫正的RGB颜色值 [0.0-1.0]
pub fn apply_gamma_correction(linear_color: &Color) -> Color {
    // 使用标准的gamma值2.2
    let gamma = 2.2;
    let inv_gamma = 1.0 / gamma;

    // 对每个颜色通道应用幂函数
    Color::new(
        linear_color.x.powf(inv_gamma),
        linear_color.y.powf(inv_gamma),
        linear_color.z.powf(inv_gamma),
    )
}

/// 从sRGB空间转换回线性RGB值（解码）
///
/// # 参数
/// * `srgb_color` - sRGB空间的RGB颜色值 [0.0-1.0]
///
/// # 返回值
/// 线性空间的RGB颜色值 [0.0-1.0]
pub fn srgb_to_linear(srgb_color: &Color) -> Color {
    // 使用标准的gamma值2.2
    let gamma = 2.2;

    // 应用逆变换
    Color::new(
        srgb_color.x.powf(gamma),
        srgb_color.y.powf(gamma),
        srgb_color.z.powf(gamma),
    )
}

/// 应用ACES色调映射，将高动态范围颜色压缩到显示范围
///
/// # 参数
/// * `color` - 线性RGB颜色值 [0.0-1.0]
/// # 返回值
/// 压缩后的RGB颜色值 [0.0-1.0]
pub fn apply_aces_tonemap(color: &Vector3<f32>) -> Vector3<f32> {
    // ACES Filmic Curve参数
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    Vector3::new(
        ((color.x * (a * color.x + b)) / (color.x * (c * color.x + d) + e)).clamp(0.0, 1.0),
        ((color.y * (a * color.y + b)) / (color.y * (c * color.y + d) + e)).clamp(0.0, 1.0),
        ((color.z * (a * color.z + b)) / (color.z * (c * color.z + d) + e)).clamp(0.0, 1.0),
    )
}

/// 将线性RGB值转换为u8数组，应用gamma矫正
///
/// # 参数
/// * `linear_color` - 线性空间的RGB颜色值 [0.0-1.0]
/// * `apply_gamma` - 是否应用gamma矫正
///
/// # 返回值
/// 一个包含三个u8值的数组，表示颜色的RGB通道
pub fn linear_rgb_to_u8(linear_color: &Color, apply_gamma: bool) -> [u8; 3] {
    let display_color = if apply_gamma {
        apply_gamma_correction(linear_color)
    } else {
        *linear_color
    };

    [
        (display_color.x * 255.0).clamp(0.0, 255.0) as u8,
        (display_color.y * 255.0).clamp(0.0, 255.0) as u8,
        (display_color.z * 255.0).clamp(0.0, 255.0) as u8,
    ]
}

/// 获取基于种子的随机颜色。
///
/// 如果`colorize`为false，返回默认的灰色。
/// 如果`colorize`为true，根据种子生成伪随机颜色
/// （对于相同的种子，结果是确定性的）。
///
/// # 参数
/// * `seed` - 随机数种子
/// * `colorize` - 是否生成彩色（否则返回默认灰色）
pub fn get_random_color(seed: u64, colorize: bool) -> Color {
    if !colorize {
        // 默认灰色
        Color::new(0.7, 0.7, 0.7)
    } else {
        // 使用种子生成确定性随机颜色
        let mut rng = StdRng::seed_from_u64(seed);
        Color::new(
            0.3 + rng.random::<f32>() * 0.4, // R 在 [0.3, 0.7) 范围内
            0.3 + rng.random::<f32>() * 0.4, // G 在 [0.3, 0.7) 范围内
            0.3 + rng.random::<f32>() * 0.4, // B 在 [0.3, 0.7) 范围内
        )
    }
}

/// 将归一化的深度图（值范围0.0-1.0）转换为使用JET色彩映射的RGB彩色图像。
///
/// 无效的深度值（NaN、无穷大）将显示为黑色像素。
///
/// # 参数
/// * `normalized_depth` - 扁平化的深度值切片（行优先）。
/// * `width` - 深度图的宽度。
/// * `height` - 深度图的高度。
/// * `apply_gamma` - 是否应用gamma矫正
///
/// # 返回值
/// 包含扁平化RGB图像数据的`Vec<u8>`（每个通道0-255）。
pub fn apply_colormap_jet(
    normalized_depth: &[f32],
    width: usize,
    height: usize,
    apply_gamma: bool,
) -> Vec<u8> {
    let num_pixels = width * height;
    if normalized_depth.len() != num_pixels {
        // 或返回一个错误Result
        panic!("Depth buffer size does not match width * height");
    }

    let mut result = vec![0u8; num_pixels * 3]; // 初始化为黑色

    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            let depth = normalized_depth[index];

            if depth.is_finite() {
                let value = depth.clamp(0.0, 1.0); // 确保值在[0, 1]范围内

                let mut r = 0.0;
                let g;
                let mut b = 0.0;

                // 应用JET色彩映射逻辑
                if value <= 0.25 {
                    // 从蓝色到青色
                    b = 1.0;
                    g = value * 4.0;
                } else if value <= 0.5 {
                    // 从青色到绿色
                    g = 1.0;
                    b = 1.0 - (value - 0.25) * 4.0;
                } else if value <= 0.75 {
                    // 从绿色到黄色
                    g = 1.0;
                    r = (value - 0.5) * 4.0;
                } else {
                    // 从黄色到红色
                    r = 1.0;
                    g = 1.0 - (value - 0.75) * 4.0;
                }

                let color = Color::new(r, g, b);
                let [r_u8, g_u8, b_u8] = linear_rgb_to_u8(&color, apply_gamma);

                // 写入结果缓冲区
                let base_index = index * 3;
                result[base_index] = r_u8;
                result[base_index + 1] = g_u8;
                result[base_index + 2] = b_u8;
            }
            // 如果深度值不是有限的，像素保持黑色（初始化为0）
        }
    }

    result
}
