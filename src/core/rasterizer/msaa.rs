use nalgebra::{Point2, Vector3};

/// MSAA采样点偏移模式
pub struct MSAAPattern {
    pub offsets: &'static [(f32, f32)],
}

impl MSAAPattern {
    /// 获取MSAA采样模式 - 使用标准的采样点分布
    pub fn get_pattern(sample_count: u32) -> Self {
        match sample_count {
            1 => Self {
                offsets: &[(0.0, 0.0)],
            },
            2 => Self {
                // 标准2x MSAA：对角线分布
                offsets: &[(-0.25, -0.25), (0.25, 0.25)],
            },
            4 => Self {
                // 标准4x MSAA：Rotated Grid模式
                offsets: &[
                    (-0.125, -0.375),
                    (0.375, -0.125),
                    (-0.375, 0.125),
                    (0.125, 0.375),
                ],
            },
            8 => Self {
                // 8x MSAA：优化的8点模式
                offsets: &[
                    (-0.0625, -0.1875),
                    (0.0625, 0.1875),
                    (-0.3125, 0.0625),
                    (0.1875, -0.3125),
                    (-0.1875, 0.3125),
                    (0.3125, -0.0625),
                    (-0.4375, -0.4375),
                    (0.4375, 0.4375),
                ],
            },
            _ => Self::get_pattern(1), // 默认无MSAA
        }
    }
}

/// MSAA采样结果
#[derive(Debug, Clone)]
pub struct MSAASample {
    pub color: Vector3<f32>,
    pub depth: f32,
    pub hit: bool,
}

impl Default for MSAASample {
    fn default() -> Self {
        Self {
            color: Vector3::zeros(),
            depth: f32::INFINITY,
            hit: false,
        }
    }
}

pub fn resolve_msaa_samples(samples: &[MSAASample], _pattern: &MSAAPattern) -> (Vector3<f32>, f32) {
    let hit_samples: Vec<_> = samples.iter().filter(|s| s.hit).collect();

    if hit_samples.is_empty() {
        return (Vector3::zeros(), f32::INFINITY);
    }

    let total_color: Vector3<f32> = hit_samples.iter().map(|s| s.color).sum();
    let avg_color = total_color / hit_samples.len() as f32;

    // 深度使用最近值（用于深度测试）
    let min_depth = hit_samples
        .iter()
        .map(|s| s.depth)
        .fold(f32::INFINITY, f32::min);

    (avg_color, min_depth)
}

/// 生成像素中心的MSAA采样点
pub fn generate_sample_points(
    pixel_x: usize,
    pixel_y: usize,
    pattern: &MSAAPattern,
) -> Vec<Point2<f32>> {
    let center_x = pixel_x as f32 + 0.5;
    let center_y = pixel_y as f32 + 0.5;

    pattern
        .offsets
        .iter()
        .map(|(dx, dy)| Point2::new(center_x + dx, center_y + dy))
        .collect()
}
