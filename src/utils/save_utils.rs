use crate::core::renderer::Renderer;
use crate::io::render_settings::RenderSettings;
use crate::material_system::color::apply_colormap_jet;
use image::ColorType;
use log::{debug, info, warn};
use std::path::Path;

/// 保存RGB图像数据到PNG文件
pub fn save_image(path: &str, data: &[u8], width: u32, height: u32) {
    match image::save_buffer(path, data, width, height, ColorType::Rgb8) {
        Ok(_) => info!("图像已保存到 {path}"),
        Err(e) => warn!("保存图像到 {path} 时出错: {e}"),
    }
}

/// 将深度缓冲数据归一化到指定的百分位数范围
pub fn normalize_depth(depth_buffer: &[f32], min_percentile: f32, max_percentile: f32) -> Vec<f32> {
    // 1. 收集所有有限的深度值
    let mut finite_depths: Vec<f32> = depth_buffer
        .iter()
        .filter(|&&d| d.is_finite())
        .cloned()
        .collect();

    let mut min_clip: f32;
    let mut max_clip: f32;

    // 2. 使用百分位数确定归一化范围
    if finite_depths.len() >= 2 {
        finite_depths.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        let min_idx = ((min_percentile / 100.0 * (finite_depths.len() - 1) as f32).round()
            as usize)
            .clamp(0, finite_depths.len() - 1);
        let max_idx = ((max_percentile / 100.0 * (finite_depths.len() - 1) as f32).round()
            as usize)
            .clamp(0, finite_depths.len() - 1);

        min_clip = finite_depths[min_idx];
        max_clip = finite_depths[max_idx];

        if (max_clip - min_clip).abs() < 1e-6 {
            min_clip = *finite_depths.first().unwrap();
            max_clip = *finite_depths.last().unwrap();
            if (max_clip - min_clip).abs() < 1e-6 {
                max_clip = min_clip + 1.0;
            }
        }
        debug!(
            "使用百分位数归一化深度: [{min_percentile:.1}%, {max_percentile:.1}%] -> [{min_clip:.3}, {max_clip:.3}]"
        );
    } else {
        warn!("没有足够的有限深度值进行百分位裁剪。使用默认范围 [0.1, 10.0]");
        min_clip = 0.1;
        max_clip = 10.0;
    }

    let range = max_clip - min_clip;
    let inv_range = if range > 1e-6 { 1.0 / range } else { 0.0 };

    depth_buffer
        .iter()
        .map(|&depth| {
            if depth.is_finite() {
                ((depth.clamp(min_clip, max_clip) - min_clip) * inv_range).clamp(0.0, 1.0)
            } else {
                1.0
            }
        })
        .collect()
}

/// 保存渲染结果（彩色图像和可选的深度图）
#[allow(clippy::too_many_arguments)]
pub fn save_render_result(
    color_data: &[u8],
    depth_data: Option<&[f32]>,
    width: usize,
    height: usize,
    output_dir: &str,
    output_name: &str,
    settings: &RenderSettings,
    save_depth: bool,
) -> Result<(), String> {
    // 保存彩色图像
    let color_path = Path::new(output_dir)
        .join(format!("{output_name}_color.png"))
        .to_str()
        .ok_or_else(|| "创建彩色输出路径字符串失败".to_string())?
        .to_string();

    save_image(&color_path, color_data, width as u32, height as u32);

    // 保存深度图（如果启用）
    if settings.use_zbuffer && save_depth {
        if let Some(depth_data_raw) = depth_data {
            let depth_normalized = normalize_depth(depth_data_raw, 1.0, 99.0);
            let depth_colored = apply_colormap_jet(
                &depth_normalized
                    .iter()
                    .map(|&d| 1.0 - d) // 反转：越近 = 越热
                    .collect::<Vec<_>>(),
                width,
                height,
                settings.use_gamma,
            );

            let depth_path = Path::new(output_dir)
                .join(format!("{output_name}_depth.png"))
                .to_str()
                .ok_or_else(|| "创建深度输出路径字符串失败".to_string())?
                .to_string();

            save_image(&depth_path, &depth_colored, width as u32, height as u32);
        }
    }

    Ok(())
}

/// 从渲染器中获取数据并保存渲染结果
pub fn save_render_with_settings(
    renderer: &Renderer,
    settings: &RenderSettings,
    output_name: Option<&str>,
) -> Result<(), String> {
    let color_data = renderer.frame_buffer.get_color_buffer_bytes();
    let depth_data = if settings.save_depth {
        Some(renderer.frame_buffer.get_depth_buffer_f32())
    } else {
        None
    };

    let width = renderer.frame_buffer.width;
    let height = renderer.frame_buffer.height;
    let output_name = output_name.unwrap_or(&settings.output);

    save_render_result(
        &color_data,
        depth_data.as_deref(),
        width,
        height,
        &settings.output_dir,
        output_name,
        settings,
        settings.save_depth,
    )
}
