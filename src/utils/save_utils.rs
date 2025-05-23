use crate::core::renderer::Renderer;
use crate::io::render_settings::RenderSettings; // 替换原来的导入
use crate::material_system::color::apply_colormap_jet;
use image::ColorType;
use std::path::Path;

/// 保存RGB图像数据到PNG文件
///
/// # 参数
/// * `path` - 输出文件路径
/// * `data` - RGB数据（u8数组）
/// * `width` - 图像宽度
/// * `height` - 图像高度
pub fn save_image(path: &str, data: &[u8], width: u32, height: u32) {
    match image::save_buffer(path, data, width, height, ColorType::Rgb8) {
        Ok(_) => println!("图像已保存到 {}", path),
        Err(e) => eprintln!("保存图像到 {} 时出错: {}", path, e),
    }
}

/// 将深度缓冲数据归一化到指定的百分位数范围
///
/// # 参数
/// * `depth_buffer` - 深度数据（f32数组）
/// * `min_percentile` - 最小百分位（例如，1.0表示第1百分位）
/// * `max_percentile` - 最大百分位（例如，99.0表示第99百分位）
///
/// # 返回值
/// 归一化后的深度数据数组（值范围在[0.0, 1.0]之间）
pub fn normalize_depth(depth_buffer: &[f32], min_percentile: f32, max_percentile: f32) -> Vec<f32> {
    // 1. 收集所有有限的深度值
    let mut finite_depths: Vec<f32> = depth_buffer
        .iter()
        .filter(|&&d| d.is_finite())
        .cloned()
        .collect();

    // 声明min_clip和max_clip为可变变量
    let mut min_clip: f32;
    let mut max_clip: f32;

    // 2. 使用百分位数确定归一化范围
    if finite_depths.len() >= 2 {
        // 需要至少两个点来定义一个范围
        // 对有限深度值进行排序
        finite_depths.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap()); // 使用不稳定排序以提高性能

        // 计算百分位数对应的索引
        let min_idx = ((min_percentile / 100.0 * (finite_depths.len() - 1) as f32).round()
            as usize)
            .clamp(0, finite_depths.len() - 1);
        let max_idx = ((max_percentile / 100.0 * (finite_depths.len() - 1) as f32).round()
            as usize)
            .clamp(0, finite_depths.len() - 1);

        min_clip = finite_depths[min_idx]; // 初始赋值
        max_clip = finite_depths[max_idx]; // 初始赋值

        // 确保 min_clip < max_clip
        if (max_clip - min_clip).abs() < 1e-6 {
            // 如果范围太小，则稍微扩大它或使用默认值
            // 为简单起见，在这种边缘情况下使用绝对最小/最大值
            min_clip = *finite_depths.first().unwrap(); // 现在允许重新赋值
            max_clip = *finite_depths.last().unwrap(); // 现在允许重新赋值
            // 即使所有值都相同，也确保最大值 > 最小值
            if (max_clip - min_clip).abs() < 1e-6 {
                max_clip = min_clip + 1.0; // 现在允许重新赋值
            }
        }
        println!(
            "使用百分位数归一化深度: [{:.1}%, {:.1}%] -> [{:.3}, {:.3}]",
            min_percentile, max_percentile, min_clip, max_clip
        );
    } else {
        // 如果没有足够的有限值，使用后备方案
        println!("警告: 没有足够的有限深度值进行百分位裁剪。使用默认范围 [0.1, 10.0]。");
        min_clip = 0.1; // 默认近平面距离
        max_clip = 10.0; // 默认远平面距离（根据需要调整）
    }

    let range = max_clip - min_clip;
    let inv_range = if range > 1e-6 { 1.0 / range } else { 0.0 }; // 避免除以零

    // 3. 使用计算的范围对原始缓冲区进行归一化
    depth_buffer
        .iter()
        .map(|&depth| {
            if depth.is_finite() {
                // 将深度限制在计算的范围内并归一化
                ((depth.clamp(min_clip, max_clip) - min_clip) * inv_range).clamp(0.0, 1.0)
            } else {
                // 将非有限值（无穷大）映射为1.0（远处）
                1.0
            }
        })
        .collect()
}

/// 保存渲染结果（彩色图像和可选的深度图）
///
/// # 参数
/// * `color_data` - 渲染的颜色数据（RGB格式的u8数组）
/// * `depth_data` - 可选的深度数据（f32数组）
/// * `width` - 图像宽度
/// * `height` - 图像高度
/// * `output_dir` - 输出目录路径
/// * `output_name` - 输出文件名（不含扩展名和后缀）
/// * `settings` - 渲染设置
/// * `save_depth` - 是否保存深度图
///
/// # 返回值
/// Result，成功为()，失败为包含错误信息的字符串
#[allow(clippy::too_many_arguments)]
pub fn save_render_result(
    color_data: &[u8],
    depth_data: Option<&[f32]>,
    width: usize,
    height: usize,
    output_dir: &str,
    output_name: &str,
    settings: &RenderSettings, // 改为使用RenderSettings
    save_depth: bool,
) -> Result<(), String> {
    // 保存彩色图像
    let color_path = Path::new(output_dir)
        .join(format!("{}_color.png", output_name))
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
                settings.use_gamma, // 使用settings.use_gamma
            );

            let depth_path = Path::new(output_dir)
                .join(format!("{}_depth.png", output_name))
                .to_str()
                .ok_or_else(|| "创建深度输出路径字符串失败".to_string())?
                .to_string();

            save_image(&depth_path, &depth_colored, width as u32, height as u32);
        }
    }

    Ok(())
}

/// 从渲染器中获取数据并保存渲染结果
///
/// 这是对 save_render_result 的便捷包装函数，适用于所有调用场景
///
/// # 参数
/// * `renderer` - 渲染器引用，用于获取渲染数据
/// * `settings` - 渲染设置引用
/// * `output_name` - 自定义输出名称（如果为None，则使用settings.output）
///
/// # 返回值
/// Result，成功为()，失败为包含错误信息的字符串
pub fn save_render_with_settings(
    // 重命名函数以避免歧义
    renderer: &Renderer,
    settings: &RenderSettings, // 替换为RenderSettings
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
