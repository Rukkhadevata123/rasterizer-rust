use crate::core::renderer::{RenderConfig, Renderer};
use crate::io::args::Args;
use crate::materials::color::apply_colormap_jet;
use crate::utils::image_utils::{normalize_depth, save_image};
use std::path::Path;

/// 保存渲染结果（彩色图像和可选的深度图）
///
/// # 参数
/// * `color_data` - 渲染的颜色数据（RGB格式的u8数组）
/// * `depth_data` - 可选的深度数据（f32数组）
/// * `width` - 图像宽度
/// * `height` - 图像高度
/// * `output_dir` - 输出目录路径
/// * `output_name` - 输出文件名（不含扩展名和后缀）
/// * `config` - 渲染配置
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
    config: &RenderConfig,
    save_depth: bool,
) -> Result<(), String> {
    // 保存彩色图像
    let color_path = Path::new(output_dir)
        .join(format!("{}_color.png", output_name))
        .to_str()
        .ok_or_else(|| "创建彩色输出路径字符串失败".to_string())?
        .to_string();

    // 使用image_utils.rs中的save_image函数
    save_image(&color_path, color_data, width as u32, height as u32);

    // 保存深度图（如果启用）
    if config.use_zbuffer && save_depth {
        if let Some(depth_data_raw) = depth_data {
            let depth_normalized = normalize_depth(depth_data_raw, 1.0, 99.0);
            let depth_colored = apply_colormap_jet(
                &depth_normalized
                    .iter()
                    .map(|&d| 1.0 - d) // 反转：越近 = 越热
                    .collect::<Vec<_>>(),
                width,
                height,
                config.apply_gamma_correction,
            );

            let depth_path = Path::new(output_dir)
                .join(format!("{}_depth.png", output_name))
                .to_str()
                .ok_or_else(|| "创建深度输出路径字符串失败".to_string())?
                .to_string();

            // 使用image_utils.rs中的save_image函数
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
/// * `args` - 命令行参数引用，包含输出路径信息
/// * `config` - 渲染配置引用
/// * `output_name` - 自定义输出名称（如果为None，则使用args.output）
///
/// # 返回值
/// Result，成功为()，失败为包含错误信息的字符串
pub fn save_render_with_args(
    renderer: &Renderer,
    args: &Args,
    config: &RenderConfig,
    output_name: Option<&str>,
) -> Result<(), String> {
    let color_data = renderer.frame_buffer.get_color_buffer_bytes();
    let depth_data = if args.save_depth {
        Some(renderer.frame_buffer.get_depth_buffer_f32())
    } else {
        None
    };

    let width = renderer.frame_buffer.width;
    let height = renderer.frame_buffer.height;
    let output_name = output_name.unwrap_or(&args.output);

    save_render_result(
        &color_data,
        depth_data.as_deref(),
        width,
        height,
        &args.output_dir,
        output_name,
        config,
        args.save_depth,
    )
}
