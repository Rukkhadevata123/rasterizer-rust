use crate::core::renderer::{RenderConfig, Renderer};
use crate::io::args::Args;
use crate::scene::scene_utils::Scene;
use crate::utils::render_output::save_render_with_args;
use std::time::Instant;

/// 渲染单帧并保存结果
///
/// 完整处理单帧渲染过程：渲染场景、保存输出、打印信息
///
/// # 参数
/// * `args` - 命令行参数引用
/// * `scene` - 场景引用
/// * `renderer` - 渲染器引用
/// * `config` - 渲染配置引用
/// * `output_name` - 输出文件名
///
/// # 返回值
/// Result，成功为()，失败为包含错误信息的字符串
pub fn render_single_frame(
    args: &Args,
    scene: &Scene,
    renderer: &Renderer,
    config: &RenderConfig,
    output_name: &str,
) -> Result<(), String> {
    let frame_start_time = Instant::now();
    println!("渲染帧: {}", output_name);

    // 渲染场景 - 克隆配置以避免可变引用问题
    let mut config_clone = config.clone();
    renderer.render_scene(scene, &mut config_clone);

    // 保存输出图像
    println!("保存 {} 的输出图像...", output_name);
    save_render_with_args(renderer, args, config, Some(output_name))?;

    // 打印材质信息（调试用）
    if let Some(model) = scene.models.first() {
        for (i, material) in model.materials.iter().enumerate() {
            if i == 0 {
                println!("材质 #{}: {}", i, material.get_name());
                println!("  漫反射颜色: {:?}", material.diffuse());
            }
        }
    }

    println!(
        "帧 {} 渲染完成，耗时 {:?}",
        output_name,
        frame_start_time.elapsed()
    );
    Ok(())
}
