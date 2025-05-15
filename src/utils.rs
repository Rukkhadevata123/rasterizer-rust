use crate::model_types::ModelData;
use nalgebra::{Point3, Vector3};

/// 将深度缓冲数据归一化到指定的百分位数范围
pub fn normalize_depth(
    depth_buffer: &[f32],
    min_percentile: f32, // 例如，1.0表示第1百分位
    max_percentile: f32, // 例如，99.0表示第99百分位
) -> Vec<f32> {
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
            "Normalizing depth using percentiles: [{:.1}%, {:.1}%] -> [{:.3}, {:.3}]",
            min_percentile, max_percentile, min_clip, max_clip
        );
    } else {
        // 如果没有足够的有限值，使用后备方案
        println!(
            "Warning: Not enough finite depth values for percentile clipping. Using default range [0.1, 10.0]."
        );
        min_clip = 0.1; // Default near // Assignment
        max_clip = 10.0; // Default far (adjust as needed) // Assignment
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

/// 保存RGB图像数据到PNG文件
pub fn save_image(path: &str, data: &[u8], width: u32, height: u32) {
    match image::save_buffer(path, data, width, height, image::ColorType::Rgb8) {
        Ok(_) => println!("Image saved to {}", path),
        Err(e) => eprintln!("Error saving image to {}: {}", path, e),
    }
}

/// 归一化和中心化模型顶点
pub fn normalize_and_center_model(model_data: &mut ModelData) -> (Vector3<f32>, f32) {
    if model_data.meshes.is_empty() {
        return (Vector3::zeros(), 1.0);
    }

    // 计算所有顶点的边界框或质心
    let mut min_coord = Point3::new(f32::MAX, f32::MAX, f32::MAX);
    let mut max_coord = Point3::new(f32::MIN, f32::MIN, f32::MIN);
    let mut vertex_sum = Vector3::zeros();
    let mut vertex_count = 0;

    for mesh in &model_data.meshes {
        for vertex in &mesh.vertices {
            min_coord = min_coord.inf(&vertex.position);
            max_coord = max_coord.sup(&vertex.position);
            vertex_sum += vertex.position.coords;
            vertex_count += 1;
        }
    }

    if vertex_count == 0 {
        return (Vector3::zeros(), 1.0);
    }

    let center = vertex_sum / (vertex_count as f32);
    let extent = max_coord - min_coord;
    let max_extent = extent.x.max(extent.y).max(extent.z);

    let scale_factor = if max_extent > 1e-6 {
        1.6 / max_extent // 缩放以大致适合[-0.8, 0.8]立方体（类似于Python的0.8因子）
    } else {
        1.0
    };

    // 对所有顶点应用变换
    for mesh in &mut model_data.meshes {
        for vertex in &mut mesh.vertices {
            vertex.position = Point3::from((vertex.position.coords - center) * scale_factor);
        }
    }

    (center, scale_factor)
}
