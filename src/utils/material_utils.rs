use crate::io::args::{Args, parse_vec3};
use crate::materials::model_types::ModelData;
use nalgebra::Vector3;

/// 应用PBR材质参数
///
/// 根据命令行参数设置模型数据的PBR材质属性，符合基于物理的渲染原则
///
/// # 参数
/// * `model_data` - 要修改的模型数据
/// * `args` - 命令行参数
pub fn apply_pbr_parameters(model_data: &mut ModelData, args: &Args) {
    if !args.use_pbr {
        return;
    }

    for material in &mut model_data.materials {
        // --- 设置PBR特有属性 ---
        material.metallic = args.metallic.clamp(0.0, 1.0);
        material.roughness = args.roughness.clamp(0.0, 1.0);
        material.ambient_occlusion = args.ambient_occlusion.clamp(0.0, 1.0);

        // --- 设置基础颜色 (在PBR中表示材质的反射率) ---
        if let Ok(base_color) = parse_vec3(&args.base_color) {
            // 直接设置基础颜色，PBR中不应预混合环境光
            material.albedo = base_color;

            // 确保能量守恒 - 非金属反射率通常不超过0.04-0.08
            if material.metallic < 0.1 {
                // 对于非金属，限制反射率范围以符合物理规律
                material.albedo = Vector3::new(
                    material.albedo.x.min(0.9),
                    material.albedo.y.min(0.9),
                    material.albedo.z.min(0.9),
                );
            }
        } else {
            println!("警告: 无法解析基础颜色, 使用默认值: {:?}", material.albedo);
        }

        // --- 设置自发光颜色 (在PBR中是独立的加性光源) ---
        if let Ok(emissive) = parse_vec3(&args.emissive) {
            material.emissive = emissive;
        }

        println!(
            "应用PBR材质 - 基础色: {:?}, 金属度: {:.2}, 粗糙度: {:.2}, 环境光遮蔽: {:.2}, 自发光: {:?}",
            material.base_color(),
            material.metallic,
            material.roughness,
            material.ambient_occlusion,
            material.emissive
        );
    }
}

/// 应用Phong材质参数
///
/// 根据命令行参数设置模型数据的Phong材质属性，符合传统Blinn-Phong着色模型
///
/// # 参数
/// * `model_data` - 要修改的模型数据
/// * `args` - 命令行参数
pub fn apply_phong_parameters(model_data: &mut ModelData, args: &Args) {
    if !args.use_phong {
        return;
    }

    for material in &mut model_data.materials {
        // --- 设置Phong特有属性 ---
        material.specular = Vector3::new(args.specular, args.specular, args.specular);
        material.shininess = args.shininess.max(1.0); // 防止shininess为0或负数

        // --- 设置漫反射颜色 (在Phong中直接影响视觉效果) ---
        if let Ok(diffuse_color) = parse_vec3(&args.diffuse_color) {
            material.albedo = diffuse_color;

            // Blinn-Phong中可以适度考虑环境光影响
            // 这是符合传统的做法，但不应过度，避免材质外观失真
            if !args.ambient_color.is_empty() && args.ambient > 0.05 {
                if let Ok(ambient_color) = parse_vec3(&args.ambient_color) {
                    // 使用小系数确保主颜色不会过度改变
                    let blend_factor = (args.ambient * 0.4).clamp(0.0, 0.3);
                    material.albedo = material.albedo.lerp(&ambient_color, blend_factor);
                }
            }
        } else {
            println!(
                "警告: 无法解析漫反射颜色, 使用默认值: {:?}",
                material.diffuse()
            );
        }

        // --- 设置自发光颜色 (与PBR保持一致) ---
        if let Ok(emissive) = parse_vec3(&args.emissive) {
            material.emissive = emissive;
        }

        println!(
            "应用Phong材质 - 漫反射: {:?}, 镜面: {:?}, 光泽度: {:.2}, 自发光: {:?}",
            material.diffuse(),
            material.specular,
            material.shininess,
            material.emissive
        );
    }
}
