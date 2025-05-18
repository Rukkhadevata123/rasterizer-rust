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

        // --- 设置环境光响应系数 ---
        // 在PBR中，环境光响应是根据材质的物理属性来确定的
        // 非金属材质会散射环境光，而金属则几乎不散射
        let ambient_response = material.ambient_occlusion * (1.0 - material.metallic);
        material.ambient_factor =
            Vector3::new(ambient_response, ambient_response, ambient_response);

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
/// 但调整为更符合物理规律的方式处理环境光
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
            // 移除对环境光的预混合，保持材质的纯净性
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

        // --- 设置环境光响应系数 ---
        // 在Blinn-Phong中，环境光系数通常是漫反射的一个比例
        // 使用0.3作为系数，符合传统渲染中的常用值
        material.ambient_factor = material.albedo * 0.3;

        println!(
            "应用Phong材质 - 漫反射: {:?}, 镜面: {:?}, 光泽度: {:.2}, 自发光: {:?}",
            material.diffuse(),
            material.specular,
            material.shininess,
            material.emissive
        );
    }
}
