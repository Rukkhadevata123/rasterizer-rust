use crate::material_system::IMaterial;
use crate::texture_utils::Texture; // 添加Texture的导入
use clap::Parser;
use nalgebra::{Matrix4, Point3, Vector3};
use std::fs;
use std::path::Path;
use std::time::Instant; // 添加IMaterial导入

// Declare modules
mod args;
mod camera;
mod color_utils;
mod interpolation;
mod lighting;
mod loaders;
mod material_system; // 添加这一行，声明新的材质系统模块
mod model_types; // Added module declaration
mod rasterizer;
mod renderer;
mod scene;
mod scene_object; // Add this line
mod texture_utils;
mod transform;
mod utils; // Add this line // 添加这一行，声明新的场景管理模块

// Use statements
use args::{Args, parse_point3, parse_vec3};
use camera::Camera;
use color_utils::apply_colormap_jet;
use lighting::Light; // Existing Light import
use loaders::load_obj_enhanced;
use model_types::ModelData;
use renderer::{RenderSettings, Renderer};
use scene::Scene; // 添加 Scene 导入
use scene_object::SceneObject;
use utils::{normalize_and_center_model, normalize_depth, save_image};

/// 创建并设置场景
fn setup_scene(mut model_data: ModelData, args: &Args) -> Result<Scene, String> {
    // 创建相机
    let aspect_ratio = args.width as f32 / args.height as f32;
    let camera_from = parse_point3(&args.camera_from)
        .map_err(|e| format!("Invalid camera_from format: {}", e))?;
    let camera_at =
        parse_point3(&args.camera_at).map_err(|e| format!("Invalid camera_at format: {}", e))?;
    let camera_up =
        parse_vec3(&args.camera_up).map_err(|e| format!("Invalid camera_up format: {}", e))?;

    let camera = Camera::new(
        camera_from,
        camera_at,
        camera_up,
        args.camera_fov,
        aspect_ratio,
        0.1,   // near plane distance
        100.0, // far plane distance
    );

    // 创建场景并设置相机
    let mut scene = Scene::new(camera);

    // 如果启用了PBR，为模型材质设置金属度和粗糙度
    if args.use_pbr {
        for material in &mut model_data.materials {
            // 确保PBR材质存在
            let pbr_material = material.ensure_pbr_material();

            // 使用命令行参数设置金属度和粗糙度
            pbr_material.metallic = args.metallic;
            pbr_material.roughness = args.roughness;

            // 解析并设置基础颜色
            if let Ok(base_color) = parse_vec3(&args.base_color) {
                pbr_material.base_color = base_color;
            } else {
                println!(
                    "警告: 无法解析基础颜色, 使用默认值: {:?}",
                    pbr_material.base_color
                );
            }

            // 设置环境光遮蔽
            pbr_material.ambient_occlusion = args.ambient_occlusion;

            // 解析并设置自发光颜色
            if let Ok(emissive) = parse_vec3(&args.emissive) {
                pbr_material.emissive = emissive;
            } else {
                println!(
                    "警告: 无法解析自发光颜色, 使用默认值: {:?}",
                    pbr_material.emissive
                );
            }

            println!(
                "应用PBR材质 - 基础色: {:?}, 金属度: {:.2}, 粗糙度: {:.2}, 环境光遮蔽: {:.2}, 自发光: {:?}",
                pbr_material.base_color,
                pbr_material.metallic,
                pbr_material.roughness,
                pbr_material.ambient_occlusion,
                pbr_material.emissive
            );
        }
    }

    // 添加模型
    let model_id = scene.add_model(model_data);

    // 添加主对象
    let main_object = SceneObject::new_default(model_id);
    scene.add_object(main_object, Some("main"));

    // 如果需要，添加更多对象实例
    if let Some(count_str) = &args.object_count {
        if let Ok(count) = count_str.parse::<usize>() {
            if count > 1 {
                // 创建环形对象阵列
                let radius = 2.0;
                scene.create_object_ring(model_id, count - 1, radius, Some("satellite"));
                println!("创建了环形排列的 {} 个附加对象", count - 1);
            }
        }
    }

    // 设置光照
    if args.no_lighting {
        println!("光照已禁用。使用环境光。");
        let _ambient_light =
            scene.create_ambient_light(Vector3::new(args.ambient, args.ambient, args.ambient));

        // 使用 get_type_name 方法 (解决警告)
        if let Some(light) = scene.lights.first() {
            println!("光源类型: {}", light.get_type_name());
        }
    } else {
        let light_intensity = Vector3::new(1.0, 1.0, 1.0) * args.diffuse;

        match args.light_type.to_lowercase().as_str() {
            "point" => {
                let light_pos = parse_point3(&args.light_pos)
                    .map_err(|e| format!("Invalid light_pos format: {}", e))?;
                let atten_parts: Vec<Result<f32, _>> = args
                    .light_atten
                    .split(',')
                    .map(|s| s.trim().parse::<f32>())
                    .collect();

                if atten_parts.len() != 3 || atten_parts.iter().any(|r| r.is_err()) {
                    return Err(format!(
                        "Invalid light_atten format: '{}'. Expected 'c,l,q'",
                        args.light_atten
                    ));
                }

                let attenuation = (
                    atten_parts[0].as_ref().map_or(0.0, |v| *v).max(0.0),
                    atten_parts[1].as_ref().map_or(0.0, |v| *v).max(0.0),
                    atten_parts[2].as_ref().map_or(0.0, |v| *v).max(0.0),
                );

                println!(
                    "使用点光源，位置: {:?}, 强度系数: {:.2}, 衰减: {:?}",
                    light_pos, args.diffuse, attenuation
                );
                scene.create_point_light(light_pos, light_intensity, attenuation);
            }
            _ => {
                // 默认为定向光
                let mut light_dir = parse_vec3(&args.light_dir)
                    .map_err(|e| format!("Invalid light_dir format: {}", e))?;
                light_dir = -light_dir.normalize(); // 朝向光源的方向

                println!(
                    "使用定向光，方向: {:?}, 强度系数: {:.2}",
                    light_dir, args.diffuse
                );
                scene.create_directional_light(light_dir, light_intensity);
            }
        }
    }

    Ok(scene)
}

/// 渲染单帧
fn render_single_frame(
    args: &Args,
    scene: &Scene,
    renderer: &Renderer,
    settings: &RenderSettings,
    output_name: &str,
) -> Result<(), String> {
    let frame_start_time = Instant::now();
    println!("渲染帧: {}", output_name);

    // 使用 renderer.render_scene 渲染整个场景
    renderer.render_scene(scene, settings);

    // --- 保存输出帧 ---
    println!("保存 {} 的输出图像...", output_name);
    let color_data = renderer.frame_buffer.get_color_buffer_bytes();
    let color_path = Path::new(&args.output_dir)
        .join(format!("{}_color.png", output_name))
        .to_str()
        .ok_or("Failed to create color output path string")?
        .to_string();
    save_image(
        &color_path,
        &color_data,
        args.width as u32,
        args.height as u32,
    );

    // 保存深度图（如果启用）
    if settings.use_zbuffer && !args.no_depth {
        let depth_data_raw = renderer.frame_buffer.get_depth_buffer_f32();
        let depth_normalized = normalize_depth(&depth_data_raw, 1.0, 99.0);
        let depth_colored = apply_colormap_jet(
            &depth_normalized
                .iter()
                .map(|&d| 1.0 - d) // 反转：越近 = 越热
                .collect::<Vec<_>>(),
            args.width,
            args.height,
            settings.apply_gamma,
        );
        let depth_path = Path::new(&args.output_dir)
            .join(format!("{}_depth.png", output_name))
            .to_str()
            .ok_or("Failed to create depth output path string")?
            .to_string();
        save_image(
            &depth_path,
            &depth_colored,
            args.width as u32,
            args.height as u32,
        );
    }

    // 添加材质信息（如果有）
    for (i, material) in scene.models[0].materials.iter().enumerate() {
        println!("材质 #{}: {}", i, material.get_name());
        println!("  漫反射颜色: {:?}", material.diffuse);

        // 检查材质是否不透明
        if !material.is_opaque() {
            println!("  透明度: {:.2}", material.get_opacity());
        }

        // 确保PBR材质存在（对每个材质调用一次，以消除警告）
        if args.use_pbr && i == 0 {
            let mut material_copy = material.clone();
            let pbr_material = material_copy.ensure_pbr_material();
            println!(
                "  已创建PBR材质 - 基础色: {:?}, 金属度: {:.2}, 粗糙度: {:.2}",
                pbr_material.base_color, pbr_material.metallic, pbr_material.roughness
            );
        }
    }

    println!(
        "帧 {} 渲染完成，耗时 {:?}",
        output_name,
        frame_start_time.elapsed()
    );
    Ok(())
}

/// 运行动画循环
fn run_animation_loop(args: &Args, scene: &mut Scene, renderer: &Renderer) -> Result<(), String> {
    let total_frames = args.total_frames;
    println!("开始动画渲染 ({} 帧)...", total_frames);

    // 计算每帧旋转增量
    let rotation_increment = 360.0 / total_frames as f32;

    for frame_num in 0..total_frames {
        let frame_start_time = Instant::now();
        println!("--- 准备帧 {} ---", frame_num);

        // 更新相机位置（使用增量旋转）
        if frame_num > 0 {
            // 克隆并旋转当前相机（只旋转一个增量）
            let mut camera = scene.active_camera.clone();
            camera.orbit_y(rotation_increment); // 每帧只旋转增量角度
            scene.set_camera(camera);

            // 为非主对象添加动画效果
            for (i, object) in scene.objects.iter_mut().enumerate() {
                if i > 0 {
                    // 统一使用局部旋转和一致的增量
                    let object_rotation_increment = rotation_increment * 0.5; // 调整速度
                    object.rotate_local(&Vector3::y_axis(), object_rotation_increment.to_radians());

                    // 使用一致的周期作为缩放变化的基础
                    let normalized_phase = (frame_num as f32 * rotation_increment).to_radians();
                    let scale_factor = 0.9 + 0.1 * normalized_phase.sin().abs();
                    object.scale_local(&Vector3::new(scale_factor, scale_factor, scale_factor));

                    // 小幅上下移动，与旋转周期协调
                    let y_offset = 0.03 * normalized_phase.sin();
                    object.translate(&Vector3::new(0.0, y_offset, 0.0));
                }
            }
        }

        // 创建渲染设置
        let settings = RenderSettings {
            projection_type: args.projection.clone(),
            use_zbuffer: !args.no_zbuffer,
            use_face_colors: args.colorize,
            use_texture: !args.no_texture,
            light: scene
                .lights
                .first()
                .map(|l| l.as_ref().to_light_enum()) // Call to_light_enum on the trait object
                .unwrap_or_else(|| {
                    Light::Ambient(Vector3::new(args.ambient, args.ambient, args.ambient))
                }),
            use_phong: args.use_phong,
            apply_gamma: !args.no_gamma,
            use_pbr: args.use_pbr,
        };

        // 打印当前渲染设置信息
        println!("使用{}渲染", settings.get_lighting_description());

        // 渲染并保存当前帧
        let frame_output_name = format!("frame_{:03}", frame_num);
        // render_single_frame(args, scene, renderer, &settings, &frame_output_name)?;
        // 直接调用 render_scene 进行渲染，然后保存图像
        renderer.render_scene(scene, &settings);

        // 保存图像的逻辑需要从 render_single_frame 中提取或复制过来
        println!("保存 {} 的输出图像...", frame_output_name);
        let color_data = renderer.frame_buffer.get_color_buffer_bytes();
        let color_path = Path::new(&args.output_dir)
            .join(format!("{}_color.png", frame_output_name))
            .to_str()
            .ok_or("Failed to create color output path string")?
            .to_string();
        save_image(
            &color_path,
            &color_data,
            args.width as u32,
            args.height as u32,
        );

        if settings.use_zbuffer && !args.no_depth {
            let depth_data_raw = renderer.frame_buffer.get_depth_buffer_f32();
            let depth_normalized = normalize_depth(&depth_data_raw, 1.0, 99.0);
            let depth_colored = apply_colormap_jet(
                &depth_normalized
                    .iter()
                    .map(|&d| 1.0 - d) // 反转：越近 = 越热
                    .collect::<Vec<_>>(),
                args.width,
                args.height,
                settings.apply_gamma,
            );
            let depth_path = Path::new(&args.output_dir)
                .join(format!("{}_depth.png", frame_output_name))
                .to_str()
                .ok_or("Failed to create depth output path string")?
                .to_string();
            save_image(
                &depth_path,
                &depth_colored,
                args.width as u32,
                args.height as u32,
            );
        }
        println!(
            "帧 {} 渲染完成，耗时 {:?}",
            frame_output_name,
            Instant::now().duration_since(frame_start_time) // Recalculate duration for the frame
        );
    }

    println!("动画渲染完成。");
    Ok(())
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    let start_time = Instant::now();

    // --- 验证输入和设置 ---
    if !Path::new(&args.obj).exists() {
        return Err(format!("错误：输入的 OBJ 文件未找到：{}", args.obj));
    }

    // 确保输出目录存在
    fs::create_dir_all(&args.output_dir)
        .map_err(|e| format!("创建输出目录 '{}' 失败：{}", args.output_dir, e))?;

    // --- 加载模型 ---
    println!("加载模型：{}", args.obj);
    let load_start = Instant::now();
    let mut model_data = load_obj_enhanced(&args.obj, &args)?;
    println!("模型加载耗时 {:?}", load_start.elapsed());

    // --- 归一化模型（一次性） ---
    println!("归一化模型...");
    let norm_start_time = Instant::now();
    let (original_center, scale_factor) = normalize_and_center_model(&mut model_data);
    let norm_duration = norm_start_time.elapsed();
    println!(
        "模型归一化耗时 {:?}。原始中心：{:.3?}，缩放系数：{:.3}",
        norm_duration, original_center, scale_factor
    );

    // --- 创建并设置场景 ---
    println!("创建场景...");
    let mut scene = setup_scene(model_data, &args)?;

    // 使用 with_transform 方法创建一个具有自定义变换的额外对象
    if args.show_debug_info {
        println!("创建自定义变换的测试对象...");

        // 创建一个自定义变换矩阵（稍微倾斜并缩放的变换）
        let custom_transform = Matrix4::new_translation(&Vector3::new(1.0, 1.5, -0.5))  // 平移
            * Matrix4::from_euler_angles(0.2, 0.3, 0.1)  // 旋转（roll, pitch, yaw）
            * Matrix4::new_scaling(0.5); // 等比缩放

        // 使用 with_transform 方法创建对象（避免dead_code警告）
        let custom_object = SceneObject::with_transform(
            0, // 使用第一个模型
            custom_transform,
            None, // 无自定义材质
        );

        scene.add_object(custom_object, Some("custom_transform_test"));
        println!("已添加自定义变换对象");
    }

    println!(
        "创建了包含 {} 个对象、{} 个光源的场景",
        scene.object_count(),
        scene.light_count()
    );

    // --- 创建渲染器 ---
    let renderer = Renderer::new(args.width, args.height);

    // --- 决定模式：动画或单帧 ---
    if args.animate {
        run_animation_loop(&args, &mut scene, &renderer)?;
    } else {
        println!("--- 准备单帧渲染 ---");

        // 创建渲染设置
        let settings = RenderSettings {
            projection_type: args.projection.clone(),
            use_zbuffer: !args.no_zbuffer,
            use_face_colors: args.colorize,
            use_texture: !args.no_texture,
            light: scene
                .lights
                .first()
                .map(|l| l.as_ref().to_light_enum()) // Call to_light_enum on the trait object
                .unwrap_or_else(|| {
                    Light::Ambient(Vector3::new(args.ambient, args.ambient, args.ambient))
                }),
            use_phong: args.use_phong,
            apply_gamma: !args.no_gamma,
            use_pbr: args.use_pbr,
        };

        // 打印当前渲染设置信息
        println!("使用{}渲染", settings.get_lighting_description());

        render_single_frame(&args, &scene, &renderer, &settings, &args.output)?;
    }

    // 最后检查一下场景状态和资源
    if args.show_debug_info {
        println!("\n--- 场景状态 ---");
        println!("模型数量: {}", scene.model_count());
        println!("对象数量: {}", scene.object_count());
        println!("光源数量: {}", scene.light_count());

        // 测试材质系统和光照模型功能
        println!("\n--- 材质系统测试 ---");
        if !scene.models.is_empty() && !scene.models[0].materials.is_empty() {
            use material_system::BlinnPhongLightingModel;

            let material = &scene.models[0].materials[0];
            let light = if let Some(l) = scene.lights.first() {
                l.as_ref().to_light_enum()
            } else {
                Light::Ambient(Vector3::new(0.2, 0.2, 0.2))
            };

            // 使用光照模型
            let lighting_model = BlinnPhongLightingModel::new();
            println!("使用光照模型: {}", lighting_model.get_model_name());

            // 创建一个简单的测试材质和光源列表以测试compute_lighting
            struct TestMaterial {
                diffuse: Vector3<f32>,
            }

            impl std::fmt::Debug for TestMaterial {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    write!(f, "TestMaterial {{ diffuse: {:?} }}", self.diffuse)
                }
            }

            impl material_system::IMaterial for TestMaterial {
                fn compute_response(
                    &self,
                    _light_dir: &Vector3<f32>,
                    _view_dir: &Vector3<f32>,
                    _normal: &Vector3<f32>,
                ) -> Vector3<f32> {
                    self.diffuse
                }

                fn get_ambient_color(&self) -> Vector3<f32> {
                    self.diffuse * 0.2
                }

                // 重写默认实现来使用此方法
                fn get_diffuse_color(&self) -> Vector3<f32> {
                    self.diffuse
                }
            }

            let test_material = TestMaterial {
                diffuse: Vector3::new(0.8, 0.6, 0.4),
            };

            // 创建测试光源
            let light_box: Box<dyn material_system::ILight> = if let Some(l) = scene.lights.first()
            {
                Box::new(l.as_ref().to_light_enum())
            } else {
                Box::new(Light::Ambient(Vector3::new(0.2, 0.2, 0.2)))
            };

            let lights: Vec<Box<dyn material_system::ILight>> = vec![light_box];

            // 计算光照（使用compute_lighting特性方法）
            let point = Point3::new(0.0, 0.0, 0.0);
            let normal = Vector3::new(0.0, 1.0, 0.0);
            let view_dir = Vector3::new(0.0, 0.0, 1.0);

            // 使用ILightingModel特性的compute_lighting方法
            let result1 = material_system::ILightingModel::compute_lighting(
                &lighting_model,
                &test_material,
                &lights,
                &point,
                &normal,
                &view_dir,
            );
            println!("compute_lighting结果: {:?}", result1);
            println!("材质漫反射颜色: {:?}", test_material.get_diffuse_color());

            // 使用特性的get_model_name方法
            let model_name = material_system::ILightingModel::get_model_name(&lighting_model);
            println!("光照模型名称 (trait方法): {}", model_name);

            // 直接测试 render_with_model 方法
            let result2 =
                lighting_model.render_with_model(point, normal, view_dir, &light, material.diffuse);
            println!("渲染结果: {:?}", result2);

            // 测试PBR材质方法
            if args.use_pbr {
                use material_system::PBRMaterial;

                // 创建一个空纹理作为默认值
                let empty_texture = Texture::default();

                let test_pbr = PBRMaterial::new(Vector3::new(0.8, 0.8, 0.8), 0.5, 0.3)
                    .with_base_color_texture(
                        material
                            .diffuse_texture
                            .clone()
                            .unwrap_or(empty_texture.clone()),
                    )
                    .with_metal_rough_ao_texture(
                        material
                            .diffuse_texture
                            .clone()
                            .unwrap_or(empty_texture.clone()),
                    )
                    .with_normal_map(material.diffuse_texture.clone().unwrap_or(empty_texture))
                    .with_emissive(Vector3::new(0.0, 0.0, 0.0), None);

                println!(
                    "创建测试PBR材质: 金属度={:.2}, 粗糙度={:.2}",
                    test_pbr.metallic, test_pbr.roughness
                );
            }
        }

        // 如果命令行指定了测试场景管理功能
        if args.test_scene_management {
            println!("\n执行场景管理测试...");
            // 测试对象查找
            if let Some(obj_id) = scene.find_object_by_name("main") {
                println!("找到主对象，ID: {}", obj_id);

                // 测试对象修改
                if let Some(obj) = scene.get_object_mut(obj_id) {
                    // 修改主对象属性
                    obj.set_position(Point3::new(0.0, 0.1, 0.0));
                    println!("已修改主对象位置");
                }
            }

            // 仅在测试模式下清除场景
            if args.test_clear_scene {
                println!("清除场景对象...");
                scene.clear_objects();
                println!("清除后对象数量: {}", scene.object_count());

                println!("清除场景光源...");
                scene.clear_lights();
                println!("清除后光源数量: {}", scene.light_count());
            }
        }

        // 测试Material的set_name和set_opacity方法
        if args.use_pbr && args.test_materials {
            println!("\n--- 材质设置测试 ---");
            if !scene.models.is_empty() && !scene.models[0].materials.is_empty() {
                // 克隆一个材质以便修改
                let mut test_material = scene.models[0].materials[0].clone();

                // 测试设置名称
                let new_name = "测试PBR材质";
                test_material.set_name(new_name.to_string());
                println!("材质名称已设置为: {}", test_material.get_name());

                // 测试设置透明度
                test_material.set_opacity(0.85);
                println!("材质透明度已设置为: {:.2}", test_material.get_opacity());
                println!("材质是否不透明: {}", test_material.is_opaque());
            }
        }
    }

    println!("总执行时间：{:?}", start_time.elapsed());
    Ok(())
}
