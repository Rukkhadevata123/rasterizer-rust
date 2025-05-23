use crate::material_system::light::LightingPreset;
use egui::{Color32, Context, RichText, Vec2};
use native_dialog::FileDialogBuilder;
use std::sync::atomic::Ordering;

use super::animation::AnimationMethods;
use super::app::RasterizerApp;
use super::core::CoreMethods;
use crate::io::render_settings::{AnimationType, RotationAxis, parse_vec3}; // 更新导入路径

use super::render_ui::RenderMethods;

/// UI组件和工具提示相关方法的特质
pub trait WidgetMethods {
    /// 绘制UI的侧边栏
    fn draw_side_panel(&mut self, ctx: &Context, ui: &mut egui::Ui);

    /// 显示错误对话框
    fn show_error_dialog_ui(&mut self, ctx: &Context);

    /// 显示工具提示
    fn add_tooltip(response: egui::Response, ctx: &Context, text: &str) -> egui::Response;
}

impl WidgetMethods for RasterizerApp {
    /// 显示错误对话框
    fn show_error_dialog_ui(&mut self, ctx: &egui::Context) {
        if self.show_error_dialog {
            egui::Window::new("错误")
                .fixed_size([400.0, 150.0])
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.vertical_centered(|ui| {
                        ui.add_space(10.0);
                        ui.label(
                            RichText::new(&self.error_message)
                                .color(Color32::from_rgb(230, 50, 50))
                                .size(16.0),
                        );
                        ui.add_space(20.0);
                        if ui.button(RichText::new("确定").size(16.0)).clicked() {
                            self.show_error_dialog = false;
                        }
                    });
                });
        }
    }

    /// 显示工具提示
    fn add_tooltip(response: egui::Response, _ctx: &egui::Context, text: &str) -> egui::Response {
        let response = response.on_hover_ui(|ui| {
            ui.add(egui::Label::new(
                RichText::new(text).size(14.0).color(Color32::LIGHT_YELLOW),
            ));
        });

        response.context_menu(|ui| {
            ui.label(text);
            if ui.button("关闭").clicked() {
                ui.close_menu();
            }
        });

        response
    }

    /// 绘制侧边栏
    fn draw_side_panel(&mut self, ctx: &Context, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            // 文件与输出设置
            ui.collapsing("文件与输出设置", |ui| {
                ui.horizontal(|ui| {
                    ui.label("OBJ文件：");
                    // 使用临时变量来处理Option<String>
                    let mut obj_text = self.settings.obj.clone().unwrap_or_default();
                    let response = ui.text_edit_singleline(&mut obj_text);
                    // 如果文本更改，更新settings.obj
                    if response.changed() {
                        if obj_text.is_empty() {
                            self.settings.obj = None;
                        } else {
                            self.settings.obj = Some(obj_text);
                        }
                    }
                    Self::add_tooltip(response, ctx, "选择要渲染的3D模型文件（.obj格式）");
                    if ui.button("浏览").clicked() {
                        self.select_obj_file();
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("输出目录：");
                    let response = ui.text_edit_singleline(&mut self.settings.output_dir);
                    Self::add_tooltip(response, ctx, "选择渲染结果保存的目录");
                    if ui.button("浏览").clicked() {
                        self.select_output_dir();
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("输出文件名：");
                    let response = ui.text_edit_singleline(&mut self.settings.output);
                    Self::add_tooltip(response, ctx, "渲染结果的文件名（不含扩展名）");
                });
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("宽度：");
                    let response = ui.add(egui::DragValue::new(&mut self.settings.width)
                    .speed(1)
                    .range(1..=4096));
                    Self::add_tooltip(response, ctx, "渲染图像的宽度（像素）");
                });

                ui.horizontal(|ui| {
                    ui.label("高度：");
                    let response = ui.add(egui::DragValue::new(&mut self.settings.height)
                    .speed(1)
                    .range(1..=4096));
                    Self::add_tooltip(response, ctx, "渲染图像的高度（像素）");
                });
                let response = ui.checkbox(&mut self.settings.save_depth, "保存深度图");
                Self::add_tooltip(response, ctx, "同时保存深度图（深度信息可视化）");
            });

            // 渲染属性设置
            ui.collapsing("渲染属性设置", |ui| {
                ui.horizontal(|ui| {
                    ui.label("投影类型：");
                    let resp1 = ui.radio_value(
                        &mut self.settings.projection,
                        "perspective".to_string(),
                                               "透视",
                    );
                    Self::add_tooltip(resp1, ctx, "使用透视投影（符合人眼观察方式）");

                    let resp2 = ui.radio_value(
                        &mut self.settings.projection,
                        "orthographic".to_string(),
                                               "正交",
                    );
                    Self::add_tooltip(resp2, ctx, "使用正交投影（无透视变形）");
                });
                ui.separator();
                let resp1 = ui.checkbox(&mut self.settings.use_zbuffer, "深度缓冲");
                Self::add_tooltip(resp1, ctx, "启用Z缓冲进行深度测试，处理物体遮挡关系");

                let resp2 = ui.checkbox(&mut self.settings.use_lighting, "启用光照");
                Self::add_tooltip(resp2, ctx, "启用光照计算，产生明暗变化");

                // 将"启用纹理"和"使用面颜色"改为互斥的单选项
                ui.horizontal(|ui| {
                    ui.label("表面颜色：");

                    // 启用纹理选项
                    let texture_response = ui.radio_value(&mut self.settings.use_texture, true, "使用纹理");
                    if texture_response.clicked() && self.settings.use_texture {
                        // 如果选择了使用纹理，关闭面颜色
                        self.settings.colorize = false;
                    }
                    Self::add_tooltip(texture_response, ctx,
                                      "使用模型的纹理贴图（如果有）\n优先级最高，会覆盖面颜色设置");

                    // 使用面颜色选项
                    let face_color_response = ui.radio_value(&mut self.settings.colorize, true, "使用面颜色");
                    if face_color_response.clicked() && self.settings.colorize {
                        // 如果选择了使用面颜色，关闭纹理
                        self.settings.use_texture = false;
                    }
                    Self::add_tooltip(face_color_response, ctx,
                                      "为每个面分配随机颜色\n仅在没有纹理或纹理被禁用时生效");

                    // 使用材质颜色选项 (实际上是关闭两者)
                    let material_color_response = ui.radio(
                        !self.settings.use_texture && !self.settings.colorize,
                        "使用材质颜色"
                    );
                    if material_color_response.clicked() {
                        self.settings.use_texture = false;
                        self.settings.colorize = false;
                    }
                    Self::add_tooltip(material_color_response, ctx,
                                      "使用材质的基本颜色（如.mtl文件中定义）\n在没有纹理且不使用面颜色时生效");
                });

                // 着色模型选择（Phong/PBR，已经是互斥的）
                ui.horizontal(|ui| {
                    ui.label("着色模型：");
                    // Phong 着色选项（逐像素着色，在 Blinn-Phong 光照模型下）
                    let phong_response = ui.radio_value(&mut self.settings.use_phong, true, "Phong着色");
                    if phong_response.clicked() && self.settings.use_phong {
                        // 如果选择了 Phong，关闭 PBR
                        self.settings.use_pbr = false;
                    }
                    Self::add_tooltip(phong_response, ctx,
                                      "使用 Phong 着色（逐像素着色）和 Blinn-Phong 光照模型\n提供高质量的光照效果，适合大多数场景");

                    // PBR 渲染选项
                    let pbr_response = ui.radio_value(&mut self.settings.use_pbr, true, "PBR渲染");
                    if pbr_response.clicked() && self.settings.use_pbr {
                        // 如果选择了 PBR，关闭 Phong
                        self.settings.use_phong = false;
                    }
                    Self::add_tooltip(pbr_response, ctx,
                                      "使用基于物理的渲染（PBR）\n提供更真实的材质效果，但需要更多的参数调整");
                });

                let resp7 = ui.checkbox(&mut self.settings.use_gamma, "Gamma校正");
                Self::add_tooltip(resp7, ctx, "应用伽马校正，使亮度显示更准确");

                let resp8 = ui.checkbox(&mut self.settings.backface_culling, "背面剔除");
                Self::add_tooltip(resp8, ctx, "剔除背向相机的三角形面，提高渲染效率");

                let resp9 = ui.checkbox(&mut self.settings.wireframe, "线框模式");
                Self::add_tooltip(resp9, ctx, "仅渲染三角形边缘，显示为线框");

                ui.separator();
                let resp10 = ui.checkbox(&mut self.settings.use_multithreading, "启用多线程渲染");
                Self::add_tooltip(resp10, ctx, "使用多线程加速渲染，提高性能");

                ui.horizontal(|ui| {
                    let resp = ui.checkbox(&mut self.settings.cull_small_triangles, "剔除小三角形");
                    Self::add_tooltip(resp, ctx, "忽略投影后面积很小的三角形，提高性能");

                    if self.settings.cull_small_triangles {
                        let resp = ui.add(
                            egui::DragValue::new(&mut self.settings.min_triangle_area)
                            .speed(0.0001)
                            .range(0.0..=1.0)
                            .prefix("面积阈值："),
                        );
                        Self::add_tooltip(resp, ctx, "小于此面积的三角形将被剔除（范围0.0-1.0）");
                    }
                });
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("纹理文件 (覆盖MTL)：");
                    let mut texture_path_str = self.settings.texture.clone().unwrap_or_default();
                    let resp = ui.text_edit_singleline(&mut texture_path_str);
                    Self::add_tooltip(resp.clone(), ctx, "选择自定义纹理，将覆盖MTL中的定义");

                    if resp.changed() {
                        if texture_path_str.is_empty() {
                            self.settings.texture = None;
                        } else {
                            self.settings.texture = Some(texture_path_str);
                        }
                    }
                    if ui.button("浏览").clicked() {
                        let result = FileDialogBuilder::default()
                        .set_title("选择纹理文件")
                        .add_filter("图像文件", ["png", "jpg", "jpeg", "bmp", "tga"])
                        .open_single_file()
                        .show();

                        match result {
                            Ok(Some(path)) => {
                                if let Some(path_str) = path.to_str() {
                                    self.settings.texture = Some(path_str.to_string());
                                    self.status_message = format!("已选择纹理: {}", path_str);
                                }
                            }
                            Ok(None) => {}
                            Err(e) => {
                                self.set_error(format!("纹理选择错误: {}", e));
                            }
                        }
                    }
                });
            });

            // 背景与环境设置
            ui.collapsing("背景与环境", |ui| {
                // 渐变背景
                let resp_grad_bg = ui.checkbox(&mut self.settings.enable_gradient_background, "启用渐变背景");
                Self::add_tooltip(resp_grad_bg, ctx, "使用渐变色作为场景背景");
                if self.settings.enable_gradient_background {
                    ui.horizontal(|ui| {
                        ui.label("顶部颜色:");
                        let top_color_rgb_vec = parse_vec3(&self.settings.gradient_top_color).unwrap_or_else(|_| nalgebra::Vector3::new(0.5, 0.7, 1.0));
                        let mut top_color_rgb = [top_color_rgb_vec.x, top_color_rgb_vec.y, top_color_rgb_vec.z];
                        if ui.color_edit_button_rgb(&mut top_color_rgb).changed() {
                            self.settings.gradient_top_color = format!("{},{},{}", top_color_rgb[0], top_color_rgb[1], top_color_rgb[2]);
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("底部颜色:");
                        let bottom_color_rgb_vec = parse_vec3(&self.settings.gradient_bottom_color).unwrap_or_else(|_| nalgebra::Vector3::new(0.1, 0.2, 0.4));
                        let mut bottom_color_rgb = [bottom_color_rgb_vec.x, bottom_color_rgb_vec.y, bottom_color_rgb_vec.z];
                        if ui.color_edit_button_rgb(&mut bottom_color_rgb).changed() {
                            self.settings.gradient_bottom_color = format!("{},{},{}", bottom_color_rgb[0], bottom_color_rgb[1], bottom_color_rgb[2]);
                        }
                    });
                }
                ui.separator();
                // 地面平面
                let resp_ground = ui.checkbox(&mut self.settings.enable_ground_plane, "启用地面平面");
                Self::add_tooltip(resp_ground, ctx, "在场景中添加一个无限延伸的地面");
                if self.settings.enable_ground_plane {
                    ui.horizontal(|ui| {
                        ui.label("地面颜色:");
                        let ground_color_rgb_vec = parse_vec3(&self.settings.ground_plane_color).unwrap_or_else(|_| nalgebra::Vector3::new(0.3, 0.5, 0.2));
                        let mut ground_color_rgb = [ground_color_rgb_vec.x, ground_color_rgb_vec.y, ground_color_rgb_vec.z];
                        if ui.color_edit_button_rgb(&mut ground_color_rgb).changed() {
                            self.settings.ground_plane_color = format!("{},{},{}", ground_color_rgb[0], ground_color_rgb[1], ground_color_rgb[2]);
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("地面高度 (Y):");
                        let mut height_value = self.settings.ground_plane_height;
                        let resp_height = ui.add(
                            egui::DragValue::new(&mut height_value)
                            .speed(0.1)
                            .range(-100.0..=-0.1) // 设置范围限制，最大值为0
                        );

                        // 如果值发生了变化，确保是负值或零
                        if resp_height.changed() {
                            self.settings.ground_plane_height = height_value.min(0.0);
                        }

                        Self::add_tooltip(resp_height, ctx, "地面平面在Y轴上的高度（世界坐标系），必须小于等于0");
                    });
                }
            });

            // 相机设置部分
            ui.collapsing("相机设置", |ui| {
                ui.horizontal(|ui| {
                    ui.label("相机位置 (x,y,z)：");
                    let resp = ui.text_edit_singleline(&mut self.settings.camera_from);
                    Self::add_tooltip(resp, ctx, "相机的位置坐标，格式为x,y,z");
                });

                ui.horizontal(|ui| {
                    ui.label("相机目标 (x,y,z)：");
                    let resp = ui.text_edit_singleline(&mut self.settings.camera_at);
                    Self::add_tooltip(resp, ctx, "相机看向的目标点坐标，格式为x,y,z");
                });

                ui.horizontal(|ui| {
                    ui.label("相机上方向 (x,y,z)：");
                    let resp = ui.text_edit_singleline(&mut self.settings.camera_up);
                    Self::add_tooltip(resp, ctx, "相机的上方向向量，格式为x,y,z");
                });

                ui.horizontal(|ui| {
                    ui.label("视场角 (度)：");
                    let resp = ui.add(egui::Slider::new(&mut self.settings.camera_fov, 10.0..=120.0));
                    Self::add_tooltip(resp, ctx, "相机视场角，值越大视野范围越广（鱼眼效果）");
                });
            });

            // 光照设置部分
            ui.collapsing("光照设置", |ui| {
                let resp = ui.checkbox(&mut self.settings.use_lighting, "启用光照")
                    .on_hover_text("总光照开关，关闭则仅使用环境光");
                Self::add_tooltip(resp, ctx, "启用或禁用方向光源");

                // 确保光源数组已初始化
                self.settings.ensure_light_arrays();

                ui.separator();

                // 环境光设置
                ui.horizontal(|ui| {
                    ui.label("环境光颜色:");
                    let ambient_color_vec = parse_vec3(&self.settings.ambient_color)
                        .unwrap_or_else(|_| nalgebra::Vector3::new(0.1, 0.1, 0.1));
                    let mut ambient_color_rgb = [ambient_color_vec.x, ambient_color_vec.y, ambient_color_vec.z];
                    let resp = ui.color_edit_button_rgb(&mut ambient_color_rgb);
                    if resp.changed() {
                        self.settings.ambient_color = format!("{},{},{}", 
                            ambient_color_rgb[0], ambient_color_rgb[1], ambient_color_rgb[2]);
                    }
                    Self::add_tooltip(resp, ctx, "环境光的颜色\n如果光照关闭，此颜色将作为基础色");
                });

                ui.horizontal(|ui| {
                    ui.label("环境光强度:");
                    let resp = ui.add(egui::Slider::new(&mut self.settings.ambient, 0.0..=1.0));
                    Self::add_tooltip(resp, ctx, "环境光的整体强度");
                });

                // 在此处添加光照预设选择器
                ui.horizontal(|ui| {
                    ui.label("光照预设:");
                    egui::ComboBox::from_id_salt("lighting_preset_combo")
                        .selected_text(match self.settings.lighting_preset {
                            LightingPreset::SingleDirectional => "单一方向光",
                            LightingPreset::ThreeDirectional => "三面方向光",
                            LightingPreset::MixedComplete => "混合光源",
                            LightingPreset::None => "无光源",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.settings.lighting_preset, LightingPreset::SingleDirectional, "单一方向光");
                            ui.selectable_value(&mut self.settings.lighting_preset, LightingPreset::ThreeDirectional, "三面方向光");
                            ui.selectable_value(&mut self.settings.lighting_preset, LightingPreset::MixedComplete, "混合光源");
                            ui.selectable_value(&mut self.settings.lighting_preset, LightingPreset::None, "无光源");
                        });

                    if ui.button("应用预设").clicked() {
                        self.settings.setup_light_sources();
                    }
                });

                if self.settings.use_lighting {
                    ui.separator();

                    // 方向光源设置
                    ui.collapsing("方向光源", |ui| {
                        for (i, light) in self.settings.directional_lights.iter_mut().enumerate() {
                            ui.group(|ui| {
                                ui.horizontal(|ui| {
                                    ui.checkbox(&mut light.enabled, format!("方向光 #{}", i + 1));

                                    if light.enabled {
                                        let resp = ui.add(egui::Slider::new(&mut light.intensity, 0.0..=2.0)
                                            .text("强度"));
                                        Self::add_tooltip(resp, ctx, "光源强度倍增因子");
                                    }
                                });

                                if light.enabled {
                                    ui.horizontal(|ui| {
                                        ui.label("方向 (x,y,z):");
                                        let resp = ui.text_edit_singleline(&mut light.direction);
                                        Self::add_tooltip(resp, ctx, "光线照射的方向，格式为x,y,z");
                                    });

                                    ui.horizontal(|ui| {
                                        ui.label("颜色:");
                                        let color_vec = parse_vec3(&light.color)
                                            .unwrap_or_else(|_| nalgebra::Vector3::new(1.0, 1.0, 1.0));
                                        let mut color_rgb = [color_vec.x, color_vec.y, color_vec.z];
                                        let resp = ui.color_edit_button_rgb(&mut color_rgb);
                                        if resp.changed() {
                                            light.color = format!("{},{},{}", 
                                                color_rgb[0], color_rgb[1], color_rgb[2]);
                                        }
                                    });
                                }
                            });
                        }
                    });

                    // 点光源设置
                    ui.collapsing("点光源", |ui| {
                        for (i, light) in self.settings.point_lights.iter_mut().enumerate() {
                            ui.group(|ui| {
                                ui.horizontal(|ui| {
                                    ui.checkbox(&mut light.enabled, format!("点光源 #{}", i + 1));

                                    if light.enabled {
                                        let resp = ui.add(egui::Slider::new(&mut light.intensity, 0.0..=5.0)
                                            .text("强度"));
                                        Self::add_tooltip(resp, ctx, "光源强度倍增因子");
                                    }
                                });

                                if light.enabled {
                                    ui.horizontal(|ui| {
                                        ui.label("位置 (x,y,z):");
                                        let resp = ui.text_edit_singleline(&mut light.position);
                                        Self::add_tooltip(resp, ctx, "光源位置，格式为x,y,z");
                                    });

                                    ui.horizontal(|ui| {
                                        ui.label("颜色:");
                                        let color_vec = parse_vec3(&light.color)
                                            .unwrap_or_else(|_| nalgebra::Vector3::new(1.0, 1.0, 1.0));
                                        let mut color_rgb = [color_vec.x, color_vec.y, color_vec.z];
                                        let resp = ui.color_edit_button_rgb(&mut color_rgb);
                                        if resp.changed() {
                                            light.color = format!("{},{},{}", 
                                                color_rgb[0], color_rgb[1], color_rgb[2]);
                                        }
                                    });

                                    // 衰减设置
                                    ui.group(|ui| {
                                        ui.label("光照衰减参数:");
                                        ui.horizontal(|ui| {
                                            ui.label("常数项:");
                                            ui.add(egui::DragValue::new(&mut light.constant_attenuation)
                                                .speed(0.05)
                                                .range(0.0..=10.0));
                                        });
                                        ui.horizontal(|ui| {
                                            ui.label("线性项:");
                                            ui.add(egui::DragValue::new(&mut light.linear_attenuation)
                                                .speed(0.01)
                                                .range(0.0..=1.0));
                                        });
                                        ui.horizontal(|ui| {
                                            ui.label("二次项:");
                                            ui.add(egui::DragValue::new(&mut light.quadratic_attenuation)
                                                .speed(0.001)
                                                .range(0.0..=0.5));
                                        });
                                        ui.label("提示: 1.0, 0.09, 0.032 为现实世界的典型值")
                                            .on_hover_text("常数项影响基础亮度\n线性项控制中距离衰减\n二次项控制远距离衰减");
                                    });
                                }
                            });
                        }
                    });
                }
            });

            // PBR材质设置部分
            if self.settings.use_pbr {
                ui.collapsing("PBR材质设置 (Physically Based Rendering)", |ui| {
                    ui.horizontal(|ui| {
                        ui.label("基础颜色 (Base Color):");
                        let base_color_vec = parse_vec3(&self.settings.base_color).unwrap_or_else(|_| nalgebra::Vector3::new(0.8, 0.8, 0.8));
                        let mut base_color_rgb = [base_color_vec.x, base_color_vec.y, base_color_vec.z];
                        let resp = ui.color_edit_button_rgb(&mut base_color_rgb);
                        if resp.changed() {
                            self.settings.base_color = format!("{},{},{}", base_color_rgb[0], base_color_rgb[1], base_color_rgb[2]);
                        }
                        Self::add_tooltip(resp, ctx, "材质的基础颜色 (Base Color)\n在PBR中代表材质的反射率或颜色");
                    });

                    ui.horizontal(|ui| {
                        ui.label("金属度 (Metallic)：");
                        let resp = ui.add(egui::Slider::new(&mut self.settings.metallic, 0.0..=1.0));
                        Self::add_tooltip(resp, ctx, "材质的金属特性 (Metallic)，0为非金属，1为纯金属\n影响材质如何反射光线和能量守恒");
                    });

                    ui.horizontal(|ui| {
                        ui.label("粗糙度 (Roughness)：");
                        let resp = ui.add(egui::Slider::new(&mut self.settings.roughness, 0.0..=1.0));
                        Self::add_tooltip(resp, ctx, "材质的粗糙程度 (Roughness)，0为完全光滑，1为完全粗糙\n影响高光的散射程度和微表面特性");
                    });

                    ui.horizontal(|ui| {
                        ui.label("环境光遮蔽 (Ambient Occlusion)：");
                        let resp = ui.add(egui::Slider::new(
                            &mut self.settings.ambient_occlusion,
                            0.0..=1.0,
                        ));
                        Self::add_tooltip(resp, ctx, "环境光遮蔽程度 (Ambient Occlusion)，0为完全遮蔽，1为无遮蔽\n模拟物体凹陷处接收较少环境光的效果");
                    });

                    ui.horizontal(|ui| {
                        ui.label("自发光颜色 (Emissive):");
                        let emissive_color_vec = parse_vec3(&self.settings.emissive).unwrap_or_else(|_| nalgebra::Vector3::new(0.0, 0.0, 0.0));
                        let mut emissive_color_rgb = [emissive_color_vec.x, emissive_color_vec.y, emissive_color_vec.z];
                        let resp = ui.color_edit_button_rgb(&mut emissive_color_rgb);
                        if resp.changed() {
                            self.settings.emissive = format!("{},{},{}", emissive_color_rgb[0], emissive_color_rgb[1], emissive_color_rgb[2]);
                        }
                        Self::add_tooltip(resp, ctx, "材质的自发光颜色 (Emissive)\n表示材质本身发出的光，不受光照影响");
                    });
                });
            }

            // Phong材质设置部分
            if self.settings.use_phong {
                ui.collapsing("Phong材质设置 (Blinn-Phong Shading)", |ui| {
                    ui.horizontal(|ui| {
                        ui.label("漫反射颜色 (Diffuse):");
                        let diffuse_color_vec = parse_vec3(&self.settings.diffuse_color).unwrap_or_else(|_| nalgebra::Vector3::new(0.8, 0.8, 0.8)); // Using a typical diffuse default
                        let mut diffuse_color_rgb = [diffuse_color_vec.x, diffuse_color_vec.y, diffuse_color_vec.z];
                        let resp = ui.color_edit_button_rgb(&mut diffuse_color_rgb);
                        if resp.changed() {
                            self.settings.diffuse_color = format!("{},{},{}", diffuse_color_rgb[0], diffuse_color_rgb[1], diffuse_color_rgb[2]);
                        }
                        Self::add_tooltip(resp, ctx, "材质的漫反射颜色 (Diffuse Color)\n决定物体表面向各个方向均匀散射的颜色");
                    });

                    ui.horizontal(|ui| {
                        ui.label("镜面反射强度 (Specular)：");
                        let resp = ui.add(egui::Slider::new(&mut self.settings.specular, 0.0..=1.0));
                        Self::add_tooltip(resp, ctx, "材质的镜面反射强度 (Specular Intensity)，0为无反射，1为最大反射\n控制高光的亮度");
                    });

                    ui.horizontal(|ui| {
                        ui.label("光泽度 (Shininess)：");
                        let resp = ui.add(egui::Slider::new(&mut self.settings.shininess, 1.0..=100.0));
                        Self::add_tooltip(resp, ctx, "材质的光泽度 (Shininess)，数值越大高光越小越集中\n也称为Phong指数，控制高光的锐利程度");
                    });

                    ui.horizontal(|ui| {
                        ui.label("自发光颜色 (Emissive):");
                        let emissive_color_vec = parse_vec3(&self.settings.emissive).unwrap_or_else(|_| nalgebra::Vector3::new(0.0, 0.0, 0.0));
                        let mut emissive_color_rgb = [emissive_color_vec.x, emissive_color_vec.y, emissive_color_vec.z];
                        let resp = ui.color_edit_button_rgb(&mut emissive_color_rgb);
                        if resp.changed() {
                            self.settings.emissive = format!("{},{},{}", emissive_color_rgb[0], emissive_color_rgb[1], emissive_color_rgb[2]);
                        }
                        Self::add_tooltip(resp, ctx, "材质的自发光颜色 (Emissive)\n表示材质本身发出的光，不受光照影响");
                    });
                });
            }

            // 动画设置部分
            ui.collapsing("动画设置", |ui| {
                ui.horizontal(|ui| {
                    ui.label("旋转圈数:");
                    let resp = ui.add(egui::DragValue::new(&mut self.settings.rotation_cycles)
                        .speed(0.1)
                        .range(0.1..=10.0));
                    Self::add_tooltip(resp, ctx, "动画完成的旋转圈数，影响生成的总帧数");
                });

                ui.horizontal(|ui| {
                    ui.label("视频生成及预渲染帧率 (FPS):");
                    let resp = ui.add(egui::DragValue::new(&mut self.settings.fps)
                        .speed(1)
                        .range(1..=60));
                    Self::add_tooltip(resp, ctx, "生成视频的每秒帧数");
                });

                let (_, seconds_per_rotation, frames_per_rotation) =
                crate::utils::render_utils::calculate_rotation_parameters(self.settings.rotation_speed, self.settings.fps);
                let total_frames = (frames_per_rotation as f32 * self.settings.rotation_cycles) as usize;
                    let total_seconds = (seconds_per_rotation * self.settings.rotation_cycles) as f32;

                ui.label(format!("估计总帧数: {} (视频长度: {:.1}秒)",
                                        total_frames, total_seconds));

                // 动画类型选择
                ui.horizontal(|ui| {
                    ui.label("动画类型:");
                    let current_animation_type = self.settings.animation_type.clone();
                    egui::ComboBox::from_id_salt("animation_type_combo")
                        .selected_text(match current_animation_type {
                            AnimationType::CameraOrbit => "相机轨道旋转",
                            AnimationType::ObjectLocalRotation => "物体局部旋转",
                            AnimationType::None => "无动画",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.settings.animation_type, AnimationType::CameraOrbit, "相机轨道旋转");
                            ui.selectable_value(&mut self.settings.animation_type, AnimationType::ObjectLocalRotation, "物体局部旋转");
                            ui.selectable_value(&mut self.settings.animation_type, AnimationType::None, "无动画");
                        });
                });

                // 旋转轴选择 (仅当动画类型不是 None 时显示)
                if self.settings.animation_type != AnimationType::None {
                    ui.horizontal(|ui| {
                        ui.label("旋转轴:");
                        let current_rotation_axis = self.settings.rotation_axis.clone();
                        egui::ComboBox::from_id_salt("rotation_axis_combo")
                            .selected_text(match current_rotation_axis {
                                RotationAxis::X => "X 轴",
                                RotationAxis::Y => "Y 轴",
                                RotationAxis::Z => "Z 轴",
                                RotationAxis::Custom => "自定义轴",
                            })
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut self.settings.rotation_axis, RotationAxis::X, "X 轴");
                                ui.selectable_value(&mut self.settings.rotation_axis, RotationAxis::Y, "Y 轴");
                                ui.selectable_value(&mut self.settings.rotation_axis, RotationAxis::Z, "Z 轴");
                                ui.selectable_value(&mut self.settings.rotation_axis, RotationAxis::Custom, "自定义轴");
                            });
                    });

                    if self.settings.rotation_axis == RotationAxis::Custom {
                        ui.horizontal(|ui| {
                            ui.label("自定义轴 (x,y,z):");
                            let resp = ui.text_edit_singleline(&mut self.settings.custom_rotation_axis);
                            Self::add_tooltip(resp, ctx, "输入自定义旋转轴，例如 1,0,0 或 0.707,0.707,0");
                        });
                    }
                }
                Self::add_tooltip(ui.label(""), ctx, "选择实时渲染和视频生成时的动画效果和旋转轴");

                // 简化预渲染模式复选框逻辑
                let pre_render_enabled = self.can_toggle_pre_render();
                let mut pre_render_value = self.pre_render_mode;

                let pre_render_resp = ui.add_enabled(
                    pre_render_enabled,
                    egui::Checkbox::new(&mut pre_render_value, "启用预渲染模式")
                );

                if pre_render_resp.changed() && pre_render_value != self.pre_render_mode {
                    super::animation::AnimationMethods::toggle_pre_render_mode(self);
                }
                Self::add_tooltip(pre_render_resp, ctx,
                        "启用后，首次开始实时渲染时会预先计算所有帧，\n然后以选定帧率无卡顿播放。\n要求更多内存，但播放更流畅。");

                ui.horizontal(|ui| {
                    ui.label("旋转速度 (实时渲染):");
                    let resp = ui.add(egui::Slider::new(&mut self.settings.rotation_speed, 0.1..=5.0));
                    Self::add_tooltip(resp, ctx, "实时渲染中的旋转速度倍率");
                });
            });

            // 按钮区域
            ui.add_space(20.0);

            // 恢复默认值与渲染按钮一行
            ui.horizontal(|ui| {
                // 恢复默认值按钮 - 使用固定宽度
                let reset_button = ui.add_sized(
                    [100.0, 40.0],  // 使用固定宽度
                    egui::Button::new(
                        RichText::new("恢复默认值")
                        .size(15.0)
                    )
                );

                if reset_button.clicked() {
                    self.reset_to_defaults();
                }

                Self::add_tooltip(reset_button, ctx, "重置所有渲染参数为默认值，保留文件路径设置");

                ui.add_space(10.0);

                // 渲染按钮
                let render_button = ui.add_sized(
                    [ui.available_width(), 40.0],
                                                 egui::Button::new(
                                                     RichText::new("开始渲染")
                                                     .size(18.0)
                                                     .strong()
                                                 )
                );

                if render_button.clicked() {
                    self.render(ctx);
                }

                Self::add_tooltip(render_button, ctx, "快捷键: Ctrl+R");
            });

            ui.add_space(10.0);            // 动画渲染和截图按钮一行
            ui.horizontal(|ui| {
                // 使用固定宽度代替计算的宽度
                let button_width = 150.0;  // 固定宽度

                // 动画渲染按钮 - 使用add_enabled和sized分开处理
                let realtime_button = ui.add_enabled(
                    self.can_render_animation(), // 使用 can_render_animation 检查是否可以渲染
                    egui::Button::new(
                        RichText::new(if self.is_realtime_rendering {
                            "停止动画渲染"
                        } else {
                            "开始动画渲染"
                        })
                        .size(15.0)
                    )
                    .min_size(Vec2::new(button_width, 40.0)) // 使用min_size设置固定大小
                );

                if realtime_button.clicked() {
                    // 如果当前在播放预渲染帧，点击时只是停止播放
                    if self.is_realtime_rendering && self.pre_render_mode {
                        self.is_realtime_rendering = false;
                        self.status_message = "已停止动画渲染".to_string();                    } 
                    // 否则切换实时渲染状态
                    else if !self.is_realtime_rendering {
                        // 使用CoreMethods中的开始动画渲染方法
                        if let Err(e) = self.start_animation_rendering() {
                            self.set_error(e);
                        }
                    } else {
                        // 使用CoreMethods中的停止动画渲染方法
                        self.stop_animation_rendering();
                    }
                }

                Self::add_tooltip(realtime_button, ctx, "启动连续动画渲染，实时显示旋转效果");

                ui.add_space(10.0);

                // 截图按钮
                let screenshot_button = ui.add_enabled(
                    self.rendered_image.is_some(),
                                                       egui::Button::new(RichText::new("截图").size(15.0))
                                                       .min_size(Vec2::new(ui.available_width(), 40.0))
                );

                if screenshot_button.clicked() {
                    match self.take_screenshot() {
                        Ok(path) => {
                            self.status_message = format!("截图已保存至 {}", path);
                        }
                        Err(e) => {
                            self.set_error(format!("截图失败: {}", e));
                        }
                    }
                }

                Self::add_tooltip(screenshot_button, ctx, "保存当前渲染结果为图片文件");
            });

            // 视频生成按钮独占一行
            ui.add_space(10.0);

            ui.horizontal(|ui| {
                let video_button_text = if self.is_generating_video {
                    let progress = self.video_progress.load(Ordering::SeqCst);

                    // 使用通用函数计算实际帧数
                    let (_, _, frames_per_rotation) =
                        crate::utils::render_utils::calculate_rotation_parameters(
                            self.settings.rotation_speed,
                            self.settings.fps
                        );
                    let total_frames = (frames_per_rotation as f32 * self.settings.rotation_cycles) as usize;

                    let percent = (progress as f32 / total_frames as f32 * 100.0).round();
                    format!("生成视频中... {}%", percent)
                } else if self.ffmpeg_available {
                    "生成视频".to_string()
                } else {
                    "生成视频 (需ffmpeg)".to_string()
                };

                let is_video_button_enabled = self.can_generate_video();

                // 计算按钮的可用宽度
                let available_w_for_buttons = ui.available_width();
                let spacing_x = ui.spacing().item_spacing.x;

                // 为"生成视频"按钮分配大约 60% 的空间，为"清空缓冲区"按钮分配大约 40%
                let video_button_width = (available_w_for_buttons - spacing_x) * 0.6;
                let clear_buffer_button_width = (available_w_for_buttons - spacing_x) * 0.4;

                // 视频生成按钮
                let video_button_response = ui.add_enabled(
                    is_video_button_enabled,
                    egui::Button::new(RichText::new(video_button_text).size(15.0))
                        .min_size(Vec2::new(video_button_width.max(80.0), 40.0))
                );

                if video_button_response.clicked() {
                    self.start_video_generation(ctx);
                }
                Self::add_tooltip(video_button_response, ctx,
                                  "在后台渲染多帧并生成MP4视频。\n需要系统安装ffmpeg。\n生成过程不会影响UI使用。");                // 清空缓冲区按钮
                // 使用can_clear_buffer函数检查是否有可清空的帧
                let is_clear_buffer_enabled = self.can_clear_buffer();

                let clear_buffer_text = RichText::new("清空缓冲区").size(15.0);
                let clear_buffer_response = ui.add_enabled(
                    is_clear_buffer_enabled,
                    egui::Button::new(clear_buffer_text)
                        .min_size(Vec2::new(clear_buffer_button_width.max(80.0), 40.0))
                );                if clear_buffer_response.clicked() {
                    // 使用CoreMethods实现
                    CoreMethods::clear_pre_rendered_frames(self);
                }
                Self::add_tooltip(clear_buffer_response, ctx,
                "清除已预渲染的动画帧，释放内存。\n请先停止动画渲染再清除缓冲区。");
            });

            // 渲染信息
            if let Some(time) = self.last_render_time {
                ui.separator();
                ui.label(format!("渲染耗时: {:.2?}", time));

                if let Some(model) = &self.model_data {
                    let triangle_count: usize =
                    model.meshes.iter().map(|m| m.indices.len() / 3).sum();
                    ui.label(format!("三角形数量: {}", triangle_count));
                }
            }
        });
    }
}
