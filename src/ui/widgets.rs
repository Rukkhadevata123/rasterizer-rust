use egui::{Color32, Context, RichText, Vec2};
use native_dialog::FileDialogBuilder;
use std::sync::atomic::Ordering;

use super::animation::AnimationMethods;
use super::app::RasterizerApp;
use super::render::RenderMethods;

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
                    let response = ui.text_edit_singleline(&mut self.args.obj);
                    Self::add_tooltip(response, ctx, "选择要渲染的3D模型文件（.obj格式）");
                    if ui.button("浏览").clicked() {
                        self.select_obj_file();
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("输出目录：");
                    let response = ui.text_edit_singleline(&mut self.args.output_dir);
                    Self::add_tooltip(response, ctx, "选择渲染结果保存的目录");
                    if ui.button("浏览").clicked() {
                        self.select_output_dir();
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("输出文件名：");
                    let response = ui.text_edit_singleline(&mut self.args.output);
                    Self::add_tooltip(response, ctx, "渲染结果的文件名（不含扩展名）");
                });
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("宽度：");
                    let response = ui.add(egui::DragValue::new(&mut self.args.width)
                        .speed(1)
                        .range(1..=4096));
                    Self::add_tooltip(response, ctx, "渲染图像的宽度（像素）");
                });

                ui.horizontal(|ui| {
                    ui.label("高度：");
                    let response = ui.add(egui::DragValue::new(&mut self.args.height)
                        .speed(1)
                        .range(1..=4096));
                    Self::add_tooltip(response, ctx, "渲染图像的高度（像素）");
                });
                let response = ui.checkbox(&mut self.args.save_depth, "保存深度图");
                Self::add_tooltip(response, ctx, "同时保存深度图（深度信息可视化）");
            });

            // 渲染属性设置
            ui.collapsing("渲染属性设置", |ui| {
                ui.horizontal(|ui| {
                    ui.label("投影类型：");
                    let resp1 = ui.radio_value(
                        &mut self.args.projection,
                        "perspective".to_string(),
                        "透视",
                    );
                    Self::add_tooltip(resp1, ctx, "使用透视投影（符合人眼观察方式）");

                    let resp2 = ui.radio_value(
                        &mut self.args.projection,
                        "orthographic".to_string(),
                        "正交",
                    );
                    Self::add_tooltip(resp2, ctx, "使用正交投影（无透视变形）");
                });
                ui.separator();
                let resp1 = ui.checkbox(&mut self.args.use_zbuffer, "深度缓冲");
                Self::add_tooltip(resp1, ctx, "启用Z缓冲进行深度测试，处理物体遮挡关系");

                let resp2 = ui.checkbox(&mut self.args.use_lighting, "启用光照");
                Self::add_tooltip(resp2, ctx, "启用光照计算，产生明暗变化");

                // 将"启用纹理"和"使用面颜色"改为互斥的单选项
                ui.horizontal(|ui| {
                    ui.label("表面颜色：");
                    
                    // 启用纹理选项
                    let texture_response = ui.radio_value(&mut self.args.use_texture, true, "使用纹理");
                    if texture_response.clicked() && self.args.use_texture {
                        // 如果选择了使用纹理，关闭面颜色
                        self.args.colorize = false;
                    }
                    Self::add_tooltip(texture_response, ctx, 
                        "使用模型的纹理贴图（如果有）\n优先级最高，会覆盖面颜色设置");
                    
                    // 使用面颜色选项
                    let face_color_response = ui.radio_value(&mut self.args.colorize, true, "使用面颜色");
                    if face_color_response.clicked() && self.args.colorize {
                        // 如果选择了使用面颜色，关闭纹理
                        self.args.use_texture = false;
                    }
                    Self::add_tooltip(face_color_response, ctx, 
                        "为每个面分配随机颜色\n仅在没有纹理或纹理被禁用时生效");
                    
                    // 使用材质颜色选项 (实际上是关闭两者)
                    let material_color_response = ui.radio(
                        !self.args.use_texture && !self.args.colorize, 
                        "使用材质颜色"
                    );
                    if material_color_response.clicked() {
                        self.args.use_texture = false;
                        self.args.colorize = false;
                    }
                    Self::add_tooltip(material_color_response, ctx, 
                        "使用材质的基本颜色（如.mtl文件中定义）\n在没有纹理且不使用面颜色时生效");
                });

                // 着色模型选择（Phong/PBR，已经是互斥的）
                ui.horizontal(|ui| {
                    ui.label("着色模型：");
                    // Phong 着色选项（逐像素着色，在 Blinn-Phong 光照模型下）
                    let phong_response = ui.radio_value(&mut self.args.use_phong, true, "Phong着色");
                    if phong_response.clicked() && self.args.use_phong {
                        // 如果选择了 Phong，关闭 PBR
                        self.args.use_pbr = false;
                    }
                    Self::add_tooltip(phong_response, ctx,
                        "使用 Phong 着色（逐像素着色）和 Blinn-Phong 光照模型\n提供高质量的光照效果，适合大多数场景");

                    // PBR 渲染选项
                    let pbr_response = ui.radio_value(&mut self.args.use_pbr, true, "PBR渲染");
                    if pbr_response.clicked() && self.args.use_pbr {
                        // 如果选择了 PBR，关闭 Phong
                        self.args.use_phong = false;
                    }
                    Self::add_tooltip(pbr_response, ctx,
                        "使用基于物理的渲染（PBR）\n提供更真实的材质效果，但需要更多的参数调整");
                });

                let resp7 = ui.checkbox(&mut self.args.use_gamma, "Gamma校正");
                Self::add_tooltip(resp7, ctx, "应用伽马校正，使亮度显示更准确");

                let resp8 = ui.checkbox(&mut self.args.backface_culling, "背面剔除");
                Self::add_tooltip(resp8, ctx, "剔除背向相机的三角形面，提高渲染效率");

                let resp9 = ui.checkbox(&mut self.args.wireframe, "线框模式");
                Self::add_tooltip(resp9, ctx, "仅渲染三角形边缘，显示为线框");

                ui.separator();
                let resp10 = ui.checkbox(&mut self.args.use_multithreading, "启用多线程渲染");
                Self::add_tooltip(resp10, ctx, "使用多线程加速渲染，提高性能");

                ui.horizontal(|ui| {
                    let resp = ui.checkbox(&mut self.args.cull_small_triangles, "剔除小三角形");
                    Self::add_tooltip(resp, ctx, "忽略投影后面积很小的三角形，提高性能");

                    if self.args.cull_small_triangles {
                        let resp = ui.add(
                            egui::DragValue::new(&mut self.args.min_triangle_area)
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
                    let mut texture_path_str = self.args.texture.clone().unwrap_or_default();
                    let resp = ui.text_edit_singleline(&mut texture_path_str);
                    Self::add_tooltip(resp.clone(), ctx, "选择自定义纹理，将覆盖MTL中的定义");

                    if resp.changed() {
                        if texture_path_str.is_empty() {
                            self.args.texture = None;
                        } else {
                            self.args.texture = Some(texture_path_str);
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
                                    self.args.texture = Some(path_str.to_string());
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

            // 相机设置部分
            ui.collapsing("相机设置", |ui| {
                ui.horizontal(|ui| {
                    ui.label("相机位置 (x,y,z)：");
                    let resp = ui.text_edit_singleline(&mut self.args.camera_from);
                    Self::add_tooltip(resp, ctx, "相机的位置坐标，格式为x,y,z");
                });

                ui.horizontal(|ui| {
                    ui.label("相机目标 (x,y,z)：");
                    let resp = ui.text_edit_singleline(&mut self.args.camera_at);
                    Self::add_tooltip(resp, ctx, "相机看向的目标点坐标，格式为x,y,z");
                });

                ui.horizontal(|ui| {
                    ui.label("相机上方向 (x,y,z)：");
                    let resp = ui.text_edit_singleline(&mut self.args.camera_up);
                    Self::add_tooltip(resp, ctx, "相机的上方向向量，格式为x,y,z");
                });

                ui.horizontal(|ui| {
                    ui.label("视场角 (度)：");
                    let resp = ui.add(egui::Slider::new(&mut self.args.camera_fov, 10.0..=120.0));
                    Self::add_tooltip(resp, ctx, "相机视场角，值越大视野范围越广（鱼眼效果）");
                });
            });

            // 光照设置部分
            ui.collapsing("光照设置", |ui| {
                let _resp = ui.checkbox(&mut self.args.use_lighting, "启用光照")
                    .on_hover_text("总光照开关，关闭则仅使用下方设置的环境光颜色/强度");

                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("光源类型：");
                    let resp1 = ui.radio_value(&mut self.args.light_type, "directional".to_string(), "定向光");
                    Self::add_tooltip(resp1, ctx, "定向光：来自无限远处的平行光（如太阳光）");

                    let resp2 = ui.radio_value(&mut self.args.light_type, "point".to_string(), "点光源");
                    Self::add_tooltip(resp2, ctx, "点光源：从一个点向四周发射的光（如灯泡）");
                });

                if self.args.light_type == "directional" {
                    ui.horizontal(|ui| {
                        ui.label("光源方向 (x,y,z)：");
                        let resp = ui.text_edit_singleline(&mut self.args.light_dir);
                        Self::add_tooltip(resp, ctx, "光线照射的方向，格式为x,y,z");
                    });
                } else if self.args.light_type == "point" {
                    ui.horizontal(|ui| {
                        ui.label("光源位置 (x,y,z)：");
                        let resp = ui.text_edit_singleline(&mut self.args.light_pos);
                        Self::add_tooltip(resp, ctx, "点光源的位置坐标，格式为x,y,z");
                    });
                    ui.horizontal(|ui| {
                        ui.label("衰减 (c,l,q)：");
                        let resp = ui.text_edit_singleline(&mut self.args.light_atten);
                        Self::add_tooltip(resp, ctx, "点光源的衰减参数，格式为常数项,线性项,二次项");
                    });
                }
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("环境光颜色 (r,g,b)：");
                    let resp = ui.text_edit_singleline(&mut self.args.ambient_color);
                    Self::add_tooltip(resp, ctx,
                        "环境光的颜色，格式为r,g,b\n如果光照关闭，或场景中没有其他光源，此颜色将作为基础色");
                });

                ui.horizontal(|ui| {
                    ui.label("环境光强度 (全局)：");
                    let resp = ui.add(egui::Slider::new(&mut self.args.ambient, 0.0..=1.0));
                    Self::add_tooltip(resp, ctx, "作为环境光颜色的倍增因子");
                });

                ui.horizontal(|ui| {
                    ui.label("漫反射强度：");
                    let resp = ui.add(egui::Slider::new(&mut self.args.diffuse, 0.0..=2.0));
                    Self::add_tooltip(resp, ctx, "物体表面漫反射的强度，影响光照明暗程度");
                });
            });

            // PBR材质设置部分
            if self.args.use_pbr {
                ui.collapsing("PBR材质设置 (Physically Based Rendering)", |ui| {
                    ui.horizontal(|ui| {
                        ui.label("基础颜色 (Base Color) [r,g,b]：");
                        let resp = ui.text_edit_singleline(&mut self.args.base_color);
                        Self::add_tooltip(resp, ctx, "材质的基础颜色 (Base Color)，格式为r,g,b\n在PBR中代表材质的反射率或颜色");
                    });

                    ui.horizontal(|ui| {
                        ui.label("金属度 (Metallic)：");
                        let resp = ui.add(egui::Slider::new(&mut self.args.metallic, 0.0..=1.0));
                        Self::add_tooltip(resp, ctx, "材质的金属特性 (Metallic)，0为非金属，1为纯金属\n影响材质如何反射光线和能量守恒");
                    });

                    ui.horizontal(|ui| {
                        ui.label("粗糙度 (Roughness)：");
                        let resp = ui.add(egui::Slider::new(&mut self.args.roughness, 0.0..=1.0));
                        Self::add_tooltip(resp, ctx, "材质的粗糙程度 (Roughness)，0为完全光滑，1为完全粗糙\n影响高光的散射程度和微表面特性");
                    });

                    ui.horizontal(|ui| {
                        ui.label("环境光遮蔽 (Ambient Occlusion)：");
                        let resp = ui.add(egui::Slider::new(
                            &mut self.args.ambient_occlusion,
                            0.0..=1.0,
                        ));
                        Self::add_tooltip(resp, ctx, "环境光遮蔽程度 (Ambient Occlusion)，0为完全遮蔽，1为无遮蔽\n模拟物体凹陷处接收较少环境光的效果");
                    });

                    ui.horizontal(|ui| {
                        ui.label("自发光颜色 (Emissive) [r,g,b]：");
                        let resp = ui.text_edit_singleline(&mut self.args.emissive);
                        Self::add_tooltip(resp, ctx, "材质的自发光颜色 (Emissive)，格式为r,g,b\n表示材质本身发出的光，不受光照影响");
                    });
                });
            }

            // Phong材质设置部分
            if self.args.use_phong {
                ui.collapsing("Phong材质设置 (Blinn-Phong Shading)", |ui| {
                    ui.horizontal(|ui| {
                        ui.label("漫反射颜色 (Diffuse) [r,g,b]：");
                        let resp = ui.text_edit_singleline(&mut self.args.diffuse_color);
                        Self::add_tooltip(resp, ctx, "材质的漫反射颜色 (Diffuse Color)，格式为r,g,b\n决定物体表面向各个方向均匀散射的颜色");
                    });

                    ui.horizontal(|ui| {
                        ui.label("镜面反射强度 (Specular)：");
                        let resp = ui.add(egui::Slider::new(&mut self.args.specular, 0.0..=1.0));
                        Self::add_tooltip(resp, ctx, "材质的镜面反射强度 (Specular Intensity)，0为无反射，1为最大反射\n控制高光的亮度");
                    });

                    ui.horizontal(|ui| {
                        ui.label("光泽度 (Shininess)：");
                        let resp = ui.add(egui::Slider::new(&mut self.args.shininess, 1.0..=100.0));
                        Self::add_tooltip(resp, ctx, "材质的光泽度 (Shininess)，数值越大高光越小越集中\n也称为Phong指数，控制高光的锐利程度");
                    });

                    ui.horizontal(|ui| {
                        ui.label("自发光颜色 (Emissive) [r,g,b]：");
                        let resp = ui.text_edit_singleline(&mut self.args.emissive);
                        Self::add_tooltip(resp, ctx, "材质的自发光颜色 (Emissive)，格式为r,g,b\n表示材质本身发出的光，不受光照影响");
                    });
                });
            }

            // 动画设置部分
            ui.collapsing("动画设置", |ui| {
                ui.horizontal(|ui| {
                    ui.label("总帧数：");
                    let resp = ui.add(egui::DragValue::new(&mut self.total_frames)
                        .speed(1)
                        .range(10..=1000));
                    Self::add_tooltip(resp, ctx, "生成视频的总帧数");
                });

                ui.horizontal(|ui| {
                    ui.label("帧率 (FPS)：");
                    let resp = ui.add(egui::DragValue::new(&mut self.fps)
                        .speed(1)
                        .range(1..=60));
                    Self::add_tooltip(resp, ctx, "生成视频的每秒帧数");
                });

                ui.horizontal(|ui| {
                    ui.label("旋转速度：");
                    let resp = ui.add(egui::Slider::new(&mut self.rotation_speed, 0.1..=5.0));
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

                ui.add_space(10.0);

                // 实时渲染和截图按钮一行
                ui.horizontal(|ui| {
                    // 使用固定宽度代替计算的宽度
                    let button_width = 150.0;  // 固定宽度

                    // 实时渲染按钮
                    let realtime_button = ui.add_sized(
                        [button_width, 40.0],
                        egui::Button::new(
                            RichText::new(if self.is_realtime_rendering {
                                "停止实时渲染"
                            } else {
                                "开始实时渲染"
                            })
                            .size(15.0)
                        )
                    );

                    if realtime_button.clicked() {
                        self.is_realtime_rendering = !self.is_realtime_rendering;
                        if self.is_realtime_rendering {
                            self.last_frame_time = None; // 重置时间计时
                            self.current_fps = 0.0;      // 重置帧率计数器
                            self.fps_history.clear();    // 清空帧率历史记录
                            self.avg_fps = 0.0;          // 重置平均帧率
                            self.status_message = "开始实时渲染...".to_string();
                        } else {
                            self.status_message = "已停止实时渲染".to_string();
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

                let video_button_text = if self.is_generating_video {
                    let progress = self.video_progress.load(Ordering::SeqCst);
                    let percent = (progress as f32 / self.total_frames as f32 * 100.0).round();
                    format!("生成视频中... {}%", percent)
                } else if self.ffmpeg_available {
                    "生成视频".to_string()
                } else {
                    "生成视频 (需安装ffmpeg)".to_string()
                };

                let is_video_enabled = !self.is_realtime_rendering && !self.is_generating_video;

                // 视频生成按钮
                let video_button = ui.add_enabled(
                    is_video_enabled,
                    egui::Button::new(RichText::new(video_button_text).size(15.0))
                        .min_size(Vec2::new(ui.available_width(), 40.0))
                );

                if video_button.clicked() {
                    self.start_video_generation(ctx);
                }

            Self::add_tooltip(video_button, ctx,
                "在后台渲染多帧并生成MP4视频。\n需要系统安装ffmpeg。\n生成过程不会影响UI使用。");

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
