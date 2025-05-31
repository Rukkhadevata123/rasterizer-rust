use crate::core::renderer::Renderer;
use crate::io::render_settings::RenderSettings;
use crate::material_system::materials::ModelData;
use crate::scene::scene_utils::Scene;
use egui::{Color32, ColorImage, RichText, Vec2};
use nalgebra::Vector3;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

// 导入其他UI模块
use super::animation::AnimationMethods;
use super::core::CoreMethods;
use super::widgets::WidgetMethods;

/// GUI应用状态 - 清晰分离TOML配置和GUI专用参数
pub struct RasterizerApp {
    // ===== TOML可配置参数 - 统一存储在settings中 =====
    /// 所有TOML可配置的渲染参数
    pub settings: RenderSettings,

    // ===== GUI专用向量字段 - 从settings字符串同步 =====
    /// GUI中物体位置控制的向量表示（与settings.object_position同步）
    pub object_position_vec: Vector3<f32>,
    /// GUI中物体旋转控制的向量表示（与settings.object_rotation同步，弧度制）
    pub object_rotation_vec: Vector3<f32>,
    /// GUI中物体缩放控制的向量表示（与settings.object_scale_xyz同步）
    pub object_scale_vec: Vector3<f32>,

    // ===== 渲染运行时状态 - 不可配置 =====
    /// 渲染器实例
    pub renderer: Renderer,
    /// 当前加载的场景
    pub scene: Option<Scene>,
    /// 当前加载的模型数据
    pub model_data: Option<ModelData>,

    // ===== GUI界面状态 - 不可配置 =====
    /// 渲染结果纹理句柄
    pub rendered_image: Option<egui::TextureHandle>,
    /// 上次渲染耗时
    pub last_render_time: Option<std::time::Duration>,
    /// 状态消息显示
    pub status_message: String,
    /// 是否显示错误对话框
    pub show_error_dialog: bool,
    /// 错误消息内容
    pub error_message: String,

    // ===== 实时渲染状态 - 不可配置 =====
    /// 当前实时帧率
    pub current_fps: f32,
    /// 帧率历史记录，用于平滑显示
    pub fps_history: Vec<f32>,
    /// 平均帧率
    pub avg_fps: f32,
    /// 是否正在实时渲染
    pub is_realtime_rendering: bool,
    /// 上一帧的时间戳
    pub last_frame_time: Option<std::time::Instant>,

    // ===== 预渲染状态 - 不可配置 =====
    /// 是否启用预渲染模式
    pub pre_render_mode: bool,
    /// 是否正在预渲染
    pub is_pre_rendering: bool,
    /// 预渲染的帧集合
    pub pre_rendered_frames: Arc<Mutex<Vec<ColorImage>>>,
    /// 当前显示的帧索引
    pub current_frame_index: usize,
    /// 预渲染进度
    pub pre_render_progress: Arc<AtomicUsize>,
    /// 全局动画计时器，用于跟踪动画总时长
    pub animation_time: f32,
    /// 预渲染一个完整周期所需的总帧数
    pub total_frames_for_pre_render_cycle: usize,

    // ===== 视频生成状态 - 不可配置 =====
    /// 是否正在生成视频
    pub is_generating_video: bool,
    /// 视频生成线程句柄
    pub video_generation_thread: Option<std::thread::JoinHandle<(bool, String)>>,
    /// 视频生成进度
    pub video_progress: Arc<AtomicUsize>,

    // ===== 相机交互设置 - 可考虑加入TOML配置 =====
    /// 平移敏感度
    pub camera_pan_sensitivity: f32,
    /// 轨道旋转敏感度
    pub camera_orbit_sensitivity: f32,
    /// 推拉缩放敏感度
    pub camera_dolly_sensitivity: f32,

    // ===== 相机交互状态 - 不可配置 =====
    /// 相机交互状态
    pub interface_interaction: InterfaceInteraction,

    // ===== 系统状态 - 不可配置 =====
    /// ffmpeg可用性检查结果
    pub ffmpeg_available: bool,
}

/// 相机交互状态
#[derive(Default)]
pub struct InterfaceInteraction {
    pub camera_is_dragging: bool,
    pub camera_is_orbiting: bool,
    pub last_mouse_pos: Option<egui::Pos2>,
    pub anything_changed: bool, // 标记相机是否发生变化，需要重新渲染
}

impl RasterizerApp {
    /// 创建新的GUI应用实例
    pub fn new(settings: RenderSettings, cc: &eframe::CreationContext<'_>) -> Self {
        // 配置字体，添加中文支持
        let mut fonts = egui::FontDefinitions::default();

        fonts.font_data.insert(
            "chinese_font".to_owned(),
            egui::FontData::from_static(include_bytes!(
                "../../assets/Noto_Sans_SC/static/NotoSansSC-Regular.ttf"
            ))
            .into(),
        );

        for (_text_style, font_ids) in fonts.families.iter_mut() {
            font_ids.push("chinese_font".to_owned());
        }

        cc.egui_ctx.set_fonts(fonts);

        // 从settings字符串初始化GUI专用向量字段
        let object_position_vec =
            if let Ok(pos) = crate::io::render_settings::parse_vec3(&settings.object_position) {
                pos
            } else {
                nalgebra::Vector3::new(0.0, 0.0, 0.0)
            };

        let object_rotation_vec =
            if let Ok(rot) = crate::io::render_settings::parse_vec3(&settings.object_rotation) {
                nalgebra::Vector3::new(rot.x.to_radians(), rot.y.to_radians(), rot.z.to_radians())
            } else {
                nalgebra::Vector3::new(0.0, 0.0, 0.0)
            };

        let object_scale_vec =
            if let Ok(scale) = crate::io::render_settings::parse_vec3(&settings.object_scale_xyz) {
                scale
            } else {
                nalgebra::Vector3::new(1.0, 1.0, 1.0)
            };

        // 创建渲染器
        let renderer = Renderer::new(settings.width, settings.height);

        // 检查ffmpeg是否可用
        let ffmpeg_available = Self::check_ffmpeg_available();

        Self {
            // ===== TOML可配置参数 =====
            settings,

            // ===== GUI专用向量字段 =====
            object_position_vec,
            object_rotation_vec,
            object_scale_vec,

            // ===== 渲染运行时状态 =====
            renderer,
            scene: None,
            model_data: None,

            // ===== GUI界面状态 =====
            rendered_image: None,
            last_render_time: None,
            status_message: String::new(),
            show_error_dialog: false,
            error_message: String::new(),

            // ===== 实时渲染状态 =====
            current_fps: 0.0,
            fps_history: Vec::new(),
            avg_fps: 0.0,
            is_realtime_rendering: false,
            last_frame_time: None,

            // ===== 预渲染状态 =====
            pre_render_mode: false,
            is_pre_rendering: false,
            pre_rendered_frames: Arc::new(Mutex::new(Vec::new())),
            current_frame_index: 0,
            pre_render_progress: Arc::new(AtomicUsize::new(0)),
            animation_time: 0.0,
            total_frames_for_pre_render_cycle: 0,

            // ===== 视频生成状态 =====
            is_generating_video: false,
            video_generation_thread: None,
            video_progress: Arc::new(AtomicUsize::new(0)),

            // ===== 相机交互设置 =====
            camera_pan_sensitivity: 1.0,
            camera_orbit_sensitivity: 1.0,
            camera_dolly_sensitivity: 1.0,

            // ===== 相机交互状态 =====
            interface_interaction: InterfaceInteraction::default(),

            // ===== 系统状态 =====
            ffmpeg_available,
        }
    }

    /// 检查ffmpeg是否可用
    fn check_ffmpeg_available() -> bool {
        std::process::Command::new("ffmpeg")
            .arg("-version")
            .output()
            .is_ok()
    }

    /// 设置错误信息并显示错误对话框
    pub fn set_error(&mut self, message: String) {
        CoreMethods::set_error(self, message.clone());
        self.error_message = message;
        self.show_error_dialog = true;
    }

    /// 简化相机交互 - 直接更新settings
    fn handle_camera_interaction(&mut self, image_response: &egui::Response, ctx: &egui::Context) {
        if let Some(scene) = &mut self.scene {
            let mut camera_changed = false;
            let mut need_clear_ground_cache = false;

            let screen_size = egui::Vec2::new(
                self.renderer.frame_buffer.width as f32,
                self.renderer.frame_buffer.height as f32,
            );

            // 处理鼠标拖拽
            if image_response.dragged() {
                if let Some(last_pos) = self.interface_interaction.last_mouse_pos {
                    let current_pos = image_response.interact_pointer_pos().unwrap_or_default();
                    let delta = current_pos - last_pos;

                    // 设置最小移动阈值，避免微小抖动触发重新渲染
                    if delta.length() < 1.0 {
                        return;
                    }

                    let is_shift_pressed = ctx.input(|i| i.modifiers.shift);

                    if is_shift_pressed && !self.interface_interaction.camera_is_orbiting {
                        self.interface_interaction.camera_is_orbiting = true;
                        self.interface_interaction.camera_is_dragging = false;
                    } else if !is_shift_pressed && !self.interface_interaction.camera_is_dragging {
                        self.interface_interaction.camera_is_dragging = true;
                        self.interface_interaction.camera_is_orbiting = false;
                    }

                    if self.interface_interaction.camera_is_orbiting && is_shift_pressed {
                        need_clear_ground_cache = scene
                            .active_camera
                            .orbit_from_screen_delta(delta, self.camera_orbit_sensitivity);
                        camera_changed = true;
                    } else if self.interface_interaction.camera_is_dragging && !is_shift_pressed {
                        need_clear_ground_cache = scene.active_camera.pan_from_screen_delta(
                            delta,
                            screen_size,
                            self.camera_pan_sensitivity,
                        );
                        camera_changed = true;
                    }
                }

                self.interface_interaction.last_mouse_pos = image_response.interact_pointer_pos();
            } else {
                self.interface_interaction.camera_is_dragging = false;
                self.interface_interaction.camera_is_orbiting = false;
                self.interface_interaction.last_mouse_pos = None;
            }

            // 处理鼠标滚轮缩放
            if image_response.hovered() {
                let scroll_delta = ctx.input(|i| i.smooth_scroll_delta.y);
                if scroll_delta.abs() > 0.1 {
                    let zoom_delta = scroll_delta * 0.01;
                    need_clear_ground_cache = scene
                        .active_camera
                        .dolly_from_scroll(zoom_delta, self.camera_dolly_sensitivity);
                    camera_changed = true;
                }
            }

            // 处理快捷键
            ctx.input(|i| {
                if i.key_pressed(egui::Key::R) {
                    need_clear_ground_cache = scene.active_camera.reset_to_default_view();
                    camera_changed = true;
                }

                if i.key_pressed(egui::Key::F) {
                    let object_center = nalgebra::Point3::new(0.0, 0.0, 0.0);
                    let object_radius = 2.0;
                    need_clear_ground_cache = scene
                        .active_camera
                        .focus_on_object(object_center, object_radius);
                    camera_changed = true;
                }
            });

            // 如果相机发生变化，直接更新settings并标记
            if camera_changed {
                // 如果相机变化，清除地面缓存（但保留背景缓存）
                if need_clear_ground_cache {
                    // 只清除地面缓存，背景缓存保留
                    self.renderer.frame_buffer.ground_cache = None;
                }

                // 直接更新settings字符串
                let pos = scene.active_camera.position();
                let target = scene.active_camera.params.target;
                let up = scene.active_camera.params.up;

                self.settings.camera_from = format!("{},{},{}", pos.x, pos.y, pos.z);
                self.settings.camera_at = format!("{},{},{}", target.x, target.y, target.z);
                self.settings.camera_up = format!("{},{},{}", up.x, up.y, up.z);

                // 统一标记
                self.interface_interaction.anything_changed = true;

                // 在非实时模式下请求重绘
                if !self.is_realtime_rendering {
                    ctx.request_repaint();
                }
            }
        }
    }

    /// 统一的资源清理方法
    fn cleanup_resources(&mut self) {
        CoreMethods::cleanup_resources(self);
    }
}

impl eframe::App for RasterizerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // 显示错误对话框（如果有）
        self.show_error_dialog_ui(ctx);

        // 检查快捷键
        if ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::R)) {
            CoreMethods::render(self, ctx);
        }

        // 执行实时渲染循环
        if self.is_realtime_rendering {
            self.perform_realtime_rendering(ctx);
        }

        // 检查视频生成进度
        if self.is_generating_video {
            if let Some(handle) = &self.video_generation_thread {
                if handle.is_finished() {
                    let result = self
                        .video_generation_thread
                        .take()
                        .unwrap()
                        .join()
                        .unwrap_or_else(|_| (false, "线程崩溃".to_string()));

                    self.is_generating_video = false;

                    if result.0 {
                        self.status_message = format!("视频生成成功: {}", result.1);
                    } else {
                        self.set_error(format!("视频生成失败: {}", result.1));
                    }

                    self.video_progress.store(0, Ordering::SeqCst);
                } else {
                    let progress = self.video_progress.load(Ordering::SeqCst);

                    let (_, _, frames_per_rotation) =
                        crate::utils::render_utils::calculate_rotation_parameters(
                            self.settings.rotation_speed,
                            self.settings.fps,
                        );
                    let total_frames =
                        (frames_per_rotation as f32 * self.settings.rotation_cycles) as usize;

                    let percent = (progress as f32 / total_frames as f32 * 100.0).round();

                    self.status_message = format!(
                        "生成视频中... ({}/{}，{:.0}%)",
                        progress, total_frames, percent
                    );

                    ctx.request_repaint_after(std::time::Duration::from_millis(500));
                }
            }
        }

        // UI布局
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("光栅化渲染器");
                ui.separator();
                ui.label(&self.status_message);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if self.is_realtime_rendering {
                        let (fps_text, fps_color) = CoreMethods::get_fps_display(self);
                        ui.label(RichText::new(&fps_text).color(fps_color));
                        ui.separator();
                    }
                    ui.label("Ctrl+R: 快速渲染");
                });
            });
        });

        egui::SidePanel::left("left_panel")
            .min_width(350.0)
            .resizable(false)
            .show(ctx, |ui| {
                self.draw_side_panel(ctx, ui);
            });

        // 中央面板 - 显示渲染结果和处理相机交互
        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(texture) = &self.rendered_image {
                let available_size = ui.available_size();
                let square_size = available_size.x.min(available_size.y) * 0.95;

                let image_aspect = self.renderer.frame_buffer.width as f32
                    / self.renderer.frame_buffer.height as f32;

                let (width, height) = if image_aspect > 1.0 {
                    (square_size, square_size / image_aspect)
                } else {
                    (square_size * image_aspect, square_size)
                };

                let image_response = ui
                    .horizontal(|ui| {
                        ui.add(
                            egui::Image::new(texture)
                                .fit_to_exact_size(Vec2::new(width, height))
                                .sense(egui::Sense::click_and_drag()),
                        )
                    })
                    .inner;

                self.handle_camera_interaction(&image_response, ctx);

                // 显示交互提示
                let overlay_rect = egui::Rect::from_min_size(
                    ui.max_rect().right_bottom() - egui::Vec2::new(220.0, 20.0),
                    egui::Vec2::new(220.0, 20.0),
                );

                ui.allocate_new_ui(
                    egui::UiBuilder::new()
                        .max_rect(overlay_rect)
                        .layout(egui::Layout::right_to_left(egui::Align::BOTTOM)),
                    |ui| {
                        ui.group(|ui| {
                            ui.label(RichText::new("🖱️ 相机交互").size(14.0).strong());
                            ui.separator();
                            ui.small("• 拖拽 - 平移相机");
                            ui.small("• Shift+拖拽 - 轨道旋转");
                            ui.small("• 滚轮 - 推拉缩放");
                            ui.small("• R键 - 重置视角");
                            ui.small("• F键 - 聚焦物体");
                            ui.separator();
                            ui.small(format!("平移敏感度: {:.1}x", self.camera_pan_sensitivity));
                            ui.small(format!("旋转敏感度: {:.1}x", self.camera_orbit_sensitivity));
                            ui.small(format!("缩放敏感度: {:.1}x", self.camera_dolly_sensitivity));
                            ui.separator();
                            ui.small(RichText::new("✅ 交互已启用").color(Color32::GREEN));
                        });
                    },
                );
            } else {
                ui.vertical_centered(|ui| {
                    ui.add_space(100.0);
                    ui.label(RichText::new("无渲染结果").size(24.0).color(Color32::GRAY));
                    ui.label(RichText::new("点击「开始渲染」按钮或按Ctrl+R").color(Color32::GRAY));
                    ui.add_space(20.0);
                    ui.label(
                        RichText::new("💡 加载模型后可在此区域进行相机交互")
                            .color(Color32::from_rgb(100, 150, 255)),
                    );
                });
            }
        });

        // 统一处理所有变化引起的重新渲染
        CoreMethods::render_if_anything_changed(self, ctx);

        // 在每帧更新结束时清理不需要的资源
        self.cleanup_resources();
    }
}

/// 启动GUI应用
pub fn start_gui(settings: RenderSettings) -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_min_inner_size([800.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Rust 光栅化渲染器",
        options,
        Box::new(|cc| Ok(Box::new(RasterizerApp::new(settings, cc)))),
    )
}
