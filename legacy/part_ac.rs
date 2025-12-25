                    let success = ffmpeg_status.is_ok_and(|s| s.success());

                    // 视频生成后清理临时文件
                    let _ = fs::remove_dir_all(&frames_dir_clone);

                    (success, video_output_path)
                });

                self.video_generation_thread = Some(thread_handle);
            }
            Err(e) => self.set_error(e),
        }
    }

    fn start_pre_rendering(&mut self, ctx: &Context) {
        if self.is_pre_rendering {
            return;
        }

        // 使用 CoreMethods 验证参数
        match self.settings.validate() {
            Ok(_) => {
                if self.scene.is_none() {
                    let obj_path = match &self.settings.obj {
                        Some(path) => path.clone(),
                        None => {
                            self.set_error("错误: 未指定OBJ文件路径".to_string());
                            self.stop_animation_rendering();
                            return;
                        }
                    };
                    match ModelLoader::load_and_create_scene(&obj_path, &self.settings) {
                        Ok((scene, model_data)) => {
                            self.scene = Some(scene);
                            self.model_data = Some(model_data);
                            self.status_message = "模型加载成功，开始预渲染...".to_string();
                        }
                        Err(e) => {
                            self.set_error(format!("加载模型失败，无法预渲染: {e}"));
                            return;
                        }
                    }
                }

                // 使用通用函数计算旋转参数
                let (_, seconds_per_rotation, frames_to_render) =
                    calculate_rotation_parameters(self.settings.rotation_speed, self.settings.fps);

                self.total_frames_for_pre_render_cycle = frames_to_render;

                self.is_pre_rendering = true;
                self.pre_rendered_frames.lock().unwrap().clear();
                self.pre_render_progress.store(0, Ordering::SeqCst);
                self.current_frame_index = 0;

                let settings_for_thread = self.settings.clone();
                let progress_arc = self.pre_render_progress.clone();
                let frames_arc = self.pre_rendered_frames.clone();
                let width = settings_for_thread.width;
                let height = settings_for_thread.height;
                let scene_clone = self.scene.as_ref().expect("场景已检查存在").clone();

                self.status_message = format!(
                    "开始预渲染动画 (0/{frames_to_render} 帧，转一圈需 {seconds_per_rotation:.1} 秒)..."
                );
                ctx.request_repaint();
                let ctx_clone = ctx.clone();

                thread::spawn(move || {
                    // 使用通用渲染函数
                    render_one_rotation_cycle(
                        scene_clone,
                        &settings_for_thread,
                        &progress_arc,
                        &ctx_clone,
                        width,
                        height,
                        |_, color_data_rgb| {
                            // 将RGB数据转换为RGBA并存储为ColorImage
                            let mut rgba_data = Vec::with_capacity(width * height * 4);
                            for chunk in color_data_rgb.chunks_exact(3) {
                                rgba_data.extend_from_slice(chunk);
                                rgba_data.push(255); // Alpha
                            }
                            let color_image =
                                ColorImage::from_rgba_unmultiplied([width, height], &rgba_data);
                            frames_arc.lock().unwrap().push(color_image);
                        },
                    );
                });
            }
            Err(e) => {
                self.set_error(e);
                self.is_pre_rendering = false;
            }
        }
    }

    fn handle_pre_rendering_tasks(&mut self, ctx: &Context) {
        let progress = self.pre_render_progress.load(Ordering::SeqCst);
        let expected_total_frames = self.total_frames_for_pre_render_cycle;

        // 使用通用函数计算参数
        let (_, seconds_per_rotation, _) =
            calculate_rotation_parameters(self.settings.rotation_speed, self.settings.fps);

        self.status_message = format!(
            "预渲染动画中... ({}/{} 帧，{:.1}%，转一圈约需 {:.1} 秒)",
            progress,
            expected_total_frames,
            if expected_total_frames > 0 {
                progress as f32 / expected_total_frames as f32 * 100.0
            } else {
                0.0
            },
            seconds_per_rotation
        );

        if progress >= expected_total_frames && expected_total_frames > 0 {
            self.is_pre_rendering = false;
            let final_frame_count = self.pre_rendered_frames.lock().unwrap().len();
            self.status_message = format!(
                "预渲染完成！已缓存 {} 帧动画 (目标 {} FPS, 转一圈 {:.1} 秒)",
                final_frame_count, self.settings.fps, seconds_per_rotation
            );
            if self.is_realtime_rendering || self.pre_render_mode {
                self.current_frame_index = 0;
                self.last_frame_time = None;
                ctx.request_repaint();
            }
        } else {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
    }

    fn play_pre_rendered_frames(&mut self, ctx: &Context) {
        let frame_to_display_idx;
        let frame_image;
        let frames_len;
        {
            let frames_guard = self.pre_rendered_frames.lock().unwrap();
            frames_len = frames_guard.len();
            if frames_len == 0 {
                self.pre_render_mode = false;
                self.status_message = "预渲染帧丢失或未生成，退出预渲染模式。".to_string();
                ctx.request_repaint();
                return;
            }
            frame_to_display_idx = self.current_frame_index % frames_len;
            frame_image = frames_guard[frame_to_display_idx].clone();
        }

        let now = Instant::now();
        let target_frame_duration = Duration::from_secs_f32(1.0 / self.settings.fps.max(1) as f32);

        if let Some(last_frame_display_time) = self.last_frame_time {
            let time_since_last_display = now.duration_since(last_frame_display_time);
            if time_since_last_display < target_frame_duration {
                let time_to_wait = target_frame_duration - time_since_last_display;
                ctx.request_repaint_after(time_to_wait);
                return;
            }
            self.update_fps_stats(time_since_last_display);
        } else {
            self.update_fps_stats(target_frame_duration);
        }
        self.last_frame_time = Some(now);

        let texture_name = format!("pre_rendered_tex_{frame_to_display_idx}");
        self.rendered_image =
            Some(ctx.load_texture(texture_name, frame_image, TextureOptions::LINEAR));
        self.current_frame_index = (self.current_frame_index + 1) % frames_len;

        // 使用通用函数计算参数
        let (_, seconds_per_rotation, _) =
            calculate_rotation_parameters(self.settings.rotation_speed, self.settings.fps);

        self.status_message = format!(
            "播放预渲染: 帧 {}/{} (目标 {} FPS, 平均 {:.1} FPS, 1圈 {:.1}秒)",
            frame_to_display_idx + 1,
            frames_len,
            self.settings.fps,
            self.avg_fps,
            seconds_per_rotation
        );
        ctx.request_repaint();
    }
}
use super::animation::AnimationMethods;
use super::core::CoreMethods;
use super::widgets::WidgetMethods;
use crate::core::renderer::Renderer;
use crate::io::render_settings::RenderSettings;
use crate::material_system::materials::Model;
use crate::scene::scene_utils::Scene;
use crate::utils::render_utils::calculate_rotation_parameters;
use egui::{Color32, ColorImage, RichText, Vec2};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// GUI应用状态
pub struct RasterizerApp {
    // TOML可配置参数
    pub settings: RenderSettings,

    // 渲染运行时状态
    pub renderer: Renderer,
    pub scene: Option<Scene>,
    pub model_data: Option<Model>,

    // GUI界面状态
    pub rendered_image: Option<egui::TextureHandle>,
    pub last_render_time: Option<std::time::Duration>,
    pub status_message: String,
    pub show_error_dialog: bool,
    pub error_message: String,
    pub is_dark_theme: bool,

    // 实时渲染状态
    pub current_fps: f32,
    pub fps_history: Vec<f32>,
    pub avg_fps: f32,
    pub is_realtime_rendering: bool,
    pub last_frame_time: Option<std::time::Instant>,

    // 预渲染状态
    pub pre_render_mode: bool,
    pub is_pre_rendering: bool,
    pub pre_rendered_frames: Arc<Mutex<Vec<ColorImage>>>,
    pub current_frame_index: usize,
    pub pre_render_progress: Arc<AtomicUsize>,
    pub animation_time: f32,
    pub total_frames_for_pre_render_cycle: usize,

    // 视频生成状态
    pub is_generating_video: bool,
    pub video_generation_thread: Option<std::thread::JoinHandle<(bool, String)>>,
    pub video_progress: Arc<AtomicUsize>,

    // 相机交互设置
    pub camera_pan_sensitivity: f32,
    pub camera_orbit_sensitivity: f32,
    pub camera_dolly_sensitivity: f32,

    // 相机交互状态
    pub interface_interaction: InterfaceInteraction,

    // 系统状态
    pub ffmpeg_available: bool,
}

/// 相机交互状态
#[derive(Default)]
pub struct InterfaceInteraction {
    pub camera_is_dragging: bool,
    pub camera_is_orbiting: bool,
    pub last_mouse_pos: Option<egui::Pos2>,
    pub anything_changed: bool, // 标记相机等是否发生变化，需要重新渲染
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

        // 浅色主题
        // cc.egui_ctx.set_visuals(egui::Visuals::light());

        // 深色主题
        cc.egui_ctx.set_visuals(egui::Visuals::dark());

        // 创建渲染器
        let renderer = Renderer::new(settings.width, settings.height);

        // 检查ffmpeg是否可用
        let ffmpeg_available = Self::check_ffmpeg_available();

        Self {
            // ===== TOML可配置参数 =====
            settings,

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
            is_dark_theme: true, // 默认使用深色主题

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

    fn handle_camera_interaction(&mut self, image_response: &egui::Response, ctx: &egui::Context) {
        if let Some(scene) = &mut self.scene {
            let mut camera_changed = false;
            let mut need_clear_ground_cache = false;

            let screen_size = Vec2::new(
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
                    // 只清除地面本体和阴影缓存
                    self.renderer.frame_buffer.invalidate_ground_base_cache();
                    self.renderer.frame_buffer.invalidate_ground_shadow_cache();
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

                    let (_, _, frames_per_rotation) = calculate_rotation_parameters(
                        self.settings.rotation_speed,
                        self.settings.fps,
                    );
                    let total_frames =
                        (frames_per_rotation as f32 * self.settings.rotation_cycles) as usize;

                    let percent = (progress as f32 / total_frames as f32 * 100.0).round();

                    self.status_message =
                        format!("生成视频中... ({progress}/{total_frames}，{percent:.0}%)");

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
                    ui.max_rect().right_bottom() - Vec2::new(220.0, 20.0),
                    Vec2::new(220.0, 20.0),
                );

                ui.scope_builder(
                    egui::UiBuilder::new()
                        .max_rect(overlay_rect)
                        .layout(egui::Layout::right_to_left(egui::Align::BOTTOM)),
                    |ui| {
                        ui.group(|ui| {
                            ui.label(RichText::new("相机交互").size(14.0).strong());
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
                            ui.small(RichText::new("交互已启用").color(Color32::GREEN));
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
                        RichText::new("加载模型后可在此区域进行相机交互")
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
use egui::{Color32, Context, RichText, Vec2};
use std::sync::atomic::Ordering;

use super::animation::AnimationMethods;
use super::app::RasterizerApp;
use super::core::CoreMethods;
use super::render_ui::RenderUIMethods;
use crate::core::renderer::Renderer;
use crate::geometry::camera::ProjectionType;
use crate::io::config_loader::TomlConfigLoader;
use crate::io::render_settings::{AnimationType, RotationAxis, parse_point3, parse_vec3};
use crate::material_system::light::Light;
use crate::utils::render_utils::calculate_rotation_parameters;

/// UI组件和工具提示相关方法的特质
pub trait WidgetMethods {
    /// 绘制UI的侧边栏
    fn draw_side_panel(&mut self, ctx: &Context, ui: &mut egui::Ui);

    /// 显示错误对话框
    fn show_error_dialog_ui(&mut self, ctx: &Context);

    /// 显示工具提示
    fn add_tooltip(response: egui::Response, ctx: &Context, text: &str) -> egui::Response;

    // === 面板函数接口 ===

    /// 绘制文件与输出设置面板
    fn ui_file_output_panel(app: &mut RasterizerApp, ui: &mut egui::Ui, ctx: &Context);

    /// 绘制渲染属性设置面板
    fn ui_render_properties_panel(app: &mut RasterizerApp, ui: &mut egui::Ui, ctx: &Context);

    /// 绘制物体变换控制面板
    fn ui_object_transform_panel(app: &mut RasterizerApp, ui: &mut egui::Ui, ctx: &Context);

    /// 绘制背景与环境设置面板
    fn ui_background_settings(app: &mut RasterizerApp, ui: &mut egui::Ui);

    /// 绘制相机设置面板
    fn ui_camera_settings_panel(app: &mut RasterizerApp, ui: &mut egui::Ui, ctx: &Context);

    /// 绘制光照设置面板
    fn ui_lighting_panel(app: &mut RasterizerApp, ui: &mut egui::Ui, ctx: &Context);

    /// 绘制PBR材质设置面板
    fn ui_pbr_material_panel(app: &mut RasterizerApp, ui: &mut egui::Ui, ctx: &Context);

    /// 绘制Phong材质设置面板
    fn ui_phong_material_panel(app: &mut RasterizerApp, ui: &mut egui::Ui, ctx: &Context);

    /// 绘制动画设置面板
    fn ui_animation_panel(app: &mut RasterizerApp, ui: &mut egui::Ui, ctx: &Context);

    /// 绘制按钮控制面板
    fn ui_button_controls_panel(app: &mut RasterizerApp, ui: &mut egui::Ui, ctx: &Context);

    /// 绘制渲染信息面板
    fn ui_render_info_panel(app: &mut RasterizerApp, ui: &mut egui::Ui);
}

impl WidgetMethods for RasterizerApp {
    /// 重构后的侧边栏
    fn draw_side_panel(&mut self, ctx: &Context, ui: &mut egui::Ui) {
        // 主题切换控件（放在侧边栏顶部）
        ui.horizontal(|ui| {
            ui.label("主题：");
            egui::ComboBox::from_id_salt("theme_switch")
                .selected_text(if self.is_dark_theme {
                    "深色"
                } else {
                    "浅色"
                })
                .show_ui(ui, |ui| {
                    if ui
                        .selectable_value(&mut self.is_dark_theme, true, "深色")
                        .clicked()
                    {
                        ctx.set_visuals(egui::Visuals::dark());
                    }
                    if ui
                        .selectable_value(&mut self.is_dark_theme, false, "浅色")
                        .clicked()
                    {
                        ctx.set_visuals(egui::Visuals::light());
                    }
                });
        });
        ui.separator();

        egui::ScrollArea::vertical().show(ui, |ui| {
            // === 核心设置组 ===
            ui.collapsing("📁 文件与输出", |ui| {
                Self::ui_file_output_panel(self, ui, ctx);
            });

            ui.collapsing("🎨 场景与视觉", |ui| {
                // 合并渲染属性和背景设置
                ui.group(|ui| {
                    ui.label(RichText::new("渲染设置").size(14.0).strong());
                    Self::ui_render_properties_panel(self, ui, ctx);
                });

                ui.separator();

                ui.group(|ui| {
                    ui.label(RichText::new("背景设置").size(14.0).strong());
                    Self::ui_background_settings(self, ui);
                });
            });

            // === 3D变换组 ===
            ui.collapsing("🔄 3D变换与相机", |ui| {
                ui.group(|ui| {
                    ui.label(RichText::new("物体变换").size(14.0).strong());
                    Self::ui_object_transform_panel(self, ui, ctx);
                });

                ui.separator();

                ui.group(|ui| {
                    ui.label(RichText::new("相机控制").size(14.0).strong());
                    Self::ui_camera_settings_panel(self, ui, ctx);
                });
            });

            // === 材质与光照组 ===
            ui.collapsing("💡 光照与材质", |ui| {
                // 先显示光照和通用材质属性
                Self::ui_lighting_panel(self, ui, ctx);

                ui.separator();

                // 然后根据着色模型显示专用设置
                if self.settings.use_pbr {
                    ui.group(|ui| {
                        ui.label(RichText::new("✨ PBR专用参数").size(14.0).strong());
                        Self::ui_pbr_material_panel(self, ui, ctx);
                    });
                }

                if self.settings.use_phong {
                    ui.group(|ui| {
                        ui.label(RichText::new("✨ Phong专用参数").size(14.0).strong());
                        Self::ui_phong_material_panel(self, ui, ctx);
                    });
                }
            });

            // === 动画与渲染组 ===
            ui.collapsing("🎬 动画与渲染", |ui| {
                ui.group(|ui| {
                    ui.label(RichText::new("动画设置").size(14.0).strong());
                    Self::ui_animation_panel(self, ui, ctx);
                });

                ui.separator();

                ui.group(|ui| {
                    ui.label(RichText::new("渲染控制").size(14.0).strong());
                    Self::ui_button_controls_panel(self, ui, ctx);
                });
            });

            // === 信息显示组 ===
            ui.collapsing("📊 渲染信息", |ui| {
                Self::ui_render_info_panel(self, ui);
            });
        });
    }

    /// 显示错误对话框
    fn show_error_dialog_ui(&mut self, ctx: &Context) {
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
    fn add_tooltip(response: egui::Response, _ctx: &Context, text: &str) -> egui::Response {
        response.on_hover_ui(|ui| {
            ui.add(egui::Label::new(
                RichText::new(text).size(14.0).color(Color32::DARK_GRAY),
            ));
        })
    }

    /// 文件与输出设置面板
    fn ui_file_output_panel(app: &mut RasterizerApp, ui: &mut egui::Ui, ctx: &Context) {
        ui.horizontal(|ui| {
            ui.label("OBJ文件：");
            let mut obj_text = app.settings.obj.clone().unwrap_or_default();
            let response = ui.text_edit_singleline(&mut obj_text);
            if response.changed() {
                if obj_text.is_empty() {
                    app.settings.obj = None;
                } else {
                    app.settings.obj = Some(obj_text);
                }

                // OBJ路径变化需要重新加载场景
                app.interface_interaction.anything_changed = true;
                app.scene = None; // 清除现有场景，强制重新加载
                app.rendered_image = None; // 清除渲染结果
            }
            Self::add_tooltip(response, ctx, "选择要渲染的3D模型文件（.obj格式）");
            if ui.button("浏览").clicked() {
                app.select_obj_file();
            }
        });

        // 配置文件管理
        ui.separator();
        ui.horizontal(|ui| {
            ui.label("配置文件：");
            if ui.button("📁 加载配置").clicked() {
                app.load_config_file();
            }
            if ui.button("💾 保存配置").clicked() {
                app.save_config_file();
            }
            if ui.button("📋 示例配置").clicked() {
                // 创建示例配置并应用
                match TomlConfigLoader::create_example_config("temp_example_for_gui.toml") {
                    Ok(_) => {
                        match TomlConfigLoader::load_from_file("temp_example_for_gui.toml") {
                            Ok(example_settings) => {
                                app.apply_loaded_config(example_settings);
                                app.status_message = "示例配置已应用".to_string();
                                // 删除临时文件
                                let _ = std::fs::remove_file("temp_example_for_gui.toml");
                            }
                            Err(e) => {
                                app.set_error(format!("加载示例配置失败: {e}"));
                            }
                        }
                    }
                    Err(e) => {
                        app.set_error(format!("创建示例配置失败: {e}"));
                    }
                }
            }
        });
        ui.small("💡 提示：加载配置会覆盖当前所有设置");

        ui.separator();

        ui.horizontal(|ui| {
            ui.label("输出目录：");
            let response = ui.text_edit_singleline(&mut app.settings.output_dir);
            Self::add_tooltip(response, ctx, "选择渲染结果保存的目录");
            if ui.button("浏览").clicked() {
                app.select_output_dir();
            }
        });

        ui.horizontal(|ui| {
            ui.label("输出文件名：");
            let response = ui.text_edit_singleline(&mut app.settings.output);
            Self::add_tooltip(response, ctx, "渲染结果的文件名（不含扩展名）");
        });

        ui.separator();

        ui.horizontal(|ui| {
            ui.label("宽度：");
            let old_width = app.settings.width;
            let response = ui.add(
                egui::DragValue::new(&mut app.settings.width)
                    .speed(1)
                    .range(1..=4096),
            );
            if app.settings.width != old_width {
                // 分辨率变化需要重新创建渲染器
                app.renderer = Renderer::new(app.settings.width, app.settings.height);
                app.rendered_image = None;
                app.interface_interaction.anything_changed = true;
            }
            Self::add_tooltip(response, ctx, "渲染图像的宽度（像素）");
        });

        ui.horizontal(|ui| {
            ui.label("高度：");
            let old_height = app.settings.height;
            let response = ui.add(
                egui::DragValue::new(&mut app.settings.height)
                    .speed(1)
                    .range(1..=4096),
            );
            if app.settings.height != old_height {
                app.renderer = Renderer::new(app.settings.width, app.settings.height);
                app.rendered_image = None;
                app.interface_interaction.anything_changed = true;
            }
            Self::add_tooltip(response, ctx, "渲染图像的高度（像素）");
        });

        let response = ui.checkbox(&mut app.settings.save_depth, "保存深度图");
        Self::add_tooltip(response, ctx, "同时保存深度图（深度信息可视化）");
    }

    /// 渲染属性设置面板
    fn ui_render_properties_panel(app: &mut RasterizerApp, ui: &mut egui::Ui, ctx: &Context) {
        ui.horizontal(|ui| {
            ui.label("投影类型：");
            let old_projection = app.settings.projection.clone();
            let resp1 = ui.radio_value(
                &mut app.settings.projection,
                "perspective".to_string(),
                "透视",
            );
            let resp2 = ui.radio_value(
                &mut app.settings.projection,
                "orthographic".to_string(),
                "正交",
            );
            if app.settings.projection != old_projection {
                app.interface_interaction.anything_changed = true;
            }
            Self::add_tooltip(resp1, ctx, "使用透视投影（符合人眼观察方式）");
            Self::add_tooltip(resp2, ctx, "使用正交投影（无透视变形）");
        });

        ui.separator();

        // 深度缓冲
        let old_zbuffer = app.settings.use_zbuffer;
        let resp1 = ui.checkbox(&mut app.settings.use_zbuffer, "深度缓冲");
        if app.settings.use_zbuffer != old_zbuffer {
            app.interface_interaction.anything_changed = true;
        }
        Self::add_tooltip(resp1, ctx, "启用Z缓冲进行深度测试，处理物体遮挡关系");

        // 表面颜色设置
        ui.horizontal(|ui| {
            ui.label("表面颜色：");

            let old_texture = app.settings.use_texture;
            let old_colorize = app.settings.colorize;

            let texture_response = ui.radio_value(&mut app.settings.use_texture, true, "使用纹理");
            if texture_response.clicked() && app.settings.use_texture {
                app.settings.colorize = false;
            }

            let face_color_response =
                ui.radio_value(&mut app.settings.colorize, true, "使用面颜色");
            if face_color_response.clicked() && app.settings.colorize {
                app.settings.use_texture = false;
            }

            let material_color_response = ui.radio(
                !app.settings.use_texture && !app.settings.colorize,
                "使用材质颜色",
            );
            if material_color_response.clicked() {
                app.settings.use_texture = false;
                app.settings.colorize = false;
            }

            if app.settings.use_texture != old_texture || app.settings.colorize != old_colorize {
                app.interface_interaction.anything_changed = true;
            }

            Self::add_tooltip(
                texture_response,
                ctx,
                "使用模型的纹理贴图（如果有）\n优先级最高，会覆盖面颜色设置",
            );
            Self::add_tooltip(
                face_color_response,
                ctx,
                "为每个面分配随机颜色\n仅在没有纹理或纹理被禁用时生效",
            );
            Self::add_tooltip(
                material_color_response,
                ctx,
                "使用材质的基本颜色（如.mtl文件中定义）\n在没有纹理且不使用面颜色时生效",
            );
        });

        // 着色模型设置
        ui.horizontal(|ui| {
            ui.label("着色模型：");
            let old_phong = app.settings.use_phong;
            let old_pbr = app.settings.use_pbr;

            let phong_response = ui.radio_value(&mut app.settings.use_phong, true, "Phong着色");
            if phong_response.clicked() && app.settings.use_phong {
                app.settings.use_pbr = false;
            }

            let pbr_response = ui.radio_value(&mut app.settings.use_pbr, true, "PBR渲染");
            if pbr_response.clicked() && app.settings.use_pbr {
                app.settings.use_phong = false;
            }

            if app.settings.use_phong != old_phong || app.settings.use_pbr != old_pbr {
                app.interface_interaction.anything_changed = true;
            }

            Self::add_tooltip(phong_response, ctx, "使用 Phong 着色（逐像素着色）和 Blinn-Phong 光照模型\n提供高质量的光照效果，适合大多数场景");
            Self::add_tooltip(pbr_response, ctx, "使用基于物理的渲染（PBR）\n提供更真实的材质效果，但需要更多的参数调整");
        });

        ui.separator();

        // 修改原有的增强光照效果组，添加阴影映射
        ui.group(|ui| {

            // 阴影映射设置
            let old_shadow_mapping = app.settings.enable_shadow_mapping;
            let resp = ui.checkbox(&mut app.settings.enable_shadow_mapping, "地面阴影映射");
            if app.settings.enable_shadow_mapping != old_shadow_mapping {
                app.interface_interaction.anything_changed = true;
            }
            Self::add_tooltip(
                resp,
                ctx,
                "启用简单阴影映射，在地面显示物体阴影\n需要至少一个方向光源\n相比软阴影更真实但需要更多计算"
            );

            if app.settings.enable_shadow_mapping {
                ui.group(|ui| {
                    ui.label(RichText::new("阴影映射参数").size(12.0).strong());

                    ui.horizontal(|ui| {
                        ui.label("阴影贴图尺寸:");
                        let old_size = app.settings.shadow_map_size;
                        let resp = ui.add(
                            egui::DragValue::new(&mut app.settings.shadow_map_size)
                                .speed(128)
                                .range(128..=10240)
                        );
                        if app.settings.shadow_map_size != old_size {
                            app.interface_interaction.anything_changed = true;
                        }
                        Self::add_tooltip(resp, ctx, "输入阴影贴图分辨率（如4096），越大越清晰但越慢");
                    });

                    ui.horizontal(|ui| {
                        ui.label("阴影偏移:");
                        let old_bias = app.settings.shadow_bias;
                        let resp = ui.add(
                            egui::Slider::new(&mut app.settings.shadow_bias, 0.0001..=0.01)
                                .step_by(0.0001)
                                .custom_formatter(|n, _| format!("{n:.4}"))
                        );
                        if (app.settings.shadow_bias - old_bias).abs() > f32::EPSILON {
                            app.interface_interaction.anything_changed = true;
                        }
                        Self::add_tooltip(resp, ctx, "防止阴影痤疮的偏移值\n值太小会出现自阴影，值太大会使阴影分离");
                    });

                    ui.horizontal(|ui| {
                        ui.label("阴影距离:");
                        let old_distance = app.settings.shadow_distance;
                        let resp = ui.add(
                            egui::Slider::new(&mut app.settings.shadow_distance, 1.0..=100.0)
                                .suffix(" 单位")
                        );
                        if (app.settings.shadow_distance - old_distance).abs() > f32::EPSILON {
                            app.interface_interaction.anything_changed = true;
                        }
                        Self::add_tooltip(resp, ctx, "阴影渲染的最大距离\n距离越大覆盖范围越广，但阴影精度可能降低");
                    });

                    // 是否启用PCF
                    let old_enable_pcf = app.settings.enable_pcf;
                    let resp = ui.checkbox(&mut app.settings.enable_pcf, "启用PCF软阴影");
                    if app.settings.enable_pcf != old_enable_pcf {
                        app.interface_interaction.anything_changed = true;
                    }
                    Self::add_tooltip(resp, ctx, "开启后阴影边缘会变软，抗锯齿但性能消耗增加");

                    if app.settings.enable_pcf {
                        // PCF类型选择
                        let old_pcf_type = app.settings.pcf_type.clone();
                        egui::ComboBox::from_id_salt("pcf_type_combo")
                            .selected_text(&app.settings.pcf_type)
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut app.settings.pcf_type, "Box".to_string(), "Box");
                                ui.selectable_value(&mut app.settings.pcf_type, "Gauss".to_string(), "Gauss");
                            });
                        if app.settings.pcf_type != old_pcf_type {
                            app.interface_interaction.anything_changed = true;
                        }

                        // kernel参数
                        let old_kernel = app.settings.pcf_kernel;
                        let resp = ui.add(
                            egui::Slider::new(&mut app.settings.pcf_kernel, 1..=10)
                                .text("PCF窗口(kernel)")
                        );
                        if app.settings.pcf_kernel != old_kernel {
                            app.interface_interaction.anything_changed = true;
                        }
                        Self::add_tooltip(resp, ctx, "采样窗口半径，越大越软，性能消耗也越高");

                        // Gauss类型时显示sigma
                        if app.settings.pcf_type == "Gauss" {
                            let old_sigma = app.settings.pcf_sigma;
                            let resp = ui.add(
                                egui::Slider::new(&mut app.settings.pcf_sigma, 0.1..=10.0)
                                    .text("高斯σ")
                            );
                            if (app.settings.pcf_sigma - old_sigma).abs() > f32::EPSILON {
                                app.interface_interaction.anything_changed = true;
                            }
                            Self::add_tooltip(resp, ctx, "高斯采样的σ参数，影响软化范围");
                        }
                    }
                });

                // 阴影映射状态提示
                if app.settings.lights.iter().any(|light| matches!(light, Light::Directional { enabled: true, .. })) {
                    ui.label(RichText::new("✅ 检测到方向光源，阴影映射可用").color(Color32::LIGHT_GREEN).size(12.0));
                } else {
                    ui.label(RichText::new("⚠️ 需要至少一个启用的方向光源").color(Color32::DARK_GRAY).size(12.0));
                }
            }
        });

        ui.separator();
        let old_gamma = app.settings.use_gamma;
        let resp7 = ui.checkbox(&mut app.settings.use_gamma, "Gamma校正");
        if app.settings.use_gamma != old_gamma {
            app.interface_interaction.anything_changed = true;
        }
        Self::add_tooltip(resp7, ctx, "应用伽马校正，使亮度显示更准确");

        // ACES色调映射开关
        let old_aces = app.settings.enable_aces;
        let resp = ui.checkbox(&mut app.settings.enable_aces, "启用ACES色调映射");
        if app.settings.enable_aces != old_aces {
            app.interface_interaction.anything_changed = true;
        }
        Self::add_tooltip(
            resp,
            ctx,
            "让高动态范围颜色更自然，避免过曝和死黑，推荐开启",
        );

        let old_backface = app.settings.backface_culling;
        let resp8 = ui.checkbox(&mut app.settings.backface_culling, "背面剔除");
        if app.settings.backface_culling != old_backface {
            app.interface_interaction.anything_changed = true;
        }
        Self::add_tooltip(resp8, ctx, "剔除背向相机的三角形面，提高渲染效率");

        let old_wireframe = app.settings.wireframe;
        let resp9 = ui.checkbox(&mut app.settings.wireframe, "线框模式");
        if app.settings.wireframe != old_wireframe {
            app.interface_interaction.anything_changed = true;
        }
        Self::add_tooltip(resp9, ctx, "仅渲染三角形边缘，显示为线框");

        // 小三角形剔除设置
        ui.horizontal(|ui| {
            let old_cull = app.settings.cull_small_triangles;
            let resp = ui.checkbox(&mut app.settings.cull_small_triangles, "剔除小三角形");
            if app.settings.cull_small_triangles != old_cull {
                app.interface_interaction.anything_changed = true;
            }
            Self::add_tooltip(resp, ctx, "忽略投影后面积很小的三角形，提高性能");

            if app.settings.cull_small_triangles {
                let old_area = app.settings.min_triangle_area;
                let resp = ui.add(
                    egui::DragValue::new(&mut app.settings.min_triangle_area)
                        .speed(0.0001)
                        .range(0.0..=1.0)
                        .prefix("面积阈值："),
                );
                if (app.settings.min_triangle_area - old_area).abs() > f32::EPSILON {
                    app.interface_interaction.anything_changed = true;
                }
                Self::add_tooltip(resp, ctx, "小于此面积的三角形将被剔除（范围0.0-1.0）");
            }
        });

        ui.separator();

        // 纹理设置
        ui.horizontal(|ui| {
            ui.label("纹理文件 (覆盖MTL)：");
            let mut texture_path_str = app.settings.texture.clone().unwrap_or_default();
            let resp = ui.text_edit_singleline(&mut texture_path_str);
            Self::add_tooltip(resp.clone(), ctx, "选择自定义纹理，将覆盖MTL中的定义");

            if resp.changed() {
                if texture_path_str.is_empty() {
                    app.settings.texture = None;
                } else {
                    app.settings.texture = Some(texture_path_str);
                }

                // 纹理变化应该立即触发重绘
                app.interface_interaction.anything_changed = true;
            }

            if ui.button("浏览").clicked() {
                app.select_texture_file(); // 调用 render_ui.rs 中的方法
            }
        });
    }

    /// 物体变换控制面板
    fn ui_object_transform_panel(app: &mut RasterizerApp, ui: &mut egui::Ui, ctx: &Context) {
        // 位置控制
        ui.group(|ui| {
            ui.label("物体位置 (x,y,z)：");
            let old = app.settings.object_position.clone();
            let resp = ui.text_edit_singleline(&mut app.settings.object_position);
            if app.settings.object_position != old {
                app.interface_interaction.anything_changed = true;
            }
            Self::add_tooltip(resp, ctx, "输入物体的世界坐标，例如 0,0,0");
        });

        // 旋转控制（度）
        ui.group(|ui| {
            ui.label("物体旋转 (x,y,z，度)：");
            let old = app.settings.object_rotation.clone();
            let resp = ui.text_edit_singleline(&mut app.settings.object_rotation);
            if app.settings.object_rotation != old {
                app.interface_interaction.anything_changed = true;
            }
            Self::add_tooltip(resp, ctx, "输入旋转角度（度），例如 0,45,0");
        });

        // 缩放控制
        ui.group(|ui| {
            ui.label("物体缩放 (x,y,z)：");
            let old = app.settings.object_scale_xyz.clone();
            let resp = ui.text_edit_singleline(&mut app.settings.object_scale_xyz);
            if app.settings.object_scale_xyz != old {
                app.interface_interaction.anything_changed = true;
            }
            Self::add_tooltip(resp, ctx, "输入缩放比例，例如 1,1,1");
            ui.horizontal(|ui| {
                ui.label("全局缩放:");
                let old_scale = app.settings.object_scale;
                let resp = ui.add(
                    egui::Slider::new(&mut app.settings.object_scale, 0.1..=5.0)
                        .logarithmic(true)
                        .text("倍率"),
                );
                if app.settings.object_scale != old_scale {
                    app.interface_interaction.anything_changed = true;
                }
                Self::add_tooltip(resp, ctx, "整体缩放倍率，影响所有轴");
            });
        });
    }

    /// 背景与环境设置面板
    fn ui_background_settings(app: &mut RasterizerApp, ui: &mut egui::Ui) {
        // 背景图片选项
        let old_bg_image = app.settings.use_background_image;
        ui.checkbox(&mut app.settings.use_background_image, "使用背景图片");
        if app.settings.use_background_image != old_bg_image {
            app.interface_interaction.anything_changed = true;
            app.renderer.frame_buffer.invalidate_background_cache(); // 失效背景缓存
        }

        if app.settings.use_background_image {
            ui.horizontal(|ui| {
                let mut path_text = app
                    .settings
                    .background_image_path
                    .clone()
                    .unwrap_or_default();
                ui.label("背景图片:");
                let response = ui.text_edit_singleline(&mut path_text);

                if response.changed() {
                    if path_text.is_empty() {
                        app.settings.background_image_path = None;
                    } else {
                        app.settings.background_image_path = Some(path_text.clone());
                        app.status_message = format!("背景图片路径已设置: {path_text}");
                    }

                    app.interface_interaction.anything_changed = true;
                    app.renderer.frame_buffer.invalidate_background_cache(); // 失效背景缓存
                }

                if ui.button("浏览...").clicked() {
                    app.select_background_image();
                }
            });
        }

        // 渐变背景设置
        let old_gradient = app.settings.enable_gradient_background;
        ui.checkbox(&mut app.settings.enable_gradient_background, "使用渐变背景");
        if app.settings.enable_gradient_background != old_gradient {
            app.interface_interaction.anything_changed = true;
            app.renderer.frame_buffer.invalidate_background_cache(); // 失效背景缓存
        }

        if app.settings.enable_gradient_background {
            if app.settings.use_background_image && app.settings.background_image_path.is_some() {
                ui.label(
                    egui::RichText::new("注意：渐变背景将覆盖在背景图片上")
                        .color(Color32::DARK_GRAY),
                );
            }

            // 使用按需计算的颜色值
            let top_color = app.settings.get_gradient_top_color_vec();
            let mut top_color_array = [top_color.x, top_color.y, top_color.z];
            if ui.color_edit_button_rgb(&mut top_color_array).changed() {
                app.settings.gradient_top_color = format!(
                    "{},{},{}",
                    top_color_array[0], top_color_array[1], top_color_array[2]
                );

                app.interface_interaction.anything_changed = true;
                app.renderer.frame_buffer.invalidate_background_cache(); // 失效背景缓存
            }
            ui.label("渐变顶部颜色");

            let bottom_color = app.settings.get_gradient_bottom_color_vec();
            let mut bottom_color_array = [bottom_color.x, bottom_color.y, bottom_color.z];
            if ui.color_edit_button_rgb(&mut bottom_color_array).changed() {
                app.settings.gradient_bottom_color = format!(
                    "{},{},{}",
                    bottom_color_array[0], bottom_color_array[1], bottom_color_array[2]
                );

                app.interface_interaction.anything_changed = true;
                app.renderer.frame_buffer.invalidate_background_cache(); // 失效背景缓存
            }
            ui.label("渐变底部颜色");
        }

        // 地面平面设置
        let old_ground = app.settings.enable_ground_plane;
        ui.checkbox(&mut app.settings.enable_ground_plane, "显示地面平面");
        if app.settings.enable_ground_plane != old_ground {
            app.interface_interaction.anything_changed = true;
        }

        if app.settings.enable_ground_plane {
            if app.settings.use_background_image && app.settings.background_image_path.is_some() {
                ui.label(
                    RichText::new("注意：地面平面将覆盖在背景图片上").color(Color32::DARK_GRAY),
                );
            }

            // 使用按需计算的地面颜色
            let ground_color = app.settings.get_ground_plane_color_vec();
            let mut ground_color_array = [ground_color.x, ground_color.y, ground_color.z];
            if ui.color_edit_button_rgb(&mut ground_color_array).changed() {
                app.settings.ground_plane_color = format!(
                    "{},{},{}",
                    ground_color_array[0], ground_color_array[1], ground_color_array[2]
                );

                app.interface_interaction.anything_changed = true;
            }
            ui.label("地面颜色");

            ui.horizontal(|ui| {
                if ui
                    .add(
                        egui::Slider::new(&mut app.settings.ground_plane_height, -10.0..=5.0)
                            .text("地面高度")
                            .step_by(0.1),
                    )
                    .changed()
                {
                    app.interface_interaction.anything_changed = true;
                }

                // 自动适配按钮
                if ui.button("自动适配").clicked() {
                    if let Some(optimal_height) = app.calculate_optimal_ground_height() {
                        app.settings.ground_plane_height = optimal_height;

                        app.interface_interaction.anything_changed = true;
                        app.status_message = format!("地面高度已自动调整为 {optimal_height:.2}");
                    } else {
                        app.status_message = "无法计算地面高度：请先加载模型".to_string();
                    }
                }
            });
        }
    }

    fn ui_camera_settings_panel(app: &mut RasterizerApp, ui: &mut egui::Ui, ctx: &Context) {
        ui.horizontal(|ui| {
            ui.label("相机位置 (x,y,z)：");
            let old = app.settings.camera_from.clone();
            let resp = ui.text_edit_singleline(&mut app.settings.camera_from);
            if app.settings.camera_from != old {
                if let Some(scene) = &mut app.scene {
                    if let Ok(from) = parse_point3(&app.settings.camera_from) {
                        scene.active_camera.params.position = from;
                        scene.active_camera.update_matrices();
                        app.interface_interaction.anything_changed = true;
                    }
                }
            }
            Self::add_tooltip(resp, ctx, "相机的位置坐标，格式为x,y,z");
        });

        ui.horizontal(|ui| {
            ui.label("相机目标 (x,y,z)：");
            let old = app.settings.camera_at.clone();
            let resp = ui.text_edit_singleline(&mut app.settings.camera_at);
            if app.settings.camera_at != old {
                if let Some(scene) = &mut app.scene {
                    if let Ok(at) = parse_point3(&app.settings.camera_at) {
                        scene.active_camera.params.target = at;
                        scene.active_camera.update_matrices();
                        app.interface_interaction.anything_changed = true;
                    }
                }
            }
            Self::add_tooltip(resp, ctx, "相机看向的目标点坐标，格式为x,y,z");
        });

        ui.horizontal(|ui| {
            ui.label("相机上方向 (x,y,z)：");
            let old = app.settings.camera_up.clone();
            let resp = ui.text_edit_singleline(&mut app.settings.camera_up);
            if app.settings.camera_up != old {
                if let Some(scene) = &mut app.scene {
                    if let Ok(up) = parse_vec3(&app.settings.camera_up) {
                        scene.active_camera.params.up = up.normalize();
                        scene.active_camera.update_matrices();
                        app.interface_interaction.anything_changed = true;
                    }
                }
            }
            Self::add_tooltip(resp, ctx, "相机的上方向向量，格式为x,y,z");
        });

        ui.horizontal(|ui| {
            ui.label("视场角 (度)：");
            let old_fov = app.settings.camera_fov;
            let resp = ui.add(egui::Slider::new(
                &mut app.settings.camera_fov,
                10.0..=120.0,
            ));
            if (app.settings.camera_fov - old_fov).abs() > 0.1 {
                if let Some(scene) = &mut app.scene {
                    if let ProjectionType::Perspective { fov_y_degrees, .. } =
                        &mut scene.active_camera.params.projection
                    {
                        *fov_y_degrees = app.settings.camera_fov;
                        scene.active_camera.update_matrices();
                        app.interface_interaction.anything_changed = true;
                    }
                }
            }
            Self::add_tooltip(resp, ctx, "相机视场角，值越大视野范围越广（鱼眼效果）");
        });
        ui.separator();

        // 相机交互控制设置（敏感度设置不需要立即响应，它们只影响交互行为）
        ui.group(|ui| {
            ui.label(RichText::new("相机交互控制").size(16.0).strong());
            ui.separator();

            ui.horizontal(|ui| {
                ui.label("平移敏感度:");
                let resp = ui.add(
                    egui::Slider::new(&mut app.camera_pan_sensitivity, 0.1..=5.0)
                        .step_by(0.1)
                        .text("倍率"),
                );
                Self::add_tooltip(
                    resp,
                    ctx,
                    "鼠标拖拽时的平移敏感度\n数值越大，鼠标移动相同距离时相机移动越快",
                );
            });

            ui.horizontal(|ui| {
                ui.label("旋转敏感度:");
                let resp = ui.add(
                    egui::Slider::new(&mut app.camera_orbit_sensitivity, 0.1..=5.0)
                        .step_by(0.1)
                        .text("倍率"),
                );
                Self::add_tooltip(
                    resp,
                    ctx,
                    "Shift+拖拽时的轨道旋转敏感度\n数值越大，鼠标移动相同距离时相机旋转角度越大",
                );
            });

            ui.horizontal(|ui| {
                ui.label("缩放敏感度:");
                let resp = ui.add(
                    egui::Slider::new(&mut app.camera_dolly_sensitivity, 0.1..=5.0)
                        .step_by(0.1)
                        .text("倍率"),
                );
                Self::add_tooltip(
                    resp,
                    ctx,
                    "鼠标滚轮的推拉缩放敏感度\n数值越大，滚轮滚动相同距离时相机前后移动越快",
                );
            });

            // 重置按钮
            ui.horizontal(|ui| {
                if ui.button("重置交互敏感度").clicked() {
                    app.camera_pan_sensitivity = 1.0;
                    app.camera_orbit_sensitivity = 1.0;
                    app.camera_dolly_sensitivity = 1.0;
                }

                // 预设敏感度按钮
                if ui.button("精确模式").clicked() {
                    app.camera_pan_sensitivity = 0.3;
                    app.camera_orbit_sensitivity = 0.3;
                    app.camera_dolly_sensitivity = 0.3;
                }

                if ui.button("快速模式").clicked() {
                    app.camera_pan_sensitivity = 2.0;
                    app.camera_orbit_sensitivity = 2.0;
                    app.camera_dolly_sensitivity = 2.0;
                }
            });

            // 交互说明
            ui.group(|ui| {
                ui.label(RichText::new("交互说明:").size(14.0).strong());
                ui.label("• 拖拽 - 平移相机视角");
                ui.label("• Shift + 拖拽 - 围绕目标旋转");
                ui.label("• 鼠标滚轮 - 推拉缩放");
                ui.label(
                    RichText::new("注意: 需要在中央渲染区域操作")
                        .size(12.0)
                        .color(Color32::DARK_GRAY),
                );
            });
        });
    }

    /// 光照设置面板
    fn ui_lighting_panel(app: &mut RasterizerApp, ui: &mut egui::Ui, ctx: &Context) {
        // 总光照开关
        let resp = ui
            .checkbox(&mut app.settings.use_lighting, "启用光照")
            .on_hover_text("总光照开关，关闭则仅使用环境光");
        if resp.changed() {
            app.interface_interaction.anything_changed = true;
        }

        ui.separator();

        // 环境光设置
        ui.horizontal(|ui| {
            ui.label("环境光颜色:");
            let ambient_color_vec = app.settings.get_ambient_color_vec();
            let mut ambient_color_rgb = [
                ambient_color_vec.x,
                ambient_color_vec.y,
                ambient_color_vec.z,
            ];
            let resp = ui.color_edit_button_rgb(&mut ambient_color_rgb);
            if resp.changed() {
                app.settings.ambient_color = format!(
                    "{},{},{}",
                    ambient_color_rgb[0], ambient_color_rgb[1], ambient_color_rgb[2]
                );
                app.interface_interaction.anything_changed = true;
            }
        });

        ui.horizontal(|ui| {
            ui.label("环境光强度:");
            let resp = ui.add(egui::Slider::new(&mut app.settings.ambient, 0.0..=1.0));
            if resp.changed() {
                app.interface_interaction.anything_changed = true;
            }
        });
        ui.separator();

        // 统一的材质通用属性控制
        ui.group(|ui| {
    ui.label(RichText::new("🎨 材质通用属性").size(16.0).strong());
    ui.separator();

    // 基础颜色（通用于PBR和Phong）
    ui.horizontal(|ui| {
        ui.label("基础颜色 (Base Color / Diffuse):");
        let base_color_vec = if app.settings.use_pbr {
            parse_vec3(&app.settings.base_color)
        } else {
            parse_vec3(&app.settings.diffuse_color)
        }.unwrap_or_else(|_| nalgebra::Vector3::new(0.8, 0.8, 0.8));

        let mut base_color_rgb = [base_color_vec.x, base_color_vec.y, base_color_vec.z];
        let resp = ui.color_edit_button_rgb(&mut base_color_rgb);
        if resp.changed() {
            let color_str = format!(
                "{:.3},{:.3},{:.3}",
                base_color_rgb[0], base_color_rgb[1], base_color_rgb[2]
            );

            // 同时更新PBR和Phong的颜色设置
            if app.settings.use_pbr {
                app.settings.base_color = color_str;
            } else {
                app.settings.diffuse_color = color_str;
            }
            app.interface_interaction.anything_changed = true;
        }
        Self::add_tooltip(
            resp,
            ctx,
            "材质的基础颜色\nPBR模式下为Base Color，Phong模式下为Diffuse Color",
        );
    });

    // 透明度控制（通用于PBR和Phong）
    ui.horizontal(|ui| {
        ui.label("透明度 (Alpha)：");
        let resp = ui.add(egui::Slider::new(&mut app.settings.alpha, 0.0..=1.0));
        if resp.changed() {
            app.interface_interaction.anything_changed = true;
        }
        Self::add_tooltip(
            resp,
            ctx,
            "材质透明度，0为完全透明，1为完全不透明\n适用于PBR和Phong着色模型\n调整此值可立即看到透明效果",
        );
    });

    // 自发光控制（通用于PBR和Phong）
    ui.horizontal(|ui| {
        ui.label("自发光颜色 (Emissive):");
        let emissive_color_vec = parse_vec3(&app.settings.emissive)
            .unwrap_or_else(|_| nalgebra::Vector3::new(0.0, 0.0, 0.0));
        let mut emissive_color_rgb = [
            emissive_color_vec.x,
            emissive_color_vec.y,
            emissive_color_vec.z,
        ];
        let resp = ui.color_edit_button_rgb(&mut emissive_color_rgb);
        if resp.changed() {
            app.settings.emissive = format!(
                "{:.3},{:.3},{:.3}",
                emissive_color_rgb[0], emissive_color_rgb[1], emissive_color_rgb[2]
            );
            app.interface_interaction.anything_changed = true;
        }
        Self::add_tooltip(
            resp,
            ctx,
            "材质的自发光颜色，表示材质本身发出的光\n不受光照影响，适用于发光物体",
        );
    });
});

        ui.separator();

        // 直接光源管理
        if app.settings.use_lighting {
            ui.horizontal(|ui| {
                if ui.button("➕ 添加方向光").clicked() {
                    app.settings.lights.push(Light::directional(
                        nalgebra::Vector3::new(0.0, -1.0, -1.0),
                        nalgebra::Vector3::new(1.0, 1.0, 1.0),
                        0.8, // 直接使用合理的默认强度
                    ));
                    app.interface_interaction.anything_changed = true;
                }

                if ui.button("➕ 添加点光源").clicked() {
                    app.settings.lights.push(Light::point(
                        nalgebra::Point3::new(0.0, 2.0, 0.0),
                        nalgebra::Vector3::new(1.0, 1.0, 1.0),
                        1.0, // 直接使用合理的默认强度
                        Some((1.0, 0.09, 0.032)),
                    ));
                    app.interface_interaction.anything_changed = true;
                }

                ui.separator();
                ui.label(format!("光源总数: {}", app.settings.lights.len()));
            });

            ui.separator();

            // 可编辑的光源列表
            let mut to_remove = Vec::new();
            for (i, light) in app.settings.lights.iter_mut().enumerate() {
                let mut light_changed = false;

                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        // 删除按钮
                        if ui.button("🗑").on_hover_text("删除此光源").clicked() {
                            to_remove.push(i);
                            app.interface_interaction.anything_changed = true;
                        }

                        // 光源类型和编号
                        match light {
                            Light::Directional { .. } => {
                                ui.label(format!("🔦 方向光 #{}", i + 1));
                            }
                            Light::Point { .. } => {
                                ui.label(format!("💡 点光源 #{}", i + 1));
                            }
                        }
                    });

                    // 光源参数编辑
                    match light {
                        Light::Directional {
                            enabled,
                            direction_str,
                            color_str,
                            intensity,
                            ..
                        } => {
                            ui.horizontal(|ui| {
                                let resp = ui.checkbox(enabled, "启用");
                                if resp.changed() {
                                    light_changed = true;
                                }

                                if *enabled {
                                    // 独立的强度控制
                                    let resp = ui.add(
                                        egui::Slider::new(intensity, 0.0..=3.0)
                                            .text("强度")
                                            .step_by(0.1),
                                    );
                                    if resp.changed() {
                                        light_changed = true;
                                    }
                                }
                            });

                            if *enabled {
                                ui.horizontal(|ui| {
                                    ui.label("方向 (x,y,z):");
                                    let resp = ui.text_edit_singleline(direction_str);
                                    if resp.changed() {
                                        light_changed = true;
                                    }
                                });

                                ui.horizontal(|ui| {
                                    ui.label("颜色:");
                                    let color_vec = parse_vec3(color_str)
                                        .unwrap_or_else(|_| nalgebra::Vector3::new(1.0, 1.0, 1.0));
                                    let mut color_rgb = [color_vec.x, color_vec.y, color_vec.z];
                                    let resp = ui.color_edit_button_rgb(&mut color_rgb);
                                    if resp.changed() {
                                        *color_str = format!(
                                            "{},{},{}",
                                            color_rgb[0], color_rgb[1], color_rgb[2]
                                        );
                                        light_changed = true;
                                    }
                                });
                            }
                        }
                        Light::Point {
                            enabled,
                            position_str,
                            color_str,
                            intensity,
                            constant_attenuation,
                            linear_attenuation,
                            quadratic_attenuation,
                            ..
                        } => {
                            ui.horizontal(|ui| {
                                let resp = ui.checkbox(enabled, "启用");
                                if resp.changed() {
                                    light_changed = true;
                                }

                                if *enabled {
                                    // 独立的强度控制
                                    let resp = ui.add(
                                        egui::Slider::new(intensity, 0.0..=10.0)
                                            .text("强度")
                                            .step_by(0.1),
                                    );
                                    if resp.changed() {
                                        light_changed = true;
                                    }
                                }
                            });

                            if *enabled {
                                ui.horizontal(|ui| {
                                    ui.label("位置 (x,y,z):");
                                    let resp = ui.text_edit_singleline(position_str);
                                    if resp.changed() {
                                        light_changed = true;
                                    }
                                });

                                ui.horizontal(|ui| {
                                    ui.label("颜色:");
                                    let color_vec = parse_vec3(color_str)
                                        .unwrap_or_else(|_| nalgebra::Vector3::new(1.0, 1.0, 1.0));
                                    let mut color_rgb = [color_vec.x, color_vec.y, color_vec.z];
                                    let resp = ui.color_edit_button_rgb(&mut color_rgb);
                                    if resp.changed() {
                                        *color_str = format!(
                                            "{},{},{}",
                                            color_rgb[0], color_rgb[1], color_rgb[2]
                                        );
                                        light_changed = true;
                                    }
                                });

                                // 衰减设置
                                ui.collapsing("衰减参数", |ui| {
                                    ui.horizontal(|ui| {
                                        ui.label("常数:");
                                        let resp = ui.add(
                                            egui::DragValue::new(constant_attenuation)
                                                .speed(0.05)
                                                .range(0.0..=10.0),
                                        );
                                        if resp.changed() {
                                            light_changed = true;
                                        }
                                    });
                                    ui.horizontal(|ui| {
                                        ui.label("线性:");
                                        let resp = ui.add(
                                            egui::DragValue::new(linear_attenuation)
                                                .speed(0.01)
                                                .range(0.0..=1.0),
                                        );
                                        if resp.changed() {
                                            light_changed = true;
                                        }
                                    });
                                    ui.horizontal(|ui| {
                                        ui.label("二次:");
                                        let resp = ui.add(
                                            egui::DragValue::new(quadratic_attenuation)
                                                .speed(0.001)
                                                .range(0.0..=0.5),
                                        );
                                        if resp.changed() {
                                            light_changed = true;
                                        }
                                    });
                                    ui.small("💡 推荐值: 常数=1.0, 线性=0.09, 二次=0.032");
                                });
                            }
                        }
                    }
                });

                if light_changed {
                    let _ = light.update_runtime_fields();
                    app.interface_interaction.anything_changed = true;
                }
            }

            // 删除标记的光源
            for &index in to_remove.iter().rev() {
                app.settings.lights.remove(index);
            }

            // 如果没有光源，显示提示
            if app.settings.lights.is_empty() {
                ui.group(|ui| {
                    ui.label("💡 提示：当前没有光源");
                    ui.label("点击上方的「➕ 添加」按钮来添加光源");
                });
            }
        }
    }

    /// PBR材质设置面板
    fn ui_pbr_material_panel(app: &mut RasterizerApp, ui: &mut egui::Ui, ctx: &Context) {
        ui.horizontal(|ui| {
            ui.label("金属度 (Metallic)：");
            let resp = ui.add(egui::Slider::new(&mut app.settings.metallic, 0.0..=1.0));
            if resp.changed() {
                app.interface_interaction.anything_changed = true;
            }
            Self::add_tooltip(resp, ctx, "材质的金属特性，0为非金属，1为纯金属");
        });

        ui.horizontal(|ui| {
            ui.label("粗糙度 (Roughness)：");
            let resp = ui.add(egui::Slider::new(&mut app.settings.roughness, 0.0..=1.0));
            if resp.changed() {
                app.interface_interaction.anything_changed = true;
            }
            Self::add_tooltip(resp, ctx, "材质的粗糙程度，影响高光的散射");
        });

        ui.horizontal(|ui| {
            ui.label("环境光遮蔽 (AO)：");
            let resp = ui.add(egui::Slider::new(
                &mut app.settings.ambient_occlusion,
                0.0..=1.0,
            ));
            if resp.changed() {
                app.interface_interaction.anything_changed = true;
            }
            Self::add_tooltip(resp, ctx, "环境光遮蔽程度，模拟凹陷处的阴影");
        });
    }

    /// 简化后的Phong材质设置面板
    fn ui_phong_material_panel(app: &mut RasterizerApp, ui: &mut egui::Ui, ctx: &Context) {
        ui.horizontal(|ui| {
            ui.label("镜面反射颜色：");
            let specular_color_vec = parse_vec3(&app.settings.specular_color)
                .unwrap_or_else(|_| nalgebra::Vector3::new(0.5, 0.5, 0.5));
            let mut specular_color_rgb = [
                specular_color_vec.x,
                specular_color_vec.y,
                specular_color_vec.z,
            ];
            let resp = ui.color_edit_button_rgb(&mut specular_color_rgb);
            if resp.changed() {
                app.settings.specular_color = format!(
                    "{:.3},{:.3},{:.3}",
                    specular_color_rgb[0], specular_color_rgb[1], specular_color_rgb[2]
                );
                app.interface_interaction.anything_changed = true;
            }
            Self::add_tooltip(resp, ctx, "高光的颜色");
        });

        ui.horizontal(|ui| {
            ui.label("漫反射强度：");
            let resp = ui.add(egui::Slider::new(
                &mut app.settings.diffuse_intensity,
                0.0..=2.0,
            ));
            if resp.changed() {
                app.interface_interaction.anything_changed = true;
            }
            Self::add_tooltip(resp, ctx, "漫反射光的强度倍数");
        });

        ui.horizontal(|ui| {
            ui.label("镜面反射强度：");
            let resp = ui.add(egui::Slider::new(
                &mut app.settings.specular_intensity,
                0.0..=2.0,
            ));
            if resp.changed() {
                app.interface_interaction.anything_changed = true;
            }
            Self::add_tooltip(resp, ctx, "高光的强度倍数");
        });

        ui.horizontal(|ui| {
            ui.label("光泽度：");
            let resp = ui.add(egui::Slider::new(&mut app.settings.shininess, 1.0..=100.0));
            if resp.changed() {
                app.interface_interaction.anything_changed = true;
            }
            Self::add_tooltip(resp, ctx, "高光的锐利程度，值越大越集中");
        });
    }

    /// 动画设置面板
    fn ui_animation_panel(app: &mut RasterizerApp, ui: &mut egui::Ui, ctx: &Context) {
        ui.horizontal(|ui| {
            ui.label("旋转圈数:");
            let resp = ui.add(
                egui::DragValue::new(&mut app.settings.rotation_cycles)
                    .speed(0.1)
                    .range(0.1..=10.0),
            );
            Self::add_tooltip(resp, ctx, "动画完成的旋转圈数，影响生成的总帧数");
        });

        ui.horizontal(|ui| {
            ui.label("视频生成及预渲染帧率 (FPS):");
            let resp = ui.add(
                egui::DragValue::new(&mut app.settings.fps)
                    .speed(1)
                    .range(1..=60),
            );
            Self::add_tooltip(resp, ctx, "生成视频的每秒帧数");
        });

        let (_, seconds_per_rotation, frames_per_rotation) =
            calculate_rotation_parameters(app.settings.rotation_speed, app.settings.fps);
        let total_frames = (frames_per_rotation as f32 * app.settings.rotation_cycles) as usize;
        let total_seconds = seconds_per_rotation * app.settings.rotation_cycles;

        ui.label(format!(
            "估计总帧数: {total_frames} (视频长度: {total_seconds:.1}秒)"
        ));

        // 动画类型选择
        ui.horizontal(|ui| {
            ui.label("动画类型:");
            let current_animation_type = app.settings.animation_type.clone();
            egui::ComboBox::from_id_salt("animation_type_combo")
                .selected_text(match current_animation_type {
                    AnimationType::CameraOrbit => "相机轨道旋转",
                    AnimationType::ObjectLocalRotation => "物体局部旋转",
                    AnimationType::None => "无动画",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut app.settings.animation_type,
                        AnimationType::CameraOrbit,
                        "相机轨道旋转",
                    );
                    ui.selectable_value(
                        &mut app.settings.animation_type,
                        AnimationType::ObjectLocalRotation,
                        "物体局部旋转",
                    );
                    ui.selectable_value(
                        &mut app.settings.animation_type,
                        AnimationType::None,
                        "无动画",
                    );
                });
        });

        // 旋转轴选择 (仅当动画类型不是 None 时显示)
        if app.settings.animation_type != AnimationType::None {
            ui.horizontal(|ui| {
                ui.label("旋转轴:");
                let current_rotation_axis = app.settings.rotation_axis.clone();
                egui::ComboBox::from_id_salt("rotation_axis_combo")
                    .selected_text(match current_rotation_axis {
                        RotationAxis::X => "X 轴",
                        RotationAxis::Y => "Y 轴",
                        RotationAxis::Z => "Z 轴",
                        RotationAxis::Custom => "自定义轴",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut app.settings.rotation_axis,
                            RotationAxis::X,
                            "X 轴",
                        );
                        ui.selectable_value(
                            &mut app.settings.rotation_axis,
                            RotationAxis::Y,
                            "Y 轴",
                        );
                        ui.selectable_value(
                            &mut app.settings.rotation_axis,
                            RotationAxis::Z,
                            "Z 轴",
                        );
                        ui.selectable_value(
                            &mut app.settings.rotation_axis,
                            RotationAxis::Custom,
                            "自定义轴",
                        );
                    });
            });

            if app.settings.rotation_axis == RotationAxis::Custom {
                ui.horizontal(|ui| {
                    ui.label("自定义轴 (x,y,z):");
                    let resp = ui.text_edit_singleline(&mut app.settings.custom_rotation_axis);
                    Self::add_tooltip(resp, ctx, "输入自定义旋转轴，例如 1,0,0 或 0.707,0.707,0");
                });
            }
        }
        Self::add_tooltip(
            ui.label(""),
            ctx,
            "选择实时渲染和视频生成时的动画效果和旋转轴",
        );

        // 简化预渲染模式复选框逻辑
        let pre_render_enabled = app.can_toggle_pre_render();
        let mut pre_render_value = app.pre_render_mode;

        let pre_render_resp = ui.add_enabled(
            pre_render_enabled,
            egui::Checkbox::new(&mut pre_render_value, "启用预渲染模式"),
        );

        if pre_render_resp.changed() && pre_render_value != app.pre_render_mode {
            app.toggle_pre_render_mode();
        }
        Self::add_tooltip(
            pre_render_resp,
            ctx,
            "启用后，首次开始实时渲染时会预先计算所有帧，\n然后以选定帧率无卡顿播放。\n要求更多内存，但播放更流畅。",
        );

        ui.horizontal(|ui| {
            ui.label("旋转速度 (实时渲染):");
            let resp = ui.add(egui::Slider::new(
                &mut app.settings.rotation_speed,
                0.1..=5.0,
            ));
            Self::add_tooltip(resp, ctx, "实时渲染中的旋转速度倍率");
        });
    }

    /// 按钮控制面板
    fn ui_button_controls_panel(app: &mut RasterizerApp, ui: &mut egui::Ui, ctx: &Context) {
        ui.add_space(20.0);

        // 计算按钮的统一宽度
        let available_width = ui.available_width();
        let spacing = ui.spacing().item_spacing.x;

        // 第一行：2个按钮等宽
        let button_width_row1 = (available_width - spacing) / 2.0;

        // 第二行：2个按钮等宽
        let button_width_row2 = (available_width - spacing) / 2.0;

        // 第三行：2个按钮等宽
        let button_width_row3 = (available_width - spacing) / 2.0;

        let button_height = 40.0;

        // === 第一行：恢复默认值 + 开始渲染 ===
        ui.horizontal(|ui| {
            // 恢复默认值按钮
            let reset_button = ui.add_sized(
                [button_width_row1, button_height],
                egui::Button::new(RichText::new("恢复默认值").size(15.0)),
            );

            if reset_button.clicked() {
                app.reset_to_defaults();
            }

            Self::add_tooltip(
                reset_button,
                ctx,
                "重置所有渲染参数为默认值，保留文件路径设置",
            );

            // 渲染按钮
            let render_button = ui.add_sized(
                [button_width_row1, button_height],
                egui::Button::new(RichText::new("开始渲染").size(18.0).strong()),
            );

            if render_button.clicked() {
                app.render(ctx);
            }

            Self::add_tooltip(render_button, ctx, "快捷键: Ctrl+R");
        });

        ui.add_space(10.0);

        // === 第二行：动画渲染 + 截图 ===
        ui.horizontal(|ui| {
            // 动画渲染按钮
            let realtime_button_text = if app.is_realtime_rendering {
                "停止动画渲染"
            } else if app.pre_render_mode {
                "开始动画渲染 (预渲染模式)"
            } else {
                "开始动画渲染 (实时模式)"
            };

            let realtime_button = ui.add_enabled(
                app.can_render_animation(),
                egui::Button::new(RichText::new(realtime_button_text).size(15.0))
                    .min_size(Vec2::new(button_width_row2, button_height)),
            );

            if realtime_button.clicked() {
                // 如果当前在播放预渲染帧，点击时只是停止播放
                if app.is_realtime_rendering && app.pre_render_mode {
                    app.is_realtime_rendering = false;
                    app.status_message = "已停止动画渲染".to_string();
                }
                // 否则切换实时渲染状态
                else if !app.is_realtime_rendering {
                    // 使用CoreMethods中的开始动画渲染方法
                    if let Err(e) = app.start_animation_rendering() {
                        app.set_error(e);
                    }
                } else {
                    // 使用CoreMethods中的停止动画渲染方法
                    app.stop_animation_rendering();
                }
            }

            // 更新工具提示文本
            let tooltip_text = if app.pre_render_mode {
                "启动动画渲染（预渲染模式）\n• 首次启动会预先计算所有帧\n• 然后以目标帧率流畅播放\n• 需要更多内存但播放更流畅"
            } else {
                "启动动画渲染（实时模式）\n• 每帧实时计算和渲染\n• 帧率取决于硬件性能\n• 内存占用较少"
            };

            Self::add_tooltip(realtime_button, ctx, tooltip_text);

            // 截图按钮
            let screenshot_button = ui.add_enabled(
                app.rendered_image.is_some(),
                egui::Button::new(RichText::new("截图").size(15.0))
                    .min_size(Vec2::new(button_width_row2, button_height)),
            );

            if screenshot_button.clicked() {
                match app.take_screenshot() {
                    Ok(path) => {
                        app.status_message = format!("截图已保存至 {path}");
                    }
                    Err(e) => {
                        app.set_error(format!("截图失败: {e}"));
                    }
                }
            }

            Self::add_tooltip(screenshot_button, ctx, "保存当前渲染结果为图片文件");
        });

        ui.add_space(10.0);

        // === 第三行：生成视频 + 清空缓冲区 ===
        ui.horizontal(|ui| {
            let video_button_text = if app.is_generating_video {
                let progress = app.video_progress.load(Ordering::SeqCst);

                // 使用通用函数计算实际帧数
                let (_, _, frames_per_rotation) =
                    calculate_rotation_parameters(app.settings.rotation_speed, app.settings.fps);
                let total_frames =
                    (frames_per_rotation as f32 * app.settings.rotation_cycles) as usize;

                let percent = (progress as f32 / total_frames as f32 * 100.0).round();
                format!("生成视频中... {percent}%")
            } else if app.ffmpeg_available {
                "生成视频".to_string()
            } else {
                "生成视频 (需ffmpeg)".to_string()
            };

            let is_video_button_enabled = app.can_generate_video();

            // 视频生成按钮
            let video_button_response = ui.add_enabled(
                is_video_button_enabled,
                egui::Button::new(RichText::new(video_button_text).size(15.0))
                    .min_size(Vec2::new(button_width_row3, button_height)),
            );

            if video_button_response.clicked() {
                app.start_video_generation(ctx);
            }
            Self::add_tooltip(
                video_button_response,
                ctx,
                "在后台渲染多帧并生成MP4视频。\n需要系统安装ffmpeg。\n生成过程不会影响UI使用。",
            );

            // 清空缓冲区按钮
            let is_clear_buffer_enabled = app.can_clear_buffer();

            let clear_buffer_response = ui.add_enabled(
                is_clear_buffer_enabled,
                egui::Button::new(RichText::new("清空缓冲区").size(15.0))
                    .min_size(Vec2::new(button_width_row3, button_height)),
            );

            if clear_buffer_response.clicked() {
                // 使用CoreMethods实现
                app.clear_pre_rendered_frames();
            }
            Self::add_tooltip(
                clear_buffer_response,
                ctx,
                "清除已预渲染的动画帧，释放内存。\n请先停止动画渲染再清除缓冲区。",
            );
        });
    }

    /// 渲染信息面板
    fn ui_render_info_panel(app: &mut RasterizerApp, ui: &mut egui::Ui) {
        // 渲染信息
        if let Some(time) = app.last_render_time {
            ui.separator();
            ui.label(format!("渲染耗时: {time:.2?}"));

            // 显示场景统计信息（直接使用SceneStats）
            if let Some(scene) = &app.scene {
                let stats = scene.get_scene_stats();
                ui.label(format!("网格数量: {}", stats.mesh_count));
                ui.label(format!("三角形数量: {}", stats.triangle_count));
                ui.label(format!("顶点数量: {}", stats.vertex_count));
                ui.label(format!("材质数量: {}", stats.material_count));
                ui.label(format!("光源数量: {}", stats.light_count));
            }
        }

        // FPS显示
        if app.is_realtime_rendering {
            let (fps_text, fps_color) = app.get_fps_display();
            ui.separator();
            ui.label(RichText::new(fps_text).color(fps_color).size(16.0));
        }
    }
}

/// 场景统计信息
#[derive(Debug, Clone)]
pub struct SceneStats {
    pub vertex_count: usize,
    pub triangle_count: usize,
    pub material_count: usize,
    pub mesh_count: usize,
    pub light_count: usize,
}
use crate::ModelLoader;
use crate::core::renderer::Renderer;
use crate::geometry::camera::ProjectionType;
use crate::io::render_settings::{RenderSettings, parse_point3, parse_vec3};
use crate::material_system::materials::apply_material_parameters;
use crate::ui::app::RasterizerApp;
use crate::utils::render_utils::calculate_rotation_parameters;
use crate::utils::save_utils::save_render_with_settings;
use egui::{Color32, Context};
use log::{debug, error, warn};
use std::fs;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use super::app::InterfaceInteraction;

/// 核心业务逻辑方法
///
/// 该trait包含应用的核心功能：
/// - 渲染和加载逻辑
/// - 状态转换与管理
/// - 错误处理
/// - 性能统计
/// - 资源管理
pub trait CoreMethods {
    // === 核心渲染和加载 ===

    /// 渲染当前场景 - 统一渲染入口
    fn render(&mut self, ctx: &Context);

    /// 在UI中显示渲染结果
    fn display_render_result(&mut self, ctx: &Context);

    /// 如果任何事情发生变化，执行重新渲染
    fn render_if_anything_changed(&mut self, ctx: &Context);

    /// 保存当前渲染结果为截图
    fn take_screenshot(&mut self) -> Result<String, String>;

    /// 智能计算地面平面的最佳高度
    fn calculate_optimal_ground_height(&self) -> Option<f32>;

    // === 状态管理 ===

    /// 设置错误信息
    fn set_error(&mut self, message: String);

    /// 将应用状态重置为默认值
    fn reset_to_defaults(&mut self);

    /// 切换预渲染模式开启/关闭状态
    fn toggle_pre_render_mode(&mut self);

    /// 清空预渲染的动画帧缓冲区
    fn clear_pre_rendered_frames(&mut self);

    // === 状态查询 ===

    /// 检查是否可以清除预渲染缓冲区
    fn can_clear_buffer(&self) -> bool;

    /// 检查是否可以切换预渲染模式
    fn can_toggle_pre_render(&self) -> bool;

    /// 检查是否可以开始或停止动画渲染
    fn can_render_animation(&self) -> bool;

    /// 检查是否可以生成视频
    fn can_generate_video(&self) -> bool;

    // === 动画状态管理 ===

    /// 开始实时渲染动画
    fn start_animation_rendering(&mut self) -> Result<(), String>;

    /// 停止实时渲染动画
    fn stop_animation_rendering(&mut self);

    // === 性能统计 ===

    /// 更新帧率统计信息
    fn update_fps_stats(&mut self, frame_time: Duration);

    /// 获取格式化的帧率显示文本和颜色
    fn get_fps_display(&self) -> (String, Color32);

    // === 资源管理 ===

    /// 执行资源清理操作
    fn cleanup_resources(&mut self);
}

impl CoreMethods for RasterizerApp {
    // === 核心渲染和加载实现 ===

    /// 渲染当前场景
    fn render(&mut self, ctx: &Context) {
        // 验证参数
        if let Err(e) = self.settings.validate() {
            self.set_error(e);
            return;
        }

        // 获取OBJ路径
        let obj_path = match &self.settings.obj {
            Some(path) => path.clone(),
            None => {
                self.set_error("错误: 未指定OBJ文件路径".to_string());
                return;
            }
        };

        self.status_message = format!("正在加载 {obj_path}...");
        ctx.request_repaint(); // 立即更新状态消息

        // 加载模型
        match ModelLoader::load_and_create_scene(&obj_path, &self.settings) {
            Ok((scene, model_data)) => {
                debug!(
                    "场景创建完成: 光源数量={}, 使用光照={}, 环境光强度={}",
                    scene.lights.len(),
                    self.settings.use_lighting,
                    self.settings.ambient
                );

                // 直接设置场景和模型数据
                self.scene = Some(scene);
                self.model_data = Some(model_data);

                self.status_message = "模型加载成功，开始渲染...".to_string();
            }
            Err(e) => {
                self.set_error(format!("加载模型失败: {e}"));
                return;
            }
        }

        self.status_message = "模型加载成功，开始渲染...".to_string();
        ctx.request_repaint();

        // 确保输出目录存在
        let output_dir = self.settings.output_dir.clone();
        if let Err(e) = fs::create_dir_all(&output_dir) {
            self.set_error(format!("创建输出目录失败: {e}"));
            return;
        }

        // 渲染
        let start_time = Instant::now();

        if let Some(scene) = &mut self.scene {
            // 渲染到帧缓冲区
            self.renderer.render_scene(scene, &self.settings);

            // 保存输出文件
            if let Err(e) = save_render_with_settings(&self.renderer, &self.settings, None) {
                warn!("保存渲染结果时发生错误: {e}");
            }

            // 更新状态
            self.last_render_time = Some(start_time.elapsed());
            let output_dir = self.settings.output_dir.clone();
            let output_name = self.settings.output.clone();
            let elapsed = self.last_render_time.unwrap();
            self.status_message =
                format!("渲染完成，耗时 {elapsed:.2?}，已保存到 {output_dir}/{output_name}");

            // 在UI中显示渲染结果
            self.display_render_result(ctx);
        }
    }

    /// 在UI中显示渲染结果
    fn display_render_result(&mut self, ctx: &Context) {
        // 从渲染器获取图像数据
        let color_data = self.renderer.frame_buffer.get_color_buffer_bytes();

        // 确保分辨率与渲染器匹配
        let width = self.renderer.frame_buffer.width;
        let height = self.renderer.frame_buffer.height;

        // 创建或更新纹理
        let rendered_texture = self.rendered_image.get_or_insert_with(|| {
            // 创建一个全黑的空白图像
            let color = Color32::BLACK;
            ctx.load_texture(
                "rendered_image",
                egui::ColorImage::new([width, height], vec![color; width * height]),
                egui::TextureOptions::default(),
            )
        });

        // 将RGB数据转换为RGBA格式
        let mut rgba_data = Vec::with_capacity(color_data.len() / 3 * 4);
        for i in (0..color_data.len()).step_by(3) {
            if i + 2 < color_data.len() {
                rgba_data.push(color_data[i]); // R
                rgba_data.push(color_data[i + 1]); // G
                rgba_data.push(color_data[i + 2]); // B
                rgba_data.push(255); // A (完全不透明)
            }
        }

        // 更新纹理，使用渲染器的实际大小
        rendered_texture.set(
            egui::ColorImage::from_rgba_unmultiplied([width, height], &rgba_data),
            egui::TextureOptions::default(),
        );
    }

    /// 统一同步入口
    fn render_if_anything_changed(&mut self, ctx: &Context) {
        if self.interface_interaction.anything_changed && self.scene.is_some() {
            if let Some(scene) = &mut self.scene {
                // 检测渲染尺寸变化
                if self.renderer.frame_buffer.width != self.settings.width
                    || self.renderer.frame_buffer.height != self.settings.height
                {
                    self.renderer.frame_buffer.invalidate_caches();
                }

                // 强制清除地面本体和阴影缓存
                self.renderer.frame_buffer.invalidate_ground_base_cache();
                self.renderer.frame_buffer.invalidate_ground_shadow_cache();

                // 统一同步所有状态

                // 1. 光源同步
                scene.set_lights(self.settings.lights.clone());

                // 2. 相机同步
                if let Ok(from) = parse_point3(&self.settings.camera_from) {
                    scene.active_camera.params.position = from;
                }
                if let Ok(at) = parse_point3(&self.settings.camera_at) {
                    scene.active_camera.params.target = at;
                }
                if let Ok(up) = parse_vec3(&self.settings.camera_up) {
                    scene.active_camera.params.up = up.normalize();
                }
                if let ProjectionType::Perspective { fov_y_degrees, .. } =
                    &mut scene.active_camera.params.projection
                {
                    *fov_y_degrees = self.settings.camera_fov;
                }
                scene.active_camera.update_matrices();

                // 3. 物体变换同步
                let (position, rotation_rad, scale) =
                    self.settings.get_object_transform_components();
                let final_scale = if self.settings.object_scale != 1.0 {
                    scale * self.settings.object_scale
                } else {
                    scale
                };
                scene.set_object_transform(position, rotation_rad, final_scale);

                // 4. 材质参数同步
                apply_material_parameters(&mut scene.object.model, &self.settings);

                // 5. 环境光同步
                scene.set_ambient(self.settings.ambient, self.settings.get_ambient_color_vec());

                // 6. 执行渲染
                self.renderer.render_scene(scene, &self.settings);
            }

            self.display_render_result(ctx);
            self.interface_interaction.anything_changed = false;
        }
    }

    /// 保存当前渲染结果为截图
    fn take_screenshot(&mut self) -> Result<String, String> {
        // 确保输出目录存在
        if let Err(e) = fs::create_dir_all(&self.settings.output_dir) {
            return Err(format!("创建输出目录失败: {e}"));
        }

        // 生成唯一的文件名（基于时间戳）
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| format!("获取时间戳失败: {e}"))?
            .as_secs();

        let snapshot_name = format!("{}_snapshot_{}", self.settings.output, timestamp);

        // 检查是否有可用的渲染结果
        if self.rendered_image.is_none() {
            return Err("没有可用的渲染结果".to_string());
        }

        // 使用共享的渲染工具函数保存截图
        save_render_with_settings(&self.renderer, &self.settings, Some(&snapshot_name))?;

        // 返回颜色图像的路径
        let color_path =
            Path::new(&self.settings.output_dir).join(format!("{snapshot_name}_color.png"));
        Ok(color_path.to_string_lossy().to_string())
    }

    fn calculate_optimal_ground_height(&self) -> Option<f32> {
        let scene = self.scene.as_ref()?;
        let model_data = self.model_data.as_ref()?;

        let mut min_y = f32::INFINITY;
        let mut has_vertices = false;

        // 计算模型在当前变换下的最低点
        for mesh in &model_data.meshes {
            for vertex in &mesh.vertices {
                let world_pos = scene.object.transform.transform_point(&vertex.position);
                min_y = min_y.min(world_pos.y);
                has_vertices = true;
            }
        }

        if !has_vertices {
            return None;
        }

        // 智能策略：让物体贴地，避免浮空
        let ground_height = if self.settings.enable_shadow_mapping {
            // 阴影映射时：让物体稍微"嵌入"地面，确保阴影可见但物体不浮空
            min_y - 0.01 // 非常小的负偏移，让物体微微贴地
        } else {
            // 无阴影时：让物体完全贴地
            min_y
        };

        Some(ground_height)
    }

    // === 状态管理实现 ===

    /// 设置错误信息
    fn set_error(&mut self, message: String) {
        error!("{message}");
        self.status_message = format!("错误: {message}");
    }

    /// 重置应用状态到默认值
    fn reset_to_defaults(&mut self) {
        // 保留当前的文件路径设置
        let obj_path = self.settings.obj.clone();
        let output_dir = self.settings.output_dir.clone();
        let output_name = self.settings.output.clone();

        let new_settings = RenderSettings {
            obj: obj_path,
            output_dir,
            output: output_name,
            ..Default::default()
        };

        // 如果渲染尺寸变化，重新创建渲染器
        if self.renderer.frame_buffer.width != new_settings.width
            || self.renderer.frame_buffer.height != new_settings.height
        {
            self.renderer = Renderer::new(new_settings.width, new_settings.height);
            self.rendered_image = None;
        }

        self.settings = new_settings;

        // 重置GUI状态
        self.camera_pan_sensitivity = 1.0;
        self.camera_orbit_sensitivity = 1.0;
        self.camera_dolly_sensitivity = 1.0;
        self.interface_interaction = InterfaceInteraction::default();

        // 重置其他状态
        self.is_realtime_rendering = false;
        self.is_pre_rendering = false;
        self.is_generating_video = false;
        self.pre_render_mode = false;
        self.animation_time = 0.0;
        self.current_frame_index = 0;
        self.last_frame_time = None;

        // 清空预渲染缓冲区
        if let Ok(mut frames) = self.pre_rendered_frames.lock() {
            frames.clear();
        }

        self.pre_render_progress.store(0, Ordering::SeqCst);
        self.video_progress.store(0, Ordering::SeqCst);

        // 重置 FPS 统计
        self.current_fps = 0.0;
        self.fps_history.clear();
        self.avg_fps = 0.0;

        self.status_message = "已重置应用状态，光源已恢复默认设置".to_string();
    }

    /// 切换预渲染模式
    fn toggle_pre_render_mode(&mut self) {
        // 统一的状态检查
        if self.is_pre_rendering || self.is_generating_video || self.is_realtime_rendering {
            self.status_message = "无法更改渲染模式: 请先停止正在进行的操作".to_string();
            return;
        }

        // 切换模式
        self.pre_render_mode = !self.pre_render_mode;

        if self.pre_render_mode {
            // 确保旋转速度合理
            if self.settings.rotation_speed.abs() < 0.01 {
                self.settings.rotation_speed = 1.0;
            }
            self.status_message = "已启用预渲染模式，开始动画渲染时将预先计算所有帧".to_string();
        } else {
            self.status_message = "已禁用预渲染模式，缓冲区中的预渲染帧仍可使用".to_string();
        }
    }

    /// 清空预渲染帧缓冲区
    fn clear_pre_rendered_frames(&mut self) {
        // 统一的状态检查逻辑
        if self.is_realtime_rendering || self.is_pre_rendering {
            self.status_message = "无法清除缓冲区: 请先停止动画渲染或等待预渲染完成".to_string();
            return;
        }

        // 执行清除操作
        let had_frames = !self.pre_rendered_frames.lock().unwrap().is_empty();
        if had_frames {
            self.pre_rendered_frames.lock().unwrap().clear();
            self.current_frame_index = 0;
            self.pre_render_progress.store(0, Ordering::SeqCst);

            if self.is_generating_video {
                let (_, _, frames_per_rotation) =
                    calculate_rotation_parameters(self.settings.rotation_speed, self.settings.fps);
                let total_frames =
                    (frames_per_rotation as f32 * self.settings.rotation_cycles) as usize;
                let progress = self.video_progress.load(Ordering::SeqCst);
                let percent = (progress as f32 / total_frames as f32 * 100.0).round();

                self.status_message =
                    format!("生成视频中... ({progress}/{total_frames}，{percent:.0}%)");
            } else {
                self.status_message = "已清空预渲染缓冲区".to_string();
            }
        } else {
            self.status_message = "缓冲区已为空".to_string();
        }
    }

    // === 状态查询实现 ===

    fn can_clear_buffer(&self) -> bool {
        !self.pre_rendered_frames.lock().unwrap().is_empty()
            && !self.is_realtime_rendering
            && !self.is_pre_rendering
    }

    fn can_toggle_pre_render(&self) -> bool {
        !self.is_pre_rendering && !self.is_generating_video && !self.is_realtime_rendering
    }

    fn can_render_animation(&self) -> bool {
        !self.is_generating_video
    }

    fn can_generate_video(&self) -> bool {
        !self.is_realtime_rendering && !self.is_generating_video && self.ffmpeg_available
    }

    // === 动画状态管理实现 ===

    fn start_animation_rendering(&mut self) -> Result<(), String> {
        if self.is_generating_video {
            return Err("无法开始动画: 视频正在生成中".to_string());
        }

        self.is_realtime_rendering = true;
        self.last_frame_time = None;
        self.current_fps = 0.0;
        self.fps_history.clear();
        self.avg_fps = 0.0;
        self.status_message = "开始动画渲染...".to_string();

        Ok(())
    }

    fn stop_animation_rendering(&mut self) {
        self.is_realtime_rendering = false;
        self.status_message = "已停止动画渲染".to_string();
    }

    // === 性能统计实现 ===

    fn update_fps_stats(&mut self, frame_time: Duration) {
        const FPS_HISTORY_SIZE: usize = 30;
        let current_fps = 1.0 / frame_time.as_secs_f32();
        self.current_fps = current_fps;

        // 更新 FPS 历史
        self.fps_history.push(current_fps);
        if self.fps_history.len() > FPS_HISTORY_SIZE {
            self.fps_history.remove(0); // 移除最早的记录
        }

        // 计算平均 FPS
        if !self.fps_history.is_empty() {
            let sum: f32 = self.fps_history.iter().sum();
            self.avg_fps = sum / self.fps_history.len() as f32;
        }
    }

    fn get_fps_display(&self) -> (String, Color32) {
        // 根据 FPS 水平选择颜色
        let fps_color = if self.avg_fps >= 30.0 {
            Color32::from_rgb(50, 220, 50) // 绿色
        } else if self.avg_fps >= 15.0 {
            Color32::from_rgb(220, 180, 50) // 黄色
        } else {
            Color32::from_rgb(220, 50, 50) // 红色
        };

        (format!("FPS: {:.1}", self.avg_fps), fps_color)
    }

    // === 资源管理实现 ===

    fn cleanup_resources(&mut self) {
        // 实际的资源清理逻辑

        // 1. 限制FPS历史记录大小，防止内存泄漏
        if self.fps_history.len() > 60 {
            self.fps_history.drain(0..30); // 保留最近30帧的数据
        }

        // 2. 清理已完成的视频生成线程
        if let Some(handle) = &self.video_generation_thread {
            if handle.is_finished() {
                // 线程已完成，标记需要在主循环中处理
                debug!("检测到已完成的视频生成线程，等待主循环处理");
            }
        }

        // 3. 在空闲状态下进行额外清理
        if !self.is_realtime_rendering && !self.is_generating_video && !self.is_pre_rendering {
            // 清理可能的临时资源
            if self.rendered_image.is_some() && self.last_render_time.is_none() {
                // 如果有渲染结果但没有最近的渲染时间，说明可能是陈旧的结果
                // 这里可以添加更多清理逻辑
            }

            // 清理预渲染进度计数器（如果没有预渲染帧）
            if self.pre_rendered_frames.lock().unwrap().is_empty() {
                self.pre_render_progress.store(0, Ordering::SeqCst);
            }
        }
    }
}
pub mod animation;
pub mod app;
pub mod core;
pub mod render_ui;
pub mod widgets;
use crate::Renderer;
use crate::io::config_loader::TomlConfigLoader;
use crate::io::model_loader::ModelLoader;
use crate::io::render_settings::RenderSettings;
use crate::ui::app::RasterizerApp;
use log::debug;
use native_dialog::FileDialogBuilder;

/// 渲染UI交互方法的特质
///
/// 该trait专门处理与文件选择和UI交互相关的功能：
/// - 文件选择对话框
/// - 背景图片处理
/// - 输出目录选择
/// - 配置文件管理
pub trait RenderUIMethods {
    /// 选择OBJ文件
    fn select_obj_file(&mut self);

    /// 选择纹理文件
    fn select_texture_file(&mut self);

    /// 选择背景图片
    fn select_background_image(&mut self);

    /// 选择输出目录
    fn select_output_dir(&mut self);

    /// 加载配置文件
    fn load_config_file(&mut self);

    /// 保存配置文件
    fn save_config_file(&mut self);

    /// 应用加载的配置到GUI
    fn apply_loaded_config(&mut self, settings: RenderSettings);
}

impl RenderUIMethods for RasterizerApp {
    /// 选择OBJ文件
    fn select_obj_file(&mut self) {
        let result = FileDialogBuilder::default()
            .set_title("选择OBJ模型文件")
            .add_filter("OBJ模型", ["obj"])
            .open_single_file()
            .show();

        match result {
            Ok(Some(path)) => {
                if let Some(path_str) = path.to_str() {
                    self.settings.obj = Some(path_str.to_string());
                    self.status_message = format!("已选择模型: {path_str}");

                    // OBJ文件变化需要重新加载场景和重新渲染
                    self.interface_interaction.anything_changed = true;
                    self.scene = None; // 清除现有场景，强制重新加载
                    self.rendered_image = None; // 清除渲染结果
                }
            }
            Ok(None) => {
                self.status_message = "文件选择被取消".to_string();
            }
            Err(e) => {
                self.set_error(format!("文件选择器错误: {e}"));
            }
        }
    }

    /// 选择纹理文件
    fn select_texture_file(&mut self) {
        let result = FileDialogBuilder::default()
            .set_title("选择纹理文件")
            .add_filter("图像文件", ["png", "jpg", "jpeg", "bmp", "tga"])
            .open_single_file()
            .show();

        match result {
            Ok(Some(path)) => {
                if let Some(path_str) = path.to_str() {
                    self.settings.texture = Some(path_str.to_string());
                    self.status_message = format!("已选择纹理: {path_str}");

                    // 纹理变化需要重新渲染
                    self.interface_interaction.anything_changed = true;
                }
            }
            Ok(None) => {
                self.status_message = "纹理选择被取消".to_string();
            }
            Err(e) => {
                self.set_error(format!("纹理选择错误: {e}"));
            }
        }
    }

    /// 选择背景图片
    fn select_background_image(&mut self) {
        let result = FileDialogBuilder::default()
            .set_title("选择背景图片")
            .add_filter("图片文件", ["png", "jpg", "jpeg", "bmp"])
            .open_single_file()
            .show();

        match result {
            Ok(Some(path)) => {
                if let Some(path_str) = path.to_str() {
                    // 只设置背景图片路径，不再直接加载到 settings
                    self.settings.background_image_path = Some(path_str.to_string());
                    self.settings.use_background_image = true;

                    // 使用 ModelLoader 验证背景图片是否有效
                    match ModelLoader::validate_resources(&self.settings) {
                        Ok(_) => {
                            self.status_message = format!("背景图片配置成功: {path_str}");

                            // 清除已渲染的图像，强制重新渲染以应用新背景
                            self.rendered_image = None;

                            debug!("背景图片路径已设置: {path_str}");
                            debug!("背景图片将在下次渲染时由 FrameBuffer 自动加载");
                        }
                        Err(e) => {
                            // 验证失败，重置背景设置
                            self.set_error(format!("背景图片验证失败: {e}"));
                            self.settings.background_image_path = None;
                            self.settings.use_background_image = false;
                        }
                    }
                }
            }
            Ok(None) => {
                self.status_message = "图片选择被取消".to_string();
            }
            Err(e) => {
                self.set_error(format!("文件选择器错误: {e}"));
            }
        }
    }

    /// 选择输出目录
    fn select_output_dir(&mut self) {
        let result = FileDialogBuilder::default()
            .set_title("选择输出目录")
            .open_single_dir()
            .show();

        match result {
            Ok(Some(path)) => {
                if let Some(path_str) = path.to_str() {
                    self.settings.output_dir = path_str.to_string();
                    self.status_message = format!("已选择输出目录: {path_str}");
                }
            }
            Ok(None) => {
                self.status_message = "目录选择被取消".to_string();
            }
            Err(e) => {
                self.set_error(format!("目录选择器错误: {e}"));
            }
        }
    }

    /// 加载配置文件
    fn load_config_file(&mut self) {
        let result = FileDialogBuilder::default()
            .set_title("加载配置文件")
            .add_filter("TOML配置文件", ["toml"])
            .open_single_file()
            .show();

        match result {
            Ok(Some(path)) => {
                if let Some(path_str) = path.to_str() {
                    match TomlConfigLoader::load_from_file(path_str) {
                        Ok(loaded_settings) => {
                            self.apply_loaded_config(loaded_settings);
                            self.status_message = format!("配置已加载: {path_str}");
                        }
                        Err(e) => {
                            self.set_error(format!("配置加载失败: {e}"));
                        }
                    }
                }
            }
            Ok(None) => {
                self.status_message = "配置加载被取消".to_string();
            }
            Err(e) => {
                self.set_error(format!("文件选择器错误: {e}"));
            }
        }
    }

    /// 保存配置文件
    fn save_config_file(&mut self) {
        let result = FileDialogBuilder::default()
            .set_title("保存配置文件")
            .add_filter("TOML配置文件", ["toml"])
            .save_single_file()
            .show();

        match result {
            Ok(Some(path)) => {
                let mut save_path = path;

                // 自动添加.toml扩展名（如果没有）
                if save_path.extension().is_none() {
                    save_path.set_extension("toml");
                }

                if let Some(path_str) = save_path.to_str() {
                    match TomlConfigLoader::save_to_file(&self.settings, path_str) {
                        Ok(_) => {
                            self.status_message = format!("配置已保存: {path_str}");
                        }
                        Err(e) => {
                            self.set_error(format!("配置保存失败: {e}"));
                        }
                    }
                }
            }
            Ok(None) => {
                self.status_message = "配置保存被取消".to_string();
            }
            Err(e) => {
                self.set_error(format!("文件选择器错误: {e}"));
            }
        }
    }

    /// 应用加载的配置到GUI
    fn apply_loaded_config(&mut self, loaded_settings: RenderSettings) {
        // 直接替换settings，无需同步GUI专用向量字段
        self.settings = loaded_settings;

        // 如果分辨率变化，重新创建渲染器
        if self.renderer.frame_buffer.width != self.settings.width
            || self.renderer.frame_buffer.height != self.settings.height
        {
            self.renderer = Renderer::new(self.settings.width, self.settings.height);
        }

        // 清除现有场景和渲染结果，强制重新加载
        self.scene = None;
        self.rendered_image = None;
        self.interface_interaction.anything_changed = true;

        debug!("配置已应用到GUI界面");
    }
}
use crate::ModelLoader;
use crate::core::renderer::Renderer;
use crate::io::render_settings::{AnimationType, RenderSettings, get_animation_axis_vector};
use crate::scene::scene_utils::Scene;
use crate::utils::render_utils::{
    animate_scene_step, calculate_rotation_delta, calculate_rotation_parameters,
};
use crate::utils::save_utils::save_image;
use egui::{ColorImage, Context, TextureOptions};
use log::debug;
use std::fs;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use super::app::RasterizerApp;
use super::core::CoreMethods;

/// 将ColorImage转换为PNG数据
pub fn frame_to_png_data(image: &ColorImage) -> Vec<u8> {
    // ColorImage是RGBA格式，我们需要转换为RGB格式
    let mut rgb_data = Vec::with_capacity(image.width() * image.height() * 3);
    for pixel in &image.pixels {
        rgb_data.push(pixel.r());
        rgb_data.push(pixel.g());
        rgb_data.push(pixel.b());
    }
    rgb_data
}

/// 渲染一圈的动画帧
///
/// # 参数
/// * `scene_copy` - 场景的克隆
/// * `settings` - 渲染参数
/// * `progress_arc` - 进度计数器
/// * `ctx_clone` - UI上下文，用于更新界面
/// * `width` - 渲染宽度
/// * `height` - 渲染高度
/// * `on_frame_rendered` - 帧渲染完成后的回调函数，参数为(帧序号, RGB颜色数据)
///
/// # 返回值
/// 渲染的总帧数
fn render_one_rotation_cycle<F>(
    mut scene_copy: Scene,
    settings: &RenderSettings,
    progress_arc: &Arc<AtomicUsize>,
    ctx_clone: &Context,
    width: usize,
    height: usize,
    mut on_frame_rendered: F,
) -> usize
where
    F: FnMut(usize, Vec<u8>),
{
    let mut thread_renderer = Renderer::new(width, height);
    let (effective_rotation_speed_dps, _, frames_to_render) =
        calculate_rotation_parameters(settings.rotation_speed, settings.fps);

    let rotation_axis_vec = get_animation_axis_vector(settings);
    let rotation_increment_rad_per_frame =
        (360.0 / frames_to_render as f32).to_radians() * effective_rotation_speed_dps.signum();

    for frame_num in 0..frames_to_render {
        progress_arc.store(frame_num, Ordering::SeqCst);

        if frame_num > 0 {
            animate_scene_step(
                &mut scene_copy,
                &settings.animation_type,
                &rotation_axis_vec,
                rotation_increment_rad_per_frame,
            );
        }

        // === 缓存失效策略 ===
        match settings.animation_type {
            AnimationType::CameraOrbit => {
                // 相机轨道动画：地面本体和阴影都依赖相机，必须全部失效
                thread_renderer.frame_buffer.invalidate_ground_base_cache();
                thread_renderer
                    .frame_buffer
                    .invalidate_ground_shadow_cache();
            }
            AnimationType::ObjectLocalRotation => {
                if settings.enable_shadow_mapping {
                    // 物体动画+阴影：只需失效阴影缓存，地面本体可复用
                    thread_renderer
                        .frame_buffer
                        .invalidate_ground_shadow_cache();
                }
                // 未开阴影时，地面缓存可复用，无需清理
            }
            AnimationType::None => {}
        }

        thread_renderer.render_scene(&mut scene_copy, settings);
        let color_data_rgb = thread_renderer.frame_buffer.get_color_buffer_bytes();
        on_frame_rendered(frame_num, color_data_rgb);

        if frame_num % (frames_to_render.max(1) / 20).max(1) == 0 {
            ctx_clone.request_repaint();
        }
    }

    progress_arc.store(frames_to_render, Ordering::SeqCst);
    ctx_clone.request_repaint();

    frames_to_render
}

/// 动画与视频生成相关方法的特质
pub trait AnimationMethods {
    /// 执行实时渲染循环
    fn perform_realtime_rendering(&mut self, ctx: &Context);

    /// 在后台生成视频
    fn start_video_generation(&mut self, ctx: &Context);

    /// 启动预渲染过程
    fn start_pre_rendering(&mut self, ctx: &Context);

    /// 处理预渲染帧
    fn handle_pre_rendering_tasks(&mut self, ctx: &Context);

    /// 播放预渲染帧
    fn play_pre_rendered_frames(&mut self, ctx: &Context);
}

impl AnimationMethods for RasterizerApp {
    /// 执行实时渲染循环
    fn perform_realtime_rendering(&mut self, ctx: &Context) {
        // 如果启用了预渲染模式且没有预渲染帧，才进入预渲染
        if self.pre_render_mode
            && !self.is_pre_rendering
            && self.pre_rendered_frames.lock().unwrap().is_empty()
        {
            // 检查模型是否已加载
            if self.scene.is_none() {
                let obj_path = match &self.settings.obj {
                    Some(path) => path.clone(),
                    None => {
                        self.set_error("错误: 未指定OBJ文件路径".to_string());
                        self.stop_animation_rendering();
                        return;
                    }
                };
                match ModelLoader::load_and_create_scene(&obj_path, &self.settings) {
                    Ok((scene, model_data)) => {
                        self.scene = Some(scene);
                        self.model_data = Some(model_data);
                        self.start_pre_rendering(ctx);
                        return;
                    }
                    Err(e) => {
                        self.set_error(format!("加载模型失败: {e}"));
                        self.stop_animation_rendering();
                        return;
                    }
                }
            } else {
                self.start_pre_rendering(ctx);
                return;
            }
        }

        // 如果正在预渲染，处理预渲染任务
        if self.is_pre_rendering {
            self.handle_pre_rendering_tasks(ctx);
            return;
        }

        // 如果启用预渲染模式且有预渲染帧，播放预渲染帧
        if self.pre_render_mode && !self.pre_rendered_frames.lock().unwrap().is_empty() {
            self.play_pre_rendered_frames(ctx);
            return;
        }

        // === 常规实时渲染（未启用预渲染模式或预渲染复选框未勾选） ===

        // 确保场景已加载
        if self.scene.is_none() {
            let obj_path = match &self.settings.obj {
                Some(path) => path.clone(),
                None => {
                    self.set_error("错误: 未指定OBJ文件路径".to_string());
                    self.stop_animation_rendering();
                    return;
                }
            };
            match ModelLoader::load_and_create_scene(&obj_path, &self.settings) {
                Ok((scene, model_data)) => {
                    self.scene = Some(scene);
                    self.model_data = Some(model_data);
                    // 注意：这里不再自动跳转到预渲染，而是继续执行实时渲染
                    self.status_message = "模型加载成功，开始实时渲染...".to_string();
                }
                Err(e) => {
                    self.set_error(format!("加载模型失败: {e}"));
                    self.stop_animation_rendering();
                    return;
                }
            }
        }

        // 检查渲染器尺寸，但避免不必要的缓存清除
        if self.renderer.frame_buffer.width != self.settings.width
            || self.renderer.frame_buffer.height != self.settings.height
        {
            self.renderer
                .resize(self.settings.width, self.settings.height);
            self.rendered_image = None;
            debug!(
                "重新创建渲染器，尺寸: {}x{}",
                self.settings.width, self.settings.height
            );
        }

        let now = Instant::now();
        let dt = if let Some(last_time) = self.last_frame_time {
            now.duration_since(last_time).as_secs_f32()
        } else {
            1.0 / 60.0 // 默认 dt
        };
        if let Some(last_time) = self.last_frame_time {
            let frame_time = now.duration_since(last_time);
            self.update_fps_stats(frame_time);
        }
        self.last_frame_time = Some(now);

        if self.is_realtime_rendering && self.settings.rotation_speed.abs() < 0.01 {
            self.settings.rotation_speed = 1.0; // 确保实时渲染时有旋转速度
        }

        self.animation_time += dt;

        if let Some(scene) = &mut self.scene {
            // 动画过程中不清除缓存
            // 物体动画不影响背景和地面（相机不动），所以缓存仍然有效

            // 使用通用函数计算旋转增量
            let rotation_delta_rad = calculate_rotation_delta(self.settings.rotation_speed, dt);
            let rotation_axis_vec = get_animation_axis_vector(&self.settings);

            // 使用通用函数执行动画步骤
            animate_scene_step(
                scene,
                &self.settings.animation_type,
                &rotation_axis_vec,
                rotation_delta_rad,
            );

            debug!(
                "实时渲染中: FPS={:.1}, 动画类型={:?}, 轴={:?}, 旋转速度={}, 角度增量={:.3}rad, Phong={}",
                self.avg_fps,
                self.settings.animation_type,
                self.settings.rotation_axis,
                self.settings.rotation_speed,
                rotation_delta_rad,
                self.settings.use_phong
            );

            match self.settings.animation_type {
                AnimationType::CameraOrbit => {
                    // 相机轨道动画：地面本体和阴影都依赖相机，必须全部失效
                    self.renderer.frame_buffer.invalidate_ground_base_cache();
                    self.renderer.frame_buffer.invalidate_ground_shadow_cache();
                }
                AnimationType::ObjectLocalRotation => {
                    if self.settings.enable_shadow_mapping {
                        // 物体动画+阴影：只需失效阴影缓存，地面本体可复用
                        self.renderer.frame_buffer.invalidate_ground_shadow_cache();
                    }
                    // 未开阴影时，地面缓存可复用，无需清理
                }
                AnimationType::None => {}
            }

            self.renderer.render_scene(scene, &self.settings);
            self.display_render_result(ctx);
            ctx.request_repaint();
        }
    }

    fn start_video_generation(&mut self, ctx: &Context) {
        if !self.ffmpeg_available {
            self.set_error("无法生成视频：未检测到ffmpeg。请安装ffmpeg后重试。".to_string());
            return;
        }
        if self.is_generating_video {
            self.status_message = "视频已在生成中，请等待完成...".to_string();
            return;
        }

        // 使用 CoreMethods 验证参数
        match self.settings.validate() {
            Ok(_) => {
                let output_dir = self.settings.output_dir.clone();
                if let Err(e) = fs::create_dir_all(&output_dir) {
                    self.set_error(format!("创建输出目录失败: {e}"));
                    return;
                }
                let frames_dir = format!(
                    "{}/temp_frames_{}",
                    output_dir,
                    chrono::Utc::now().timestamp_millis()
                );
                if let Err(e) = fs::create_dir_all(&frames_dir) {
                    self.set_error(format!("创建帧目录失败: {e}"));
                    return;
                }

                // 计算旋转参数，获取视频帧数
                let (_, _, frames_per_rotation) =
                    calculate_rotation_parameters(self.settings.rotation_speed, self.settings.fps);

                let total_frames =
                    (frames_per_rotation as f32 * self.settings.rotation_cycles) as usize;

                // 如果场景未加载，尝试加载
                if self.scene.is_none() {
                    let obj_path = match &self.settings.obj {
                        Some(path) => path.clone(),
                        None => {
                            self.set_error("错误: 未指定OBJ文件路径".to_string());
                            return;
                        }
                    };
                    match ModelLoader::load_and_create_scene(&obj_path, &self.settings) {
                        Ok((scene, model_data)) => {
                            self.scene = Some(scene);
                            self.model_data = Some(model_data);
                            self.status_message = "模型加载成功，开始生成视频...".to_string();
                        }
                        Err(e) => {
                            self.set_error(format!("加载模型失败，无法生成视频: {e}"));
                            return;
                        }
                    }
                }

                let settings_for_thread = self.settings.clone();
                let video_progress_arc = self.video_progress.clone();
                let fps = self.settings.fps;
                let scene_clone = self.scene.as_ref().expect("场景已检查").clone();

                // 检查是否有预渲染帧
                let has_pre_rendered_frames = {
                    let frames_guard = self.pre_rendered_frames.lock().unwrap();
                    !frames_guard.is_empty()
                };

                // 如果没有预渲染帧，那么我们需要同时为预渲染缓冲区生成帧
                let frames_for_pre_render = if !has_pre_rendered_frames {
                    Some(self.pre_rendered_frames.clone())
                } else {
                    None
                };

                // 设置渲染状态
                self.is_generating_video = true;
                video_progress_arc.store(0, Ordering::SeqCst);

                // 更新状态消息
                self.status_message = format!(
                    "开始生成视频 (0/{} 帧，{:.1} 秒时长)...",
                    total_frames,
                    total_frames as f32 / fps as f32
                );

                ctx.request_repaint();
                let ctx_clone = ctx.clone();
                let video_filename = format!("{}.mp4", settings_for_thread.output);
                let video_output_path = format!("{output_dir}/{video_filename}");
                let frames_dir_clone = frames_dir.clone();

                // 如果有预渲染帧，复制到线程中
                let pre_rendered_frames_clone = if has_pre_rendered_frames {
                    let frames_guard = self.pre_rendered_frames.lock().unwrap();
                    Some(frames_guard.clone())
                } else {
                    None
                };

                let thread_handle = thread::spawn(move || {
                    let width = settings_for_thread.width;
                    let height = settings_for_thread.height;
                    let mut rendered_frames = Vec::new();

                    // 使用预渲染帧或重新渲染
                    if let Some(frames) = pre_rendered_frames_clone {
                        // 使用预渲染帧
                        let pre_rendered_count = frames.len();

                        for frame_num in 0..total_frames {
                            video_progress_arc.store(frame_num, Ordering::SeqCst);

                            // 计算当前帧在哪个圈和圈内的位置
                            let cycle_position = frame_num % frames_per_rotation;

                            // 将圈内位置映射到预渲染帧索引
                            // 这处理了预渲染帧数量可能与理论帧数不匹配的情况
                            let pre_render_idx =
                                (cycle_position * pre_rendered_count) / frames_per_rotation;

                            let frame = &frames[pre_render_idx.min(pre_rendered_count - 1)]; // 避免越界访问

                            // 将ColorImage转换为PNG并保存
                            let frame_path = format!("{frames_dir_clone}/frame_{frame_num:04}.png");
                            let color_data = frame_to_png_data(frame);
                            save_image(&frame_path, &color_data, width as u32, height as u32);

                            if frame_num % (total_frames.max(1) / 20).max(1) == 0 {
                                ctx_clone.request_repaint();
                            }
                        }
                    } else {
                        // 使用通用渲染函数渲染一圈或部分圈
                        let frames_arc = frames_for_pre_render.clone();

                        let rendered_frame_count = render_one_rotation_cycle(
                            scene_clone,
                            &settings_for_thread,
                            &video_progress_arc,
                            &ctx_clone,
                            width,
                            height,
                            |frame_num, color_data_rgb| {
                                // 保存RGB数据用于后续复用
                                rendered_frames.push(color_data_rgb.clone());

                                // 同时为视频保存PNG文件
                                let frame_path =
                                    format!("{frames_dir_clone}/frame_{frame_num:04}.png");
                                save_image(
                                    &frame_path,
                                    &color_data_rgb,
                                    width as u32,
                                    height as u32,
                                );

                                // 如果需要同时保存到预渲染缓冲区
                                if let Some(ref frames_arc) = frames_arc {
                                    // 转换为RGBA格式以用于预渲染帧
                                    let mut rgba_data = Vec::with_capacity(width * height * 4);
                                    for chunk in color_data_rgb.chunks_exact(3) {
                                        rgba_data.extend_from_slice(chunk);
                                        rgba_data.push(255); // Alpha
                                    }
                                    let color_image = ColorImage::from_rgba_unmultiplied(
                                        [width, height],
                                        &rgba_data,
                                    );
                                    frames_arc.lock().unwrap().push(color_image);
                                }
                            },
                        );

                        // 如果需要多于一圈，使用前面渲染的帧复用
                        if rendered_frame_count < total_frames {
                            for frame_num in rendered_frame_count..total_frames {
                                video_progress_arc.store(frame_num, Ordering::SeqCst);

                                // 复用之前渲染的帧
                                let source_frame_idx = frame_num % rendered_frame_count;
                                let source_data = &rendered_frames[source_frame_idx];

                                // 保存为图片文件
                                let frame_path =
                                    format!("{frames_dir_clone}/frame_{frame_num:04}.png");
                                save_image(&frame_path, source_data, width as u32, height as u32);

                                if frame_num % (total_frames.max(1) / 20).max(1) == 0 {
                                    ctx_clone.request_repaint();
                                }
                            }
                        }
                    }

                    video_progress_arc.store(total_frames, Ordering::SeqCst);
                    ctx_clone.request_repaint();

                    // 使用ffmpeg将帧序列合成为视频，并解决阻塞问题
                    let frames_pattern = format!("{frames_dir_clone}/frame_%04d.png");
                    let ffmpeg_status = std::process::Command::new("ffmpeg")
                        .args([
                            "-y",
                            "-framerate",
                            &fps.to_string(),
                            "-i",
                            &frames_pattern,
                            "-c:v",
                            "libx264",
                            "-pix_fmt",
                            "yuv420p",
                            "-crf",
                            "23",
                            &video_output_path,
                        ])
                        .status();
