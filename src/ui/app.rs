use crate::core::renderer::Renderer;
use crate::io::args::Args; // 确保导入 AnimationType 和 RotationAxis
use crate::material_system::materials::ModelData;
use crate::scene::scene_utils::Scene;
use clap::Parser;
use egui::{Color32, ColorImage, RichText, Vec2};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

// 导入其他UI模块
use super::animation::AnimationMethods;
use super::core::CoreMethods;
use super::render_ui::RenderMethods;
use super::widgets::WidgetMethods;

/// GUI应用状态
pub struct RasterizerApp {
    // 渲染相关
    pub renderer: Renderer,
    pub scene: Option<Scene>,
    pub model_data: Option<ModelData>,

    // 命令行参数（所有渲染参数现在都在这里）
    pub args: Args,

    // UI状态
    pub rendered_image: Option<egui::TextureHandle>,
    pub last_render_time: Option<std::time::Duration>,
    pub status_message: String,
    pub show_error_dialog: bool,
    pub error_message: String,

    // 实时渲染性能统计
    pub current_fps: f32,      // 当前实时帧率
    pub fps_history: Vec<f32>, // 帧率历史记录，用于平滑显示
    pub avg_fps: f32,          // 平均帧率

    // 实时渲染状态
    pub is_realtime_rendering: bool,
    pub last_frame_time: Option<std::time::Instant>,

    // 预渲染相关字段
    pub pre_render_mode: bool,  // 是否启用预渲染模式
    pub is_pre_rendering: bool, // 是否正在预渲染
    pub pre_rendered_frames: Arc<Mutex<Vec<ColorImage>>>, // 预渲染的帧集合
    pub current_frame_index: usize, // 当前显示的帧索引
    pub pre_render_progress: Arc<AtomicUsize>, // 预渲染进度
    pub animation_time: f32,    // 全局动画计时器，用于跟踪动画总时长
    pub total_frames_for_pre_render_cycle: usize, // 预渲染一个完整周期所需的总帧数

    // 视频生成状态
    pub is_generating_video: bool,
    pub video_generation_thread: Option<std::thread::JoinHandle<(bool, String)>>,
    pub video_progress: Arc<AtomicUsize>,

    // ffmpeg 检查结果
    pub ffmpeg_available: bool,
}

impl RasterizerApp {
    /// 创建新的GUI应用实例
    pub fn new(args: Args, cc: &eframe::CreationContext<'_>) -> Self {
        // 配置字体，添加中文支持
        let mut fonts = egui::FontDefinitions::default();

        // 尝试添加一个支持中文的字体
        // 注意：这里使用系统中可能存在的中文字体，如果字体不存在，可能需要调整
        fonts.font_data.insert(
            "chinese_font".to_owned(),
            egui::FontData::from_static(include_bytes!(
                "../../assets/Noto_Sans_SC/static/NotoSansSC-Regular.ttf"
            ))
            .into(),
        );

        // 将中文字体添加到所有文本样式中
        for (_text_style, font_ids) in fonts.families.iter_mut() {
            font_ids.push("chinese_font".to_owned());
        }

        // 应用字体设置
        cc.egui_ctx.set_fonts(fonts);

        // 确保args中的Phong着色开启，PBR关闭（这是一个浅复制，不会改变原始args）
        let mut args_copy = args.clone();
        args_copy.use_phong = true;
        args_copy.use_pbr = false; // 确保PBR默认关闭

        // 创建渲染器
        let renderer = Renderer::new(args_copy.width, args_copy.height);

        // 检查ffmpeg是否可用
        let ffmpeg_available = Self::check_ffmpeg_available();

        // 设置初始状态
        Self {
            renderer,
            scene: None,
            model_data: None,
            args: args_copy,

            rendered_image: None,
            last_render_time: None,
            status_message: String::new(),
            show_error_dialog: false,
            error_message: String::new(),

            current_fps: 0.0,        // 初始化当前帧率
            fps_history: Vec::new(), // 初始化帧率历史记录
            avg_fps: 0.0,            // 初始化平均帧率

            is_realtime_rendering: false,
            last_frame_time: None,

            // 预渲染字段初始化
            pre_render_mode: false, // 默认禁用
            is_pre_rendering: false,
            pre_rendered_frames: Arc::new(Mutex::new(Vec::new())),
            current_frame_index: 0,
            pre_render_progress: Arc::new(AtomicUsize::new(0)),
            animation_time: 0.0,
            total_frames_for_pre_render_cycle: 0,

            is_generating_video: false,
            video_generation_thread: None,
            video_progress: Arc::new(AtomicUsize::new(0)),

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
        // 使用CoreMethods中的实现
        CoreMethods::set_error(self, message.clone());
        // 额外设置UI特有的错误对话框
        self.error_message = message;
        self.show_error_dialog = true;
    }

    /// 重置所有参数为默认值
    pub fn reset_to_defaults(&mut self) {
        // 保留当前的文件路径和输出设置
        let obj_path = self.args.obj.clone();
        let output_dir = self.args.output_dir.clone();
        let output_name = self.args.output.clone();

        // 创建新的默认Args实例
        let mut new_args = Args::parse_from(["program_name", "--obj", &obj_path].iter());

        // 恢复保留的路径
        new_args.output_dir = output_dir;
        new_args.output = output_name;
        new_args.use_phong = true; // 确保Phong着色默认开启
        new_args.use_pbr = false; // 确保PBR渲染默认关闭

        // 如果宽度或高度发生变化，需要重新创建渲染器
        if self.renderer.frame_buffer.width != new_args.width
            || self.renderer.frame_buffer.height != new_args.height
        {
            // 创建新的渲染器，使用新的宽高
            self.renderer = Renderer::new(new_args.width, new_args.height);
            // 清除已渲染的图像
            self.rendered_image = None;
        }

        // 更新Args对象
        self.args = new_args;

        // 使用CoreMethods中的实现重置应用状态
        CoreMethods::reset_to_defaults(self);
    }

    // 注意：clear_pre_rendered_frames, update_fps_stats 和 get_fps_display
    // 方法已删除，直接使用 CoreMethods trait 的实现
}

impl eframe::App for RasterizerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // 显示错误对话框（如果有）
        self.show_error_dialog_ui(ctx);

        // 检查快捷键
        if ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::R)) {
            self.render(ctx);
        }

        // 执行实时渲染循环
        if self.is_realtime_rendering {
            self.perform_realtime_rendering(ctx);
        }

        // 检查视频生成进度
        if self.is_generating_video {
            // 检查线程是否已完成
            if let Some(handle) = &self.video_generation_thread {
                if handle.is_finished() {
                    // 线程已完成，更新状态
                    let progress = self.video_progress.load(Ordering::SeqCst);
                    if progress >= self.args.total_frames {
                        // 视频生成成功完成
                        self.status_message = format!(
                            "视频生成完成，已保存到 {}/{}.mp4",
                            self.args.output_dir, self.args.output
                        );
                        // 重置线程句柄和状态
                        self.video_generation_thread = None;
                        self.is_generating_video = false;
                    }
                } else {
                    // 线程仍在运行，更新进度显示
                    let progress = self.video_progress.load(Ordering::SeqCst);
                    let percent = (progress as f32 / self.args.total_frames as f32 * 100.0).round();
                    self.status_message = format!(
                        "后台生成视频中... ({}/{}，{:.0}%)",
                        progress, self.args.total_frames, percent
                    );
                    ctx.request_repaint_after(std::time::Duration::from_millis(500));
                }
            }
        }

        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("光栅化渲染器");
                ui.separator();
                ui.label(&self.status_message);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // 仅在实时渲染时显示帧率
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
                // 使用带有滚动区域的侧边栏
                self.draw_side_panel(ctx, ui);
            });

        // 中央面板 - 显示渲染结果
        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(texture) = &self.rendered_image {
                // 计算图像尺寸，确保显示为正方形或接近正方形
                let available_size = ui.available_size();

                // 取可用空间中较小的一边作为正方形的边长
                let square_size = available_size.x.min(available_size.y) * 0.95; // 留一点边距

                // 根据图像的实际宽高比进行调整
                let image_aspect = self.renderer.frame_buffer.width as f32
                    / self.renderer.frame_buffer.height as f32;

                let (width, height) = if image_aspect > 1.0 {
                    // 宽大于高
                    (square_size, square_size / image_aspect)
                } else {
                    // 高大于宽
                    (square_size * image_aspect, square_size)
                };

                // 居中显示图像
                ui.horizontal(|ui| {
                    ui.add(egui::Image::new(texture).fit_to_exact_size(Vec2::new(width, height)));
                });
            } else {
                // 显示空白提示
                ui.vertical_centered(|ui| {
                    ui.add_space(100.0);
                    ui.label(RichText::new("无渲染结果").size(24.0).color(Color32::GRAY));
                    ui.label(RichText::new("点击「开始渲染」按钮或按Ctrl+R").color(Color32::GRAY));
                });
            }
        });

        // 在每帧更新结束时清理不需要的资源
        self.cleanup_resources();
    }
}

/// 启动GUI应用
pub fn start_gui(args: Args) -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1350.0, 900.0])
            .with_min_inner_size([1100.0, 700.0]), // 增加最小宽度
        ..Default::default()
    };

    eframe::run_native(
        "光栅化渲染器",
        options,
        Box::new(|cc| Ok(Box::new(RasterizerApp::new(args, cc)))),
    )
}
