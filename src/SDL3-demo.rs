use sdl3::event::Event;
use sdl3::keyboard::Keycode;
use sdl3::pixels::{Color, PixelMasks};
use sdl3::rect::{Point, Rect};
use sdl3::render::{Canvas, Texture, TextureCreator};
use sdl3::surface::Surface;
use sdl3::ttf::Font;
use sdl3::video::{Window, WindowContext};
use std::path::Path;
use std::time::Duration;

// Simple UI widget types
enum WidgetType {
    Button,
    Slider,
    Checkbox,
    TextInput,
    Label,
    Dropdown,
}

// UI widget structure
struct Widget {
    widget_type: WidgetType,
    rect: Rect,
    label: String,
    value: String,
    is_active: bool,
    is_hovered: bool,
    is_checked: bool,
    min_value: f32,
    max_value: f32,
    current_value: f32,
}

impl Widget {
    fn new_button(x: i32, y: i32, width: u32, height: u32, label: &str) -> Self {
        Widget {
            widget_type: WidgetType::Button,
            rect: Rect::new(x, y, width, height),
            label: label.to_string(),
            value: String::new(),
            is_active: false,
            is_hovered: false,
            is_checked: false,
            min_value: 0.0,
            max_value: 0.0,
            current_value: 0.0,
        }
    }

    fn new_slider(
        x: i32,
        y: i32,
        width: u32,
        height: u32,
        label: &str,
        min: f32,
        max: f32,
        current: f32,
    ) -> Self {
        Widget {
            widget_type: WidgetType::Slider,
            rect: Rect::new(x, y, width, height),
            label: label.to_string(),
            value: format!("{:.2}", current),
            is_active: false,
            is_hovered: false,
            is_checked: false,
            min_value: min,
            max_value: max,
            current_value: current,
        }
    }

    fn new_checkbox(x: i32, y: i32, width: u32, height: u32, label: &str, checked: bool) -> Self {
        Widget {
            widget_type: WidgetType::Checkbox,
            rect: Rect::new(x, y, width, height),
            label: label.to_string(),
            value: String::new(),
            is_active: false,
            is_hovered: false,
            is_checked: checked,
            min_value: 0.0,
            max_value: 0.0,
            current_value: 0.0,
        }
    }

    fn new_text_input(x: i32, y: i32, width: u32, height: u32, label: &str, value: &str) -> Self {
        Widget {
            widget_type: WidgetType::TextInput,
            rect: Rect::new(x, y, width, height),
            label: label.to_string(),
            value: value.to_string(),
            is_active: false,
            is_hovered: false,
            is_checked: false,
            min_value: 0.0,
            max_value: 0.0,
            current_value: 0.0,
        }
    }

    fn new_label(x: i32, y: i32, width: u32, height: u32, text: &str) -> Self {
        Widget {
            widget_type: WidgetType::Label,
            rect: Rect::new(x, y, width, height),
            label: text.to_string(),
            value: String::new(),
            is_active: false,
            is_hovered: false,
            is_checked: false,
            min_value: 0.0,
            max_value: 0.0,
            current_value: 0.0,
        }
    }

    fn new_dropdown(x: i32, y: i32, width: u32, height: u32, label: &str, value: &str) -> Self {
        Widget {
            widget_type: WidgetType::Dropdown,
            rect: Rect::new(x, y, width, height),
            label: label.to_string(),
            value: value.to_string(),
            is_active: false,
            is_hovered: false,
            is_checked: false,
            min_value: 0.0,
            max_value: 0.0,
            current_value: 0.0,
        }
    }

    fn contains_point(&self, x: i32, y: i32) -> bool {
        self.rect.contains_point(Point::new(x, y))
    }
}

// Render widget
fn render_widget(
    canvas: &mut Canvas<Window>,
    font: &Font,
    texture_creator: &TextureCreator<WindowContext>,
    widget: &Widget,
) -> Result<(), String> {
    match widget.widget_type {
        WidgetType::Button => render_button(canvas, font, texture_creator, widget),
        WidgetType::Slider => render_slider(canvas, font, texture_creator, widget),
        WidgetType::Checkbox => render_checkbox(canvas, font, texture_creator, widget),
        WidgetType::TextInput => render_text_input(canvas, font, texture_creator, widget),
        WidgetType::Label => render_label(canvas, font, texture_creator, widget),
        WidgetType::Dropdown => render_dropdown(canvas, font, texture_creator, widget),
    }
}

fn render_button(
    canvas: &mut Canvas<Window>,
    font: &Font,
    texture_creator: &TextureCreator<WindowContext>,
    widget: &Widget,
) -> Result<(), String> {
    // Button background
    let bg_color = if widget.is_active {
        Color::RGB(100, 100, 255)
    } else if widget.is_hovered {
        Color::RGB(150, 150, 255)
    } else {
        Color::RGB(100, 100, 200)
    };
    canvas.set_draw_color(bg_color);
    canvas.fill_rect(widget.rect).map_err(|e| e.to_string())?;

    // Button border
    canvas.set_draw_color(Color::RGB(50, 50, 150));
    canvas
        .draw_rect(widget.rect.into())
        .map_err(|e| e.to_string())?;

    // Button text
    let surface = font
        .render(&widget.label)
        .blended(Color::RGB(255, 255, 255))
        .map_err(|e| e.to_string())?;
    let texture = texture_creator
        .create_texture_from_surface(&surface)
        .map_err(|e| e.to_string())?;

    let text_width = surface.width();
    let text_height = surface.height();
    let text_rect = Rect::new(
        widget.rect.x + ((widget.rect.width() as i32 - text_width as i32) / 2),
        widget.rect.y + ((widget.rect.height() as i32 - text_height as i32) / 2),
        text_width,
        text_height,
    );

    canvas
        .copy(&texture, None, text_rect)
        .map_err(|e| e.to_string())?;
    Ok(())
}

fn render_slider(
    canvas: &mut Canvas<Window>,
    font: &Font,
    texture_creator: &TextureCreator<WindowContext>,
    widget: &Widget,
) -> Result<(), String> {
    // Draw slider track
    canvas.set_draw_color(Color::RGB(200, 200, 200));
    let track_rect = Rect::new(
        widget.rect.x + 100,
        widget.rect.y + (widget.rect.height() as i32 / 2) - 2,
        widget.rect.width() - 100,
        4,
    );
    canvas.fill_rect(track_rect).map_err(|e| e.to_string())?;

    // Draw slider handle
    let percentage =
        (widget.current_value - widget.min_value) / (widget.max_value - widget.min_value);
    let handle_pos = track_rect.x + ((track_rect.width() as f32 * percentage) as i32); // 修复浮点数精度问题
    let handle_rect = Rect::new(handle_pos - 5, widget.rect.y, 10, widget.rect.height());

    canvas.set_draw_color(if widget.is_active {
        Color::RGB(100, 100, 255)
    } else if widget.is_hovered {
        Color::RGB(150, 150, 255)
    } else {
        Color::RGB(100, 100, 200)
    });
    canvas.fill_rect(handle_rect).map_err(|e| e.to_string())?;

    // Draw label and current value
    let label_surface = font
        .render(&widget.label)
        .blended(Color::RGB(255, 255, 255))
        .map_err(|e| e.to_string())?;
    let label_texture = texture_creator
        .create_texture_from_surface(&label_surface)
        .map_err(|e| e.to_string())?;
    let label_rect = Rect::new(
        widget.rect.x,
        widget.rect.y + (widget.rect.height() as i32 / 2) - (label_surface.height() as i32 / 2),
        label_surface.width(),
        label_surface.height(),
    );
    canvas
        .copy(&label_texture, None, label_rect)
        .map_err(|e| e.to_string())?;

    // Display current value
    let value_text = format!("{:.2}", widget.current_value);
    let value_surface = font
        .render(&value_text)
        .blended(Color::RGB(255, 255, 255))
        .map_err(|e| e.to_string())?;
    let value_texture = texture_creator
        .create_texture_from_surface(&value_surface)
        .map_err(|e| e.to_string())?;
    let value_rect = Rect::new(
        track_rect.x + track_rect.width() as i32 + 10,
        widget.rect.y + (widget.rect.height() as i32 / 2) - (value_surface.height() as i32 / 2),
        value_surface.width(),
        value_surface.height(),
    );
    canvas
        .copy(&value_texture, None, value_rect)
        .map_err(|e| e.to_string())?;

    Ok(())
}

fn render_checkbox(
    canvas: &mut Canvas<Window>,
    font: &Font,
    texture_creator: &TextureCreator<WindowContext>,
    widget: &Widget,
) -> Result<(), String> {
    // Draw checkbox
    let box_size = 20;
    let box_rect = Rect::new(widget.rect.x, widget.rect.y, box_size, box_size);

    // Checkbox background
    canvas.set_draw_color(Color::RGB(255, 255, 255));
    canvas.fill_rect(box_rect).map_err(|e| e.to_string())?;

    // Checkbox border
    canvas.set_draw_color(Color::RGB(100, 100, 100));
    canvas
        .draw_rect(box_rect.into())
        .map_err(|e| e.to_string())?;

    // If checked, draw checkmark
    if widget.is_checked {
        canvas.set_draw_color(Color::RGB(0, 100, 0));
        canvas
            .draw_line(
                (box_rect.x + 3, box_rect.y + 10),
                (box_rect.x + 8, box_rect.y + 15),
            )
            .map_err(|e| e.to_string())?;
        canvas
            .draw_line(
                (box_rect.x + 8, box_rect.y + 15),
                (box_rect.x + 17, box_rect.y + 5),
            )
            .map_err(|e| e.to_string())?;
    }

    // Draw label
    let label_surface = font
        .render(&widget.label)
        .blended(Color::RGB(255, 255, 255))
        .map_err(|e| e.to_string())?;
    let label_texture = texture_creator
        .create_texture_from_surface(&label_surface)
        .map_err(|e| e.to_string())?;
    let label_rect = Rect::new(
        widget.rect.x + box_size as i32 + 10,
        widget.rect.y,
        label_surface.width(),
        label_surface.height(),
    );
    canvas
        .copy(&label_texture, None, label_rect)
        .map_err(|e| e.to_string())?;

    Ok(())
}

fn render_text_input(
    canvas: &mut Canvas<Window>,
    font: &Font,
    texture_creator: &TextureCreator<WindowContext>,
    widget: &Widget,
) -> Result<(), String> {
    // Draw label
    let label_surface = font
        .render(&widget.label)
        .blended(Color::RGB(255, 255, 255))
        .map_err(|e| e.to_string())?;
    let label_texture = texture_creator
        .create_texture_from_surface(&label_surface)
        .map_err(|e| e.to_string())?;
    let label_rect = Rect::new(
        widget.rect.x,
        widget.rect.y,
        label_surface.width(),
        label_surface.height(),
    );
    canvas
        .copy(&label_texture, None, label_rect)
        .map_err(|e| e.to_string())?;

    // Draw input box
    let input_rect = Rect::new(
        widget.rect.x + 100,
        widget.rect.y,
        widget.rect.width() - 100,
        widget.rect.height(),
    );

    // Input box background
    canvas.set_draw_color(Color::RGB(255, 255, 255));
    canvas.fill_rect(input_rect).map_err(|e| e.to_string())?;

    // Input box border
    canvas.set_draw_color(if widget.is_active {
        Color::RGB(100, 100, 255)
    } else {
        Color::RGB(150, 150, 150)
    });
    canvas
        .draw_rect(input_rect.into())
        .map_err(|e| e.to_string())?;

    // Draw input text
    if !widget.value.is_empty() {
        let value_surface = font
            .render(&widget.value)
            .blended(Color::RGB(0, 0, 0))
            .map_err(|e| e.to_string())?;
        let value_texture = texture_creator
            .create_texture_from_surface(&value_surface)
            .map_err(|e| e.to_string())?;
        let padding = 5;
        let value_rect = Rect::new(
            input_rect.x + padding,
            input_rect.y + (input_rect.height() as i32 - value_surface.height() as i32) / 2,
            value_surface.width(),
            value_surface.height(),
        );
        canvas
            .copy(&value_texture, None, value_rect)
            .map_err(|e| e.to_string())?;
    }

    Ok(())
}

fn render_label(
    canvas: &mut Canvas<Window>,
    font: &Font,
    texture_creator: &TextureCreator<WindowContext>,
    widget: &Widget,
) -> Result<(), String> {
    let surface = font
        .render(&widget.label)
        .blended(Color::RGB(255, 255, 255))
        .map_err(|e| e.to_string())?;
    let texture = texture_creator
        .create_texture_from_surface(&surface)
        .map_err(|e| e.to_string())?;
    canvas
        .copy(&texture, None, widget.rect)
        .map_err(|e| e.to_string())?;
    Ok(())
}

fn render_dropdown(
    canvas: &mut Canvas<Window>,
    font: &Font,
    texture_creator: &TextureCreator<WindowContext>,
    widget: &Widget,
) -> Result<(), String> {
    // Draw label
    let label_surface = font
        .render(&widget.label)
        .blended(Color::RGB(255, 255, 255))
        .map_err(|e| e.to_string())?;
    let label_texture = texture_creator
        .create_texture_from_surface(&label_surface)
        .map_err(|e| e.to_string())?;
    let label_rect = Rect::new(
        widget.rect.x,
        widget.rect.y,
        label_surface.width(),
        label_surface.height(),
    );
    canvas
        .copy(&label_texture, None, label_rect)
        .map_err(|e| e.to_string())?;

    // Draw dropdown box
    let dropdown_rect = Rect::new(
        widget.rect.x + 100,
        widget.rect.y,
        widget.rect.width() - 100,
        widget.rect.height(),
    );

    // Dropdown background
    canvas.set_draw_color(Color::RGB(230, 230, 230));
    canvas.fill_rect(dropdown_rect).map_err(|e| e.to_string())?;

    // Dropdown border
    canvas.set_draw_color(Color::RGB(150, 150, 150));
    canvas
        .draw_rect(dropdown_rect.into())
        .map_err(|e| e.to_string())?;

    // Draw current selected value
    let value_surface = font
        .render(&widget.value)
        .blended(Color::RGB(0, 0, 0))
        .map_err(|e| e.to_string())?;
    let value_texture = texture_creator
        .create_texture_from_surface(&value_surface)
        .map_err(|e| e.to_string())?;
    let padding = 5;
    let value_rect = Rect::new(
        dropdown_rect.x + padding,
        dropdown_rect.y + (dropdown_rect.height() as i32 - value_surface.height() as i32) / 2,
        value_surface.width(),
        value_surface.height(),
    );
    canvas
        .copy(&value_texture, None, value_rect)
        .map_err(|e| e.to_string())?;

    // Draw dropdown arrow
    let arrow_size = 10;
    let arrow_x = dropdown_rect.x + dropdown_rect.width() as i32 - arrow_size - 5;
    let arrow_y = dropdown_rect.y + (dropdown_rect.height() as i32 - arrow_size) / 2;

    canvas.set_draw_color(Color::RGB(100, 100, 100));
    // Draw arrow
    canvas
        .draw_line(
            (arrow_x, arrow_y),
            (arrow_x + arrow_size / 2, arrow_y + arrow_size),
        )
        .map_err(|e| e.to_string())?;
    canvas
        .draw_line(
            (arrow_x + arrow_size / 2, arrow_y + arrow_size),
            (arrow_x + arrow_size, arrow_y),
        )
        .map_err(|e| e.to_string())?;

    // 如果下拉框是打开状态，绘制选项列表
    if widget.is_active {
        let options = if widget.label == "Projection Type:" {
            vec!["Perspective", "Orthographic"]
        } else {
            vec![]
        };

        if !options.is_empty() {
            let option_height = 30;
            let list_height = options.len() as u32 * option_height;

            // 绘制选项列表背景
            let list_rect = Rect::new(
                dropdown_rect.x,
                dropdown_rect.y + dropdown_rect.height() as i32,
                dropdown_rect.width(),
                list_height,
            );

            canvas.set_draw_color(Color::RGB(240, 240, 240));
            canvas.fill_rect(list_rect).map_err(|e| e.to_string())?;
            canvas.set_draw_color(Color::RGB(100, 100, 100));
            canvas
                .draw_rect(list_rect.into())
                .map_err(|e| e.to_string())?;

            // 绘制每个选项
            for (i, option) in options.iter().enumerate() {
                let option_rect = Rect::new(
                    list_rect.x,
                    list_rect.y + (i as i32 * option_height as i32),
                    list_rect.width(),
                    option_height,
                );

                // 如果是当前选中的值，高亮显示
                if *option == widget.value {
                    canvas.set_draw_color(Color::RGB(200, 200, 255));
                    canvas.fill_rect(option_rect).map_err(|e| e.to_string())?;
                }

                // 绘制选项文本
                let option_surface = font
                    .render(option)
                    .blended(Color::RGB(0, 0, 0))
                    .map_err(|e| e.to_string())?;
                let option_texture = texture_creator
                    .create_texture_from_surface(&option_surface)
                    .map_err(|e| e.to_string())?;
                let option_text_rect = Rect::new(
                    option_rect.x + padding,
                    option_rect.y
                        + (option_rect.height() as i32 - option_surface.height() as i32) / 2,
                    option_surface.width(),
                    option_surface.height(),
                );

                canvas
                    .copy(&option_texture, None, option_text_rect)
                    .map_err(|e| e.to_string())?;
            }
        }
    }

    Ok(())
}

// Create a dummy rendering preview
fn create_dummy_render(
    texture_creator: &TextureCreator<WindowContext>,
    width: u32,
    height: u32,
) -> Result<Texture, String> {
    // SDL3 surface creation method may differ from SDL2
    let mut surface = Surface::new(
        width,
        height,
        sdl3::pixels::PixelFormat::from_masks(PixelMasks {
            bpp: 32,
            rmask: 0xFF000000,
            gmask: 0x00FF0000,
            bmask: 0x0000FF00,
            amask: 0x000000FF,
        }),
    )
    .map_err(|e| e.to_string())?;

    // In SDL3 you might need a different way to modify pixels
    // This is a simplified method, you might need to adjust according to SDL3 API
    let pitch = width * 4; // RGBA32 format, 4 bytes per pixel
    let data = unsafe { surface.without_lock_mut().unwrap() };

    for y in 0..height {
        for x in 0..width {
            let r = (x * 255 / width) as u8;
            let g = (y * 255 / height) as u8;
            let b = 128u8;

            let offset = (y * pitch + x * 4) as usize;
            if offset + 3 < data.len() {
                data[offset] = r; // R
                data[offset + 1] = g; // G
                data[offset + 2] = b; // B
                data[offset + 3] = 255; // A
            }
        }
    }

    // Create texture
    let texture = texture_creator
        .create_texture_from_surface(&surface)
        .map_err(|e| e.to_string())?;

    Ok(texture)
}

struct DropdownState {
    is_open: bool,
    options: Vec<String>,
}

fn main() -> Result<(), String> {
    let sdl_context = sdl3::init().map_err(|e| e.to_string())?;
    let video_subsystem = sdl_context.video().map_err(|e| e.to_string())?;
    let ttf_context = sdl3::ttf::init().map_err(|e| e.to_string())?;

    let mut text_input_active = false;
    let mut dropdown_open: Option<usize> = None;
    let mut dropdown_options = vec![vec!["Perspective", "Orthographic"]];

    let window = video_subsystem
        .window("Rasterizer SDL3 Demo", 1280, 720)
        .position_centered()
        .opengl()
        .build()
        .map_err(|e| e.to_string())?;

    let mut canvas = window.into_canvas();
    canvas.set_draw_color(Color::RGB(45, 45, 48));

    // Assume using system default monospace font, specify font file path when using
    let font_path = "/nix/store/9xz6bdv0yyzin2qa2pw0w1qy1ygipi8y-noto-64351-tex/fonts/truetype/google/noto/NotoSans-Regular.ttf";
    let font = ttf_context
        .load_font(Path::new(font_path), 14.0)
        .map_err(|e| format!("Could not load font: {}", e))?;

    let texture_creator = canvas.texture_creator();

    // Create preview area
    let preview_width = 640;
    let preview_height = 480;
    let preview_rect = Rect::new(620, 20, preview_width, preview_height);

    // Create sample rendering result
    let preview_texture = create_dummy_render(&texture_creator, preview_width, preview_height)?;

    // Create UI widgets
    let mut widgets = vec![
        // Title label
        Widget::new_label(20, 20, 200, 30, "Rasterizer Control Panel"),
        // Basic rendering settings
        Widget::new_label(20, 70, 200, 20, "Basic Rendering Settings"),
        Widget::new_text_input(20, 100, 350, 30, "OBJ File Path:", "obj/simple/bunny.obj"),
        Widget::new_slider(20, 140, 350, 30, "Width:", 100.0, 2048.0, 1024.0),
        Widget::new_slider(20, 180, 350, 30, "Height:", 100.0, 2048.0, 1024.0),
        Widget::new_dropdown(20, 220, 350, 30, "Projection Type:", "Perspective"),
        Widget::new_checkbox(20, 260, 350, 30, "Disable Z-Buffer", false),
        // Camera settings
        Widget::new_label(20, 310, 200, 20, "Camera Settings"),
        Widget::new_text_input(20, 340, 350, 30, "Camera Position:", "0,0,3"),
        Widget::new_text_input(20, 380, 350, 30, "Camera Target:", "0,0,0"),
        Widget::new_text_input(20, 420, 350, 30, "Camera Up Direction:", "0,1,0"),
        Widget::new_slider(20, 460, 350, 30, "Field of View:", 10.0, 120.0, 45.0),
        // Material settings
        Widget::new_label(20, 510, 200, 20, "Material Settings"),
        Widget::new_checkbox(20, 540, 350, 30, "Use PBR Materials", false),
        Widget::new_slider(20, 580, 350, 30, "Metalness:", 0.0, 1.0, 0.0),
        Widget::new_slider(20, 620, 350, 30, "Roughness:", 0.0, 1.0, 0.5),
        // Render buttons
        Widget::new_button(620, 520, 640, 40, "Start Rendering"),
        Widget::new_button(620, 570, 640, 40, "Save Rendering Result"),
        Widget::new_button(620, 620, 640, 40, "Reset Settings"),
    ];

    // Disable some widgets (as example)
    widgets[16].is_active = false; // Metalness slider
    widgets[17].is_active = false; // Roughness slider

    // 添加下拉框选项
    let mut dropdown_states = vec![DropdownState {
        is_open: false,
        options: vec!["Perspective".to_string(), "Orthographic".to_string()],
    }];

    let mut active_widget: Option<usize> = None;
    let mut active_text_input: Option<usize> = None; // 跟踪当前活动的文本输入框
    let mut event_pump = sdl_context.event_pump().map_err(|e| e.to_string())?;

    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => break 'running,
                Event::KeyDown {
                    keycode: Some(key), ..
                } => {
                    if key == Keycode::Escape {
                        break 'running;
                    }

                    // 处理文本输入
                    if let Some(idx) = active_text_input {
                        if let WidgetType::TextInput = widgets[idx].widget_type {
                            match key {
                                Keycode::Backspace => {
                                    // 删除最后一个字符
                                    if !widgets[idx].value.is_empty() {
                                        widgets[idx].value.pop();
                                    }
                                }
                                _ => {
                                    // 忽略其他控制键
                                }
                            }
                        }
                    }
                }
                Event::TextInput { text, .. } => {
                    // 处理文本输入
                    if let Some(idx) = active_text_input {
                        if let WidgetType::TextInput = widgets[idx].widget_type {
                            widgets[idx].value.push_str(&text);
                        }
                    }
                }
                Event::MouseMotion { x, y, .. } => {
                    // Update hover state
                    for widget in widgets.iter_mut() {
                        widget.is_hovered = widget.contains_point(x as i32, y as i32);
                    }

                    // Update slider dragging
                    if let Some(idx) = active_widget {
                        if let WidgetType::Slider = widgets[idx].widget_type {
                            let slider = &mut widgets[idx];
                            let track_start = slider.rect.x + 100;
                            let track_width = slider.rect.width() - 100;
                            let track_end = track_start + track_width as i32;

                            let x_i32 = x as i32;
                            if x_i32 < track_start {
                                slider.current_value = slider.min_value;
                            } else if x_i32 > track_end {
                                slider.current_value = slider.max_value;
                            } else {
                                let percentage = (x_i32 - track_start) as f32 / track_width as f32;
                                slider.current_value = slider.min_value
                                    + percentage * (slider.max_value - slider.min_value);
                            }
                        }
                    }
                }
                Event::MouseButtonDown { x, y, .. } => {
                    // 先检查是否需要关闭已打开的下拉框
                    let mut close_dropdown = true;
                    let mut click_on_dropdown_options = false;

                    // 检查是否点击了下拉菜单选项
                    if let Some(idx) = active_widget {
                        if let WidgetType::Dropdown = widgets[idx].widget_type {
                            let dropdown_rect = Rect::new(
                                widgets[idx].rect.x + 100,
                                widgets[idx].rect.y,
                                widgets[idx].rect.width() - 100,
                                widgets[idx].rect.height(),
                            );

                            let options = if widgets[idx].label == "Projection Type:" {
                                vec!["Perspective", "Orthographic"]
                            } else {
                                vec![]
                            };

                            if !options.is_empty() {
                                let option_height = 30;
                                let list_rect = Rect::new(
                                    dropdown_rect.x,
                                    dropdown_rect.y + dropdown_rect.height() as i32,
                                    dropdown_rect.width(),
                                    options.len() as u32 * option_height,
                                );

                                // 检查点击是否在下拉选项区域内
                                if x >= list_rect.x as f32
                                    && x < (list_rect.x as f32 + list_rect.width() as f32)
                                    && y >= list_rect.y as f32
                                    && y < (list_rect.y as f32 + list_rect.height() as f32)
                                {
                                    // 计算点击了哪个选项
                                    let option_idx =
                                        ((y - list_rect.y as f32) / option_height as f32) as usize;
                                    if option_idx < options.len() {
                                        widgets[idx].value = options[option_idx].to_string();
                                        widgets[idx].is_active = false; // 选择后关闭下拉框
                                        active_widget = None;
                                    }
                                    click_on_dropdown_options = true;
                                    close_dropdown = false;
                                }
                            }
                        }
                    }

                    // 如果点击在下拉选项上，跳过其他处理
                    if click_on_dropdown_options {
                        continue;
                    }

                    // 取消激活所有文本框
                    if active_text_input.is_some() {
                        if let Some(idx) = active_text_input {
                            widgets[idx].is_active = false;
                        }
                        active_text_input = None;
                    }

                    // 如果需要关闭下拉框，并且之前有打开的下拉框
                    if close_dropdown && active_widget.is_some() {
                        if let Some(idx) = active_widget {
                            if let WidgetType::Dropdown = widgets[idx].widget_type {
                                widgets[idx].is_active = false;
                                active_widget = None;
                            }
                        }
                    }

                    // Activate clicked widget and store PBR state
                    let mut idx_to_activate = None;
                    let mut pbr_checked = false;
                    let mut pbr_checkbox_clicked = false;
                    let mut reset_button_clicked = false;

                    // First pass: find which widget was clicked
                    for (i, widget) in widgets.iter_mut().enumerate() {
                        if widget.contains_point(x as i32, y as i32) {
                            // 特殊处理文本输入框
                            if let WidgetType::TextInput = widget.widget_type {
                                widget.is_active = true;
                                active_text_input = Some(i);
                            }
                            // 特殊处理下拉框
                            else if let WidgetType::Dropdown = widget.widget_type {
                                widget.is_active = !widget.is_active; // 切换下拉框打开/关闭状态
                                idx_to_activate = Some(i);
                            }
                            // 处理其他类型的控件
                            else {
                                widget.is_active = true;
                                idx_to_activate = Some(i);

                                // Handle checkbox click
                                if let WidgetType::Checkbox = widget.widget_type {
                                    widget.is_checked = !widget.is_checked;

                                    // If PBR checkbox was toggled
                                    if widget.label == "Use PBR Materials" {
                                        println!(
                                            "PBR Material checkbox state: {}",
                                            widget.is_checked
                                        );
                                        pbr_checked = widget.is_checked;
                                        pbr_checkbox_clicked = true;
                                    }
                                }

                                // Handle button click
                                if let WidgetType::Button = widget.widget_type {
                                    println!("Button clicked: {}", widget.label);

                                    // Check if reset button was clicked
                                    if widget.label == "Reset Settings" {
                                        reset_button_clicked = true;
                                    }
                                }
                            }
                            break;
                        }
                    }

                    // Update active widget
                    if idx_to_activate.is_some() {
                        active_widget = idx_to_activate;
                    }

                    // Second pass: update dependent widgets if needed
                    if pbr_checkbox_clicked {
                        widgets[16].is_active = pbr_checked;
                        widgets[17].is_active = pbr_checked;
                    }

                    // Handle reset button action
                    if reset_button_clicked {
                        for w in widgets.iter_mut() {
                            if let WidgetType::Slider = w.widget_type {
                                if w.label == "Width:" || w.label == "Height:" {
                                    w.current_value = 1024.0;
                                } else if w.label == "Field of View:" {
                                    w.current_value = 45.0;
                                } else if w.label == "Metalness:" {
                                    w.current_value = 0.0;
                                } else if w.label == "Roughness:" {
                                    w.current_value = 0.5;
                                }
                            }
                        }
                    }
                }
                Event::MouseButtonUp { .. } => {
                    // Deactivate state
                    if let Some(idx) = active_widget {
                        if let WidgetType::Button = widgets[idx].widget_type {
                            widgets[idx].is_active = false;
                        }
                    }
                    active_widget = None;
                }
                _ => {}
            }
        }

        // Clear screen
        canvas.clear();

        // Draw preview area background
        canvas.set_draw_color(Color::RGB(30, 30, 30));
        canvas.fill_rect(preview_rect).map_err(|e| e.to_string())?;

        // Draw preview
        canvas
            .copy(&preview_texture, None, preview_rect)
            .map_err(|e| e.to_string())?;

        // Draw preview area border
        canvas.set_draw_color(Color::RGB(150, 150, 150));
        canvas
            .draw_rect(preview_rect.into())
            .map_err(|e| e.to_string())?;

        // Draw control panel background
        let panel_rect = Rect::new(10, 10, 590, 700);
        canvas.set_draw_color(Color::RGB(60, 60, 60));
        canvas.fill_rect(panel_rect).map_err(|e| e.to_string())?;
        canvas.set_draw_color(Color::RGB(100, 100, 100));
        canvas
            .draw_rect(panel_rect.into())
            .map_err(|e| e.to_string())?;

        // Draw separator lines
        canvas.set_draw_color(Color::RGB(100, 100, 100));
        canvas
            .draw_line((20, 60), (580, 60))
            .map_err(|e| e.to_string())?;
        canvas
            .draw_line((20, 300), (580, 300))
            .map_err(|e| e.to_string())?;
        canvas
            .draw_line((20, 500), (580, 500))
            .map_err(|e| e.to_string())?;

        // Draw all widgets
        for widget in &widgets {
            render_widget(&mut canvas, &font, &texture_creator, widget)?;
        }

        // Update screen
        canvas.present();

        // Limit frame rate
        std::thread::sleep(Duration::from_millis(16));
    }

    Ok(())
}
