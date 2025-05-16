// UI模块主文件
// 声明子模块
pub mod animation;
pub mod app;
pub mod render;
pub mod widgets;

// 为了方便直接调用，从app模块导出启动函数
pub use app::start_gui;
