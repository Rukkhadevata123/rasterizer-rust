//! # 三角形光栅化模块
//!
//! 高效的软件光栅化器，支持多种着色模型和并行渲染

pub mod color_calculator;
pub mod lighting_effects;
pub mod pixel_processor;
pub mod texture_sampler;
pub mod triangle_data;

// 重新导出主要类型和函数
pub use pixel_processor::{rasterize_pixel, rasterize_triangle};
pub use triangle_data::{TextureSource, TriangleData, VertexRenderData};
