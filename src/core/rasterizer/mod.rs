// 重新导出主要组件
pub mod pixel_processor;
pub mod shading;
pub mod triangle_data;

// 重新导出关键类型和函数
pub use pixel_processor::{rasterize_pixel, rasterize_triangle};
pub use triangle_data::{TextureSource, TriangleData, VertexRenderData};
