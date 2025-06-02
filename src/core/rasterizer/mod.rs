// 重新导出主要组件
pub mod msaa;
pub mod pixel_processor;
pub mod shading;
pub mod triangle_data;

// 重新导出关键类型和函数
pub use triangle_data::{TextureSource, TriangleData, VertexRenderData};
