use crate::material_system::color::Color;
use crate::material_system::light::Light;
use crate::material_system::materials::MaterialView;
use crate::material_system::texture::Texture;
use nalgebra::{Point2, Point3, Vector2, Vector3};

/// 顶点渲染数据
#[derive(Debug, Clone)]
pub struct VertexRenderData {
    pub pix: Point2<f32>,
    pub z_view: f32,
    pub texcoord: Option<Vector2<f32>>,
    pub normal_view: Option<Vector3<f32>>,
    pub position_view: Option<Point3<f32>>,
}

/// 纹理来源枚举
#[derive(Debug, Clone)]
pub enum TextureSource<'a> {
    None,
    Image(&'a Texture),
    FaceColor(u64),
    SolidColor(Vector3<f32>),
}

/// 三角形光栅化数据
pub struct TriangleData<'a> {
    pub vertices: [VertexRenderData; 3],
    pub base_color: Color,
    pub texture_source: TextureSource<'a>,
    pub material_view: Option<MaterialView<'a>>,
    pub lights: &'a [Light],
    pub ambient_intensity: f32,
    pub ambient_color: Vector3<f32>,
    pub is_perspective: bool,
}

/// 屏幕空间包围盒
#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    pub min_x: usize,
    pub min_y: usize,
    pub max_x: usize,
    pub max_y: usize,
}

impl BoundingBox {
    pub fn from_triangle(triangle: &TriangleData, width: usize, height: usize) -> Option<Self> {
        let v0 = &triangle.vertices[0].pix;
        let v1 = &triangle.vertices[1].pix;
        let v2 = &triangle.vertices[2].pix;

        let min_x = v0.x.min(v1.x).min(v2.x).floor().max(0.0) as usize;
        let min_y = v0.y.min(v1.y).min(v2.y).floor().max(0.0) as usize;
        let max_x = v0.x.max(v1.x).max(v2.x).ceil().min(width as f32) as usize;
        let max_y = v0.y.max(v1.y).max(v2.y).ceil().min(height as f32) as usize;

        if max_x <= min_x || max_y <= min_y {
            None
        } else {
            Some(Self {
                min_x,
                min_y,
                max_x,
                max_y,
            })
        }
    }

    pub fn for_each_pixel<F>(&self, mut callback: F)
    where
        F: FnMut(usize, usize),
    {
        for y in self.min_y..self.max_y {
            for x in self.min_x..self.max_x {
                callback(x, y);
            }
        }
    }
}

impl<'a> TriangleData<'a> {
    pub fn is_valid(&self) -> bool {
        let v0 = &self.vertices[0].pix;
        let v1 = &self.vertices[1].pix;
        let v2 = &self.vertices[2].pix;

        let area = 0.5 * ((v1.x - v0.x) * (v2.y - v0.y) - (v2.x - v0.x) * (v1.y - v0.y)).abs();
        area > 1e-6
    }
}
