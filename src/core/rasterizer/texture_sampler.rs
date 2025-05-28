use super::triangle_data::{TextureSource, TriangleData};
use crate::geometry::interpolation::interpolate_texcoords;
use crate::material_system::color::Color;
use nalgebra::Vector3;

pub fn sample_texture(triangle: &TriangleData, bary: Vector3<f32>) -> Color {
    match &triangle.texture_source {
        TextureSource::Image(tex) => {
            if let (Some(tc1), Some(tc2), Some(tc3)) = (
                triangle.vertices[0].texcoord,
                triangle.vertices[1].texcoord,
                triangle.vertices[2].texcoord,
            ) {
                let tc = interpolate_texcoords(
                    bary,
                    tc1,
                    tc2,
                    tc3,
                    triangle.vertices[0].z_view,
                    triangle.vertices[1].z_view,
                    triangle.vertices[2].z_view,
                    triangle.is_perspective,
                );

                let color_array = tex.sample(tc.x, tc.y);
                Color::new(color_array[0], color_array[1], color_array[2])
            } else {
                Color::new(1.0, 1.0, 1.0)
            }
        }
        TextureSource::FaceColor(seed) => {
            let color = crate::material_system::color::get_random_color(*seed, true);
            Color::new(color.x, color.y, color.z)
        }
        TextureSource::SolidColor(color) => Color::new(color.x, color.y, color.z),
        TextureSource::None => Color::new(1.0, 1.0, 1.0),
    }
}
