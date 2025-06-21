use super::triangle_data::{TextureSource, TriangleData};
use crate::geometry::interpolation::{
    interpolate_normal, interpolate_position, interpolate_texcoords,
};
use crate::io::render_settings::RenderSettings;
use crate::material_system::color::Color;
use nalgebra::{Point3, Vector3};

/// 统一的像素颜色计算
#[allow(clippy::too_many_arguments)]
pub fn calculate_pixel_color(
    triangle: &TriangleData,
    bary: Vector3<f32>,
    settings: &RenderSettings,
    use_phong_or_pbr: bool,
    use_texture: bool,
    ambient_contribution: &Color,
) -> Color {
    if use_phong_or_pbr {
        calculate_advanced_shading(triangle, bary, settings, use_texture, ambient_contribution)
    } else {
        calculate_basic_shading(triangle, bary, settings, use_texture, ambient_contribution)
    }
}

fn calculate_advanced_shading(
    triangle: &TriangleData,
    bary: Vector3<f32>,
    settings: &RenderSettings,
    use_texture: bool,
    ambient_contribution: &Color,
) -> Color {
    let material_view = match &triangle.material_view {
        Some(mat_view) => mat_view,
        None => {
            return calculate_basic_shading(
                triangle,
                bary,
                settings,
                use_texture,
                ambient_contribution,
            );
        }
    };

    let interp_normal = interpolate_normal(
        bary,
        triangle.vertices[0].normal_view.unwrap(),
        triangle.vertices[1].normal_view.unwrap(),
        triangle.vertices[2].normal_view.unwrap(),
        triangle.is_perspective,
        triangle.vertices[0].z_view,
        triangle.vertices[1].z_view,
        triangle.vertices[2].z_view,
    );

    let interp_position = interpolate_position(
        bary,
        triangle.vertices[0].position_view.unwrap(),
        triangle.vertices[1].position_view.unwrap(),
        triangle.vertices[2].position_view.unwrap(),
        triangle.is_perspective,
        triangle.vertices[0].z_view,
        triangle.vertices[1].z_view,
        triangle.vertices[2].z_view,
    );

    let view_dir = (-interp_position.coords).normalize();
    let ao_factor = calculate_enhanced_ao(triangle, bary, &interp_normal, settings);

    let mut total_direct_light = Vector3::zeros();

    for light in triangle.lights {
        let light_dir = light.get_direction(&interp_position);
        let light_intensity = light.get_intensity(&interp_position);

        let shadow_factor = calculate_simple_shadow_factor(
            &light_dir,
            &interp_normal,
            triangle,
            &interp_position,
            settings,
        );

        let response = material_view.compute_response(&light_dir, &view_dir, &interp_normal);

        total_direct_light += Vector3::new(
            response.x * light_intensity.x * shadow_factor,
            response.y * light_intensity.y * shadow_factor,
            response.z * light_intensity.z * shadow_factor,
        );
    }

    let direct_light = Color::new(
        total_direct_light.x,
        total_direct_light.y,
        total_direct_light.z,
    );
    let ao_ambient = apply_ao_to_ambient(ambient_contribution, ao_factor);

    let surface_color = if use_texture {
        sample_texture(triangle, bary)
    } else {
        triangle.base_color
    };

    if settings.use_lighting {
        surface_color.component_mul(&(direct_light + ao_ambient))
    } else {
        surface_color.component_mul(&ao_ambient)
    }
}

fn calculate_basic_shading(
    triangle: &TriangleData,
    bary: Vector3<f32>,
    settings: &RenderSettings,
    use_texture: bool,
    ambient_contribution: &Color,
) -> Color {
    let surface_color = if use_texture {
        sample_texture(triangle, bary)
    } else {
        triangle.base_color
    };

    if settings.use_lighting {
        let ao_factor = if settings.enhanced_ao {
            calculate_simple_ao_factor(triangle, bary, settings)
        } else {
            1.0
        };

        let ao_ambient = apply_ao_to_ambient(ambient_contribution, ao_factor);
        surface_color.component_mul(&ao_ambient)
    } else {
        surface_color
    }
}

// ===== 纹理采样功能 =====
fn sample_texture(triangle: &TriangleData, bary: Vector3<f32>) -> Color {
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

// ===== 光照效果功能 =====
pub fn calculate_ambient_contribution(triangle: &TriangleData) -> Color {
    let ambient_color = triangle.ambient_color;
    let ambient_intensity = triangle.ambient_intensity;

    let ambient = Color::new(
        ambient_color.x * ambient_intensity,
        ambient_color.y * ambient_intensity,
        ambient_color.z * ambient_intensity,
    );

    if let Some(material_view) = &triangle.material_view {
        let material = match material_view {
            crate::material_system::materials::MaterialView::BlinnPhong(material) => material,
            crate::material_system::materials::MaterialView::PBR(material) => material,
        };

        return Color::new(
            material.ambient_factor.x * ambient.x,
            material.ambient_factor.y * ambient.y,
            material.ambient_factor.z * ambient.z,
        );
    }

    ambient
}

fn apply_ao_to_ambient(ambient: &Color, ao_factor: f32) -> Color {
    Color::new(
        ambient.x * ao_factor,
        ambient.y * ao_factor,
        ambient.z * ao_factor,
    )
}

fn calculate_enhanced_ao(
    triangle: &TriangleData,
    bary: Vector3<f32>,
    interp_normal: &Vector3<f32>,
    settings: &RenderSettings,
) -> f32 {
    if !settings.enhanced_ao {
        return 1.0;
    }

    let up_factor = {
        let raw_up = (interp_normal.y + 1.0) * 0.5;
        raw_up.powf(1.5)
    };

    let edge_proximity = {
        let min_bary = bary.x.min(bary.y).min(bary.z);
        let edge_factor = (min_bary * 2.0).min(1.0);
        0.6 + 0.4 * edge_factor
    };

    if let (Some(n0), Some(n1), Some(n2)) = (
        triangle.vertices[0].normal_view,
        triangle.vertices[1].normal_view,
        triangle.vertices[2].normal_view,
    ) {
        let normal_variance = (n0 - n1).magnitude() + (n1 - n2).magnitude() + (n2 - n0).magnitude();
        let curvature_factor = (1.0 - (normal_variance * 0.4).min(0.7)).max(0.1);

        let center_distance = (bary - Vector3::new(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)).magnitude();
        let position_factor = 1.0 - (center_distance * 0.5).min(0.3);

        let base_ao = (up_factor * 0.5
            + curvature_factor * 0.3
            + edge_proximity * 0.15
            + position_factor * 0.05)
            .clamp(0.05, 1.0);

        let enhanced_strength = settings.ao_strength * 1.5;
        let final_ao = 1.0 - ((1.0 - base_ao) * enhanced_strength.min(1.0));
        final_ao.clamp(0.05, 1.0)
    } else {
        let base_ao = (up_factor * 0.7 + edge_proximity * 0.3).clamp(0.3, 1.0);
        let enhanced_strength = settings.ao_strength * 1.2;
        1.0 - ((1.0 - base_ao) * enhanced_strength.min(1.0))
    }
}

fn calculate_simple_shadow_factor(
    light_dir: &Vector3<f32>,
    surface_normal: &Vector3<f32>,
    triangle: &TriangleData,
    interp_position: &Point3<f32>,
    settings: &RenderSettings,
) -> f32 {
    if !settings.soft_shadows {
        return 1.0;
    }

    let ndl = surface_normal.dot(light_dir).max(0.0);

    let edge_factor = if ndl < 0.3 {
        (ndl / 0.3).powf(0.7)
    } else {
        1.0
    };

    let depth_factor = if interp_position.z < -2.0 {
        0.8 + 0.2 * ((-interp_position.z - 2.0) / 8.0).min(1.0)
    } else {
        1.0
    };

    let local_occlusion = if let (Some(n0), Some(n1), Some(n2)) = (
        triangle.vertices[0].normal_view,
        triangle.vertices[1].normal_view,
        triangle.vertices[2].normal_view,
    ) {
        let normal_variance = (n0 - n1).magnitude() + (n1 - n2).magnitude() + (n2 - n0).magnitude();
        let occlusion_strength = (normal_variance * 0.3).min(0.4);
        1.0 - occlusion_strength
    } else {
        1.0
    };

    let base_shadow = edge_factor * depth_factor * local_occlusion;
    let final_shadow = 1.0 - ((1.0 - base_shadow) * settings.shadow_strength);
    final_shadow.clamp(0.1, 1.0)
}

fn calculate_simple_ao_factor(
    triangle: &TriangleData,
    bary: Vector3<f32>,
    settings: &RenderSettings,
) -> f32 {
    if let (Some(n0), Some(n1), Some(n2)) = (
        triangle.vertices[0].normal_view,
        triangle.vertices[1].normal_view,
        triangle.vertices[2].normal_view,
    ) {
        let interp_normal = interpolate_normal(
            bary,
            n0,
            n1,
            n2,
            triangle.is_perspective,
            triangle.vertices[0].z_view,
            triangle.vertices[1].z_view,
            triangle.vertices[2].z_view,
        );
        calculate_enhanced_ao(triangle, bary, &interp_normal, settings)
    } else {
        1.0
    }
}
