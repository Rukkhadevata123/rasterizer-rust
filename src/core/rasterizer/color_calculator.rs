use super::lighting_effects::{
    apply_ao_to_ambient, calculate_enhanced_ao, calculate_simple_shadow_factor,
};
use super::texture_sampler::sample_texture;
use super::triangle_data::TriangleData;
use crate::geometry::interpolation::{interpolate_normal, interpolate_position};
use crate::io::render_settings::RenderSettings;
use crate::material_system::color::Color;
use nalgebra::Vector3;

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
