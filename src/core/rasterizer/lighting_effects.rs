use super::triangle_data::TriangleData;
use crate::io::render_settings::RenderSettings;
use crate::material_system::color::Color;
use crate::material_system::materials::MaterialView;
use nalgebra::{Point3, Vector3};

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
            MaterialView::BlinnPhong(material) => material,
            MaterialView::PBR(material) => material,
        };

        return Color::new(
            material.ambient_factor.x * ambient.x,
            material.ambient_factor.y * ambient.y,
            material.ambient_factor.z * ambient.z,
        );
    }

    ambient
}

pub fn apply_ao_to_ambient(ambient: &Color, ao_factor: f32) -> Color {
    Color::new(
        ambient.x * ao_factor,
        ambient.y * ao_factor,
        ambient.z * ao_factor,
    )
}

pub fn calculate_enhanced_ao(
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

pub fn calculate_simple_shadow_factor(
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
