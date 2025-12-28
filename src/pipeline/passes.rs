use crate::core::color::{aces_tone_mapping, linear_to_srgb};
use crate::core::framebuffer::FrameBuffer;
use crate::core::math::transform::TransformFactory;
use crate::io::config::Config;
use crate::pipeline::renderer::{ClearOptions, Renderer};
use crate::pipeline::shaders::pbr::PbrShader;
use crate::pipeline::shaders::shadow::ShadowShader;
use crate::scene::context::RenderContext;
use crate::scene::texture::Texture;
use nalgebra::{Matrix4, Point3, Vector3};
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::atomic::Ordering;

/// Executes the Shadow Mapping Pass.
pub fn render_shadow_pass(
    config: &Config,
    context: &RenderContext,
    shadow_renderer: &mut Renderer,
) -> (Option<Arc<Vec<f32>>>, Matrix4<f32>) {
    if !config.render.use_shadows {
        return (None, Matrix4::identity());
    }

    let light_target = Point3::new(0.0, 0.0, 0.0);
    let light_dir = (light_target - context.shadow_light_pos).normalize();
    let light_up = if light_dir.y.abs() > 0.9 {
        Vector3::z()
    } else {
        Vector3::y()
    };

    let light_view = TransformFactory::view(&context.shadow_light_pos, &light_target, &light_up);
    let ortho_size = config.render.shadow_ortho_size;
    let light_proj =
        TransformFactory::orthographic(-ortho_size, ortho_size, -ortho_size, ortho_size, 0.1, 50.0);
    let light_space_matrix = light_proj * light_view;

    shadow_renderer.clear_with_options(ClearOptions {
        depth: f32::INFINITY,
        ..Default::default()
    });

    for obj in &context.scene_objects {
        let shader = ShadowShader::new(obj.transform, light_view, light_proj);
        shadow_renderer.draw_model(&obj.model, &shader);
    }

    let shadow_depth_data: Vec<f32> = shadow_renderer
        .framebuffer
        .depth_buffer
        .iter()
        .map(|atomic| f32::from_bits(atomic.load(Ordering::Relaxed)))
        .collect();

    (Some(Arc::new(shadow_depth_data)), light_space_matrix)
}

/// Executes the Main Rendering Pass.
pub fn render_main_pass(
    config: &Config,
    context: &RenderContext,
    renderer: &mut Renderer,
    shadow_map: Option<Arc<Vec<f32>>>,
    light_space_matrix: Matrix4<f32>,
) {
    let bg_texture = if let Some(path) = &config.render.background_image {
        Texture::load(path).ok()
    } else {
        None
    };

    let (gradient, color) = if let Some(c) = config.render.background_color {
        (None, Vector3::from(c))
    } else if let (Some(top), Some(bottom)) = (
        config.render.background_gradient_top,
        config.render.background_gradient_bottom,
    ) {
        (
            Some((Vector3::from(top), Vector3::from(bottom))),
            Vector3::zeros(),
        )
    } else {
        (None, Vector3::zeros())
    };

    renderer.clear_with_options(ClearOptions {
        color,
        gradient,
        texture: bg_texture.as_ref(),
        depth: f32::INFINITY,
    });

    let ambient_light = Vector3::from(config.render.ambient_light);

    for obj in &context.scene_objects {
        let mut shader = PbrShader::new(
            obj.transform,
            context.camera.view_matrix(),
            context.camera.projection_matrix(),
            context.camera.position,
        );

        shader.lights = context.lights.clone();
        shader.ambient_light = ambient_light;
        shader.shadow_map = shadow_map.clone();
        shader.shadow_map_size = config.render.shadow_map_size;
        shader.light_space_matrix = light_space_matrix;
        shader.shadow_bias = config.render.shadow_bias;
        shader.use_pcf = config.render.use_pcf;
        shader.pcf_kernel_size = config.render.pcf_kernel_size;

        renderer.draw_model(&obj.model, &shader);
    }
}

/// Post-processing: Tone Mapping -> Gamma Correction -> u32 Buffer.
pub fn post_process_to_buffer(framebuffer: &FrameBuffer, buffer: &mut [u32], config: &Config) {
    buffer
        .par_chunks_mut(framebuffer.width)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, pixel) in row.iter_mut().enumerate() {
                if let Some(color) = framebuffer.get_pixel(x, y) {
                    let exposed = color * config.render.exposure;
                    let mapped = if config.render.use_aces {
                        aces_tone_mapping(exposed)
                    } else {
                        exposed
                    };
                    let srgb = linear_to_srgb(mapped);

                    let r = (srgb.x.clamp(0.0, 1.0) * 255.0) as u32;
                    let g = (srgb.y.clamp(0.0, 1.0) * 255.0) as u32;
                    let b = (srgb.z.clamp(0.0, 1.0) * 255.0) as u32;

                    *pixel = (255 << 24) | (r << 16) | (g << 8) | b;
                } else {
                    *pixel = 0;
                }
            }
        });
}
