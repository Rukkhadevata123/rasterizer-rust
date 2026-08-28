use crate::core::color::{aces_tone_mapping, linear_to_srgb};
use crate::core::framebuffer::FrameBuffer;
use crate::core::geometry::Vertex;
use crate::core::math::transform::TransformFactory;
use crate::core::rasterizer::{BlendMode, CullMode, RenderState};
use crate::error::AssetError;
use crate::io::config::Config;
use crate::pipeline::renderer::{ClearOptions, RenderGeometry, RenderQueue, Renderer};
use crate::pipeline::shaders::pbr::PbrShader;
use crate::pipeline::shaders::shadow::ShadowShader;
use crate::scene::context::RenderContext;
use crate::scene::material::{AlphaMode, Material};
use crate::scene::texture::{
    MinFilter, SamplerState, TexCoordSet, TextureBinding, TextureImage, TextureUsage,
};
use nalgebra::{Matrix4, Point3, Vector3, Vector4};
use rayon::prelude::*;
use std::sync::Arc;

pub struct ShadowPassOutput {
    pub depth: Option<Arc<Vec<f32>>>,
    pub size: usize,
    pub light_space_matrix: Matrix4<f32>,
    pub light_index: Option<usize>,
}

impl ShadowPassOutput {
    fn disabled() -> Self {
        Self {
            depth: None,
            size: 0,
            light_space_matrix: Matrix4::identity(),
            light_index: None,
        }
    }
}

pub fn render_shadow_pass(
    config: &Config,
    context: &RenderContext,
    shadow_renderer: &mut Renderer,
) -> ShadowPassOutput {
    if !config.render.use_shadows {
        return ShadowPassOutput::disabled();
    }

    let Some(shadow_light) = context.shadow_light else {
        return ShadowPassOutput::disabled();
    };

    let light_target = Point3::origin();
    let light_dir = (light_target - shadow_light.position).normalize();
    let light_up = if light_dir.y.abs() > 0.9 {
        Vector3::z()
    } else {
        Vector3::y()
    };

    let light_view = TransformFactory::view(&shadow_light.position, &light_target, &light_up);
    let ortho_size = config.render.shadow_ortho_size;
    let light_projection =
        TransformFactory::orthographic(-ortho_size, ortho_size, -ortho_size, ortho_size, 0.1, 50.0);
    let light_space_matrix = light_projection * light_view;

    shadow_renderer.clear_with_options(ClearOptions {
        depth: f32::INFINITY,
        ..Default::default()
    });
    let shadow_state = RenderState::default();
    let shaders: Vec<ShadowShader> = context
        .scene_objects
        .iter()
        .map(|object| ShadowShader::new(object.transform, light_view, light_projection))
        .collect();
    let mut shadow_queue = RenderQueue::default();

    for (shader_index, object) in context.scene_objects.iter().enumerate() {
        for mesh in &object.model.meshes {
            let material = object.model.materials.get(mesh.material_id);
            let pbr_material = material.map(|material| match material {
                Material::Pbr(material) => material,
            });
            if matches!(pbr_material, Some(material) if material.alpha_mode == AlphaMode::Blend) {
                continue;
            }
            let command_state = RenderState {
                cull_mode: if pbr_material.is_some_and(|material| material.double_sided) {
                    CullMode::None
                } else {
                    shadow_state.cull_mode
                },
                ..shadow_state
            };
            shadow_queue.push(
                shader_index,
                RenderGeometry::Mesh(mesh),
                material,
                command_state,
                0.0,
            );
        }
    }
    shadow_renderer.draw_queue(&shadow_queue, &shaders);

    ShadowPassOutput {
        depth: Some(Arc::new(shadow_renderer.framebuffer.depth_values())),
        size: shadow_renderer.framebuffer.width,
        light_space_matrix,
        light_index: Some(shadow_light.light_index),
    }
}
/// Executes the Main Rendering Pass.
pub fn render_main_pass(
    config: &Config,
    context: &RenderContext,
    renderer: &mut Renderer,
    shadow: &ShadowPassOutput,
    state: RenderState,
) -> Result<(), AssetError> {
    let bg_texture = if let Some(path) = &config.render.background_image {
        let background_path = config.resolve_path(path);
        let image =
            TextureImage::load(&background_path, config.render.use_mipmap).map_err(|source| {
                AssetError::BackgroundImage {
                    path: background_path,
                    source,
                }
            })?;
        Some(TextureBinding::new(
            Arc::new(image),
            SamplerState {
                min_filter: MinFilter::LinearMipmapLinear,
                ..Default::default()
            },
            TexCoordSet::TexCoord0,
            TextureUsage::Color,
        ))
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

    // Helper to create configured PBR shader
    let create_pbr_shader = |model: Matrix4<f32>| -> PbrShader {
        let mut shader = PbrShader::new(
            model,
            context.camera.view_matrix(),
            context.camera.projection_matrix(),
            context.camera.position,
        );

        shader.lights = context.lights.clone();
        shader.ambient_light = ambient_light;
        shader.shadow_map = shadow.depth.clone();
        shader.shadow_map_size = shadow.size;
        shader.shadow_light_index = shadow.light_index;
        shader.light_space_matrix = shadow.light_space_matrix;
        shader.shadow_bias = config.render.shadow_bias;
        shader.use_pcf = config.render.use_pcf;
        shader.pcf_kernel_size = config.render.pcf_kernel_size;
        shader
    };

    let opaque_state = RenderState {
        blend_mode: BlendMode::Opaque,
        depth_write: true,
        ..state
    };
    let transparent_state = RenderState {
        blend_mode: BlendMode::Alpha,
        depth_write: false,
        ..state
    };
    let mut shaders: Vec<PbrShader> = context
        .scene_objects
        .iter()
        .map(|object| create_pbr_shader(object.transform))
        .collect();
    let transparent_shader_index = shaders.len();
    shaders.push(create_pbr_shader(Matrix4::identity()));
    let mut opaque_queue = RenderQueue::default();
    let mut masked_queue = RenderQueue::default();
    let mut transparent_queue = RenderQueue::default();

    for (shader_index, obj) in context.scene_objects.iter().enumerate() {
        for mesh in &obj.model.meshes {
            let material = if mesh.material_id < obj.model.materials.len() {
                Some(&obj.model.materials[mesh.material_id])
            } else {
                None
            };

            let pbr_material = material.map(|material| match material {
                Material::Pbr(material) => material,
            });
            let alpha_mode = pbr_material.map_or(AlphaMode::Opaque, |material| material.alpha_mode);
            let command_state = |state: RenderState| RenderState {
                cull_mode: if pbr_material.is_some_and(|material| material.double_sided) {
                    CullMode::None
                } else {
                    state.cull_mode
                },
                ..state
            };

            if alpha_mode == AlphaMode::Blend {
                let model_matrix = obj.transform;
                let view_matrix = context.camera.view_matrix();

                // Calculate Normal Matrix for transforming normals/tangents correctly
                let model_3x3 = model_matrix.fixed_view::<3, 3>(0, 0).into_owned();
                let normal_matrix = model_3x3.try_inverse().unwrap_or(model_3x3).transpose();

                // Pre-transform all vertices to World Space
                let transform_vertex = |v: &Vertex| -> Vertex {
                    let pos_world = model_matrix.transform_point(&v.position);
                    let n_world = (normal_matrix * v.normal).normalize();
                    let t_xyz_local = Vector3::new(v.tangent.x, v.tangent.y, v.tangent.z);
                    let t_xyz_world = (normal_matrix * t_xyz_local).normalize();
                    let t_world =
                        Vector4::new(t_xyz_world.x, t_xyz_world.y, t_xyz_world.z, v.tangent.w);

                    let mut new_v = *v;
                    new_v.position = pos_world;
                    new_v.normal = n_world;
                    new_v.tangent = t_world;
                    new_v
                };

                // Use Rayon for parallel vertex transformation
                let world_vertices: Vec<Vertex> =
                    mesh.vertices.par_iter().map(transform_vertex).collect();

                for chunk in mesh.indices.chunks(3) {
                    if chunk.len() < 3 {
                        continue;
                    }
                    // Use transformed vertices directly
                    let v0_world = world_vertices[chunk[0] as usize];
                    let v1_world = world_vertices[chunk[1] as usize];
                    let v2_world = world_vertices[chunk[2] as usize];

                    if let Some(mat) = material {
                        // Calculate Z in View Space using the centroid of World Space vertices
                        let centroid_world = (v0_world.position.coords
                            + v1_world.position.coords
                            + v2_world.position.coords)
                            / 3.0;
                        let centroid_view =
                            view_matrix * Point3::from(centroid_world).to_homogeneous();

                        transparent_queue.push(
                            transparent_shader_index,
                            RenderGeometry::Triangle([v0_world, v1_world, v2_world]),
                            Some(mat),
                            command_state(transparent_state),
                            centroid_view.z,
                        );
                    }
                }
            } else if matches!(alpha_mode, AlphaMode::Mask(_)) {
                masked_queue.push(
                    shader_index,
                    RenderGeometry::Mesh(mesh),
                    material,
                    command_state(opaque_state),
                    0.0,
                );
            } else {
                opaque_queue.push(
                    shader_index,
                    RenderGeometry::Mesh(mesh),
                    material,
                    command_state(opaque_state),
                    0.0,
                );
            }
        }
    }

    renderer.draw_queues(&[&opaque_queue, &masked_queue], &shaders);
    transparent_queue.sort_transparent();
    renderer.draw_queue(&transparent_queue, &shaders);

    Ok(())
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
