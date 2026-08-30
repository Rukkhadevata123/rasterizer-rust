use crate::core::color::{aces_tone_mapping, linear_to_srgb};
use crate::core::math::transform::{TangentFrameTransform, TransformFactory};
use crate::core::pipeline_state::{
    BlendState, ColorTargetState, CullMode, DepthStencilState, GraphicsPipelineState,
    PrimitiveState,
};
use crate::error::AssetError;
use crate::io::config::Config;
use crate::pipeline::renderer::{
    ClearOptions, FrameResources, RenderGeometry, RenderPhase, RenderTarget, Renderer,
};
use crate::pipeline::shaders::pbr::PbrShader;
use crate::pipeline::shaders::shadow::ShadowShader;
use crate::scene::context::RenderScene;
use crate::scene::material::{AlphaMode, Material};
use nalgebra::{Matrix4, Point3, Vector3, Vector4};
use rayon::prelude::*;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub struct ShadowPassOutput {
    pub depth: Option<Arc<Vec<f32>>>,
    pub size: usize,
    pub light_space_matrix: Matrix4<f32>,
    pub light_index: Option<usize>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ShadowPassTimings {
    pub pass_setup: Duration,
    pub recording: Duration,
    pub attachment_processing: Duration,
    pub backend_preparation: Duration,
    pub rasterization: Duration,
    pub submission_total: Duration,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MainPassTimings {
    pub pass_setup: Duration,
    pub recording: Duration,
    pub attachment_processing: Duration,
    pub backend_preparation: Duration,
    pub opaque_masked_rasterization: Duration,
    pub transparent_rasterization: Duration,
    pub submission_total: Duration,
}

struct ShadowCamera {
    view: Matrix4<f32>,
    projection: Matrix4<f32>,
}

fn configured_pbr_shader<'resources>(
    config: &Config,
    context: &'resources RenderScene,
    shadow: &'resources ShadowPassOutput,
    model: Matrix4<f32>,
    tangent_frame_transform: TangentFrameTransform,
) -> PbrShader<'resources> {
    let mut shader = PbrShader::new_with_tangent_frame_transform(
        model,
        context.camera.view_matrix(),
        context.camera.projection_matrix(),
        context.camera.position,
        tangent_frame_transform,
    );
    shader.lights = &context.lights;
    shader.ambient_light = Vector3::from(config.render.ambient_light);
    shader.shadow_map = shadow.depth.as_deref().map(Vec::as_slice);
    shader.shadow_map_size = shadow.size;
    shader.shadow_light_index = shadow.light_index;
    shader.light_space_matrix = shadow.light_space_matrix;
    shader.shadow_constant_bias = config.render.shadow_constant_bias;
    shader.shadow_slope_bias = config.render.shadow_slope_bias;
    shader.use_pcf = config.render.use_pcf;
    shader.pcf_kernel_size = config.render.pcf_kernel_size;
    shader
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

fn shadow_camera(
    config: &Config,
    context: &RenderScene,
    light_direction: Vector3<f32>,
) -> ShadowCamera {
    let inverse_view_projection = (context.camera.projection_matrix()
        * context.camera.view_matrix())
    .try_inverse()
    .unwrap_or_else(Matrix4::identity);
    let mut corners = Vec::with_capacity(8);
    for z in [-1.0, 1.0] {
        for y in [-1.0, 1.0] {
            for x in [-1.0, 1.0] {
                let homogeneous = inverse_view_projection * Vector4::new(x, y, z, 1.0);
                if homogeneous.w.abs() > f32::EPSILON {
                    corners.push(Point3::from(homogeneous.xyz() / homogeneous.w));
                }
            }
        }
    }

    let max_distance = config.render.shadow_ortho_size;
    for corner in &mut corners {
        let offset = *corner - context.camera.position;
        if offset.norm() > max_distance {
            *corner = context.camera.position + offset.normalize() * max_distance;
        }
    }
    let center = corners
        .iter()
        .fold(Vector3::zeros(), |sum, corner| sum + corner.coords)
        / corners.len().max(1) as f32;
    let center = Point3::from(center);
    let light_up = if light_direction.y.abs() > 0.9 {
        Vector3::z()
    } else {
        Vector3::y()
    };
    let view = TransformFactory::view(
        &(center - light_direction * max_distance),
        &center,
        &light_up,
    );

    let mut min = Vector3::repeat(f32::INFINITY);
    let mut max = Vector3::repeat(f32::NEG_INFINITY);
    let mut include = |point: Point3<f32>| {
        let point = view.transform_point(&point);
        min = min.zip_map(&point.coords, f32::min);
        max = max.zip_map(&point.coords, f32::max);
    };
    for corner in corners {
        include(corner);
    }
    for object in &context.scene_objects {
        for mesh in &object.model.meshes {
            for vertex in &mesh.vertices {
                include(object.transform().transform_point(&vertex.position));
            }
        }
    }

    let padding = 0.1;
    let projection = TransformFactory::orthographic(
        min.x - padding,
        max.x + padding,
        min.y - padding,
        max.y + padding,
        (-max.z - padding).max(0.01),
        (-min.z + padding).max(0.02),
    );
    ShadowCamera { view, projection }
}

pub fn render_shadow_pass(
    config: &Config,
    context: &RenderScene,
    shadow_renderer: &mut Renderer,
    shadow_target: &mut RenderTarget,
    resources: &mut FrameResources,
) -> ShadowPassOutput {
    render_shadow_pass_profiled(config, context, shadow_renderer, shadow_target, resources).0
}

pub fn render_shadow_pass_profiled(
    config: &Config,
    context: &RenderScene,
    shadow_renderer: &mut Renderer,
    shadow_target: &mut RenderTarget,
    resources: &mut FrameResources,
) -> (ShadowPassOutput, ShadowPassTimings) {
    let pass_started = Instant::now();
    if !config.render.use_shadows {
        return (
            ShadowPassOutput::disabled(),
            ShadowPassTimings {
                pass_setup: pass_started.elapsed(),
                ..Default::default()
            },
        );
    }

    let Some(shadow_light) = context.shadow_light else {
        return (
            ShadowPassOutput::disabled(),
            ShadowPassTimings {
                pass_setup: pass_started.elapsed(),
                ..Default::default()
            },
        );
    };

    let light_direction = context
        .lights
        .get(shadow_light.light_index)
        .and_then(|light| match light {
            crate::scene::light::Light::Directional { direction, .. } => Some(*direction),
            crate::scene::light::Light::Point { .. } => None,
        })
        .unwrap_or_else(|| (Point3::origin() - shadow_light.position).normalize());
    let ShadowCamera {
        view: light_view,
        projection: light_projection,
    } = shadow_camera(config, context, light_direction);
    let light_space_matrix = light_projection * light_view;

    let initial_setup = pass_started.elapsed();
    let attachment_started = Instant::now();
    shadow_renderer.clear_with_options(
        shadow_target,
        ClearOptions {
            depth: f32::INFINITY,
            ..Default::default()
        },
    );
    let attachment_processing = attachment_started.elapsed();

    let setup_started = Instant::now();
    let shadow_state = GraphicsPipelineState {
        color_target: None,
        ..Default::default()
    };
    let shaders: Vec<ShadowShader> = context
        .scene_objects
        .iter()
        .map(|object| ShadowShader::new(object.transform(), light_view, light_projection))
        .collect();
    let pass_setup = initial_setup + setup_started.elapsed();

    let recording_started = Instant::now();
    let mut shadow_phase = RenderPhase::with_capacity(
        context
            .scene_objects
            .iter()
            .map(|object| object.model.meshes.len())
            .sum(),
    );

    for (shader_index, object) in context.scene_objects.iter().enumerate() {
        for mesh in &object.model.meshes {
            let material = object.model.materials.get(mesh.material_id);
            let pbr_material = material.map(|material| match material {
                Material::Pbr(material) => material,
            });
            if matches!(pbr_material, Some(material) if material.alpha_mode == AlphaMode::Blend) {
                continue;
            }
            let command_state = GraphicsPipelineState {
                primitive: PrimitiveState {
                    cull_mode: if pbr_material.is_some_and(|material| material.double_sided) {
                        CullMode::None
                    } else {
                        shadow_state.primitive.cull_mode
                    },
                    front_face: object.front_face(),
                    ..shadow_state.primitive
                },
                ..shadow_state
            };
            shadow_phase.push(
                shader_index,
                RenderGeometry::Mesh(mesh),
                material,
                command_state,
                0.0,
            );
        }
    }
    let recording = recording_started.elapsed();
    let draw_timings = shadow_renderer.draw_phase_profiled(shadow_target, &shadow_phase, &shaders);

    let output = ShadowPassOutput {
        depth: Some(resources.shadow_depth_snapshot(shadow_target)),
        size: shadow_target.framebuffer().width,
        light_space_matrix,
        light_index: Some(shadow_light.light_index),
    };
    (
        output,
        ShadowPassTimings {
            pass_setup,
            recording,
            attachment_processing,
            backend_preparation: draw_timings.backend_preparation,
            rasterization: draw_timings.rasterization,
            submission_total: draw_timings.submission_total,
        },
    )
}
/// Executes the Main Rendering Pass.
pub fn render_main_pass(
    config: &Config,
    context: &RenderScene,
    renderer: &mut Renderer,
    target: &mut RenderTarget,
    resources: &mut FrameResources,
    shadow: &ShadowPassOutput,
    state: GraphicsPipelineState,
) -> Result<(), AssetError> {
    render_main_pass_profiled(config, context, renderer, target, resources, shadow, state)
        .map(|_| ())
}

pub fn render_main_pass_profiled(
    config: &Config,
    context: &RenderScene,
    renderer: &mut Renderer,
    target: &mut RenderTarget,
    resources: &mut FrameResources,
    shadow: &ShadowPassOutput,
    state: GraphicsPipelineState,
) -> Result<MainPassTimings, AssetError> {
    let pass_started = Instant::now();
    let bg_texture = if let Some(path) = &config.render.background_image {
        let background_path = config.resolve_path(path);
        Some(
            resources
                .background_texture(&background_path, config.render.use_mipmap)
                .map_err(|source| AssetError::BackgroundImage {
                    path: background_path,
                    source,
                })?,
        )
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

    let initial_setup = pass_started.elapsed();
    let attachment_started = Instant::now();
    renderer.clear_with_options(
        target,
        ClearOptions {
            color,
            gradient,
            texture: bg_texture.as_deref(),
            depth: f32::INFINITY,
        },
    );
    let attachment_processing = attachment_started.elapsed();

    let setup_started = Instant::now();
    let opaque_state = GraphicsPipelineState {
        color_target: Some(ColorTargetState { blend: None }),
        depth_stencil: state.depth_stencil.map(|depth_stencil| DepthStencilState {
            depth_write_enabled: true,
            ..depth_stencil
        }),
        ..state
    };
    let transparent_state = GraphicsPipelineState {
        color_target: Some(ColorTargetState {
            blend: Some(BlendState::Alpha),
        }),
        depth_stencil: state.depth_stencil.map(|depth_stencil| DepthStencilState {
            depth_write_enabled: false,
            ..depth_stencil
        }),
        ..state
    };
    let mut shaders: Vec<PbrShader<'_>> = context
        .scene_objects
        .iter()
        .map(|object| {
            configured_pbr_shader(
                config,
                context,
                shadow,
                object.transform(),
                object.tangent_frame_transform(),
            )
        })
        .collect();
    let transparent_shader_index = shaders.len();
    shaders.push(configured_pbr_shader(
        config,
        context,
        shadow,
        Matrix4::identity(),
        TangentFrameTransform::new(nalgebra::Matrix3::identity()),
    ));

    let phase_counts = context
        .scene_objects
        .iter()
        .fold([0; 3], |mut counts, object| {
            for mesh in &object.model.meshes {
                let alpha_mode = object
                    .model
                    .materials
                    .get(mesh.material_id)
                    .map(|material| match material {
                        Material::Pbr(material) => material.alpha_mode,
                    })
                    .unwrap_or(AlphaMode::Opaque);
                match alpha_mode {
                    AlphaMode::Opaque => counts[0] += 1,
                    AlphaMode::Mask(_) => counts[1] += 1,
                    AlphaMode::Blend => counts[2] += mesh.indices.len() / 3,
                }
            }
            counts
        });
    let pass_setup = initial_setup + setup_started.elapsed();

    let recording_started = Instant::now();
    let mut opaque_phase = RenderPhase::with_capacity(phase_counts[0]);
    let mut masked_phase = RenderPhase::with_capacity(phase_counts[1]);
    let mut transparent_phase = RenderPhase::with_capacity(phase_counts[2]);

    for (shader_index, obj) in context.scene_objects.iter().enumerate() {
        for (mesh_index, mesh) in obj.model.meshes.iter().enumerate() {
            let material = if mesh.material_id < obj.model.materials.len() {
                Some(&obj.model.materials[mesh.material_id])
            } else {
                None
            };

            let pbr_material = material.map(|material| match material {
                Material::Pbr(material) => material,
            });
            let alpha_mode = pbr_material.map_or(AlphaMode::Opaque, |material| material.alpha_mode);
            let command_state = |state: GraphicsPipelineState| GraphicsPipelineState {
                primitive: PrimitiveState {
                    cull_mode: if pbr_material.is_some_and(|material| material.double_sided) {
                        CullMode::None
                    } else {
                        state.primitive.cull_mode
                    },
                    front_face: obj.front_face(),
                    ..state.primitive
                },
                ..state
            };

            if alpha_mode == AlphaMode::Blend {
                let view_matrix = context.camera.view_matrix();
                let world_vertices = obj
                    .transparent_world_vertices(mesh_index)
                    .expect("transparent meshes cache world-space vertices");

                for chunk in mesh.indices.chunks(3) {
                    if chunk.len() < 3 {
                        continue;
                    }
                    // Use transformed vertices directly
                    let indices = [chunk[0], chunk[1], chunk[2]];
                    let v0_world = world_vertices[indices[0] as usize];
                    let v1_world = world_vertices[indices[1] as usize];
                    let v2_world = world_vertices[indices[2] as usize];

                    if let Some(mat) = material {
                        // Calculate Z in View Space using the centroid of World Space vertices
                        let centroid_world = (v0_world.position.coords
                            + v1_world.position.coords
                            + v2_world.position.coords)
                            / 3.0;
                        let centroid_view =
                            view_matrix * Point3::from(centroid_world).to_homogeneous();

                        transparent_phase.push(
                            transparent_shader_index,
                            RenderGeometry::IndexedTriangle {
                                vertices: world_vertices,
                                indices,
                                cache_vertices: mesh.reuses_vertices(),
                            },
                            Some(mat),
                            command_state(transparent_state),
                            centroid_view.z,
                        );
                    }
                }
            } else if matches!(alpha_mode, AlphaMode::Mask(_)) {
                masked_phase.push(
                    shader_index,
                    RenderGeometry::Mesh(mesh),
                    material,
                    command_state(opaque_state),
                    0.0,
                );
            } else {
                opaque_phase.push(
                    shader_index,
                    RenderGeometry::Mesh(mesh),
                    material,
                    command_state(opaque_state),
                    0.0,
                );
            }
        }
    }
    let mut recording = recording_started.elapsed();

    let opaque_masked =
        renderer.draw_phases_profiled(target, &[&opaque_phase, &masked_phase], &shaders);
    let sorting_started = Instant::now();
    transparent_phase.sort_transparent();
    recording += sorting_started.elapsed();
    let transparent = renderer.draw_phase_profiled(target, &transparent_phase, &shaders);

    Ok(MainPassTimings {
        pass_setup,
        recording,
        attachment_processing,
        backend_preparation: opaque_masked.backend_preparation + transparent.backend_preparation,
        opaque_masked_rasterization: opaque_masked.rasterization,
        transparent_rasterization: transparent.rasterization,
        submission_total: opaque_masked.submission_total + transparent.submission_total,
    })
}

/// Post-processing: Tone Mapping -> Gamma Correction -> u32 Buffer.
pub fn post_process_to_buffer(target: &RenderTarget, buffer: &mut [u32], config: &Config) {
    let framebuffer = target.framebuffer();
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
