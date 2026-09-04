use crate::core::color::{aces_tone_mapping, linear_to_srgb};
use crate::core::math::transform::TransformFactory;
use crate::core::pipeline_state::{
    BlendState, ColorTargetState, CullMode, DepthStencilState, FrontFace, GraphicsPipeline,
    GraphicsPipelineState, PrimitiveState, VertexProgramId,
};
use crate::io::config::Config;
use crate::pipeline::renderer::{
    BackgroundPass, BackgroundSource, FrameResources, GraphicsQueue, LoadOp, MainHdrTarget,
    Operations, PresentBuffer, RenderDevice, RenderGeometry, RenderPassDescriptor, RenderTarget,
};
use crate::pipeline::shaders::pbr::{
    PbrDrawContext, PbrFrameBindings, PbrMaterialBindings, PbrObjectBindings, PbrShader,
    PbrShadowBindings, PbrShadowBindingsDescriptor,
};
use crate::pipeline::shaders::shadow::{
    ShadowDrawContext, ShadowFrameBindings, ShadowMaterialBindings, ShadowObjectBindings,
    ShadowShader,
};
use crate::scene::AssetError;
use crate::scene::context::RenderScene;
use crate::scene::material::{AlphaMode, Material, PbrMaterial};
use nalgebra::{Matrix4, Point3, Vector3, Vector4};
use rayon::prelude::*;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;

struct ShadowMapOutput {
    depth: Arc<Vec<f32>>,
    size: usize,
    light_space_matrix: Matrix4<f32>,
    light_index: usize,
}

pub struct ShadowPassOutput {
    map: Option<ShadowMapOutput>,
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

fn pipeline_state_for_material(
    state: GraphicsPipelineState,
    front_face: FrontFace,
    double_sided: bool,
) -> GraphicsPipelineState {
    GraphicsPipelineState {
        primitive: PrimitiveState {
            cull_mode: if double_sided {
                CullMode::None
            } else {
                state.primitive.cull_mode
            },
            front_face,
            ..state.primitive
        },
        ..state
    }
}
struct ShadowCamera {
    view: Matrix4<f32>,
    projection: Matrix4<f32>,
}

impl ShadowPassOutput {
    pub const fn disabled() -> Self {
        Self { map: None }
    }

    pub fn is_enabled(&self) -> bool {
        self.map.is_some()
    }

    pub fn depth(&self) -> Option<&[f32]> {
        self.map.as_ref().map(|map| map.depth.as_slice())
    }

    pub fn depth_snapshot(&self) -> Option<Arc<Vec<f32>>> {
        self.map.as_ref().map(|map| Arc::clone(&map.depth))
    }

    pub fn size(&self) -> Option<usize> {
        self.map.as_ref().map(|map| map.size)
    }

    pub fn light_space_matrix(&self) -> Option<Matrix4<f32>> {
        self.map.as_ref().map(|map| map.light_space_matrix)
    }

    pub fn light_index(&self) -> Option<usize> {
        self.map.as_ref().map(|map| map.light_index)
    }

    fn enabled(
        depth: Arc<Vec<f32>>,
        size: usize,
        light_space_matrix: Matrix4<f32>,
        light_index: usize,
    ) -> Self {
        debug_assert_eq!(size.checked_mul(size), Some(depth.len()));
        Self {
            map: Some(ShadowMapOutput {
                depth,
                size,
                light_space_matrix,
                light_index,
            }),
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
    queue: &mut GraphicsQueue,
    shadow_target: &mut RenderTarget,
    resources: &mut FrameResources,
) -> ShadowPassOutput {
    render_shadow_pass_profiled(config, context, queue, shadow_target, resources).0
}

pub fn render_shadow_pass_profiled(
    config: &Config,
    context: &RenderScene,
    queue: &mut GraphicsQueue,
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
    let setup_started = Instant::now();
    let shadow_state = GraphicsPipelineState {
        color_target: None,
        ..Default::default()
    };
    let shadow_shader = ShadowShader;
    let frame_bindings = ShadowFrameBindings::new(light_view, light_projection);
    let object_bindings: Vec<_> = context
        .scene_objects
        .iter()
        .map(|object| ShadowObjectBindings::new(object.transform()))
        .collect();
    let shadow_vertex_program_id = VertexProgramId::new();
    let shadow_pipelines = [
        GraphicsPipeline::new(
            shadow_shader,
            pipeline_state_for_material(shadow_state, FrontFace::CounterClockwise, false),
            shadow_vertex_program_id,
        ),
        GraphicsPipeline::new(
            shadow_shader,
            pipeline_state_for_material(shadow_state, FrontFace::Clockwise, false),
            shadow_vertex_program_id,
        ),
        GraphicsPipeline::new(
            shadow_shader,
            pipeline_state_for_material(shadow_state, FrontFace::CounterClockwise, true),
            shadow_vertex_program_id,
        ),
        GraphicsPipeline::new(
            shadow_shader,
            pipeline_state_for_material(shadow_state, FrontFace::Clockwise, true),
            shadow_vertex_program_id,
        ),
    ];
    let pass_setup = initial_setup + setup_started.elapsed();

    let recording_started = Instant::now();
    let device = RenderDevice::new();
    let mut encoder = device.create_command_encoder("shadow");
    {
        let mut pass = encoder
            .begin_render_pass(
                RenderPassDescriptor {
                    label: Some("shadow"),
                    target: shadow_target,
                    color_ops: None,
                    depth_ops: Some(Operations {
                        load: LoadOp::Clear(f32::INFINITY),
                    }),
                },
                None,
            )
            .expect("the built-in shadow pass descriptor must remain valid");

        for (object_binding_index, object) in context.scene_objects.iter().enumerate() {
            for mesh in &object.model.meshes {
                let material = object.model.materials.get(mesh.material_id);
                let pbr_material = material.map(|material| match material {
                    Material::Pbr(material) => material,
                });
                if matches!(pbr_material, Some(material) if material.alpha_mode == AlphaMode::Blend)
                {
                    continue;
                }
                let command_state = pipeline_state_for_material(
                    shadow_state,
                    object.front_face(),
                    pbr_material.is_some_and(|material| material.double_sided),
                );
                let pipeline_index =
                    usize::from(command_state.primitive.front_face == FrontFace::Clockwise)
                        + 2 * usize::from(command_state.primitive.cull_mode == CullMode::None);
                pass.set_pipeline(&shadow_pipelines[pipeline_index]);
                pass.set_draw_bindings(ShadowDrawContext::new(
                    &frame_bindings,
                    &object_bindings[object_binding_index],
                    ShadowMaterialBindings::new(material),
                ));
                pass.draw_mesh(mesh, 0.0)
                    .expect("the built-in shadow draw must remain valid");
            }
        }
        pass.end()
            .expect("the built-in shadow pass must end successfully");
    }
    let command_buffer = encoder
        .finish()
        .expect("the built-in shadow command buffer must be complete");
    let recording = recording_started.elapsed();
    let submission = queue
        .submit(command_buffer)
        .expect("the built-in shadow submission must succeed");
    let output = ShadowPassOutput::enabled(
        resources.shadow_depth_snapshot(shadow_target),
        shadow_target.readback().width(),
        light_space_matrix,
        shadow_light.light_index,
    );
    (
        output,
        ShadowPassTimings {
            pass_setup,
            recording,
            attachment_processing: submission.attachment_processing,
            backend_preparation: submission.backend_preparation,
            rasterization: submission.rasterization,
            submission_total: submission.submission_total,
        },
    )
}
/// Executes the Main Rendering Pass.
pub fn render_main_pass(
    config: &Config,
    context: &RenderScene,
    queue: &mut GraphicsQueue,
    target: &mut MainHdrTarget,
    resources: &mut FrameResources,
    shadow: &ShadowPassOutput,
    state: GraphicsPipelineState,
) -> Result<(), AssetError> {
    render_main_pass_profiled(config, context, queue, target, resources, shadow, state).map(|_| ())
}

pub fn render_main_pass_profiled(
    config: &Config,
    context: &RenderScene,
    queue: &mut GraphicsQueue,
    target: &mut MainHdrTarget,
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

    let (background, color) = if let Some(texture) = bg_texture.as_deref() {
        (
            Some(BackgroundPass {
                source: BackgroundSource::Texture(texture),
            }),
            Vector3::zeros(),
        )
    } else if let Some(c) = config.render.background_color {
        (None, Vector3::from(c))
    } else if let (Some(top), Some(bottom)) = (
        config.render.background_gradient_top,
        config.render.background_gradient_bottom,
    ) {
        (
            Some(BackgroundPass {
                source: BackgroundSource::Gradient {
                    top: Vector3::from(top),
                    bottom: Vector3::from(bottom),
                },
            }),
            Vector3::zeros(),
        )
    } else {
        (None, Vector3::zeros())
    };

    let initial_setup = pass_started.elapsed();
    let color_ops = background.is_none().then_some(Operations {
        load: LoadOp::Clear(color),
    });
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
    let pbr_shader = PbrShader;
    let mut frame_bindings = PbrFrameBindings::new(
        context.camera.view_matrix(),
        context.camera.projection_matrix(),
        context.camera.position,
    );
    frame_bindings.lights = &context.lights;
    frame_bindings.ambient_light = Vector3::from(config.render.ambient_light);
    frame_bindings.shadow = shadow.map.as_ref().and_then(|shadow| {
        PbrShadowBindings::new(PbrShadowBindingsDescriptor {
            depth: shadow.depth.as_slice(),
            size: shadow.size,
            light_index: shadow.light_index,
            light_space_matrix: shadow.light_space_matrix,
            constant_bias: config.render.shadow_constant_bias,
            slope_bias: config.render.shadow_slope_bias,
            use_pcf: config.render.use_pcf,
            pcf_kernel_size: config.render.pcf_kernel_size,
        })
        .ok()
    });
    let object_bindings: Vec<_> = context
        .scene_objects
        .iter()
        .map(|object| {
            PbrObjectBindings::new_with_tangent_frame_transform(
                object.transform(),
                object.tangent_frame_transform(),
            )
        })
        .collect();
    let transparent_object_binding = PbrObjectBindings::new(Matrix4::identity());
    let fallback_material = PbrMaterial::default();
    let pbr_vertex_program_id = VertexProgramId::new();
    let pbr_pipelines = [opaque_state, transparent_state].map(|base_state| {
        [FrontFace::CounterClockwise, FrontFace::Clockwise].map(|front_face| {
            [false, true].map(|double_sided| {
                GraphicsPipeline::new(
                    pbr_shader,
                    pipeline_state_for_material(base_state, front_face, double_sided),
                    pbr_vertex_program_id,
                )
            })
        })
    });
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
    let device = RenderDevice::new();
    let mut encoder = device.create_command_encoder("main");
    {
        let mut pass = encoder
            .begin_render_pass(
                RenderPassDescriptor {
                    label: Some("main"),
                    target: target.render_target_mut(),
                    color_ops,
                    depth_ops: Some(Operations {
                        load: LoadOp::Clear(f32::INFINITY),
                    }),
                },
                background,
            )
            .expect("the built-in main pass descriptor must remain valid");
        pass.reserve_draws(phase_counts[0] + phase_counts[1]);

        for record_masked in [false, true] {
            for (object_binding_index, obj) in context.scene_objects.iter().enumerate() {
                for mesh in &obj.model.meshes {
                    let material = obj.model.materials.get(mesh.material_id);
                    let pbr_material = material.map(|material| match material {
                        Material::Pbr(material) => material,
                    });
                    let alpha_mode =
                        pbr_material.map_or(AlphaMode::Opaque, |material| material.alpha_mode);
                    let selected = if record_masked {
                        matches!(alpha_mode, AlphaMode::Mask(_))
                    } else {
                        alpha_mode == AlphaMode::Opaque
                    };
                    if !selected {
                        continue;
                    }

                    let front_face_index = usize::from(obj.front_face() == FrontFace::Clockwise);
                    let double_sided = pbr_material.is_some_and(|material| material.double_sided);
                    let pipeline = &pbr_pipelines[0][front_face_index][usize::from(double_sided)];
                    debug_assert_eq!(
                        pipeline.state(),
                        pipeline_state_for_material(opaque_state, obj.front_face(), double_sided)
                    );
                    pass.set_pipeline(pipeline);
                    pass.set_draw_bindings(PbrDrawContext::new(
                        &frame_bindings,
                        &object_bindings[object_binding_index],
                        PbrMaterialBindings::new(material, &fallback_material),
                    ));
                    pass.draw_mesh(mesh, 0.0)
                        .expect("the built-in opaque or masked draw must remain valid");
                }
            }
        }
        pass.finish_phase("opaque-masked");
        pass.reserve_draws(phase_counts[2]);

        let view_matrix = context.camera.view_matrix();
        for obj in &context.scene_objects {
            for (mesh_index, mesh) in obj.model.meshes.iter().enumerate() {
                let Some(material) = obj.model.materials.get(mesh.material_id) else {
                    continue;
                };
                let Material::Pbr(pbr_material) = material;
                if pbr_material.alpha_mode != AlphaMode::Blend {
                    continue;
                }

                let front_face_index = usize::from(obj.front_face() == FrontFace::Clockwise);
                let pipeline =
                    &pbr_pipelines[1][front_face_index][usize::from(pbr_material.double_sided)];
                debug_assert_eq!(
                    pipeline.state(),
                    pipeline_state_for_material(
                        transparent_state,
                        obj.front_face(),
                        pbr_material.double_sided,
                    )
                );
                pass.set_pipeline(pipeline);
                pass.set_draw_bindings(PbrDrawContext::new(
                    &frame_bindings,
                    &transparent_object_binding,
                    PbrMaterialBindings::new(Some(material), &fallback_material),
                ));

                let world_vertices = obj
                    .transparent_world_vertices(mesh_index)
                    .expect("transparent meshes cache world-space vertices");
                for chunk in mesh.indices.chunks(3) {
                    if chunk.len() < 3 {
                        continue;
                    }
                    let indices = [chunk[0], chunk[1], chunk[2]];
                    let v0_world = world_vertices[indices[0] as usize];
                    let v1_world = world_vertices[indices[1] as usize];
                    let v2_world = world_vertices[indices[2] as usize];
                    let centroid_world = (v0_world.position.coords
                        + v1_world.position.coords
                        + v2_world.position.coords)
                        / 3.0;
                    let centroid_view = view_matrix * Point3::from(centroid_world).to_homogeneous();
                    pass.draw(
                        RenderGeometry::IndexedTriangle {
                            vertices: world_vertices,
                            indices,
                            cache_vertices: mesh.reuses_vertices(),
                        },
                        centroid_view.z,
                    )
                    .expect("the built-in transparent draw must remain valid");
                }
            }
        }
        pass.sort_transparent();
        pass.finish_phase("transparent");
        pass.end()
            .expect("the built-in main pass must end successfully");
    }
    let command_buffer = encoder
        .finish()
        .expect("the built-in main command buffer must be complete");
    let recording = recording_started.elapsed();
    let submission = queue
        .submit(command_buffer)
        .expect("the built-in main submission must succeed");
    let opaque_masked = submission
        .phase("opaque-masked")
        .expect("the main submission must report its opaque/masked phase");
    let transparent = submission
        .phase("transparent")
        .expect("the main submission must report its transparent phase");

    Ok(MainPassTimings {
        pass_setup,
        recording,
        attachment_processing: submission.attachment_processing,
        backend_preparation: submission.backend_preparation,
        opaque_masked_rasterization: opaque_masked.rasterization,
        transparent_rasterization: transparent.rasterization,
        submission_total: submission.submission_total,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TonemapOperator {
    None,
    Aces,
}

#[derive(Debug, Error, PartialEq)]
pub enum ResolveTonemapError {
    #[error(
        "resolve-tonemap pass '{label}' source dimensions {source_width}x{source_height} do not match destination dimensions {destination_width}x{destination_height}"
    )]
    DimensionMismatch {
        label: String,
        source_width: usize,
        source_height: usize,
        destination_width: usize,
        destination_height: usize,
    },
    #[error(
        "resolve-tonemap pass '{label}' exposure must be finite and non-negative, got {exposure}"
    )]
    InvalidExposure { label: String, exposure: f32 },
}

pub struct ResolveTonemapPassDescriptor<'a> {
    pub label: Option<&'a str>,
    pub source: &'a MainHdrTarget,
    pub destination: &'a mut PresentBuffer,
    pub exposure: f32,
    pub tonemap: TonemapOperator,
}

pub fn execute_resolve_tonemap_pass(
    descriptor: ResolveTonemapPassDescriptor<'_>,
) -> Result<(), ResolveTonemapError> {
    let ResolveTonemapPassDescriptor {
        label,
        source,
        destination,
        exposure,
        tonemap,
    } = descriptor;
    let label = label.unwrap_or("<unnamed>").to_owned();
    let source_readback = source.readback();
    if (source_readback.width(), source_readback.height())
        != (destination.width(), destination.height())
    {
        return Err(ResolveTonemapError::DimensionMismatch {
            label,
            source_width: source_readback.width(),
            source_height: source_readback.height(),
            destination_width: destination.width(),
            destination_height: destination.height(),
        });
    }
    if !exposure.is_finite() || exposure < 0.0 {
        return Err(ResolveTonemapError::InvalidExposure { label, exposure });
    }

    let width = source_readback.width();
    destination
        .pixels_mut()
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, pixel) in row.iter_mut().enumerate() {
                if let Some(color) = source_readback.color(x, y) {
                    let exposed = color * exposure;
                    let mapped = match tonemap {
                        TonemapOperator::None => exposed,
                        TonemapOperator::Aces => aces_tone_mapping(exposed),
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
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::renderer::PresentBuffer;

    fn pack(color: Vector3<f32>) -> u32 {
        let r = (color.x.clamp(0.0, 1.0) * 255.0) as u32;
        let g = (color.y.clamp(0.0, 1.0) * 255.0) as u32;
        let b = (color.z.clamp(0.0, 1.0) * 255.0) as u32;
        (255 << 24) | (r << 16) | (g << 8) | b
    }

    #[test]
    fn resolve_tonemap_pass_resolves_exposes_and_packs_in_one_pass() {
        let mut source = MainHdrTarget::new(1, 1, 2).expect("HDR target should be valid");
        let samples = source.framebuffer_mut().samples_mut();
        samples[0].color = Vector3::new(0.0, 0.0, 0.0);
        samples[1].color = Vector3::new(0.5, 0.0, 0.0);
        samples[2].color = Vector3::new(0.0, 0.5, 0.0);
        samples[3].color = Vector3::new(0.5, 0.5, 1.0);
        let mut destination = PresentBuffer::new(1, 1).expect("present target should be valid");

        execute_resolve_tonemap_pass(ResolveTonemapPassDescriptor {
            label: Some("resolve-test"),
            source: &source,
            destination: &mut destination,
            exposure: 2.0,
            tonemap: TonemapOperator::None,
        })
        .expect("resolve-tonemap pass should succeed");

        let expected_linear = Vector3::new(0.5, 0.5, 0.5);
        assert_eq!(
            destination.pixels(),
            &[pack(linear_to_srgb(expected_linear))]
        );
    }

    #[test]
    fn resolve_tonemap_pass_applies_aces_when_requested() {
        let mut source = MainHdrTarget::new(1, 1, 1).expect("HDR target should be valid");
        source.framebuffer_mut().samples_mut()[0].color = Vector3::new(2.0, 1.0, 0.5);
        let mut destination = PresentBuffer::new(1, 1).expect("present target should be valid");

        execute_resolve_tonemap_pass(ResolveTonemapPassDescriptor {
            label: Some("aces-test"),
            source: &source,
            destination: &mut destination,
            exposure: 1.0,
            tonemap: TonemapOperator::Aces,
        })
        .expect("ACES pass should succeed");

        let expected = linear_to_srgb(aces_tone_mapping(Vector3::new(2.0, 1.0, 0.5)));
        assert_eq!(destination.pixels(), &[pack(expected)]);
    }

    #[test]
    fn resolve_tonemap_validation_happens_before_destination_writes() {
        let source = MainHdrTarget::new(2, 1, 1).expect("HDR target should be valid");
        let mut destination = PresentBuffer::new(1, 1).expect("present target should be valid");

        let error = execute_resolve_tonemap_pass(ResolveTonemapPassDescriptor {
            label: Some("mismatch"),
            source: &source,
            destination: &mut destination,
            exposure: 1.0,
            tonemap: TonemapOperator::None,
        })
        .expect_err("mismatched dimensions should be rejected");
        assert!(matches!(
            error,
            ResolveTonemapError::DimensionMismatch { label, .. } if label == "mismatch"
        ));
        assert_eq!(destination.pixels(), &[0]);

        let source = MainHdrTarget::new(1, 1, 1).expect("HDR target should be valid");
        let error = execute_resolve_tonemap_pass(ResolveTonemapPassDescriptor {
            label: Some("invalid-exposure"),
            source: &source,
            destination: &mut destination,
            exposure: f32::NAN,
            tonemap: TonemapOperator::None,
        })
        .expect_err("non-finite exposure should be rejected");
        assert!(matches!(
            error,
            ResolveTonemapError::InvalidExposure { label, .. } if label == "invalid-exposure"
        ));
        assert_eq!(destination.pixels(), &[0]);

        let error = execute_resolve_tonemap_pass(ResolveTonemapPassDescriptor {
            label: Some("negative-exposure"),
            source: &source,
            destination: &mut destination,
            exposure: -1.0,
            tonemap: TonemapOperator::None,
        })
        .expect_err("negative exposure should be rejected");
        assert!(matches!(
            error,
            ResolveTonemapError::InvalidExposure { label, .. } if label == "negative-exposure"
        ));
        assert_eq!(destination.pixels(), &[0]);
    }
}
