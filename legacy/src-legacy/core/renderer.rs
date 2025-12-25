use crate::core::frame_buffer::FrameBuffer;
use crate::core::rasterizer::Rasterizer;
use crate::core::shadow_map::ShadowMap;
use crate::geometry::camera::Camera;
use crate::geometry::transform::{
    TransformFactory, clip_to_screen, compute_normal_matrix, point_to_clip, transform_normal,
    transform_point,
};
use crate::io::render_settings::RenderSettings;
use crate::material_system::light::Light;
use crate::scene::scene_object::SceneObject;
use crate::scene::scene_utils::Scene;
use log::debug;
use nalgebra::{Point2, Point3, Vector3};
use rayon::prelude::*;
use std::time::Instant;

pub struct TransformedGeometry {
    pub screen_coords: Vec<Point2<f32>>,
    pub view_coords: Vec<Point3<f32>>,
    pub view_normals: Vec<Vector3<f32>>,
    pub mesh_offsets: Vec<usize>,
}

pub fn transform_geometry(
    scene_object: &SceneObject,
    camera: &mut Camera,
    frame_width: usize,
    frame_height: usize,
) -> TransformedGeometry {
    camera.update_matrices();

    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut mesh_offsets = vec![0];
    for mesh in &scene_object.model.meshes {
        vertices.extend(mesh.vertices.iter().map(|v| v.position));
        normals.extend(mesh.vertices.iter().map(|v| v.normal));
        mesh_offsets.push(vertices.len());
    }

    let model_matrix = &scene_object.transform;
    let view_matrix = &camera.view_matrix();
    let projection_matrix = &camera.projection_matrix();

    let model_view = TransformFactory::model_view(model_matrix, view_matrix);
    let mvp = TransformFactory::model_view_projection(model_matrix, view_matrix, projection_matrix);
    let normal_matrix = compute_normal_matrix(&model_view);

    let view_positions = vertices
        .par_iter()
        .map(|v| transform_point(v, &model_view))
        .collect();
    let screen_coords = vertices
        .par_iter()
        .map(|v| {
            let clip = point_to_clip(v, &mvp);
            clip_to_screen(&clip, frame_width as f32, frame_height as f32)
        })
        .collect();
    let view_normals = normals
        .par_iter()
        .map(|n| transform_normal(n, &normal_matrix))
        .collect();

    TransformedGeometry {
        screen_coords,
        view_coords: view_positions,
        view_normals,
        mesh_offsets,
    }
}

pub struct Renderer {
    pub frame_buffer: FrameBuffer,
    shadow_map: Option<ShadowMap>,
    last_frame_time: Option<std::time::Duration>,
}

impl Renderer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            frame_buffer: FrameBuffer::new(width, height),
            shadow_map: None,
            last_frame_time: None,
        }
    }

    pub fn resize(&mut self, width: usize, height: usize) {
        if self.frame_buffer.width != width || self.frame_buffer.height != height {
            debug!(
                "渲染器尺寸变化: {}x{} -> {}x{}",
                self.frame_buffer.width, self.frame_buffer.height, width, height
            );
            self.frame_buffer = FrameBuffer::new(width, height);
        }
    }

    pub fn render_scene(&mut self, scene: &mut Scene, settings: &RenderSettings) {
        let frame_start = Instant::now();

        self.resize(settings.width, settings.height);

        if settings.enable_shadow_mapping {
            self.generate_shadow_map(scene, settings);
        }

        self.frame_buffer
            .clear(settings, &scene.active_camera, self.shadow_map.as_ref());

        let geometry = transform_geometry(
            &scene.object,
            &mut scene.active_camera,
            self.frame_buffer.width,
            self.frame_buffer.height,
        );

        let triangles = Rasterizer::prepare_triangles(
            &scene.object.model,
            &geometry,
            None,
            settings,
            &scene.lights,
            scene.ambient_intensity,
            scene.ambient_color,
        );

        Rasterizer::rasterize_triangles(
            &triangles,
            self.frame_buffer.width,
            self.frame_buffer.height,
            &self.frame_buffer.depth_buffer,
            &self.frame_buffer.color_buffer,
            settings,
            &self.frame_buffer,
        );

        self.last_frame_time = Some(frame_start.elapsed());
        debug!(
            "渲染完成 '{}': {} 三角形, 耗时: {:?}",
            scene.object.model.name,
            triangles.len(),
            self.last_frame_time.unwrap()
        );
    }

    fn generate_shadow_map(&mut self, scene: &Scene, settings: &RenderSettings) {
        if self.shadow_map.is_none()
            || self.shadow_map.as_ref().unwrap().size != settings.shadow_map_size
        {
            self.shadow_map = Some(ShadowMap::new(settings.shadow_map_size));
            debug!(
                "创建新阴影贴图: {}x{}",
                settings.shadow_map_size, settings.shadow_map_size
            );
        }

        let shadow_map = self.shadow_map.as_mut().unwrap();

        if let Some(directional_light) = scene
            .lights
            .iter()
            .find(|light| matches!(light, Light::Directional { enabled: true, .. }))
        {
            let scene_bounds = Self::compute_scene_bounds(scene, settings);
            shadow_map.generate(&scene.object, directional_light, scene_bounds);

            if shadow_map.is_valid {
                debug!("阴影贴图已更新");
            } else {
                debug!("阴影贴图生成失败");
            }
        } else {
            shadow_map.is_valid = false;
            debug!("未找到可用的方向光源，跳过阴影贴图生成");
        }
    }

    fn compute_scene_bounds(
        scene: &Scene,
        settings: &RenderSettings,
    ) -> (nalgebra::Point3<f32>, f32) {
        let mut min_pos = nalgebra::Point3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut max_pos =
            nalgebra::Point3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
        let mut has_vertices = false;

        for mesh in &scene.object.model.meshes {
            for vertex in &mesh.vertices {
                let world_pos = scene.object.transform.transform_point(&vertex.position);

                min_pos.x = min_pos.x.min(world_pos.x);
                min_pos.y = min_pos.y.min(world_pos.y);
                min_pos.z = min_pos.z.min(world_pos.z);

                max_pos.x = max_pos.x.max(world_pos.x);
                max_pos.y = max_pos.y.max(world_pos.y);
                max_pos.z = max_pos.z.max(world_pos.z);

                has_vertices = true;
            }
        }

        if !has_vertices {
            debug!("场景无顶点数据，使用默认阴影边界");
            return (
                nalgebra::Point3::new(0.0, 0.0, 0.0),
                settings.shadow_distance,
            );
        }

        let center = nalgebra::Point3::new(
            (min_pos.x + max_pos.x) * 0.5,
            (min_pos.y + max_pos.y) * 0.5,
            (min_pos.z + max_pos.z) * 0.5,
        );

        let size = max_pos - min_pos;
        let radius = (size.x.max(size.y).max(size.z) * 0.6).max(settings.shadow_distance * 0.5);

        let ground_extended_radius = if settings.enable_ground_plane {
            let ground_distance = (center.y - settings.ground_plane_height).abs() + radius;
            radius.max(ground_distance)
        } else {
            radius
        };

        debug!(
            "场景边界: 中心({:.2}, {:.2}, {:.2}), 半径: {:.2}",
            center.x, center.y, center.z, ground_extended_radius
        );

        (center, ground_extended_radius)
    }
}

impl Default for Renderer {
    fn default() -> Self {
        Self::new(800, 600)
    }
}
