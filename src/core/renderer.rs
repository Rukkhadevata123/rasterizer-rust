use crate::core::{
    frame_buffer::FrameBuffer,
    geometry::transform_geometry,
    rasterizer::{prepare_triangles, rasterize_triangles},
    shadow_map::ShadowMap,
};
use crate::io::render_settings::RenderSettings;
use crate::material_system::light::Light;
use crate::scene::scene_utils::Scene;
use log::debug;
use std::time::Instant;

pub struct Renderer {
    pub frame_buffer: FrameBuffer,
    last_frame_time: Option<std::time::Duration>,
    shadow_map: Option<ShadowMap>,
}

impl Renderer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            frame_buffer: FrameBuffer::new(width, height),
            last_frame_time: None,
            shadow_map: None,
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

        if self.frame_buffer.width != settings.width || self.frame_buffer.height != settings.height
        {
            self.resize(settings.width, settings.height);
        }

        // 1. 阴影贴图生成
        if settings.enable_shadow_mapping {
            self.generate_shadow_map(scene, settings);
        }

        // 2. 清空帧缓冲区
        self.frame_buffer
            .clear(settings, &scene.active_camera, self.shadow_map.as_ref());

        // 3. 几何变换
        let geometry_result = transform_geometry(
            &scene.object,
            &mut scene.active_camera,
            self.frame_buffer.width,
            self.frame_buffer.height,
        );

        // 4. 三角形准备
        let triangles = prepare_triangles(
            &scene.object.model_data,
            &geometry_result,
            None,
            settings,
            &scene.lights,
            scene.ambient_intensity,
            scene.ambient_color,
        );

        // 5. 光栅化
        rasterize_triangles(
            &triangles,
            self.frame_buffer.width,
            self.frame_buffer.height,
            &self.frame_buffer.depth_buffer,
            &self.frame_buffer.color_buffer,
            settings,
            &self.frame_buffer,
        );

        let frame_time = frame_start.elapsed();
        self.last_frame_time = Some(frame_time);

        if log::log_enabled!(log::Level::Debug) {
            debug!(
                "渲染完成 '{}': {} 三角形, 耗时: {:?}{}",
                scene.object.model_data.name,
                triangles.len(),
                frame_time,
                if settings.enable_shadow_mapping
                    && self.shadow_map.as_ref().is_some_and(|sm| sm.is_valid)
                {
                    " (含阴影)"
                } else {
                    ""
                }
            );
        }
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

        for mesh in &scene.object.model_data.meshes {
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

    pub fn invalidate_background_cache(&mut self) {
        self.frame_buffer.invalidate_caches();
    }
}

impl Default for Renderer {
    fn default() -> Self {
        Self::new(800, 600)
    }
}
