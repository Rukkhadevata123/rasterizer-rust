pub use frame_buffer::FrameBuffer;
use geometry_processor::GeometryProcessor;
use triangle_processor::TriangleProcessor;

use crate::core::rasterizer::rasterize_triangle;
use crate::geometry::camera::Camera;
use crate::io::render_settings::RenderSettings;
use crate::material_system::materials::ModelData;
use crate::scene::scene_object::SceneObject;
use rayon::prelude::*;
use std::time::Instant;

use super::{frame_buffer, geometry_processor, triangle_processor};

/// 渲染器结构体 - 负责高层次渲染流程
pub struct Renderer {
    pub frame_buffer: FrameBuffer,
}

impl Renderer {
    /// 创建一个新的渲染器实例
    pub fn new(width: usize, height: usize) -> Self {
        Renderer {
            frame_buffer: FrameBuffer::new(width, height),
        }
    }

    /// 渲染一个场景，包含多个模型和对象
    pub fn render_scene(
        &self,
        scene: &mut crate::scene::scene_utils::Scene,
        settings: &RenderSettings,
    ) {
        self.frame_buffer.clear(settings, &scene.active_camera);

        // 渲染场景中的每个对象
        for object in &scene.objects {
            if object.model_id < scene.models.len() {
                let model = &scene.models[object.model_id];
                self.render(model, object, &mut scene.active_camera, settings);
            } else {
                println!("警告：对象引用了无效的模型 ID {}", object.model_id);
            }
        }
    }

    /// 渲染一个场景对象
    pub fn render(
        &self,
        model_data: &ModelData,
        scene_object: &SceneObject,
        camera: &mut Camera,
        settings: &RenderSettings,
    ) {
        let start_time = Instant::now();
        println!("渲染场景对象...");

        // 材质准备
        let material_override = scene_object
            .material_id
            .and_then(|id| model_data.materials.get(id));

        // 几何变换
        println!("变换顶点...");
        let transform_start = Instant::now();

        let (all_pixel_coords, all_view_coords, all_view_normals, mesh_vertex_offsets) =
            GeometryProcessor::transform_geometry(
                model_data,
                scene_object,
                camera,
                self.frame_buffer.width,
                self.frame_buffer.height,
            );

        let transform_duration = transform_start.elapsed();

        // 调试信息
        if !all_view_coords.is_empty() {
            let z_min = all_view_coords
                .iter()
                .map(|p| p.z)
                .fold(f32::INFINITY, f32::min);
            let z_max = all_view_coords
                .iter()
                .map(|p| p.z)
                .fold(f32::NEG_INFINITY, f32::max);
            println!("视图空间Z范围: [{:.3}, {:.3}]", z_min, z_max);
        }

        // 三角形准备
        println!("准备三角形数据...");
        let triangles_to_render = TriangleProcessor::prepare_triangles(
            model_data,
            &all_pixel_coords,
            &all_view_coords,
            &all_view_normals,
            &mesh_vertex_offsets,
            material_override,
            settings,
        );

        // 光栅化
        println!("光栅化网格...");
        let raster_start = Instant::now();

        if settings.use_multithreading {
            triangles_to_render.par_iter().for_each(|triangle_data| {
                rasterize_triangle(
                    triangle_data,
                    self.frame_buffer.width,
                    self.frame_buffer.height,
                    &self.frame_buffer.depth_buffer,
                    &self.frame_buffer.color_buffer,
                    settings,
                );
            });
        } else {
            triangles_to_render.iter().for_each(|triangle_data| {
                rasterize_triangle(
                    triangle_data,
                    self.frame_buffer.width,
                    self.frame_buffer.height,
                    &self.frame_buffer.depth_buffer,
                    &self.frame_buffer.color_buffer,
                    settings,
                );
            });
        }

        // 性能统计
        let raster_duration = raster_start.elapsed();
        let total_duration = start_time.elapsed();

        println!(
            "渲染完成. 变换: {:?}, 光栅化: {:?}, 总时间: {:?}",
            transform_duration, raster_duration, total_duration
        );
        println!("渲染了 {} 个三角形。", triangles_to_render.len());
    }
}
