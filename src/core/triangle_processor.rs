use crate::core::geometry_processor::GeometryResult;
use crate::core::rasterizer::{TextureSource, VertexRenderData};
use crate::geometry::culling::{is_backface, should_cull_small_triangle};
use crate::io::render_settings::RenderSettings;
use crate::material_system::light::Light;
use crate::material_system::materials::{Material, MaterialView, ModelData, Vertex};
use nalgebra::{Point2, Point3, Vector3};
use rayon::prelude::*;

// 重新导出 TriangleData，使其对外可见
pub use crate::core::rasterizer::TriangleData;

/// 三角形处理器，负责三角形数据准备和剔除
pub struct TriangleProcessor;

impl TriangleProcessor {
    /// 准备所有要渲染的三角形 - 使用新的GeometryResult结构
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_triangles<'a>(
        model_data: &'a ModelData,
        geometry_result: &GeometryResult,
        material_override: Option<&'a Material>,
        settings: &'a RenderSettings,
        lights: &'a [Light],
        ambient_intensity: f32,
        ambient_color: Vector3<f32>,
    ) -> Vec<TriangleData<'a>> {
        model_data
            .meshes
            .par_iter()
            .enumerate()
            .flat_map(|(mesh_idx, mesh)| {
                let vertex_offset = geometry_result.mesh_offsets[mesh_idx];
                let material_opt = material_override
                    .or_else(|| mesh.material_id.and_then(|id| model_data.materials.get(id)));

                mesh.indices
                    .chunks_exact(3)
                    .enumerate()
                    .filter_map(move |(face_idx, indices)| {
                        let global_face_index = (mesh_idx * 1000 + face_idx) as u64;

                        Self::process_triangle(
                            indices,
                            &mesh.vertices,
                            vertex_offset,
                            global_face_index,
                            geometry_result,
                            material_opt,
                            settings,
                            lights,
                            ambient_intensity,
                            ambient_color,
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// 处理单个三角形 - 更新为使用GeometryResult
    #[allow(clippy::too_many_arguments)]
    fn process_triangle<'a>(
        indices: &[u32],
        vertices: &[Vertex],
        vertex_offset: usize,
        global_face_index: u64,
        geometry_result: &GeometryResult,
        material_opt: Option<&'a Material>,
        settings: &'a RenderSettings,
        lights: &'a [Light],
        ambient_intensity: f32,
        ambient_color: Vector3<f32>,
    ) -> Option<TriangleData<'a>> {
        // 提取顶点索引
        let i0 = indices[0] as usize;
        let i1 = indices[1] as usize;
        let i2 = indices[2] as usize;

        // 计算全局索引
        let global_i0 = vertex_offset + i0;
        let global_i1 = vertex_offset + i1;
        let global_i2 = vertex_offset + i2;

        // 检查索引有效性 - 使用GeometryResult的字段
        if global_i0 >= geometry_result.screen_coords.len()
            || global_i1 >= geometry_result.screen_coords.len()
            || global_i2 >= geometry_result.screen_coords.len()
        {
            return None;
        }

        // 获取坐标 - 使用GeometryResult的字段
        let pix0 = geometry_result.screen_coords[global_i0];
        let pix1 = geometry_result.screen_coords[global_i1];
        let pix2 = geometry_result.screen_coords[global_i2];

        let view_pos0 = geometry_result.view_coords[global_i0];
        let view_pos1 = geometry_result.view_coords[global_i1];
        let view_pos2 = geometry_result.view_coords[global_i2];

        // 背面剔除
        if settings.backface_culling && is_backface(&view_pos0, &view_pos1, &view_pos2) {
            return None;
        }

        // 小三角形剔除
        if settings.cull_small_triangles
            && should_cull_small_triangle(&pix0, &pix1, &pix2, settings.min_triangle_area)
        {
            return None;
        }

        // 确定纹理和颜色
        let texture_source =
            Self::determine_texture_source(settings, material_opt, global_face_index);
        let base_color = Self::determine_base_color(settings, &texture_source, material_opt);

        // 创建材质视图
        let material_view = material_opt.map(|m| {
            if settings.use_pbr {
                MaterialView::PBR(m)
            } else {
                MaterialView::BlinnPhong(m)
            }
        });

        // 创建顶点数据 - 使用GeometryResult的字段
        let vertex_data = [
            Self::create_vertex_render_data(
                &pix0,
                view_pos0,
                &vertices[i0],
                global_i0,
                &texture_source,
                &geometry_result.view_normals,
            ),
            Self::create_vertex_render_data(
                &pix1,
                view_pos1,
                &vertices[i1],
                global_i1,
                &texture_source,
                &geometry_result.view_normals,
            ),
            Self::create_vertex_render_data(
                &pix2,
                view_pos2,
                &vertices[i2],
                global_i2,
                &texture_source,
                &geometry_result.view_normals,
            ),
        ];

        Some(TriangleData {
            vertices: vertex_data,
            base_color,
            texture_source,
            material_view,
            lights,
            ambient_intensity,
            ambient_color,
            is_perspective: settings.is_perspective(),
        })
    }

    fn determine_texture_source<'a>(
        settings: &RenderSettings,
        material_opt: Option<&'a Material>,
        global_face_index: u64,
    ) -> TextureSource<'a> {
        if !settings.use_texture {
            return if settings.colorize {
                TextureSource::FaceColor(global_face_index)
            } else {
                TextureSource::None
            };
        }

        // 优先级：PNG材质 > 面随机颜色 > 固体颜色
        if let Some(tex) = material_opt.and_then(|m| m.texture.as_ref()) {
            TextureSource::Image(tex)
        } else if settings.colorize {
            TextureSource::FaceColor(global_face_index)
        } else {
            let color = material_opt.map_or_else(|| Vector3::new(0.7, 0.7, 0.7), |m| m.diffuse());
            TextureSource::SolidColor(color)
        }
    }

    fn determine_base_color(
        _settings: &RenderSettings,
        texture_source: &TextureSource,
        material_opt: Option<&Material>,
    ) -> Vector3<f32> {
        match texture_source {
            TextureSource::FaceColor(_) => Vector3::new(1.0, 1.0, 1.0),
            _ => material_opt.map_or_else(|| Vector3::new(0.7, 0.7, 0.7), |m| m.diffuse()),
        }
    }

    fn create_vertex_render_data(
        pix: &Point2<f32>,
        view_pos: Point3<f32>,
        vertex: &Vertex,
        global_index: usize,
        texture_source: &TextureSource,
        all_view_normals: &[Vector3<f32>],
    ) -> VertexRenderData {
        VertexRenderData {
            pix: Point2::new(pix.x, pix.y),
            z_view: view_pos.z,
            texcoord: if matches!(texture_source, TextureSource::Image(_)) {
                Some(vertex.texcoord)
            } else {
                None
            },
            normal_view: Some(all_view_normals[global_index]),
            position_view: Some(view_pos),
        }
    }
}
