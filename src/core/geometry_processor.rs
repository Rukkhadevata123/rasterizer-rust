use crate::geometry::camera::Camera;
use crate::geometry::transform::{vertex_pipeline_parallel, vertex_pipeline_serial};
use crate::io::render_settings::RenderSettings;
use crate::scene::scene_object::SceneObject;
use nalgebra::{Point2, Point3, Vector3};

/// 几何变换结果类型（重新导出以保持兼容性）
pub type GeometryTransformResult = (
    Vec<Point2<f32>>,  // 屏幕坐标
    Vec<Point3<f32>>,  // 视图空间坐标
    Vec<Vector3<f32>>, // 视图空间法线
    Vec<usize>,        // 网格顶点偏移量
);

/// 几何处理器，负责顶点变换和几何数据准备
pub struct GeometryProcessor;

impl GeometryProcessor {
    /// 执行完整的几何变换管线 - 根据设置选择串行或并行
    pub fn transform_geometry(
        scene_object: &SceneObject,
        camera: &mut Camera,
        frame_width: usize,
        frame_height: usize,
        settings: &RenderSettings,
    ) -> GeometryTransformResult {
        // 确保相机矩阵是最新的
        camera.update_matrices();

        // 收集所有顶点和法线
        let mut all_vertices_model: Vec<Point3<f32>> = Vec::new();
        let mut all_normals_model: Vec<Vector3<f32>> = Vec::new();
        let mut mesh_vertex_offsets: Vec<usize> = vec![0];

        for mesh in &scene_object.model_data.meshes {
            all_vertices_model.extend(mesh.vertices.iter().map(|v| v.position));
            all_normals_model.extend(mesh.vertices.iter().map(|v| v.normal));
            mesh_vertex_offsets.push(all_vertices_model.len());
        }

        // 获取变换矩阵
        let model_matrix = scene_object.transform;
        let view_matrix = camera.view_matrix();
        let projection_matrix = camera.projection_matrix();

        // 根据设置选择串行或并行变换
        let (all_pixel_coords, all_view_coords, all_view_normals) = if settings.use_multithreading {
            vertex_pipeline_parallel(
                &all_vertices_model,
                &all_normals_model,
                &model_matrix,
                &view_matrix,
                &projection_matrix,
                frame_width,
                frame_height,
            )
        } else {
            vertex_pipeline_serial(
                &all_vertices_model,
                &all_normals_model,
                &model_matrix,
                &view_matrix,
                &projection_matrix,
                frame_width,
                frame_height,
            )
        };

        (
            all_pixel_coords,
            all_view_coords,
            all_view_normals,
            mesh_vertex_offsets,
        )
    }
}
