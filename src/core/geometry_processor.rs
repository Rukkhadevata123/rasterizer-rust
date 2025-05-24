use crate::geometry::camera::Camera;
use crate::geometry::transform::transform_pipeline_batch_parallel;
use crate::material_system::materials::ModelData;
use crate::scene::scene_object::SceneObject;
use nalgebra::{Point2, Point3, Vector3};

/// 几何变换结果类型
pub type GeometryTransformResult = (
    Vec<Point2<f32>>,  // 屏幕坐标
    Vec<Point3<f32>>,  // 视图空间坐标
    Vec<Vector3<f32>>, // 视图空间法线
    Vec<usize>,        // 网格顶点偏移量
);

/// 几何处理器，负责顶点变换和几何数据准备
pub struct GeometryProcessor;

impl GeometryProcessor {
    /// 执行完整的几何变换管线
    pub fn transform_geometry(
        model_data: &ModelData,
        scene_object: &SceneObject,
        camera: &mut Camera,
        frame_width: usize,
        frame_height: usize,
    ) -> GeometryTransformResult {
        // 确保相机矩阵是最新的
        camera.update_matrices();

        // 获取对象的模型矩阵
        let model_matrix = scene_object.transform;

        // 收集所有顶点和法线
        let vertex_count = Self::estimate_vertex_count(model_data);
        let mut all_vertices_model: Vec<Point3<f32>> = Vec::with_capacity(vertex_count);
        let mut all_normals_model: Vec<Vector3<f32>> = Vec::with_capacity(vertex_count);
        let mut mesh_vertex_offsets: Vec<usize> = vec![0];

        for mesh in &model_data.meshes {
            all_vertices_model.extend(mesh.vertices.iter().map(|v| v.position));
            all_normals_model.extend(mesh.vertices.iter().map(|v| v.normal));
            mesh_vertex_offsets.push(all_vertices_model.len());
        }

        // 获取相机变换矩阵
        let view_matrix = camera.view_matrix();
        let projection_matrix = camera.projection_matrix();

        // 执行批量变换
        let (all_pixel_coords, all_view_coords, all_view_normals) =
            transform_pipeline_batch_parallel(
                &all_vertices_model,
                &all_normals_model,
                &model_matrix,
                &view_matrix,
                &projection_matrix,
                frame_width,
                frame_height,
            );

        (
            all_pixel_coords,
            all_view_coords,
            all_view_normals,
            mesh_vertex_offsets,
        )
    }

    /// 估算模型的总顶点数
    fn estimate_vertex_count(model_data: &ModelData) -> usize {
        model_data
            .meshes
            .iter()
            .map(|mesh| mesh.vertices.len())
            .sum()
    }
}
