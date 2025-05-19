use crate::material_system::materials::ModelData;
use nalgebra::{Point3, Vector3};

/// 归一化和中心化模型顶点
pub fn normalize_and_center_model(model_data: &mut ModelData) -> (Vector3<f32>, f32) {
    if model_data.meshes.is_empty() {
        return (Vector3::zeros(), 1.0);
    }

    // 计算所有顶点的边界框或质心
    let mut min_coord = Point3::new(f32::MAX, f32::MAX, f32::MAX);
    let mut max_coord = Point3::new(f32::MIN, f32::MIN, f32::MIN);
    let mut vertex_sum = Vector3::zeros();
    let mut vertex_count = 0;

    for mesh in &model_data.meshes {
        for vertex in &mesh.vertices {
            min_coord = min_coord.inf(&vertex.position);
            max_coord = max_coord.sup(&vertex.position);
            vertex_sum += vertex.position.coords;
            vertex_count += 1;
        }
    }

    if vertex_count == 0 {
        return (Vector3::zeros(), 1.0);
    }

    let center = vertex_sum / (vertex_count as f32);
    let extent = max_coord - min_coord;
    let max_extent = extent.x.max(extent.y).max(extent.z);

    let scale_factor = if max_extent > 1e-6 {
        1.6 / max_extent // 缩放以大致适合[-0.8, 0.8]立方体（类似于Python的0.8因子）
    } else {
        1.0
    };

    // 对所有顶点应用变换
    for mesh in &mut model_data.meshes {
        for vertex in &mut mesh.vertices {
            vertex.position = Point3::from((vertex.position.coords - center) * scale_factor);
        }
    }

    (center, scale_factor)
}
