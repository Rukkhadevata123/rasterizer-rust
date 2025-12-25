use nalgebra::{Point2, Point3};

/// 计算二维三角形面积
///
/// # 参数
/// * `v0`, `v1`, `v2` - 三角形的三个顶点（屏幕坐标）
///
/// # 返回值
/// 三角形面积
#[inline]
pub fn calculate_triangle_area(v0: &Point2<f32>, v1: &Point2<f32>, v2: &Point2<f32>) -> f32 {
    ((v1.x - v0.x) * (v2.y - v0.y) - (v2.x - v0.x) * (v1.y - v0.y)).abs() * 0.5
}

/// 检查面积是否合理
/// # 参数
/// * `v0`, `v1`, `v2` - 三角形的三个顶点（屏幕坐标）
///
/// # 返回值
/// 如果三角形面积大于一个很小的阈值（1e-6），返回true
///
#[inline]
pub fn is_valid_triangle(v0: &Point2<f32>, v1: &Point2<f32>, v2: &Point2<f32>) -> bool {
    let area = calculate_triangle_area(v0, v1, v2);
    area > 1e-6
}

/// 检查三角形是否应该被剔除（面积过小）
///
/// # 参数
/// * `v0`, `v1`, `v2` - 三角形的三个顶点（屏幕坐标）
/// * `min_area` - 最小面积阈值
///
/// # 返回值
/// 如果三角形应被剔除，返回true
#[inline]
pub fn should_cull_small_triangle(
    v0: &Point2<f32>,
    v1: &Point2<f32>,
    v2: &Point2<f32>,
    min_area: f32,
) -> bool {
    calculate_triangle_area(v0, v1, v2) < min_area
}

/// 进行背面剔除判断
///
/// # 参数
/// * `v0`, `v1`, `v2` - 三角形的三个顶点（视图空间坐标）
///
/// # 返回值
/// 如果三角形是背面（应被剔除），返回true
#[inline]
pub fn is_backface(v0: &Point3<f32>, v1: &Point3<f32>, v2: &Point3<f32>) -> bool {
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let face_normal = edge1.cross(&edge2).normalize();
    let view_dir = (v0 - Point3::origin()).normalize();

    // 如果法线与视线方向夹角大于90度，则是背面
    face_normal.dot(&view_dir) > -1e-6
}

/// 检查像素是否在三角形边缘附近（用于线框渲染模式）
///
/// # 参数
/// * `pixel_point` - 像素中心点坐标
/// * `v0`, `v1`, `v2` - 三角形的三个顶点（屏幕坐标）
/// * `edge_threshold` - 边缘检测阈值（像素距离边缘的最大距离）
///
/// # 返回值
/// 如果像素在三角形任意边缘附近，返回true
#[inline]
pub fn is_on_triangle_edge(
    pixel_point: Point2<f32>,
    v0: Point2<f32>,
    v1: Point2<f32>,
    v2: Point2<f32>,
    edge_threshold: f32,
) -> bool {
    // 计算点到线段的距离
    let dist_to_edge = |p: Point2<f32>, edge_start: Point2<f32>, edge_end: Point2<f32>| -> f32 {
        let edge_vec = edge_end - edge_start.coords;
        let edge_length_sq = edge_vec.coords.norm_squared();

        // 如果边长为0，直接返回点到起点的距离
        if edge_length_sq < 1e-6 {
            return (p - edge_start.coords).coords.norm();
        }

        // 计算投影比例
        let t =
            ((p - edge_start.coords).coords.dot(&edge_vec.coords) / edge_length_sq).clamp(0.0, 1.0);

        // 计算投影点
        let projection = Point2::new(edge_start.x + t * edge_vec.x, edge_start.y + t * edge_vec.y);

        // 返回点到投影点的距离
        (p - projection.coords).coords.norm()
    };

    // 检查点到三条边的距离是否小于阈值
    dist_to_edge(pixel_point, v0, v1) <= edge_threshold
        || dist_to_edge(pixel_point, v1, v2) <= edge_threshold
        || dist_to_edge(pixel_point, v2, v0) <= edge_threshold
}
