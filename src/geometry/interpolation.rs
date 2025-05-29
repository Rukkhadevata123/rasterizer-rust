use nalgebra::{Point2, Point3, Vector2, Vector3};
use std::ops::{Add, Mul};

const EPSILON: f32 = 1e-5; // 浮点比较的小值

/// 计算点p相对于三角形(v1, v2, v3)的重心坐标(alpha, beta, gamma)
/// 如果三角形退化则返回None
/// Alpha对应v1, Beta对应v2, Gamma对应v3
pub fn barycentric_coordinates(
    p: Point2<f32>,
    v1: Point2<f32>,
    v2: Point2<f32>,
    v3: Point2<f32>,
) -> Option<Vector3<f32>> {
    let e1 = v2 - v1;
    let e2 = v3 - v1;
    let p_v1 = p - v1;

    // 主三角形面积(乘2)，使用2D叉积行列式
    let total_area_x2 = e1.x * e2.y - e1.y * e2.x;

    if total_area_x2.abs() < EPSILON {
        return None; // 退化三角形
    }

    let inv_total_area_x2 = 1.0 / total_area_x2;

    // 子三角形(p, v3, v1)面积/总面积 -> v2的重心坐标(beta)
    let area2_x2 = p_v1.x * e2.y - p_v1.y * e2.x;
    let beta = area2_x2 * inv_total_area_x2;

    // 子三角形(p, v1, v2)面积/总面积 -> v3的重心坐标(gamma)
    let area3_x2 = e1.x * p_v1.y - e1.y * p_v1.x;
    let gamma = area3_x2 * inv_total_area_x2;

    // v1的重心坐标(alpha)
    let alpha = 1.0 - beta - gamma;

    Some(Vector3::new(alpha, beta, gamma))
}

/// 检查重心坐标是否表示点在三角形内部
#[inline(always)]
pub fn is_inside_triangle(bary: Vector3<f32>) -> bool {
    bary.x >= -EPSILON && bary.y >= -EPSILON && bary.z >= -EPSILON
}

/// 通用的透视校正插值函数，适用于任意可线性组合的类型
#[allow(clippy::too_many_arguments)]
fn perspective_correct_interpolate<T>(
    bary: Vector3<f32>,
    v1: T,
    v2: T,
    v3: T,
    z1_view: f32,
    z2_view: f32,
    z3_view: f32,
    is_perspective: bool,
) -> T
where
    T: Copy + Add<Output = T> + Mul<f32, Output = T>,
{
    if !is_perspective {
        // 正交投影：线性插值
        v1 * bary.x + v2 * bary.y + v3 * bary.z
    } else {
        // 透视投影：插值 attribute/z
        let inv_z1 = if z1_view.abs() > EPSILON {
            1.0 / z1_view
        } else {
            0.0
        };
        let inv_z2 = if z2_view.abs() > EPSILON {
            1.0 / z2_view
        } else {
            0.0
        };
        let inv_z3 = if z3_view.abs() > EPSILON {
            1.0 / z3_view
        } else {
            0.0
        };

        let interpolated_inv_z = bary.x * inv_z1 + bary.y * inv_z2 + bary.z * inv_z3;

        if interpolated_inv_z.abs() > EPSILON {
            let inv_z = 1.0 / interpolated_inv_z;
            // 插值 attr/z 并乘以插值后的 z
            (v1 * (bary.x * inv_z1) + v2 * (bary.y * inv_z2) + v3 * (bary.z * inv_z3)) * inv_z
        } else {
            // 回退到线性插值
            v1 * bary.x + v2 * bary.y + v3 * bary.z
        }
    }
}

/// 使用重心坐标插值深度(z)，带透视校正
/// 采用视空间Z值(通常为负值)
/// 返回正值深度用于缓冲区比较，无效则返回f32::INFINITY
pub fn interpolate_depth(
    bary: Vector3<f32>,
    z1_view: f32,
    z2_view: f32,
    z3_view: f32,
    is_perspective: bool,
) -> f32 {
    if !is_inside_triangle(bary) {
        return f32::INFINITY;
    }

    let interpolated_z = perspective_correct_interpolate(
        bary,
        z1_view,
        z2_view,
        z3_view,
        z1_view,
        z2_view,
        z3_view,
        is_perspective,
    );

    // 返回正值深度用于缓冲区(较小值表示更近)
    if interpolated_z > -EPSILON {
        // 检查是否在近平面后方或非常接近
        f32::INFINITY
    } else {
        -interpolated_z
    }
}

/// 使用重心坐标插值纹理坐标(UV)，带透视校正
/// 采用视空间Z值进行校正
#[allow(clippy::too_many_arguments)]
pub fn interpolate_texcoords(
    bary: Vector3<f32>,
    tc1: Vector2<f32>,
    tc2: Vector2<f32>,
    tc3: Vector2<f32>,
    z1_view: f32,
    z2_view: f32,
    z3_view: f32,
    is_perspective: bool,
) -> Vector2<f32> {
    perspective_correct_interpolate(
        bary,
        tc1,
        tc2,
        tc3,
        z1_view,
        z2_view,
        z3_view,
        is_perspective,
    )
}

/// 使用重心坐标插值法线向量，带透视校正
/// 采用视空间Z值进行校正
#[allow(clippy::too_many_arguments)]
pub fn interpolate_normal(
    bary: Vector3<f32>,
    n1: Vector3<f32>,
    n2: Vector3<f32>,
    n3: Vector3<f32>,
    is_perspective: bool,
    z1_view: f32,
    z2_view: f32,
    z3_view: f32,
) -> Vector3<f32> {
    let result = perspective_correct_interpolate(
        bary,
        n1,
        n2,
        n3,
        z1_view,
        z2_view,
        z3_view,
        is_perspective,
    );

    // 归一化结果
    if result.norm_squared() > EPSILON {
        result.normalize()
    } else {
        Vector3::z() // 使用默认Z方向作为备用
    }
}

/// 使用重心坐标插值视空间位置，带透视校正
/// 采用视空间Z值进行校正
#[allow(clippy::too_many_arguments)]
pub fn interpolate_position(
    bary: Vector3<f32>,
    p1: Point3<f32>,
    p2: Point3<f32>,
    p3: Point3<f32>,
    is_perspective: bool,
    z1_view: f32,
    z2_view: f32,
    z3_view: f32,
) -> Point3<f32> {
    // 通过向量计算插值，然后转回点
    let coords1 = p1.coords;
    let coords2 = p2.coords;
    let coords3 = p3.coords;

    let result = perspective_correct_interpolate(
        bary,
        coords1,
        coords2,
        coords3,
        z1_view,
        z2_view,
        z3_view,
        is_perspective,
    );

    Point3::from(result)
}

/// 使用重心坐标插值切线向量，带透视校正
/// 采用视空间Z值进行校正
#[allow(clippy::too_many_arguments)]
pub fn interpolate_tangent(
    bary: Vector3<f32>,
    t1: Vector3<f32>,
    t2: Vector3<f32>,
    t3: Vector3<f32>,
    is_perspective: bool,
    z1_view: f32,
    z2_view: f32,
    z3_view: f32,
) -> Vector3<f32> {
    let result = perspective_correct_interpolate(
        bary,
        t1,
        t2,
        t3,
        z1_view,
        z2_view,
        z3_view,
        is_perspective,
    );
    result.try_normalize(1e-6).unwrap_or_else(Vector3::x)
}

/// 使用重心坐标插值副切线向量，带透视校正
/// 采用视空间Z值进行校正
#[allow(clippy::too_many_arguments)]
pub fn interpolate_bitangent(
    bary: Vector3<f32>,
    b1: Vector3<f32>,
    b2: Vector3<f32>,
    b3: Vector3<f32>,
    is_perspective: bool,
    z1_view: f32,
    z2_view: f32,
    z3_view: f32,
) -> Vector3<f32> {
    let result = perspective_correct_interpolate(
        bary,
        b1,
        b2,
        b3,
        z1_view,
        z2_view,
        z3_view,
        is_perspective,
    );
    result.try_normalize(1e-6).unwrap_or_else(Vector3::y)
}
