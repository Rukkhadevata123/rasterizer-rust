use nalgebra::{Matrix3, Matrix4, Point2, Point3, Rotation3, Unit, Vector3, Vector4};

//------------------------ 第1层：创建变换矩阵 ------------------------//

/// 变换矩阵工厂，提供创建各种变换矩阵的静态方法
pub struct TransformFactory;

impl TransformFactory {
    /// 创建绕任意轴旋转的变换矩阵
    pub fn rotation(axis: &Vector3<f32>, angle_rad: f32) -> Matrix4<f32> {
        let axis_unit = Unit::new_normalize(*axis);
        Matrix4::from(Rotation3::from_axis_angle(&axis_unit, angle_rad))
    }

    /// 创建绕X轴旋转的变换矩阵
    pub fn rotation_x(angle_rad: f32) -> Matrix4<f32> {
        Self::rotation(&Vector3::x_axis(), angle_rad)
    }

    /// 创建绕Y轴旋转的变换矩阵
    pub fn rotation_y(angle_rad: f32) -> Matrix4<f32> {
        Self::rotation(&Vector3::y_axis(), angle_rad)
    }

    /// 创建绕Z轴旋转的变换矩阵
    pub fn rotation_z(angle_rad: f32) -> Matrix4<f32> {
        Self::rotation(&Vector3::z_axis(), angle_rad)
    }

    /// 创建平移矩阵
    pub fn translation(translation: &Vector3<f32>) -> Matrix4<f32> {
        Matrix4::new_translation(translation)
    }

    /// 创建平移矩阵（从点）
    pub fn translation_from_point(position: &Point3<f32>) -> Matrix4<f32> {
        Self::translation(&position.coords)
    }

    /// 创建均匀缩放矩阵
    pub fn scaling(scale: f32) -> Matrix4<f32> {
        Matrix4::new_scaling(scale)
    }

    /// 创建非均匀缩放矩阵
    pub fn scaling_nonuniform(scale: &Vector3<f32>) -> Matrix4<f32> {
        Matrix4::new_nonuniform_scaling(scale)
    }

    /// 创建视图矩阵 (lookAt)
    pub fn view(eye: &Point3<f32>, target: &Point3<f32>, up: &Vector3<f32>) -> Matrix4<f32> {
        Matrix4::look_at_rh(eye, target, &Unit::new_normalize(*up))
    }

    /// 创建透视投影矩阵
    pub fn perspective(aspect_ratio: f32, fov_y_rad: f32, near: f32, far: f32) -> Matrix4<f32> {
        Matrix4::new_perspective(aspect_ratio, fov_y_rad, near, far)
    }

    /// 创建正交投影矩阵
    pub fn orthographic(
        left: f32,
        right: f32,
        bottom: f32,
        top: f32,
        near: f32,
        far: f32,
    ) -> Matrix4<f32> {
        Matrix4::new_orthographic(left, right, bottom, top, near, far)
    }

    /// 创建完整的模型-视图-投影矩阵
    pub fn model_view_projection(
        model: &Matrix4<f32>,
        view: &Matrix4<f32>,
        projection: &Matrix4<f32>,
    ) -> Matrix4<f32> {
        projection * view * model
    }

    /// 创建模型-视图矩阵
    pub fn model_view(model: &Matrix4<f32>, view: &Matrix4<f32>) -> Matrix4<f32> {
        view * model
    }

    /// 从位置创建平移矩阵的便捷方法
    pub fn position_matrix(position: Point3<f32>) -> Matrix4<f32> {
        Self::translation(&position.coords)
    }

    /// 设置矩阵的平移部分
    pub fn set_translation(matrix: &mut Matrix4<f32>, translation: &Vector3<f32>) {
        matrix.m14 = translation.x;
        matrix.m24 = translation.y;
        matrix.m34 = translation.z;
    }

    /// 从矩阵提取平移部分
    pub fn extract_translation(matrix: &Matrix4<f32>) -> Vector3<f32> {
        Vector3::new(matrix.m14, matrix.m24, matrix.m34)
    }

    /// 从矩阵提取位置
    pub fn extract_position(matrix: &Matrix4<f32>) -> Point3<f32> {
        Point3::new(matrix.m14, matrix.m24, matrix.m34)
    }
}

//------------------------ 第2层：基础变换函数 ------------------------//

/// 计算法线变换矩阵（模型-视图矩阵的逆转置）
#[inline]
pub fn compute_normal_matrix(model_view_matrix: &Matrix4<f32>) -> Matrix3<f32> {
    model_view_matrix.try_inverse().map_or_else(
        || {
            eprintln!("警告：模型-视图矩阵不可逆，使用单位矩阵代替法线矩阵。");
            Matrix3::identity()
        },
        |inv| inv.transpose().fixed_view::<3, 3>(0, 0).into_owned(),
    )
}

/// 将3D点从一个空间变换到另一个空间
#[inline]
pub fn transform_point(point: &Point3<f32>, matrix: &Matrix4<f32>) -> Point3<f32> {
    let homogeneous_point = point.to_homogeneous();
    let transformed_homogeneous = matrix * homogeneous_point;

    if transformed_homogeneous.w.abs() < 1e-9 {
        Point3::new(
            transformed_homogeneous.x,
            transformed_homogeneous.y,
            transformed_homogeneous.z,
        )
    } else {
        Point3::new(
            transformed_homogeneous.x / transformed_homogeneous.w,
            transformed_homogeneous.y / transformed_homogeneous.w,
            transformed_homogeneous.z / transformed_homogeneous.w,
        )
    }
}

/// 将法线向量从一个空间变换到另一个空间
#[inline]
pub fn transform_normal(normal: &Vector3<f32>, normal_matrix: &Matrix3<f32>) -> Vector3<f32> {
    (normal_matrix * normal).normalize()
}

/// 应用透视除法 (将裁剪空间坐标转换为NDC坐标)
#[inline]
pub fn apply_perspective_division(clip: &Vector4<f32>) -> Point3<f32> {
    let w = clip.w;
    if w.abs() > 1e-6 {
        Point3::new(clip.x / w, clip.y / w, clip.z / w)
    } else {
        Point3::origin()
    }
}

/// 将NDC坐标转换为屏幕像素坐标
#[inline]
pub fn ndc_to_screen(ndc_x: f32, ndc_y: f32, width: f32, height: f32) -> Point2<f32> {
    Point2::new(
        (ndc_x * 0.5 + 0.5) * width,
        (1.0 - (ndc_y * 0.5 + 0.5)) * height,
    )
}

/// 将NDC点转换为屏幕坐标
#[inline]
pub fn ndc_point_to_screen(ndc: &Point3<f32>, width: f32, height: f32) -> Point2<f32> {
    ndc_to_screen(ndc.x, ndc.y, width, height)
}

//------------------------ 第3层：组合变换函数 ------------------------//

/// 将裁剪空间点转换为屏幕像素坐标
#[inline]
pub fn clip_to_screen(clip: &Vector4<f32>, width: f32, height: f32) -> Point2<f32> {
    let ndc = apply_perspective_division(clip);
    ndc_point_to_screen(&ndc, width, height)
}

/// 将点转换为齐次裁剪坐标
#[inline]
pub fn point_to_clip(point: &Point3<f32>, matrix: &Matrix4<f32>) -> Vector4<f32> {
    matrix * point.to_homogeneous()
}

/// 将点从模型空间直接变换到屏幕空间
#[inline]
pub fn transform_point_to_screen(
    point: &Point3<f32>,
    model_view_projection: &Matrix4<f32>,
    width: f32,
    height: f32,
) -> Point2<f32> {
    let clip = point_to_clip(point, model_view_projection);
    clip_to_screen(&clip, width, height)
}

//------------------------ 第4层：批量变换函数 ------------------------//

/// 对点集合应用变换矩阵
pub fn transform_points(points: &[Point3<f32>], matrix: &Matrix4<f32>) -> Vec<Point3<f32>> {
    points
        .iter()
        .map(|point| transform_point(point, matrix))
        .collect()
}

/// 将法线向量批量从一个空间变换到另一个空间
pub fn transform_normals(
    normals: &[Vector3<f32>],
    normal_matrix: &Matrix3<f32>,
) -> Vec<Vector3<f32>> {
    normals
        .iter()
        .map(|normal| transform_normal(normal, normal_matrix))
        .collect()
}

/// 将世界坐标点转换为裁剪空间坐标（齐次坐标）
pub fn world_to_clip(
    world_points: &[Point3<f32>],
    view_projection_matrix: &Matrix4<f32>,
) -> Vec<Vector4<f32>> {
    world_points
        .iter()
        .map(|point| point_to_clip(point, view_projection_matrix))
        .collect()
}

/// 将裁剪空间坐标转换为NDC坐标
pub fn clip_to_ndc(clip_coords: &[Vector4<f32>]) -> Vec<Point3<f32>> {
    clip_coords.iter().map(apply_perspective_division).collect()
}

/// 将NDC坐标转换为屏幕像素坐标
pub fn ndc_to_pixel(ndc_coords: &[Point3<f32>], width: f32, height: f32) -> Vec<Point2<f32>> {
    ndc_coords
        .iter()
        .map(|ndc| ndc_point_to_screen(ndc, width, height))
        .collect()
}

/// 将世界坐标直接转换为NDC坐标
pub fn world_to_ndc(
    world_points: &[Point3<f32>],
    view_projection_matrix: &Matrix4<f32>,
) -> Vec<Point3<f32>> {
    let clip_coords = world_to_clip(world_points, view_projection_matrix);
    clip_to_ndc(&clip_coords)
}

/// 将世界坐标点直接转换为屏幕坐标
pub fn world_to_screen(
    world_points: &[Point3<f32>],
    view_projection_matrix: &Matrix4<f32>,
    width: f32,
    height: f32,
) -> Vec<Point2<f32>> {
    let ndc_coords = world_to_ndc(world_points, view_projection_matrix);
    ndc_to_pixel(&ndc_coords, width, height)
}

//------------------------ 第5层：完整渲染管线变换 ------------------------//

/// 执行完整的渲染管线变换（模型→视图→裁剪→NDC→屏幕）
/// 智能选择版本：自动使用并行版本（如果可用）或串行版本

/// 串行版本：适用于小数据量或调试
fn transform_pipeline_batch_serial(
    vertices_model: &[Point3<f32>],
    normals_model: &[Vector3<f32>],
    model_matrix: &Matrix4<f32>,
    view_matrix: &Matrix4<f32>,
    projection_matrix: &Matrix4<f32>,
    screen_width: usize,
    screen_height: usize,
) -> (Vec<Point2<f32>>, Vec<Point3<f32>>, Vec<Vector3<f32>>) {
    // 预计算变换矩阵 - 使用工厂方法
    let model_view = TransformFactory::model_view(model_matrix, view_matrix);
    let mvp = TransformFactory::model_view_projection(model_matrix, view_matrix, projection_matrix);
    let normal_matrix = compute_normal_matrix(&model_view);

    // 批量变换到视图空间 - 复用第4层函数
    let view_positions = transform_points(vertices_model, &model_view);

    // 批量变换到屏幕空间 - 直接使用MVP矩阵，更高效
    let screen_coords = vertices_model
        .iter()
        .map(|vertex| {
            transform_point_to_screen(vertex, &mvp, screen_width as f32, screen_height as f32)
        })
        .collect::<Vec<Point2<f32>>>();

    // 变换法线 - 复用第4层函数
    let view_normals = transform_normals(normals_model, &normal_matrix);

    (screen_coords, view_positions, view_normals)
}

pub fn transform_pipeline_batch_parallel(
    vertices_model: &[Point3<f32>],
    normals_model: &[Vector3<f32>],
    model_matrix: &Matrix4<f32>,
    view_matrix: &Matrix4<f32>,
    projection_matrix: &Matrix4<f32>,
    screen_width: usize,
    screen_height: usize,
) -> (Vec<Point2<f32>>, Vec<Point3<f32>>, Vec<Vector3<f32>>) {
    use rayon::prelude::*;

    // 预计算变换矩阵
    let model_view = TransformFactory::model_view(model_matrix, view_matrix);
    let mvp = TransformFactory::model_view_projection(model_matrix, view_matrix, projection_matrix);
    let normal_matrix = compute_normal_matrix(&model_view);

    // 并行变换 - 根据数据量智能选择并行策略
    let vertex_count = vertices_model.len();

    // 对于小数据量，并行开销可能大于收益，使用串行
    if vertex_count < 1000 {
        return transform_pipeline_batch_serial(
            vertices_model,
            normals_model,
            model_matrix,
            view_matrix,
            projection_matrix,
            screen_width,
            screen_height,
        );
    }

    // 大数据量使用并行处理
    let view_positions = vertices_model
        .par_iter()
        .map(|vertex| transform_point(vertex, &model_view))
        .collect();

    let screen_coords = vertices_model
        .par_iter()
        .map(|vertex| {
            transform_point_to_screen(vertex, &mvp, screen_width as f32, screen_height as f32)
        })
        .collect();

    let view_normals = normals_model
        .par_iter()
        .map(|normal| transform_normal(normal, &normal_matrix))
        .collect();

    (screen_coords, view_positions, view_normals)
}
