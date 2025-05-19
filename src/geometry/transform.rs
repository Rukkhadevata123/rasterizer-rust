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
        // Matrix4::from_euler_angles(angle_rad, 0.0, 0.0)
        Self::rotation(&Vector3::x_axis(), angle_rad)
    }

    /// 创建绕Y轴旋转的变换矩阵
    pub fn rotation_y(angle_rad: f32) -> Matrix4<f32> {
        // Matrix4::from_euler_angles(0.0, angle_rad, 0.0)
        Self::rotation(&Vector3::y_axis(), angle_rad)
    }

    /// 创建绕Z轴旋转的变换矩阵
    pub fn rotation_z(angle_rad: f32) -> Matrix4<f32> {
        // Matrix4::from_euler_angles(0.0, 0.0, angle_rad)
        Self::rotation(&Vector3::z_axis(), angle_rad)
    }

    /// 创建平移矩阵
    pub fn translation(translation: &Vector3<f32>) -> Matrix4<f32> {
        Matrix4::new_translation(translation)
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
}

//------------------------ 第1层：基础变换函数 ------------------------//

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
    let homogeneous_point = point.to_homogeneous(); // 转换为齐次坐标 (x, y, z, 1)
    let transformed_homogeneous = matrix * homogeneous_point;

    // 对于仿射变换（如模型、视图变换），w分量应为1
    // Point3::from_homogeneous 会处理 w 分量（如果它不是1，会进行透视除法）
    // 如果 w 接近0，表示点在无穷远处，这里根据具体情况处理或返回原点/错误
    if transformed_homogeneous.w.abs() < 1e-9 {
        // 避免除以零
        // 对于模型/视图变换，w通常是1。如果w是0，则点在无穷远处。
        // 返回非齐次部分可能是一个备选方案，或者根据上下文处理。
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
        Point3::origin() // 避免除以零
    }
}

/// 将NDC坐标转换为屏幕像素坐标
#[inline]
pub fn ndc_to_screen(ndc_x: f32, ndc_y: f32, width: f32, height: f32) -> Point2<f32> {
    Point2::new(
        (ndc_x * 0.5 + 0.5) * width,
        (1.0 - (ndc_y * 0.5 + 0.5)) * height, // Y轴翻转，因为NDC的Y向上，屏幕Y向下
    )
}

//------------------------ 第2层：组合变换函数 ------------------------//

/// 将裁剪空间点转换为屏幕像素坐标
#[inline]
pub fn clip_to_screen(clip: &Vector4<f32>, width: f32, height: f32) -> Point2<f32> {
    let ndc = apply_perspective_division(clip);
    ndc_to_screen(ndc.x, ndc.y, width, height)
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

/// 将点从模型空间直接变换到屏幕空间
#[inline]
pub fn transform_point_to_screen(
    point: &Point3<f32>,
    model_view_projection: &Matrix4<f32>,
    width: f32,
    height: f32,
) -> Point2<f32> {
    let clip = model_view_projection * point.to_homogeneous();
    clip_to_screen(&clip, width, height)
}

//------------------------ 第3层：批量变换函数 ------------------------//

/// 对点集合应用变换矩阵
pub fn transform_points(points: &[Point3<f32>], matrix: &Matrix4<f32>) -> Vec<Point3<f32>> {
    points
        .iter()
        .map(|point| transform_point(point, matrix))
        .collect()
}

/// 将世界坐标点转换为裁剪空间坐标（齐次坐标）
pub fn world_to_clip(
    world_points: &[Point3<f32>],
    view_projection_matrix: &Matrix4<f32>,
) -> Vec<Vector4<f32>> {
    world_points
        .iter()
        .map(|point| view_projection_matrix * point.to_homogeneous())
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
        .map(|ndc| ndc_to_screen(ndc.x, ndc.y, width, height))
        .collect()
}

//------------------------ 第4层：完整渲染管线变换 ------------------------//

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
    let clip_coords = world_to_clip(world_points, view_projection_matrix);
    let ndc_coords = clip_to_ndc(&clip_coords);
    ndc_to_pixel(&ndc_coords, width, height)
}

/// 执行完整的渲染管线变换（模型→视图→裁剪→NDC→屏幕）
/// 此版本使用了优化的直接变换，适合于光栅化渲染器
pub fn transform_pipeline_batch(
    vertices_model: &[Point3<f32>],
    normals_model: &[Vector3<f32>],
    model_matrix: &Matrix4<f32>,
    view_matrix: &Matrix4<f32>,
    projection_matrix: &Matrix4<f32>,
    screen_width: usize,
    screen_height: usize,
) -> (Vec<Point2<f32>>, Vec<Point3<f32>>, Vec<Vector3<f32>>) {
    // 预计算变换矩阵
    let model_view = view_matrix * model_matrix;
    let mvp = projection_matrix * model_view;
    let normal_matrix = compute_normal_matrix(&model_view);

    // 预分配结果向量
    let mut screen_coords = Vec::with_capacity(vertices_model.len());
    let mut view_coords = Vec::with_capacity(vertices_model.len());

    // 使用直接变换进行性能优化
    for vertex in vertices_model {
        // 变换到裁剪空间
        let clip = mvp * vertex.to_homogeneous();

        // 转换到屏幕空间 - 使用第3层函数
        let pixel = clip_to_screen(&clip, screen_width as f32, screen_height as f32);
        screen_coords.push(pixel);

        // 计算视图空间坐标 - 使用第2层函数
        let view_pos = transform_point(vertex, &model_view); // model_view 变换到视图空间
        view_coords.push(view_pos);
    }

    // 变换法线 - 使用第3层函数
    let view_normals = transform_normals(normals_model, &normal_matrix);

    (screen_coords, view_coords, view_normals)
}
