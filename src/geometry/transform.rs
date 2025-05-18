use nalgebra::{Matrix3, Matrix4, Point3, Rotation3, Unit, Vector3, Vector4};

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
        Matrix4::from_euler_angles(angle_rad, 0.0, 0.0)
    }
    /// 创建绕Y轴旋转的变换矩阵
    pub fn rotation_y(angle_rad: f32) -> Matrix4<f32> {
        Matrix4::from_euler_angles(0.0, angle_rad, 0.0)
    }

    /// 创建绕Z轴旋转的变换矩阵
    pub fn rotation_z(angle_rad: f32) -> Matrix4<f32> {
        Matrix4::from_euler_angles(0.0, 0.0, angle_rad)
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

// 以下是统一的坐标变换函数

/// 计算法线变换矩阵（模型-视图矩阵的逆转置）
pub fn compute_normal_matrix(model_view_matrix: &Matrix4<f32>) -> Matrix3<f32> {
    model_view_matrix.try_inverse().map_or_else(
        || {
            eprintln!("警告：模型-视图矩阵不可逆，使用单位矩阵代替法线矩阵。");
            Matrix3::identity()
        },
        |inv| inv.transpose().fixed_view::<3, 3>(0, 0).into_owned(),
    )
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

/// 将裁剪空间坐标转换为NDC坐标（透视除法）
pub fn clip_to_ndc(clip_coords: &[Vector4<f32>]) -> Vec<Point3<f32>> {
    clip_coords
        .iter()
        .map(|clip| {
            let w = clip.w;
            if w.abs() > 1e-8 {
                Point3::new(clip.x / w, clip.y / w, clip.z / w)
            } else {
                Point3::origin() // 避免除以零
            }
        })
        .collect()
}

/// 将NDC坐标转换为屏幕像素坐标
pub fn ndc_to_pixel(ndc_coords: &[Point3<f32>], width: f32, height: f32) -> Vec<Point3<f32>> {
    ndc_coords
        .iter()
        .map(|ndc| {
            let screen_x = (ndc.x + 1.0) * 0.5 * width;
            // 翻转Y轴：NDC中+1是顶部，屏幕坐标中0是顶部
            let screen_y = (1.0 - (ndc.y + 1.0) * 0.5) * height;
            Point3::new(screen_x, screen_y, ndc.z)
        })
        .collect()
}

/// 将法线向量从一个空间变换到另一个空间
pub fn transform_normals(
    normals: &[Vector3<f32>],
    normal_matrix: &Matrix3<f32>,
) -> Vec<Vector3<f32>> {
    normals
        .iter()
        .map(|normal| (normal_matrix * normal).normalize())
        .collect()
}

/// 将世界坐标直接转换为NDC坐标（组合了world_to_clip和clip_to_ndc）
pub fn world_to_ndc(
    world_points: &[Point3<f32>],
    view_projection_matrix: &Matrix4<f32>,
) -> Vec<Point3<f32>> {
    let clip_coords = world_to_clip(world_points, view_projection_matrix);
    clip_to_ndc(&clip_coords)
}

/// 将世界坐标直接转换为屏幕坐标（组合了多个变换步骤）
pub fn world_to_screen(
    world_points: &[Point3<f32>],
    view_projection_matrix: &Matrix4<f32>,
    width: f32,
    height: f32,
) -> Vec<Point3<f32>> {
    let ndc_coords = world_to_ndc(world_points, view_projection_matrix);
    ndc_to_pixel(&ndc_coords, width, height)
}
