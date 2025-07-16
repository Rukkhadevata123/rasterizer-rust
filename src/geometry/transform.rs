use log::warn;
use nalgebra::{Matrix3, Matrix4, Point2, Point3, Rotation3, Unit, Vector3, Vector4};

//=================================
// 变换矩阵创建工厂
//=================================

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

    /// 创建缩放矩阵
    pub fn scaling_nonuniform(scale: &Vector3<f32>) -> Matrix4<f32> {
        Matrix4::new_nonuniform_scaling(scale)
    }

    /// 创建视图矩阵 (Look-At)
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

    /// 创建MVP矩阵（Model-View-Projection）
    pub fn model_view_projection(
        model: &Matrix4<f32>,
        view: &Matrix4<f32>,
        projection: &Matrix4<f32>,
    ) -> Matrix4<f32> {
        projection * view * model
    }

    /// 创建MV矩阵（Model-View）
    pub fn model_view(model: &Matrix4<f32>, view: &Matrix4<f32>) -> Matrix4<f32> {
        view * model
    }
}

//=================================
// 核心变换函数
//=================================

/// 计算法线变换矩阵（模型-视图矩阵的逆转置）
///
/// 法线向量需要特殊处理：使用变换矩阵的逆转置来保持垂直性
#[inline]
pub fn compute_normal_matrix(model_view_matrix: &Matrix4<f32>) -> Matrix3<f32> {
    model_view_matrix.try_inverse().map_or_else(
        || {
            warn!("模型-视图矩阵不可逆，使用单位矩阵代替法线矩阵");
            Matrix3::identity()
        },
        |inv| inv.transpose().fixed_view::<3, 3>(0, 0).into_owned(),
    )
}

/// 3D点变换：将点从一个坐标空间变换到另一个坐标空间
///
/// 使用齐次坐标进行变换，最后执行齐次除法得到3D点
#[inline]
pub fn transform_point(point: &Point3<f32>, matrix: &Matrix4<f32>) -> Point3<f32> {
    let homogeneous_point = point.to_homogeneous();
    let transformed_homogeneous = matrix * homogeneous_point;

    // 执行齐次除法，处理w分量
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

/// 法线向量变换：使用法线矩阵变换法线向量并归一化
#[inline]
pub fn transform_normal(normal: &Vector3<f32>, normal_matrix: &Matrix3<f32>) -> Vector3<f32> {
    (normal_matrix * normal).normalize()
}

/// 透视除法：将裁剪空间坐标转换为NDC坐标
///
/// 裁剪空间 → NDC（标准化设备坐标）：除以w分量
#[inline]
pub fn apply_perspective_division(clip: &Vector4<f32>) -> Point3<f32> {
    let w = clip.w;
    if w.abs() > 1e-6 {
        Point3::new(clip.x / w, clip.y / w, clip.z / w)
    } else {
        Point3::origin()
    }
}

/// NDC到屏幕坐标转换
///
/// NDC范围[-1,1] → 屏幕像素坐标[0,width/height]
/// 注意Y轴翻转：NDC的+Y向上，屏幕坐标的+Y向下
#[inline]
pub fn ndc_to_screen(ndc_x: f32, ndc_y: f32, width: f32, height: f32) -> Point2<f32> {
    Point2::new(
        (ndc_x * 0.5 + 0.5) * width,
        (1.0 - (ndc_y * 0.5 + 0.5)) * height,
    )
}

/// 裁剪空间到屏幕坐标的完整转换
///
/// 组合透视除法和视口变换：裁剪空间 → NDC → 屏幕坐标
#[inline]
pub fn clip_to_screen(clip: &Vector4<f32>, width: f32, height: f32) -> Point2<f32> {
    let ndc = apply_perspective_division(clip);
    ndc_to_screen(ndc.x, ndc.y, width, height)
}

/// 点到裁剪坐标的转换
///
/// 将3D点转换为齐次裁剪坐标（用于后续透视除法）
#[inline]
pub fn point_to_clip(point: &Point3<f32>, matrix: &Matrix4<f32>) -> Vector4<f32> {
    matrix * point.to_homogeneous()
}
