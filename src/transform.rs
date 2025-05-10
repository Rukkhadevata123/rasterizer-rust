use nalgebra::{Matrix3, Matrix4, Point3, Vector3, Vector4};

/// 将世界坐标转换为视图坐标
///
/// # 参数
/// * `world_coords` - 世界坐标中的点坐标数组
/// * `view_matrix` - 视图变换矩阵
///
/// # 返回值
/// 视图坐标中的点坐标数组
pub fn world_to_view(world_coords: &[Point3<f32>], view_matrix: &Matrix4<f32>) -> Vec<Point3<f32>> {
    world_coords
        .iter()
        .map(|world_point| {
            let view_h = view_matrix * world_point.to_homogeneous();
            Point3::from_homogeneous(view_h).unwrap_or_else(|| Point3::origin())
        })
        .collect()
}

/// 将世界坐标转换为裁剪坐标（齐次坐标系统）
///
/// # 参数
/// * `world_coords` - 世界坐标中的点坐标数组
/// * `view_projection_matrix` - 视图投影变换矩阵（视图矩阵与投影矩阵的乘积）
///
/// # 返回值
/// 裁剪坐标（齐次坐标）
pub fn world_to_clip(world_coords: &[Point3<f32>], view_projection_matrix: &Matrix4<f32>) -> Vec<Vector4<f32>> {
    world_coords
        .iter()
        .map(|world_point| {
            view_projection_matrix * world_point.to_homogeneous()
        })
        .collect()
}

/// 将裁剪坐标转换为NDC坐标（透视除法）
///
/// # 参数
/// * `clip_coords` - 裁剪坐标（齐次坐标）
///
/// # 返回值
/// 归一化设备坐标(NDC)
pub fn clip_to_ndc(clip_coords: &[Vector4<f32>]) -> Vec<Point3<f32>> {
    clip_coords
        .iter()
        .map(|clip_h| {
            let w = clip_h.w;
            if w.abs() > 1e-8 {
                Point3::new(clip_h.x / w, clip_h.y / w, clip_h.z / w)
            } else {
                Point3::origin() // 避免除以零
            }
        })
        .collect()
}

/// 计算法线变换矩阵
/// 法线需要使用视图矩阵的逆转置矩阵进行变换，以保持法线垂直于表面
///
/// # 参数
/// * `view_matrix` - 视图变换矩阵
///
/// # 返回值
/// 用于变换法线的3x3矩阵
pub fn compute_normal_matrix(view_matrix: &Matrix4<f32>) -> Matrix3<f32> {
    view_matrix.try_inverse().map_or_else(
        || {
            println!("警告：视图矩阵不可逆，使用单位矩阵代替法线矩阵。");
            Matrix3::identity()
        },
        |inv_view| inv_view.transpose().fixed_view::<3, 3>(0, 0).into_owned(),
    )
}

/// 变换法线向量
///
/// # 参数
/// * `world_normals` - 世界坐标中的法线向量数组
/// * `normal_matrix` - 法线变换矩阵
///
/// # 返回值
/// 变换后的法线向量数组（通常是视图空间中的法线）
pub fn transform_normals(world_normals: &[Vector3<f32>], normal_matrix: &Matrix3<f32>) -> Vec<Vector3<f32>> {
    world_normals
        .iter()
        .map(|normal| {
            (normal_matrix * normal).normalize()
        })
        .collect()
}

/// 将世界坐标直接转换为NDC坐标（结合了world_to_clip和clip_to_ndc）
///
/// # 参数
/// * `world_coords` - 世界坐标中的点坐标数组
/// * `view_projection_matrix` - 视图投影变换矩阵
///
/// # 返回值
/// 归一化设备坐标(NDC)
pub fn world_to_ndc(world_coords: &[Point3<f32>], view_projection_matrix: &Matrix4<f32>) -> Vec<Point3<f32>> {
    // 使用 world_to_clip 函数将世界坐标转换为裁剪坐标
    let clip_coords = world_to_clip(world_coords, view_projection_matrix);
    
    // 使用 clip_to_ndc 函数将裁剪坐标转换为NDC坐标
    clip_to_ndc(&clip_coords)
}

/// 将归一化设备坐标(NDC)转换为屏幕像素坐标。
/// NDC范围假定为[-1, 1]（x和y轴）。
/// 像素坐标原点(0,0)通常在左上角。
///
/// # 参数
/// * `ndc_coords` - NDC空间中的点坐标数组 (x, y, z)。一般不会用到z值，但会保留。
/// * `width` - 目标屏幕/图像的宽度（像素）。
/// * `height` - 目标屏幕/图像的高度（像素）。
///
/// # 返回值
/// 像素空间中的坐标数组。
/// X从[-1, 1]映射到[0, width]。
/// Y从[-1, 1]映射到[height, 0]（Y轴翻转）。
/// Z通常原样传递。
pub fn ndc_to_pixel(ndc_coords: &[Point3<f32>], width: f32, height: f32) -> Vec<Point3<f32>> {
    ndc_coords
        .iter()
        .map(|ndc| {
            let pixel_x = (ndc.x + 1.0) * width / 2.0;
            // 翻转Y轴：NDC中+1是顶部，像素坐标中0是顶部
            let pixel_y = height - (ndc.y + 1.0) * height / 2.0;
            // let pixel_y = (ndc.y + 1.0) * height / 2.0; // 如果Y轴不需要翻转，则使用此行

            Point3::new(pixel_x, pixel_y, ndc.z) // 传递Z值
        })
        .collect()
}
