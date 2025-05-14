//! # 三角形光栅化器
//!
//! 本模块实现了一个基于三角形的软件光栅化器，支持以下特性：
//! - 基于重心坐标的插值
//! - 深度缓冲和深度测试
//! - 多种着色模型：平面着色(Flat)、Gouraud着色和Phong着色
//! - 纹理映射与透视校正插值
//! - Blinn-Phong光照模型
//! - Gamma校正
//!
//! 光栅化器使用原子操作处理深度缓冲以支持并行渲染，使用互斥锁保护颜色缓冲区。

use crate::color_utils::{Color, linear_rgb_to_u8, srgb_to_linear};
use crate::interpolation::{
    barycentric_coordinates, interpolate_depth, interpolate_normal, interpolate_position,
    interpolate_texcoords, is_inside_triangle,
};
use crate::lighting::{Light, SimpleMaterial, calculate_blinn_phong};
use crate::material_system::IMaterial; // 添加 IMaterial trait 导入
use crate::texture_utils::Texture;
use atomic_float::AtomicF32;
use nalgebra::{Point2, Point3, Vector2, Vector3};
use std::sync::Mutex;
use std::sync::atomic::Ordering;

/// 单个三角形光栅化所需的输入数据
///
/// 包含三角形的几何信息（顶点位置、法线）、材质属性、纹理坐标和光照信息。
/// 该结构体的字段分为几个逻辑组：
/// - 屏幕空间坐标 (v*_pix)
/// - 视图空间深度值 (z*_view)
/// - 颜色与光照属性
/// - 纹理相关属性
/// - 渲染设置
/// - Phong着色所需的额外数据
pub struct TriangleData<'a> {
    // 屏幕空间坐标 (像素坐标)
    pub v1_pix: Point2<f32>, // 第一个顶点的屏幕坐标 (x,y)
    pub v2_pix: Point2<f32>, // 第二个顶点的屏幕坐标 (x,y)
    pub v3_pix: Point2<f32>, // 第三个顶点的屏幕坐标 (x,y)

    // 视图空间 Z 值 - 用于深度测试和透视校正插值
    pub z1_view: f32, // 第一个顶点在视图空间的 z 坐标（通常为负值）
    pub z2_view: f32, // 第二个顶点在视图空间的 z 坐标
    pub z3_view: f32, // 第三个顶点在视图空间的 z 坐标

    // 颜色与光照属性
    pub base_color: Color, // 基础颜色（来自材质的漫反射颜色或面颜色）
    pub lit_color: Color,  // 预计算的光照贡献值（用于非 Phong 着色模式）

    // 纹理相关属性
    pub tc1: Option<Vector2<f32>>, // 第一个顶点的纹理坐标 (u,v)，可能不存在
    pub tc2: Option<Vector2<f32>>, // 第二个顶点的纹理坐标 (u,v)
    pub tc3: Option<Vector2<f32>>, // 第三个顶点的纹理坐标 (u,v)
    pub texture: Option<&'a Texture>, // 三角形使用的纹理引用，借用生命周期为 'a

    // 渲染设置
    pub is_perspective: bool, // 是否使用透视投影（影响插值方式）

    // Phong 着色所需的额外数据
    pub n1_view: Option<Vector3<f32>>, // 第一个顶点在视图空间的法线向量
    pub n2_view: Option<Vector3<f32>>, // 第二个顶点在视图空间的法线向量
    pub n3_view: Option<Vector3<f32>>, // 第三个顶点在视图空间的法线向量

    pub v1_view: Option<Point3<f32>>, // 第一个顶点在视图空间的完整坐标 (x,y,z)
    pub v2_view: Option<Point3<f32>>, // 第二个顶点在视图空间的完整坐标
    pub v3_view: Option<Point3<f32>>, // 第三个顶点在视图空间的完整坐标

    pub material: Option<SimpleMaterial>, // 三角形材质属性（环境光、漫反射、高光等）
    pub light: Option<Light>,             // 光源信息（位置、强度、颜色等）

    pub use_phong: bool, // 是否对此三角形使用 Phong 着色（逐像素光照计算）
    // 若为 false，则使用 lit_color（预计算的面或顶点光照）

    // 添加 PBR 材质支持
    pub pbr_material: Option<&'a crate::material_system::PBRMaterial>, // PBR 材料引用
}

/// 光栅化器配置参数
///
/// 控制光栅化过程中的各种功能开关，包括深度测试、光照计算、透视校正等。
#[derive(Debug, Clone, Copy)]
pub struct RasterizerConfig {
    /// 是否启用深度缓冲和深度测试
    pub use_zbuffer: bool,
    /// 是否应用光照计算
    pub use_lighting: bool,
    /// 是否使用透视校正插值
    pub use_perspective: bool,
    /// 是否使用Phong着色（逐像素光照计算）
    pub use_phong: bool,
    /// 是否使用基于物理的渲染 (PBR)
    pub use_pbr: bool,
    /// 是否使用纹理映射
    pub use_texture: bool,
    /// 是否应用gamma校正（sRGB空间转换）
    pub apply_gamma_correction: bool,
}

impl Default for RasterizerConfig {
    fn default() -> Self {
        Self {
            use_zbuffer: true,
            use_lighting: true,
            use_perspective: true,
            use_phong: false,
            use_pbr: false,
            use_texture: true,
            apply_gamma_correction: true,
        }
    }
}

/// 光栅化单个三角形到帧缓冲区
///
/// 该函数实现了三角形光栅化的核心算法，包括：
/// 1. 计算三角形包围盒
/// 2. 对包围盒中的每个像素进行处理
/// 3. 计算重心坐标，判断像素是否在三角形内
/// 4. 对于三角形内的像素，进行深度测试
/// 5. 计算最终颜色（基于着色模型、纹理和光照）
/// 6. 写入颜色到帧缓冲区
///
/// # 参数
/// * `triangle` - 包含三角形数据的结构体
/// * `width` - 帧缓冲区宽度（像素）
/// * `height` - 帧缓冲区高度（像素）
/// * `depth_buffer` - 深度缓冲区（使用原子操作支持并行）
/// * `color_buffer` - 颜色缓冲区（使用互斥锁保护）
/// * `config` - 光栅化器配置参数
pub fn rasterize_triangle(
    triangle: &TriangleData,
    width: usize,
    height: usize,
    depth_buffer: &[AtomicF32],
    color_buffer: &Mutex<Vec<u8>>,
    config: &RasterizerConfig,
) {
    // 1. 计算三角形包围盒
    let min_x = triangle
        .v1_pix
        .x
        .min(triangle.v2_pix.x)
        .min(triangle.v3_pix.x)
        .floor()
        .max(0.0) as usize;
    let min_y = triangle
        .v1_pix
        .y
        .min(triangle.v2_pix.y)
        .min(triangle.v3_pix.y)
        .floor()
        .max(0.0) as usize;
    let max_x = triangle
        .v1_pix
        .x
        .max(triangle.v2_pix.x)
        .max(triangle.v3_pix.x)
        .ceil()
        .min(width as f32) as usize;
    let max_y = triangle
        .v1_pix
        .y
        .max(triangle.v2_pix.y)
        .max(triangle.v3_pix.y)
        .ceil()
        .min(height as f32) as usize;

    // 检查无效的包围盒（宽度或高度为0）
    if max_x <= min_x || max_y <= min_y {
        return;
    }

    // 可选：对面积过小的三角形提前退出
    // let area_x2 = ((triangle.v2_pix.x - triangle.v1_pix.x) * (triangle.v3_pix.y - triangle.v1_pix.y)
    //             - (triangle.v3_pix.x - triangle.v1_pix.x) * (triangle.v2_pix.y - triangle.v1_pix.y)).abs();
    // if area_x2 < 1e-3 { return; }

    // 2. 遍历包围盒中的每个像素
    for y in min_y..max_y {
        for x in min_x..max_x {
            // 计算像素中心点坐标
            let pixel_center = Point2::new(x as f32 + 0.5, y as f32 + 0.5);
            let pixel_index = y * width + x;

            // 3. 计算重心坐标
            if let Some(bary) = barycentric_coordinates(
                pixel_center,
                triangle.v1_pix,
                triangle.v2_pix,
                triangle.v3_pix,
            ) {
                // 4. 检查像素是否在三角形内
                if is_inside_triangle(bary) {
                    // 5. 插值深度值
                    let interpolated_depth = interpolate_depth(
                        bary,
                        triangle.z1_view,
                        triangle.z2_view,
                        triangle.z3_view,
                        config.use_perspective && triangle.is_perspective,
                    );

                    // 检查深度是否有效（不在相机后方且不太远）
                    if interpolated_depth.is_finite() && interpolated_depth < f32::INFINITY {
                        // 6. 深度测试（使用原子操作）
                        let current_depth_atomic = &depth_buffer[pixel_index];
                        let previous_depth = current_depth_atomic.load(Ordering::Relaxed);

                        // 根据配置决定是否执行深度测试
                        if !config.use_zbuffer || interpolated_depth < previous_depth {
                            // 尝试原子更新深度值
                            // fetch_min返回更新前的值
                            let old_depth_before_update = current_depth_atomic
                                .fetch_min(interpolated_depth, Ordering::Relaxed);

                            // 只有当当前线程成功更新了深度值时才写入颜色
                            if !config.use_zbuffer || old_depth_before_update > interpolated_depth {
                                // 7. 计算最终颜色
                                let final_color = calculate_pixel_color(triangle, bary, config);

                                // 8. 写入颜色到缓冲区（使用互斥锁）
                                {
                                    // 互斥锁作用域
                                    let mut cbuf_guard = color_buffer.lock().unwrap();
                                    let buffer_start_index = pixel_index * 3;
                                    if buffer_start_index + 2 < cbuf_guard.len() {
                                        // 使用gamma校正函数转换颜色
                                        let [r, g, b] = linear_rgb_to_u8(
                                            &final_color,
                                            config.apply_gamma_correction,
                                        );
                                        cbuf_guard[buffer_start_index] = r;
                                        cbuf_guard[buffer_start_index + 1] = g;
                                        cbuf_guard[buffer_start_index + 2] = b;
                                    }
                                } // 互斥锁在此处释放
                            }
                        }
                    }
                }
            }
        }
    }
}

/// 计算像素的最终颜色值
///
/// 根据三角形数据、重心坐标和配置参数计算像素颜色。
/// 处理三种主要的着色模式：
/// 1. PBR 着色（基于物理的渲染）
/// 2. Phong着色（逐像素光照计算）
/// 3. 预计算光照（Flat或Gouraud着色）
///
/// # 参数
/// * `triangle` - 三角形数据
/// * `bary` - 像素的重心坐标
/// * `config` - 光栅化器配置
///
/// # 返回值
/// 计算得到的像素颜色（线性RGB空间）
fn calculate_pixel_color(
    triangle: &TriangleData,
    bary: Vector3<f32>,
    config: &RasterizerConfig,
) -> Color {
    // 首先检查是否使用 PBR 渲染
    if config.use_pbr && triangle.pbr_material.is_some() && triangle.light.is_some() {
        // --- PBR 着色（基于物理的渲染）---
        // 获取 PBR 材质和光源
        let pbr_material = triangle.pbr_material.unwrap();
        let light = triangle.light.as_ref().unwrap();

        // 确保有法线和位置数据
        if triangle.n1_view.is_none() || triangle.v1_view.is_none() {
            // 如果缺少法线或位置数据，回退到基本着色
            return if config.use_lighting {
                triangle.base_color.component_mul(&triangle.lit_color)
            } else {
                triangle.base_color
            };
        }

        // 插值法线
        let interp_normal = interpolate_normal(
            bary,
            triangle.n1_view.unwrap(),
            triangle.n2_view.unwrap(),
            triangle.n3_view.unwrap(),
            triangle.is_perspective,
            triangle.z1_view,
            triangle.z2_view,
            triangle.z3_view,
        );

        // 插值视图空间位置
        let interp_position = interpolate_position(
            bary,
            triangle.v1_view.unwrap(),
            triangle.v2_view.unwrap(),
            triangle.v3_view.unwrap(),
            triangle.is_perspective,
            triangle.z1_view,
            triangle.z2_view,
            triangle.z3_view,
        );

        // 计算视线方向
        let view_dir = (-interp_position.coords).normalize();

        // 根据光源类型计算光照方向和强度
        let (light_dir, light_intensity) = match light {
            Light::Directional {
                direction,
                intensity,
            } => (direction.normalize(), *intensity),
            Light::Point {
                position,
                intensity,
                attenuation,
            } => {
                let dir_to_light = (*position - interp_position).normalize();
                let distance = (*position - interp_position).magnitude();

                // 计算衰减
                let (a, b, c) = *attenuation;
                let attenuation_factor = 1.0 / (a + b * distance + c * distance * distance);

                (dir_to_light, *intensity * attenuation_factor)
            }
            Light::Ambient(intensity) => {
                // 环境光没有明确的方向，使用法线作为方向
                (interp_normal, *intensity)
            }
        };

        // 计算 PBR 光照响应
        let pbr_response = pbr_material.compute_response(&light_dir, &view_dir, &interp_normal);

        // 将 Vector3 转换为 Color 并应用光照强度
        let pbr_color = Color::new(
            pbr_response.x * light_intensity.x,
            pbr_response.y * light_intensity.y,
            pbr_response.z * light_intensity.z,
        );

        // 处理纹理（如果有）
        if should_use_texture(triangle, config) {
            let texel_color = sample_texture(triangle, bary, config);

            // 将纹理颜色与 PBR 光照结果相乘
            if config.use_lighting {
                texel_color.component_mul(&pbr_color)
            } else {
                texel_color
            }
        } else {
            // 无纹理，直接使用 PBR 光照结果
            if config.use_lighting {
                let base_pbr_color = Color::new(
                    pbr_material.base_color.x,
                    pbr_material.base_color.y,
                    pbr_material.base_color.z,
                );
                base_pbr_color.component_mul(&pbr_color)
            } else {
                Color::new(
                    pbr_material.base_color.x,
                    pbr_material.base_color.y,
                    pbr_material.base_color.z,
                )
            }
        }
    }
    // 判断是否使用Phong着色
    else if config.use_phong
        && triangle.use_phong
        && triangle.n1_view.is_some()
        && triangle.material.is_some()
        && triangle.light.is_some()
        && triangle.v1_view.is_some()
    {
        // --- Phong着色（逐像素光照）---

        // 插值法线
        let interp_normal = interpolate_normal(
            bary,
            triangle.n1_view.unwrap(),
            triangle.n2_view.unwrap(),
            triangle.n3_view.unwrap(),
            triangle.is_perspective,
            triangle.z1_view,
            triangle.z2_view,
            triangle.z3_view,
        );

        // 插值视图空间位置
        let interp_position = interpolate_position(
            bary,
            triangle.v1_view.unwrap(),
            triangle.v2_view.unwrap(),
            triangle.v3_view.unwrap(),
            triangle.is_perspective,
            triangle.z1_view,
            triangle.z2_view,
            triangle.z3_view,
        );

        // 计算视线方向
        let view_dir = (-interp_position.coords).normalize();

        // 计算Blinn-Phong光照
        let light = triangle.light.as_ref().unwrap();
        let material = triangle.material.as_ref().unwrap();
        let pixel_lit_color =
            calculate_blinn_phong(interp_position, interp_normal, view_dir, light, material);

        // 处理纹理
        if should_use_texture(triangle, config) {
            let texel_color = sample_texture(triangle, bary, config);

            // 应用光照（如果启用）
            if config.use_lighting {
                texel_color.component_mul(&pixel_lit_color)
            } else {
                texel_color
            }
        } else {
            // 无纹理，使用基础颜色
            if config.use_lighting {
                triangle.base_color.component_mul(&pixel_lit_color)
            } else {
                triangle.base_color
            }
        }
    } else {
        // --- 使用预计算的光照（Flat/Gouraud着色）---

        // 处理纹理
        if should_use_texture(triangle, config) {
            let texel_color = sample_texture(triangle, bary, config);

            // 应用光照（如果启用）
            if config.use_lighting {
                texel_color.component_mul(&triangle.lit_color)
            } else {
                texel_color
            }
        } else {
            // 无纹理，使用基础颜色+预计算光照
            if config.use_lighting {
                triangle.base_color.component_mul(&triangle.lit_color)
            } else {
                triangle.base_color
            }
        }
    }
}

/// 检查是否应该对三角形使用纹理
///
/// # 参数
/// * `triangle` - 三角形数据
/// * `config` - 光栅化器配置
///
/// # 返回值
/// 如果应该使用纹理，返回true
fn should_use_texture(triangle: &TriangleData, config: &RasterizerConfig) -> bool {
    config.use_texture
        && triangle.texture.is_some()
        && triangle.tc1.is_some()
        && triangle.tc2.is_some()
        && triangle.tc3.is_some()
}

/// 从纹理中采样并应用gamma校正（如需要）
///
/// # 参数
/// * `triangle` - 三角形数据，包含纹理和纹理坐标
/// * `bary` - 像素的重心坐标
/// * `config` - 光栅化器配置
///
/// # 返回值
/// 采样得到的纹理颜色（线性RGB空间）
fn sample_texture(triangle: &TriangleData, bary: Vector3<f32>, config: &RasterizerConfig) -> Color {
    // 获取三角形的纹理坐标
    let tc1 = triangle.tc1.unwrap();
    let tc2 = triangle.tc2.unwrap();
    let tc3 = triangle.tc3.unwrap();
    let tex = triangle.texture.unwrap();

    // 插值纹理坐标
    let interp_tc = interpolate_texcoords(
        bary,
        tc1,
        tc2,
        tc3,
        triangle.z1_view,
        triangle.z2_view,
        triangle.z3_view,
        config.use_perspective && triangle.is_perspective,
    );

    // 采样纹理
    let texel = tex.sample(interp_tc.x, interp_tc.y);
    let texel_color = Color::new(texel[0], texel[1], texel[2]);

    // 如果需要gamma校正，将sRGB纹理颜色转换为线性空间
    if config.apply_gamma_correction {
        srgb_to_linear(&texel_color)
    } else {
        texel_color
    }
}
