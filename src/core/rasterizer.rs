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
//! 光栅化器使用原子操作处理深度缓冲和颜色缓冲区以支持高效的并行渲染。

use crate::core::renderer::RenderConfig; // 直接导入 RenderConfig
use crate::geometry::interpolation::{
    barycentric_coordinates, interpolate_depth, interpolate_normal, interpolate_position,
    interpolate_texcoords, is_inside_triangle,
};
use crate::materials::color::{Color, linear_rgb_to_u8};
use crate::materials::material_system::{Light, MaterialView};
use crate::materials::texture::{Texture, TextureData};
use atomic_float::AtomicF32;
use nalgebra::{Point2, Point3, Vector2, Vector3};
use std::sync::atomic::{AtomicU8, Ordering};

/// 顶点渲染数据，组织单个顶点的所有渲染属性
#[derive(Debug, Clone)]
pub struct VertexRenderData {
    pub pix: Point2<f32>,                   // 屏幕空间坐标 (x,y)
    pub z_view: f32,                        // 视图空间 z 值
    pub texcoord: Option<Vector2<f32>>,     // 纹理坐标
    pub normal_view: Option<Vector3<f32>>,  // 视图空间法线
    pub position_view: Option<Point3<f32>>, // 视图空间位置
}

/// 单个三角形光栅化所需的输入数据
///
/// 包含三角形的几何信息（顶点位置、法线）、材质属性、纹理坐标和光照信息。
/// 该结构体的字段分为几个逻辑组：
/// - 顶点数据 (包含屏幕坐标、深度值、法线、纹理坐标等)
/// - 纹理与材质属性
/// - 渲染设置
pub struct TriangleData<'a> {
    // 三个顶点数据
    pub vertices: [VertexRenderData; 3],

    // 颜色与光照属性
    pub base_color: Color, // 基础颜色
    pub lit_color: Color,  // 预计算的光照贡献值

    // 纹理与材质
    pub texture_data: TextureData,               // 统一的纹理来源
    pub texture_ref: Option<&'a Texture>,        // 可选的纹理引用
    pub material_view: Option<MaterialView<'a>>, // 材质视图

    // 光照信息
    pub light: Option<Light>, // 光源信息

    // 渲染设置
    pub is_perspective: bool, // 是否使用透视投影
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
/// * `color_buffer` - 颜色缓冲区（使用原子操作支持并行）
/// * `config` - 光栅化器配置参数
pub fn rasterize_triangle(
    triangle: &TriangleData,
    width: usize,
    height: usize,
    depth_buffer: &[AtomicF32],
    color_buffer: &[AtomicU8],
    config: &RenderConfig, // 直接使用 RenderConfig
) {
    // 1. 计算三角形包围盒
    let min_x = triangle.vertices[0]
        .pix
        .x
        .min(triangle.vertices[1].pix.x)
        .min(triangle.vertices[2].pix.x)
        .floor()
        .max(0.0) as usize;
    let min_y = triangle.vertices[0]
        .pix
        .y
        .min(triangle.vertices[1].pix.y)
        .min(triangle.vertices[2].pix.y)
        .floor()
        .max(0.0) as usize;
    let max_x = triangle.vertices[0]
        .pix
        .x
        .max(triangle.vertices[1].pix.x)
        .max(triangle.vertices[2].pix.x)
        .ceil()
        .min(width as f32) as usize;
    let max_y = triangle.vertices[0]
        .pix
        .y
        .max(triangle.vertices[1].pix.y)
        .max(triangle.vertices[2].pix.y)
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

    // 线框模式的边缘检测阈值（像素单位）
    const EDGE_THRESHOLD: f32 = 1.0;

    // 2. 遍历包围盒中的每个像素
    for y in min_y..max_y {
        for x in min_x..max_x {
            // 计算像素中心点坐标
            let pixel_center = Point2::new(x as f32 + 0.5, y as f32 + 0.5);
            let pixel_index = y * width + x;

            // 3. 计算重心坐标
            if let Some(bary) = barycentric_coordinates(
                pixel_center,
                triangle.vertices[0].pix,
                triangle.vertices[1].pix,
                triangle.vertices[2].pix,
            ) {
                // 4. 检查像素是否在三角形内
                if is_inside_triangle(bary) {
                    // 如果是线框模式，检查像素是否在三角形边缘附近
                    // 如果不在边缘附近，则不渲染该像素
                    if config.use_wireframe
                        && !is_on_triangle_edge(
                            pixel_center,
                            triangle.vertices[0].pix,
                            triangle.vertices[1].pix,
                            triangle.vertices[2].pix,
                            EDGE_THRESHOLD,
                        )
                    {
                        continue;
                    }

                    // 5. 插值深度值
                    let interpolated_depth = interpolate_depth(
                        bary,
                        triangle.vertices[0].z_view,
                        triangle.vertices[1].z_view,
                        triangle.vertices[2].z_view,
                        config.is_perspective() && triangle.is_perspective,
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

                                // 8. 写入颜色到缓冲区（使用原子操作）
                                {
                                    // 原子操作写入颜色
                                    let buffer_start_index = pixel_index * 3;
                                    if buffer_start_index + 2 < color_buffer.len() {
                                        // 使用gamma校正函数转换颜色
                                        let [r, g, b] = linear_rgb_to_u8(
                                            &final_color,
                                            config.apply_gamma_correction,
                                        );
                                        color_buffer[buffer_start_index]
                                            .store(r, Ordering::Relaxed);
                                        color_buffer[buffer_start_index + 1]
                                            .store(g, Ordering::Relaxed);
                                        color_buffer[buffer_start_index + 2]
                                            .store(b, Ordering::Relaxed);
                                    }
                                }
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
    config: &RenderConfig,
) -> Color {
    // 计算环境光贡献
    let ambient_contribution = calculate_ambient_contribution(triangle, config);

    // 使用传入的基础颜色 - 不再在这里重新判断是否使用面颜色
    // 面颜色已经在渲染器中通过纹理系统处理
    let base_color = triangle.base_color;

    // 首先检查是否使用逐像素着色（PBR或Phong模式）
    if (config.use_pbr || config.use_phong)
        && triangle.vertices[0].normal_view.is_some()
        && triangle.vertices[0].position_view.is_some()
        && triangle.light.is_some()
    {
        // --- 使用材质视图进行PBR或Phong着色 ---
        // 确保有法线和位置数据
        if triangle.vertices[0].normal_view.is_none()
            || triangle.vertices[0].position_view.is_none()
        {
            // 如果缺少法线或位置数据，回退到基本着色
            return get_base_color_with_ambient(triangle, base_color, config);
        }

        // 获取必要的数据
        let light = triangle.light.as_ref().unwrap();

        // 创建适当的材质视图
        let material_view = if config.use_pbr && triangle.material_view.is_some() {
            triangle.material_view.as_ref().unwrap()
        } else if triangle.material_view.is_some() {
            // 这里直接使用已有的material_view，无需创建新的
            triangle.material_view.as_ref().unwrap()
        } else {
            // 没有材质数据，回退到基本着色
            return get_base_color_with_ambient(triangle, base_color, config);
        };

        // 插值法线
        let interp_normal = interpolate_normal(
            bary,
            triangle.vertices[0].normal_view.unwrap(),
            triangle.vertices[1].normal_view.unwrap(),
            triangle.vertices[2].normal_view.unwrap(),
            triangle.is_perspective,
            triangle.vertices[0].z_view,
            triangle.vertices[1].z_view,
            triangle.vertices[2].z_view,
        );

        // 插值视图空间位置
        let interp_position = interpolate_position(
            bary,
            triangle.vertices[0].position_view.unwrap(),
            triangle.vertices[1].position_view.unwrap(),
            triangle.vertices[2].position_view.unwrap(),
            triangle.is_perspective,
            triangle.vertices[0].z_view,
            triangle.vertices[1].z_view,
            triangle.vertices[2].z_view,
        );

        // 计算视线方向
        let view_dir = (-interp_position.coords).normalize();

        // 计算直接光照贡献
        let light_dir = light.get_direction(&interp_position);
        let light_intensity = light.get_intensity(&interp_position);

        // 计算材质响应
        let response = material_view.compute_response(&light_dir, &view_dir, &interp_normal);

        // 转换为颜色
        let direct_light = Color::new(
            response.x * light_intensity.x,
            response.y * light_intensity.y,
            response.z * light_intensity.z,
        );

        // 处理纹理和应用光照
        if should_use_texture(triangle, config) {
            let texel_color = sample_texture(triangle, bary);

            if config.use_lighting {
                // 结合直接光照和环境光
                texel_color.component_mul(&(direct_light + ambient_contribution))
            } else {
                // 只使用环境光
                texel_color.component_mul(&ambient_contribution)
            }
        } else {
            // 无纹理，使用基础颜色
            if config.use_lighting {
                // 结合直接光照和环境光
                base_color.component_mul(&(direct_light + ambient_contribution))
            } else {
                // 只使用环境光
                base_color.component_mul(&ambient_contribution)
            }
        }
    } else {
        // --- 使用预计算的光照（Flat/Gouraud着色）---

        // 处理纹理
        if should_use_texture(triangle, config) {
            let texel_color = sample_texture(triangle, bary);

            // 应用光照
            if config.use_lighting {
                // 使用预计算的光照颜色和环境光贡献
                texel_color.component_mul(&(triangle.lit_color + ambient_contribution))
            } else {
                // 只使用环境光贡献
                texel_color.component_mul(&ambient_contribution)
            }
        } else {
            // 无纹理，使用基础颜色
            if config.use_lighting {
                // 使用预计算的光照颜色和环境光贡献
                base_color.component_mul(&(triangle.lit_color + ambient_contribution))
            } else {
                // 只使用环境光贡献
                base_color.component_mul(&ambient_contribution)
            }
        }
    }
}

/// 计算环境光贡献
///
/// 基于场景环境光设置和材质特性计算环境光贡献
///
/// # 参数
/// * `triangle` - 三角形数据
/// * `config` - 光栅化器配置
///
/// # 返回值
/// 环境光贡献（颜色）
fn calculate_ambient_contribution(triangle: &TriangleData, config: &RenderConfig) -> Color {
    // 获取环境光颜色和强度
    let ambient_color = config.ambient_color;
    let ambient_intensity = config.ambient_intensity;

    // 结合环境光颜色和强度
    let ambient = Color::new(
        ambient_color.x * ambient_intensity,
        ambient_color.y * ambient_intensity,
        ambient_color.z * ambient_intensity,
    );

    // 如果有材质视图，考虑材质对环境光的特殊响应
    if let Some(material_view) = &triangle.material_view {
        let ambient_response = material_view.get_ambient_color();
        return Color::new(
            ambient_response.x * ambient.x,
            ambient_response.y * ambient.y,
            ambient_response.z * ambient.z,
        );
    }

    // 返回纯环境光颜色
    ambient
}

/// 获取基本颜色并应用环境光
///
/// 这个辅助函数在无法进行完整光照计算时提供基本颜色
/// 即使在use_lighting=false时也会应用环境光
///
/// # 参数
/// * `triangle` - 三角形数据
/// * `base_color` - 基础颜色（可能是面颜色或材质颜色）
/// * `config` - 光栅化器配置
///
/// # 返回值
/// 基本颜色（应用环境光后）
fn get_base_color_with_ambient(
    triangle: &TriangleData,
    base_color: Color,
    config: &RenderConfig,
) -> Color {
    // 计算环境光贡献
    let ambient_contribution = calculate_ambient_contribution(triangle, config);

    // 应用环境光或预计算光照
    if config.use_lighting {
        base_color.component_mul(&(triangle.lit_color + ambient_contribution))
    } else {
        base_color.component_mul(&ambient_contribution)
    }
}

/// 检查像素是否在三角形边缘附近
///
/// # 参数
/// * `pixel_point` - 像素中心点坐标
/// * `v1` - 三角形第一个顶点
/// * `v2` - 三角形第二个顶点
/// * `v3` - 三角形第三个顶点
/// * `edge_threshold` - 边缘检测阈值（像素距离边缘的最大距离）
///
/// # 返回值
/// 如果像素在三角形任意边缘附近，返回true
fn is_on_triangle_edge(
    pixel_point: Point2<f32>,
    v1: Point2<f32>,
    v2: Point2<f32>,
    v3: Point2<f32>,
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

        // 计算投影点 - 使用向量加法而不是点加法
        let projection = Point2::new(edge_start.x + t * edge_vec.x, edge_start.y + t * edge_vec.y);

        // 返回点到投影点的距离
        (p - projection.coords).coords.norm()
    };

    // 检查点到三条边的距离是否小于阈值
    dist_to_edge(pixel_point, v1, v2) <= edge_threshold
        || dist_to_edge(pixel_point, v2, v3) <= edge_threshold
        || dist_to_edge(pixel_point, v3, v1) <= edge_threshold
}

/// 采样纹理并返回颜色。使用统一的sample方法。
///
/// # 参数
/// * `triangle` - 三角形数据，包含纹理
/// * `bary` - 像素的重心坐标
///
/// # 返回值
/// 采样得到的颜色（线性RGB空间，[0,1]范围）
fn sample_texture(triangle: &TriangleData, bary: Vector3<f32>) -> Color {
    // 根据纹理来源类型处理
    match &triangle.texture_data {
        TextureData::Image(_) => {
            // 对于图像纹理，使用真实的Texture对象进行采样
            if let (Some(tc1), Some(tc2), Some(tc3)) = (
                triangle.vertices[0].texcoord,
                triangle.vertices[1].texcoord,
                triangle.vertices[2].texcoord,
            ) {
                if let Some(tex) = triangle.texture_ref {
                    // 使用透视校正的插值函数
                    let tc = interpolate_texcoords(
                        bary,
                        tc1,
                        tc2,
                        tc3,
                        triangle.vertices[0].z_view,
                        triangle.vertices[1].z_view,
                        triangle.vertices[2].z_view,
                        triangle.is_perspective,
                    );

                    // 采样纹理
                    let color_array = tex.sample(tc.x, tc.y);
                    Color::new(color_array[0], color_array[1], color_array[2])
                } else {
                    // 纹理引用为空，回退到默认颜色
                    Color::new(1.0, 1.0, 1.0)
                }
            } else {
                // 缺少纹理坐标，回退到默认颜色
                Color::new(1.0, 1.0, 1.0)
            }
        }
        TextureData::FaceColor(seed) => {
            // 使用面索引生成颜色
            let color = crate::materials::color::get_random_color(*seed, true);
            Color::new(color.x, color.y, color.z)
        }
        TextureData::SolidColor(color) => {
            // 使用固定颜色
            Color::new(color.x, color.y, color.z)
        }
        TextureData::None => {
            // 无纹理，返回白色
            Color::new(1.0, 1.0, 1.0)
        }
    }
}

/// 判断是否应该使用纹理（根据三角形数据和配置）
fn should_use_texture(triangle: &TriangleData, config: &RenderConfig) -> bool {
    // 面颜色模式特殊处理 - 即使没有纹理对象也返回true
    if config.use_face_colors {
        return true;
    }

    // 常规纹理判断
    config.use_texture
        && match triangle.texture_data {
            TextureData::Image(_) => true,
            TextureData::SolidColor(_) => true,
            TextureData::FaceColor(_) => false, // 面颜色模式不算作纹理
            TextureData::None => false,
        }
}
