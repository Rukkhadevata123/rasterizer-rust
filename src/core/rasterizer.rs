//! # 三角形光栅化器
//!
//! 本模块实现了一个基于三角形的软件光栅化器，专注于像素级处理：
//! - 基于重心坐标的插值
//! - 深度缓冲和深度测试
//! - 多种着色模型处理：平面着色(Flat)、Gouraud着色和Phong着色
//! - 纹理采样与透视校正插值
//! - 着色计算 (Blinn-Phong和PBR)
//! - 增强环境光遮蔽和软阴影
//! - Gamma校正
//!
//! 光栅化器使用原子操作处理深度缓冲和颜色缓冲区以支持高效的并行渲染。

use crate::geometry::culling::is_on_triangle_edge;
use crate::geometry::interpolation::{
    barycentric_coordinates, interpolate_depth, interpolate_normal, interpolate_position,
    interpolate_texcoords, is_inside_triangle,
};
use crate::io::render_settings::RenderSettings; // 直接导入 RenderSettings
use crate::material_system::color::{Color, linear_rgb_to_u8};
use crate::material_system::light::Light;
use crate::material_system::materials::MaterialView;
use crate::material_system::texture::Texture;
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

/// 为 TextureSource 实现 Clone 特性，解决所有权问题
/// 这使得我们可以在方法之间传递 TextureSource 而不必担心所有权转移
#[derive(Debug, Clone)]
pub enum TextureSource<'a> {
    None,
    Image(&'a Texture),
    FaceColor(u64),
    SolidColor(Vector3<f32>),
}

/// 单个三角形光栅化所需的输入数据
///
/// 包含三角形的几何信息（顶点位置、法线）、材质属性、纹理坐标和光照信息。
/// 所有决策（如使用哪种纹理来源）已经在渲染器中做出。
pub struct TriangleData<'a> {
    // 三个顶点数据
    pub vertices: [VertexRenderData; 3],

    // 颜色属性
    pub base_color: Color, // 基础颜色

    // 纹理与材质
    pub texture_source: TextureSource<'a>, // 统一的纹理来源
    pub material_view: Option<MaterialView<'a>>, // 材质视图

    // 光照信息
    pub lights: &'a [Light], // 多光源数组引用

    // 环境光信息
    pub ambient_intensity: f32,
    pub ambient_color: Vector3<f32>,

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
/// * `settings` - 渲染设置参数
pub fn rasterize_triangle(
    triangle: &TriangleData,
    width: usize,
    height: usize,
    depth_buffer: &[AtomicF32],
    color_buffer: &[AtomicU8],
    settings: &RenderSettings,
) {
    // 1. 计算三角形包围盒 - 优化实现，减少重复计算
    let v0 = &triangle.vertices[0].pix;
    let v1 = &triangle.vertices[1].pix;
    let v2 = &triangle.vertices[2].pix;

    // 使用SIMD友好的min/max计算
    let min_x = v0.x.min(v1.x).min(v2.x).floor().max(0.0) as usize;
    let min_y = v0.y.min(v1.y).min(v2.y).floor().max(0.0) as usize;
    let max_x = v0.x.max(v1.x).max(v2.x).ceil().min(width as f32) as usize;
    let max_y = v0.y.max(v1.y).max(v2.y).ceil().min(height as f32) as usize;

    // 检查无效的包围盒（宽度或高度为0）
    if max_x <= min_x || max_y <= min_y {
        return;
    }

    // 线框模式的边缘检测阈值（像素单位）
    const EDGE_THRESHOLD: f32 = 1.0;

    // 预计算与光照相关的常量
    let use_phong_or_pbr = (settings.use_pbr || settings.use_phong)
        && triangle.vertices[0].normal_view.is_some()
        && triangle.vertices[0].position_view.is_some()
        && !triangle.lights.is_empty();

    // 预计算纹理使用决策
    let use_texture = matches!(
        triangle.texture_source,
        TextureSource::Image(_) | TextureSource::FaceColor(_) | TextureSource::SolidColor(_)
    );

    // 提前计算环境光贡献，避免每个像素重复计算
    let ambient_contribution = calculate_ambient_contribution(triangle);

    // 2. 遍历包围盒中的每个像素
    for y in min_y..max_y {
        for x in min_x..max_x {
            // 计算像素中心点坐标
            let pixel_center = Point2::new(x as f32 + 0.5, y as f32 + 0.5);
            let pixel_index = y * width + x;

            // 3. 计算重心坐标
            if let Some(bary) = barycentric_coordinates(pixel_center, *v0, *v1, *v2) {
                // 4. 检查像素是否在三角形内
                if is_inside_triangle(bary) {
                    // 线框模式特殊处理
                    if settings.wireframe
                        && !is_on_triangle_edge(pixel_center, *v0, *v1, *v2, EDGE_THRESHOLD)
                    {
                        continue;
                    }

                    // 5. 插值深度值
                    let interpolated_depth = interpolate_depth(
                        bary,
                        triangle.vertices[0].z_view,
                        triangle.vertices[1].z_view,
                        triangle.vertices[2].z_view,
                        settings.is_perspective() && triangle.is_perspective,
                    );

                    // 检查深度是否有效（不在相机后方且不太远）
                    if interpolated_depth.is_finite() && interpolated_depth < f32::INFINITY {
                        // 6. 深度测试（使用原子操作）
                        let current_depth_atomic = &depth_buffer[pixel_index];

                        // 优化深度测试逻辑，减少原子操作
                        if !settings.use_zbuffer {
                            // 不使用深度测试，直接更新颜色
                            let final_color = calculate_pixel_color(
                                triangle,
                                bary,
                                settings,
                                use_phong_or_pbr,
                                use_texture,
                                &ambient_contribution,
                            );
                            write_pixel_color(
                                pixel_index,
                                &final_color,
                                color_buffer,
                                settings.use_gamma,
                            );
                        } else {
                            // 进行深度测试
                            let previous_depth = current_depth_atomic.load(Ordering::Relaxed);

                            if interpolated_depth < previous_depth {
                                // 尝试原子更新深度值
                                let old_depth = current_depth_atomic
                                    .fetch_min(interpolated_depth, Ordering::Relaxed);

                                // 只有当当前线程成功更新了深度值时才写入颜色
                                if old_depth > interpolated_depth {
                                    let final_color = calculate_pixel_color(
                                        triangle,
                                        bary,
                                        settings,
                                        use_phong_or_pbr,
                                        use_texture,
                                        &ambient_contribution,
                                    );
                                    write_pixel_color(
                                        pixel_index,
                                        &final_color,
                                        color_buffer,
                                        settings.use_gamma,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// 将颜色写入到帧缓冲区
#[inline]
fn write_pixel_color(
    pixel_index: usize,
    color: &Color,
    color_buffer: &[AtomicU8],
    apply_gamma: bool,
) {
    let buffer_start_index = pixel_index * 3;
    if buffer_start_index + 2 < color_buffer.len() {
        // 使用gamma校正函数转换颜色
        let [r, g, b] = linear_rgb_to_u8(color, apply_gamma);
        color_buffer[buffer_start_index].store(r, Ordering::Relaxed);
        color_buffer[buffer_start_index + 1].store(g, Ordering::Relaxed);
        color_buffer[buffer_start_index + 2].store(b, Ordering::Relaxed);
    }
}

/// 计算增强的环境光遮蔽因子 - 更明显的效果
fn calculate_enhanced_ao(
    triangle: &TriangleData,
    bary: Vector3<f32>,
    interp_normal: &Vector3<f32>,
    settings: &RenderSettings,
) -> f32 {
    if !settings.enhanced_ao {
        return 1.0; // 禁用时返回无遮蔽
    }

    // 更激进的基础AO计算
    // 基于法线朝向 - 增强对比度
    let up_factor = {
        let raw_up = (interp_normal.y + 1.0) * 0.5;
        // 使用幂函数增强对比度
        raw_up.powf(1.5) // 让朝下的表面更暗
    };

    // 更明显的边缘遮蔽效果
    let edge_proximity = {
        let min_bary = bary.x.min(bary.y).min(bary.z);
        let edge_factor = (min_bary * 2.0).min(1.0); // 减少乘数，让边缘效果更强
        0.6 + 0.4 * edge_factor // 边缘区域AO更强
    };

    // 计算法线变化量（凹陷区域法线变化大）
    if let (Some(n0), Some(n1), Some(n2)) = (
        triangle.vertices[0].normal_view,
        triangle.vertices[1].normal_view,
        triangle.vertices[2].normal_view,
    ) {
        // 增强法线变化检测
        let normal_variance = (n0 - n1).magnitude() + (n1 - n2).magnitude() + (n2 - n0).magnitude();

        // 更激进的曲率因子
        let curvature_factor = (1.0 - (normal_variance * 0.4).min(0.7)).max(0.1); // 增强变化检测

        // 更明显的位置相关AO
        let center_distance = (bary - Vector3::new(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)).magnitude();
        let position_factor = 1.0 - (center_distance * 0.5).min(0.3); // 增强中心遮蔽

        // 重新调整权重，让效果更明显
        let base_ao = (up_factor * 0.5 +           // 增加法线影响
            curvature_factor * 0.3 +
            edge_proximity * 0.15 +
            position_factor * 0.05)
            .clamp(0.05, 1.0); // 🔥 降低最小值，允许更暗的阴影

        // 应用用户设置，但增强效果
        let enhanced_strength = settings.ao_strength * 1.5; // 放大用户设置的效果
        let final_ao = 1.0 - ((1.0 - base_ao) * enhanced_strength.min(1.0));
        final_ao.clamp(0.05, 1.0) // 🔥 允许更暗的阴影
    } else {
        // 没有法线信息，只使用基础遮蔽和边缘因子
        let base_ao = (up_factor * 0.7 + edge_proximity * 0.3).clamp(0.3, 1.0);
        let enhanced_strength = settings.ao_strength * 1.2;
        1.0 - ((1.0 - base_ao) * enhanced_strength.min(1.0))
    }
}

/// 计算光源的简单软阴影因子
fn calculate_simple_shadow_factor(
    light_dir: &Vector3<f32>,
    surface_normal: &Vector3<f32>,
    triangle: &TriangleData,
    interp_position: &Point3<f32>,
    settings: &RenderSettings,
) -> f32 {
    if !settings.soft_shadows {
        return 1.0; // 禁用时返回无阴影
    }

    // 1. 基础因子：光线与法线的角度
    let ndl = surface_normal.dot(light_dir).max(0.0);

    // 2. 边缘softening：在grazing angle处产生softer shadows
    let edge_factor = if ndl < 0.3 {
        // 在边缘处应用soft transition
        (ndl / 0.3).powf(0.7) // 非线性过渡
    } else {
        1.0
    };

    // 3. 深度相关的遮蔽：距离相机较远的区域更容易被遮蔽
    let depth_factor = if interp_position.z < -2.0 {
        // 远处物体有更多环境遮蔽
        0.8 + 0.2 * ((-interp_position.z - 2.0) / 8.0).min(1.0)
    } else {
        1.0
    };

    // 4. 基于法线变化的局部遮蔽
    let local_occlusion = if let (Some(n0), Some(n1), Some(n2)) = (
        triangle.vertices[0].normal_view,
        triangle.vertices[1].normal_view,
        triangle.vertices[2].normal_view,
    ) {
        // 计算法线变化，变化大的地方更容易产生阴影
        let normal_variance = (n0 - n1).magnitude() + (n1 - n2).magnitude() + (n2 - n0).magnitude();

        // 法线变化大的区域有更多局部遮蔽
        let occlusion_strength = (normal_variance * 0.3).min(0.4);
        1.0 - occlusion_strength
    } else {
        1.0
    };

    // 5. 组合所有因子
    let base_shadow = edge_factor * depth_factor * local_occlusion;

    // 应用用户设置的阴影强度
    let final_shadow = 1.0 - ((1.0 - base_shadow) * settings.shadow_strength);
    final_shadow.clamp(0.1, 1.0) // 确保不会完全黑
}

/// 应用AO到环境光的新函数
fn apply_ao_to_ambient(ambient: &Color, ao_factor: f32) -> Color {
    Color::new(
        ambient.x * ao_factor,
        ambient.y * ao_factor,
        ambient.z * ao_factor,
    )
}

/// 计算像素的最终颜色值
///
/// 根据三角形数据、重心坐标和配置参数计算像素颜色。
/// 处理三种主要的着色模式：
/// 1. PBR 着色（基于物理的渲染）
/// 2. Phong着色（逐像素光照计算）
/// 3. 预计算光照（Flat或Gouraud着色）
///
/// 新增功能：增强AO和软阴影
///
/// # 参数
/// * `triangle` - 三角形数据
/// * `bary` - 像素的重心坐标
/// * `settings` - 渲染设置
/// * `use_phong_or_pbr` - 是否使用Phong或PBR着色（预计算的标志）
/// * `use_texture` - 是否使用纹理（预计算的标志）
/// * `ambient_contribution` - 预计算的环境光贡献
///
/// # 返回值
/// 计算得到的像素颜色（线性RGB空间）
fn calculate_pixel_color(
    triangle: &TriangleData,
    bary: Vector3<f32>,
    settings: &RenderSettings,
    use_phong_or_pbr: bool,
    use_texture: bool,
    ambient_contribution: &Color,
) -> Color {
    // 使用传入的基础颜色
    let base_color = triangle.base_color;

    // 使用预计算的标记判断着色模式
    if use_phong_or_pbr {
        // --- 使用材质视图进行PBR或Phong着色 ---

        // 获取材质视图
        let material_view = if let Some(mat_view) = &triangle.material_view {
            mat_view
        } else {
            // 没有材质数据，回退到基本着色
            return base_color.component_mul(ambient_contribution);
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

        // 计算增强的AO因子
        let ao_factor = calculate_enhanced_ao(triangle, bary, &interp_normal, settings);

        // 为每个光源计算软阴影
        let mut total_direct_light = Vector3::zeros();

        // 遍历所有光源
        for light in triangle.lights {
            // 计算光线方向和强度
            let light_dir = light.get_direction(&interp_position);
            let light_intensity = light.get_intensity(&interp_position);

            // 计算此光源的软阴影因子
            let shadow_factor = calculate_simple_shadow_factor(
                &light_dir,
                &interp_normal,
                triangle,
                &interp_position,
                settings,
            );

            // 计算材质对该光源的响应
            let response = material_view.compute_response(&light_dir, &view_dir, &interp_normal);

            // 应用软阴影因子到光照
            total_direct_light += Vector3::new(
                response.x * light_intensity.x * shadow_factor,
                response.y * light_intensity.y * shadow_factor,
                response.z * light_intensity.z * shadow_factor,
            );
        }

        // 转换为颜色
        let direct_light = Color::new(
            total_direct_light.x,
            total_direct_light.y,
            total_direct_light.z,
        );

        // 应用AO到环境光
        let ao_ambient = apply_ao_to_ambient(ambient_contribution, ao_factor);

        // 处理纹理和应用光照
        if use_texture {
            let texel_color = sample_texture(triangle, bary);

            if settings.use_lighting {
                // 结合直接光照和AO环境光
                texel_color.component_mul(&(direct_light + ao_ambient))
            } else {
                // 只使用AO环境光
                texel_color.component_mul(&ao_ambient)
            }
        } else {
            // 无纹理，使用基础颜色
            if settings.use_lighting {
                // 结合直接光照和AO环境光
                base_color.component_mul(&(direct_light + ao_ambient))
            } else {
                // 只使用AO环境光
                base_color.component_mul(&ao_ambient)
            }
        }
    } else {
        // --- 使用预计算的光照（Flat/Gouraud着色）或无光照 ---

        // 获取表面颜色（从纹理或基础颜色）
        let surface_color = if use_texture {
            sample_texture(triangle, bary)
        } else {
            base_color
        };

        // 为Blinn-Phong模式也应用简单AO
        if settings.use_lighting {
            // 为非PBR/Phong模式计算简单AO
            let ao_factor = if settings.enhanced_ao {
                // 计算法线（如果可用）
                let interp_normal = if let (Some(n0), Some(n1), Some(n2)) = (
                    triangle.vertices[0].normal_view,
                    triangle.vertices[1].normal_view,
                    triangle.vertices[2].normal_view,
                ) {
                    interpolate_normal(
                        bary,
                        n0,
                        n1,
                        n2,
                        triangle.is_perspective,
                        triangle.vertices[0].z_view,
                        triangle.vertices[1].z_view,
                        triangle.vertices[2].z_view,
                    )
                } else {
                    Vector3::new(0.0, 1.0, 0.0) // 默认向上法线
                };

                calculate_enhanced_ao(triangle, bary, &interp_normal, settings)
            } else {
                1.0 // 禁用AO时不应用遮蔽
            };

            // 应用AO到环境光
            let ao_ambient = apply_ao_to_ambient(ambient_contribution, ao_factor);
            surface_color.component_mul(&ao_ambient)
        } else {
            // 只使用表面颜色
            surface_color
        }
    }
}

/// 计算环境光贡献
///
/// 基于场景环境光设置和材质特性计算环境光贡献
///
/// # 参数
/// * `triangle` - 三角形数据
///
/// # 返回值
/// 环境光贡献（颜色）
fn calculate_ambient_contribution(triangle: &TriangleData) -> Color {
    // 获取环境光颜色和强度
    let ambient_color = triangle.ambient_color;
    let ambient_intensity = triangle.ambient_intensity;

    // 结合环境光颜色和强度
    let ambient = Color::new(
        ambient_color.x * ambient_intensity,
        ambient_color.y * ambient_intensity,
        ambient_color.z * ambient_intensity,
    );

    // 如果有材质，直接使用其 ambient_factor 属性
    if let Some(material_view) = &triangle.material_view {
        // 获取材质实际引用
        let material = match material_view {
            MaterialView::BlinnPhong(material) => material,
            MaterialView::PBR(material) => material,
        };

        // 使用材质的 ambient_factor 属性
        return Color::new(
            material.ambient_factor.x * ambient.x,
            material.ambient_factor.y * ambient.y,
            material.ambient_factor.z * ambient.z,
        );
    }

    // 返回纯环境光颜色
    ambient
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
    match &triangle.texture_source {
        TextureSource::Image(tex) => {
            // 对于图像纹理，使用真实的Texture对象进行采样
            if let (Some(tc1), Some(tc2), Some(tc3)) = (
                triangle.vertices[0].texcoord,
                triangle.vertices[1].texcoord,
                triangle.vertices[2].texcoord,
            ) {
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
                // 缺少纹理坐标，回退到默认颜色
                Color::new(1.0, 1.0, 1.0)
            }
        }
        TextureSource::FaceColor(seed) => {
            // 使用面索引生成颜色
            let color = crate::material_system::color::get_random_color(*seed, true);
            Color::new(color.x, color.y, color.z)
        }
        TextureSource::SolidColor(color) => {
            // 使用固定颜色
            Color::new(color.x, color.y, color.z)
        }
        TextureSource::None => {
            // 无纹理，返回白色
            Color::new(1.0, 1.0, 1.0)
        }
    }
}
