use crate::core::frame_buffer::FrameBuffer;
use crate::core::renderer::GeometryResult;
use crate::geometry::culling::{is_backface, is_on_triangle_edge, should_cull_small_triangle};
use crate::geometry::interpolation::{
    barycentric_coordinates, interpolate_depth, interpolate_normal, interpolate_position,
    interpolate_texcoords, is_inside_triangle,
};
use crate::io::render_settings::RenderSettings;
use crate::material_system::color::{get_random_color, linear_rgb_to_u8};
use crate::material_system::light::Light;
use crate::material_system::materials::{Material, MaterialView, ModelData, Vertex};
use crate::material_system::texture::Texture;
use atomic_float::AtomicF32;
use nalgebra::{Point2, Point3, Vector2, Vector3};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU8, Ordering};

// ===== 数据结构区域 =====

#[derive(Debug, Clone)]
pub struct VertexRenderData {
    pub pix: Point2<f32>,
    pub z_view: f32,
    pub texcoord: Option<Vector2<f32>>,
    pub normal_view: Option<Vector3<f32>>,
    pub position_view: Option<Point3<f32>>,
}

#[derive(Debug, Clone)]
pub enum TextureSource<'a> {
    None,
    Image(&'a Texture),
    FaceColor(u64),
    SolidColor(Vector3<f32>),
}

pub struct TriangleData<'a> {
    pub vertices: [VertexRenderData; 3],
    pub base_color: Vector3<f32>,
    pub texture_source: TextureSource<'a>,
    pub material_view: Option<MaterialView<'a>>,
    pub lights: &'a [Light],
    pub ambient_intensity: f32,
    pub ambient_color: Vector3<f32>,
    pub is_perspective: bool,
}

#[derive(Debug, Clone, Copy)]
enum RenderStrategy {
    LargeTrianglePixelParallel,
    SmallTriangleParallel,
    Mixed,
}

// ===== 核心功能实现 =====

impl<'a> TriangleData<'a> {
    pub fn is_valid(&self) -> bool {
        estimate_triangle_area(self) > 1e-6
    }
}

// ===== 三角形准备 =====

#[allow(clippy::too_many_arguments)]
pub fn prepare_triangles<'a>(
    model_data: &'a ModelData,
    geometry_result: &GeometryResult,
    material_override: Option<&'a Material>,
    settings: &'a RenderSettings,
    lights: &'a [Light],
    ambient_intensity: f32,
    ambient_color: Vector3<f32>,
) -> Vec<TriangleData<'a>> {
    model_data
        .meshes
        .par_iter()
        .enumerate()
        .flat_map(|(mesh_idx, mesh)| {
            let vertex_offset = geometry_result.mesh_offsets[mesh_idx];
            let material_opt = material_override
                .or_else(|| mesh.material_id.and_then(|id| model_data.materials.get(id)));

            mesh.indices
                .chunks_exact(3)
                .enumerate()
                .filter_map(move |(face_idx, indices)| {
                    let global_face_index = (mesh_idx * 1000 + face_idx) as u64;
                    process_triangle(
                        indices,
                        &mesh.vertices,
                        vertex_offset,
                        global_face_index,
                        geometry_result,
                        material_opt,
                        settings,
                        lights,
                        ambient_intensity,
                        ambient_color,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn process_triangle<'a>(
    indices: &[u32],
    vertices: &[Vertex],
    vertex_offset: usize,
    global_face_index: u64,
    geometry_result: &GeometryResult,
    material_opt: Option<&'a Material>,
    settings: &'a RenderSettings,
    lights: &'a [Light],
    ambient_intensity: f32,
    ambient_color: Vector3<f32>,
) -> Option<TriangleData<'a>> {
    let i0 = indices[0] as usize;
    let i1 = indices[1] as usize;
    let i2 = indices[2] as usize;

    let global_i0 = vertex_offset + i0;
    let global_i1 = vertex_offset + i1;
    let global_i2 = vertex_offset + i2;

    if global_i0 >= geometry_result.screen_coords.len()
        || global_i1 >= geometry_result.screen_coords.len()
        || global_i2 >= geometry_result.screen_coords.len()
    {
        return None;
    }

    let pix0 = geometry_result.screen_coords[global_i0];
    let pix1 = geometry_result.screen_coords[global_i1];
    let pix2 = geometry_result.screen_coords[global_i2];

    let view_pos0 = geometry_result.view_coords[global_i0];
    let view_pos1 = geometry_result.view_coords[global_i1];
    let view_pos2 = geometry_result.view_coords[global_i2];

    // 剔除检查
    if settings.backface_culling && is_backface(&view_pos0, &view_pos1, &view_pos2) {
        return None;
    }

    if settings.cull_small_triangles
        && should_cull_small_triangle(&pix0, &pix1, &pix2, settings.min_triangle_area)
    {
        return None;
    }

    // 确定纹理和颜色
    let texture_source = determine_texture_source(settings, material_opt, global_face_index);
    let base_color = determine_base_color(&texture_source, material_opt);

    let material_view = material_opt.map(|m| {
        if settings.use_pbr {
            MaterialView::PBR(m)
        } else {
            MaterialView::BlinnPhong(m)
        }
    });

    let vertex_data = [
        create_vertex_render_data(
            &pix0,
            view_pos0,
            &vertices[i0],
            global_i0,
            &texture_source,
            geometry_result,
        ),
        create_vertex_render_data(
            &pix1,
            view_pos1,
            &vertices[i1],
            global_i1,
            &texture_source,
            geometry_result,
        ),
        create_vertex_render_data(
            &pix2,
            view_pos2,
            &vertices[i2],
            global_i2,
            &texture_source,
            geometry_result,
        ),
    ];

    Some(TriangleData {
        vertices: vertex_data,
        base_color,
        texture_source,
        material_view,
        lights,
        ambient_intensity,
        ambient_color,
        is_perspective: settings.is_perspective(),
    })
}

fn determine_texture_source<'a>(
    settings: &RenderSettings,
    material_opt: Option<&'a Material>,
    global_face_index: u64,
) -> TextureSource<'a> {
    if !settings.use_texture {
        return if settings.colorize {
            TextureSource::FaceColor(global_face_index)
        } else {
            TextureSource::None
        };
    }

    if let Some(tex) = material_opt.and_then(|m| m.texture.as_ref()) {
        if tex.is_face_color() {
            TextureSource::FaceColor(global_face_index)
        } else {
            TextureSource::Image(tex)
        }
    } else if settings.colorize {
        TextureSource::FaceColor(global_face_index)
    } else {
        let color = material_opt.map_or_else(|| Vector3::new(0.7, 0.7, 0.7), |m| m.diffuse());
        TextureSource::SolidColor(color)
    }
}

fn determine_base_color(
    texture_source: &TextureSource,
    material_opt: Option<&Material>,
) -> Vector3<f32> {
    match texture_source {
        TextureSource::FaceColor(_) => Vector3::new(1.0, 1.0, 1.0),
        _ => material_opt.map_or_else(|| Vector3::new(0.7, 0.7, 0.7), |m| m.diffuse()),
    }
}

fn create_vertex_render_data(
    pix: &Point2<f32>,
    view_pos: Point3<f32>,
    vertex: &Vertex,
    global_index: usize,
    texture_source: &TextureSource,
    geometry_result: &GeometryResult,
) -> VertexRenderData {
    VertexRenderData {
        pix: *pix,
        z_view: view_pos.z,
        texcoord: if matches!(texture_source, TextureSource::Image(_)) {
            Some(vertex.texcoord)
        } else {
            None
        },
        normal_view: Some(geometry_result.view_normals[global_index]),
        position_view: Some(view_pos),
    }
}

// ===== 并行光栅化 =====

pub fn rasterize_triangles(
    triangles: &[TriangleData],
    width: usize,
    height: usize,
    depth_buffer: &[AtomicF32],
    color_buffer: &[AtomicU8],
    settings: &RenderSettings,
    frame_buffer: &FrameBuffer,
) {
    if triangles.is_empty() {
        return;
    }

    let strategy = choose_strategy(triangles, width, height);

    match strategy {
        RenderStrategy::LargeTrianglePixelParallel => {
            triangles.par_iter().for_each(|triangle| {
                rasterize_triangle_pixel_parallel(
                    triangle,
                    width,
                    height,
                    depth_buffer,
                    color_buffer,
                    settings,
                    frame_buffer,
                );
            });
        }
        RenderStrategy::SmallTriangleParallel => {
            triangles.par_iter().for_each(|triangle| {
                rasterize_triangle(
                    triangle,
                    width,
                    height,
                    depth_buffer,
                    color_buffer,
                    settings,
                    frame_buffer,
                );
            });
        }
        RenderStrategy::Mixed => {
            let screen_area = (width * height) as f32;
            let large_threshold = screen_area * 0.001;

            let (large_indices, small_indices): (Vec<_>, Vec<_>) = triangles
                .iter()
                .enumerate()
                .partition(|(_, tri)| estimate_triangle_area(tri) > large_threshold);

            rayon::join(
                || {
                    large_indices.par_iter().for_each(|(idx, _)| {
                        rasterize_triangle_pixel_parallel(
                            &triangles[*idx],
                            width,
                            height,
                            depth_buffer,
                            color_buffer,
                            settings,
                            frame_buffer,
                        );
                    });
                },
                || {
                    small_indices.par_iter().for_each(|(idx, _)| {
                        rasterize_triangle(
                            &triangles[*idx],
                            width,
                            height,
                            depth_buffer,
                            color_buffer,
                            settings,
                            frame_buffer,
                        );
                    });
                },
            );
        }
    }
}

fn choose_strategy(triangles: &[TriangleData], width: usize, height: usize) -> RenderStrategy {
    let screen_area = (width * height) as f32;
    let triangle_count = triangles.len();

    let avg_triangle_size = if triangle_count > 0 {
        let total_area: f32 = triangles
            .iter()
            .take(triangle_count.min(50))
            .map(estimate_triangle_area)
            .sum();
        total_area / triangle_count.min(50) as f32
    } else {
        0.0
    };

    if triangle_count > 2000 && avg_triangle_size > screen_area * 0.0005 {
        RenderStrategy::Mixed
    } else if avg_triangle_size > 500.0 || triangle_count < 100 {
        RenderStrategy::LargeTrianglePixelParallel
    } else {
        RenderStrategy::SmallTriangleParallel
    }
}

fn estimate_triangle_area(triangle: &TriangleData) -> f32 {
    let v0 = &triangle.vertices[0].pix;
    let v1 = &triangle.vertices[1].pix;
    let v2 = &triangle.vertices[2].pix;
    0.5 * ((v1.x - v0.x) * (v2.y - v0.y) - (v2.x - v0.x) * (v1.y - v0.y)).abs()
}

fn rasterize_triangle_pixel_parallel(
    triangle: &TriangleData,
    width: usize,
    height: usize,
    depth_buffer: &[AtomicF32],
    color_buffer: &[AtomicU8],
    settings: &RenderSettings,
    frame_buffer: &FrameBuffer,
) {
    let v0 = &triangle.vertices[0].pix;
    let v1 = &triangle.vertices[1].pix;
    let v2 = &triangle.vertices[2].pix;

    let min_x = v0.x.min(v1.x).min(v2.x).floor().max(0.0) as usize;
    let min_y = v0.y.min(v1.y).min(v2.y).floor().max(0.0) as usize;
    let max_x = v0.x.max(v1.x).max(v2.x).ceil().min(width as f32) as usize;
    let max_y = v0.y.max(v1.y).max(v2.y).ceil().min(height as f32) as usize;

    if max_x <= min_x || max_y <= min_y {
        return;
    }

    (min_y..max_y).into_par_iter().for_each(|y| {
        for x in min_x..max_x {
            let pixel_center = Point2::new(x as f32 + 0.5, y as f32 + 0.5);
            let pixel_index = y * width + x;
            rasterize_pixel(
                triangle,
                pixel_center,
                pixel_index,
                x,
                y,
                depth_buffer,
                color_buffer,
                settings,
                frame_buffer,
            );
        }
    });
}

// ===== 像素处理 =====

pub fn rasterize_triangle(
    triangle: &TriangleData,
    width: usize,
    height: usize,
    depth_buffer: &[AtomicF32],
    color_buffer: &[AtomicU8],
    settings: &RenderSettings,
    frame_buffer: &FrameBuffer,
) {
    if !triangle.is_valid() {
        return;
    }

    let (min_x, min_y, max_x, max_y) = compute_bounding_box(triangle, width, height);
    if max_x <= min_x || max_y <= min_y {
        return;
    }

    let use_phong_or_pbr = (settings.use_pbr || settings.use_phong)
        && triangle.vertices[0].normal_view.is_some()
        && triangle.vertices[0].position_view.is_some()
        && !triangle.lights.is_empty();

    let use_texture = !matches!(triangle.texture_source, TextureSource::None);
    let ambient_contribution = calculate_ambient_contribution(triangle);

    for y in min_y..max_y {
        for x in min_x..max_x {
            let pixel_index = y * width + x;
            let pixel_center = Point2::new(x as f32 + 0.5, y as f32 + 0.5);
            process_pixel(
                triangle,
                pixel_center,
                pixel_index,
                x,
                y,
                use_phong_or_pbr,
                use_texture,
                &ambient_contribution,
                depth_buffer,
                color_buffer,
                settings,
                frame_buffer,
            );
        }
    }
}

fn compute_bounding_box(
    triangle: &TriangleData,
    width: usize,
    height: usize,
) -> (usize, usize, usize, usize) {
    let v0 = &triangle.vertices[0].pix;
    let v1 = &triangle.vertices[1].pix;
    let v2 = &triangle.vertices[2].pix;

    let min_x = v0.x.min(v1.x).min(v2.x).floor().max(0.0) as usize;
    let min_y = v0.y.min(v1.y).min(v2.y).floor().max(0.0) as usize;
    let max_x = v0.x.max(v1.x).max(v2.x).ceil().min(width as f32) as usize;
    let max_y = v0.y.max(v1.y).max(v2.y).ceil().min(height as f32) as usize;

    (min_x, min_y, max_x, max_y)
}

#[allow(clippy::too_many_arguments)]
pub fn rasterize_pixel(
    triangle: &TriangleData,
    pixel_center: Point2<f32>,
    pixel_index: usize,
    pixel_x: usize,
    pixel_y: usize,
    depth_buffer: &[AtomicF32],
    color_buffer: &[AtomicU8],
    settings: &RenderSettings,
    frame_buffer: &FrameBuffer,
) {
    let use_phong_or_pbr = (settings.use_pbr || settings.use_phong)
        && triangle.vertices[0].normal_view.is_some()
        && triangle.vertices[0].position_view.is_some()
        && !triangle.lights.is_empty();

    let use_texture = !matches!(triangle.texture_source, TextureSource::None);
    let ambient_contribution = calculate_ambient_contribution(triangle);

    process_pixel(
        triangle,
        pixel_center,
        pixel_index,
        pixel_x,
        pixel_y,
        use_phong_or_pbr,
        use_texture,
        &ambient_contribution,
        depth_buffer,
        color_buffer,
        settings,
        frame_buffer,
    );
}

#[allow(clippy::too_many_arguments)]
fn process_pixel(
    triangle: &TriangleData,
    pixel_center: Point2<f32>,
    pixel_index: usize,
    pixel_x: usize,
    pixel_y: usize,
    use_phong_or_pbr: bool,
    use_texture: bool,
    ambient_contribution: &Vector3<f32>,
    depth_buffer: &[AtomicF32],
    color_buffer: &[AtomicU8],
    settings: &RenderSettings,
    frame_buffer: &FrameBuffer,
) {
    let v0 = &triangle.vertices[0].pix;
    let v1 = &triangle.vertices[1].pix;
    let v2 = &triangle.vertices[2].pix;

    let bary = match barycentric_coordinates(pixel_center, *v0, *v1, *v2) {
        Some(bary) => bary,
        None => return,
    };

    if !is_inside_triangle(bary) {
        return;
    }

    let final_alpha = get_effective_alpha(triangle, settings);
    if final_alpha <= 0.01 {
        return;
    }

    if settings.wireframe && !is_on_triangle_edge(pixel_center, *v0, *v1, *v2, 1.0) {
        return;
    }

    let interpolated_depth = interpolate_depth(
        bary,
        triangle.vertices[0].z_view,
        triangle.vertices[1].z_view,
        triangle.vertices[2].z_view,
        settings.is_perspective() && triangle.is_perspective,
    );

    if !interpolated_depth.is_finite() || interpolated_depth >= f32::INFINITY {
        return;
    }

    if settings.use_zbuffer {
        let current_depth_atomic = &depth_buffer[pixel_index];
        let old_depth = current_depth_atomic.fetch_min(interpolated_depth, Ordering::Relaxed);
        if old_depth <= interpolated_depth {
            return;
        }
    }

    let material_color = calculate_pixel_color(
        triangle,
        bary,
        settings,
        use_phong_or_pbr,
        use_texture,
        ambient_contribution,
    );
    let final_color =
        apply_alpha_blending(&material_color, final_alpha, pixel_x, pixel_y, frame_buffer);

    write_pixel_color(pixel_index, &final_color, color_buffer, settings.use_gamma);
}

// ===== 着色计算 =====

#[allow(clippy::too_many_arguments)]
fn calculate_pixel_color(
    triangle: &TriangleData,
    bary: Vector3<f32>,
    settings: &RenderSettings,
    use_phong_or_pbr: bool,
    use_texture: bool,
    ambient_contribution: &Vector3<f32>,
) -> Vector3<f32> {
    if use_phong_or_pbr {
        calculate_advanced_shading(triangle, bary, settings, use_texture, ambient_contribution)
    } else {
        calculate_basic_shading(triangle, bary, settings, use_texture, ambient_contribution)
    }
}

fn calculate_advanced_shading(
    triangle: &TriangleData,
    bary: Vector3<f32>,
    settings: &RenderSettings,
    use_texture: bool,
    ambient_contribution: &Vector3<f32>,
) -> Vector3<f32> {
    let material_view = match &triangle.material_view {
        Some(mat_view) => mat_view,
        None => {
            return calculate_basic_shading(
                triangle,
                bary,
                settings,
                use_texture,
                ambient_contribution,
            );
        }
    };

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

    let view_dir = (-interp_position.coords).normalize();

    let mut total_direct_light = Vector3::zeros();

    for light in triangle.lights {
        let light_dir = light.get_direction(&interp_position);
        let light_intensity = light.get_intensity(&interp_position);

        let response = material_view.compute_response(&light_dir, &view_dir, &interp_normal);

        total_direct_light += Vector3::new(
            response.x * light_intensity.x,
            response.y * light_intensity.y,
            response.z * light_intensity.z,
        );
    }

    let direct_light = total_direct_light;

    let surface_color = if use_texture {
        sample_texture(triangle, bary)
    } else {
        triangle.base_color
    };

    if settings.use_lighting {
        surface_color.component_mul(&(direct_light + *ambient_contribution))
    } else {
        surface_color.component_mul(ambient_contribution)
    }
}

fn calculate_basic_shading(
    triangle: &TriangleData,
    bary: Vector3<f32>,
    settings: &RenderSettings,
    use_texture: bool,
    ambient_contribution: &Vector3<f32>,
) -> Vector3<f32> {
    let surface_color = if use_texture {
        sample_texture(triangle, bary)
    } else {
        triangle.base_color
    };

    if settings.use_lighting {
        surface_color.component_mul(ambient_contribution)
    } else {
        surface_color
    }
}

fn sample_texture(triangle: &TriangleData, bary: Vector3<f32>) -> Vector3<f32> {
    match &triangle.texture_source {
        TextureSource::Image(tex) => {
            if let (Some(tc1), Some(tc2), Some(tc3)) = (
                triangle.vertices[0].texcoord,
                triangle.vertices[1].texcoord,
                triangle.vertices[2].texcoord,
            ) {
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
                let color_array = tex.sample(tc.x, tc.y);
                Vector3::new(color_array[0], color_array[1], color_array[2])
            } else {
                Vector3::new(1.0, 1.0, 1.0)
            }
        }
        TextureSource::FaceColor(seed) => get_random_color(*seed, true),
        TextureSource::SolidColor(color) => *color,
        TextureSource::None => Vector3::new(1.0, 1.0, 1.0),
    }
}

fn calculate_ambient_contribution(triangle: &TriangleData) -> Vector3<f32> {
    let ambient_color = triangle.ambient_color;
    let ambient_intensity = triangle.ambient_intensity;

    let ambient = Vector3::new(
        ambient_color.x * ambient_intensity,
        ambient_color.y * ambient_intensity,
        ambient_color.z * ambient_intensity,
    );

    if let Some(material_view) = &triangle.material_view {
        let material = match material_view {
            MaterialView::BlinnPhong(material) => material,
            MaterialView::PBR(material) => material,
        };

        return Vector3::new(
            material.ambient_factor.x * ambient.x,
            material.ambient_factor.y * ambient.y,
            material.ambient_factor.z * ambient.z,
        );
    }

    ambient
}

// ===== Alpha和颜色处理 =====

fn get_effective_alpha(triangle: &TriangleData, settings: &RenderSettings) -> f32 {
    let material_alpha = if let Some(material_view) = &triangle.material_view {
        match material_view {
            MaterialView::BlinnPhong(material) => material.alpha,
            MaterialView::PBR(material) => material.alpha,
        }
    } else {
        1.0
    };

    (material_alpha * settings.alpha).clamp(0.0, 1.0)
}

fn apply_alpha_blending(
    material_color: &Vector3<f32>,
    alpha: f32,
    pixel_x: usize,
    pixel_y: usize,
    frame_buffer: &FrameBuffer,
) -> Vector3<f32> {
    if alpha >= 1.0 {
        return *material_color;
    }

    if alpha <= 0.0 {
        return if let Some(bg_color) = frame_buffer.get_pixel_color(pixel_x, pixel_y) {
            bg_color
        } else {
            Vector3::new(0.0, 0.0, 0.0)
        };
    }

    let background_color = frame_buffer.get_pixel_color_as_color(pixel_x, pixel_y);

    Vector3::new(
        material_color.x * alpha + background_color.x * (1.0 - alpha),
        material_color.y * alpha + background_color.y * (1.0 - alpha),
        material_color.z * alpha + background_color.z * (1.0 - alpha),
    )
}

#[inline]
fn write_pixel_color(
    pixel_index: usize,
    color: &Vector3<f32>,
    color_buffer: &[AtomicU8],
    apply_gamma: bool,
) {
    let buffer_start_index = pixel_index * 3;
    if buffer_start_index + 2 < color_buffer.len() {
        let [r, g, b] = linear_rgb_to_u8(color, apply_gamma);
        color_buffer[buffer_start_index].store(r, Ordering::Relaxed);
        color_buffer[buffer_start_index + 1].store(g, Ordering::Relaxed);
        color_buffer[buffer_start_index + 2].store(b, Ordering::Relaxed);
    }
}
