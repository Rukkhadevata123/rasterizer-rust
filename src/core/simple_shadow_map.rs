use crate::geometry::interpolation::barycentric_coordinates;
use crate::geometry::transform::{TransformFactory, transform_point};
use crate::material_system::light::Light;
use crate::scene::scene_object::SceneObject;
use log::debug;
use nalgebra::{Matrix4, Point2, Point3, Vector3};

/// 极简阴影贴图 - 专为地面阴影设计
#[derive(Debug, Clone)]
pub struct SimpleShadowMap {
    /// 深度缓冲区（线性存储）
    pub depth_buffer: Vec<f32>,
    /// 阴影贴图尺寸
    pub size: usize,
    /// 光源的视图投影矩阵
    pub light_view_proj_matrix: Matrix4<f32>,
    /// 是否有效
    pub is_valid: bool,
}

impl SimpleShadowMap {
    pub fn new(size: usize) -> Self {
        Self {
            depth_buffer: vec![f32::INFINITY; size * size],
            size,
            light_view_proj_matrix: Matrix4::identity(),
            is_valid: false,
        }
    }

    /// 生成阴影贴图
    pub fn generate(
        &mut self,
        scene_object: &SceneObject,
        directional_light: &Light,
        scene_bounds: (Point3<f32>, f32), // (center, radius)
    ) -> bool {
        // 只支持方向光
        if let Light::Directional {
            direction, enabled, ..
        } = directional_light
        {
            if !enabled {
                return false;
            }

            self.setup_light_matrices(direction, scene_bounds);
            self.render_shadow_casters(scene_object);
            self.is_valid = true;
            debug!("阴影贴图生成完成: {}x{}", self.size, self.size);
            true
        } else {
            false
        }
    }

    /// 设置光源矩阵
    fn setup_light_matrices(
        &mut self,
        light_direction: &Vector3<f32>,
        (scene_center, scene_radius): (Point3<f32>, f32),
    ) {
        // 计算光源位置（在场景后方）
        let light_pos = scene_center - light_direction * scene_radius * 2.0;

        // 构建光源视图矩阵
        let up = if light_direction.y.abs() > 0.9 {
            Vector3::x() // 如果光线接近垂直，使用X轴
        } else {
            Vector3::y()
        };

        let light_view = TransformFactory::view(&light_pos, &scene_center, &up);

        // 正交投影（覆盖场景）
        let ortho_size = scene_radius * 1.2;
        let light_proj = TransformFactory::orthographic(
            -ortho_size,
            ortho_size,
            -ortho_size,
            ortho_size,
            0.1,
            scene_radius * 4.0,
        );

        self.light_view_proj_matrix = light_proj * light_view;
    }

    /// 渲染阴影投射物（简化版本）
    fn render_shadow_casters(&mut self, scene_object: &SceneObject) {
        // 清空深度缓冲区
        self.depth_buffer.fill(f32::INFINITY);

        // 预计算完整变换矩阵
        let full_transform = self.light_view_proj_matrix * scene_object.transform;

        let mut triangles_processed = 0;
        let mut triangles_culled = 0;

        // 简化：只渲染主要几何体的深度
        for mesh in &scene_object.model_data.meshes {
            for indices in mesh.indices.chunks_exact(3) {
                let v0 = mesh.vertices[indices[0] as usize].position;
                let v1 = mesh.vertices[indices[1] as usize].position;
                let v2 = mesh.vertices[indices[2] as usize].position;

                // 直接变换到光源空间
                let transformed_v0 = transform_point(&v0, &full_transform);
                let transformed_v1 = transform_point(&v1, &full_transform);
                let transformed_v2 = transform_point(&v2, &full_transform);

                // 简单的三角形光栅化到阴影贴图
                if self.rasterize_shadow_triangle(transformed_v0, transformed_v1, transformed_v2) {
                    triangles_processed += 1;
                } else {
                    triangles_culled += 1;
                }
            }
        }

        debug!(
            "阴影三角形: 处理 {}, 剔除 {}",
            triangles_processed, triangles_culled
        );
    }

    /// 简化的三角形阴影光栅化 - 修复版本
    fn rasterize_shadow_triangle(
        &mut self,
        v0: Point3<f32>,
        v1: Point3<f32>,
        v2: Point3<f32>,
    ) -> bool {
        // 视锥剔除：检查三角形是否完全在NDC范围外
        let all_outside_left = v0.x < -1.0 && v1.x < -1.0 && v2.x < -1.0;
        let all_outside_right = v0.x > 1.0 && v1.x > 1.0 && v2.x > 1.0;
        let all_outside_bottom = v0.y < -1.0 && v1.y < -1.0 && v2.y < -1.0;
        let all_outside_top = v0.y > 1.0 && v1.y > 1.0 && v2.y > 1.0;
        let all_outside_near = v0.z < -1.0 && v1.z < -1.0 && v2.z < -1.0;
        let all_outside_far = v0.z > 1.0 && v1.z > 1.0 && v2.z > 1.0;

        if all_outside_left
            || all_outside_right
            || all_outside_bottom
            || all_outside_top
            || all_outside_near
            || all_outside_far
        {
            return false; // 被剔除
        }

        // 转换到屏幕坐标系统
        let screen_coords = [
            self.ndc_to_shadow_coord(v0.x, v0.y),
            self.ndc_to_shadow_coord(v1.x, v1.y),
            self.ndc_to_shadow_coord(v2.x, v2.y),
        ];

        // 计算包围盒
        let bbox = self.compute_triangle_bbox(&screen_coords);
        if bbox.is_empty() {
            return false;
        }

        // 简化的点-三角形测试和深度写入
        for y in bbox.min_y..=bbox.max_y {
            for x in bbox.min_x..=bbox.max_x {
                // 重心坐标计算
                if let Some(bary) = barycentric_coordinates(
                    Point2::new(x as f32, y as f32),
                    Point2::new(screen_coords[0].0, screen_coords[0].1),
                    Point2::new(screen_coords[1].0, screen_coords[1].1),
                    Point2::new(screen_coords[2].0, screen_coords[2].1),
                ) {
                    if bary.x >= 0.0 && bary.y >= 0.0 && bary.z >= 0.0 {
                        let depth = bary.x * v0.z + bary.y * v1.z + bary.z * v2.z;
                        let shadow_index = (y as usize) * self.size + (x as usize);

                        if shadow_index < self.depth_buffer.len()
                            && depth < self.depth_buffer[shadow_index]
                        {
                            self.depth_buffer[shadow_index] = depth;
                        }
                    }
                }
            }
        }

        true // 成功处理
    }

    /// NDC到阴影贴图坐标转换
    #[inline]
    fn ndc_to_shadow_coord(&self, ndc_x: f32, ndc_y: f32) -> (f32, f32) {
        let coord = |ndc: f32| (ndc + 1.0) * 0.5 * self.size as f32;
        (coord(ndc_x), coord(ndc_y))
    }

    /// 包围盒计算
    fn compute_triangle_bbox(&self, screen_coords: &[(f32, f32); 3]) -> TriangleBoundingBox {
        let coords_i32: [i32; 6] = [
            screen_coords[0].0 as i32,
            screen_coords[0].1 as i32,
            screen_coords[1].0 as i32,
            screen_coords[1].1 as i32,
            screen_coords[2].0 as i32,
            screen_coords[2].1 as i32,
        ];

        let min_x = coords_i32[0].min(coords_i32[2]).min(coords_i32[4]).max(0);
        let max_x = coords_i32[0]
            .max(coords_i32[2])
            .max(coords_i32[4])
            .min(self.size as i32 - 1);
        let min_y = coords_i32[1].min(coords_i32[3]).min(coords_i32[5]).max(0);
        let max_y = coords_i32[1]
            .max(coords_i32[3])
            .max(coords_i32[5])
            .min(self.size as i32 - 1);

        TriangleBoundingBox {
            min_x,
            max_x,
            min_y,
            max_y,
        }
    }

    /// 采样阴影贴图深度
    pub fn sample_depth(&self, u: f32, v: f32) -> f32 {
        if !self.is_valid || !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
            return f32::INFINITY;
        }

        let x = (u * (self.size - 1) as f32) as usize;
        let y = (v * (self.size - 1) as f32) as usize;
        let index = y * self.size + x;

        self.depth_buffer
            .get(index)
            .copied()
            .unwrap_or(f32::INFINITY)
    }

    /// 计算阴影因子
    pub fn compute_shadow_factor(
        &self,
        world_pos: &Point3<f32>,
        model_matrix: &Matrix4<f32>,
        bias: f32,
    ) -> f32 {
        if !self.is_valid {
            return 1.0;
        }

        // 直接变换到光源空间
        let full_transform = self.light_view_proj_matrix * model_matrix;
        let light_space_pos = transform_point(world_pos, &full_transform);

        // 转换到UV坐标
        let shadow_u = (light_space_pos.x + 1.0) * 0.5;
        let shadow_v = (light_space_pos.y + 1.0) * 0.5;

        // 边界检查
        if !(0.0..=1.0).contains(&shadow_u) || !(0.0..=1.0).contains(&shadow_v) {
            return 1.0;
        }

        // 深度比较
        let current_depth = light_space_pos.z;
        let shadow_depth = self.sample_depth(shadow_u, shadow_v);

        if current_depth - bias > shadow_depth {
            0.2 // 在阴影中
        } else {
            1.0 // 不在阴影中
        }
    }
}

/// 辅助结构：三角形包围盒
#[derive(Debug)]
struct TriangleBoundingBox {
    min_x: i32,
    max_x: i32,
    min_y: i32,
    max_y: i32,
}

impl TriangleBoundingBox {
    fn is_empty(&self) -> bool {
        self.min_x > self.max_x || self.min_y > self.max_y
    }
}
