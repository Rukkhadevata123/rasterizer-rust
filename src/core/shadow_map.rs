use crate::geometry::interpolation::barycentric_coordinates;
use crate::geometry::transform::{TransformFactory, transform_point};
use crate::material_system::light::Light;
use crate::scene::scene_object::SceneObject;
use log::debug;
use nalgebra::{Matrix4, Point2, Point3, Vector3};

/// 简化阴影贴图
#[derive(Debug, Clone)]
pub struct ShadowMap {
    pub depth_buffer: Vec<f32>,
    pub size: usize,
    pub light_view_proj_matrix: Matrix4<f32>,
    pub is_valid: bool,
}

impl ShadowMap {
    pub fn new(size: usize) -> Self {
        Self {
            depth_buffer: vec![f32::INFINITY; size * size],
            size,
            light_view_proj_matrix: Matrix4::identity(),
            is_valid: false,
        }
    }

    pub fn generate(
        &mut self,
        scene_object: &SceneObject,
        directional_light: &Light,
        scene_bounds: (Point3<f32>, f32),
    ) -> bool {
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

    fn setup_light_matrices(
        &mut self,
        light_direction: &Vector3<f32>,
        (scene_center, scene_radius): (Point3<f32>, f32),
    ) {
        let light_pos = scene_center - light_direction * scene_radius * 2.0;

        let up = if light_direction.y.abs() > 0.9 {
            Vector3::x()
        } else {
            Vector3::y()
        };

        let light_view = TransformFactory::view(&light_pos, &scene_center, &up);
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

    fn render_shadow_casters(&mut self, scene_object: &SceneObject) {
        self.depth_buffer.fill(f32::INFINITY);
        let full_transform = self.light_view_proj_matrix * scene_object.transform;

        let mut triangles_processed = 0;
        let mut triangles_culled = 0;

        for mesh in &scene_object.model.meshes {
            for indices in mesh.indices.chunks_exact(3) {
                let vertices = [
                    mesh.vertices[indices[0] as usize].position,
                    mesh.vertices[indices[1] as usize].position,
                    mesh.vertices[indices[2] as usize].position,
                ];

                let transformed_vertices = [
                    transform_point(&vertices[0], &full_transform),
                    transform_point(&vertices[1], &full_transform),
                    transform_point(&vertices[2], &full_transform),
                ];

                if self.is_triangle_outside_frustum(&transformed_vertices) {
                    triangles_culled += 1;
                    continue;
                }

                let screen_coords = [
                    self.ndc_to_shadow_coord(transformed_vertices[0].x, transformed_vertices[0].y),
                    self.ndc_to_shadow_coord(transformed_vertices[1].x, transformed_vertices[1].y),
                    self.ndc_to_shadow_coord(transformed_vertices[2].x, transformed_vertices[2].y),
                ];

                self.rasterize_triangle(&transformed_vertices, &screen_coords);
                triangles_processed += 1;
            }
        }

        debug!("阴影三角形: 处理 {triangles_processed}, 剔除 {triangles_culled}");
    }

    #[inline]
    fn is_triangle_outside_frustum(&self, vertices: &[Point3<f32>; 3]) -> bool {
        let outside_bounds = |get_axis: fn(&Point3<f32>) -> f32| {
            vertices.iter().all(|v| get_axis(v) < -1.0)
                || vertices.iter().all(|v| get_axis(v) > 1.0)
        };

        outside_bounds(|v| v.x) || outside_bounds(|v| v.y) || outside_bounds(|v| v.z)
    }

    fn rasterize_triangle(&mut self, vertices: &[Point3<f32>; 3], screen_coords: &[(f32, f32); 3]) {
        let (min_x, max_x) = screen_coords
            .iter()
            .map(|(x, _)| *x as i32)
            .fold((i32::MAX, i32::MIN), |(min_x, max_x), x| {
                (min_x.min(x), max_x.max(x))
            });

        let (min_y, max_y) = screen_coords
            .iter()
            .map(|(_, y)| *y as i32)
            .fold((i32::MAX, i32::MIN), |(min_y, max_y), y| {
                (min_y.min(y), max_y.max(y))
            });

        let min_x = min_x.max(0);
        let max_x = max_x.min(self.size as i32 - 1);
        let min_y = min_y.max(0);
        let max_y = max_y.min(self.size as i32 - 1);

        if max_x <= min_x || max_y <= min_y {
            return;
        }

        let triangle_points = [
            Point2::new(screen_coords[0].0, screen_coords[0].1),
            Point2::new(screen_coords[1].0, screen_coords[1].1),
            Point2::new(screen_coords[2].0, screen_coords[2].1),
        ];

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let pixel_point = Point2::new(x as f32, y as f32);

                if let Some(bary) = barycentric_coordinates(
                    pixel_point,
                    triangle_points[0],
                    triangle_points[1],
                    triangle_points[2],
                ) {
                    if bary.x >= 0.0 && bary.y >= 0.0 && bary.z >= 0.0 {
                        let depth = bary.x * vertices[0].z
                            + bary.y * vertices[1].z
                            + bary.z * vertices[2].z;
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
    }

    #[inline]
    fn ndc_to_shadow_coord(&self, ndc_x: f32, ndc_y: f32) -> (f32, f32) {
        let coord = |ndc: f32| (ndc + 1.0) * 0.5 * self.size as f32;
        (coord(ndc_x), coord(ndc_y))
    }

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

    pub fn compute_shadow_factor(
        &self,
        world_pos: &Point3<f32>,
        model_matrix: &Matrix4<f32>,
        bias: f32,
    ) -> f32 {
        if !self.is_valid {
            return 1.0;
        }

        let full_transform = self.light_view_proj_matrix * model_matrix;
        let light_space_pos = transform_point(world_pos, &full_transform);

        let shadow_coords = (
            (light_space_pos.x + 1.0) * 0.5,
            (light_space_pos.y + 1.0) * 0.5,
        );

        if !(0.0..=1.0).contains(&shadow_coords.0) || !(0.0..=1.0).contains(&shadow_coords.1) {
            return 1.0;
        }

        let current_depth = light_space_pos.z;
        let shadow_depth = self.sample_depth(shadow_coords.0, shadow_coords.1);

        if current_depth - bias > shadow_depth {
            0.2
        } else {
            1.0
        }
    }
}
