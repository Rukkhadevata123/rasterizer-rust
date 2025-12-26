use crate::core::math::transform::TransformFactory;
use nalgebra::{Matrix4, Point3, Vector3};

/// Manages the View and Projection matrices.
#[derive(Debug, Clone)]
pub struct Camera {
    // --- Parameters ---
    pub position: Point3<f32>,
    pub target: Point3<f32>,
    pub up: Vector3<f32>,

    pub fov_y_rad: f32,
    pub aspect_ratio: f32,
    pub near: f32,
    pub far: f32,

    // --- Cached Matrices ---
    view_matrix: Matrix4<f32>,
    projection_matrix: Matrix4<f32>,
}

impl Camera {
    /// Creates a new perspective camera.
    pub fn new(
        position: Point3<f32>,
        target: Point3<f32>,
        up: Vector3<f32>,
        fov_y_rad: f32,
        aspect_ratio: f32,
        near: f32,
        far: f32,
    ) -> Self {
        let mut cam = Self {
            position,
            target,
            up,
            fov_y_rad,
            aspect_ratio,
            near,
            far,
            view_matrix: Matrix4::identity(),
            projection_matrix: Matrix4::identity(),
        };
        cam.update_matrices();
        cam
    }

    /// Recalculates View and Projection matrices based on current parameters.
    pub fn update_matrices(&mut self) {
        self.view_matrix = TransformFactory::view(&self.position, &self.target, &self.up);
        self.projection_matrix =
            TransformFactory::perspective(self.aspect_ratio, self.fov_y_rad, self.near, self.far);
    }

    pub fn view_matrix(&self) -> Matrix4<f32> {
        self.view_matrix
    }

    pub fn projection_matrix(&self) -> Matrix4<f32> {
        self.projection_matrix
    }

    /// Orbits the camera around the target point.
    /// Useful for simple model viewers.
    pub fn orbit(&mut self, axis: Vector3<f32>, angle_rad: f32) {
        let rotation = TransformFactory::rotation(&axis, angle_rad);
        let current_offset = self.position - self.target;
        let new_offset = rotation.transform_vector(&current_offset);
        self.position = self.target + new_offset;
        self.update_matrices();
    }
    // TODO: Future: Implement camera movement methods (pan, zoom, etc.)
}
