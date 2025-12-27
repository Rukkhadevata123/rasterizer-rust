use crate::core::math::transform::TransformFactory;
use nalgebra::{Matrix4, Point3, Vector3};

#[derive(Debug, Clone)]
pub enum ProjectionType {
    Perspective { fov_y_rad: f32, aspect_ratio: f32 },
    Orthographic { height: f32, aspect_ratio: f32 },
}

/// Manages the View and Projection matrices.
#[derive(Debug, Clone)]
pub struct Camera {
    // --- Common Parameters ---
    pub position: Point3<f32>,
    pub target: Point3<f32>,
    pub up: Vector3<f32>,
    pub near: f32,
    pub far: f32,

    // --- Projection Specifics ---
    pub projection_type: ProjectionType,

    // --- Cached Matrices ---
    view_matrix: Matrix4<f32>,
    projection_matrix: Matrix4<f32>,
}

impl Camera {
    pub fn new_perspective(
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
            near,
            far,
            projection_type: ProjectionType::Perspective {
                fov_y_rad,
                aspect_ratio,
            },
            view_matrix: Matrix4::identity(),
            projection_matrix: Matrix4::identity(),
        };
        cam.update_matrices();
        cam
    }

    pub fn new_orthographic(
        position: Point3<f32>,
        target: Point3<f32>,
        up: Vector3<f32>,
        height: f32, // View height
        aspect_ratio: f32,
        near: f32,
        far: f32,
    ) -> Self {
        let mut cam = Self {
            position,
            target,
            up,
            near,
            far,
            projection_type: ProjectionType::Orthographic {
                height,
                aspect_ratio,
            },
            view_matrix: Matrix4::identity(),
            projection_matrix: Matrix4::identity(),
        };
        cam.update_matrices();
        cam
    }

    /// Recalculates View and Projection matrices based on current parameters.
    pub fn update_matrices(&mut self) {
        // 1. View Matrix (Same for both)
        self.view_matrix = TransformFactory::view(&self.position, &self.target, &self.up);

        // 2. Projection Matrix (Depends on type)
        self.projection_matrix = match self.projection_type {
            ProjectionType::Perspective {
                fov_y_rad,
                aspect_ratio,
            } => TransformFactory::perspective(aspect_ratio, fov_y_rad, self.near, self.far),

            ProjectionType::Orthographic {
                height,
                aspect_ratio,
            } => {
                let half_height = height / 2.0;
                let half_width = half_height * aspect_ratio;

                TransformFactory::orthographic(
                    -half_width,
                    half_width, // Left, Right
                    -half_height,
                    half_height, // Bottom, Top
                    self.near,
                    self.far,
                )
            }
        };
    }

    pub fn view_matrix(&self) -> Matrix4<f32> {
        self.view_matrix
    }

    pub fn projection_matrix(&self) -> Matrix4<f32> {
        self.projection_matrix
    }
}
