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
    pub position: Point3<f32>,
    pub target: Point3<f32>,
    pub up: Vector3<f32>,
    pub near: f32,
    pub far: f32,
    pub projection_type: ProjectionType,
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
        height: f32,
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

    /// Recalculates the view and projection matrices from the current parameters.
    pub fn update_matrices(&mut self) {
        self.view_matrix = TransformFactory::view(&self.position, &self.target, &self.up);
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
                    half_width,
                    -half_height,
                    half_height,
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
