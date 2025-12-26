use nalgebra::{Matrix4, Point2, Point3, Vector3, Vector4};

//=================================
// Transform Matrix Factory
//=================================

/// Factory for creating various transformation matrices.
/// Manually implemented to ensure control over the coordinate system (Right-Handed).
/// And more educational than using nalgebra's built-in functions directly.
pub struct TransformFactory;

#[rustfmt::skip]
impl TransformFactory {
    /// Creates a rotation matrix around an arbitrary axis using Rodrigues' rotation formula.
    pub fn rotation(axis: &Vector3<f32>, angle_rad: f32) -> Matrix4<f32> {
        let axis_unit = axis.normalize();
        let x = axis_unit.x;
        let y = axis_unit.y;
        let z = axis_unit.z;
        let c = angle_rad.cos();
        let s = angle_rad.sin();
        let t = 1.0 - c;

        Matrix4::new(
            t * x * x + c,     t * x * y - z * s, t * x * z + y * s, 0.0,
            t * x * y + z * s, t * y * y + c,     t * y * z - x * s, 0.0,
            t * x * z - y * s, t * y * z + x * s, t * z * z + c,     0.0,
            0.0,               0.0,               0.0,               1.0,
        )
    }

    /// Creates a rotation matrix around the X-axis.
    pub fn rotation_x(angle_rad: f32) -> Matrix4<f32> {
        let c = angle_rad.cos();
        let s = angle_rad.sin();
        Matrix4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, c,  -s,   0.0,
            0.0, s,   c,   0.0,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    /// Creates a rotation matrix around the Y-axis.
    pub fn rotation_y(angle_rad: f32) -> Matrix4<f32> {
        let c = angle_rad.cos();
        let s = angle_rad.sin();
        Matrix4::new(
            c,   0.0, s,   0.0,
            0.0, 1.0, 0.0, 0.0,
           -s,   0.0, c,   0.0,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    /// Creates a rotation matrix around the Z-axis.
    pub fn rotation_z(angle_rad: f32) -> Matrix4<f32> {
        let c = angle_rad.cos();
        let s = angle_rad.sin();
        Matrix4::new(
            c,  -s,   0.0, 0.0,
            s,   c,   0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    /// Creates a translation matrix.
    pub fn translation(translation: &Vector3<f32>) -> Matrix4<f32> {
        Matrix4::new(
            1.0, 0.0, 0.0, translation.x,
            0.0, 1.0, 0.0, translation.y,
            0.0, 0.0, 1.0, translation.z,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    /// Creates a non-uniform scaling matrix.
    pub fn scaling_nonuniform(scale: &Vector3<f32>) -> Matrix4<f32> {
        Matrix4::new(
            scale.x, 0.0,     0.0,     0.0,
            0.0,     scale.y, 0.0,     0.0,
            0.0,     0.0,     scale.z, 0.0,
            0.0,     0.0,     0.0,     1.0,
        )
    }

    /// Creates a View matrix (Look-At, Right-Handed).
    /// Transforms world space coordinates to camera/view space.
    pub fn view(eye: &Point3<f32>, target: &Point3<f32>, up: &Vector3<f32>) -> Matrix4<f32> {
        // In RHS, camera looks down -Z
        let z_axis = (eye - target).normalize(); 
        let x_axis = up.cross(&z_axis).normalize();
        let y_axis = z_axis.cross(&x_axis);

        // Rotation matrix from world to view
        let rotation = Matrix4::new(
            x_axis.x, x_axis.y, x_axis.z, 0.0,
            y_axis.x, y_axis.y, y_axis.z, 0.0,
            z_axis.x, z_axis.y, z_axis.z, 0.0,
            0.0,      0.0,      0.0,      1.0,
        );

        // Translation matrix to move camera to origin
        let translation = Self::translation(&-eye.coords);

        rotation * translation
    }

    /// Creates a Perspective Projection matrix (Right-Handed).
    /// Maps view frustum to NDC [-1, 1].
    pub fn perspective(aspect_ratio: f32, fov_y_rad: f32, near: f32, far: f32) -> Matrix4<f32> {
        let f = 1.0 / (fov_y_rad / 2.0).tan();
        let nf = 1.0 / (near - far);

        Matrix4::new(
            f / aspect_ratio, 0.0, 0.0,                          0.0,
            0.0,              f,   0.0,                          0.0,
            0.0,              0.0, (far + near) * nf,            2.0 * far * near * nf,
            0.0,              0.0, -1.0,                         0.0,
        )
    }

    /// Creates an Orthographic Projection matrix (Right-Handed).
    pub fn orthographic(
        left: f32,
        right: f32,
        bottom: f32,
        top: f32,
        near: f32,
        far: f32,
    ) -> Matrix4<f32> {
        let rl = 1.0 / (right - left);
        let tb = 1.0 / (top - bottom);
        let nf = 1.0 / (near - far);

        Matrix4::new(
            2.0 * rl,      0.0,           0.0,          -(right + left) * rl,
            0.0,           2.0 * tb,      0.0,          -(top + bottom) * tb,
            0.0,           0.0,           2.0 * nf,     (far + near) * nf,
            0.0,           0.0,           0.0,          1.0,
        )
    }
}

//=================================
// Core Transformation Functions
//=================================

/// Performs perspective division: Clip Space -> NDC.
#[inline]
pub fn apply_perspective_division(clip: &Vector4<f32>) -> Point3<f32> {
    let w = clip.w;
    if w.abs() > 1e-6 {
        Point3::new(clip.x / w, clip.y / w, clip.z / w)
    } else {
        Point3::origin()
    }
}

/// Converts NDC coordinates to Screen coordinates (Viewport Transform).
/// Note: Y-axis is flipped (NDC +Y is up, Screen +Y is down).
#[inline]
pub fn ndc_to_screen(ndc_x: f32, ndc_y: f32, width: f32, height: f32) -> Point2<f32> {
    Point2::new(
        (ndc_x + 1.0) * 0.5 * width,
        (1.0 - (ndc_y + 1.0) * 0.5) * height,
    )
}
