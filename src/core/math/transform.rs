use nalgebra::{Matrix3, Matrix4, Point2, Point3, Vector3, Vector4};

/// Constructs matrices using the renderer's right-handed coordinate conventions.
pub(crate) struct TransformFactory;

#[rustfmt::skip]
impl TransformFactory {
    pub(crate) fn rotation_x(angle_rad: f32) -> Matrix4<f32> {
        let c = angle_rad.cos();
        let s = angle_rad.sin();
        Matrix4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, c,  -s,   0.0,
            0.0, s,   c,   0.0,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    pub(crate) fn rotation_y(angle_rad: f32) -> Matrix4<f32> {
        let c = angle_rad.cos();
        let s = angle_rad.sin();
        Matrix4::new(
            c,   0.0, s,   0.0,
            0.0, 1.0, 0.0, 0.0,
           -s,   0.0, c,   0.0,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    pub(crate) fn rotation_z(angle_rad: f32) -> Matrix4<f32> {
        let c = angle_rad.cos();
        let s = angle_rad.sin();
        Matrix4::new(
            c,  -s,   0.0, 0.0,
            s,   c,   0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    pub(crate) fn translation(translation: &Vector3<f32>) -> Matrix4<f32> {
        Matrix4::new(
            1.0, 0.0, 0.0, translation.x,
            0.0, 1.0, 0.0, translation.y,
            0.0, 0.0, 1.0, translation.z,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    pub(crate) fn scaling_nonuniform(scale: &Vector3<f32>) -> Matrix4<f32> {
        Matrix4::new(
            scale.x, 0.0,     0.0,     0.0,
            0.0,     scale.y, 0.0,     0.0,
            0.0,     0.0,     scale.z, 0.0,
            0.0,     0.0,     0.0,     1.0,
        )
    }

    /// Builds a view matrix with camera forward along negative Z.
    pub(crate) fn view(eye: &Point3<f32>, target: &Point3<f32>, up: &Vector3<f32>) -> Matrix4<f32> {
        let z_axis = (eye - target).normalize(); 
        let x_axis = up.cross(&z_axis).normalize();
        let y_axis = z_axis.cross(&x_axis);

        let rotation = Matrix4::new(
            x_axis.x, x_axis.y, x_axis.z, 0.0,
            y_axis.x, y_axis.y, y_axis.z, 0.0,
            z_axis.x, z_axis.y, z_axis.z, 0.0,
            0.0,      0.0,      0.0,      1.0,
        );

        let translation = Self::translation(&-eye.coords);

        rotation * translation
    }

    /// Builds a right-handed perspective projection with depth mapped to NDC `[-1, 1]`.
    pub(crate) fn perspective(aspect_ratio: f32, fov_y_rad: f32, near: f32, far: f32) -> Matrix4<f32> {
        let f = 1.0 / (fov_y_rad / 2.0).tan();
        let nf = 1.0 / (near - far);

        Matrix4::new(
            f / aspect_ratio, 0.0, 0.0,                          0.0,
            0.0,              f,   0.0,                          0.0,
            0.0,              0.0, (far + near) * nf,            2.0 * far * near * nf,
            0.0,              0.0, -1.0,                         0.0,
        )
    }

    /// Builds a right-handed orthographic projection with depth mapped to NDC `[-1, 1]`.
    pub(crate) fn orthographic(
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

/// Divides clip-space XYZ by W, returning the origin when W is too close to zero.
#[inline]
pub(crate) fn apply_perspective_division(clip: &Vector4<f32>) -> Point3<f32> {
    let w = clip.w;
    if w.abs() > 1e-6 {
        Point3::new(clip.x / w, clip.y / w, clip.z / w)
    } else {
        Point3::origin()
    }
}

/// Converts NDC coordinates to screen coordinates, flipping NDC +Y into screen -Y.
#[inline]
pub(crate) fn ndc_to_screen(ndc_x: f32, ndc_y: f32, width: f32, height: f32) -> Point2<f32> {
    Point2::new(
        (ndc_x + 1.0) * 0.5 * width,
        (1.0 - (ndc_y + 1.0) * 0.5) * height,
    )
}

/// Precomputed normal/tangent transform for a linear object-space transform.
#[derive(Clone, Copy)]
pub struct TangentFrameTransform {
    linear: Matrix3<f32>,
    normal: Matrix3<f32>,
    orientation: f32,
}

impl TangentFrameTransform {
    pub fn new(linear: Matrix3<f32>) -> Self {
        let normal = linear
            .try_inverse()
            .map(|matrix| matrix.transpose())
            .unwrap_or(linear);
        let orientation = if linear.determinant() < 0.0 {
            -1.0
        } else {
            1.0
        };
        Self {
            linear,
            normal,
            orientation,
        }
    }

    pub fn transform(
        &self,
        normal: Vector3<f32>,
        tangent: Vector4<f32>,
    ) -> (Vector3<f32>, Vector4<f32>) {
        let transformed_normal = (self.normal * normal).normalize();
        let tangent_direction = tangent.xyz();
        if tangent_direction.norm_squared() <= f32::EPSILON {
            return (transformed_normal, tangent);
        }

        let transformed_tangent = self.linear * tangent_direction;
        let orthogonal_tangent =
            transformed_tangent - transformed_normal * transformed_normal.dot(&transformed_tangent);
        let tangent_direction = orthogonal_tangent
            .try_normalize(f32::EPSILON)
            .unwrap_or_else(Vector3::zeros);

        (
            transformed_normal,
            Vector4::new(
                tangent_direction.x,
                tangent_direction.y,
                tangent_direction.z,
                tangent.w * self.orientation,
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_point_approx(actual: Point3<f32>, expected: Point3<f32>) {
        assert!(
            (actual - expected).norm() < 1e-5,
            "expected {expected:?}, got {actual:?}"
        );
    }

    #[test]
    fn translation_moves_points_without_affecting_directions() {
        let matrix = TransformFactory::translation(&Vector3::new(2.0, -3.0, 4.0));
        assert_point_approx(
            matrix.transform_point(&Point3::new(1.0, 2.0, 3.0)),
            Point3::new(3.0, -1.0, 7.0),
        );
        assert_eq!(
            matrix.transform_vector(&Vector3::new(1.0, 2.0, 3.0)),
            Vector3::new(1.0, 2.0, 3.0)
        );
    }

    #[test]
    fn object_transform_composes_scale_rotation_and_translation() {
        let transform = TransformFactory::translation(&Vector3::new(1.0, 2.0, 3.0))
            * TransformFactory::rotation_y(std::f32::consts::FRAC_PI_2)
            * TransformFactory::scaling_nonuniform(&Vector3::new(2.0, 1.0, 1.0));

        assert_point_approx(
            transform.transform_point(&Point3::new(1.0, 0.0, 0.0)),
            Point3::new(1.0, 2.0, 1.0),
        );
    }

    #[test]
    fn tangent_frame_handles_non_uniform_and_mirrored_transforms() {
        let non_uniform = Matrix3::new(2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.5);
        let (normal, tangent) = TangentFrameTransform::new(non_uniform)
            .transform(Vector3::z(), Vector4::new(1.0, 1.0, 0.0, 1.0));
        assert_point_approx(Point3::from(normal), Point3::new(0.0, 0.0, 1.0));
        assert!((tangent.xyz() - Vector3::new(2.0, 3.0, 0.0).normalize()).norm() < 1e-5);
        assert_eq!(tangent.w, 1.0);

        let mirrored = Matrix3::new(-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        let (_, tangent) = TangentFrameTransform::new(mirrored)
            .transform(Vector3::z(), Vector4::new(1.0, 0.0, 0.0, 1.0));
        assert!((tangent.xyz() + Vector3::x()).norm() < 1e-5);
        assert_eq!(tangent.w, -1.0);
    }

    #[test]
    fn tangent_frame_preserves_missing_tangents() {
        let (_, tangent) = TangentFrameTransform::new(Matrix3::identity())
            .transform(Vector3::z(), Vector4::new(0.0, 0.0, 0.0, 1.0));
        assert_eq!(tangent, Vector4::new(0.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn view_maps_camera_to_origin_and_target_down_negative_z() {
        let eye = Point3::new(0.0, 0.0, 5.0);
        let target = Point3::origin();
        let view = TransformFactory::view(&eye, &target, &Vector3::y());

        assert_point_approx(view.transform_point(&eye), Point3::origin());
        assert_point_approx(view.transform_point(&target), Point3::new(0.0, 0.0, -5.0));
    }

    #[test]
    fn perspective_maps_near_and_far_planes_to_ndc_range() {
        let projection = TransformFactory::perspective(1.0, 60.0_f32.to_radians(), 0.1, 100.0);
        let near = apply_perspective_division(&(projection * Vector4::new(0.0, 0.0, -0.1, 1.0)));
        let far = apply_perspective_division(&(projection * Vector4::new(0.0, 0.0, -100.0, 1.0)));

        assert!((near.z + 1.0).abs() < 1e-4, "near z was {}", near.z);
        assert!((far.z - 1.0).abs() < 1e-4, "far z was {}", far.z);
    }

    #[test]
    fn viewport_flips_y_axis() {
        assert_eq!(ndc_to_screen(-1.0, 1.0, 100.0, 50.0), Point2::new(0.0, 0.0));
        assert_eq!(
            ndc_to_screen(1.0, -1.0, 100.0, 50.0),
            Point2::new(100.0, 50.0)
        );
    }
}
