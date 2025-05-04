use nalgebra::{Point2, Point3, Vector2, Vector3};

const EPSILON: f32 = 1e-5; // Small value for float comparisons

/// Calculates barycentric coordinates (alpha, beta, gamma) for point p
/// with respect to the 2D triangle (v1, v2, v3).
/// Returns None if the triangle is degenerate.
/// Alpha corresponds to v1, Beta to v2, Gamma to v3.
pub fn barycentric_coordinates(
    p: Point2<f32>,
    v1: Point2<f32>,
    v2: Point2<f32>,
    v3: Point2<f32>,
) -> Option<Vector3<f32>> {
    let e1 = v2 - v1;
    let e2 = v3 - v1;
    let p_v1 = p - v1;

    // Area of the main triangle (times 2) using 2D cross product determinant
    let total_area_x2 = e1.x * e2.y - e1.y * e2.x;

    if total_area_x2.abs() < EPSILON {
        return None; // Degenerate triangle
    }

    let inv_total_area_x2 = 1.0 / total_area_x2;

    // Area of subtriangle opposite v2 (p, v3, v1) / total_area -> bary for v2 (beta)
    let area2_x2 = p_v1.x * e2.y - p_v1.y * e2.x;
    let beta = area2_x2 * inv_total_area_x2;

    // Area of subtriangle opposite v3 (p, v1, v2) / total_area -> bary for v3 (gamma)
    let area3_x2 = e1.x * p_v1.y - e1.y * p_v1.x;
    let gamma = area3_x2 * inv_total_area_x2;

    // Bary for v1 (alpha)
    let alpha = 1.0 - beta - gamma;

    Some(Vector3::new(alpha, beta, gamma))
}

/// Checks if the barycentric coordinates indicate the point is inside the triangle.
#[inline(always)]
pub fn is_inside_triangle(bary: Vector3<f32>) -> bool {
    bary.x >= -EPSILON && bary.y >= -EPSILON && bary.z >= -EPSILON
    // Optional stricter check: && (bary.x + bary.y + bary.z <= 1.0 + EPSILON)
    // Usually the >= -epsilon check is sufficient if barycentric_coordinates is correct.
}

/// Interpolates depth (z) using barycentric coordinates, with perspective correction.
/// Takes view-space Z values (typically negative).
/// Returns positive depth for buffer comparison, or f32::INFINITY if invalid.
pub fn interpolate_depth(
    bary: Vector3<f32>,
    z1_view: f32,
    z2_view: f32,
    z3_view: f32,
    is_perspective: bool,
) -> f32 {
    if !is_inside_triangle(bary) {
        return f32::INFINITY;
    }

    let interpolated_z = if !is_perspective {
        // Orthographic: Linear interpolation
        bary.x * z1_view + bary.y * z2_view + bary.z * z3_view
    } else {
        // Perspective: Interpolate 1/z
        let inv_z1 = if z1_view.abs() > EPSILON {
            1.0 / z1_view
        } else {
            0.0
        };
        let inv_z2 = if z2_view.abs() > EPSILON {
            1.0 / z2_view
        } else {
            0.0
        };
        let inv_z3 = if z3_view.abs() > EPSILON {
            1.0 / z3_view
        } else {
            0.0
        };

        let interpolated_inv_z = bary.x * inv_z1 + bary.y * inv_z2 + bary.z * inv_z3;

        if interpolated_inv_z.abs() > EPSILON {
            1.0 / interpolated_inv_z
        } else {
            // Fallback to linear if perspective correction fails (e.g., division by zero)
            bary.x * z1_view + bary.y * z2_view + bary.z * z3_view
        }
    };

    // Return positive depth for buffer (smaller is closer)
    // If interpolated_z is positive (behind camera in view space), map to infinity?
    if interpolated_z > -EPSILON {
        // Check if behind or very close to near plane
        f32::INFINITY
    } else {
        -interpolated_z
    }
}

/// Interpolates texture coordinates (UV) using barycentric coordinates, with perspective correction.
/// Takes view-space Z values for correction.
pub fn interpolate_texcoords(
    bary: Vector3<f32>,
    tc1: Vector2<f32>,
    tc2: Vector2<f32>,
    tc3: Vector2<f32>,
    z1_view: f32,
    z2_view: f32,
    z3_view: f32,
    is_perspective: bool,
) -> Vector2<f32> {
    // Note: Assumes is_inside_triangle check happened before calling this

    if !is_perspective {
        // Orthographic: Linear interpolation
        tc1 * bary.x + tc2 * bary.y + tc3 * bary.z
    } else {
        // Perspective: Interpolate attribute/z
        let inv_z1 = if z1_view.abs() > EPSILON {
            1.0 / z1_view
        } else {
            0.0
        };
        let inv_z2 = if z2_view.abs() > EPSILON {
            1.0 / z2_view
        } else {
            0.0
        };
        let inv_z3 = if z3_view.abs() > EPSILON {
            1.0 / z3_view
        } else {
            0.0
        };

        let interpolated_inv_z = bary.x * inv_z1 + bary.y * inv_z2 + bary.z * inv_z3;

        if interpolated_inv_z.abs() > EPSILON {
            let inv_z = 1.0 / interpolated_inv_z;
            // Interpolate tc/z
            let interp_tc_over_z =
                tc1 * (bary.x * inv_z1) + tc2 * (bary.y * inv_z2) + tc3 * (bary.z * inv_z3);
            // Multiply by interpolated z (which is 1/interpolated_inv_z)
            interp_tc_over_z * inv_z
        } else {
            // Fallback to linear interpolation
            tc1 * bary.x + tc2 * bary.y + tc3 * bary.z
        }
    }
}
