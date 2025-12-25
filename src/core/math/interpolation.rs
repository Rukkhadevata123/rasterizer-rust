use nalgebra::{Point2, Vector3};
use std::ops::{Add, Mul};

const EPSILON: f32 = 1e-5;

/// Calculates the barycentric coordinates (alpha, beta, gamma) of point p
/// with respect to triangle (v1, v2, v3).
///
/// Returns `None` if the triangle is degenerate (area is near zero).
///
/// # Returns
/// A Vector3 where:
/// - x: alpha (weight for v1)
/// - y: beta  (weight for v2)
/// - z: gamma (weight for v3)
pub fn barycentric_coordinates(
    p: Point2<f32>,
    v1: Point2<f32>,
    v2: Point2<f32>,
    v3: Point2<f32>,
) -> Option<Vector3<f32>> {
    let e1 = v2 - v1;
    let e2 = v3 - v1;
    let p_v1 = p - v1;

    // Calculate the determinant (2x area of the triangle)
    let total_area_x2 = e1.x * e2.y - e1.y * e2.x;

    if total_area_x2.abs() < EPSILON {
        return None; // Degenerate triangle
    }

    let inv_total_area_x2 = 1.0 / total_area_x2;

    // Calculate weight for v2 (beta)
    // Area of sub-triangle (p, v3, v1)
    let area2_x2 = p_v1.x * e2.y - p_v1.y * e2.x;
    let beta = area2_x2 * inv_total_area_x2;

    // Calculate weight for v3 (gamma)
    // Area of sub-triangle (p, v1, v2)
    let area3_x2 = e1.x * p_v1.y - e1.y * p_v1.x;
    let gamma = area3_x2 * inv_total_area_x2;

    // Calculate weight for v1 (alpha)
    let alpha = 1.0 - beta - gamma;

    Some(Vector3::new(alpha, beta, gamma))
}

/// Checks if the barycentric coordinates represent a point inside the triangle.
/// Returns true if alpha, beta, and gamma are all >= 0.
#[inline(always)]
pub fn is_inside_triangle(bary: Vector3<f32>) -> bool {
    bary.x >= -EPSILON && bary.y >= -EPSILON && bary.z >= -EPSILON
}

/// Performs perspective-correct interpolation for any attribute (color, UV, normal).
///
/// # Arguments
/// * `bary` - Barycentric coordinates in screen space.
/// * `v1`, `v2`, `v3` - Attribute values at the vertices.
/// * `w1`, `w2`, `w3` - Clip space W coordinates (usually View Space Depth) at vertices.
pub fn perspective_correct_interpolate<T>(
    bary: Vector3<f32>,
    v1: T,
    v2: T,
    v3: T,
    w1: f32,
    w2: f32,
    w3: f32,
) -> T
where
    T: Copy + Add<Output = T> + Mul<f32, Output = T>,
{
    // 1. Calculate 1/w for each vertex
    let inv_w1 = if w1.abs() > EPSILON { 1.0 / w1 } else { 1.0 };
    let inv_w2 = if w2.abs() > EPSILON { 1.0 / w2 } else { 1.0 };
    let inv_w3 = if w3.abs() > EPSILON { 1.0 / w3 } else { 1.0 };

    // 2. Interpolate 1/w linearly in screen space
    let inv_w = bary.x * inv_w1 + bary.y * inv_w2 + bary.z * inv_w3;

    // 3. Calculate w at the current pixel
    let w = if inv_w.abs() > EPSILON {
        1.0 / inv_w
    } else {
        1.0
    };

    // 4. Interpolate (Attribute / w) linearly
    // We multiply by w at the end to recover the correct attribute value
    // Formula: Result = ( (v1/w1)*alpha + (v2/w2)*beta + (v3/w3)*gamma ) * w

    let term1 = v1 * (inv_w1 * bary.x);
    let term2 = v2 * (inv_w2 * bary.y);
    let term3 = v3 * (inv_w3 * bary.z);

    (term1 + term2 + term3) * w
}
