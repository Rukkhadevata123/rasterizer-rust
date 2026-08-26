use nalgebra::{Point2, Vector3};

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

/// Compute perspective-correct barycentric coordinates (alpha', beta', gamma').
///
/// The corrected barycentrics are defined as:
///   wa = alpha * (1/w1), wb = beta * (1/w2), wc = gamma * (1/w3)
///   sum = wa + wb + wc
///   alpha' = wa / sum, ...
///
/// Returns `None` when numerical instability is detected (sum near zero).
pub fn perspective_correct_barycentric(
    bary: Vector3<f32>,
    w1: f32,
    w2: f32,
    w3: f32,
) -> Option<Vector3<f32>> {
    // Avoid division by extremely small w values: clamp behavior promotes robustness
    let inv_w1 = if w1.abs() > EPSILON { 1.0 / w1 } else { 1.0 };
    let inv_w2 = if w2.abs() > EPSILON { 1.0 / w2 } else { 1.0 };
    let inv_w3 = if w3.abs() > EPSILON { 1.0 / w3 } else { 1.0 };

    let wa = bary.x * inv_w1;
    let wb = bary.y * inv_w2;
    let wc = bary.z * inv_w3;

    let sum = wa + wb + wc;
    if sum.abs() < EPSILON {
        return None;
    }
    let inv_sum = 1.0 / sum;
    Some(Vector3::new(wa * inv_sum, wb * inv_sum, wc * inv_sum))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_approx(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn returns_expected_barycentric_coordinates() {
        let bary = barycentric_coordinates(
            Point2::new(0.25, 0.25),
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.0, 1.0),
        )
        .expect("triangle is not degenerate");

        assert_approx(bary.x, 0.5);
        assert_approx(bary.y, 0.25);
        assert_approx(bary.z, 0.25);
        assert!(is_inside_triangle(bary));
    }

    #[test]
    fn rejects_degenerate_triangle() {
        let bary = barycentric_coordinates(
            Point2::new(0.5, 0.0),
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 0.0),
        );

        assert!(bary.is_none());
    }

    #[test]
    fn perspective_correction_favors_smaller_w() {
        let corrected = perspective_correct_barycentric(
            Vector3::new(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
            1.0,
            2.0,
            4.0,
        )
        .expect("weights have a finite sum");

        assert_approx(corrected.x, 4.0 / 7.0);
        assert_approx(corrected.y, 2.0 / 7.0);
        assert_approx(corrected.z, 1.0 / 7.0);
        assert_approx(corrected.sum(), 1.0);
    }
}
