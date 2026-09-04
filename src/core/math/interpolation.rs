use nalgebra::Vector3;

const EPSILON: f32 = 1e-5;

/// Compute perspective-correct barycentric coordinates (alpha', beta', gamma').
///
/// The corrected barycentrics are defined as:
///   wa = alpha * (1/w1), wb = beta * (1/w2), wc = gamma * (1/w3)
///   sum = wa + wb + wc
///   alpha' = wa / sum, ...
///
/// Returns `None` when numerical instability is detected (sum near zero).
pub(crate) fn perspective_correct_barycentric(
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
