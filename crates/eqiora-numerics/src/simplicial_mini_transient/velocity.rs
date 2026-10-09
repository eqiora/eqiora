//! Velocity reconstruction and its exact directional linearization.
pub(super) fn evaluate_velocity<const D: usize>(
    coefficients: &[[f64; D]],
    basis: &[f64],
    gradients: &[Vec<f64>],
) -> ([f64; D], [[f64; D]; D]) {
    let mut value = [0.0; D];
    let mut gradient = [[0.0; D]; D];
    for local in 0..coefficients.len() {
        for component in 0..D {
            value[component] += coefficients[local][component] * basis[local];
            for axis in 0..D {
                gradient[component][axis] +=
                    coefficients[local][component] * gradients[local][axis];
            }
        }
    }
    (value, gradient)
}

pub(super) fn evaluate_velocity_tangent<const D: usize>(
    coefficients: &[[f64; D]],
    coefficient_tangents: &[[f64; D]],
    basis: &[f64],
    gradients: &[Vec<f64>],
    gradient_tangents: &[Vec<f64>],
) -> ([f64; D], [[f64; D]; D]) {
    let mut value = [0.0; D];
    let mut gradient = [[0.0; D]; D];
    for local in 0..coefficients.len() {
        for component in 0..D {
            value[component] += coefficient_tangents[local][component] * basis[local];
            for axis in 0..D {
                gradient[component][axis] += coefficient_tangents[local][component]
                    * gradients[local][axis]
                    + coefficients[local][component] * gradient_tangents[local][axis];
            }
        }
    }
    (value, gradient)
}
