use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;

use crate::quadrature::gauss_legendre_axis;
use crate::{QuadraturePoint, QuadratureRule, ReferenceCell};

/// Construct the centroid rule on a unit simplex of runtime dimension `d`.
///
/// The point has barycentric coordinates `1 / (d + 1)` and weight `1 / d!`,
/// so the rule integrates every affine scalar function exactly in any
/// representable dimension.
///
/// # Errors
/// Returns `EQ0804` when the dimension is zero or `d!` cannot be represented
/// as a finite positive `f64` weight.
pub fn simplex_centroid_rule(dimension: usize) -> Result<QuadratureRule, Diagnostic> {
    // 170! is finite in binary64; 171! is not. Reject before iteration so an
    // adversarial runtime dimension cannot turn validation into unbounded work.
    if dimension == 0 || dimension > 170 {
        return Err(invalid_quadrature(
            "simplex centroid rule requires a dimension in 1..=170",
        ));
    }
    let reference_cell = ReferenceCell::simplex(dimension)?;
    let mut factorial = 1.0;
    for factor in 2..=dimension {
        factorial *= factor as f64;
        if !factorial.is_finite() {
            return Err(invalid_quadrature(
                "simplex centroid weight underflows its finite factorial contract",
            ));
        }
    }
    let weight = 1.0 / factorial;
    if weight == 0.0 {
        return Err(invalid_quadrature(
            "simplex centroid weight underflows to zero",
        ));
    }
    QuadratureRule::new(
        reference_cell,
        Some(1),
        vec![QuadraturePoint {
            coordinates: vec![1.0 / (dimension + 1) as f64; dimension],
            weight,
        }],
    )
}

/// Construct a positive tensor-product Duffy rule on a unit simplex.
///
/// Cube coordinates `t_i` on `[0, 1]^d` map to simplex coordinates
///
/// `x_i = t_i * product_{j < i}(1 - t_j)`.
///
/// The Jacobian is `product_i (1 - t_i)^(d - i - 1)`. With `n`
/// Gauss--Legendre points per cube axis, the rule integrates every
/// total-degree polynomial through `2n - d` exactly. Dimension is an explicit
/// part of the rule, so triangles and tetrahedra use the same construction
/// without pretending that their exactness is identical.
///
/// # Errors
/// Returns `EQ0804` for dimension zero, insufficient or unsupported axis
/// exactness, point-count overflow, or allocation failure.
pub fn simplex_duffy_gauss_legendre(
    dimension: usize,
    points_per_axis: usize,
) -> Result<QuadratureRule, Diagnostic> {
    let reference_cell = ReferenceCell::simplex(dimension)?;
    let axis = gauss_legendre_axis(points_per_axis)?;
    let exponent = u32::try_from(dimension)
        .map_err(|_| invalid_quadrature("simplex Duffy dimension exceeds u32 capacity"))?;
    let point_count = points_per_axis
        .checked_pow(exponent)
        .ok_or_else(|| invalid_quadrature("simplex Duffy point count overflows usize"))?;
    let exactness = points_per_axis
        .checked_mul(2)
        .and_then(|value| value.checked_sub(dimension))
        .ok_or_else(|| {
            invalid_quadrature(
                "simplex Duffy axis rule is too small for a non-negative exactness contract",
            )
        })?;
    let mut points = Vec::new();
    points
        .try_reserve_exact(point_count)
        .map_err(|_| invalid_quadrature("simplex Duffy allocation exceeds platform capacity"))?;
    for linear_index in 0..point_count {
        let mut remainder = linear_index;
        let mut axis_indices = vec![0; dimension];
        for index in axis_indices.iter_mut().rev() {
            *index = remainder % points_per_axis;
            remainder /= points_per_axis;
        }
        let mut coordinates = Vec::new();
        coordinates.try_reserve_exact(dimension).map_err(|_| {
            invalid_quadrature("simplex Duffy point dimension exceeds platform capacity")
        })?;
        let mut remaining = 1.0;
        let mut tensor_weight = 1.0;
        let mut jacobian = 1.0;
        for (coordinate_axis, &axis_index) in axis_indices.iter().enumerate() {
            let (symmetric_coordinate, symmetric_weight) = axis[axis_index];
            let coordinate = 0.5 * (symmetric_coordinate + 1.0);
            tensor_weight *= 0.5 * symmetric_weight;
            coordinates.push(remaining * coordinate);
            let complement = 1.0 - coordinate;
            for _ in (coordinate_axis + 1)..dimension {
                jacobian *= complement;
            }
            remaining *= complement;
        }
        points.push(QuadraturePoint {
            coordinates,
            weight: tensor_weight * jacobian,
        });
    }
    QuadratureRule::new(reference_cell, Some(exactness), points)
}

fn invalid_quadrature(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_QUADRATURE, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integrates_affine_coordinates_in_runtime_dimensions() {
        for dimension in 1..=8 {
            let rule = simplex_centroid_rule(dimension).unwrap();
            let volume = rule.points()[0].weight;
            assert_eq!(rule.polynomial_exactness(), Some(1));
            for axis in 0..dimension {
                let integral = rule
                    .points()
                    .iter()
                    .map(|point| point.weight * point.coordinates[axis])
                    .sum::<f64>();
                assert!((integral - volume / (dimension + 1) as f64).abs() < 1.0e-15);
            }
        }
    }

    #[test]
    fn rejects_zero_and_unrepresentable_dimension() {
        assert_eq!(
            simplex_centroid_rule(0).unwrap_err().code(),
            codes::INVALID_QUADRATURE
        );
        assert_eq!(
            simplex_centroid_rule(usize::MAX).unwrap_err().code(),
            codes::INVALID_QUADRATURE
        );
    }

    #[test]
    fn triangle_duffy_rule_integrates_total_degree_four() {
        let rule = simplex_duffy_gauss_legendre(2, 3).unwrap();
        assert_eq!(rule.polynomial_exactness(), Some(4));
        assert_eq!(rule.points().len(), 9);
        for x_degree in 0_i32..=4 {
            for y_degree in 0_i32..=(4 - x_degree) {
                let actual = rule
                    .points()
                    .iter()
                    .map(|point| {
                        point.weight
                            * point.coordinates[0].powi(x_degree)
                            * point.coordinates[1].powi(y_degree)
                    })
                    .sum::<f64>();
                let exact =
                    factorial(x_degree) * factorial(y_degree) / factorial(x_degree + y_degree + 2);
                assert!((actual - exact).abs() < 2.0e-15);
            }
        }
    }

    #[test]
    fn five_by_five_triangle_duffy_rule_integrates_total_degree_eight() {
        let rule = simplex_duffy_gauss_legendre(2, 5).unwrap();
        assert_eq!(rule.polynomial_exactness(), Some(8));
        assert_eq!(rule.points().len(), 25);
        for x_degree in 0_i32..=8 {
            for y_degree in 0_i32..=(8 - x_degree) {
                let actual = rule
                    .points()
                    .iter()
                    .map(|point| {
                        point.weight
                            * point.coordinates[0].powi(x_degree)
                            * point.coordinates[1].powi(y_degree)
                    })
                    .sum::<f64>();
                let exact =
                    factorial(x_degree) * factorial(y_degree) / factorial(x_degree + y_degree + 2);
                assert!((actual - exact).abs() < 3.0e-15);
            }
        }
    }

    #[test]
    fn six_by_six_by_six_tetrahedron_rule_integrates_total_degree_eight() {
        let rule = simplex_duffy_gauss_legendre(3, 6).unwrap();
        assert_eq!(rule.polynomial_exactness(), Some(9));
        assert_eq!(rule.points().len(), 216);
        for x_degree in 0_i32..=8 {
            for y_degree in 0_i32..=(8 - x_degree) {
                for z_degree in 0_i32..=(8 - x_degree - y_degree) {
                    let actual = rule
                        .points()
                        .iter()
                        .map(|point| {
                            point.weight
                                * point.coordinates[0].powi(x_degree)
                                * point.coordinates[1].powi(y_degree)
                                * point.coordinates[2].powi(z_degree)
                        })
                        .sum::<f64>();
                    let exact = factorial(x_degree) * factorial(y_degree) * factorial(z_degree)
                        / factorial(x_degree + y_degree + z_degree + 3);
                    assert!(
                        (actual - exact).abs() < 1.0e-14,
                        "failed monomial ({x_degree}, {y_degree}, {z_degree}): {actual} != {exact}",
                    );
                }
            }
        }
    }

    #[test]
    fn seven_by_seven_by_seven_tetrahedron_rule_integrates_total_degree_eleven() {
        let rule = simplex_duffy_gauss_legendre(3, 7).unwrap();
        assert_eq!(rule.polynomial_exactness(), Some(11));
        assert_eq!(rule.points().len(), 343);
        for x_degree in 0_i32..=11 {
            for y_degree in 0_i32..=(11 - x_degree) {
                for z_degree in 0_i32..=(11 - x_degree - y_degree) {
                    let actual = rule
                        .points()
                        .iter()
                        .map(|point| {
                            point.weight
                                * point.coordinates[0].powi(x_degree)
                                * point.coordinates[1].powi(y_degree)
                                * point.coordinates[2].powi(z_degree)
                        })
                        .sum::<f64>();
                    let exact = factorial(x_degree) * factorial(y_degree) * factorial(z_degree)
                        / factorial(x_degree + y_degree + z_degree + 3);
                    assert!(
                        (actual - exact).abs() < 2.0e-14,
                        "failed monomial ({x_degree}, {y_degree}, {z_degree}): {actual} != {exact}",
                    );
                }
            }
        }
    }

    #[test]
    fn simplex_duffy_rejects_an_axis_rule_without_dimension_exactness() {
        let error = simplex_duffy_gauss_legendre(3, 1).unwrap_err();
        assert_eq!(error.code(), codes::INVALID_QUADRATURE);
        assert!(error.message().contains("too small"));
    }

    fn factorial(value: i32) -> f64 {
        (2..=value).map(f64::from).product()
    }
}
