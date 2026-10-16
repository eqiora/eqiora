use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;

/// Family of a reference integration cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReferenceCellFamily {
    /// Unique zero-dimensional cell.
    Point,
    /// Unit simplex in an arbitrary positive dimension.
    Simplex,
    /// Tensor-product cell `[-1, 1]^d` in an arbitrary positive dimension.
    Hypercube,
}

/// Validated reference-cell topology with an explicit runtime dimension.
///
/// Runtime dimension is intentional: imported meshes and mixed-cell artifacts
/// must be inspectable before backend lowering specializes a kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReferenceCell {
    family: ReferenceCellFamily,
    dimension: usize,
}

impl ReferenceCell {
    /// The unique reference point.
    #[must_use]
    pub const fn point() -> Self {
        Self {
            family: ReferenceCellFamily::Point,
            dimension: 0,
        }
    }

    /// Unit simplex of any positive dimension.
    ///
    /// # Errors
    /// Returns `EQ0804` for dimension zero.
    pub fn simplex(dimension: usize) -> Result<Self, Diagnostic> {
        positive_dimension(ReferenceCellFamily::Simplex, dimension)
    }

    /// Tensor-product reference cell `[-1, 1]^d` of any positive dimension.
    ///
    /// # Errors
    /// Returns `EQ0804` for dimension zero.
    pub fn hypercube(dimension: usize) -> Result<Self, Diagnostic> {
        positive_dimension(ReferenceCellFamily::Hypercube, dimension)
    }

    /// Canonical reference segment `[-1, 1]`.
    #[must_use]
    pub const fn segment() -> Self {
        Self {
            family: ReferenceCellFamily::Hypercube,
            dimension: 1,
        }
    }

    /// Cell family.
    #[must_use]
    pub const fn family(self) -> ReferenceCellFamily {
        self.family
    }

    /// Topological dimension.
    #[must_use]
    pub const fn dimension(self) -> usize {
        self.dimension
    }

    /// Whether finite coordinates lie in this reference cell.
    #[must_use]
    pub fn contains(self, coordinates: &[f64]) -> bool {
        if coordinates.len() != self.dimension
            || coordinates.iter().any(|coordinate| !coordinate.is_finite())
        {
            return false;
        }
        match self.family {
            ReferenceCellFamily::Point => coordinates.is_empty(),
            ReferenceCellFamily::Simplex => {
                let tolerance = 64.0 * f64::EPSILON * self.dimension as f64;
                coordinates.iter().all(|coordinate| *coordinate >= 0.0)
                    && coordinates.iter().sum::<f64>() <= 1.0 + tolerance
            }
            ReferenceCellFamily::Hypercube => coordinates
                .iter()
                .all(|coordinate| *coordinate >= -1.0 && *coordinate <= 1.0),
        }
    }
}

/// One weighted sample on a reference cell.
#[derive(Debug, Clone, PartialEq)]
pub struct QuadraturePoint {
    /// Runtime-dimensional reference coordinates.
    pub coordinates: Vec<f64>,
    /// Reference-cell weight.
    pub weight: f64,
}

/// Validated quadrature rule tied to one reference-cell topology.
#[derive(Debug, Clone, PartialEq)]
pub struct QuadratureRule {
    reference_cell: ReferenceCell,
    polynomial_exactness: Option<usize>,
    points: Vec<QuadraturePoint>,
}

impl QuadratureRule {
    /// Construct a custom rule in any dimension.
    ///
    /// `polynomial_exactness` is declarative evidence, not inferred. Use
    /// `None` for rules without a polynomial exactness contract.
    ///
    /// # Errors
    /// Returns `EQ0804` for an empty rule, non-finite data, inconsistent point
    /// dimensions, points outside the reference cell, or non-positive total
    /// measure.
    pub fn new(
        reference_cell: ReferenceCell,
        polynomial_exactness: Option<usize>,
        points: Vec<QuadraturePoint>,
    ) -> Result<Self, Diagnostic> {
        if points.is_empty() {
            return Err(invalid_quadrature(
                "quadrature rule requires at least one point",
            ));
        }
        for point in &points {
            if !point.weight.is_finite() || !reference_cell.contains(&point.coordinates) {
                return Err(invalid_quadrature(
                    "quadrature point is non-finite or outside its reference cell",
                ));
            }
        }
        let weight_sum = points.iter().map(|point| point.weight).sum::<f64>();
        if !weight_sum.is_finite() || weight_sum <= 0.0 {
            return Err(invalid_quadrature(
                "quadrature weights must have finite positive total measure",
            ));
        }
        Ok(Self {
            reference_cell,
            polynomial_exactness,
            points,
        })
    }

    /// Reference-cell topology and dimension.
    #[must_use]
    pub const fn reference_cell(&self) -> ReferenceCell {
        self.reference_cell
    }

    /// Declared maximum total polynomial degree integrated exactly.
    #[must_use]
    pub const fn polynomial_exactness(&self) -> Option<usize> {
        self.polynomial_exactness
    }

    /// Ordered quadrature samples.
    #[must_use]
    pub fn points(&self) -> &[QuadraturePoint] {
        &self.points
    }

    /// Canonical measure-one rule for a reference point.
    #[must_use]
    pub fn point() -> Self {
        Self {
            reference_cell: ReferenceCell::point(),
            polynomial_exactness: Some(0),
            points: vec![QuadraturePoint {
                coordinates: Vec::new(),
                weight: 1.0,
            }],
        }
    }

    /// Gauss–Legendre rule on `[-1, 1]` from the Legendre recurrence.
    ///
    /// # Errors
    /// Returns `EQ0804` for zero points, index/allocation overflow, or failed root convergence.
    pub fn gauss_legendre(point_count: usize) -> Result<Self, Diagnostic> {
        let points = gauss_legendre_axis(point_count)?;
        Self::new(
            ReferenceCell::segment(),
            Some(2 * point_count - 1),
            points
                .into_iter()
                .map(|(coordinate, weight)| QuadraturePoint {
                    coordinates: vec![coordinate],
                    weight,
                })
                .collect(),
        )
    }

    /// Tensor-product Gauss–Legendre rule on `[-1, 1]^d`.
    ///
    /// # Errors
    /// Returns `EQ0804` for dimension zero, an unsupported one-dimensional
    /// point count, point-count overflow, or allocation failure.
    pub fn tensor_product_gauss_legendre(
        dimension: usize,
        points_per_axis: usize,
    ) -> Result<Self, Diagnostic> {
        let reference_cell = ReferenceCell::hypercube(dimension)?;
        let axis = gauss_legendre_axis(points_per_axis)?;
        let exponent = u32::try_from(dimension).map_err(|_| {
            invalid_quadrature("tensor-product quadrature dimension exceeds u32 capacity")
        })?;
        let point_count = points_per_axis.checked_pow(exponent).ok_or_else(|| {
            invalid_quadrature("tensor-product quadrature point count overflows usize")
        })?;
        let mut points = Vec::new();
        points.try_reserve_exact(point_count).map_err(|_| {
            invalid_quadrature("tensor-product quadrature allocation exceeds platform capacity")
        })?;
        for linear_index in 0..point_count {
            let mut remainder = linear_index;
            let mut coordinates = Vec::new();
            coordinates.try_reserve_exact(dimension).map_err(|_| {
                invalid_quadrature("tensor-product point dimension exceeds platform capacity")
            })?;
            let mut weight = 1.0;
            for _ in 0..dimension {
                let axis_index = remainder % points_per_axis;
                remainder /= points_per_axis;
                coordinates.push(axis[axis_index].0);
                weight *= axis[axis_index].1;
            }
            points.push(QuadraturePoint {
                coordinates,
                weight,
            });
        }
        debug_assert_eq!(points.len(), point_count);
        Self::new(reference_cell, Some(2 * points_per_axis - 1), points)
    }
}

fn positive_dimension(
    family: ReferenceCellFamily,
    dimension: usize,
) -> Result<ReferenceCell, Diagnostic> {
    if dimension == 0 {
        Err(invalid_quadrature(
            "simplex and hypercube reference cells require positive dimension",
        ))
    } else {
        Ok(ReferenceCell { family, dimension })
    }
}

pub(crate) fn gauss_legendre_axis(point_count: usize) -> Result<Vec<(f64, f64)>, Diagnostic> {
    if point_count == 0
        || point_count.checked_mul(2).is_none()
        || (point_count as f64) > 2_f64.powi(52)
    {
        return Err(invalid_quadrature(
            "Gauss-Legendre degree exceeds exact index capacity",
        ));
    }
    let mut points = Vec::new();
    points
        .try_reserve_exact(point_count)
        .map_err(|_| invalid_quadrature("Gauss-Legendre allocation exceeds platform capacity"))?;
    points.resize(point_count, (0.0, 0.0));
    for index in 0..point_count.div_ceil(2) {
        let center = index == point_count - 1 - index;
        let mut root = if center {
            0.0
        } else {
            (std::f64::consts::PI * (index as f64 + 0.75) / (point_count as f64 + 0.5)).cos()
        };
        let mut converged = center;
        for _ in 0..64 {
            if converged {
                break;
            }
            let (value, derivative) = legendre_value_derivative(point_count, root);
            let correction = value / derivative;
            root -= correction;
            if !root.is_finite() || root.abs() >= 1.0 {
                break;
            }
            converged = correction.abs() <= 4.0 * f64::EPSILON * (1.0 + root.abs());
        }
        if !converged {
            return Err(invalid_quadrature(
                "Gauss-Legendre root iteration did not converge",
            ));
        }
        let (_, derivative) = legendre_value_derivative(point_count, root);
        // Legendre roots and weights: https://dlmf.nist.gov/3.5.v.
        let weight = 2.0 / ((1.0 - root * root) * derivative * derivative);
        if !weight.is_finite() || weight <= 0.0 {
            return Err(invalid_quadrature(
                "Gauss-Legendre weight lost finite positivity",
            ));
        }
        points[index] = (if center { 0.0 } else { -root }, weight);
        points[point_count - 1 - index] = (root, weight);
    }
    if points.windows(2).any(|pair| pair[0].0 >= pair[1].0) {
        return Err(invalid_quadrature(
            "Gauss-Legendre roots are not distinct and ordered",
        ));
    }
    Ok(points)
}

/// Differentiate the three-term Legendre recurrence together with its value.
fn legendre_value_derivative(degree: usize, x: f64) -> (f64, f64) {
    let (mut previous, mut value) = (1.0, x);
    let (mut previous_derivative, mut derivative) = (0.0, 1.0);
    for order in 2..=degree {
        let k = order as f64;
        let next = ((2.0 * k - 1.0) * x * value - (k - 1.0) * previous) / k;
        let next_derivative =
            ((2.0 * k - 1.0) * (value + x * derivative) - (k - 1.0) * previous_derivative) / k;
        (previous, value) = (value, next);
        (previous_derivative, derivative) = (derivative, next_derivative);
    }
    (value, derivative)
}

fn invalid_quadrature(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_QUADRATURE, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauss_rules_integrate_independent_monomial_moments() {
        for n in 1..=64 {
            let rule = QuadratureRule::gauss_legendre(n).unwrap();
            assert_eq!(rule.polynomial_exactness(), Some(2 * n - 1));
            assert_eq!(rule.points().len(), n);
            for degree in 0..2 * n {
                let actual = rule
                    .points()
                    .iter()
                    .map(|point| point.weight * point.coordinates[0].powi(degree as i32))
                    .sum::<f64>();
                let expected = if degree % 2 == 0 {
                    2.0 / (degree + 1) as f64
                } else {
                    0.0
                };
                // Preserve the existing low-order analytic precision checks.
                let tolerance = if (4..=7).contains(&n) {
                    (n - 2) as f64 * 1e-15
                } else {
                    64.0 * n as f64 * f64::EPSILON
                };
                assert!(
                    (actual - expected).abs() < tolerance,
                    "n={n}, degree={degree}: {actual} != {expected}"
                );
            }
            assert!(
                rule.points()
                    .iter()
                    .all(|point| point.weight > 0.0 && point.coordinates[0].abs() < 1.0)
            );
        }
        assert!(QuadratureRule::gauss_legendre(0).is_err());
        assert!(QuadratureRule::gauss_legendre(usize::MAX).is_err());
        // Independent analytic four-point root and weight (not the recurrence).
        let analytic = QuadratureRule::gauss_legendre(4).unwrap();
        let first = &analytic.points()[0];
        let root = -((3.0 + 2.0 * (6.0_f64 / 5.0).sqrt()) / 7.0).sqrt();
        let weight = (18.0 - 30.0_f64.sqrt()) / 36.0;
        assert!((first.coordinates[0] - root).abs() < 2e-15);
        assert!((first.weight - weight).abs() < 2e-15);
    }

    #[test]
    fn tensor_product_rule_has_arbitrary_runtime_dimension() {
        let rule = QuadratureRule::tensor_product_gauss_legendre(5, 2).unwrap();
        assert_eq!(rule.reference_cell().dimension(), 5);
        assert_eq!(rule.points().len(), 32);
        assert!(
            rule.points()
                .iter()
                .all(|point| point.coordinates.len() == 5)
        );
        assert!(
            (rule.points().iter().map(|point| point.weight).sum::<f64>() - 32.0).abs() < 1.0e-13
        );
    }

    #[test]
    fn custom_rule_checks_reference_cell_and_coordinates() {
        assert_eq!(
            QuadratureRule::new(
                ReferenceCell::point(),
                None,
                vec![QuadraturePoint {
                    coordinates: vec![0.0],
                    weight: 1.0,
                }],
            )
            .unwrap_err()
            .code(),
            codes::INVALID_QUADRATURE
        );
    }
}
