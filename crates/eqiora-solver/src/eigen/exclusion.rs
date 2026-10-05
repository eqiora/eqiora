//! A declared embedding determines an excluded space, not a gauge assertion.
use super::*;
use eqiora_core::DimExponents;

impl HermitianEigenproblem<'_> {
    /// Dimensionless complementary metric projector
    /// `E = I - P (Pᴴ B P)⁻¹ Pᴴ B` of an explicitly declared embedding.
    ///
    /// The full original metric may be singular or indefinite; its pullback
    /// must be positive definite. E describes directions excluded by P and
    /// has rank `full_extent - admitted_extent`. It is generally an oblique
    /// projector in coefficient coordinates. This does not assert that A or B
    /// vanishes on those directions, or that they are physical gauge freedoms.
    /// Cholesky triangular solves avoid explicitly forming an inverse.
    /// `tolerance` checks the B-orthonormality of the lifted coordinate basis.
    pub fn excluded_metric_projector(
        operator: &ValueLiteral,
        metric: &ValueLiteral,
        embedding: &ValueLiteral,
        tolerance: f64,
    ) -> Result<ValueLiteral, Diagnostic> {
        let original = HermitianEigenproblem::pencil(operator, metric)?;
        let (_, restricted_metric) = Self::pullback(operator, metric, embedding)?;
        let (source, target) = embedding.value_type().map_bases().expect("checked map");
        let (n, k) = (target.extent() as usize, source.extent() as usize);
        let lower = metric::cholesky(&restricted_metric, k)?;
        let mut normalized = vec![Complex64::new(0., 0.); n * k];
        // U Lᴴ=P, hence each physical row requires a forward triangular solve.
        // L has dimension sqrt(PᴴBP), so U has the original mode dimension.
        for row in 0..n {
            for column in 0..k {
                let mut value = coefficient(embedding, row * k + column);
                for inner in 0..column {
                    value -= normalized[row * k + inner] * lower[column * k + inner].conj();
                }
                normalized[row * k + column] = value / lower[column * k + column].re;
            }
        }
        let modes = (0..k)
            .map(|column| {
                ValueLiteral::new(
                    original.mode_type().clone(),
                    (0..n).map(|row| {
                        let z = normalized[row * k + column];
                        (z.re, z.im)
                    }),
                )
                .map_err(|e| invalid(e.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let included = original.metric_projector(&modes, tolerance)?;
        let ty = ValueType::linear_map(
            target,
            target,
            original.mode_type().scalar_domain(),
            DimExponents::DIMENSIONLESS,
        )
        .map_err(|e| invalid(e.to_string()))?;
        ValueLiteral::new(
            ty,
            (0..n * n).map(|index| {
                let z = coefficient(&included, index);
                (f64::from(index / n == index % n) - z.re, -z.im)
            }),
        )
        .map_err(|e| invalid(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::{FiniteBasis, Id, ScalarDomain};

    #[test]
    fn excluded_projector_retains_metric_geometry_without_asserting_nullity() {
        let full = FiniteBasis::new(Id::new(), 2).unwrap();
        let reduced = FiniteBasis::new(Id::new(), 1).unwrap();
        let matrix_type =
            ValueType::linear_map(full, full, ScalarDomain::Real, DimExponents::DIMENSIONLESS)
                .unwrap();
        let embedding_type = ValueType::linear_map(
            reduced,
            full,
            ScalarDomain::Real,
            DimExponents::DIMENSIONLESS,
        )
        .unwrap();
        // Common-kernel exclusion, ordinary zero-eigenvalue exclusion, an
        // oblique metric complement, and an excluded indefinite direction.
        for (a, b, p, expected) in [
            (
                [1., -1., -1., 1.],
                [1., -1., -1., 1.],
                [1., -1.],
                [0.5, 0.5, 0.5, 0.5],
            ),
            (
                [1., -1., -1., 1.],
                [1., 0., 0., 1.],
                [1., -1.],
                [0.5, 0.5, 0.5, 0.5],
            ),
            (
                [1., 0., 0., 4.],
                [1., 0., 0., 4.],
                [1., 1.],
                [0.8, -0.8, -0.2, 0.2],
            ),
            (
                [-1., 0., 0., 4.],
                [-1., 0., 0., 4.],
                [0., 1.],
                [1., 0., 0., 0.],
            ),
        ] {
            let matrix =
                |v: [f64; 4]| ValueLiteral::new(matrix_type.clone(), v.map(|x| (x, 0.))).unwrap();
            let embedding = ValueLiteral::new(embedding_type.clone(), p.map(|x| (x, 0.))).unwrap();
            let e = HermitianEigenproblem::excluded_metric_projector(
                &matrix(a),
                &matrix(b),
                &embedding,
                1e-12,
            )
            .unwrap();
            assert_eq!(e.value_type(), &matrix_type);
            for (i, x) in expected.into_iter().enumerate() {
                assert!((e.component(i).unwrap().0 - x).abs() < 1e-12);
                assert_eq!(e.component(i).unwrap().1, 0.);
            }
        }
    }

    #[test]
    fn complex_scaled_coordinates_preserve_the_same_excluded_space() {
        let full = FiniteBasis::new(Id::new(), 6).unwrap();
        let reduced = FiniteBasis::new(Id::new(), 5).unwrap();
        let ty = ValueType::linear_map(
            full,
            full,
            ScalarDomain::Complex,
            DimExponents::DIMENSIONLESS,
        )
        .unwrap();
        let metric = ValueLiteral::new(
            ty,
            (0..36).map(|i| (if i / 6 == i % 6 { (i / 6) as f64 } else { 0. }, 0.)),
        )
        .unwrap();
        let ty = ValueType::linear_map(
            reduced,
            full,
            ScalarDomain::Complex,
            DimExponents::DIMENSIONLESS,
        )
        .unwrap();
        for scale in [1., 1e-100, 1e100] {
            for mixed in [false, true] {
                // P=i*scale*[e2..e6]*T, where T is identity or has i on its
                // superdiagonal. Both T are invertible. The excluded space is
                // span(e1), independent of scale, phase and coordinate mixing.
                // The mixed case has non-real off-diagonal Cholesky entries.
                let p = ValueLiteral::new(
                    ty.clone(),
                    (0..30).map(|i| {
                        let (row, column) = (i / 5, i % 5);
                        if row == column + 1 {
                            (0., scale)
                        } else if mixed && row > 0 && row == column {
                            (-scale, 0.)
                        } else {
                            (0., 0.)
                        }
                    }),
                )
                .unwrap();
                let e =
                    HermitianEigenproblem::excluded_metric_projector(&metric, &metric, &p, 1e-12)
                        .unwrap();
                for i in 0..36 {
                    let (re, im) = e.component(i).unwrap();
                    assert!((re - f64::from(i == 0)).abs() < 1e-12);
                    assert!(im.abs() < 1e-12);
                }
            }
        }
    }
}
