//! Typed congruence retains an explicitly authored coordinate embedding.
use super::*;
use eqiora_core::ScalarDomain;

impl HermitianEigenproblem<'_> {
    /// Pull back complete Hermitian matrices along an explicit map `u=Pq`.
    ///
    /// Returns `(Pᴴ A P, Pᴴ B P)` in P's exact source basis. The restricted
    /// metric must have positive finite Cholesky pivots; the original metric
    /// may be singular or indefinite only outside this admitted space. No
    /// coordinates are chosen, deleted, shifted or regularized implicitly.
    /// This does not prove that reduced eigenvectors satisfy the original
    /// equation: lifted candidates still require original-pencil verification.
    pub fn pullback(
        operator: &ValueLiteral,
        metric: &ValueLiteral,
        embedding: &ValueLiteral,
    ) -> Result<(ValueLiteral, ValueLiteral), Diagnostic> {
        let original = HermitianEigenproblem::pencil(operator, metric)?;
        let (source, target) = embedding
            .value_type()
            .map_bases()
            .ok_or_else(|| invalid("spectral embedding must be a typed finite linear map"))?;
        if original.mode_type().coordinate_basis() != Some(target)
            || source.extent() > target.extent()
            || (embedding.value_type().scalar_domain() == ScalarDomain::Complex
                && original.mode_type().scalar_domain() == ScalarDomain::Real)
        {
            return Err(invalid(
                "spectral embedding has incompatible bases or scalar domain",
            ));
        }
        let n = original.dimension();
        let k = source.extent() as usize;
        let pull = |matrix: &ValueLiteral| -> Result<ValueLiteral, Diagnostic> {
            let dimension = matrix
                .value_type()
                .dimension()
                .mul(embedding.value_type().dimension())
                .and_then(|value| value.mul(embedding.value_type().dimension()))
                .ok_or_else(|| invalid("spectral pullback dimension overflowed"))?;
            let ty = ValueType::linear_map(
                source,
                source,
                original.mode_type().scalar_domain(),
                dimension,
            )
            .map_err(|e| invalid(e.to_string()))?;
            let mut entries = vec![(0., 0.); k * k];
            let mut action = vec![Complex64::new(0., 0.); n * k];
            for row in 0..n {
                for column in 0..k {
                    for inner in 0..n {
                        action[row * k + column] += coefficient(matrix, row * n + inner)
                            * coefficient(embedding, inner * k + column);
                    }
                }
            }
            // Both input triangles were validated above. Form one triangle of
            // the congruence and its conjugate, preserving its mathematical
            // Hermitian identity despite floating-point summation order.
            for row in 0..k {
                for column in 0..=row {
                    let mut value = Complex64::new(0., 0.);
                    for i in 0..n {
                        value +=
                            coefficient(embedding, i * k + row).conj() * action[i * k + column];
                    }
                    if !value.re.is_finite() || !value.im.is_finite() {
                        return Err(invalid("spectral pullback produced nonfinite arithmetic"));
                    }
                    if row == column {
                        value.im = 0.;
                    }
                    entries[row * k + column] = (value.re, value.im);
                    entries[column * k + row] = (value.re, -value.im);
                }
            }
            ValueLiteral::new(ty, entries).map_err(|e| invalid(e.to_string()))
        };
        let a = pull(operator)?;
        let b = pull(metric)?;
        HermitianEigenproblem::new(&a, &b)?;
        Ok((a, b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::{DimExponents, FiniteBasis, Id};

    #[test]
    fn singular_original_metric_is_admitted_only_through_positive_restriction() {
        let full = FiniteBasis::new(Id::new(), 2).unwrap();
        let reduced = FiniteBasis::new(Id::new(), 1).unwrap();
        let matrix = |entries| {
            ValueLiteral::new(
                ValueType::linear_map(full, full, ScalarDomain::Real, DimExponents::DIMENSIONLESS)
                    .unwrap(),
                entries,
            )
            .unwrap()
        };
        let a = matrix([(1., 0.), (-1., 0.), (-1., 0.), (1., 0.)]);
        let p = ValueLiteral::new(
            ValueType::linear_map(
                reduced,
                full,
                ScalarDomain::Real,
                DimExponents::DIMENSIONLESS,
            )
            .unwrap(),
            [(1., 0.), (-1., 0.)],
        )
        .unwrap();
        assert!(HermitianEigenproblem::new(&a, &a).is_err());
        let (ar, br) = HermitianEigenproblem::pullback(&a, &a, &p).unwrap();
        assert_eq!(ar.component(0), Some((4., 0.)));
        assert_eq!(br, ar);
        assert_eq!(ar.value_type().map_bases(), Some((reduced, reduced)));
        let (lambda_type, mode_type) = a
            .value_type()
            .hermitian_eigenpair_types(a.value_type())
            .unwrap();
        let lambda = ValueLiteral::from_real(lambda_type, 1.).unwrap();
        let mode = ValueLiteral::new(mode_type, [(0.5, 0.), (-0.5, 0.)]).unwrap();
        assert_eq!(
            HermitianEigenproblem::original_eigenpair_defects(&a, &a, &lambda, &mode).unwrap(),
            (0., 0.)
        );
        let projector =
            HermitianEigenproblem::original_metric_projector(&a, &a, &[mode], 1e-12).unwrap();
        assert_eq!(
            projector.components().unwrap().collect::<Vec<_>>(),
            [(0.5, 0.), (-0.5, 0.), (-0.5, 0.), (0.5, 0.)]
        );
        let null = ValueLiteral::new(p.value_type().clone(), [(1., 0.), (1., 0.)]).unwrap();
        assert!(HermitianEigenproblem::pullback(&a, &a, &null).is_err());
        let false_hermitian = matrix([(1., 0.), (7., 0.), (-1., 0.), (1., 0.)]);
        assert!(HermitianEigenproblem::pullback(&false_hermitian, &a, &p).is_err());
    }

    #[test]
    fn complex_congruence_keeps_five_admitted_coordinates_and_physical_units() {
        let full = FiniteBasis::new(Id::new(), 6).unwrap();
        let reduced = FiniteBasis::new(Id::new(), 5).unwrap();
        let mass = DimExponents::from_integers([1, 0, 0, 0, 0, 0, 0]).unwrap();
        let ty = ValueType::linear_map(full, full, ScalarDomain::Complex, mass).unwrap();
        let b = ValueLiteral::new(
            ty,
            (0..6)
                .flat_map(|i| (0..6).map(move |j| (if i == j { (i * i) as f64 } else { 0. }, 0.))),
        )
        .unwrap();
        // P=i[e2,...,e6]. Conjugation in Pᴴ is essential: PᵀBP is negative.
        let p = ValueLiteral::new(
            ValueType::linear_map(
                reduced,
                full,
                ScalarDomain::Complex,
                DimExponents::DIMENSIONLESS,
            )
            .unwrap(),
            (0..6).flat_map(|i| (0..5).map(move |j| (0., if i == j + 1 { 1. } else { 0. }))),
        )
        .unwrap();
        let (ar, br) = HermitianEigenproblem::pullback(&b, &b, &p).unwrap();
        assert_eq!(ar, br);
        assert_eq!(br.value_type().dimension(), mass);
        assert_eq!(br.value_type().map_bases(), Some((reduced, reduced)));
        for i in 0..5 {
            for j in 0..5 {
                assert_eq!(
                    br.component(i * 5 + j),
                    Some((
                        if i == j {
                            ((i + 1) * (i + 1)) as f64
                        } else {
                            0.
                        },
                        0.
                    ))
                );
            }
        }
    }

    #[test]
    fn original_verification_exposes_a_false_projected_eigenpair() {
        let full = FiniteBasis::new(Id::new(), 2).unwrap();
        let reduced = FiniteBasis::new(Id::new(), 1).unwrap();
        let ty = ValueType::linear_map(full, full, ScalarDomain::Real, DimExponents::DIMENSIONLESS)
            .unwrap();
        let a = ValueLiteral::new(ty.clone(), [(2., 0.), (1., 0.), (1., 0.), (2., 0.)]).unwrap();
        let b = ValueLiteral::new(ty, [(1., 0.), (0., 0.), (0., 0.), (1., 0.)]).unwrap();
        let p = ValueLiteral::new(
            ValueType::linear_map(
                reduced,
                full,
                ScalarDomain::Real,
                DimExponents::DIMENSIONLESS,
            )
            .unwrap(),
            [(1., 0.), (0., 0.)],
        )
        .unwrap();
        let (ar, br) = HermitianEigenproblem::pullback(&a, &b, &p).unwrap();
        let restricted = HermitianEigenproblem::new(&ar, &br).unwrap();
        let lambda = ValueLiteral::from_real(restricted.eigenvalue_type().clone(), 2.).unwrap();
        let q = ValueLiteral::new(restricted.mode_type().clone(), [(1., 0.)]).unwrap();
        assert_eq!(restricted.eigenpair_defects(&lambda, &q).unwrap(), (0., 0.));
        let (_, mode_type) = a
            .value_type()
            .hermitian_eigenpair_types(b.value_type())
            .unwrap();
        let u = ValueLiteral::new(mode_type, [(1., 0.), (0., 0.)]).unwrap();
        let (residual, normalization) =
            HermitianEigenproblem::original_eigenpair_defects(&a, &b, &lambda, &u).unwrap();
        // ||(0,1)|| / (||(2,1)|| + 2 ||(1,0)||).
        assert!((residual - 1. / (5_f64.sqrt() + 2.)).abs() < 1e-14);
        assert_eq!(normalization, 0.);
    }
}
