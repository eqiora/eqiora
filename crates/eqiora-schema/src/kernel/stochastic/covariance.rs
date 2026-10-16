//! Exact covariance admission; no sampler, eigenvalue tolerance or nearest-PSD repair.
use super::invalid;
use crate::kernel::pure_operator::ExactRational;
use eqiora_core::{Diagnostic, DimExponents, ScalarDomain, ValueFrame, ValueLiteral};

pub(super) fn validate(value: &ValueLiteral) -> Result<usize, Diagnostic> {
    decompose(value, |_, _, _, _| Ok(()))
}

/// Retain the same exact Schur decomposition used for PSD admission. Each
/// column of L sqrt(D) is rounded once, with no pivot tolerance or PSD repair.
pub(super) fn factor(value: &ValueLiteral) -> Result<Vec<f64>, Diagnostic> {
    let mut factor = Vec::new();
    decompose(value, |n, k, pivot, matrix| {
        if factor.is_empty() {
            factor.resize(n * n, 0.);
        }
        if pivot.is_zero() {
            return Ok(());
        }
        for i in k..n {
            let entry = &matrix[i * n + k];
            let magnitude = entry
                .checked_mul(entry.clone())
                .and_then(|square| square.checked_div(pivot.clone()))
                .and_then(|square| square.sqrt_to_f64())
                .map_err(|_| {
                    invalid("Wiener covariance factor exceeds the exact proof or binary64 range")
                })?;
            factor[i * n + k] = if entry < &ExactRational::integer(0) {
                -magnitude
            } else {
                magnitude
            };
        }
        Ok(())
    })?;
    Ok(factor)
}

fn decompose(
    value: &ValueLiteral,
    mut retain: impl FnMut(usize, usize, &ExactRational, &[ExactRational]) -> Result<(), Diagnostic>,
) -> Result<usize, Diagnostic> {
    let ty = value.value_type();
    if ty.scalar_domain() != ScalarDomain::Real
        || ty.dimension() != DimExponents::DIMENSIONLESS
        || ty.frame() != ValueFrame::Invariant
        || ty.finite_bases().next().is_some()
    {
        return Err(invalid(
            "Wiener covariance requires finite real dimensionless channel values",
        ));
    }
    let n = match ty.shape().extents() {
        [] => 1,
        [rows, columns] if rows == columns && ty.array_rank() == 2 => rows.get() as usize,
        _ => {
            return Err(invalid(
                "Wiener covariance requires a scalar or a square channel array",
            ));
        }
    };
    // No stochastic evaluation occurs here. The exact binary input is the
    // covariance; even one negative subnormal eigenvalue must be rejected.
    let mut matrix = value
        .components()
        .expect("real value")
        .map(|(re, _)| ExactRational::from_binary64(re).expect("finite ValueLiteral"))
        .collect::<Vec<_>>();
    for i in 0..n {
        for j in 0..i {
            if matrix[i * n + j] != matrix[j * n + i] {
                return Err(invalid("Wiener covariance must be exactly symmetric"));
            }
        }
    }
    let zero = ExactRational::integer(0);
    for k in 0..n {
        let pivot = matrix[k * n + k].clone();
        if pivot < zero {
            return Err(invalid("Wiener covariance must be positive semidefinite"));
        }
        retain(n, k, &pivot, &matrix)?;
        if pivot.is_zero() {
            // A PSD matrix with zero diagonal has a zero row and column;
            // otherwise its corresponding 2-by-2 principal minor is negative.
            if (k + 1..n).any(|j| !matrix[k * n + j].is_zero()) {
                return Err(invalid(
                    "Wiener covariance has a nonzero coupling at a zero pivot",
                ));
            }
            continue;
        }
        for i in k + 1..n {
            for j in i..n {
                let reduction = matrix[i * n + k]
                    .checked_mul(matrix[k * n + j].clone())
                    .and_then(|product| product.checked_div(pivot.clone()))
                    .and_then(|quotient| quotient.checked_neg());
                let updated = reduction
                    .and_then(|reduction| matrix[i * n + j].checked_add(reduction))
                    .map_err(|_| {
                        invalid("Wiener covariance exceeds the exact arithmetic proof budget")
                    })?;
                matrix[i * n + j] = updated.clone();
                matrix[j * n + i] = updated;
            }
        }
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::{ValueShape, ValueType};

    fn covariance(n: u32, values: &[f64]) -> ValueLiteral {
        let ty = ValueType::shaped(
            ScalarDomain::Real,
            DimExponents::DIMENSIONLESS,
            ValueShape::new([n, n]).unwrap(),
            ValueFrame::Invariant,
        )
        .unwrap();
        ValueLiteral::new(ty, values.iter().map(|&value| (value, 0.))).unwrap()
    }

    #[test]
    fn exact_psd_accepts_degenerate_and_correlated_channels() {
        // The last covariance is B B^T for rows (1,0), (0,1), (1,1).
        for (n, values) in [
            (2, vec![0., 0., 0., 1.]),
            (2, vec![1., 1., 1., 1.]),
            (2, vec![1., -0.5, -0.5, 1.]),
            (3, vec![1., 0., 1., 0., 1., 1., 1., 1., 2.]),
        ] {
            assert_eq!(validate(&covariance(n, &values)).unwrap(), n as usize);
        }
    }

    #[test]
    fn exact_psd_rejects_asymmetry_and_negative_principal_minors() {
        for values in [
            [1., 0., 0.1, 1.],
            [1., 2., 2., 1.],
            [0., 1., 1., 0.],
            [1., 0., 0., -f64::from_bits(1)],
        ] {
            assert!(validate(&covariance(2, &values)).is_err());
        }
        // A one-ULP increase above unit correlation has determinant 1-rho^2<0.
        let rho = f64::from_bits(1f64.to_bits() + 1);
        assert!(validate(&covariance(2, &[1., rho, rho, 1.])).is_err());
    }

    #[test]
    fn numerical_factors_retain_exact_degeneracy_and_signed_coupling() {
        for (matrix, expected) in [
            ([0., 0., 0., 0.], [0., 0., 0., 0.]),
            ([0., 0., 0., 1.], [0., 0., 0., 1.]),
            ([1., 1., 1., 1.], [1., 0., 1., 0.]),
            ([1., -0.5, -0.5, 0.5], [1., 0., -0.5, 0.5]),
        ] {
            assert_eq!(factor(&covariance(2, &matrix)).unwrap(), expected);
        }
        // Independently specified rows of B: (1,0,0), (0,1,0), (1,1,0).
        assert_eq!(
            factor(&covariance(3, &[1., 0., 1., 0., 1., 1., 1., 1., 2.])).unwrap(),
            [1., 0., 0., 0., 1., 0., 1., 1., 0.]
        );
        let subnormal = f64::from_bits(1);
        let extreme = covariance(2, &[subnormal, 0., 0., f64::MAX]);
        let result = factor(&extreme).unwrap();
        assert_eq!(result, [subnormal.sqrt(), 0., 0., f64::MAX.sqrt()]);
        // This exact matrix is positive definite (MAX - subnormal^2 > 0),
        // but its factor entry subnormal/sqrt(MAX) cannot be stored in f64.
        // A numerical realization must reject without narrowing Model meaning.
        let unrepresentable = covariance(2, &[f64::MAX, subnormal, subnormal, 1.]);
        assert_eq!(validate(&unrepresentable).unwrap(), 2);
        assert!(factor(&unrepresentable).is_err());
        let rho = f64::from_bits(1f64.to_bits() + 1);
        assert!(factor(&covariance(2, &[1., rho, rho, 1.])).is_err());
    }
}
