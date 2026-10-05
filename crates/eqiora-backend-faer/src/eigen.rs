use eqiora_core::{Diagnostic, ScalarDomain, ValueLiteral};
use eqiora_solver::HermitianEigenproblem;
use faer::{
    Mat, Side,
    traits::{
        ComplexField,
        math_utils::{imag, is_finite, real},
    },
};

/// Compute the complete finite spectrum of an admitted Hermitian pencil.
///
/// Uses Cholesky reduction followed by dense self-adjoint decomposition, for
/// real or complex binary64 coefficients. Returned typed eigenpairs are
/// numerical candidates, sorted by eigenvalue. The owning Plan must apply its
/// residual and normalization tolerances, selection and convergence policy;
/// this function does not certify a Result or select a unique phase/basis.
pub(super) fn hermitian_eigenpairs(
    problem: &HermitianEigenproblem<'_>,
) -> Result<Vec<(ValueLiteral, ValueLiteral)>, Diagnostic> {
    if problem.mode_type().scalar_domain() == ScalarDomain::Real {
        decompose::<f64>(problem, |(real, _)| real)
    } else {
        decompose::<faer::c64>(problem, |(real, imaginary)| faer::c64::new(real, imaginary))
    }
}

fn decompose<T: ComplexField<Real = f64>>(
    problem: &HermitianEigenproblem<'_>,
    convert: impl Fn((f64, f64)) -> T,
) -> Result<Vec<(ValueLiteral, ValueLiteral)>, Diagnostic> {
    let n = problem.dimension();
    let matrix = |literal: &ValueLiteral| {
        Mat::<T>::from_fn(n, n, |i, j| {
            convert(literal.component(i * n + j).expect("admitted matrix"))
        })
    };
    let metric = matrix(problem.metric());
    let factor = metric
        .llt(Side::Lower)
        .map_err(|error| super::solve_failed(format!("faer metric Cholesky failed: {error:?}")))?;
    // B=L Lᴴ, C=L^-1 A L^-H, u=L^-H q. Right division is obtained
    // through the adjoint, avoiding an explicit inverse or doubled-real model.
    let mut left = matrix(problem.operator());
    factor.L().solve_lower_triangular_in_place(left.as_mut());
    let mut reduced = left.adjoint().to_owned();
    factor.L().solve_lower_triangular_in_place(reduced.as_mut());
    if (0..n).any(|i| (0..n).any(|j| !is_finite(&reduced[(i, j)]))) {
        return Err(super::solve_failed(
            "Hermitian reduction produced nonfinite coefficients",
        ));
    }
    let decomposition = reduced.self_adjoint_eigen(Side::Lower).map_err(|error| {
        super::solve_failed(format!(
            "faer Hermitian eigendecomposition failed: {error:?}"
        ))
    })?;
    let mut modes = decomposition.U().to_owned();
    factor
        .L()
        .adjoint()
        .solve_upper_triangular_in_place(modes.as_mut());
    (0..n)
        .map(|column| {
            let eigenvalue = ValueLiteral::from_real(
                problem.eigenvalue_type().clone(),
                real(&decomposition.S()[column]),
            )
            .map_err(|error| super::solve_failed(format!("invalid eigenvalue: {error}")))?;
            let mode = ValueLiteral::new(
                problem.mode_type().clone(),
                (0..n).map(|row| (real(&modes[(row, column)]), imag(&modes[(row, column)]))),
            )
            .map_err(|error| super::solve_failed(format!("invalid eigenmode: {error}")))?;
            Ok((eigenvalue, mode))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::{DimExponents, FiniteBasis, Id, ValueType};
    use eqiora_solver::LinearSolverBackend;

    #[test]
    fn hermitian_eigenpairs_preserve_repeated_subspaces_with_complex_metric() {
        // B=L Lᴴ and A=L diag(1,3) Lᴴ with L=[[1,0],[i,2]].
        // Three copies exercise repeated eigenspaces and a nonreal, nondiagonal
        // metric. The invariant is B-orthogonality, never particular phases.
        let basis = FiniteBasis::new(Id::new(), 6).unwrap();
        let ty = ValueType::linear_map(
            basis,
            basis,
            ScalarDomain::Complex,
            DimExponents::DIMENSIONLESS,
        )
        .unwrap();
        let mut a = vec![(0., 0.); 36];
        let mut b = a.clone();
        for block in 0..3 {
            let i = 2 * block;
            for entries in [&mut a, &mut b] {
                entries[i * 6 + i] = (1., 0.);
                entries[i * 6 + i + 1] = (0., -1.);
                entries[(i + 1) * 6 + i] = (0., 1.);
            }
            a[(i + 1) * 6 + i + 1] = (13., 0.);
            b[(i + 1) * 6 + i + 1] = (5., 0.);
        }
        let a = ValueLiteral::new(ty.clone(), a).unwrap();
        let b = ValueLiteral::new(ty, b).unwrap();
        let problem = HermitianEigenproblem::new(&a, &b).unwrap();
        let pairs = crate::FaerLinearSolver
            .hermitian_eigenpairs(&problem)
            .unwrap();
        let complex = |(real, imaginary)| faer::c64::new(real, imaginary);
        assert_eq!(pairs.len(), 6);
        for (i, ((value, mode), expected)) in pairs.iter().zip([1., 1., 1., 3., 3., 3.]).enumerate()
        {
            assert!((value.component(0).unwrap().0 - expected).abs() < 1.0e-12);
            let (residual, normalization) = problem.eigenpair_defects(value, mode).unwrap();
            assert!(residual < 1.0e-12 && normalization < 1.0e-12);
            for (j, (_, other)) in pairs.iter().enumerate() {
                let mut product = faer::c64::new(0., 0.);
                for row in 0..6 {
                    for col in 0..6 {
                        product += complex(mode.component(row).unwrap()).conj()
                            * complex(b.component(row * 6 + col).unwrap())
                            * complex(other.component(col).unwrap());
                    }
                }
                assert!((product - if i == j { 1. } else { 0. }).norm() < 1.0e-12);
            }
        }
    }

    #[test]
    fn hermitian_eigenpairs_solve_real_and_complex_metric_pencils() {
        // C has eigenvalues 1 and 3; diagonal metric entries are 2 and 8.
        // The complex version changes off-diagonal 4 into 4i and -4i.
        let basis = FiniteBasis::new(Id::new(), 2).unwrap();
        for domain in [ScalarDomain::Real, ScalarDomain::Complex] {
            let ty =
                ValueType::linear_map(basis, basis, domain, DimExponents::DIMENSIONLESS).unwrap();
            let off = if domain == ScalarDomain::Real {
                (4., 0.)
            } else {
                (0., 4.)
            };
            let a =
                ValueLiteral::new(ty.clone(), [(4., 0.), off, (off.0, -off.1), (16., 0.)]).unwrap();
            let b = ValueLiteral::new(ty, [(2., 0.), (0., 0.), (0., 0.), (8., 0.)]).unwrap();
            let problem = HermitianEigenproblem::new(&a, &b).unwrap();
            let backend: &dyn LinearSolverBackend = &crate::FaerLinearSolver;
            backend.require_hermitian_eigenproblem(&problem).unwrap();
            let pairs = backend.hermitian_eigenpairs(&problem).unwrap();
            let unsupported: &dyn LinearSolverBackend = &eqiora_solver::REFERENCE_LINEAR_SOLVER;
            assert!(
                unsupported
                    .require_hermitian_eigenproblem(&problem)
                    .is_err()
            );
            assert!(unsupported.hermitian_eigenpairs(&problem).is_err());
            assert_eq!(pairs.len(), 2);
            for ((value, mode), expected) in pairs.iter().zip([1., 3.]) {
                assert!((value.component(0).unwrap().0 - expected).abs() < 1.0e-12);
                let (residual, normalization) = problem.eigenpair_defects(value, mode).unwrap();
                assert!(residual < 1.0e-12, "residual {residual}");
                assert!(normalization < 1.0e-12, "normalization {normalization}");
            }
        }
    }
}
