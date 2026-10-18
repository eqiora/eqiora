//! Affine flux, mixed traces and their derivatives share the same face closure.
use super::*;

#[test]
fn mixed_trace_affine_flux_and_direction_match_independent_schur_elimination() {
    let mesh = CartesianMesh::from_axes(vec![vec![0., 2.]]).unwrap();
    let coefficient = |i: usize, j: usize, _, _: &[f64]| Ok([[2., 1.], [3., 4.]][i][j]);
    let prescribed = |i: usize, _, x: &[f64]| Ok([5., 7.][i] * x[0]);
    let boundary = |i: usize, _, side, _: &[f64]| {
        Ok(match (side, i) {
            (BoundarySide::Lower, _) => CartesianBoundaryValue::Essential([1., 3.][i]),
            (_, 0) => CartesianBoundaryValue::Essential(5.),
            _ => CartesianBoundaryValue::Natural(16.),
        })
    };
    let coefficient_partial = |i: usize, j: usize, _, _: &[f64]| Ok([[1., -2.], [2., 1.]][i][j]);
    let prescribed_partial = |i: usize, _, x: &[f64]| Ok([1., -3.][i] * x[0]);
    let boundary_partial = |i: usize, _, side, _: &[f64]| {
        Ok(match (side, i) {
            (BoundarySide::Lower, _) => CartesianBoundaryValue::Essential([1., -2.][i]),
            (_, 0) => CartesianBoundaryValue::Essential(3.),
            _ => CartesianBoundaryValue::Natural(4.),
        })
    };
    for partial in [false, true] {
        let mut flux = FvmFlux::new(&coefficient, &prescribed, &boundary);
        if partial {
            flux =
                flux.with_direction(&coefficient_partial, &prescribed_partial, &boundary_partial);
        }
        let assembly = finalize_elliptic_cartesian_fvm(
            &mesh,
            2,
            Some(flux),
            &|_, _, _| Ok(0.),
            &|i: usize, _| Ok(if partial { [-1., 3.][i] } else { [-5., -7.][i] }),
            &QuadratureRule::gauss_legendre(1).unwrap(),
            &QuadratureRule::point(),
            &REFERENCE_ASSEMBLY_BACKEND,
            LinearOperatorProperties::General,
        )
        .unwrap();
        // u=(1+2x,3-x), du=(1+x,-2+x), g=(5x,7x), dg=(x,-3x).
        // At the right face P=T_DN/T_NN=1/4 and dP=-9/16.
        // Its effective coefficient is 5/4, derivative 35/16; its load
        // is 67/4, derivative 289/16 after restoring the prescribed flux.
        let (matrix, rhs) = if partial {
            ([[51. / 16., -2.], [2., 1.]], [177. / 16., 10.])
        } else {
            ([[13. / 4., 1.], [3., 4.]], [47. / 4., 17.])
        };
        for (i, row) in matrix.iter().enumerate() {
            for (j, expected) in row.iter().enumerate() {
                assert!(
                    (assembly.linear_system.matrix().entry(i, j).unwrap() - expected).abs() < 1e-12
                );
            }
            assert!((assembly.linear_system.rhs()[i] - rhs[i]).abs() < 1e-12);
        }
    }
}

#[test]
fn prescribed_flux_preserves_natural_reconstruction_and_total_balance() {
    let mesh = CartesianMesh::from_axes(vec![vec![0., 1., 2.]]).unwrap();
    let assembly = finalize_elliptic_cartesian_fvm(
        &mesh,
        1,
        Some(FvmFlux::new(
            &|_, _, _, _| Ok(1.),
            &|_, _, x| Ok(x[0]),
            &|_, _, side, _| {
                Ok(match side {
                    BoundarySide::Lower => CartesianBoundaryValue::Essential(1.),
                    BoundarySide::Upper => CartesianBoundaryValue::Natural(4.),
                })
            },
        )),
        &|_, _, _| Ok(0.),
        &|_, _| Ok(-1.),
        &QuadratureRule::gauss_legendre(1).unwrap(),
        &QuadratureRule::point(),
        &REFERENCE_ASSEMBLY_BACKEND,
        LinearOperatorProperties::SymmetricPositiveDefinite,
    )
    .unwrap();
    let (system, state) = assembly.into_canonical(None).unwrap();
    let solved = state
        .solve(
            LinearSolveRequest::new(
                &REFERENCE_LINEAR_SOLVER,
                SolverPlan::new(
                    LinearSolver::ConjugateGradient,
                    1e-13,
                    1e-13,
                    NonZeroUsize::new(32).unwrap(),
                )
                .unwrap(),
            ),
            system,
        )
        .unwrap();
    // u=1+2x, q=u'+x, -q'=-1. Right trace is 5, not u_cell+q/(K/d).
    for (actual, expected) in solved.cell_values().iter().zip([2., 4.]) {
        assert!((actual - expected).abs() < 1e-12);
    }
    assert!(
        solved
            .reconstruction()
            .l2_error(
                &|x| 1. + 2. * x[0],
                &QuadratureRule::gauss_legendre(2).unwrap()
            )
            .unwrap()
            < 1e-12
    );
    assert!((solved.integrated_source() + solved.boundary_flux_sum()).abs() < 1e-12);
    assert!((solved.integrated_source() + 2.).abs() < 1e-12);
}

#[test]
fn axis_coefficients_and_prescribed_flux_keep_nonuniform_face_measures() {
    let mesh = CartesianMesh::from_axes(vec![vec![0., 1., 3.], vec![0., 2., 5.]]).unwrap();
    let exact = |x: &[f64]| 1. + 2. * x[0] - 3. * x[1];
    let assembly = finalize_elliptic_cartesian_fvm(
        &mesh,
        1,
        Some(FvmFlux::new(
            &|_, _, axis: usize, _| Ok([2., 5.][axis]),
            &|_, axis: usize, x| Ok([7., 11.][axis] * x[axis]),
            &|_, _, _, x| Ok(CartesianBoundaryValue::Essential(exact(x))),
        )),
        &|_, _, _| Ok(0.),
        &|_, _| Ok(-18.),
        &QuadratureRule::tensor_product_gauss_legendre(2, 1).unwrap(),
        &QuadratureRule::gauss_legendre(2).unwrap(),
        &REFERENCE_ASSEMBLY_BACKEND,
        LinearOperatorProperties::SymmetricPositiveDefinite,
    )
    .unwrap();
    let (system, state) = assembly.into_canonical(None).unwrap();
    let solved = state
        .solve(
            LinearSolveRequest::new(
                &REFERENCE_LINEAR_SOLVER,
                SolverPlan::new(
                    LinearSolver::ConjugateGradient,
                    1e-13,
                    1e-13,
                    NonZeroUsize::new(32).unwrap(),
                )
                .unwrap(),
            ),
            system,
        )
        .unwrap();
    for (x, actual) in solved.cell_centers().iter().zip(solved.cell_values()) {
        assert!((actual - exact(x)).abs() < 1e-11);
    }
    assert!((solved.integrated_source() + 270.).abs() < 1e-10);
    assert!((solved.integrated_source() + solved.boundary_flux_sum()).abs() < 1e-10);
}
