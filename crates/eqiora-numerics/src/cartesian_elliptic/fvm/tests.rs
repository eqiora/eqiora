use super::*;
use eqiora_solver::{LinearSolver, REFERENCE_LINEAR_SOLVER, SolverPlan};
use std::num::NonZeroUsize;

fn solve(
    axes: Vec<f64>,
    source: f64,
    fluxes: [f64; 2],
    mean: Option<f64>,
    algorithm: LinearSolver,
) -> Result<ScalarEllipticCartesianFvmSolution, Diagnostic> {
    let mesh = CartesianMesh::from_axes(vec![axes])?;
    let assembly = finalize_scalar_elliptic_cartesian_fvm(
        &mesh,
        &|_| 1.,
        &|_| source,
        &|_, side, _| {
            CartesianBoundaryValue::Natural(match side {
                BoundarySide::Lower => fluxes[0],
                BoundarySide::Upper => fluxes[1],
            })
        },
        &QuadratureRule::gauss_legendre(1)?,
        &QuadratureRule::point(),
        &REFERENCE_ASSEMBLY_BACKEND,
    )?;
    let (system, state) = assembly.into_canonical(mean)?;
    assert_eq!(system.properties(), LinearOperatorProperties::Symmetric);
    let plan = SolverPlan::new(algorithm, 1e-12, 1e-12, NonZeroUsize::new(128).unwrap())?;
    state.solve(
        LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, plan),
        system,
    )
}

#[test]
fn affine_neumann_tpfa_uses_cell_measure_reference_and_preserves_gradient() {
    // u=2x-1+c, integral(u)=c, outward grad(u)=[-2,2].
    let axes = vec![0., 0.125, 0.25, 0.5, 1.];
    for mean in [0., 3.] {
        let solved = solve(
            axes.clone(),
            0.,
            [-2., 2.],
            Some(mean),
            LinearSolver::MinimumResidual,
        )
        .unwrap();
        for (center, value) in solved.cell_centers().iter().zip(solved.cell_values()) {
            assert!((value - (2. * center[0] - 1. + mean)).abs() < 1e-10);
        }
        assert!(
            solved
                .reconstruction()
                .l2_error(
                    &|x| 2. * x[0] - 1. + mean,
                    &QuadratureRule::gauss_legendre(2).unwrap()
                )
                .unwrap()
                < 1e-10
        );
        assert!(solved.original_residual_norm() < 1e-10);
        assert_eq!(solved.compatibility_residual(), Some(0.));
        assert!(solved.gauge_residual().unwrap().abs() < 1e-10);
        assert!(solved.gauge_multiplier().unwrap().abs() < 1e-10);
    }
}

#[test]
fn polynomial_neumann_tpfa_has_independently_derived_discrete_mean() {
    // u=x²-x+1/6 has -u''=-2, conormal endpoint loads [1,1].
    // Midpoint quadrature underestimates its mean by h²/12, so the
    // zero-integral cell-constant solution adds h²/12 to each midpoint value.
    for cells in [2, 4, 8] {
        let h = 1. / cells as f64;
        let solved = solve(
            (0..=cells).map(|i| i as f64 * h).collect(),
            -2.,
            [1., 1.],
            Some(0.),
            LinearSolver::MinimumResidual,
        )
        .unwrap();
        for (center, value) in solved.cell_centers().iter().zip(solved.cell_values()) {
            let x = center[0];
            assert!((value - (x * x - x + 1. / 6. + h * h / 12.)).abs() < 1e-10);
        }
        assert!((solved.integrated_source() + solved.boundary_flux_sum()).abs() < 1e-12);
        assert!(solved.cell_values().iter().sum::<f64>().abs() * h < 1e-10);
    }
}

#[test]
fn neumann_tpfa_never_repairs_missing_reference_incompatible_load_or_solver() {
    let axes = vec![0., 0.25, 0.5, 0.75, 1.];
    for (source, flux, mean, solver) in [
        (0., [-2., 2.], None, LinearSolver::MinimumResidual),
        (1., [-2., 2.], Some(0.), LinearSolver::MinimumResidual),
        (-2., [-1., 1.], Some(0.), LinearSolver::MinimumResidual),
        (0., [-2., 2.], Some(0.), LinearSolver::ConjugateGradient),
        (0., [-2., 2.], Some(f64::NAN), LinearSolver::MinimumResidual),
    ] {
        assert!(solve(axes.clone(), source, flux, mean, solver).is_err());
    }
}
