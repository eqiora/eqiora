use super::*;
use eqiora_solver::{LinearSolver, REFERENCE_LINEAR_SOLVER, SolverPlan};
use std::num::NonZeroUsize;

fn request() -> LinearSolveRequest<'static> {
    LinearSolveRequest::new(
        &REFERENCE_LINEAR_SOLVER,
        SolverPlan::new(
            LinearSolver::MinimumResidual,
            1e-12,
            1e-12,
            NonZeroUsize::new(100).unwrap(),
        )
        .unwrap(),
    )
}

#[test]
fn affine_neumann_solution_uses_physical_mean_on_nonuniform_mesh() {
    // u=2x-1 on [0,1], k=3: f=0 and outward k*u' is [-6,6].
    // Its spatial mean is zero, but its arithmetic nodal mean is not.
    let mesh = LineMesh::from_vertices(vec![0., 0.125, 0.25, 0.5, 1.]).unwrap();
    for mean in [0., 3.] {
        let solved = solve_scalar_elliptic_linear_fem(
            &mesh,
            &|_| 3.,
            &|_| 0.,
            ScalarBoundaryPair1d::pure_neumann(-6., 6., mean).unwrap(),
            &QuadratureRule::gauss_legendre(2).unwrap(),
            request(),
        )
        .unwrap();
        for (vertex, &value) in mesh.vertices().zip(solved.field().values()) {
            let x = mesh.vertex_coordinate(vertex).unwrap();
            assert!((value - (2. * x - 1. + mean)).abs() < 1e-10);
        }
        assert!(
            solved
                .cell_gradients()
                .iter()
                .all(|gradient| (gradient - 2.).abs() < 1e-10)
        );
        assert_eq!(solved.endpoint_reactions(), [None, None]);
        assert_eq!(solved.compatibility_residual(), Some(0.));
        assert!(solved.gauge_multiplier().unwrap().abs() < 1e-10);
        assert!(solved.gauge_residual().unwrap().abs() < 1e-10);
        assert!(solved.residual_norm() < 1e-10);
    }
}

#[test]
fn polynomial_source_balance_and_discrete_zero_mean_are_independent() {
    // u=x²-x+1/6, -u''=-2, outward u'=[1,1], integral(f)+sum(flux)=0.
    // P1 stiffness reproduces the nodal quadratic up to a constant. On each
    // uniform cell its interpolation has excess integral h³/6, so subtract
    // h²/6 from the continuum constant to enforce the exact P1 mean.
    for cells in [2, 4, 8] {
        let h = 1. / cells as f64;
        let mesh = LineMesh::uniform(0., 1., cells).unwrap();
        let solved = solve_scalar_elliptic_linear_fem(
            &mesh,
            &|_| 1.,
            &|_| -2.,
            ScalarBoundaryPair1d::pure_neumann(1., 1., 0.).unwrap(),
            &QuadratureRule::gauss_legendre(2).unwrap(),
            request(),
        )
        .unwrap();
        let mut integral = 0.;
        for (i, &value) in solved.field().values().iter().enumerate() {
            let x = i as f64 * h;
            let expected = x * x - x + (1. - h * h) / 6.;
            assert!((value - expected).abs() < 1e-10);
            integral += if i == 0 || i == cells {
                0.5 * h * value
            } else {
                h * value
            };
        }
        assert!(integral.abs() < 1e-10);
        assert!(solved.residual_norm() < 1e-10);
    }
}

#[test]
fn incompatible_source_and_wrong_outward_boundary_sign_are_not_repaired() {
    let mesh = LineMesh::uniform(0., 1., 4).unwrap();
    for (source, lower, upper) in [(-1., 1., 1.), (-2., -1., 1.), (-2., 1., -1.)] {
        let error = solve_scalar_elliptic_linear_fem(
            &mesh,
            &|_| 1.,
            &|_| source,
            ScalarBoundaryPair1d::pure_neumann(lower, upper, 0.).unwrap(),
            &QuadratureRule::gauss_legendre(2).unwrap(),
            request(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("load is incompatible"));
    }
}

#[test]
fn no_implicit_gauge_or_spd_solver_substitution() {
    assert!(
        ScalarBoundaryPair1d::new(
            ScalarBoundaryCondition1d::Natural(-1.),
            ScalarBoundaryCondition1d::Natural(1.),
        )
        .is_err()
    );
    let mesh = LineMesh::uniform(0., 1., 4).unwrap();
    let cg = LinearSolveRequest::new(
        &REFERENCE_LINEAR_SOLVER,
        SolverPlan::new(
            LinearSolver::ConjugateGradient,
            1e-12,
            1e-12,
            NonZeroUsize::new(100).unwrap(),
        )
        .unwrap(),
    );
    assert!(
        solve_scalar_elliptic_linear_fem(
            &mesh,
            &|_| 1.,
            &|_| 0.,
            ScalarBoundaryPair1d::pure_neumann(-1., 1., 0.).unwrap(),
            &QuadratureRule::gauss_legendre(2).unwrap(),
            cg,
        )
        .is_err()
    );
}
