use std::num::NonZeroUsize;

use eqiora_solver::{LinearSolver, REFERENCE_LINEAR_SOLVER, SolverPlan};

use super::*;

#[test]
fn q1_diffusion_lowers_to_anonymous_uniform_local_action() {
    let mesh = CartesianMesh::from_axes(vec![vec![0.0, 0.2, 1.0], vec![-1.0, 0.5]]).unwrap();
    let rule = QuadratureRule::tensor_product_gauss_legendre(2, 2).unwrap();
    let action = lower_cartesian_q1_diffusion_local_action(&mesh, &|_: &[f64]| 1.7, &rule).unwrap();
    let input = vec![1.0; action.input_len()];
    let mut output = vec![f64::NAN; action.output_len()];

    action.apply_reference(&input, &mut output).unwrap();

    assert_eq!(action.entity_count(), 2);
    assert_eq!((action.rows(), action.columns()), (4, 4));
    assert!(output.iter().all(|value| value.abs() < 8.0e-15));
}

#[test]
fn both_methods_reproduce_a_linear_harmonic_field_on_a_nonuniform_grid() {
    let mesh =
        CartesianMesh::from_axes(vec![vec![0.0, 0.2, 0.65, 1.0], vec![-1.0, -0.1, 0.4, 2.0]])
            .unwrap();
    let cell_rule = QuadratureRule::tensor_product_gauss_legendre(2, 2).unwrap();
    let facet_rule = QuadratureRule::gauss_legendre(2).unwrap();
    let plan = SolverPlan::new(
        LinearSolver::ConjugateGradient,
        1.0e-12,
        1.0e-14,
        NonZeroUsize::new(256).unwrap(),
    )
    .unwrap();
    let solver = LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, plan);
    let exact = |coordinate: &[f64]| 2.0 + coordinate[0] - 0.5 * coordinate[1];
    let source = |_: &[f64]| 0.0;

    let fem = solve_scalar_elliptic_cartesian_fem(&mesh, 1.0, &source, &exact, &cell_rule, solver)
        .unwrap();
    let fvm = solve_scalar_elliptic_cartesian_fvm(
        &mesh,
        1.0,
        &source,
        &exact,
        &cell_rule,
        &facet_rule,
        solver,
    )
    .unwrap();

    assert!(fem.field().l2_error(&exact, &cell_rule).unwrap() < 2.0e-13);
    let dual_rule = QuadratureRule::tensor_product_gauss_legendre(2, 2).unwrap();
    assert!(fvm.reconstruction().l2_error(&exact, &dual_rule).unwrap() < 2.0e-13);
    assert!(fem.boundary_reaction_sum().abs() < 2.0e-12);
    assert!(fvm.boundary_flux_sum().abs() < 2.0e-12);
    assert!((fem.boundary_reaction_sum() + fem.integrated_source()).abs() < 2.0e-12);
    assert!((fvm.boundary_flux_sum() + fvm.integrated_source()).abs() < 2.0e-12);
}
