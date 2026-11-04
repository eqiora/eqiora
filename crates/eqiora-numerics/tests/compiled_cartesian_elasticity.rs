use std::num::NonZeroUsize;

use eqiora_compiler::compile;
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_meshing::{CartesianMesh, MeshTopology, QuadratureRule};
use eqiora_numerics::solid::{
    lower_cartesian_q1_linear_elasticity_local_action_2d, lower_isotropic_elasticity_cartesian_2d,
    solve_cartesian_q1_linear_elasticity_2d,
};
use eqiora_sem::KernelProgram;
use eqiora_solver::{LinearSolveRequest, LinearSolver, REFERENCE_LINEAR_SOLVER, SolverPlan};

const SOURCE: &str =
    include_str!("../../../verify/solid/isotropic-elasticity-2d/models/linear-load.eqi");

#[test]
fn public_elasticity_construction_and_scatter_smoke() {
    let program = compile_program();
    let model = lower_isotropic_elasticity_cartesian_2d(&program).unwrap();
    let mesh = CartesianMesh::uniform(model.bounds(), &[2, 2]).unwrap();
    let quadrature = QuadratureRule::tensor_product_gauss_legendre(2, 2).unwrap();

    let local_action = lower_cartesian_q1_linear_elasticity_local_action_2d(
        &mesh,
        model.shear_modulus(),
        model.first_lame_parameter(),
        &quadrature,
    )
    .unwrap();
    assert_eq!(local_action.entity_count(), mesh.entity_count(2).unwrap());
    assert!(
        local_action
            .coefficients()
            .iter()
            .all(|value| value.is_finite())
    );

    let plan = SolverPlan::new(
        LinearSolver::ConjugateGradient,
        1.0e-15,
        1.0e-15,
        NonZeroUsize::new(64).unwrap(),
    )
    .unwrap();
    let solution = solve_cartesian_q1_linear_elasticity_2d(
        &mesh,
        model.shear_modulus(),
        model.first_lame_parameter(),
        model.load_potential_expression(),
        &quadrature,
        LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, plan),
    )
    .unwrap();
    assert_eq!(solution.displacement().mesh(), &mesh);
    assert!(
        solution
            .displacement()
            .values()
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(
        solution
            .boundary_reaction()
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(
        solution
            .integrated_body_force()
            .iter()
            .all(|value| value.is_finite())
    );
}

fn compile_program() -> KernelProgram {
    let mut compiled = compile("compiled-cartesian-elasticity.eqi", SOURCE).unwrap();
    let (transaction, model, _) = compiled.remove(0).into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    KernelProgram::from_snapshot(&store.snapshot(), model).unwrap()
}

fn program_from(source: &str) -> KernelProgram {
    let mut compiled = compile("optional-solid-load.eqi", source).unwrap();
    let (transaction, model, _) = compiled.remove(0).into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    KernelProgram::from_snapshot(&store.snapshot(), model).unwrap()
}

fn unforced_source() -> String {
    SOURCE
        .replace("  variable load_potential: kg / (m * s ^ 2) on body;\n", "")
        .replace("  parameter pressure_gradient: kg / (m ^ 2 * s ^ 2) = 1;\n", "")
        .replace("  relation load on body {\n    load_potential - pressure_gradient * coordinate(0) = 0;\n  }\n", "")
        .replace(" - grad(load_potential)", "")
}

#[test]
fn unforced_elasticity_has_no_load_identity_and_solves_the_homogeneous_problem() {
    let direct =
        lower_isotropic_elasticity_cartesian_2d(&program_from(&unforced_source())).unwrap();
    let legacy =
        lower_isotropic_elasticity_cartesian_2d(&program_from(&SOURCE.replace("= 1;", "= 0;")))
            .unwrap();
    assert_eq!(direct.load_potential(), None);
    assert_eq!(direct.load_definition_relation(), None);
    assert!(
        direct
            .load_potential_expression()
            .parameter_fields()
            .is_empty()
    );
    assert!(legacy.load_potential().is_some());
    for model in [&direct, &legacy] {
        assert_eq!(
            model.conservative_body_force(&[0.3, 0.7]).unwrap(),
            [0.0; 2]
        );
        let mesh = CartesianMesh::uniform(model.bounds(), &[2, 2]).unwrap();
        let quadrature = QuadratureRule::tensor_product_gauss_legendre(2, 2).unwrap();
        let plan = SolverPlan::new(
            LinearSolver::ConjugateGradient,
            1.0e-15,
            1.0e-15,
            NonZeroUsize::new(64).unwrap(),
        )
        .unwrap();
        let solution = solve_cartesian_q1_linear_elasticity_2d(
            &mesh,
            model.shear_modulus(),
            model.first_lame_parameter(),
            model.load_potential_expression(),
            &quadrature,
            LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, plan),
        )
        .unwrap();
        // The unique zero-Dirichlet solution of homogeneous coercive elasticity is zero.
        assert!(
            solution
                .displacement()
                .values()
                .iter()
                .all(|value| value.abs() < 1e-14)
        );
        assert!(
            solution
                .integrated_body_force()
                .iter()
                .all(|value| value.abs() < 1e-14)
        );
        assert!(
            solution
                .boundary_reaction()
                .iter()
                .all(|value| value.abs() < 1e-14)
        );
    }
    let forced = lower_isotropic_elasticity_cartesian_2d(&program_from(SOURCE)).unwrap();
    // Independently differentiate q = pressure_gradient * x, pressure_gradient = 1.
    assert_eq!(
        forced.conservative_body_force(&[0.3, 0.7]).unwrap(),
        [1.0, 0.0]
    );
}

#[test]
fn unforced_elasticity_rejects_orphan_load_and_extra_volume_equations() {
    for source in [
        SOURCE.replace(" - grad(load_potential)", ""),
        SOURCE.replace("  relation load on body {\n    load_potential - pressure_gradient * coordinate(0) = 0;\n  }\n", ""),
        unforced_source().replace("  relation balance on body {", "  relation extra on body { div(displacement) = 0; }\n  relation balance on body {"),
    ] {
        let diagnostic = lower_isotropic_elasticity_cartesian_2d(&program_from(&source)).unwrap_err();
        assert_eq!(diagnostic.code(), eqiora_core::diagnostic::codes::INVALID_SPATIAL_LOWERING);
    }
}
