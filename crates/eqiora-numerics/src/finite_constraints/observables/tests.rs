//! Independently derived nonlinear integral residuals and Parameter partials.
use super::*;
use crate::finite_constraints::{
    ConstraintRef, ConstraintTolerance, FiniteConstraintEnforcement, lower_finite_constraints,
};
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_realization::NonlinearSolvePlan;
use eqiora_solver::{
    LinearSolveRequest, LinearSolver, REFERENCE_LINEAR_SOLVER, ReductionPolicy, SolverPlan,
};
use std::num::NonZeroUsize;

#[test]
fn integral_candidates_and_ad_share_the_exact_parameter_point() {
    let source = "model Root(support body:interval(m)) {
        parameter p:1=4; parameter k:1=1; variable w:1;
        observable mass:m=integral(k*w*w,measure(body));
        relation root {mass=2[m]*p; inequality(w>=0);}
    }";
    let length = eqiora_core::DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let compiled = eqiora_compiler::CompiledModel::compile_selected(
        "root.eqi",
        source,
        "Root",
        &[(
            "body",
            eqiora_compiler::StaticBindingValue::CoordinateInterval(
                eqiora_schema::kernel::AxisBounds::new(
                    DynQuantity::new(0.0, length),
                    DynQuantity::new(2.0, length),
                )
                .unwrap(),
            ),
        )],
    )
    .unwrap();
    let p = compiled.symbols().get("p").unwrap().downcast().unwrap();
    let k = compiled.symbols().get("k").unwrap().downcast().unwrap();
    let relation = compiled.symbols().get("root").unwrap().downcast().unwrap();
    let (transaction, model, _) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let kernel = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let policy = FiniteConstraintEnforcement::strict_interior(vec![
        ConstraintTolerance::inequality(
            ConstraintRef::new(relation, 1),
            DynQuantity::new(1e-8, eqiora_core::DimExponents::DIMENSIONLESS),
        )
        .unwrap(),
    ])
    .unwrap();
    let problem = lower_finite_constraints(&kernel, Some(&policy), true).unwrap();
    let nonlinear =
        NonlinearSolvePlan::new(0.0, 1e-12, NonZeroUsize::new(32).unwrap(), 16).unwrap();
    let linear = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1e-13,
        1e-15,
        NonZeroUsize::new(8).unwrap(),
    )
    .unwrap()
    .with_reduction(ReductionPolicy::Reproducible);
    // Constant-density integral on [0,2] gives R=2kw²-2p.
    // Thus R_w=4kw, R_p=-2, R_k=2w², w=sqrt(p/k).
    // Vary both a Parameter inside the integral and one outside it; then replay
    // the original point to falsify accidental mutation or stale default values.
    for (p_value, k_value, w) in [(4.0, 1.0, 2.0), (9.0, 4.0, 1.5), (4.0, 1.0, 2.0)] {
        let point = problem.at_parameters(&[p, k], &[p_value, k_value]).unwrap();
        let solved = point
            .solve_at_point(
                &[1.0],
                nonlinear,
                LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, linear),
            )
            .unwrap();
        assert!((solved.values[0] - w).abs() <= 1e-12);
        assert!(point.original_residual(&[w]).unwrap()[0].abs() <= 1e-12);
        let (actions, _) = point.equality_jacobian(&[w], &[p, k]).unwrap();
        assert_eq!(actions.values, [0.0]);
        assert_eq!(actions.unknown_jacobian, [4.0 * k_value * w]);
        assert_eq!(actions.parameter_jacobian, [-2.0, 2.0 * w * w]);
    }
}
