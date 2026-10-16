//! Independently derived nonlinear integral residuals and Parameter partials.
use super::*;
use crate::finite_constraints::{
    ConstraintRef, ConstraintTolerance, FiniteConstraintEnforcement, lower_finite_constraints,
};
use eqiora_core::DynQuantity;
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

#[test]
fn explicit_random_variable_realizations_keep_scoped_points_and_parameter_partials() {
    let source = "model Sample(support noise:interval(1)) {
        parameter location:1=2; parameter scale:1=3; parameter draw:1=0.5;
        coordinate z:1 on noise from noise[0];
        distribution base:normalize_density(measure(noise),density=1);
        observable x:1=random_variable(base,location+scale*z);
        observable mean:1=expectation(x); observable alias:1=mean;
        observable sample:1=evaluate(x,at=(z=draw))+evaluate(x,at=(z=1));
        variable result:1; relation output {result=sample;}
    }";
    let bindings = [(
        "noise",
        eqiora_compiler::StaticBindingValue::CoordinateInterval(
            eqiora_schema::kernel::AxisBounds::new(
                DynQuantity::new(-2.0, eqiora_core::DimExponents::DIMENSIONLESS),
                DynQuantity::new(2.0, eqiora_core::DimExponents::DIMENSIONLESS),
            )
            .unwrap(),
        ),
    )];
    let foreign = source.replace(
        "observable mean:1=expectation(x);",
        "distribution other:normalize_density(measure(noise),density=1);
         observable foreign:1=random_variable(other,evaluate(x,at=(z=draw)));
         observable mean:1=expectation(x);",
    );
    let compiled_foreign = eqiora_compiler::CompiledModel::compile_selected(
        "foreign.eqi",
        &foreign,
        "Sample",
        &bindings,
    )
    .unwrap();
    let (transaction, model, _) = compiled_foreign.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let errors = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.message().contains("exact probability-law context")),
        "{errors:?}"
    );
    let compiled =
        eqiora_compiler::CompiledModel::compile_selected("sample.eqi", source, "Sample", &bindings)
            .unwrap();
    let selected = ["location", "scale", "draw"]
        .map(|name| compiled.symbols().get(name).unwrap().downcast().unwrap());
    let x = compiled.symbols().get("x").unwrap().downcast().unwrap();
    let noise = compiled.symbols().get("noise").unwrap().downcast().unwrap();
    let moments =
        ["mean", "alias"].map(|name| compiled.symbols().get(name).unwrap().downcast().unwrap());
    let (transaction, model, _) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let kernel = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let evaluation_point = eqiora_sem::EvaluationPoint::new(
        &kernel,
        noise,
        vec![(
            (noise.erase(), 0),
            DynQuantity::new(0.5, eqiora_core::DimExponents::DIMENSIONLESS),
        )],
        None,
    )
    .unwrap();
    let realized = kernel
        .evaluate_observable_with_points(x, Some(&evaluation_point), &mut |input, _| match input {
            eqiora_sem::EvaluationInput::Value(SymbolRef::Parameter(id)) => {
                Ok(kernel.typed_value(id.erase()).unwrap().clone())
            }
            _ => panic!("only immutable parameters require resolution"),
        })
        .unwrap();
    assert_eq!(realized.real_scalar_value().unwrap().value(), 3.5);
    assert!(
        kernel
            .evaluate_observable_with_points(x, None, &mut |_, _| unreachable!())
            .is_err()
    );
    for moment in moments {
        let mut builder = ExprDagBuilder::new();
        let root = builder.symbol(SymbolRef::Observable(moment)).unwrap();
        let error = expand(&kernel, &builder.finish([root]).unwrap()).unwrap_err();
        assert!(
            error.message().contains("differentiation assumptions"),
            "{error:?}"
        );
    }
    let problem = lower_finite_constraints(&kernel, None, true).unwrap();
    // Two bindings of the same random variable give 2*location+scale*(draw+1).
    // Every derivative is of this exact finite realization, not a population mean.
    for [location, scale, draw] in [[2.0, 3.0, 0.5], [-1.0, 2.0, -0.5], [2.0, 3.0, 0.5]] {
        let point = problem
            .at_parameters(&selected, &[location, scale, draw])
            .unwrap();
        let result = 2.0 * location + scale * (draw + 1.0);
        assert_eq!(point.original_residual(&[result]).unwrap(), [0.0]);
        let (actions, _) = point.equality_jacobian(&[result], &selected).unwrap();
        assert_eq!(actions.values, [0.0]);
        assert_eq!(actions.unknown_jacobian, [1.0]);
        assert_eq!(actions.parameter_jacobian, [-2.0, -(draw + 1.0), -scale]);
    }
    let outside = problem.at_parameters(&selected, &[2.0, 3.0, 2.5]).unwrap();
    assert!(outside.original_residual(&[14.5]).is_err());
    assert!(outside.equality_jacobian(&[14.5], &selected).is_err());

    let mut builder = ExprDagBuilder::new();
    let unbound = builder.symbol(SymbolRef::Observable(x)).unwrap();
    let error = expand(&kernel, &builder.finish([unbound]).unwrap()).unwrap_err();
    assert!(
        error.message().contains("every exact law coordinate"),
        "{error:?}"
    );
}
