//! Independent finite and uniform population moments through accepted Common Results.
use eqiora_artifact::ModelEnvelope;
use eqiora_backend_faer::FaerLinearSolver;
use eqiora_compiler::{CompiledModel, StaticBindingValue};
use eqiora_core::{DimExponents, DynQuantity};
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_numerics::{CommonAlgebraicPlan, CommonLinearRequest, CommonSolvePolicy};
use eqiora_schema::kernel::AxisBounds;
use eqiora_solver::{LinearSolver, LinearSolverBackend, ReductionPolicy, SolverPlan};
use std::collections::HashMap;
use std::num::NonZeroUsize;

#[test]
fn normalized_uniform_and_finite_moments_keep_units_and_result_lineage() {
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let source = r#"enum Outcome {Low,High}
      model M(support position:interval(m)) {
        parameter scale:1=1;
        variable anchor:1; relation fixed {anchor=scale;}
        distribution finite:normalize_mass(Outcome,weights=[1,3]);
        observable x:m=random_variable(finite,if outcome(finite)==Outcome.Low then 0[m] else 2[m]);
        observable finite_mean:m=expectation(x);
        observable finite_variance:m^2=variance(x);
        observable finite_alias:m=2*finite_mean;
        distribution uniform:normalize_density(measure(position),density=2[1/m]);
        coordinate coordinate_x:m on position from position[0];
        observable u:m=random_variable(uniform,coordinate_x);
        observable v:s=random_variable(uniform,3[s]-1[s/m]*u);
        observable rebound:m=random_variable(uniform,
          u+evaluate(u,at=(coordinate_x=0[m]))+evaluate(u,at=(coordinate_x=3[m])));
        observable rebound_mean:m=expectation(rebound);
        observable shifted:m=random_variable(uniform,
          u+pullback(u,from=(coordinate_x),at=(coordinate_x=0[m]))
           +pullback(u,from=(coordinate_x),at=(coordinate_x=3[m])));
        observable shifted_mean:m=expectation(shifted);
        observable shifted_variance:m^2=variance(shifted);
        observable mean:m=expectation(u);
        observable variance:m^2=variance(u);
        observable covariance:m*s=covariance(u,v);
      }"#;
    // The proportional density 2/m on [0,3m] normalizes to 1/(3m).
    // Integrating 1, x and x² gives mean=3/2 m, variance=3/4 m²;
    // Cov(x,3s-(s/m)x)=-3/4 m*s independently of implementation output.
    let compiled = CompiledModel::compile_selected(
        "probability.eqi",
        source,
        "M",
        &[(
            "position",
            StaticBindingValue::CoordinateInterval(
                AxisBounds::new(DynQuantity::new(0.0, length), DynQuantity::new(3.0, length))
                    .unwrap(),
            ),
        )],
    )
    .unwrap();
    let symbols = compiled.symbols().clone();
    let (transaction, model, _) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = eqiora_sem::KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    // Reusing u across three coordinate bindings must preserve x+0+3, not
    // substitute its first cached value three times. The shared budget must
    // include the nested bindings and must survive successive quadrature calls.
    let point = eqiora_sem::EvaluationPoint::new(
        &program,
        symbols.get("position").unwrap().downcast().unwrap(),
        vec![(
            (symbols.get("position").unwrap(), 0),
            DynQuantity::new(1.0, length),
        )],
        None,
    )
    .unwrap();
    let shifted_mean = symbols.get("rebound_mean").unwrap().downcast().unwrap();
    let work = std::cell::Cell::new(1_000_000);
    let values = program
        .evaluate_observable_integrands_with_points(shifted_mean, &point, &work, &mut |_, _| {
            panic!("coordinate-only random variables")
        })
        .unwrap();
    assert_eq!(values[0].real_scalar_value().unwrap().value(), 4.0);
    let used = 1_000_000 - work.get();
    assert!(used > 0);
    for available in [0, used - 1] {
        let error = program
            .evaluate_observable_integrands_with_points(
                shifted_mean,
                &point,
                &std::cell::Cell::new(available),
                &mut |_, _| panic!("coordinate-only random variables"),
            )
            .unwrap_err();
        assert!(error.message().contains("work budget"), "{error:?}");
    }
    work.set(used);
    program
        .evaluate_observable_integrands_with_points(shifted_mean, &point, &work, &mut |_, _| {
            panic!("coordinate-only random variables")
        })
        .unwrap();
    assert_eq!(work.get(), 0);
    assert!(
        program
            .evaluate_observable_integrands_with_points(
                shifted_mean,
                &point,
                &work,
                &mut |_, _| panic!("coordinate-only random variables")
            )
            .is_err()
    );
    let artifact = ModelEnvelope::from_program(&program).unwrap();
    let solver = &FaerLinearSolver;
    let plan = CommonAlgebraicPlan::resolve(
        &artifact,
        CommonSolvePolicy::Linear(
            CommonLinearRequest::exact(
                SolverPlan::new(
                    LinearSolver::SparseLu,
                    1e-12,
                    1e-12,
                    NonZeroUsize::new(10).unwrap(),
                )
                .unwrap()
                .with_reduction(ReductionPolicy::Fast),
                solver.provider(),
            )
            .unwrap(),
        ),
        None,
        &[],
        None,
        solver,
    )
    .unwrap();
    let result = plan
        .run_result(&plan.initial_state(&[]).unwrap(), solver)
        .unwrap();
    // Ordinary finite sensitivities still compile; a moment or an alias of one
    // must not acquire a zero derivative by silently treating it as a constant.
    let document = eqiora::api::ModelDocument::replay(&artifact.canonical_json().unwrap()).unwrap();
    let parameter = document
        .parameter_ref(&symbols.get("scale").unwrap().ulid().to_string())
        .unwrap();
    for name in ["anchor", "finite_mean", "finite_alias"] {
        let selection = symbols.get(name).unwrap().ulid().to_string();
        let output = if name == "anchor" {
            document.field_ref(&selection).unwrap().erase()
        } else {
            document.observable_ref(&selection).unwrap().erase()
        };
        let differentiated = eqiora::api::DifferentiableProgram::compile(
            eqiora_numerics::ResolvedCommonPlan::Algebraic(Box::new(plan.clone())),
            std::slice::from_ref(&parameter),
            &[output],
            None,
            solver,
        );
        if name == "anchor" {
            differentiated.unwrap();
        } else {
            let errors = differentiated.unwrap_err();
            assert!(
                errors
                    .iter()
                    .any(|error| error.message().contains("instantaneous")
                        || error
                            .message()
                            .contains("separately admitted differentiation")),
                "{name}: {errors:?}"
            );
        }
    }
    // Native point evaluation supports Evaluate; the quadrature regularity
    // projection currently admits Pullback but not Evaluate. Keep that boundary
    // explicit while testing the same rebinding semantics through both owners.
    let rules = result
        .plan()
        .observation_quadratures(&[shifted_mean], 2)
        .unwrap();
    let error = result
        .observe(&artifact, shifted_mean, &rules, None)
        .unwrap_err();
    assert!(error.message().contains("scalar Operator IR"), "{error:?}");
    for (name, expected, units, continuous) in [
        ("finite_mean", 1.5, [0, 1, 0, 0, 0, 0, 0], false),
        ("finite_alias", 3.0, [0, 1, 0, 0, 0, 0, 0], false),
        ("finite_variance", 0.75, [0, 2, 0, 0, 0, 0, 0], false),
        ("shifted_mean", 4.5, [0, 1, 0, 0, 0, 0, 0], true),
        ("shifted_variance", 0.75, [0, 2, 0, 0, 0, 0, 0], true),
        ("mean", 1.5, [0, 1, 0, 0, 0, 0, 0], true),
        ("variance", 0.75, [0, 2, 0, 0, 0, 0, 0], true),
        ("covariance", -0.75, [0, 1, 1, 0, 0, 0, 0], true),
    ] {
        let observable = symbols.get(name).unwrap().downcast().unwrap();
        let rules = if continuous {
            result
                .plan()
                .observation_quadratures(&[observable], 2)
                .unwrap()
        } else {
            HashMap::new()
        };
        let observed = result.observe(&artifact, observable, &rules, None).unwrap();
        let value = observed.value().real_scalar_value().unwrap();
        assert!(
            (value.value() - expected).abs() <= 64.0 * f64::EPSILON * expected.abs(),
            "{name}: {}",
            value.value()
        );
        assert_eq!(value.dim(), DimExponents::from_integers(units).unwrap());
        assert_eq!(observed.result_identity(), result.identity());
        assert_eq!(observed.quadratures(), &rules);
        if continuous {
            assert!(
                result
                    .observe(&artifact, observable, &HashMap::new(), None)
                    .is_err()
            );
        }
    }
}
