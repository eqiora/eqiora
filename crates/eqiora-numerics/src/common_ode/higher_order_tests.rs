use super::*;
use eqiora_core::DynQuantity;
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_solver::REFERENCE_LINEAR_SOLVER;

#[test]
fn higher_order_plan_preserves_coordinates_units_controls_and_state_replay() {
    let source = "model Oscillator() { parameter k:1/s^2=4; state x:m; initial { x=1[m]; derivative(x)=2[m/s]; } relation motion { derivative(derivative(x))+k*x=0[m/s^2]; } }";
    let (transaction, model_id, symbols) = eqiora_compiler::compile("oscillator.eqi", source)
        .unwrap()
        .remove(0)
        .into_parts();
    let field = symbols.get("x").unwrap().downcast().unwrap();
    let parameter = symbols.get("k").unwrap().downcast().unwrap();
    let relation = symbols.get("motion").unwrap().downcast().unwrap();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let kernel = KernelProgram::from_snapshot(&store.snapshot(), model_id).unwrap();
    let model = ModelEnvelope::from_program(&kernel).unwrap();
    let backend = eqiora_time::TimeBackendCapabilities::new(
        eqiora_time::TimeBackendIdentity::new("eqiora.test.time", "1"),
        &[
            eqiora_core::ScalarDomain::Real,
            eqiora_core::ScalarDomain::Complex,
        ],
        &[eqiora_core::ScalarType::F64],
    );
    let tolerance = |order, value| {
        CommonTimeTolerance::new(
            eqiora_core::TimeStateCoordinate::new(field, order, 0, false),
            value,
        )
        .unwrap()
    };
    let policy = |entries| {
        CommonOdePolicy::new(eqiora_time::TimeMethod::Tsitouras45, 0.01, 1e-9, entries).unwrap()
    };
    let temporal = policy(vec![tolerance(1, 2e-11), tolerance(0, 1e-11)]);
    let plan = CommonOdePlan::resolve(&model, &kernel, temporal.clone(), backend).unwrap();
    assert_eq!(
        plan.state_coordinates().collect::<Vec<_>>(),
        [
            eqiora_core::TimeStateCoordinate::new(field, 0, 0, false),
            eqiora_core::TimeStateCoordinate::new(field, 1, 0, false)
        ]
    );
    // DimExponents follows kg, m, s, A, K, mol, cd (core quantity contract).
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let velocity = DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).unwrap();
    assert_eq!(plan.state_dimensions(), &[length, velocity]);
    assert_eq!(plan.ordered_absolute_tolerances, [1e-11, 2e-11]);
    let initial = plan.initial_state(0.0).unwrap();
    assert_eq!(
        initial.state_coordinates(),
        &[
            eqiora_core::TimeStateCoordinate::new(field, 0, 0, false),
            eqiora_core::TimeStateCoordinate::new(field, 1, 0, false)
        ]
    );
    assert_eq!(initial.values(), &[1., 2.]);
    assert_eq!(
        CommonOdeState::from_bytes(&initial.to_bytes().unwrap(), &plan).unwrap(),
        initial
    );
    let resolved = crate::ResolvedCommonPlan::Ode(Box::new(plan.clone()));
    let bytes = resolved.to_bytes().unwrap();
    let replay =
        crate::ResolvedCommonPlan::from_bytes(&bytes, &REFERENCE_LINEAR_SOLVER, backend).unwrap();
    assert_eq!(replay, resolved);
    for inspected in [&resolved, &replay] {
        let form = inspected.formulation().unwrap();
        assert_eq!(form.source_relation(), Some(relation));
        assert_eq!(
            form.state_coordinates(),
            &[
                eqiora_core::TimeStateCoordinate::new(field, 0, 0, false),
                eqiora_core::TimeStateCoordinate::new(field, 1, 0, false)
            ]
        );
        assert_eq!(
            form.effective(),
            crate::FormulationKind::FirstOrderEvolution
        );
    }
    assert!(
        CommonOdePlan::resolve(&model, &kernel, policy(vec![tolerance(0, 1e-11)]), backend)
            .is_err()
    );
    assert!(
        CommonOdePlan::resolve(
            &model,
            &kernel,
            policy(vec![tolerance(0, 1e-11), tolerance(2, 2e-11)]),
            backend
        )
        .is_err()
    );
    assert!(
        CommonOdePolicy::new(
            eqiora_time::TimeMethod::Tsitouras45,
            0.01,
            1e-9,
            vec![tolerance(0, 1e-11); 2]
        )
        .is_err()
    );
    // d(x)/d(k) has m*s², d(velocity)/d(k) has m*s.
    let displacement_sensitivity = DimExponents::from_integers([0, 1, 2, 0, 0, 0, 0]).unwrap();
    let velocity_sensitivity = DimExponents::from_integers([0, 1, 1, 0, 0, 0, 0]).unwrap();
    let sensitivities = vec![
        (
            eqiora_core::TimeStateCoordinate::new(field, 0, 0, false),
            parameter,
            DynQuantity::new(3e-11, displacement_sensitivity),
        ),
        (
            eqiora_core::TimeStateCoordinate::new(field, 1, 0, false),
            parameter,
            DynQuantity::new(4e-11, velocity_sensitivity),
        ),
    ];
    let sensitive = temporal
        .clone()
        .with_forward_sensitivities(1e-9, sensitivities.clone())
        .unwrap();
    let sensitive = CommonOdePlan::resolve(&model, &kernel, sensitive, backend).unwrap();
    let resolved = crate::ResolvedCommonPlan::Ode(Box::new(sensitive));
    assert_eq!(
        crate::ResolvedCommonPlan::from_bytes(
            &resolved.to_bytes().unwrap(),
            &REFERENCE_LINEAR_SOLVER,
            backend
        )
        .unwrap(),
        resolved
    );
    let mut wrong_units = sensitivities;
    wrong_units[1].2 = DynQuantity::new(4e-11, displacement_sensitivity);
    let wrong = temporal
        .with_forward_sensitivities(1e-9, wrong_units)
        .unwrap();
    assert!(CommonOdePlan::resolve(&model, &kernel, wrong, backend).is_err());
}
