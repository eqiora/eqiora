use super::*;
use eqiora::backends::diffsol::DIFFSOL_TIME_BACKEND;
use eqiora_numerics::{
    CommonOdePlan, CommonOdeRunRequest, CommonTrajectory, CommonTsitouras45,
    CommonTsitourasTolerance, ResolvedCommonPlan, TimeFunctionalQuadrature,
};

fn fixture(rate: u32) -> (ModelEnvelope, KernelProgram) {
    let source = format!(
        r#"
model Decay() {{
  state x: 1;
  initial {{ x = 3; }}
  parameter rate: 1/s = {rate};
  relation flow {{ derivative(x) = -rate*x; }}
  observable sample: 1 = x;
  observable weighted: 1/s = rate*x;
}}
"#
    );
    let compiled = eqiora::compiler::compile("functional-decay.eqi", &source)
        .unwrap()
        .pop()
        .unwrap();
    let (transaction, model, _) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let kernel = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    (ModelEnvelope::from_program(&kernel).unwrap(), kernel)
}

use eqiora::sem::KernelProgram;

#[test]
fn ordinary_time_functional_uses_native_steps_and_exact_trajectory_lineage() {
    let (model, kernel) = fixture(2);
    let field = kernel
        .nodes()
        .find_map(|node| match node {
            KernelNode::Field(value) => Some(value.id()),
            _ => None,
        })
        .unwrap();
    let observable = kernel
        .nodes()
        .find_map(|node| match node {
            KernelNode::Observable(value)
                if value.value_type().dimension() == DimExponents::DIMENSIONLESS =>
            {
                Some(value.id())
            }
            _ => None,
        })
        .unwrap();
    let temporal = CommonTsitouras45::new(
        1e-3,
        1e-11,
        vec![CommonTsitourasTolerance::new(field, 1e-13).unwrap()],
    )
    .unwrap();
    let plan = CommonOdePlan::resolve(&model, &kernel, temporal, DIFFSOL_TIME_BACKEND).unwrap();
    assert_eq!(plan.field_ids().len(), 1, "functional adds no ODE state");
    let run = |output_times| {
        let request = CommonOdeRunRequest::new(
            plan.clone(),
            plan.initial_state().unwrap(),
            1.0,
            output_times,
        )
        .unwrap();
        let solution = DiffsolTimeBackend::new()
            .solve(&request.problem().unwrap(), request.time_plan())
            .unwrap();
        CommonTrajectory::accept_ode(request, solution).unwrap()
    };
    let sparse = run(vec![0.25]);
    let dense = run((1..10).map(|index| f64::from(index) / 10.0).collect());
    assert_eq!(sparse.ode_states().unwrap().len(), 1);
    assert_eq!(
        sparse.ode_history(),
        dense.ode_history(),
        "requested samples do not choose accepted integration steps"
    );
    let rule = TimeFunctionalQuadrature::AcceptedStepSimpson;
    let integrated = sparse
        .observe_time_integral(&model, observable, rule)
        .unwrap();
    let dense_integrated = dense
        .observe_time_integral(&model, observable, rule)
        .unwrap();
    assert_eq!(integrated.value(), dense_integrated.value());
    assert_eq!(integrated.trajectory_identity(), sparse.identity());
    assert_eq!(integrated.quadrature(), Some(rule));
    assert_eq!(integrated.interval_s(), [0.0, 1.0]);
    assert_eq!(integrated.endpoint_convention(), "fixed-interval-dt");
    // Solve x'= -2x, x(0)=3 independently: x=3exp(-2t), J=3(1-exp(-2))/2.
    let expected = 1.5 * (1.0 - (-2.0_f64).exp());
    assert!((integrated.value().real_scalar_value().unwrap().value() - expected).abs() < 1e-8);
    assert_eq!(
        integrated.value().value_type().dimension(),
        DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).unwrap()
    );
    let terminal = sparse.observe_terminal(&model, observable).unwrap();
    assert!(
        (terminal.value().real_scalar_value().unwrap().value() - 3.0 * (-2.0_f64).exp()).abs()
            < 1e-9
    );
    assert_eq!(terminal.quadrature(), None);
    assert_eq!(terminal.interval_s(), [1.0, 1.0]);
    assert_eq!(terminal.endpoint_convention(), "terminal-after-events");
    let request =
        CommonOdeRunRequest::new(plan.clone(), plan.initial_state().unwrap(), 1.0, vec![0.25])
            .unwrap();
    let solution = DiffsolTimeBackend::new()
        .solve_forward_sensitivities(
            &request.forward_sensitivity_problem().unwrap(),
            request.time_plan(),
            &ForwardSensitivityPlan::new(1e-11, vec![1e-13]).unwrap(),
        )
        .unwrap();
    let (sensitive, tangent) =
        CommonTrajectory::accept_ode_forward_sensitivities(request, solution).unwrap();
    let rate = plan.parameter_ids()[0];
    let rate_dimension = DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).unwrap();
    let direction = [(rate, DynQuantity::new(1.0, rate_dimension))];
    let derivative = sensitive
        .observe_time_integral_parameter_jvp(&model, observable, rule, &tangent, direction)
        .unwrap();
    // d[3(1-exp(-a))/a]/da = 3*((a+1)*exp(-a)-1)/a^2 at a=2.
    let expected_derivative = 0.75 * (3.0 * (-2.0_f64).exp() - 1.0);
    assert!(
        (derivative.value().real_scalar_value().unwrap().value() - expected_derivative).abs()
            < 1e-8
    );
    assert!(derivative.is_parameter_jvp());
    assert_eq!(
        derivative.trajectory_identity(),
        tangent.trajectory_identity()
    );
    let weighted = kernel
        .nodes()
        .find_map(|node| match node {
            KernelNode::Observable(value) if value.value_type().dimension() == rate_dimension => {
                Some(value.id())
            }
            _ => None,
        })
        .unwrap();
    // Integral(a*x)=3(1-exp(-a)); derivative=3exp(-a), including direct a dependence.
    let direct = sensitive
        .observe_time_integral_parameter_jvp(&model, weighted, rule, &tangent, direction)
        .unwrap();
    assert!(
        (direct.value().real_scalar_value().unwrap().value() - 3.0 * (-2.0_f64).exp()).abs() < 1e-8
    );
    assert!(
        dense
            .observe_time_integral_parameter_jvp(&model, observable, rule, &tangent, direction)
            .is_err()
    );
    assert!(
        sensitive
            .observe_time_integral_parameter_jvp(
                &model,
                observable,
                rule,
                &tangent,
                [(rate, DynQuantity::new(1.0, DimExponents::DIMENSIONLESS))]
            )
            .is_err()
    );
    let resolved = ResolvedCommonPlan::Ode(Box::new(plan));
    let bytes = sparse.to_bytes().unwrap();
    let replayed = CommonTrajectory::from_bytes(&bytes, &resolved).unwrap();
    assert_eq!(replayed, sparse);
    assert_eq!(
        replayed
            .observe_time_integral(&model, observable, rule)
            .unwrap(),
        integrated
    );
    let mut forged: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    forged["payload"]["history"][0]["midpoint_state"][0] = serde_json::json!(8.0);
    assert!(
        CommonTrajectory::from_bytes(&serde_json::to_vec(&forged).unwrap(), &resolved).is_err()
    );
    assert!(
        sparse
            .observe_time_integral(&fixture(3).0, observable, rule)
            .is_err()
    );
    assert!(
        sparse
            .observe_time_integral(&model, Id::new(), rule)
            .is_err()
    );
}

#[test]
fn registered_reset_functional_retains_event_sides_and_ignores_output_cadence() {
    let compiled = eqiora::compiler::compile(
        "reset-functional.eqi",
        r#"
model Reset() {
  state x: 1;
  parameter threshold: 1 = 0.4;
  initial { x = 0; }
  relation flow { derivative(x) = 1[1/s]; }
  event hit = crossing(x-threshold, direction=rising);
  relation reset at hit { next(x) = 0; }
  observable sample: 1 = x;
}
"#,
    )
    .unwrap()
    .pop()
    .unwrap();
    let (transaction, model_id, _) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let kernel = KernelProgram::from_snapshot(&store.snapshot(), model_id).unwrap();
    let model = ModelEnvelope::from_program(&kernel).unwrap();
    let field = kernel
        .nodes()
        .find_map(|node| {
            if let KernelNode::Field(field) = node {
                Some(field.id())
            } else {
                None
            }
        })
        .unwrap();
    let event = kernel
        .nodes()
        .find_map(|node| match node {
            KernelNode::Activation(activation)
                if matches!(activation.kind(), ActivationKind::Event { .. }) =>
            {
                Some(activation.id())
            }
            _ => None,
        })
        .unwrap();
    let observable = kernel
        .nodes()
        .find_map(|node| {
            if let KernelNode::Observable(value) = node {
                Some(value.id())
            } else {
                None
            }
        })
        .unwrap();
    let temporal = CommonTsitouras45::new(
        1e-3,
        1e-11,
        vec![CommonTsitourasTolerance::new(field, 1e-13).unwrap()],
    )
    .unwrap()
    .with_events(
        4,
        vec![(event, DynQuantity::new(1e-9, DimExponents::DIMENSIONLESS))],
    )
    .unwrap();
    let plan = CommonOdePlan::resolve(&model, &kernel, temporal, DIFFSOL_TIME_BACKEND).unwrap();
    let run = |outputs| {
        let request =
            CommonOdeRunRequest::new(plan.clone(), plan.initial_state().unwrap(), 0.7, outputs)
                .unwrap();
        request
            .run_with_events(|problem, roots, plan| {
                DiffsolTimeBackend::new().solve_until_root(problem, roots, plan)
            })
            .unwrap()
    };
    let sparse = run(vec![0.2]);
    let dense = run(vec![0.1, 0.2, 0.3, 0.5, 0.6]);
    assert_eq!(sparse.ode_history(), dense.ode_history());
    let history = sparse.ode_history().unwrap();
    assert_eq!(history.events().len(), 1);
    assert!((history.events()[0].proposal().time() - 0.4).abs() < 1e-10);
    assert_eq!(history.events()[0].after_state(), &[0.0]);
    let integral = sparse
        .observe_time_integral(
            &model,
            observable,
            TimeFunctionalQuadrature::AcceptedStepSimpson,
        )
        .unwrap();
    // Integrate t on [0,p], then t-p on [p,T]: (p^2 + (T-p)^2)/2.
    assert!((integral.value().real_scalar_value().unwrap().value() - 0.125).abs() < 1e-10);
    let terminal = sparse.observe_terminal(&model, observable).unwrap();
    assert!((terminal.value().real_scalar_value().unwrap().value() - 0.3).abs() < 1e-10);
    let resolved = ResolvedCommonPlan::Ode(Box::new(plan.clone()));
    let bytes = sparse.to_bytes().unwrap();
    let replayed = CommonTrajectory::from_bytes(&bytes, &resolved).unwrap();
    assert_eq!(replayed, sparse);
    let mut forged: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    forged["payload"]["events"][0]["after_state"][0] = serde_json::json!(0.1);
    assert!(
        CommonTrajectory::from_bytes(&serde_json::to_vec(&forged).unwrap(), &resolved).is_err()
    );
    let request =
        CommonOdeRunRequest::new(plan.clone(), plan.initial_state().unwrap(), 0.7, vec![0.6])
            .unwrap();
    assert!(
        request.forward_sensitivity_problem().is_err(),
        "smooth entry must not ignore events"
    );
    let (sensitive, tangent) = request
        .run_with_event_forward_sensitivities(|problem, roots, plan| {
            DiffsolTimeBackend::new().solve_until_root_forward_sensitivities(
                problem,
                roots,
                plan,
                &ForwardSensitivityPlan::new(1e-11, vec![1e-13]).unwrap(),
            )
        })
        .unwrap();
    assert_eq!(
        tangent.parameters().len(),
        1,
        "guard-only threshold is an actual derivative coordinate"
    );
    let direction = [(
        tangent.parameters()[0],
        DynQuantity::new(1.0, DimExponents::DIMENSIONLESS),
    )];
    let derivative = sensitive
        .observe_time_integral_parameter_jvp(
            &model,
            observable,
            TimeFunctionalQuadrature::AcceptedStepSimpson,
            &tangent,
            direction,
        )
        .unwrap();
    // dJ/dp = p - (T-p) = 2p-T, combining the smooth post-reset variation and moving boundary.
    assert!((derivative.value().real_scalar_value().unwrap().value() - 0.1).abs() < 1e-9);
    let result = eqiora_numerics::CommonResult::accept_trajectory_with_parameter_sensitivity(
        0.0,
        sensitive.clone(),
        tangent.clone(),
    )
    .unwrap();
    let result_bytes = result.to_bytes().unwrap();
    let restored_result =
        eqiora_numerics::CommonResult::from_bytes(&result_bytes, &resolved).unwrap();
    assert_eq!(restored_result, result);
    let restored_product = restored_result
        .trajectory()
        .unwrap()
        .observe_time_integral_parameter_jvp(
            &model,
            observable,
            TimeFunctionalQuadrature::AcceptedStepSimpson,
            restored_result.parameter_sensitivity().unwrap(),
            direction,
        )
        .unwrap();
    assert_eq!(restored_product, derivative);
    let mut forged_product: serde_json::Value = serde_json::from_slice(&result_bytes).unwrap();
    forged_product["content"]["payload"]["parameter_sensitivity"]["event_time_gradients"][0][0] =
        serde_json::json!(0.0);
    assert!(
        eqiora_numerics::CommonResult::from_bytes(
            &serde_json::to_vec(&forged_product).unwrap(),
            &resolved
        )
        .is_err()
    );
    let terminal_derivative = sensitive
        .observe_terminal_parameter_jvp(&model, observable, &tangent, direction)
        .unwrap();
    assert!(
        (terminal_derivative
            .value()
            .real_scalar_value()
            .unwrap()
            .value()
            + 1.0)
            .abs()
            < 1e-9
    );
    let repeated =
        CommonOdeRunRequest::new(plan.clone(), plan.initial_state().unwrap(), 0.95, vec![0.9])
            .unwrap();
    let (repeated, repeated_tangent) = repeated
        .run_with_event_forward_sensitivities(|problem, roots, plan| {
            DiffsolTimeBackend::new().solve_until_root_forward_sensitivities(
                problem,
                roots,
                plan,
                &ForwardSensitivityPlan::new(1e-11, vec![1e-13]).unwrap(),
            )
        })
        .unwrap();
    assert_eq!(repeated.ode_history().unwrap().events().len(), 2);
    let repeated_derivative = repeated
        .observe_time_integral_parameter_jvp(
            &model,
            observable,
            TimeFunctionalQuadrature::AcceptedStepSimpson,
            &repeated_tangent,
            direction,
        )
        .unwrap();
    // n=2 full ramps plus the final partial ramp: J=n*p^2/2+(T-n*p)^2/2,
    // dJ/dp=n*((n+1)*p-T) = 0.5 for p=.4,T=.95.
    assert!(
        (repeated_derivative
            .value()
            .real_scalar_value()
            .unwrap()
            .value()
            - 0.5)
            .abs()
            < 1e-9
    );
    assert!(
        repeated
            .observe_time_integral_parameter_jvp(
                &model,
                observable,
                TimeFunctionalQuadrature::AcceptedStepSimpson,
                &tangent,
                direction
            )
            .is_err()
    );
    let bounded = CommonOdePlan::resolve(
        &model,
        &kernel,
        plan.temporal()
            .clone()
            .with_events(
                1,
                vec![(event, DynQuantity::new(1e-9, DimExponents::DIMENSIONLESS))],
            )
            .unwrap(),
        DIFFSOL_TIME_BACKEND,
    )
    .unwrap();
    let bounded_request = CommonOdeRunRequest::new(
        bounded.clone(),
        bounded.initial_state().unwrap(),
        0.95,
        vec![0.9],
    )
    .unwrap();
    assert!(
        bounded_request
            .run_with_events(|problem, roots, plan| DiffsolTimeBackend::new()
                .solve_until_root(problem, roots, plan))
            .is_err()
    );
    let endpoint =
        CommonOdeRunRequest::new(plan.clone(), plan.initial_state().unwrap(), 0.4, vec![0.4])
            .unwrap();
    let rejection = endpoint
        .run_with_event_forward_sensitivities(|_, roots, _| {
            use eqiora::time::{
                AcceptedTimeHistory, TimeExecutionReport, TimeHistoryStep, TimeRootOutcome,
                TimeRootSensitivityOutcome,
            };
            let report = TimeExecutionReport::new(
                DIFFSOL_TIME_BACKEND,
                TimeMethod::Tsitouras45,
                TimeEquationClass::ExplicitOde,
                InitialConditionPolicy::Provided,
            );
            let proposal =
                RootProposal::accepted(roots.registration(), 0.4, 0, 1, vec![0.4], 1, report)?;
            let history = AcceptedTimeHistory::accepted(
                1,
                vec![TimeHistoryStep::accepted(
                    0.0,
                    0.4,
                    vec![0.0],
                    vec![0.2],
                    vec![0.4],
                )?],
                vec![],
            )?;
            let tangent = AcceptedTimeHistory::accepted(
                1,
                vec![TimeHistoryStep::accepted(
                    0.0,
                    0.4,
                    vec![0.0],
                    vec![0.0],
                    vec![0.0],
                )?],
                vec![],
            )?;
            TimeRootSensitivityOutcome::accepted(
                TimeRootOutcome::localized(proposal, history, None)?,
                1,
                tangent,
            )
        })
        .unwrap_err();
    assert!(rejection.message().contains("endpoint"));
    let resumed = CommonOdeRunRequest::new(
        plan.clone(),
        sensitive.ode_states().unwrap()[0].clone(),
        0.9,
        vec![0.8],
    )
    .unwrap();
    assert!(
        resumed
            .run_with_event_forward_sensitivities(|_, _, _| panic!(
                "resumed sensitivity must be rejected before backend execution"
            ))
            .is_err()
    );
}
