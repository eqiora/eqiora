use super::*;
use eqiora::backends::diffsol::DiffsolTimeBackend;
use eqiora_numerics::{
    CommonOdePlan, CommonOdePolicy, CommonTimeRunRequest, CommonTrajectory, ResolvedCommonPlan,
    TimeObservationInterpolation,
};

fn fixture(rate: u32) -> (ModelEnvelope, KernelProgram) {
    let source = format!(
        r#"
model Decay() {{
  state x: 1;
  initial {{ x = 3; }}
  parameter rate: 1/s = {rate};
  relation flow {{ derivative(x) = -rate*x; }}
  observable position: m = x*1[m];
  observable sample: 1 = position/1[m];
  observable weighted: 1/s = rate*sample;
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
    let temporal = CommonOdePolicy::new(
        eqiora_time::TimeMethod::Tsitouras45,
        1e-3,
        1e-11,
        vec![(
            eqiora_core::TimeStateCoordinate::new(field, 0, 0, false),
            1e-13,
        )],
    )
    .unwrap();
    let plan = CommonOdePlan::resolve(&model, &kernel, temporal, DiffsolTimeBackend::CAPABILITIES)
        .unwrap();
    assert_eq!(
        plan.state_coordinates().len(),
        1,
        "functional adds no ODE state"
    );
    let run = |output_times| {
        let request = CommonTimeRunRequest::new(
            plan.clone(),
            plan.initial_state(0.0).unwrap(),
            1.0,
            output_times,
        )
        .unwrap();
        let solution = DiffsolTimeBackend::new()
            .solve(&request.problem().unwrap(), request.time_plan().unwrap())
            .unwrap();
        CommonTrajectory::accept_ode(request, solution).unwrap()
    };
    let sparse = run(vec![0.25]);
    let dense = run((1..10).map(|index| f64::from(index) / 10.0).collect());
    assert_eq!(sparse.time_states().unwrap().len(), 1);
    assert_eq!(
        sparse.ode_history(),
        dense.ode_history(),
        "requested samples do not choose accepted integration steps"
    );
    let rule = TimeObservationInterpolation::NativeMidpointQuadratic;
    let integrated = sparse
        .observe_time_integral(&model, observable, rule, None, &Default::default())
        .unwrap();
    let dense_integrated = dense
        .observe_time_integral(&model, observable, rule, None, &Default::default())
        .unwrap();
    assert_eq!(integrated.value(), dense_integrated.value());
    assert_eq!(integrated.trajectory_identity(), sparse.identity());
    assert_eq!(integrated.interpolation(), Some(rule));
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
    assert_eq!(terminal.interpolation(), None);
    assert_eq!(terminal.interval_s(), [1.0, 1.0]);
    assert_eq!(terminal.endpoint_convention(), "terminal-after-events");
    let request = CommonTimeRunRequest::new(
        plan.clone(),
        plan.initial_state(0.0).unwrap(),
        1.0,
        vec![0.25],
    )
    .unwrap();
    let solution = DiffsolTimeBackend::new()
        .solve_forward_sensitivities(
            &request.forward_sensitivity_problem().unwrap(),
            request.time_plan().unwrap(),
            &ForwardSensitivityPlan::new(1e-11, vec![1e-13]).unwrap(),
        )
        .unwrap();
    let (sensitive, tangent) =
        CommonTrajectory::accept_ode_forward_sensitivities(request, solution).unwrap();
    let rate = plan.parameter_ids()[0];
    let rate_dimension = DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).unwrap();
    let direction = [(rate, DynQuantity::new(1.0, rate_dimension))];
    let derivative = sensitive
        .observe_time_integral_parameter_jvp(&model, observable, rule, &tangent, direction, None)
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
        .observe_time_integral_parameter_jvp(&model, weighted, rule, &tangent, direction, None)
        .unwrap();
    assert!(
        (direct.value().real_scalar_value().unwrap().value() - 3.0 * (-2.0_f64).exp()).abs() < 1e-8
    );
    assert!(
        dense
            .observe_time_integral_parameter_jvp(
                &model, observable, rule, &tangent, direction, None
            )
            .is_err()
    );
    assert!(
        sensitive
            .observe_time_integral_parameter_jvp(
                &model,
                observable,
                rule,
                &tangent,
                [(rate, DynQuantity::new(1.0, DimExponents::DIMENSIONLESS))],
                None
            )
            .is_err()
    );
    let resolved = ResolvedCommonPlan::Time(Box::new(plan.into()));
    let result = eqiora_numerics::CommonResult::accept_trajectory_with_parameter_sensitivity(
        0.,
        sensitive.clone(),
        tangent.clone(),
    )
    .unwrap();
    let result_bytes = result.to_bytes().unwrap();
    let replay = eqiora_numerics::CommonResult::from_bytes(&result_bytes, &resolved).unwrap();
    for candidate in [&result, &replay] {
        let state = &candidate.trajectory().unwrap().time_states().unwrap()[0];
        for (output, expected) in [
            (observable, -0.75 * (-0.5_f64).exp()),
            (weighted, 1.5 * (-0.5_f64).exp()),
        ] {
            let product = candidate
                .observe_parameter_jvp(&model, output, 0, direction)
                .unwrap();
            assert!((product.value().real_scalar_value().unwrap().value() - expected).abs() < 1e-9);
            assert_eq!(product.output(), Some((0, 0.25, state.identity())));
            assert_eq!(product.result_identity(), candidate.identity());
        }
        assert!(
            candidate
                .observe_parameter_jvp(&model, observable, 1, direction)
                .is_err()
        );
        assert!(
            candidate
                .observe_parameter_jvp(&fixture(3).0, observable, 0, direction)
                .is_err()
        );
        assert!(
            candidate
                .observe_parameter_jvp(
                    &model,
                    observable,
                    0,
                    [(rate, DynQuantity::new(1., DimExponents::DIMENSIONLESS))]
                )
                .is_err()
        );
        assert_eq!(candidate.to_bytes().unwrap(), result_bytes);
    }
    let bytes = sparse.to_bytes().unwrap();
    let replayed = CommonTrajectory::from_bytes(&bytes, &resolved).unwrap();
    assert_eq!(replayed, sparse);
    assert_eq!(
        replayed
            .observe_time_integral(&model, observable, rule, None, &Default::default())
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
            .observe_time_integral(&fixture(3).0, observable, rule, None, &Default::default())
            .is_err()
    );
    assert!(
        sparse
            .observe_time_integral(&model, Id::new(), rule, None, &Default::default())
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
    let temporal = CommonOdePolicy::new(
        eqiora_time::TimeMethod::Tsitouras45,
        1e-3,
        1e-11,
        vec![(
            eqiora_core::TimeStateCoordinate::new(field, 0, 0, false),
            1e-13,
        )],
    )
    .unwrap()
    .with_events(
        4,
        vec![(event, DynQuantity::new(1e-9, DimExponents::DIMENSIONLESS))],
    )
    .unwrap();
    let plan = CommonOdePlan::resolve(&model, &kernel, temporal, DiffsolTimeBackend::CAPABILITIES)
        .unwrap();
    let run = |outputs| {
        let request =
            CommonTimeRunRequest::new(plan.clone(), plan.initial_state(0.0).unwrap(), 0.7, outputs)
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
    // A closed window includes both event sides even when the event is exactly
    // an endpoint. The pre-reset maximum is .4, the post-reset minimum is zero.
    let event_time = history.events()[0].proposal().time();
    for rule in [
        TimeObservationInterpolation::EndpointLinear,
        TimeObservationInterpolation::NativeMidpointQuadratic,
    ] {
        for window in [[0.1, event_time], [event_time, 0.6]] {
            let min = sparse
                .observe_time_minimum(&model, observable, rule, Some(window), &Default::default())
                .unwrap();
            let max = sparse
                .observe_time_maximum(&model, observable, rule, Some(window), &Default::default())
                .unwrap();
            assert_eq!(min.value().real_scalar_value().unwrap().value(), 0.);
            assert!((max.value().real_scalar_value().unwrap().value() - 0.4).abs() < 1e-10);
            assert_eq!(min.extremum_time_s(), Some(event_time));
            assert_eq!(max.extremum_time_s(), Some(event_time));
        }
    }

    let integral = sparse
        .observe_time_integral(
            &model,
            observable,
            TimeObservationInterpolation::NativeMidpointQuadratic,
            None,
            &Default::default(),
        )
        .unwrap();
    // Integrate t on [0,p], then t-p on [p,T]: (p^2 + (T-p)^2)/2.
    assert!((integral.value().real_scalar_value().unwrap().value() - 0.125).abs() < 1e-10);
    let terminal = sparse.observe_terminal(&model, observable).unwrap();
    assert!((terminal.value().real_scalar_value().unwrap().value() - 0.3).abs() < 1e-10);
    let resolved = ResolvedCommonPlan::Time(Box::new((plan.clone()).into()));
    let bytes = sparse.to_bytes().unwrap();
    let replayed = CommonTrajectory::from_bytes(&bytes, &resolved).unwrap();
    assert_eq!(replayed, sparse);
    let mut forged: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    forged["payload"]["events"][0]["after_state"][0] = serde_json::json!(0.1);
    assert!(
        CommonTrajectory::from_bytes(&serde_json::to_vec(&forged).unwrap(), &resolved).is_err()
    );
    let request = CommonTimeRunRequest::new(
        plan.clone(),
        plan.initial_state(0.0).unwrap(),
        0.7,
        vec![0.6],
    )
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
            TimeObservationInterpolation::NativeMidpointQuadratic,
            &tangent,
            direction,
            None,
        )
        .unwrap();
    // dJ/dp = p - (T-p) = 2p-T, combining the smooth post-reset variation and moving boundary.
    assert!((derivative.value().real_scalar_value().unwrap().value() - 0.1).abs() < 1e-9);
    // x=t before p=.4, x=t-p after it. Only a window containing the
    // moving jump receives its boundary contribution to the Parameter product.
    for (interval, integral, derivative) in [
        ([0.1, 0.3], 0.04, 0.0),
        ([0.1, 0.6], 0.095, 0.2),
        ([0.5, 0.6], 0.015, -0.1),
    ] {
        let rule = TimeObservationInterpolation::NativeMidpointQuadratic;
        let value = sensitive
            .observe_time_integral(
                &model,
                observable,
                rule,
                Some(interval),
                &Default::default(),
            )
            .unwrap();
        let product = sensitive
            .observe_time_integral_parameter_jvp(
                &model,
                observable,
                rule,
                &tangent,
                direction,
                Some(interval),
            )
            .unwrap();
        assert!((value.value().real_scalar_value().unwrap().value() - integral).abs() < 1e-9);
        assert!((product.value().real_scalar_value().unwrap().value() - derivative).abs() < 1e-9);
    }
    let event_time = sensitive.ode_history().unwrap().events()[0]
        .proposal()
        .time();
    for interval in [[0.1, event_time], [event_time, 0.6]] {
        let error = sensitive
            .observe_time_integral_parameter_jvp(
                &model,
                observable,
                TimeObservationInterpolation::NativeMidpointQuadratic,
                &tangent,
                direction,
                Some(interval),
            )
            .unwrap_err();
        assert!(
            error
                .message()
                .contains("moving event meets a fixed interval endpoint")
        );
    }
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
            TimeObservationInterpolation::NativeMidpointQuadratic,
            restored_result.parameter_sensitivity().unwrap(),
            direction,
            None,
        )
        .unwrap();
    assert_eq!(restored_product, derivative);
    for candidate in [&result, &restored_result] {
        let product = candidate
            .observe_parameter_jvp(&model, observable, 0, direction)
            .unwrap();
        assert!((product.value().real_scalar_value().unwrap().value() + 1.).abs() < 1e-9);
        assert_eq!(product.output().unwrap().1, 0.6);
    }
    // Sampling does not change native event localization. At the exact nominal
    // event time a moving jump has no two-sided fixed-time derivative.
    let event_request = CommonTimeRunRequest::new(
        plan.clone(),
        plan.initial_state(0.).unwrap(),
        0.7,
        vec![event_time],
    )
    .unwrap();
    let (event_trajectory, event_products) = event_request
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
        event_trajectory.ode_history().unwrap().events()[0]
            .proposal()
            .time(),
        event_time
    );
    let event_result = eqiora_numerics::CommonResult::accept_trajectory_with_parameter_sensitivity(
        0.,
        event_trajectory,
        event_products,
    )
    .unwrap();
    assert!(
        event_result
            .observe_parameter_jvp(&model, observable, 0, direction)
            .unwrap_err()
            .message()
            .contains("stable observation side")
    );
    assert_eq!(
        event_result
            .observe_parameter_jvp(&model, observable, 0, [])
            .unwrap()
            .value()
            .real_scalar_value()
            .unwrap()
            .value(),
        0.
    );
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
    let repeated = CommonTimeRunRequest::new(
        plan.clone(),
        plan.initial_state(0.0).unwrap(),
        0.95,
        vec![0.9],
    )
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
            TimeObservationInterpolation::NativeMidpointQuadratic,
            &repeated_tangent,
            direction,
            None,
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
                TimeObservationInterpolation::NativeMidpointQuadratic,
                &tangent,
                direction,
                None
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
        DiffsolTimeBackend::CAPABILITIES,
    )
    .unwrap();
    let bounded_request = CommonTimeRunRequest::new(
        bounded.clone(),
        bounded.initial_state(0.0).unwrap(),
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
    let endpoint = CommonTimeRunRequest::new(
        plan.clone(),
        plan.initial_state(0.0).unwrap(),
        0.4,
        vec![0.4],
    )
    .unwrap();
    // Feed an independently authored exact ramp to the ordinary event acceptance
    // path. The terminal reset side has no following step to expose it.
    let terminal_event = endpoint
        .clone()
        .run_with_events(|_, roots, _| {
            use eqiora::time::{
                AcceptedTimeHistory, TimeExecutionReport, TimeHistoryStep, TimeRootOutcome,
            };
            let report = TimeExecutionReport::new(
                DiffsolTimeBackend::CAPABILITIES.identity(),
                TimeMethod::Tsitouras45,
                TimeEquationClass::ExplicitOde,
                InitialConditionPolicy::Provided,
            );
            let proposal =
                RootProposal::accepted(roots.registration(), 0.4, 0, 1, vec![0.4], 1, report)?;
            let history = AcceptedTimeHistory::accepted(
                1,
                vec![TimeHistoryStep::accepted(
                    0.,
                    0.4,
                    vec![0.],
                    vec![0.2],
                    vec![0.4],
                    vec![0., 0.2, 0.4],
                    vec![vec![0.], vec![0.2], vec![0.4]],
                )?],
                vec![],
            )?;
            TimeRootOutcome::localized(proposal, history, None)
        })
        .unwrap();
    for rule in [
        TimeObservationInterpolation::EndpointLinear,
        TimeObservationInterpolation::NativeMidpointQuadratic,
    ] {
        let minimum = terminal_event
            .observe_time_minimum(
                &model,
                observable,
                rule,
                Some([0.1, 0.4]),
                &Default::default(),
            )
            .unwrap();
        assert_eq!(minimum.value().real_scalar_value().unwrap().value(), 0.);
        assert_eq!(minimum.extremum_time_s(), Some(0.4));
    }
    let rejection = endpoint
        .run_with_event_forward_sensitivities(|_, roots, _| {
            use eqiora::time::{
                AcceptedTimeHistory, TimeExecutionReport, TimeHistoryStep, TimeRootOutcome,
            };
            let report = TimeExecutionReport::new(
                DiffsolTimeBackend::CAPABILITIES.identity(),
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
                    vec![0.0, 0.2, 0.4],
                    vec![vec![0.0], vec![0.2], vec![0.4]],
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
                    vec![0.0, 0.2, 0.4],
                    vec![vec![0.0], vec![0.0], vec![0.0]],
                )?],
                vec![],
            )?;
            TimeRootOutcome::localized(proposal, history, None)?.with_sensitivities(
                1,
                tangent,
                vec![],
            )
        })
        .unwrap_err();
    assert!(rejection.message().contains("endpoint"));
    let resumed = CommonTimeRunRequest::new(
        plan.clone(),
        sensitive.time_states().unwrap()[0].clone(),
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
