//! Fixed-window quadrature has analytic meaning independent of retained outputs.
#![cfg(feature = "diffsol")]

use eqiora::api::ModelDocument;
use eqiora::artifact::ModelEnvelope;
use eqiora::backends::diffsol::DiffsolTimeBackend;
use eqiora::runtime::FirstOrderProgram;
use eqiora::time::{ForwardSensitivityPlan, ImplicitMidpointTimeBackend, TimeMethod};
use eqiora::{DimExponents, DynQuantity};
use eqiora_numerics::{
    CommonOdePlan, CommonOdePolicy, CommonTimeRunRequest, CommonTrajectory, ResolvedCommonPlan,
    TimeObservationInterpolation,
};

const SOURCE: &str = r#"model Ramp() {
    state x:m;
    initial { x=0[m]; }
    parameter rate:m/s=2;
    relation flow { derivative(x)=rate; }
    observable square:m^2=x*x;
    observable bowl:m^2=(x-0.75[m])*(x-0.75[m]);
    observable plateau:m=7[m];
}"#;

#[test]
fn clipped_quadrature_is_exact_for_a_quadratic_on_two_execution_backends() {
    let document = ModelDocument::compile("window-ramp.eqi", SOURCE).unwrap();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let flow = FirstOrderProgram::lower(
        document.program(),
        document.aliases()["flow"].downcast().unwrap(),
    )
    .unwrap();
    let observable = document.aliases()["square"].downcast().unwrap();
    let rule = TimeObservationInterpolation::NativeMidpointQuadratic;
    for method in [TimeMethod::ImplicitMidpoint, TimeMethod::Tsitouras45] {
        let capabilities = if method == TimeMethod::ImplicitMidpoint {
            ImplicitMidpointTimeBackend::CAPABILITIES
        } else {
            DiffsolTimeBackend::CAPABILITIES
        };
        let temporal = CommonOdePolicy::new(
            method,
            0.25,
            1e-11,
            flow.state_coordinates()
                .iter()
                .map(|&coordinate| (coordinate, 1e-13))
                .collect(),
        )
        .unwrap();
        let plan =
            CommonOdePlan::resolve(&model, document.program(), temporal, capabilities).unwrap();
        let request = |outputs| {
            CommonTimeRunRequest::new(plan.clone(), plan.initial_state(0.).unwrap(), 1., outputs)
                .unwrap()
        };
        let run = |outputs| {
            let request = request(outputs);
            let problem = request.problem().unwrap();
            let solution = if method == TimeMethod::ImplicitMidpoint {
                ImplicitMidpointTimeBackend::new().solve(&problem, request.time_plan().unwrap())
            } else {
                DiffsolTimeBackend::new().solve(&problem, request.time_plan().unwrap())
            }
            .unwrap();
            CommonTrajectory::accept_ode(request, solution).unwrap()
        };
        let sparse = run(vec![0.25]);
        let dense = run(vec![0.125, 0.25, 0.5, 0.875, 1.]);
        assert_eq!(sparse.ode_history(), dense.ode_history());
        let before = sparse.clone();
        let bytes = sparse.to_bytes().unwrap();
        let replay = CommonTrajectory::from_bytes(
            &bytes,
            &ResolvedCommonPlan::Time(Box::new((plan.clone()).into())),
        )
        .unwrap();
        assert_eq!(replay.to_bytes().unwrap(), bytes);
        // x=2t is represented exactly by both methods; its square is quadratic.
        // The integral is 4*(b^3-a^3)/3, with 2e-12 m^2*s binary64 allowance
        // fixed before execution. The principal window cuts accepted steps.
        for interval in [[0.125_f64, 0.875], [0., 0.125], [0.875, 1.], [0., 1.]] {
            let expected = 4. * (interval[1].powi(3) - interval[0].powi(3)) / 3.;
            let value = sparse
                .observe_time_integral(
                    &model,
                    observable,
                    rule,
                    Some(interval),
                    &Default::default(),
                )
                .unwrap();
            let other = dense
                .observe_time_integral(
                    &model,
                    observable,
                    rule,
                    Some(interval),
                    &Default::default(),
                )
                .unwrap();
            assert_eq!(value.value(), other.value());
            assert_eq!(
                value.value(),
                replay
                    .observe_time_integral(
                        &model,
                        observable,
                        rule,
                        Some(interval),
                        &Default::default()
                    )
                    .unwrap()
                    .value()
            );
            assert_eq!(value.interval_s(), interval);
            assert_eq!(value.trajectory_identity(), sparse.identity());
            assert!((value.value().real_scalar_value().unwrap().value() - expected).abs() < 2e-12);
            assert_eq!(
                value.value().value_type().dimension(),
                DimExponents::from_integers([0, 2, 1, 0, 0, 0, 0]).unwrap()
            );
        }
        if method == TimeMethod::ImplicitMidpoint {
            // x=2t, h=1/4. The endpoint interpolant integrates x^2 to
            // four times (1/3+h^2/6), and its clipped integral to 4*59/256.
            for (window, expected) in [(None, 11. / 8.), (Some([0.125, 0.875]), 59. / 64.)] {
                let value = sparse
                    .observe_time_integral(
                        &model,
                        observable,
                        TimeObservationInterpolation::EndpointLinear,
                        window,
                        &Default::default(),
                    )
                    .unwrap();
                assert!(
                    (value.value().real_scalar_value().unwrap().value() - expected).abs() < 2e-12
                );
            }
        }
        // x=2t, so the bowl has minimum zero at t=3/8 and maximum
        // (2-3/4)^2=25/16 at t=1. Endpoint-linear interpolation at h=1/4
        // instead has equal minima 1/16 at t=1/4 and 1/2: choose 1/4.
        let bowl = document.aliases()["bowl"].downcast().unwrap();
        for trajectory in [&sparse, &dense, &replay] {
            let minimum = trajectory
                .observe_time_minimum(&model, bowl, rule, None, &Default::default())
                .unwrap();
            assert!(minimum.value().real_scalar_value().unwrap().value().abs() < 2e-12);
            assert!((minimum.extremum_time_s().unwrap() - 0.375).abs() < 2e-12);
            assert_eq!(minimum.evaluation_kind(), "time-minimum");
            assert_eq!(
                minimum.value().value_type().dimension(),
                DimExponents::from_integers([0, 2, 0, 0, 0, 0, 0]).unwrap()
            );
            let maximum = trajectory
                .observe_time_maximum(&model, bowl, rule, None, &Default::default())
                .unwrap();
            assert!(
                (maximum.value().real_scalar_value().unwrap().value() - 25. / 16.).abs() < 2e-12
            );
            assert_eq!(maximum.extremum_time_s(), Some(1.));
            let plateau = trajectory
                .observe_time_minimum(
                    &model,
                    document.aliases()["plateau"].downcast().unwrap(),
                    rule,
                    Some([0.13, 0.87]),
                    &Default::default(),
                )
                .unwrap();
            assert_eq!(plateau.value().real_scalar_value().unwrap().value(), 7.);
            assert_eq!(plateau.extremum_time_s(), Some(0.13));
            if method == TimeMethod::ImplicitMidpoint {
                let linear = trajectory
                    .observe_time_minimum(
                        &model,
                        bowl,
                        TimeObservationInterpolation::EndpointLinear,
                        None,
                        &Default::default(),
                    )
                    .unwrap();
                assert_eq!(
                    linear.value().real_scalar_value().unwrap().value(),
                    1. / 16.
                );
                assert_eq!(linear.extremum_time_s(), Some(0.25));
            }
        }
        for interval in [
            [-0.1, 0.5],
            [0.5, 1.1],
            [0.5, 0.5],
            [0.7, 0.2],
            [f64::NAN, 1.],
            [0., f64::INFINITY],
        ] {
            assert!(
                sparse
                    .observe_time_integral(
                        &model,
                        observable,
                        rule,
                        Some(interval),
                        &Default::default()
                    )
                    .is_err()
            );
        }
        assert_eq!(
            sparse, before,
            "queries must preserve accepted history and restart states"
        );
        let foreign =
            ModelDocument::compile("foreign-window.eqi", &SOURCE.replace("m/s=2", "m/s=3"))
                .unwrap();
        assert!(
            sparse
                .observe_time_integral(
                    &ModelEnvelope::from_program(foreign.program()).unwrap(),
                    observable,
                    rule,
                    Some([0.125, 0.875]),
                    &Default::default()
                )
                .is_err()
        );
        if method == TimeMethod::Tsitouras45 {
            let request = request(vec![0.25]);
            let solution = DiffsolTimeBackend::new()
                .solve_forward_sensitivities(
                    &request.forward_sensitivity_problem().unwrap(),
                    request.time_plan().unwrap(),
                    &ForwardSensitivityPlan::new(1e-11, vec![1e-13]).unwrap(),
                )
                .unwrap();
            let (trajectory, sensitivity) =
                CommonTrajectory::accept_ode_forward_sensitivities(request, solution).unwrap();
            let value = trajectory
                .observe_time_integral_parameter_jvp(
                    &model,
                    observable,
                    rule,
                    &sensitivity,
                    [(
                        plan.parameter_ids()[0],
                        DynQuantity::new(
                            1.,
                            DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).unwrap(),
                        ),
                    )],
                    Some([0.125, 0.875]),
                )
                .unwrap();
            // d[r^2*(b^3-a^3)/3]/dr at r=2, in direction 1 m/s.
            assert!((value.value().real_scalar_value().unwrap().value() - 0.890625).abs() < 2e-12);
            assert_eq!(value.interval_s(), [0.125, 0.875]);
        }
    }
}
