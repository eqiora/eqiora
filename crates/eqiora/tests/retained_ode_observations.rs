//! Requested ODE output occurrences use the shared typed observation owner.
#![cfg(feature = "diffsol")]

use eqiora::DimExponents;
use eqiora::api::ModelDocument;
use eqiora::artifact::ModelEnvelope;
use eqiora::backends::diffsol::DiffsolTimeBackend;
use eqiora::runtime::FirstOrderProgram;
use eqiora::time::{ImplicitMidpointTimeBackend, TimeMethod};
use eqiora_numerics::{
    CommonOdePlan, CommonOdePolicy, CommonResult, CommonTimeRunRequest, CommonTrajectory,
};

#[test]
fn retained_outputs_preserve_time_type_lineage_and_replay_on_both_backends() {
    let document = ModelDocument::compile(
        "outputs.eqi",
        r#"model Ramp() {
        state x:m; initial {x=0[m];}
        relation flow {derivative(x)=2[m/s];}
        observable square:m^2=x*x;
        observable clock:s=time();
        observable speed:m/s=derivative(x);
    }"#,
    )
    .unwrap();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let flow = FirstOrderProgram::lower(
        document.program(),
        document.aliases()["flow"].downcast().unwrap(),
    )
    .unwrap();
    let square = document.aliases()["square"].downcast().unwrap();
    let clock = document.aliases()["clock"].downcast().unwrap();
    let speed = document.aliases()["speed"].downcast().unwrap();
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
        let times = vec![0.125, 0.3, 0.875];
        let request = CommonTimeRunRequest::new(
            plan.clone(),
            plan.initial_state(0.).unwrap(),
            1.,
            times.clone(),
        )
        .unwrap();
        let problem = request.problem().unwrap();
        let solution = if method == TimeMethod::ImplicitMidpoint {
            ImplicitMidpointTimeBackend::new().solve(&problem, request.time_plan().unwrap())
        } else {
            DiffsolTimeBackend::new().solve(&problem, request.time_plan().unwrap())
        }
        .unwrap();
        let trajectory = CommonTrajectory::accept_ode(request, solution).unwrap();
        let result = CommonResult::accept_trajectory(0., trajectory).unwrap();
        let bytes = result.to_bytes().unwrap();
        let replay = CommonResult::from_bytes(&bytes, result.plan()).unwrap();
        for candidate in [&result, &replay] {
            for (index, &time) in times.iter().enumerate() {
                let state = &candidate.trajectory().unwrap().time_states().unwrap()[index];
                for (observable, expected, dimension) in [
                    (square, 4. * time * time, [0, 2, 0, 0, 0, 0, 0]),
                    (clock, time, [0, 0, 1, 0, 0, 0, 0]),
                    (speed, 2., [0, 1, -1, 0, 0, 0, 0]),
                ] {
                    let observed = candidate
                        .observe(&model, observable, &Default::default(), Some(index))
                        .unwrap();
                    let value = observed.value().real_scalar_value().unwrap();
                    assert!((value.value() - expected).abs() < 2e-12);
                    assert_eq!(value.dim(), DimExponents::from_integers(dimension).unwrap());
                    assert_eq!(observed.output(), Some((index, time, state.identity())));
                    assert_eq!(observed.result_identity(), candidate.identity());
                }
            }
            for index in [None, Some(times.len())] {
                assert!(
                    candidate
                        .observe(&model, square, &Default::default(), index)
                        .is_err()
                );
            }
            let rules = std::collections::HashMap::from([(
                eqiora::Id::new(),
                eqiora::meshing::QuadratureRule::point(),
            )]);
            assert!(candidate.observe(&model, square, &rules, Some(0)).is_err());
            assert_eq!(candidate.to_bytes().unwrap(), bytes);
            // The last retained time is 7/8, while the accepted terminal time is 1.
            let terminal = candidate
                .trajectory()
                .unwrap()
                .observe_terminal(&model, square)
                .unwrap();
            assert!((terminal.value().real_scalar_value().unwrap().value() - 4.).abs() < 2e-12);
        }
    }
}
