//! Ordinary native stochastic Plan/Run/Checkpoint/Trajectory/Result lifecycle.
use eqiora::api::ModelDocument;
use eqiora::artifact::ModelEnvelope;
use eqiora_numerics::{
    CommonResult, CommonTimeCheckpoint, CommonTimePlan, CommonTimeRunRequest, CommonTrajectory,
    ResolvedCommonPlan, SamplingIdentity, TimeObservationInterpolation,
};
use std::{collections::HashMap, ops::ControlFlow};

const WIENER: &str = r#"model Process() {
    noise W: wiener(covariance=1);
    state x:1; state y:1;
    initial { x=1; y=-2; }
    law first for x { calculus ito; drift 2[1/s]; diffusion W=1[1/s^(1/2)]; }
    law second for y { calculus ito; drift -4[1/s]; diffusion W=-2[1/s^(1/2)]; }
    observable value:1=x;
}"#;
const POISSON: &str = r#"model Process() {
    noise N: poisson(intensity=8[1/s]);
    state x:1; state y:1;
    initial { x=1; y=0; }
    law first for x { jump N=x; }
    law second for y { jump N=1; }
    observable value:1=x;
}"#;

fn fixture(source: &str) -> (ModelDocument, CommonTimeRunRequest) {
    let document = ModelDocument::compile("process.eqi", source).unwrap();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let plan = if source == WIENER {
        CommonTimePlan::resolve_euler_maruyama(&model, document.program(), 4).unwrap()
    } else {
        CommonTimePlan::resolve_poisson_events(&model, document.program(), 64).unwrap()
    };
    let request = CommonTimeRunRequest::sampled(
        plan.clone(),
        plan.initial_state(2.).unwrap(),
        3.,
        vec![2.25, 2.5, 2.75],
        [31; 32],
        SamplingIdentity::new(b"native", b"path", &[5, 2], None).unwrap(),
    )
    .unwrap();
    (document, request)
}

#[test]
fn native_process_results_survive_recreated_plans_and_repeated_atomic_restart() {
    for source in [WIENER, POISSON] {
        let (document, request) = fixture(source);
        let model = request.plan().model_artifact();
        let resolved = ResolvedCommonPlan::Time(Box::new(request.plan().clone()));
        let plan_bytes = resolved.to_bytes().unwrap();
        let mut final_progress = None;
        let whole = request
            .advance_process(None, |state, accepted_steps| {
                final_progress = Some((state.time_s(), accepted_steps));
                // Even a terminal cancellation must report the completed horizon.
                if state.time_s() == request.until_s() {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            })
            .unwrap()
            .continue_value()
            .unwrap();
        let (final_time, accepted_steps) = final_progress.unwrap();
        assert_eq!(final_time, request.until_s());
        assert!(accepted_steps > 0);
        let mut saved = None;
        let mut next_stop = 1;
        let restarted = loop {
            match request
                .advance_process(saved.as_ref(), |_, count| {
                    if count >= next_stop {
                        ControlFlow::Break(())
                    } else {
                        ControlFlow::Continue(())
                    }
                })
                .unwrap()
            {
                ControlFlow::Continue(trajectory) => break trajectory,
                ControlFlow::Break(checkpoint) => {
                    assert_eq!(checkpoint.accepted_steps(), next_stop);
                    assert!(checkpoint.state().unwrap().time_s() < request.until_s());
                    assert_eq!(checkpoint.state().unwrap().source_kind(), "checkpoint");
                    let plan = ResolvedCommonPlan::from_bytes(
                        &plan_bytes,
                        &eqiora::solver::REFERENCE_LINEAR_SOLVER,
                        &[eqiora::time::ImplicitMidpointTimeBackend::CAPABILITIES],
                    )
                    .unwrap();
                    let bytes = checkpoint.to_bytes().unwrap();
                    let reopened =
                        CommonTimeCheckpoint::from_bytes(&bytes, plan.as_time().unwrap()).unwrap();
                    assert_eq!(reopened, checkpoint);
                    assert_eq!(reopened.to_bytes().unwrap(), bytes);
                    saved = Some(reopened);
                    next_stop += 1;
                }
            }
        };
        assert_eq!(whole, restarted);
        assert_eq!(whole.to_bytes().unwrap(), restarted.to_bytes().unwrap());
        assert!(whole.ode_history().is_none());
        let states = whole.time_states().unwrap();
        assert_eq!(
            states
                .iter()
                .map(|state| state.time_s())
                .collect::<Vec<_>>(),
            request.output_times_s()
        );
        let x = document.aliases()["x"].downcast().unwrap();
        let y = document.aliases()["y"].downcast().unwrap();
        let observable = document.aliases()["value"].downcast().unwrap();
        let result = CommonResult::accept_trajectory(0., whole.clone()).unwrap();
        assert_eq!(result.family_name(), "time");
        for (index, state) in states.iter().enumerate() {
            let x_value = state
                .field_value(model, x, 0)
                .unwrap()
                .real_scalar_value()
                .unwrap()
                .value();
            let y_value = state
                .field_value(model, y, 0)
                .unwrap()
                .real_scalar_value()
                .unwrap()
                .value();
            if source == WIENER {
                // Both authored drift and diffusion give d(y+2x)=0.
                assert_eq!(y_value + 2. * x_value, 0.);
            } else {
                // Each arrival maps x -> 2x, y -> y+1, from (1,0).
                assert_eq!(x_value, 2_f64.powi(y_value as i32));
            }
            assert_eq!(
                result
                    .observe(model, observable, &HashMap::new(), Some(index))
                    .unwrap()
                    .value()
                    .real_scalar_value()
                    .unwrap()
                    .value(),
                x_value
            );
        }
        let terminal = whole.observe_terminal(model, observable).unwrap();
        assert_eq!(terminal.interval_s(), [3., 3.]);
        assert!(
            whole
                .observe_time_integral(
                    model,
                    observable,
                    TimeObservationInterpolation::NativeMidpointQuadratic,
                    None,
                    &HashMap::new()
                )
                .is_err()
        );
        let bytes = whole.to_bytes().unwrap();
        assert_eq!(
            CommonTrajectory::from_bytes(&bytes, &resolved).unwrap(),
            whole
        );
        let result_bytes = result.to_bytes().unwrap();
        assert_eq!(
            CommonResult::from_bytes(&result_bytes, &resolved)
                .unwrap()
                .to_bytes()
                .unwrap(),
            result_bytes
        );
    }
}

#[test]
fn process_resume_rejects_a_different_request_and_noncanonical_checkpoint() {
    let (_, request) = fixture(WIENER);
    let checkpoint = request
        .advance_process(None, |_, n| {
            if n == 2 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        })
        .unwrap()
        .break_value()
        .unwrap();
    let bytes = checkpoint.to_bytes().unwrap();
    let (master, sampling) = request.sampling().unwrap();
    let different = CommonTimeRunRequest::sampled(
        request.plan().clone(),
        request.state().clone(),
        request.until_s(),
        vec![2.5],
        *master,
        sampling.clone(),
    )
    .unwrap();
    assert!(
        different
            .advance_process(Some(&checkpoint), |_, _| ControlFlow::Continue(()))
            .unwrap_err()
            .message()
            .contains("different exact time Run")
    );
    let mut noncanonical = bytes.clone();
    noncanonical.push(b'\n');
    assert!(CommonTimeCheckpoint::from_bytes(&noncanonical, request.plan()).is_err());
    for end in [0, bytes.len() / 2, bytes.len() - 1] {
        assert!(CommonTimeCheckpoint::from_bytes(&bytes[..end], request.plan()).is_err());
    }
    let (_, other) = fixture(POISSON);
    assert!(CommonTimeCheckpoint::from_bytes(&bytes, other.plan()).is_err());
}
