use super::super::super::without_draws;
use super::super::tests::{POISSON, WIENER, request};
use super::*;
use crate::{CommonTrajectory, ResolvedCommonPlan};
use artifact::WireProcessHistory;

fn complete(run: &Arc<ProcessRun>) -> ProcessCursor {
    let mut cursor = run.start();
    // Coarse partitions include several arrivals per Poisson step, exercising
    // stored intermediate predecessors rather than only one jump per step.
    for elapsed in [0.25, 0.5, 0.75, 1.] {
        cursor
            .accept(cursor.trial(&exact(elapsed)).unwrap())
            .unwrap();
    }
    cursor
}

#[test]
fn process_trajectory_reopens_stored_stencils_without_drawing_or_restarting() {
    for (source, method) in [
        (WIENER, ProcessMethod::EulerMaruyama { depth: 8 }),
        (POISSON, ProcessMethod::PoissonEvents { max_events: 64 }),
    ] {
        let run = request(source, method);
        let cursor = complete(&run);
        let history = ProcessHistory::from_cursor(cursor.clone()).unwrap();
        let trajectory = CommonTrajectory::accept_process(history).unwrap();
        let bytes = trajectory.to_bytes().unwrap();
        let plan = ResolvedCommonPlan::Time(Box::new(run.request.plan().clone()));
        let reopened = without_draws(|| CommonTrajectory::from_bytes(&bytes, &plan)).unwrap();
        assert_eq!(reopened, trajectory);
        assert_eq!(reopened.to_bytes().unwrap(), bytes);
        assert_eq!(reopened.time_states().unwrap(), cursor.outputs);
        let result = crate::CommonResult::accept_trajectory(0., trajectory).unwrap();
        let bytes = result.to_bytes().unwrap();
        let restored = without_draws(|| crate::CommonResult::from_bytes(&bytes, &plan)).unwrap();
        assert_eq!(restored.to_bytes().unwrap(), bytes);

        // The same guard proves this is a draw falsifier, not merely a label:
        // the existing strict continuation path really does attempt a draw.
        let checkpoint = cursor.checkpoint_bytes().unwrap();
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                without_draws(|| {
                    ProcessCursor::from_bytes(&checkpoint, run.request.plan()).unwrap()
                })
            }))
            .is_err()
        );
        assert_eq!(
            ProcessCursor::from_bytes(&checkpoint, run.request.plan()).unwrap(),
            cursor
        );
    }
}

#[test]
fn process_history_rejects_reidentified_state_channel_partition_and_completion_mutants() {
    for (source, method) in [
        (WIENER, ProcessMethod::EulerMaruyama { depth: 8 }),
        (POISSON, ProcessMethod::PoissonEvents { max_events: 64 }),
    ] {
        let run = request(source, method);
        let cursor = complete(&run);
        let original = ProcessHistory::from_cursor(cursor.clone()).unwrap();
        let reject = |history: &ProcessHistory, message: &str| {
            // Capture and replay a new canonical payload; stale hashes never
            // substitute for original-equation and membership admission.
            let wire = WireProcessHistory::capture(history).unwrap();
            let error = without_draws(|| wire.replay(run.request.plan())).unwrap_err();
            assert!(error.message().contains(message), "{error:?}");
        };
        let mut changed = original.clone();
        changed.steps[0].state = run
            .program()
            .accept_state(0.25, vec![123.], "result")
            .unwrap();
        reject(&changed, "original-equation stencil");
        let mut changed = original.clone();
        changed.steps.insert(1, changed.steps[0].clone());
        reject(&changed, "advance its accepted position");
        let mut changed = original.clone();
        changed.steps[0].elapsed = exact(0.125);
        reject(&changed, "physical time");
        let mut changed = original.clone();
        changed.steps.pop();
        let wire = WireProcessHistory::capture(&changed).unwrap();
        let partial = without_draws(|| wire.replay(run.request.plan())).unwrap();
        assert!(
            CommonTrajectory::accept_process(partial)
                .unwrap_err()
                .message()
                .contains("complete original horizon")
        );
        let mut changed = original.clone();
        match &mut changed.steps[0].inputs {
            StepInputs::Wiener(increments) => {
                increments[0].values[0] += exact(1.);
                reject(&changed, "original-equation stencil");
                let mut changed = original.clone();
                let StepInputs::Wiener(increments) = &mut changed.steps[0].inputs else {
                    unreachable!()
                };
                increments.push(increments[0].clone());
                reject(&changed, "foreign Wiener increment group");
                let mut changed = original.clone();
                let StepInputs::Wiener(increments) = &mut changed.steps[0].inputs else {
                    unreachable!()
                };
                increments.clear();
                reject(&changed, "omits a declared noise group");
            }
            StepInputs::Poisson(stencils) => {
                // Force the malformed stencil independently of how many arrivals
                // this particular sampled path happened to contain.
                let event = JumpEvent {
                    elapsed: exact(0.125),
                    group: run.program().noises[0].id,
                    component: 0,
                };
                *stencils = vec![JumpStencil {
                    events: vec![event.clone(), event.clone()],
                    state: run
                        .program()
                        .accept_state(0.125, vec![2.], "result")
                        .unwrap(),
                }];
                reject(&changed, "duplicated, unordered or foreign channels");
                let mut changed = original.clone();
                changed.steps[0].inputs = StepInputs::Poisson(vec![JumpStencil {
                    events: vec![event],
                    state: run
                        .program()
                        .accept_state(0.125, vec![123.], "result")
                        .unwrap(),
                }]);
                reject(&changed, "original-equation stencil");
            }
        }
    }
}

#[test]
fn stored_driving_inputs_still_obey_the_model_covariance_and_zero_rate() {
    let source = r#"model Process() {
        noise W: wiener(covariance=[[1,-1],[-1,1]]);
        state x: 1;
        initial { x=0; }
        law dynamics for x {
            calculus ito;
            drift 0[1/s];
            diffusion W=[1[1/s^(1/2)],1[1/s^(1/2)]];
        }
    }"#;
    let run = request(source, ProcessMethod::EulerMaruyama { depth: 8 });
    let mut history = ProcessHistory::from_cursor(complete(&run)).unwrap();
    assert_eq!(history.state().values(), [0.]);
    // For covariance [[1,-1],[-1,1]], dW1+dW2 is exactly zero,
    // regardless of the particular recorded independent driving coordinates.
    let StepInputs::Wiener(inputs) = &mut history.steps[0].inputs else {
        unreachable!()
    };
    inputs[0].values = vec![exact(3.), exact(7.)];
    without_draws(|| {
        WireProcessHistory::capture(&history)
            .unwrap()
            .replay(run.request.plan())
    })
    .unwrap();
    history.steps[0].state = run
        .program()
        .accept_state(0.25, vec![1.], "result")
        .unwrap();
    assert!(
        WireProcessHistory::capture(&history)
            .unwrap()
            .replay(run.request.plan())
            .unwrap_err()
            .message()
            .contains("original-equation stencil")
    );

    let source = POISSON.replace("8[1/s]", "0[1/s]");
    let run = request(&source, ProcessMethod::PoissonEvents { max_events: 64 });
    let mut history = ProcessHistory::from_cursor(complete(&run)).unwrap();
    let StepInputs::Poisson(stencils) = &mut history.steps[0].inputs else {
        unreachable!()
    };
    assert!(stencils.is_empty());
    stencils.push(JumpStencil {
        events: vec![JumpEvent {
            elapsed: exact(0.125),
            group: run.program().noises[0].id,
            component: 0,
        }],
        state: run
            .program()
            .accept_state(0.125, vec![2.], "result")
            .unwrap(),
    });
    assert!(
        without_draws(|| WireProcessHistory::capture(&history)
            .unwrap()
            .replay(run.request.plan()))
        .unwrap_err()
        .message()
        .contains("foreign channels")
    );
}

#[test]
fn process_history_preserves_and_validates_the_original_execution_provenance() {
    let run = request(WIENER, ProcessMethod::EulerMaruyama { depth: 8 });
    let history = ProcessHistory::from_cursor(complete(&run)).unwrap();
    let wire = WireProcessHistory::capture(&history).unwrap();
    let original = serde_json::to_value(&wire).unwrap();
    for field in ["sampler", "target_arch", "target_os", "arithmetic"] {
        let mut changed = original.clone();
        changed["request"][field] = serde_json::json!("foreign");
        let changed: WireProcessHistory = serde_json::from_value(changed).unwrap();
        let error = without_draws(|| changed.replay(run.request.plan())).unwrap_err();
        assert!(
            error
                .message()
                .contains("sampler or arithmetic environment"),
            "{error:?}"
        );
    }
}
