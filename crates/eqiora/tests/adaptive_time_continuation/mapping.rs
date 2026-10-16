//! Independent maps reuse the ordinary adaptive lifecycle fixtures.
use super::*;

#[path = "mapping/rejections.rs"]
mod rejections;

#[test]
fn event_budget_failure_does_not_terminate_an_independent_inflight_member() {
    use eqiora::api::{EvaluationMapExecutionPolicy, EvaluationMapPlan, EvaluationMapRetention};
    use eqiora_numerics::CommonResult;

    let template = fixture(Some(1), 1e-9, 1e-3);
    let plan = template.plan().clone();
    let preparation = CommonTimeRunRequest::new(
        plan.clone(),
        plan.initial_state(0.).unwrap(),
        0.5,
        vec![0.5],
    )
    .unwrap();
    let prepared = finish(&preparation, None).unwrap();
    let states = [
        prepared.time_states().unwrap()[0].clone(),
        plan.initial_state(0.5).unwrap(),
    ];
    let requests = states
        .into_iter()
        .map(|state| {
            CommonTimeRunRequest::new(plan.clone(), state, 1.5, vec![0.6, 0.8, 1.5]).unwrap()
        })
        .collect::<Vec<_>>();
    // The first member hits a second reset before 1.5; the second only hits one.
    assert!(
        finish(&requests[0], None)
            .unwrap_err()
            .message()
            .contains("accepted-event budget")
    );
    let expected = finish(&requests[1], None).unwrap();
    assert_eq!(expected.ode_history().unwrap().events().len(), 1);
    let policy = EvaluationMapExecutionPolicy::new(
        2,
        eqiora::realization::Target::HostCpu {
            threads: std::num::NonZeroUsize::new(2).unwrap(),
        },
        EvaluationMapRetention::Retain,
        1 << 28,
    )
    .unwrap();
    let map = EvaluationMapPlan::from_time_requests(
        Arc::new(plan),
        &requests,
        None,
        policy,
        1 << 20,
        |request, saved| {
            finish(request, saved)
                .and_then(|trajectory| CommonResult::accept_trajectory(0., trajectory))
                .map(ControlFlow::Continue)
                .map_err(|error| vec![error])
        },
    )
    .unwrap();
    let report = map.execute().unwrap_err();
    assert_eq!(report.stopped_index(), 0);
    assert!(
        report.diagnostics()[0]
            .message()
            .contains("accepted-event budget")
    );
    assert!(report.time_result(0).is_none());
    assert_eq!(report.plan().time_requests().unwrap(), requests);
    assert_eq!(
        report
            .time_result(1)
            .unwrap()
            .trajectory()
            .unwrap()
            .to_bytes()
            .unwrap(),
        expected.to_bytes().unwrap()
    );
}

#[test]
fn mapped_decay_keeps_distinct_adaptive_histories_events_and_restarts() {
    use eqiora::api::{EvaluationMapExecutionPolicy, EvaluationMapPlan};
    use eqiora_numerics::CommonResult;

    let template = fixture(Some(4), 1e-9, 1e-3);
    let plan = template.plan().clone();
    // Both physical states come from ordinary owners: one has already decayed,
    // the other has its authored initial value at the same physical time.
    let preparation = CommonTimeRunRequest::new(
        plan.clone(),
        plan.initial_state(0.).unwrap(),
        0.5,
        vec![0.5],
    )
    .unwrap();
    let prepared = finish(&preparation, None).unwrap();
    let initial = [
        prepared.time_states().unwrap()[0].clone(),
        plan.initial_state(0.5).unwrap(),
    ];
    assert_ne!(initial[0].values(), initial[1].values());
    let requests = initial
        .into_iter()
        .map(|state| {
            CommonTimeRunRequest::new(plan.clone(), state, 2.2, vec![0.6, 0.8, 1., 1.5, 2.1])
                .unwrap()
        })
        .collect::<Vec<_>>();
    let expected = requests
        .iter()
        .map(|request| finish(request, None).unwrap())
        .collect::<Vec<_>>();
    let histories = expected
        .iter()
        .map(|value| value.ode_history().unwrap())
        .collect::<Vec<_>>();
    assert_ne!(histories[0].steps(), histories[1].steps());
    assert_ne!(
        histories[0].events()[0].proposal().time(),
        histories[1].events()[0].proposal().time()
    );
    let execute = |request: &CommonTimeRunRequest, saved: Option<&CommonTimeCheckpoint>| {
        finish(request, saved)
            .and_then(|trajectory| CommonResult::accept_trajectory(0., trajectory))
            .map(ControlFlow::Continue)
            .map_err(|error| vec![error])
    };
    let policy = eqiora::api::EvaluationMapExecutionPolicy::new(
        2,
        eqiora::realization::Target::HostCpu {
            threads: std::num::NonZeroUsize::new(2).unwrap(),
        },
        eqiora::api::EvaluationMapRetention::Retain,
        1 << 28,
    )
    .unwrap();
    let map = EvaluationMapPlan::from_time_requests(
        Arc::new(plan.clone()),
        &requests,
        None,
        policy,
        1 << 20,
        execute,
    )
    .unwrap();
    let complete = map.execute().unwrap();
    for (index, expected) in expected.iter().enumerate() {
        assert_eq!(
            complete
                .time_result(index)
                .unwrap()
                .trajectory()
                .unwrap()
                .to_bytes()
                .unwrap(),
            expected.to_bytes().unwrap()
        );
    }
    // Observe each member exactly at its first reset. Output cadence must not
    // move the root or substitute the pre-reset state for the committed state.
    let mut event_times = histories
        .iter()
        .map(|history| history.events()[0].proposal().time())
        .collect::<Vec<_>>();
    event_times.sort_by(f64::total_cmp);
    let event_requests = requests
        .iter()
        .map(|request| {
            CommonTimeRunRequest::new(
                plan.clone(),
                request.state().clone(),
                request.until_s(),
                event_times.clone(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let event_map = EvaluationMapPlan::from_time_requests(
        Arc::new(plan.clone()),
        &event_requests,
        None,
        policy,
        1 << 20,
        execute,
    )
    .unwrap()
    .execute()
    .unwrap();
    for (index, request) in event_requests.iter().enumerate() {
        let trajectory = event_map.time_result(index).unwrap().trajectory().unwrap();
        assert_eq!(
            trajectory.to_bytes().unwrap(),
            finish(request, None).unwrap().to_bytes().unwrap()
        );
        let event = &trajectory.ode_history().unwrap().events()[0];
        let sample = event_times
            .iter()
            .position(|&time| time == event.proposal().time())
            .unwrap();
        assert_eq!(trajectory.time_states().unwrap()[sample].values(), &[1.]);
        assert_eq!(
            trajectory.time_states().unwrap()[sample].values(),
            event.after_state()
        );
        assert_ne!(event.after_state(), event.proposal().state());
    }
    // Stop only the first occurrence, after its first reset. Other members must
    // not inherit that event budget, solver checkpoint or physical state.
    let stopped = EvaluationMapPlan::from_time_requests(
        Arc::new(plan.clone()),
        &requests,
        None,
        EvaluationMapExecutionPolicy::retained(1 << 28),
        1 << 20,
        |request, _| Ok(ControlFlow::Break(stop(request, 0.95))),
    )
    .unwrap()
    .execute()
    .unwrap_err();
    let checkpoint = reopen(stopped.time_checkpoint(0).unwrap().clone());
    assert_eq!(checkpoint.accepted_events(), 1);
    let checkpoints = [Some(checkpoint), None];
    let resumed = EvaluationMapPlan::from_time_requests(
        Arc::new(plan),
        &requests,
        Some(&checkpoints),
        policy,
        1 << 20,
        execute,
    )
    .unwrap()
    .execute()
    .unwrap();
    for (index, expected) in expected.iter().enumerate() {
        assert_eq!(
            resumed
                .time_result(index)
                .unwrap()
                .trajectory()
                .unwrap()
                .to_bytes()
                .unwrap(),
            expected.to_bytes().unwrap()
        );
    }
}
