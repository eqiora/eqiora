//! Composition checks against separately executed ordinary stochastic Runs.
use super::*;
use crate::{EvaluationMapProducts, ModelDocument};
use eqiora_artifact::ModelEnvelope;
use eqiora_numerics::SamplingIdentity;
use eqiora_realization::Target;
use std::{
    num::NonZeroUsize,
    sync::atomic::{AtomicUsize, Ordering},
};

const MEMBER_LIMIT: usize = 1 << 20;
const LIMIT: usize = 1 << 28;

fn fixture() -> (Arc<CommonTimePlan>, Vec<CommonTimeRunRequest>) {
    let components =
        include_str!("../../../packages/Eqiora.Stochastic.Processes/src/processes.eqi");
    let composition = include_str!("../../../../../examples/stochastic-processes/src/main.eqi")
        .replace(
            "import Eqiora.Stochastic.Processes.processes as processes;",
            "",
        )
        .replace("processes.", "");
    let document =
        ModelDocument::compile("mapped.eqi", &format!("{components}\n{composition}")).unwrap();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let plan =
        Arc::new(CommonTimePlan::resolve_euler_maruyama(&model, document.program(), 3).unwrap());
    let requests = [2u32, 1, 2, 3]
        .into_iter()
        .map(|sample| {
            CommonTimeRunRequest::sampled(
                (*plan).clone(),
                plan.initial_state(0.).unwrap(),
                1.,
                vec![0.25, 0.5, 1.],
                [83; 32],
                SamplingIdentity::new(&sample.to_le_bytes(), b"mapped", &[], None).unwrap(),
            )
            .unwrap()
        })
        .collect();
    (plan, requests)
}

fn policy(
    chunk: usize,
    workers: usize,
    retention: EvaluationMapRetention,
) -> EvaluationMapExecutionPolicy {
    EvaluationMapExecutionPolicy::new(
        chunk,
        Target::HostCpu {
            threads: NonZeroUsize::new(workers).unwrap(),
        },
        retention,
        LIMIT,
    )
    .unwrap()
}

fn run(
    request: &CommonTimeRunRequest,
    checkpoint: Option<&CommonTimeCheckpoint>,
) -> Result<ControlFlow<CommonTimeCheckpoint, CommonResult>, Vec<Diagnostic>> {
    advance(request, checkpoint, usize::MAX)
}

fn advance(
    request: &CommonTimeRunRequest,
    checkpoint: Option<&CommonTimeCheckpoint>,
    stop: usize,
) -> Result<ControlFlow<CommonTimeCheckpoint, CommonResult>, Vec<Diagnostic>> {
    match request
        .advance_process(checkpoint, |_, count| {
            if count >= stop {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        })
        .map_err(|error| vec![error])?
    {
        ControlFlow::Break(checkpoint) => Ok(ControlFlow::Break(checkpoint)),
        ControlFlow::Continue(trajectory) => CommonResult::accept_trajectory(0., trajectory)
            .map(ControlFlow::Continue)
            .map_err(|error| vec![error]),
    }
}

#[test]
fn independent_processes_preserve_order_repeats_and_recompute() {
    let (plan, requests) = fixture();
    let expected = requests
        .iter()
        .map(|request| run(request, None).unwrap().continue_value().unwrap())
        .collect::<Vec<_>>();
    assert_ne!(expected[0].identity(), expected[1].identity());
    assert_eq!(expected[0].identity(), expected[2].identity());
    for (chunk, workers) in [(1, 1), (2, 2), (4, 3)] {
        for retention in [
            EvaluationMapRetention::Retain,
            EvaluationMapRetention::Recompute,
        ] {
            let map = EvaluationMapPlan::from_time_requests(
                plan.clone(),
                &requests,
                None,
                policy(chunk, workers, retention),
                MEMBER_LIMIT,
                run,
            )
            .unwrap();
            let mut delivered = Vec::new();
            let complete = map
                .execute_with_delivery(
                    || false,
                    |index, occurrence| {
                        let EvaluationMapOccurrence::TimeAccepted(result) = occurrence else {
                            panic!("time Result")
                        };
                        delivered.push(index);
                        assert_eq!(
                            result.to_bytes().unwrap(),
                            expected[index].to_bytes().unwrap()
                        );
                    },
                )
                .unwrap();
            assert_eq!(delivered, [0, 1, 2, 3]);
            complete.validate().unwrap();
            assert_eq!(complete.len(), requests.len());
            assert_eq!(complete.plan().time_requests().unwrap(), requests);
            assert!(EvaluationMapProducts::new(&complete, &[4], &[], &[0], LIMIT).is_err());
            for (index, expected) in expected.iter().enumerate() {
                assert_eq!(complete.time_evidence(index), Some(expected.identity()));
                match retention {
                    EvaluationMapRetention::Retain => {
                        assert_eq!(
                            complete.time_result(index).unwrap().to_bytes().unwrap(),
                            expected.to_bytes().unwrap()
                        );
                        assert!(complete.recompute_time(index).is_err());
                    }
                    EvaluationMapRetention::Recompute => {
                        assert!(complete.time_result(index).is_none());
                        assert_eq!(
                            complete.recompute_time(index).unwrap().to_bytes().unwrap(),
                            expected.to_bytes().unwrap()
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn recomputation_ignores_wall_time_but_rejects_changed_acceptance() {
    use crate::evaluation_map::outcomes::{CompleteMembers, Completed};

    let (plan, requests) = fixture();
    let calls = AtomicUsize::new(0);
    let map = EvaluationMapPlan::from_time_requests(
        plan,
        &requests,
        None,
        policy(1, 1, EvaluationMapRetention::Recompute),
        MEMBER_LIMIT,
        move |request, saved| {
            let value = run(request, saved)?.continue_value().unwrap();
            let elapsed = calls.fetch_add(1, Ordering::SeqCst) as f64;
            Ok(ControlFlow::Continue(
                value.with_elapsed_seconds(elapsed).unwrap(),
            ))
        },
    )
    .unwrap();
    let mut complete = map.execute().unwrap();
    let recomputed = complete.recompute_time(0).unwrap();
    assert_ne!(Some(recomputed.identity()), complete.time_evidence(0));
    assert_eq!(
        Some(recomputed.with_elapsed_seconds(0.).unwrap().identity()),
        complete.time_evidence(0)
    );
    let CompleteMembers::Time(Completed::Released(evidence)) = &mut complete.members else {
        panic!("released time members");
    };
    evidence.swap(0, 1);
    assert!(
        complete.recompute_time(0).unwrap_err()[0]
            .message()
            .contains("changed its original acceptance")
    );
    assert!(complete.time_result(0).is_none());
}

#[test]
fn interrupted_member_retains_atomic_checkpoint_and_original_sample() {
    let (plan, requests) = fixture();
    let calls = Arc::new(AtomicUsize::new(0));
    let called = calls.clone();
    let map = EvaluationMapPlan::from_time_requests(
        plan.clone(),
        &requests,
        None,
        policy(2, 1, EvaluationMapRetention::Retain),
        MEMBER_LIMIT,
        move |request, checkpoint| {
            called.fetch_add(1, Ordering::SeqCst);
            advance(request, checkpoint, 2)
        },
    )
    .unwrap();
    let report = map.execute().unwrap_err();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(report.is_cancelled());
    assert_eq!(report.stopped_index(), 0);
    assert!(matches!(
        report.occurrence(1),
        Some(EvaluationMapOccurrence::NotStarted)
    ));
    let checkpoint = report.time_checkpoint(0).unwrap();
    assert_eq!(checkpoint.request().identity(), requests[0].identity());
    assert_eq!(checkpoint.accepted_steps(), 2);
    let checkpoint =
        CommonTimeCheckpoint::from_bytes(&checkpoint.to_bytes().unwrap(), &plan).unwrap();
    let checkpoints = [Some(checkpoint.clone()), None, None, None];
    let resumed = EvaluationMapPlan::from_time_requests(
        plan.clone(),
        &requests,
        Some(&checkpoints),
        policy(4, 2, EvaluationMapRetention::Retain),
        MEMBER_LIMIT,
        run,
    )
    .unwrap()
    .execute()
    .unwrap();
    for (index, request) in requests.iter().enumerate() {
        assert_eq!(
            resumed.time_result(index).unwrap().to_bytes().unwrap(),
            run(request, None)
                .unwrap()
                .continue_value()
                .unwrap()
                .to_bytes()
                .unwrap()
        );
    }
    let wrong = [None, Some(checkpoint), None, None];
    assert!(
        EvaluationMapPlan::from_time_requests(
            plan,
            &requests,
            Some(&wrong),
            policy(1, 1, EvaluationMapRetention::Retain),
            MEMBER_LIMIT,
            run
        )
        .is_err()
    );
}

#[test]
fn failed_or_foreign_member_cannot_replace_its_occurrence() {
    let (plan, requests) = fixture();
    let wrong = run(&requests[1], None).unwrap();
    let map = EvaluationMapPlan::from_time_requests(
        plan.clone(),
        &requests,
        None,
        policy(1, 1, EvaluationMapRetention::Retain),
        MEMBER_LIMIT,
        move |_, _| Ok(wrong.clone()),
    )
    .unwrap();
    let report = map.execute().unwrap_err();
    assert!(!report.is_cancelled());
    assert!(
        report.diagnostics()[0]
            .message()
            .contains("exact requested Run")
    );
    assert!(report.time_result(0).is_none());
    assert!(matches!(
        report.occurrence(1),
        Some(EvaluationMapOccurrence::NotStarted)
    ));
    let fail_id = requests[1].identity().to_owned();
    let map = EvaluationMapPlan::from_time_requests(
        plan,
        &requests,
        None,
        policy(4, 1, EvaluationMapRetention::Recompute),
        MEMBER_LIMIT,
        move |request, saved| {
            if request.identity() == fail_id {
                Err(vec![invalid("deliberate member failure")])
            } else {
                run(request, saved)
            }
        },
    )
    .unwrap();
    let report = map.execute().unwrap_err();
    assert_eq!(report.stopped_index(), 1);
    assert!(report.recompute_time(0).is_ok());
    assert!(report.recompute_time(1).is_err());
    assert!(matches!(
        report.occurrence(2),
        Some(EvaluationMapOccurrence::NotStarted)
    ));
    assert_eq!(report.plan().time_requests().unwrap()[1], requests[1]);
}

#[test]
fn cancellation_does_not_run_inactive_members_or_consume_their_paths() {
    let (plan, requests) = fixture();
    let calls = Arc::new(AtomicUsize::new(0));
    let called = calls.clone();
    let map = EvaluationMapPlan::from_time_requests(
        plan,
        &requests,
        None,
        policy(4, 3, EvaluationMapRetention::Retain),
        MEMBER_LIMIT,
        move |request, saved| {
            called.fetch_add(1, Ordering::SeqCst);
            run(request, saved)
        },
    )
    .unwrap();
    let report = map.execute_with_cancellation(|| true).unwrap_err();
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(report.is_cancelled());
    assert!(report.time_checkpoint(0).is_none());
    let result = map.execute().unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 4);
    for (index, request) in requests.iter().enumerate() {
        assert_eq!(
            result.time_result(index).unwrap().identity(),
            run(request, None)
                .unwrap()
                .continue_value()
                .unwrap()
                .identity()
        );
    }
}

#[test]
fn admission_rejects_retiming_and_storage_overflow_before_dispatch() {
    let (plan, mut requests) = fixture();
    let request = &requests[1];
    let (master, identity) = request.sampling().unwrap();
    requests[1] = CommonTimeRunRequest::sampled(
        (*plan).clone(),
        request.state().clone(),
        1.,
        vec![0.5, 1.],
        *master,
        identity.clone(),
    )
    .unwrap();
    let invalid = EvaluationMapPlan::from_time_requests(
        plan.clone(),
        &requests,
        None,
        policy(1, 1, EvaluationMapRetention::Retain),
        MEMBER_LIMIT,
        |_, _| panic!("admission must not execute"),
    );
    assert!(
        invalid
            .unwrap_err()
            .message()
            .contains("fixed observation times")
    );
    requests[1] = requests[0].clone();
    for limit in [0, usize::MAX] {
        assert!(
            EvaluationMapPlan::from_time_requests(
                plan.clone(),
                &requests,
                None,
                policy(1, 1, EvaluationMapRetention::Retain),
                limit,
                run
            )
            .is_err()
        );
    }
    let bounded = EvaluationMapPlan::from_time_requests(
        plan.clone(),
        &requests,
        None,
        policy(1, 1, EvaluationMapRetention::Retain),
        1,
        run,
    )
    .unwrap();
    assert!(
        bounded.execute().unwrap_err().diagnostics()[0]
            .message()
            .contains("encoded byte limit")
    );
    let empty = EvaluationMapPlan::from_time_requests(
        plan,
        &[],
        None,
        policy(1, 1, EvaluationMapRetention::Retain),
        MEMBER_LIMIT,
        |_, _| panic!("empty map"),
    )
    .unwrap()
    .execute()
    .unwrap();
    assert!(empty.is_empty());
    empty.validate().unwrap();
}
