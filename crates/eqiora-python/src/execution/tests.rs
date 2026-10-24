use std::sync::Arc;
use std::thread;

use eqiora::Diagnostic;
use eqiora::diagnostic::codes;

use super::{
    CacheClaim, MaterializationClaim, PyRunStatus, ResultCache, RunFailure, RunShared, RunState,
    RunTerminal, RunTerminalKind,
};

#[test]
fn run_states_follow_the_explicit_branching_transition_table() {
    let mut completed = RunState::accepted();
    completed.transition(PyRunStatus::Running);
    completed.transition(PyRunStatus::Cancelling);
    completed.transition(PyRunStatus::Completed);
    assert_eq!(
        completed.history,
        [
            PyRunStatus::Created,
            PyRunStatus::Validating,
            PyRunStatus::Queued,
            PyRunStatus::Running,
            PyRunStatus::Cancelling,
            PyRunStatus::Completed,
        ]
    );
    assert!(!completed.integrity_failed);

    completed.transition(PyRunStatus::Cancelled);
    assert!(completed.integrity_failed);
    assert_eq!(completed.status, PyRunStatus::Failed);
    assert!(matches!(
        completed.terminal,
        Some(RunTerminal::Failed(RunFailure::Internal(_)))
    ));

    let mut cancelled = RunState::accepted();
    cancelled.transition(PyRunStatus::Cancelling);
    cancelled.transition(PyRunStatus::Cancelled);
    assert_eq!(cancelled.history.len(), 5);
    assert!(!cancelled.integrity_failed);
}

#[test]
fn every_waiter_observes_one_failed_terminal() {
    let shared = Arc::new(RunShared::new());
    let waiters: Vec<_> = (0..2)
        .map(|_| {
            let shared = Arc::clone(&shared);
            thread::spawn(move || shared.wait_terminal_kind())
        })
        .collect();
    shared.finish(RunTerminal::Failed(RunFailure::Execution(vec![
        Diagnostic::error(codes::NONSQUARE_SYSTEM, "probe"),
    ])));
    for waiter in waiters {
        assert!(matches!(
            waiter.join().expect("waiter must not panic"),
            RunTerminalKind::Failed(RunFailure::Execution(_))
        ));
    }
}

#[test]
fn abandoned_result_materialization_fails_and_wakes_future_callers() {
    let cache = Arc::new(ResultCache::new());
    assert!(matches!(cache.claim(), CacheClaim::Materialize));
    {
        let _claim = MaterializationClaim::new(Arc::clone(&cache));
    }
    let CacheClaim::Failed(diagnostics) = cache.claim() else {
        panic!("an abandoned materializer must leave one stable failure");
    };
    assert_eq!(diagnostics[0].code().to_string(), "EQ0002");
    assert_eq!(
        diagnostics[0].message(),
        "the completed native Result could not be materialized"
    );
}

#[test]
fn progress_streams_are_bounded_and_keep_their_segment_prefix() {
    use std::sync::mpsc::TrySendError;

    let shared = RunShared::new();
    shared.mark_running();
    let first = shared.observe_progress(7);
    first.try_send((2, 0.25)).unwrap();
    assert!(matches!(
        first.try_send((3, 0.5)),
        Err(TrySendError::Full(_))
    ));
    let progress = shared.progress_snapshot().unwrap();
    assert_eq!(progress.accepted_steps, 9);
    assert_eq!(progress.model_time_bits, 0.25_f64.to_bits());
    assert_eq!(progress.maximum_steps, None);

    first.try_send((4, 0.75)).unwrap();
    let next = shared.observe_progress(11);
    assert!(matches!(
        first.try_send((5, 1.0)),
        Err(TrySendError::Disconnected(_))
    ));
    // Replacing the source retains its last queued accepted boundary.
    assert_eq!(shared.progress_snapshot().unwrap().accepted_steps, 11);
    next.try_send((2, 1.25)).unwrap();
    assert!(shared.request_cancellation());
    assert!(shared.cancellation_requested());
    assert_eq!(shared.progress_snapshot().unwrap().accepted_steps, 13);
}

#[test]
fn final_progress_cannot_be_overwritten_by_a_queued_observation() {
    use std::sync::mpsc::TrySendError;

    let shared = RunShared::new();
    shared.mark_running();
    let sender = shared.observe_progress(10);
    sender.try_send((2, 0.5)).unwrap();
    let final_progress = super::PyRunProgress {
        accepted_steps: 20,
        maximum_steps: None,
        model_time_bits: 1.0_f64.to_bits(),
    };
    shared.publish_progress(final_progress.clone());
    assert!(matches!(
        sender.try_send((3, 0.75)),
        Err(TrySendError::Disconnected(_))
    ));
    assert_eq!(shared.progress_snapshot(), Some(final_progress));
}

#[test]
fn terminal_failure_retains_the_last_observation_and_closes_the_stream() {
    use std::sync::mpsc::TrySendError;

    let shared = RunShared::new();
    shared.mark_running();
    let sender = shared.observe_progress(3);
    sender.try_send((2, 0.5)).unwrap();
    shared.finish(RunTerminal::Failed(RunFailure::Execution(vec![])));
    assert_eq!(shared.progress_snapshot().unwrap().accepted_steps, 5);
    assert!(matches!(
        sender.try_send((3, 0.75)),
        Err(TrySendError::Disconnected(_))
    ));
    let late = shared.observe_progress(9);
    assert!(matches!(
        late.try_send((1, 1.0)),
        Err(TrySendError::Disconnected(_))
    ));
    assert_eq!(shared.progress_snapshot().unwrap().accepted_steps, 5);
}
