//! One bounded executor: serial and host workers share member admission.

use std::sync::{Mutex, mpsc};

use super::EvaluationMapOccurrenceId;
use super::{
    CompleteEvaluationMap, EvaluationMapExecutionPolicy, EvaluationMapPlan, EvaluationMapRetention,
    EvaluationMapTerminalReport, invalid, outcomes::Outcome, validate_member,
};
use crate::{DifferentiableEvaluation, DifferentiableProgram, DifferentiableValues};
use eqiora_core::Diagnostic;

type MemberResult = Result<DifferentiableEvaluation, Vec<Diagnostic>>;

pub(super) fn execute<E, C, D>(
    plan: &EvaluationMapPlan,
    evaluator: &E,
    should_cancel: &mut C,
    deliver: &mut D,
    observe_resident: &mut impl FnMut(usize),
) -> Result<CompleteEvaluationMap, EvaluationMapTerminalReport>
where
    E: Fn(&DifferentiableProgram, &DifferentiableValues) -> MemberResult + Sync,
    C: FnMut() -> bool,
    D: FnMut(EvaluationMapOccurrenceId, &DifferentiableEvaluation),
{
    let (execution, outcomes) = execute_members(
        plan.len(),
        plan.policy(),
        &|index| evaluate(plan, evaluator, index),
        should_cancel,
        deliver,
        &DifferentiableEvaluation::primal_evidence,
        observe_resident,
    );
    CompleteEvaluationMap::finish(plan, execution, outcomes)
}

/// Scheduling owns occurrence order and lifetime, not a member's numerical method.
/// The caller admits each result and supplies its original acceptance evidence.
pub(super) fn execute_members<M, R, S, E, C, D, F>(
    count: usize,
    policy: EvaluationMapExecutionPolicy,
    evaluator: &E,
    should_cancel: &mut C,
    deliver: &mut D,
    release: &F,
    observe_resident: &mut impl FnMut(usize),
) -> (ulid::Ulid, Vec<Outcome<M, R, S>>)
where
    M: Send,
    R: Send,
    S: Send,
    E: Fn(usize) -> Outcome<M, R, S> + Sync,
    C: FnMut() -> bool,
    D: FnMut(EvaluationMapOccurrenceId, &M),
    F: Fn(&M) -> R,
{
    let execution = ulid::Ulid::generate();
    let mut outcomes = (0..count).map(|_| Outcome::NotStarted).collect::<Vec<_>>();
    let mut resident = 0usize;
    for start in (0..count).step_by(policy.chunk_size()) {
        let end = start.saturating_add(policy.chunk_size()).min(count);
        let stopped = if policy.workers() == 1 {
            serial_chunk(evaluator, should_cancel, start, &mut outcomes[start..end])
        } else {
            parallel_chunk(
                policy.workers(),
                evaluator,
                should_cancel,
                start,
                &mut outcomes[start..end],
            )
        };
        resident += outcomes[start..end]
            .iter()
            .filter(|state| matches!(state, Outcome::Accepted(_)))
            .count();
        observe_resident(resident);
        for (offset, state) in outcomes[start..end].iter_mut().enumerate() {
            if let Outcome::Accepted(member) = state {
                deliver(
                    EvaluationMapOccurrenceId::new(execution, start + offset),
                    member,
                );
                if policy.retention() == EvaluationMapRetention::Recompute {
                    *state = Outcome::Released(Box::new(release(member)));
                    resident -= 1;
                }
            }
        }
        observe_resident(resident);
        if stopped {
            break;
        }
    }
    (execution, outcomes)
}

fn stopped<M, R, S>(state: &Outcome<M, R, S>) -> bool {
    matches!(state, Outcome::Failed(_) | Outcome::Cancelled(_))
}

fn serial_chunk<M, R, S, E, C>(
    evaluator: &E,
    should_cancel: &mut C,
    start: usize,
    states: &mut [Outcome<M, R, S>],
) -> bool
where
    E: Fn(usize) -> Outcome<M, R, S>,
    C: FnMut() -> bool,
{
    for (offset, state) in states.iter_mut().enumerate() {
        if should_cancel() {
            *state = Outcome::Cancelled(None);
            return true;
        }
        *state = evaluator(start + offset);
        if stopped(state) {
            return true;
        }
    }
    false
}

fn parallel_chunk<M, R, S, E, C>(
    worker_limit: usize,
    evaluator: &E,
    should_cancel: &mut C,
    start: usize,
    states: &mut [Outcome<M, R, S>],
) -> bool
where
    M: Send,
    R: Send,
    S: Send,
    E: Fn(usize) -> Outcome<M, R, S> + Sync,
    C: FnMut() -> bool,
{
    // Both channels and the ordering buffer are bounded by admitted counts.
    // Every evaluator owns its numerical state; no provider session crosses tasks.
    let workers = worker_limit.min(states.len());
    let (tasks_tx, tasks_rx) = mpsc::sync_channel::<usize>(workers);
    let tasks_rx = Mutex::new(tasks_rx);
    let (results_tx, results_rx) = mpsc::sync_channel(workers);
    std::thread::scope(|scope| {
        for _ in 0..workers {
            let tasks = &tasks_rx;
            let results = results_tx.clone();
            if let Err(error) = std::thread::Builder::new().spawn_scoped(scope, move || {
                loop {
                    let task = {
                        tasks
                            .lock()
                            .expect("map task receiver is not poisoned")
                            .recv()
                    };
                    let Ok(index) = task else {
                        break;
                    };
                    // Unwind cannot strand the coordinator or trigger a retry.
                    let state = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        evaluator(start + index)
                    }))
                    .unwrap_or_else(|_| {
                        Outcome::Failed(vec![invalid(
                            "evaluation map worker panicked; occurrence was not accepted",
                        )])
                    });
                    if results.send((index, state)).is_err() {
                        break;
                    }
                }
            }) {
                // No task has been dispatched. Join already-created idle workers.
                states[0] = Outcome::Failed(vec![invalid(format!(
                    "evaluation map could not start a host worker for chunk at occurrence {start}: {error}"
                ))]);
                drop(tasks_tx);
                return true;
            }
        }
        drop(results_tx);
        let (mut next, mut active, mut halted) = (0, 0, false);
        loop {
            while !halted && next < states.len() && active < workers {
                if should_cancel() {
                    states[next] = Outcome::Cancelled(None);
                    halted = true;
                    break;
                }
                tasks_tx
                    .send(next)
                    .expect("scoped map workers own the receiver");
                next += 1;
                active += 1;
            }
            if active == 0 {
                break;
            }
            let (index, state) = results_rx
                .recv()
                .expect("every dispatched map task returns one outcome");
            active -= 1;
            halted |= stopped(&state);
            states[index] = state;
        }
        drop(tasks_tx);
        halted
    })
}

fn evaluate<E>(plan: &EvaluationMapPlan, evaluator: &E, index: usize) -> Outcome
where
    E: Fn(&DifferentiableProgram, &DifferentiableValues) -> MemberResult,
{
    let point = &plan.points().expect("steady execution selected")[index];
    match evaluator(plan.program().expect("steady execution selected"), point).and_then(|member| {
        validate_member(plan, point, &member).map_err(|error| vec![error])?;
        plan.program()
            .expect("steady execution selected")
            .prepare_map_member(member)
            .map_err(|error| vec![error])
    }) {
        Ok(member) => Outcome::Accepted(Box::new(member)),
        Err(mut errors) => {
            if errors.is_empty() {
                errors.push(invalid("evaluation failed without a diagnostic"));
            }
            Outcome::Failed(errors)
        }
    }
}
