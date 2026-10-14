//! One bounded executor: serial and host workers share member admission.

use std::sync::{Mutex, mpsc};

use super::{
    CompleteEvaluationMap, EvaluationMapPlan, EvaluationMapRetention, EvaluationMapTerminalReport,
    invalid, outcomes::Outcome, validate_member,
};
use crate::{DifferentiableEvaluation, DifferentiableProgram};
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
    E: Fn(&DifferentiableProgram, &[f64]) -> MemberResult + Sync,
    C: FnMut() -> bool,
    D: FnMut(usize, &DifferentiableEvaluation),
{
    let mut outcomes = vec![Outcome::NotStarted; plan.points().len()];
    let mut resident = 0usize;
    for start in (0..plan.points().len()).step_by(plan.policy().chunk_size()) {
        let end = start
            .saturating_add(plan.policy().chunk_size())
            .min(plan.points().len());
        let stopped = if plan.policy().workers() == 1 {
            serial_chunk(
                plan,
                evaluator,
                should_cancel,
                start,
                &mut outcomes[start..end],
            )
        } else {
            parallel_chunk(
                plan,
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
                deliver(start + offset, member);
                if plan.policy().retention() == EvaluationMapRetention::Recompute {
                    *state = Outcome::Released(Box::new(member.primal_evidence()));
                    resident -= 1;
                }
            }
        }
        observe_resident(resident);
        if stopped {
            break;
        }
    }
    CompleteEvaluationMap::finish(plan, outcomes)
}

fn serial_chunk<E, C>(
    plan: &EvaluationMapPlan,
    evaluator: &E,
    should_cancel: &mut C,
    start: usize,
    states: &mut [Outcome],
) -> bool
where
    E: Fn(&DifferentiableProgram, &[f64]) -> MemberResult,
    C: FnMut() -> bool,
{
    for (offset, state) in states.iter_mut().enumerate() {
        if should_cancel() {
            *state = Outcome::Cancelled;
            return true;
        }
        *state = evaluate(plan, evaluator, start + offset);
        if matches!(state, Outcome::Failed(_)) {
            return true;
        }
    }
    false
}

fn parallel_chunk<E, C>(
    plan: &EvaluationMapPlan,
    evaluator: &E,
    should_cancel: &mut C,
    start: usize,
    states: &mut [Outcome],
) -> bool
where
    E: Fn(&DifferentiableProgram, &[f64]) -> MemberResult + Sync,
    C: FnMut() -> bool,
{
    // Both channels and the ordering buffer are bounded by admitted counts.
    // Workers create no nested executor: each evaluator uses its own one-thread
    // reference solve and immutable Program. No provider session crosses tasks.
    let workers = plan.policy().workers().min(states.len());
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
                    // A worker unwind must not strand the coordinator or trigger
                    // a retry. Publish this exact occurrence as failed instead.
                    let state = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        evaluate(plan, evaluator, start + index)
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
                // No task has been dispatched yet. Closing the channel joins
                // already-created idle workers without retry or fallback.
                states[0] = Outcome::Failed(vec![invalid(format!(
                    "evaluation map could not start a host worker for chunk at occurrence {start}: {error}"
                ))]);
                drop(tasks_tx);
                return true;
            }
        }
        drop(results_tx);
        let (mut next, mut active, mut stopped) = (0, 0, false);
        loop {
            while !stopped && next < states.len() && active < workers {
                if should_cancel() {
                    states[next] = Outcome::Cancelled;
                    stopped = true;
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
            stopped |= matches!(state, Outcome::Failed(_));
            states[index] = state;
        }
        drop(tasks_tx);
        stopped
    })
}

fn evaluate<E>(plan: &EvaluationMapPlan, evaluator: &E, index: usize) -> Outcome
where
    E: Fn(&DifferentiableProgram, &[f64]) -> MemberResult,
{
    let point = &plan.points()[index];
    match evaluator(&plan.program, point.coordinates()).and_then(|member| {
        validate_member(plan, point, &member).map_err(|error| vec![error])?;
        plan.program
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
