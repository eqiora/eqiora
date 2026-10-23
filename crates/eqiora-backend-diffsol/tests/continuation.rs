#![cfg(feature = "diffsol-runtime")]

use eqiora_backend_diffsol::DiffsolTimeBackend;
use eqiora_core::Diagnostic;
use eqiora_time::{
    ImplicitDaeInitialization, TimeCheckpoint, TimeEquationClass, TimeMethod, TimePlan,
    TimeProblem, TimeSystem,
};
use std::cell::Cell;
use std::ops::ControlFlow;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc::sync_channel,
};

struct ForcedDecay {
    cancellation: Arc<AtomicBool>,
    calls: Cell<usize>,
    cancel_at: usize,
    cancel_rejected_trial: bool,
    previous_time: Cell<f64>,
    rejected_trial: Cell<bool>,
}

impl ForcedDecay {
    fn new(cancellation: Arc<AtomicBool>, cancel_at: usize, cancel_rejected_trial: bool) -> Self {
        Self {
            cancellation,
            calls: Cell::new(0),
            cancel_at,
            cancel_rejected_trial,
            previous_time: Cell::new(0.0),
            rejected_trial: Cell::new(false),
        }
    }
}
impl TimeSystem for ForcedDecay {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(&self, time: f64, state: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        let previous = self.previous_time.replace(time);
        // Tsit45 stage times within one trial increase. A retry after the
        // deliberately oversized first trial returns from t >= 0.9 to t < 0.5.
        if previous >= 0.9 && time < 0.5 {
            self.rejected_trial.set(true);
            if self.cancel_rejected_trial {
                self.cancellation.store(true, Ordering::Release);
            }
        }
        self.calls.set(self.calls.get() + 1);
        if self.calls.get() == self.cancel_at {
            self.cancellation.store(true, Ordering::Release);
        }
        output[0] = -25.0 * state[0] + time.sin();
        Ok(())
    }
    fn rhs_jvp(
        &self,
        _: f64,
        _: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        output[0] = -25.0 * direction[0];
        Ok(())
    }
}
fn problem(system: &ForcedDecay) -> TimeProblem<'_> {
    TimeProblem::new(
        system,
        TimeEquationClass::ExplicitOde,
        ImplicitDaeInitialization::accepted(vec![1.0], vec![-25.0]).unwrap(),
    )
    .unwrap()
}
fn plan() -> TimePlan {
    TimePlan::new(
        TimeMethod::Tsitouras45,
        0.0,
        1.0,
        1e-8,
        vec![1e-10],
        vec![0.01, 0.05, 0.1, 0.3, 1.0, 2.0],
    )
    .unwrap()
}
fn initial(problem: &TimeProblem<'_>, plan: &TimePlan) -> TimeCheckpoint {
    // Independent initial derivative and deliberately oversized next trial;
    // continuation must retain this controller decision, including on rollback.
    TimeCheckpoint::initial(
        problem,
        plan,
        DiffsolTimeBackend::IDENTITY,
        None,
        1.0,
        vec![-25.0],
    )
    .unwrap()
}

#[test]
fn interruption_preserves_exact_adaptive_history_and_output_samples() {
    let backend = DiffsolTimeBackend::new();
    let plan = plan();
    let cancel = Arc::new(AtomicBool::new(false));
    let system = ForcedDecay::new(cancel.clone(), usize::MAX, false);
    let problem = problem(&system);
    let initial = initial(&problem, &plan);
    let (sender, receiver) = sync_channel(1);
    let baseline = backend
        .advance(
            &problem,
            &plan,
            None,
            Some(&initial),
            &cancel,
            Some(&sender),
        )
        .unwrap()
        .continue_value()
        .unwrap();
    assert!(
        system.rejected_trial.get(),
        "fixture must reject an oversized adaptive trial"
    );
    let (accepted_steps, time) = receiver.try_recv().unwrap();
    assert_eq!(accepted_steps, 1);
    assert_eq!(
        time.to_bits(),
        baseline.history().steps()[0].end_time().to_bits()
    );
    assert!(
        receiver.try_recv().is_err(),
        "bounded undrained progress must not accumulate steps"
    );
    for cancel_at in [130, system.calls.get() - 3] {
        let cancel = Arc::new(AtomicBool::new(false));
        let interrupted = ForcedDecay::new(cancel.clone(), cancel_at, false);
        let interrupted_problem = crate::problem(&interrupted);
        let saved = backend
            .advance(
                &interrupted_problem,
                &plan,
                None,
                Some(&initial),
                &cancel,
                None,
            )
            .unwrap()
            .break_value()
            .unwrap();
        assert!(saved.time() > 0.0 && saved.time() < 2.0);
        let prefix = saved.history().unwrap();
        assert_eq!(
            prefix.steps(),
            &baseline.history().steps()[..prefix.steps().len()]
        );
        assert_eq!(saved.state(), prefix.steps().last().unwrap().end_state());
        assert!(
            !saved.samples().is_empty(),
            "fixture retains requested output values"
        );
        assert!(
            saved
                .clone()
                .with_history(
                    prefix.clone(),
                    vec![],
                    saved.next_step(),
                    saved.derivative().to_vec()
                )
                .is_err()
        );
        assert!(
            saved
                .clone()
                .with_history(
                    prefix.clone(),
                    saved.samples().to_vec(),
                    saved.next_step(),
                    vec![]
                )
                .is_err()
        );
        let truncated =
            eqiora_time::AcceptedTimeHistory::accepted(1, prefix.steps()[1..].to_vec(), vec![])
                .unwrap();
        assert!(
            saved
                .clone()
                .with_history(
                    truncated,
                    saved.samples().to_vec(),
                    saved.next_step(),
                    saved.derivative().to_vec()
                )
                .is_err()
        );
        cancel.store(false, Ordering::Release);
        let resumed = backend
            .advance(
                &interrupted_problem,
                &plan,
                None,
                Some(&saved),
                &cancel,
                None,
            )
            .unwrap()
            .continue_value()
            .unwrap();
        // Same provider/build and controls: every accepted step, native midpoint,
        // output and root outcome must be bit-identical, not merely close.
        assert_eq!(resumed, baseline);
        assert_eq!(outcome_bits(&resumed), outcome_bits(&baseline));
    }
    for (index, &time) in plan.output_times().iter().enumerate() {
        let exact = (627.0 * (-25.0 * time).exp() + 25.0 * time.sin() - time.cos()) / 626.0;
        let actual = baseline.samples().unwrap().state(index).unwrap()[0];
        assert!(
            (actual - exact).abs() < 2e-8,
            "analytic forced-decay solution at {time}"
        );
    }
}

#[test]
fn cancellation_inside_a_rejected_trial_does_not_replace_the_checkpoint() {
    let cancel = Arc::new(AtomicBool::new(false));
    let system = ForcedDecay::new(cancel.clone(), usize::MAX, true);
    let problem = problem(&system);
    let plan = plan();
    let initial = initial(&problem, &plan);
    let stopped = DiffsolTimeBackend::new()
        .advance(&problem, &plan, None, Some(&initial), &cancel, None)
        .unwrap();
    assert!(system.rejected_trial.get());
    assert_eq!(stopped, ControlFlow::Break(initial));
}

#[test]
fn incomplete_controller_or_changed_request_rejects_before_callbacks() {
    let cancel = Arc::new(AtomicBool::new(true));
    let system = ForcedDecay::new(cancel.clone(), usize::MAX, false);
    let problem = problem(&system);
    let plan = plan();
    assert!(
        TimeCheckpoint::initial(
            &problem,
            &plan,
            DiffsolTimeBackend::IDENTITY,
            None,
            1.0,
            vec![]
        )
        .is_err()
    );
    let initial = initial(&problem, &plan);
    let changed = TimePlan::new(
        TimeMethod::Tsitouras45,
        0.0,
        1.0,
        1e-7,
        vec![1e-10],
        plan.output_times().to_vec(),
    )
    .unwrap();
    assert!(
        DiffsolTimeBackend::new()
            .advance(&problem, &changed, None, Some(&initial), &cancel, None)
            .is_err()
    );
    assert_eq!(system.calls.get(), 0);
    let foreign = eqiora_time::TimeBackendIdentity::new("test.foreign", "1");
    assert!(
        initial
            .validate_for(&problem, &plan, foreign, None)
            .is_err()
    );
    assert_eq!(
        DiffsolTimeBackend::new()
            .advance(&problem, &plan, None, Some(&initial), &cancel, None)
            .unwrap(),
        ControlFlow::Break(initial)
    );
}

fn outcome_bits(outcome: &eqiora_time::TimeRootOutcome) -> Vec<u64> {
    let mut values = Vec::new();
    for step in outcome.history().steps() {
        values.extend([step.start_time().to_bits(), step.end_time().to_bits()]);
        values.extend(
            step.start_state()
                .iter()
                .chain(step.midpoint_state())
                .chain(step.end_state())
                .map(|value| value.to_bits()),
        );
    }
    if let Some(samples) = outcome.samples() {
        for (index, time) in samples.times().iter().enumerate() {
            values.push(time.to_bits());
            values.extend(
                samples
                    .state(index)
                    .unwrap()
                    .iter()
                    .map(|value| value.to_bits()),
            );
        }
    }
    if let Some(proposal) = outcome.proposal() {
        values.push(proposal.time().to_bits());
        values.extend(proposal.state().iter().map(|value| value.to_bits()));
    }
    values
}

struct Threshold;
impl eqiora_time::RootFunctions for Threshold {
    fn count(&self) -> usize {
        1
    }
    fn evaluate(&self, _: f64, state: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        output[0] = state[0] - 0.5;
        Ok(())
    }
}

#[test]
fn resumed_native_prefix_localizes_the_same_registered_event() {
    use eqiora_time::{
        RegisteredRootProblem, RootActivationGroup, RootRegistrationId, RootRegistrationProof,
    };
    let proof = RootRegistrationProof::new(vec![
        RootActivationGroup::new(vec![eqiora_core::Id::<
            eqiora_core::entity::kinds::Activation,
        >::new()])
        .unwrap(),
    ])
    .unwrap();
    let roots =
        RegisteredRootProblem::new(RootRegistrationId::from_sha256([19; 32]), proof, &Threshold)
            .unwrap();
    let backend = DiffsolTimeBackend::new();
    let plan = plan();
    let cancel = Arc::new(AtomicBool::new(false));
    let system = ForcedDecay::new(cancel.clone(), usize::MAX, false);
    let baseline = backend
        .solve_until_root(&problem(&system), &roots, &plan)
        .unwrap();
    let interrupted = ForcedDecay::new(cancel.clone(), 25, false);
    let problem = problem(&interrupted);
    let saved = backend
        .advance(&problem, &plan, Some(&roots), None, &cancel, None)
        .unwrap()
        .break_value()
        .unwrap();
    assert!(saved.time() > 0.0 && saved.time() < baseline.proposal().unwrap().time());
    assert!(
        saved
            .validate_for(&problem, &plan, DiffsolTimeBackend::IDENTITY, None)
            .is_err()
    );
    cancel.store(false, Ordering::Release);
    let resumed = backend
        .advance(&problem, &plan, Some(&roots), Some(&saved), &cancel, None)
        .unwrap()
        .continue_value()
        .unwrap();
    assert_eq!(resumed, baseline);
    assert_eq!(outcome_bits(&resumed), outcome_bits(&baseline));
    let event = resumed.proposal().unwrap();
    let t = event.time();
    // Independently evaluate the closed-form trajectory at the localized guard.
    let exact = (627.0 * (-25.0 * t).exp() + 25.0 * t.sin() - t.cos()) / 626.0;
    assert!((exact - 0.5).abs() < 2e-8);
}
