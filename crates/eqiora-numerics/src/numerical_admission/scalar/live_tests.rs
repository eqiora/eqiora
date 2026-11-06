use super::*;
use crate::common_result::evidence::{CommonSolveEvidence, CommonSolveHistory};
use eqiora_solver::REFERENCE_LINEAR_SOLVER;

pub(in crate::numerical_admission) fn finalization_count() -> usize {
    STATIC_FINALIZATIONS.with(std::cell::Cell::get)
}

pub(in crate::numerical_admission) fn check_replay_count(plan: &CommonSpatialPlan) {
    let count = || crate::numerical_admission::native::ADMISSION_REPLAYS.with(std::cell::Cell::get);
    let before = count();
    let result = plan.run_result(None, &REFERENCE_LINEAR_SOLVER).unwrap();
    assert_eq!(
        count() - before,
        2,
        "one live entry replay and the unchanged publication replay; no second helper replay"
    );
    let before = count();
    assert_eq!(
        crate::CommonResult::from_bytes(&result.to_bytes().unwrap(), result.plan()).unwrap(),
        result
    );
    assert_eq!(count() - before, 1, "artifact replay remains independent");
    let before = count();
    plan.admission
        .execute_linear(&REFERENCE_LINEAR_SOLVER, None)
        .unwrap();
    assert_eq!(
        count() - before,
        1,
        "direct test admission callers still replay"
    );
}

// Called with independently specified Poisson fixtures; no production mutation hooks.
pub(in crate::numerical_admission) fn check(plan: &CommonSpatialPlan) {
    let count = || STATIC_FINALIZATIONS.with(std::cell::Cell::get);
    let before = count();
    let result = plan.run_result(None, &REFERENCE_LINEAR_SOLVER).unwrap();
    assert_eq!(count() - before, 1, "live execution must finalize once");
    check_attribution(plan, &result);
    let before = count();
    assert_eq!(
        crate::CommonResult::from_bytes(&result.to_bytes().unwrap(), result.plan()).unwrap(),
        result
    );
    assert_eq!(
        count() - before,
        1,
        "artifact replay must independently finalize"
    );

    for mutate in [
        |out: &mut CommonLinearRunOutput<f64>| out.fields[0].2[1] += 1.,
        |out: &mut CommonLinearRunOutput<f64>| out.fields[0].2[0] = 1.,
        |out: &mut CommonLinearRunOutput<f64>| out.fields.clear(),
        |out: &mut CommonLinearRunOutput<f64>| out.fields[0].0 = eqiora_core::Id::new(),
        |out: &mut CommonLinearRunOutput<f64>| out.fields[0].3 = Space::tetrahedral_edge(),
        |out: &mut CommonLinearRunOutput<f64>| out.fields[0].2[1] = f64::NAN,
        |out: &mut CommonLinearRunOutput<f64>| {
            out.fields[0].2.pop();
        },
    ] {
        let mut live = LiveStaticExecution::execute(plan, None, &REFERENCE_LINEAR_SOLVER).unwrap();
        mutate(live.output.as_mut().unwrap());
        assert!(live.publish().is_err(), "corrupted live output must reject");
    }

    let live = LiveStaticExecution::execute(plan, None, &REFERENCE_LINEAR_SOLVER).unwrap();
    let output = live.output.as_ref().unwrap();
    let values = output
        .fields
        .iter()
        .map(|(id, _, values, _)| (id.erase(), values.clone()))
        .collect();
    let mut solve = CommonSolveEvidence::from_report(&output.solve_report);
    solve.initial_residual_norm = f64::from_bits(solve.initial_residual_norm.to_bits() + 1);
    let execution = CommonSpatialExecution {
        solve: CommonSolveHistory::Linear(Box::new(solve)),
        assembly: crate::common_result::evidence::CommonAssemblyEvidence::from_report(
            &output.assembly_report,
        ),
    };
    assert!(
        live.validate(&values, &execution, None)
            .unwrap_err()
            .message()
            .contains("initial residual")
    );
    // Failure retains the owned equation and output for a subsequent honest validation.
    let mut execution = execution;
    execution.solve = CommonSolveHistory::Linear(Box::new(CommonSolveEvidence::from_report(
        &output.solve_report,
    )));
    live.validate(&values, &execution, None).unwrap();
}

pub(in crate::numerical_admission) fn check_initial(
    plan: &CommonSpatialPlan,
    initial: &CommonState,
    changed: &CommonState,
    foreign: &CommonState,
) {
    let result = plan
        .run_result(Some(initial), &REFERENCE_LINEAR_SOLVER)
        .unwrap();
    assert_eq!(
        crate::CommonResult::from_bytes(&result.to_bytes().unwrap(), result.plan()).unwrap(),
        result
    );
    let mut live =
        LiveStaticExecution::execute(plan, Some(initial), &REFERENCE_LINEAR_SOLVER).unwrap();
    live.initial = Some(changed);
    assert!(
        live.publish()
            .err()
            .unwrap()
            .message()
            .contains("initial residual")
    );
    assert!(LiveStaticExecution::execute(plan, Some(foreign), &REFERENCE_LINEAR_SOLVER).is_err());
}

// Observe real execution rather than asserting a global callsite inventory.
fn check_attribution(plan: &CommonSpatialPlan, plain: &crate::CommonResult) {
    use std::sync::{Arc, Mutex};
    use tracing::{Id, Metadata, Subscriber, field::Visit, span};

    #[derive(Default)]
    struct Capture {
        spans: Vec<(String, Option<u64>)>,
        stack: Vec<u64>,
    }
    #[derive(Clone)]
    struct Recorder(Arc<Mutex<Capture>>);
    #[derive(Default)]
    struct Phase(String);
    impl Visit for Phase {
        fn record_debug(&mut self, _: &tracing::field::Field, _: &dyn std::fmt::Debug) {}
        fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
            if field.name() == "phase" {
                self.0 = value.to_owned();
            }
        }
    }
    impl Subscriber for Recorder {
        fn enabled(&self, _: &Metadata<'_>) -> bool {
            true
        }
        fn new_span(&self, attrs: &span::Attributes<'_>) -> Id {
            let mut phase = Phase::default();
            attrs.record(&mut phase);
            let mut capture = self.0.lock().unwrap();
            let parent = attrs.parent().map(Id::into_u64).or_else(|| {
                attrs
                    .is_contextual()
                    .then(|| capture.stack.last().copied())
                    .flatten()
            });
            capture.spans.push((phase.0, parent));
            Id::from_u64(capture.spans.len() as u64)
        }
        fn record(&self, _: &Id, _: &span::Record<'_>) {}
        fn record_follows_from(&self, _: &Id, _: &Id) {}
        fn event(&self, _: &tracing::Event<'_>) {}
        fn enter(&self, id: &Id) {
            self.0.lock().unwrap().stack.push(id.into_u64());
        }
        fn exit(&self, id: &Id) {
            assert_eq!(self.0.lock().unwrap().stack.pop(), Some(id.into_u64()));
        }
    }
    let capture = Arc::new(Mutex::new(Capture::default()));
    let profiled = tracing::subscriber::with_default(Recorder(capture.clone()), || {
        plan.run_result(None, &REFERENCE_LINEAR_SOLVER).unwrap()
    });
    assert_eq!(profiled.field_count(), plain.field_count());
    for field in 0..plain.field_count() {
        assert_eq!(profiled.field(field), plain.field(field));
        assert_eq!(
            profiled.field_block_count(field),
            plain.field_block_count(field)
        );
        for block in 0..plain.field_block_count(field) {
            assert_eq!(
                profiled.field_block(field, block),
                plain.field_block(field, block)
            );
        }
    }
    let capture = capture.lock().unwrap();
    let find = |name: &str| {
        let matches: Vec<_> = capture
            .spans
            .iter()
            .enumerate()
            .filter(|(_, (phase, _))| phase == name)
            .collect();
        assert_eq!(
            matches.len(),
            1,
            "one {name} scope in this ordinary scalar execution"
        );
        matches[0]
    };
    let (acceptance, _) = find("scalar_live_acceptance");
    for child in [
        "scalar_result_packing",
        "scalar_live_validation",
        "scalar_result_publication",
    ] {
        assert_eq!(
            find(child).1.1,
            Some(acceptance as u64 + 1),
            "{child} belongs to live acceptance"
        );
    }
    for phase in [
        "scalar_plan_reauthentication",
        "scalar_structure_preflight",
        "scalar_problem_preparation",
        "scalar_solution_validation",
        "scalar_coordinate_recovery",
    ] {
        assert_eq!(
            find(phase).1.1,
            find("scalar_live_acceptance").1.1,
            "{phase} precedes acceptance"
        );
    }
    assert!(capture.stack.is_empty(), "all scopes close after execution");
}
