#![cfg(feature = "diffsol")]

use eqiora::api::ModelDocument;
use eqiora::artifact::ModelEnvelope;
use eqiora::backends::diffsol::DiffsolTimeBackend;
use eqiora::runtime::FirstOrderProgram;
use eqiora::time::{InitialConditionPolicy, TimeMethod, TimeProblem, TimeSystem};
use eqiora::{Diagnostic, DimExponents, DynQuantity};
use eqiora_numerics::{
    CommonOdePlan, CommonOdePolicy, CommonTimeCheckpoint, CommonTimeRunRequest, CommonTrajectory,
    ResolvedCommonPlan,
};
use std::ops::ControlFlow;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

fn fixture(events: Option<usize>, relative_tolerance: f64) -> CommonTimeRunRequest {
    let mut source =
        "model Decay(){state x:1;initial{x=1;}relation flow{derivative(x)=-1[1/s]*x;}".to_owned();
    if events.is_some() {
        source.push_str(
            "event hit=crossing(x-0.5,direction=falling);relation reset at hit{next(x)=1;}",
        );
    }
    source.push('}');
    let document = ModelDocument::compile("continuation.eqi", &source).unwrap();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let flow = FirstOrderProgram::lower(
        document.program(),
        document.aliases()["flow"].downcast().unwrap(),
    )
    .unwrap();
    let mut policy = CommonOdePolicy::new(
        TimeMethod::Tsitouras45,
        1e-3,
        relative_tolerance,
        flow.state_coordinates()
            .iter()
            .map(|&coordinate| (coordinate, 1e-11))
            .collect(),
    )
    .unwrap();
    if let Some(maximum) = events {
        policy = policy
            .with_events(
                maximum,
                vec![(
                    document.aliases()["hit"].downcast().unwrap(),
                    DynQuantity::new(1e-8, DimExponents::DIMENSIONLESS),
                )],
            )
            .unwrap();
    }
    let plan = CommonOdePlan::resolve(
        &model,
        document.program(),
        policy,
        DiffsolTimeBackend::CAPABILITIES,
    )
    .unwrap();
    CommonTimeRunRequest::new(
        plan.clone(),
        plan.initial_state(0.0).unwrap(),
        2.2,
        vec![0.1, 0.5, 0.8, 1.0, 1.5, 2.1],
    )
    .unwrap()
}

struct StopAtTime<'a> {
    system: &'a dyn TimeSystem,
    token: &'a Arc<AtomicBool>,
    time: f64,
}
impl TimeSystem for StopAtTime<'_> {
    fn dimension(&self) -> usize {
        self.system.dimension()
    }
    fn rhs(&self, time: f64, state: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        let result = self.system.rhs(time, state, output);
        if time >= self.time {
            self.token.store(true, Ordering::Release);
        }
        result
    }
    fn rhs_jvp(
        &self,
        time: f64,
        state: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        self.system.rhs_jvp(time, state, direction, output)
    }
}

fn stop(request: &CommonTimeRunRequest, time: f64) -> CommonTimeCheckpoint {
    let token = Arc::new(AtomicBool::new(false));
    request
        .advance(None, |problem, roots, plan, saved, _| {
            let system = StopAtTime {
                system: problem.system(),
                token: &token,
                time,
            };
            let problem = TimeProblem::new(
                &system,
                problem.equation_class(),
                InitialConditionPolicy::Provided,
                problem.initial_state().to_vec(),
            )?;
            DiffsolTimeBackend::new().advance(&problem, plan, roots, saved, &token, None)
        })
        .unwrap()
        .break_value()
        .unwrap()
}

fn finish(
    request: &CommonTimeRunRequest,
    saved: Option<&CommonTimeCheckpoint>,
) -> Result<CommonTrajectory, Diagnostic> {
    let token = Arc::new(AtomicBool::new(false));
    request
        .advance(saved, |problem, roots, plan, saved, _| {
            DiffsolTimeBackend::new().advance(problem, plan, roots, saved, &token, None)
        })?
        .continue_value()
        .ok_or_else(|| {
            Diagnostic::error(
                eqiora::diagnostic::codes::NUMERICAL_SOLVE_FAILED,
                "unexpected checkpoint",
            )
        })
}

fn reopen(saved: CommonTimeCheckpoint) -> CommonTimeCheckpoint {
    let resolved = ResolvedCommonPlan::Time(Box::new((saved.request().plan().clone()).into()));
    let plan_bytes = resolved.to_bytes().unwrap();
    let path = std::env::temp_dir().join(format!(
        "eqiora-adaptive-{}-{}.json",
        std::process::id(),
        saved.identity()
    ));
    std::fs::write(&path, saved.to_bytes().unwrap()).unwrap();
    drop(saved);
    // Reconstruct the Plan and checkpoint through their ordinary codecs, then
    // create a fresh native solver. No live controller or callback survives.
    let plan = ResolvedCommonPlan::from_bytes(
        &plan_bytes,
        &eqiora::solver::REFERENCE_LINEAR_SOLVER,
        DiffsolTimeBackend::CAPABILITIES,
    )
    .unwrap();
    let ResolvedCommonPlan::Time(plan) = plan else {
        panic!("ODE Plan")
    };
    let bytes = std::fs::read(&path).unwrap();
    std::fs::remove_file(path).unwrap();
    let reopened = CommonTimeCheckpoint::from_bytes(&bytes, &plan).unwrap();
    assert_eq!(reopened.to_bytes().unwrap(), bytes);
    reopened
}

#[test]
fn persisted_smooth_and_reset_continuations_reproduce_the_complete_result() {
    for events in [None, Some(4)] {
        let request = fixture(events, 1e-9);
        let baseline = finish(&request, None).unwrap();
        if events.is_some() {
            let ordinary = request
                .run_with_events(|problem, roots, plan| {
                    DiffsolTimeBackend::new().solve_until_root(problem, roots, plan)
                })
                .unwrap();
            assert_eq!(ordinary.to_bytes().unwrap(), baseline.to_bytes().unwrap());
        }
        let saved = stop(&request, 0.95);
        assert!(saved.accepted_steps() > 0);
        assert_eq!(saved.accepted_events(), usize::from(events.is_some()));
        assert!(saved.state().unwrap().time_s() < 0.95);
        assert_eq!(saved.state().unwrap().source_kind(), "checkpoint");
        let state_bytes = saved.state().unwrap().to_bytes().unwrap();
        let state = request
            .ode_plan()
            .unwrap()
            .state_from_bytes(&state_bytes)
            .unwrap();
        assert_eq!(state, saved.state().unwrap());
        let saved = reopen(saved);
        let resumed = finish(saved.request(), Some(&saved)).unwrap();
        assert_eq!(
            resumed.to_bytes().unwrap(),
            baseline.to_bytes().unwrap(),
            "every accepted binary64 stencil, event receipt and requested output must match"
        );
        for state in resumed.time_states().unwrap() {
            let cycles = if events.is_some() {
                (state.time_s() / 2.0_f64.ln()).floor()
            } else {
                0.0
            };
            let exact = (-(state.time_s() - cycles * 2.0_f64.ln())).exp();
            assert!((state.values()[0] - exact).abs() < 2e-8);
        }
        for (index, event) in resumed.ode_history().unwrap().events().iter().enumerate() {
            assert!((event.proposal().time() - (index + 1) as f64 * 2.0_f64.ln()).abs() < 2e-8);
        }
    }
}

#[test]
fn cancellation_at_a_committed_reset_preserves_its_budget_and_post_state() {
    let request = fixture(Some(1), 1e-9);
    let token = Arc::new(AtomicBool::new(false));
    let mut segment = 0;
    let saved = request
        .advance(None, |problem, roots, plan, saved, _| {
            if segment == 1 {
                token.store(true, Ordering::Release);
            }
            segment += 1;
            DiffsolTimeBackend::new().advance(problem, plan, roots, saved, &token, None)
        })
        .unwrap()
        .break_value()
        .unwrap();
    assert_eq!(saved.accepted_events(), 1);
    assert_eq!(saved.state().unwrap().values(), &[1.0]);
    assert!((saved.state().unwrap().time_s() - 2.0_f64.ln()).abs() < 2e-8);
    let saved = reopen(saved);
    assert!(
        finish(saved.request(), Some(&saved))
            .unwrap_err()
            .message()
            .contains("accepted-event budget")
    );
    assert!(
        finish(&request, None)
            .unwrap_err()
            .message()
            .contains("accepted-event budget")
    );
}

#[test]
fn incomplete_foreign_and_forged_checkpoints_reach_their_semantic_rejection() {
    let request = fixture(Some(4), 1e-9);
    let saved = stop(&request, 0.95);
    let bytes = saved.to_bytes().unwrap();
    let original: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let reject = |value: serde_json::Value, message: &str| {
        let error =
            CommonTimeCheckpoint::from_bytes(&serde_json::to_vec(&value).unwrap(), request.plan())
                .unwrap_err();
        assert!(
            error.message().contains(message),
            "expected {message:?}, got {error:?}"
        );
    };
    let mut value = original.clone();
    value["payload"]["segment"]
        .as_object_mut()
        .unwrap()
        .remove("derivative");
    reject(value, "derivative");
    let mut value = original.clone();
    value["payload"]["segment"]["derivative"] = serde_json::json!([]);
    reject(value, "controller state");
    let mut value = original.clone();
    value["payload"]["segment"]["steps"] = serde_json::json!([]);
    reject(value, "samples lack");
    let mut value = original.clone();
    value["payload"]["committed_steps"] = serde_json::json!([]);
    reject(value, "reset receipts lack");
    let mut value = original.clone();
    value["payload"]["committed_events"][0]["after_state"][0] = serde_json::json!(0.75);
    reject(value, "canonical registered reset");
    let mut value = original.clone();
    value["payload"]["segment"]["report"]["backend_version"] = serde_json::json!("foreign");
    reject(value, "backend differs");
    let mut value = original;
    value["payload"]["request"]["plan_identity"] = serde_json::json!("foreign");
    reject(value, "different exact Plan");
    for end in [0, 1, bytes.len() / 2, bytes.len() - 1] {
        assert!(CommonTimeCheckpoint::from_bytes(&bytes[..end], request.plan()).is_err());
    }
    let foreign = fixture(Some(4), 1e-7);
    assert_ne!(request.plan().identity(), foreign.plan().identity());
    assert!(CommonTimeCheckpoint::from_bytes(&bytes, foreign.plan()).is_err());
    let changed = CommonTimeRunRequest::new(
        request.plan().clone(),
        request.state().clone(),
        3.0,
        vec![1.0],
    )
    .unwrap();
    let result: Result<ControlFlow<CommonTimeCheckpoint, CommonTrajectory>, _> = changed
        .advance(Some(&saved), |_, _, _, _, _| {
            panic!("foreign request must reject before backend execution")
        });
    assert!(
        result
            .unwrap_err()
            .message()
            .contains("different exact ODE Run")
    );
}
