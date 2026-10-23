use super::*;
use crate::{CommonOdePlan, CommonOdePolicy};
use eqiora_artifact::ModelEnvelope;
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_schema::kernel::KernelNode;
use eqiora_sem::KernelProgram;
use eqiora_time::{
    RootProposal, RootRegistrationId, TimeBackendIdentity, TimeEquationClass,
    TimeEventDiscontinuity, TimeExecutionReport, TimeHistoryStep, TimeMethod,
};

fn report() -> TimeExecutionReport {
    TimeExecutionReport::new(
        TimeBackendIdentity::new("test.history", "1"),
        TimeMethod::Tsitouras45,
        TimeEquationClass::ExplicitOde,
        InitialConditionPolicy::Provided,
    )
}
fn step(start: f64, end: f64, left: f64, right: f64) -> TimeHistoryStep {
    TimeHistoryStep::accepted(
        start,
        end,
        vec![left],
        vec![(left + right) * 0.5],
        vec![right],
        eqiora_time::TimeDenseOutput::from_samples(
            vec![start, start + (end - start) * 0.5, end],
            vec![vec![left], vec![(left + right) * 0.5], vec![right]],
        )
        .unwrap(),
    )
    .unwrap()
}
fn request() -> CommonTimeRunRequest {
    let compiled = eqiora_compiler::compile(
        "boundary.eqi",
        "model M(){state x:1;initial{x=0;}relation flow{derivative(x)=1[1/s];}}",
    )
    .unwrap()
    .pop()
    .unwrap();
    let (transaction, id, _) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let kernel = KernelProgram::from_snapshot(&store.snapshot(), id).unwrap();
    let model = ModelEnvelope::from_program(&kernel).unwrap();
    let field = kernel
        .nodes()
        .find_map(|node| match node {
            KernelNode::Field(field) => Some(field.id()),
            _ => None,
        })
        .unwrap();
    let plan = CommonOdePlan::resolve(
        &model,
        &kernel,
        CommonOdePolicy::new(
            eqiora_time::TimeMethod::Tsitouras45,
            0.01,
            1e-9,
            vec![(
                eqiora_core::TimeStateCoordinate::new(field, 0, 0, false),
                1e-11,
            )],
        )
        .unwrap(),
        eqiora_time::TimeBackendCapabilities::new(
            report().backend_identity(),
            &[eqiora_core::ScalarDomain::Real],
            &[eqiora_core::ScalarType::F64],
        ),
    )
    .unwrap();
    CommonTimeRunRequest::new(
        plan.clone(),
        plan.initial_state(0.0).unwrap(),
        1.0,
        vec![0.5],
    )
    .unwrap()
}
#[test]
fn forged_hidden_terminal_and_replayed_midpoint_samples_are_rejected() {
    let request = request();
    let history = AcceptedTimeHistory::accepted(1, vec![step(0.0, 1.0, 0.0, 1.0)], vec![]).unwrap();
    let solution = |values| {
        TimeSolution::accepted_with_history(1, vec![0.5, 1.0], values, report(), history.clone())
            .unwrap()
    };
    assert!(CommonTrajectory::accept_ode(request.clone(), solution(vec![0.5, 1.0])).is_ok());
    // The terminal sample is required for execution but absent from public output.
    assert!(CommonTrajectory::accept_ode(request.clone(), solution(vec![0.5, 999.0])).is_err());
    assert!(CommonTrajectory::accept_ode(request.clone(), solution(vec![999.0, 1.0])).is_err());
    let forged = request
        .ode_plan()
        .unwrap()
        .accept_state(0.5, vec![999.0], "result")
        .unwrap();
    assert!(CommonTrajectory::accept_ode_states(request, vec![forged], history).is_err());
}
#[test]
fn exact_event_output_uses_post_reset_side_including_at_terminal() {
    let proposal = RootProposal::accepted(
        RootRegistrationId::from_sha256([7; 32]),
        1.0,
        0,
        1,
        vec![1.0],
        1,
        report(),
    )
    .unwrap();
    let event = TimeEventDiscontinuity::accepted(proposal, vec![0.0]).unwrap();
    for steps in [
        vec![step(0.0, 1.0, 0.0, 1.0)],
        vec![step(0.0, 1.0, 0.0, 1.0), step(1.0, 2.0, 0.0, 1.0)],
    ] {
        let history = AcceptedTimeHistory::accepted(1, steps, vec![event.clone()]).unwrap();
        assert!(validate(&history, 1.0, &[0.0]).is_ok());
        assert!(validate(&history, 1.0, &[1.0]).is_err());
        assert!(validate(&history, 0.0, &[999.0]).is_err());
        assert!(validate(&history, 0.5, &[999.0]).is_err());
        // The explicit fixture polynomial also constrains off-stencil values.
        assert!(validate(&history, 0.25, &[0.25]).is_ok());
        assert!(validate(&history, 0.25, &[999.]).is_err());
    }
}

#[test]
fn retained_dense_output_rejects_forged_off_stencil_samples() {
    let dense = eqiora_time::TimeDenseOutput::from_samples(
        vec![0., 0.25, 0.5, 0.75, 1.],
        vec![
            vec![0.],
            vec![1. / 256.],
            vec![1. / 16.],
            vec![81. / 256.],
            vec![1.],
        ],
    )
    .unwrap();
    let step =
        TimeHistoryStep::accepted(0., 1., vec![0.], vec![1. / 16.], vec![1.], dense).unwrap();
    let sample = step.sample(0.125).unwrap();
    // The analytic quartic value is independent; the same degree-five roundoff
    // allowance as the primitive test separates accuracy from exact replay.
    assert!((sample[0] - 1. / 4096.).abs() <= 128. * f64::EPSILON);
    let history = AcceptedTimeHistory::accepted(1, vec![step], vec![]).unwrap();
    assert!(validate(&history, 0.125, &sample).is_ok());
    assert!(validate(&history, 0.125, &[999.]).is_err());
}
