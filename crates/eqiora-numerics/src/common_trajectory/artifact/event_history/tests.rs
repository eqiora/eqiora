use super::*;
use crate::common_ode::{CommonEventPolicy, CommonGuardTolerance};
use crate::{CommonOdePlan, CommonTsitouras45, CommonTsitourasTolerance};
use eqiora_artifact::ModelEnvelope;
use eqiora_core::DynQuantity;
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_schema::kernel::{ActivationKind, KernelNode};
use eqiora_sem::KernelProgram;
use eqiora_time::TimeBackendIdentity;

fn fixture(max_events: usize) -> CommonOdePlan {
    let source = "model M(){state x:1;initial{x=1;}relation flow{derivative(x)=-1[1/s];}event zero=crossing(x,direction=falling);relation reset at zero{next(x)=1;}}";
    let compiled = eqiora_compiler::compile("event-codec.eqi", source)
        .unwrap()
        .pop()
        .unwrap();
    let (transaction, model, _) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let kernel = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let model = ModelEnvelope::from_program(&kernel).unwrap();
    let fields = kernel
        .nodes()
        .filter_map(|node| match node {
            KernelNode::Field(field) => {
                Some(CommonTsitourasTolerance::new(field.id(), 1e-11).unwrap())
            }
            _ => None,
        })
        .collect();
    let guards = kernel
        .nodes()
        .filter_map(|node| match node {
            KernelNode::Activation(activation)
                if matches!(activation.kind(), ActivationKind::Event { .. }) =>
            {
                Some(
                    CommonGuardTolerance::new(
                        activation.id(),
                        DynQuantity::new(1e-8, eqiora_core::DimExponents::DIMENSIONLESS),
                    )
                    .unwrap(),
                )
            }
            _ => None,
        })
        .collect();
    CommonOdePlan::resolve(
        &model,
        &kernel,
        CommonTsitouras45::new(1e-3, 1e-9, fields)
            .unwrap()
            .with_event_policy(CommonEventPolicy::new(max_events, guards).unwrap()),
        TimeBackendIdentity::new("eqiora.test.time", "1"),
    )
    .unwrap()
}
fn event(plan: &CommonOdePlan) -> TimeEventDiscontinuity {
    let roots = plan.root_set().unwrap().unwrap();
    let proposal = RootProposal::accepted(
        roots.registration(),
        1.0,
        0,
        1,
        vec![0.0],
        1,
        TimeExecutionReport::new(
            plan.backend(),
            TimeMethod::Tsitouras45,
            TimeEquationClass::ExplicitOde,
            InitialConditionPolicy::Provided,
        ),
    )
    .unwrap();
    TimeEventDiscontinuity::accepted(proposal, vec![1.0]).unwrap()
}
#[test]
fn exact_event_receipt_round_trips_all_proposal_fields_and_reset_sides() {
    let plan = fixture(2);
    let event = event(&plan);
    let wire = encode(std::slice::from_ref(&event)).unwrap();
    let bytes = serde_json::to_vec(&wire).unwrap();
    let decoded: Vec<WireEvent> = serde_json::from_slice(&bytes).unwrap();
    let replayed = replay(&decoded, &plan).unwrap();
    assert_eq!(replayed, vec![event]);
    assert_eq!(
        serde_json::to_vec(&encode(&replayed).unwrap()).unwrap(),
        bytes
    );
}
#[test]
fn altered_registration_report_index_shape_and_budget_fail_before_history_acceptance() {
    let plan = fixture(1);
    let wire = encode(&[event(&plan)]).unwrap();
    let mut invalid = wire.clone();
    invalid[0].registration_sha256[0] ^= 1;
    assert!(replay(&invalid, &plan).is_err());
    for version in [false, true] {
        let mut invalid = wire.clone();
        if version {
            invalid[0].report.backend_version = "2".to_owned();
        } else {
            invalid[0].report.backend = "eqiora.other.time".to_owned();
        }
        assert!(replay(&invalid, &plan).is_err());
    }
    let mut invalid = wire.clone();
    invalid[0].root_index = 1;
    assert!(replay(&invalid, &plan).is_err());
    let mut invalid = wire.clone();
    invalid[0].before_state.clear();
    assert!(replay(&invalid, &plan).is_err());
    let mut invalid = wire.clone();
    invalid[0].after_state.push(0.0);
    assert!(replay(&invalid, &plan).is_err());
    let mut invalid = wire.clone();
    invalid.push(wire[0].clone());
    assert!(replay(&invalid, &plan).is_err());
    let mut invalid = wire.clone();
    invalid[0].time = f64::NAN;
    assert!(replay(&invalid, &plan).is_err());
    for (key, value) in [
        ("method", "bdf"),
        ("equation_class", "mass-matrix"),
        ("initial_condition", "consistent"),
    ] {
        let mut json = serde_json::to_value(&wire).unwrap();
        json[0]["report"][key] = serde_json::json!(value);
        assert!(serde_json::from_value::<Vec<WireEvent>>(json).is_err());
    }
}
