use super::*;
use crate::ResolvedCommonPlan;
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_solver::REFERENCE_LINEAR_SOLVER;

const BALL: &str = r#"
model Ball() {
  state height: m;
  state velocity: m/s;
  parameter ground: m = 0;
  parameter restitution: 1 = 0.8;
  initial { height = 1[m]; velocity = 0[m/s]; }
  relation flight { derivative(height) = velocity; derivative(velocity) = -9.81[m/s^2]; }
  event impact_h = crossing(height-ground, direction=falling);
  event impact_v = crossing(height-ground, direction=falling);
  relation reset_h at impact_h { next(height)=ground; }
  relation reset_v at impact_v { next(velocity)=-restitution*pre(velocity); }
}
"#;
fn fixture(source: &str) -> (ModelEnvelope, KernelProgram) {
    let compiled = eqiora_compiler::compile("event-policy.eqi", source)
        .unwrap()
        .pop()
        .unwrap();
    let (transaction, model, _) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let kernel = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    (ModelEnvelope::from_program(&kernel).unwrap(), kernel)
}
fn temporal(kernel: &KernelProgram) -> CommonTsitouras45 {
    CommonTsitouras45::new(
        1e-3,
        1e-9,
        kernel
            .nodes()
            .filter_map(|node| match node {
                KernelNode::Field(field) => {
                    Some(CommonTsitourasTolerance::new(field.id(), 1e-11).unwrap())
                }
                _ => None,
            })
            .collect(),
    )
    .unwrap()
}
fn entries(kernel: &KernelProgram) -> Vec<CommonGuardTolerance> {
    kernel
        .nodes()
        .filter_map(|node| match node {
            KernelNode::Activation(activation)
                if matches!(activation.kind(), ActivationKind::Event { .. }) =>
            {
                Some(
                    CommonGuardTolerance::new(
                        activation.id(),
                        DynQuantity::new(
                            1e-8,
                            DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap(),
                        ),
                    )
                    .unwrap(),
                )
            }
            _ => None,
        })
        .collect()
}
fn backend() -> TimeBackendIdentity {
    TimeBackendIdentity::new("eqiora.test.time", "1")
}

#[test]
fn event_policy_authenticates_group_units_plan_bytes_and_closed_execution_entry() {
    let (model, kernel) = fixture(BALL);
    let controls = temporal(&kernel);
    assert!(CommonOdePlan::resolve(&model, &kernel, controls.clone(), backend()).is_err());
    let policy = CommonEventPolicy::new(8, entries(&kernel)).unwrap();
    let plan = CommonOdePlan::resolve(
        &model,
        &kernel,
        controls.clone().with_event_policy(policy.clone()),
        backend(),
    )
    .unwrap();
    assert_eq!(plan.field_ids().len(), 2);
    let roots = plan.root_set().unwrap().unwrap();
    assert_eq!(roots.events().len(), 1);
    assert_eq!(roots.events()[0].activations().len(), 2);
    assert_eq!(
        roots.events()[0].parameter_fields().len(),
        2,
        "guard/reset-only real Parameters stay in event columns"
    );
    assert_eq!(
        plan.guard_tolerance(0).unwrap(),
        policy.guard_tolerances()[0].quantity()
    );
    assert!(plan.guard_tolerance(1).is_err());
    let resolved = ResolvedCommonPlan::Ode(Box::new(plan.clone()));
    let bytes = resolved.to_bytes().unwrap();
    assert_eq!(
        ResolvedCommonPlan::from_bytes(&bytes, &REFERENCE_LINEAR_SOLVER, backend()).unwrap(),
        resolved
    );
    let changed = CommonOdePlan::resolve(
        &model,
        &kernel,
        controls.with_event_policy(CommonEventPolicy::new(9, entries(&kernel)).unwrap()),
        backend(),
    )
    .unwrap();
    assert_ne!(plan.identity(), changed.identity());
    assert_eq!(plan.state_space_identity(), changed.state_space_identity());
    let request =
        CommonOdeRunRequest::new(plan.clone(), plan.initial_state().unwrap(), 1.0, vec![1.0])
            .unwrap();
    assert!(
        request
            .problem()
            .err()
            .unwrap()
            .message()
            .contains("event driver")
    );
    assert!(
        request
            .forward_sensitivity_problem()
            .err()
            .unwrap()
            .message()
            .contains("event sensitivity driver")
    );
    let mut wire: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    wire["temporal"]["events"]["max_events"] = serde_json::json!(9);
    assert!(
        ResolvedCommonPlan::from_bytes(
            &serde_json::to_vec(&wire).unwrap(),
            &REFERENCE_LINEAR_SOLVER,
            backend()
        )
        .is_err()
    );
}

#[test]
fn event_policy_rejects_missing_foreign_dimension_and_group_tolerance_drift() {
    let (model, kernel) = fixture(BALL);
    let exact = entries(&kernel);
    let controls = temporal(&kernel);
    assert!(CommonEventPolicy::new(0, exact.clone()).is_err());
    assert!(CommonEventPolicy::new(8, vec![]).is_err());
    assert!(CommonEventPolicy::new(8, vec![exact[0], exact[0]]).is_err());
    for entries in [
        vec![exact[0]],
        vec![
            exact[0],
            CommonGuardTolerance::new(Id::new(), exact[1].quantity()).unwrap(),
        ],
        vec![
            exact[0],
            CommonGuardTolerance::new(
                exact[1].activation(),
                DynQuantity::new(1e-8, DimExponents::DIMENSIONLESS),
            )
            .unwrap(),
        ],
        vec![
            exact[0],
            CommonGuardTolerance::new(
                exact[1].activation(),
                DynQuantity::new(2e-8, exact[1].quantity().dim()),
            )
            .unwrap(),
        ],
    ] {
        let policy = CommonEventPolicy::new(8, entries).unwrap();
        assert!(
            CommonOdePlan::resolve(
                &model,
                &kernel,
                controls.clone().with_event_policy(policy),
                backend()
            )
            .is_err()
        );
    }
    let (smooth_model, smooth_kernel) =
        fixture("model Smooth(){state x:1;initial{x=1;}relation flow{derivative(x)=-1[1/s]*x;}}");
    assert!(
        CommonOdePlan::resolve(
            &smooth_model,
            &smooth_kernel,
            temporal(&smooth_kernel).with_event_policy(CommonEventPolicy::new(8, exact).unwrap()),
            backend()
        )
        .is_err()
    );
}
