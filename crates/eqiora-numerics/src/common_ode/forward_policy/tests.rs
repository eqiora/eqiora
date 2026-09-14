use super::*;
use crate::ResolvedCommonPlan;
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_solver::REFERENCE_LINEAR_SOLVER;

fn fixture(
    events: bool,
) -> (
    ModelEnvelope,
    KernelProgram,
    CommonTsitouras45,
    Vec<CommonSensitivityTolerance>,
) {
    let source = if events {
        "model M(){state x:m;parameter rate:m/s=1;parameter threshold:m=0.4;initial{x=0[m];}relation flow{derivative(x)=rate;}event hit=crossing(x-threshold,direction=rising);relation reset at hit{next(x)=0[m];}}"
    } else {
        "model M(){state x:m;parameter rate:m/s=1;initial{x=0[m];}relation flow{derivative(x)=rate;}}"
    };
    let compiled = eqiora_compiler::compile("forward-policy.eqi", source)
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
            KernelNode::Field(field) => Some(field),
            _ => None,
        })
        .unwrap();
    let mut temporal = CommonTsitouras45::new(
        0.001,
        1e-9,
        vec![CommonTsitourasTolerance::new(field.id(), 1e-11).unwrap()],
    )
    .unwrap();
    if events {
        let activation = kernel
            .nodes()
            .find_map(|node| match node {
                KernelNode::Activation(activation)
                    if matches!(activation.kind(), ActivationKind::Event { .. }) =>
                {
                    Some(activation.id())
                }
                _ => None,
            })
            .unwrap();
        temporal = temporal.with_event_policy(
            CommonEventPolicy::new(
                4,
                vec![
                    CommonGuardTolerance::new(
                        activation,
                        DynQuantity::new(1e-9, field.dimension()),
                    )
                    .unwrap(),
                ],
            )
            .unwrap(),
        );
    }
    let entries = kernel
        .nodes()
        .filter_map(|node| match node {
            KernelNode::Parameter(parameter) => Some(parameter.id()),
            _ => None,
        })
        .enumerate()
        .map(|(index, parameter)| {
            let dimension = field
                .dimension()
                .div(
                    kernel
                        .typed_value(parameter.into())
                        .unwrap()
                        .value_type()
                        .dimension(),
                )
                .unwrap();
            CommonSensitivityTolerance::new(
                field.id(),
                parameter,
                DynQuantity::new((index + 1) as f64 * 1e-11, dimension),
            )
            .unwrap()
        })
        .collect();
    (model, kernel, temporal, entries)
}
fn resolve(
    model: &ModelEnvelope,
    kernel: &KernelProgram,
    temporal: CommonTsitouras45,
    entries: Vec<CommonSensitivityTolerance>,
) -> Result<CommonOdePlan, Diagnostic> {
    CommonOdePlan::resolve(
        model,
        kernel,
        temporal.with_forward_sensitivity_policy(CommonForwardSensitivity::new(1e-9, entries)?),
        TimeBackendIdentity::new("test.forward", "1"),
    )
}
#[test]
fn controls_bind_units_global_event_coordinates_and_exact_plan_replay() {
    for events in [false, true] {
        let (model, kernel, temporal, entries) = fixture(events);
        let plain = CommonOdePlan::resolve(
            &model,
            &kernel,
            temporal.clone(),
            TimeBackendIdentity::new("test.forward", "1"),
        )
        .unwrap();
        let plan = resolve(&model, &kernel, temporal.clone(), entries.clone()).unwrap();
        assert_ne!(plain.identity(), plan.identity());
        assert_eq!(plain.state_space_identity(), plan.state_space_identity());
        assert_eq!(plan.parameter_ids().len(), if events { 2 } else { 1 });
        let expected = plan
            .parameter_ids()
            .iter()
            .map(|id| {
                entries
                    .iter()
                    .find(|entry| entry.parameter() == *id)
                    .unwrap()
                    .quantity()
                    .value()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            plan.forward_sensitivity_plan()
                .unwrap()
                .absolute_tolerances(),
            expected
        );
        let resolved = ResolvedCommonPlan::Ode(Box::new(plan));
        let bytes = resolved.to_bytes().unwrap();
        let reopened = ResolvedCommonPlan::from_bytes(
            &bytes,
            &REFERENCE_LINEAR_SOLVER,
            TimeBackendIdentity::new("test.forward", "1"),
        )
        .unwrap();
        assert_eq!(reopened, resolved);
        assert_eq!(reopened.to_bytes().unwrap(), bytes);
        let mut changed = entries.clone();
        let entry = changed[0];
        changed[0] = CommonSensitivityTolerance::new(
            entry.field(),
            entry.parameter(),
            DynQuantity::new(entry.quantity().value() * 2.0, entry.quantity().dim()),
        )
        .unwrap();
        assert_ne!(
            resolve(&model, &kernel, temporal, changed)
                .unwrap()
                .identity(),
            resolved.identity()
        );
    }
}
#[test]
fn incomplete_foreign_duplicate_and_wrong_dimension_controls_fail_closed() {
    let (model, kernel, temporal, entries) = fixture(true);
    assert!(resolve(&model, &kernel, temporal.clone(), entries[..1].to_vec()).is_err());
    let entry = entries[0];
    assert!(CommonForwardSensitivity::new(1e-9, vec![entry, entry]).is_err());
    assert!(CommonForwardSensitivity::new(f64::NAN, entries.clone()).is_err());
    assert!(
        CommonSensitivityTolerance::new(
            entry.field(),
            entry.parameter(),
            DynQuantity::new(0.0, entry.quantity().dim())
        )
        .is_err()
    );
    let mut wrong = entries.clone();
    wrong[0] = CommonSensitivityTolerance::new(
        entry.field(),
        entry.parameter(),
        DynQuantity::new(
            1e-11,
            DimExponents::from_integers([1, 1, 1, 0, 0, 0, 0]).unwrap(),
        ),
    )
    .unwrap();
    assert!(resolve(&model, &kernel, temporal.clone(), wrong).is_err());
    let mut foreign = entries;
    foreign[0] =
        CommonSensitivityTolerance::new(entry.field(), Id::new(), entry.quantity()).unwrap();
    assert!(resolve(&model, &kernel, temporal, foreign).is_err());
}
