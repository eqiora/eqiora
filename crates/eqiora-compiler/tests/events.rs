use eqiora_compiler::compile;
use eqiora_graph::Op;
use eqiora_schema::kernel::{ActivationKind, EventDirection, KernelNode};

#[test]
fn declared_event_has_one_identity_shared_by_reset_relations() {
    let source = "model M(){state x:m;event hit=crossing(x,direction=falling);relation a at hit{next(x)=0[m];}relation b at hit{pre(x)=0[m];}}";
    let models = compile("event.eqi", source).unwrap();
    let events = models[0]
        .transaction()
        .ops()
        .iter()
        .filter_map(|op| match op {
            Op::DefineKernelNode {
                node: KernelNode::Activation(value),
            } if matches!(
                value.kind(),
                ActivationKind::Event {
                    priority: 0,
                    direction: EventDirection::Falling,
                    ..
                }
            ) =>
            {
                Some(value.id())
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(events.len(), 1);
    let activations = models[0]
        .transaction()
        .ops()
        .iter()
        .filter(|op| {
            matches!(
                op,
                Op::DefineKernelNode {
                    node: KernelNode::Activation(_)
                }
            )
        })
        .count();
    assert_eq!(activations, 1);
    let rising = compile("event.eqi", &source.replace("falling", "rising")).unwrap();
    let rising_event = rising[0]
        .transaction()
        .ops()
        .iter()
        .find_map(|op| match op {
            Op::DefineKernelNode {
                node: KernelNode::Activation(value),
            } => Some(value.id()),
            _ => None,
        })
        .unwrap();
    assert_ne!(events[0], rising_event);
}

#[test]
fn guard_and_activation_kind_are_checked_before_lowering() {
    for guard in ["true", "[1,2]", "math.complex(1,2)", "pre(x)", "next(x)"] {
        let source = format!(
            "model M(){{state x:m;event hit=crossing({guard},direction=any);relation r at hit{{next(x)=0[m];}}}}"
        );
        let errors = compile("invalid-event.eqi", &source).unwrap_err();
        assert!(
            errors.iter().all(|error| error.source_span().is_some()),
            "{errors:?}"
        );
    }
    for declaration in [
        "state y:m at hit;",
        "variable y:m at hit;",
        "event hit=crossing(x,direction=rising);",
    ] {
        let source = format!(
            "model M(){{state x:m;event hit=crossing(x,direction=falling);{declaration}relation r at hit{{next(x)=0[m];}}}}"
        );
        assert!(
            compile("invalid-activation.eqi", &source).is_err(),
            "{declaration}"
        );
    }
}
