//! Explicit whole-owner arbitration, independent of declaration and graph order.
use eqiora::api::ModelDocument;
use eqiora::sem::{Interpreter, ReferenceConfig};

fn model(events: &[&str]) -> ModelDocument {
    ModelDocument::compile(
        "event-priority.eqi",
        &format!(
            "model Priority(){{state x:1;state y:1;state z:1;\
             initial{{x=0;y=0;z=0;}}\
             relation flow{{derivative(x)=0[1/s];derivative(y)=0[1/s];derivative(z)=0[1/s];}}\
             {}}}",
            events.join("\n")
        ),
    )
    .unwrap()
}

fn config() -> ReferenceConfig {
    ReferenceConfig::new(2.0, 0.25).unwrap()
}

const HIGH: &str =
    "event high=crossing(time()-1[s],direction=rising,priority=2);relation a at high{next(x)=1;}";
const MIDDLE: &str = "event middle=crossing(time()-1[s],direction=rising,priority=1);relation b at middle{next(x)=2;next(y)=2;next(z)=2;}";
const LOW: &str =
    "event low=crossing(time()-1[s],direction=rising,priority=-1);relation c at low{next(y)=3;}";

#[test]
fn descending_groups_suppress_whole_owners_without_transitive_conflict_groups() {
    // High claims x; middle loses all of x/y/z; low is still free to claim y.
    // Choosing one owner per connected conflict component would lose low.
    for events in [
        [HIGH, MIDDLE, LOW],
        [LOW, HIGH, MIDDLE],
        [MIDDLE, LOW, HIGH],
    ] {
        let model = model(&events);
        let replay = ModelDocument::replay(&model.canonical_json().unwrap()).unwrap();
        for program in [model.program(), replay.program()] {
            let interpreter = Interpreter::new();
            let mut session = interpreter
                .execution_session(program, config(), [])
                .unwrap();
            let mut boundaries = 0;
            while session.advance().unwrap() {
                if !session.activation_sequence().is_empty() {
                    boundaries += 1;
                    let actual = session
                        .activation_sequence()
                        .iter()
                        .flatten()
                        .copied()
                        .collect::<std::collections::BTreeSet<_>>();
                    let expected = [model.aliases()["high"], model.aliases()["low"]]
                        .into_iter()
                        .collect();
                    assert_eq!(actual, expected);
                    assert_eq!(session.activation_sequence().len(), 1);
                    let checkpoint = session.checkpoint();
                    session = interpreter.resume_execution(program, &checkpoint).unwrap();
                }
            }
            assert_eq!(
                boundaries, 1,
                "suppressed owners must not immediately replay"
            );
            for (name, expected) in [("x", 1.0), ("y", 3.0), ("z", 0.0)] {
                assert_eq!(
                    session
                        .field(model.aliases()[name])
                        .unwrap()
                        .real_scalar_value()
                        .unwrap()
                        .value(),
                    expected
                );
            }
        }
    }
}

#[test]
fn equal_priority_conflicts_only_among_surviving_owners() {
    let peer = MIDDLE
        .replace("middle", "peer")
        .replace("relation b", "relation peer_reset");
    let document = model(&[HIGH, MIDDLE, &peer]);
    let mut session = Interpreter::new()
        .execution_session(document.program(), config(), [])
        .unwrap();
    while session.advance().unwrap() {}
    assert_eq!(
        session
            .field(document.aliases()["x"])
            .unwrap()
            .real_scalar_value()
            .unwrap()
            .value(),
        1.0
    );
    assert_eq!(
        session
            .field(document.aliases()["y"])
            .unwrap()
            .real_scalar_value()
            .unwrap()
            .value(),
        0.0
    );

    for events in [[MIDDLE, peer.as_str()], [peer.as_str(), MIDDLE]] {
        let document = model(&events);
        let mut session = Interpreter::new()
            .execution_session(document.program(), config(), [])
            .unwrap();
        for _ in 0..3 {
            assert!(session.advance().unwrap());
        }
        let before = session.progress();
        let errors = session.advance().unwrap_err();
        assert!(
            errors[0].message().contains("equal event priority"),
            "{errors:?}"
        );
        assert_eq!(session.progress(), before);
        assert_eq!(
            session
                .field(document.aliases()["x"])
                .unwrap()
                .real_scalar_value()
                .unwrap()
                .value(),
            0.0
        );
    }
}

#[test]
fn inconsistent_winner_rolls_back_instead_of_falling_back_to_loser() {
    let broken = HIGH.replace("next(x)=1;", "next(x)=1;next(x)=4;");
    let document = model(&[&broken, MIDDLE]);
    let mut session = Interpreter::new()
        .execution_session(document.program(), config(), [])
        .unwrap();
    for _ in 0..3 {
        assert!(session.advance().unwrap());
    }
    let before = session.progress();
    assert!(session.advance().is_err());
    assert_eq!(session.progress(), before);
    for name in ["x", "y", "z"] {
        assert_eq!(
            session
                .field(document.aliases()[name])
                .unwrap()
                .real_scalar_value()
                .unwrap()
                .value(),
            0.0
        );
    }
}

#[test]
fn priority_is_required_persisted_meaning_and_source_identity() {
    let original = model(&[HIGH]);
    let changed = model(&[&HIGH.replace("priority=2", "priority=3")]);
    assert_ne!(original.aliases()["high"], changed.aliases()["high"]);
    assert!(!original.structurally_equivalent(&changed).unwrap());
    let omitted = model(&[&HIGH.replace(",priority=2", "")]);
    let explicit_zero = model(&[&HIGH.replace("priority=2", "priority=0")]);
    assert!(omitted.structurally_equivalent(&explicit_zero).unwrap());
    assert_eq!(
        omitted.canonical_json().unwrap(),
        explicit_zero.canonical_json().unwrap()
    );
    let bytes = original.canonical_json().unwrap();
    let text = String::from_utf8(bytes).unwrap();
    assert!(text.contains("\"priority\":2"));
    assert!(
        ModelDocument::replay(
            text.replace("\"priority\":2,", "")
                .replace(",\"priority\":2", "")
                .as_bytes()
        )
        .is_err()
    );
    assert!(
        ModelDocument::replay(
            text.replace("model-envelope/v28", "model-envelope/v27")
                .as_bytes()
        )
        .is_err()
    );
}

#[test]
fn suppressed_event_can_rearm_for_a_later_genuine_crossing() {
    let later = "event later=crossing((time()-1[s])*(time()-2[s])*(time()-3[s]),direction=rising,priority=1);relation later_reset at later{next(x)=2;}";
    let document = model(&[HIGH, later]);
    let interpreter = Interpreter::new();
    let mut session = interpreter
        .execution_session(
            document.program(),
            ReferenceConfig::new(3.5, 0.25).unwrap(),
            [],
        )
        .unwrap();
    let mut events = Vec::new();
    while session.advance().unwrap() {
        if !session.activation_sequence().is_empty() {
            events.push((
                session.progress().model_time(),
                session.activation_sequence().to_vec(),
                session
                    .field(document.aliases()["x"])
                    .unwrap()
                    .real_scalar_value()
                    .unwrap()
                    .value(),
            ));
            session = interpreter
                .resume_execution(document.program(), &session.checkpoint())
                .unwrap();
        }
    }
    // The cubic is negative before1, positive between1/2, negative between2/3.
    // Rising roots are1 and3; the priority winner consumes only the first one.
    assert_eq!(
        events,
        vec![
            (1.0, vec![vec![document.aliases()["high"]]], 1.0),
            (3.0, vec![vec![document.aliases()["later"]]], 2.0)
        ]
    );
}

#[test]
fn canonical_execution_rejects_priority_on_a_grouped_peer() {
    use eqiora::artifact::{ModelEnvelope, RootRegistrationEnvelopeV1, TimeLoweringEnvelopeV1};
    use eqiora::runtime::{CanonicalEventProgram, CpuProgram, FirstOrderProgram};
    let source = "model M(){state x:1;initial{x=0;}relation flow{derivative(x)=1[1/s];}event a=crossing(x-1,direction=rising);event b=crossing(x-1,direction=rising,priority=1);relation reset at b{next(x)=0;}}";
    for priority in [0, 1] {
        let document = ModelDocument::compile(
            "group.eqi",
            &source.replace("priority=1", &format!("priority={priority}")),
        )
        .unwrap();
        let cpu = CpuProgram::lower(document.program()).unwrap();
        let flow = document.aliases()["flow"].downcast().unwrap();
        let first = FirstOrderProgram::lower(&cpu, flow).unwrap();
        let model = ModelEnvelope::from_program(document.program()).unwrap();
        let lowering =
            TimeLoweringEnvelopeV1::from_proof(&model, document.program(), first.lowering_proof())
                .unwrap();
        let registration = RootRegistrationEnvelopeV1::new(&model, document.program(), &lowering);
        for owner in ["a", "b"] {
            let event = CanonicalEventProgram::lower(
                &cpu,
                flow,
                document.aliases()[owner].downcast().unwrap(),
            );
            if priority == 0 {
                assert!(event.is_ok(), "{event:?}");
            } else {
                assert!(
                    event
                        .unwrap_err()
                        .message()
                        .contains("does not support event priority")
                );
            }
        }
        if priority == 0 {
            assert!(registration.is_ok(), "{registration:?}");
        } else {
            assert!(
                registration
                    .unwrap_err()
                    .message()
                    .contains("does not support event priority")
            );
        }
    }
}
