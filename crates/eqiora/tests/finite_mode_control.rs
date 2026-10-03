//! Finite modes reuse typed state, event owners and simultaneous reset equations.
use eqiora::api::ModelDocument;
use eqiora::sem::{Interpreter, ReferenceConfig};

const CONTROLLER: &str = r#"
enum Mode {Heating,Cooling,Fault}
model Controller(){
 state mode:Mode; state temperature:K; state command:K/s; state cycles:integer;
 initial{mode=Mode.Heating;temperature=20[K];command=1[K/s];cycles=0;}
 relation flow{
  derivative(temperature)=case mode{Mode.Heating=>command,Mode.Cooling=>command,Mode.Fault=>0[K/s]};
  derivative(command)=0[K/s^2];
 }
 event upper=crossing(temperature-22[K],direction=rising);
 event lower=crossing(temperature-18[K],direction=falling);
 relation cool at upper{
  next(mode)=case pre(mode){Mode.Heating=>Mode.Cooling,Mode.Cooling=>Mode.Cooling,Mode.Fault=>Mode.Fault};
  next(command)=case pre(mode){Mode.Heating=>-1[K/s],Mode.Cooling=>pre(command),Mode.Fault=>pre(command)};
  next(cycles)=case pre(mode){Mode.Heating=>pre(cycles)+1,Mode.Cooling=>pre(cycles),Mode.Fault=>pre(cycles)};
 }
 relation heat at lower{
  next(mode)=case pre(mode){Mode.Heating=>Mode.Heating,Mode.Cooling=>Mode.Heating,Mode.Fault=>Mode.Fault};
  next(command)=case pre(mode){Mode.Heating=>pre(command),Mode.Cooling=>1[K/s],Mode.Fault=>pre(command)};
 }
 // EXTRA_TRANSITIONS
}
"#;
const FAULT: &str = r#"
 event failure=crossing(time()-2.5[s],direction=rising);
 event recovery=crossing(time()-4[s],direction=rising);
 relation fail at failure{next(mode)=Mode.Fault;next(cycles)=0;}
 relation recover at recovery{next(mode)=Mode.Heating;next(command)=1[K/s];}
"#;

fn model(extra: &str) -> ModelDocument {
    ModelDocument::compile(
        "finite-mode.eqi",
        &CONTROLLER.replace("// EXTRA_TRANSITIONS", extra),
    )
    .unwrap()
}

fn config(end: f64) -> ReferenceConfig {
    ReferenceConfig::new(end, 0.25)
        .unwrap()
        .with_nonlinear_tolerances(1e-12, 0.0)
        .unwrap()
        .with_event_tolerances(1e-11, 1e-10)
        .unwrap()
}

#[test]
fn thermostat_and_fault_controller_share_typed_modes_and_restart_semantics() {
    // Each segment has slope +1, -1 or 0 K/s. Integrating those constants
    // gives these event times/temperatures without consulting execution output.
    for (extra, end, expected, final_temperature) in [
        (
            "",
            12.0,
            vec![
                (2.0, 1, 22.0, -1.0, 1),
                (6.0, 0, 18.0, 1.0, 1),
                (10.0, 1, 22.0, -1.0, 2),
            ],
            20.0,
        ),
        (
            FAULT,
            10.0,
            vec![
                (2.0, 1, 22.0, -1.0, 1),
                (2.5, 2, 21.5, -1.0, 0),
                (4.0, 0, 21.5, 1.0, 0),
                (4.5, 1, 22.0, -1.0, 1),
                (8.5, 0, 18.0, 1.0, 1),
            ],
            19.5,
        ),
    ] {
        let model = model(extra);
        let interpreter = Interpreter::new();
        let mut session = interpreter
            .execution_session(model.program(), config(end), [])
            .unwrap();
        let mode = model.aliases()["mode"];
        let temperature = model.aliases()["temperature"];
        let command = model.aliases()["command"];
        let cycles = model.aliases()["cycles"];
        assert_eq!(session.field(mode).unwrap().enum_tag(), Some(0));
        assert_eq!(
            session.field(cycles).unwrap().integer_scalar_value(),
            Some(0)
        );
        let mut transitions = Vec::new();
        // Fewer than 64 accepted quarter-second steps and five events. This
        // budget exceeds accumulated residual, guard and localization errors
        // for these unit-slope, affine segments (no truncation error).
        let tolerance = 64.0 * (1e-12 + 1e-10 + 1e-11);
        while session.advance().unwrap() {
            if !session.activation_sequence().is_empty() {
                transitions.push((
                    session.progress().model_time(),
                    session.field(mode).unwrap().enum_tag().unwrap(),
                    session
                        .field(temperature)
                        .unwrap()
                        .real_scalar_value()
                        .unwrap()
                        .value(),
                    session
                        .field(command)
                        .unwrap()
                        .real_scalar_value()
                        .unwrap()
                        .value(),
                    session
                        .field(cycles)
                        .unwrap()
                        .integer_scalar_value()
                        .unwrap(),
                ));
                // Checkpointing at each transition must retain the new mode,
                // held command, reset memory, and event arming without replay.
                session = interpreter
                    .resume_execution(model.program(), &session.checkpoint())
                    .unwrap();
            }
            if session.field(mode).unwrap().enum_tag() == Some(2) {
                assert!(
                    (session
                        .field(temperature)
                        .unwrap()
                        .real_scalar_value()
                        .unwrap()
                        .value()
                        - 21.5)
                        .abs()
                        <= tolerance
                );
                assert_eq!(
                    session
                        .field(command)
                        .unwrap()
                        .real_scalar_value()
                        .unwrap()
                        .value(),
                    -1.0
                );
                assert_eq!(
                    session.field(cycles).unwrap().integer_scalar_value(),
                    Some(0)
                );
            }
        }
        assert_eq!(
            transitions.len(),
            expected.len(),
            "restart must not repeat transitions"
        );
        for (actual, expected) in transitions.into_iter().zip(expected) {
            assert!((actual.0 - expected.0).abs() <= tolerance);
            assert_eq!(actual.1, expected.1);
            assert!((actual.2 - expected.2).abs() <= tolerance);
            assert!((actual.3 - expected.3).abs() <= tolerance);
            assert_eq!(actual.4, expected.4);
        }
        assert!(
            (session
                .field(temperature)
                .unwrap()
                .real_scalar_value()
                .unwrap()
                .value()
                - final_temperature)
                .abs()
                <= tolerance
        );
    }
}

#[test]
fn competing_mode_transitions_reject_without_declaration_order_priority() {
    let first = "event a=crossing(time()-1[s],direction=rising);relation first at a{next(mode)=Mode.Cooling;}";
    let second = "event b=crossing(time()-1[s],direction=rising);relation second at b{next(mode)=Mode.Fault;}";
    for extra in [format!("{first}{second}"), format!("{second}{first}")] {
        let model = model(&extra);
        let mut session = Interpreter::new()
            .execution_session(model.program(), config(2.0), [])
            .unwrap();
        for _ in 0..3 {
            assert!(session.advance().unwrap());
        }
        let before = session.progress();
        let mode = model.aliases()["mode"];
        let temperature = model.aliases()["temperature"];
        let retained = session.field(temperature).unwrap();
        let errors = session.advance().unwrap_err();
        assert!(
            errors[0]
                .message()
                .contains("conflicting activation ownership"),
            "{errors:?}"
        );
        let owners = errors[0].message();
        for owner in ["a", "b", "mode"] {
            assert!(
                owners.contains(&model.aliases()[owner].to_string()),
                "{owners}"
            );
        }
        assert_eq!(session.progress(), before);
        assert_eq!(session.field(mode).unwrap().enum_tag(), Some(0));
        assert_eq!(session.field(temperature).unwrap(), retained);
    }
}

#[test]
fn inconsistent_reset_cannot_publish_a_new_mode() {
    let model = model(
        "event failure=crossing(time()-1[s],direction=rising);relation fail at failure{next(mode)=Mode.Fault;next(command)=1[K/s];next(command)=-1[K/s];}",
    );
    let mut session = Interpreter::new()
        .execution_session(model.program(), config(2.0), [])
        .unwrap();
    for _ in 0..3 {
        assert!(session.advance().unwrap());
    }
    let before = session.progress();
    assert!(session.advance().is_err());
    assert_eq!(session.progress(), before);
    assert_eq!(
        session.field(model.aliases()["mode"]).unwrap().enum_tag(),
        Some(0)
    );
    assert_eq!(
        session
            .field(model.aliases()["command"])
            .unwrap()
            .real_scalar_value()
            .unwrap()
            .value(),
        1.0
    );
}

#[test]
fn mode_initialization_never_chooses_an_implicit_first_member() {
    let source = CONTROLLER.replace("mode=Mode.Heating;", "");
    let model = ModelDocument::compile("missing-initial-mode.eqi", &source).unwrap();
    let error = Interpreter::new()
        .execution_session(model.program(), config(1.0), [])
        .expect_err("an enum member must be selected explicitly");
    assert!(
        error
            .iter()
            .any(|error| error.message().contains("explicit initial assignment")),
        "{error:?}"
    );
}

#[test]
fn fault_priority_suppresses_the_whole_coincident_thermostat_transition() {
    let model = model(
        "event failure=crossing(time()-2[s],direction=rising,priority=10);relation fail at failure{next(mode)=Mode.Fault;}",
    );
    let interpreter = Interpreter::new();
    let mut session = interpreter
        .execution_session(model.program(), config(3.0), [])
        .unwrap();
    let mut boundaries = 0;
    while session.advance().unwrap() {
        if !session.activation_sequence().is_empty() {
            boundaries += 1;
            assert_eq!(
                session.activation_sequence(),
                &[vec![model.aliases()["failure"]]]
            );
            session = interpreter
                .resume_execution(model.program(), &session.checkpoint())
                .unwrap();
        }
    }
    // Temperature reaches22 at2s. The fault wins over upper, so the complete
    // cooling reset (mode, command and cycles) loses. Fault then holds22 K.
    assert_eq!(boundaries, 1);
    assert_eq!(
        session.field(model.aliases()["mode"]).unwrap().enum_tag(),
        Some(2)
    );
    assert_eq!(
        session
            .field(model.aliases()["cycles"])
            .unwrap()
            .integer_scalar_value(),
        Some(0)
    );
    for (name, expected) in [("temperature", 22.0), ("command", 1.0)] {
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
