//! Explicit Sample/Hold uses initialized State memory and simultaneous tick equations.
use eqiora::api::ModelDocument;
use eqiora::compiler::{ModelSymbols, compile};
use eqiora::graph::{GraphStore, InMemoryGraphStore};
use eqiora::runtime::{CpuExecutor, CpuProgram};
use eqiora::sem::{Interpreter, KernelProgram, ReferenceConfig, Trajectory};

const SOURCE: &str = r#"
model Ramp() {
  clock tick = periodic(1[s], phase = 0.5[s]);
  state x: V;
  state memory: V at tick;
  variable held: V;
  variable before: V at tick;
  variable after: V at tick;
  initial { x = 0[V]; memory = -1[V]; }
  relation ramp { derivative(x) = 1[V/s]; }
  relation output { held = hold(memory); }
  relation update at tick {
    next(memory) = sample(x, tick);
    before = pre(memory);
    after = next(memory);
  }
}
"#;

fn admit(source: &str) -> (KernelProgram, ModelSymbols) {
    let compiled = compile("temporal-conversions.eqi", source)
        .unwrap()
        .pop()
        .unwrap();
    let (transaction, model, symbols) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    (
        KernelProgram::from_snapshot(&store.snapshot(), model).unwrap(),
        symbols,
    )
}

fn at(trajectory: &Trajectory, symbols: &ModelSymbols, name: &str, time: f64) -> f64 {
    let field = symbols.get(name).unwrap();
    trajectory
        .samples()
        .iter()
        .find(|sample| sample.coordinate() == field && sample.time() == time)
        .unwrap_or_else(|| panic!("missing {name} at {time}"))
        .value()
        .value()
}

#[test]
fn delayed_sample_and_hold_observe_initialized_pre_and_committed_post_values() {
    let (program, symbols) = admit(SOURCE);
    // Quarter-second steps and half-second phase are exactly binary representable.
    let config = ReferenceConfig::new(1.75, 0.25).unwrap();
    let reference = Interpreter::new().run(&program, config).unwrap();
    let cpu = CpuExecutor::new()
        .run(&CpuProgram::lower(&program).unwrap(), config)
        .unwrap();
    assert_eq!(reference, cpu);
    // x(t)=t V. No tick occurs before0.5; memory then changes to0.5 and1.5 V.
    for (time, expected) in [
        (0.0, -1.0),
        (0.25, -1.0),
        (0.5, 0.5),
        (0.75, 0.5),
        (1.25, 0.5),
        (1.5, 1.5),
        (1.75, 1.5),
    ] {
        assert!((at(&reference, &symbols, "held", time) - expected).abs() < 1e-10);
    }
    for (time, pre, post) in [(0.5, -1.0, 0.5), (1.5, 0.5, 1.5)] {
        assert!((at(&reference, &symbols, "before", time) - pre).abs() < 1e-10);
        assert!((at(&reference, &symbols, "after", time) - post).abs() < 1e-10);
    }
}

#[test]
fn missing_hold_memory_initialization_rejects_before_execution() {
    let (program, _) = admit(&SOURCE.replace(" memory = -1[V];", ""));
    let config = ReferenceConfig::new(0.25, 0.25).unwrap();
    assert!(
        Interpreter::new()
            .initialize(&program, 0.0, config)
            .is_err()
    );
    assert!(Interpreter::new().run(&program, config).is_err());
}

#[test]
fn sampling_phase_is_model_meaning_while_numerical_steps_are_not() {
    let document = ModelDocument::compile("temporal-conversions.eqi", SOURCE).unwrap();
    let changed = ModelDocument::compile(
        "temporal-conversions.eqi",
        &SOURCE.replace("phase = 0.5[s]", "phase = 0.75[s]"),
    )
    .unwrap();
    assert!(!document.structurally_equivalent(&changed).unwrap());
    let bytes = document.canonical_json().unwrap();
    let identity = document.artifact_reference().unwrap();
    for step in [0.25, 0.125] {
        Interpreter::new()
            .run(
                document.program(),
                ReferenceConfig::new(1.75, step).unwrap(),
            )
            .unwrap();
        assert_eq!(document.artifact_reference().unwrap(), identity);
        assert_eq!(document.canonical_json().unwrap(), bytes);
    }
}

fn coincident_source(reverse: bool, conflict: bool) -> String {
    let mut declarations = vec![
        "clock tick = periodic(1[s], phase = 1[s]);",
        "state x: V; state y: V; state z: V; state memory: V at tick; variable held: V; variable observed: V at tick;",
        "initial { x = 0[V]; y = 0[V]; z = 0[V]; memory = 0[V]; }",
        "relation flow { derivative(x) = 1[V/s]; derivative(y) = 2[V/s]; derivative(z) = 0[V/s]; }",
        "relation output { held = hold(memory); }",
        "relation update at tick { next(memory) = sample(x + y, tick); observed = next(memory); }",
        "event first = crossing(x - 1[V], direction = rising);",
        "event second = crossing(y - 2[V], direction = rising);",
        "event cascade = crossing(x - 5[V], direction = rising);",
        "relation reset_first at first { next(x) = 10[V]; }",
        "relation reset_second at second { next(y) = 20[V]; }",
        "relation reset_cascade at cascade { next(z) = 7[V]; }",
    ];
    if conflict {
        declarations.push("event competing = crossing(2*x - 2[V], direction = rising);");
        declarations.push("relation reset_competing at competing { next(x) = 11[V]; }");
    }
    if reverse {
        declarations.reverse();
    }
    format!("model Coincidence() {{ {} }}", declarations.join("\n"))
}

fn boundary(
    session: &eqiora::sem::ExecutionSession,
    symbols: &ModelSymbols,
) -> (f64, Vec<f64>, Vec<Vec<eqiora::RawId>>) {
    (
        session.progress().model_time(),
        ["x", "y", "z", "memory", "held"]
            .iter()
            .map(|name| {
                session
                    .field(symbols.get(name).unwrap())
                    .unwrap()
                    .real_scalar_value()
                    .unwrap()
                    .value()
            })
            .collect(),
        session.activation_sequence().to_vec(),
    )
}

#[test]
fn coincident_sampler_reads_left_state_and_restart_preserves_stabilized_microsteps() {
    for reverse in [false, true] {
        let (program, symbols) = admit(&coincident_source(reverse, false));
        let config = ReferenceConfig::new(1.25, 0.25).unwrap();
        let interpreter = Interpreter::new();
        let mut session = interpreter
            .execution_session(&program, config, vec![])
            .unwrap();
        for _ in 0..3 {
            assert!(session.advance().unwrap());
        }
        assert_eq!(session.progress().model_time(), 0.75);
        let before = session.checkpoint();
        assert!(session.advance().unwrap());
        let accepted = boundary(&session, &symbols);
        assert_eq!(accepted.0, 1.0);
        // A0 left x=1,y=2; reset right x=10,y=20. The sampler retains1+2,
        // and a subsequent event-only microstep sets z=7 without resampling.
        for (actual, expected) in accepted.1.iter().zip([10.0, 20.0, 7.0, 3.0, 3.0]) {
            assert!((actual - expected).abs() < 1e-10);
        }
        assert_eq!(accepted.2.len(), 2);
        assert_eq!(accepted.2[0].len(), 3);
        assert!(accepted.2[0].contains(&symbols.get("first").unwrap()));
        assert!(accepted.2[0].contains(&symbols.get("second").unwrap()));
        assert_eq!(accepted.2[1], vec![symbols.get("cascade").unwrap()]);
        // A clocked Variable remains present through every microstep at this instant.
        let observed = symbols.get("observed").unwrap();
        let tick_value = session
            .field(observed)
            .expect("tick value survives cascade");
        assert!((tick_value.real_scalar_value().unwrap().value() - 3.0).abs() < 1e-10);
        let after = session.checkpoint();
        let mut resumed_before = interpreter.resume_execution(&program, &before).unwrap();
        assert!(resumed_before.advance().unwrap());
        assert_eq!(boundary(&resumed_before, &symbols), accepted);
        assert_eq!(resumed_before.field(observed), Some(tick_value.clone()));
        let mut resumed_after = interpreter.resume_execution(&program, &after).unwrap();
        assert_eq!(resumed_after.field(observed), Some(tick_value));
        assert!(session.advance().unwrap());
        assert!(resumed_before.advance().unwrap());
        assert!(resumed_after.advance().unwrap());
        let final_boundary = boundary(&session, &symbols);
        assert_eq!(final_boundary.0, 1.25);
        // Presence ends at another physical instant, not at an event-only microstep.
        assert!(session.field(observed).is_none());
        assert!(resumed_before.field(observed).is_none());
        assert!(resumed_after.field(observed).is_none());
        // One quarter second of the unchanged flow after stabilized t=1.
        for (actual, expected) in final_boundary.1.iter().zip([10.25, 20.5, 7.0, 3.0, 3.0]) {
            assert!((actual - expected).abs() < 1e-10);
        }
        assert_eq!(boundary(&resumed_before, &symbols), final_boundary);
        assert_eq!(boundary(&resumed_after, &symbols), final_boundary);
        assert_eq!(resumed_before.next_tick(), session.next_tick());
        assert_eq!(resumed_after.next_tick(), session.next_tick());
        assert!(!session.advance().unwrap());
    }
}

#[test]
fn conflicting_coincident_reset_owners_reject_without_advancing_state_or_calendar() {
    let (program, symbols) = admit(&coincident_source(false, true));
    let mut session = Interpreter::new()
        .execution_session(&program, ReferenceConfig::new(1.25, 0.25).unwrap(), vec![])
        .unwrap();
    for _ in 0..3 {
        assert!(session.advance().unwrap());
    }
    let accepted = boundary(&session, &symbols);
    let tick = session.next_tick();
    let diagnostics = session.advance().unwrap_err();
    let report = format!("{diagnostics:?}");
    for event in ["first", "competing"] {
        assert!(
            report.contains(&symbols.get(event).unwrap().to_string()),
            "{report}"
        );
    }
    assert_eq!(boundary(&session, &symbols), accepted);
    assert_eq!(session.next_tick(), tick);
}
