//! Source event declarations preserve explicit reset equations and alias meaning.
use eqiora::compiler::{ModelSymbols, compile};
use eqiora::graph::{GraphStore, InMemoryGraphStore};
use eqiora::kernel::KernelNode;
use eqiora::runtime::{CpuExecutor, CpuProgram};
use eqiora::sem::{Interpreter, KernelProgram, ReferenceConfig, Trajectory};

const BALL: &str = r#"
model Ball() {
  state height: m;
  state velocity: m/s;
  initial { height = 1[m]; velocity = 0[m/s]; }
  relation flight {
    derivative(height) = velocity;
    derivative(velocity) = -9.81[m/s^2];
  }
  event impact = crossing(height, direction = falling);
  let rebound at impact = -0.8 * pre(velocity);
  relation reset_height at impact { next(height) = 0[m]; }
  relation reset_velocity at impact { next(velocity) = rebound; }
}
"#;

fn admit(source: &str) -> (KernelProgram, ModelSymbols) {
    let compiled = compile("authored-events.eqi", source)
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

fn series(trajectory: &Trajectory, symbols: &ModelSymbols, name: &str) -> Vec<(f64, f64)> {
    let field = symbols.get(name).unwrap();
    trajectory
        .samples()
        .iter()
        .filter(|sample| sample.coordinate() == field)
        .map(|sample| (sample.time(), sample.value().value()))
        .collect()
}

fn run(program: &KernelProgram, end: f64) -> Trajectory {
    let config = ReferenceConfig::new(end, 0.01)
        .unwrap()
        .with_event_tolerances(1e-11, 1e-10)
        .unwrap();
    let reference = Interpreter::new().run(program, config).unwrap();
    let cpu = CpuExecutor::new()
        .run(&CpuProgram::lower(program).unwrap(), config)
        .unwrap();
    assert_eq!(reference, cpu);
    reference
}

#[test]
fn source_event_alias_preserves_equations_and_restitution() {
    let explicit = BALL
        .replace("  let rebound at impact = -0.8 * pre(velocity);\n", "")
        .replace(
            "next(velocity) = rebound",
            "next(velocity) = -0.8 * pre(velocity)",
        );
    let (aliased, alias_symbols) = admit(BALL);
    let (direct, direct_symbols) = admit(&explicit);
    assert!(alias_symbols.get("rebound").is_none());
    for program in [&aliased, &direct] {
        assert_eq!(
            program
                .nodes()
                .filter(|node| matches!(node, KernelNode::Field(_)))
                .count(),
            2
        );
        assert_eq!(
            program
                .nodes()
                .filter(|node| matches!(node, KernelNode::Relation(_)))
                .count(),
            4
        );
    }
    let alias_run = run(&aliased, 0.7);
    let direct_run = run(&direct, 0.7);
    for field in ["height", "velocity"] {
        let left = series(&alias_run, &alias_symbols, field);
        let right = series(&direct_run, &direct_symbols, field);
        assert_eq!(left.len(), right.len());
        for ((ta, a), (tb, b)) in left.iter().zip(&right) {
            assert!((ta - tb).abs() < 1e-10);
            assert!((a - b).abs() < 1e-8);
        }
    }
    let velocities = series(&alias_run, &alias_symbols, "velocity");
    let impact = velocities
        .windows(2)
        .find(|pair| pair[0].0 == pair[1].0)
        .unwrap();
    assert!(impact[0].1 < 0.0);
    // The authored restitution law reverses the incoming speed and retains 80%.
    assert!((impact[1].1 + 0.8 * impact[0].1).abs() < 1e-8);
    assert!(impact[1].1 > 0.0);
}

#[test]
fn component_events_keep_their_occurrence_owned_states() {
    let source = format!(
        "{} model Root() {{ instance ball: Ball(); }}",
        BALL.replace("model Ball()", "component Ball()")
    );
    let (program, symbols) = admit(&source);
    let trajectory = run(&program, 0.7);
    let velocities = series(&trajectory, &symbols, "ball.velocity");
    let impact = velocities
        .windows(2)
        .find(|pair| pair[0].0 == pair[1].0)
        .unwrap();
    assert!((impact[1].1 + 0.8 * impact[0].1).abs() < 1e-8);
    assert!(impact[0].1 < 0.0 && impact[1].1 > 0.0);
}

#[test]
fn component_event_identities_do_not_merge_equal_guards() {
    let source = format!(
        "{} model Root() {{ instance first: Ball(); instance second: Ball(); }}",
        BALL.replace("model Ball()", "component Ball()")
    );
    let (program, symbols) = admit(&source);
    let first = symbols.get("first.impact").unwrap();
    let second = symbols.get("second.impact").unwrap();
    assert_ne!(first, second);
    for (prefix, event) in [("first", first), ("second", second)] {
        let Some(KernelNode::Activation(activation)) = program.node(event) else {
            panic!("authored event must own an Activation");
        };
        let eqiora::kernel::ActivationKind::Event { guard, .. } = activation.kind() else {
            panic!("event cannot become periodic");
        };
        let height = symbols.get(&format!("{prefix}.height")).unwrap();
        assert!(matches!(
            guard.nodes(),
            [eqiora::kernel::ExprNode::Symbol(eqiora::kernel::SymbolRef::Field(id))]
                if id.erase() == height
        ));
        let resets = program.edges().iter().filter(|edge| {
            edge.from() == event && edge.kind() == eqiora::graph::EdgeKind::Activates
        });
        let actual = resets
            .map(|edge| edge.to())
            .collect::<std::collections::BTreeSet<_>>();
        let expected = ["reset_height", "reset_velocity"]
            .map(|name| symbols.get(&format!("{prefix}.{name}")).unwrap());
        assert_eq!(actual, expected.into_iter().collect());
    }
}

#[test]
fn explicit_thermostat_memory_switches_at_separate_thresholds() {
    let source = r#"
model Thermostat() {
  state temperature: K;
  state slope: K/s;
  initial { temperature = 20[K]; slope = 1[K/s]; }
  relation flow { derivative(temperature) = slope; derivative(slope) = 0[K/s^2]; }
  event upper = crossing(temperature - 22[K], direction = rising);
  event lower = crossing(temperature - 18[K], direction = falling);
  let cooling at upper = -pre(slope);
  let heating at lower = -pre(slope);
  relation cool at upper { next(slope) = cooling; }
  relation heat at lower { next(slope) = heating; }
}
"#;
    let (program, symbols) = admit(source);
    let trajectory = run(&program, 12.0);
    let slopes = series(&trajectory, &symbols, "slope");
    let switches = slopes
        .windows(2)
        .filter(|pair| pair[0].0 == pair[1].0)
        .collect::<Vec<_>>();
    assert_eq!(switches.len(), 3);
    // Unit speed: 20 -> 22 takes 2 s; each full 4 K traversal then takes 4 s.
    for (pair, (time, after)) in switches.iter().zip([(2.0, -1.0), (6.0, 1.0), (10.0, -1.0)]) {
        assert!((pair[0].0 - time).abs() < 1e-8);
        assert!((pair[1].1 - after).abs() < 1e-10);
    }
    assert!(
        (trajectory
            .last_value(symbols.get("temperature").unwrap())
            .unwrap()
            .value()
            - 20.0)
            .abs()
            < 1e-8
    );
}

#[test]
fn event_aliases_do_not_authorize_illegal_evolution_contexts() {
    for source in [
        BALL.replace(
            "derivative(height) = velocity",
            "derivative(height) = rebound",
        ),
        BALL.replace(
            "initial { height = 1[m]; velocity = 0[m/s]; }",
            "initial { height = 1[m]; velocity = next(velocity); }",
        ),
        BALL.replace(
            "derivative(height) = velocity",
            "derivative(height) = next(velocity)",
        ),
        BALL.replace(
            "  event impact",
            "  event other = crossing(height, direction = rising);\n  event impact",
        )
        .replace(
            "relation reset_velocity at impact",
            "relation reset_velocity at other",
        ),
    ] {
        assert!(
            compile("invalid-event-scope.eqi", &source).is_err(),
            "accepted invalid event scope: {source}"
        );
    }
}
