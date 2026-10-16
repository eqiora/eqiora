use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroUsize;

use eqiora::artifact::ModelEnvelope;
use eqiora::compiler::{ModelSymbols, compile};
use eqiora::entity::kinds;
use eqiora::graph::{EdgeKind, GraphStore, InMemoryGraphStore};
use eqiora::kernel::KernelNode;
use eqiora::sem::{KernelProgram, PhysicalUnknown};
use eqiora::solver::{
    LinearSolveRequest, LinearSolver, PreconditionerPolicy, ReductionPolicy, SolverPlan,
};
use eqiora::{Id, RawId};
use eqiora_backend_faer::FaerLinearSolver;
use eqiora_numerics::{
    scalar::ScalarPhysicalAffineSolution, scalar::lower_scalar_physical_affine,
    scalar::solve_scalar_physical_affine,
};

const SOURCE: &str = include_str!(
    "../../../verify/language/component-elaboration/models/hierarchical-parallel-dc.eqi"
);
const PERMUTED_SOURCE: &str = include_str!(
    "../../../verify/language/component-elaboration/models/hierarchical-parallel-dc-permuted.eqi"
);
const EXPLICIT_SOURCE: &str =
    include_str!("../../../verify/language/component-elaboration/models/explicit-parallel-dc.eqi");

const VALUE_TOLERANCE: f64 = 2.0e-11;
const RESIDUAL_TOLERANCE: f64 = 1.2e-11;

const HIERARCHICAL_PORTS: [(&str, f64, f64); 7] = [
    ("circuit.source.positive", 12.0, -9.0),
    ("circuit.source.negative", 0.0, 9.0),
    ("circuit.resistor_two.positive", 12.0, 6.0),
    ("circuit.resistor_two.negative", 0.0, -6.0),
    ("circuit.resistor_four.positive", 12.0, 3.0),
    ("circuit.resistor_four.negative", 0.0, -3.0),
    ("circuit.ground.terminal", 0.0, 0.0),
];

const EXPLICIT_CORRESPONDENCE: [(&str, &str); 15] = [
    ("connector::Pin", "electrical"),
    ("supply_voltage", "supply_voltage"),
    ("resistance_two", "resistance_two"),
    ("resistance_four", "resistance_four"),
    ("circuit.source.positive", "source_positive"),
    ("circuit.source.negative", "source_negative"),
    ("circuit.resistor_two.positive", "resistor_two_positive"),
    ("circuit.resistor_two.negative", "resistor_two_negative"),
    ("circuit.resistor_four.positive", "resistor_four_positive"),
    ("circuit.resistor_four.negative", "resistor_four_negative"),
    ("circuit.ground.terminal", "ground_terminal"),
    ("circuit.source.law", "voltage_source"),
    ("circuit.resistor_two.law", "two_ohm_resistor"),
    ("circuit.resistor_four.law", "four_ohm_resistor"),
    ("circuit.ground.law", "explicit_ground"),
];

fn admit(compiled: eqiora::compiler::CompiledModel) -> (KernelProgram, ModelSymbols) {
    let model = compiled.model();
    let symbols = compiled.symbols().clone();
    let (transaction, _, _) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).expect("atomic hierarchy commit");
    let program = KernelProgram::from_snapshot(&store.snapshot(), model)
        .expect("closed hierarchical physical model");
    (program, symbols)
}

fn port(symbols: &ModelSymbols, name: &str) -> Id<kinds::Port> {
    symbols
        .get(name)
        .and_then(RawId::downcast)
        .unwrap_or_else(|| panic!("fixture Port `{name}`"))
}

fn selected_connection(program: &KernelProgram, member: Id<kinds::Port>) -> Id<kinds::Connection> {
    program
        .nodes()
        .find_map(|node| {
            let KernelNode::Connection(connection) = node else {
                return None;
            };
            program
                .edges()
                .iter()
                .any(|edge| {
                    edge.kind() == EdgeKind::Connects
                        && edge.from() == connection.id().erase()
                        && edge.to() == member.erase()
                })
                .then_some(connection.id())
        })
        .expect("selected Port belongs to one Connection")
}

fn solve(
    program: &KernelProgram,
    symbols: &ModelSymbols,
    selected_port: &str,
) -> ScalarPhysicalAffineSolution {
    let connection = selected_connection(program, port(symbols, selected_port));
    let problem = lower_scalar_physical_affine(program, connection, None)
        .expect("complete affine physical closure");
    assert_eq!(problem.canonical_system().rows(), 14);
    assert_eq!(problem.canonical_system().columns(), 14);
    let mut roots = problem
        .composed_system()
        .junctions()
        .iter()
        .map(|junction| junction.dag().roots().len())
        .collect::<Vec<_>>();
    roots.sort_unstable();
    assert_eq!(roots, [3, 4]);

    let plan = SolverPlan::new(
        LinearSolver::SparseLu,
        1.0e-12,
        1.0e-14,
        NonZeroUsize::new(1).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Identity)
    .with_reduction(ReductionPolicy::Fast);
    let solution = solve_scalar_physical_affine(
        &problem,
        &vec![0.0; problem.canonical_system().rows()],
        LinearSolveRequest::new(&FaerLinearSolver, plan),
    )
    .expect("hierarchical physical solve");
    assert!(solution.reference_residual_norm() <= RESIDUAL_TOLERANCE);
    solution
}

fn physical_value(solution: &ScalarPhysicalAffineSolution, port: Id<kinds::Port>) -> (f64, f64) {
    (
        solution
            .value(PhysicalUnknown::Across(port))
            .expect("across value"),
        solution
            .value(PhysicalUnknown::Through(port))
            .expect("through value"),
    )
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= VALUE_TOLERANCE,
        "expected {expected:e}, received {actual:e}"
    );
}

fn assert_identical_semantics_after_identity_normalization(
    hierarchy: &(KernelProgram, ModelSymbols),
    explicit: &(KernelProgram, ModelSymbols),
) {
    let hierarchy_envelope = ModelEnvelope::from_program(&hierarchy.0).unwrap();
    let explicit_envelope = ModelEnvelope::from_program(&explicit.0).unwrap();
    let mut hierarchy_value: serde_json::Value =
        serde_json::from_slice(&hierarchy_envelope.canonical_json().unwrap()).unwrap();
    let explicit_value: serde_json::Value =
        serde_json::from_slice(&explicit_envelope.canonical_json().unwrap()).unwrap();

    let mut identities = BTreeMap::from([(
        hierarchy.0.model().ulid().to_string(),
        explicit.0.model().ulid().to_string(),
    )]);
    for (hierarchical_name, explicit_name) in EXPLICIT_CORRESPONDENCE {
        let hierarchical = hierarchy
            .1
            .get(hierarchical_name)
            .unwrap_or_else(|| panic!("hierarchical symbol `{hierarchical_name}`"));
        let explicit = explicit
            .1
            .get(explicit_name)
            .unwrap_or_else(|| panic!("explicit symbol `{explicit_name}`"));
        assert!(
            identities
                .insert(hierarchical.ulid().to_string(), explicit.ulid().to_string(),)
                .is_none()
        );
    }

    let hierarchy_activations = activations_by_relation(&hierarchy_value);
    let explicit_activations = activations_by_relation(&explicit_value);
    assert_eq!(hierarchy_activations.len(), explicit_activations.len());
    for (relation, activation) in hierarchy_activations {
        let explicit_relation = identities.get(&relation).expect("mapped Relation");
        let explicit_activation = explicit_activations
            .get(explicit_relation)
            .expect("corresponding Activation");
        assert!(
            identities
                .insert(activation, explicit_activation.clone())
                .is_none()
        );
    }

    let hierarchy_connections = connections_by_members(&hierarchy_value);
    let explicit_connections = connections_by_members(&explicit_value);
    let explicit_connections_by_members = explicit_connections
        .into_iter()
        .map(|(connection, members)| (members, connection))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        hierarchy_connections.len(),
        explicit_connections_by_members.len()
    );
    for (connection, members) in hierarchy_connections {
        let normalized_members = members
            .into_iter()
            .map(|member| identities.get(&member).expect("mapped Port").clone())
            .collect::<BTreeSet<_>>();
        let explicit_connection = explicit_connections_by_members
            .get(&normalized_members)
            .expect("Connection with the same normalized member set");
        assert!(
            identities
                .insert(connection, explicit_connection.clone())
                .is_none()
        );
    }

    let hierarchy_ids = collect_model_ulids(&hierarchy_value);
    let explicit_ids = collect_model_ulids(&explicit_value);
    assert_eq!(
        identities.keys().cloned().collect::<BTreeSet<_>>(),
        hierarchy_ids,
        "identity normalization must cover every hierarchical semantic entity"
    );
    assert_eq!(
        identities.values().cloned().collect::<BTreeSet<_>>(),
        explicit_ids,
        "identity normalization must be a complete bijection"
    );

    rewrite_model_ulids(&mut hierarchy_value, &identities);
    let normalized = ModelEnvelope::from_json(
        &serde_json::to_vec(&hierarchy_value).unwrap(),
        Default::default(),
    )
    .unwrap();
    assert_eq!(
        normalized.canonical_json().unwrap(),
        explicit_envelope.canonical_json().unwrap()
    );
    assert_eq!(
        normalized.digest().unwrap(),
        explicit_envelope.digest().unwrap()
    );
}

fn activations_by_relation(model: &serde_json::Value) -> BTreeMap<String, String> {
    let mut activations = BTreeMap::new();
    for edge in model["edges"].as_array().expect("edge array") {
        if edge["kind"].as_str() != Some("activates") {
            continue;
        }
        let activation = id_ulid(&edge["from"]);
        let relation = id_ulid(&edge["to"]);
        assert_eq!(edge["from"]["kind"].as_str(), Some("activation"));
        assert_eq!(edge["to"]["kind"].as_str(), Some("relation"));
        assert!(activations.insert(relation, activation).is_none());
    }
    activations
}

fn connections_by_members(model: &serde_json::Value) -> BTreeMap<String, BTreeSet<String>> {
    let mut connections = BTreeMap::<_, BTreeSet<_>>::new();
    for edge in model["edges"].as_array().expect("edge array") {
        if edge["kind"].as_str() != Some("connects") {
            continue;
        }
        assert_eq!(edge["from"]["kind"].as_str(), Some("connection"));
        assert_eq!(edge["to"]["kind"].as_str(), Some("port"));
        assert!(
            connections
                .entry(id_ulid(&edge["from"]))
                .or_default()
                .insert(id_ulid(&edge["to"]))
        );
    }
    connections
}

fn collect_model_ulids(model: &serde_json::Value) -> BTreeSet<String> {
    fn collect(value: &serde_json::Value, identities: &mut BTreeSet<String>) {
        match value {
            serde_json::Value::Object(object) => {
                if let Some(ulid) = object.get("ulid") {
                    identities.insert(ulid.as_str().expect("typed ID ULID").to_owned());
                }
                for child in object.values() {
                    collect(child, identities);
                }
            }
            serde_json::Value::Array(values) => {
                for child in values {
                    collect(child, identities);
                }
            }
            _ => {}
        }
    }

    let mut identities =
        BTreeSet::from([model["model_ulid"].as_str().expect("Model ULID").to_owned()]);
    collect(model, &mut identities);
    identities
}

fn rewrite_model_ulids(value: &mut serde_json::Value, identities: &BTreeMap<String, String>) {
    match value {
        serde_json::Value::Object(object) => {
            for key in ["model_ulid", "ulid"] {
                if let Some(value) = object.get_mut(key) {
                    let source = value.as_str().expect("ULID string");
                    *value = identities
                        .get(source)
                        .unwrap_or_else(|| panic!("unmapped semantic identity `{source}`"))
                        .clone()
                        .into();
                }
            }
            for child in object.values_mut() {
                rewrite_model_ulids(child, identities);
            }
        }
        serde_json::Value::Array(values) => {
            for child in values {
                rewrite_model_ulids(child, identities);
            }
        }
        _ => {}
    }
}

fn id_ulid(value: &serde_json::Value) -> String {
    value["ulid"].as_str().expect("typed ID ULID").to_owned()
}

#[test]
fn nested_components_lower_to_one_flat_kernel_and_solve_the_analytic_dc_case() {
    let mut compiled = compile("hierarchical-parallel-dc.eqi", SOURCE).unwrap();
    let compiled = compiled.pop().expect("one root model");
    let provenance = compiled.provenance().expect("hierarchy provenance").clone();

    let resistor_two = compiled
        .symbols()
        .get("circuit.resistor_two.positive")
        .unwrap();
    let resistor_four = compiled
        .symbols()
        .get("circuit.resistor_four.positive")
        .unwrap();
    assert_ne!(resistor_two, resistor_four);
    let two_source = provenance.get_by_graph_id(resistor_two).unwrap();
    let four_source = provenance.get_by_graph_id(resistor_four).unwrap();
    assert_eq!(two_source.definition_span(), four_source.definition_span());
    assert_ne!(two_source.instance_span(), four_source.instance_span());
    assert_eq!(two_source.binding_spans().len(), 4);
    assert_eq!(four_source.binding_spans().len(), 4);

    let (program, symbols) = admit(compiled);
    for aliases in [
        [
            "supply_voltage",
            "circuit.supply_voltage",
            "circuit.source.voltage",
        ],
        [
            "resistance_two",
            "circuit.resistance_two",
            "circuit.resistor_two.resistance",
        ],
        [
            "resistance_four",
            "circuit.resistance_four",
            "circuit.resistor_four.resistance",
        ],
    ] {
        let identity = symbols.get(aliases[0]).expect("root Parameter identity");
        assert!(
            aliases[1..]
                .iter()
                .all(|alias| symbols.get(alias) == Some(identity))
        );
    }
    let counts = program.nodes().fold([0_usize; 8], |mut counts, node| {
        let index = match node {
            KernelNode::Domain(_) => 0,
            KernelNode::Parameter(_) => 1,
            KernelNode::Port(_) => 2,
            KernelNode::Relation(_) => 3,
            KernelNode::Activation(_) => 4,
            KernelNode::Connection(_) => 5,
            KernelNode::Field(_) => 6,
            KernelNode::ClockDomain(_) => 7,
            _ => return counts,
        };
        counts[index] += 1;
        counts
    });
    assert_eq!(counts, [1, 3, 7, 4, 4, 2, 0, 0]);
    assert_eq!(provenance.len(), 22);
    assert!(
        program
            .nodes()
            .all(|node| provenance.get_by_graph_id(node.id()).is_some())
    );
    assert!(program.boundary().is_empty());
    assert_eq!(
        program
            .edges()
            .iter()
            .filter(|edge| edge.kind() == EdgeKind::DependsOn)
            .count(),
        10
    );
    assert_eq!(
        program
            .edges()
            .iter()
            .filter(|edge| edge.kind() == EdgeKind::HasPort)
            .count(),
        7
    );
    assert_eq!(
        program
            .edges()
            .iter()
            .filter(|edge| edge.kind() == EdgeKind::Activates)
            .count(),
        4
    );
    assert_eq!(
        program
            .edges()
            .iter()
            .filter(|edge| edge.kind() == EdgeKind::Connects)
            .count(),
        7
    );
    let mut relation_roots = program
        .nodes()
        .filter_map(|node| match node {
            KernelNode::Relation(relation) => Some(relation.equation_sides().unwrap().len()),
            _ => None,
        })
        .collect::<Vec<_>>();
    relation_roots.sort_unstable();
    assert_eq!(relation_roots, [1, 2, 2, 2]);

    let envelope = ModelEnvelope::from_program(&program).unwrap();
    let bytes = envelope.canonical_json().unwrap();
    let digest = envelope.digest().unwrap();
    let decoded = ModelEnvelope::from_json(&bytes, Default::default()).unwrap();
    let reconstructed = ModelEnvelope::from_program(&decoded.to_program().unwrap()).unwrap();
    assert_eq!(reconstructed.canonical_json().unwrap(), bytes);
    assert_eq!(reconstructed.digest().unwrap(), digest);

    let solution = solve(&program, &symbols, "circuit.source.positive");
    for (name, expected_across, expected_through) in HIERARCHICAL_PORTS {
        let (across, through) = physical_value(&solution, port(&symbols, name));
        assert_close(across, expected_across);
        assert_close(through, expected_through);
    }
}

#[test]
fn semantic_permutation_preserves_exact_ids_transaction_and_model_bytes() {
    let mut first = compile("first/location.eqi", SOURCE).unwrap();
    let mut second = compile("relocated.eqi", PERMUTED_SOURCE).unwrap();
    let first = first.pop().unwrap();
    let second = second.pop().unwrap();
    assert_eq!(first.model(), second.model());
    assert_eq!(
        first.symbols().iter().collect::<Vec<_>>(),
        second.symbols().iter().collect::<Vec<_>>()
    );
    assert_eq!(first.transaction().ops(), second.transaction().ops());
    assert_ne!(first.provenance(), second.provenance());

    let (first_program, _) = admit(first);
    let (second_program, _) = admit(second);
    let first_model = ModelEnvelope::from_program(&first_program).unwrap();
    let second_model = ModelEnvelope::from_program(&second_program).unwrap();
    assert_eq!(
        first_model.canonical_json().unwrap(),
        second_model.canonical_json().unwrap()
    );
    assert_eq!(
        first_model.digest().unwrap(),
        second_model.digest().unwrap()
    );
}

#[test]
fn hierarchy_and_explicit_flat_source_have_identical_normalized_semantics() {
    let hierarchy = admit(compile("hierarchy.eqi", SOURCE).unwrap().pop().unwrap());
    let flat = admit(
        compile("explicit-parallel-dc.eqi", EXPLICIT_SOURCE)
            .unwrap()
            .pop()
            .unwrap(),
    );
    assert_identical_semantics_after_identity_normalization(&hierarchy, &flat);
}

#[test]
fn invalid_hierarchy_is_rejected_before_any_transaction_is_exposed() {
    let cases = [
        "component C(\n  parameter p: 1\n) { } model m() { instance c: C(); }",
        "component C(\n  parameter p: 1 = 1\n) { } model m() { instance c: C(q = 2); }",
        "component C() { parameter p: 1 = 1; } model m() { instance c: C(p = 2); }",
        "component C(\n  parameter p: m = 1\n) { } model m() { parameter q: s = 2; instance c: C(p = q); }",
        "component C(\n  parameter p: 1 = 1\n) { } model m() { instance c: C(p = 2, p = 3); }",
        "component A() { instance b: B(); } component B() { instance a: A(); } model m() { instance a: A(); }",
        r#"
connector A {
  across potential: 1;
  through flow: 1;
}
connector B {
  across potential: 1;
  through flow: 1;
}
component Left(
  port p: A
) { relation r { p.potential = 0; } }
component Right(
  port p: B
) { relation r { p.potential = 0; } }
model m() { instance left: Left(); instance right: Right(); connect left.p, right.p; }
"#,
        r#"
component C() { port hidden: signal input 1; relation r { hidden = 0; } }
model m() { port source: signal output 1; instance c: C(); connect source -> c.hidden; }
"#,
    ];
    for source in cases {
        assert!(compile("invalid-hierarchy.eqi", source).is_err());
    }
}
