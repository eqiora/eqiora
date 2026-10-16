//! A fixed resistor chain retains ordinary equations and exact endpoint ownership.
use eqiora::RawId;
use eqiora::api::ModelDocument;
use eqiora::graph::EdgeKind;
use eqiora::kernel::{ExprDag, ExprId, ExprNode, KernelNode, SymbolRef};
use eqiora::sem::{Interpreter, PhysicalUnknown, ReferenceConfig};
use std::collections::{BTreeMap, BTreeSet};

// Same local across/through convention as component-elaboration's DC fixture.
const DEFINITIONS: &str = r#"
connector Pin {
  across voltage: V;
  through current: A;
}
component VoltageSource(parameter voltage: V, port positive: Pin, port negative: Pin) {
  relation law {
    positive.voltage - negative.voltage - voltage = 0;
    positive.current + negative.current = 0;
  }
}
component Resistor(parameter resistance: Ohm, port positive: Pin, port negative: Pin) {
  relation law {
    positive.voltage - negative.voltage - resistance * positive.current = 0;
    positive.current + negative.current = 0;
  }
}
component Ground(port terminal: Pin) {
  relation law { terminal.voltage = 0; }
}
"#;

fn source(indexed: bool, permuted: bool) -> String {
    let declarations = if indexed {
        "instance cell[i in Stages]: Resistor(resistance = 2[Ohm]);".to_owned()
    } else {
        let mut cells = (0..3)
            .map(|i| format!("instance cell{i}: Resistor(resistance = 2[Ohm]);"))
            .collect::<Vec<_>>();
        if permuted {
            cells.reverse();
        }
        cells.join("\n")
    };
    let cell = |i| {
        if indexed {
            format!("cell[index(Stages,{i})]")
        } else {
            format!("cell{i}")
        }
    };
    let mut fragments = vec![
        format!("connect supply.positive, {}.positive;", cell(0)),
        format!(
            "connect supply.negative, {}.negative, ground.terminal;",
            cell(2)
        ),
    ];
    if indexed {
        fragments.push("connect [j in Links] cell[index(Stages, ordinal(j))].negative, cell[index(Stages, ordinal(j) + 1)].positive;".into());
    } else {
        for i in 0..2 {
            fragments.push(format!(
                "connect {}.negative, {}.positive;",
                cell(i),
                cell(i + 1)
            ));
        }
    }
    if permuted {
        fragments.reverse();
    }
    format!(
        "{DEFINITIONS}\nmodel Chain() {{\nindexset Stages = range(3);\nindexset Links = range(2);\n{declarations}\ninstance supply: VoltageSource(voltage = 12[V]);\ninstance ground: Ground();\n{}\n}}",
        fragments.join("\n")
    )
}

fn canonical_name(name: &str) -> String {
    let mut name = name.to_owned();
    for i in 0..3 {
        name = name.replace(&format!("cell{i}."), &format!("cell[{i}]."));
    }
    name
}

fn names(document: &ModelDocument) -> BTreeMap<RawId, String> {
    document
        .aliases()
        .iter()
        .map(|(name, id)| (*id, canonical_name(name)))
        .collect()
}

fn expression(dag: &ExprDag, id: ExprId, names: &BTreeMap<RawId, String>) -> String {
    match &dag.nodes()[id.index() as usize] {
        ExprNode::Constant(value) => format!("{value:?}"),
        ExprNode::Symbol(symbol) => {
            let (kind, id) = match symbol {
                SymbolRef::Across(id) => ("across", id.erase()),
                SymbolRef::Through(id) => ("through", id.erase()),
                SymbolRef::Parameter(id) => ("parameter", id.erase()),
                other => panic!("unexpected resistor equation symbol {other:?}"),
            };
            format!("{kind}({})", names[&id])
        }
        ExprNode::Add(a, b) => format!(
            "({}+{})",
            expression(dag, *a, names),
            expression(dag, *b, names)
        ),
        ExprNode::Sub(a, b) => format!(
            "({}-{})",
            expression(dag, *a, names),
            expression(dag, *b, names)
        ),
        ExprNode::Mul(a, b) => format!(
            "({}*{})",
            expression(dag, *a, names),
            expression(dag, *b, names)
        ),
        other => panic!("unexpected resistor equation node {other:?}"),
    }
}

fn equations(document: &ModelDocument) -> BTreeMap<String, Vec<(String, String)>> {
    let names = names(document);
    document
        .program()
        .nodes()
        .filter_map(|node| {
            let KernelNode::Relation(relation) = node else {
                return None;
            };
            Some((
                names[&relation.id().erase()].clone(),
                relation
                    .equation_sides()
                    .unwrap()
                    .map(|(a, b)| {
                        (
                            expression(relation.expression(), a, &names),
                            expression(relation.expression(), b, &names),
                        )
                    })
                    .collect(),
            ))
        })
        .collect()
}

fn connections(document: &ModelDocument) -> BTreeSet<BTreeSet<String>> {
    let names = names(document);
    document
        .program()
        .nodes()
        .filter_map(|node| {
            let KernelNode::Connection(connection) = node else {
                return None;
            };
            Some(
                document
                    .program()
                    .edges()
                    .iter()
                    .filter(|edge| {
                        edge.kind() == EdgeKind::Connects && edge.from() == connection.id().erase()
                    })
                    .map(|edge| names[&edge.to()].clone())
                    .collect(),
            )
        })
        .collect()
}

#[test]
fn explicit_and_indexed_resistors_preserve_equations_connections_and_dc_solution() {
    let explicit = ModelDocument::compile("explicit-chain.eqi", &source(false, false)).unwrap();
    let expected_equations = equations(&explicit);
    let expected_connections = connections(&explicit);
    assert_eq!(expected_equations.len(), 5);
    assert_eq!(expected_connections.len(), 4);
    for (indexed, permuted) in [(false, false), (false, true), (true, false), (true, true)] {
        let document = ModelDocument::compile("chain.eqi", &source(indexed, permuted)).unwrap();
        assert_eq!(equations(&document), expected_equations);
        assert_eq!(connections(&document), expected_connections);
        let names = names(&document);
        let lookup = |name: &str| {
            *names
                .iter()
                .find(|(_, actual)| actual.as_str() == name)
                .unwrap()
                .0
        };
        let laws = (0..3)
            .map(|i| lookup(&format!("cell[{i}].law")))
            .collect::<BTreeSet<_>>();
        let ports = (0..3)
            .flat_map(|i| {
                [
                    lookup(&format!("cell[{i}].positive")),
                    lookup(&format!("cell[{i}].negative")),
                ]
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(laws.len(), 3);
        assert_eq!(ports.len(), 6);
        let trajectory = Interpreter::new()
            .run(document.program(), ReferenceConfig::new(0.0, 1.0).unwrap())
            .unwrap();
        for i in 0..3 {
            let positive = lookup(&format!("cell[{i}].positive")).downcast().unwrap();
            let negative = lookup(&format!("cell[{i}].negative")).downcast().unwrap();
            let across = |port| {
                trajectory
                    .last_physical_value(PhysicalUnknown::Across(port))
                    .unwrap()
                    .value()
            };
            let through = |port| {
                trajectory
                    .last_physical_value(PhysicalUnknown::Through(port))
                    .unwrap()
                    .value()
            };
            // 12 V / (2+2+2) Ohm = 2 A. Into each positive terminal is +2 A.
            assert!((across(positive) - across(negative) - 4.0).abs() < 1e-10);
            assert!((through(positive) - 2.0).abs() < 1e-10);
            assert!((through(negative) + 2.0).abs() < 1e-10);
        }
    }
}

#[test]
fn indexed_connections_reject_foreign_sets_bad_neighbors_and_missing_members_locally() {
    let valid = source(true, false);
    ModelDocument::compile("valid-chain.eqi", &valid).unwrap();
    for invalid in [
        valid.replace("index(Stages, ordinal(j))", "index(Links, ordinal(j))"),
        valid.replace("range(2)", "range(3)"),
        valid.replace("ordinal(j) + 1", "ordinal(j) - 1"),
        valid.replace("].negative", "].missing"),
        valid.replace("ordinal(j) + 1", "time"),
        valid.replace("range(3)", "range(2147483647)"),
    ] {
        let diagnostics = ModelDocument::compile("invalid-chain.eqi", &invalid).unwrap_err();
        assert!(
            diagnostics
                .iter()
                .any(|diagnostic| diagnostic.source_span().is_some()),
            "{diagnostics:?}"
        );
    }
}

#[test]
fn distinct_static_chain_sizes_share_the_same_two_terminal_contract() {
    let source = format!(
        r#"{DEFINITIONS}
component Chain(parameter n:integer,port positive:Pin,port negative:Pin) {{
    indexset Stages=range(n);
    indexset Links=range(n-1);
    instance cell[i in Stages]:Resistor(resistance=2[Ohm]);
    connect positive,cell[index(Stages,0)].positive;
    connect negative,cell[index(Stages,n-1)].negative;
    connect [i in Links] cell[index(Stages,ordinal(i))].negative,cell[index(Stages,ordinal(i)+1)].positive;
}}
model Pair() {{
    instance short:Chain(n=2);
    instance long:Chain(n=3);
    instance a:VoltageSource(voltage=12[V]);
    instance b:VoltageSource(voltage=12[V]);
    instance ga:Ground();
    instance gb:Ground();
    connect short.positive,a.positive;
    connect short.negative,a.negative,ga.terminal;
    connect long.positive,b.positive;
    connect long.negative,b.negative,gb.terminal;
}}"#
    );
    let document = ModelDocument::compile("two-sizes.eqi", &source).unwrap();
    let trajectory = Interpreter::new()
        .run(document.program(), ReferenceConfig::new(0.0, 1.0).unwrap())
        .unwrap();
    let mut owners = BTreeSet::new();
    let mut endpoints = BTreeSet::new();
    for (name, count, current, voltage) in [("short", 2, 3.0, 6.0), ("long", 3, 2.0, 4.0)] {
        for index in 0..count {
            let positive = document.aliases()[&format!("{name}.cell[{index}].positive")];
            let negative = document.aliases()[&format!("{name}.cell[{index}].negative")];
            assert!(owners.insert(document.aliases()[&format!("{name}.cell[{index}].law")]));
            assert!(endpoints.insert(positive));
            assert!(endpoints.insert(negative));
            let across = |id: RawId| {
                trajectory
                    .last_physical_value(PhysicalUnknown::Across(id.downcast().unwrap()))
                    .unwrap()
                    .value()
            };
            let through = |id: RawId| {
                trajectory
                    .last_physical_value(PhysicalUnknown::Through(id.downcast().unwrap()))
                    .unwrap()
                    .value()
            };
            // Series law: I=12/(2*n); each 2-ohm resistor drops 12/n volts.
            // The small linear solve uses the same absolute 1e-10 oracle as above.
            assert!((through(positive) - current).abs() < 1e-10);
            assert!((through(negative) + current).abs() < 1e-10);
            assert!((across(positive) - across(negative) - voltage).abs() < 1e-10);
        }
    }
    assert_eq!(owners.len(), 5);
    assert_eq!(endpoints.len(), 10);
}
