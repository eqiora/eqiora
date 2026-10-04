//! Derived declarations retain typed meaning without extra solve unknowns.
use eqiora_graph::Op;
use eqiora_schema::kernel::{KernelNode, ObservableMeasure, ObservableReduction};

fn observables(source: &str) -> crate::CompiledModel {
    crate::compile("observable.eqi", source)
        .expect("Observable source compiles")
        .remove(0)
}

#[test]
fn divider_value_is_a_retained_expression_not_an_unknown() {
    let model = observables(
        "model Divider() { parameter supply: V = 12; parameter top: Ohm = 1000; parameter bottom: Ohm = 2000; variable potential: V; relation source_value { potential = supply; } observable output: V = potential * bottom / (top + bottom); }",
    );
    let id = model.symbols().get("output").unwrap();
    assert_eq!(id.kind(), eqiora_core::EntityKind::Observable);
    let nodes = model
        .transaction()
        .ops()
        .iter()
        .filter_map(|op| match op {
            Op::DefineKernelNode { node } => Some(node),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        nodes
            .iter()
            .filter(|node| matches!(node, KernelNode::Observable(_)))
            .count(),
        1
    );
    assert_eq!(
        nodes
            .iter()
            .filter(|node| matches!(node, KernelNode::Field(_)))
            .count(),
        1
    );
    assert_eq!(
        nodes
            .iter()
            .filter(|node| matches!(node, KernelNode::Relation(_)))
            .count(),
        1
    );
}

#[test]
fn spatial_integrals_keep_exact_measure_and_reject_wrong_output_type() {
    let source = "model Integral() { variable anchor: 1; relation law { anchor = 0; } domain body = box(0, 2); domain wall = boundary(body, axis = 0, side = upper); parameter temperature: K = 3; observable heat: K*m = integral(temperature, measure(body)); observable face: K = integral(temperature, measure(wall)); }";
    let model = observables(source);
    let definitions = model
        .transaction()
        .ops()
        .iter()
        .filter_map(|op| match op {
            Op::DefineKernelNode {
                node: KernelNode::Observable(value),
            } => Some(value),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(definitions.len(), 2);
    assert!(definitions.iter().any(|value| matches!(
        value.reduction(),
        ObservableReduction::SpatialIntegral {
            measure: ObservableMeasure::Volume,
            ..
        }
    )));
    assert!(definitions.iter().any(|value| matches!(
        value.reduction(),
        ObservableReduction::SpatialIntegral {
            measure: ObservableMeasure::Boundary,
            ..
        }
    )));
    assert!(crate::compile("wrong.eqi", &source.replace("heat: K*m", "heat: K")).is_err());
    assert!(
        crate::compile(
            "wrong.eqi",
            &source.replace(
                "integral(temperature, measure(body))",
                "boundary_integral(body, temperature)"
            )
        )
        .is_err()
    );
}

#[test]
fn component_observables_are_inspectable_but_cannot_be_read_as_ports() {
    let source = "component Sensor() { parameter offset: 1 = 2; observable local: 1 = offset + 1; } model Main() { variable anchor: 1; relation law { anchor = 0; } instance sensor: Sensor(); observable total: 1 = 8; }";
    let model = observables(source);
    assert_eq!(
        model.symbols().get("sensor.local").unwrap().kind(),
        eqiora_core::EntityKind::Observable
    );
    assert!(
        crate::compile(
            "private.eqi",
            &source.replace("total: 1 = 8", "total: 1 = sensor.local")
        )
        .is_err()
    );
}

#[test]
fn observable_source_and_factory_round_trip_identity() {
    let source = "component Probe() { observable local: 1 = 2; } model Main() { // derived\n observable output: 1 = 8; }";
    let document = eqiora_lang::parse("original.eqi", source)
        .into_document()
        .unwrap();
    let formatted = eqiora_lang::format(&document);
    let reparsed = eqiora_lang::parse("formatted.eqi", &formatted)
        .into_document()
        .unwrap();
    assert_eq!(eqiora_lang::format(&reparsed), formatted);
    let first = crate::source_identity::LocalSourceIdentity::from_document(&document).unwrap();
    let second = crate::source_identity::LocalSourceIdentity::from_document(&reparsed).unwrap();
    assert_eq!(first, second);
}

#[test]
fn field_integrals_require_exact_support_and_explicit_boundary_trace() {
    let source = "model Heat() { domain body = box(0, 2); domain other = box(0, 2); domain wall = boundary(body, axis = 0, side = upper); variable temperature: K on body; relation heat on body { temperature = 3[K]; } observable content: K*m = integral(temperature, measure(body)); observable face: K = integral(trace(temperature), measure(wall)); }";
    observables(source);
    for changed in [
        source.replace(
            "integral(temperature, measure(body))",
            "integral(temperature, measure(other))",
        ),
        source.replace(
            "integral(trace(temperature), measure(wall))",
            "integral(temperature, measure(wall))",
        ),
        source.replace(
            "content: K*m = integral(temperature, measure(body))",
            "content: K = temperature",
        ),
        source.replace(
            "content: K*m = integral(temperature, measure(body))",
            "content: K*m = integral(temperature, measure(body)) + 1",
        ),
    ] {
        assert!(crate::compile("invalid.eqi", &changed).is_err());
    }
}

#[test]
fn declared_scalar_domain_context_reaches_literal_roots() {
    observables(
        "model Exact() { variable anchor: 1; relation law { anchor = 0; } observable count: integer = 9007199254740993; observable flag: bool = true; }",
    );
}

#[test]
fn unused_component_observable_bodies_are_checked() {
    let source = |expression: &str| {
        format!(
            "component Unused() {{ observable reading: 1 = {expression}; }} model Main() {{ variable anchor: 1; relation law {{ anchor = 0; }} }}"
        )
    };
    observables(&source("1"));
    for expression in [
        "missing",
        "integral(1, measure(missing))",
        "integral(1)",
        "1[m]",
    ] {
        assert!(crate::compile("unused.eqi", &source(expression)).is_err());
    }
}

#[test]
fn unused_component_integral_types_use_the_shared_measure_contract() {
    let source = "component Unused(support body: volume(ambient_dimension = 2)) { observable area: m^2 = integral(1, measure(body)); } model Main() { variable anchor: 1; relation law { anchor = 0; } }";
    observables(source);
    for invalid in [
        source.replace("area: m^2", "area: m"),
        source.replace("measure(body)", "body"),
    ] {
        assert!(crate::compile("unused.eqi", &invalid).is_err());
    }
}

#[test]
fn full_vector_integral_reference_retains_its_frame_without_an_output_support() {
    observables(
        "model VectorIntegral() { domain body=box(0,1,0,1); variable f:vector<K,2> on body; relation r on body {f=f;} observable flux:vector<K*m^2,2>=integral(f,measure(body)); observable total:vector<K*m^4,2>=integral(flux,measure(body)); }",
    );
}
