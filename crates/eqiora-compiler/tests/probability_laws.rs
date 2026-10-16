//! Source probability declarations retain exact support and have no time activation.
use eqiora_graph::Op;
use eqiora_schema::kernel::{KernelNode, RelationMeaning};

#[test]
fn named_mass_round_trips_without_becoming_an_equation() {
    let source = r#"enum Outcome {Low,High}
        model M() {
            distribution law: normalize_mass(Outcome, weights = [1,3]);
        }"#;
    let parsed = eqiora_lang::parse("probability.eqi", source)
        .into_document()
        .unwrap();
    let formatted = eqiora_lang::format(&parsed);
    let reparsed = eqiora_lang::parse("probability.eqi", &formatted)
        .into_document()
        .unwrap();
    assert_eq!(eqiora_lang::format(&reparsed), formatted);
    for source in [source, &formatted] {
        let compiled = eqiora_compiler::compile("probability.eqi", source)
            .unwrap_or_else(|errors| panic!("{errors:?}"));
        let nodes = compiled[0]
            .transaction()
            .ops()
            .iter()
            .filter_map(|op| match op {
                Op::DefineKernelNode { node } => Some(node),
                _ => None,
            })
            .collect::<Vec<_>>();
        let outcomes = nodes
            .iter()
            .find_map(|node| match node {
                KernelNode::Enum(value) => Some(value.id()),
                _ => None,
            })
            .unwrap();
        let law = nodes
            .iter()
            .find_map(|node| match node {
                KernelNode::Relation(value) => Some(value),
                _ => None,
            })
            .unwrap();
        assert!(matches!(law.meaning(), RelationMeaning::ProbabilityMass {
            outcomes: actual, weights,
        } if *actual == outcomes && law.expression().roots() == [*weights]));
        assert!(law.equation_sides().is_err());
        assert!(
            !nodes
                .iter()
                .any(|node| matches!(node, KernelNode::Activation(_)))
        );
    }
}

#[test]
fn probability_support_must_be_a_named_enum() {
    for support in ["1", "real", "Missing"] {
        let source =
            format!("model M() {{ distribution law: normalize_mass({support}, weights=[1,3]); }}");
        assert!(eqiora_compiler::compile("probability.eqi", &source).is_err());
    }
}

#[test]
fn uniform_density_retains_its_exact_coordinate_measure() {
    use eqiora_compiler::{CompiledModel, StaticBindingValue};
    use eqiora_core::{DimExponents, DynQuantity};
    use eqiora_schema::kernel::{AxisBounds, ObservableMeasure};
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let bounds =
        AxisBounds::new(DynQuantity::new(0.0, length), DynQuantity::new(3.0, length)).unwrap();
    let source = "model M(support position:interval(m)) { distribution law: normalize_density(measure(position), density=1[1/m]); }";
    let formatted = eqiora_lang::format(
        &eqiora_lang::parse("density.eqi", source)
            .into_document()
            .unwrap(),
    );
    for source in [source, &formatted] {
        let compiled = CompiledModel::compile_selected(
            "density.eqi",
            source,
            "M",
            &[("position", StaticBindingValue::CoordinateInterval(bounds))],
        )
        .unwrap_or_else(|errors| panic!("{errors:?}"));
        let support = compiled.symbols().get("position").unwrap();
        assert!(compiled.transaction().ops().iter().any(|op| matches!(op,
            Op::DefineKernelNode { node: KernelNode::Relation(law) }
            if matches!(law.meaning(), RelationMeaning::ProbabilityDensity {
                domain, measure: ObservableMeasure::Volume, ..
            } if domain.erase() == support))));
        assert!(!compiled.transaction().ops().iter().any(|op| matches!(
            op,
            Op::DefineKernelNode {
                node: KernelNode::Activation(_)
            }
        )));
    }
}

#[test]
fn unused_components_still_require_exact_probability_support() {
    for declaration in [
        "distribution law:normalize_mass(1, weights=[1,3]);",
        "distribution law:normalize_density(measure(missing), density=1[1/m]);",
    ] {
        let source =
            format!("component Unused() {{ {declaration} }} model M() {{ parameter p:1=1; }}");
        assert!(
            eqiora_compiler::compile("unused.eqi", &source).is_err(),
            "{source}"
        );
    }
}

#[test]
fn moment_declarations_require_typed_random_variables_under_one_law() {
    let declarations = "enum Outcome {Low,High} model M() { distribution first:normalize_mass(Outcome,weights=[1,3]); distribution second:normalize_mass(Outcome,weights=[1,3]); observable a:m=random_variable(first,2[m]); observable b:s=random_variable(second,3[s]); observable ordinary:m=2[m];";
    for invalid in [
        "observable bad:m=expectation(ordinary);",
        "observable bad:m=variance(a);",
        "observable bad:m*s=covariance(a,b);",
        "observable bad:m=expectation(a, a);",
        "observable bad:m=expectation(a+ordinary);",
    ] {
        let source = format!("{declarations} {invalid} }}");
        assert!(
            eqiora_compiler::compile("moments.eqi", &source).is_err(),
            "{source}"
        );
    }
    let source = format!(
        "{declarations} observable y:s=random_variable(first,3[s]); observable paired:m*s=covariance(a,y); }}"
    );
    let formatted = eqiora_lang::format(
        &eqiora_lang::parse("moments.eqi", &source)
            .into_document()
            .unwrap(),
    );
    for source in [&source, &formatted] {
        eqiora_compiler::compile("moments.eqi", source)
            .unwrap_or_else(|errors| panic!("{errors:?}"));
    }
}

#[test]
fn population_derivatives_require_explicit_interchange_assumptions() {
    let declarations = r#"enum Outcome {Low,High} model M() {
        parameter scale:1=1;
        distribution law:normalize_mass(Outcome,weights=[1,3]);
        observable x:m=random_variable(law,2[m]*scale);
        observable population:m=expectation(x);
        observable alias:m=population;
    "#;
    for differentiated in ["x", "population", "2*population", "alias"] {
        let source =
            format!("{declarations} observable bad:m=partial({differentiated},wrt=scale); }}");
        assert!(
            eqiora_compiler::compile("probability-partial.eqi", &source).is_err(),
            "{differentiated} must not silently exchange probability reduction and differentiation"
        );
    }
}
