//! Source, persisted identity, and execution share the same coordinate pullback.
use eqiora_api::ModelDocument;
use eqiora_core::{DimExponents, DynQuantity};
use eqiora_schema::kernel::KernelNode;

#[test]
fn mixed_position_velocity_map_keeps_each_jacobian_entry_dimensioned() {
    use eqiora_compiler::StaticBindingValue;
    use eqiora_schema::kernel::AxisBounds;
    let interval = |time| {
        let unit = DimExponents::from_integers([0, 1, time, 0, 0, 0, 0]).unwrap();
        StaticBindingValue::CoordinateInterval(
            AxisBounds::new(DynQuantity::new(0.0, unit), DynQuantity::new(4.0, unit)).unwrap(),
        )
    };
    for (selected, unit, time, expected) in [("xi", "m/s", -1, 0.5), ("nu", "m", 0, 2.25)] {
        let source = format!(
            r#"model Phase(
            support position:interval(m), support velocity:interval(m/s),
            support target_position:interval(m), support target_velocity:interval(m/s)) {{
            support reference:product(position,velocity);
            support target:product(target_position,target_velocity);
            coordinate xi:m on reference from position;
            coordinate nu:m/s on reference from velocity;
            coordinate x:m on target from target_position;
            coordinate v:m/s on target from target_velocity;
            variable anchor:1;
            relation r {{anchor=0;}}
            observable derivative:{unit}=evaluate(partial(
                pullback(x*v,from=(xi,nu),at=(x=xi+2[s]*nu,v=nu)),wrt={selected}),
                at=(xi=0.25[m],nu=0.5[m/s]));
        }}"#
        );
        let document = ModelDocument::compile_selected(
            "phase-map.eqi",
            &source,
            "Phase",
            &[
                ("position", interval(0)),
                ("velocity", interval(-1)),
                ("target_position", interval(0)),
                ("target_velocity", interval(-1)),
            ],
        )
        .unwrap();
        let replay = ModelDocument::replay(&document.canonical_json().unwrap()).unwrap();
        let observable = replay
            .program()
            .nodes()
            .find_map(|node| match node {
                KernelNode::Observable(value) => Some(value.id()),
                _ => None,
            })
            .unwrap();
        let actual = replay
            .program()
            .evaluate_finite_observable(observable, &mut |_| None)
            .unwrap();
        // x*v=xi*nu+2s*nu², hence derivatives nu and xi+4s*nu.
        let dimension = DimExponents::from_integers([0, 1, time, 0, 0, 0, 0]).unwrap();
        assert_eq!(
            actual.real_scalar_value().unwrap(),
            DynQuantity::new(expected, dimension)
        );
    }
}

#[test]
fn authored_pullback_survives_model_replay_with_independent_polynomial_value() {
    let source = r#"model M() {
        domain reference=box(0,1,0,1);
        domain body=box(0,3,0,3);
        coordinate xi:m on reference from reference[0];
        coordinate eta:m on reference from reference[1];
        coordinate x:m on body from body[0];
        coordinate y:m on body from body[1];
        variable anchor:1;
        relation r {anchor=0;}
        observable sample:m^2=evaluate(
            pullback(x*x+x*y,from=(xi,eta),at=(x=2*xi+eta,y=3*eta)),
            at=(xi=0.25[m],eta=0.5[m]));
    }"#;
    let document = ModelDocument::compile("pullback.eqi", source).unwrap();
    let bytes = document.canonical_json().unwrap();
    let mut obsolete: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(obsolete["schema"], "eqiora.model-envelope/v43");
    obsolete["schema"] = serde_json::json!("eqiora.model-envelope/v39");
    let error = ModelDocument::replay(&serde_json::to_vec(&obsolete).unwrap()).unwrap_err();
    assert!(error.iter().any(|diagnostic| {
        diagnostic
            .message()
            .contains("unsupported eqiora.model-envelope/v43 schema")
    }));

    // The alternate affine map meets the original at this sample, but has a
    // different Jacobian. A point value cannot identify the entire map.
    let alternate = ModelDocument::compile(
        "pullback.eqi",
        &source.replace("2*xi+eta", "xi+eta+0.25[m]"),
    )
    .unwrap();
    assert_ne!(
        document.structural_fingerprint().unwrap(),
        alternate.structural_fingerprint().unwrap()
    );
    let replay = ModelDocument::replay(&document.canonical_json().unwrap()).unwrap();
    assert_eq!(document.program(), replay.program());
    assert_eq!(
        document.structural_fingerprint().unwrap(),
        replay.structural_fingerprint().unwrap()
    );
    for model in [&document, &replay, &alternate] {
        let observable = model
            .program()
            .nodes()
            .find_map(|node| match node {
                KernelNode::Observable(value) => Some(value.id()),
                _ => None,
            })
            .unwrap();
        let actual = model
            .program()
            .evaluate_finite_observable(observable, &mut |_| None)
            .unwrap();
        // u(2ξ+η,3η)=4ξ²+10ξη+4η²; at (1/4,1/2), u=5/2 m².
        let area = DimExponents::from_integers([0, 2, 0, 0, 0, 0, 0]).unwrap();
        assert_eq!(
            actual.real_scalar_value().unwrap(),
            DynQuantity::new(2.5, area)
        );
    }
}

#[test]
fn authored_chain_rule_uses_every_nonsymmetric_jacobian_entry() {
    for (selected, expression, unit, expected) in [
        ("xi", "pulled", "m", 7.0),
        ("eta", "pulled", "m", 6.5),
        ("xi", "xi*pulled", "m^2", 4.25),
        ("eta", "xi*pulled", "m^2", 1.625),
    ] {
        let expression = expression.replace(
            "pulled",
            "pullback(x*x+x*y,from=(xi,eta),at=(x=2*xi+eta,y=3*eta))",
        );
        let source = format!(
            r#"model M() {{
            domain reference=box(0,1,0,1);
            domain body=box(0,3,0,3);
            coordinate xi:m on reference from reference[0];
            coordinate eta:m on reference from reference[1];
            coordinate x:m on body from body[0];
            coordinate y:m on body from body[1];
            variable anchor:1;
            relation r {{anchor=0;}}
            observable derivative:{unit}=evaluate(partial(
                {expression},wrt={selected}),
                at=(xi=0.25[m],eta=0.5[m]));
        }}"#
        );
        let document = ModelDocument::compile("chain-rule.eqi", &source).unwrap();
        let replay = ModelDocument::replay(&document.canonical_json().unwrap()).unwrap();
        let observable = replay
            .program()
            .nodes()
            .find_map(|node| match node {
                KernelNode::Observable(value) => Some(value.id()),
                _ => None,
            })
            .unwrap();
        let actual = replay
            .program()
            .evaluate_finite_observable(observable, &mut |_| None)
            .unwrap();
        // ∂ξ u=8ξ+10η; ∂η u=10ξ+8η, independently from the expanded polynomial.
        let length =
            DimExponents::from_integers([0, if unit == "m" { 1 } else { 2 }, 0, 0, 0, 0, 0])
                .unwrap();
        assert_eq!(
            actual.real_scalar_value().unwrap(),
            DynQuantity::new(expected, length)
        );
    }
}

#[test]
fn sibling_pullback_derivatives_keep_distinct_generated_expression_contexts() {
    // Each division creates a reciprocal during formalization. Those synthetic
    // inputs must remain independent across both map directions and siblings.
    for (mapped_x, expected) in [("2*xi+eta", 13.5), ("xi/2+eta", 6.0)] {
        let pulled = format!("pullback(x*x+x*y,from=(xi,eta),at=(x={mapped_x},y=3*eta))");
        let source = format!(
            r#"model M() {{
        domain reference=box(0,1,0,1);
        domain body=box(0,3,0,3);
        coordinate xi:m on reference from reference[0];
        coordinate eta:m on reference from reference[1];
        coordinate x:m on body from body[0];
        coordinate y:m on body from body[1];
        variable anchor:1;
        relation r {{anchor=0;}}
        observable sum:m=evaluate(partial({pulled},wrt=xi)+partial({pulled},wrt=eta),
            at=(xi=0.25[m],eta=0.5[m]));
    }}"#
        );
        let document = ModelDocument::compile("siblings.eqi", &source).unwrap();
        let observable = document
            .program()
            .nodes()
            .find_map(|node| match node {
                KernelNode::Observable(value) => Some(value.id()),
                _ => None,
            })
            .unwrap();
        let actual = document
            .program()
            .evaluate_finite_observable(observable, &mut |_| None)
            .unwrap();
        // Original map: 7 + 13/2. Half-x map: u=xi²/4+(5/2)xi*eta+4eta²; partials 11/8 + 37/8.
        let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
        assert_eq!(
            actual.real_scalar_value().unwrap(),
            DynQuantity::new(expected, length)
        );
    }
}

#[test]
fn authored_jacobian_factors_survive_replay_with_signed_affine_values() {
    for (mapping, sign) in [("2*xi+eta", 1.0), ("2[m]-2*xi+eta", -1.0)] {
        for (operator, expected) in [
            ("jacobian_determinant", sign * 6.0),
            ("volume_jacobian", 6.0),
            ("map_orientation", sign),
        ] {
            let source = format!(
                r#"model M() {{
                domain reference=box(0,1,0,1);
                domain body=box(0,3,0,3);
                coordinate xi:m on reference from reference[0];
                coordinate eta:m on reference from reference[1];
                coordinate x:m on body from body[0];
                coordinate y:m on body from body[1];
                variable anchor:1;
                relation r {{anchor=0;}}
                observable factor:1=evaluate(
                    {operator}(from=(xi,eta),at=(x={mapping},y=3*eta)),
                    at=(xi=0.25[m],eta=0.5[m]));
            }}"#
            );
            let document = ModelDocument::compile("jacobian.eqi", &source).unwrap();
            let replay = ModelDocument::replay(&document.canonical_json().unwrap()).unwrap();
            assert_eq!(document.program(), replay.program());
            let observable = replay
                .program()
                .nodes()
                .find_map(|node| match node {
                    KernelNode::Observable(value) => Some(value.id()),
                    _ => None,
                })
                .unwrap();
            let actual = replay
                .program()
                .evaluate_finite_observable(observable, &mut |_| None)
                .unwrap()
                .real_scalar_value()
                .unwrap();
            // J=[±2,1;0,3]. LU/log-exp evaluation is rounded; its sign is exact.
            assert_eq!(actual.dim(), DimExponents::DIMENSIONLESS);
            assert!(
                (actual.value() - expected).abs() <= 64.0 * f64::EPSILON * expected.abs(),
                "{operator}: {actual:?}"
            );
        }
    }
}

#[test]
fn diagonal_maps_execute_beyond_four_coordinates() {
    use eqiora_compiler::StaticBindingValue;
    use eqiora_schema::kernel::AxisBounds;
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let interval = |upper| {
        StaticBindingValue::CoordinateInterval(
            AxisBounds::new(
                DynQuantity::new(0.0, length),
                DynQuantity::new(upper, length),
            )
            .unwrap(),
        )
    };
    for n in [1, 2, 5, 6, 9, 16] {
        let names = (0..2 * n).map(|i| format!("a{i}")).collect::<Vec<_>>();
        let declarations = names
            .iter()
            .map(|name| format!("support {name}:interval(m)"))
            .collect::<Vec<_>>()
            .join(",");
        let mut source = format!("model M({declarations}) {{");
        let (reference, target) = if n == 1 {
            ("a0", "a1")
        } else {
            source.push_str(&format!(
                "support reference:product({});support target:product({});",
                names[..n].join(","),
                names[n..].join(",")
            ));
            ("reference", "target")
        };
        for i in 0..n {
            source.push_str(&format!("coordinate q{i}:m on {reference} from a{i}; coordinate x{i}:m on {target} from a{};", n+i));
        }
        let coordinates = (0..n)
            .map(|i| format!("q{i}"))
            .collect::<Vec<_>>()
            .join(",");
        let mapping = (0..n)
            .map(|i| format!("x{i}=2*q{i}"))
            .collect::<Vec<_>>()
            .join(",");
        let point = (0..n)
            .map(|i| format!("q{i}=0.25[m]"))
            .collect::<Vec<_>>()
            .join(",");
        source.push_str(&format!("variable anchor:1;relation r{{anchor=0;}}observable factor:1=evaluate(volume_jacobian(from=({coordinates}),at=({mapping})),at=({point}));}}"));
        let bindings = names
            .iter()
            .enumerate()
            .map(|(i, name)| (name.as_str(), interval(if i < n { 1.0 } else { 2.0 })))
            .collect::<Vec<_>>();
        let document =
            ModelDocument::compile_selected("diagonal.eqi", &source, "M", &bindings).unwrap();
        let replay = ModelDocument::replay(&document.canonical_json().unwrap()).unwrap();
        let observable = replay
            .program()
            .nodes()
            .find_map(|node| match node {
                KernelNode::Observable(value) => Some(value.id()),
                _ => None,
            })
            .unwrap();
        let actual = replay
            .program()
            .evaluate_finite_observable(observable, &mut |_| None)
            .unwrap()
            .real_scalar_value()
            .unwrap();
        // Each independent axis doubles, so the volume scale is the product 2^n.
        let expected = 2.0_f64.powi(n as i32);
        assert_eq!(actual.dim(), DimExponents::DIMENSIONLESS);
        assert!(
            (actual.value() - expected).abs() <= 64.0 * n as f64 * f64::EPSILON * expected,
            "n={n}: {actual:?}"
        );
    }
}
