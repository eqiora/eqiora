//! Moving-coordinate time rates use the same retained map and scalar calculus.
use eqiora_api::ModelDocument;
use eqiora_core::{DimExponents, DynQuantity, ScalarDomain, ValueLiteral, ValueType};
use eqiora_schema::kernel::{ExprNode, KernelNode, SymbolRef};

fn source(expression: &str, unit: &str) -> String {
    format!(
        r#"model Moving() {{
        domain reference=box(0,1,0,1);
        domain body=box(0,4,0,4);
        coordinate xi:m on reference from reference[0];
        coordinate eta:m on reference from reference[1];
        coordinate x:m on body from body[0];
        coordinate y:m on body from body[1];
        state anchor:1;
        initial {{ anchor=0; }}
        relation evolution {{ derivative(anchor)=0; }}
        observable rate:{unit}=evaluate({expression},at=(xi=0.25[m],eta=0.5[m]));
        }}"#
    )
}

fn evaluate(source: &str, time: f64) -> DynQuantity {
    let document = ModelDocument::compile("moving-pullback.eqi", source).unwrap();
    let bytes = document.canonical_json().unwrap();
    let document = ModelDocument::replay(&bytes).unwrap();
    assert_eq!(document.canonical_json().unwrap(), bytes);
    let observable = document
        .program()
        .nodes()
        .find_map(|node| match node {
            KernelNode::Observable(value) => Some(value.id()),
            _ => None,
        })
        .unwrap();
    let seconds = DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).unwrap();
    document
        .program()
        .evaluate_finite_observable(observable, &mut |symbol| {
            matches!(symbol, SymbolRef::Time).then(|| {
                ValueLiteral::from_real(
                    ValueType::scalar(ScalarDomain::Real, seconds).unwrap(),
                    time,
                )
                .unwrap()
            })
        })
        .unwrap()
        .real_scalar_value()
        .unwrap()
}

#[test]
fn moving_pullback_rate_and_relative_transport_have_independent_values() {
    let q = "x+2*y-2[m/s]*time()";
    for map in [
        "x=(1+0.5[1/s]*time())*xi,y=eta+0.25[m/s]*time()",
        "y=eta+0.25[m/s]*time(),x=(1+0.5[1/s]*time())*xi",
    ] {
        let pull = |value| format!("pullback({value},from=(xi,eta),at=({map}))");
        let rate = format!("derivative({})", pull(q));
        let material = format!(
            "{rate}+(2[m/s]-0.5[1/s]*xi)*{}-0.25[m/s]*{}",
            pull(&format!("partial({q},wrt=x)")),
            pull(&format!("partial({q},wrt=y)")),
        );
        // q-hat=(1+t/2)*xi+2*eta-3t/2. Its reference rate is xi/2-3/2.
        // For physical v=(2,0), (v-w).grad(q)=3/2-xi/2, independently.
        let velocity = DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).unwrap();
        for time in [0.0, 1.0] {
            assert_eq!(
                evaluate(&source(&rate, "m/s"), time),
                DynQuantity::new(-1.375, velocity)
            );
            assert_eq!(
                evaluate(&source(&material, "m/s"), time),
                DynQuantity::new(0.0, velocity)
            );
        }
    }
    let fixed = format!("derivative(pullback({q},from=(xi,eta),at=(x=xi,y=eta)))");
    assert_eq!(evaluate(&source(&fixed, "m/s"), 1.0).value(), -2.0);
}

#[test]
fn compound_mapped_values_keep_product_and_higher_time_rules() {
    let pulled = "pullback(x-2[m/s]*time(),from=(xi,eta),at=(x=(1+0.5[1/s]*time())*xi,y=eta))";
    // At t=1, q-hat=-13/8 and its rate=-15/8. d(q-hat²)/dt=195/32.
    let squared = format!("derivative({pulled}*{pulled})");
    assert_eq!(
        evaluate(&source(&squared, "m^2/s"), 1.0).value(),
        195.0 / 32.0
    );
    let second = format!("derivative(derivative({pulled}))");
    assert_eq!(evaluate(&source(&second, "m/s^2"), 1.0).value(), 0.0);
}

#[test]
fn mapped_unknown_state_retains_its_rate_and_coordinate_partials() {
    let pulled = "pullback(q,from=(xi,eta),at=(x=(1+0.5[1/s]*time())*xi,y=eta))";
    let with_field = |expression: &str, unit| {
        source(expression, unit).replace(
            "state anchor:1;",
            "state q:m on body in smooth; state anchor:1;",
        )
    };
    let document = ModelDocument::compile(
        "mapped-state.eqi",
        &with_field(&format!("derivative({pulled})"), "m/s"),
    )
    .unwrap();
    let replay = ModelDocument::replay(&document.canonical_json().unwrap()).unwrap();
    let nodes = replay
        .program()
        .nodes()
        .find_map(|node| match node {
            KernelNode::Observable(value) => Some(value.expression().nodes()),
            _ => None,
        })
        .unwrap();
    assert!(
        nodes
            .iter()
            .any(|node| matches!(node, ExprNode::Symbol(SymbolRef::Derivative(_, _))))
    );
    assert!(
        nodes
            .iter()
            .any(|node| matches!(node, ExprNode::CoordinatePartial { .. }))
    );
    // No represented mixed space-time Field derivative is available in this profile.
    let errors = ModelDocument::compile(
        "mixed-rate.eqi",
        &with_field(&format!("derivative(derivative({pulled}))"), "m/s^2"),
    )
    .unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.message().contains("mixed-derivative representation"))
    );
    let algebraic =
        with_field(&format!("derivative({pulled})"), "m/s").replace("state q:m", "variable q:m");
    assert!(ModelDocument::compile("algebraic-rate.eqi", &algebraic).is_err());
}

#[test]
fn map_volume_rate_retains_expansion_reflection_and_inventory_terms() {
    for reflection in [false, true] {
        let x = if reflection {
            "(1+0.5[1/s]*time())*(1[m]-xi)"
        } else {
            "(1+0.5[1/s]*time())*xi"
        };
        let map = format!("from=(xi,eta),at=(x={x},y=(1+0.5[1/s]*time())*eta)");
        // lambda=1+t/2: |J|=lambda², d|J|/dt=lambda. Reflection only
        // negates the signed determinant, never the positive inventory.
        for (time, rate) in [(0.0, 1.0), (1.0, 1.5)] {
            for (factor, expected) in [
                ("volume_jacobian", rate),
                (
                    "jacobian_determinant",
                    if reflection { -rate } else { rate },
                ),
                ("map_orientation", 0.0),
            ] {
                let expression = format!("derivative({factor}({map}))");
                assert_eq!(
                    evaluate(&source(&expression, "1/s"), time).value(),
                    expected
                );
            }
            let inventory = format!("derivative(2[kg/m^2]*volume_jacobian({map}))");
            assert_eq!(
                evaluate(&source(&inventory, "kg/m^2/s"), time).value(),
                2.0 * rate
            );
        }
    }
    // A constant physical density has zero scalar rate, distinct from inventory.
    assert_eq!(
        evaluate(&source("derivative(2[kg/m^2])", "kg/m^2/s"), 1.0).value(),
        0.0
    );
    let fixed = "derivative(volume_jacobian(from=(xi,eta),at=(x=xi,y=eta)))";
    assert_eq!(evaluate(&source(fixed, "1/s"), 1.0).value(), 0.0);
}

#[test]
fn map_rate_model_rejects_the_displaced_epoch() {
    let document = ModelDocument::compile(
        "rate-epoch.eqi",
        &source(
            "derivative(volume_jacobian(from=(xi,eta),at=(x=(1+0.5[1/s]*time())*xi,y=eta)))",
            "1/s",
        ),
    )
    .unwrap();
    let mut wire: serde_json::Value =
        serde_json::from_slice(&document.canonical_json().unwrap()).unwrap();
    assert_eq!(wire["schema"], "eqiora.model-envelope/v45");
    wire["schema"] = serde_json::json!("eqiora.model-envelope/v44");
    let errors = ModelDocument::replay(&serde_json::to_vec(&wire).unwrap()).unwrap_err();
    assert!(errors.iter().any(|error| {
        error
            .message()
            .contains("unsupported eqiora.model-envelope/v45")
    }));
}

#[test]
fn map_rate_fingerprint_retains_ordered_directions() {
    let expression = "derivative(volume_jacobian(from=(xi,eta),at=(x=(1+0.5[1/s]*time())*xi,y=(1+0.25[1/s]*time())*eta)))";
    let document = ModelDocument::compile("rate-identity.eqi", &source(expression, "1/s")).unwrap();
    let bytes = document.canonical_json().unwrap();
    let replay = ModelDocument::replay(&bytes).unwrap();
    assert!(document.structurally_equivalent(&replay).unwrap());

    let mut changed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(reverse_directions(&mut changed), 1);
    // Both directions have length/time units and the same support. Swapping
    // their row roles remains a valid action but must change its meaning.
    let changed = ModelDocument::replay(&serde_json::to_vec(&changed).unwrap()).unwrap();
    assert!(!document.structurally_equivalent(&changed).unwrap());
    assert!(
        ModelDocument::compile(
            "higher-rate.eqi",
            &source(&format!("derivative({expression})"), "1/s^2"),
        )
        .is_err()
    );
}

#[test]
fn reference_inventory_law_replays_an_independent_map_rate_correspondence() {
    for storage in [
        "2[kg/m^2]*volume_jacobian(from=(xi,eta),at=(x=(1+0.5[1/s]*time())*xi,y=(1+0.5[1/s]*time())*eta))",
        "pullback(2[kg/m^2],from=(xi,eta),at=(x=(1+0.5[1/s]*time())*xi,y=eta))*volume_jacobian(from=(xi,eta),at=(x=(1+0.5[1/s]*time())*xi,y=eta))",
    ] {
        let source = source("0", "1").replace(
            "observable rate:1=",
            &format!("law inventory on reference {{ storage {storage}; flux -0[kg/m/s]*grad(xi); source 0[kg/m^2/s]; }} observable rate:1="),
        );
        // This tests admission of d(storage)/dt, not satisfaction or numerical
        // execution of the authored balance with its deliberately zero source.
        let document = ModelDocument::compile("mapped-law.eqi", &source).unwrap();
        let replay = ModelDocument::replay(&document.canonical_json().unwrap()).unwrap();
        let relation = replay
            .program()
            .nodes()
            .find_map(|node| match node {
                KernelNode::Relation(relation)
                    if matches!(
                        relation.meaning(),
                        eqiora_schema::kernel::RelationMeaning::Conservation(_)
                    ) =>
                {
                    Some(relation)
                }
                _ => None,
            })
            .unwrap();
        let eqiora_schema::kernel::RelationMeaning::Conservation(terms) = relation.meaning() else {
            unreachable!()
        };
        let (storage, accumulation) = terms.storage().unwrap();
        relation
            .expression()
            .verify_time_derivative(storage, accumulation)
            .unwrap();

        // The swapped directions still have the right dimensions and support.
        // Replay must reject their false correspondence to the retained storage.
        let mut wrong: serde_json::Value =
            serde_json::from_slice(&document.canonical_json().unwrap()).unwrap();
        assert!(reverse_directions(&mut wrong) > 0);
        let errors = ModelDocument::replay(&serde_json::to_vec(&wrong).unwrap()).unwrap_err();
        assert!(errors.iter().any(|error| {
            error
                .message()
                .contains("storage accumulation correspondence")
        }));
    }
}

fn reverse_directions(value: &mut serde_json::Value) -> usize {
    match value {
        serde_json::Value::Object(object) => {
            if object.get("op").and_then(serde_json::Value::as_str)
                == Some("coordinate-map-factor-action")
            {
                object["directions"].as_array_mut().unwrap().reverse();
                1
            } else {
                object.values_mut().map(reverse_directions).sum()
            }
        }
        serde_json::Value::Array(values) => values.iter_mut().map(reverse_directions).sum(),
        _ => 0,
    }
}

#[test]
fn unknown_mapped_density_storage_reaches_law_admission_and_replay() {
    let map = "from=(xi,eta),at=(x=(1+0.5[1/s]*time())*xi,y=eta)";
    let storage = format!("pullback(q,{map})*volume_jacobian({map})");
    let source = source("0", "1")
        .replace("state anchor:1;", "state q:kg/m^2 on body in smooth; state anchor:1;")
        .replace("observable rate:1=", &format!("law inventory on reference {{ storage {storage}; flux -0[kg/m/s]*grad(xi); source 0[kg/m^2/s]; }} observable rate:1="));
    let document = ModelDocument::compile("mapped-density-law.eqi", &source).unwrap();
    let bytes = document.canonical_json().unwrap();
    let replay = ModelDocument::replay(&bytes).unwrap();
    assert_eq!(replay.canonical_json().unwrap(), bytes);
    // This reaches unknown-density storage correspondence, not a numerical solve.
    assert!(
        replay
            .program()
            .nodes()
            .any(|node| matches!(node, KernelNode::Relation(relation)
        if matches!(relation.meaning(), eqiora_schema::kernel::RelationMeaning::Conservation(_))))
    );
}
