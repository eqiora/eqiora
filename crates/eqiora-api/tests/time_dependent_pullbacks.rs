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
