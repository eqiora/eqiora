use super::*;
use eqiora_core::{DimExponents, ScalarDomain};
use eqiora_schema::kernel::{ComparisonOp, UnaryMathFunction};
use serde_json::json;

fn definition() -> PureOperatorDefinition {
    let area = DimExponents::from_integers([0, 2, 0, 0, 0, 0, 0]).unwrap();
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let real = PureValueClass::invariant_scalar()
        .with_scalar_domain(ScalarDomain::Real)
        .unwrap();
    let mut b =
        CalculusBuilder::new([real.with_dimension(area)], real.with_dimension(length)).unwrap();
    let x = b
        .push(CalculusNode::FormalComponent {
            formal: 0,
            axes: Box::new([]),
        })
        .unwrap();
    let zero = b
        .push(CalculusNode::Rational {
            value: ExactRational::new(0, 1).unwrap(),
            dimension: area,
        })
        .unwrap();
    let nonnegative = b
        .push(CalculusNode::Compare(ComparisonOp::GreaterEqual, x, zero))
        .unwrap();
    let no = b.push(CalculusNode::Boolean(false)).unwrap();
    let yes = b.push(CalculusNode::Not(no)).unwrap();
    let both = b.push(CalculusNode::And(nonnegative, yes)).unwrap();
    let either = b.push(CalculusNode::Or(both, no)).unwrap();
    let sqrt = b
        .push(CalculusNode::UnaryMath(UnaryMathFunction::Sqrt, x))
        .unwrap();
    let one = b
        .push(CalculusNode::Rational {
            value: ExactRational::new(1, 1).unwrap(),
            dimension: length,
        })
        .unwrap();
    let selected = b
        .push(CalculusNode::Select {
            condition: both,
            then_value: sqrt,
            else_value: one,
        })
        .unwrap();
    let required = b
        .push(CalculusNode::Require {
            condition: either,
            value: selected,
        })
        .unwrap();
    b.finish(required).unwrap()
}

#[test]
fn mixed_calculus_wire_retains_typed_rationals_and_ordered_guard_roles() {
    let original = definition();
    let wire = WirePureOperatorDefinition::encode(&original);
    let json = serde_json::to_value(&wire).unwrap();
    assert_eq!(
        json["nodes"][1],
        json!({"op":"rational","numerator":0,"denominator":1,"dimension":[[0,1],[2,1],[0,1],[0,1],[0,1],[0,1],[0,1]]})
    );
    assert_eq!(
        json["nodes"][2],
        json!({"op":"compare","comparison":"greater-equal","left":0,"right":1})
    );
    assert_eq!(json["nodes"][3], json!({"op":"boolean","value":false}));
    assert_eq!(
        json["nodes"][7],
        json!({"op":"unary-math","function":"sqrt","value":0})
    );
    assert_eq!(
        json["nodes"][9],
        json!({"op":"select","condition":5,"then_value":7,"else_value":8})
    );
    assert_eq!(
        json["nodes"][10],
        json!({"op":"require","condition":6,"value":9})
    );
    let replay: WirePureOperatorDefinition = serde_json::from_value(json).unwrap();
    assert_eq!(replay.rebuild_and_validate_digest().unwrap(), original);
}

#[test]
fn mixed_calculus_wire_rejects_invalid_types_units_references_and_identity() {
    let original = serde_json::to_value(WirePureOperatorDefinition::encode(&definition())).unwrap();
    for mutation in 0..8 {
        let mut invalid = original.clone();
        match mutation {
            0 => invalid["nodes"][9]["condition"] = json!(99),
            1 => invalid["nodes"][10]["condition"] = json!(7),
            2 => invalid["nodes"][9]["else_value"] = json!(1),
            3 => invalid["nodes"][7]["function"] = json!("sin"),
            4 => invalid["nodes"][1]["dimension"][1] = json!([2, 2]),
            5 => invalid["nodes"][3]["value"] = json!(0),
            6 => invalid["nodes"][2]["comparison"] = json!("less-equal"),
            _ => {
                invalid["nodes"][1]
                    .as_object_mut()
                    .unwrap()
                    .remove("dimension");
            }
        }
        let rejected = match serde_json::from_value::<WirePureOperatorDefinition>(invalid) {
            Err(_) => true,
            Ok(wire) => wire.rebuild_and_validate_digest().is_err(),
        };
        assert!(rejected, "mutation {mutation}");
    }
}

#[test]
fn ordered_partial_history_round_trips_even_when_both_mixed_values_are_zero() {
    let class = PureValueClass::invariant_scalar()
        .with_dimension(DimExponents::DIMENSIONLESS)
        .with_scalar_domain(ScalarDomain::Real)
        .unwrap();
    let make = |order: [u16; 2]| {
        let mut builder = CalculusBuilder::new([class, class], class).unwrap();
        let x = builder
            .push(CalculusNode::FormalComponent {
                formal: 0,
                axes: Box::new([]),
            })
            .unwrap();
        let first = builder.partial(x, order[0]).unwrap();
        let second = builder.partial(first, order[1]).unwrap();
        builder.finish(second).unwrap()
    };
    let xy = make([0, 1]);
    let yx = make([1, 0]);
    assert_ne!(xy.digest(), yx.digest());
    for definition in [xy, yx] {
        let wire = WirePureOperatorDefinition::encode(&definition);
        let json = serde_json::to_value(&wire).unwrap();
        let replay: WirePureOperatorDefinition = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(replay.rebuild_and_validate_digest().unwrap(), definition);
        let mut third = json;
        let old_root = third["root"].as_u64().unwrap();
        let nodes = third["nodes"].as_array_mut().unwrap();
        let next = nodes.len();
        nodes.push(json!({"op":"differentiated", "source":old_root, "value":old_root, "wrt":0}));
        third["root"] = json!(next);
        let invalid: WirePureOperatorDefinition = serde_json::from_value(third).unwrap();
        let error = invalid.rebuild_and_validate_digest().unwrap_err();
        assert!(error.message().contains("derivative order"), "{error:?}");
    }
}

#[test]
fn retained_derivative_wire_rejects_a_false_value_before_digest_comparison() {
    let class = PureValueClass::invariant_scalar()
        .with_dimension(DimExponents::DIMENSIONLESS)
        .with_scalar_domain(ScalarDomain::Real)
        .unwrap();
    let mut builder = CalculusBuilder::new([class], class).unwrap();
    let x = builder
        .push(CalculusNode::FormalComponent {
            formal: 0,
            axes: Box::new([]),
        })
        .unwrap();
    let derivative = builder.partial(x, 0).unwrap();
    let definition = builder.finish(derivative).unwrap();
    let mut wire = serde_json::to_value(WirePureOperatorDefinition::encode(&definition)).unwrap();
    // d(x)/dx=1, independently; substitute x while keeping valid references and units.
    wire["nodes"][derivative.index() as usize]["value"] = json!(x.index());
    let invalid: WirePureOperatorDefinition = serde_json::from_value(wire).unwrap();
    let error = invalid.rebuild_and_validate_digest().unwrap_err();
    assert!(
        error.message().contains("retained derivative value"),
        "{error:?}"
    );
}
