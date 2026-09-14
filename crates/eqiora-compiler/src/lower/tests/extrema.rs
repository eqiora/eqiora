use super::*;
use eqiora_core::{ValueLiteral, ValueType};
use eqiora_schema::kernel::{ComparisonOp, ExprNode};

#[test]
fn native_extrema_lower_exact_integer_operands_in_authored_order() {
    let range = TextRange::new(0, 0);
    let literal = |value| {
        LoweringExpression::literal(
            ValueLiteral::from_integer(
                ValueType::scalar(
                    eqiora_core::ScalarDomain::Integer,
                    DimExponents::DIMENSIONLESS,
                )
                .expect("admitted numeric scalar type"),
                value,
            )
            .unwrap(),
            range,
        )
    };
    for minimum in [true, false] {
        let equation = LoweringEquation {
            kind: eqiora_schema::kernel::RelationConditionKind::Equality,
            left: LoweringExpression::extremum(
                minimum,
                literal(9_007_199_254_740_993),
                literal(9_007_199_254_740_992),
                range,
            ),
            right: literal(if minimum {
                9_007_199_254_740_992
            } else {
                9_007_199_254_740_993
            }),
            contextual_left_zero: false,
            contextual_right_zero: false,
            range,
        };
        let result = expression::lower_relation(
            "native",
            range,
            &ActivationSyntax::Continuous,
            None,
            &[equation],
            false,
            &BTreeMap::new(),
        )
        .unwrap();
        let (left, right) = result
            .expression
            .nodes()
            .iter()
            .find_map(|node| match node {
                ExprNode::Select { condition, then_value, else_value } => {
                    let expected = if minimum { ComparisonOp::LessEqual } else { ComparisonOp::GreaterEqual };
                    assert!(matches!(result.expression.node(*condition), Some(ExprNode::Compare(op, a, b)) if *op == expected && a == then_value && b == else_value));
                    Some((*then_value, *else_value))
                },
                _ => None,
            })
            .unwrap();
        for (id, expected) in [
            (left, 9_007_199_254_740_993),
            (right, 9_007_199_254_740_992),
        ] {
            assert!(
                matches!(result.expression.node(id),Some(ExprNode::Constant(value)) if value.integer_scalar_value()==Some(expected))
            );
        }
    }
}
