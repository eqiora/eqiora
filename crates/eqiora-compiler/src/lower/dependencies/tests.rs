use super::*;
use crate::math::tensor::Operation;
use eqiora_lang::TextRange;

#[test]
fn tensor_arguments_retain_symbol_and_structural_dependencies() {
    let range = TextRange::new(0, 1);
    let expression = LoweringExpression::tensor(
        Operation::Component(vec![0]),
        vec![
            LoweringExpression::name("input".into(), range)
                .with_structural_parameters(["extent".into()]),
        ],
        range,
    );
    assert_eq!(
        expression.referenced_names(),
        BTreeSet::from(["input".into(), "extent".into()])
    );
    assert_eq!(
        expression.structural_parameters(),
        BTreeSet::from(["extent".into()])
    );
}

fn port(index: usize) -> LoweringExpression {
    let range = TextRange::new(0, 1);
    LoweringExpression::call(
        "across".into(),
        LoweringExpression::name(format!("port{index}"), range),
        range,
    )
    .with_structural_parameters([format!("size{index}")])
}

#[test]
fn both_analyses_visit_every_nested_operand() {
    use eqiora_core::ValueLiteral;
    use eqiora_schema::kernel::CoordinateMapFactor;
    let range = TextRange::new(0, 1);
    let expressions = [
        (
            LoweringExpression::tensor(Operation::Outer, vec![port(0), port(1)], range),
            2,
        ),
        (
            LoweringExpression::case(
                port(0),
                vec![
                    (ValueLiteral::boolean(false), port(1)),
                    (ValueLiteral::boolean(true), port(2)),
                ],
                range,
            ),
            3,
        ),
        (LoweringExpression::require(port(0), port(1), range), 2),
        (
            LoweringExpression::select(port(0), port(1), port(2), range),
            3,
        ),
        (LoweringExpression::partial(port(0), port(1), range), 2),
        (
            LoweringExpression::evaluate_at(port(0), vec![(port(1), port(2))], None, range),
            3,
        ),
        (
            LoweringExpression::pullback(port(0), vec![port(1)], vec![(port(2), port(3))], range),
            4,
        ),
        (
            LoweringExpression::coordinate_map_factor(
                CoordinateMapFactor::SignedJacobian,
                vec![port(0)],
                vec![(port(1), port(2))],
                range,
            ),
            3,
        ),
    ];
    for (expression, count) in expressions {
        let ports = (0..count)
            .map(|index| format!("port{index}"))
            .collect::<BTreeSet<_>>();
        let structural = (0..count)
            .map(|index| format!("size{index}"))
            .collect::<BTreeSet<_>>();
        assert_eq!(expression.structural_parameters(), structural);
        assert_eq!(
            expression.referenced_names(),
            ports.union(&structural).cloned().collect()
        );
        let mut found = BTreeSet::new();
        assert!(expression.collect_physical_port_names(&mut found));
        assert_eq!(found, ports);
    }
}

#[test]
fn shared_nodes_keep_each_edges_structural_parameters() {
    let range = TextRange::new(0, 1);
    let shared = port(0);
    let expression = LoweringExpression::array(
        vec![
            shared
                .clone()
                .with_structural_parameters(["left_extent".into()]),
            shared.with_structural_parameters(["right_extent".into()]),
        ],
        range,
    );
    assert_eq!(
        expression.structural_parameters(),
        BTreeSet::from(["size0".into(), "left_extent".into(), "right_extent".into(),])
    );
    assert_eq!(
        expression.referenced_names(),
        BTreeSet::from([
            "port0".into(),
            "size0".into(),
            "left_extent".into(),
            "right_extent".into(),
        ])
    );
    let mut found = BTreeSet::new();
    assert!(expression.collect_physical_port_names(&mut found));
    assert_eq!(found, BTreeSet::from(["port0".into()]));
}

#[test]
fn physical_port_discovery_keeps_its_unsupported_boundary() {
    use std::sync::Arc;
    let range = TextRange::new(0, 1);
    let action =
        LoweringExpression::coordinate_map_factor_action(port(0), port(1), vec![port(2)], range);
    assert_eq!(
        action.referenced_names(),
        BTreeSet::from([
            "port0".into(),
            "port1".into(),
            "port2".into(),
            "size0".into(),
            "size1".into(),
            "size2".into(),
        ])
    );
    assert!(!action.collect_physical_port_names(&mut BTreeSet::new()));
    for node in [
        LoweringExpressionNode::UnknownMath("unknown".into()),
        LoweringExpressionNode::InvalidValue("invalid"),
        LoweringExpressionNode::Unsupported,
    ] {
        let expression = LoweringExpression {
            node: Arc::new(node),
            range,
            structural_parameters: None,
        };
        assert!(!expression.collect_physical_port_names(&mut BTreeSet::new()));
    }
}
