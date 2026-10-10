use super::*;

#[test]
fn boundary_operator_wire_requires_and_retains_each_exact_target() {
    use eqiora_core::Id;
    let first = Id::<kinds::Domain>::new();
    let second = Id::<kinds::Domain>::new();
    let mut builder = ExprDagBuilder::new();
    let value = builder.symbol(SymbolRef::Field(Id::new())).unwrap();
    let trace = builder.trace(value, first).unwrap();
    let normal = builder.normal_component(value, second).unwrap();
    let expression = builder.finish([trace, normal]).unwrap();
    let wire = WireExpression::encode(&expression).unwrap();
    let json = serde_json::to_value(&wire).unwrap();
    let replay: WireExpression = serde_json::from_value(json.clone()).unwrap();
    assert_eq!(replay.decode().unwrap(), expression);
    for target in [first, second] {
        assert!(
            wire.semantic_references()
                .iter()
                .any(|id| { id.typed::<kinds::Domain>().ok() == Some(target) })
        );
    }
    // Neither operator may recover its target from an enclosing Relation.
    for index in [1, 2] {
        let mut missing = json.clone();
        missing["nodes"][index]
            .as_object_mut()
            .unwrap()
            .remove("on");
        assert!(serde_json::from_value::<WireExpression>(missing).is_err());
    }
}

#[test]
fn coordinate_factor_wire_retains_factor_kind_and_map_bindings() {
    use eqiora_core::Id;
    for factor in [
        CoordinateMapFactor::SignedJacobian,
        CoordinateMapFactor::VolumeScale,
        CoordinateMapFactor::Orientation,
    ] {
        let source = Id::<kinds::Domain>::new();
        let target = Id::<kinds::Domain>::new();
        let mut builder = ExprDagBuilder::new();
        let xi = builder.coordinate(source, source, 0).unwrap();
        let x = builder.coordinate(target, target, 0).unwrap();
        let root = builder
            .coordinate_map_factor(factor, vec![xi], vec![(x, xi)])
            .unwrap();
        let expression = builder.finish([root]).unwrap();
        let wire = WireExpression::encode(&expression).unwrap();
        let bytes = serde_json::to_vec(&wire).unwrap();
        let replay: WireExpression = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(replay.decode().unwrap(), expression);
        let mut malformed = replay;
        if let WireExpressionNode::CoordinateMapFactor { at, .. } = &mut malformed.nodes[2] {
            at[0].1 = 2;
        }
        assert!(malformed.decode().is_err());
    }
}

#[test]
fn pullback_wire_rejects_forward_references_and_empty_inventories() {
    use eqiora_core::Id;
    let source = Id::<kinds::Domain>::new();
    let target = Id::<kinds::Domain>::new();
    let mut builder = ExprDagBuilder::new();
    let xi = builder.coordinate(source, source, 0).unwrap();
    let x = builder.coordinate(target, target, 0).unwrap();
    let pulled = builder.pullback(x, vec![xi], vec![(x, xi)]).unwrap();
    let expression = builder.finish([pulled]).unwrap();
    let wire = WireExpression::encode(&expression).unwrap();
    assert_eq!(wire.decode().unwrap(), expression);
    for (value, source, at) in [
        (2, vec![0], vec![(1, 0)]),
        (1, vec![2], vec![(1, 0)]),
        (1, vec![0], vec![(2, 0)]),
        (1, vec![0], vec![(1, 2)]),
        (1, vec![], vec![(1, 0)]),
        (1, vec![0], vec![]),
    ] {
        let mut malformed = wire.clone();
        malformed.nodes[2] = WireExpressionNode::Pullback { value, source, at };
        assert!(malformed.decode().is_err());
    }
}

#[cfg(test)]
mod typed_operation_tests {
    use super::*;
    use eqiora_core::{DimExponents, ScalarDomain, ValueLiteral, ValueType};

    #[test]
    fn array_index_and_complex_preserve_order_and_shared_operands() {
        let ty = ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
            .expect("valid fixture scalar type");
        let mut builder = ExprDagBuilder::new();
        let one = builder
            .constant(ValueLiteral::from_real(ty.clone(), 1.0).unwrap())
            .unwrap();
        let two = builder
            .constant(ValueLiteral::from_real(ty, 2.0).unwrap())
            .unwrap();
        let array = builder.array([two, one, two]).unwrap();
        let selected = builder.index(array, 1).unwrap();
        let complex = builder.complex(selected, two).unwrap();
        let expression = builder.finish([complex]).unwrap();
        let wire = WireExpression::encode(&expression).unwrap();
        let json = serde_json::to_value(&wire).unwrap();
        assert_eq!(
            json["nodes"][2],
            serde_json::json!({"op":"array","elements":[1,0,1]})
        );
        assert_eq!(
            json["nodes"][3],
            serde_json::json!({"op":"index","value":2,"index":1})
        );
        assert_eq!(
            json["nodes"][4],
            serde_json::json!({"op":"complex","real":3,"imag":1})
        );
        assert_eq!(wire.decode().unwrap(), expression);
        for bad in [
            WireExpressionNode::Array { elements: vec![2] },
            WireExpressionNode::Index { value: 2, index: 0 },
            WireExpressionNode::Complex { real: 0, imag: 2 },
        ] {
            let mut malformed = wire.clone();
            malformed.nodes[2] = bad;
            assert!(
                malformed.decode().is_err(),
                "forward/self operands must be rejected"
            );
        }
    }
}

#[cfg(test)]
mod transition_tests {
    use super::*;
    use eqiora_core::{Id, entity::kinds};

    #[test]
    fn transition_wire_retains_clock_and_operand_identity() {
        let clock = Id::<kinds::ClockDomain>::new();
        let field = Id::<kinds::Field>::new();
        let mut builder = ExprDagBuilder::new();
        let state = builder.symbol(SymbolRef::Field(field)).unwrap();
        let held = builder.hold(state).unwrap();
        let sampled = builder.sample(held, clock).unwrap();
        let expression = builder.finish([sampled]).unwrap();
        let wire = WireExpression::encode(&expression).unwrap();
        let encoded = serde_json::to_value(&wire).unwrap();
        assert_eq!(encoded["nodes"][1]["op"], "hold");
        assert_eq!(encoded["nodes"][1]["value"], 0);
        assert_eq!(encoded["nodes"][2]["op"], "sample");
        assert_eq!(encoded["nodes"][2]["value"], 1);
        assert_eq!(wire.decode().unwrap(), expression);
        let mut wrong_clock = wire.clone();
        if let WireExpressionNode::Sample { clock, .. } = &mut wrong_clock.nodes[2] {
            *clock = WireId::from_raw(field.erase());
        }
        assert!(wrong_clock.decode().is_err());
        let mut wrong_operand = wire;
        if let WireExpressionNode::Hold { value } = &mut wrong_operand.nodes[1] {
            *value = 2;
        }
        assert!(wrong_operand.decode().is_err());
    }
}

#[cfg(test)]
mod integer_operation_tests {
    use super::*;
    use eqiora_core::{DimExponents, ScalarDomain, ValueLiteral, ValueType};

    #[test]
    fn discrete_operations_preserve_shared_operands_and_reject_forward_references() {
        let ty = ValueType::scalar(ScalarDomain::Integer, DimExponents::DIMENSIONLESS)
            .expect("valid fixture scalar type");
        let mut builder = ExprDagBuilder::new();
        let left = builder
            .constant(ValueLiteral::from_integer(ty.clone(), 9_007_199_254_740_993).unwrap())
            .unwrap();
        let right = builder
            .constant(ValueLiteral::from_integer(ty, 2).unwrap())
            .unwrap();
        let quotient = builder.quotient(left, right).unwrap();
        let remainder = builder.remainder(left, right).unwrap();
        let real = builder.to_real(quotient).unwrap();
        let integer = builder.to_integer(real).unwrap();
        let expression = builder.finish([remainder, integer]).unwrap();
        let wire = WireExpression::encode(&expression).unwrap();
        let json = serde_json::to_value(&wire).unwrap();
        for (index, expected) in [
            (2, serde_json::json!({"op":"quotient","left":0,"right":1})),
            (3, serde_json::json!({"op":"remainder","left":0,"right":1})),
            (4, serde_json::json!({"op":"to-real","value":2})),
            (5, serde_json::json!({"op":"to-integer","value":4})),
        ] {
            assert_eq!(json["nodes"][index], expected);
        }
        assert_eq!(wire.decode().unwrap(), expression);
        for bad in [
            WireExpressionNode::Quotient { left: 2, right: 1 },
            WireExpressionNode::Remainder { left: 0, right: 2 },
            WireExpressionNode::ToReal { value: 2 },
            WireExpressionNode::ToInteger { value: 2 },
        ] {
            let mut malformed = wire.clone();
            malformed.nodes[2] = bad;
            assert!(malformed.decode().is_err());
        }
    }
    #[test]
    fn ordinal_wire_retains_the_exact_index_set_reference() {
        let set = eqiora_core::Id::new();
        let index = ValueLiteral::from_integer(ValueType::index(set, 4).unwrap(), 2).unwrap();
        let mut builder = ExprDagBuilder::new();
        let value = builder.constant(index).unwrap();
        let ordinal = builder.ordinal(value).unwrap();
        let expression = builder.finish([ordinal]).unwrap();
        let wire = WireExpression::encode(&expression).unwrap();
        let json = serde_json::to_value(&wire).unwrap();
        assert_eq!(
            json["nodes"][1],
            serde_json::json!({"op":"ordinal","value":0})
        );
        assert_eq!(
            wire.semantic_references(),
            vec![&WireId::from_raw(set.erase())]
        );
        assert_eq!(wire.decode().unwrap(), expression);
    }
    #[test]
    fn exact_comparison_and_boolean_operations_preserve_ordered_edges() {
        use eqiora_core::{DimExponents, ScalarDomain, ValueLiteral, ValueType};
        let mut builder = ExprDagBuilder::new();
        let ty = ValueType::scalar(ScalarDomain::Integer, DimExponents::DIMENSIONLESS)
            .expect("valid fixture scalar type");
        let a = builder
            .constant(ValueLiteral::from_integer(ty.clone(), 9_007_199_254_740_992).unwrap())
            .unwrap();
        let b = builder
            .constant(ValueLiteral::from_integer(ty, 9_007_199_254_740_993).unwrap())
            .unwrap();
        let operations = [
            (ComparisonOp::Equal, "equal"),
            (ComparisonOp::NotEqual, "not-equal"),
            (ComparisonOp::Less, "less"),
            (ComparisonOp::LessEqual, "less-equal"),
            (ComparisonOp::Greater, "greater"),
            (ComparisonOp::GreaterEqual, "greater-equal"),
        ];
        let mut roots = Vec::new();
        for (op, _) in operations {
            roots.push(builder.compare(op, a, b).unwrap());
        }
        let negated = builder.not(roots[0]).unwrap();
        let both = builder.and(roots[1], negated).unwrap();
        let either = builder.or(both, roots[2]).unwrap();
        roots.extend([negated, both, either]);
        let expression = builder.finish(roots).unwrap();
        let wire = WireExpression::encode(&expression).unwrap();
        let json = serde_json::to_value(&wire).unwrap();
        for (index, (_, spelling)) in operations.into_iter().enumerate() {
            assert_eq!(
                json["nodes"][index + 2],
                serde_json::json!({
                    "op":"compare", "comparison":spelling, "left":0, "right":1,
                })
            );
        }
        assert_eq!(json["nodes"][8], serde_json::json!({"op":"not","value":2}));
        assert_eq!(
            json["nodes"][9],
            serde_json::json!({"op":"and","left":3,"right":8})
        );
        assert_eq!(
            json["nodes"][10],
            serde_json::json!({"op":"or","left":9,"right":4})
        );
        assert_eq!(wire.decode().unwrap(), expression);
        let mut invalid = wire;
        invalid.nodes[2] = WireExpressionNode::Compare {
            comparison: WireComparisonOp::Equal,
            left: 0,
            right: 2,
        };
        assert!(
            invalid.decode().is_err(),
            "self references are not earlier operands"
        );
    }
}

#[test]
fn finite_operations_preserve_operand_roles_and_reject_forward_references() {
    use eqiora_core::{DimExponents, FiniteBasis, Id, ScalarDomain, ValueLiteral, ValueType};
    use eqiora_schema::kernel::{FiniteBinaryOperation as B, FiniteUnaryOperation as U};
    let basis = FiniteBasis::new(Id::new(), 2).unwrap();
    let mut builder = ExprDagBuilder::new();
    let map = builder
        .constant(
            ValueLiteral::from_real(
                ValueType::linear_map(
                    basis,
                    basis,
                    ScalarDomain::Complex,
                    DimExponents::DIMENSIONLESS,
                )
                .unwrap(),
                0.0,
            )
            .unwrap(),
        )
        .unwrap();
    let vector = builder
        .constant(
            ValueLiteral::from_real(
                ValueType::coordinates(basis, ScalarDomain::Complex, DimExponents::DIMENSIONLESS)
                    .unwrap(),
                0.0,
            )
            .unwrap(),
        )
        .unwrap();
    let transpose = builder.finite_unary(U::Transpose, map).unwrap();
    let adjoint = builder.finite_unary(U::Adjoint, vector).unwrap();
    let applied = builder.finite_binary(B::Apply, map, vector).unwrap();
    let composed = builder.finite_binary(B::Compose, map, map).unwrap();
    let paired = builder.finite_binary(B::Pair, adjoint, vector).unwrap();
    let product = builder
        .finite_binary(B::TensorProduct, vector, vector)
        .unwrap();
    let permutation = builder
        .finite_unary(U::PermuteFactors([1, 0]), product)
        .unwrap();
    let expression = builder
        .finish([transpose, applied, composed, paired, permutation])
        .unwrap();
    let wire = WireExpression::encode(&expression).unwrap();
    let json = serde_json::to_value(&wire).unwrap();
    assert_eq!(
        json["nodes"][2],
        serde_json::json!({"op":"finite-transpose","value":0})
    );
    assert_eq!(
        json["nodes"][3],
        serde_json::json!({"op":"finite-adjoint","value":1})
    );
    assert_eq!(
        json["nodes"][4],
        serde_json::json!({"op":"finite-apply","left":0,"right":1})
    );
    assert_eq!(
        json["nodes"][5],
        serde_json::json!({"op":"finite-compose","left":0,"right":0})
    );
    assert_eq!(
        json["nodes"][6],
        serde_json::json!({"op":"finite-pair","left":3,"right":1})
    );
    assert_eq!(
        json["nodes"][7],
        serde_json::json!({"op":"finite-tensor-product","left":1,"right":1})
    );
    assert_eq!(
        json["nodes"][8],
        serde_json::json!({"op":"finite-permutation","value":7,"order":[1,0]})
    );
    let restored: WireExpression = serde_json::from_value(json).unwrap();
    assert_eq!(restored.decode().unwrap(), expression);
    for invalid in [
        WireExpressionNode::FiniteTranspose { value: 2 },
        WireExpressionNode::FiniteAdjoint { value: 2 },
        WireExpressionNode::FiniteApply { left: 0, right: 2 },
        WireExpressionNode::FiniteCompose { left: 2, right: 0 },
        WireExpressionNode::FinitePair { left: 0, right: 2 },
        WireExpressionNode::FiniteTensorProduct { left: 0, right: 2 },
        WireExpressionNode::FinitePermutation {
            value: 2,
            order: [1, 0],
        },
    ] {
        let mut invalid_wire = wire.clone();
        invalid_wire.nodes[2] = invalid;
        assert!(invalid_wire.decode().is_err());
    }
}

#[test]
fn coordinate_factor_action_wire_retains_exact_operands_and_rejects_bad_rows() {
    use eqiora_core::{DimExponents, DynQuantity, Id};
    let reference = Id::<kinds::Domain>::new();
    let target = Id::<kinds::Domain>::new();
    let mut builder = ExprDagBuilder::new();
    let xi = builder.coordinate(reference, reference, 0).unwrap();
    let x = builder.coordinate(target, target, 0).unwrap();
    let time = builder.symbol(SymbolRef::Time).unwrap();
    let rate = builder
        .constant(DynQuantity::new(
            1.0,
            DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).unwrap(),
        ))
        .unwrap();
    let factor = builder
        .coordinate_map_factor(CoordinateMapFactor::VolumeScale, vec![xi], vec![(x, xi)])
        .unwrap();
    let action = builder
        .coordinate_map_factor_action(factor, time, vec![rate])
        .unwrap();
    let expression = builder.finish([action]).unwrap();
    let wire = WireExpression::encode(&expression).unwrap();
    let bytes = serde_json::to_vec(&wire).unwrap();
    let replay: WireExpression = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(replay.decode().unwrap(), expression);
    for (value, parameter, directions) in [
        (5, 2, vec![3]),
        (4, 5, vec![3]),
        (4, 2, vec![5]),
        (4, 2, vec![]),
    ] {
        let mut malformed = replay.clone();
        malformed.nodes[5] = WireExpressionNode::CoordinateMapFactorAction {
            value,
            parameter,
            directions,
        };
        assert!(malformed.decode().is_err());
    }
}
