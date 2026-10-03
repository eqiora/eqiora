//! Explicit uniform spatial component construction on an existing Cartesian frame.
use super::*;
use eqiora_core::{ValueFrame, ValueShape};
use eqiora_schema::kernel::typing::{ExpressionType, SpatialSupport};

pub(super) fn arguments(arguments: &eqiora_lang::CallArguments) -> Option<(&Expr, &Expr)> {
    let bindings = arguments.named()?;
    if bindings.len() != 2 {
        return None;
    }
    let mut frame = None;
    let mut components = None;
    for binding in bindings {
        match binding.name() {
            "frame" if frame.is_none() => frame = Some(binding.value()),
            "components" if components.is_none() => components = Some(binding.value()),
            _ => return None,
        }
    }
    Some((frame?, components?))
}

pub(super) fn frame_name(expression: &Expr) -> Option<&str> {
    let ExprKind::Call {
        callee,
        arguments: bindings,
    } = expression.kind()
    else {
        return None;
    };
    if callee.as_str() != "tensor_value" {
        return None;
    }
    let (frame, _) = arguments(bindings)?;
    match frame.kind() {
        ExprKind::Name(name) => Some(name),
        ExprKind::Path(path) => Some(path.as_str()),
        _ => None,
    }
}

pub(super) fn evaluate(
    file: &str,
    expression: &Expr,
    context: ExpressionContext<'_>,
    resolve: &mut impl FnMut(&str, TextRange) -> Result<SymbolicParameterValue, Diagnostic>,
    (resolve_clock, resolve_frame): StaticContexts<'_>,
    target: Option<&ValueType>,
    evaluate_values: bool,
) -> Result<EvaluatedParameter, Diagnostic> {
    let error = |message: &str| {
        source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            expression.range(),
            message,
        )
    };
    let frame = frame_name(expression).ok_or_else(|| {
        error("tensor_value requires a direct frame name and explicit component array")
    })?;
    let support = resolve_frame(frame).ok_or_else(|| {
        error("tensor_value frame is not an existing Cartesian volume or boundary")
    })?;
    let dimensions = match support {
        SpatialSupport::Volume { dimensions, .. } | SpatialSupport::Boundary { dimensions, .. } => {
            dimensions
        }
        SpatialSupport::Interface { .. } => {
            return Err(error(
                "tensor_value requires a declared volume or boundary frame",
            ));
        }
    };
    let ExprKind::Call {
        arguments: bindings,
        ..
    } = expression.kind()
    else {
        unreachable!()
    };
    let (_, components) = arguments(bindings).expect("validated tensor_value bindings");
    let (rank, leaves) = component_leaves(components, dimensions).map_err(error)?;
    let scalar_target = target
        .map(|value| ValueType::scalar(value.scalar_domain(), value.dimension()))
        .transpose()
        .map_err(|violation| error(&violation.to_string()))?;
    let mut operands = Vec::with_capacity(leaves.len());
    for leaf in leaves {
        if has_named_component_reference(leaf) {
            return Err(error(
                "tensor_value components must be closed expressions without named value references; bind an already typed tensor value for Parameter arithmetic",
            ));
        }
        let operand = match &scalar_target {
            Some(target) if evaluate_values => expression_eval::evaluate_initializer(
                file,
                leaf,
                context,
                resolve,
                target.clone(),
                "tensor component initializer",
                (&mut *resolve_clock, &mut *resolve_frame),
            )?,
            _ => expression_eval::evaluate_mode(
                file,
                leaf,
                context,
                resolve,
                (&mut *resolve_clock, &mut *resolve_frame),
                None,
                evaluate_values,
            )?,
        };
        if !operand.value_type.value_type().shape().is_scalar()
            || !matches!(
                operand.value_type.value_type().scalar_domain(),
                ScalarDomain::Real | ScalarDomain::Complex
            )
        {
            return Err(error(
                "tensor_value components require real or complex scalars",
            ));
        }
        operands.push(operand);
    }
    let types = operands
        .iter()
        .map(|value| ExpressionType::<()>::new(value.value_type.value_type().clone(), None))
        .collect::<Vec<_>>();
    let mut common = types[0].clone();
    for value in &types[1..] {
        common = common
            .equation(value.clone())
            .map_err(|violation| error(&violation.to_string()))?;
    }
    let extent = u32::try_from(dimensions)
        .map_err(|_| error("frame dimension is outside the portable shape range"))?;
    let shape = ValueShape::new(std::iter::repeat_n(extent, rank))
        .map_err(|violation| error(&violation.to_string()))?;
    let value_type = ValueType::shaped(
        common.value_type.scalar_domain(),
        common.dimension(),
        shape,
        ValueFrame::SpatialCartesian,
    )
    .map_err(|violation| error(&violation.to_string()))?;
    if let Some(target) = target {
        ExpressionType::<()>::new(value_type.clone(), None)
            .equation(ExpressionType::new(target.clone(), None))
            .map_err(|violation| error(&violation.to_string()))?;
    }
    let value = operands
        .iter()
        .map(|operand| operand.value.as_ref().and_then(|value| value.component(0)))
        .collect::<Option<Vec<_>>>()
        .map(|components| {
            ValueLiteral::new(value_type.clone(), components)
                .map_err(|violation| error(&violation.to_string()))
        })
        .transpose()?;
    Ok(EvaluatedParameter {
        expression: value
            .as_ref()
            .map(|value| LoweringExpression::literal(value.clone(), expression.range())),
        value,
        value_type: if operands
            .iter()
            .all(|value| matches!(value.value_type, EvaluatedType::Known(_)))
        {
            EvaluatedType::Known(value_type)
        } else {
            EvaluatedType::Deferred(value_type)
        },
        bare_literal: false,
        lineage: Some(ParameterLineage::Constant),
    })
}

fn has_named_component_reference(expression: &Expr) -> bool {
    let mut pending = vec![expression];
    while let Some(value) = pending.pop() {
        if value.resolved_enum().is_some() {
            continue;
        }
        match value.kind() {
            ExprKind::Name(_) | ExprKind::Member { .. } => return true,
            ExprKind::Path(path)
                if crate::math::constant(path).is_none() && path.as_str() != "math.i" =>
            {
                return true;
            }
            ExprKind::Unary { value, .. } => pending.push(value),
            ExprKind::Binary { left, right, .. } => pending.extend([left.as_ref(), right.as_ref()]),
            ExprKind::Call { arguments, .. } => pending.extend(arguments.expressions()),
            ExprKind::Case { value, arms } => {
                pending.push(value.as_ref());
                pending.extend(arms.iter().map(eqiora_lang::CaseArm::value));
            }
            ExprKind::Select {
                condition,
                then_value,
                else_value,
            } => {
                pending.extend([condition.as_ref(), then_value.as_ref(), else_value.as_ref()]);
            }
            ExprKind::Array(elements) => pending.extend(elements),
            ExprKind::Index { value, index } => pending.extend([value.as_ref(), index.as_ref()]),
            ExprKind::Slice {
                value,
                lower,
                upper,
            } => pending.extend([value.as_ref(), lower.as_ref(), upper.as_ref()]),
            _ => {}
        }
    }
    false
}

fn component_leaves(expression: &Expr, extent: usize) -> Result<(usize, Vec<&Expr>), &'static str> {
    let mut first = expression;
    let mut rank = 0;
    while let ExprKind::Array(values) = first.kind() {
        rank += 1;
        if rank > 4 {
            return Err("tensor_value admits spatial ranks one through four");
        }
        first = values.first().ok_or("tensor_value axes must be nonempty")?;
    }
    if rank == 0 {
        return Err("tensor_value components require an explicit nested array");
    }
    fn collect<'a>(
        value: &'a Expr,
        extent: usize,
        rank: usize,
        leaves: &mut Vec<&'a Expr>,
    ) -> Result<(), &'static str> {
        match (rank, value.kind()) {
            (0, ExprKind::Array(_)) => Err(
                "tensor_value component arrays must be rectangular with matching ambient extents",
            ),
            (0, _) => {
                leaves.push(value);
                Ok(())
            }
            (_, ExprKind::Array(values)) if values.len() == extent && extent != 0 => {
                for value in values {
                    collect(value, extent, rank - 1, leaves)?;
                }
                Ok(())
            }
            _ => Err(
                "tensor_value component arrays must be rectangular with matching ambient extents",
            ),
        }
    }
    let mut leaves = Vec::new();
    collect(expression, extent, rank, &mut leaves)?;
    Ok((rank, leaves))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn expression(components: &str) -> Expr {
        let source = format!(
            "model M() {{ let value = tensor_value(frame = grid, components = {components}); }}"
        );
        let document = eqiora_lang::parse("tensor.eqi", &source)
            .into_document()
            .unwrap();
        let eqiora_lang::Item::Let(value) = &document.models()[0].items()[0] else {
            panic!("tensor value");
        };
        value.value().clone()
    }

    #[test]
    fn tensor_bindings_resolve_roles_independently_of_order_and_reject_other_shapes() {
        for (call, accepted) in [
            ("tensor_value(components = [1,2], frame = grid)", true),
            ("tensor_value(frame = grid, components = [1,2])", true),
            ("tensor_value(grid, [1,2])", false),
            ("tensor_value(frame = grid, other = [1,2])", false),
            ("tensor_value(frame = grid)", false),
            (
                "tensor_value(frame = grid, components = [1,2], other = 0)",
                false,
            ),
        ] {
            let source = format!("model M() {{ let value = {call}; }}");
            let document = match eqiora_lang::parse("bindings.eqi", &source).into_document() {
                Ok(document) => document,
                Err(diagnostics) => {
                    assert!(
                        !accepted,
                        "valid tensor bindings rejected: {call}: {diagnostics:?}"
                    );
                    continue;
                }
            };
            let eqiora_lang::Item::Let(value) = &document.models()[0].items()[0] else {
                panic!("tensor binding fixture");
            };
            assert_eq!(
                frame_name(value.value()),
                accepted.then_some("grid"),
                "{call}"
            );
            if accepted {
                let ExprKind::Call {
                    arguments: bindings,
                    ..
                } = value.value().kind()
                else {
                    panic!("tensor call");
                };
                let (_, components) = arguments(bindings).unwrap();
                assert!(matches!(components.kind(), ExprKind::Array(values) if values.len() == 2));
            }
        }
    }

    fn construct(
        components: &str,
        target: Option<&ValueType>,
        boundary: bool,
        evaluate_values: bool,
    ) -> Result<EvaluatedParameter, Diagnostic> {
        evaluate(
            "tensor.eqi",
            &expression(components),
            ExpressionContext::Default,
            &mut |_, _| {
                Ok(SymbolicParameterValue {
                    value: None,
                    value_type: ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
                        .expect("valid numeric scalar type"),
                    expression: None,
                    lineage: None,
                })
            },
            (&mut |_| None, &mut |name| {
                (name == "grid").then(|| {
                    if boundary {
                        SpatialSupport::Boundary {
                            domain: "wall".into(),
                            parent: "grid".into(),
                            dimensions: 2,
                        }
                    } else {
                        SpatialSupport::Volume {
                            domain: "grid".into(),
                            dimensions: 2,
                        }
                    }
                })
            }),
            target,
            evaluate_values,
        )
    }
    #[test]
    fn explicit_components_preserve_order_frame_dimension_and_complex_parts() {
        let dim = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
        let target = ValueType::shaped(
            ScalarDomain::Real,
            dim,
            ValueShape::new([2, 2]).unwrap(),
            ValueFrame::SpatialCartesian,
        )
        .unwrap();
        let evaluated = construct("[[1,2],[3,4]]", Some(&target), true, true).unwrap();
        let literal = evaluated.value.unwrap();
        assert_eq!(literal.value_type(), &target);
        assert_eq!(
            literal.components().unwrap().collect::<Vec<_>>(),
            vec![(1., 0.), (2., 0.), (3., 0.), (4., 0.)]
        );
        let complex = construct("[math.complex(1,2),3]", None, false, true)
            .unwrap()
            .value
            .unwrap();
        assert_eq!(complex.component(0), Some((1., 2.)));
        assert_eq!(complex.component(1), Some((3., 0.)));
        assert_eq!(complex.value_type().array_rank(), 0);
        assert_eq!(complex.value_type().frame(), ValueFrame::SpatialCartesian);
        assert!(construct("[[1+2,2],[3,4]]", Some(&target), false, true).is_err());
    }
    #[test]
    fn constant_lineage_does_not_authorize_named_component_folding() {
        for components in ["[p,0]", "[alias,0]", "[owner.p,0]", "[p[0],0]"] {
            let error = evaluate(
                "tensor.eqi",
                &expression(components),
                ExpressionContext::Default,
                &mut |_, _| {
                    let value = ValueLiteral::from_real(
                        ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
                            .expect("valid numeric scalar type"),
                        2.0,
                    )
                    .unwrap();
                    Ok(SymbolicParameterValue {
                        value: Some(value.clone()),
                        value_type: value.value_type().clone(),
                        expression: Some(LoweringExpression::literal(value, TextRange::new(0, 0))),
                        lineage: Some(ParameterLineage::Constant),
                    })
                },
                (&mut |_| None, &mut |_| {
                    Some(SpatialSupport::Volume {
                        domain: "grid".into(),
                        dimensions: 2,
                    })
                }),
                None,
                true,
            )
            .unwrap_err();
            assert!(
                error.message().contains("without named value references"),
                "{components}: {error:?}"
            );
        }
        assert!(construct("[math.pi,1+2]", None, false, true).is_ok());
    }

    #[test]
    fn rejects_wrong_rank_ragged_shape_and_live_components_without_dummy_values() {
        for components in [
            "[1]",
            "[[1,2],[3]]",
            "[true,false]",
            "[live,0]",
            "[math.complex(real = 1, imaginary = 2),0]",
            "[quotient(left = 4, right = 2),0]",
        ] {
            assert!(
                construct(components, None, false, true).is_err(),
                "{components}"
            );
        }
        let rank_three = construct("[[[1,2],[3,4]],[[5,6],[7,8]]]", None, false, true)
            .unwrap()
            .value
            .unwrap();
        assert_eq!(rank_three.value_type().shape().rank(), 3);
        assert_eq!(rank_three.component(6), Some((7.0, 0.0)));
        let mut rank_five = "1".to_owned();
        for _ in 0..5 {
            rank_five = format!("[{rank_five},{rank_five}]");
        }
        assert!(
            construct(&rank_five, None, false, true)
                .unwrap_err()
                .message()
                .contains("one through four")
        );
        let skipped = construct("[1/0,2]", None, false, false).unwrap();
        assert!(skipped.value.is_none());
        assert!(skipped.expression.is_none());
        assert!(construct("[1/0,2]", None, false, true).is_err());
    }
}
