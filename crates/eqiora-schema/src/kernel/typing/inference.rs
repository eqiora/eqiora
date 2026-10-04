//! Infer expression arena nodes from already inferred operands.

use super::*;

pub(super) enum NodeInference<I, E> {
    Typed(ExpressionType<I>),
    Unavailable,
    Symbol { symbol: SymbolRef, error: E },
    Type(TypeViolation<I>),
}

pub(super) fn infer_node<I: Clone + Eq, E>(
    expression: &ExprDag,
    node: &ExprNode,
    inferred: &[Option<ExpressionType<I>>],
    relation_support: Option<&SpatialSupport<I>>,
    symbol_type: &mut impl FnMut(SymbolRef) -> Result<ExpressionType<I>, E>,
) -> NodeInference<I, E> {
    let typed = match node {
        ExprNode::Evaluate { value, at, side } => {
            return point_evaluation(expression, *value, at, side.is_some(), inferred);
        }
        ExprNode::Require { condition, value } => {
            let Some((condition, value)) = inferred_binary(inferred, *condition, *value) else {
                return NodeInference::Unavailable;
            };
            condition.require(value)
        }
        ExprNode::Select {
            condition,
            then_value,
            else_value,
        } => {
            let Some(condition) = inferred_type(inferred, *condition) else {
                return NodeInference::Unavailable;
            };
            let Some((then_value, else_value)) =
                inferred_binary(inferred, *then_value, *else_value)
            else {
                return NodeInference::Unavailable;
            };
            condition.select(then_value, else_value)
        }
        ExprNode::Compare(op, left, right) => {
            let Some((left, right)) = inferred_binary(inferred, *left, *right) else {
                return NodeInference::Unavailable;
            };
            left.compare(*op, right)
        }
        ExprNode::Not(value) => {
            let Some(value) = inferred_type(inferred, *value) else {
                return NodeInference::Unavailable;
            };
            value.logical_not()
        }
        ExprNode::And(left, right) | ExprNode::Or(left, right) => {
            let Some((left, right)) = inferred_binary(inferred, *left, *right) else {
                return NodeInference::Unavailable;
            };
            left.and(right)
        }
        ExprNode::Constant(value) => Ok(ExpressionType::new(value.value_type().clone(), None)),
        ExprNode::Symbol(symbol) => {
            return match symbol_type(*symbol) {
                Ok(value) => NodeInference::Typed(value),
                Err(error) => NodeInference::Symbol {
                    symbol: *symbol,
                    error,
                },
            };
        }
        ExprNode::Array { elements } => {
            let Some(elements) = elements
                .iter()
                .map(|id| inferred_type(inferred, *id))
                .collect::<Option<Vec<_>>>()
            else {
                return NodeInference::Unavailable;
            };
            ExpressionType::array(&elements)
        }
        ExprNode::Index {
            value,
            index: position,
        } => {
            let Some(value) = inferred_type(inferred, *value) else {
                return NodeInference::Unavailable;
            };
            value.index(*position)
        }
        ExprNode::Complex { real, imag } => {
            let Some((real, imag)) = inferred_binary(inferred, *real, *imag) else {
                return NodeInference::Unavailable;
            };
            real.complex(imag)
        }
        ExprNode::Neg(value) => {
            let Some(value) = inferred_type(inferred, *value) else {
                return NodeInference::Unavailable;
            };
            if matches!(
                value.value_type.scalar_domain(),
                eqiora_core::ScalarDomain::Boolean | eqiora_core::ScalarDomain::Enum
            ) || value.value_type.is_count()
                || value.value_type.index_set().is_some()
            {
                Err(TypeViolation::ScalarDomainMismatch)
            } else {
                Ok(value)
            }
        }
        ExprNode::Sample { value, .. } | ExprNode::Hold(value) => {
            return inferred_type(inferred, *value)
                .map_or(NodeInference::Unavailable, NodeInference::Typed);
        }
        ExprNode::Add(left, right) | ExprNode::Sub(left, right) => {
            let Some((left, right)) = inferred_binary(inferred, *left, *right) else {
                return NodeInference::Unavailable;
            };
            if matches!(
                left.value_type.scalar_domain(),
                eqiora_core::ScalarDomain::Boolean | eqiora_core::ScalarDomain::Enum
            ) || matches!(
                right.value_type.scalar_domain(),
                eqiora_core::ScalarDomain::Boolean | eqiora_core::ScalarDomain::Enum
            ) {
                return NodeInference::Type(TypeViolation::ScalarDomainMismatch);
            }
            if matches!(node, ExprNode::Add(_, _)) {
                left.sum(right)
            } else {
                additive(&left, &right)
            }
        }
        ExprNode::Mul(left, right) => {
            let Some((left, right)) = inferred_binary(inferred, *left, *right) else {
                return NodeInference::Unavailable;
            };
            multiply(&left, &right)
        }
        ExprNode::Div(left, right) => {
            let Some((left, right)) = inferred_binary(inferred, *left, *right) else {
                return NodeInference::Unavailable;
            };
            divide(&left, &right)
        }
        ExprNode::Quotient(left, right) | ExprNode::Remainder(left, right) => {
            let Some((left, right)) = inferred_binary(inferred, *left, *right) else {
                return NodeInference::Unavailable;
            };
            left.integer_quotient(right)
        }
        ExprNode::Ordinal(value) => {
            let Some(value) = inferred_type(inferred, *value) else {
                return NodeInference::Unavailable;
            };
            value.ordinal()
        }
        ExprNode::ToReal(value) | ExprNode::ToInteger(value) => {
            let Some(value) = inferred_type(inferred, *value) else {
                return NodeInference::Unavailable;
            };
            if matches!(node, ExprNode::ToReal(_)) {
                value.to_real()
            } else {
                value.to_integer()
            }
        }
        ExprNode::PowI(base, exponent) => {
            let Some(base) = inferred_type(inferred, *base) else {
                return NodeInference::Unavailable;
            };
            power(&base, *exponent)
        }
        ExprNode::UnaryMath(function, value) => {
            let Some(value) = inferred_type(inferred, *value) else {
                return NodeInference::Unavailable;
            };
            unary_math(*function, &value)
        }
        ExprNode::FiniteUnary(operation, value) => {
            let Some(value) = inferred_type(inferred, *value) else {
                return NodeInference::Unavailable;
            };
            value.finite_unary(*operation)
        }
        ExprNode::FiniteBinary(operation, left, right) => {
            let Some((left, right)) = inferred_binary(inferred, *left, *right) else {
                return NodeInference::Unavailable;
            };
            left.finite_binary(*operation, right)
        }
        ExprNode::CoordinatePartial { value, wrt } => {
            if !matches!(
                expression.node(*wrt),
                Some(ExprNode::Symbol(SymbolRef::Coordinate { .. }))
            ) {
                return NodeInference::Type(TypeViolation::CoordinatePartialRequiresCoordinate);
            }
            let Some((value, selected)) = inferred_binary(inferred, *value, *wrt) else {
                return NodeInference::Unavailable;
            };
            if !matches!(
                selected.support,
                Some(SpatialSupport::Volume { .. } | SpatialSupport::Coordinates { .. })
            ) {
                return NodeInference::Type(TypeViolation::CoordinatePartialRequiresCoordinate);
            }
            if [&value, &selected].iter().any(|ty| {
                !ty.shape().is_scalar()
                    || ty.value_type.array_rank() != 0
                    || ty.value_type.scalar_domain() != eqiora_core::ScalarDomain::Real
            }) {
                return NodeInference::Type(TypeViolation::RootRequiresRealScalar);
            }
            match combine_additive_support(&value.support, &selected.support) {
                Err(error) => Err(error),
                Ok(support) => value
                    .dimension()
                    .div(selected.dimension())
                    .map(|dimension| ExpressionType::scalar(dimension, support))
                    .ok_or(TypeViolation::DimensionOverflow {
                        operation: "coordinate partial",
                    }),
            }
        }
        ExprNode::Gradient(value) => {
            let Some(value) = inferred_type(inferred, *value) else {
                return NodeInference::Unavailable;
            };
            gradient(&value)
        }
        ExprNode::Divergence(value) => {
            let Some(value) = inferred_type(inferred, *value) else {
                return NodeInference::Unavailable;
            };
            divergence(&value)
        }
        ExprNode::SymmetricPart(value) => {
            let Some(value) = inferred_type(inferred, *value) else {
                return NodeInference::Unavailable;
            };
            symmetric_part(&value)
        }
        ExprNode::IsotropicLift(value) => {
            let Some(value) = inferred_type(inferred, *value) else {
                return NodeInference::Unavailable;
            };
            isotropic_lift(&value)
        }
        ExprNode::Trace(value) => {
            let Some(value) = inferred_type(inferred, *value) else {
                return NodeInference::Unavailable;
            };
            trace(&value, relation_support)
        }
        ExprNode::NormalComponent(value) => {
            let Some(value) = inferred_type(inferred, *value) else {
                return NodeInference::Unavailable;
            };
            normal(&value, relation_support)
        }
        ExprNode::PureOperatorApplication(application) => {
            let mut arguments = Vec::with_capacity(application.arguments().len());
            for argument in application.arguments() {
                let Some(argument_type) = inferred_type(inferred, *argument) else {
                    return NodeInference::Unavailable;
                };
                arguments.push(argument_type);
            }
            let definition = expression
                .definition(application.definition())
                .expect("ExprDag keeps every pure application definition in its closed table");
            definition
                .instantiate(&arguments)
                .map(|application| application.result_type().clone())
                .map_err(TypeViolation::PureOperatorApplication)
        }
    };
    match typed {
        Ok(value) => NodeInference::Typed(value),
        Err(error) => NodeInference::Type(error),
    }
}

fn point_evaluation<I: Clone + Eq, E>(
    expression: &ExprDag,
    value: ExprId,
    at: &[(ExprId, ExprId)],
    side: bool,
    inferred: &[Option<ExpressionType<I>>],
) -> NodeInference<I, E> {
    let invalid = || NodeInference::Type(TypeViolation::PointEvaluationRequiresExactCoordinates);
    let Some(value) = inferred_type(inferred, value) else {
        return NodeInference::Unavailable;
    };
    if at.is_empty() || at.len() > 64 || (side && at.len() != 1) {
        return invalid();
    }
    let mut support = None;
    let mut axes = std::collections::BTreeSet::new();
    for (coordinate, point) in at {
        let Some(ExprNode::Symbol(SymbolRef::Coordinate { factor, axis, .. })) =
            expression.node(*coordinate)
        else {
            return invalid();
        };
        if !axes.insert((factor.erase(), *axis)) {
            return invalid();
        }
        let Some((coordinate, point)) = inferred_binary(inferred, *coordinate, *point) else {
            return NodeInference::Unavailable;
        };
        let Some(selected @ (SpatialSupport::Coordinates { .. } | SpatialSupport::Volume { .. })) =
            coordinate.support.as_ref()
        else {
            return invalid();
        };
        if selected.intrinsic_dimensions() != at.len()
            || support
                .as_ref()
                .is_some_and(|expected| expected != selected)
            || value
                .support
                .as_ref()
                .is_some_and(|expected| expected != selected)
            || point.support.is_some()
            || point.dimension() != coordinate.dimension()
            || !point.shape().is_scalar()
            || point.value_type.array_rank() != 0
            || point.value_type.scalar_domain() != eqiora_core::ScalarDomain::Real
            || point.value_type.frame() != eqiora_core::ValueFrame::Invariant
        {
            return invalid();
        }
        support = Some(selected.clone());
    }
    NodeInference::Typed(ExpressionType::new(value.value_type, None))
}

fn inferred_binary<I: Clone>(
    inferred: &[Option<ExpressionType<I>>],
    left: ExprId,
    right: ExprId,
) -> Option<(ExpressionType<I>, ExpressionType<I>)> {
    Some((
        inferred_type(inferred, left)?,
        inferred_type(inferred, right)?,
    ))
}

pub(super) fn inferred_type<I: Clone>(
    inferred: &[Option<ExpressionType<I>>],
    id: ExprId,
) -> Option<ExpressionType<I>> {
    usize::try_from(id.index())
        .ok()
        .and_then(|index| inferred.get(index))
        .cloned()
        .flatten()
}
