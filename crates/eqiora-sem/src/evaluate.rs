//! Scalar expression-DAG evaluation for one explicit semantic context.

use std::collections::BTreeMap;

use eqiora_core::diagnostic::codes;
use eqiora_core::{
    Diagnostic, DynQuantity, GraphPath, RawId, ScalarDomain, ValueLiteral, ValueType,
};
use eqiora_schema::kernel::{ExprDag, ExprId, ExprNode, SymbolRef};

use crate::{ExpressionBackend, KernelProgram, PhysicalUnknown};

/// Canonical typed DAG evaluator used to independently inspect numerical candidates.
/// Symbol resolution and root selection use the existing `ExpressionBackend` contract.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ReferenceExpressionBackend;

impl ExpressionBackend for ReferenceExpressionBackend {
    fn evaluate(
        &self,
        owner: RawId,
        expression: &ExprDag,
        roots: &[ExprId],
        resolve: &mut dyn FnMut(SymbolRef) -> Option<ValueLiteral>,
    ) -> Result<Vec<ValueLiteral>, Diagnostic> {
        evaluate_selected(owner, expression, roots, resolve)
    }
}

pub(crate) struct EvalContext<'a> {
    pub(crate) program: &'a KernelProgram,
    pub(crate) typed_fields: &'a BTreeMap<RawId, ValueLiteral>,
    pub(crate) typed_ports: &'a BTreeMap<RawId, ValueLiteral>,
    pub(crate) typed_next: &'a BTreeMap<RawId, ValueLiteral>,
    pub(crate) time: f64,
    pub(crate) fields: &'a BTreeMap<RawId, f64>,
    pub(crate) field_candidates: &'a BTreeMap<RawId, f64>,
    pub(crate) derivatives: &'a BTreeMap<RawId, f64>,
    pub(crate) next_fields: &'a BTreeMap<RawId, f64>,
    pub(crate) ports: &'a BTreeMap<RawId, f64>,
    pub(crate) port_candidates: &'a BTreeMap<RawId, f64>,
    pub(crate) signal_sources: &'a BTreeMap<RawId, RawId>,
    pub(crate) physical: &'a BTreeMap<PhysicalUnknown, f64>,
    pub(crate) physical_candidates: &'a BTreeMap<PhysicalUnknown, f64>,
}

pub(crate) fn evaluate_expression(
    owner: RawId,
    expression: &ExprDag,
    resolve: &mut dyn FnMut(SymbolRef) -> Option<ValueLiteral>,
) -> Result<Vec<ValueLiteral>, Diagnostic> {
    evaluate_selected(owner, expression, expression.roots(), resolve)
}

fn evaluate_selected(
    owner: RawId,
    expression: &ExprDag,
    roots: &[ExprId],
    resolve: &mut dyn FnMut(SymbolRef) -> Option<ValueLiteral>,
) -> Result<Vec<ValueLiteral>, Diagnostic> {
    enum Frame {
        Demand(ExprId),
        Apply(ExprId),
        Logical(ExprId),
        Branch(ExprId),
    }
    let mut component_work = 0usize;
    let mut values = vec![None; expression.nodes().len()];
    for &root in roots {
        let mut pending = vec![Frame::Demand(root)];
        while let Some(frame) = pending.pop() {
            let id = match frame {
                Frame::Demand(id) | Frame::Apply(id) | Frame::Logical(id) | Frame::Branch(id) => id,
            };
            let index = id.index() as usize;
            let Some(node) = expression.nodes().get(index) else {
                return Err(Diagnostic::error(
                    codes::INVALID_EXPRESSION_DAG,
                    "requested expression root is unavailable",
                ));
            };
            if values[index].is_some() {
                continue;
            }
            if let ExprNode::Select { condition, .. } | ExprNode::Require { condition, .. } = node {
                if matches!(frame, Frame::Demand(_)) {
                    pending.push(Frame::Branch(id));
                    pending.push(Frame::Demand(*condition));
                    continue;
                }
                if matches!(frame, Frame::Branch(_)) {
                    let condition = boolean(operand(&values, *condition, owner)?)?;
                    let selected = match node {
                        ExprNode::Select {
                            then_value,
                            else_value,
                            ..
                        } => {
                            if condition {
                                *then_value
                            } else {
                                *else_value
                            }
                        }
                        ExprNode::Require { value, .. } if condition => *value,
                        _ => {
                            return Err(Diagnostic::error(
                                codes::NONFINITE_EVALUATION,
                                "required expression domain condition is false",
                            ));
                        }
                    };
                    pending.push(Frame::Apply(id));
                    pending.push(Frame::Demand(selected));
                    continue;
                }
            }
            if let ExprNode::And(left, right) | ExprNode::Or(left, right) = node {
                if matches!(frame, Frame::Demand(_)) {
                    pending.push(Frame::Logical(id));
                    pending.push(Frame::Demand(*left));
                    continue;
                }
                if matches!(frame, Frame::Logical(_)) {
                    let left = boolean(operand(&values, *left, owner)?)?;
                    if matches!(node, ExprNode::And(_, _)) && !left
                        || matches!(node, ExprNode::Or(_, _)) && left
                    {
                        values[index] = Some(ValueLiteral::boolean(left));
                    } else {
                        pending.push(Frame::Apply(id));
                        pending.push(Frame::Demand(*right));
                    }
                    continue;
                }
            }
            if matches!(frame, Frame::Demand(_)) {
                pending.push(Frame::Apply(id));
                match node {
                    ExprNode::Constant(_) | ExprNode::Symbol(_) => {}
                    ExprNode::PureOperatorApplication(application) => {
                        check_component_work(component_work, application.arguments().len())?;
                        pending.extend(
                            application
                                .arguments()
                                .iter()
                                .rev()
                                .copied()
                                .map(Frame::Demand),
                        );
                    }
                    ExprNode::Array { elements } => {
                        check_component_work(component_work, elements.len())?;
                        pending.extend(elements.iter().rev().copied().map(Frame::Demand));
                    }
                    ExprNode::UnaryMath(
                        eqiora_schema::kernel::UnaryMathFunction::Sqrt
                        | eqiora_schema::kernel::UnaryMathFunction::Sin,
                        value,
                    )
                    | ExprNode::Index { value, .. }
                    | ExprNode::Sample { value, .. }
                    | ExprNode::Hold(value)
                    | ExprNode::Neg(value)
                    | ExprNode::PowI(value, _)
                    | ExprNode::ToReal(value)
                    | ExprNode::ToInteger(value)
                    | ExprNode::Ordinal(value)
                    | ExprNode::Not(value) => pending.push(Frame::Demand(*value)),
                    ExprNode::Compare(_, a, b)
                    | ExprNode::Add(a, b)
                    | ExprNode::Sub(a, b)
                    | ExprNode::Mul(a, b)
                    | ExprNode::Div(a, b)
                    | ExprNode::Quotient(a, b)
                    | ExprNode::Remainder(a, b) => {
                        pending.push(Frame::Demand(*b));
                        pending.push(Frame::Demand(*a));
                    }
                    _ => {
                        return Err(Diagnostic::error(
                            codes::NOT_IMPLEMENTED,
                            "expression node is outside the reference execution profile",
                        ));
                    }
                }
                continue;
            }
            let value = match node {
                ExprNode::Select {
                    condition,
                    then_value,
                    else_value,
                } => {
                    let selected = if boolean(operand(&values, *condition, owner)?)? {
                        then_value
                    } else {
                        else_value
                    };
                    operand(&values, *selected, owner)?.clone()
                }
                ExprNode::Require { value, .. } => operand(&values, *value, owner)?.clone(),
                ExprNode::UnaryMath(eqiora_schema::kernel::UnaryMathFunction::Sin, value) => {
                    let value = real(operand(&values, *value, owner)?)?;
                    if value.dim() != eqiora_core::DimExponents::DIMENSIONLESS {
                        return Err(Diagnostic::error(
                            codes::NONFINITE_EVALUATION,
                            "sine requires a dimensionless real scalar",
                        ));
                    }
                    literal(DynQuantity::new(
                        value.value().sin(),
                        eqiora_core::DimExponents::DIMENSIONLESS,
                    ))?
                }
                ExprNode::UnaryMath(eqiora_schema::kernel::UnaryMathFunction::Sqrt, value) => {
                    let value = real(operand(&values, *value, owner)?)?;
                    if value.value() < 0. {
                        return Err(Diagnostic::error(
                            codes::NONFINITE_EVALUATION,
                            "real square root requires a nonnegative argument",
                        ));
                    }
                    let dimension = value.dim().pow(1, 2).ok_or_else(|| {
                        Diagnostic::error(
                            codes::NONFINITE_EVALUATION,
                            "square-root dimension exceeds bounds",
                        )
                    })?;
                    literal(DynQuantity::new(value.value().sqrt(), dimension))?
                }

                ExprNode::PureOperatorApplication(application) => {
                    let definition = expression
                        .definition(application.definition())
                        .expect("checked definition");
                    let arguments = application
                        .arguments()
                        .iter()
                        .map(|id| operand(&values, *id, owner))
                        .collect::<Result<Vec<_>, _>>()?;
                    let cost = definition.nodes().len() + arguments.len();
                    check_component_work(component_work, cost)?;
                    component_work += cost;
                    evaluate_pure_operator(owner, definition, &arguments)?
                }
                ExprNode::Array { elements } => {
                    let elements = elements
                        .iter()
                        .map(|id| operand(&values, *id, owner))
                        .collect::<Result<Vec<_>, _>>()?;
                    for element in &elements {
                        require_channels(element.value_type())?;
                    }
                    let types = elements
                        .iter()
                        .map(|value| {
                            eqiora_schema::kernel::typing::ExpressionType::<()>::new(
                                value.value_type().clone(),
                                None,
                            )
                        })
                        .collect::<Vec<_>>();
                    let ty = eqiora_schema::kernel::typing::ExpressionType::array(&types).map_err(
                        |error| Diagnostic::error(codes::INVALID_EXPRESSION_DAG, error.to_string()),
                    )?;
                    let count = ty
                        .value_type
                        .shape()
                        .component_count()
                        .expect("checked type");
                    check_component_work(component_work, count)?;
                    ValueLiteral::array(&elements).map_err(discrete_error)?
                }
                ExprNode::Index { value, index } => {
                    let value = operand(&values, *value, owner)?;
                    require_channels(value.value_type())?;
                    let ty = eqiora_schema::kernel::typing::ExpressionType::<()>::new(
                        value.value_type().clone(),
                        None,
                    )
                    .index(*index)
                    .map_err(|error| {
                        Diagnostic::error(codes::INVALID_EXPRESSION_DAG, error.to_string())
                    })?;
                    check_component_work(
                        component_work,
                        ty.value_type
                            .shape()
                            .component_count()
                            .expect("checked type"),
                    )?;
                    value.index(*index).map_err(discrete_error)?
                }
                ExprNode::Not(value) => {
                    ValueLiteral::boolean(!boolean(operand(&values, *value, owner)?)?)
                }
                ExprNode::And(_, right) | ExprNode::Or(_, right) => {
                    ValueLiteral::boolean(boolean(operand(&values, *right, owner)?)?)
                }
                ExprNode::Compare(op, left, right) => compare(
                    *op,
                    operand(&values, *left, owner)?,
                    operand(&values, *right, owner)?,
                )?,
                ExprNode::Constant(value) => {
                    check_component_work(component_work, value.component_count())?;
                    value.clone()
                }
                ExprNode::Symbol(symbol) => resolve(*symbol).ok_or_else(|| {
                    Diagnostic::error(
                        codes::MISSING_EXECUTION_INPUT,
                        format!("no reference-execution value is available for {symbol:?}"),
                    )
                    .with_graph_path(expression_path(owner, index))
                })?,
                ExprNode::Sample { value, .. } | ExprNode::Hold(value) => {
                    operand(&values, *value, owner)?.clone()
                }
                ExprNode::Neg(value) => {
                    let value = operand(&values, *value, owner)?;
                    require_scalar_arithmetic(value)?;
                    if value.value_type().scalar_domain() == ScalarDomain::Integer {
                        value.checked_neg().map_err(discrete_error)?
                    } else {
                        literal(-real(value)?)?
                    }
                }
                ExprNode::Add(left, right) => {
                    let left = operand(&values, *left, owner)?;
                    let right = operand(&values, *right, owner)?;
                    require_scalar_arithmetic(left)?;
                    require_scalar_arithmetic(right)?;
                    if left.value_type().scalar_domain() == ScalarDomain::Integer {
                        left.checked_add(right).map_err(discrete_error)?
                    } else {
                        literal(real(left)?.try_add(real(right)?)?)?
                    }
                }
                ExprNode::Sub(left, right) => {
                    let left = operand(&values, *left, owner)?;
                    let right = operand(&values, *right, owner)?;
                    require_scalar_arithmetic(left)?;
                    require_scalar_arithmetic(right)?;
                    if left.value_type().scalar_domain() == ScalarDomain::Integer {
                        left.checked_sub(right).map_err(discrete_error)?
                    } else {
                        literal(real(left)?.try_sub(real(right)?)?)?
                    }
                }
                ExprNode::Mul(left, right) => {
                    let left = operand(&values, *left, owner)?;
                    let right = operand(&values, *right, owner)?;
                    require_scalar_arithmetic(left)?;
                    require_scalar_arithmetic(right)?;
                    if left.value_type().scalar_domain() == ScalarDomain::Integer {
                        left.checked_mul(right).map_err(discrete_error)?
                    } else {
                        literal(real(left)?.try_mul(real(right)?)?)?
                    }
                }
                ExprNode::Div(left, right) => literal(
                    real(operand(&values, *left, owner)?)?
                        .try_div(real(operand(&values, *right, owner)?)?)?,
                )?,
                ExprNode::Quotient(a, b) => operand(&values, *a, owner)?
                    .checked_quotient(operand(&values, *b, owner)?)
                    .map_err(discrete_error)?,
                ExprNode::Remainder(a, b) => operand(&values, *a, owner)?
                    .checked_remainder(operand(&values, *b, owner)?)
                    .map_err(discrete_error)?,
                ExprNode::ToReal(value) => operand(&values, *value, owner)?
                    .to_real()
                    .map_err(discrete_error)?,
                ExprNode::Ordinal(value) => operand(&values, *value, owner)?
                    .ordinal()
                    .map_err(discrete_error)?,
                ExprNode::ToInteger(value) => operand(&values, *value, owner)?
                    .to_integer()
                    .map_err(discrete_error)?,
                ExprNode::PowI(base, exponent) => {
                    let base = real(operand(&values, *base, owner)?)?;
                    let dimension = base.dim().pow(*exponent, 1).ok_or_else(|| {
                        Diagnostic::error(
                            codes::DIMENSION_MISMATCH,
                            "power dimension exceeds bounds",
                        )
                    })?;
                    literal(DynQuantity::new(base.value().powi(*exponent), dimension))?
                }
                _ => {
                    return Err(Diagnostic::error(
                        codes::NOT_IMPLEMENTED,
                        "expression node is newer than this reference interpreter",
                    )
                    .with_graph_path(expression_path(owner, index)));
                }
            };
            check_component_work(component_work, value.component_count())?;
            component_work += value.component_count();
            values[index] = Some(value);
        }
    }

    roots
        .iter()
        .map(|root| operand(&values, *root, owner).cloned())
        .collect()
}

pub(crate) fn resolve_symbol(symbol: SymbolRef, context: &EvalContext<'_>) -> Option<ValueLiteral> {
    if let SymbolRef::Parameter(id) = symbol {
        return context.program.typed_value(id.erase()).cloned();
    }
    let discrete = match symbol {
        SymbolRef::Field(id) | SymbolRef::Pre(id) => context.typed_fields.get(&id.erase()),
        SymbolRef::Next(id) => context.typed_next.get(&id.erase()),
        SymbolRef::Port(id) => context.typed_ports.get(
            context
                .signal_sources
                .get(&id.erase())
                .unwrap_or(&id.erase()),
        ),
        _ => None,
    };
    if let Some(value) = discrete {
        return Some(value.clone());
    }
    let value = match symbol {
        SymbolRef::Field(id) => context
            .field_candidates
            .get(&id.erase())
            .or_else(|| context.fields.get(&id.erase()))
            .copied(),
        SymbolRef::Derivative(id) => context.derivatives.get(&id.erase()).copied(),
        SymbolRef::Pre(id) => context.fields.get(&id.erase()).copied(),
        SymbolRef::Next(id) => context.next_fields.get(&id.erase()).copied(),
        SymbolRef::Parameter(id) => context.program.value(id.erase()).map(|value| value.value()),
        SymbolRef::Port(id) => {
            let port = context
                .signal_sources
                .get(&id.erase())
                .copied()
                .unwrap_or_else(|| id.erase());
            context
                .port_candidates
                .get(&port)
                .or_else(|| context.ports.get(&port))
                .copied()
        }
        SymbolRef::Across(id) => resolve_physical(PhysicalUnknown::Across(id), context),
        SymbolRef::Through(id) => resolve_physical(PhysicalUnknown::Through(id), context),
        SymbolRef::Time => Some(context.time),
        _ => None,
    }?;
    ValueLiteral::from_real(context.program.execution_symbol_type(symbol)?, value).ok()
}

fn resolve_physical(unknown: PhysicalUnknown, context: &EvalContext<'_>) -> Option<f64> {
    context
        .physical_candidates
        .get(&unknown)
        .or_else(|| context.physical.get(&unknown))
        .copied()
}

fn operand(
    values: &[Option<ValueLiteral>],
    id: ExprId,
    owner: RawId,
) -> Result<&ValueLiteral, Diagnostic> {
    usize::try_from(id.index())
        .ok()
        .and_then(|index| values.get(index))
        .and_then(Option::as_ref)
        .ok_or_else(|| {
            Diagnostic::error(
                codes::INVALID_EXPRESSION_DAG,
                format!("expression operand {} is unavailable", id.index()),
            )
            .with_graph_path(expression_path(
                owner,
                usize::try_from(id.index()).unwrap_or(usize::MAX),
            ))
        })
}

fn discrete_error(error: eqiora_core::InvalidValueLiteral) -> Diagnostic {
    Diagnostic::error(
        codes::NOT_IMPLEMENTED,
        format!("exact discrete operation rejected: {error:?}"),
    )
}

pub(crate) fn real(value: &ValueLiteral) -> Result<DynQuantity, Diagnostic> {
    value.real_scalar_value().ok_or_else(|| {
        Diagnostic::error(
            codes::NOT_IMPLEMENTED,
            "numerical reference execution requires a real scalar value",
        )
    })
}

pub(crate) fn literal(value: DynQuantity) -> Result<ValueLiteral, Diagnostic> {
    ValueLiteral::from_real(
        ValueType::scalar(ScalarDomain::Real, value.dim())
            .map_err(|error| Diagnostic::error(codes::DIMENSION_MISMATCH, error.to_string()))?,
        value.value(),
    )
    .map_err(|_| {
        Diagnostic::error(
            codes::NONFINITE_EVALUATION,
            "reference expression produced a nonfinite value",
        )
    })
}

pub(crate) fn numerical_differences(values: Vec<ValueLiteral>) -> Result<Vec<f64>, Diagnostic> {
    let (pairs, remainder) = values.as_chunks::<2>();
    if !remainder.is_empty() {
        return Err(Diagnostic::error(
            codes::INVALID_EXPRESSION_DAG,
            "numerical equation evaluation returned an incomplete side pair",
        ));
    }
    pairs
        .iter()
        .map(|sides| {
            literal(real(&sides[0])?.try_sub(real(&sides[1])?)?)
                .and_then(|value| real(&value).map(|value| value.value()))
        })
        .collect()
}

pub(crate) fn real_values(values: Vec<ValueLiteral>) -> Result<Vec<f64>, Diagnostic> {
    values
        .iter()
        .map(|value| real(value).map(|value| value.value()))
        .collect()
}

fn expression_path(owner: RawId, index: usize) -> GraphPath {
    GraphPath::new([
        "semantic".to_owned(),
        format!("{:?}", owner.kind()),
        owner.to_string(),
        "expression".to_owned(),
        index.to_string(),
    ])
}

fn boolean(value: &ValueLiteral) -> Result<bool, Diagnostic> {
    value.as_bool().ok_or_else(|| {
        Diagnostic::error(
            codes::NOT_IMPLEMENTED,
            "logical execution requires a scalar Boolean",
        )
    })
}

fn compare(
    op: eqiora_schema::kernel::ComparisonOp,
    left: &ValueLiteral,
    right: &ValueLiteral,
) -> Result<ValueLiteral, Diagnostic> {
    use eqiora_schema::kernel::ComparisonOp;
    use std::cmp::Ordering;
    let value = match op {
        ComparisonOp::Equal => left.checked_equal(right),
        ComparisonOp::NotEqual => left.checked_equal(right).map(|value| !value),
        ComparisonOp::Less => left
            .checked_order(right)
            .map(|value| value == Ordering::Less),
        ComparisonOp::LessEqual => left
            .checked_order(right)
            .map(|value| value != Ordering::Greater),
        ComparisonOp::Greater => left
            .checked_order(right)
            .map(|value| value == Ordering::Greater),
        ComparisonOp::GreaterEqual => left
            .checked_order(right)
            .map(|value| value != Ordering::Less),
    }
    .map_err(discrete_error)?;
    Ok(ValueLiteral::boolean(value))
}

fn require_channels(value_type: &ValueType) -> Result<(), Diagnostic> {
    if value_type.frame() != eqiora_core::ValueFrame::Invariant
        || !matches!(
            value_type.scalar_domain(),
            ScalarDomain::Real | ScalarDomain::Integer
        )
        || value_type.array_rank() != value_type.shape().rank()
        || value_type.finite_space().is_some()
        || value_type.index_set().is_some()
    {
        return Err(Diagnostic::error(
            codes::NOT_IMPLEMENTED,
            "channel execution requires invariant real or integer values",
        ));
    }
    Ok(())
}
fn component_budget_error() -> Diagnostic {
    Diagnostic::error(
        codes::NOT_IMPLEMENTED,
        "expression evaluation exceeds the one-million component work budget",
    )
}
fn check_component_work(used: usize, next: usize) -> Result<(), Diagnostic> {
    if used.checked_add(next).is_none_or(|total| total > 1_000_000) {
        Err(component_budget_error())
    } else {
        Ok(())
    }
}

fn require_scalar_arithmetic(value: &ValueLiteral) -> Result<(), Diagnostic> {
    if value.value_type().array_rank() > 0 {
        return Err(Diagnostic::error(
            codes::NOT_IMPLEMENTED,
            "channel arrays require explicit indexing before arithmetic",
        ));
    }
    Ok(())
}

fn evaluate_pure_operator(
    owner: RawId,
    definition: &eqiora_schema::kernel::pure_operator::PureOperatorDefinition,
    arguments: &[&ValueLiteral],
) -> Result<ValueLiteral, Diagnostic> {
    use eqiora_schema::kernel::{ExprDagBuilder, typing::ExpressionType};
    let types = arguments
        .iter()
        .map(|value| ExpressionType::<()>::new(value.value_type().clone(), None))
        .collect::<Vec<_>>();
    if types
        .iter()
        .any(|ty| ty.value_type.scalar_domain() != ScalarDomain::Real || !ty.shape().is_scalar())
    {
        return Err(Diagnostic::error(
            codes::NOT_IMPLEMENTED,
            "pure operator execution requires real scalar arguments",
        ));
    }
    let instance = definition
        .instantiate(&types)
        .map_err(|error| Diagnostic::error(codes::NOT_IMPLEMENTED, error.to_string()))?;
    let mut builder = ExprDagBuilder::new();
    let arguments = arguments
        .iter()
        .map(|value| builder.constant((*value).clone()))
        .collect::<Result<Vec<_>, _>>()?;
    let root = builder.project_scalar_operator(&instance, &arguments, 1_000_000)?;
    let dag = builder.finish([root])?;
    evaluate_selected(owner, &dag, &[root], &mut |_| None)?
        .pop()
        .ok_or_else(|| {
            Diagnostic::error(
                codes::INVALID_EXPRESSION_DAG,
                "pure operator result is absent",
            )
        })
}

#[cfg(test)]
mod piecewise_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::{DimExponents, Id, ScalarDomain, ValueLiteral, ValueType, entity::kinds};
    use eqiora_schema::kernel::ExprDagBuilder;

    #[test]
    fn channels_construct_index_and_preserve_integer_low_bits() {
        let mut builder = ExprDagBuilder::new();
        let ty = ValueType::scalar(ScalarDomain::Integer, DimExponents::DIMENSIONLESS)
            .expect("valid scalar type");
        let high = builder
            .constant(ValueLiteral::from_integer(ty.clone(), 9_007_199_254_740_993).unwrap())
            .unwrap();
        let one = builder
            .constant(ValueLiteral::from_integer(ty, 1).unwrap())
            .unwrap();
        let channels = builder.array([high, one]).unwrap();
        let selected = builder.index(channels, 0).unwrap();
        let incremented = builder.add(selected, one).unwrap();
        let result = builder.array([incremented, selected]).unwrap();
        let dag = builder.finish([result]).unwrap();
        let values = evaluate_selected(
            Id::<kinds::Relation>::new().erase(),
            &dag,
            &[result],
            &mut |_| None,
        )
        .unwrap();
        assert_eq!(
            values[0].integer_components().unwrap().collect::<Vec<_>>(),
            vec![9_007_199_254_740_994, 9_007_199_254_740_993]
        );
    }
    #[test]
    fn channel_intermediate_budget_and_direct_array_arithmetic_reject() {
        let mut builder = ExprDagBuilder::new();
        let ty = ValueType::scalar(ScalarDomain::Integer, DimExponents::DIMENSIONLESS)
            .expect("valid scalar type")
            .array(600_000)
            .unwrap();
        let input = builder
            .constant(ValueLiteral::from_integer(ty, 0).unwrap())
            .unwrap();
        let result = builder.array([input, input]).unwrap();
        let dag = builder.finish([result]).unwrap();
        assert!(
            evaluate_selected(
                Id::<kinds::Relation>::new().erase(),
                &dag,
                &[result],
                &mut |_| None
            )
            .unwrap_err()
            .message()
            .contains("component")
        );
        let mut builder = ExprDagBuilder::new();
        let ty = ValueType::scalar(ScalarDomain::Integer, DimExponents::DIMENSIONLESS)
            .expect("valid scalar type")
            .array(2)
            .unwrap();
        let input = builder
            .constant(ValueLiteral::from_integer(ty, 0).unwrap())
            .unwrap();
        let result = builder.add(input, input).unwrap();
        let dag = builder.finish([result]).unwrap();
        assert!(
            evaluate_selected(
                Id::<kinds::Relation>::new().erase(),
                &dag,
                &[result],
                &mut |_| None
            )
            .unwrap_err()
            .message()
            .contains("indexing")
        );
    }
    #[test]
    fn logical_demand_skips_errors_but_does_not_poison_shared_roots() {
        use eqiora_schema::kernel::ComparisonOp;
        let owner = Id::<kinds::Relation>::new().erase();
        let gate = Id::<kinds::Parameter>::new();
        let integer = ValueType::scalar(ScalarDomain::Integer, DimExponents::DIMENSIONLESS)
            .expect("valid scalar type");
        let mut dag = ExprDagBuilder::new();
        let condition = dag.symbol(SymbolRef::Parameter(gate)).unwrap();
        let one = dag
            .constant(ValueLiteral::from_integer(integer.clone(), 1).unwrap())
            .unwrap();
        let zero = dag
            .constant(ValueLiteral::from_integer(integer, 0).unwrap())
            .unwrap();
        let bad = dag.quotient(one, zero).unwrap();
        let bad_predicate = dag.compare(ComparisonOp::Equal, bad, zero).unwrap();
        let conjunction = dag.and(condition, bad_predicate).unwrap();
        let disjunction = dag.or(condition, bad_predicate).unwrap();
        let negated = dag.not(condition).unwrap();
        let dag = dag.finish([conjunction, disjunction, negated]).unwrap();

        for (roots, gate_value, expected) in [
            (
                vec![conjunction, conjunction, negated],
                false,
                Some(vec![false, false, true]),
            ),
            (vec![disjunction], true, Some(vec![true])),
            (vec![conjunction, bad_predicate], false, None),
            (vec![conjunction], true, None),
            (vec![disjunction], false, None),
        ] {
            let mut reads = 0;
            let mut resolve = |symbol| {
                assert_eq!(symbol, SymbolRef::Parameter(gate));
                reads += 1;
                Some(ValueLiteral::boolean(gate_value))
            };
            let roots = roots.as_slice();
            let result = evaluate_selected(owner, &dag, roots, &mut resolve);
            assert_eq!(
                reads, 1,
                "one memoized node is read once within one demand context"
            );
            if let Some(expected) = expected {
                assert_eq!(
                    result
                        .unwrap()
                        .iter()
                        .map(|value| value.as_bool().unwrap())
                        .collect::<Vec<_>>(),
                    expected
                );
            } else {
                assert!(result.is_err());
            }
        }
    }

    #[test]
    fn reference_evaluation_does_not_narrow_typed_constants() {
        let owner = Id::<kinds::Relation>::new().erase();
        let real = ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
            .expect("valid scalar type");
        for value_type in [
            real.clone(),
            ValueType::scalar(ScalarDomain::Complex, real.dimension()).expect("valid scalar type"),
            real.array(3).unwrap(),
        ] {
            let is_real_scalar =
                value_type.scalar_domain() == ScalarDomain::Real && value_type.shape().is_scalar();
            let mut builder = ExprDagBuilder::new();
            let value = ValueLiteral::from_real(value_type, 0.0).unwrap();
            let root = builder.constant(value.clone()).unwrap();
            let result =
                evaluate_expression(owner, &builder.finish([root]).unwrap(), &mut |_| None)
                    .unwrap();
            assert_eq!(result, vec![value]);
            assert_eq!(real_values(result).is_ok(), is_real_scalar);
        }
    }
    #[test]
    fn exact_integer_operations_and_checked_conversion_use_one_backend() {
        let owner = Id::<kinds::Relation>::new().erase();
        let ty = ValueType::scalar(ScalarDomain::Integer, DimExponents::DIMENSIONLESS)
            .expect("valid scalar type");
        let mut dag = ExprDagBuilder::new();
        let a = dag
            .constant(ValueLiteral::from_integer(ty.clone(), -7).unwrap())
            .unwrap();
        let b = dag
            .constant(ValueLiteral::from_integer(ty, 3).unwrap())
            .unwrap();
        let q = dag.quotient(a, b).unwrap();
        let r = dag.remainder(a, b).unwrap();
        let real = dag.to_real(q).unwrap();
        let integer = dag.to_integer(real).unwrap();
        let index_type = ValueType::index(Id::new(), 3).unwrap();
        let index = dag
            .constant(ValueLiteral::from_integer(index_type, 2).unwrap())
            .unwrap();
        let ordinal = dag.ordinal(index).unwrap();
        let dag = dag.finish([q, r, real, integer, ordinal]).unwrap();
        let values = evaluate_expression(owner, &dag, &mut |_| None).unwrap();
        assert_eq!(values[0].integer_scalar_value(), Some(-2));
        assert_eq!(values[1].integer_scalar_value(), Some(-1));
        assert_eq!(values[2].real_scalar_value().unwrap().value(), -2.);
        assert_eq!(values[3].integer_scalar_value(), Some(-2));
        assert_eq!(values[4].integer_scalar_value(), Some(2));
        assert!(real_values(values).is_err());
        let mut dag = ExprDagBuilder::new();
        let a = dag
            .constant(DynQuantity::new(0.5, DimExponents::DIMENSIONLESS))
            .unwrap();
        let converted = dag.to_integer(a).unwrap();
        assert!(
            evaluate_expression(owner, &dag.finish([converted]).unwrap(), &mut |_| None).is_err()
        );
    }
    #[test]
    fn extrema_keep_adjacent_integers_exact_and_evaluate_both_operands() {
        use eqiora_core::{Id, entity::kinds};
        use eqiora_schema::kernel::ExprDagBuilder;
        let integer = |n| {
            ValueLiteral::from_integer(
                ValueType::scalar(
                    ScalarDomain::Integer,
                    eqiora_core::DimExponents::DIMENSIONLESS,
                )
                .expect("valid scalar type"),
                n,
            )
            .unwrap()
        };
        let mut builder = ExprDagBuilder::new();
        let first = builder.constant(integer(9_007_199_254_740_993)).unwrap();
        let second = builder.constant(integer(9_007_199_254_740_992)).unwrap();
        let smallest = builder.min(first, second).unwrap();
        let largest = builder.max(first, second).unwrap();
        let tie = builder.min(first, first).unwrap();
        let roots = [smallest, largest, tie];
        let dag = builder.finish(roots).unwrap();
        assert_eq!(
            evaluate_selected(
                Id::<kinds::Relation>::new().erase(),
                &dag,
                &roots,
                &mut |_| None
            )
            .unwrap(),
            vec![
                integer(9_007_199_254_740_992),
                integer(9_007_199_254_740_993),
                integer(9_007_199_254_740_993)
            ]
        );
        let mut builder = ExprDagBuilder::new();
        let zero = builder.constant(integer(0)).unwrap();
        let max = builder.constant(integer(i64::MAX)).unwrap();
        let one = builder.constant(integer(1)).unwrap();
        let bad = builder.add(max, one).unwrap();
        // Even though every valid positive candidate would lose to zero, overflow is demanded.
        let selected = builder.min(zero, bad).unwrap();
        let roots = [selected];
        let dag = builder.finish(roots).unwrap();
        assert!(
            evaluate_selected(
                Id::<kinds::Relation>::new().erase(),
                &dag,
                &roots,
                &mut |_| None
            )
            .is_err()
        );
    }
}
