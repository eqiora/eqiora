//! Scalar value projection from validated semantic expression operands.

use super::*;
use eqiora_schema::kernel::{ExprDag, ExprId, ExprNode};

impl ScalarOperatorIr {
    /// Lower a canonical expression DAG into dense symbol slots and scalar
    /// instructions without changing operation order. Sample/Hold reuse their
    /// operand value; activation and memory admission remain with the validated
    /// KernelProgram retained by the executor, not this scalar value projection.
    ///
    /// # Errors
    /// Returns `EQ0701` if an operand/root index is inconsistent with the DAG
    /// contract.
    pub fn lower(expression: &ExprDag) -> Result<Self, Diagnostic> {
        let definitions = expression
            .definitions()
            .iter()
            .map(|(digest, definition)| (*digest, definition.clone()))
            .collect::<Vec<_>>();
        let mut typed_constants = Vec::new();
        let mut array_operands = Vec::new();
        let mut symbols = Vec::new();
        let mut symbol_slots = HashMap::new();
        let mut instructions = Vec::with_capacity(expression.nodes().len());
        let mut values = Vec::with_capacity(expression.nodes().len());
        for (index, node) in expression.nodes().iter().enumerate() {
            if let ExprNode::Sample { value, .. } | ExprNode::Hold(value) = node {
                values.push(value_id(*value, &values)?);
                continue;
            }
            let instruction = match node {
                ExprNode::Constant(value) => {
                    if let Some(real) = value.real_scalar_value() {
                        Instruction::Constant(real)
                    } else if matches!(
                        value.value_type().scalar_domain(),
                        eqiora_core::ScalarDomain::Integer
                            | eqiora_core::ScalarDomain::Boolean
                            | eqiora_core::ScalarDomain::Complex
                            | eqiora_core::ScalarDomain::Enum
                    ) || (value.value_type().scalar_domain()
                        == eqiora_core::ScalarDomain::Real
                        && value.value_type().frame() == eqiora_core::ValueFrame::Invariant)
                    {
                        let slot =
                            u32::try_from(typed_constants.len()).map_err(|_| ir_size_error())?;
                        typed_constants.push(value.clone());
                        Instruction::TypedConstant(slot)
                    } else {
                        return Err(ir_builder_error(
                            "scalar IR requires real scalar or exact discrete constants",
                        ));
                    }
                }
                ExprNode::PureOperatorApplication(application) => {
                    let definition = definitions
                        .binary_search_by_key(&application.definition(), |(digest, _)| *digest)
                        .map_err(|_| ir_builder_error("pure operator definition is unavailable"))?;
                    let retained = &definitions[definition].1;
                    if !retained.result_rule().is_invariant_scalar()
                        || retained
                            .formals()
                            .iter()
                            .any(|formal| !formal.is_invariant_scalar())
                    {
                        return Err(ir_builder_error(
                            "tensor pure operators require component expansion before scalar execution",
                        ));
                    }
                    let arguments = application.arguments();
                    if array_operands
                        .len()
                        .checked_add(arguments.len())
                        .is_none_or(|n| n > 1_000_000)
                    {
                        return Err(ir_builder_error(
                            "operator argument count exceeds the component budget",
                        ));
                    }
                    let start = u32::try_from(array_operands.len()).map_err(|_| ir_size_error())?;
                    for argument in arguments {
                        array_operands.push(value_id(*argument, &values)?);
                    }
                    Instruction::PureOperator {
                        definition: definition as u32,
                        start,
                        len: arguments.len() as u32,
                    }
                }
                ExprNode::Array { elements } => {
                    if array_operands
                        .len()
                        .checked_add(elements.len())
                        .is_none_or(|count| count > 1_000_000)
                    {
                        return Err(ir_builder_error(
                            "array operand count exceeds the component budget",
                        ));
                    }
                    let start = u32::try_from(array_operands.len()).map_err(|_| ir_size_error())?;
                    let len = u32::try_from(elements.len()).map_err(|_| ir_size_error())?;
                    for element in elements {
                        array_operands.push(value_id(*element, &values)?);
                    }
                    Instruction::Array { start, len }
                }
                ExprNode::Select {
                    condition,
                    then_value,
                    else_value,
                } => Instruction::Select {
                    condition: value_id(*condition, &values)?,
                    then_value: value_id(*then_value, &values)?,
                    else_value: value_id(*else_value, &values)?,
                },
                ExprNode::Require { condition, value } => Instruction::Require {
                    condition: value_id(*condition, &values)?,
                    value: value_id(*value, &values)?,
                },
                ExprNode::UnaryMath(eqiora_schema::kernel::UnaryMathFunction::Abs, value) => {
                    let value = value_id(*value, &values)?;
                    let first = u32::try_from(instructions.len()).map_err(|_| ir_size_error())?;
                    let condition = first.checked_add(1).ok_or_else(ir_size_error)?;
                    let negative = first.checked_add(2).ok_or_else(ir_size_error)?;
                    // Derive a same-dimension zero and retain the existing point-bound
                    // comparison policy: a demanded switch at zero has no derivative.
                    instructions.push(Instruction::Sub(value, value));
                    instructions.push(Instruction::Compare(
                        eqiora_schema::kernel::ComparisonOp::GreaterEqual,
                        value,
                        ValueId(first),
                    ));
                    instructions.push(Instruction::Neg(value));
                    Instruction::Select {
                        condition: ValueId(condition),
                        then_value: value,
                        else_value: ValueId(negative),
                    }
                }
                ExprNode::UnaryMath(eqiora_schema::kernel::UnaryMathFunction::Sin, value) => {
                    Instruction::Sin(value_id(*value, &values)?)
                }
                ExprNode::UnaryMath(eqiora_schema::kernel::UnaryMathFunction::Sqrt, value) => {
                    Instruction::Sqrt(value_id(*value, &values)?)
                }
                ExprNode::Index { value, index } => {
                    Instruction::Index(value_id(*value, &values)?, *index)
                }
                ExprNode::Compare(op, a, b) => {
                    Instruction::Compare(*op, value_id(*a, &values)?, value_id(*b, &values)?)
                }
                ExprNode::Not(a) => Instruction::Not(value_id(*a, &values)?),
                ExprNode::And(a, b) => {
                    Instruction::And(value_id(*a, &values)?, value_id(*b, &values)?)
                }
                ExprNode::Or(a, b) => {
                    Instruction::Or(value_id(*a, &values)?, value_id(*b, &values)?)
                }
                ExprNode::Quotient(a, b) => {
                    Instruction::Quotient(value_id(*a, &values)?, value_id(*b, &values)?)
                }
                ExprNode::Remainder(a, b) => {
                    Instruction::Remainder(value_id(*a, &values)?, value_id(*b, &values)?)
                }
                ExprNode::ToReal(value) => Instruction::ToReal(value_id(*value, &values)?),
                ExprNode::Ordinal(value) => Instruction::Ordinal(value_id(*value, &values)?),
                ExprNode::ToInteger(value) => Instruction::ToInteger(value_id(*value, &values)?),
                ExprNode::Symbol(symbol) => {
                    let next_slot = u32::try_from(symbols.len()).map_err(|_| ir_size_error())?;
                    let slot = *symbol_slots.entry(*symbol).or_insert_with(|| {
                        symbols.push(*symbol);
                        SymbolSlot(next_slot)
                    });
                    Instruction::Read(slot)
                }
                ExprNode::Neg(value) => Instruction::Neg(value_id(*value, &values)?),
                ExprNode::Add(left, right) => {
                    Instruction::Add(value_id(*left, &values)?, value_id(*right, &values)?)
                }
                ExprNode::Sub(left, right) => {
                    Instruction::Sub(value_id(*left, &values)?, value_id(*right, &values)?)
                }
                ExprNode::Mul(left, right) => {
                    Instruction::Mul(value_id(*left, &values)?, value_id(*right, &values)?)
                }
                ExprNode::Div(left, right) => {
                    Instruction::Div(value_id(*left, &values)?, value_id(*right, &values)?)
                }
                ExprNode::PowI(base, exponent) => {
                    Instruction::PowI(value_id(*base, &values)?, *exponent)
                }
                _ => {
                    return Err(Diagnostic::error(
                        codes::INVALID_OPERATOR_IR,
                        "expression node is newer than scalar Operator IR",
                    )
                    .with_graph_path(ir_path(index)));
                }
            };
            values.push(ValueId(
                u32::try_from(instructions.len()).map_err(|_| ir_size_error())?,
            ));
            instructions.push(instruction);
        }
        let roots = expression
            .roots()
            .iter()
            .map(|root| value_id(*root, &values))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            source_values: values,
            typed_constants,
            definitions,
            array_operands,
            symbols,
            instructions,
            roots,
        })
    }
}

fn value_id(id: ExprId, values: &[ValueId]) -> Result<ValueId, Diagnostic> {
    usize::try_from(id.index())
        .ok()
        .and_then(|index| values.get(index))
        .copied()
        .ok_or_else(|| invalid_value(id, values.len()))
}

fn invalid_value(id: ExprId, upper_bound: usize) -> Diagnostic {
    Diagnostic::error(
        codes::INVALID_OPERATOR_IR,
        format!(
            "expression value {} is not below the instruction bound {upper_bound}",
            id.index()
        ),
    )
    .with_graph_path(ir_path(upper_bound))
}
