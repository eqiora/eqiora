//! Typed execution of the existing scalar SSA instruction stream.
use super::*;
use eqiora_core::{DynQuantity, ScalarDomain, ValueLiteral, ValueType};

impl ScalarOperatorIr {
    /// Evaluate requested original DAG roots without projecting exact discrete values
    /// through floating point. Unreachable symbol slots are never resolved.
    ///
    /// # Errors
    /// Rejects missing operands, incompatible values, and checked arithmetic failures.
    pub fn evaluate_typed(
        &self,
        roots: &[eqiora_schema::kernel::ExprId],
        resolve: &mut dyn FnMut(SymbolRef) -> Option<ValueLiteral>,
    ) -> Result<Vec<ValueLiteral>, Diagnostic> {
        self.evaluate_trace(roots, resolve)
            .map(|(values, _)| values)
    }

    pub(super) fn evaluate_trace(
        &self,
        roots: &[eqiora_schema::kernel::ExprId],
        resolve: &mut dyn FnMut(SymbolRef) -> Option<ValueLiteral>,
    ) -> Result<(Vec<ValueLiteral>, Vec<Option<ValueLiteral>>), Diagnostic> {
        let roots = roots
            .iter()
            .map(|id| {
                self.source_values
                    .get(id.index() as usize)
                    .copied()
                    .ok_or_else(|| ir_builder_error("typed source root is unavailable"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        enum Frame {
            Demand(ValueId),
            Apply(ValueId),
            Logical(ValueId),
            Branch(ValueId),
        }
        let mut component_work = 0usize;
        let mut values: Vec<Option<ValueLiteral>> = vec![None; self.instructions.len()];
        for &root in &roots {
            let mut pending = vec![Frame::Demand(root)];
            while let Some(frame) = pending.pop() {
                let id = match frame {
                    Frame::Demand(id)
                    | Frame::Apply(id)
                    | Frame::Logical(id)
                    | Frame::Branch(id) => id,
                };
                let index = id.0 as usize;
                let instruction = *self
                    .instructions
                    .get(index)
                    .ok_or_else(|| ir_builder_error("typed operand is unavailable"))?;
                if values[index].is_some() {
                    continue;
                }
                if let Instruction::Select { condition, .. }
                | Instruction::Require { condition, .. } = instruction
                {
                    if matches!(frame, Frame::Demand(_)) {
                        pending.push(Frame::Branch(id));
                        pending.push(Frame::Demand(condition));
                        continue;
                    }
                    if matches!(frame, Frame::Branch(_)) {
                        let test = boolean(
                            values[condition.0 as usize]
                                .as_ref()
                                .expect("demanded condition"),
                        )?;
                        let selected = match instruction {
                            Instruction::Select {
                                then_value,
                                else_value,
                                ..
                            } => {
                                if test {
                                    then_value
                                } else {
                                    else_value
                                }
                            }
                            Instruction::Require { value, .. } if test => value,
                            _ => {
                                return Err(ir_builder_error(
                                    "required expression domain condition is false",
                                ));
                            }
                        };
                        pending.push(Frame::Apply(id));
                        pending.push(Frame::Demand(selected));
                        continue;
                    }
                }
                if let Instruction::And(left, right) | Instruction::Or(left, right) = instruction {
                    if matches!(frame, Frame::Demand(_)) {
                        pending.push(Frame::Logical(id));
                        pending.push(Frame::Demand(left));
                        continue;
                    }
                    if matches!(frame, Frame::Logical(_)) {
                        let left =
                            boolean(values[left.0 as usize].as_ref().expect("demanded left"))?;
                        if matches!(instruction, Instruction::And(_, _)) && !left
                            || matches!(instruction, Instruction::Or(_, _)) && left
                        {
                            values[index] = Some(ValueLiteral::boolean(left));
                        } else {
                            pending.push(Frame::Apply(id));
                            pending.push(Frame::Demand(right));
                        }
                        continue;
                    }
                }
                if matches!(frame, Frame::Demand(_)) {
                    pending.push(Frame::Apply(id));
                    match instruction {
                        Instruction::And(_, _)
                        | Instruction::Or(_, _)
                        | Instruction::Select { .. }
                        | Instruction::Require { .. } => {
                            unreachable!("logical demand is staged separately")
                        }
                        Instruction::Constant(_)
                        | Instruction::TypedConstant(_)
                        | Instruction::Read(_) => {}
                        Instruction::Array { start, len }
                        | Instruction::PureOperator { start, len, .. } => {
                            let operands = self
                                .array_operands
                                .get(start as usize..start as usize + len as usize)
                                .ok_or_else(|| {
                                    ir_builder_error("array operand range unavailable")
                                })?;
                            pending.extend(operands.iter().rev().copied().map(Frame::Demand));
                        }
                        Instruction::Sin(a)
                        | Instruction::Sqrt(a)
                        | Instruction::Index(a, _)
                        | Instruction::Neg(a)
                        | Instruction::PowI(a, _)
                        | Instruction::ToReal(a)
                        | Instruction::ToInteger(a)
                        | Instruction::Ordinal(a)
                        | Instruction::Not(a) => pending.push(Frame::Demand(a)),
                        Instruction::Compare(_, a, b)
                        | Instruction::Add(a, b)
                        | Instruction::Sub(a, b)
                        | Instruction::Mul(a, b)
                        | Instruction::Div(a, b)
                        | Instruction::Quotient(a, b)
                        | Instruction::Remainder(a, b) => {
                            pending.push(Frame::Demand(b));
                            pending.push(Frame::Demand(a));
                        }
                    }
                    continue;
                }
                let read = |id: ValueId| {
                    values
                        .get(id.0 as usize)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| ir_builder_error("typed scalar operand is unavailable"))
                };
                let value = match instruction {
                    Instruction::Select {
                        condition,
                        then_value,
                        else_value,
                    } => read(if boolean(read(condition)?)? {
                        then_value
                    } else {
                        else_value
                    })?
                    .clone(),
                    Instruction::Require { value, .. } => read(value)?.clone(),
                    Instruction::Sin(value) => {
                        let value = real(read(value)?)?;
                        if value.dim() != eqiora_core::DimExponents::DIMENSIONLESS {
                            return Err(ir_builder_error(
                                "sine requires a dimensionless real scalar",
                            ));
                        }
                        literal(DynQuantity::new(
                            value.value().sin(),
                            eqiora_core::DimExponents::DIMENSIONLESS,
                        ))?
                    }
                    Instruction::Sqrt(value) => {
                        let value = real(read(value)?)?;
                        if value.value() < 0. {
                            return Err(ir_builder_error(
                                "real square root requires a nonnegative argument",
                            ));
                        }
                        let dimension = value.dim().pow(1, 2).ok_or_else(|| {
                            ir_builder_error("square-root dimension exceeds bounds")
                        })?;
                        literal(DynQuantity::new(value.value().sqrt(), dimension))?
                    }

                    Instruction::PureOperator {
                        definition,
                        start,
                        len,
                    } => {
                        let definition = &self.definitions[definition as usize].1;
                        let arguments = self.array_operands
                            [start as usize..start as usize + len as usize]
                            .iter()
                            .map(|id| read(*id))
                            .collect::<Result<Vec<_>, _>>()?;
                        let cost = definition.nodes().len() + arguments.len();
                        check_component_work(component_work, cost)?;
                        component_work += cost;
                        evaluate_pure_operator(definition, &arguments)?
                    }
                    Instruction::Array { start, len } => {
                        let elements = self.array_operands
                            [start as usize..start as usize + len as usize]
                            .iter()
                            .map(|id| read(*id))
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
                        let ty = eqiora_schema::kernel::typing::ExpressionType::array(&types)
                            .map_err(|error| ir_builder_error(error.to_string()))?;
                        let count = ty
                            .value_type
                            .shape()
                            .component_count()
                            .expect("checked type");
                        check_component_work(component_work, count)?;
                        ValueLiteral::array(&elements).map_err(discrete_error)?
                    }
                    Instruction::Index(value, index) => {
                        let value = read(value)?;
                        require_channels(value.value_type())?;
                        let ty = eqiora_schema::kernel::typing::ExpressionType::<()>::new(
                            value.value_type().clone(),
                            None,
                        )
                        .index(index)
                        .map_err(|error| ir_builder_error(error.to_string()))?;
                        check_component_work(
                            component_work,
                            ty.value_type
                                .shape()
                                .component_count()
                                .expect("checked type"),
                        )?;
                        value.index(index).map_err(discrete_error)?
                    }
                    Instruction::Not(value) => ValueLiteral::boolean(!boolean(read(value)?)?),
                    Instruction::And(_, right) | Instruction::Or(_, right) => {
                        ValueLiteral::boolean(boolean(read(right)?)?)
                    }
                    Instruction::Compare(op, left, right) => {
                        compare(op, read(left)?, read(right)?)?
                    }
                    Instruction::Constant(value) => literal(value)?,
                    Instruction::Read(slot) => self
                        .symbols
                        .get(slot_index(slot, index)?)
                        .and_then(|symbol| resolve(*symbol))
                        .ok_or_else(|| ir_builder_error("typed scalar input is unavailable"))?,
                    Instruction::TypedConstant(slot) => {
                        let value = self
                            .typed_constants
                            .get(slot as usize)
                            .ok_or_else(|| ir_builder_error("typed constant is unavailable"))?;
                        check_component_work(component_work, value.component_count())?;
                        value.clone()
                    }
                    Instruction::Quotient(a, b) => read(a)?
                        .checked_quotient(read(b)?)
                        .map_err(discrete_error)?,
                    Instruction::Remainder(a, b) => read(a)?
                        .checked_remainder(read(b)?)
                        .map_err(discrete_error)?,
                    Instruction::ToReal(a) => read(a)?.to_real().map_err(discrete_error)?,
                    Instruction::ToInteger(a) => read(a)?.to_integer().map_err(discrete_error)?,
                    Instruction::Ordinal(a) => read(a)?.ordinal().map_err(discrete_error)?,
                    Instruction::Neg(value) => {
                        let value = read(value)?;
                        require_scalar_arithmetic(value)?;
                        if value.value_type().scalar_domain() == ScalarDomain::Integer {
                            value.checked_neg().map_err(discrete_error)?
                        } else {
                            literal(-real(value)?)?
                        }
                    }
                    Instruction::Add(left, right) => {
                        let left = read(left)?;
                        let right = read(right)?;
                        require_scalar_arithmetic(left)?;
                        require_scalar_arithmetic(right)?;
                        if left.value_type().scalar_domain() == ScalarDomain::Integer {
                            left.checked_add(right).map_err(discrete_error)?
                        } else {
                            literal(real(left)?.try_add(real(right)?)?)?
                        }
                    }
                    Instruction::Sub(left, right) => {
                        let left = read(left)?;
                        let right = read(right)?;
                        require_scalar_arithmetic(left)?;
                        require_scalar_arithmetic(right)?;
                        if left.value_type().scalar_domain() == ScalarDomain::Integer {
                            left.checked_sub(right).map_err(discrete_error)?
                        } else {
                            literal(real(left)?.try_sub(real(right)?)?)?
                        }
                    }
                    Instruction::Mul(left, right) => {
                        let left = read(left)?;
                        let right = read(right)?;
                        require_scalar_arithmetic(left)?;
                        require_scalar_arithmetic(right)?;
                        if left.value_type().scalar_domain() == ScalarDomain::Integer {
                            left.checked_mul(right).map_err(discrete_error)?
                        } else {
                            literal(real(left)?.try_mul(real(right)?)?)?
                        }
                    }
                    Instruction::Div(left, right) => {
                        literal(real(read(left)?)?.try_div(real(read(right)?)?)?)?
                    }
                    Instruction::PowI(base, exponent) => {
                        let base = real(read(base)?)?;
                        let dimension = base.dim().pow(exponent, 1).ok_or_else(|| {
                            ir_builder_error("typed power dimension exceeds bounds")
                        })?;
                        literal(DynQuantity::new(base.value().powi(exponent), dimension))?
                    }
                };
                check_component_work(component_work, value.component_count())?;
                component_work += value.component_count();
                values[index] = Some(value);
            }
        }
        let results = roots
            .iter()
            .map(|root| {
                values
                    .get(root.0 as usize)
                    .and_then(Option::as_ref)
                    .cloned()
                    .ok_or_else(|| ir_builder_error("typed scalar root is unavailable"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((results, values))
    }
}

fn real(value: &ValueLiteral) -> Result<DynQuantity, Diagnostic> {
    value.real_scalar_value().ok_or_else(|| {
        Diagnostic::error(
            codes::NOT_IMPLEMENTED,
            "scalar IR numerical operation requires a real scalar",
        )
    })
}

fn literal(value: DynQuantity) -> Result<ValueLiteral, Diagnostic> {
    ValueLiteral::from_real(
        ValueType::scalar(ScalarDomain::Real, value.dim()).expect("numeric scalar type"),
        value.value(),
    )
    .map_err(|_| {
        Diagnostic::error(
            codes::NONFINITE_EVALUATION,
            "typed scalar operation produced a nonfinite value",
        )
    })
}

fn discrete_error(error: eqiora_core::InvalidValueLiteral) -> Diagnostic {
    ir_builder_error(format!("exact discrete operation rejected: {error:?}"))
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
        return Err(ir_builder_error(
            "channel execution requires invariant real or integer values",
        ));
    }
    Ok(())
}
fn component_budget_error() -> Diagnostic {
    ir_builder_error("typed evaluation exceeds the one-million component work budget")
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
        return Err(ir_builder_error(
            "channel arrays require explicit indexing before arithmetic",
        ));
    }
    Ok(())
}

fn evaluate_pure_operator(
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
        return Err(ir_builder_error(
            "pure operator execution requires real scalar arguments",
        ));
    }
    let instance = definition
        .instantiate(&types)
        .map_err(|error| ir_builder_error(error.to_string()))?;
    let mut builder = ExprDagBuilder::new();
    let arguments = arguments
        .iter()
        .map(|value| builder.constant((*value).clone()))
        .collect::<Result<Vec<_>, _>>()?;
    let root = builder.project_operator_component(
        &instance,
        &arguments.iter().map(|id| [*id]).collect::<Vec<_>>(),
        &[],
        1_000_000,
    )?;
    let dag = builder.finish([root])?;
    let ir = ScalarOperatorIr::lower(&dag)?;
    ir.evaluate_typed(&[root], &mut |_| None)?
        .pop()
        .ok_or_else(|| ir_builder_error("pure operator result is absent"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::{DimExponents, Id, entity::kinds};
    use eqiora_schema::kernel::ExprDagBuilder;

    #[test]
    fn channels_construct_index_and_preserve_integer_low_bits() {
        let mut builder = ExprDagBuilder::new();
        let ty = ValueType::scalar(ScalarDomain::Integer, DimExponents::DIMENSIONLESS)
            .expect("numeric scalar type");
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
        let values = ScalarOperatorIr::lower(&dag)
            .unwrap()
            .evaluate_typed(&[result], &mut |_| None)
            .unwrap();
        assert_eq!(
            values[0].integer_components().unwrap().collect::<Vec<_>>(),
            vec![9_007_199_254_740_994, 9_007_199_254_740_993]
        );
        let ir = ScalarOperatorIr::lower(&dag).unwrap();
        assert!(ir.evaluate(&[]).is_err());
        assert!(ir.linearize(&[], &[]).is_err());
    }
    #[test]
    fn channel_intermediate_budget_and_direct_array_arithmetic_reject() {
        let mut builder = ExprDagBuilder::new();
        let ty = ValueType::scalar(ScalarDomain::Integer, DimExponents::DIMENSIONLESS)
            .expect("numeric scalar type")
            .array(600_000)
            .unwrap();
        let input = builder
            .constant(ValueLiteral::from_integer(ty, 0).unwrap())
            .unwrap();
        let result = builder.array([input, input]).unwrap();
        let dag = builder.finish([result]).unwrap();
        assert!(
            ScalarOperatorIr::lower(&dag)
                .unwrap()
                .evaluate_typed(&[result], &mut |_| None)
                .unwrap_err()
                .message()
                .contains("component")
        );
        let mut builder = ExprDagBuilder::new();
        let ty = ValueType::scalar(ScalarDomain::Integer, DimExponents::DIMENSIONLESS)
            .expect("numeric scalar type")
            .array(2)
            .unwrap();
        let input = builder
            .constant(ValueLiteral::from_integer(ty, 0).unwrap())
            .unwrap();
        let result = builder.add(input, input).unwrap();
        let dag = builder.finish([result]).unwrap();
        assert!(
            ScalarOperatorIr::lower(&dag)
                .unwrap()
                .evaluate_typed(&[result], &mut |_| None)
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
            .expect("numeric scalar type");
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
        let ir = ScalarOperatorIr::lower(&dag).unwrap();
        let _ = owner;
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
            let result = ir.evaluate_typed(roots, &mut resolve);
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
    fn requested_integer_rhs_skips_unresolved_target_and_retains_low_bit() {
        let ty = ValueType::scalar(ScalarDomain::Integer, DimExponents::DIMENSIONLESS)
            .expect("numeric scalar type");
        let mut dag = ExprDagBuilder::new();
        let target = dag
            .symbol(SymbolRef::Next(Id::<kinds::Field>::new()))
            .unwrap();
        let value = dag
            .constant(ValueLiteral::from_integer(ty.clone(), 9_007_199_254_740_993).unwrap())
            .unwrap();
        let one = dag
            .constant(ValueLiteral::from_integer(ty, 1).unwrap())
            .unwrap();
        let rhs = dag.add(value, one).unwrap();
        let residual = dag.sub(target, rhs).unwrap();
        let dag = dag.finish([residual]).unwrap();
        let ir = ScalarOperatorIr::lower(&dag).unwrap();
        let result = ir
            .evaluate_typed(&[rhs], &mut |_| {
                panic!("unused assignment target must not be resolved")
            })
            .unwrap();
        assert_eq!(
            result[0].integer_scalar_value(),
            Some(9_007_199_254_740_994)
        );
        assert!(ir.evaluate(&[0.]).is_err());
    }

    #[test]
    fn quotient_remainder_and_explicit_conversion_preserve_declared_semantics() {
        let ty = ValueType::scalar(ScalarDomain::Integer, DimExponents::DIMENSIONLESS)
            .expect("numeric scalar type");
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
        let ir = ScalarOperatorIr::lower(&dag).unwrap();
        let values = ir.evaluate_typed(dag.roots(), &mut |_| None).unwrap();
        assert_eq!(values[0].integer_scalar_value(), Some(-2));
        assert_eq!(values[1].integer_scalar_value(), Some(-1));
        assert_eq!(values[2].real_scalar_value().unwrap().value(), -2.);
        assert_eq!(values[3].integer_scalar_value(), Some(-2));
        assert_eq!(values[4].integer_scalar_value(), Some(2));
    }
    #[test]
    fn extrema_keep_adjacent_integers_exact_and_evaluate_both_operands() {
        use eqiora_schema::kernel::ExprDagBuilder;
        let integer = |n| {
            ValueLiteral::from_integer(
                ValueType::scalar(
                    ScalarDomain::Integer,
                    eqiora_core::DimExponents::DIMENSIONLESS,
                )
                .expect("numeric scalar type"),
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
            ScalarOperatorIr::lower(&dag)
                .unwrap()
                .evaluate_typed(&roots, &mut |_| None)
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
            ScalarOperatorIr::lower(&dag)
                .unwrap()
                .evaluate_typed(&roots, &mut |_| None)
                .is_err()
        );
    }

    #[test]
    fn extrema_refuse_numerical_evaluation_and_derivatives() {
        use eqiora_schema::kernel::ExprDagBuilder;
        let mut builder = ExprDagBuilder::new();
        let a = builder
            .constant(DynQuantity::new(
                1.,
                eqiora_core::DimExponents::DIMENSIONLESS,
            ))
            .unwrap();
        let b = builder
            .constant(DynQuantity::new(
                2.,
                eqiora_core::DimExponents::DIMENSIONLESS,
            ))
            .unwrap();
        let root = builder.max(a, b).unwrap();
        let ir = ScalarOperatorIr::lower(&builder.finish([root]).unwrap()).unwrap();
        assert!(ir.evaluate(&[]).is_err());
        assert!(ir.linearize(&[], &[]).is_err());
    }
}
