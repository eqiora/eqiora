//! Whole typed discrete updates share the expression backend with numeric initialization.
use super::*;
use eqiora_core::{ScalarDomain, ValueLiteral, ValueType};
use eqiora_schema::kernel::{ExprDag, ExprId};

/// Channel axes retain invariant scalar elements; spatial axes stay unsupported.
pub(super) fn supported_type(value: &ValueType) -> bool {
    let channels = value.array_rank() > 0
        && value.array_rank() == value.shape().rank()
        && value.frame() == eqiora_core::ValueFrame::Invariant
        && matches!(
            value.scalar_domain(),
            ScalarDomain::Real | ScalarDomain::Integer
        );
    channels
        || (value.scalar_domain() == ScalarDomain::Real && value.shape().is_scalar())
        || *value == ValueType::boolean()
        || (value.scalar_domain() == ScalarDomain::Enum
            && value.shape().is_scalar()
            && value.array_rank() == 0)
        || (value.scalar_domain() == ScalarDomain::Integer && value.array_rank() == 0)
}

pub(super) fn requires_typed_assignment(program: &KernelProgram, symbol: SymbolRef) -> bool {
    program.execution_symbol_type(symbol).is_some_and(|value| {
        (value.array_rank() > 0
            && match symbol {
                SymbolRef::Field(id) | SymbolRef::Pre(id) => {
                    is_clocked_variable(program, id.erase())
                }
                _ => true,
            })
            || matches!(
                value.scalar_domain(),
                ScalarDomain::Integer | ScalarDomain::Boolean | ScalarDomain::Enum
            )
    })
}

pub(super) fn requires_typed_assignment_id(program: &KernelProgram, id: RawId) -> bool {
    match program.node(id) {
        Some(KernelNode::Field(field)) => {
            requires_typed_assignment(program, SymbolRef::Field(field.id()))
        }
        Some(KernelNode::Port(port)) => {
            requires_typed_assignment(program, SymbolRef::Port(port.id()))
        }
        _ => false,
    }
}

fn assignment(
    program: &KernelProgram,
    dag: &ExprDag,
    a: ExprId,
    b: ExprId,
) -> Option<(SymbolRef, ExprId)> {
    for (target, rhs) in [(a, b), (b, a)] {
        if let Some(ExprNode::Symbol(symbol)) = dag.nodes().get(target.index() as usize)
            && matches!(
                symbol,
                SymbolRef::Field(_) | SymbolRef::Pre(_) | SymbolRef::Next(_) | SymbolRef::Port(_)
            )
            && requires_typed_assignment(program, *symbol)
        {
            return Some((*symbol, rhs));
        }
    }
    None
}

pub(super) fn numerical_roots(
    program: &KernelProgram,
    relation: &eqiora_schema::kernel::RelationDef,
) -> Vec<ExprId> {
    relation
        .equation_sides()
        .filter(|(a, b)| assignment(program, relation.expression(), *a, *b).is_none())
        .flat_map(|(a, b)| [a, b])
        .collect()
}

pub(super) fn typed_fields(
    program: &KernelProgram,
    state: &RuntimeState,
) -> Result<BTreeMap<RawId, ValueLiteral>, Vec<Diagnostic>> {
    let mut fields = state.typed_fields.clone();
    for (&id, &value) in &state.fields {
        let Some(KernelNode::Field(field)) = program.node(id) else {
            continue;
        };
        fields.insert(
            id,
            ValueLiteral::from_real(field.value_type().clone(), value)
                .map_err(|_| vec![execution_error("accepted Field value is not finite", 0.0)])?,
        );
    }
    Ok(fields)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn stage(
    program: &KernelProgram,
    plan: &ExecutionPlan,
    state: &mut RuntimeState,
    relations: &BTreeSet<RawId>,
    time: f64,
    initial: bool,
    backend: &impl ExpressionBackend,
) -> Result<(), Diagnostic> {
    let mut pending = Vec::new();
    let mut targets = std::collections::HashSet::new();
    for &owner in relations {
        let Some(KernelNode::Relation(relation)) = program.node(owner) else {
            continue;
        };
        for (left, right) in relation.equation_sides() {
            if let Some((target, rhs)) = assignment(program, relation.expression(), left, right) {
                let target = if initial {
                    match target {
                        SymbolRef::Pre(id) => SymbolRef::Field(id),
                        other => other,
                    }
                } else {
                    target
                };
                let allowed = match target {
                    SymbolRef::Next(_) => !initial,
                    SymbolRef::Field(id) => initial || is_clocked_variable(program, id.erase()),
                    SymbolRef::Port(id) => {
                        !initial
                            && is_output_port(program, id.erase())
                            && !plan.signal_sources.contains_key(&id.erase())
                    }
                    _ => false,
                };
                if !allowed || !targets.insert(target) {
                    return Err(execution_error(
                        "typed direct assignments require one direct target at its owning activation",
                        time,
                    ));
                }
                pending.push((owner, relation.expression(), target, rhs));
            }
        }
    }
    while !pending.is_empty() {
        let before = pending.len();
        let mut waiting = Vec::new();
        for (owner, dag, target, rhs) in pending {
            let empty = BTreeMap::new();
            let context = EvalContext {
                program,
                time,
                fields: &state.fields,
                field_candidates: &empty,
                derivatives: &state.derivatives,
                next_fields: &empty,
                ports: &state.ports,
                port_candidates: &empty,
                signal_sources: &plan.signal_sources,
                physical: &state.physical,
                physical_candidates: &BTreeMap::new(),
                typed_fields: &state.typed_fields,
                typed_ports: &state.typed_ports,
                typed_next: &state.typed_next,
            };
            let mut missing = false;
            let result = backend.evaluate(owner, dag, &[rhs], &mut |symbol| {
                let value = if initial
                    && !matches!(symbol, SymbolRef::Parameter(_))
                    && !requires_typed_assignment(program, symbol)
                {
                    None
                } else {
                    evaluate::resolve_symbol(symbol, &context)
                };
                missing |= value.is_none();
                value
            });
            if missing {
                waiting.push((owner, dag, target, rhs));
                continue;
            }
            let mut result = result?;
            if result.len() != 1 {
                return Err(execution_error(
                    "direct assignment requires exactly one evaluated right side",
                    time,
                ));
            }
            let value = result
                .pop()
                .ok_or_else(|| execution_error("exact assignment has no value", time))?;
            if Some(value.value_type().clone()) != program.execution_symbol_type(target) {
                return Err(execution_error(
                    "exact assignment value differs from its complete target type",
                    time,
                ));
            }
            match target {
                SymbolRef::Next(id) => {
                    state.typed_next.insert(id.erase(), value);
                }
                SymbolRef::Field(id) => {
                    state.typed_fields.insert(id.erase(), value);
                }
                SymbolRef::Port(id) => {
                    state.typed_ports.insert(id.erase(), value);
                }
                _ => unreachable!(),
            }
        }
        if waiting.len() == before {
            return Err(execution_error(
                "typed direct assignments are cyclic or lack an accepted input",
                time,
            ));
        }
        pending = waiting;
    }
    if initial
        && plan.fields.iter().any(|id| {
            requires_typed_assignment_id(program, *id)
                && !is_clocked_variable(program, *id)
                && !state.typed_fields.contains_key(id)
        })
    {
        return Err(execution_error(
            "typed State requires an explicit initial assignment",
            time,
        ));
    }
    Ok(())
}

/// Bound complete retained state/Port/Parameter payloads before plan/state allocation.
/// The same scalar-component cap applies to sampled input and output retention.
pub(super) const MAX_COMPONENTS: usize = 1_000_000;

pub(super) fn validate_storage_budget(program: &KernelProgram) -> Result<(), Diagnostic> {
    let types = program.nodes().filter_map(|node| match node {
        KernelNode::Field(value) => Some(value.value_type()),
        KernelNode::Port(value) => value.signal_contract().map(|(_, value_type)| value_type),
        KernelNode::Parameter(value) => Some(value.value_type()),
        _ => None,
    });
    validate_component_total(types, MAX_COMPONENTS)
}

fn validate_component_total<'a>(
    types: impl Iterator<Item = &'a ValueType>,
    limit: usize,
) -> Result<(), Diagnostic> {
    let mut components = 0usize;
    for value_type in types {
        components = value_type
            .shape()
            .component_count()
            .and_then(|count| components.checked_add(count))
            .filter(|total| *total <= limit)
            .ok_or_else(|| {
                execution_error(
                    "reference value storage exceeds the scalar-component budget",
                    0.0,
                )
            })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn channel_profile_keeps_domain_and_spatial_axes_distinct() {
        use eqiora_core::{DimExponents, ValueFrame, ValueShape};
        let real = ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
            .expect("valid scalar type");
        assert!(supported_type(
            &real.clone().array(2).unwrap().array(3).unwrap()
        ));
        assert!(supported_type(
            &ValueType::scalar(ScalarDomain::Integer, DimExponents::DIMENSIONLESS)
                .expect("valid scalar type")
                .array(2)
                .unwrap()
        ));
        assert!(!supported_type(
            &ValueType::scalar(ScalarDomain::Complex, DimExponents::DIMENSIONLESS)
                .expect("valid scalar type")
                .array(2)
                .unwrap()
        ));
        assert!(ValueType::boolean().array(2).is_err());
        let vector = ValueType::shaped(
            ScalarDomain::Real,
            DimExponents::DIMENSIONLESS,
            ValueShape::new([2]).unwrap(),
            ValueFrame::SpatialCartesian,
        )
        .unwrap();
        assert!(!supported_type(&vector));
        assert!(!supported_type(&vector.array(2).unwrap()));
    }

    #[test]
    fn storage_budget_counts_complete_shapes_before_allocating_values() {
        let scalar =
            ValueType::scalar(ScalarDomain::Real, eqiora_core::DimExponents::DIMENSIONLESS)
                .expect("valid scalar type");
        let array = scalar.clone().array(3).unwrap();
        assert!(validate_component_total([&array, &scalar].into_iter(), 4).is_ok());
        assert!(validate_component_total([&array, &scalar].into_iter(), 3).is_err());
        let huge = scalar.array(1_000_001).unwrap();
        assert!(validate_component_total([&huge].into_iter(), MAX_COMPONENTS).is_err());
    }
}
