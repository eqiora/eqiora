use std::collections::{HashMap, HashSet};

use eqiora_core::diagnostic::codes;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, GraphPath, Id, RawId};
use eqiora_ir::{ScalarOperatorIr, SymbolicLinearityFailure};
use eqiora_schema::kernel::SymbolRef;

pub(super) fn validate_guard_symbols(
    owner: RawId,
    operator: &ScalarOperatorIr,
    states: &[Id<kinds::Field>],
) -> Result<(), Diagnostic> {
    let state_set = states.iter().copied().collect::<HashSet<_>>();
    if operator.residual_count() != 1
        || operator.symbols().iter().any(|symbol| match symbol {
            SymbolRef::Field(field) => !state_set.contains(field),
            SymbolRef::Parameter(_) | SymbolRef::Time => false,
            _ => true,
        })
    {
        Err(invalid_event(
            owner,
            "event guard must be one scalar expression of flow state, Parameter, and time",
        ))
    } else {
        Ok(())
    }
}

pub(super) fn validate_reset_symbols(
    owner: RawId,
    operator: &ScalarOperatorIr,
    states: &[Id<kinds::Field>],
) -> Result<(), Diagnostic> {
    let state_set = states.iter().copied().collect::<HashSet<_>>();
    if operator.symbols().iter().any(|symbol| match symbol {
        SymbolRef::Pre(field) | SymbolRef::Next(field) => !state_set.contains(field),
        SymbolRef::Parameter(_) | SymbolRef::Time => false,
        _ => true,
    }) {
        Err(invalid_event(
            owner,
            "event reset must be an implicit Relation of Pre, Next, Parameter, and time",
        ))
    } else {
        Ok(())
    }
}

pub(super) fn append_operator_parameters(
    parameters: &mut Vec<Id<kinds::Parameter>>,
    operator: &ScalarOperatorIr,
) {
    for symbol in operator.symbols() {
        if let SymbolRef::Parameter(parameter) = *symbol
            && !parameters.contains(&parameter)
        {
            parameters.push(parameter);
        }
    }
}

pub(super) fn state_coordinate(
    owner: RawId,
    states: &HashMap<Id<kinds::Field>, usize>,
    field: Id<kinds::Field>,
) -> Result<usize, Diagnostic> {
    states
        .get(&field)
        .copied()
        .ok_or_else(|| invalid_event(owner, "event references a Field outside the flow state"))
}

pub(super) fn next_structure_error(owner: RawId, failure: SymbolicLinearityFailure) -> Diagnostic {
    invalid_event(
        owner,
        format!("cannot prove constant implicit reset Next Jacobian: {failure:?}"),
    )
}

pub(super) fn invalid_event(owner: RawId, message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_TIME_LOWERING, message).with_graph_path(GraphPath::new([
        "hybrid-lowering".to_owned(),
        owner.to_string(),
    ]))
}
