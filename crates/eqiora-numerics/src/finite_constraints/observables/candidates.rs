//! Numerical Observable values for canonical reevaluation of original Relation operands.
use super::*;
use crate::finite_constraints::FiniteConstraintProblem;
use eqiora_core::ValueLiteral;
use eqiora_ir::ScalarOperatorIr;

pub(in crate::finite_constraints) fn candidates(
    problem: &FiniteConstraintProblem,
    relation: Id<kinds::Relation>,
    fields: &[(Id<kinds::Field>, ValueLiteral)],
) -> Result<Vec<(Id<kinds::Observable>, ValueLiteral)>, Diagnostic> {
    let kernel = &problem.kernel;
    let Some(KernelNode::Relation(definition)) = kernel.node(relation.erase()) else {
        return Err(invalid("original Relation is absent from its Model"));
    };
    let mut seen = std::collections::BTreeSet::new();
    let ids = definition
        .expression()
        .nodes()
        .iter()
        .filter_map(|node| match node {
            ExprNode::Symbol(SymbolRef::Observable(id)) if seen.insert(id.erase()) => Some(*id),
            _ => None,
        })
        .collect::<Vec<_>>();
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    let mut builder = ExprDagBuilder::new();
    let roots = ids
        .iter()
        .map(|id| builder.symbol(SymbolRef::Observable(*id)))
        .collect::<Result<Vec<_>, _>>()?;
    let expression = expand(kernel, &builder.finish(roots)?)?;
    let operator = ScalarOperatorIr::lower(&expression)?;
    let values = operator.evaluate_typed(expression.roots(), &mut |symbol| match symbol {
        SymbolRef::Field(id) => fields
            .iter()
            .find(|(candidate, _)| *candidate == id)
            .map(|(_, value)| value.clone()),
        SymbolRef::Parameter(id) => problem
            .parameter_candidates
            .iter()
            .find(|(candidate, _)| *candidate == id)
            .map(|(_, value)| value.clone())
            .or_else(|| match kernel.node(id.erase()) {
                Some(KernelNode::Parameter(parameter)) => Some(parameter.value().clone()),
                _ => None,
            }),
        _ => None,
    })?;
    ids.into_iter()
        .zip(values)
        .map(|(id, value)| {
            let Some(KernelNode::Observable(definition)) = kernel.node(id.erase()) else {
                return Err(invalid("original Observable is absent from its Model"));
            };
            if value.value_type() != definition.value_type() {
                return Err(invalid(
                    "finite Observable candidate has an incompatible value type",
                ));
            }
            Ok((id, value))
        })
        .collect()
}
