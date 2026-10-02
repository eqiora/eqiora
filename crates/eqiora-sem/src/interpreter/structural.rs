//! Necessary equation balance for the already admitted scalar continuous system.
//! A Field and its derivative share a column here: this is incidence matching,
//! not index reduction, numerical rank, or a choice of solved variable.

use super::*;
use eqiora_schema::kernel::{ExprDag, ExprId};

pub(super) fn validate(program: &KernelProgram, plan: &ExecutionPlan) -> Result<(), Diagnostic> {
    let variables = plan
        .differential_fields
        .union(&plan.algebraic_fields)
        .copied()
        .map(Variable::Field)
        .chain(plan.continuous_ports.iter().copied().map(Variable::Port))
        .chain(
            plan.physical_unknowns
                .iter()
                .copied()
                .map(Variable::Physical),
        )
        .collect::<BTreeSet<_>>()
        .into_iter()
        .enumerate()
        .map(|(index, variable)| (variable, index))
        .collect::<BTreeMap<_, _>>();
    let mut rows = Vec::new();
    let mut owners = Vec::new();
    for &owner in &plan.continuous_relations {
        let Some(KernelNode::Relation(relation)) = program.node(owner) else {
            return Err(execution_error("validated Relation is unavailable", 0.0));
        };
        for (index, (left, right)) in relation.equation_sides().enumerate() {
            rows.push(incidence(
                relation.expression(),
                &[left, right],
                plan,
                &variables,
            )?);
            owners.push((owner, index));
        }
    }
    for system in &plan.physical_systems {
        for junction in system.junctions() {
            for (index, &root) in junction.dag().roots().iter().enumerate() {
                rows.push(incidence(junction.dag(), &[root], plan, &variables)?);
                owners.push((junction.connection().erase(), index));
            }
        }
    }
    if rows.len() != variables.len() {
        return Err(Diagnostic::error(
            codes::NONSQUARE_SYSTEM,
            format!("continuous structural balance requires a square system; found {} equations and {} unknowns (initial equations are separate)", rows.len(), variables.len()),
        ).with_graph_path(GraphPath::new(["model".to_owned(), program.model().to_string()])));
    }
    let matched = maximum_matching(&rows, variables.len());
    if let Some(row) = matched.iter().position(Option::is_none) {
        let rank = matched.iter().filter(|entry| entry.is_some()).count();
        let unmatched = variables
            .iter()
            .filter(|(_, column)| !matched.contains(&Some(**column)))
            .map(|(variable, _)| format!("{variable:?}"))
            .collect::<Vec<_>>()
            .join(", ");
        let (owner, equation) = owners[row];
        return Err(Diagnostic::error(
            codes::NONLINEAR_SOLVE_FAILED,
            format!("continuous system is structurally singular: incidence rank {rank} for {} unknowns; unmatched equation {}; unmatched unknowns: {unmatched}. A full structural rank would still require numerical regularity", variables.len(), equation + 1),
        ).with_graph_path(kernel_path(owner)));
    }
    Ok(())
}

fn incidence(
    dag: &ExprDag,
    roots: &[ExprId],
    plan: &ExecutionPlan,
    variables: &BTreeMap<Variable, usize>,
) -> Result<Vec<usize>, Diagnostic> {
    let mut pending = roots.to_vec();
    let mut seen = vec![false; dag.nodes().len()];
    let mut columns = BTreeSet::new();
    while let Some(id) = pending.pop() {
        let index = id.index() as usize;
        if std::mem::replace(&mut seen[index], true) {
            continue;
        }
        match &dag.nodes()[index] {
            ExprNode::Symbol(symbol) => {
                let variable = match symbol {
                    SymbolRef::Field(field) | SymbolRef::Derivative(field) => {
                        Some(Variable::Field(field.erase()))
                    }
                    SymbolRef::Port(port) => Some(Variable::Port(
                        plan.signal_sources
                            .get(&port.erase())
                            .copied()
                            .unwrap_or_else(|| port.erase()),
                    )),
                    SymbolRef::Across(port) => {
                        Some(Variable::Physical(PhysicalUnknown::Across(*port)))
                    }
                    SymbolRef::Through(port) => {
                        Some(Variable::Physical(PhysicalUnknown::Through(*port)))
                    }
                    _ => None,
                };
                if let Some(column) = variable.and_then(|variable| variables.get(&variable)) {
                    columns.insert(*column);
                }
            }
            ExprNode::Constant(_) | ExprNode::SpatialCoordinate(_) => {}
            ExprNode::Require { condition, value } => pending.extend([*condition, *value]),
            ExprNode::Select {
                condition,
                then_value,
                else_value,
            } => pending.extend([*condition, *then_value, *else_value]),
            ExprNode::Array { elements } => pending.extend(elements),
            ExprNode::Index { value, .. }
            | ExprNode::Sample { value, .. }
            | ExprNode::Not(value)
            | ExprNode::Ordinal(value)
            | ExprNode::ToReal(value)
            | ExprNode::ToInteger(value)
            | ExprNode::Hold(value)
            | ExprNode::Neg(value)
            | ExprNode::PowI(value, _)
            | ExprNode::UnaryMath(_, value)
            | ExprNode::Gradient(value)
            | ExprNode::Divergence(value)
            | ExprNode::SymmetricPart(value)
            | ExprNode::IsotropicLift(value)
            | ExprNode::Trace(value)
            | ExprNode::NormalComponent(value) => pending.push(*value),
            ExprNode::Complex {
                real: left,
                imag: right,
            }
            | ExprNode::Compare(_, left, right)
            | ExprNode::And(left, right)
            | ExprNode::Or(left, right)
            | ExprNode::Add(left, right)
            | ExprNode::Sub(left, right)
            | ExprNode::Mul(left, right)
            | ExprNode::Div(left, right)
            | ExprNode::Quotient(left, right)
            | ExprNode::Remainder(left, right) => pending.extend([*left, *right]),
            ExprNode::PureOperatorApplication(application) => {
                pending.extend(application.arguments())
            }
            _ => {
                return Err(Diagnostic::error(
                    codes::NOT_IMPLEMENTED,
                    "expression is newer than scalar structural incidence analysis",
                ));
            }
        }
    }
    Ok(columns.into_iter().collect())
}

/// Augment along alternating paths, without recursion or source-order pivots.
fn maximum_matching(rows: &[Vec<usize>], columns: usize) -> Vec<Option<usize>> {
    let mut row_match = vec![None; rows.len()];
    let mut column_match = vec![None; columns];
    for start in 0..rows.len() {
        let mut parents = vec![None; columns];
        let mut pending = std::collections::VecDeque::from([start]);
        let mut endpoint = None;
        'search: while let Some(row) = pending.pop_front() {
            for &column in &rows[row] {
                if parents[column].is_some() {
                    continue;
                }
                parents[column] = Some(row);
                if let Some(next) = column_match[column] {
                    pending.push_back(next);
                } else {
                    endpoint = Some(column);
                    break 'search;
                }
            }
        }
        while let Some(column) = endpoint {
            let row = parents[column].expect("alternating path has a predecessor");
            endpoint = row_match[row].replace(column);
            column_match[column] = Some(row);
        }
    }
    row_match
}

#[cfg(test)]
mod tests {
    use super::maximum_matching;

    #[test]
    fn all_three_by_three_patterns_agree_with_distinct_column_assignments() {
        // Independent exhaustive assignment oracle, including unassigned rows.
        // This exercises alternating-path reassignment and deficient subblocks.
        for pattern in 0..512 {
            let rows = (0..3)
                .map(|row| {
                    (0..3)
                        .filter(|column| pattern & (1 << (3 * row + column)) != 0)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let mut expected = 0;
            for first in 0..4 {
                for second in 0..4 {
                    for third in 0..4 {
                        let assignment = [first, second, third];
                        if assignment.iter().enumerate().all(|(row, &column)| {
                            column == 3
                                || (rows[row].contains(&column)
                                    && !assignment[..row].contains(&column))
                        }) {
                            expected = expected
                                .max(assignment.iter().filter(|&&column| column != 3).count());
                        }
                    }
                }
            }
            let matched = maximum_matching(&rows, 3);
            assert_eq!(
                matched.iter().filter(|column| column.is_some()).count(),
                expected,
                "pattern={pattern}"
            );
            for (row, column) in matched.iter().enumerate() {
                if let Some(column) = column {
                    assert!(rows[row].contains(column));
                    assert!(!matched[..row].contains(&Some(*column)));
                }
            }
        }
    }
}
