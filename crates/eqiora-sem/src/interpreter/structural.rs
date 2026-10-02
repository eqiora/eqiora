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
    let matched = maximum_matching(&rows, variables.len());
    let rank = matched.iter().filter(|entry| entry.is_some()).count();
    if rows.len() != variables.len() || rank != variables.len() {
        let (over, under) = deficient_blocks(&rows, variables.len(), &matched);
        let describe = |(block_rows, block_columns): &(BTreeSet<usize>, BTreeSet<usize>)| {
            let equations = block_rows
                .iter()
                .map(|&row| {
                    let (owner, equation) = owners[row];
                    format!("{owner} equation {}", equation + 1)
                })
                .collect::<Vec<_>>()
                .join(", ");
            let unknowns = variables
                .iter()
                .filter(|(_, column)| block_columns.contains(column))
                .map(|(variable, _)| match variable {
                    Variable::Field(id) | Variable::Port(id) => id.to_string(),
                    Variable::Derivative(id) => format!("derivative({id})"),
                    Variable::NextField(id) => format!("next({id})"),
                    Variable::Physical(PhysicalUnknown::Across(port)) => {
                        format!("across({})", port.erase())
                    }
                    Variable::Physical(PhysicalUnknown::Through(port)) => {
                        format!("through({})", port.erase())
                    }
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "{} equations [{equations}], {} unknowns [{unknowns}]",
                block_rows.len(),
                block_columns.len()
            )
        };
        let blocks = format!(
            "overdetermined block: {}; underdetermined block: {}",
            describe(&over),
            describe(&under)
        );
        let (code, message) = if rows.len() != variables.len() {
            (
                codes::NONSQUARE_SYSTEM,
                format!(
                    "continuous structural balance requires a square system; found {} equations and {} unknowns (initial equations are separate); {blocks}",
                    rows.len(),
                    variables.len()
                ),
            )
        } else {
            (
                codes::NONLINEAR_SOLVE_FAILED,
                format!(
                    "continuous system is structurally singular: incidence rank {rank} for {} unknowns; unmatched equation and unknown blocks: {blocks}. A full structural rank would still require numerical regularity",
                    variables.len()
                ),
            )
        };
        let path = over
            .0
            .iter()
            .next()
            .or_else(|| under.0.iter().next())
            .map(|&row| kernel_path(owners[row].0))
            .unwrap_or_else(|| GraphPath::new(["model".to_owned(), program.model().to_string()]));
        return Err(Diagnostic::error(code, message).with_graph_path(path));
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

/// The coarse deficient blocks of a maximum matching are invariant under the
/// chosen matching. Alternating paths from free rows identify excess equations;
/// paths from free columns in the transposed graph identify missing equations.
/// Balanced components are deliberately absent from both diagnostic blocks.
fn deficient_blocks(
    rows: &[Vec<usize>],
    columns: usize,
    matched: &[Option<usize>],
) -> (IncidenceBlock, IncidenceBlock) {
    let mut transposed = vec![Vec::new(); columns];
    let mut column_match = vec![None; columns];
    for (row, neighbors) in rows.iter().enumerate() {
        for &column in neighbors {
            transposed[column].push(row);
        }
        if let Some(column) = matched[row] {
            column_match[column] = Some(row);
        }
    }
    let over = alternating_block(rows, matched, &column_match);
    let (under_columns, under_rows) = alternating_block(&transposed, &column_match, matched);
    (over, (under_rows, under_columns))
}

type IncidenceBlock = (BTreeSet<usize>, BTreeSet<usize>);

fn alternating_block(
    edges: &[Vec<usize>],
    matched: &[Option<usize>],
    opposite_match: &[Option<usize>],
) -> IncidenceBlock {
    let mut reached = matched
        .iter()
        .enumerate()
        .filter_map(|(index, entry)| entry.is_none().then_some(index))
        .collect::<BTreeSet<_>>();
    let mut pending = reached
        .iter()
        .copied()
        .collect::<std::collections::VecDeque<_>>();
    let mut opposite = BTreeSet::new();
    while let Some(index) = pending.pop_front() {
        for &neighbor in &edges[index] {
            if opposite.insert(neighbor)
                && let Some(next) = opposite_match[neighbor]
                && reached.insert(next)
            {
                pending.push_back(next);
            }
        }
    }
    (reached, opposite)
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
    use super::{deficient_blocks, maximum_matching};
    use std::collections::BTreeSet;

    #[test]
    fn rectangular_and_empty_blocks_keep_the_balanced_core_separate() {
        let rows = vec![vec![0], vec![0], vec![1]];
        let (over, under) = deficient_blocks(&rows, 2, &maximum_matching(&rows, 2));
        assert_eq!(over, (BTreeSet::from([0, 1]), BTreeSet::from([0])));
        assert_eq!(under, (BTreeSet::new(), BTreeSet::new()));
        let rows = vec![vec![0, 1], vec![2]];
        let (over, under) = deficient_blocks(&rows, 3, &maximum_matching(&rows, 3));
        assert_eq!(over, (BTreeSet::new(), BTreeSet::new()));
        assert_eq!(under, (BTreeSet::from([0]), BTreeSet::from([0, 1])));
        assert_eq!(
            deficient_blocks(&[], 1, &[]).1,
            (BTreeSet::new(), BTreeSet::from([0]))
        );
        assert_eq!(
            deficient_blocks(&[vec![]], 0, &[None]).0,
            (BTreeSet::from([0]), BTreeSet::new())
        );
    }

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
            let mut assignments = Vec::new();
            for first in 0..4 {
                for second in 0..4 {
                    for third in 0..4 {
                        let assignment = [first, second, third];
                        if assignment.iter().enumerate().all(|(row, &column)| {
                            column == 3
                                || (rows[row].contains(&column)
                                    && !assignment[..row].contains(&column))
                        }) {
                            assignments.push(assignment);
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
            // Independent oracle: a row/column is in a deficient block exactly
            // when some maximum assignment leaves it free. Enumerate all such
            // assignments, without walking an alternating path.
            let mut free_rows = BTreeSet::new();
            let mut free_columns = BTreeSet::new();
            for assignment in assignments.iter().filter(|assignment| {
                assignment.iter().filter(|&&column| column != 3).count() == expected
            }) {
                free_rows.extend((0..3).filter(|&row| assignment[row] == 3));
                free_columns.extend((0..3).filter(|column| !assignment.contains(column)));
            }
            let (over, under) = deficient_blocks(&rows, 3, &matched);
            assert_eq!(over.0, free_rows, "over rows, pattern={pattern}");
            assert_eq!(under.1, free_columns, "under columns, pattern={pattern}");
            assert_eq!(
                over.1,
                free_rows
                    .iter()
                    .flat_map(|&row| rows[row].iter().copied())
                    .collect()
            );
            assert_eq!(
                under.0,
                (0..3)
                    .filter(|&row| rows[row].iter().any(|column| free_columns.contains(column)))
                    .collect()
            );
            assert!(over.0.is_disjoint(&under.0));
            assert!(over.1.is_disjoint(&under.1));
            for (row, column) in matched.iter().enumerate() {
                if let Some(column) = column {
                    assert!(rows[row].contains(column));
                    assert!(!matched[..row].contains(&Some(*column)));
                }
            }
        }
    }
}
