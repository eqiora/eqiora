//! Regular equations alone must locally determine an admitted continuous DAE.
use super::*;
mod affine;
mod compatibility;
use eqiora_time::ConstantDerivativeMatrixProof;

pub(super) fn validate<'a>(
    program: &'a KernelProgram,
    plan: &'a ExecutionPlan,
    context: &EvalContext<'_>,
    settings: solver::NonlinearSettings,
) -> Result<(), Diagnostic> {
    if plan.algebraic_fields.is_empty()
        && plan.continuous_ports.is_empty()
        && plan.physical_systems.is_empty()
        && affine::regular_ode(
            program,
            &plan.differential_orders,
            &plan.continuous_relations,
        )?
    {
        // Full-rank constant mass determines every initial rate; there is no
        // algebraic constraint whose tangent would require additional rates.
        return Ok(());
    }
    let values = plan
        .differential_orders
        .keys()
        .chain(&plan.algebraic_fields)
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(Variable::Field)
        .chain(plan.continuous_ports.iter().copied().map(Variable::Port))
        .chain(
            plan.physical_unknowns
                .iter()
                .copied()
                .map(Variable::Physical),
        )
        .collect::<Vec<_>>();
    let n = values.len();
    if n == 0 {
        return Ok(());
    }
    let rates = values
        .iter()
        .enumerate()
        .filter_map(|(column, value)| match value {
            Variable::Field(field) => Some((
                column,
                SymbolRef::Derivative(
                    field.downcast().expect("Field"),
                    plan.differential_orders
                        .get(field)
                        .copied()
                        .unwrap_or(std::num::NonZeroU32::MIN),
                ),
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    let variables = values
        .iter()
        .copied()
        .chain(rates.iter().map(|(_, symbol)| match symbol {
            SymbolRef::Derivative(field, order) => Variable::Derivative(field.erase(), *order),
            _ => unreachable!(),
        }))
        .collect::<Vec<_>>();
    let columns = values
        .iter()
        .copied()
        .enumerate()
        .map(|(column, value)| (value, column))
        .collect::<BTreeMap<_, _>>();
    let mut equations = Vec::new();
    let mut operators = Vec::new();
    let mut append = |owner,
                      expression: &'a ExprDag,
                      roots: &[ExprId],
                      paired: bool|
     -> Result<(), Diagnostic> {
        let operator = point_operator(program, owner, expression, roots, &variables, context)?;
        let rows = differentiate(&operator, &variables, context, roots.len())?;
        let width = if paired { 2 } else { 1 };
        for (ordinal, sides) in roots.chunks_exact(width).enumerate() {
            let root = ordinal * width;
            let jacobian = if paired {
                rows[root]
                    .iter()
                    .zip(&rows[root + 1])
                    .map(|(left, right)| left - right)
                    .collect()
            } else {
                rows[root].clone()
            };
            equations.push(Equation {
                owner,
                expression,
                sides: sides.to_vec(),
                operator: operators.len(),
                root,
                paired,
                jacobian,
                incidence: structural::incidence::variables(
                    expression,
                    sides,
                    &plan.signal_sources,
                )?
                .into_iter()
                .map(|variable| match variable {
                    Variable::Derivative(field, _) => Variable::Field(field),
                    other => other,
                })
                .filter_map(|variable| columns.get(&variable).copied())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect(),
            });
        }
        operators.push(operator);
        Ok(())
    };
    for &owner in &plan.continuous_relations {
        let Some(KernelNode::Relation(relation)) = program.node(owner) else {
            return Err(execution_error("continuous Relation is unavailable", 0.0));
        };
        append(
            owner,
            relation.expression(),
            &direct_assignments::numerical_roots(program, relation),
            true,
        )?;
    }
    for system in &plan.physical_systems {
        for junction in system.junctions() {
            append(
                junction.owner().erase(),
                junction.dag(),
                junction.dag().roots(),
                false,
            )?;
        }
    }
    for (block_rows, block_columns) in connected_blocks(&equations, n) {
        check_block(
            &equations,
            &operators,
            &values,
            &rates,
            &block_rows,
            &block_columns,
            plan,
        )
        .and_then(|()| {
            compatibility::validate(
                &equations,
                &values,
                &rates,
                &block_rows,
                &block_columns,
                context,
                settings,
            )
        })
        .map_err(|error| {
            Diagnostic::error(codes::NONLINEAR_SOLVE_FAILED, error.message())
                .with_graph_path(kernel_path(equations[block_rows[0]].owner))
        })?;
    }
    Ok(())
}

struct Equation<'a> {
    expression: &'a ExprDag,
    sides: Vec<ExprId>,
    owner: RawId,
    operator: usize,
    root: usize,
    paired: bool,
    jacobian: Vec<f64>,
    incidence: Vec<usize>,
}

fn connected_blocks(equations: &[Equation<'_>], count: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
    let mut seen = BTreeSet::new();
    let mut blocks = Vec::new();
    for start in 0..count {
        if seen.contains(&start) {
            continue;
        }
        let mut columns = BTreeSet::from([start]);
        let mut rows = BTreeSet::new();
        let mut pending = vec![start];
        while let Some(column) = pending.pop() {
            seen.insert(column);
            for (row, equation) in equations.iter().enumerate() {
                if equation.incidence.contains(&column) && rows.insert(row) {
                    for &neighbor in &equation.incidence {
                        if columns.insert(neighbor) {
                            pending.push(neighbor);
                        }
                    }
                }
            }
        }
        blocks.push((rows.into_iter().collect(), columns.into_iter().collect()));
    }
    blocks
}

fn check_block(
    equations: &[Equation<'_>],
    operators: &[ScalarOperatorIr],
    values: &[Variable],
    rates: &[(usize, SymbolRef)],
    rows: &[usize],
    columns: &[usize],
    plan: &ExecutionPlan,
) -> Result<(), Diagnostic> {
    let n = columns.len();
    if rows.len() != n {
        return Err(execution_error(
            "continuous regularity requires square incidence blocks",
            0.0,
        ));
    }
    let selected_rates = rates
        .iter()
        .filter(|(column, _)| columns.contains(column))
        .copied()
        .collect::<Vec<_>>();
    let symbols = selected_rates
        .iter()
        .map(|(_, symbol)| *symbol)
        .collect::<Vec<_>>();
    let mut proofs = BTreeMap::new();
    let mut constant = true;
    for &row in rows {
        let operator = equations[row].operator;
        if let std::collections::btree_map::Entry::Vacant(entry) = proofs.entry(operator) {
            match operators[operator].constant_symbol_jacobian(&symbols) {
                Ok(proof) => {
                    entry.insert(proof);
                }
                Err(_) => {
                    constant = false;
                    break;
                }
            }
        }
    }
    if constant {
        let mut mass = vec![0.0; n * n];
        let mut state = Vec::with_capacity(n * n);
        for (local_row, &row) in rows.iter().enumerate() {
            let equation = &equations[row];
            let proof = &proofs[&equation.operator];
            for (rate, (column, _)) in selected_rates.iter().enumerate() {
                let local_column = columns
                    .iter()
                    .position(|candidate| candidate == column)
                    .expect("block rate");
                let coefficient = |root| proof.coefficients()[root * selected_rates.len() + rate];
                mass[local_row * n + local_column] = coefficient(equation.root)
                    - if equation.paired {
                        coefficient(equation.root + 1)
                    } else {
                        0.0
                    };
            }
            state.extend(columns.iter().map(|&column| equation.jacobian[column]));
        }
        let proof = ConstantDerivativeMatrixProof::new(n, mass)?;
        if proof.exact_rank() < n
            && columns.iter().any(|&column| match values[column] {
                Variable::Field(field) => plan
                    .differential_orders
                    .get(&field)
                    .is_some_and(|order| order.get() > 1),
                _ => false,
            })
        {
            return Err(execution_error(
                "rank-deficient higher-order block requires an explicit index-reduction formulation",
                0.0,
            ));
        }
        proof.require_index_one_regularity(&state)
    } else {
        // Only this connected equation block falls back to its authored
        // rate/algebraic partition; unrelated descriptors keep their proof.
        let selected = rows
            .iter()
            .flat_map(|&row| {
                columns.iter().map(move |&column| match values[column] {
                    Variable::Field(field) if plan.differential_orders.contains_key(&field) => {
                        let rate = rates
                            .iter()
                            .position(|(value_column, _)| *value_column == column)
                            .expect("Field rate");
                        equations[row].jacobian[values.len() + rate]
                    }
                    _ => equations[row].jacobian[column],
                })
            })
            .collect::<Vec<_>>();
        let rank = ConstantDerivativeMatrixProof::new(n, selected)?.exact_rank();
        if rank != n {
            return Err(Diagnostic::error(
                codes::NONLINEAR_SOLVE_FAILED,
                format!(
                    "continuous equations have an unsupported high-index or singular differential/algebraic partition: local regularity rank {rank}, required {n}"
                ),
            ));
        }
        Ok(())
    }
}
