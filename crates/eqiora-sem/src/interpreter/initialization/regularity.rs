//! Accepted-point initial Jacobian through the existing scalar Operator IR AD.
use super::*;
use eqiora_ir::{DifferentiationRole, LinearizedRelation, RelationTangent, ScalarOperatorIr};
use eqiora_schema::kernel::typing::RootContract;
use eqiora_schema::kernel::{ExprDag, ExprDagBuilder, ExprId};

pub(super) fn validate(
    program: &KernelProgram,
    plan: &ExecutionPlan,
    state: &RuntimeState,
    variables: &[Variable],
    solution: &[f64],
    relations: &BTreeSet<RawId>,
    tangents: &[tangent::Tangent],
) -> Result<(), Diagnostic> {
    if variables.is_empty() {
        return Ok(());
    }
    let candidates = candidate_maps(variables, solution, state);
    let mut initial_state = state.clone();
    initial_state.fields.clone_from(&candidates.fields);
    let context = EvalContext {
        program,
        time: 0.0,
        typed_fields: &initial_state.typed_fields,
        typed_ports: &initial_state.typed_ports,
        typed_next: &initial_state.typed_next,
        fields: &initial_state.fields,
        field_candidates: &candidates.fields,
        derivatives: &candidates.derivatives,
        next_fields: &candidates.next_fields,
        ports: &initial_state.ports,
        port_candidates: &candidates.ports,
        signal_sources: &plan.signal_sources,
        physical: &initial_state.physical,
        physical_candidates: &candidates.physical,
    };
    let mut matrix = Vec::new();
    for &owner in relations {
        let Some(KernelNode::Relation(relation)) = program.node(owner) else {
            return Err(execution_error(
                "validated initial Relation is unavailable",
                0.0,
            ));
        };
        let roots = direct_assignments::numerical_roots(program, relation);
        if roots.is_empty() {
            continue;
        }
        let rows = jacobian(
            program,
            owner,
            relation.expression(),
            &roots,
            variables,
            &context,
        )?;
        matrix.extend(rows.as_chunks::<2>().0.iter().map(|pair| {
            pair[0]
                .iter()
                .zip(&pair[1])
                .map(|(left, right)| left - right)
                .collect()
        }));
    }
    for system in &plan.physical_systems {
        for junction in system.junctions() {
            matrix.extend(jacobian(
                program,
                junction.connection().erase(),
                junction.dag(),
                junction.dag().roots(),
                variables,
                &context,
            )?);
        }
    }
    for tangent in tangents {
        matrix.push(
            variables
                .iter()
                .map(|variable| match variable {
                    Variable::Derivative(field) => tangent
                        .coefficients
                        .iter()
                        .find_map(|(id, value)| (id == field).then_some(*value))
                        .unwrap_or(0.0),
                    _ => 0.0,
                })
                .collect(),
        );
    }
    if matrix.len() != variables.len() {
        return Err(execution_error(
            "initial Jacobian differs from the admitted square solve",
            0.0,
        ));
    }
    solver::solve_linear(matrix, vec![0.0; variables.len()]).ok_or_else(|| {
        Diagnostic::error(codes::NONLINEAR_SOLVE_FAILED,
            "initial Jacobian is singular at the accepted point (Operator IR automatic differentiation)")
            .with_graph_path(execution_path("initialization", 0.0))
    })?;
    Ok(())
}

fn coordinate(
    symbol: SymbolRef,
    variables: &[Variable],
    context: &EvalContext<'_>,
) -> Option<usize> {
    let variable = match symbol {
        SymbolRef::Field(id) | SymbolRef::Pre(id) => Variable::Field(id.erase()),
        SymbolRef::Derivative(id) => Variable::Derivative(id.erase()),
        SymbolRef::Next(id) => Variable::NextField(id.erase()),
        SymbolRef::Port(id) => Variable::Port(
            context
                .signal_sources
                .get(&id.erase())
                .copied()
                .unwrap_or(id.erase()),
        ),
        SymbolRef::Across(id) => Variable::Physical(PhysicalUnknown::Across(id)),
        SymbolRef::Through(id) => Variable::Physical(PhysicalUnknown::Through(id)),
        _ => return None,
    };
    variables
        .iter()
        .position(|candidate| *candidate == variable)
}

fn jacobian(
    program: &KernelProgram,
    owner: RawId,
    expression: &ExprDag,
    roots: &[ExprId],
    variables: &[Variable],
    context: &EvalContext<'_>,
) -> Result<Vec<Vec<f64>>, Diagnostic> {
    // Bind only frozen inputs. This is a point projection, never a mutation of
    // authored equations, identities, properties, or the accepted Model.
    let mut builder = ExprDagBuilder::new();
    for node in expression.nodes() {
        match node {
            ExprNode::Symbol(symbol) if coordinate(*symbol, variables, context).is_none() => {
                let value = evaluate::resolve_symbol(*symbol, context).ok_or_else(|| {
                    execution_error("initial Jacobian has an unavailable frozen input", 0.0)
                })?;
                builder.constant(value)?;
            }
            ExprNode::PureOperatorApplication(application) => {
                let definition = expression
                    .definition(application.definition())
                    .expect("validated pure operator definition");
                builder.pure_operator(definition, application.arguments().iter().copied())?;
            }
            _ => {
                builder.push(node.clone())?;
            }
        }
    }
    for (&id, release) in expression.properties() {
        builder.bind_property(id, release.clone())?;
    }
    let typed = program
        .type_derived_residual(
            builder.finish(roots.iter().copied())?,
            owner,
            None,
            if owner.kind() == eqiora_core::EntityKind::Relation {
                RootContract::InitialConditions
            } else {
                RootContract::ComponentwiseResidual
            },
        )
        .map_err(|errors| errors.into_iter().next().expect("typing failure"))?;
    let operator = ScalarOperatorIr::lower_typed_scalar(&typed)?;
    let inputs = operator
        .symbols()
        .iter()
        .map(|&symbol| {
            evaluate::resolve_symbol(symbol, context).ok_or_else(|| {
                execution_error("initial Jacobian has an unavailable active input", 0.0)
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let columns = operator
        .symbols()
        .iter()
        .map(|&symbol| {
            coordinate(symbol, variables, context)
                .expect("unbound symbols are active initial coordinates")
        })
        .collect::<Vec<_>>();
    let linearization =
        operator.linearize_typed(&inputs, &vec![DifferentiationRole::Unknown; inputs.len()])?;
    let mut rows = vec![vec![0.0; variables.len()]; roots.len()];
    for column in 0..variables.len() {
        let direction = columns
            .iter()
            .map(|&mapped| if mapped == column { 1.0 } else { 0.0 })
            .collect::<Vec<_>>();
        let mut output = vec![0.0; roots.len()];
        linearization.jvp(RelationTangent::Unknown(&direction), &mut output)?;
        for (row, value) in rows.iter_mut().zip(output) {
            row[column] = value;
        }
    }
    Ok(rows)
}
