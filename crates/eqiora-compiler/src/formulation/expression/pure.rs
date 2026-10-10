//! Structural projection of already checked scalar calculus, without differentiation.
use super::*;
use eqiora_schema::kernel::pure_operator::{CalculusNode, CalculusNodeId, PureOperatorDefinition};

pub(super) fn project(
    dag: &ExprDag,
    application: &eqiora_schema::kernel::PureOperatorApplication,
    remaining: &mut usize,
    depth: usize,
) -> Result<AuthoredFormExpressionV1, ProjectionFailure> {
    let definition = dag
        .definition(application.definition())
        .ok_or_else(|| rejection("missing pure definition"))?;
    if !definition.result_rule().is_invariant_scalar()
        || definition
            .formals()
            .iter()
            .any(|formal| !formal.is_invariant_scalar())
    {
        return Err(ProjectionFailure::Unsupported);
    }
    *remaining = remaining
        .checked_sub(definition.nodes().len())
        .ok_or(ProjectionFailure::Unsupported)?;
    for node in definition.nodes() {
        if !matches!(
            node,
            CalculusNode::Rational { .. }
                | CalculusNode::FormalComponent { .. }
                | CalculusNode::BoundInput(_)
                | CalculusNode::Differentiated { .. }
                | CalculusNode::Neg(_)
                | CalculusNode::Add(..)
                | CalculusNode::Mul(..)
        ) {
            return Err(ProjectionFailure::Unsupported);
        }
    }
    // Validate every argument before projecting the result, including canceled inputs.
    for argument in application.arguments() {
        from_dag(dag, *argument, remaining, depth + 1)?;
    }
    project_node(
        dag,
        definition,
        application.arguments(),
        definition.root(),
        remaining,
        depth,
    )
}

fn project_node(
    dag: &ExprDag,
    definition: &PureOperatorDefinition,
    arguments: &[ExprId],
    id: CalculusNodeId,
    remaining: &mut usize,
    depth: usize,
) -> Result<AuthoredFormExpressionV1, ProjectionFailure> {
    use AuthoredFormExpressionV1 as E;
    if *remaining == 0 || depth > 128 {
        return Err(ProjectionFailure::Unsupported);
    }
    *remaining -= 1;
    let mut child =
        |id| project_node(dag, definition, arguments, id, remaining, depth + 1).map(Box::new);
    Ok(match &definition.nodes()[id.index() as usize] {
        CalculusNode::FormalComponent { formal, axes } if axes.is_empty() => {
            from_dag(dag, arguments[usize::from(*formal)], remaining, depth + 1)?
        }
        CalculusNode::Rational { value, dimension } => E::Rational {
            numerator: value.numerator(),
            denominator: value.denominator(),
            dimension: dimension.exponents(),
        },
        // Canonical definition validation has already checked the retained derivative.
        CalculusNode::BoundInput(value) | CalculusNode::Differentiated { value, .. } => {
            *child(*value)?
        }
        CalculusNode::Neg(value) => E::Neg {
            value: child(*value)?,
        },
        CalculusNode::Add(left, right) => E::Add {
            left: child(*left)?,
            right: child(*right)?,
        },
        CalculusNode::Mul(left, right) => E::Mul {
            left: child(*left)?,
            right: child(*right)?,
        },
        _ => return Err(ProjectionFailure::Unsupported),
    })
}
