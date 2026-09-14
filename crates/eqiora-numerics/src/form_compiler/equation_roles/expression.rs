use std::collections::BTreeSet;

use eqiora_core::{Diagnostic, RawId};
use eqiora_schema::kernel::{ExprDag, ExprId, ExprNode, SymbolRef};

pub(super) fn strip_sign(dag: &ExprDag, mut id: ExprId) -> ExprId {
    while let Some(ExprNode::Neg(inner)) = dag.node(id) {
        id = *inner;
    }
    id
}

pub(super) fn field(dag: &ExprDag, id: ExprId) -> Option<RawId> {
    match dag.node(id)? {
        ExprNode::Symbol(SymbolRef::Field(field)) => Some(field.erase()),
        _ => None,
    }
}

pub(super) fn kinematic(dag: &ExprDag, root: ExprId) -> Option<(RawId, RawId)> {
    let ExprNode::Sub(left, right) = dag.node(strip_sign(dag, root))? else {
        return None;
    };
    [(*left, *right), (*right, *left)]
        .into_iter()
        .find_map(|(left, right)| {
            let ExprNode::Symbol(SymbolRef::Derivative(state)) = dag.node(left)? else {
                return None;
            };
            Some((state.erase(), field(dag, right)?))
        })
}

pub(super) fn coefficient_dependencies(dag: &ExprDag, root: ExprId) -> Option<BTreeSet<RawId>> {
    let mut dependencies = BTreeSet::new();
    let mut pending = vec![root];
    let mut visited = BTreeSet::new();
    while let Some(id) = pending.pop() {
        if !visited.insert(id) {
            continue;
        }
        match dag.node(id)? {
            ExprNode::Symbol(SymbolRef::Field(field)) => {
                dependencies.insert(field.erase());
            }
            ExprNode::Constant(_)
            | ExprNode::Symbol(SymbolRef::Parameter(_))
            | ExprNode::SpatialCoordinate(_) => {}
            ExprNode::Neg(value) | ExprNode::PowI(value, _) | ExprNode::UnaryMath(_, value) => {
                pending.push(*value)
            }
            ExprNode::Add(left, right)
            | ExprNode::Sub(left, right)
            | ExprNode::Mul(left, right)
            | ExprNode::Div(left, right) => pending.extend([*left, *right]),
            _ => return None,
        }
    }
    Some(dependencies)
}

pub(super) fn principal(
    dag: &ExprDag,
    root: ExprId,
) -> Result<(BTreeSet<RawId>, BTreeSet<RawId>), Diagnostic> {
    let mut trials = BTreeSet::new();
    let mut multipliers = BTreeSet::new();
    let mut pending = vec![(root, false)];
    let mut visited = BTreeSet::new();
    while let Some((id, in_divergence)) = pending.pop() {
        if !visited.insert((id, in_divergence)) {
            continue;
        }
        match dag.node(id).expect("validated residual DAG") {
            ExprNode::Symbol(SymbolRef::Derivative(field)) => {
                trials.insert(field.erase());
            }
            ExprNode::Gradient(value) => {
                if let Some(field) = field(dag, *value) {
                    if in_divergence {
                        trials.insert(field);
                    } else {
                        multipliers.insert(field);
                    }
                }
                pending.push((*value, in_divergence));
            }
            ExprNode::IsotropicLift(value) => {
                if in_divergence && let Some(field) = field(dag, *value) {
                    multipliers.insert(field);
                }
                pending.push((*value, in_divergence));
            }
            ExprNode::Divergence(value) => pending.push((*value, true)),
            ExprNode::PureOperatorApplication(application) if in_divergence => {
                let dyadic = eqiora_ir::PureOperatorDefinition::dyadic_product()
                    .expect("canonical closed dyadic definition");
                let arguments = application.arguments();
                let trial = arguments.first().and_then(|argument| field(dag, *argument));
                if application.definition() != dyadic.digest()
                    || arguments.len() != 2
                    || trial.is_none()
                    || arguments.get(1).and_then(|argument| field(dag, *argument)) != trial
                {
                    return Err(Diagnostic::error(
                        eqiora_core::diagnostic::codes::INVALID_REALIZATION,
                        "conservative dyadic role requires its exact definition and identical Field arguments",
                    ));
                }
                trials.insert(trial.expect("checked exact dyadic trial"));
            }
            ExprNode::Neg(value)
            | ExprNode::PowI(value, _)
            | ExprNode::UnaryMath(_, value)
            | ExprNode::SymmetricPart(value) => pending.push((*value, in_divergence)),
            ExprNode::Add(left, right)
            | ExprNode::Sub(left, right)
            | ExprNode::Mul(left, right)
            | ExprNode::Div(left, right) => {
                pending.extend([(*left, in_divergence), (*right, in_divergence)])
            }
            ExprNode::Constant(_)
            | ExprNode::Symbol(SymbolRef::Field(_) | SymbolRef::Parameter(_))
            | ExprNode::SpatialCoordinate(_) => {}
            _ => {
                return Err(Diagnostic::error(
                    eqiora_core::diagnostic::codes::INVALID_REALIZATION,
                    "equation role contains an unsupported operator",
                ));
            }
        }
    }
    Ok((trials, multipliers))
}
