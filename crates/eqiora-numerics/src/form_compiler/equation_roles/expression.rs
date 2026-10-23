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
            let ExprNode::Symbol(SymbolRef::Derivative(state, std::num::NonZeroU32::MIN)) =
                dag.node(left)?
            else {
                return None;
            };
            Some((state.erase(), field(dag, right)?))
        })
}

pub(in crate::form_compiler) fn coefficient_dependencies(
    dag: &ExprDag,
    root: ExprId,
) -> Option<BTreeSet<RawId>> {
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
            | ExprNode::Symbol(SymbolRef::Coordinate { .. })
            | ExprNode::Symbol(SymbolRef::Time) => {}
            ExprNode::Neg(value)
            | ExprNode::PowI(value, _)
            | ExprNode::UnaryMath(_, value)
            | ExprNode::FiniteUnary(_, value) => pending.push(*value),
            ExprNode::Complex {
                real: left,
                imag: right,
            }
            | ExprNode::Add(left, right)
            | ExprNode::Sub(left, right)
            | ExprNode::Mul(left, right)
            | ExprNode::Div(left, right)
            | ExprNode::FiniteBinary(_, left, right) => pending.extend([*left, *right]),
            ExprNode::CoordinateMapFactor { at, .. } => {
                pending.extend(at.iter().map(|(_, mapped)| mapped));
            }
            ExprNode::Pullback { value, at, .. } => {
                pending.push(*value);
                pending.extend(at.iter().map(|(_, mapped)| mapped));
            }
            ExprNode::PureOperatorApplication(application) => {
                pending.extend(application.arguments())
            }
            node @ (ExprNode::Select { .. }
            | ExprNode::Require { .. }
            | ExprNode::Compare(..)
            | ExprNode::Not(_)
            | ExprNode::And(..)
            | ExprNode::Or(..)) => {
                node.try_for_each_operand(|operand| {
                    pending.push(operand);
                    Ok::<_, std::convert::Infallible>(())
                })
                .expect("infallible operand visitor");
            }
            ExprNode::Array { elements } => pending.extend(elements),
            ExprNode::Index { value, .. } => pending.push(*value),
            _ => return None,
        }
    }
    Some(dependencies)
}

pub(super) fn principal(
    dag: &ExprDag,
    root: ExprId,
    coefficients: &BTreeSet<RawId>,
) -> Result<(BTreeSet<RawId>, BTreeSet<RawId>), Diagnostic> {
    let mut trials = BTreeSet::new();
    let mut multipliers = BTreeSet::new();
    let mut flux_inputs = BTreeSet::new();
    let mut pending = vec![(root, false)];
    let mut visited = BTreeSet::new();
    while let Some((id, in_divergence)) = pending.pop() {
        if !visited.insert((id, in_divergence)) {
            continue;
        }
        match dag.node(id).expect("validated residual DAG") {
            ExprNode::Symbol(SymbolRef::Derivative(field, std::num::NonZeroU32::MIN)) => {
                trials.insert(field.erase());
            }
            ExprNode::Gradient(value) => {
                // grad(div(u)) owns the vector principal trial just as
                // div(grad(u)) does; a bare grad(p) remains a multiplier.
                if let Some(ExprNode::Divergence(inner)) = dag.node(*value)
                    && let Some(field) = field(dag, *inner)
                {
                    trials.insert(field);
                }
                if let Some(fields) = coefficient_dependencies(dag, *value) {
                    // A pointwise composition inside a gradient retains its
                    // unknown operands' differential role. Its affinity and
                    // exact component action belong to the shared lowerer.
                    let unknowns = fields.difference(coefficients).copied();
                    if in_divergence {
                        trials.extend(unknowns);
                    } else {
                        multipliers.extend(unknowns);
                    }
                } else {
                    pending.push((*value, in_divergence));
                }
            }
            ExprNode::IsotropicLift(value) => {
                if in_divergence && let Some(field) = field(dag, *value) {
                    multipliers.insert(field);
                }
                pending.push((*value, in_divergence));
            }
            ExprNode::Divergence(value) => pending.push((*value, true)),
            ExprNode::PureOperatorApplication(_)
                if super::super::vector_curl::curl_curl_field(dag, id).is_some() =>
            {
                trials.insert(
                    super::super::vector_curl::curl_curl_field(dag, id)
                        .expect("checked vector curl-curl Field"),
                );
            }
            ExprNode::PureOperatorApplication(_)
                if super::super::planar_curl::gradient(dag, id).is_some() =>
            {
                let gradient = super::super::planar_curl::gradient(dag, id)
                    .expect("checked planar composition");
                pending.push((gradient, true));
            }
            ExprNode::PureOperatorApplication(_)
                if coefficient_dependencies(dag, id)
                    .is_some_and(|dependencies| dependencies.is_subset(coefficients)) =>
            {
                // Prescribed pointwise calculus is coefficient data, not a
                // nonlinear dyadic trial. Scalar/type admission remains with
                // the coefficient lowerer; unknown Field inputs cannot enter.
            }
            ExprNode::PureOperatorApplication(application) => {
                // Roles follow exact operands. Affinity and component actions
                // remain the responsibility of the numerical operator lowerer.
                pending.extend(
                    application
                        .arguments()
                        .iter()
                        .map(|id| (*id, in_divergence)),
                );
            }
            ExprNode::CoordinateMapFactor { at, .. } => {
                pending.extend(at.iter().map(|(_, mapped)| (*mapped, in_divergence)));
            }
            ExprNode::Pullback { value, at, .. } => {
                pending.push((*value, in_divergence));
                pending.extend(at.iter().map(|(_, mapped)| (*mapped, in_divergence)));
            }
            node @ (ExprNode::Select { .. }
            | ExprNode::Require { .. }
            | ExprNode::Compare(..)
            | ExprNode::Not(_)
            | ExprNode::And(..)
            | ExprNode::Or(..)) => {
                // Roles conservatively include every branch and predicate.
                // Lazy evaluation and admissible derivatives belong downstream.
                node.try_for_each_operand(|operand| {
                    pending.push((operand, in_divergence));
                    Ok::<_, std::convert::Infallible>(())
                })
                .expect("infallible operand visitor");
            }
            ExprNode::Array { elements } => {
                pending.extend(elements.iter().map(|id| (*id, in_divergence)));
            }
            ExprNode::Index { value, .. } => pending.push((*value, in_divergence)),
            ExprNode::Neg(value)
            | ExprNode::PowI(value, _)
            | ExprNode::UnaryMath(_, value)
            | ExprNode::SymmetricPart(value)
            | ExprNode::FiniteUnary(_, value) => pending.push((*value, in_divergence)),
            ExprNode::Complex {
                real: left,
                imag: right,
            }
            | ExprNode::Add(left, right)
            | ExprNode::Sub(left, right)
            | ExprNode::Mul(left, right)
            | ExprNode::Div(left, right)
            | ExprNode::FiniteBinary(_, left, right) => {
                pending.extend([(*left, in_divergence), (*right, in_divergence)])
            }
            ExprNode::Symbol(SymbolRef::Field(field)) if in_divergence => {
                if !coefficients.contains(&field.erase()) {
                    flux_inputs.insert(field.erase());
                }
            }
            ExprNode::Constant(_)
            | ExprNode::Symbol(SymbolRef::Field(_) | SymbolRef::Parameter(_))
            | ExprNode::Symbol(SymbolRef::Coordinate { .. })
            | ExprNode::Symbol(SymbolRef::Time) => {}
            _ => {
                return Err(Diagnostic::error(
                    eqiora_core::diagnostic::codes::INVALID_REALIZATION,
                    "equation role contains an unsupported operator",
                ));
            }
        }
    }
    // Pointwise flux inputs anchor a conservation equation only when no
    // gradient, time derivative or multiplier already identifies its role.
    // Constitutive coefficients must not become extra principal trials.
    if trials.is_empty() && multipliers.is_empty() {
        trials.extend(flux_inputs);
    }
    if trials.is_empty()
        && multipliers.is_empty()
        && let Some(fields) = coefficient_dependencies(dag, root)
    {
        // A pointwise equation needs no differential anchor. Its complete
        // remaining Field inventory must still select one unambiguous trial;
        // affinity, exact component type and Space admission belong downstream.
        trials.extend(fields.difference(coefficients).copied());
    }
    Ok((trials, multipliers))
}
