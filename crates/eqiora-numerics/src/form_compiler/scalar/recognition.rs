//! Closed expression recognizers for the scalar primal Galerkin slice.

use eqiora_core::{Diagnostic, RawId};
use eqiora_schema::kernel::{ExprDag, ExprId, ExprNode, SymbolRef};

use super::{BoundaryNodes, VolumeNodes, certificate_error, push_operands};
use crate::form_compiler::vocabulary::WeakSign;

pub(super) fn recognize_volume(
    expression: &ExprDag,
    owner: RawId,
    field: RawId,
) -> Result<VolumeNodes, Diagnostic> {
    let root = expression.roots()[0];
    let (divergence, flux, divergence_sign, source) =
        recognize_volume_top_roles(expression, root).or_else(|| {
            let view = crate::additive_residual::AdditiveResidualView::derive(
                expression, root, owner,
            )
            .ok()?;
            if view.leaves().len() != 2 {
                return None;
            }
            let divergences = view
                .leaves()
                .iter()
                .filter_map(|leaf| match expression.node(leaf.value()) {
                    Some(ExprNode::Divergence(flux)) => Some((leaf, *flux)),
                    _ => None,
                })
                .collect::<Vec<_>>();
            let [(operator, flux)] = divergences.as_slice() else {
                return None;
            };
            let source = view
                .leaves()
                .iter()
                .find(|leaf| leaf.value() != operator.value())?;
            let source_sign = if multiplicative_sign_is_negative(expression, *flux) {
                match operator.sign() {
                    crate::additive_residual::AdditiveSign::Positive => {
                        crate::additive_residual::AdditiveSign::Negative
                    }
                    crate::additive_residual::AdditiveSign::Negative => {
                        crate::additive_residual::AdditiveSign::Positive
                    }
                }
            } else {
                operator.sign()
            };
            if source.sign() != source_sign {
                return None;
            }
            Some((
                operator.value(),
                *flux,
                weak_divergence_sign(operator.sign()),
                source.value(),
            ))
        })
        .ok_or_else(|| {
            crate::additive_residual::AdditiveResidualView::derive(expression, root, owner)
                .map(|view| {
                    view.mismatch(
                        "volume residual requires one diffusion divergence and one equally signed source, up to whole-equation reversal",
                    )
                })
                .unwrap_or_else(|diagnostic| diagnostic)
        })?;
    let gradients = gradient_nodes(expression, flux, field);
    if gradients.len() != 1 {
        return Err(certificate_error(
            owner,
            "constitutive flux must contain exactly one gradient of the unknown",
        ));
    }
    Ok(VolumeNodes {
        root,
        divergence,
        bilinear_flux: flux,
        divergence_sign,
        gradient: gradients[0],
        source,
    })
}

fn recognize_volume_top_roles(
    expression: &ExprDag,
    root: ExprId,
) -> Option<(ExprId, ExprId, WeakSign, ExprId)> {
    let negative_divergence = |value| {
        let ExprNode::Neg(divergence) = expression.node(value)? else {
            return None;
        };
        let ExprNode::Divergence(flux) = expression.node(*divergence)? else {
            return None;
        };
        Some((*divergence, *flux))
    };
    let positive_divergence = |value| {
        let ExprNode::Divergence(flux) = expression.node(value)? else {
            return None;
        };
        Some((value, *flux))
    };
    let zero_source = || {
        expression
            .nodes()
            .iter()
            .enumerate()
            .find_map(|(index, node)| {
                matches!(node, ExprNode::Constant(value) if value.is_zero())
                    .then(|| expression.node_id(u32::try_from(index).ok()?))
                    .flatten()
            })
    };
    match expression.node(root)? {
        ExprNode::Neg(divergence)
            if matches!(expression.node(*divergence), Some(ExprNode::Divergence(_))) =>
        {
            let (divergence, flux) = negative_divergence(root)?;
            Some((divergence, flux, WeakSign::Positive, zero_source()?))
        }
        ExprNode::Divergence(flux) => Some((root, *flux, WeakSign::Negative, zero_source()?)),
        ExprNode::Sub(left, right) => {
            if let Some((divergence, flux)) = negative_divergence(*left) {
                Some((divergence, flux, WeakSign::Positive, *right))
            } else if let Some((divergence, flux)) = positive_divergence(*left) {
                Some((divergence, flux, WeakSign::Negative, *right))
            } else {
                negative_divergence(*right)
                    .map(|(divergence, flux)| (divergence, flux, WeakSign::Negative, *left))
            }
        }
        ExprNode::Add(left, right) => {
            if let (Some((divergence, flux)), Some(ExprNode::Neg(source))) =
                (negative_divergence(*left), expression.node(*right))
            {
                Some((divergence, flux, WeakSign::Positive, *source))
            } else {
                positive_divergence(*left)
                    .map(|(divergence, flux)| (divergence, flux, WeakSign::Negative, *right))
            }
        }
        _ => None,
    }
}

fn multiplicative_sign_is_negative(expression: &ExprDag, value: ExprId) -> bool {
    match expression.node(value) {
        Some(ExprNode::Neg(value)) => !multiplicative_sign_is_negative(expression, *value),
        Some(ExprNode::Mul(left, right)) => {
            multiplicative_sign_is_negative(expression, *left)
                ^ multiplicative_sign_is_negative(expression, *right)
        }
        _ => false,
    }
}

const fn weak_divergence_sign(sign: crate::additive_residual::AdditiveSign) -> WeakSign {
    match sign {
        crate::additive_residual::AdditiveSign::Positive => WeakSign::Negative,
        crate::additive_residual::AdditiveSign::Negative => WeakSign::Positive,
    }
}

fn gradient_nodes(expression: &ExprDag, value: ExprId, field: RawId) -> Vec<ExprId> {
    match expression.node(value) {
        Some(ExprNode::Gradient(argument))
            if matches!(
                expression.node(*argument),
                Some(ExprNode::Symbol(SymbolRef::Field(id))) if id.erase() == field
            ) =>
        {
            vec![value]
        }
        Some(ExprNode::Mul(left, right)) => {
            let mut nodes = gradient_nodes(expression, *left, field);
            nodes.extend(gradient_nodes(expression, *right, field));
            nodes
        }
        _ => Vec::new(),
    }
}

pub(super) fn validate_source_expression(
    expression: &ExprDag,
    source: ExprId,
    owner: RawId,
) -> Result<(), Diagnostic> {
    let mut pending = vec![source];
    let mut reached = vec![false; expression.nodes().len()];
    while let Some(value) = pending.pop() {
        let index = usize::try_from(value.index()).expect("ExprDag indices fit usize");
        if reached[index] {
            continue;
        }
        reached[index] = true;
        let node = expression.node(value).expect("ExprDag owns every operand");
        if !matches!(
            node,
            ExprNode::Constant(_)
                | ExprNode::Symbol(SymbolRef::Parameter(_))
                | ExprNode::Neg(_)
                | ExprNode::Add(_, _)
                | ExprNode::Sub(_, _)
                | ExprNode::Mul(_, _)
                | ExprNode::PowI(_, _)
                | ExprNode::SpatialCoordinate(_)
                | ExprNode::UnaryMath(eqiora_schema::kernel::UnaryMathFunction::Sin, _)
        ) {
            return Err(certificate_error(
                owner,
                "source term is not an unknown-independent scalar spatial expression",
            ));
        }
        push_operands(node, &mut pending);
    }
    Ok(())
}

pub(super) fn recognize_essential_trace(
    expression: &ExprDag,
    owner: RawId,
    field: RawId,
) -> Result<Option<BoundaryNodes>, Diagnostic> {
    if expression.roots().len() != 1 {
        return Err(certificate_error(
            owner,
            "boundary Relation requires one residual root",
        ));
    }
    let root = expression.roots()[0];
    let direct = match expression.node(root) {
        Some(ExprNode::Trace(argument)) => Some((root, *argument, None)),
        Some(ExprNode::Sub(trace, datum)) => {
            if let Some(ExprNode::Trace(argument)) = expression.node(*trace) {
                Some((*trace, *argument, Some(*datum)))
            } else if let Some(ExprNode::Trace(argument)) = expression.node(*datum) {
                Some((*datum, *argument, Some(*trace)))
            } else {
                None
            }
        }
        _ => None,
    };
    let (trace_node, trace, datum) = if let Some(direct) = direct {
        direct
    } else {
        let view = crate::additive_residual::AdditiveResidualView::derive(expression, root, owner)?;
        let traces = view
            .leaves()
            .iter()
            .filter_map(|leaf| match expression.node(leaf.value()) {
                Some(ExprNode::Trace(argument)) => Some((leaf, *argument)),
                _ => None,
            })
            .collect::<Vec<_>>();
        let [(trace_leaf, trace)] = traces.as_slice() else {
            return Ok(None);
        };
        let data = view
            .leaves()
            .iter()
            .filter(|leaf| leaf.value() != trace_leaf.value())
            .collect::<Vec<_>>();
        let datum = match data.as_slice() {
            [] => None,
            [datum] if datum.sign().is_opposite(trace_leaf.sign()) => Some(datum.value()),
            _ => return Ok(None),
        };
        (trace_leaf.value(), *trace, datum)
    };
    if !matches!(
        expression.node(trace),
        Some(ExprNode::Symbol(SymbolRef::Field(id))) if id.erase() == field
    ) {
        return Ok(None);
    }
    if let Some(datum) = datum {
        validate_source_expression(expression, datum, owner).map_err(|_| {
            certificate_error(
                owner,
                "essential trace datum is not an unknown-independent scalar spatial expression",
            )
        })?;
    }
    Ok(Some(BoundaryNodes { trace: trace_node }))
}

pub(super) struct BoundaryFlux {
    pub(super) normal: ExprId,
    pub(super) datum: Option<(ExprId, bool)>,
}

/// Bind the complete constitutive flux before retaining its prescribed datum.
/// The returned sign pairs that datum with the positive weak bilinear flux;
/// neither a whole-equation reversal nor a negated constitutive flux loses sign.
pub(super) fn recognize_flux(
    boundary: &eqiora_schema::kernel::typing::TypedResidual<RawId>,
    owner: RawId,
    volume: &eqiora_schema::kernel::typing::TypedResidual<RawId>,
    volume_nodes: VolumeNodes,
) -> Result<Option<BoundaryFlux>, Diagnostic> {
    use eqiora_compiler::AuthoredFormExpressionV1 as Expression;
    let dag = boundary.expression();
    let [root] = dag.roots() else {
        return Ok(None);
    };
    let view = crate::additive_residual::AdditiveResidualView::derive(dag, *root, owner)?;
    if !(1..=2).contains(&view.leaves().len()) {
        return Ok(None);
    }
    let operators = view
        .leaves()
        .iter()
        .filter_map(|leaf| match dag.node(leaf.value()) {
            Some(ExprNode::NormalComponent(flux)) => Some((leaf, *flux)),
            _ => None,
        })
        .collect::<Vec<_>>();
    let [(operator, flux)] = operators.as_slice() else {
        return Ok(None);
    };
    let volume_flux = volume_nodes.bilinear_flux;
    if boundary.node_type(*flux).map(|t| &t.value_type)
        != volume.node_type(volume_flux).map(|t| &t.value_type)
    {
        return Ok(None);
    }
    let Some(actual) = Expression::from_expression(dag, *flux)? else {
        return Ok(None);
    };
    let Some(expected) = Expression::from_expression(volume.expression(), volume_flux)? else {
        return Ok(None);
    };
    let (actual, actual_negative) = super::authored::product_sign(actual);
    let (expected, volume_negative) = super::authored::product_sign(expected);
    if !super::authored::equivalent(&actual, &expected) {
        return Ok(None);
    }
    let data = view
        .leaves()
        .iter()
        .find(|leaf| leaf.value() != operator.value())
        .map(|datum| {
            validate_source_expression(dag, datum.value(), owner)?;
            // a*n(F)+b*g=0 => n(F)=-(b/a)g. Convert F to the weak flux.
            let negative = (datum.sign() == operator.sign())
                ^ actual_negative
                ^ volume_negative
                ^ (volume_nodes.divergence_sign == WeakSign::Negative);
            Ok((datum.value(), negative))
        })
        .transpose()?;
    Ok(Some(BoundaryFlux {
        normal: operator.value(),
        datum: data,
    }))
}
