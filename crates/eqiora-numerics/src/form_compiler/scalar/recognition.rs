//! Closed expression recognizers for the scalar primal Galerkin slice.

use eqiora_core::{Diagnostic, RawId};
use eqiora_schema::kernel::{ExprDag, ExprId, ExprNode, SymbolRef};

use super::{BoundaryNodes, VolumeNodes, certificate_error, push_operands};
use crate::form_compiler::vocabulary::{DiffusionRule, WeakSign};

pub(super) fn recognize_volume(
    expression: &ExprDag,
    owner: RawId,
    field: RawId,
    complex_trial: bool,
    dimension: usize,
) -> Result<VolumeNodes, Diagnostic> {
    use crate::additive_residual::{AdditiveResidualView, AdditiveSign};
    use crate::form_compiler::vocabulary::PrimalValueTerm;
    let root = expression.roots()[0];
    // Keep independent forcing expressions intact: splitting their integrals
    // would require additional algebra to replay non-polynomial authored forms.
    let view = AdditiveResidualView::derive_preserving(expression, root, owner, &|value| {
        value_degree(expression, value, field, owner, false, complex_trial)
            .is_ok_and(|degree| degree == 0)
    })?;
    let operators = view
        .leaves()
        .iter()
        .filter_map(|leaf| {
            if let Some(ExprNode::Divergence(flux)) = expression.node(leaf.value()) {
                Some((leaf, *flux, DiffusionRule::Divergence))
            } else if dimension == 2 {
                super::super::planar_curl::gradient(expression, leaf.value())
                    .map(|gradient| (leaf, gradient, DiffusionRule::PlanarScalarCurlCurl))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let [(operator, flux, diffusion_rule)] = operators.as_slice() else {
        return Err(
            view.mismatch("volume residual requires exactly one admitted diffusion operator")
        );
    };
    let gradients = gradient_nodes(expression, *flux, field);
    if gradients.len() != 1 {
        return Err(certificate_error(
            owner,
            "constitutive flux must contain exactly one gradient of the unknown",
        ));
    }
    if value_degree(expression, *flux, field, owner, true, complex_trial)? != 1 {
        return Err(certificate_error(
            owner,
            "diffusion flux requires linear trial dependence",
        ));
    }
    let mut values = Vec::new();
    for leaf in view.leaves() {
        if leaf.value() == operator.value() {
            continue;
        }
        let degree = value_degree(expression, leaf.value(), field, owner, false, complex_trial)?;
        if degree > 1 {
            return Err(certificate_error(
                owner,
                "value pairing requires linear trial dependence",
            ));
        }
        let trial_dependent = degree == 1;
        // Integration by parts changes only the divergence sign. Loads move
        // to the right side; trial-dependent values remain on the left.
        let positive = (leaf.sign() == AdditiveSign::Positive) == trial_dependent;
        values.push(PrimalValueTerm {
            source_node: leaf.value(),
            trial_dependent,
            sign: if positive {
                WeakSign::Positive
            } else {
                WeakSign::Negative
            },
        });
    }
    // Preserve an explicit typed zero source occurrence when residual projection
    // has elided it. It contributes no mathematical term.
    if !values.iter().any(|term| !term.trial_dependent)
        && let Some(zero) = expression
            .nodes()
            .iter()
            .enumerate()
            .find_map(|(index, node)| {
                matches!(node, ExprNode::Constant(value) if value.is_zero())
                    .then(|| expression.node_id(u32::try_from(index).ok()?))
                    .flatten()
            })
    {
        values.push(PrimalValueTerm {
            source_node: zero,
            trial_dependent: false,
            sign: WeakSign::Positive,
        });
    }
    Ok(VolumeNodes {
        root,
        divergence: operator.value(),
        diffusion_rule: *diffusion_rule,
        bilinear_flux: *flux,
        divergence_sign: if (operator.sign() == AdditiveSign::Negative)
            == (*diffusion_rule == DiffusionRule::Divergence)
        {
            WeakSign::Positive
        } else {
            WeakSign::Negative
        },
        gradient: gradients[0],
        values,
    })
}

/// Classify local values without evaluating coefficients or sampling the trial.
/// The memoized postorder traversal visits each source node at most once.
pub(in crate::form_compiler) fn value_degree(
    dag: &ExprDag,
    root: ExprId,
    field: RawId,
    owner: RawId,
    allow_gradient: bool,
    complex_trial: bool,
) -> Result<u8, Diagnostic> {
    let mut degrees = vec![None; dag.nodes().len()];
    let mut pending = vec![(root, false)];
    while let Some((id, ready)) = pending.pop() {
        let index = id.index() as usize;
        if degrees[index].is_some() {
            continue;
        }
        let node = dag.node(id).expect("validated source DAG");
        if !ready {
            pending.push((id, true));
            let mut operands = Vec::new();
            push_operands(node, &mut operands);
            pending.extend(operands.into_iter().map(|operand| (operand, false)));
            continue;
        }
        let degree = |id: ExprId| degrees[id.index() as usize].expect("postorder operand");
        let result = match node {
            ExprNode::Constant(_)
            | ExprNode::Symbol(SymbolRef::Parameter(_))
            | ExprNode::Symbol(SymbolRef::Coordinate { .. }) => 0,
            ExprNode::Symbol(SymbolRef::Field(id)) if id.erase() == field => 1,
            ExprNode::Gradient(value) if allow_gradient => degree(*value),
            ExprNode::Neg(value) => degree(*value),
            ExprNode::Add(a, b) | ExprNode::Sub(a, b) | ExprNode::Complex { real: a, imag: b } => {
                let (a, b) = (degree(*a), degree(*b));
                // An opaque affine sum cannot be recorded as a homogeneous
                // bilinear term. Top-level sums are separated by the residual view.
                if a == b { a } else { 2 }
            }
            ExprNode::Mul(a, b) => (degree(*a) + degree(*b)).min(2),
            ExprNode::PowI(value, power) => match (degree(*value), *power) {
                (0, _) | (_, 0) => 0,
                (value, 1) => value,
                _ => 2,
            },
            ExprNode::UnaryMath(eqiora_schema::kernel::UnaryMathFunction::Conj, value)
                if !complex_trial =>
            {
                degree(*value)
            }
            ExprNode::UnaryMath(_, value) if degree(*value) == 0 => 0,
            _ => {
                return Err(certificate_error(
                    owner,
                    "value pairing contains an unsupported or antilinear trial operation",
                ));
            }
        };
        degrees[index] = Some(result);
    }
    Ok(degrees[root.index() as usize].expect("root degree"))
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
        Some(
            ExprNode::Neg(value)
            | ExprNode::UnaryMath(eqiora_schema::kernel::UnaryMathFunction::Conj, value),
        ) => gradient_nodes(expression, *value, field),
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
                | ExprNode::Complex { .. }
                | ExprNode::Symbol(SymbolRef::Parameter(_))
                | ExprNode::Neg(_)
                | ExprNode::Add(_, _)
                | ExprNode::Sub(_, _)
                | ExprNode::Mul(_, _)
                | ExprNode::PowI(_, _)
                | ExprNode::Symbol(SymbolRef::Coordinate { .. })
                | ExprNode::UnaryMath(
                    eqiora_schema::kernel::UnaryMathFunction::Sin
                        | eqiora_schema::kernel::UnaryMathFunction::Conj,
                    _
                )
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
        Some(ExprNode::Trace {
            value: argument, ..
        }) => Some((root, *argument, None)),
        Some(ExprNode::Sub(trace, datum)) => {
            if let Some(ExprNode::Trace {
                value: argument, ..
            }) = expression.node(*trace)
            {
                Some((*trace, *argument, Some(*datum)))
            } else if let Some(ExprNode::Trace {
                value: argument, ..
            }) = expression.node(*datum)
            {
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
                Some(ExprNode::Trace {
                    value: argument, ..
                }) => Some((leaf, *argument)),
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
    boundary_domain: RawId,
    parent: RawId,
    volume: &eqiora_schema::kernel::typing::TypedResidual<RawId>,
    volume_nodes: &VolumeNodes,
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
            Some(ExprNode::NormalComponent { value: flux, .. }) => Some((leaf, *flux)),
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
    let Some(mut actual) = Expression::from_expression(dag, *flux)? else {
        return Ok(None);
    };
    let Some(expected) = Expression::from_expression(volume.expression(), volume_flux)? else {
        return Ok(None);
    };
    // boundary_inventory has authenticated BoundaryOf(boundary_domain, parent).
    // Compare the restriction in that parent's chart without discarding factor identity.
    parent_coordinates(
        &mut actual,
        &boundary_domain.ulid().to_string(),
        &parent.ulid().to_string(),
    );
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

fn parent_coordinates(
    value: &mut eqiora_compiler::AuthoredFormExpressionV1,
    boundary: &str,
    parent: &str,
) {
    use eqiora_compiler::AuthoredFormExpressionV1 as E;
    match value {
        E::Coordinate {
            support_ulid,
            factor_ulid,
            ..
        } if support_ulid == boundary && factor_ulid == parent => {
            *support_ulid = parent.to_owned();
        }
        E::Neg { value } | E::Gradient { value } | E::Sin { value } => {
            parent_coordinates(value, boundary, parent)
        }
        E::Add { left, right }
        | E::Sub { left, right }
        | E::Mul { left, right }
        | E::Div { left, right } => {
            parent_coordinates(left, boundary, parent);
            parent_coordinates(right, boundary, parent);
        }
        E::Pow { base, .. } => parent_coordinates(base, boundary, parent),
        _ => {}
    }
}
