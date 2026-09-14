//! Admission of retained physical conservation meaning on fixed volume support.

use super::*;
use eqiora_core::{ScalarDomain, ValueFrame};
use eqiora_schema::kernel::ConservationTerms;

pub(super) fn validate_conservation_types(
    owner: RawId,
    terms: ConservationTerms,
    typed: &TypedResidual<RawId>,
    support: Option<&SpatialSupport<RawId>>,
    environment: TypingEnvironment<'_>,
    diagnostics: &mut Vec<Diagnostic>,
) {
    let Some(support @ SpatialSupport::Volume { .. }) = support else {
        diagnostics.push(kernel_error(
            owner,
            "fixed-domain conservation Law requires an exact volume support",
        ));
        return;
    };
    for (role, value) in [("flux", terms.flux()), ("source", terms.source())]
        .into_iter()
        .chain(
            terms
                .storage()
                .into_iter()
                .flat_map(|(value, accumulation)| {
                    [("storage", value), ("accumulation", accumulation)]
                }),
        )
    {
        let Some(value_type) = typed.node_type(value) else {
            diagnostics.push(kernel_error(
                owner,
                format!("Law {role} expression is missing"),
            ));
            continue;
        };
        if value_type.value_type.scalar_domain() != ScalarDomain::Real {
            diagnostics.push(kernel_error(
                owner,
                format!("initial scalar conservation Law requires real {role}"),
            ));
        }
        if role != "flux"
            && (!value_type.shape().is_scalar() || value_type.frame() != ValueFrame::Invariant)
        {
            diagnostics.push(kernel_error(
                owner,
                format!("scalar conservation Law requires invariant scalar {role}"),
            ));
        }
        if value_type
            .support
            .as_ref()
            .is_some_and(|actual| actual != support)
        {
            diagnostics.push(kernel_error(
                owner,
                format!("Law {role} uses a foreign support"),
            ));
        }
    }
    if let Some((stored, accumulation)) = terms.storage() {
        if let Err(error) = typed
            .expression()
            .verify_time_derivative(stored, accumulation)
        {
            diagnostics.push(kernel_error(
                owner,
                format!("Law storage accumulation correspondence failed: {error}"),
            ));
        }
        validate_storage_fields(owner, typed.expression(), stored, environment, diagnostics);
    }
    if let Some((stored, accumulation)) = terms.storage()
        && let (Some(value), Some(accumulation)) =
            (typed.node_type(stored), typed.node_type(accumulation))
    {
        let time = DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).expect("time dimension");
        if accumulation.dimension().mul(time) != Some(value.dimension()) {
            diagnostics.push(relation_dimension_error(
                owner,
                "Law storage dimension must equal accumulation dimension times time",
            ));
        }
    }
}

/// Validate storage Field eligibility even when its derivative cancels to zero.
fn validate_storage_fields(
    owner: RawId,
    expression: &ExprDag,
    storage: eqiora_schema::kernel::ExprId,
    environment: TypingEnvironment<'_>,
    diagnostics: &mut Vec<Diagnostic>,
) {
    let mut pending = vec![storage];
    let mut visited = BTreeSet::new();
    while let Some(id) = pending.pop() {
        if !visited.insert(id) {
            continue;
        }
        match expression.node(id) {
            Some(ExprNode::Symbol(SymbolRef::Field(field))) => {
                if symbol_type(
                    SymbolRef::Derivative(*field),
                    environment.nodes,
                    environment.edges,
                    environment.spatial_supports,
                )
                .is_err()
                {
                    diagnostics.push(kernel_error(
                        owner,
                        "Law storage may read only continuous state Fields",
                    ));
                }
            }
            Some(ExprNode::Neg(value) | ExprNode::PowI(value, _)) => pending.push(*value),
            Some(
                ExprNode::Add(left, right)
                | ExprNode::Sub(left, right)
                | ExprNode::Mul(left, right),
            ) => {
                pending.extend([*left, *right]);
            }
            Some(ExprNode::PureOperatorApplication(application)) => {
                pending.extend(application.arguments())
            }
            // The independent derivative checker rejects every other storage operator.
            _ => {}
        }
    }
}
