//! Probability-law admission and binding, separate from realized samples.

use super::*;
use eqiora_core::{ScalarDomain, ValueFrame};
use eqiora_schema::kernel::{ObservableMeasure, RelationMeaning};

mod binding;
pub(super) use binding::validate_bindings;
mod provenance;
pub(super) use provenance::validate_sample_dependencies;
mod evaluation;
pub(super) use evaluation::{evaluate_finite_moment, evaluate_integrands};

pub(super) fn domain(
    distribution: Id<kinds::Relation>,
    nodes: &BTreeMap<RawId, KernelNode>,
) -> Result<Option<Id<kinds::Domain>>, Diagnostic> {
    match nodes.get(&distribution.erase()) {
        Some(KernelNode::Relation(law)) => match law.meaning() {
            RelationMeaning::ProbabilityMass { .. } => Ok(None),
            RelationMeaning::ProbabilityDensity { domain, .. } => Ok(Some(*domain)),
            _ => Err(kernel_error(
                distribution.erase(),
                "random variable requires an exact probability law",
            )),
        },
        _ => Err(kernel_error(
            distribution.erase(),
            "probability law is outside the exact Model",
        )),
    }
}

pub(super) fn validate_types(
    owner: RawId,
    meaning: &RelationMeaning,
    typed: &TypedResidual<RawId>,
    environment: TypingEnvironment<'_>,
    diagnostics: &mut Vec<Diagnostic>,
) {
    if let Err(error) = validate(owner, meaning, typed, environment) {
        diagnostics.push(error);
    }
}

fn validate(
    owner: RawId,
    meaning: &RelationMeaning,
    typed: &TypedResidual<RawId>,
    environment: TypingEnvironment<'_>,
) -> Result<(), Diagnostic> {
    if typed.expression().nodes().iter().any(|node| {
        matches!(node, ExprNode::Symbol(symbol) if !matches!(symbol, SymbolRef::Parameter(_)))
    }) {
        return Err(kernel_error(owner, "probability weights must be immutable Parameter expressions; conditional laws require separate admission"));
    }
    let values = crate::evaluate::evaluate_expression(owner, typed.expression(), &mut |symbol| {
        let SymbolRef::Parameter(id) = symbol else {
            return None;
        };
        environment.values.get(&id.erase()).cloned()
    })?;
    let value = &values[0];
    let ty = value.value_type();
    if ty.scalar_domain() != ScalarDomain::Real
        || ty.frame() != ValueFrame::Invariant
        || ty.finite_bases().next().is_some()
    {
        return Err(kernel_error(
            owner,
            "probability weights require invariant real values",
        ));
    }
    match meaning {
        RelationMeaning::ProbabilityMass { outcomes, weights } => {
            let Some(KernelNode::Enum(outcomes)) = environment.nodes.get(&outcomes.erase()) else {
                return Err(kernel_error(
                    owner,
                    "probability mass requires an exact named-outcome declaration",
                ));
            };
            if ty.dimension() != DimExponents::DIMENSIONLESS
                || ty.array_rank() != 1
                || !matches!(ty.shape().extents(), [count] if count.get() as usize == outcomes.members().len())
                || typed
                    .node_type(*weights)
                    .expect("typed weights")
                    .support
                    .is_some()
                || !edge_targets(environment.edges, owner, EdgeKind::AppliesOn).is_empty()
            {
                return Err(kernel_error(
                    owner,
                    "probability mass requires one dimensionless weight per exact named outcome",
                ));
            }
        }
        RelationMeaning::ProbabilityDensity {
            domain,
            measure,
            density,
        } => {
            let support = environment
                .spatial_supports
                .get(&domain.erase())
                .ok_or_else(|| kernel_error(owner, "probability density support is unavailable"))?;
            let root = typed.node_type(*density).expect("typed density");
            if !ty.shape().is_scalar()
                || ty.array_rank() != 0
                || measure
                    .output_type(root, support, support, None)?
                    .dimension()
                    != DimExponents::DIMENSIONLESS
                || edge_targets(environment.edges, owner, EdgeKind::AppliesOn)
                    != BTreeSet::from([domain.erase()])
            {
                return Err(kernel_error(
                    owner,
                    "probability density requires inverse-measure units on its exact support",
                ));
            }
            let SpatialSupport::Coordinates { factors, .. } = support else {
                return Err(kernel_error(
                    owner,
                    "uniform density admission requires bounded coordinate factors",
                ));
            };
            if *measure != ObservableMeasure::Volume
                || factors.iter().any(|(factor, _, axes)| {
                    *axes != 1
                        || !matches!(environment.nodes.get(factor), Some(KernelNode::Domain(domain))
                    if matches!(domain.kind(), DomainKind::CoordinateInterval { .. }))
                })
            {
                return Err(kernel_error(
                    owner,
                    "uniform density admission requires the product measure of bounded coordinate intervals",
                ));
            }
            // Each admitted interval has finite, strictly ordered bounds. A positive
            // constant density therefore has finite positive integral on their finite
            // product. No quadrature tolerance or rounded normalization test is used.
        }
        _ => unreachable!("probability admission dispatch"),
    }
    eqiora_schema::kernel::ProbabilityMoment::validate_weights(
        value
            .components()
            .expect("real weights")
            .map(|(weight, _)| weight),
    )
    .map_err(|error| kernel_error(owner, error.message()))
}
