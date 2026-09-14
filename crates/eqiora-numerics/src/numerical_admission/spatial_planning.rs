//! Private spatial-policy decision plane for common root resolution.
//!
//! This module admits caller spatial intent against already-recognized Model
//! meaning. It does not select a solver, derive a Formulation, authenticate
//! Mesh resources, construct a Realization, or seal a Plan.

use super::*;

pub(super) fn resolve_scalar(
    request: CommonSpatialRequest,
) -> Result<NativeSpatialPolicy, Diagnostic> {
    let CommonSpatialRequest::Uniform(spatial) = request else {
        return Err(invalid(
            "scalar-elliptic mathematics does not admit Domain-scoped spatial policies",
        ));
    };
    match spatial {
        CommonSpatialPolicy::Q1 => Ok(NativeSpatialPolicy::ScalarQ1),
        CommonSpatialPolicy::CellCenteredTpfa => Ok(NativeSpatialPolicy::ScalarTpfa),
        CommonSpatialPolicy::MiniP1 => Err(invalid(
            "scalar-elliptic Model mathematics is incompatible with MINI/P1",
        )),
        CommonSpatialPolicy::CellCentered => Err(invalid(
            "scalar-elliptic Model mathematics is incompatible with incompressible CellCentered",
        )),
        CommonSpatialPolicy::P1 => Err(invalid(
            "scalar-elliptic Model mathematics is incompatible with simplex P1",
        )),
    }
}

pub(super) fn resolve_elasticity(
    request: CommonSpatialRequest,
) -> Result<NativeSpatialPolicy, Diagnostic> {
    let CommonSpatialRequest::Uniform(spatial) = request else {
        return Err(invalid(
            "linear-elasticity mathematics does not admit Domain-scoped spatial policies",
        ));
    };
    if spatial != CommonSpatialPolicy::Q1 {
        return Err(invalid(
            "linear-elasticity mathematics requires the admitted Cartesian Q1 policy",
        ));
    }
    Ok(NativeSpatialPolicy::ElasticityQ1)
}

pub(super) fn resolve_stokes(
    request: CommonSpatialRequest,
) -> Result<StokesSpatialDecision, Diagnostic> {
    let CommonSpatialRequest::Uniform(spatial) = request else {
        return Err(invalid(
            "steady-Stokes mathematics does not admit Domain-scoped spatial policies",
        ));
    };
    if spatial != CommonSpatialPolicy::MiniP1 {
        return Err(invalid(
            "steady-Stokes Model mathematics requires the admitted MINI/P1 policy",
        ));
    }
    Ok(StokesSpatialDecision)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct StokesSpatialDecision;

impl StokesSpatialDecision {
    pub(super) const fn with_scaling(
        self,
        scales: IncompressibleFlowScaleProfile2d,
    ) -> NativeSpatialPolicy {
        NativeSpatialPolicy::StokesMiniP1(scales)
    }
}

pub(super) fn resolve_transient(
    request: CommonSpatialRequest,
) -> Result<TransientSpatialDecision, Diagnostic> {
    let CommonSpatialRequest::Uniform(spatial) = request else {
        return Err(invalid(
            "transient-flow mathematics does not admit Domain-scoped spatial policies",
        ));
    };
    match spatial {
        CommonSpatialPolicy::MiniP1 => Ok(TransientSpatialDecision::MiniP1),
        CommonSpatialPolicy::CellCentered => Ok(TransientSpatialDecision::CellCentered),
        CommonSpatialPolicy::Q1 | CommonSpatialPolicy::CellCenteredTpfa => Err(invalid(
            "transient incompressible-flow mathematics requires MINI/P1 or CellCentered",
        )),
        CommonSpatialPolicy::P1 => Err(invalid(
            "transient incompressible-flow mathematics does not admit standalone P1",
        )),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TransientSpatialDecision {
    MiniP1,
    CellCentered,
}

impl TransientSpatialDecision {
    pub(super) const fn with_scaling(
        self,
        scales: IncompressibleFlowScaleProfile2d,
    ) -> NativeSpatialPolicy {
        match self {
            Self::MiniP1 => NativeSpatialPolicy::TransientMiniP1(scales),
            Self::CellCentered => NativeSpatialPolicy::TransientCellCentered(scales),
        }
    }
}

pub(super) fn require_fixed_reference_fsi(
    model: &ModelEnvelope,
    canonical: &FixedReferenceFsiCartesianModel2d,
    request: CommonSpatialRequest,
) -> Result<(), Diagnostic> {
    let CommonSpatialRequest::Scoped(bindings) = request else {
        return Err(invalid(
            "fixed-reference FSI mathematics requires exact Domain-scoped spatial policies",
        ));
    };
    let expected = canonical
        .fluids()
        .map(|fluid| (fluid.domain(), CommonSpatialPolicy::MiniP1))
        .chain(
            canonical
                .solids()
                .map(|solid| (solid.continuum().domain(), CommonSpatialPolicy::P1)),
        )
        .collect::<BTreeMap<_, _>>();
    let model_digest = model.digest()?;
    let mut actual = BTreeMap::new();
    for binding in bindings {
        if binding.model() != &model_digest {
            return Err(invalid(
                "FSI scoped spatial policy carries a foreign or stale exact Model reference",
            ));
        }
        if actual
            .insert(binding.domain().erase(), binding.policy())
            .is_some()
        {
            return Err(invalid("FSI scoped spatial policy repeats one DomainRef"));
        }
    }
    if actual != expected {
        return Err(invalid(
            "FSI scoped spatial policies must completely and exclusively bind MiniP1 to fluid and P1 to solid",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_decisions_are_closed_before_mesh_admission() {
        assert_eq!(
            resolve_scalar(CommonSpatialRequest::Uniform(CommonSpatialPolicy::Q1)).unwrap(),
            NativeSpatialPolicy::ScalarQ1
        );
        assert_eq!(
            resolve_scalar(CommonSpatialRequest::Uniform(
                CommonSpatialPolicy::CellCenteredTpfa,
            ))
            .unwrap(),
            NativeSpatialPolicy::ScalarTpfa
        );
        assert_eq!(
            resolve_elasticity(CommonSpatialRequest::Uniform(CommonSpatialPolicy::Q1)).unwrap(),
            NativeSpatialPolicy::ElasticityQ1
        );
        assert!(resolve_stokes(CommonSpatialRequest::Uniform(CommonSpatialPolicy::MiniP1)).is_ok());
        assert_eq!(
            resolve_transient(CommonSpatialRequest::Uniform(CommonSpatialPolicy::MiniP1)).unwrap(),
            TransientSpatialDecision::MiniP1
        );
        assert_eq!(
            resolve_transient(CommonSpatialRequest::Uniform(
                CommonSpatialPolicy::CellCentered
            ))
            .unwrap(),
            TransientSpatialDecision::CellCentered
        );
    }

    #[test]
    fn uniform_decisions_reject_foreign_and_scoped_requests() {
        assert!(resolve_scalar(CommonSpatialRequest::Uniform(CommonSpatialPolicy::P1)).is_err());
        assert!(
            resolve_elasticity(CommonSpatialRequest::Uniform(CommonSpatialPolicy::MiniP1)).is_err()
        );
        assert!(resolve_stokes(CommonSpatialRequest::Uniform(CommonSpatialPolicy::Q1)).is_err());
        assert!(
            resolve_transient(CommonSpatialRequest::Uniform(
                CommonSpatialPolicy::CellCenteredTpfa,
            ))
            .is_err()
        );
        assert!(resolve_scalar(CommonSpatialRequest::Scoped(Vec::new())).is_err());
        assert!(resolve_transient(CommonSpatialRequest::Scoped(Vec::new())).is_err());
    }
}
