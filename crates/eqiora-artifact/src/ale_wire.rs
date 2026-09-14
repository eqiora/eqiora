//! Shared validation and encoding of the current ALE Realization payload.

use eqiora_core::Diagnostic;
use eqiora_realization::{
    FixedTopologyAleCoupledRealizationPlan, FixedTopologyAleCoupledRealizationRequirements,
    ResolvedFixedTopologyAleCoupledRealization,
};
use serde::{Deserialize, Serialize};
use ulid::Ulid;

use crate::realization_v7::wire::{WireLayoutArtifacts, WireQuadratureCodec};
use crate::{
    ArtifactDigest, CANONICAL_ENCODING, CanonicalModelArtifact, LayoutArtifacts,
    RealizationDecoderLimits, SimplicialMeshEnvelopeV1, check_json_limits, invalid_artifact,
};

pub(crate) mod wire;

use self::wire::WireAleRequirements;

pub(crate) fn validate_ale_mesh_quality_gate(
    plan: &FixedTopologyAleCoupledRealizationPlan,
    mesh: &SimplicialMeshEnvelopeV1,
) -> Result<(), Diagnostic> {
    let mesh_gate = mesh.mesh().quality_gate().minimum_mean_ratio();
    let realization_gate = plan.mesh_motion().quality_gate().minimum_mean_ratio();
    if mesh_gate.to_bits() != realization_gate.to_bits() {
        return Err(invalid_artifact(
            "reference mesh quality gate differs from the ALE mesh-motion quality gate",
        ));
    }
    Ok(())
}

pub(crate) fn validate_requirements_plan(
    requirements: &FixedTopologyAleCoupledRealizationRequirements,
    plan: &FixedTopologyAleCoupledRealizationPlan,
) -> Result<(), Diagnostic> {
    let motion = plan.mesh_motion();
    let pullback = plan.pullback();
    if motion.fluid_domain() != requirements.fluid_domain()
        || motion.solid_domain() != requirements.solid_domain()
        || motion.solid_displacement() != requirements.solid_displacement()
        || !matches!(requirements.coupled().trace_quotients(), [quotient] if quotient.connection() == motion.interface())
        || plan.fluid_time_step().relation() != requirements.fluid_relation()
        || plan.fluid_time_step().state() != requirements.fluid_velocity()
        || pullback.relation() != requirements.fluid_relation()
        || pullback.velocity() != requirements.fluid_velocity()
        || plan.solid_kinematic_relation() != requirements.solid_kinematic_relation()
    {
        return Err(invalid_artifact(
            "ALE plan differs from the exact lowerer Domain, Field, Relation, or Connection roles",
        ));
    }
    let coupled = plan.coupled();
    let eliminated = coupled
        .time_step()
        .eliminated_states()
        .iter()
        .map(|state| state.pair())
        .collect::<Vec<_>>();
    let represented = coupled
        .represented_physical_fields()
        .map_err(|error| invalid_artifact(error.to_string()))?;
    let selected_domains = coupled
        .spatial()
        .domains()
        .iter()
        .map(|domain| {
            eqiora_realization::DomainFieldInventory::new(
                domain.domain(),
                represented
                    .iter()
                    .filter(|field| field.domain() == domain.domain())
                    .map(|field| field.field()),
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| invalid_artifact(error.to_string()))?;
    if requirements.coupled().domains() != selected_domains
        || requirements.coupled().trace_quotients() != coupled.spatial().trace_quotients()
        || requirements.coupled().eliminated_states() != eliminated
    {
        return Err(invalid_artifact(
            "ALE common plan does not bind the exact required Domain, Field, trace, and eliminated-state inventory",
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireAleEnvelope<Q> {
    pub(crate) schema: String,
    pub(crate) encoding: String,
    pub(crate) model_sha256: String,
    pub(crate) model_ulid: String,
    pub(crate) semantic_revision: u64,
    pub(crate) source: WireAleSource,
    pub(crate) requirements: WireAleRequirements,
    pub(crate) plan: wire::WireAlePlanWith<Q>,
    pub(crate) layout_artifacts: WireLayoutArtifacts,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(crate) enum WireAleSource {
    Explicit { realization_revision: u64 },
}

impl<Q> WireAleEnvelope<Q>
where
    Q: WireQuadratureCodec + Clone + PartialEq,
{
    pub(crate) fn from_resolved(
        schema: &str,
        model: &impl CanonicalModelArtifact,
        resolved: &ResolvedFixedTopologyAleCoupledRealization,
        layout_artifacts: LayoutArtifacts,
    ) -> Result<Self, Diagnostic> {
        let model = model.artifact_reference()?;
        if model.model() != resolved.model()
            || model.semantic_revision() != resolved.semantic_revision()
        {
            return Err(invalid_artifact(
                "resolved ALE realization does not identify the supplied Model artifact and source revision",
            ));
        }
        require_layout_artifacts(
            resolved
                .requirements()
                .coupled()
                .execution()
                .vector_layout(),
            &layout_artifacts,
        )?;
        resolved.portable_graph().map_err(|error| {
            invalid_artifact(format!(
                "cannot project resolved fixed-topology ALE graph: {error}",
            ))
        })?;
        let wire = Self {
            schema: schema.to_owned(),
            encoding: CANONICAL_ENCODING.to_owned(),
            model_sha256: model.artifact().to_string(),
            model_ulid: resolved.model().ulid().to_string(),
            semantic_revision: resolved.semantic_revision().get(),
            source: WireAleSource::Explicit {
                realization_revision: resolved.realization_revision().get(),
            },
            requirements: WireAleRequirements::encode(resolved.requirements())?,
            plan: wire::WireAlePlanWith::<Q>::encode(resolved.requirements(), resolved.plan())?,
            layout_artifacts: WireLayoutArtifacts::encode(layout_artifacts),
        };
        wire.validate(schema)?;
        Ok(wire)
    }

    pub(crate) fn from_json(
        schema: &str,
        bytes: &[u8],
        limits: RealizationDecoderLimits,
    ) -> Result<Self, Diagnostic>
    where
        Q: for<'de> Deserialize<'de>,
    {
        check_json_limits(bytes, limits.json)?;
        let wire: Self = serde_json::from_slice(bytes).map_err(|error| {
            invalid_artifact(format!("invalid ALE realization envelope JSON: {error}"))
        })?;
        wire.requirements.validate_limits(limits)?;
        wire.plan.validate_limits(limits)?;
        wire.validate(schema)?;
        Ok(wire)
    }

    pub(crate) fn validate(&self, schema: &str) -> Result<(), Diagnostic> {
        if self.schema != schema || self.encoding != CANONICAL_ENCODING {
            return Err(invalid_artifact(format!(
                "unsupported {schema} schema or canonical encoding",
            )));
        }
        ArtifactDigest::from_hex(self.model_sha256.clone())?;
        parse_ulid(&self.model_ulid)?;
        let requirements = self.requirements.clone().decode()?;
        let plan = self.plan.clone().decode(&requirements)?;
        validate_requirements_plan(&requirements, &plan)?;
        if WireAleRequirements::encode(&requirements)? != self.requirements
            || wire::WireAlePlanWith::<Q>::encode(&requirements, &plan)? != self.plan
        {
            return Err(invalid_artifact(
                "ALE realization arrays or graph values are not in canonical closed form",
            ));
        }
        let artifacts = self.layout_artifacts.decode_validated()?;
        require_layout_artifacts(
            requirements.coupled().execution().vector_layout(),
            &artifacts,
        )?;
        Ok(())
    }
}

pub(crate) fn require_layout_artifacts(
    layout: eqiora_realization::VectorLayoutKind,
    artifacts: &LayoutArtifacts,
) -> Result<(), Diagnostic> {
    if matches!(
        (layout, artifacts),
        (
            eqiora_realization::VectorLayoutKind::Replicated,
            LayoutArtifacts::Replicated
        ) | (
            eqiora_realization::VectorLayoutKind::Distributed,
            LayoutArtifacts::Distributed { .. }
        )
    ) {
        Ok(())
    } else {
        Err(invalid_artifact(
            "ALE realization layout artifacts contradict the admitted vector-layout requirement",
        ))
    }
}

pub(crate) fn parse_ulid(value: &str) -> Result<Ulid, Diagnostic> {
    let ulid: Ulid = value
        .parse()
        .map_err(|_| invalid_artifact("model ULID is malformed"))?;
    if ulid.to_string() != value {
        return Err(invalid_artifact("model ULID is not in canonical spelling"));
    }
    Ok(ulid)
}
