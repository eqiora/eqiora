//! Exact physical normalization across an admitted remesh seam.

use super::*;
use eqiora_solver::AlgebraicBlock;

pub(super) fn validate_normalization_closure(
    source_scales: AleFsiRemeshScaleProfile2d,
    target_scales: AleFsiRemeshScaleProfile2d,
    normalization: RemeshNormalizationWitnessV1,
    projections: &[RemeshProjectionEvidenceEnvelopeV1],
) -> Result<(), Diagnostic> {
    if source_scales != target_scales || normalization.scales() != source_scales {
        return Err(invalid_artifact(
            "remesh source, target, and physical evidence must share one exact normalization profile",
        ));
    }
    validate_evidence_projection_normalization(normalization, projections)
}

pub(super) fn validate_evidence_projection_normalization(
    normalization: RemeshNormalizationWitnessV1,
    projections: &[RemeshProjectionEvidenceEnvelopeV1],
) -> Result<(), Diagnostic> {
    if projections
        .iter()
        .map(RemeshProjectionEvidenceEnvelopeV1::plan)
        .collect::<Result<Vec<_>, _>>()?
        .iter()
        .any(|plan| plan.scales() != normalization.scales())
    {
        return Err(invalid_artifact(
            "remesh projection scales differ from exact physical normalization evidence",
        ));
    }
    Ok(())
}

pub(super) fn realization_remesh_scales(
    realization: &crate::RealizationEnvelopeV6,
) -> Result<AleFsiRemeshScaleProfile2d, Diagnostic> {
    let plan = realization.plan()?;
    let requirements = realization.requirements()?;
    let fluid_velocity = requirements.fluid_velocity();
    let solid_velocity = requirements
        .coupled()
        .eliminated_states()
        .iter()
        .find(|state| state.state() == requirements.solid_displacement())
        .ok_or_else(|| invalid_artifact("remesh driver lacks exact state/rate binding"))?
        .rate();
    let fluid_pressure = requirements
        .coupled()
        .domains()
        .iter()
        .find(|domain| domain.domain() == requirements.fluid_domain())
        .map(|domain| {
            domain
                .fields()
                .iter()
                .copied()
                .filter(|field| *field != fluid_velocity)
                .collect::<Vec<_>>()
        })
        .filter(|fields| fields.len() == 1)
        .and_then(|fields| fields.into_iter().next())
        .ok_or_else(|| {
            invalid_artifact("ALE remesh scale replay requires one exact fluid pressure Field")
        })?;
    let scale_for = |field| {
        plan.coupled()
            .scaling()
            .block_scales()
            .iter()
            .find_map(|entry| {
                (entry.block() == AlgebraicBlock::Field(field)).then_some(entry.scale().quantity())
            })
            .ok_or_else(|| invalid_artifact("ALE remesh Field has no exact block scale"))
    };
    let length = plan
        .coupled()
        .spatial()
        .coordinate_length_scale()
        .quantity();
    let fluid_velocity_scale = scale_for(fluid_velocity)?;
    let solid_velocity_scale = scale_for(solid_velocity)?;
    let pressure = scale_for(fluid_pressure)?;
    let displacement = plan
        .coupled()
        .time_step()
        .eliminated_states()
        .iter()
        .find(|state| state.pair().state() == requirements.solid_displacement())
        .ok_or_else(|| invalid_artifact("remesh driver lacks exact state scale"))?
        .state_scale()
        .quantity();
    if fluid_velocity_scale != solid_velocity_scale || displacement != length {
        return Err(invalid_artifact(
            "ALE remesh requires equal fluid/solid velocity scales and displacement/length scales",
        ));
    }
    AleFsiRemeshScaleProfile2d::new(length, fluid_velocity_scale, pressure)
        .map_err(|error| invalid_artifact(error.message()))
}
