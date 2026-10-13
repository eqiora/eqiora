//! Physical normalization across an accepted remesh seam.

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

pub(super) struct RemeshRealization {
    pub(super) fluid_velocity: Id<kinds::Field>,
    pub(super) solid_velocity: Id<kinds::Field>,
    pub(super) pressure: Id<kinds::Field>,
    pub(super) displacement: Id<kinds::Field>,
    scales: AleFsiRemeshScaleProfile2d,
}

pub(super) fn realization_remesh_scales(
    realization: &crate::RealizationEnvelope,
) -> Result<AleFsiRemeshScaleProfile2d, Diagnostic> {
    Ok(remesh_realization(realization)?.scales)
}

pub(super) fn remesh_realization(
    realization: &crate::RealizationEnvelope,
) -> Result<RemeshRealization, Diagnostic> {
    use eqiora_realization::{CoordinateTreatment, GeometryActionNode, TransformationNode};
    let graph = realization.graph();
    let [GeometryActionNode::P1HarmonicExtension { drivers, .. }] = graph.geometry_actions() else {
        return Err(invalid_artifact(
            "remesh requires one exact harmonic geometry action",
        ));
    };
    let [driver] = drivers.as_slice() else {
        return Err(invalid_artifact(
            "remesh requires one exact displacement driver",
        ));
    };
    let state = graph
        .field(*driver)
        .ok_or_else(|| invalid_artifact("remesh driver is absent"))?;
    let pullbacks = graph
        .transformations()
        .iter()
        .filter_map(|node| match node {
            TransformationNode::GclCompatibleAlePullback { velocity, .. } => Some(*velocity),
            _ => None,
        })
        .collect::<Vec<_>>();
    let [fluid_velocity] = pullbacks.as_slice() else {
        return Err(invalid_artifact(
            "remesh requires one exact ALE fluid velocity",
        ));
    };
    let fluid_velocity = graph
        .field(*fluid_velocity)
        .ok_or_else(|| invalid_artifact("remesh fluid velocity is absent"))?;
    let eliminated = graph
        .transformations()
        .iter()
        .filter_map(|node| match node {
            TransformationNode::BackwardEulerElimination {
                state,
                rate,
                state_scale,
                ..
            } if state == driver => Some((*rate, *state_scale)),
            _ => None,
        })
        .collect::<Vec<_>>();
    let [(rate, state_scale)] = eliminated.as_slice() else {
        return Err(invalid_artifact(
            "remesh driver lacks one exact state/rate binding",
        ));
    };
    let solid_velocity = graph
        .field(*rate)
        .ok_or_else(|| invalid_artifact("remesh driver rate is absent"))?;
    let pressures = graph
        .fields()
        .iter()
        .filter(|node| {
            node.domain() == fluid_velocity.domain() && node.field() != fluid_velocity.field()
        })
        .collect::<Vec<_>>();
    let [pressure] = pressures.as_slice() else {
        return Err(invalid_artifact(
            "ALE remesh scale replay requires one exact fluid pressure Field",
        ));
    };
    let scaling = graph.systems()[0]
        .congruence_scaling()
        .ok_or_else(|| invalid_artifact("remesh requires explicit physical block scales"))?;
    let scale_for = |field| {
        scaling
            .block_scales()
            .iter()
            .find_map(|entry| {
                (entry.block() == AlgebraicBlock::Field(field)).then_some(entry.scale().quantity())
            })
            .ok_or_else(|| invalid_artifact("ALE remesh Field has no exact block scale"))
    };
    let length_for = |domain| match graph.domain(domain).map(|node| node.coordinates()) {
        Some(CoordinateTreatment::Scaled(length)) => Ok(length.quantity()),
        _ => Err(invalid_artifact(
            "remesh Domain has no exact physical length scale",
        )),
    };
    let length = length_for(fluid_velocity.domain())?;
    let velocity = scale_for(fluid_velocity.field())?;
    if length_for(state.domain())? != length
        || state_scale.quantity() != length
        || scale_for(solid_velocity.field())? != velocity
    {
        return Err(invalid_artifact(
            "ALE remesh requires equal fluid/solid velocity scales and displacement/length scales",
        ));
    }
    let scales = AleFsiRemeshScaleProfile2d::new(length, velocity, scale_for(pressure.field())?)
        .map_err(|error| invalid_artifact(error.message()))?;
    Ok(RemeshRealization {
        fluid_velocity: fluid_velocity.field(),
        solid_velocity: solid_velocity.field(),
        pressure: pressure.field(),
        displacement: state.field(),
        scales,
    })
}
