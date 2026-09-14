use eqiora_core::{Diagnostic, ScalarType};
use eqiora_distributed::DistributedLinearSystem;
use eqiora_realization::VectorLayoutKind;

use super::{DistributedLayoutEnvelopeV1, LinearSystemEnvelopeV1, PartitionEnvelopeV1};
use crate::{
    ExecutionTopologyV1, LayoutArtifacts, ModelEnvelope, RealizationEnvelopeV1, RunManifestV2,
    invalid_artifact,
};

/// Validate the external links in one distributed spatial artifact content DAG.
///
/// This check proves exact content linkage from the run through its
/// Realization and Model, and from that Realization through the distributed
/// layout to its complete system and unique-owner partition. The returned
/// system is freshly reconstructed from the linked complete algebra and owner
/// map; callers can reuse it for later admission instead of deriving the
/// layout a second time.
///
/// This artifact-layer check deliberately does not prove that `system` was
/// lowered from `model` under `realization`. Semantic derivation replay belongs
/// in the public facade integration, where the canonical lowerer is available.
/// In particular, the admitted spatial dimension is not the algebraic degree-
/// of-freedom count and can only be rechecked against the replayed mesh and
/// lowering, while this function proves the system/partition algebraic
/// dimension agreement.
///
/// # Errors
/// Returns `EQ0901` if any content identity, typed model identity, semantic
/// revision, distributed requirement, algebraic shape/scalar, derived layout,
/// execution partition count, target, worker count, or reduction policy is
/// inconsistent.
pub fn validate_distributed_content_dag(
    model: &ModelEnvelope,
    realization: &RealizationEnvelopeV1,
    run: &RunManifestV2,
    system: &LinearSystemEnvelopeV1,
    partition: &PartitionEnvelopeV1,
    layout: &DistributedLayoutEnvelopeV1,
) -> Result<DistributedLinearSystem, Diagnostic> {
    if realization.model_artifact() != model.digest()?
        || realization.model()? != model.model()?
        || realization.semantic_revision().get() != model.source_revision()
    {
        return Err(invalid_artifact(
            "realization model digest, typed identity, or semantic revision differs from the linked model artifact",
        ));
    }

    run.validate_against(realization)?;

    let expected_layout = layout.digest()?;
    let expected_partition = partition.digest()?;
    match realization.layout_artifacts() {
        LayoutArtifacts::Distributed { layout, partition }
            if layout == expected_layout && partition == expected_partition => {}
        _ => {
            return Err(invalid_artifact(
                "realization does not reference the supplied distributed layout and partition artifacts exactly",
            ));
        }
    }

    let requirements = realization.requirements()?;
    if requirements.vector_layout() != VectorLayoutKind::Distributed
        || requirements.scalar_type() != ScalarType::F64
    {
        return Err(invalid_artifact(
            "distributed content DAG requires an admitted distributed f64 realization",
        ));
    }

    // Fresh derivation proves the system/partition global dimension and f64
    // scalar agreement before comparing every stored local and halo record.
    let distributed = layout.validate_against(system, partition)?;
    let execution = run.execution();
    match execution.topology()? {
        ExecutionTopologyV1::Distributed { partitions, .. }
            if partitions == distributed.partition().count() => {}
        ExecutionTopologyV1::Distributed { partitions, .. } => {
            return Err(invalid_artifact(format!(
                "run declares {partitions} partitions for a linked partition artifact with {}",
                distributed.partition().count(),
            )));
        }
        _ => {
            return Err(invalid_artifact(
                "distributed content DAG requires distributed execution topology",
            ));
        }
    }

    Ok(distributed)
}
