//! Replay one exact zero-duration representation change.
use super::*;
use crate::{
    MeshRevisionOverlapEnvelopeV1, RemeshTransferReceiptEnvelopeV1, ValidatedRemeshGeometrySource,
};
impl SpatialStateEnvelope {
    /// Construct the first target state at an exact same-time remesh seam.
    ///
    /// # Errors
    /// Returns `EQ0901` for any stale resource, incomplete Field inventory,
    /// invalid geometry/overlap/receipt replay, or changed step/time.
    #[allow(clippy::too_many_arguments)]
    pub fn remesh<M: ReplayableCanonicalModelArtifact>(
        source: &ValidatedRemeshGeometrySource<'_, M>,
        target_context: &ValidatedMovingSpatialContext<'_, M>,
        target_geometry_state: &GeometryStateEnvelope,
        overlap: &MeshRevisionOverlapEnvelopeV1,
        receipt: &RemeshTransferReceiptEnvelopeV1,
        source_snapshots: &[FieldSnapshotEnvelopeV2],
        target_snapshots: &[FieldSnapshotEnvelopeV2],
    ) -> Result<Self, Diagnostic> {
        if target_geometry_state.origin() != GeometryStateOriginKind::Remesh
            || target_geometry_state.step() != source.state().step()
            || target_geometry_state.time_s() != source.state().time_s()
        {
            return Err(invalid_artifact(
                "first target state must be a same-coordinate remesh representation",
            ));
        }
        let (fields, target_solid_displacement) =
            state_inputs(target_context, target_geometry_state, target_snapshots)?;
        overlap.validate_against(
            source,
            target_context,
            target_geometry_state,
            target_solid_displacement,
        )?;
        receipt.validate_against(
            source,
            overlap,
            target_context,
            target_geometry_state,
            source_snapshots,
            target_snapshots,
        )?;
        Self::finish(
            target_context,
            target_geometry_state,
            fields,
            WireOrigin::Remesh {
                anchor: RemeshStateAnchor {
                    source_spatial_state_sha256: source.state().digest()?.to_string(),
                    overlap_sha256: overlap.digest()?.to_string(),
                    transfer_receipt_sha256: receipt.digest()?.to_string(),
                },
            },
        )
    }

    /// Replay and compare a remesh-origin state.
    ///
    /// # Errors
    /// Returns `EQ0901` for any substituted dependency.
    #[allow(clippy::too_many_arguments)]
    pub fn validate_against_remesh<M: ReplayableCanonicalModelArtifact>(
        &self,
        source: &ValidatedRemeshGeometrySource<'_, M>,
        target_context: &ValidatedMovingSpatialContext<'_, M>,
        target_geometry_state: &GeometryStateEnvelope,
        overlap: &MeshRevisionOverlapEnvelopeV1,
        receipt: &RemeshTransferReceiptEnvelopeV1,
        source_snapshots: &[FieldSnapshotEnvelopeV2],
        target_snapshots: &[FieldSnapshotEnvelopeV2],
    ) -> Result<(), Diagnostic> {
        self.require_equal(&Self::remesh(
            source,
            target_context,
            target_geometry_state,
            overlap,
            receipt,
            source_snapshots,
            target_snapshots,
        )?)
    }
}
