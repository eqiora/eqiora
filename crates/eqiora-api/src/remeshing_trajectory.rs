//! Exact replay of one remeshing trajectory and its optional storage projection.

use std::collections::{BTreeMap, BTreeSet};

use eqiora_artifact::{
    ArtifactDigest, DiscreteFieldEnvelopeV1, FieldSnapshotEnvelopeV2,
    GeometryRevisionAssociationEnvelopeV1, GeometryStateEnvelope, MeshRevisionOverlapEnvelopeV1,
    RemeshTransferReceiptEnvelopeV1, ReplayableCanonicalModelArtifact, SpatialStateEnvelope,
    SpatialStateOriginKind, SpatialTrajectoryEnvelopeV2, SpatialTrajectoryEnvelopeV3,
    SpatialTrajectorySegmentEnvelopeV2, SpatialTrajectorySegmentEnvelopeV3,
    ValidatedMovingSpatialContext,
};
#[cfg(feature = "hdf5")]
use eqiora_artifact::{
    ExternalAdapterIdentityV1, ExternalRuntimeComponentV1, ExternalRuntimeRoleV1,
    TemporalStorageBlockPresentationV1, XdmfHdf5TrajectoryFieldV1, XdmfHdf5TrajectoryFrameV1,
    XdmfHdf5TrajectoryStorageEnvelope,
};
use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
#[cfg(feature = "hdf5")]
use eqiora_io_hdf5::{
    Hdf5DatasetWrite, Hdf5RuntimeIdentity, Hdf5WriteLimits, write_hdf5_file_image,
};
#[cfg(feature = "hdf5")]
use eqiora_io_xml::{
    XdmfTemporalExportLimits, XdmfTemporalExportPlan, XdmfTemporalField, XdmfTemporalFrame,
};
#[cfg(feature = "hdf5")]
use eqiora_meshing::DiscreteFieldAssociation;

#[cfg(feature = "hdf5")]
const ADAPTER_ID: &str = "eqiora.xdmf-hdf5.trajectory-file-image";
#[cfg(feature = "hdf5")]
const ADAPTER_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, Clone)]
pub(crate) struct ReplayedField<'a> {
    pub(crate) snapshot: &'a FieldSnapshotEnvelopeV2,
    pub(crate) blocks: Vec<&'a DiscreteFieldEnvelopeV1>,
}

#[derive(Debug, Clone)]
pub(crate) struct MovingSpatialFrame<'a> {
    pub(crate) state: &'a SpatialStateEnvelope,
    pub(crate) geometry: &'a GeometryStateEnvelope,
    pub(crate) fields: Vec<ReplayedField<'a>>,
}

/// Complete borrowed dependency replay for one V2-to-V3 trajectory root.
///
/// This is deliberately not a universal trajectory interface. It represents
/// the exact first V2-to-V3 replay profile and adds no durable identity.
#[derive(Debug)]
pub struct RemeshingTrajectoryReplayInputV1<'a, M: ReplayableCanonicalModelArtifact> {
    source_context: &'a ValidatedMovingSpatialContext<'a, M>,
    target_context: &'a ValidatedMovingSpatialContext<'a, M>,
    source_frames: Vec<MovingSpatialFrame<'a>>,
    trajectory: &'a SpatialTrajectoryEnvelopeV3,
    target_frames: Vec<MovingSpatialFrame<'a>>,
}

impl<'a, M: ReplayableCanonicalModelArtifact> RemeshingTrajectoryReplayInputV1<'a, M> {
    /// Replay every root, segment, state, geometry, snapshot, and block edge.
    ///
    /// # Errors
    /// Returns `EQ0901` for any missing, stale, substituted, reordered, or
    /// cross-context dependency.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        source_context: &'a ValidatedMovingSpatialContext<'a, M>,
        source_root: &'a SpatialTrajectoryEnvelopeV2,
        source_segments: &'a [SpatialTrajectorySegmentEnvelopeV2],
        source_states: &'a [SpatialStateEnvelope],
        source_geometry_states: &'a [GeometryStateEnvelope],
        target_context: &'a ValidatedMovingSpatialContext<'a, M>,
        trajectory: &'a SpatialTrajectoryEnvelopeV3,
        target_segments: &'a [SpatialTrajectorySegmentEnvelopeV3],
        target_states: &'a [SpatialStateEnvelope],
        target_geometry_states: &'a [GeometryStateEnvelope],
        snapshots: &'a [FieldSnapshotEnvelopeV2],
        blocks: &'a [DiscreteFieldEnvelopeV1],
        association: &'a GeometryRevisionAssociationEnvelopeV1,
        overlap: &'a MeshRevisionOverlapEnvelopeV1,
        receipt: &'a RemeshTransferReceiptEnvelopeV1,
    ) -> Result<Self, Diagnostic> {
        let source_segment_index = index_catalog(
            source_segments,
            SpatialTrajectorySegmentEnvelopeV2::digest,
            "source trajectory segment",
        )?;
        let source_segments = resolve_catalog(
            &source_segment_index,
            &source_root.segment_artifacts(),
            "source trajectory segment",
        )?
        .into_iter()
        .cloned()
        .collect::<Vec<_>>();
        source_root.validate_segments(source_context, &source_segments)?;
        let mut source_replay = SpatialTrajectoryEnvelopeV2::start(
            source_context,
            source_segments
                .first()
                .ok_or_else(|| replay_error("remeshing replay source segment set is empty"))?,
        )?;
        for segment in &source_segments[1..] {
            source_replay =
                SpatialTrajectoryEnvelopeV2::extend(source_context, &source_replay, segment)?;
        }
        if &source_replay != source_root {
            return Err(replay_error(
                "remeshing replay source root differs from complete immutable-prefix replay",
            ));
        }
        let target_segment_index = index_catalog(
            target_segments,
            SpatialTrajectorySegmentEnvelopeV3::digest,
            "target trajectory segment",
        )?;
        let target_segments = resolve_catalog(
            &target_segment_index,
            &trajectory.segment_artifacts(),
            "target trajectory segment",
        )?
        .into_iter()
        .cloned()
        .collect::<Vec<_>>();
        trajectory.validate_segments(source_root, &target_segments)?;

        let source_state_index = index_catalog(
            source_states,
            SpatialStateEnvelope::digest,
            "source spatial state",
        )?;
        let source_state_ids = source_segments
            .iter()
            .flat_map(SpatialTrajectorySegmentEnvelopeV2::state_artifacts)
            .collect::<Vec<_>>();
        let ordered_source_states = resolve_catalog(
            &source_state_index,
            &source_state_ids,
            "source spatial state",
        )?;
        let target_state_index = index_catalog(
            target_states,
            SpatialStateEnvelope::digest,
            "target spatial state",
        )?;
        let target_state_ids = target_segments
            .iter()
            .flat_map(SpatialTrajectorySegmentEnvelopeV3::state_artifacts)
            .collect::<Vec<_>>();
        let ordered_target_states = resolve_catalog(
            &target_state_index,
            &target_state_ids,
            "target spatial state",
        )?;
        let source_geometry_index = index_catalog(
            source_geometry_states,
            GeometryStateEnvelope::digest,
            "source geometry state",
        )?;
        let target_geometry_index = index_catalog(
            target_geometry_states,
            GeometryStateEnvelope::digest,
            "target geometry state",
        )?;
        let snapshot_index =
            index_catalog(snapshots, FieldSnapshotEnvelopeV2::digest, "Field snapshot")?;
        let block_index = index_catalog(
            blocks,
            DiscreteFieldEnvelopeV1::digest,
            "DiscreteField block",
        )?;
        let mut used_snapshots = BTreeSet::new();
        let mut used_blocks = BTreeSet::new();
        let source_frames = ordered_source_states
            .iter()
            .map(|state| {
                Ok(MovingSpatialFrame {
                    state,
                    geometry: resolve_one(
                        &source_geometry_index,
                        &state.geometry_state_artifact(),
                        "source geometry state",
                    )?,
                    fields: replay_fields(
                        &state.fields(),
                        &snapshot_index,
                        &block_index,
                        &mut used_snapshots,
                        &mut used_blocks,
                    )?,
                })
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        let target_frames = ordered_target_states
            .iter()
            .map(|state| {
                Ok(MovingSpatialFrame {
                    state,
                    geometry: resolve_one(
                        &target_geometry_index,
                        &state.geometry_state_artifact(),
                        "target geometry state",
                    )?,
                    fields: replay_fields(
                        &state.fields(),
                        &snapshot_index,
                        &block_index,
                        &mut used_snapshots,
                        &mut used_blocks,
                    )?,
                })
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        require_complete_catalog(
            source_segment_index.len(),
            source_segments.len(),
            "source trajectory segment",
        )?;
        require_complete_catalog(
            target_segment_index.len(),
            target_segments.len(),
            "target trajectory segment",
        )?;
        require_complete_catalog(
            source_state_index.len(),
            source_frames.len(),
            "source spatial state",
        )?;
        require_complete_catalog(
            target_state_index.len(),
            target_frames.len(),
            "target spatial state",
        )?;
        require_complete_catalog(
            source_geometry_index.len(),
            source_frames.len(),
            "source geometry state",
        )?;
        require_complete_catalog(
            target_geometry_index.len(),
            target_frames.len(),
            "target geometry state",
        )?;
        require_complete_catalog(snapshot_index.len(), used_snapshots.len(), "Field snapshot")?;
        require_complete_catalog(block_index.len(), used_blocks.len(), "DiscreteField block")?;

        validate_source_segments(source_context, &source_segments, &source_frames)?;
        validate_target_segments(target_context, &target_segments, &target_frames)?;
        validate_source_frames(source_context, &source_frames)?;
        validate_target_frames(target_context, &target_frames)?;

        let source_tip = source_frames
            .last()
            .ok_or_else(|| replay_error("remeshing replay source trajectory is empty"))?;
        let target_first = target_frames
            .first()
            .ok_or_else(|| replay_error("remeshing replay target trajectory is empty"))?;
        if trajectory.source_state() != source_tip.state.digest()?
            || target_first.state.origin() != SpatialStateOriginKind::Remesh
            || target_first
                .state
                .remesh_anchor()
                .map(eqiora_artifact::RemeshStateAnchor::source_spatial_state)
                != Some(source_tip.state.digest()?)
            || target_first.state.step() != source_tip.state.step()
            || target_first.state.time_s() != source_tip.state.time_s()
        {
            return Err(replay_error(
                "remeshing replay target does not replace the exact same-coordinate source tip",
            ));
        }
        let source_snapshots = snapshot_objects(&source_tip.fields);
        let source_predecessor = source_frames
            .len()
            .checked_sub(2)
            .and_then(|index| source_frames.get(index))
            .map(|frame| (frame.state, frame.geometry));
        let remesh_source = eqiora_artifact::ValidatedRemeshGeometrySource::new(
            source_context,
            source_tip.state,
            source_tip.geometry,
            source_predecessor,
            &source_snapshots,
            driver_blocks(&source_tip.fields, &source_tip.geometry.driver_snapshot())?,
            association,
        )?;
        let target_snapshots = snapshot_objects(&target_first.fields);
        target_first.state.validate_against_remesh(
            &remesh_source,
            target_context,
            target_first.geometry,
            overlap,
            receipt,
            &source_snapshots,
            &target_snapshots,
        )?;
        for pair in target_frames.windows(2) {
            let current_snapshots = snapshot_objects(&pair[1].fields);
            pair[1].state.validate_against(
                target_context,
                pair[1].geometry,
                Some((pair[0].state, pair[0].geometry)),
                &current_snapshots,
                driver_blocks(&pair[1].fields, &pair[1].geometry.driver_snapshot())?,
            )?;
        }
        let external_frame_count = source_frames
            .len()
            .checked_sub(1)
            .and_then(|count| count.checked_add(target_frames.len()))
            .ok_or_else(|| replay_error("remeshing replay frame count overflows usize"))?;
        if external_frame_count < 2 {
            return Err(replay_error(
                "remeshing replay requires at least two frames after remesh replacement",
            ));
        }
        Ok(Self {
            source_context,
            target_context,
            source_frames,
            trajectory,
            target_frames,
        })
    }

    /// Exact remeshing-aware trajectory root being projected.
    #[must_use]
    pub const fn trajectory(&self) -> &'a SpatialTrajectoryEnvelopeV3 {
        self.trajectory
    }

    pub(crate) const fn source_context(&self) -> &'a ValidatedMovingSpatialContext<'a, M> {
        self.source_context
    }

    pub(crate) const fn target_context(&self) -> &'a ValidatedMovingSpatialContext<'a, M> {
        self.target_context
    }

    pub(crate) fn source_frames(&self) -> &[MovingSpatialFrame<'a>] {
        &self.source_frames
    }

    pub(crate) fn target_frames(&self) -> &[MovingSpatialFrame<'a>] {
        &self.target_frames
    }
}

fn index_catalog<'a, T>(
    items: &'a [T],
    digest: impl Fn(&T) -> Result<ArtifactDigest, Diagnostic>,
    label: &str,
) -> Result<BTreeMap<ArtifactDigest, &'a T>, Diagnostic> {
    let mut index = BTreeMap::new();
    for item in items {
        let identity = digest(item)?;
        if index.insert(identity, item).is_some() {
            return Err(replay_error(format!(
                "remeshing replay {label} catalog contains a duplicate identity"
            )));
        }
    }
    Ok(index)
}

fn resolve_catalog<'a, T>(
    index: &BTreeMap<ArtifactDigest, &'a T>,
    identities: &[ArtifactDigest],
    label: &str,
) -> Result<Vec<&'a T>, Diagnostic> {
    identities
        .iter()
        .map(|identity| resolve_one(index, identity, label))
        .collect()
}

fn resolve_one<'a, T>(
    index: &BTreeMap<ArtifactDigest, &'a T>,
    identity: &ArtifactDigest,
    label: &str,
) -> Result<&'a T, Diagnostic> {
    index
        .get(identity)
        .copied()
        .ok_or_else(|| replay_error(format!("remeshing replay omits exact {label} `{identity}`")))
}

fn replay_fields<'a>(
    references: &[(
        eqiora_core::Id<eqiora_core::entity::kinds::Domain>,
        eqiora_core::Id<eqiora_core::entity::kinds::Field>,
        ArtifactDigest,
    )],
    snapshots: &BTreeMap<ArtifactDigest, &'a FieldSnapshotEnvelopeV2>,
    blocks: &BTreeMap<ArtifactDigest, &'a DiscreteFieldEnvelopeV1>,
    used_snapshots: &mut BTreeSet<ArtifactDigest>,
    used_blocks: &mut BTreeSet<ArtifactDigest>,
) -> Result<Vec<ReplayedField<'a>>, Diagnostic> {
    references
        .iter()
        .map(|(support, field, snapshot_identity)| {
            let snapshot = resolve_one(snapshots, snapshot_identity, "Field snapshot")?;
            if snapshot.support_domain() != *support
                || snapshot.field() != *field
                || snapshot.digest()? != *snapshot_identity
            {
                return Err(replay_error(
                    "remeshing replay snapshot differs from its exact SpatialState Field edge",
                ));
            }
            used_snapshots.insert(snapshot_identity.clone());
            let mut resolved_blocks = Vec::new();
            for (association, block_identity) in snapshot.block_artifacts() {
                let block = resolve_one(blocks, &block_identity, "DiscreteField block")?;
                if block.association() != association || block.digest()? != block_identity {
                    return Err(replay_error(
                        "remeshing replay block differs from its exact FieldSnapshot edge",
                    ));
                }
                used_blocks.insert(block_identity);
                resolved_blocks.push(block);
            }
            Ok(ReplayedField {
                snapshot,
                blocks: resolved_blocks,
            })
        })
        .collect()
}

fn require_complete_catalog(declared: usize, used: usize, label: &str) -> Result<(), Diagnostic> {
    if declared == used {
        Ok(())
    } else {
        Err(replay_error(format!(
            "remeshing replay {label} catalog has {declared} declarations but exact replay uses {used}"
        )))
    }
}

/// Independent native HDF5 and pure XDMF producer budgets.
#[cfg(feature = "hdf5")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct XdmfHdf5TrajectoryExportLimits {
    /// Complete HDF5 file-image writer budgets.
    pub hdf5: Hdf5WriteLimits,
    /// Complete XDMF Temporal Collection renderer budgets.
    pub xdmf: XdmfTemporalExportLimits,
}

/// Freshly derived trajectory storage projection ready for caller persistence.
#[cfg(feature = "hdf5")]
#[derive(Debug)]
pub struct XdmfHdf5TrajectoryExportArtifactsV1 {
    envelope: XdmfHdf5TrajectoryStorageEnvelope,
    envelope_digest: ArtifactDigest,
    xdmf_bytes: Vec<u8>,
    hdf5_bytes: Vec<u8>,
}

#[cfg(feature = "hdf5")]
impl XdmfHdf5TrajectoryExportArtifactsV1 {
    /// Exact format-specific storage lineage.
    #[must_use]
    pub const fn envelope(&self) -> &XdmfHdf5TrajectoryStorageEnvelope {
        &self.envelope
    }

    /// Exact envelope identity computed during fresh derivation.
    #[must_use]
    pub const fn envelope_digest(&self) -> &ArtifactDigest {
        &self.envelope_digest
    }

    /// Complete deterministic XDMF metadata bytes.
    #[must_use]
    pub fn xdmf_bytes(&self) -> &[u8] {
        &self.xdmf_bytes
    }

    /// Complete native HDF5 file-image bytes.
    #[must_use]
    pub fn hdf5_bytes(&self) -> &[u8] {
        &self.hdf5_bytes
    }
}

/// Opaque proof of fresh equality with independently loaded storage outputs.
#[cfg(feature = "hdf5")]
#[derive(Debug)]
pub struct VerifiedXdmfHdf5TrajectoryExportV1 {
    artifacts: XdmfHdf5TrajectoryExportArtifactsV1,
}

#[cfg(feature = "hdf5")]
impl VerifiedXdmfHdf5TrajectoryExportV1 {
    /// Fresh exact replay artifacts.
    #[must_use]
    pub const fn artifacts(&self) -> &XdmfHdf5TrajectoryExportArtifactsV1 {
        &self.artifacts
    }
}

/// Project one exact durable remeshing trajectory into XDMF/HDF5 bytes.
///
/// The display locator is rendered but never opened. The returned values grant
/// no filesystem or network authority.
///
/// # Errors
/// Returns `EQ0811` for projection drift or a producer budget excess. Exact
/// artifact and adapter diagnostics retain their narrower codes.
#[cfg(feature = "hdf5")]
pub fn export_xdmf_hdf5_trajectory_v1<M: ReplayableCanonicalModelArtifact>(
    input: &RemeshingTrajectoryReplayInputV1<'_, M>,
    hdf5_display_locator: &str,
    limits: XdmfHdf5TrajectoryExportLimits,
) -> Result<XdmfHdf5TrajectoryExportArtifactsV1, Diagnostic> {
    derive_export(input, hdf5_display_locator, limits)
}

/// Freshly replay one expected envelope and complete output pair.
///
/// # Errors
/// Returns `EQ0811` for any substituted envelope, trajectory dependency,
/// XDMF document, HDF5 image, or producer result.
#[cfg(feature = "hdf5")]
pub fn verify_xdmf_hdf5_trajectory_storage_v1<M: ReplayableCanonicalModelArtifact>(
    expected_envelope: &XdmfHdf5TrajectoryStorageEnvelope,
    expected_xdmf_bytes: &[u8],
    expected_hdf5_bytes: &[u8],
    input: &RemeshingTrajectoryReplayInputV1<'_, M>,
    hdf5_display_locator: &str,
    limits: XdmfHdf5TrajectoryExportLimits,
) -> Result<VerifiedXdmfHdf5TrajectoryExportV1, Diagnostic> {
    expected_envelope.validate_outputs(
        input.trajectory,
        expected_xdmf_bytes,
        expected_hdf5_bytes,
    )?;
    let artifacts = derive_export(input, hdf5_display_locator, limits)?;
    if expected_envelope != &artifacts.envelope
        || expected_xdmf_bytes != artifacts.xdmf_bytes
        || expected_hdf5_bytes != artifacts.hdf5_bytes
    {
        return Err(export_error(
            "persisted XDMF/HDF5 trajectory storage differs from fresh exact replay",
        ));
    }
    Ok(VerifiedXdmfHdf5TrajectoryExportV1 { artifacts })
}

#[cfg(feature = "hdf5")]
fn derive_export<M: ReplayableCanonicalModelArtifact>(
    input: &RemeshingTrajectoryReplayInputV1<'_, M>,
    locator: &str,
    limits: XdmfHdf5TrajectoryExportLimits,
) -> Result<XdmfHdf5TrajectoryExportArtifactsV1, Diagnostic> {
    let mut projection = TemporalProjection::default();
    let mut storage_frames = Vec::new();

    let frames = input
        .source_frames
        .iter()
        .take(input.source_frames.len() - 1)
        .map(|frame| (frame, input.source_context.mesh()))
        .chain(
            input
                .target_frames
                .iter()
                .map(|frame| (frame, input.target_context.mesh())),
        );
    for (frame, mesh) in frames {
        let ordinal = portable_ordinal(projection.frames.len())?;
        projection.push_frame(
            ordinal,
            frame.state.time_s(),
            frame.geometry.digest()?,
            mesh,
            frame.geometry.current_coordinates_m(),
            &frame.fields,
        )?;
        storage_frames.push(XdmfHdf5TrajectoryFrameV1::from_state(
            ordinal,
            frame.state,
            storage_fields(&frame.fields)?,
        )?);
    }

    let writes = dataset_writes(&projection.datasets)?;
    let hdf5 = write_hdf5_file_image(&writes, limits.hdf5)?;
    let xdmf = XdmfTemporalExportPlan::new(locator, projection.frames, limits.xdmf)?;
    let xdmf_bytes = xdmf.metadata_bytes().to_vec();
    let runtime_stack = runtime_stack(hdf5.runtime())?;
    let hdf5_bytes = hdf5.into_bytes();
    let envelope = XdmfHdf5TrajectoryStorageEnvelope::new(
        ExternalAdapterIdentityV1::new(ADAPTER_ID, ADAPTER_VERSION)?,
        runtime_stack,
        input.trajectory,
        &xdmf_bytes,
        &hdf5_bytes,
        storage_frames,
    )?;
    let envelope_digest = envelope.digest()?;
    Ok(XdmfHdf5TrajectoryExportArtifactsV1 {
        envelope,
        envelope_digest,
        xdmf_bytes,
        hdf5_bytes,
    })
}

fn validate_source_segments<M: ReplayableCanonicalModelArtifact>(
    context: &ValidatedMovingSpatialContext<'_, M>,
    segments: &[SpatialTrajectorySegmentEnvelopeV2],
    frames: &[MovingSpatialFrame<'_>],
) -> Result<(), Diagnostic> {
    let mut offset = 0_usize;
    for segment in segments {
        let end = offset
            .checked_add(segment.state_count())
            .ok_or_else(|| replay_error("source trajectory state count overflows usize"))?;
        let states = frames
            .get(offset..end)
            .ok_or_else(|| replay_error("source trajectory is missing exact frame states"))?
            .iter()
            .map(|frame| frame.state.clone())
            .collect::<Vec<_>>();
        segment.validate_against(context, &states)?;
        offset = end;
    }
    if offset != frames.len() {
        return Err(replay_error(
            "source frame inventory exceeds the exact trajectory segments",
        ));
    }
    Ok(())
}

fn validate_target_segments<M: ReplayableCanonicalModelArtifact>(
    context: &ValidatedMovingSpatialContext<'_, M>,
    segments: &[SpatialTrajectorySegmentEnvelopeV3],
    frames: &[MovingSpatialFrame<'_>],
) -> Result<(), Diagnostic> {
    let mut offset = 0_usize;
    for segment in segments {
        let count = segment.state_artifacts().len();
        let end = offset
            .checked_add(count)
            .ok_or_else(|| replay_error("target trajectory state count overflows usize"))?;
        let states = frames
            .get(offset..end)
            .ok_or_else(|| replay_error("target trajectory is missing exact frame states"))?
            .iter()
            .map(|frame| frame.state.clone())
            .collect::<Vec<_>>();
        segment.validate_states(context, &states)?;
        offset = end;
    }
    if offset != frames.len() {
        return Err(replay_error(
            "target frame inventory exceeds the exact trajectory segments",
        ));
    }
    Ok(())
}

fn validate_source_frames<M: ReplayableCanonicalModelArtifact>(
    context: &ValidatedMovingSpatialContext<'_, M>,
    frames: &[MovingSpatialFrame<'_>],
) -> Result<(), Diagnostic> {
    for (index, frame) in frames.iter().enumerate() {
        if frame.state.geometry_state_artifact() != frame.geometry.digest()?
            || frame.state.step() != frame.geometry.step()
            || frame.state.time_s() != frame.geometry.time_s()
            || frame.state.reference_mesh_artifact() != context.mesh().digest()?
            || frame.geometry.reference_mesh_artifact() != context.mesh().digest()?
        {
            return Err(replay_error(
                "source temporal frame differs from its exact state, geometry, or mesh",
            ));
        }
        let snapshots = snapshot_objects(&frame.fields);
        frame.state.validate_against(
            context,
            frame.geometry,
            index
                .checked_sub(1)
                .and_then(|previous| frames.get(previous))
                .map(|previous| (previous.state, previous.geometry)),
            &snapshots,
            driver_blocks(&frame.fields, &frame.geometry.driver_snapshot())?,
        )?;
        for field in &frame.fields {
            field
                .snapshot
                .validate_against_moving(context, field.blocks.iter().copied())?;
        }
    }
    Ok(())
}

fn snapshot_objects(fields: &[ReplayedField<'_>]) -> Vec<FieldSnapshotEnvelopeV2> {
    fields.iter().map(|field| field.snapshot.clone()).collect()
}

fn field_by_digest<'a>(
    fields: &'a [ReplayedField<'a>],
    digest: &ArtifactDigest,
) -> Result<&'a ReplayedField<'a>, Diagnostic> {
    for field in fields {
        if field.snapshot.digest()? == *digest {
            return Ok(field);
        }
    }
    Err(replay_error(
        "temporal frame omits its exact geometry-driver snapshot",
    ))
}

fn validate_target_frames<M: ReplayableCanonicalModelArtifact>(
    context: &ValidatedMovingSpatialContext<'_, M>,
    frames: &[MovingSpatialFrame<'_>],
) -> Result<(), Diagnostic> {
    for frame in frames {
        if frame.state.geometry_state_artifact() != frame.geometry.digest()?
            || frame.state.step() != frame.geometry.step()
            || frame.state.time_s() != frame.geometry.time_s()
            || frame.state.reference_mesh_artifact() != context.mesh().digest()?
            || frame.geometry.reference_mesh_artifact() != context.mesh().digest()?
        {
            return Err(replay_error(
                "target temporal frame differs from its exact state, geometry, or mesh",
            ));
        }
        for field in &frame.fields {
            field
                .snapshot
                .validate_against_moving(context, field.blocks.iter().copied())?;
        }
    }
    Ok(())
}

#[cfg(feature = "hdf5")]
fn storage_fields(
    fields: &[ReplayedField<'_>],
) -> Result<Vec<XdmfHdf5TrajectoryFieldV1>, Diagnostic> {
    fields
        .iter()
        .map(|field| storage_field(field.snapshot, &field.blocks))
        .collect()
}

#[cfg(feature = "hdf5")]
fn storage_field(
    snapshot: &FieldSnapshotEnvelopeV2,
    blocks: &[&DiscreteFieldEnvelopeV1],
) -> Result<XdmfHdf5TrajectoryFieldV1, Diagnostic> {
    XdmfHdf5TrajectoryFieldV1::new(
        snapshot,
        blocks
            .iter()
            .map(|block| {
                (
                    *block,
                    match block.association() {
                        DiscreteFieldAssociation::Vertex => {
                            TemporalStorageBlockPresentationV1::XdmfNodeAttribute
                        }
                        DiscreteFieldAssociation::Cell => {
                            TemporalStorageBlockPresentationV1::Hidden
                        }
                    },
                )
            })
            .collect(),
    )
}

#[cfg(feature = "hdf5")]
#[derive(Debug, Default)]
struct TemporalProjection {
    datasets: BTreeMap<String, OwnedDataset>,
    frames: Vec<XdmfTemporalFrame>,
}

#[cfg(feature = "hdf5")]
impl TemporalProjection {
    fn push_frame(
        &mut self,
        ordinal: u64,
        time_s: f64,
        geometry_state: ArtifactDigest,
        mesh: &eqiora_artifact::SimplicialMeshEnvelopeV1,
        current_coordinates: &[Vec<f64>],
        fields: &[ReplayedField<'_>],
    ) -> Result<(), Diagnostic> {
        if mesh.dimension() != 2
            || mesh.mesh().cells().iter().any(|cell| cell.len() != 3)
            || current_coordinates.len() != mesh.mesh().vertices().len()
            || current_coordinates.iter().any(|point| point.len() != 2)
        {
            return Err(export_error(
                "first temporal export profile requires complete current 2D affine Tri3 geometry",
            ));
        }
        let mesh_digest = mesh.digest()?;
        let topology_path = format!("/meshes/{mesh_digest}/topology");
        let topology = mesh
            .mesh()
            .cells()
            .iter()
            .flatten()
            .map(|vertex| {
                u64::try_from(*vertex)
                    .map_err(|_| export_error("mesh vertex index exceeds portable HDF5 u64"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        insert_dataset(
            &mut self.datasets,
            topology_path.clone(),
            OwnedDataset::U64 {
                shape: vec![portable_extent(mesh.mesh().cells().len())?, 3],
                values: topology,
            },
        )?;

        let geometry_path = format!("/geometry/{geometry_state}/coordinates");
        insert_dataset(
            &mut self.datasets,
            geometry_path.clone(),
            OwnedDataset::F64 {
                shape: vec![portable_extent(current_coordinates.len())?, 2],
                values: current_coordinates.iter().flatten().copied().collect(),
            },
        )?;

        let mut xdmf_fields = Vec::new();
        for field in fields {
            for block in &field.blocks {
                let digest = block.digest()?;
                let path = format!("/fields/{digest}/values");
                let components = block.component_shape().component_count()?;
                let shape = if components == 1 {
                    vec![portable_extent(block.entity_count()?)?]
                } else {
                    vec![
                        portable_extent(block.entity_count()?)?,
                        portable_extent(components)?,
                    ]
                };
                insert_dataset(
                    &mut self.datasets,
                    path.clone(),
                    OwnedDataset::F64 {
                        shape,
                        values: block.values().to_vec(),
                    },
                )?;
                if block.association() == DiscreteFieldAssociation::Vertex {
                    xdmf_fields.push(XdmfTemporalField::new(
                        field.snapshot.field().ulid().to_string(),
                        block.association(),
                        block.component_shape(),
                        path,
                    )?);
                }
            }
        }
        self.frames.push(XdmfTemporalFrame::new(
            ordinal,
            time_s,
            2,
            mesh.mesh().vertices().len(),
            mesh.mesh().cells().len(),
            geometry_path,
            topology_path,
            xdmf_fields,
        )?);
        Ok(())
    }
}

#[cfg(feature = "hdf5")]
#[derive(Debug, Clone, PartialEq)]
enum OwnedDataset {
    U64 { shape: Vec<u64>, values: Vec<u64> },
    F64 { shape: Vec<u64>, values: Vec<f64> },
}

#[cfg(feature = "hdf5")]
fn insert_dataset(
    datasets: &mut BTreeMap<String, OwnedDataset>,
    path: String,
    dataset: OwnedDataset,
) -> Result<(), Diagnostic> {
    if let Some(existing) = datasets.get(&path) {
        if existing != &dataset {
            return Err(export_error(
                "one content-addressed HDF5 path resolves to different arrays",
            ));
        }
    } else {
        datasets.insert(path, dataset);
    }
    Ok(())
}

#[cfg(feature = "hdf5")]
fn dataset_writes(
    datasets: &BTreeMap<String, OwnedDataset>,
) -> Result<Vec<Hdf5DatasetWrite<'_>>, Diagnostic> {
    datasets
        .iter()
        .map(|(path, dataset)| match dataset {
            OwnedDataset::U64 { shape, values } => {
                Hdf5DatasetWrite::u64(path, shape.clone(), values)
            }
            OwnedDataset::F64 { shape, values } => {
                Hdf5DatasetWrite::f64(path, shape.clone(), values)
            }
        })
        .collect()
}

#[cfg(feature = "hdf5")]
fn runtime_stack(
    runtime: &Hdf5RuntimeIdentity,
) -> Result<Vec<ExternalRuntimeComponentV1>, Diagnostic> {
    Ok(vec![
        ExternalRuntimeComponentV1::new(
            ExternalRuntimeRoleV1::RustBinding,
            runtime.binding_id(),
            runtime.binding_version(),
        )?,
        ExternalRuntimeComponentV1::new(
            ExternalRuntimeRoleV1::NativeStorageLibrary,
            runtime.native_library_id(),
            runtime.native_library_version(),
        )?,
    ])
}

#[cfg(feature = "hdf5")]
fn portable_ordinal(value: usize) -> Result<u64, Diagnostic> {
    u64::try_from(value).map_err(|_| export_error("temporal frame ordinal exceeds portable u64"))
}

#[cfg(feature = "hdf5")]
fn portable_extent(value: usize) -> Result<u64, Diagnostic> {
    u64::try_from(value).map_err(|_| export_error("HDF5 extent exceeds portable u64"))
}

fn replay_error(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_ARTIFACT, message)
}

#[cfg(feature = "hdf5")]
fn export_error(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_EXTERNAL_DATA_EXPORT, message)
}

fn driver_blocks<'a>(
    fields: &'a [ReplayedField<'a>],
    digest: &ArtifactDigest,
) -> Result<impl Iterator<Item = &'a DiscreteFieldEnvelopeV1>, Diagnostic> {
    Ok(field_by_digest(fields, digest)?.blocks.iter().copied())
}
