//! Exact replay of one remeshing trajectory and its optional storage projection.

use std::collections::{BTreeMap, BTreeSet};

#[cfg(feature = "hdf5")]
use crate::xdmf_export::{ExportField, Projection, export_error, physical_field_information};
use eqiora_artifact::{
    ArtifactDigest, DiscreteFieldEnvelopeV1, FieldSnapshotEnvelopeV2,
    GeometryRevisionAssociationEnvelopeV1, GeometryStateEnvelope, MeshRevisionOverlapEnvelopeV1,
    RemeshTransferReceiptEnvelopeV1, ReplayableCanonicalModelArtifact, SpatialStateEnvelope,
    SpatialStateOriginKind, SpatialTrajectoryEnvelope, SpatialTrajectorySegmentEnvelope,
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
use eqiora_io_hdf5::{Hdf5RuntimeIdentity, Hdf5WriteLimits};
#[cfg(feature = "hdf5")]
use eqiora_io_xml::XdmfExportLimits;
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

/// Complete borrowed dependency replay for publication across one 2D remesh.
/// The trajectory includes the entire source and target prefix.
#[derive(Debug)]
pub struct RemeshingTrajectoryReplayInput<'a, M: ReplayableCanonicalModelArtifact> {
    source_context: &'a ValidatedMovingSpatialContext<'a, M>,
    target_context: &'a ValidatedMovingSpatialContext<'a, M>,
    source_frames: Vec<MovingSpatialFrame<'a>>,
    trajectory: &'a SpatialTrajectoryEnvelope,
    target_frames: Vec<MovingSpatialFrame<'a>>,
}

impl<'a, M: ReplayableCanonicalModelArtifact> RemeshingTrajectoryReplayInput<'a, M> {
    /// Replay every root, segment, state, geometry, snapshot, and block edge.
    ///
    /// # Errors
    /// Returns `EQ0901` for any missing, stale, substituted, reordered, or
    /// cross-context dependency.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        source_context: &'a ValidatedMovingSpatialContext<'a, M>,
        target_context: &'a ValidatedMovingSpatialContext<'a, M>,
        trajectory: &'a SpatialTrajectoryEnvelope,
        segments: &'a [SpatialTrajectorySegmentEnvelope],
        states: &'a [SpatialStateEnvelope],
        geometry_states: &'a [GeometryStateEnvelope],
        snapshots: &'a [FieldSnapshotEnvelopeV2],
        blocks: &'a [DiscreteFieldEnvelopeV1],
        association: &'a GeometryRevisionAssociationEnvelopeV1,
        overlap: &'a MeshRevisionOverlapEnvelopeV1,
        receipt: &'a RemeshTransferReceiptEnvelopeV1,
    ) -> Result<Self, Diagnostic> {
        let segment_index = index_catalog(
            segments,
            SpatialTrajectorySegmentEnvelope::digest,
            "trajectory segment",
        )?;
        let segments = resolve_catalog(
            &segment_index,
            &trajectory.segment_artifacts(),
            "trajectory segment",
        )?
        .into_iter()
        .cloned()
        .collect::<Vec<_>>();
        require_complete_catalog(segment_index.len(), segments.len(), "trajectory segment")?;
        trajectory.validate_segments(&segments)?;
        let seams = segments
            .iter()
            .enumerate()
            .filter(|(_, segment)| segment.origin() == SpatialStateOriginKind::Remesh)
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        let [seam] = seams.as_slice() else {
            return Err(replay_error("publication requires exactly one remesh seam"));
        };
        let (source_segments, target_segments) = segments.split_at(*seam);

        let state_index = index_catalog(states, SpatialStateEnvelope::digest, "spatial state")?;
        let state_ids = segments
            .iter()
            .flat_map(SpatialTrajectorySegmentEnvelope::state_artifacts)
            .collect::<Vec<_>>();
        let ordered_states = resolve_catalog(&state_index, &state_ids, "spatial state")?;
        require_complete_catalog(state_index.len(), ordered_states.len(), "spatial state")?;
        let geometry_index = index_catalog(
            geometry_states,
            GeometryStateEnvelope::digest,
            "geometry state",
        )?;
        let used_geometries = ordered_states
            .iter()
            .map(|state| state.geometry_state_artifact())
            .collect::<BTreeSet<_>>();
        require_complete_catalog(
            geometry_index.len(),
            used_geometries.len(),
            "geometry state",
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
        let mut source_frames = ordered_states
            .iter()
            .map(|state| {
                Ok(MovingSpatialFrame {
                    state,
                    geometry: resolve_one(
                        &geometry_index,
                        &state.geometry_state_artifact(),
                        "geometry state",
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
        let source_count = source_segments
            .iter()
            .map(SpatialTrajectorySegmentEnvelope::state_count)
            .sum();
        let target_frames = source_frames.split_off(source_count);
        require_complete_catalog(snapshot_index.len(), used_snapshots.len(), "Field snapshot")?;
        require_complete_catalog(block_index.len(), used_blocks.len(), "DiscreteField block")?;

        validate_segments(source_context, source_segments, &source_frames)?;
        validate_segments(target_context, target_segments, &target_frames)?;
        validate_source_frames(source_context, &source_frames)?;
        validate_target_frames(target_context, &target_frames)?;

        let source_tip = source_frames
            .last()
            .ok_or_else(|| replay_error("remeshing replay source trajectory is empty"))?;
        let target_first = target_frames
            .first()
            .ok_or_else(|| replay_error("remeshing replay target trajectory is empty"))?;
        if target_first.state.origin() != SpatialStateOriginKind::Remesh
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
    pub const fn trajectory(&self) -> &'a SpatialTrajectoryEnvelope {
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
pub struct XdmfHdf5ExportLimits {
    /// Complete HDF5 file-image writer budgets.
    pub hdf5: Hdf5WriteLimits,
    /// Complete XDMF Temporal Collection renderer budgets.
    pub xdmf: XdmfExportLimits,
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

#[cfg(feature = "hdf5")]
impl<M: ReplayableCanonicalModelArtifact> RemeshingTrajectoryReplayInput<'_, M> {
    /// Freshly replay one expected envelope and complete output pair.
    ///
    /// # Errors
    /// Returns `EQ0811` for any substituted envelope, trajectory dependency,
    /// XDMF document, HDF5 image, or producer result.
    pub fn verify_storage(
        &self,
        expected_envelope: &XdmfHdf5TrajectoryStorageEnvelope,
        expected_xdmf_bytes: &[u8],
        expected_hdf5_bytes: &[u8],
        hdf5_display_locator: &str,
        limits: XdmfHdf5ExportLimits,
    ) -> Result<VerifiedXdmfHdf5TrajectoryExportV1, Diagnostic> {
        expected_envelope.validate_outputs(
            self.trajectory,
            expected_xdmf_bytes,
            expected_hdf5_bytes,
        )?;
        let artifacts = self.export(hdf5_display_locator, limits)?;
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

    /// Project this completely replayed trajectory into XDMF/HDF5 bytes.
    /// The display locator is rendered but never opened; no filesystem authority is granted.
    pub fn export(
        &self,
        locator: &str,
        limits: XdmfHdf5ExportLimits,
    ) -> Result<XdmfHdf5TrajectoryExportArtifactsV1, Diagnostic> {
        let mut projection = Projection::default();
        let mut storage_frames = Vec::new();

        let frames = self
            .source_frames
            .iter()
            .take(self.source_frames.len() - 1)
            .map(|frame| (frame, self.source_context.mesh()))
            .chain(
                self.target_frames
                    .iter()
                    .map(|frame| (frame, self.target_context.mesh())),
            );
        for (frame, mesh) in frames {
            let ordinal = portable_ordinal(projection.frame_count())?;
            let mut fields = Vec::new();
            for field in &frame.fields {
                for block in &field.blocks {
                    fields.push(ExportField {
                        name: field.snapshot.field().ulid().to_string(),
                        identity: block.digest()?,
                        association: block.association(),
                        shape: block.component_shape(),
                        values: block.values(),
                        visible: block.association() == DiscreteFieldAssociation::Vertex,
                        information: field_information(field)?,
                    });
                }
            }
            projection.push_frame(
                ordinal,
                Some(frame.state.time_s()),
                frame.geometry.digest()?,
                mesh,
                frame.geometry.current_coordinates_m(),
                &fields,
                BTreeMap::from([
                    (
                        "eqiora.trajectory".into(),
                        self.trajectory.digest()?.to_string(),
                    ),
                    ("eqiora.state".into(), frame.state.digest()?.to_string()),
                ]),
            )?;
            storage_frames.push(XdmfHdf5TrajectoryFrameV1::from_state(
                ordinal,
                frame.state,
                storage_fields(&frame.fields)?,
            )?);
        }

        let (xdmf, hdf5) = projection.finish(locator, limits.xdmf, limits.hdf5)?;
        let xdmf_bytes = xdmf.metadata_bytes().to_vec();
        let runtime_stack = runtime_stack(hdf5.runtime())?;
        let hdf5_bytes = hdf5.into_bytes();
        let envelope = XdmfHdf5TrajectoryStorageEnvelope::new(
            ExternalAdapterIdentityV1::new(ADAPTER_ID, ADAPTER_VERSION)?,
            runtime_stack,
            self.trajectory,
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
}

#[cfg(feature = "hdf5")]
fn field_information(field: &ReplayedField<'_>) -> Result<BTreeMap<String, String>, Diagnostic> {
    let snapshot = field.snapshot;
    let mut information = physical_field_information(
        snapshot.dimension(),
        snapshot.frame(),
        DiscreteFieldAssociation::Vertex,
    );
    information.extend([
        (
            "eqiora.field-snapshot".into(),
            snapshot.digest()?.to_string(),
        ),
        ("eqiora.model".into(), snapshot.model_artifact().to_string()),
        (
            "eqiora.realization".into(),
            snapshot.realization_artifact().to_string(),
        ),
        (
            "eqiora.support-domain".into(),
            snapshot.support_domain().ulid().to_string(),
        ),
    ]);
    if let Some(bubbles) = field
        .blocks
        .iter()
        .find(|block| block.association() == DiscreteFieldAssociation::Cell)
    {
        information.insert(
            "eqiora.hidden-cell-bubbles".into(),
            format!("/fields/{}/values", bubbles.digest()?),
        );
    }
    Ok(information)
}

fn validate_segments<M: ReplayableCanonicalModelArtifact>(
    context: &ValidatedMovingSpatialContext<'_, M>,
    segments: &[SpatialTrajectorySegmentEnvelope],
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

fn replay_error(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_ARTIFACT, message)
}

fn driver_blocks<'a>(
    fields: &'a [ReplayedField<'a>],
    digest: &ArtifactDigest,
) -> Result<impl Iterator<Item = &'a DiscreteFieldEnvelopeV1>, Diagnostic> {
    Ok(field_by_digest(fields, digest)?.blocks.iter().copied())
}
