//! Exact lineage for one external temporal-storage projection.

use std::collections::BTreeSet;
use std::str::FromStr;

use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, Id};
use eqiora_meshing::DiscreteFieldAssociation;
use serde::{Deserialize, Serialize};
use ulid::Ulid;

use crate::discrete_field::WireAssociation;
use crate::external_import::wire::{WireAdapter, WireRuntimeComponent, WireRuntimeRole};
use crate::{
    ArtifactDigest, CANONICAL_ENCODING, DiscreteFieldEnvelopeV1, ExternalAdapterIdentityV1,
    ExternalRuntimeComponentV1, FieldSnapshotEnvelopeV2, SpatialStateEnvelope,
    SpatialStateTopology, SpatialTrajectoryEnvelopeV3, StorageChunkSha256V1, check_json_limits,
    invalid_artifact,
};

const SCHEMA: &str = "eqiora.xdmf-hdf5-trajectory-storage/v2";

/// Semantic work budgets for external trajectory-storage artifacts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrajectoryStorageDecoderLimits {
    /// Common JSON syntax admission.
    pub json: crate::JsonDecoderLimits,
    /// Maximum runtime components in one storage envelope.
    pub max_trajectory_storage_runtime_entries: usize,
    /// Maximum frames in one storage envelope.
    pub max_trajectory_storage_frames: usize,
    /// Maximum Field entries summed across one storage envelope.
    pub max_trajectory_storage_fields: usize,
    /// Maximum coefficient blocks summed across one storage envelope.
    pub max_trajectory_storage_blocks: usize,
    /// Maximum dynamic UTF-8 text bytes in one storage envelope.
    pub max_trajectory_storage_text_bytes: usize,
    /// Maximum complete XDMF document bytes asserted by one storage envelope.
    pub max_xdmf_storage_bytes: u64,
    /// Maximum complete HDF5 file-image bytes asserted by one storage envelope.
    pub max_hdf5_storage_bytes: u64,
}

impl Default for TrajectoryStorageDecoderLimits {
    fn default() -> Self {
        Self {
            json: crate::JsonDecoderLimits::default(),
            max_trajectory_storage_runtime_entries: 32,
            max_trajectory_storage_frames: 16_384,
            max_trajectory_storage_fields: 1_000_000,
            max_trajectory_storage_blocks: 2_000_000,
            max_trajectory_storage_text_bytes: 64 * 1024 * 1024,
            max_xdmf_storage_bytes: 16 * 1024 * 1024,
            max_hdf5_storage_bytes: 512 * 1024 * 1024,
        }
    }
}
const SEAM_POLICY: &str = "target-replaces-source-at-remesh";

/// Truthful presentation status of one losslessly stored coefficient block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TemporalStorageBlockPresentationV1 {
    /// Exposed as a node-centered XDMF Attribute.
    XdmfNodeAttribute,
    /// Stored in HDF5 but deliberately absent from XDMF presentation.
    Hidden,
}

/// One exact coefficient block and its canonical HDF5 location.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XdmfHdf5TrajectoryBlockV1 {
    wire: WireBlock,
}

impl XdmfHdf5TrajectoryBlockV1 {
    /// Mesh association of the coefficient block.
    #[must_use]
    pub const fn association(&self) -> DiscreteFieldAssociation {
        self.wire.association.decode()
    }

    /// Exact logical DiscreteField artifact.
    #[must_use]
    pub fn logical_field_artifact(&self) -> ArtifactDigest {
        parse_digest(&self.wire.logical_discrete_field_sha256)
    }

    /// Canonical content-addressed HDF5 dataset path.
    #[must_use]
    pub fn dataset_path(&self) -> &str {
        &self.wire.hdf5_dataset_path
    }

    /// Truthful XDMF presentation status.
    #[must_use]
    pub const fn presentation(&self) -> TemporalStorageBlockPresentationV1 {
        self.wire.presentation
    }
}

/// One Semantic Field snapshot and all of its losslessly stored blocks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XdmfHdf5TrajectoryFieldV1 {
    wire: WireField,
}

impl XdmfHdf5TrajectoryFieldV1 {
    /// Construct one exact snapshot and complete coefficient inventory.
    ///
    /// # Errors
    /// Returns `EQ0901` for substituted, missing, duplicate, cross-mesh, or
    /// untruthfully presented coefficient blocks.
    pub fn new(
        snapshot: &FieldSnapshotEnvelopeV2,
        blocks: Vec<(&DiscreteFieldEnvelopeV1, TemporalStorageBlockPresentationV1)>,
    ) -> Result<Self, Diagnostic> {
        let mut blocks = blocks
            .into_iter()
            .map(|(block, presentation)| {
                if block.mesh_artifact() != snapshot.mesh_artifact() {
                    return Err(invalid_artifact(
                        "temporal storage coefficient block references a different snapshot mesh",
                    ));
                }
                let digest = block.digest()?;
                let wire = WireBlock {
                    association: WireAssociation::encode(block.association()),
                    logical_discrete_field_sha256: digest.to_string(),
                    hdf5_dataset_path: format!("/fields/{digest}/values"),
                    presentation,
                };
                validate_block(&wire)?;
                Ok(wire)
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        blocks.sort_by_key(|block| block.association);
        let wire = WireField {
            support_domain_ulid: snapshot.support_domain().ulid().to_string(),
            field_ulid: snapshot.field().ulid().to_string(),
            snapshot_sha256: snapshot.digest()?.to_string(),
            blocks,
        };
        validate_field(&wire)?;
        let expected = snapshot.block_artifacts();
        let actual = wire
            .blocks
            .iter()
            .map(|block| {
                (
                    block.association.decode(),
                    ArtifactDigest::from_hex(block.logical_discrete_field_sha256.clone())
                        .expect("validated temporal storage block digest"),
                )
            })
            .collect::<Vec<_>>();
        if actual != expected {
            return Err(invalid_artifact(
                "temporal storage blocks differ from the complete exact Field snapshot",
            ));
        }
        Ok(Self { wire })
    }

    /// Semantic support Domain.
    #[must_use]
    pub fn support_domain(&self) -> Id<kinds::Domain> {
        parse_id(&self.wire.support_domain_ulid, "support Domain")
            .expect("validated temporal storage Domain")
    }

    /// Semantic Field identity.
    #[must_use]
    pub fn field(&self) -> Id<kinds::Field> {
        parse_id(&self.wire.field_ulid, "Field").expect("validated temporal storage Field")
    }

    /// Exact FieldSnapshot artifact.
    #[must_use]
    pub fn snapshot_artifact(&self) -> ArtifactDigest {
        parse_digest(&self.wire.snapshot_sha256)
    }

    /// Canonically association-ordered coefficient blocks.
    #[must_use]
    pub fn blocks(&self) -> Vec<XdmfHdf5TrajectoryBlockV1> {
        self.wire
            .blocks
            .iter()
            .cloned()
            .map(|wire| XdmfHdf5TrajectoryBlockV1 { wire })
            .collect()
    }
}

/// One canonical external frame after applying the closed remesh seam policy.
#[derive(Debug, Clone, PartialEq)]
pub struct XdmfHdf5TrajectoryFrameV1 {
    wire: WireFrame,
}

impl XdmfHdf5TrajectoryFrameV1 {
    /// Construct one external frame from the exact current state.
    /// # Errors
    /// Rejects a missing, substituted or reordered Field.
    pub fn from_state(
        ordinal: u64,
        state: &SpatialStateEnvelope,
        fields: Vec<XdmfHdf5TrajectoryFieldV1>,
    ) -> Result<Self, Diagnostic> {
        let step = state.step();
        let time_s = state.time_s();
        let state_kind = state.topology();
        let spatial_state = state.digest()?;
        let reference_mesh = state.reference_mesh_artifact();
        let geometry_state = state.geometry_state_artifact();
        let expected_fields = state.fields();
        let mut fields = fields
            .into_iter()
            .map(|field| field.wire)
            .collect::<Vec<_>>();
        fields.sort_by(|left, right| left.field_ulid.cmp(&right.field_ulid));
        let wire = WireFrame {
            ordinal,
            step,
            time_s,
            state_kind,
            spatial_state_sha256: spatial_state.to_string(),
            reference_mesh_sha256: reference_mesh.to_string(),
            geometry_state_sha256: geometry_state.to_string(),
            fields,
        };
        validate_frame(&wire)?;
        let actual_fields = wire
            .fields
            .iter()
            .map(|field| {
                Ok((
                    parse_id(&field.support_domain_ulid, "support Domain")?,
                    parse_id(&field.field_ulid, "Field")?,
                    ArtifactDigest::from_hex(field.snapshot_sha256.clone())?,
                ))
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        if actual_fields != expected_fields {
            return Err(invalid_artifact(
                "temporal storage Fields differ from the complete exact SpatialState",
            ));
        }
        Ok(Self { wire })
    }

    /// External frame ordinal.
    #[must_use]
    pub const fn ordinal(&self) -> u64 {
        self.wire.ordinal
    }

    /// Accepted trajectory step.
    #[must_use]
    pub const fn step(&self) -> u64 {
        self.wire.step
    }

    /// Accepted coherent-SI time.
    #[must_use]
    pub const fn time_s(&self) -> f64 {
        self.wire.time_s
    }

    /// Durable state generation.
    #[must_use]
    pub const fn state_kind(&self) -> SpatialStateTopology {
        self.wire.state_kind
    }

    /// Exact SpatialState artifact.
    #[must_use]
    pub fn spatial_state_artifact(&self) -> ArtifactDigest {
        parse_digest(&self.wire.spatial_state_sha256)
    }

    /// Exact reference mesh artifact.
    #[must_use]
    pub fn reference_mesh_artifact(&self) -> ArtifactDigest {
        parse_digest(&self.wire.reference_mesh_sha256)
    }

    /// Exact current GeometryState artifact.
    #[must_use]
    pub fn geometry_state_artifact(&self) -> ArtifactDigest {
        parse_digest(&self.wire.geometry_state_sha256)
    }

    /// Semantic Fields in canonical identity order.
    #[must_use]
    pub fn fields(&self) -> Vec<XdmfHdf5TrajectoryFieldV1> {
        self.wire
            .fields
            .iter()
            .cloned()
            .map(|wire| XdmfHdf5TrajectoryFieldV1 { wire })
            .collect()
    }
}

/// Exact format-specific lineage for one XDMF/HDF5 trajectory projection.
#[derive(Debug, Clone, PartialEq)]
pub struct XdmfHdf5TrajectoryStorageEnvelope {
    wire: WireEnvelope,
    adapter: ExternalAdapterIdentityV1,
    runtime_stack: Vec<ExternalRuntimeComponentV1>,
}

impl XdmfHdf5TrajectoryStorageEnvelope {
    /// Capture one complete output from an exact remeshing-aware trajectory.
    ///
    /// Raw digests and byte counts are always derived from the supplied
    /// complete payloads.
    ///
    /// # Errors
    /// Returns `EQ0901` for a noncanonical frame sequence, invalid runtime
    /// identity, Field inventory drift, or an untruthful block presentation.
    pub fn new(
        adapter: ExternalAdapterIdentityV1,
        runtime_stack: Vec<ExternalRuntimeComponentV1>,
        trajectory: &SpatialTrajectoryEnvelopeV3,
        xdmf_bytes: &[u8],
        hdf5_bytes: &[u8],
        frames: Vec<XdmfHdf5TrajectoryFrameV1>,
    ) -> Result<Self, Diagnostic> {
        Self::finish(
            adapter,
            runtime_stack,
            trajectory.digest()?,
            xdmf_bytes,
            hdf5_bytes,
            frames,
            TrajectoryStorageDecoderLimits::default(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn finish(
        adapter: ExternalAdapterIdentityV1,
        runtime_stack: Vec<ExternalRuntimeComponentV1>,
        trajectory: ArtifactDigest,
        xdmf_bytes: &[u8],
        hdf5_bytes: &[u8],
        mut frames: Vec<XdmfHdf5TrajectoryFrameV1>,
        limits: TrajectoryStorageDecoderLimits,
    ) -> Result<Self, Diagnostic> {
        frames.sort_by_key(XdmfHdf5TrajectoryFrameV1::ordinal);
        let wire = WireEnvelope {
            schema: SCHEMA.to_owned(),
            encoding: CANONICAL_ENCODING.to_owned(),
            adapter: WireAdapter::from(&adapter),
            runtime_stack: runtime_stack
                .iter()
                .map(WireRuntimeComponent::from)
                .collect(),
            trajectory_v3_sha256: trajectory.to_string(),
            remesh_seam_policy: SEAM_POLICY.to_owned(),
            xdmf: WirePayload::new(xdmf_bytes)?,
            hdf5: WirePayload::new(hdf5_bytes)?,
            frames: frames.into_iter().map(|frame| frame.wire).collect(),
        };
        validate_wire(&wire, limits)?;
        Ok(Self {
            wire,
            adapter,
            runtime_stack,
        })
    }

    /// Decode one closed bounded storage-lineage DTO without opening outputs.
    ///
    /// # Errors
    /// Returns `EQ0901` for unknown, malformed, noncanonical, or over-budget
    /// data.
    pub fn from_json(
        bytes: &[u8],
        limits: TrajectoryStorageDecoderLimits,
    ) -> Result<Self, Diagnostic> {
        check_json_limits(bytes, limits.json)?;
        let wire: WireEnvelope = serde_json::from_slice(bytes).map_err(|error| {
            invalid_artifact(format!(
                "invalid XDMF/HDF5 trajectory storage JSON: {error}"
            ))
        })?;
        let (adapter, runtime_stack) = validate_wire(&wire, limits)?;
        Ok(Self {
            wire,
            adapter,
            runtime_stack,
        })
    }

    /// Deterministic canonical JSON bytes.
    ///
    /// # Errors
    /// Returns `EQ0901` if serialization unexpectedly fails.
    pub fn canonical_json(&self) -> Result<Vec<u8>, Diagnostic> {
        serde_json::to_vec(&self.wire).map_err(|error| {
            invalid_artifact(format!(
                "cannot serialize XDMF/HDF5 trajectory storage: {error}"
            ))
        })
    }

    /// Domain-separated storage-envelope identity.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
    }

    /// Exact format adapter identity.
    #[must_use]
    pub const fn adapter(&self) -> &ExternalAdapterIdentityV1 {
        &self.adapter
    }

    /// Exact native runtime stack in outer-to-inner order.
    #[must_use]
    pub fn runtime_stack(&self) -> &[ExternalRuntimeComponentV1] {
        &self.runtime_stack
    }

    /// Exact remeshing-aware trajectory artifact.
    #[must_use]
    pub fn trajectory_artifact(&self) -> ArtifactDigest {
        parse_digest(&self.wire.trajectory_v3_sha256)
    }

    /// Fixed same-coordinate seam presentation policy.
    #[must_use]
    pub fn remesh_seam_policy(&self) -> &str {
        &self.wire.remesh_seam_policy
    }

    /// Raw XDMF metadata identity and byte count.
    #[must_use]
    pub fn xdmf_payload(&self) -> (StorageChunkSha256V1, u64) {
        (
            StorageChunkSha256V1::from_hex(self.wire.xdmf.raw_sha256.clone())
                .expect("validated XDMF payload digest"),
            self.wire.xdmf.byte_count,
        )
    }

    /// Raw HDF5 file-image identity and byte count.
    #[must_use]
    pub fn hdf5_payload(&self) -> (StorageChunkSha256V1, u64) {
        (
            StorageChunkSha256V1::from_hex(self.wire.hdf5.raw_sha256.clone())
                .expect("validated HDF5 payload digest"),
            self.wire.hdf5.byte_count,
        )
    }

    /// Canonical external frames.
    #[must_use]
    pub fn frames(&self) -> Vec<XdmfHdf5TrajectoryFrameV1> {
        self.wire
            .frames
            .iter()
            .cloned()
            .map(|wire| XdmfHdf5TrajectoryFrameV1 { wire })
            .collect()
    }

    /// Recompute the trajectory and complete output-byte identities.
    ///
    /// # Errors
    /// Returns `EQ0901` for any substituted trajectory, XML, or HDF5 image.
    pub fn validate_outputs(
        &self,
        trajectory: &SpatialTrajectoryEnvelopeV3,
        xdmf_bytes: &[u8],
        hdf5_bytes: &[u8],
    ) -> Result<(), Diagnostic> {
        let expected = Self::finish(
            self.adapter.clone(),
            self.runtime_stack.clone(),
            trajectory.digest()?,
            xdmf_bytes,
            hdf5_bytes,
            self.frames(),
            TrajectoryStorageDecoderLimits::default(),
        )?;
        if expected == *self {
            Ok(())
        } else {
            Err(invalid_artifact(
                "XDMF/HDF5 trajectory storage differs from exact output replay",
            ))
        }
    }
}

fn validate_wire(
    wire: &WireEnvelope,
    limits: TrajectoryStorageDecoderLimits,
) -> Result<(ExternalAdapterIdentityV1, Vec<ExternalRuntimeComponentV1>), Diagnostic> {
    if wire.schema != SCHEMA
        || wire.encoding != CANONICAL_ENCODING
        || wire.remesh_seam_policy != SEAM_POLICY
    {
        return Err(invalid_artifact(
            "unsupported XDMF/HDF5 trajectory storage schema, encoding, or seam policy",
        ));
    }
    let adapter = wire.adapter.decode()?;
    if wire.runtime_stack.len() != 2
        || wire.runtime_stack.len() > limits.max_trajectory_storage_runtime_entries
        || wire.runtime_stack[0].role != WireRuntimeRole::RustBinding
        || wire.runtime_stack[1].role != WireRuntimeRole::NativeStorageLibrary
    {
        return Err(invalid_artifact(
            "XDMF/HDF5 trajectory runtime stack must be one bounded binding-to-native pair",
        ));
    }
    let runtime_stack = wire
        .runtime_stack
        .iter()
        .map(WireRuntimeComponent::decode)
        .collect::<Result<Vec<_>, _>>()?;
    let mut runtime_identities = BTreeSet::new();
    if runtime_stack
        .iter()
        .any(|entry| !runtime_identities.insert((entry.role(), entry.implementation())))
    {
        return Err(invalid_artifact(
            "XDMF/HDF5 trajectory runtime identities must be unique",
        ));
    }
    ArtifactDigest::from_hex(wire.trajectory_v3_sha256.clone())?;
    validate_payload(&wire.xdmf)?;
    validate_payload(&wire.hdf5)?;
    if wire.xdmf.byte_count > limits.max_xdmf_storage_bytes
        || wire.hdf5.byte_count > limits.max_hdf5_storage_bytes
    {
        return Err(invalid_artifact(
            "trajectory-storage payload byte count exceeds its format-specific decoder limit",
        ));
    }
    if wire.frames.len() < 2 || wire.frames.len() > limits.max_trajectory_storage_frames {
        return Err(invalid_artifact(
            "XDMF/HDF5 trajectory frame count is invalid or over budget",
        ));
    }

    let mut total_fields = 0_usize;
    let mut total_blocks = 0_usize;
    let mut total_text = wire.adapter.id.len() + wire.adapter.version.len();
    for runtime in &wire.runtime_stack {
        total_text = checked_add(
            total_text,
            runtime.implementation.len(),
            "export text bytes",
        )?;
        total_text = checked_add(total_text, runtime.version.len(), "export text bytes")?;
    }
    for (index, frame) in wire.frames.iter().enumerate() {
        validate_frame(frame)?;
        if usize::try_from(frame.ordinal).ok() != Some(index) {
            return Err(invalid_artifact(
                "XDMF/HDF5 trajectory frame ordinals must be contiguous from zero",
            ));
        }
        total_fields = checked_add(total_fields, frame.fields.len(), "export Field count")?;
        for field in &frame.fields {
            total_text = checked_add(total_text, field.support_domain_ulid.len(), "export text")?;
            total_text = checked_add(total_text, field.field_ulid.len(), "export text")?;
            total_blocks = checked_add(total_blocks, field.blocks.len(), "export block count")?;
            for block in &field.blocks {
                total_text = checked_add(total_text, block.hdf5_dataset_path.len(), "export text")?;
            }
        }
    }
    if total_fields > limits.max_trajectory_storage_fields
        || total_blocks > limits.max_trajectory_storage_blocks
        || total_text > limits.max_trajectory_storage_text_bytes
    {
        return Err(invalid_artifact(
            "XDMF/HDF5 trajectory nested inventory exceeds a decoder limit",
        ));
    }
    for pair in wire.frames.windows(2) {
        if pair[0].step >= pair[1].step || pair[0].time_s >= pair[1].time_s {
            return Err(invalid_artifact(
                "XDMF/HDF5 trajectory step and time must increase strictly after remesh replacement",
            ));
        }
    }
    let first_v3 = wire
        .frames
        .iter()
        .position(|frame| frame.state_kind == SpatialStateTopology::Remeshed)
        .ok_or_else(|| {
            invalid_artifact("external trajectory omits its V3 remesh representation")
        })?;
    if wire.frames[first_v3..]
        .iter()
        .any(|frame| frame.state_kind != SpatialStateTopology::Remeshed)
    {
        return Err(invalid_artifact(
            "external trajectory may contain only a V2 prefix followed by one nonempty V3 suffix",
        ));
    }
    let inventory = field_inventory(&wire.frames[0]);
    if wire
        .frames
        .iter()
        .skip(1)
        .any(|frame| field_inventory(frame) != inventory)
    {
        return Err(invalid_artifact(
            "XDMF/HDF5 trajectory frames must retain one exact Field/block presentation inventory",
        ));
    }
    let mut states = BTreeSet::new();
    if wire
        .frames
        .iter()
        .any(|frame| !states.insert(&frame.spatial_state_sha256))
    {
        return Err(invalid_artifact(
            "XDMF/HDF5 trajectory spatial states must be unique",
        ));
    }
    Ok((adapter, runtime_stack))
}

fn validate_frame(frame: &WireFrame) -> Result<(), Diagnostic> {
    if !frame.time_s.is_finite()
        || frame.time_s < 0.0
        || (frame.time_s == 0.0 && frame.time_s.is_sign_negative())
        || frame.fields.is_empty()
    {
        return Err(invalid_artifact(
            "XDMF/HDF5 trajectory frame coordinate or Field inventory is invalid",
        ));
    }
    for digest in [
        &frame.spatial_state_sha256,
        &frame.reference_mesh_sha256,
        &frame.geometry_state_sha256,
    ] {
        ArtifactDigest::from_hex(digest.clone())?;
    }
    for field in &frame.fields {
        validate_field(field)?;
    }
    if frame
        .fields
        .windows(2)
        .any(|pair| pair[0].field_ulid >= pair[1].field_ulid)
    {
        return Err(invalid_artifact(
            "XDMF/HDF5 trajectory Fields must be unique and canonical",
        ));
    }
    Ok(())
}

fn validate_field(field: &WireField) -> Result<(), Diagnostic> {
    parse_id::<kinds::Domain>(&field.support_domain_ulid, "support Domain")?;
    parse_id::<kinds::Field>(&field.field_ulid, "Field")?;
    ArtifactDigest::from_hex(field.snapshot_sha256.clone())?;
    if field.blocks.is_empty()
        || !field.blocks.iter().any(|block| {
            block.presentation == TemporalStorageBlockPresentationV1::XdmfNodeAttribute
        })
    {
        return Err(invalid_artifact(
            "XDMF/HDF5 trajectory Field must have stored blocks and a truthful nodal presentation",
        ));
    }
    for block in &field.blocks {
        validate_block(block)?;
    }
    if field
        .blocks
        .windows(2)
        .any(|pair| pair[0].association >= pair[1].association)
    {
        return Err(invalid_artifact(
            "XDMF/HDF5 trajectory coefficient associations must be unique and canonical",
        ));
    }
    Ok(())
}

fn validate_block(block: &WireBlock) -> Result<(), Diagnostic> {
    let digest = ArtifactDigest::from_hex(block.logical_discrete_field_sha256.clone())?;
    if block.hdf5_dataset_path != format!("/fields/{digest}/values") {
        return Err(invalid_artifact(
            "XDMF/HDF5 trajectory block path is not its canonical content-addressed path",
        ));
    }
    if block.presentation == TemporalStorageBlockPresentationV1::XdmfNodeAttribute
        && block.association != WireAssociation::Vertex
    {
        return Err(invalid_artifact(
            "only vertex-associated values may be presented as XDMF node Attributes",
        ));
    }
    Ok(())
}

fn validate_payload(payload: &WirePayload) -> Result<(), Diagnostic> {
    StorageChunkSha256V1::from_hex(payload.raw_sha256.clone())?;
    if payload.byte_count == 0 {
        return Err(invalid_artifact(
            "external-export payload byte count must be positive",
        ));
    }
    Ok(())
}

type FieldInventoryEntry<'a> = (
    &'a str,
    &'a str,
    Vec<(WireAssociation, TemporalStorageBlockPresentationV1)>,
);

fn field_inventory(frame: &WireFrame) -> Vec<FieldInventoryEntry<'_>> {
    frame
        .fields
        .iter()
        .map(|field| {
            (
                field.support_domain_ulid.as_str(),
                field.field_ulid.as_str(),
                field
                    .blocks
                    .iter()
                    .map(|block| (block.association, block.presentation))
                    .collect(),
            )
        })
        .collect()
}

fn parse_digest(value: &str) -> ArtifactDigest {
    ArtifactDigest::from_hex(value.to_owned()).expect("validated artifact digest")
}

fn parse_id<K: eqiora_core::Entity>(value: &str, label: &str) -> Result<Id<K>, Diagnostic> {
    let ulid = Ulid::from_str(value)
        .map_err(|_| invalid_artifact(format!("temporal storage {label} ULID is malformed")))?;
    if ulid.to_string() != value {
        return Err(invalid_artifact(format!(
            "temporal storage {label} ULID spelling is noncanonical"
        )));
    }
    Ok(Id::from_ulid(ulid))
}

fn checked_add(left: usize, right: usize, label: &str) -> Result<usize, Diagnostic> {
    left.checked_add(right)
        .ok_or_else(|| invalid_artifact(format!("{label} overflows usize")))
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireEnvelope {
    schema: String,
    encoding: String,
    adapter: WireAdapter,
    runtime_stack: Vec<WireRuntimeComponent>,
    trajectory_v3_sha256: String,
    remesh_seam_policy: String,
    xdmf: WirePayload,
    hdf5: WirePayload,
    frames: Vec<WireFrame>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WirePayload {
    raw_sha256: String,
    byte_count: u64,
}

impl WirePayload {
    fn new(bytes: &[u8]) -> Result<Self, Diagnostic> {
        Ok(Self {
            raw_sha256: StorageChunkSha256V1::from_bytes(bytes).to_string(),
            byte_count: u64::try_from(bytes.len()).map_err(|_| {
                invalid_artifact("external-export payload length exceeds portable u64")
            })?,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireFrame {
    ordinal: u64,
    step: u64,
    time_s: f64,
    state_kind: SpatialStateTopology,
    spatial_state_sha256: String,
    reference_mesh_sha256: String,
    geometry_state_sha256: String,
    fields: Vec<WireField>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireField {
    support_domain_ulid: String,
    field_ulid: String,
    snapshot_sha256: String,
    blocks: Vec<WireBlock>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireBlock {
    association: WireAssociation,
    logical_discrete_field_sha256: String,
    hdf5_dataset_path: String,
    presentation: TemporalStorageBlockPresentationV1,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ExternalRuntimeRoleV1;

    fn digest(byte: u8) -> ArtifactDigest {
        ArtifactDigest::from_hex(format!("{byte:02x}").repeat(32)).unwrap()
    }

    fn field(snapshot: u8, vertex: u8, cell: Option<u8>) -> XdmfHdf5TrajectoryFieldV1 {
        let vertex = digest(vertex);
        let mut blocks = vec![WireBlock {
            association: WireAssociation::Vertex,
            logical_discrete_field_sha256: vertex.to_string(),
            hdf5_dataset_path: format!("/fields/{vertex}/values"),
            presentation: TemporalStorageBlockPresentationV1::XdmfNodeAttribute,
        }];
        if let Some(cell) = cell {
            let cell = digest(cell);
            blocks.push(WireBlock {
                association: WireAssociation::Cell,
                logical_discrete_field_sha256: cell.to_string(),
                hdf5_dataset_path: format!("/fields/{cell}/values"),
                presentation: TemporalStorageBlockPresentationV1::Hidden,
            });
        }
        let wire = WireField {
            support_domain_ulid: "01ARZ3NDEKTSV4RRFFQ69G5FAV".to_owned(),
            field_ulid: "01ARZ3NDEKTSV4RRFFQ69G5FAW".to_owned(),
            snapshot_sha256: digest(snapshot).to_string(),
            blocks,
        };
        validate_field(&wire).unwrap();
        XdmfHdf5TrajectoryFieldV1 { wire }
    }

    fn frame(
        ordinal: u64,
        state_kind: SpatialStateTopology,
        seed: u8,
    ) -> XdmfHdf5TrajectoryFrameV1 {
        let wire = WireFrame {
            ordinal,
            step: ordinal,
            time_s: ordinal as f64 * 0.25,
            state_kind,
            spatial_state_sha256: digest(seed).to_string(),
            reference_mesh_sha256: digest(seed + 1).to_string(),
            geometry_state_sha256: digest(seed + 2).to_string(),
            fields: vec![field(seed + 3, seed + 4, Some(seed + 5)).wire],
        };
        validate_frame(&wire).unwrap();
        XdmfHdf5TrajectoryFrameV1 { wire }
    }

    #[test]
    fn closed_wire_roundtrips_and_hides_cell_basis_coefficients() {
        let adapter = ExternalAdapterIdentityV1::new("eqiora.xdmf-hdf5-trajectory", "1").unwrap();
        let runtime = vec![
            ExternalRuntimeComponentV1::new(
                ExternalRuntimeRoleV1::RustBinding,
                "hdf5-metno",
                "0.13.0",
            )
            .unwrap(),
            ExternalRuntimeComponentV1::new(
                ExternalRuntimeRoleV1::NativeStorageLibrary,
                "hdf5",
                "2.0.0",
            )
            .unwrap(),
        ];
        let value = XdmfHdf5TrajectoryStorageEnvelope::finish(
            adapter,
            runtime,
            digest(1),
            b"<Xdmf/>",
            b"hdf5",
            vec![
                frame(0, SpatialStateTopology::Reference, 10),
                frame(1, SpatialStateTopology::Remeshed, 20),
            ],
            TrajectoryStorageDecoderLimits::default(),
        )
        .unwrap();
        let bytes = value.canonical_json().unwrap();
        let decoded = XdmfHdf5TrajectoryStorageEnvelope::from_json(
            &bytes,
            TrajectoryStorageDecoderLimits::default(),
        )
        .unwrap();
        let mut retired: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        retired["schema"] = serde_json::json!("eqiora.xdmf-hdf5-trajectory-storage/v1");
        assert!(
            XdmfHdf5TrajectoryStorageEnvelope::from_json(
                &serde_json::to_vec(&retired).unwrap(),
                Default::default()
            )
            .is_err()
        );
        assert_eq!(decoded, value);
        assert_eq!(decoded.canonical_json().unwrap(), bytes);
        assert_eq!(
            decoded.frames()[0].fields()[0].blocks()[1].presentation(),
            TemporalStorageBlockPresentationV1::Hidden
        );
    }

    #[test]
    fn seam_order_presentation_and_resource_excess_fail_closed() {
        let cell = digest(1);
        assert!(
            validate_block(&WireBlock {
                association: WireAssociation::Cell,
                logical_discrete_field_sha256: cell.to_string(),
                hdf5_dataset_path: format!("/fields/{cell}/values"),
                presentation: TemporalStorageBlockPresentationV1::XdmfNodeAttribute,
            })
            .is_err()
        );
        let adapter = ExternalAdapterIdentityV1::new("eqiora.xdmf", "1").unwrap();
        let runtime = vec![
            ExternalRuntimeComponentV1::new(ExternalRuntimeRoleV1::RustBinding, "binding", "1")
                .unwrap(),
            ExternalRuntimeComponentV1::new(
                ExternalRuntimeRoleV1::NativeStorageLibrary,
                "native",
                "1",
            )
            .unwrap(),
        ];
        let reversed = vec![
            frame(0, SpatialStateTopology::Remeshed, 10),
            frame(1, SpatialStateTopology::Reference, 20),
        ];
        assert!(
            XdmfHdf5TrajectoryStorageEnvelope::finish(
                adapter.clone(),
                runtime.clone(),
                digest(1),
                b"x",
                b"h",
                reversed,
                TrajectoryStorageDecoderLimits::default(),
            )
            .is_err()
        );
        let limits = TrajectoryStorageDecoderLimits {
            max_trajectory_storage_frames: 1,
            ..TrajectoryStorageDecoderLimits::default()
        };
        assert!(
            XdmfHdf5TrajectoryStorageEnvelope::finish(
                adapter,
                runtime,
                digest(1),
                b"x",
                b"h",
                vec![
                    frame(0, SpatialStateTopology::Reference, 10),
                    frame(1, SpatialStateTopology::Remeshed, 20),
                ],
                limits,
            )
            .is_err()
        );
    }
}
