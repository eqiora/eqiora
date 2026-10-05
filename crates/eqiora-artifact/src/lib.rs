//! Versioned, deterministic Evidence & Artifact Graph payloads.
//!
//! Wire DTOs are deliberately separate from Semantic Kernel Rust types.
//! Decoding reconstructs validated definitions and commits one typed graph
//! transaction; deserialization never bypasses an existing invariant.

mod ale_wire;
mod cad;
mod cartesian_mesh;
mod cartesian_q1_field_snapshot;
mod dimension;
mod discrete_field;
mod distributed;
mod external_import;
mod geometry_definition;
mod geometry_identity;
mod geometry_mesh_correspondence;
mod geometry_revision_association;
mod geometry_state;
mod geometry_state_reference;
mod geometry_state_v2;
mod geometry_state_v3;
mod implicit_time;
mod implicit_time_lineage;
mod json_preflight;
mod mesh;
mod mesh_production_lineage;
mod mesh_revision_overlap;
mod model;
mod model_reference;
mod model_transaction;
mod model_transaction_wire;
mod model_wire;
mod physical_exposure;
mod property_table;
mod realization;
mod realization_reference;
mod realization_v6;
mod realization_v7;
mod realization_v8;
mod remesh_transfer;
mod resolved_array;
mod root_registration;
mod run_v2;
mod semantic_fingerprint;
mod spatial_data;
mod spatial_state_v2;
mod spatial_state_v3;
mod spatial_trajectory_v2;
mod spatial_trajectory_v3;
mod time;
mod xdmf_hdf5_trajectory_storage;

use std::collections::BTreeMap;
use std::fmt;

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub use cad::{CadBuildEvidenceEnvelopeV1, CadDesignEnvelopeV1};
pub use cartesian_mesh::{CartesianMeshDecoderLimits, CartesianMeshEnvelopeV1};
pub use cartesian_q1_field_snapshot::CartesianQ1FieldSnapshotEnvelopeV2;
pub use discrete_field::{DiscreteFieldEnvelopeV1, FieldDecoderLimits};
pub use distributed::{
    DistributedDecoderLimits, DistributedLayoutEnvelopeV1, LinearSystemEnvelopeV1,
    PartitionEnvelopeV1, validate_distributed_content_dag,
};
pub use external_import::{
    ExternalAdapterIdentityV1, ExternalImportDecoderLimits, ExternalImportManifestV1,
    ExternalImportObservationV1, ExternalImportSelectionV1, ExternalImportSourceV1,
    ExternalRuntimeComponentV1, ExternalRuntimeRoleV1, RawSourceSha256, ResolvedImportArrayV1,
    SelectedSourceEntityV1, StructuralSelectorV1,
};
pub use geometry_definition::{GeometryDefinitionDecoderLimits, GeometryDefinitionV1};
pub use geometry_identity::{
    CartesianGeometryBodyV1, CartesianGeometryBoundaryV1, GeometryDecoderLimits, GeometryEntityV1,
    GeometryIdentityEnvelopeV1,
};
pub use geometry_mesh_correspondence::{
    ConservingGeometryInterfaceV1, GeometryMeshCorrespondenceEnvelopeV1,
};
pub use geometry_revision_association::{
    GeometryAssociationArtifactError, GeometryRevisionAssociationEnvelopeV1,
};
pub use geometry_state::GeometryStateEnvelopeV1;
pub use geometry_state_reference::ReplayableFixedTopologyGeometryStateArtifact;
pub use geometry_state_v2::{
    GeometryStateEnvelopeV2, GeometryStateOriginKindV2, ValidatedRemeshGeometrySourceV2,
};
pub use geometry_state_v3::GeometryStateEnvelopeV3;
pub use implicit_time::{
    GeneralImplicitTimeLoweringEnvelopeV3, ImplicitTimeInitialDataEnvelopeV1,
    ImplicitTimeRunManifestV1,
};
pub use implicit_time_lineage::{ImplicitTimeCheckpointEnvelopeV1, ImplicitTimeRestartManifestV1};
pub use json_preflight::JsonDecoderLimits;
pub(crate) use json_preflight::check_json_limits;
pub use mesh::{MeshDecoderLimits, SimplicialMeshEnvelopeV1};
pub use mesh_production_lineage::{
    AffineTriangleMeshCellsV1, CartesianMeshCellsV2, GmshMeshPolicyV1,
    MeshProductionLineageEnvelopeV1,
};
pub use mesh_revision_overlap::MeshRevisionOverlapEnvelopeV1;
pub use model::ModelDecoderLimits;
pub use model_reference::{
    AcceptedModelArtifact, CanonicalModelArtifact, ModelArtifactReference,
    ReplayableCanonicalModelArtifact, ReplayedCanonicalModel,
};
pub use model_transaction_wire::ModelTransactionEnvelope;
pub use model_wire::ModelEnvelope;
pub use physical_exposure::{
    PhysicalExposureCatalogEnvelopeV1, PhysicalExposureContractV1, PhysicalExposureDecoderLimits,
    PhysicalExposureObservationBindingV1, PhysicalExposureProjectionV1, PhysicalExposureQuantityV1,
    PhysicalExposureSourceOriginV1, PhysicalExposureSourceSpanV1,
};
pub use property_table::decode_real_table;
pub use realization::{LayoutArtifacts, RealizationDecoderLimits, RealizationEnvelopeV1};
pub use realization_reference::{
    CanonicalRealizationArtifact, RealizationArtifactReference,
    ReplayableFixedTopologyAleRealizationArtifact,
};
pub use realization_v6::RealizationEnvelopeV6;
pub use realization_v7::RealizationEnvelopeV7;
pub use realization_v8::RealizationEnvelopeV8;
pub use remesh_transfer::{
    BoundedRemeshDefectV1, FieldTransferReceiptV1, RemeshDecoderLimits, RemeshFieldRoleV1,
    RemeshIntegrationChartV1, RemeshNormalizationWitnessV1, RemeshProjectionActionV1,
    RemeshProjectionEvidenceEnvelopeV1, RemeshProjectionExecutionModeV1, RemeshTransferEvidenceV1,
    RemeshTransferLawV1, RemeshTransferReceiptEnvelopeV1,
};
pub use resolved_array::{
    ResolvedArrayDecoderLimits, ResolvedArrayLimits, ResolvedArrayScalarV1, ResolvedArrayV1,
};
pub use root_registration::RootRegistrationEnvelopeV1;
pub use run_v2::{
    DistributedTransportV1, ExecutionProvenanceFingerprintV1, ExecutionProvenanceV1,
    ExecutionTopologyV1, MpiThreadSupportV1, RunManifestV2,
};
pub use semantic_fingerprint::{
    SemanticFingerprintGeneration, StructuralSemanticFingerprint, structurally_equivalent,
};
pub use spatial_data::{
    DiscreteFieldStorageEnvelopeV1, FieldSnapshotEnvelopeV2, MlDatasetChannelStatisticsV1,
    MlDatasetDecoderLimits, MlDatasetDescriptorRoleV1, MlDatasetEnvelopeV2,
    MlDatasetFieldDescriptorV1, MlDatasetObservationReferenceV1, MlDatasetSampleSplitV1,
    MlDatasetSampleV1, MlDatasetStateKindV1, MlDatasetStateReferenceV1, StorageChunkSha256V1,
    StorageChunkV1, TrajectoryDecoderLimits,
};
pub use spatial_state_v2::{SpatialStateEnvelopeV2, ValidatedMovingSpatialContextV2};
pub use spatial_state_v3::{SpatialStateEnvelopeV3, SpatialStateOriginKindV3};
pub use spatial_trajectory_v2::{SpatialTrajectoryEnvelopeV2, SpatialTrajectorySegmentEnvelopeV2};
pub use spatial_trajectory_v3::{
    SpatialTrajectoryEnvelopeV3, SpatialTrajectorySegmentEnvelopeV3,
    SpatialTrajectorySegmentOriginKindV3,
};
pub use time::{TimeDecoderLimits, TimeLoweringEnvelopeV3, TimeRunManifestV1};
pub use xdmf_hdf5_trajectory_storage::{
    TemporalStorageBlockPresentationV1, TemporalStorageStateKindV1, TrajectoryStorageDecoderLimits,
    XdmfHdf5TrajectoryBlockV1, XdmfHdf5TrajectoryFieldV1, XdmfHdf5TrajectoryFrameV1,
    XdmfHdf5TrajectoryStorageEnvelopeV1,
};

const RUN_SCHEMA: &str = "eqiora.run-manifest/v1";
pub(crate) const CANONICAL_ENCODING: &str = "eqiora.canonical-json/v1";

/// A lowercase SHA-256 content identity.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ArtifactDigest(String);

impl ArtifactDigest {
    /// Encode complete SHA-256 bytes as canonical lowercase hexadecimal.
    #[must_use]
    pub fn from_sha256(bytes: [u8; 32]) -> Self {
        let mut encoded = String::with_capacity(64);
        for byte in bytes {
            use fmt::Write as _;
            write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
        }
        Self(encoded)
    }

    /// Parse a canonical 64-character lowercase SHA-256 digest.
    ///
    /// # Errors
    /// Returns `EQ0901` for any other representation.
    pub fn from_hex(value: impl Into<String>) -> Result<Self, Diagnostic> {
        let value = value.into();
        if value.len() != 64
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(invalid_artifact(
                "artifact digest must be 64 lowercase hexadecimal SHA-256 characters",
            ));
        }
        Ok(Self(value))
    }

    /// Canonical lowercase hexadecimal form.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Complete SHA-256 bytes.
    #[must_use]
    pub fn sha256_bytes(&self) -> [u8; 32] {
        let mut bytes = [0_u8; 32];
        for (index, pair) in self.0.as_bytes().as_chunks::<2>().0.iter().enumerate() {
            bytes[index] = (hex_nibble(pair[0]) << 4) | hex_nibble(pair[1]);
        }
        bytes
    }

    pub(crate) fn compute(domain: &[u8], bytes: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(domain);
        hasher.update([0]);
        hasher.update(bytes);
        let digest = hasher.finalize();
        Self::from_sha256(digest.into())
    }
}

fn hex_nibble(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        _ => unreachable!("ArtifactDigest always contains validated lowercase hexadecimal"),
    }
}

impl fmt::Display for ArtifactDigest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Reproducible inputs and outputs of one execution.
///
/// Wall-clock timestamps and host paths are intentionally absent. A caller may
/// attach those as provenance metadata without changing run identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunManifestV1 {
    wire: WireRunManifestV1,
}

impl RunManifestV1 {
    /// Start a run manifest for one semantic artifact revision and executor.
    ///
    /// # Errors
    /// Returns `EQ0901` for empty executor identity.
    pub fn new(
        model: ArtifactDigest,
        semantic_revision: u64,
        executor: impl Into<String>,
        executor_version: impl Into<String>,
    ) -> Result<Self, Diagnostic> {
        let executor = executor.into();
        let executor_version = executor_version.into();
        validate_text("executor", &executor)?;
        validate_text("executor version", &executor_version)?;
        Ok(Self {
            wire: WireRunManifestV1 {
                schema: RUN_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                model_sha256: model.0,
                semantic_revision,
                realization_sha256: None,
                executor,
                executor_version,
                numerical_settings: BTreeMap::new(),
                output_sha256: Vec::new(),
            },
        })
    }

    /// Reference an independently versioned Realization artifact.
    #[must_use]
    pub fn with_realization(mut self, realization: ArtifactDigest) -> Self {
        self.wire.realization_sha256 = Some(realization.0);
        self
    }

    /// Add one deterministic numerical setting.
    ///
    /// # Errors
    /// Returns `EQ0901` unless the key is lowercase dotted/kebab ASCII and the
    /// value is non-empty without control characters.
    pub fn with_numerical_setting(
        mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Result<Self, Diagnostic> {
        let key = key.into();
        let value = value.into();
        validate_setting_key(&key)?;
        validate_text("numerical setting", &value)?;
        if self
            .wire
            .numerical_settings
            .insert(key.clone(), value)
            .is_some()
        {
            return Err(invalid_artifact(format!(
                "duplicate numerical setting `{key}`"
            )));
        }
        Ok(self)
    }

    /// Add a content-addressed output artifact.
    #[must_use]
    pub fn with_output(mut self, output: ArtifactDigest) -> Self {
        self.wire.output_sha256.push(output.0);
        self.wire.output_sha256.sort();
        self.wire.output_sha256.dedup();
        self
    }

    /// Decode and validate a run manifest.
    ///
    /// # Errors
    /// Returns `EQ0901` for oversized, malformed, unknown-version, duplicate,
    /// or non-canonical field data.
    pub fn from_json(bytes: &[u8], limits: JsonDecoderLimits) -> Result<Self, Diagnostic> {
        check_json_limits(bytes, limits)?;
        let mut wire: WireRunManifestV1 = serde_json::from_slice(bytes)
            .map_err(|error| invalid_artifact(format!("invalid run manifest JSON: {error}")))?;
        if wire.schema != RUN_SCHEMA || wire.encoding != CANONICAL_ENCODING {
            return Err(invalid_artifact(
                "unsupported run-manifest schema or canonical encoding",
            ));
        }
        ArtifactDigest::from_hex(wire.model_sha256.clone())?;
        if let Some(realization) = &wire.realization_sha256 {
            ArtifactDigest::from_hex(realization.clone())?;
        }
        validate_text("executor", &wire.executor)?;
        validate_text("executor version", &wire.executor_version)?;
        for (key, value) in &wire.numerical_settings {
            validate_setting_key(key)?;
            validate_text("numerical setting", value)?;
        }
        for output in &wire.output_sha256 {
            ArtifactDigest::from_hex(output.clone())?;
        }
        let original_output_count = wire.output_sha256.len();
        wire.output_sha256.sort();
        wire.output_sha256.dedup();
        if wire.output_sha256.len() != original_output_count {
            return Err(invalid_artifact(
                "run manifest contains duplicate output artifact digests",
            ));
        }
        Ok(Self { wire })
    }

    /// Deterministic canonical JSON bytes.
    ///
    /// # Errors
    /// Returns `EQ0901` if serialization unexpectedly fails.
    pub fn canonical_json(&self) -> Result<Vec<u8>, Diagnostic> {
        serde_json::to_vec(&self.wire)
            .map_err(|error| invalid_artifact(format!("cannot serialize run manifest: {error}")))
    }

    /// Domain-separated SHA-256 identity of canonical manifest bytes.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            RUN_SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
    }

    /// Referenced semantic-model digest.
    #[must_use]
    pub fn model(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.model_sha256.clone())
    }

    /// Semantic revision executed within the referenced Model artifact.
    #[must_use]
    pub const fn semantic_revision(&self) -> u64 {
        self.wire.semantic_revision
    }

    /// Sorted content-addressed output artifacts.
    #[must_use]
    pub fn outputs(&self) -> Vec<ArtifactDigest> {
        self.wire
            .output_sha256
            .iter()
            .cloned()
            .map(ArtifactDigest)
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireRunManifestV1 {
    schema: String,
    encoding: String,
    model_sha256: String,
    semantic_revision: u64,
    realization_sha256: Option<String>,
    executor: String,
    executor_version: String,
    numerical_settings: BTreeMap<String, String>,
    output_sha256: Vec<String>,
}

pub(crate) fn validate_setting_key(key: &str) -> Result<(), Diagnostic> {
    if key.is_empty()
        || !key.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'.' | b'-' | b'_')
        })
    {
        return Err(invalid_artifact(
            "numerical setting keys must be non-empty lowercase dotted/kebab/snake ASCII",
        ));
    }
    Ok(())
}

pub(crate) fn validate_text(label: &str, value: &str) -> Result<(), Diagnostic> {
    if value.trim().is_empty() || value.chars().any(char::is_control) {
        return Err(invalid_artifact(format!(
            "{label} must be non-empty and contain no control characters"
        )));
    }
    Ok(())
}

pub(crate) fn invalid_artifact(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_ARTIFACT, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_manifest_is_ordered_versioned_and_content_addressed() {
        let model = ArtifactDigest::compute(b"model", b"one");
        let output = ArtifactDigest::compute(b"output", b"two");
        let manifest = RunManifestV1::new(model.clone(), 7, "eqiora-reference", "0.1.0")
            .unwrap()
            .with_numerical_setting("solver.max-step", "0.01 s")
            .unwrap()
            .with_output(output.clone())
            .with_output(output);
        let bytes = manifest.canonical_json().unwrap();
        let decoded = RunManifestV1::from_json(&bytes, JsonDecoderLimits::default()).unwrap();
        assert_eq!(decoded.canonical_json().unwrap(), bytes);
        assert_eq!(decoded.model(), model);
        assert_eq!(decoded.digest().unwrap(), manifest.digest().unwrap());
    }

    #[test]
    fn malformed_digest_and_duplicate_setting_are_rejected() {
        assert_eq!(
            ArtifactDigest::from_hex("ABC").unwrap_err().code(),
            codes::INVALID_ARTIFACT
        );
        assert_eq!(
            ArtifactDigest::from_hex("01".repeat(32))
                .unwrap()
                .sha256_bytes(),
            [1; 32]
        );
        let model = ArtifactDigest::compute(b"model", b"one");
        let manifest = RunManifestV1::new(model, 1, "executor", "v1")
            .unwrap()
            .with_numerical_setting("step", "one")
            .unwrap();
        assert_eq!(
            manifest
                .with_numerical_setting("step", "two")
                .unwrap_err()
                .code(),
            codes::INVALID_ARTIFACT
        );
    }
}
