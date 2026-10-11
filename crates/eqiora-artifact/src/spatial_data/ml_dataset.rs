//! Storage-independent ML Dataset manifests derived from remeshing trajectories.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::str::FromStr;

use crate::dimension::WireDimension;

use eqiora_core::ValueFrame;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, DimExponents, Id, ValueShape};
use eqiora_meshing::DiscreteFieldAssociation;
use serde::{Deserialize, Serialize};
use ulid::Ulid;

use crate::{
    ArtifactDigest, CANONICAL_ENCODING, FieldSnapshotEnvelopeV2, JsonDecoderLimits,
    SpatialStateEnvelopeV2, SpatialStateEnvelopeV3, SpatialTrajectoryEnvelopeV3, check_json_limits,
    invalid_artifact,
};

const SCHEMA: &str = "eqiora.ml-dataset-envelope/v2";
const SEAM_POLICY: &str = "target-replaces-source-at-remesh";

/// Semantic work budgets for the ML Dataset artifact family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MlDatasetDecoderLimits {
    /// Common JSON syntax admission.
    pub json: JsonDecoderLimits,
    /// Maximum rank of one Dataset Field descriptor shape.
    pub max_value_shape_rank: usize,
    /// Maximum scalar components in one Dataset Field descriptor shape.
    pub max_value_shape_components: usize,
    /// Maximum typed Field descriptors in one derived ML Dataset.
    pub max_ml_dataset_descriptors: usize,
    /// Maximum samples in one derived ML Dataset.
    pub max_ml_dataset_samples: usize,
    /// Maximum state references summed across all Dataset windows.
    pub max_ml_dataset_window_states: usize,
    /// Maximum selected snapshot references summed across all samples.
    pub max_ml_dataset_observations: usize,
    /// Maximum coefficient-block references summed across one Dataset.
    pub max_ml_dataset_blocks: usize,
    /// Maximum population-normalization channels in one Dataset.
    pub max_ml_dataset_normalization_channels: usize,
}

impl Default for MlDatasetDecoderLimits {
    fn default() -> Self {
        Self {
            json: JsonDecoderLimits::default(),
            max_value_shape_rank: 8,
            max_value_shape_components: 4_096,
            max_ml_dataset_descriptors: 100_000,
            max_ml_dataset_samples: 1_000_000,
            max_ml_dataset_window_states: 16_000_000,
            max_ml_dataset_observations: 16_000_000,
            max_ml_dataset_blocks: 32_000_000,
            max_ml_dataset_normalization_channels: 6_400_000,
        }
    }
}

/// Semantic use of one exact Field selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MlDatasetDescriptorRoleV1 {
    /// Model input exposed to a learner.
    Feature,
    /// Supervised value to be predicted.
    Target,
}

/// Durable state generation represented in one strict-time sample window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MlDatasetStateKindV1 {
    /// Pre-remesh state from the immutable V2 source prefix.
    MovingV2,
    /// Seam replacement or continuation state on the V3 target topology.
    RemeshedV3,
}

/// Closed Dataset partition semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MlDatasetSampleSplitV1 {
    /// Samples used to fit normalization and learning parameters.
    Training,
    /// Later samples used for model selection.
    Validation,
    /// Latest held-out samples used only for final evaluation.
    Test,
}

/// Typed Field selection at one fixed offset inside every sample window.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlDatasetFieldDescriptorV1 {
    wire: WireDescriptor,
}

impl MlDatasetFieldDescriptorV1 {
    /// Derive a descriptor's physical meaning from one validated snapshot.
    ///
    /// Later snapshots may live on different mesh revisions, but must retain
    /// this exact Field, support, dimension, shape, and frame.
    #[must_use]
    pub fn from_snapshot(
        role: MlDatasetDescriptorRoleV1,
        window_offset: u32,
        snapshot: &FieldSnapshotEnvelopeV2,
    ) -> Self {
        Self {
            wire: WireDescriptor {
                role,
                window_offset,
                field_ulid: snapshot.field().ulid().to_string(),
                support_domain_ulid: snapshot.support_domain().ulid().to_string(),
                physical: WirePhysicalType::encode(
                    snapshot.dimension(),
                    &snapshot.value_shape(),
                    snapshot.frame(),
                ),
            },
        }
    }

    /// Feature or target role.
    #[must_use]
    pub const fn role(&self) -> MlDatasetDescriptorRoleV1 {
        self.wire.role
    }

    /// Zero-based state offset inside every fixed-length sample window.
    #[must_use]
    pub const fn window_offset(&self) -> u32 {
        self.wire.window_offset
    }

    /// Exact Semantic Field identity.
    #[must_use]
    pub fn field(&self) -> Id<kinds::Field> {
        parse_id(&self.wire.field_ulid, "Field").expect("validated ML Dataset Field")
    }

    /// Exact Semantic Domain support.
    #[must_use]
    pub fn support_domain(&self) -> Id<kinds::Domain> {
        parse_id(&self.wire.support_domain_ulid, "support Domain")
            .expect("validated ML Dataset support Domain")
    }

    /// Coherent-SI physical dimension.
    #[must_use]
    pub const fn dimension(&self) -> DimExponents {
        self.wire.physical.dimension.decode()
    }

    /// Storage-independent mathematical value shape.
    #[must_use]
    pub fn value_shape(&self) -> ValueShape {
        self.wire
            .physical
            .value_shape
            .decode()
            .expect("validated ML Dataset value shape")
    }

    /// Coordinate-frame meaning of components.
    #[must_use]
    pub const fn frame(&self) -> ValueFrame {
        self.wire.physical.frame.decode()
    }

    fn cmp_key(&self, other: &Self) -> Ordering {
        self.wire
            .role
            .cmp(&other.wire.role)
            .then_with(|| self.wire.window_offset.cmp(&other.wire.window_offset))
            .then_with(|| self.wire.field_ulid.cmp(&other.wire.field_ulid))
    }

    fn matches_snapshot(&self, snapshot: &FieldSnapshotEnvelopeV2) -> bool {
        self.field() == snapshot.field()
            && self.support_domain() == snapshot.support_domain()
            && self.dimension() == snapshot.dimension()
            && self.value_shape() == snapshot.value_shape()
            && self.frame() == snapshot.frame()
    }
}

/// Exact trajectory-state identity retained by one sample window.
#[derive(Debug, Clone, PartialEq)]
pub struct MlDatasetStateReferenceV1 {
    wire: WireStateReference,
}

impl MlDatasetStateReferenceV1 {
    /// Capture one V2 source-prefix state at its strict-time frame ordinal.
    ///
    /// The V2 source tip must be omitted when its same-coordinate V3 remesh
    /// replacement is present.
    ///
    /// # Errors
    /// Returns `EQ0901` if the frame ordinal is not portable.
    pub fn from_v2(ordinal: usize, state: &SpatialStateEnvelopeV2) -> Result<Self, Diagnostic> {
        Self::new(
            ordinal,
            state.step(),
            state.time_s(),
            MlDatasetStateKindV1::MovingV2,
            state.digest()?,
            state.reference_mesh_artifact(),
            state.geometry_state_artifact(),
        )
    }

    /// Capture one V3 seam replacement or continuation state.
    ///
    /// # Errors
    /// Returns `EQ0901` if the frame ordinal is not portable.
    pub fn from_v3(ordinal: usize, state: &SpatialStateEnvelopeV3) -> Result<Self, Diagnostic> {
        Self::new(
            ordinal,
            state.step(),
            state.time_s(),
            MlDatasetStateKindV1::RemeshedV3,
            state.digest()?,
            state.reference_mesh_artifact(),
            state.geometry_state_artifact(),
        )
    }

    fn new(
        ordinal: usize,
        step: u64,
        time_s: f64,
        state_kind: MlDatasetStateKindV1,
        state: ArtifactDigest,
        mesh: ArtifactDigest,
        geometry_state: ArtifactDigest,
    ) -> Result<Self, Diagnostic> {
        let value = Self {
            wire: WireStateReference {
                frame_ordinal: u64::try_from(ordinal).map_err(|_| {
                    invalid_artifact("ML Dataset frame ordinal exceeds portable u64")
                })?,
                step,
                time_s,
                state_kind,
                spatial_state_sha256: state.to_string(),
                reference_mesh_sha256: mesh.to_string(),
                geometry_state_sha256: geometry_state.to_string(),
            },
        };
        validate_state(&value.wire)?;
        Ok(value)
    }

    /// Strict-time frame ordinal.
    #[must_use]
    pub const fn frame_ordinal(&self) -> u64 {
        self.wire.frame_ordinal
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
    pub const fn state_kind(&self) -> MlDatasetStateKindV1 {
        self.wire.state_kind
    }

    /// Exact SpatialState artifact.
    #[must_use]
    pub fn spatial_state_artifact(&self) -> ArtifactDigest {
        parse_digest(&self.wire.spatial_state_sha256)
    }

    /// Exact mesh revision for this possibly ragged frame.
    #[must_use]
    pub fn reference_mesh_artifact(&self) -> ArtifactDigest {
        parse_digest(&self.wire.reference_mesh_sha256)
    }

    /// Exact current GeometryState artifact.
    #[must_use]
    pub fn geometry_state_artifact(&self) -> ArtifactDigest {
        parse_digest(&self.wire.geometry_state_sha256)
    }
}

/// Exact selected snapshot and complete coefficient-block inventory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlDatasetObservationReferenceV1 {
    wire: WireObservationReference,
}

impl MlDatasetObservationReferenceV1 {
    /// Select one typed V2 snapshot without copying numerical values.
    ///
    /// # Errors
    /// Returns `EQ0901` for a descriptor mismatch or a snapshot not named by
    /// the exact state.
    pub fn from_v2(
        descriptor_ordinal: usize,
        descriptor: &MlDatasetFieldDescriptorV1,
        state: &SpatialStateEnvelopeV2,
        snapshot: &FieldSnapshotEnvelopeV2,
    ) -> Result<Self, Diagnostic> {
        Self::from_state(
            descriptor_ordinal,
            descriptor,
            state.digest()?,
            state.reference_mesh_artifact(),
            &state.fields(),
            snapshot,
        )
    }

    /// Select one typed V3 snapshot without copying numerical values.
    ///
    /// # Errors
    /// Returns `EQ0901` for a descriptor mismatch or a snapshot not named by
    /// the exact state.
    pub fn from_v3(
        descriptor_ordinal: usize,
        descriptor: &MlDatasetFieldDescriptorV1,
        state: &SpatialStateEnvelopeV3,
        snapshot: &FieldSnapshotEnvelopeV2,
    ) -> Result<Self, Diagnostic> {
        Self::from_state(
            descriptor_ordinal,
            descriptor,
            state.digest()?,
            state.reference_mesh_artifact(),
            &state.fields(),
            snapshot,
        )
    }

    fn from_state(
        descriptor_ordinal: usize,
        descriptor: &MlDatasetFieldDescriptorV1,
        state: ArtifactDigest,
        mesh: ArtifactDigest,
        state_fields: &[(Id<kinds::Domain>, Id<kinds::Field>, ArtifactDigest)],
        snapshot: &FieldSnapshotEnvelopeV2,
    ) -> Result<Self, Diagnostic> {
        let snapshot_digest = snapshot.digest()?;
        if !descriptor.matches_snapshot(snapshot)
            || snapshot.mesh_artifact() != mesh
            || !state_fields.contains(&(
                snapshot.support_domain(),
                snapshot.field(),
                snapshot_digest.clone(),
            ))
        {
            return Err(invalid_artifact(
                "ML Dataset observation does not match its descriptor or exact SpatialState",
            ));
        }
        let value = Self {
            wire: WireObservationReference {
                descriptor_ordinal: u32::try_from(descriptor_ordinal).map_err(|_| {
                    invalid_artifact("ML Dataset descriptor ordinal exceeds portable u32")
                })?,
                spatial_state_sha256: state.to_string(),
                reference_mesh_sha256: mesh.to_string(),
                snapshot_sha256: snapshot_digest.to_string(),
                blocks: snapshot
                    .block_artifacts()
                    .into_iter()
                    .map(|(association, digest)| WireBlockReference {
                        association: association.into(),
                        discrete_field_sha256: digest.to_string(),
                    })
                    .collect(),
            },
        };
        validate_observation(&value.wire)?;
        Ok(value)
    }

    /// Descriptor ordinal in canonical Dataset order.
    #[must_use]
    pub const fn descriptor_ordinal(&self) -> u32 {
        self.wire.descriptor_ordinal
    }

    /// Exact selected FieldSnapshot artifact.
    #[must_use]
    pub fn snapshot_artifact(&self) -> ArtifactDigest {
        parse_digest(&self.wire.snapshot_sha256)
    }

    /// Exact association-ordered DiscreteField block identities.
    #[must_use]
    pub fn blocks(&self) -> Vec<(DiscreteFieldAssociation, ArtifactDigest)> {
        self.wire
            .blocks
            .iter()
            .map(|block| {
                (
                    block.association.into(),
                    parse_digest(&block.discrete_field_sha256),
                )
            })
            .collect()
    }
}

/// One fixed-length strict-time sample with ragged spatial references.
#[derive(Debug, Clone, PartialEq)]
pub struct MlDatasetSampleV1 {
    wire: WireSample,
}

impl MlDatasetSampleV1 {
    /// Construct one sample in exact descriptor order.
    ///
    /// Windows may cross the V2-to-V3 remesh boundary. They retain the exact
    /// mesh identity per frame and never assert dense or shape-compatible
    /// arrays across revisions.
    ///
    /// # Errors
    /// Returns `EQ0901` for an empty/noncontiguous window, descriptor drift,
    /// duplicate observations, or a cross-wired snapshot state.
    pub fn new(
        ordinal: usize,
        split: MlDatasetSampleSplitV1,
        descriptors: &[MlDatasetFieldDescriptorV1],
        states: Vec<MlDatasetStateReferenceV1>,
        observations: Vec<MlDatasetObservationReferenceV1>,
    ) -> Result<Self, Diagnostic> {
        let value = Self {
            wire: WireSample {
                ordinal: u64::try_from(ordinal).map_err(|_| {
                    invalid_artifact("ML Dataset sample ordinal exceeds portable u64")
                })?,
                split,
                start_frame_ordinal: states.first().map_or(0, |state| state.wire.frame_ordinal),
                states: states.into_iter().map(|state| state.wire).collect(),
                observations: observations
                    .into_iter()
                    .map(|observation| observation.wire)
                    .collect(),
            },
        };
        validate_sample(&value.wire, descriptors)?;
        Ok(value)
    }

    /// Canonical sample ordinal.
    #[must_use]
    pub const fn ordinal(&self) -> u64 {
        self.wire.ordinal
    }

    /// Closed Dataset partition.
    #[must_use]
    pub const fn split(&self) -> MlDatasetSampleSplitV1 {
        self.wire.split
    }

    /// Explicit start in the source strict-time frame sequence.
    #[must_use]
    pub const fn start_frame_ordinal(&self) -> u64 {
        self.wire.start_frame_ordinal
    }

    /// Complete ordered state window.
    #[must_use]
    pub fn states(&self) -> Vec<MlDatasetStateReferenceV1> {
        self.wire
            .states
            .iter()
            .cloned()
            .map(|wire| MlDatasetStateReferenceV1 { wire })
            .collect()
    }

    /// One exact snapshot reference per descriptor.
    #[must_use]
    pub fn observations(&self) -> Vec<MlDatasetObservationReferenceV1> {
        self.wire
            .observations
            .iter()
            .cloned()
            .map(|wire| MlDatasetObservationReferenceV1 { wire })
            .collect()
    }
}

/// Training-only population standardization parameters for one scalar channel.
#[derive(Debug, Clone, PartialEq)]
pub struct MlDatasetChannelStatisticsV1 {
    wire: WireChannelStatistics,
}

impl MlDatasetChannelStatisticsV1 {
    /// Record a population mean and standard deviation derived from training.
    ///
    /// Zero variance is represented explicitly as a constant channel with a
    /// scale of one; all other channels use the population standard deviation
    /// as their scale.
    ///
    /// # Errors
    /// Returns `EQ0901` for an empty population, invalid scalar, or nonportable
    /// ordinal.
    pub fn population_standard_score(
        descriptor_ordinal: usize,
        association: DiscreteFieldAssociation,
        component: u32,
        population_count: u64,
        mean: f64,
        population_standard_deviation: f64,
        constant: bool,
    ) -> Result<Self, Diagnostic> {
        let value = Self {
            wire: WireChannelStatistics {
                descriptor_ordinal: u32::try_from(descriptor_ordinal).map_err(|_| {
                    invalid_artifact("ML Dataset descriptor ordinal exceeds portable u32")
                })?,
                association: association.into(),
                component,
                population_count,
                mean,
                population_standard_deviation,
                scale: if constant {
                    1.0
                } else {
                    population_standard_deviation
                },
                constant,
            },
        };
        validate_statistics(&value.wire)?;
        Ok(value)
    }

    /// Descriptor ordinal in canonical Dataset order.
    #[must_use]
    pub const fn descriptor_ordinal(&self) -> u32 {
        self.wire.descriptor_ordinal
    }

    /// Coefficient-block association.
    #[must_use]
    pub const fn association(&self) -> DiscreteFieldAssociation {
        self.wire.association.into_public()
    }

    /// Scalar component within the descriptor's mathematical value shape.
    #[must_use]
    pub const fn component(&self) -> u32 {
        self.wire.component
    }

    /// Number of active training values used for this channel.
    #[must_use]
    pub const fn population_count(&self) -> u64 {
        self.wire.population_count
    }

    /// Training-population mean.
    #[must_use]
    pub const fn mean(&self) -> f64 {
        self.wire.mean
    }

    /// Training-population standard deviation (`ddof = 0`).
    #[must_use]
    pub const fn population_standard_deviation(&self) -> f64 {
        self.wire.population_standard_deviation
    }

    /// Applied divisor, equal to one for a constant channel.
    #[must_use]
    pub const fn scale(&self) -> f64 {
        self.wire.scale
    }

    /// Whether the training population has zero variance.
    #[must_use]
    pub const fn is_constant(&self) -> bool {
        self.wire.constant
    }
}

/// Immutable logical Dataset derived from one exact remeshing trajectory.
///
/// This manifest contains only identity, type, split, window, snapshot, block,
/// and normalization lineage. Numerical values, active-entity indices, memory
/// layouts, device choices, and storage paths belong to materialization and
/// storage adapters.
#[derive(Debug, Clone, PartialEq)]
pub struct MlDatasetEnvelopeV2 {
    wire: WireEnvelope,
}

impl MlDatasetEnvelopeV2 {
    /// Close one Dataset manifest over an exact V3 trajectory root.
    ///
    /// Descriptor order is canonical: feature before target, then window
    /// offset, then Field identity. Samples must be ordered and partitioned in
    /// time as training, validation, then test. Sliding windows may overlap
    /// only inside the same partition.
    ///
    /// # Errors
    /// Returns `EQ0901` for noncanonical types, windows, splits, references,
    /// statistics, limits, or trajectory identities.
    pub fn new(
        trajectory: &SpatialTrajectoryEnvelopeV3,
        descriptors: Vec<MlDatasetFieldDescriptorV1>,
        samples: Vec<MlDatasetSampleV1>,
        statistics: Vec<MlDatasetChannelStatisticsV1>,
    ) -> Result<Self, Diagnostic> {
        let window_length = samples.first().map_or(0, |sample| sample.wire.states.len());
        let value = Self {
            wire: WireEnvelope {
                schema: SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                trajectory_v3_sha256: trajectory.digest()?.to_string(),
                source_prefix_v2_sha256: trajectory.source_prefix().to_string(),
                remesh_seam_policy: SEAM_POLICY.to_owned(),
                window_length: u32::try_from(window_length).map_err(|_| {
                    invalid_artifact("ML Dataset window length exceeds portable u32")
                })?,
                descriptors: descriptors
                    .into_iter()
                    .map(|descriptor| descriptor.wire)
                    .collect(),
                samples: samples.into_iter().map(|sample| sample.wire).collect(),
                normalization: WireNormalization {
                    method: WireNormalizationMethod::PopulationStandardScore,
                    accumulator: WireAccumulatorProfile::OrderedWelfordBinary64V1,
                    statistics_source: MlDatasetSampleSplitV1::Training,
                    channels: statistics
                        .into_iter()
                        .map(|statistics| statistics.wire)
                        .collect(),
                },
            },
        };
        value.validate_local(MlDatasetDecoderLimits::default())?;
        Ok(value)
    }

    /// Decode one bounded closed manifest without loading trajectories or data.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed, unknown, noncanonical, or over-budget
    /// wire data.
    pub fn from_json(bytes: &[u8], limits: MlDatasetDecoderLimits) -> Result<Self, Diagnostic> {
        check_json_limits(bytes, limits.json)?;
        let wire = serde_json::from_slice(bytes)
            .map_err(|error| invalid_artifact(format!("invalid ML Dataset JSON: {error}")))?;
        let value = Self { wire };
        value.validate_local(limits)?;
        Ok(value)
    }

    /// Deterministic canonical JSON bytes containing no numerical Field values.
    ///
    /// # Errors
    /// Returns `EQ0901` if serialization unexpectedly fails.
    pub fn canonical_json(&self) -> Result<Vec<u8>, Diagnostic> {
        serde_json::to_vec(&self.wire)
            .map_err(|error| invalid_artifact(format!("cannot serialize ML Dataset: {error}")))
    }

    /// Domain-separated immutable Dataset identity.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
    }

    /// Exact remeshing-aware trajectory root.
    #[must_use]
    pub fn trajectory_artifact(&self) -> ArtifactDigest {
        parse_digest(&self.wire.trajectory_v3_sha256)
    }

    /// Exact immutable V2 source prefix retained by that V3 root.
    #[must_use]
    pub fn source_prefix_artifact(&self) -> ArtifactDigest {
        parse_digest(&self.wire.source_prefix_v2_sha256)
    }

    /// Fixed sample-window length.
    #[must_use]
    pub const fn window_length(&self) -> u32 {
        self.wire.window_length
    }

    /// Canonically ordered typed descriptors.
    #[must_use]
    pub fn descriptors(&self) -> Vec<MlDatasetFieldDescriptorV1> {
        self.wire
            .descriptors
            .iter()
            .cloned()
            .map(|wire| MlDatasetFieldDescriptorV1 { wire })
            .collect()
    }

    /// Strict-time samples in canonical ordinal order.
    #[must_use]
    pub fn samples(&self) -> Vec<MlDatasetSampleV1> {
        self.wire
            .samples
            .iter()
            .cloned()
            .map(|wire| MlDatasetSampleV1 { wire })
            .collect()
    }

    /// Training-only population normalization channels.
    #[must_use]
    pub fn statistics(&self) -> Vec<MlDatasetChannelStatisticsV1> {
        self.wire
            .normalization
            .channels
            .iter()
            .cloned()
            .map(|wire| MlDatasetChannelStatisticsV1 { wire })
            .collect()
    }

    /// Check the exact source roots without loading any Dataset values.
    ///
    /// # Errors
    /// Returns `EQ0901` for trajectory substitution or source-prefix drift.
    pub fn validate_trajectory(
        &self,
        trajectory: &SpatialTrajectoryEnvelopeV3,
    ) -> Result<(), Diagnostic> {
        if self.trajectory_artifact() != trajectory.digest()?
            || self.source_prefix_artifact() != trajectory.source_prefix()
        {
            return Err(invalid_artifact(
                "ML Dataset references a different remeshing trajectory root",
            ));
        }
        Ok(())
    }

    fn validate_local(&self, limits: MlDatasetDecoderLimits) -> Result<(), Diagnostic> {
        validate_wire(&self.wire, limits)
    }
}

fn validate_wire(wire: &WireEnvelope, limits: MlDatasetDecoderLimits) -> Result<(), Diagnostic> {
    if wire.schema != SCHEMA
        || wire.encoding != CANONICAL_ENCODING
        || wire.remesh_seam_policy != SEAM_POLICY
        || wire.normalization.method != WireNormalizationMethod::PopulationStandardScore
        || wire.normalization.accumulator != WireAccumulatorProfile::OrderedWelfordBinary64V1
        || wire.normalization.statistics_source != MlDatasetSampleSplitV1::Training
    {
        return Err(invalid_artifact(
            "unsupported ML Dataset schema, encoding, seam, or normalization policy",
        ));
    }
    ArtifactDigest::from_hex(wire.trajectory_v3_sha256.clone())?;
    ArtifactDigest::from_hex(wire.source_prefix_v2_sha256.clone())?;
    if wire.window_length == 0 {
        return Err(invalid_artifact(
            "ML Dataset window length must be positive",
        ));
    }
    if wire.descriptors.is_empty() || wire.descriptors.len() > limits.max_ml_dataset_descriptors {
        return Err(invalid_artifact(
            "ML Dataset descriptor count is empty or exceeds its decoder limit",
        ));
    }
    for descriptor in &wire.descriptors {
        validate_descriptor(descriptor, wire.window_length, limits)?;
    }
    if wire.descriptors.windows(2).any(|pair| {
        let left = MlDatasetFieldDescriptorV1 {
            wire: pair[0].clone(),
        };
        let right = MlDatasetFieldDescriptorV1 {
            wire: pair[1].clone(),
        };
        left.cmp_key(&right) != Ordering::Less
    }) {
        return Err(invalid_artifact(
            "ML Dataset descriptors must be unique in role-offset-Field canonical order",
        ));
    }
    if !wire
        .descriptors
        .iter()
        .any(|descriptor| descriptor.role == MlDatasetDescriptorRoleV1::Feature)
        || !wire
            .descriptors
            .iter()
            .any(|descriptor| descriptor.role == MlDatasetDescriptorRoleV1::Target)
    {
        return Err(invalid_artifact(
            "ML Dataset requires at least one feature and one target descriptor",
        ));
    }
    if wire.samples.len() < 3 || wire.samples.len() > limits.max_ml_dataset_samples {
        return Err(invalid_artifact(
            "ML Dataset requires bounded nonempty training, validation, and test samples",
        ));
    }

    let descriptors = wire
        .descriptors
        .iter()
        .cloned()
        .map(|wire| MlDatasetFieldDescriptorV1 { wire })
        .collect::<Vec<_>>();
    let mut total_states = 0_usize;
    let mut total_observations = 0_usize;
    let mut total_blocks = 0_usize;
    let mut state_splits = BTreeMap::<&str, MlDatasetSampleSplitV1>::new();
    let mut strict_time_frames = BTreeMap::<u64, &WireStateReference>::new();
    let mut split_bounds = BTreeMap::<MlDatasetSampleSplitV1, (u64, u64)>::new();
    let mut association_inventory = vec![None::<Vec<WireAssociation>>; descriptors.len()];
    for (index, sample) in wire.samples.iter().enumerate() {
        if usize::try_from(sample.ordinal).ok() != Some(index)
            || sample.states.len() != wire.window_length as usize
        {
            return Err(invalid_artifact(
                "ML Dataset sample ordinals or fixed window lengths are noncanonical",
            ));
        }
        validate_sample(sample, &descriptors)?;
        total_states = checked_add(total_states, sample.states.len(), "ML Dataset states")?;
        total_observations = checked_add(
            total_observations,
            sample.observations.len(),
            "ML Dataset observations",
        )?;
        for state in &sample.states {
            if state_splits
                .insert(&state.spatial_state_sha256, sample.split)
                .is_some_and(|prior| prior != sample.split)
            {
                return Err(invalid_artifact(
                    "ML Dataset partitions overlap through one exact SpatialState",
                ));
            }
            if strict_time_frames
                .insert(state.frame_ordinal, state)
                .is_some_and(|prior| prior != state)
            {
                return Err(invalid_artifact(
                    "ML Dataset assigns different state identities to one frame ordinal",
                ));
            }
        }
        let first = sample.states[0].frame_ordinal;
        let last = sample.states[sample.states.len() - 1].frame_ordinal;
        split_bounds
            .entry(sample.split)
            .and_modify(|bounds| {
                bounds.0 = bounds.0.min(first);
                bounds.1 = bounds.1.max(last);
            })
            .or_insert((first, last));
        for observation in &sample.observations {
            total_blocks =
                checked_add(total_blocks, observation.blocks.len(), "ML Dataset blocks")?;
            let descriptor = usize::try_from(observation.descriptor_ordinal)
                .map_err(|_| invalid_artifact("ML Dataset descriptor ordinal exceeds usize"))?;
            let associations = observation
                .blocks
                .iter()
                .map(|block| block.association)
                .collect::<Vec<_>>();
            match &association_inventory[descriptor] {
                None => association_inventory[descriptor] = Some(associations),
                Some(expected) if expected == &associations => {}
                Some(_) => {
                    return Err(invalid_artifact(
                        "ML Dataset coefficient-block inventory changes across samples",
                    ));
                }
            }
        }
    }
    if total_states > limits.max_ml_dataset_window_states
        || total_observations > limits.max_ml_dataset_observations
        || total_blocks > limits.max_ml_dataset_blocks
    {
        return Err(invalid_artifact(
            "ML Dataset nested sample inventory exceeds a decoder limit",
        ));
    }
    for pair in strict_time_frames
        .values()
        .copied()
        .collect::<Vec<_>>()
        .windows(2)
    {
        if pair[0].step >= pair[1].step
            || pair[0].time_s >= pair[1].time_s
            || (pair[0].state_kind == MlDatasetStateKindV1::RemeshedV3
                && pair[1].state_kind == MlDatasetStateKindV1::MovingV2)
        {
            return Err(invalid_artifact(
                "ML Dataset selected frames must preserve one strict V2-prefix/V3-suffix time sequence",
            ));
        }
    }
    let training = split_bounds
        .get(&MlDatasetSampleSplitV1::Training)
        .ok_or_else(|| invalid_artifact("ML Dataset has no training partition"))?;
    let validation = split_bounds
        .get(&MlDatasetSampleSplitV1::Validation)
        .ok_or_else(|| invalid_artifact("ML Dataset has no validation partition"))?;
    let test = split_bounds
        .get(&MlDatasetSampleSplitV1::Test)
        .ok_or_else(|| invalid_artifact("ML Dataset has no test partition"))?;
    if training.1 >= validation.0 || validation.1 >= test.0 {
        return Err(invalid_artifact(
            "ML Dataset partitions must be disjoint and ordered training-validation-test in time",
        ));
    }
    if wire
        .samples
        .windows(2)
        .any(|pair| pair[0].start_frame_ordinal >= pair[1].start_frame_ordinal)
    {
        return Err(invalid_artifact(
            "ML Dataset sample starts must increase strictly",
        ));
    }

    validate_all_statistics(
        &wire.normalization.channels,
        &descriptors,
        &association_inventory,
        limits,
    )
}

fn validate_descriptor(
    descriptor: &WireDescriptor,
    window_length: u32,
    limits: MlDatasetDecoderLimits,
) -> Result<(), Diagnostic> {
    if descriptor.window_offset >= window_length {
        return Err(invalid_artifact(
            "ML Dataset descriptor offset lies outside its sample window",
        ));
    }
    parse_id::<kinds::Field>(&descriptor.field_ulid, "Field")?;
    parse_id::<kinds::Domain>(&descriptor.support_domain_ulid, "support Domain")?;
    let shape = descriptor.physical.value_shape.decode()?;
    if shape.rank() > limits.max_value_shape_rank
        || shape
            .component_count()
            .is_none_or(|count| count == 0 || count > limits.max_value_shape_components)
    {
        return Err(invalid_artifact(
            "ML Dataset descriptor value shape exceeds a decoder limit",
        ));
    }
    Ok(())
}

fn validate_sample(
    sample: &WireSample,
    descriptors: &[MlDatasetFieldDescriptorV1],
) -> Result<(), Diagnostic> {
    if sample.states.is_empty()
        || sample.start_frame_ordinal != sample.states[0].frame_ordinal
        || sample.observations.len() != descriptors.len()
    {
        return Err(invalid_artifact(
            "ML Dataset sample has an empty window or incomplete descriptor observations",
        ));
    }
    for state in &sample.states {
        validate_state(state)?;
    }
    for pair in sample.states.windows(2) {
        if pair[1].frame_ordinal
            != pair[0]
                .frame_ordinal
                .checked_add(1)
                .ok_or_else(|| invalid_artifact("ML Dataset strict-time frame ordinal overflows"))?
            || pair[0].step >= pair[1].step
            || pair[0].time_s >= pair[1].time_s
            || (pair[0].state_kind == MlDatasetStateKindV1::RemeshedV3
                && pair[1].state_kind == MlDatasetStateKindV1::MovingV2)
        {
            return Err(invalid_artifact(
                "ML Dataset sample states must be contiguous in strict V2-prefix/V3-suffix time",
            ));
        }
    }
    for (index, observation) in sample.observations.iter().enumerate() {
        validate_observation(observation)?;
        if usize::try_from(observation.descriptor_ordinal).ok() != Some(index) {
            return Err(invalid_artifact(
                "ML Dataset observations must follow exact descriptor order",
            ));
        }
        let descriptor = &descriptors[index];
        let state = &sample.states[descriptor.window_offset() as usize];
        if observation.spatial_state_sha256 != state.spatial_state_sha256
            || observation.reference_mesh_sha256 != state.reference_mesh_sha256
        {
            return Err(invalid_artifact(
                "ML Dataset observation is not attached to its descriptor-offset state",
            ));
        }
    }
    Ok(())
}

fn validate_state(state: &WireStateReference) -> Result<(), Diagnostic> {
    if !is_canonical_time(state.time_s) {
        return Err(invalid_artifact(
            "ML Dataset state time must be finite nonnegative canonical time",
        ));
    }
    for digest in [
        &state.spatial_state_sha256,
        &state.reference_mesh_sha256,
        &state.geometry_state_sha256,
    ] {
        ArtifactDigest::from_hex(digest.clone())?;
    }
    Ok(())
}

fn validate_observation(observation: &WireObservationReference) -> Result<(), Diagnostic> {
    for digest in [
        &observation.spatial_state_sha256,
        &observation.reference_mesh_sha256,
        &observation.snapshot_sha256,
    ] {
        ArtifactDigest::from_hex(digest.clone())?;
    }
    if observation.blocks.is_empty() {
        return Err(invalid_artifact(
            "ML Dataset observation must retain at least one coefficient block",
        ));
    }
    for block in &observation.blocks {
        ArtifactDigest::from_hex(block.discrete_field_sha256.clone())?;
    }
    if observation
        .blocks
        .windows(2)
        .any(|pair| pair[0].association >= pair[1].association)
    {
        return Err(invalid_artifact(
            "ML Dataset observation blocks must be unique in canonical association order",
        ));
    }
    Ok(())
}

fn validate_all_statistics(
    channels: &[WireChannelStatistics],
    descriptors: &[MlDatasetFieldDescriptorV1],
    associations: &[Option<Vec<WireAssociation>>],
    limits: MlDatasetDecoderLimits,
) -> Result<(), Diagnostic> {
    if channels.is_empty() || channels.len() > limits.max_ml_dataset_normalization_channels {
        return Err(invalid_artifact(
            "ML Dataset normalization channels are empty or exceed their decoder limit",
        ));
    }
    for channel in channels {
        validate_statistics(channel)?;
    }
    if channels
        .windows(2)
        .any(|pair| statistics_key(&pair[0]) >= statistics_key(&pair[1]))
    {
        return Err(invalid_artifact(
            "ML Dataset normalization channels must be unique in descriptor-block-component order",
        ));
    }
    let mut expected = Vec::new();
    for (descriptor_ordinal, descriptor) in descriptors.iter().enumerate() {
        let component_count = descriptor
            .value_shape()
            .component_count()
            .expect("validated descriptor component count");
        let descriptor_ordinal = u32::try_from(descriptor_ordinal)
            .map_err(|_| invalid_artifact("ML Dataset descriptor ordinal exceeds portable u32"))?;
        for association in associations[usize::try_from(descriptor_ordinal).expect("u32 to usize")]
            .as_ref()
            .expect("every descriptor has one observation")
        {
            for component in 0..component_count {
                expected.push((
                    descriptor_ordinal,
                    *association,
                    u32::try_from(component).map_err(|_| {
                        invalid_artifact("ML Dataset component ordinal exceeds portable u32")
                    })?,
                ));
            }
        }
    }
    let actual = channels.iter().map(statistics_key).collect::<Vec<_>>();
    if actual != expected {
        return Err(invalid_artifact(
            "ML Dataset normalization channels do not cover every descriptor block component",
        ));
    }
    Ok(())
}

fn validate_statistics(statistics: &WireChannelStatistics) -> Result<(), Diagnostic> {
    if statistics.population_count == 0
        || !statistics.mean.is_finite()
        || !statistics.population_standard_deviation.is_finite()
        || statistics.population_standard_deviation < 0.0
        || !statistics.scale.is_finite()
        || statistics.scale <= 0.0
        || is_negative_zero(statistics.mean)
        || is_negative_zero(statistics.population_standard_deviation)
        || is_negative_zero(statistics.scale)
    {
        return Err(invalid_artifact(
            "ML Dataset population statistics contain an invalid scalar or empty population",
        ));
    }
    let expected_scale = if statistics.constant {
        1.0
    } else {
        statistics.population_standard_deviation
    };
    if (statistics.constant && statistics.population_standard_deviation != 0.0)
        || (!statistics.constant && statistics.population_standard_deviation <= 0.0)
        || statistics.scale != expected_scale
    {
        return Err(invalid_artifact(
            "ML Dataset constant-channel marker and population scale are inconsistent",
        ));
    }
    Ok(())
}

fn statistics_key(statistics: &WireChannelStatistics) -> (u32, WireAssociation, u32) {
    (
        statistics.descriptor_ordinal,
        statistics.association,
        statistics.component,
    )
}

fn checked_add(left: usize, right: usize, label: &str) -> Result<usize, Diagnostic> {
    left.checked_add(right)
        .ok_or_else(|| invalid_artifact(format!("{label} count overflows usize")))
}

fn parse_digest(value: &str) -> ArtifactDigest {
    ArtifactDigest::from_hex(value.to_owned()).expect("validated artifact digest")
}

fn parse_id<K: eqiora_core::Entity>(value: &str, label: &str) -> Result<Id<K>, Diagnostic> {
    let ulid = Ulid::from_str(value)
        .map_err(|_| invalid_artifact(format!("ML Dataset {label} ULID is malformed")))?;
    if ulid.to_string() != value {
        return Err(invalid_artifact(format!(
            "ML Dataset {label} ULID spelling is noncanonical"
        )));
    }
    Ok(Id::from_ulid(ulid))
}

fn is_canonical_time(value: f64) -> bool {
    value.is_finite() && value >= 0.0 && !is_negative_zero(value)
}

fn is_negative_zero(value: f64) -> bool {
    value == 0.0 && value.is_sign_negative()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireEnvelope {
    schema: String,
    encoding: String,
    trajectory_v3_sha256: String,
    source_prefix_v2_sha256: String,
    remesh_seam_policy: String,
    window_length: u32,
    descriptors: Vec<WireDescriptor>,
    samples: Vec<WireSample>,
    normalization: WireNormalization,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireDescriptor {
    role: MlDatasetDescriptorRoleV1,
    window_offset: u32,
    field_ulid: String,
    support_domain_ulid: String,
    physical: WirePhysicalType,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WirePhysicalType {
    unit_system: WireUnitSystem,
    dimension: WireDimension,
    value_shape: WireValueShape,
    frame: WireFrame,
}

impl WirePhysicalType {
    fn encode(dimension: DimExponents, shape: &ValueShape, frame: ValueFrame) -> Self {
        Self {
            unit_system: WireUnitSystem::CoherentSi,
            dimension: WireDimension::encode(dimension),
            value_shape: WireValueShape::encode(shape),
            frame: WireFrame::encode(frame),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireUnitSystem {
    CoherentSi,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireValueShape {
    extents: Vec<u32>,
}

impl WireValueShape {
    fn encode(value: &ValueShape) -> Self {
        Self {
            extents: value.extents().iter().map(|extent| extent.get()).collect(),
        }
    }

    fn decode(&self) -> Result<ValueShape, Diagnostic> {
        ValueShape::new(self.extents.iter().copied())
            .map_err(|_| invalid_artifact("ML Dataset value shape contains a zero extent"))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireFrame {
    Invariant,
    SpatialCartesian,
}

impl WireFrame {
    const fn encode(value: ValueFrame) -> Self {
        match value {
            ValueFrame::Invariant => Self::Invariant,
            ValueFrame::SpatialCartesian => Self::SpatialCartesian,
        }
    }

    const fn decode(self) -> ValueFrame {
        match self {
            Self::Invariant => ValueFrame::Invariant,
            Self::SpatialCartesian => ValueFrame::SpatialCartesian,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireSample {
    ordinal: u64,
    split: MlDatasetSampleSplitV1,
    start_frame_ordinal: u64,
    states: Vec<WireStateReference>,
    observations: Vec<WireObservationReference>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireStateReference {
    frame_ordinal: u64,
    step: u64,
    time_s: f64,
    state_kind: MlDatasetStateKindV1,
    spatial_state_sha256: String,
    reference_mesh_sha256: String,
    geometry_state_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireObservationReference {
    descriptor_ordinal: u32,
    spatial_state_sha256: String,
    reference_mesh_sha256: String,
    snapshot_sha256: String,
    blocks: Vec<WireBlockReference>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireBlockReference {
    association: WireAssociation,
    discrete_field_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireNormalization {
    method: WireNormalizationMethod,
    accumulator: WireAccumulatorProfile,
    statistics_source: MlDatasetSampleSplitV1,
    channels: Vec<WireChannelStatistics>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireNormalizationMethod {
    PopulationStandardScore,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireAccumulatorProfile {
    OrderedWelfordBinary64V1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireChannelStatistics {
    descriptor_ordinal: u32,
    association: WireAssociation,
    component: u32,
    population_count: u64,
    mean: f64,
    population_standard_deviation: f64,
    scale: f64,
    constant: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireAssociation {
    Vertex,
    Cell,
}

impl WireAssociation {
    const fn into_public(self) -> DiscreteFieldAssociation {
        match self {
            Self::Vertex => DiscreteFieldAssociation::Vertex,
            Self::Cell => DiscreteFieldAssociation::Cell,
        }
    }
}

impl From<DiscreteFieldAssociation> for WireAssociation {
    fn from(value: DiscreteFieldAssociation) -> Self {
        match value {
            DiscreteFieldAssociation::Vertex => Self::Vertex,
            DiscreteFieldAssociation::Cell => Self::Cell,
        }
    }
}

impl From<WireAssociation> for DiscreteFieldAssociation {
    fn from(value: WireAssociation) -> Self {
        value.into_public()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(seed: u8) -> String {
        format!("{seed:02x}").repeat(32)
    }

    fn descriptor(role: MlDatasetDescriptorRoleV1, field: &str) -> WireDescriptor {
        WireDescriptor {
            role,
            window_offset: 0,
            field_ulid: field.to_owned(),
            support_domain_ulid: "01ARZ3NDEKTSV4RRFFQ69G5FAV".to_owned(),
            physical: WirePhysicalType::encode(
                DimExponents::DIMENSIONLESS,
                &ValueShape::scalar(),
                ValueFrame::Invariant,
            ),
        }
    }

    fn state(frame: u64, seed: u8, kind: MlDatasetStateKindV1) -> WireStateReference {
        WireStateReference {
            frame_ordinal: frame,
            step: frame,
            time_s: frame as f64,
            state_kind: kind,
            spatial_state_sha256: digest(seed),
            reference_mesh_sha256: digest(if kind == MlDatasetStateKindV1::MovingV2 {
                80
            } else {
                81
            }),
            geometry_state_sha256: digest(seed + 20),
        }
    }

    fn observation(
        descriptor: u32,
        state: &WireStateReference,
        seed: u8,
    ) -> WireObservationReference {
        WireObservationReference {
            descriptor_ordinal: descriptor,
            spatial_state_sha256: state.spatial_state_sha256.clone(),
            reference_mesh_sha256: state.reference_mesh_sha256.clone(),
            snapshot_sha256: digest(seed),
            blocks: vec![WireBlockReference {
                association: WireAssociation::Vertex,
                discrete_field_sha256: digest(seed + 1),
            }],
        }
    }

    fn fixture() -> WireEnvelope {
        let descriptors = vec![
            descriptor(
                MlDatasetDescriptorRoleV1::Feature,
                "01ARZ3NDEKTSV4RRFFQ69G5FAW",
            ),
            descriptor(
                MlDatasetDescriptorRoleV1::Target,
                "01ARZ3NDEKTSV4RRFFQ69G5FAX",
            ),
        ];
        let mut samples = Vec::new();
        for (index, split) in [
            MlDatasetSampleSplitV1::Training,
            MlDatasetSampleSplitV1::Validation,
            MlDatasetSampleSplitV1::Test,
        ]
        .into_iter()
        .enumerate()
        {
            let state = state(
                index as u64,
                10 + index as u8,
                if index == 0 {
                    MlDatasetStateKindV1::MovingV2
                } else {
                    MlDatasetStateKindV1::RemeshedV3
                },
            );
            samples.push(WireSample {
                ordinal: index as u64,
                split,
                start_frame_ordinal: index as u64,
                states: vec![state.clone()],
                observations: vec![
                    observation(0, &state, 30 + index as u8 * 4),
                    observation(1, &state, 32 + index as u8 * 4),
                ],
            });
        }
        WireEnvelope {
            schema: SCHEMA.to_owned(),
            encoding: CANONICAL_ENCODING.to_owned(),
            trajectory_v3_sha256: digest(1),
            source_prefix_v2_sha256: digest(2),
            remesh_seam_policy: SEAM_POLICY.to_owned(),
            window_length: 1,
            descriptors,
            samples,
            normalization: WireNormalization {
                method: WireNormalizationMethod::PopulationStandardScore,
                accumulator: WireAccumulatorProfile::OrderedWelfordBinary64V1,
                statistics_source: MlDatasetSampleSplitV1::Training,
                channels: vec![
                    WireChannelStatistics {
                        descriptor_ordinal: 0,
                        association: WireAssociation::Vertex,
                        component: 0,
                        population_count: 4,
                        mean: 2.0,
                        population_standard_deviation: 0.0,
                        scale: 1.0,
                        constant: true,
                    },
                    WireChannelStatistics {
                        descriptor_ordinal: 1,
                        association: WireAssociation::Vertex,
                        component: 0,
                        population_count: 4,
                        mean: 3.0,
                        population_standard_deviation: 2.0,
                        scale: 2.0,
                        constant: false,
                    },
                ],
            },
        }
    }

    #[test]
    fn closed_wire_roundtrips_without_values_or_storage_layout() {
        let wire = fixture();
        validate_wire(&wire, MlDatasetDecoderLimits::default()).unwrap();
        let bytes = serde_json::to_vec(&wire).unwrap();
        let decoded =
            MlDatasetEnvelopeV2::from_json(&bytes, MlDatasetDecoderLimits::default()).unwrap();
        assert_eq!(decoded.canonical_json().unwrap(), bytes);
        let json = String::from_utf8(bytes).unwrap();
        assert!(!json.contains("values"));
        assert!(!json.contains("path"));
        assert!(decoded.statistics()[0].is_constant());
        assert_eq!(decoded.statistics()[0].scale(), 1.0);
    }

    #[test]
    fn rational_dimension_roundtrips_and_old_schema_is_rejected() {
        let mut wire = fixture();
        let dimension =
            DimExponents::from_rationals([(0, 1), (-1, 2), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)])
                .unwrap();
        wire.descriptors[0].physical.dimension = WireDimension::encode(dimension);
        let bytes = serde_json::to_vec(&wire).unwrap();
        let decoded = MlDatasetEnvelopeV2::from_json(&bytes, Default::default()).unwrap();
        assert_eq!(decoded.descriptors()[0].dimension(), dimension);
        assert_eq!(decoded.canonical_json().unwrap(), bytes);

        wire.schema = "eqiora.ml-dataset-envelope/v1".to_owned();
        assert!(
            MlDatasetEnvelopeV2::from_json(&serde_json::to_vec(&wire).unwrap(), Default::default())
                .is_err()
        );
    }

    #[test]
    fn split_overlap_statistics_drift_and_limits_fail_closed() {
        let mut overlap = fixture();
        overlap.samples[1].states[0].spatial_state_sha256 =
            overlap.samples[0].states[0].spatial_state_sha256.clone();
        assert!(validate_wire(&overlap, MlDatasetDecoderLimits::default()).is_err());

        let mut leaked_scale = fixture();
        leaked_scale.normalization.channels[0].scale = 0.0;
        assert!(validate_wire(&leaked_scale, MlDatasetDecoderLimits::default()).is_err());

        let mut reordered = fixture();
        reordered.descriptors.swap(0, 1);
        assert!(validate_wire(&reordered, MlDatasetDecoderLimits::default()).is_err());

        for limits in [
            MlDatasetDecoderLimits {
                max_ml_dataset_descriptors: 1,
                ..MlDatasetDecoderLimits::default()
            },
            MlDatasetDecoderLimits {
                max_ml_dataset_samples: 2,
                ..MlDatasetDecoderLimits::default()
            },
            MlDatasetDecoderLimits {
                max_ml_dataset_window_states: 2,
                ..MlDatasetDecoderLimits::default()
            },
            MlDatasetDecoderLimits {
                max_ml_dataset_observations: 5,
                ..MlDatasetDecoderLimits::default()
            },
            MlDatasetDecoderLimits {
                max_ml_dataset_blocks: 5,
                ..MlDatasetDecoderLimits::default()
            },
            MlDatasetDecoderLimits {
                max_ml_dataset_normalization_channels: 1,
                ..MlDatasetDecoderLimits::default()
            },
        ] {
            assert!(validate_wire(&fixture(), limits).is_err());
        }
    }

    #[test]
    fn one_window_may_cross_the_exact_v2_to_v3_seam_without_dense_shape_claims() {
        let mut descriptors = vec![
            MlDatasetFieldDescriptorV1 {
                wire: descriptor(
                    MlDatasetDescriptorRoleV1::Feature,
                    "01ARZ3NDEKTSV4RRFFQ69G5FAW",
                ),
            },
            MlDatasetFieldDescriptorV1 {
                wire: descriptor(
                    MlDatasetDescriptorRoleV1::Target,
                    "01ARZ3NDEKTSV4RRFFQ69G5FAX",
                ),
            },
        ];
        descriptors[1].wire.window_offset = 1;
        let source = state(0, 10, MlDatasetStateKindV1::MovingV2);
        let target = state(1, 11, MlDatasetStateKindV1::RemeshedV3);
        let sample = WireSample {
            ordinal: 0,
            split: MlDatasetSampleSplitV1::Training,
            start_frame_ordinal: 0,
            states: vec![source.clone(), target.clone()],
            observations: vec![observation(0, &source, 30), observation(1, &target, 32)],
        };
        validate_sample(&sample, &descriptors).unwrap();
        assert_ne!(source.reference_mesh_sha256, target.reference_mesh_sha256);
    }

    #[test]
    fn noncanonical_seam_and_unknown_fields_fail_closed() {
        let mut seam = fixture();
        seam.samples[2].states[0].state_kind = MlDatasetStateKindV1::MovingV2;
        let bytes = serde_json::to_vec(&seam).unwrap();
        assert!(MlDatasetEnvelopeV2::from_json(&bytes, MlDatasetDecoderLimits::default()).is_err());

        let bytes = serde_json::to_vec(&fixture()).unwrap();
        let mut json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        json.as_object_mut()
            .unwrap()
            .insert("storage_path".to_owned(), serde_json::json!("hidden"));
        assert!(
            MlDatasetEnvelopeV2::from_json(
                &serde_json::to_vec(&json).unwrap(),
                MlDatasetDecoderLimits::default(),
            )
            .is_err()
        );
    }
}
