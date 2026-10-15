//! Bounded, storage-independent ML Dataset derivation and CPU materialization.

use std::collections::BTreeMap;

use eqiora_artifact::{
    ArtifactDigest, DiscreteFieldEnvelopeV1, FieldSnapshotEnvelopeV2, MlDatasetChannelStatisticsV1,
    MlDatasetDecoderLimits, MlDatasetDescriptorRoleV1, MlDatasetEnvelope,
    MlDatasetFieldDescriptorV1, MlDatasetObservationReferenceV1, MlDatasetSampleSplitV1,
    MlDatasetSampleV1, MlDatasetStateReferenceV1, ReplayableCanonicalModelArtifact,
};
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, Id};
use eqiora_meshing::DiscreteFieldAssociation;

use crate::remeshing_trajectory::{
    MovingSpatialFrame, RemeshingTrajectoryReplayInput, ReplayedField,
};

/// One semantic Field selected for one role and window offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MlDatasetFieldSelectionV1 {
    role: MlDatasetDescriptorRoleV1,
    window_offset: u32,
    field: Id<kinds::Field>,
}

impl MlDatasetFieldSelectionV1 {
    /// Select one Field without asserting its support, units, shape, or frame.
    ///
    /// Those properties are derived from exact snapshots during replay.
    #[must_use]
    pub const fn new(
        role: MlDatasetDescriptorRoleV1,
        window_offset: u32,
        field: Id<kinds::Field>,
    ) -> Self {
        Self {
            role,
            window_offset,
            field,
        }
    }

    /// Feature or target role.
    #[must_use]
    pub const fn role(self) -> MlDatasetDescriptorRoleV1 {
        self.role
    }

    /// Zero-based position inside each sample window.
    #[must_use]
    pub const fn window_offset(self) -> u32 {
        self.window_offset
    }

    /// Exact Semantic Field identity.
    #[must_use]
    pub const fn field(self) -> Id<kinds::Field> {
        self.field
    }
}

/// One explicit sample start and its closed partition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MlDatasetSampleSelectionV1 {
    start_frame_ordinal: usize,
    split: MlDatasetSampleSplitV1,
}

impl MlDatasetSampleSelectionV1 {
    /// Select a window by its first strict-time frame.
    #[must_use]
    pub const fn new(start_frame_ordinal: usize, split: MlDatasetSampleSplitV1) -> Self {
        Self {
            start_frame_ordinal,
            split,
        }
    }

    /// First strict-time frame selected by this sample.
    #[must_use]
    pub const fn start_frame_ordinal(self) -> usize {
        self.start_frame_ordinal
    }

    /// Training, validation, or test partition.
    #[must_use]
    pub const fn split(self) -> MlDatasetSampleSplitV1 {
        self.split
    }
}

/// Closed authoring intent for one deterministic Dataset derivation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlDatasetDerivationPlanV1 {
    window_length: u32,
    fields: Vec<MlDatasetFieldSelectionV1>,
    samples: Vec<MlDatasetSampleSelectionV1>,
}

impl MlDatasetDerivationPlanV1 {
    /// Normalize declaration order and reject duplicate or incomplete intent.
    ///
    /// # Errors
    /// Returns `EQ0901` unless the plan has a positive window, at least one
    /// feature and target, unique descriptors, and unique sample starts.
    pub fn new(
        window_length: u32,
        fields: impl IntoIterator<Item = MlDatasetFieldSelectionV1>,
        samples: impl IntoIterator<Item = MlDatasetSampleSelectionV1>,
    ) -> Result<Self, Diagnostic> {
        if window_length == 0 {
            return Err(dataset_error("ML Dataset window length must be positive"));
        }
        let decoder_limits = MlDatasetDecoderLimits::default();
        let mut normalized_fields = Vec::new();
        for field in fields {
            if normalized_fields.len() == decoder_limits.max_ml_dataset_descriptors {
                return Err(dataset_error(
                    "ML Dataset Field selections exceed the manifest descriptor budget",
                ));
            }
            normalized_fields.push(field);
        }
        let mut fields = normalized_fields;
        fields.sort_by(|left, right| {
            left.role
                .cmp(&right.role)
                .then_with(|| left.window_offset.cmp(&right.window_offset))
                .then_with(|| left.field.ulid().cmp(&right.field.ulid()))
        });
        if fields.is_empty()
            || fields.windows(2).any(|pair| {
                pair[0].role == pair[1].role
                    && pair[0].window_offset == pair[1].window_offset
                    && pair[0].field == pair[1].field
            })
            || fields
                .iter()
                .any(|selection| selection.window_offset >= window_length)
        {
            return Err(dataset_error(
                "ML Dataset Fields must be unique and select offsets inside the window",
            ));
        }
        if !fields
            .iter()
            .any(|selection| selection.role == MlDatasetDescriptorRoleV1::Feature)
            || !fields
                .iter()
                .any(|selection| selection.role == MlDatasetDescriptorRoleV1::Target)
        {
            return Err(dataset_error(
                "ML Dataset requires at least one feature and target selection",
            ));
        }
        let mut normalized_samples = Vec::new();
        for sample in samples {
            if normalized_samples.len() == decoder_limits.max_ml_dataset_samples {
                return Err(dataset_error(
                    "ML Dataset samples exceed the manifest sample budget",
                ));
            }
            normalized_samples.push(sample);
        }
        let mut samples = normalized_samples;
        samples.sort_by_key(|sample| sample.start_frame_ordinal);
        if samples.len() < 3
            || samples
                .windows(2)
                .any(|pair| pair[0].start_frame_ordinal == pair[1].start_frame_ordinal)
        {
            return Err(dataset_error(
                "ML Dataset requires at least three unique sample starts",
            ));
        }
        let window_length_usize = usize::try_from(window_length)
            .map_err(|_| dataset_error("ML Dataset window length exceeds local usize"))?;
        let window_states = samples
            .len()
            .checked_mul(window_length_usize)
            .ok_or_else(|| dataset_error("ML Dataset selected state count overflows usize"))?;
        let observations = samples
            .len()
            .checked_mul(fields.len())
            .ok_or_else(|| dataset_error("ML Dataset observation count overflows usize"))?;
        if samples.len() > decoder_limits.max_ml_dataset_samples
            || window_states > decoder_limits.max_ml_dataset_window_states
            || observations > decoder_limits.max_ml_dataset_observations
        {
            return Err(dataset_error(
                "ML Dataset sample intent exceeds a manifest resource budget",
            ));
        }
        Ok(Self {
            window_length,
            fields,
            samples,
        })
    }

    /// Fixed number of strict-time frames per sample.
    #[must_use]
    pub const fn window_length(&self) -> u32 {
        self.window_length
    }

    /// Canonically ordered Field selections.
    #[must_use]
    pub fn fields(&self) -> &[MlDatasetFieldSelectionV1] {
        &self.fields
    }

    /// Strictly start-ordered samples.
    #[must_use]
    pub fn samples(&self) -> &[MlDatasetSampleSelectionV1] {
        &self.samples
    }
}

/// Explicit work budgets for owned CPU materialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MlDatasetMaterializationLimitsV1 {
    /// Maximum materialized coefficient blocks.
    pub max_blocks: usize,
    /// Maximum active entity indices summed across all blocks.
    pub max_active_entities: usize,
    /// Maximum normalized scalar values summed across all blocks.
    pub max_scalar_values: usize,
}

impl Default for MlDatasetMaterializationLimitsV1 {
    fn default() -> Self {
        Self {
            max_blocks: 1_000_000,
            max_active_entities: 16_000_000,
            max_scalar_values: 64_000_000,
        }
    }
}

/// One owned normalized ragged block with complete logical lineage.
#[derive(Debug, Clone, PartialEq)]
pub struct MlDatasetBlockArrayV1 {
    descriptor_ordinal: u32,
    role: MlDatasetDescriptorRoleV1,
    window_offset: u32,
    field: Id<kinds::Field>,
    state_artifact: ArtifactDigest,
    snapshot_artifact: ArtifactDigest,
    block_artifact: ArtifactDigest,
    mesh_artifact: ArtifactDigest,
    association: DiscreteFieldAssociation,
    active_entity_indices: Vec<usize>,
    component_count: usize,
    values: Vec<f64>,
}

impl MlDatasetBlockArrayV1 {
    /// Descriptor ordinal in canonical Dataset order.
    #[must_use]
    pub const fn descriptor_ordinal(&self) -> u32 {
        self.descriptor_ordinal
    }

    /// Feature or target role.
    #[must_use]
    pub const fn role(&self) -> MlDatasetDescriptorRoleV1 {
        self.role
    }

    /// Selected state offset inside the sample window.
    #[must_use]
    pub const fn window_offset(&self) -> u32 {
        self.window_offset
    }

    /// Exact Semantic Field identity.
    #[must_use]
    pub const fn field(&self) -> Id<kinds::Field> {
        self.field
    }

    /// Exact selected SpatialState artifact.
    #[must_use]
    pub const fn state_artifact(&self) -> &ArtifactDigest {
        &self.state_artifact
    }

    /// Exact selected FieldSnapshot artifact.
    #[must_use]
    pub const fn snapshot_artifact(&self) -> &ArtifactDigest {
        &self.snapshot_artifact
    }

    /// Exact selected logical coefficient block.
    #[must_use]
    pub const fn block_artifact(&self) -> &ArtifactDigest {
        &self.block_artifact
    }

    /// Exact mesh owning the active entity indices.
    #[must_use]
    pub const fn mesh_artifact(&self) -> &ArtifactDigest {
        &self.mesh_artifact
    }

    /// Vertex or Cell coefficient association.
    #[must_use]
    pub const fn association(&self) -> DiscreteFieldAssociation {
        self.association
    }

    /// Sorted active support entities on the exact mesh.
    #[must_use]
    pub fn active_entity_indices(&self) -> &[usize] {
        &self.active_entity_indices
    }

    /// Number of mathematical components per active entity.
    #[must_use]
    pub const fn component_count(&self) -> usize {
        self.component_count
    }

    /// Owned entity-major, component-minor standardized values.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }
}

/// One sample's owned blocks in canonical descriptor and association order.
#[derive(Debug, Clone, PartialEq)]
pub struct MlDatasetSampleArraysV1 {
    ordinal: u64,
    split: MlDatasetSampleSplitV1,
    blocks: Vec<MlDatasetBlockArrayV1>,
}

impl MlDatasetSampleArraysV1 {
    /// Canonical sample ordinal.
    #[must_use]
    pub const fn ordinal(&self) -> u64 {
        self.ordinal
    }

    /// Closed Dataset partition.
    #[must_use]
    pub const fn split(&self) -> MlDatasetSampleSplitV1 {
        self.split
    }

    /// Owned ragged blocks in canonical order.
    #[must_use]
    pub fn blocks(&self) -> &[MlDatasetBlockArrayV1] {
        &self.blocks
    }
}

/// Complete bounded owned CPU projection of one logical Dataset.
#[derive(Debug, Clone, PartialEq)]
pub struct MlDatasetMaterializationV1 {
    dataset_artifact: ArtifactDigest,
    samples: Vec<MlDatasetSampleArraysV1>,
    scalar_count: usize,
}

impl MlDatasetMaterializationV1 {
    /// Exact logical Dataset artifact from which values were derived.
    #[must_use]
    pub const fn dataset_artifact(&self) -> &ArtifactDigest {
        &self.dataset_artifact
    }

    /// Strict source-time sample order.
    #[must_use]
    pub fn samples(&self) -> &[MlDatasetSampleArraysV1] {
        &self.samples
    }

    /// Complete owned normalized scalar count.
    #[must_use]
    pub const fn scalar_count(&self) -> usize {
        self.scalar_count
    }
}

/// Freshly derived logical artifact and its explicit owned CPU projection.
#[derive(Debug, Clone, PartialEq)]
pub struct MlDatasetArtifactsV1 {
    envelope: MlDatasetEnvelope,
    envelope_digest: ArtifactDigest,
    materialization: MlDatasetMaterializationV1,
}

impl MlDatasetArtifactsV1 {
    /// Storage-independent logical Dataset manifest.
    #[must_use]
    pub const fn envelope(&self) -> &MlDatasetEnvelope {
        &self.envelope
    }

    /// Exact logical Dataset identity computed during fresh derivation.
    #[must_use]
    pub const fn envelope_digest(&self) -> &ArtifactDigest {
        &self.envelope_digest
    }

    /// Explicitly owned ragged CPU values.
    #[must_use]
    pub const fn materialization(&self) -> &MlDatasetMaterializationV1 {
        &self.materialization
    }
}

/// Opaque proof of equality with one fresh trajectory derivation.
#[derive(Debug)]
pub struct VerifiedMlDatasetV1 {
    artifacts: MlDatasetArtifactsV1,
}

impl VerifiedMlDatasetV1 {
    /// Fresh exact replay artifacts.
    #[must_use]
    pub const fn artifacts(&self) -> &MlDatasetArtifactsV1 {
        &self.artifacts
    }
}

/// Derive one logical Dataset and bounded owned CPU materialization.
///
/// # Errors
/// Returns `EQ0901` for any stale dependency, type drift, invalid window or
/// split, non-finite statistic, or work-budget excess.
pub fn derive_ml_dataset_v1<M: ReplayableCanonicalModelArtifact>(
    input: &RemeshingTrajectoryReplayInput<'_, M>,
    plan: &MlDatasetDerivationPlanV1,
    limits: MlDatasetMaterializationLimitsV1,
) -> Result<MlDatasetArtifactsV1, Diagnostic> {
    derive_dataset(input, plan, limits)
}

/// Re-derive and compare a manifest and previously derived in-memory CPU arrays.
///
/// # Errors
/// Returns `EQ0901` for any substituted dependency, manifest, statistic,
/// active support, ordering, or materialized scalar.
pub fn verify_ml_dataset_v1<M: ReplayableCanonicalModelArtifact>(
    expected_envelope: &MlDatasetEnvelope,
    expected_materialization: &MlDatasetMaterializationV1,
    input: &RemeshingTrajectoryReplayInput<'_, M>,
    plan: &MlDatasetDerivationPlanV1,
    limits: MlDatasetMaterializationLimitsV1,
) -> Result<VerifiedMlDatasetV1, Diagnostic> {
    expected_envelope.validate_trajectory(input.trajectory())?;
    let artifacts = derive_dataset(input, plan, limits)?;
    if expected_envelope != &artifacts.envelope
        || expected_materialization != &artifacts.materialization
    {
        return Err(dataset_error(
            "retained ML Dataset differs from fresh exact trajectory derivation",
        ));
    }
    Ok(VerifiedMlDatasetV1 { artifacts })
}

fn derive_dataset<M: ReplayableCanonicalModelArtifact>(
    input: &RemeshingTrajectoryReplayInput<'_, M>,
    plan: &MlDatasetDerivationPlanV1,
    limits: MlDatasetMaterializationLimitsV1,
) -> Result<MlDatasetArtifactsV1, Diagnostic> {
    let frames = strict_time_frames(input)?;
    let window_length = usize::try_from(plan.window_length)
        .map_err(|_| dataset_error("ML Dataset window length exceeds local usize"))?;
    for sample in &plan.samples {
        let end = sample
            .start_frame_ordinal
            .checked_add(window_length)
            .ok_or_else(|| dataset_error("ML Dataset sample window overflows usize"))?;
        if end > frames.len() {
            return Err(dataset_error(
                "ML Dataset sample window exceeds the exact strict-time trajectory",
            ));
        }
    }

    let first_sample = plan
        .samples
        .first()
        .ok_or_else(|| dataset_error("ML Dataset has no samples"))?;
    let descriptors = plan
        .fields
        .iter()
        .map(|selection| {
            let frame = selected_frame(&frames, *first_sample, *selection)?;
            let field = frame.field(selection.field).ok_or_else(|| {
                dataset_error("ML Dataset Field is absent from a selected trajectory frame")
            })?;
            Ok(MlDatasetFieldDescriptorV1::from_snapshot(
                selection.role,
                selection.window_offset,
                field.snapshot,
            ))
        })
        .collect::<Result<Vec<_>, Diagnostic>>()?;
    preflight_work(input, plan, &frames, &descriptors, limits)?;

    let samples = plan
        .samples
        .iter()
        .enumerate()
        .map(|(ordinal, selection)| {
            derive_sample(ordinal, *selection, window_length, &frames, &descriptors)
        })
        .collect::<Result<Vec<_>, Diagnostic>>()?;
    let statistics = derive_statistics(input, plan, &frames, &descriptors, limits)?;
    let envelope = MlDatasetEnvelope::new(
        input.trajectory(),
        descriptors.clone(),
        samples,
        statistics.clone(),
    )?;
    let envelope_digest = envelope.digest()?;
    let materialization = materialize(
        input,
        plan,
        &frames,
        &descriptors,
        &statistics,
        envelope_digest.clone(),
        limits,
    )?;
    Ok(MlDatasetArtifactsV1 {
        envelope,
        envelope_digest,
        materialization,
    })
}

impl<'data> MovingSpatialFrame<'data> {
    fn field(&self, field: Id<kinds::Field>) -> Option<&ReplayedField<'data>> {
        self.fields()
            .iter()
            .find(|candidate| candidate.snapshot.field() == field)
    }
    fn fields(&self) -> &[ReplayedField<'data>] {
        &self.fields
    }
    fn state_reference(&self, ordinal: usize) -> Result<MlDatasetStateReferenceV1, Diagnostic> {
        MlDatasetStateReferenceV1::from_state(ordinal, self.state)
    }
    fn observation_reference(
        &self,
        descriptor_ordinal: usize,
        descriptor: &MlDatasetFieldDescriptorV1,
        snapshot: &FieldSnapshotEnvelopeV2,
    ) -> Result<MlDatasetObservationReferenceV1, Diagnostic> {
        MlDatasetObservationReferenceV1::from_state(
            descriptor_ordinal,
            descriptor,
            self.state,
            snapshot,
        )
    }
    fn state_artifact(&self) -> Result<ArtifactDigest, Diagnostic> {
        self.state.digest()
    }
    fn mesh_artifact(&self) -> ArtifactDigest {
        self.state.reference_mesh_artifact()
    }
    fn time_s(&self) -> f64 {
        self.state.time_s()
    }
    fn context<'a, M: ReplayableCanonicalModelArtifact>(
        &self,
        input: &'a RemeshingTrajectoryReplayInput<'data, M>,
    ) -> &'a eqiora_artifact::ValidatedMovingSpatialContext<'data, M> {
        match self.state.topology() {
            eqiora_artifact::SpatialStateTopology::Reference => input.source_context(),
            eqiora_artifact::SpatialStateTopology::Remeshed => input.target_context(),
        }
    }
}

fn strict_time_frames<'replay, 'data, M: ReplayableCanonicalModelArtifact>(
    input: &'replay RemeshingTrajectoryReplayInput<'data, M>,
) -> Result<Vec<&'replay MovingSpatialFrame<'data>>, Diagnostic> {
    let frames = input
        .source_frames()
        .iter()
        .take(input.source_frames().len().saturating_sub(1))
        .chain(input.target_frames().iter())
        .collect::<Vec<_>>();
    if frames.len() < 3
        || frames
            .windows(2)
            .any(|pair| pair[0].time_s() >= pair[1].time_s())
    {
        return Err(dataset_error(
            "ML Dataset requires at least three strict-time frames after remesh replacement",
        ));
    }
    Ok(frames)
}

fn selected_frame<'replay, 'data>(
    frames: &[&'replay MovingSpatialFrame<'data>],
    sample: MlDatasetSampleSelectionV1,
    field: MlDatasetFieldSelectionV1,
) -> Result<&'replay MovingSpatialFrame<'data>, Diagnostic> {
    let offset = usize::try_from(field.window_offset)
        .map_err(|_| dataset_error("ML Dataset window offset exceeds local usize"))?;
    let ordinal = sample
        .start_frame_ordinal
        .checked_add(offset)
        .ok_or_else(|| dataset_error("ML Dataset selected frame overflows usize"))?;
    frames
        .get(ordinal)
        .copied()
        .ok_or_else(|| dataset_error("ML Dataset selected frame is absent"))
}

fn derive_sample(
    ordinal: usize,
    selection: MlDatasetSampleSelectionV1,
    window_length: usize,
    frames: &[&MovingSpatialFrame<'_>],
    descriptors: &[MlDatasetFieldDescriptorV1],
) -> Result<MlDatasetSampleV1, Diagnostic> {
    let states = frames
        .iter()
        .copied()
        .skip(selection.start_frame_ordinal)
        .take(window_length)
        .enumerate()
        .map(|(offset, frame)| frame.state_reference(selection.start_frame_ordinal + offset))
        .collect::<Result<Vec<_>, Diagnostic>>()?;
    let observations = descriptors
        .iter()
        .enumerate()
        .map(|(descriptor_ordinal, descriptor)| {
            let frame_ordinal =
                selection
                    .start_frame_ordinal
                    .checked_add(usize::try_from(descriptor.window_offset()).map_err(|_| {
                        dataset_error("ML Dataset window offset exceeds local usize")
                    })?)
                    .ok_or_else(|| dataset_error("ML Dataset observation frame overflows usize"))?;
            let frame = frames
                .get(frame_ordinal)
                .copied()
                .ok_or_else(|| dataset_error("ML Dataset observation frame is absent"))?;
            let field = frame.field(descriptor.field()).ok_or_else(|| {
                dataset_error("ML Dataset Field is absent from an observation frame")
            })?;
            frame.observation_reference(descriptor_ordinal, descriptor, field.snapshot)
        })
        .collect::<Result<Vec<_>, Diagnostic>>()?;
    MlDatasetSampleV1::new(ordinal, selection.split, descriptors, states, observations)
}

fn preflight_work<M: ReplayableCanonicalModelArtifact>(
    input: &RemeshingTrajectoryReplayInput<'_, M>,
    plan: &MlDatasetDerivationPlanV1,
    frames: &[&MovingSpatialFrame<'_>],
    descriptors: &[MlDatasetFieldDescriptorV1],
    limits: MlDatasetMaterializationLimitsV1,
) -> Result<(), Diagnostic> {
    let mut total_blocks = 0_usize;
    let mut total_active_entities = 0_usize;
    let mut total_scalars = 0_usize;
    for sample in &plan.samples {
        for (selection, descriptor) in plan.fields.iter().zip(descriptors) {
            let frame = selected_frame(frames, *sample, *selection)?;
            let field = frame.field(descriptor.field()).ok_or_else(|| {
                dataset_error("ML Dataset preflight Field is absent from its exact frame")
            })?;
            for block in &field.blocks {
                let active_entities =
                    active_entity_count(input, frame, field.snapshot, block.association())?;
                let components = block.component_shape().component_count()?;
                total_blocks = checked_add(total_blocks, 1, "preflight blocks")?;
                total_active_entities = checked_add(
                    total_active_entities,
                    active_entities,
                    "preflight active entities",
                )?;
                total_scalars = checked_add(
                    total_scalars,
                    active_entities.checked_mul(components).ok_or_else(|| {
                        dataset_error("ML Dataset preflight scalar count overflows usize")
                    })?,
                    "preflight scalars",
                )?;
                if total_blocks > limits.max_blocks
                    || total_active_entities > limits.max_active_entities
                    || total_scalars > limits.max_scalar_values
                {
                    return Err(dataset_error(
                        "ML Dataset derivation exceeds an explicit pre-allocation work budget",
                    ));
                }
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, Default)]
struct OnlinePopulation {
    count: u64,
    mean: f64,
    squared_deviation_sum: f64,
    first: Option<f64>,
    all_equal: bool,
}

impl OnlinePopulation {
    fn push(&mut self, value: f64) -> Result<(), Diagnostic> {
        if !value.is_finite() {
            return Err(dataset_error("ML Dataset observation is not finite"));
        }
        match self.first {
            None => {
                self.first = Some(value);
                self.all_equal = true;
            }
            Some(first) if first == value => {}
            Some(_) => self.all_equal = false,
        }
        self.count = self
            .count
            .checked_add(1)
            .ok_or_else(|| dataset_error("ML Dataset population count overflows u64"))?;
        let count = self.count as f64;
        let delta = value - self.mean;
        self.mean += delta / count;
        let delta_after = value - self.mean;
        self.squared_deviation_sum += delta * delta_after;
        if !self.mean.is_finite() || !self.squared_deviation_sum.is_finite() {
            return Err(dataset_error(
                "ML Dataset normalization accumulation is not finite",
            ));
        }
        Ok(())
    }

    fn finish(self) -> Result<(u64, f64, f64, bool), Diagnostic> {
        if self.count == 0 || self.squared_deviation_sum < 0.0 {
            return Err(dataset_error(
                "ML Dataset normalization channel has an invalid population",
            ));
        }
        let mean = positive_zero(self.mean);
        let deviation = positive_zero((self.squared_deviation_sum / self.count as f64).sqrt());
        if !deviation.is_finite() {
            return Err(dataset_error(
                "ML Dataset population standard deviation is not finite",
            ));
        }
        Ok((self.count, mean, deviation, self.all_equal))
    }
}

fn derive_statistics<M: ReplayableCanonicalModelArtifact>(
    input: &RemeshingTrajectoryReplayInput<'_, M>,
    plan: &MlDatasetDerivationPlanV1,
    frames: &[&MovingSpatialFrame<'_>],
    descriptors: &[MlDatasetFieldDescriptorV1],
    limits: MlDatasetMaterializationLimitsV1,
) -> Result<Vec<MlDatasetChannelStatisticsV1>, Diagnostic> {
    let mut channels = BTreeMap::<(usize, u8, u32), OnlinePopulation>::new();
    let mut total_blocks = 0_usize;
    let mut total_active_entities = 0_usize;
    let mut total_scalars = 0_usize;
    for sample in plan
        .samples
        .iter()
        .filter(|sample| sample.split == MlDatasetSampleSplitV1::Training)
    {
        for (descriptor_ordinal, (selection, descriptor)) in
            plan.fields.iter().zip(descriptors).enumerate()
        {
            let frame = selected_frame(frames, *sample, *selection)?;
            let field = frame.field(descriptor.field()).ok_or_else(|| {
                dataset_error("ML Dataset training Field is absent from its exact frame")
            })?;
            for block in &field.blocks {
                let association = block.association();
                let association_key = association_order(association);
                let components = block.component_shape().component_count()?;
                let active = active_entities(input, frame, field.snapshot, association)?;
                total_blocks = checked_add(total_blocks, 1, "normalization blocks")?;
                total_active_entities = checked_add(
                    total_active_entities,
                    active.len(),
                    "normalization active entities",
                )?;
                total_scalars = checked_add(
                    total_scalars,
                    active.len().checked_mul(components).ok_or_else(|| {
                        dataset_error("ML Dataset normalization scalar count overflows usize")
                    })?,
                    "normalization scalars",
                )?;
                if total_blocks > limits.max_blocks
                    || total_active_entities > limits.max_active_entities
                    || total_scalars > limits.max_scalar_values
                {
                    return Err(dataset_error(
                        "ML Dataset normalization exceeds an explicit work budget",
                    ));
                }
                for entity in active {
                    for component in 0..components {
                        let index = entity
                            .checked_mul(components)
                            .and_then(|index| index.checked_add(component))
                            .ok_or_else(|| {
                                dataset_error("ML Dataset coefficient index overflows usize")
                            })?;
                        let value = *block.values().get(index).ok_or_else(|| {
                            dataset_error("ML Dataset active coefficient is absent")
                        })?;
                        channels
                            .entry((
                                descriptor_ordinal,
                                association_key,
                                u32::try_from(component).map_err(|_| {
                                    dataset_error("ML Dataset component exceeds portable u32")
                                })?,
                            ))
                            .or_default()
                            .push(value)?;
                    }
                }
            }
        }
    }
    channels
        .into_iter()
        .map(
            |((descriptor_ordinal, association, component), population)| {
                let (count, mean, standard_deviation, constant) = population.finish()?;
                MlDatasetChannelStatisticsV1::population_standard_score(
                    descriptor_ordinal,
                    association_from_order(association),
                    component,
                    count,
                    mean,
                    standard_deviation,
                    constant,
                )
            },
        )
        .collect()
}

#[derive(Debug)]
struct PlannedBlock<'replay, 'data> {
    descriptor_ordinal: usize,
    descriptor: &'replay MlDatasetFieldDescriptorV1,
    frame: &'replay MovingSpatialFrame<'data>,
    field: &'replay ReplayedField<'data>,
    block: &'data DiscreteFieldEnvelopeV1,
    active_entities: Vec<usize>,
}

fn materialize<M: ReplayableCanonicalModelArtifact>(
    input: &RemeshingTrajectoryReplayInput<'_, M>,
    plan: &MlDatasetDerivationPlanV1,
    frames: &[&MovingSpatialFrame<'_>],
    descriptors: &[MlDatasetFieldDescriptorV1],
    statistics: &[MlDatasetChannelStatisticsV1],
    dataset_artifact: ArtifactDigest,
    limits: MlDatasetMaterializationLimitsV1,
) -> Result<MlDatasetMaterializationV1, Diagnostic> {
    let stats = statistics
        .iter()
        .map(|statistics| {
            (
                (
                    usize::try_from(statistics.descriptor_ordinal()).expect("u32 fits usize"),
                    association_order(statistics.association()),
                    statistics.component(),
                ),
                statistics,
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut planned_samples = Vec::with_capacity(plan.samples.len());
    let mut total_blocks = 0_usize;
    let mut total_active_entities = 0_usize;
    let mut total_scalars = 0_usize;
    for sample in &plan.samples {
        let mut blocks = Vec::new();
        for (descriptor_ordinal, (selection, descriptor)) in
            plan.fields.iter().zip(descriptors).enumerate()
        {
            let frame = selected_frame(frames, *sample, *selection)?;
            let field = frame.field(descriptor.field()).ok_or_else(|| {
                dataset_error("ML Dataset materialization Field is absent from its exact frame")
            })?;
            for block in &field.blocks {
                let active_entities =
                    active_entities(input, frame, field.snapshot, block.association())?;
                let components = block.component_shape().component_count()?;
                let scalars = active_entities
                    .len()
                    .checked_mul(components)
                    .ok_or_else(|| {
                        dataset_error("ML Dataset block scalar count overflows usize")
                    })?;
                total_blocks = checked_add(total_blocks, 1, "materialized blocks")?;
                total_active_entities = checked_add(
                    total_active_entities,
                    active_entities.len(),
                    "materialized active entities",
                )?;
                total_scalars = checked_add(total_scalars, scalars, "materialized scalars")?;
                if total_blocks > limits.max_blocks
                    || total_active_entities > limits.max_active_entities
                    || total_scalars > limits.max_scalar_values
                {
                    return Err(dataset_error(
                        "ML Dataset materialization exceeds an explicit work budget",
                    ));
                }
                blocks.push(PlannedBlock {
                    descriptor_ordinal,
                    descriptor,
                    frame,
                    field,
                    block,
                    active_entities,
                });
            }
        }
        planned_samples.push((*sample, blocks));
    }

    let mut samples = Vec::with_capacity(planned_samples.len());
    for (ordinal, (sample, blocks)) in planned_samples.into_iter().enumerate() {
        let mut materialized_blocks = Vec::with_capacity(blocks.len());
        for planned in blocks {
            let components = planned.block.component_shape().component_count()?;
            let mut values = Vec::with_capacity(planned.active_entities.len() * components);
            for &entity in &planned.active_entities {
                for component in 0..components {
                    let source_index = entity
                        .checked_mul(components)
                        .and_then(|index| index.checked_add(component))
                        .ok_or_else(|| {
                            dataset_error("ML Dataset coefficient index overflows usize")
                        })?;
                    let source =
                        *planned.block.values().get(source_index).ok_or_else(|| {
                            dataset_error("ML Dataset active coefficient is absent")
                        })?;
                    let key = (
                        planned.descriptor_ordinal,
                        association_order(planned.block.association()),
                        u32::try_from(component).map_err(|_| {
                            dataset_error("ML Dataset component exceeds portable u32")
                        })?,
                    );
                    let statistics = stats.get(&key).ok_or_else(|| {
                        dataset_error("ML Dataset normalization channel is absent")
                    })?;
                    let normalized =
                        positive_zero((source - statistics.mean()) / statistics.scale());
                    if !normalized.is_finite() {
                        return Err(dataset_error(
                            "ML Dataset standardized coefficient is not finite",
                        ));
                    }
                    values.push(normalized);
                }
            }
            materialized_blocks.push(MlDatasetBlockArrayV1 {
                descriptor_ordinal: u32::try_from(planned.descriptor_ordinal).map_err(|_| {
                    dataset_error("ML Dataset descriptor ordinal exceeds portable u32")
                })?,
                role: planned.descriptor.role(),
                window_offset: planned.descriptor.window_offset(),
                field: planned.descriptor.field(),
                state_artifact: planned.frame.state_artifact()?,
                snapshot_artifact: planned.field.snapshot.digest()?,
                block_artifact: planned.block.digest()?,
                mesh_artifact: planned.frame.mesh_artifact(),
                association: planned.block.association(),
                active_entity_indices: planned.active_entities,
                component_count: components,
                values,
            });
        }
        samples.push(MlDatasetSampleArraysV1 {
            ordinal: u64::try_from(ordinal)
                .map_err(|_| dataset_error("ML Dataset sample ordinal exceeds portable u64"))?,
            split: sample.split,
            blocks: materialized_blocks,
        });
    }
    Ok(MlDatasetMaterializationV1 {
        dataset_artifact,
        samples,
        scalar_count: total_scalars,
    })
}

fn active_entities<M: ReplayableCanonicalModelArtifact>(
    input: &RemeshingTrajectoryReplayInput<'_, M>,
    frame: &MovingSpatialFrame<'_>,
    snapshot: &FieldSnapshotEnvelopeV2,
    association: DiscreteFieldAssociation,
) -> Result<Vec<usize>, Diagnostic> {
    snapshot.active_entities_against_moving(frame.context(input), association)
}

fn active_entity_count<M: ReplayableCanonicalModelArtifact>(
    input: &RemeshingTrajectoryReplayInput<'_, M>,
    frame: &MovingSpatialFrame<'_>,
    snapshot: &FieldSnapshotEnvelopeV2,
    association: DiscreteFieldAssociation,
) -> Result<usize, Diagnostic> {
    snapshot.active_entity_count_against_moving(frame.context(input), association)
}

const fn association_order(association: DiscreteFieldAssociation) -> u8 {
    match association {
        DiscreteFieldAssociation::Vertex => 0,
        DiscreteFieldAssociation::Cell => 1,
    }
}

const fn association_from_order(order: u8) -> DiscreteFieldAssociation {
    match order {
        0 => DiscreteFieldAssociation::Vertex,
        1 => DiscreteFieldAssociation::Cell,
        _ => unreachable!(),
    }
}

fn checked_add(left: usize, right: usize, label: &str) -> Result<usize, Diagnostic> {
    left.checked_add(right)
        .ok_or_else(|| dataset_error(format!("ML Dataset {label} count overflows usize")))
}

fn positive_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

fn dataset_error(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(eqiora_core::diagnostic::codes::INVALID_ARTIFACT, message)
}
