//! Immutable moving-spatial trajectory manifests.

use std::collections::BTreeSet;
use std::str::FromStr;

use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, Id};
use serde::{Deserialize, Serialize};
use ulid::Ulid;

use crate::spatial_state_v2::{SpatialStateEnvelopeV2, ValidatedMovingSpatialContextV2};
use crate::{
    ArtifactDigest, CANONICAL_ENCODING, ReplayableCanonicalModelArtifact, TrajectoryDecoderLimits,
    check_json_limits, invalid_artifact,
};

const SEGMENT_SCHEMA: &str = "eqiora.spatial-trajectory-segment/v2";
const TRAJECTORY_SCHEMA: &str = "eqiora.spatial-trajectory/v2";
const MAX_EXACT_F64_INTEGER: u64 = 1_u64 << 53;

/// Nonempty ordered sequence of fixed-topology moving states.
///
/// Every state shares one immutable reference context and complete Field
/// inventory. GeometryState predecessor identities must form an exact chain;
/// step/time monotonicity alone is insufficient.
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialTrajectorySegmentEnvelopeV2 {
    wire: WireTrajectorySegmentV2,
}

impl SpatialTrajectorySegmentEnvelopeV2 {
    /// Build one segment from states already ordered in accepted-state order.
    ///
    /// State order is semantic and is never sorted by this constructor.
    ///
    /// # Errors
    /// Returns `EQ0901` for empty, reordered, nonconsecutive, cross-resource,
    /// incomplete-Field, or broken GeometryState input.
    pub fn new<M: ReplayableCanonicalModelArtifact>(
        context: &ValidatedMovingSpatialContextV2<'_, M>,
        states: &[SpatialStateEnvelopeV2],
    ) -> Result<Self, Diagnostic> {
        let first = states
            .first()
            .ok_or_else(|| invalid_artifact("moving trajectory segment must contain a state"))?;
        let expected_fields = context_field_inventory(context);
        validate_state_context(context, first, &expected_fields)?;
        for state in states {
            validate_state_context(context, state, &expected_fields)?;
        }
        for pair in states.windows(2) {
            require_adjacent_states(&pair[0], &pair[1])?;
        }

        let value = Self {
            wire: WireTrajectorySegmentV2 {
                schema: SEGMENT_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                reference: context_reference(context)?,
                fields: expected_fields,
                states: states
                    .iter()
                    .map(WireStateReferenceV2::from_state)
                    .collect::<Result<Vec<_>, Diagnostic>>()?,
            },
        };
        value.validate_local(TrajectoryDecoderLimits::default())?;
        Ok(value)
    }

    /// Decode a closed segment without resolving its state dependencies.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed, oversized, unknown, or noncanonical data.
    pub fn from_json(bytes: &[u8], limits: TrajectoryDecoderLimits) -> Result<Self, Diagnostic> {
        check_json_limits(bytes, limits.json)?;
        let wire = serde_json::from_slice(bytes).map_err(|error| {
            invalid_artifact(format!(
                "invalid spatial-trajectory-segment/v2 JSON: {error}"
            ))
        })?;
        let value = Self { wire };
        value.validate_local(limits)?;
        Ok(value)
    }

    /// Deterministic canonical JSON bytes.
    ///
    /// # Errors
    /// Returns `EQ0901` if serialization unexpectedly fails.
    pub fn canonical_json(&self) -> Result<Vec<u8>, Diagnostic> {
        serde_json::to_vec(&self.wire).map_err(|error| {
            invalid_artifact(format!(
                "cannot serialize spatial trajectory segment v2: {error}"
            ))
        })
    }

    /// Domain-separated segment identity.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            SEGMENT_SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
    }

    /// First accepted step.
    #[must_use]
    pub fn first_step(&self) -> u64 {
        self.wire.states[0].step
    }

    /// Last accepted step.
    #[must_use]
    pub fn last_step(&self) -> u64 {
        self.wire.states[self.wire.states.len() - 1].step
    }

    /// First accepted coherent-SI time.
    #[must_use]
    pub fn first_time_s(&self) -> f64 {
        self.wire.states[0].time_s
    }

    /// Last accepted coherent-SI time.
    #[must_use]
    pub fn last_time_s(&self) -> f64 {
        self.wire.states[self.wire.states.len() - 1].time_s
    }

    /// GeometryState immediately preceding the first state, when any.
    #[must_use]
    pub fn first_geometry_predecessor(&self) -> Option<ArtifactDigest> {
        self.wire.states[0]
            .predecessor_geometry_state_sha256
            .clone()
            .map(ArtifactDigest)
    }

    /// First current GeometryState artifact.
    #[must_use]
    pub fn first_geometry_state(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.states[0].geometry_state_sha256.clone())
    }

    /// Last current GeometryState artifact.
    #[must_use]
    pub fn last_geometry_state(&self) -> ArtifactDigest {
        ArtifactDigest(
            self.wire.states[self.wire.states.len() - 1]
                .geometry_state_sha256
                .clone(),
        )
    }

    /// Number of accepted states.
    #[must_use]
    pub fn state_count(&self) -> usize {
        self.wire.states.len()
    }

    /// Ordered exact moving-state artifact references.
    #[must_use]
    pub fn state_artifacts(&self) -> Vec<ArtifactDigest> {
        self.wire
            .states
            .iter()
            .map(|state| ArtifactDigest(state.state_sha256.clone()))
            .collect()
    }

    /// Ordered `(step, time, state, GeometryState)` index.
    #[must_use]
    pub fn states(&self) -> Vec<(u64, f64, ArtifactDigest, ArtifactDigest)> {
        self.wire
            .states
            .iter()
            .map(|state| {
                (
                    state.step,
                    state.time_s,
                    ArtifactDigest(state.state_sha256.clone()),
                    ArtifactDigest(state.geometry_state_sha256.clone()),
                )
            })
            .collect()
    }

    /// Exact selected Semantic Field identities.
    #[must_use]
    pub fn fields(&self) -> Vec<Id<kinds::Field>> {
        field_ids(&self.wire.fields)
    }

    /// Exact reference Model artifact.
    #[must_use]
    pub fn model_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.reference.model_sha256.clone())
    }

    /// Reference Model semantic revision.
    #[must_use]
    pub const fn semantic_revision(&self) -> u64 {
        self.wire.reference.semantic_revision
    }

    /// Exact ALE Realization artifact.
    #[must_use]
    pub fn realization_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.reference.realization_sha256.clone())
    }

    /// Exact reference Geometry Identity artifact.
    #[must_use]
    pub fn reference_geometry_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.reference.reference_geometry_sha256.clone())
    }

    /// Exact correspondence artifact.
    #[must_use]
    pub fn correspondence_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.reference.correspondence_sha256.clone())
    }

    /// Exact immutable reference mesh artifact.
    #[must_use]
    pub fn reference_mesh_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.reference.reference_mesh_sha256.clone())
    }

    /// Rebuild and compare this segment from exact state objects.
    ///
    /// # Errors
    /// Returns `EQ0901` for missing, reordered, substituted, or cross-wired
    /// state content.
    pub fn validate_against<M: ReplayableCanonicalModelArtifact>(
        &self,
        context: &ValidatedMovingSpatialContextV2<'_, M>,
        states: &[SpatialStateEnvelopeV2],
    ) -> Result<(), Diagnostic> {
        let expected = Self::new(context, states)?;
        if self != &expected {
            return Err(invalid_artifact(
                "moving trajectory segment differs from exact state replay",
            ));
        }
        Ok(())
    }

    fn validate_context<M: ReplayableCanonicalModelArtifact>(
        &self,
        context: &ValidatedMovingSpatialContextV2<'_, M>,
    ) -> Result<(), Diagnostic> {
        if self.wire.reference != context_reference(context)?
            || self.wire.fields != context_field_inventory(context)
        {
            return Err(invalid_artifact(
                "moving trajectory segment differs from its exact common context",
            ));
        }
        let duration = context.duration();
        for state in &self.wire.states {
            if state.time_s != normalize_zero((state.step as f64) * duration) {
                return Err(invalid_artifact(
                    "moving trajectory state time differs from the ALE fixed-step duration",
                ));
            }
        }
        Ok(())
    }

    fn validate_local(&self, limits: TrajectoryDecoderLimits) -> Result<(), Diagnostic> {
        if self.wire.schema != SEGMENT_SCHEMA || self.wire.encoding != CANONICAL_ENCODING {
            return Err(invalid_artifact(
                "unsupported spatial-trajectory-segment/v2 schema or encoding",
            ));
        }
        validate_reference(&self.wire.reference)?;
        validate_field_inventory(&self.wire.fields, limits.max_spatial_state_fields)?;
        if self.wire.states.is_empty()
            || self.wire.states.len() > limits.max_trajectory_segment_states
        {
            return Err(invalid_artifact(
                "moving trajectory segment state count is empty or exceeds the decoder limit",
            ));
        }
        for state in &self.wire.states {
            state.validate()?;
        }
        if self
            .wire
            .states
            .iter()
            .map(|state| state.state_sha256.as_str())
            .collect::<BTreeSet<_>>()
            .len()
            != self.wire.states.len()
        {
            return Err(invalid_artifact(
                "moving trajectory segment contains a duplicate state artifact",
            ));
        }
        for pair in self.wire.states.windows(2) {
            pair[0].require_adjacent(&pair[1])?;
        }
        Ok(())
    }
}

/// Immutable root over a complete prefix of moving-state segments.
///
/// Extending a trajectory creates a new root retaining the exact previous
/// segment prefix. The root contains no Run reference, so a Run may name it as
/// output without introducing a digest cycle.
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialTrajectoryEnvelopeV2 {
    wire: WireSpatialTrajectoryV2,
}

impl SpatialTrajectoryEnvelopeV2 {
    /// Publish the first root of a complete trajectory beginning at state zero.
    ///
    /// # Errors
    /// Returns `EQ0901` unless the first segment starts at `(0, 0)` with no
    /// GeometryState predecessor and matches the exact common context.
    pub fn start<M: ReplayableCanonicalModelArtifact>(
        context: &ValidatedMovingSpatialContextV2<'_, M>,
        segment: &SpatialTrajectorySegmentEnvelopeV2,
    ) -> Result<Self, Diagnostic> {
        segment.validate_context(context)?;
        if segment.first_step() != 0
            || segment.first_time_s() != 0.0
            || segment.first_geometry_predecessor().is_some()
        {
            return Err(invalid_artifact(
                "moving trajectory root must begin at initial state and GeometryState",
            ));
        }
        let value = Self {
            wire: WireSpatialTrajectoryV2 {
                schema: TRAJECTORY_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                generation: 0,
                previous_root_sha256: None,
                reference: segment.wire.reference.clone(),
                fields: segment.wire.fields.clone(),
                segments: vec![WireSegmentReferenceV2::from_segment(segment)?],
            },
        };
        value.validate_local(TrajectoryDecoderLimits::default())?;
        Ok(value)
    }

    /// Publish a new root retaining the complete previous segment prefix.
    ///
    /// # Errors
    /// Returns `EQ0901` for resource/Field drift, a broken immutable prefix,
    /// a skipped state, or a GeometryState predecessor mismatch.
    pub fn extend<M: ReplayableCanonicalModelArtifact>(
        context: &ValidatedMovingSpatialContextV2<'_, M>,
        previous: &Self,
        segment: &SpatialTrajectorySegmentEnvelopeV2,
    ) -> Result<Self, Diagnostic> {
        previous.validate_context(context)?;
        segment.validate_context(context)?;
        if segment.wire.reference != previous.wire.reference
            || segment.wire.fields != previous.wire.fields
        {
            return Err(invalid_artifact(
                "moving trajectory extension changes common resources or Field inventory",
            ));
        }
        let prior = previous
            .wire
            .segments
            .last()
            .expect("validated moving trajectory owns a segment");
        let next_step = prior
            .last_step
            .checked_add(1)
            .ok_or_else(|| invalid_artifact("moving trajectory step overflows u64"))?;
        if segment.first_step() != next_step
            || segment.first_time_s() <= prior.last_time_s
            || segment.first_geometry_predecessor() != Some(previous.last_geometry_state())
        {
            return Err(invalid_artifact(
                "moving trajectory extension breaks state or GeometryState continuity",
            ));
        }
        let mut segments = previous.wire.segments.clone();
        segments.push(WireSegmentReferenceV2::from_segment(segment)?);
        let value = Self {
            wire: WireSpatialTrajectoryV2 {
                schema: TRAJECTORY_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                generation: previous.wire.generation.checked_add(1).ok_or_else(|| {
                    invalid_artifact("moving trajectory generation overflows u64")
                })?,
                previous_root_sha256: Some(previous.digest()?.to_string()),
                reference: previous.wire.reference.clone(),
                fields: previous.wire.fields.clone(),
                segments,
            },
        };
        value.validate_local(TrajectoryDecoderLimits::default())?;
        Ok(value)
    }

    /// Decode a closed moving-trajectory root without resolving dependencies.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed, oversized, unknown, or noncanonical data.
    pub fn from_json(bytes: &[u8], limits: TrajectoryDecoderLimits) -> Result<Self, Diagnostic> {
        check_json_limits(bytes, limits.json)?;
        let wire = serde_json::from_slice(bytes).map_err(|error| {
            invalid_artifact(format!("invalid spatial-trajectory/v2 JSON: {error}"))
        })?;
        let value = Self { wire };
        value.validate_local(limits)?;
        Ok(value)
    }

    /// Deterministic canonical JSON bytes.
    ///
    /// # Errors
    /// Returns `EQ0901` if serialization unexpectedly fails.
    pub fn canonical_json(&self) -> Result<Vec<u8>, Diagnostic> {
        serde_json::to_vec(&self.wire).map_err(|error| {
            invalid_artifact(format!("cannot serialize spatial trajectory v2: {error}"))
        })
    }

    /// Domain-separated immutable-root identity.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            TRAJECTORY_SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
    }

    /// Root generation, starting at zero.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.wire.generation
    }

    /// Exact previous immutable root, when this is an extension.
    #[must_use]
    pub fn previous_root(&self) -> Option<ArtifactDigest> {
        self.wire.previous_root_sha256.clone().map(ArtifactDigest)
    }

    /// Ordered complete segment prefix.
    #[must_use]
    pub fn segment_artifacts(&self) -> Vec<ArtifactDigest> {
        self.wire
            .segments
            .iter()
            .map(|segment| ArtifactDigest(segment.segment_sha256.clone()))
            .collect()
    }

    /// First accepted step represented by this root.
    #[must_use]
    pub fn first_step(&self) -> u64 {
        self.wire.segments[0].first_step
    }

    /// Last accepted step represented by this root.
    #[must_use]
    pub fn last_step(&self) -> u64 {
        self.wire.segments[self.wire.segments.len() - 1].last_step
    }

    /// Last exact current GeometryState represented by this root.
    #[must_use]
    pub fn last_geometry_state(&self) -> ArtifactDigest {
        ArtifactDigest(
            self.wire.segments[self.wire.segments.len() - 1]
                .last_geometry_state_sha256
                .clone(),
        )
    }

    /// Exact selected Semantic Field identities.
    #[must_use]
    pub fn fields(&self) -> Vec<Id<kinds::Field>> {
        field_ids(&self.wire.fields)
    }

    /// Exact reference Model artifact.
    #[must_use]
    pub fn model_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.reference.model_sha256.clone())
    }

    /// Reference Model semantic revision.
    #[must_use]
    pub const fn semantic_revision(&self) -> u64 {
        self.wire.reference.semantic_revision
    }

    /// Exact ALE Realization artifact.
    #[must_use]
    pub fn realization_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.reference.realization_sha256.clone())
    }

    /// Exact reference Geometry Identity artifact.
    #[must_use]
    pub fn reference_geometry_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.reference.reference_geometry_sha256.clone())
    }

    /// Exact correspondence artifact.
    #[must_use]
    pub fn correspondence_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.reference.correspondence_sha256.clone())
    }

    /// Exact immutable reference mesh artifact.
    #[must_use]
    pub fn reference_mesh_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.reference.reference_mesh_sha256.clone())
    }

    /// Rebuild the complete immutable prefix and compare its prior-root edge.
    ///
    /// Segment input order is exact and is never normalized.
    ///
    /// # Errors
    /// Returns `EQ0901` for a missing, reordered, substituted, cross-wired, or
    /// non-prefix segment, or for an incorrect prior root.
    pub fn validate_against<M: ReplayableCanonicalModelArtifact>(
        &self,
        context: &ValidatedMovingSpatialContextV2<'_, M>,
        previous: Option<&Self>,
        segments: &[SpatialTrajectorySegmentEnvelopeV2],
    ) -> Result<(), Diagnostic> {
        let first = segments
            .first()
            .ok_or_else(|| invalid_artifact("moving trajectory replay requires one segment"))?;
        let mut replay = Self::start(context, first)?;
        for segment in &segments[1..] {
            replay = Self::extend(context, &replay, segment)?;
        }
        if self != &replay {
            return Err(invalid_artifact(
                "moving trajectory root differs from exact segment-prefix replay",
            ));
        }
        match (previous, self.previous_root()) {
            (None, None) => Ok(()),
            (Some(previous), Some(digest))
                if previous.digest()? == digest
                    && previous.wire.segments.len() < self.wire.segments.len()
                    && previous.wire.segments
                        == self.wire.segments[..previous.wire.segments.len()] =>
            {
                Ok(())
            }
            _ => Err(invalid_artifact(
                "moving trajectory previous root or immutable segment prefix differs",
            )),
        }
    }

    /// Validate exact segment objects without requiring the prior root object.
    ///
    /// # Errors
    /// Returns `EQ0901` for missing, reordered, substituted, or cross-lineage
    /// segment dependencies.
    pub fn validate_segments<M: ReplayableCanonicalModelArtifact>(
        &self,
        context: &ValidatedMovingSpatialContextV2<'_, M>,
        segments: &[SpatialTrajectorySegmentEnvelopeV2],
    ) -> Result<(), Diagnostic> {
        if segments.len() != self.wire.segments.len() {
            return Err(invalid_artifact(
                "moving trajectory root is missing one or more segments",
            ));
        }
        for (expected, segment) in self.wire.segments.iter().zip(segments) {
            segment.validate_context(context)?;
            if expected != &WireSegmentReferenceV2::from_segment(segment)?
                || segment.wire.reference != self.wire.reference
                || segment.wire.fields != self.wire.fields
            {
                return Err(invalid_artifact(
                    "moving trajectory segment identity, range, resources, or Field inventory differs",
                ));
            }
        }
        for pair in segments.windows(2) {
            let next_step = pair[0]
                .last_step()
                .checked_add(1)
                .ok_or_else(|| invalid_artifact("moving trajectory step overflows u64"))?;
            if pair[1].first_step() != next_step
                || pair[1].first_geometry_predecessor() != Some(pair[0].last_geometry_state())
            {
                return Err(invalid_artifact(
                    "moving trajectory segments break exact GeometryState continuity",
                ));
            }
        }
        Ok(())
    }

    fn validate_context<M: ReplayableCanonicalModelArtifact>(
        &self,
        context: &ValidatedMovingSpatialContextV2<'_, M>,
    ) -> Result<(), Diagnostic> {
        if self.wire.reference != context_reference(context)?
            || self.wire.fields != context_field_inventory(context)
        {
            return Err(invalid_artifact(
                "moving trajectory root differs from its exact common context",
            ));
        }
        Ok(())
    }

    fn validate_local(&self, limits: TrajectoryDecoderLimits) -> Result<(), Diagnostic> {
        if self.wire.schema != TRAJECTORY_SCHEMA || self.wire.encoding != CANONICAL_ENCODING {
            return Err(invalid_artifact(
                "unsupported spatial-trajectory/v2 schema or encoding",
            ));
        }
        validate_reference(&self.wire.reference)?;
        validate_field_inventory(&self.wire.fields, limits.max_spatial_state_fields)?;
        if let Some(previous) = &self.wire.previous_root_sha256 {
            ArtifactDigest::from_hex(previous.clone())?;
        }
        if (self.wire.generation == 0) != self.wire.previous_root_sha256.is_none() {
            return Err(invalid_artifact(
                "moving trajectory generation zero must be exactly the root without a predecessor",
            ));
        }
        if self.wire.segments.is_empty()
            || self.wire.segments.len() > limits.max_trajectory_segments
            || u64::try_from(self.wire.segments.len() - 1).ok() != Some(self.wire.generation)
        {
            return Err(invalid_artifact(
                "moving trajectory segment count is empty, excessive, or inconsistent with generation",
            ));
        }
        let mut total_states = 0_usize;
        for segment in &self.wire.segments {
            segment.validate(limits.max_trajectory_segment_states)?;
            total_states = total_states
                .checked_add(usize::try_from(segment.state_count).map_err(|_| {
                    invalid_artifact("moving trajectory segment state count exceeds local usize")
                })?)
                .ok_or_else(|| {
                    invalid_artifact("moving trajectory aggregate state count overflows")
                })?;
        }
        if total_states > limits.max_trajectory_states {
            return Err(invalid_artifact(
                "moving trajectory aggregate state count exceeds the decoder limit",
            ));
        }
        let first = &self.wire.segments[0];
        if first.first_step != 0
            || first.first_time_s != 0.0
            || first.first_geometry_predecessor_sha256.is_some()
        {
            return Err(invalid_artifact(
                "moving trajectory prefix must begin at initial state and GeometryState",
            ));
        }
        for pair in self.wire.segments.windows(2) {
            if pair[0]
                .last_step
                .checked_add(1)
                .is_none_or(|next| next != pair[1].first_step)
                || pair[0].last_time_s >= pair[1].first_time_s
                || pair[1].first_geometry_predecessor_sha256
                    != Some(pair[0].last_geometry_state_sha256.clone())
            {
                return Err(invalid_artifact(
                    "moving trajectory segment summaries break state or GeometryState continuity",
                ));
            }
        }
        Ok(())
    }
}

fn validate_state_context<M: ReplayableCanonicalModelArtifact>(
    context: &ValidatedMovingSpatialContextV2<'_, M>,
    state: &SpatialStateEnvelopeV2,
    fields: &[WireFieldIdentityV2],
) -> Result<(), Diagnostic> {
    if state.model_artifact() != *context.model_reference().artifact()
        || state.semantic_revision() != context.model_reference().semantic_revision().get()
        || state.realization_artifact() != context.realization_artifact()?
        || state.reference_geometry_artifact() != context.geometry().digest()?
        || state.correspondence_artifact() != context.correspondence().digest()?
        || state.reference_mesh_artifact() != context.mesh().digest()?
        || state_field_inventory(state) != fields
    {
        return Err(invalid_artifact(
            "moving trajectory state differs from exact common resources or Field inventory",
        ));
    }
    let duration = context.duration();
    if state.time_s() != normalize_zero((state.step() as f64) * duration) {
        return Err(invalid_artifact(
            "moving trajectory state time differs from exact fixed-step Realization",
        ));
    }
    Ok(())
}

fn require_adjacent_states(
    previous: &SpatialStateEnvelopeV2,
    next: &SpatialStateEnvelopeV2,
) -> Result<(), Diagnostic> {
    let expected_step = previous
        .step()
        .checked_add(1)
        .ok_or_else(|| invalid_artifact("moving trajectory step overflows u64"))?;
    if next.step() != expected_step
        || next.time_s() <= previous.time_s()
        || next.predecessor_geometry_state() != Some(previous.geometry_state_artifact())
    {
        return Err(invalid_artifact(
            "moving trajectory states must be consecutive in state and GeometryState lineage",
        ));
    }
    Ok(())
}

fn context_reference<M: ReplayableCanonicalModelArtifact>(
    context: &ValidatedMovingSpatialContextV2<'_, M>,
) -> Result<WireReferenceLineageV2, Diagnostic> {
    Ok(WireReferenceLineageV2 {
        model_sha256: context.model_reference().artifact().to_string(),
        semantic_revision: context.model_reference().semantic_revision().get(),
        realization_sha256: context.realization_artifact()?.to_string(),
        reference_geometry_sha256: context.geometry().digest()?.to_string(),
        correspondence_sha256: context.correspondence().digest()?.to_string(),
        reference_mesh_sha256: context.mesh().digest()?.to_string(),
    })
}

fn context_field_inventory<M: ReplayableCanonicalModelArtifact>(
    context: &ValidatedMovingSpatialContextV2<'_, M>,
) -> Vec<WireFieldIdentityV2> {
    context
        .represented_fields()
        .iter()
        .map(|entry| WireFieldIdentityV2 {
            support_domain_ulid: entry.domain().ulid().to_string(),
            field_ulid: entry.field().ulid().to_string(),
        })
        .collect()
}

fn state_field_inventory(state: &SpatialStateEnvelopeV2) -> Vec<WireFieldIdentityV2> {
    state
        .fields()
        .into_iter()
        .map(|(domain, field, _)| WireFieldIdentityV2 {
            support_domain_ulid: domain.ulid().to_string(),
            field_ulid: field.ulid().to_string(),
        })
        .collect()
}

fn field_ids(fields: &[WireFieldIdentityV2]) -> Vec<Id<kinds::Field>> {
    fields
        .iter()
        .map(|entry| {
            Id::from_ulid(
                Ulid::from_str(&entry.field_ulid).expect("validated moving trajectory Field ULID"),
            )
        })
        .collect()
}

fn validate_reference(reference: &WireReferenceLineageV2) -> Result<(), Diagnostic> {
    for digest in [
        &reference.model_sha256,
        &reference.realization_sha256,
        &reference.reference_geometry_sha256,
        &reference.correspondence_sha256,
        &reference.reference_mesh_sha256,
    ] {
        ArtifactDigest::from_hex(digest.clone())?;
    }
    Ok(())
}

fn validate_field_inventory(
    fields: &[WireFieldIdentityV2],
    limit: usize,
) -> Result<(), Diagnostic> {
    if fields.is_empty() || fields.len() > limit {
        return Err(invalid_artifact(
            "moving trajectory Field inventory is empty or exceeds the decoder limit",
        ));
    }
    let mut prior = None;
    for entry in fields {
        let domain = parse_ulid(&entry.support_domain_ulid, "support Domain")?;
        let field = parse_ulid(&entry.field_ulid, "Field")?;
        if prior.is_some_and(|prior| prior >= field) {
            return Err(invalid_artifact(
                "moving trajectory Fields must be unique and in canonical Field identity order",
            ));
        }
        let _ = domain;
        prior = Some(field);
    }
    Ok(())
}

fn validate_time(time_s: f64) -> Result<(), Diagnostic> {
    if !time_s.is_finite() || time_s < 0.0 || is_negative_zero(time_s) {
        Err(invalid_artifact(
            "moving trajectory accepted time must be finite, nonnegative, and canonical",
        ))
    } else {
        Ok(())
    }
}

fn parse_ulid(value: &str, label: &str) -> Result<Ulid, Diagnostic> {
    let parsed = Ulid::from_str(value)
        .map_err(|_| invalid_artifact(format!("moving trajectory {label} ULID is malformed")))?;
    if parsed.to_string() != value {
        return Err(invalid_artifact(format!(
            "moving trajectory {label} ULID is not in canonical spelling"
        )));
    }
    Ok(parsed)
}

const fn normalize_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

fn is_negative_zero(value: f64) -> bool {
    value == 0.0 && value.is_sign_negative()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireTrajectorySegmentV2 {
    schema: String,
    encoding: String,
    reference: WireReferenceLineageV2,
    fields: Vec<WireFieldIdentityV2>,
    states: Vec<WireStateReferenceV2>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireSpatialTrajectoryV2 {
    schema: String,
    encoding: String,
    generation: u64,
    previous_root_sha256: Option<String>,
    reference: WireReferenceLineageV2,
    fields: Vec<WireFieldIdentityV2>,
    segments: Vec<WireSegmentReferenceV2>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireReferenceLineageV2 {
    model_sha256: String,
    semantic_revision: u64,
    realization_sha256: String,
    reference_geometry_sha256: String,
    correspondence_sha256: String,
    reference_mesh_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireFieldIdentityV2 {
    support_domain_ulid: String,
    field_ulid: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireStateReferenceV2 {
    step: u64,
    time_s: f64,
    state_sha256: String,
    geometry_state_sha256: String,
    predecessor_geometry_state_sha256: Option<String>,
}

impl WireStateReferenceV2 {
    fn from_state(state: &SpatialStateEnvelopeV2) -> Result<Self, Diagnostic> {
        Ok(Self {
            step: state.step(),
            time_s: state.time_s(),
            state_sha256: state.digest()?.to_string(),
            geometry_state_sha256: state.geometry_state_artifact().to_string(),
            predecessor_geometry_state_sha256: state
                .predecessor_geometry_state()
                .map(|digest| digest.to_string()),
        })
    }

    fn validate(&self) -> Result<(), Diagnostic> {
        if self.step > MAX_EXACT_F64_INTEGER {
            return Err(invalid_artifact(
                "moving trajectory step cannot be represented exactly as binary64",
            ));
        }
        validate_time(self.time_s)?;
        ArtifactDigest::from_hex(self.state_sha256.clone())?;
        ArtifactDigest::from_hex(self.geometry_state_sha256.clone())?;
        if let Some(predecessor) = &self.predecessor_geometry_state_sha256 {
            ArtifactDigest::from_hex(predecessor.clone())?;
        }
        if self.predecessor_geometry_state_sha256.is_none()
            != (self.step == 0 && self.time_s == 0.0)
        {
            return Err(invalid_artifact(
                "moving trajectory state coordinate and GeometryState predecessor disagree",
            ));
        }
        Ok(())
    }

    fn require_adjacent(&self, next: &Self) -> Result<(), Diagnostic> {
        if self
            .step
            .checked_add(1)
            .is_none_or(|step| next.step != step)
            || next.time_s <= self.time_s
            || next.predecessor_geometry_state_sha256 != Some(self.geometry_state_sha256.clone())
        {
            return Err(invalid_artifact(
                "moving trajectory state references break exact GeometryState continuity",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireSegmentReferenceV2 {
    first_step: u64,
    last_step: u64,
    first_time_s: f64,
    last_time_s: f64,
    state_count: u64,
    first_geometry_predecessor_sha256: Option<String>,
    first_geometry_state_sha256: String,
    last_geometry_state_sha256: String,
    segment_sha256: String,
}

impl WireSegmentReferenceV2 {
    fn from_segment(segment: &SpatialTrajectorySegmentEnvelopeV2) -> Result<Self, Diagnostic> {
        Ok(Self {
            first_step: segment.first_step(),
            last_step: segment.last_step(),
            first_time_s: segment.first_time_s(),
            last_time_s: segment.last_time_s(),
            state_count: u64::try_from(segment.state_count())
                .map_err(|_| invalid_artifact("moving trajectory state count exceeds u64"))?,
            first_geometry_predecessor_sha256: segment
                .first_geometry_predecessor()
                .map(|digest| digest.to_string()),
            first_geometry_state_sha256: segment.first_geometry_state().to_string(),
            last_geometry_state_sha256: segment.last_geometry_state().to_string(),
            segment_sha256: segment.digest()?.to_string(),
        })
    }

    fn validate(&self, state_limit: usize) -> Result<(), Diagnostic> {
        ArtifactDigest::from_hex(self.first_geometry_state_sha256.clone())?;
        ArtifactDigest::from_hex(self.last_geometry_state_sha256.clone())?;
        ArtifactDigest::from_hex(self.segment_sha256.clone())?;
        if let Some(predecessor) = &self.first_geometry_predecessor_sha256 {
            ArtifactDigest::from_hex(predecessor.clone())?;
        }
        validate_time(self.first_time_s)?;
        validate_time(self.last_time_s)?;
        let state_count = usize::try_from(self.state_count)
            .map_err(|_| invalid_artifact("moving trajectory state count exceeds local usize"))?;
        if self.state_count == 0
            || state_count > state_limit
            || self.first_step > self.last_step
            || self.first_time_s > self.last_time_s
            || self
                .last_step
                .checked_sub(self.first_step)
                .and_then(|span| span.checked_add(1))
                != Some(self.state_count)
            || self.first_geometry_predecessor_sha256.is_none()
                != (self.first_step == 0 && self.first_time_s == 0.0)
        {
            return Err(invalid_artifact(
                "moving trajectory segment summary has an invalid contiguous range",
            ));
        }
        Ok(())
    }
}
