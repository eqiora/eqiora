//! One immutable trajectory across exact continuous and remeshing state edges.
mod wire;
use crate::spatial_state::{
    context::MovingSpatialReference, lineage::StateLineage, wire::WireOrigin,
};
use crate::{
    ArtifactDigest, CANONICAL_ENCODING, ReplayableCanonicalModelArtifact, SpatialStateEnvelope,
    SpatialStateOriginKind, TrajectoryDecoderLimits, ValidatedMovingSpatialContext,
    invalid_artifact,
};
use eqiora_core::{Diagnostic, Id, entity::kinds};
use std::collections::BTreeSet;
use wire::*;

const SEGMENT_SCHEMA: &str = "eqiora.spatial-trajectory-segment/v4";
const TRAJECTORY_SCHEMA: &str = "eqiora.spatial-trajectory/v4";

/// Nonempty consecutive states sharing one exact reference context and Field inventory.
/// A representation change starts a new segment; subsequent states continue that representation.
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialTrajectorySegmentEnvelope {
    wire: WireSegment,
}
impl SpatialTrajectorySegmentEnvelope {
    /// Capture consecutive states in one fully replayed reference context.
    ///
    /// # Errors
    /// Returns `EQ0901` for missing, reordered, cross-context or incomplete states.
    pub fn new<M: ReplayableCanonicalModelArtifact>(
        context: &ValidatedMovingSpatialContext<'_, M>,
        states: &[SpatialStateEnvelope],
    ) -> Result<Self, Diagnostic> {
        let first = states
            .first()
            .ok_or_else(|| invalid_artifact("trajectory segment requires at least one state"))?;
        let fields = FieldIdentity::from_state(first);
        let expected_fields = context
            .represented_fields()
            .iter()
            .map(|field| FieldIdentity {
                support_domain_ulid: field.domain().ulid().to_string(),
                field_ulid: field.field().ulid().to_string(),
            })
            .collect::<Vec<_>>();
        if fields != expected_fields {
            return Err(invalid_artifact("trajectory omits a represented Field"));
        }
        for state in states {
            state.require_context(context)?;
            if FieldIdentity::from_state(state) != fields
                || state.time_s() != (state.step() as f64) * context.duration()
            {
                return Err(invalid_artifact(
                    "trajectory state changes its Field inventory or fixed-step time",
                ));
            }
        }
        let value = Self {
            wire: WireSegment {
                schema: SEGMENT_SCHEMA.into(),
                encoding: CANONICAL_ENCODING.into(),
                spatial_dimension: first.spatial_dimension(),
                reference: MovingSpatialReference::from_context(context)?,
                fields,
                states: states
                    .iter()
                    .map(StateLineage::from_state)
                    .collect::<Result<_, _>>()?,
            },
        };
        value.validate_local(TrajectoryDecoderLimits::default())?;
        Ok(value)
    }
    /// Decode bounded local projections without resolving their dependencies.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed data, invalid state edges or exceeded budgets.
    pub fn from_json(bytes: &[u8], limits: TrajectoryDecoderLimits) -> Result<Self, Diagnostic> {
        limits.json.check(bytes)?;
        let value = Self {
            wire: serde_json::from_slice(bytes).map_err(|error| {
                invalid_artifact(format!("invalid trajectory segment: {error}"))
            })?,
        };
        value.validate_local(limits)?;
        Ok(value)
    }
    /// Deterministic canonical bytes.
    ///
    /// # Errors
    /// Returns `EQ0901` if serialization fails.
    pub fn canonical_json(&self) -> Result<Vec<u8>, Diagnostic> {
        serde_json::to_vec(&self.wire).map_err(|error| {
            invalid_artifact(format!("cannot serialize trajectory segment: {error}"))
        })
    }
    /// Domain-separated identity.
    ///
    /// # Errors
    /// Returns `EQ0901` if serialization fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            SEGMENT_SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
    }
    /// Number of states.
    #[must_use]
    pub fn state_count(&self) -> usize {
        self.wire.states.len()
    }
    /// First accepted step.
    #[must_use]
    pub fn first_step(&self) -> u64 {
        self.wire.states[0].step
    }
    /// Last accepted step.
    #[must_use]
    pub fn last_step(&self) -> u64 {
        self.wire.states[self.state_count() - 1].step
    }
    /// First accepted time in coherent SI.
    #[must_use]
    pub fn first_time_s(&self) -> f64 {
        self.wire.states[0].time_s
    }
    /// Last accepted time in coherent SI.
    #[must_use]
    pub fn last_time_s(&self) -> f64 {
        self.wire.states[self.state_count() - 1].time_s
    }
    /// Immediate origin of the first state.
    #[must_use]
    pub fn origin(&self) -> SpatialStateOriginKind {
        self.wire.states[0].origin.kind()
    }
    /// Exact ordered state identities.
    #[must_use]
    pub fn state_artifacts(&self) -> Vec<ArtifactDigest> {
        self.wire
            .states
            .iter()
            .map(|state| ArtifactDigest(state.state_sha256.clone()))
            .collect()
    }
    /// Ordered `(step, time, state, GeometryState)` projection.
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
    /// Replay exact state objects and their common reference context.
    ///
    /// # Errors
    /// Returns `EQ0901` for missing, substituted or reordered dependencies.
    pub fn validate_against<M: ReplayableCanonicalModelArtifact>(
        &self,
        context: &ValidatedMovingSpatialContext<'_, M>,
        states: &[SpatialStateEnvelope],
    ) -> Result<(), Diagnostic> {
        if self != &Self::new(context, states)? {
            return Err(invalid_artifact(
                "trajectory segment differs from exact state replay",
            ));
        }
        Ok(())
    }
    fn validate_local(&self, limits: TrajectoryDecoderLimits) -> Result<(), Diagnostic> {
        if self.wire.schema != SEGMENT_SCHEMA
            || self.wire.encoding != CANONICAL_ENCODING
            || self.wire.states.is_empty()
            || self.wire.states.len() > limits.max_trajectory_segment_states
            || !matches!(self.wire.spatial_dimension, 2 | 3)
        {
            return Err(invalid_artifact("unsupported or empty trajectory segment"));
        }
        FieldIdentity::validate_inventory(&self.wire.fields, limits.max_spatial_state_fields)?;
        SegmentReference::from_segment(self)?.validate(limits)?;
        let mut identities = BTreeSet::new();
        for state in &self.wire.states {
            state.validate(self.wire.spatial_dimension)?;
            if !identities.insert(&state.state_sha256) {
                return Err(invalid_artifact("trajectory segment repeats a state"));
            }
        }
        for pair in self.wire.states.windows(2) {
            pair[0].require_continuation(&pair[1])?;
        }
        Ok(())
    }
}

/// Complete immutable prefix of state segments, including representation changes.
/// Each append retains every prior segment and binds the exact previous root.
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialTrajectoryEnvelope {
    wire: WireTrajectory,
}
impl SpatialTrajectoryEnvelope {
    /// Start a full trajectory at its initial state.
    ///
    /// # Errors
    /// Returns `EQ0901` unless the segment begins at the initial `(0, 0)` state.
    pub fn start(segment: &SpatialTrajectorySegmentEnvelope) -> Result<Self, Diagnostic> {
        Self::finish(None, vec![SegmentReference::from_segment(segment)?])
    }
    /// Append one segment through an exact continuous or same-time remesh edge.
    ///
    /// # Errors
    /// Returns `EQ0901` for a broken state edge or changed semantic context.
    pub fn extend(
        previous: &Self,
        segment: &SpatialTrajectorySegmentEnvelope,
    ) -> Result<Self, Diagnostic> {
        let mut segments = previous.wire.segments.clone();
        segments.push(SegmentReference::from_segment(segment)?);
        Self::finish(Some(previous.digest()?), segments)
    }
    fn finish(
        previous: Option<ArtifactDigest>,
        segments: Vec<SegmentReference>,
    ) -> Result<Self, Diagnostic> {
        let generation = u64::try_from(segments.len().saturating_sub(1))
            .map_err(|_| invalid_artifact("trajectory generation overflows"))?;
        let value = Self {
            wire: WireTrajectory {
                schema: TRAJECTORY_SCHEMA.into(),
                encoding: CANONICAL_ENCODING.into(),
                generation,
                previous_root_sha256: previous.map(|digest| digest.to_string()),
                segments,
            },
        };
        value.validate_local(TrajectoryDecoderLimits::default())?;
        Ok(value)
    }
    /// Decode bounded local summaries without resolving dependencies.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed summaries, invalid edges or exceeded budgets.
    pub fn from_json(bytes: &[u8], limits: TrajectoryDecoderLimits) -> Result<Self, Diagnostic> {
        limits.json.check(bytes)?;
        let value = Self {
            wire: serde_json::from_slice(bytes)
                .map_err(|error| invalid_artifact(format!("invalid trajectory: {error}")))?,
        };
        value.validate_local(limits)?;
        Ok(value)
    }
    /// Deterministic canonical bytes.
    ///
    /// # Errors
    /// Returns `EQ0901` if serialization fails.
    pub fn canonical_json(&self) -> Result<Vec<u8>, Diagnostic> {
        serde_json::to_vec(&self.wire)
            .map_err(|error| invalid_artifact(format!("cannot serialize trajectory: {error}")))
    }
    /// Domain-separated identity.
    ///
    /// # Errors
    /// Returns `EQ0901` if serialization fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            TRAJECTORY_SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
    }
    /// Append generation, starting at zero and including remesh segments.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.wire.generation
    }
    /// Exact prior root.
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
    /// Exact semantic Model shared by all representations.
    #[must_use]
    pub fn model_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.segments[0].reference.model_sha256.clone())
    }
    /// Semantic revision shared by all representations.
    #[must_use]
    pub fn semantic_revision(&self) -> u64 {
        self.wire.segments[0].reference.semantic_revision
    }
    /// Exact represented semantic Fields.
    #[must_use]
    pub fn fields(&self) -> Vec<Id<kinds::Field>> {
        self.wire.segments[0]
            .fields
            .iter()
            .map(|field| {
                crate::spatial_state::wire::parse_id(&field.field_ulid, "Field")
                    .expect("validated trajectory Field")
            })
            .collect()
    }
    /// First accepted step.
    #[must_use]
    pub fn first_step(&self) -> u64 {
        self.wire.segments[0].first.step
    }
    /// Last accepted step.
    #[must_use]
    pub fn last_step(&self) -> u64 {
        self.wire.segments[self.wire.segments.len() - 1].last.step
    }
    /// Last exact GeometryState.
    #[must_use]
    pub fn last_geometry_state(&self) -> ArtifactDigest {
        ArtifactDigest(
            self.wire.segments[self.wire.segments.len() - 1]
                .last
                .geometry_state_sha256
                .clone(),
        )
    }
    /// Replay all immutable roots from the ordered complete segment prefix.
    ///
    /// # Errors
    /// Returns `EQ0901` for missing or substituted segments, broken edges or a false prior root.
    pub fn validate_segments(
        &self,
        segments: &[SpatialTrajectorySegmentEnvelope],
    ) -> Result<(), Diagnostic> {
        let first = segments
            .first()
            .ok_or_else(|| invalid_artifact("trajectory replay has no segments"))?;
        let mut replay = Self::start(first)?;
        for segment in &segments[1..] {
            replay = Self::extend(&replay, segment)?;
        }
        if self != &replay {
            return Err(invalid_artifact(
                "trajectory differs from complete immutable-prefix replay",
            ));
        }
        Ok(())
    }
    /// Replay the full prefix and resolve its exact immediate predecessor.
    ///
    /// # Errors
    /// Returns `EQ0901` for a substituted prior root or segment.
    pub fn validate_against(
        &self,
        previous: Option<&Self>,
        segments: &[SpatialTrajectorySegmentEnvelope],
    ) -> Result<(), Diagnostic> {
        self.validate_segments(segments)?;
        if previous.map(Self::digest).transpose()? != self.previous_root() {
            return Err(invalid_artifact("trajectory previous root differs"));
        }
        Ok(())
    }
    fn validate_local(&self, limits: TrajectoryDecoderLimits) -> Result<(), Diagnostic> {
        if self.wire.schema != TRAJECTORY_SCHEMA
            || self.wire.encoding != CANONICAL_ENCODING
            || self.wire.segments.is_empty()
            || self.wire.segments.len() > limits.max_trajectory_segments
            || u64::try_from(self.wire.segments.len() - 1).ok() != Some(self.generation())
            || (self.generation() == 0) != self.wire.previous_root_sha256.is_none()
        {
            return Err(invalid_artifact(
                "trajectory schema, generation or segment count is invalid",
            ));
        }
        if let Some(previous) = &self.wire.previous_root_sha256 {
            ArtifactDigest::from_hex(previous.clone())?;
        }
        let mut identities = BTreeSet::new();
        let mut total = 0_usize;
        for segment in &self.wire.segments {
            segment.validate(limits)?;
            if !identities.insert(&segment.segment_sha256) {
                return Err(invalid_artifact("trajectory repeats a segment"));
            }
            total = total
                .checked_add(segment.state_count)
                .ok_or_else(|| invalid_artifact("trajectory state count overflows"))?;
        }
        if total > limits.max_trajectory_states {
            return Err(invalid_artifact(
                "trajectory aggregate state count exceeds its bound",
            ));
        }
        if !matches!(self.wire.segments[0].first.origin, WireOrigin::Initial {}) {
            return Err(invalid_artifact(
                "trajectory must begin at its initial state",
            ));
        }
        for pair in self.wire.segments.windows(2) {
            pair[0].require_next(&pair[1])?;
        }
        Ok(())
    }
}
