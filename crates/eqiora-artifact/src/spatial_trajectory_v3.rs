//! Immutable trajectory append chains across one V2-to-V3 remesh seam.

mod wire;
use wire::*;

use std::collections::BTreeSet;

use eqiora_core::Diagnostic;

use crate::{
    ArtifactDigest, CANONICAL_ENCODING, ReplayableCanonicalModelArtifact, SpatialStateEnvelope,
    SpatialStateOriginKind, SpatialTrajectoryEnvelopeV2, SpatialTrajectorySegmentEnvelopeV2,
    TrajectoryDecoderLimits, ValidatedMovingSpatialContext, check_json_limits, invalid_artifact,
};

const SEGMENT_SCHEMA: &str = "eqiora.spatial-trajectory-segment/v3";
const TRAJECTORY_SCHEMA: &str = "eqiora.spatial-trajectory/v3";

/// Closed origin of one target-side trajectory segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialTrajectorySegmentOriginKindV3 {
    /// First target segment paired to the finalized V2 source prefix.
    Remesh,
    /// Later target-only append after one exact V3 predecessor.
    Continuation,
}

/// Nonempty immutable target-state segment retaining the exact V2 source root.
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialTrajectorySegmentEnvelopeV3 {
    wire: WireTrajectorySegmentV3,
}

impl SpatialTrajectorySegmentEnvelopeV3 {
    /// Construct the first target segment at the same coordinate as the V2 tip.
    ///
    /// # Errors
    /// Returns `EQ0901` unless the source root is completely replayed, its last
    /// state is the exact transition source, and the first target state is the
    /// matching remesh-origin state at the same step/time.
    #[allow(clippy::too_many_arguments)]
    pub fn remesh<S: ReplayableCanonicalModelArtifact, T: ReplayableCanonicalModelArtifact>(
        source_context: &ValidatedMovingSpatialContext<'_, S>,
        source_prefix: &SpatialTrajectoryEnvelopeV2,
        source_segments: &[SpatialTrajectorySegmentEnvelopeV2],
        source_state: &SpatialStateEnvelope,
        target_context: &ValidatedMovingSpatialContext<'_, T>,
        target_states: &[SpatialStateEnvelope],
    ) -> Result<Self, Diagnostic> {
        source_prefix.validate_segments(source_context, source_segments)?;
        let source_tip = source_segments
            .last()
            .and_then(|segment| segment.state_artifacts().last().cloned())
            .ok_or_else(|| invalid_artifact("remesh trajectory source prefix is empty"))?;
        if source_tip != source_state.digest()? {
            return Err(invalid_artifact(
                "remesh trajectory source state is not the exact V2 prefix tip",
            ));
        }
        let first = target_states
            .first()
            .ok_or_else(|| invalid_artifact("remesh trajectory segment is empty"))?;
        if first.origin() != SpatialStateOriginKind::Remesh
            || first.predecessor().is_some()
            || first
                .remesh_anchor()
                .map(crate::RemeshStateAnchor::source_spatial_state)
                != Some(source_tip)
            || first.step() != source_state.step()
            || first.time_s() != source_state.time_s()
        {
            return Err(invalid_artifact(
                "first V3 state is not the same-coordinate representation of the V2 tip",
            ));
        }
        validate_target_states(target_context, target_states)?;
        Self::finish(
            source_prefix,
            source_state,
            target_states,
            WireSegmentOriginV3::Remesh {
                source_step: source_state.step(),
                source_time_s: source_state.time_s(),
            },
        )
    }

    /// Construct a later target-only segment after one exact predecessor.
    ///
    /// # Errors
    /// Returns `EQ0901` for a broken state edge, changed transition anchor, or
    /// target-context drift.
    pub fn continuation<M: ReplayableCanonicalModelArtifact>(
        context: &ValidatedMovingSpatialContext<'_, M>,
        source_prefix: &SpatialTrajectoryEnvelopeV2,
        source_state: &SpatialStateEnvelope,
        predecessor: &SpatialStateEnvelope,
        states: &[SpatialStateEnvelope],
    ) -> Result<Self, Diagnostic> {
        let first = states
            .first()
            .ok_or_else(|| invalid_artifact("V3 trajectory continuation is empty"))?;
        predecessor.require_adjacent(first)?;
        validate_target_states(context, states)?;
        Self::finish(
            source_prefix,
            source_state,
            states,
            WireSegmentOriginV3::Continuation {
                predecessor_spatial_state_v3_sha256: predecessor.digest()?.to_string(),
            },
        )
    }

    fn finish(
        source_prefix: &SpatialTrajectoryEnvelopeV2,
        source_state: &SpatialStateEnvelope,
        target_states: &[SpatialStateEnvelope],
        origin: WireSegmentOriginV3,
    ) -> Result<Self, Diagnostic> {
        let first = &target_states[0];
        let anchor = first.remesh_anchor().ok_or_else(|| {
            invalid_artifact("remesh trajectory requires its complete transition anchor")
        })?;
        if target_states
            .iter()
            .any(|state| state.remesh_anchor() != Some(anchor))
        {
            return Err(invalid_artifact(
                "remesh trajectory changes its immutable transition anchor",
            ));
        }
        let value = Self {
            wire: WireTrajectorySegmentV3 {
                schema: SEGMENT_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                source_prefix_v2_sha256: source_prefix.digest()?.to_string(),
                source_state_v2_sha256: source_state.digest()?.to_string(),
                target_reference: WireTargetReferenceV3::from_state(first),
                overlap_sha256: anchor.overlap_artifact().to_string(),
                transfer_receipt_sha256: anchor.transfer_receipt_artifact().to_string(),
                origin,
                states: target_states
                    .iter()
                    .map(WireTargetStateReferenceV3::from_state)
                    .collect::<Result<_, _>>()?,
            },
        };
        value.validate_local(TrajectoryDecoderLimits::default())?;
        Ok(value)
    }

    /// Decode bounded segment data without resolving state dependencies.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed, oversized, unknown, or noncanonical data.
    pub fn from_json(bytes: &[u8], limits: TrajectoryDecoderLimits) -> Result<Self, Diagnostic> {
        check_json_limits(bytes, limits.json)?;
        let wire = serde_json::from_slice(bytes).map_err(|error| {
            invalid_artifact(format!(
                "invalid spatial-trajectory-segment/v3 JSON: {error}"
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
                "cannot serialize spatial trajectory segment v3: {error}"
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

    /// Closed segment origin.
    #[must_use]
    pub const fn origin(&self) -> SpatialTrajectorySegmentOriginKindV3 {
        match self.wire.origin {
            WireSegmentOriginV3::Remesh { .. } => SpatialTrajectorySegmentOriginKindV3::Remesh,
            WireSegmentOriginV3::Continuation { .. } => {
                SpatialTrajectorySegmentOriginKindV3::Continuation
            }
        }
    }

    /// Exact immutable V2 source prefix.
    #[must_use]
    pub fn source_prefix(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.source_prefix_v2_sha256.clone())
    }

    /// Exact finalized V2 source state.
    #[must_use]
    pub fn source_state(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.source_state_v2_sha256.clone())
    }

    /// First target step.
    #[must_use]
    pub fn first_step(&self) -> u64 {
        self.wire.states[0].step
    }

    /// Last target step.
    #[must_use]
    pub fn last_step(&self) -> u64 {
        self.wire.states[self.wire.states.len() - 1].step
    }

    /// First target time.
    #[must_use]
    pub fn first_time_s(&self) -> f64 {
        self.wire.states[0].time_s
    }

    /// Last target time.
    #[must_use]
    pub fn last_time_s(&self) -> f64 {
        self.wire.states[self.wire.states.len() - 1].time_s
    }

    /// Last target state.
    #[must_use]
    pub fn last_state(&self) -> ArtifactDigest {
        ArtifactDigest(
            self.wire.states[self.wire.states.len() - 1]
                .state_sha256
                .clone(),
        )
    }

    /// Ordered exact target states.
    #[must_use]
    pub fn state_artifacts(&self) -> Vec<ArtifactDigest> {
        self.wire
            .states
            .iter()
            .map(|state| ArtifactDigest(state.state_sha256.clone()))
            .collect()
    }

    /// Replay exact target state objects.
    ///
    /// # Errors
    /// Returns `EQ0901` for reordered, substituted, or cross-context states.
    pub fn validate_states<M: ReplayableCanonicalModelArtifact>(
        &self,
        context: &ValidatedMovingSpatialContext<'_, M>,
        states: &[SpatialStateEnvelope],
    ) -> Result<(), Diagnostic> {
        validate_target_states(context, states)?;
        let references = states
            .iter()
            .map(WireTargetStateReferenceV3::from_state)
            .collect::<Result<Vec<_>, _>>()?;
        if self.wire.target_reference != WireTargetReferenceV3::from_state(&states[0])
            || self.wire.states != references
        {
            return Err(invalid_artifact(
                "V3 trajectory segment differs from exact state replay",
            ));
        }
        Ok(())
    }

    fn validate_local(&self, limits: TrajectoryDecoderLimits) -> Result<(), Diagnostic> {
        if self.wire.schema != SEGMENT_SCHEMA || self.wire.encoding != CANONICAL_ENCODING {
            return Err(invalid_artifact(
                "unsupported spatial-trajectory-segment/v3 schema or encoding",
            ));
        }
        for digest in [
            &self.wire.source_prefix_v2_sha256,
            &self.wire.source_state_v2_sha256,
            &self.wire.overlap_sha256,
            &self.wire.transfer_receipt_sha256,
        ] {
            ArtifactDigest::from_hex(digest.clone())?;
        }
        self.wire.target_reference.validate()?;
        self.wire.origin.validate()?;
        if self.wire.states.is_empty()
            || self.wire.states.len() > limits.max_remesh_trajectory_states
        {
            return Err(invalid_artifact(
                "V3 trajectory segment is empty or exceeds its state budget",
            ));
        }
        for state in &self.wire.states {
            state.validate()?;
            if state.overlap_sha256 != self.wire.overlap_sha256
                || state.transfer_receipt_sha256 != self.wire.transfer_receipt_sha256
            {
                return Err(invalid_artifact(
                    "V3 trajectory state changes the segment transition anchor",
                ));
            }
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
                "V3 trajectory segment contains a duplicate state",
            ));
        }
        for pair in self.wire.states.windows(2) {
            pair[0].require_adjacent(&pair[1])?;
        }
        match self.wire.origin {
            WireSegmentOriginV3::Remesh {
                source_step,
                source_time_s,
            } if self.wire.states[0].origin == WireStateOriginV3::Remesh
                && self.wire.states[0].predecessor_sha256.is_none()
                && self.wire.states[0].step == source_step
                && self.wire.states[0].time_s == source_time_s => {}
            WireSegmentOriginV3::Continuation {
                ref predecessor_spatial_state_v3_sha256,
            } if self.wire.states[0].origin == WireStateOriginV3::Continuous
                && self.wire.states[0].predecessor_sha256.as_ref()
                    == Some(predecessor_spatial_state_v3_sha256) => {}
            _ => {
                return Err(invalid_artifact(
                    "V3 trajectory segment origin does not close its first state edge",
                ));
            }
        }
        Ok(())
    }
}

/// Immutable V3 root retaining the exact V2 prefix and complete V3 segment prefix.
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialTrajectoryEnvelopeV3 {
    wire: WireSpatialTrajectoryV3,
}

impl SpatialTrajectoryEnvelopeV3 {
    /// Publish the first root containing the remesh segment.
    ///
    /// # Errors
    /// Returns `EQ0901` unless the segment retains the exact source root and
    /// begins with a remesh-origin pair.
    pub fn start(
        source_prefix: &SpatialTrajectoryEnvelopeV2,
        segment: &SpatialTrajectorySegmentEnvelopeV3,
    ) -> Result<Self, Diagnostic> {
        if segment.origin() != SpatialTrajectorySegmentOriginKindV3::Remesh
            || segment.source_prefix() != source_prefix.digest()?
        {
            return Err(invalid_artifact(
                "V3 trajectory root must start from the exact V2 prefix remesh segment",
            ));
        }
        let value = Self::finish(0, None, source_prefix, vec![segment_reference(segment)?])?;
        Ok(value)
    }

    /// Publish a new immutable root by appending one continuation segment.
    ///
    /// # Errors
    /// Returns `EQ0901` for a changed V2 prefix, broken V3 state edge, or
    /// repeated/substituted segment.
    pub fn extend(
        source_prefix: &SpatialTrajectoryEnvelopeV2,
        previous: &Self,
        segment: &SpatialTrajectorySegmentEnvelopeV3,
    ) -> Result<Self, Diagnostic> {
        if previous.source_prefix() != source_prefix.digest()?
            || segment.source_prefix() != previous.source_prefix()
            || segment.source_state() != previous.source_state()
            || segment.origin() != SpatialTrajectorySegmentOriginKindV3::Continuation
            || segment.wire.origin.predecessor()
                != Some(
                    previous
                        .wire
                        .segments
                        .last()
                        .expect("validated root")
                        .last_state_sha256
                        .as_str(),
                )
        {
            return Err(invalid_artifact(
                "V3 trajectory extension breaks its immutable source or target prefix",
            ));
        }
        let mut segments = previous.wire.segments.clone();
        segments.push(segment_reference(segment)?);
        Self::finish(
            previous
                .wire
                .generation
                .checked_add(1)
                .ok_or_else(|| invalid_artifact("V3 trajectory generation overflows"))?,
            Some(previous.digest()?.to_string()),
            source_prefix,
            segments,
        )
    }

    fn finish(
        generation: u64,
        previous_root_sha256: Option<String>,
        source_prefix: &SpatialTrajectoryEnvelopeV2,
        segments: Vec<WireSegmentReferenceV3>,
    ) -> Result<Self, Diagnostic> {
        let first = &segments[0];
        let value = Self {
            wire: WireSpatialTrajectoryV3 {
                schema: TRAJECTORY_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                generation,
                previous_root_sha256,
                source_prefix_v2_sha256: source_prefix.digest()?.to_string(),
                source_state_v2_sha256: first.source_state_v2_sha256.clone(),
                target_reference: first.target_reference.clone(),
                overlap_sha256: first.overlap_sha256.clone(),
                transfer_receipt_sha256: first.transfer_receipt_sha256.clone(),
                segments,
            },
        };
        value.validate_local(TrajectoryDecoderLimits::default())?;
        Ok(value)
    }

    /// Decode a bounded immutable root.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed, oversized, unknown, or noncanonical data.
    pub fn from_json(bytes: &[u8], limits: TrajectoryDecoderLimits) -> Result<Self, Diagnostic> {
        check_json_limits(bytes, limits.json)?;
        let wire = serde_json::from_slice(bytes).map_err(|error| {
            invalid_artifact(format!("invalid spatial-trajectory/v3 JSON: {error}"))
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
            invalid_artifact(format!("cannot serialize spatial trajectory v3: {error}"))
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

    /// Root generation.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.wire.generation
    }

    /// Exact previous V3 root.
    #[must_use]
    pub fn previous_root(&self) -> Option<ArtifactDigest> {
        self.wire.previous_root_sha256.clone().map(ArtifactDigest)
    }

    /// Exact immutable V2 source prefix.
    #[must_use]
    pub fn source_prefix(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.source_prefix_v2_sha256.clone())
    }

    /// Exact finalized V2 source state.
    #[must_use]
    pub fn source_state(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.source_state_v2_sha256.clone())
    }

    /// Complete ordered V3 segment prefix.
    #[must_use]
    pub fn segment_artifacts(&self) -> Vec<ArtifactDigest> {
        self.wire
            .segments
            .iter()
            .map(|segment| ArtifactDigest(segment.segment_sha256.clone()))
            .collect()
    }

    /// Replay the immutable V3 segment prefix and exact V2 source root.
    ///
    /// # Errors
    /// Returns `EQ0901` for a missing, reordered, substituted, or cyclic edge.
    pub fn validate_segments(
        &self,
        source_prefix: &SpatialTrajectoryEnvelopeV2,
        segments: &[SpatialTrajectorySegmentEnvelopeV3],
    ) -> Result<(), Diagnostic> {
        if self.source_prefix() != source_prefix.digest()?
            || segments.len() != self.wire.segments.len()
        {
            return Err(invalid_artifact(
                "V3 trajectory root differs from its exact source or segment count",
            ));
        }
        for (reference, segment) in self.wire.segments.iter().zip(segments) {
            if reference != &segment_reference(segment)? {
                return Err(invalid_artifact(
                    "V3 trajectory segment differs from its exact root reference",
                ));
            }
        }
        let mut replay = Self::start(source_prefix, &segments[0])?;
        for segment in &segments[1..] {
            replay = Self::extend(source_prefix, &replay, segment)?;
        }
        if self == &replay {
            Ok(())
        } else {
            Err(invalid_artifact(
                "V3 trajectory root differs from immutable-prefix replay",
            ))
        }
    }

    fn validate_local(&self, limits: TrajectoryDecoderLimits) -> Result<(), Diagnostic> {
        if self.wire.schema != TRAJECTORY_SCHEMA || self.wire.encoding != CANONICAL_ENCODING {
            return Err(invalid_artifact(
                "unsupported spatial-trajectory/v3 schema or encoding",
            ));
        }
        for digest in [
            &self.wire.source_prefix_v2_sha256,
            &self.wire.source_state_v2_sha256,
            &self.wire.overlap_sha256,
            &self.wire.transfer_receipt_sha256,
        ] {
            ArtifactDigest::from_hex(digest.clone())?;
        }
        self.wire.target_reference.validate()?;
        match (&self.wire.previous_root_sha256, self.wire.generation) {
            (None, 0) => {}
            (Some(digest), 1..) => {
                ArtifactDigest::from_hex(digest.clone())?;
            }
            _ => {
                return Err(invalid_artifact(
                    "V3 trajectory root generation and previous edge are inconsistent",
                ));
            }
        }
        if self.wire.segments.is_empty()
            || self.wire.segments.len() > limits.max_remesh_trajectory_segments
        {
            return Err(invalid_artifact(
                "V3 trajectory root is empty or exceeds its segment budget",
            ));
        }
        let mut total_states = 0_usize;
        let mut segment_ids = BTreeSet::new();
        for (index, segment) in self.wire.segments.iter().enumerate() {
            segment.validate()?;
            if !segment_ids.insert(&segment.segment_sha256)
                || segment.source_prefix_v2_sha256 != self.wire.source_prefix_v2_sha256
                || segment.source_state_v2_sha256 != self.wire.source_state_v2_sha256
                || segment.target_reference != self.wire.target_reference
                || segment.overlap_sha256 != self.wire.overlap_sha256
                || segment.transfer_receipt_sha256 != self.wire.transfer_receipt_sha256
                || (index == 0 && segment.origin != WireSegmentOriginTagV3::Remesh)
                || (index > 0 && segment.origin != WireSegmentOriginTagV3::Continuation)
            {
                return Err(invalid_artifact(
                    "V3 trajectory root contains a stale, duplicate, or misclassified segment",
                ));
            }
            total_states = total_states
                .checked_add(segment.state_count)
                .ok_or_else(|| invalid_artifact("V3 trajectory state count overflows"))?;
        }
        if total_states > limits.max_remesh_trajectory_states {
            return Err(invalid_artifact(
                "V3 trajectory root exceeds its aggregate state budget",
            ));
        }
        for pair in self.wire.segments.windows(2) {
            if pair[1].first_predecessor_sha256.as_ref() != Some(&pair[0].last_state_sha256)
                || pair[1].first_step
                    != pair[0]
                        .last_step
                        .checked_add(1)
                        .ok_or_else(|| invalid_artifact("V3 trajectory step overflows"))?
                || pair[1].first_time_s <= pair[0].last_time_s
            {
                return Err(invalid_artifact(
                    "V3 trajectory segments break the exact target append chain",
                ));
            }
        }
        Ok(())
    }
}

fn validate_target_states<M: ReplayableCanonicalModelArtifact>(
    context: &ValidatedMovingSpatialContext<'_, M>,
    states: &[SpatialStateEnvelope],
) -> Result<(), Diagnostic> {
    if states.is_empty() {
        return Err(invalid_artifact("V3 trajectory segment is empty"));
    }
    for state in states {
        state.require_context(context)?;
    }
    for pair in states.windows(2) {
        pair[0].require_adjacent(&pair[1])?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target_reference() -> WireTargetReferenceV3 {
        WireTargetReferenceV3 {
            model_sha256: "00".repeat(32),
            semantic_revision: 1,
            realization_sha256: "11".repeat(32),
            geometry_sha256: "22".repeat(32),
            correspondence_sha256: "33".repeat(32),
            mesh_sha256: "44".repeat(32),
        }
    }

    fn segment() -> SpatialTrajectorySegmentEnvelopeV3 {
        SpatialTrajectorySegmentEnvelopeV3 {
            wire: WireTrajectorySegmentV3 {
                schema: SEGMENT_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                source_prefix_v2_sha256: "55".repeat(32),
                source_state_v2_sha256: "66".repeat(32),
                target_reference: target_reference(),
                overlap_sha256: "77".repeat(32),
                transfer_receipt_sha256: "88".repeat(32),
                origin: WireSegmentOriginV3::Remesh {
                    source_step: 5,
                    source_time_s: 0.5,
                },
                states: vec![
                    WireTargetStateReferenceV3 {
                        state_sha256: "99".repeat(32),
                        origin: WireStateOriginV3::Remesh,
                        step: 5,
                        time_s: 0.5,
                        geometry_state_sha256: "aa".repeat(32),
                        predecessor_sha256: None,
                        source_state_v2_sha256: "66".repeat(32),
                        overlap_sha256: "77".repeat(32),
                        transfer_receipt_sha256: "88".repeat(32),
                    },
                    WireTargetStateReferenceV3 {
                        state_sha256: "bb".repeat(32),
                        origin: WireStateOriginV3::Continuous,
                        step: 6,
                        time_s: 0.6,
                        geometry_state_sha256: "cc".repeat(32),
                        predecessor_sha256: Some("99".repeat(32)),
                        source_state_v2_sha256: "66".repeat(32),
                        overlap_sha256: "77".repeat(32),
                        transfer_receipt_sha256: "88".repeat(32),
                    },
                ],
            },
        }
    }

    fn root(segment: &SpatialTrajectorySegmentEnvelopeV3) -> SpatialTrajectoryEnvelopeV3 {
        SpatialTrajectoryEnvelopeV3 {
            wire: WireSpatialTrajectoryV3 {
                schema: TRAJECTORY_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                generation: 0,
                previous_root_sha256: None,
                source_prefix_v2_sha256: "55".repeat(32),
                source_state_v2_sha256: "66".repeat(32),
                target_reference: target_reference(),
                overlap_sha256: "77".repeat(32),
                transfer_receipt_sha256: "88".repeat(32),
                segments: vec![segment_reference(segment).unwrap()],
            },
        }
    }

    #[test]
    fn trajectory_v3_wires_roundtrip_and_digests_are_frozen() {
        let segment = segment();
        segment
            .validate_local(TrajectoryDecoderLimits::default())
            .unwrap();
        let segment_bytes = segment.canonical_json().unwrap();
        assert_eq!(
            SpatialTrajectorySegmentEnvelopeV3::from_json(
                &segment_bytes,
                TrajectoryDecoderLimits::default(),
            )
            .unwrap(),
            segment
        );
        assert_eq!(
            segment.digest().unwrap().to_string(),
            "d3eecff525ef6734e4f23fba7bb85a16111d4ac4080494b1c497c7aaa75715a0"
        );

        let root = root(&segment);
        root.validate_local(TrajectoryDecoderLimits::default())
            .unwrap();
        let root_bytes = root.canonical_json().unwrap();
        assert_eq!(
            SpatialTrajectoryEnvelopeV3::from_json(&root_bytes, TrajectoryDecoderLimits::default())
                .unwrap(),
            root
        );
        assert_eq!(
            root.digest().unwrap().to_string(),
            "1d054468bc22731691d6449e7e2cb05837ba78351c840a5ba602210ee8bbedc0"
        );
    }

    #[test]
    fn trajectory_v3_rejects_broken_edges_and_resource_excess() {
        let value = segment();
        let mut broken = value.clone();
        broken.wire.states[1].predecessor_sha256 = Some("dd".repeat(32));
        assert!(
            broken
                .validate_local(TrajectoryDecoderLimits::default())
                .is_err()
        );

        let limits = TrajectoryDecoderLimits {
            max_remesh_trajectory_states: 1,
            ..TrajectoryDecoderLimits::default()
        };
        assert!(SpatialTrajectorySegmentEnvelopeV3::from_json(
            &value.canonical_json().unwrap(),
            limits,
        )
        .is_err());

        let root = root(&value);
        let root_bytes = root.canonical_json().unwrap();
        let segment_limits = TrajectoryDecoderLimits {
            max_remesh_trajectory_segments: 0,
            ..TrajectoryDecoderLimits::default()
        };
        assert!(SpatialTrajectoryEnvelopeV3::from_json(&root_bytes, segment_limits).is_err());

        let aggregate_state_limits = TrajectoryDecoderLimits {
            max_remesh_trajectory_states: 1,
            ..TrajectoryDecoderLimits::default()
        };
        assert!(
            SpatialTrajectoryEnvelopeV3::from_json(&root_bytes, aggregate_state_limits).is_err()
        );
    }
}
