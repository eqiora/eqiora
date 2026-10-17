//! Fail-closed selection retention across exact geometry revisions.

use std::collections::{BTreeMap, BTreeSet};
use std::str::FromStr;

use eqiora_core::Diagnostic;
use eqiora_core::Id;
use eqiora_core::entity::kinds;
use eqiora_geometry::{BodyAssociationCandidate, RetainedGeometryAssociation, RetentionRejection};
use eqiora_schema::kernel::BoundarySide;
use serde::{Deserialize, Serialize};
use ulid::Ulid;

use crate::{
    ArtifactDigest, CANONICAL_ENCODING, GeometryDecoderLimits, GeometryIdentityEnvelopeV1,
    GeometryMeshCorrespondenceEnvelopeV1, ReplayableCanonicalModelArtifact,
    SimplicialMeshEnvelopeV1, invalid_artifact,
};

const ASSOCIATION_SCHEMA: &str = "eqiora.geometry-revision-association-envelope/v1";

/// Content-bound total one-to-one selection association between two exact
/// geometry revisions.
///
/// Domain ULID equality is neither required nor used as retention evidence.
/// Missing, split, merged, and ambiguous candidate relations return their
/// typed [`RetentionRejection`] before an artifact can be constructed.
#[derive(Clone, Debug, PartialEq)]
pub struct GeometryRevisionAssociationEnvelopeV1 {
    wire: WireGeometryRevisionAssociationV1,
}

impl GeometryRevisionAssociationEnvelopeV1 {
    /// Validate a total body bijection and derive boundary pairs by retained
    /// parent plus exact Cartesian `(axis, side)` role.
    ///
    /// # Errors
    /// Returns `EQ0901` for stale input artifacts. Returns a typed retention
    /// rejection for missing, split, merged, ambiguous, unknown, or
    /// boundary-incompatible candidates.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        source_model: &impl ReplayableCanonicalModelArtifact,
        source_geometry: &GeometryIdentityEnvelopeV1,
        source_correspondence: &GeometryMeshCorrespondenceEnvelopeV1,
        source_mesh: &SimplicialMeshEnvelopeV1,
        target_model: &impl ReplayableCanonicalModelArtifact,
        target_geometry: &GeometryIdentityEnvelopeV1,
        target_correspondence: &GeometryMeshCorrespondenceEnvelopeV1,
        target_mesh: &SimplicialMeshEnvelopeV1,
        candidates: Vec<BodyAssociationCandidate>,
    ) -> Result<Self, GeometryAssociationArtifactError> {
        source_correspondence
            .validate_against(source_geometry, source_model, source_mesh)
            .map_err(GeometryAssociationArtifactError::Artifact)?;
        target_correspondence
            .validate_against(target_geometry, target_model, target_mesh)
            .map_err(GeometryAssociationArtifactError::Artifact)?;
        let source = source_correspondence
            .typed_correspondence(source_geometry, source_mesh)
            .map_err(GeometryAssociationArtifactError::Artifact)?;
        let target = target_correspondence
            .typed_correspondence(target_geometry, target_mesh)
            .map_err(GeometryAssociationArtifactError::Artifact)?;
        let association = RetainedGeometryAssociation::validate(&source, &target, candidates)
            .map_err(GeometryAssociationArtifactError::Retention)?;
        let wire = WireGeometryRevisionAssociationV1 {
            schema: ASSOCIATION_SCHEMA.to_owned(),
            encoding: CANONICAL_ENCODING.to_owned(),
            source_geometry_sha256: source_geometry
                .digest()
                .map_err(GeometryAssociationArtifactError::Artifact)?
                .to_string(),
            source_correspondence_sha256: source_correspondence
                .digest()
                .map_err(GeometryAssociationArtifactError::Artifact)?
                .to_string(),
            target_geometry_sha256: target_geometry
                .digest()
                .map_err(GeometryAssociationArtifactError::Artifact)?
                .to_string(),
            target_correspondence_sha256: target_correspondence
                .digest()
                .map_err(GeometryAssociationArtifactError::Artifact)?
                .to_string(),
            bodies: association
                .bodies()
                .iter()
                .map(|pair| WireBodyPair {
                    source_domain_ulid: pair.source().ulid().to_string(),
                    target_domain_ulid: pair.target().ulid().to_string(),
                })
                .collect(),
            boundaries: association
                .boundaries()
                .iter()
                .map(|pair| {
                    Ok(WireBoundaryPair {
                        source_domain_ulid: pair.source().ulid().to_string(),
                        target_domain_ulid: pair.target().ulid().to_string(),
                        source_parent_ulid: pair.source_parent().ulid().to_string(),
                        target_parent_ulid: pair.target_parent().ulid().to_string(),
                        axis: u64::try_from(pair.axis()).map_err(|_| {
                            GeometryAssociationArtifactError::Artifact(invalid_artifact(
                                "geometry association axis exceeds portable u64",
                            ))
                        })?,
                        side: WireBoundarySide::encode(pair.side()),
                    })
                })
                .collect::<Result<Vec<_>, GeometryAssociationArtifactError>>()?,
        };
        let envelope = Self { wire };
        envelope
            .validate_local(GeometryDecoderLimits::default())
            .map_err(GeometryAssociationArtifactError::Artifact)?;
        Ok(envelope)
    }

    /// Decode bounded wire data. Exact source and target replay remains
    /// required before the association is trusted.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed, oversized, or noncanonical data.
    pub fn from_json(bytes: &[u8], limits: GeometryDecoderLimits) -> Result<Self, Diagnostic> {
        limits.json.check(bytes)?;
        let wire = serde_json::from_slice(bytes).map_err(|error| {
            invalid_artifact(format!(
                "invalid geometry revision association JSON: {error}"
            ))
        })?;
        let envelope = Self { wire };
        envelope.validate_local(limits)?;
        Ok(envelope)
    }

    /// Recompute the complete one-to-one association from exact resources.
    ///
    /// # Errors
    /// Returns an artifact or typed retention error for any drift.
    #[allow(clippy::too_many_arguments)]
    pub fn validate_against(
        &self,
        source_model: &impl ReplayableCanonicalModelArtifact,
        source_geometry: &GeometryIdentityEnvelopeV1,
        source_correspondence: &GeometryMeshCorrespondenceEnvelopeV1,
        source_mesh: &SimplicialMeshEnvelopeV1,
        target_model: &impl ReplayableCanonicalModelArtifact,
        target_geometry: &GeometryIdentityEnvelopeV1,
        target_correspondence: &GeometryMeshCorrespondenceEnvelopeV1,
        target_mesh: &SimplicialMeshEnvelopeV1,
    ) -> Result<(), GeometryAssociationArtifactError> {
        let candidates = self
            .wire
            .bodies
            .iter()
            .map(|pair| {
                Ok(BodyAssociationCandidate::new(
                    decode_domain(&pair.source_domain_ulid)?,
                    decode_domain(&pair.target_domain_ulid)?,
                ))
            })
            .collect::<Result<Vec<_>, Diagnostic>>()
            .map_err(GeometryAssociationArtifactError::Artifact)?;
        let expected = Self::new(
            source_model,
            source_geometry,
            source_correspondence,
            source_mesh,
            target_model,
            target_geometry,
            target_correspondence,
            target_mesh,
            candidates,
        )?;
        if self != &expected {
            return Err(GeometryAssociationArtifactError::Artifact(
                invalid_artifact(
                    "geometry revision association differs from exact resource replay",
                ),
            ));
        }
        Ok(())
    }

    /// Deterministic canonical JSON bytes.
    ///
    /// # Errors
    /// Returns `EQ0901` if serialization unexpectedly fails.
    pub fn canonical_json(&self) -> Result<Vec<u8>, Diagnostic> {
        serde_json::to_vec(&self.wire).map_err(|error| {
            invalid_artifact(format!(
                "cannot serialize geometry revision association: {error}"
            ))
        })
    }

    /// Domain-separated identity of the complete association proof.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            ASSOCIATION_SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
    }

    /// Target body retained from one exact source body, if present.
    #[must_use]
    pub fn retained_body_target(&self, source: Id<kinds::Domain>) -> Option<Id<kinds::Domain>> {
        self.wire
            .bodies
            .iter()
            .find(|pair| pair.source_domain_ulid == source.ulid().to_string())
            .and_then(|pair| decode_domain(&pair.target_domain_ulid).ok())
    }

    /// Target boundary retained from one exact source boundary, if present.
    #[must_use]
    pub fn retained_boundary_target(&self, source: Id<kinds::Domain>) -> Option<Id<kinds::Domain>> {
        self.wire
            .boundaries
            .iter()
            .find(|pair| pair.source_domain_ulid == source.ulid().to_string())
            .and_then(|pair| decode_domain(&pair.target_domain_ulid).ok())
    }

    fn validate_local(&self, limits: GeometryDecoderLimits) -> Result<(), Diagnostic> {
        if self.wire.schema != ASSOCIATION_SCHEMA || self.wire.encoding != CANONICAL_ENCODING {
            return Err(invalid_artifact(
                "unsupported geometry revision association schema or encoding",
            ));
        }
        for digest in [
            &self.wire.source_geometry_sha256,
            &self.wire.source_correspondence_sha256,
            &self.wire.target_geometry_sha256,
            &self.wire.target_correspondence_sha256,
        ] {
            ArtifactDigest::from_hex(digest.clone())?;
        }
        if self.wire.bodies.is_empty()
            || self.wire.bodies.len() > limits.max_geometry_revision_associations
            || self.wire.boundaries.len() > limits.max_geometry_entities
            || !self
                .wire
                .bodies
                .windows(2)
                .all(|pair| pair[0].source_domain_ulid < pair[1].source_domain_ulid)
        {
            return Err(invalid_artifact(
                "geometry association body pairs are empty, oversized, or noncanonical",
            ));
        }
        let mut source_bodies = BTreeSet::new();
        let mut target_bodies = BTreeSet::new();
        let mut target_by_source = BTreeMap::new();
        for pair in &self.wire.bodies {
            parse_ulid(&pair.source_domain_ulid, "association source body")?;
            parse_ulid(&pair.target_domain_ulid, "association target body")?;
            if !source_bodies.insert(pair.source_domain_ulid.as_str())
                || !target_bodies.insert(pair.target_domain_ulid.as_str())
                || target_by_source
                    .insert(
                        pair.source_domain_ulid.as_str(),
                        pair.target_domain_ulid.as_str(),
                    )
                    .is_some()
            {
                return Err(invalid_artifact(
                    "geometry association body pairs must form a canonical bijection",
                ));
            }
        }
        if self.wire.boundaries.is_empty()
            || !self
                .wire
                .boundaries
                .windows(2)
                .all(|pair| boundary_pair_key(&pair[0]) < boundary_pair_key(&pair[1]))
        {
            return Err(invalid_artifact(
                "geometry association boundary pairs are empty or noncanonical",
            ));
        }
        let mut source_boundaries = BTreeSet::new();
        let mut target_boundaries = BTreeSet::new();
        let mut roles_by_source_parent = BTreeMap::<&str, BTreeSet<(u64, u8)>>::new();
        for pair in &self.wire.boundaries {
            parse_ulid(&pair.source_domain_ulid, "association source boundary")?;
            parse_ulid(&pair.target_domain_ulid, "association target boundary")?;
            parse_ulid(&pair.source_parent_ulid, "association source parent")?;
            parse_ulid(&pair.target_parent_ulid, "association target parent")?;
            if source_bodies.contains(pair.source_domain_ulid.as_str())
                || target_bodies.contains(pair.target_domain_ulid.as_str())
                || !source_boundaries.insert(pair.source_domain_ulid.as_str())
                || !target_boundaries.insert(pair.target_domain_ulid.as_str())
                || target_by_source.get(pair.source_parent_ulid.as_str())
                    != Some(&pair.target_parent_ulid.as_str())
                || !roles_by_source_parent
                    .entry(pair.source_parent_ulid.as_str())
                    .or_default()
                    .insert((pair.axis, pair.side.key()))
            {
                return Err(invalid_artifact(
                    "geometry association boundaries must be unique and derived from exact retained parents",
                ));
            }
        }
        let Some(expected_roles) = roles_by_source_parent.values().next() else {
            return Err(invalid_artifact(
                "geometry association requires boundary roles",
            ));
        };
        if !complete_cartesian_roles(expected_roles)
            || roles_by_source_parent.len() != self.wire.bodies.len()
            || roles_by_source_parent
                .values()
                .any(|roles| roles != expected_roles)
        {
            return Err(invalid_artifact(
                "geometry association boundary roles must be complete and common to every retained body",
            ));
        }
        Ok(())
    }
}

/// Error boundary separating artifact integrity from typed retention failure.
#[derive(Debug)]
pub enum GeometryAssociationArtifactError {
    /// Exact artifact replay or wire validation failed.
    Artifact(Diagnostic),
    /// Candidate relation was not a total one-to-one retained association.
    Retention(RetentionRejection),
}

impl std::fmt::Display for GeometryAssociationArtifactError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Artifact(error) => error.fmt(formatter),
            Self::Retention(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for GeometryAssociationArtifactError {}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireGeometryRevisionAssociationV1 {
    schema: String,
    encoding: String,
    source_geometry_sha256: String,
    source_correspondence_sha256: String,
    target_geometry_sha256: String,
    target_correspondence_sha256: String,
    bodies: Vec<WireBodyPair>,
    boundaries: Vec<WireBoundaryPair>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireBodyPair {
    source_domain_ulid: String,
    target_domain_ulid: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireBoundaryPair {
    source_domain_ulid: String,
    target_domain_ulid: String,
    source_parent_ulid: String,
    target_parent_ulid: String,
    axis: u64,
    side: WireBoundarySide,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireBoundarySide {
    Lower,
    Upper,
}

impl WireBoundarySide {
    const fn encode(side: BoundarySide) -> Self {
        match side {
            BoundarySide::Lower => Self::Lower,
            BoundarySide::Upper => Self::Upper,
        }
    }

    const fn key(self) -> u8 {
        match self {
            Self::Lower => 0,
            Self::Upper => 1,
        }
    }
}

fn boundary_pair_key(pair: &WireBoundaryPair) -> (&str, u64, u8, &str) {
    (
        &pair.source_parent_ulid,
        pair.axis,
        pair.side.key(),
        &pair.source_domain_ulid,
    )
}

fn complete_cartesian_roles(roles: &BTreeSet<(u64, u8)>) -> bool {
    !roles.is_empty()
        && roles.len().is_multiple_of(2)
        && roles.iter().copied().enumerate().all(|(index, role)| {
            u64::try_from(index / 2).is_ok_and(|axis| role == (axis, u8::from(index % 2 != 0)))
        })
}

fn parse_ulid(value: &str, label: &str) -> Result<Ulid, Diagnostic> {
    Ulid::from_str(value).map_err(|_| invalid_artifact(format!("{label} ULID is invalid")))
}

fn decode_domain(value: &str) -> Result<Id<kinds::Domain>, Diagnostic> {
    parse_ulid(value, "geometry association Domain").map(Id::from_ulid)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::complete_cartesian_roles;

    #[test]
    fn cartesian_roles_are_dense_paired_and_resource_bounded() {
        assert!(complete_cartesian_roles(&BTreeSet::from([
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ])));
        assert!(!complete_cartesian_roles(&BTreeSet::from([
            (0, 0),
            (0, 1),
            (2, 0),
            (2, 1),
        ])));
        assert!(!complete_cartesian_roles(&BTreeSet::from([
            (u64::MAX, 0),
            (u64::MAX, 1),
        ])));
    }
}
