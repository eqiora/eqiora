//! Content-bound identity of semantic geometry selections.
//!
//! Semantic [`Domain`](eqiora_schema::kernel::DomainKind) identities remain
//! the source of model meaning. This artifact gives one exact Model revision
//! a closed geometry-entity catalog without exposing adapter face numbers or
//! adding geometry objects to the Semantic Kernel.

use std::collections::{BTreeMap, BTreeSet};
use std::str::FromStr;

use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, Id};
use eqiora_geometry::GeometryEntity;
use eqiora_graph::EdgeKind;
use eqiora_schema::kernel::{BoundarySide, DomainKind, KernelNode};
use serde::{Deserialize, Serialize};
use ulid::Ulid;

use crate::{
    ArtifactDigest, CANONICAL_ENCODING, ReplayableCanonicalModelArtifact, check_json_limits,
    invalid_artifact,
};

const GEOMETRY_IDENTITY_SCHEMA: &str = "eqiora.geometry-identity-envelope/v1";

/// Semantic work budgets shared by Geometry Identity and correspondence artifacts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeometryDecoderLimits {
    /// Common JSON syntax admission.
    pub json: crate::JsonDecoderLimits,
    /// Maximum body and boundary entities in one Geometry Identity.
    pub max_geometry_entities: usize,
    /// Maximum cell and facet memberships in one geometry correspondence.
    pub max_geometry_mesh_memberships: usize,
    /// Maximum decisions in one cross-revision geometry association.
    pub max_geometry_revision_associations: usize,
}

impl Default for GeometryDecoderLimits {
    fn default() -> Self {
        Self {
            json: crate::JsonDecoderLimits::default(),
            max_geometry_entities: 1_000_000,
            max_geometry_mesh_memberships: 16_000_000,
            max_geometry_revision_associations: 1_000_000,
        }
    }
}

/// One exact Cartesian body selected from the Semantic Model.
#[derive(Clone, Debug, PartialEq)]
pub struct CartesianGeometryBodyV1 {
    domain: Id<kinds::Domain>,
    entity: GeometryEntity,
    bounds_m: Vec<(f64, f64)>,
}

impl CartesianGeometryBodyV1 {
    /// Exact Semantic Domain identity.
    #[must_use]
    pub const fn domain(&self) -> Id<kinds::Domain> {
        self.domain
    }

    /// Artifact-local full-dimensional geometry entity.
    #[must_use]
    pub const fn entity(&self) -> GeometryEntity {
        self.entity
    }

    /// Cartesian axis bounds in coherent SI metres.
    #[must_use]
    pub fn bounds_m(&self) -> &[(f64, f64)] {
        &self.bounds_m
    }
}

/// One exact oriented boundary selected from the Semantic Model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CartesianGeometryBoundaryV1 {
    domain: Id<kinds::Domain>,
    parent: Id<kinds::Domain>,
    entity: GeometryEntity,
    parent_entity: GeometryEntity,
    axis: usize,
    side: BoundarySide,
}

impl CartesianGeometryBoundaryV1 {
    /// Exact boundary Domain identity.
    #[must_use]
    pub const fn domain(self) -> Id<kinds::Domain> {
        self.domain
    }

    /// Exact parent body Domain identity.
    #[must_use]
    pub const fn parent(self) -> Id<kinds::Domain> {
        self.parent
    }

    /// Artifact-local codimension-one geometry entity.
    #[must_use]
    pub const fn entity(self) -> GeometryEntity {
        self.entity
    }

    /// Artifact-local parent body entity.
    #[must_use]
    pub const fn parent_entity(self) -> GeometryEntity {
        self.parent_entity
    }

    /// Cartesian normal axis.
    #[must_use]
    pub const fn axis(self) -> usize {
        self.axis
    }

    /// Parent-outward side. The physical orientation is derived from this
    /// role and the exact parent, never supplied as an independent sign.
    #[must_use]
    pub const fn side(self) -> BoundarySide {
        self.side
    }
}

/// Versioned geometry identity for complete Cartesian bodies in one exact
/// Model artifact.
#[derive(Clone, Debug, PartialEq)]
pub struct GeometryIdentityEnvelopeV1 {
    wire: WireGeometryIdentityV1,
}

impl GeometryIdentityEnvelopeV1 {
    /// Derive a canonical geometry catalog from exact Cartesian body Domains.
    ///
    /// Every body must have exactly one `CartesianBoundary` child for each
    /// `(axis, side)` role. Body input order is non-semantic. `tolerance_m` is
    /// the producer's coherent-SI classification precision for this geometry
    /// revision. It is part of geometry identity and is reused by every
    /// geometry-to-mesh membership decision; it is not mesh quality policy.
    ///
    /// # Errors
    /// Returns `EQ0901` for an invalid Model replay, empty/duplicate/non-box
    /// selection, incomplete exterior, mixed dimension, or invalid tolerance.
    pub fn new(
        model: &impl ReplayableCanonicalModelArtifact,
        bodies: impl IntoIterator<Item = Id<kinds::Domain>>,
        tolerance_m: f64,
    ) -> Result<Self, Diagnostic> {
        if !tolerance_m.is_finite() || tolerance_m <= 0.0 {
            return Err(invalid_artifact(
                "geometry tolerance must be finite and positive in metres",
            ));
        }
        let replay = model.replay_model()?;
        let program = replay.program();
        let model_reference = replay.artifact_reference();
        let mut bodies = bodies.into_iter().collect::<Vec<_>>();
        bodies.sort_by_key(Id::ulid);
        if bodies.is_empty()
            || bodies
                .windows(2)
                .any(|pair| pair[0].ulid() == pair[1].ulid())
        {
            return Err(invalid_artifact(
                "geometry identity requires nonempty unique body Domains",
            ));
        }

        let mut wire_bodies = Vec::with_capacity(bodies.len());
        let mut next_boundary_index = 0_usize;
        let mut common_dimension = None;
        for (body_index, body) in bodies.into_iter().enumerate() {
            let Some(KernelNode::Domain(_)) = program.node(body.erase()) else {
                return Err(invalid_artifact(
                    "geometry body identity does not name a retained Domain",
                ));
            };
            let bounds = program.resolved_cartesian_bounds(body).map_err(|_| {
                invalid_artifact("geometry identity v1 admits only Cartesian box bodies")
            })?;
            let dimension = bounds.len();
            if common_dimension
                .replace(dimension)
                .is_some_and(|value| value != dimension)
            {
                return Err(invalid_artifact(
                    "one geometry identity revision requires a common body dimension",
                ));
            }
            let body_entity = WireGeometryEntity::new(dimension, body_index)?;
            let mut role_domains = BTreeMap::new();
            for edge in program
                .edges()
                .iter()
                .filter(|edge| edge.kind() == EdgeKind::BoundaryOf && edge.to() == body.erase())
            {
                let Some(KernelNode::Domain(boundary)) = program.node(edge.from()) else {
                    return Err(invalid_artifact(
                        "BoundaryOf source is not a retained Domain",
                    ));
                };
                let DomainKind::CartesianBoundary { axis, side } = boundary.kind() else {
                    return Err(invalid_artifact(
                        "geometry identity v1 requires Cartesian boundary children",
                    ));
                };
                if role_domains.insert((*axis, *side), boundary.id()).is_some() {
                    return Err(invalid_artifact(
                        "geometry body has more than one boundary for one Cartesian role",
                    ));
                }
            }
            let expected_roles = (0..dimension)
                .flat_map(|axis| {
                    [BoundarySide::Lower, BoundarySide::Upper]
                        .into_iter()
                        .map(move |side| (axis, side))
                })
                .collect::<Vec<_>>();
            if role_domains.len() != expected_roles.len()
                || expected_roles
                    .iter()
                    .any(|role| !role_domains.contains_key(role))
            {
                return Err(invalid_artifact(
                    "geometry body requires one exact boundary for every Cartesian axis/side",
                ));
            }
            let mut wire_boundaries = Vec::with_capacity(expected_roles.len());
            for (axis, side) in expected_roles {
                let boundary = role_domains[&(axis, side)];
                wire_boundaries.push(WireGeometryBoundary {
                    domain_ulid: boundary.ulid().to_string(),
                    entity: WireGeometryEntity::new(dimension - 1, next_boundary_index)?,
                    parent_entity: body_entity,
                    axis: u64::try_from(axis)
                        .map_err(|_| invalid_artifact("geometry axis exceeds portable u64"))?,
                    side: WireBoundarySide::encode(side),
                    orientation: WireBoundaryOrientation::ParentOutward,
                });
                next_boundary_index = next_boundary_index
                    .checked_add(1)
                    .ok_or_else(|| invalid_artifact("geometry boundary count overflows usize"))?;
            }
            wire_bodies.push(WireGeometryBody {
                domain_ulid: body.ulid().to_string(),
                entity: body_entity,
                bounds_m: bounds
                    .iter()
                    .map(|axis| WireAxisBounds {
                        lower_m: canonical_geometry_scalar(axis.lower().value()),
                        upper_m: canonical_geometry_scalar(axis.upper().value()),
                    })
                    .collect(),
                boundaries: wire_boundaries,
            });
        }
        let dimension = common_dimension.expect("nonempty body set has one dimension");
        let boundary_keys = wire_bodies
            .iter()
            .flat_map(|body| {
                body.boundaries
                    .iter()
                    .map(move |boundary| boundary_embedding_key(&body.bounds_m, boundary))
            })
            .collect::<BTreeSet<_>>();
        let boundary_entities = boundary_keys
            .into_iter()
            .enumerate()
            .map(|(index, key)| Ok((key, WireGeometryEntity::new(dimension - 1, index)?)))
            .collect::<Result<BTreeMap<_, _>, Diagnostic>>()?;
        for body in &mut wire_bodies {
            let bounds = body.bounds_m.clone();
            for boundary in &mut body.boundaries {
                boundary.entity = boundary_entities[&boundary_embedding_key(&bounds, boundary)];
            }
        }

        let envelope = Self {
            wire: WireGeometryIdentityV1 {
                schema: GEOMETRY_IDENTITY_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                model_sha256: model_reference.artifact().to_string(),
                model_ulid: model_reference.model().ulid().to_string(),
                semantic_revision: model_reference.semantic_revision().get(),
                producer: WireGeometryProducer::SemanticCartesianV1,
                length_unit: WireLengthUnit::Metre,
                tolerance_m,
                bodies: wire_bodies,
            },
        };
        envelope.validate_local(GeometryDecoderLimits::default())?;
        Ok(envelope)
    }

    /// Decode bounded wire data without trusting its external Model reference.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed, noncanonical, or oversized data.
    pub fn from_json(bytes: &[u8], limits: GeometryDecoderLimits) -> Result<Self, Diagnostic> {
        check_json_limits(bytes, limits.json)?;
        let wire = serde_json::from_slice(bytes).map_err(|error| {
            invalid_artifact(format!("invalid geometry identity JSON: {error}"))
        })?;
        let envelope = Self { wire };
        envelope.validate_local(limits)?;
        Ok(envelope)
    }

    /// Replay exact Model membership, parentage, roles, bounds, and digest.
    ///
    /// # Errors
    /// Returns `EQ0901` for any Model or semantic geometry drift.
    pub fn validate_against(
        &self,
        model: &impl ReplayableCanonicalModelArtifact,
    ) -> Result<(), Diagnostic> {
        let body_ids = self
            .wire
            .bodies
            .iter()
            .map(|body| decode_domain(&body.domain_ulid))
            .collect::<Result<Vec<_>, _>>()?;
        let expected = Self::new(model, body_ids, self.wire.tolerance_m)?;
        if self != &expected {
            return Err(invalid_artifact(
                "geometry identity differs from the exact Model-derived Cartesian catalog",
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
            invalid_artifact(format!("cannot serialize geometry identity: {error}"))
        })
    }

    /// Domain-separated content identity of this exact geometry revision.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            GEOMETRY_IDENTITY_SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
    }

    /// Exact referenced Model artifact.
    #[must_use]
    pub fn model_artifact(&self) -> ArtifactDigest {
        ArtifactDigest::from_hex(self.wire.model_sha256.clone())
            .expect("validated geometry Model digest")
    }

    /// Producer-owned coherent-SI classification precision in metres.
    ///
    /// This value is part of the geometry digest. Consumers must reuse it for
    /// geometry membership classification rather than accept a second,
    /// potentially contradictory mesh-local tolerance.
    #[must_use]
    pub const fn tolerance_m(&self) -> f64 {
        self.wire.tolerance_m
    }

    /// Common topological and coordinate dimension.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.wire.bodies[0].bounds_m.len()
    }

    /// Canonically ordered selected bodies.
    #[must_use]
    pub fn bodies(&self) -> Vec<CartesianGeometryBodyV1> {
        self.wire
            .bodies
            .iter()
            .map(|body| CartesianGeometryBodyV1 {
                domain: decode_domain(&body.domain_ulid).expect("validated geometry body"),
                entity: body.entity.decode().expect("validated geometry entity"),
                bounds_m: body
                    .bounds_m
                    .iter()
                    .map(|axis| (axis.lower_m, axis.upper_m))
                    .collect(),
            })
            .collect()
    }

    /// Canonically ordered complete boundary catalog.
    #[must_use]
    pub fn boundaries(&self) -> Vec<CartesianGeometryBoundaryV1> {
        self.wire
            .bodies
            .iter()
            .flat_map(|body| {
                let parent = decode_domain(&body.domain_ulid).expect("validated geometry body");
                body.boundaries
                    .iter()
                    .map(move |boundary| CartesianGeometryBoundaryV1 {
                        domain: decode_domain(&boundary.domain_ulid)
                            .expect("validated geometry boundary"),
                        parent,
                        entity: boundary.entity.decode().expect("validated geometry entity"),
                        parent_entity: boundary
                            .parent_entity
                            .decode()
                            .expect("validated geometry parent entity"),
                        axis: usize::try_from(boundary.axis).expect("validated geometry axis"),
                        side: boundary.side.decode(),
                    })
            })
            .collect()
    }

    fn validate_local(&self, limits: GeometryDecoderLimits) -> Result<(), Diagnostic> {
        if self.wire.schema != GEOMETRY_IDENTITY_SCHEMA
            || self.wire.encoding != CANONICAL_ENCODING
            || self.wire.producer != WireGeometryProducer::SemanticCartesianV1
            || self.wire.length_unit != WireLengthUnit::Metre
        {
            return Err(invalid_artifact(
                "unsupported geometry identity schema, encoding, producer, or unit",
            ));
        }
        ArtifactDigest::from_hex(self.wire.model_sha256.clone())?;
        parse_ulid(&self.wire.model_ulid, "geometry Model")?;
        if !self.wire.tolerance_m.is_finite() || self.wire.tolerance_m <= 0.0 {
            return Err(invalid_artifact(
                "geometry tolerance must be finite and positive in metres",
            ));
        }
        if self.wire.bodies.is_empty() || self.wire.bodies.len() > limits.max_geometry_entities {
            return Err(invalid_artifact(
                "geometry body count is empty or exceeds decoder limits",
            ));
        }
        let dimension = self.wire.bodies[0].bounds_m.len();
        if dimension == 0 {
            return Err(invalid_artifact("geometry dimension must be positive"));
        }
        let mut domains = BTreeSet::new();
        let mut entities = BTreeSet::new();
        let expected_boundary_entities = self
            .wire
            .bodies
            .iter()
            .flat_map(|body| {
                body.boundaries
                    .iter()
                    .map(move |boundary| boundary_embedding_key(&body.bounds_m, boundary))
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .enumerate()
            .map(|(index, key)| (key, GeometryEntity::new(dimension - 1, index)))
            .collect::<BTreeMap<_, _>>();
        let mut boundary_count = 0_usize;
        for (body_index, body) in self.wire.bodies.iter().enumerate() {
            parse_ulid(&body.domain_ulid, "geometry body")?;
            if !domains.insert(body.domain_ulid.as_str())
                || body.entity.decode()? != GeometryEntity::new(dimension, body_index)
                || body.bounds_m.len() != dimension
                || body.bounds_m.iter().any(|axis| {
                    !axis.lower_m.is_finite()
                        || !axis.upper_m.is_finite()
                        || is_negative_zero(axis.lower_m)
                        || is_negative_zero(axis.upper_m)
                        || axis.upper_m <= axis.lower_m
                })
            {
                return Err(invalid_artifact(
                    "geometry bodies must be unique, canonical, finite Cartesian boxes",
                ));
            }
            entities.insert(body.entity);
            if body.boundaries.len() != 2 * dimension {
                return Err(invalid_artifact(
                    "geometry body does not retain its complete Cartesian exterior",
                ));
            }
            for (role_index, boundary) in body.boundaries.iter().enumerate() {
                parse_ulid(&boundary.domain_ulid, "geometry boundary")?;
                let expected_axis = role_index / 2;
                let expected_side = if role_index % 2 == 0 {
                    WireBoundarySide::Lower
                } else {
                    WireBoundarySide::Upper
                };
                if !domains.insert(boundary.domain_ulid.as_str())
                    || boundary.axis
                        != u64::try_from(expected_axis)
                            .map_err(|_| invalid_artifact("geometry axis exceeds u64"))?
                    || boundary.side != expected_side
                    || boundary.orientation != WireBoundaryOrientation::ParentOutward
                    || boundary.parent_entity != body.entity
                    || boundary.entity.decode()?
                        != expected_boundary_entities
                            [&boundary_embedding_key(&body.bounds_m, boundary)]
                {
                    return Err(invalid_artifact(
                        "geometry boundaries must be unique and canonically parent-outward",
                    ));
                }
                boundary_count = boundary_count
                    .checked_add(1)
                    .ok_or_else(|| invalid_artifact("geometry boundary count overflows usize"))?;
            }
        }
        if boundary_count > limits.max_geometry_entities
            || entities.len() + expected_boundary_entities.len() > limits.max_geometry_entities
            || !strictly_sorted_by_domain(&self.wire.bodies)
        {
            return Err(invalid_artifact(
                "geometry entity count or canonical body order is invalid",
            ));
        }
        Ok(())
    }
}

// The wire DTOs stay private so future CAD adapters can add a new explicit
// schema without turning this bounded Cartesian contract into an anything-box.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireGeometryIdentityV1 {
    schema: String,
    encoding: String,
    model_sha256: String,
    model_ulid: String,
    semantic_revision: u64,
    producer: WireGeometryProducer,
    length_unit: WireLengthUnit,
    tolerance_m: f64,
    bodies: Vec<WireGeometryBody>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireGeometryProducer {
    SemanticCartesianV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireLengthUnit {
    Metre,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireGeometryBody {
    domain_ulid: String,
    entity: WireGeometryEntity,
    bounds_m: Vec<WireAxisBounds>,
    boundaries: Vec<WireGeometryBoundary>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireAxisBounds {
    lower_m: f64,
    upper_m: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireGeometryBoundary {
    domain_ulid: String,
    entity: WireGeometryEntity,
    parent_entity: WireGeometryEntity,
    axis: u64,
    side: WireBoundarySide,
    orientation: WireBoundaryOrientation,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireGeometryEntity {
    dimension: u64,
    index: u64,
}

impl WireGeometryEntity {
    pub(super) fn new(dimension: usize, index: usize) -> Result<Self, Diagnostic> {
        Ok(Self {
            dimension: u64::try_from(dimension)
                .map_err(|_| invalid_artifact("geometry dimension exceeds portable u64"))?,
            index: u64::try_from(index)
                .map_err(|_| invalid_artifact("geometry entity index exceeds portable u64"))?,
        })
    }

    pub(super) fn decode(self) -> Result<GeometryEntity, Diagnostic> {
        Ok(GeometryEntity::new(
            usize::try_from(self.dimension)
                .map_err(|_| invalid_artifact("geometry dimension exceeds local usize"))?,
            usize::try_from(self.index)
                .map_err(|_| invalid_artifact("geometry entity index exceeds local usize"))?,
        ))
    }
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

    const fn decode(self) -> BoundarySide {
        match self {
            Self::Lower => BoundarySide::Lower,
            Self::Upper => BoundarySide::Upper,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireBoundaryOrientation {
    ParentOutward,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct BoundaryEmbeddingKey {
    axis: u64,
    coordinate_bits: u64,
    tangential_bounds_bits: Vec<(u64, u64)>,
}

fn boundary_embedding_key(
    bounds_m: &[WireAxisBounds],
    boundary: &WireGeometryBoundary,
) -> BoundaryEmbeddingKey {
    let axis = usize::try_from(boundary.axis).expect("validated geometry axis");
    let coordinate = match boundary.side {
        WireBoundarySide::Lower => bounds_m[axis].lower_m,
        WireBoundarySide::Upper => bounds_m[axis].upper_m,
    };
    BoundaryEmbeddingKey {
        axis: boundary.axis,
        coordinate_bits: canonical_geometry_scalar(coordinate).to_bits(),
        tangential_bounds_bits: bounds_m
            .iter()
            .enumerate()
            .filter(|(candidate, _)| *candidate != axis)
            .map(|(_, bounds)| {
                (
                    canonical_geometry_scalar(bounds.lower_m).to_bits(),
                    canonical_geometry_scalar(bounds.upper_m).to_bits(),
                )
            })
            .collect(),
    }
}

const fn canonical_geometry_scalar(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

fn is_negative_zero(value: f64) -> bool {
    value == 0.0 && value.is_sign_negative()
}

fn parse_ulid(value: &str, label: &str) -> Result<Ulid, Diagnostic> {
    Ulid::from_str(value).map_err(|_| invalid_artifact(format!("{label} ULID is invalid")))
}

fn decode_domain(value: &str) -> Result<Id<kinds::Domain>, Diagnostic> {
    parse_ulid(value, "geometry Domain").map(Id::from_ulid)
}

fn strictly_sorted_by_domain(bodies: &[WireGeometryBody]) -> bool {
    bodies
        .windows(2)
        .all(|pair| pair[0].domain_ulid < pair[1].domain_ulid)
}

#[cfg(test)]
mod tests {
    use super::canonical_geometry_scalar;

    #[test]
    fn canonical_geometry_scalar_erases_signed_zero() {
        assert_eq!(canonical_geometry_scalar(-0.0).to_bits(), 0.0_f64.to_bits());
        assert_eq!(
            canonical_geometry_scalar(1.25).to_bits(),
            1.25_f64.to_bits()
        );
    }
}
