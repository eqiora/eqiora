use std::collections::BTreeSet;
use std::num::NonZeroU16;
use std::str::FromStr;

use crate::dimension::WireDimension;

use eqiora_core::ValueFrame;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, DimExponents, Id, ValueShape};
use eqiora_graph::EdgeKind;
use eqiora_meshing::{DiscreteFieldAssociation, DiscreteFieldShape, MeshEntity, MeshTopology};
use eqiora_realization::SpaceFamily;
use eqiora_schema::kernel::KernelNode;
use serde::{Deserialize, Serialize};
use ulid::Ulid;

use crate::{
    ArtifactDigest, CANONICAL_ENCODING, DiscreteFieldEnvelopeV1, FieldDecoderLimits,
    GeometryMeshCorrespondenceEnvelopeV1, ModelArtifactReference, ReplayableCanonicalModelArtifact,
    ReplayableFixedTopologyAleRealizationArtifact, SimplicialMeshEnvelopeV1,
    ValidatedMovingSpatialContextV2, check_json_limits, invalid_artifact,
};

const FIELD_SNAPSHOT_SCHEMA: &str = "eqiora.field-snapshot-envelope/v2";

/// Validated lineage shared by coefficient snapshots across spatial contexts.
///
/// This stays private: it removes duplicated validation machinery without
/// turning an implementation seam into another public extension point.
pub(super) trait ValidatedFieldSnapshotContext {
    fn model_reference(&self) -> &ModelArtifactReference;
    fn program(&self) -> &eqiora_sem::KernelProgram;
    fn realization_artifact(&self) -> Result<ArtifactDigest, Diagnostic>;
    fn geometry_artifact(&self) -> Result<ArtifactDigest, Diagnostic>;
    fn correspondence(&self) -> &GeometryMeshCorrespondenceEnvelopeV1;
    fn mesh(&self) -> &SimplicialMeshEnvelopeV1;
    fn active_cells(&self, domain: Id<kinds::Domain>) -> Result<Vec<usize>, Diagnostic>;
    fn realized_field_space(
        &self,
        field: Id<kinds::Field>,
    ) -> Result<(Id<kinds::Domain>, SpaceFamily), Diagnostic>;
}

/// Logical snapshot of one exact Semantic Field in one exact spatial Realization.
///
/// Numerical values remain normalized `DiscreteFieldEnvelopeV1` leaves. A P1
/// Field has one vertex block; a simplex P1-bubble Field has a vertex block
/// and a top-cell bubble block. Values outside the exact Domain closure are
/// canonical positive zero, so every block retains the canonical mesh entity
/// order while the Domain support remains exact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldSnapshotEnvelopeV2 {
    wire: WireFieldSnapshotV2,
}

impl FieldSnapshotEnvelopeV2 {
    /// Bind normalized coefficient blocks in a fixed-topology moving context.
    ///
    /// Geometry motion changes the enclosing state lineage, not the
    /// reference-mesh coefficient representation.
    ///
    /// # Errors
    /// Returns `EQ0901` when the exact ALE Realization, immutable reference
    /// topology, semantic Field, and normalized coefficient blocks disagree.
    pub fn new_moving<
        M: ReplayableCanonicalModelArtifact,
        R: ReplayableFixedTopologyAleRealizationArtifact,
    >(
        context: &ValidatedMovingSpatialContextV2<'_, M, R>,
        field: Id<kinds::Field>,
        blocks: &[DiscreteFieldEnvelopeV1],
    ) -> Result<Self, Diagnostic> {
        Self::new_in_context(context, field, blocks)
    }

    pub(super) fn new_in_context<'a>(
        context: &impl ValidatedFieldSnapshotContext,
        field: Id<kinds::Field>,
        blocks: impl IntoIterator<Item = &'a DiscreteFieldEnvelopeV1>,
    ) -> Result<Self, Diagnostic> {
        let blocks = blocks.into_iter().collect::<Vec<_>>();
        let correspondence = context.correspondence();
        let mesh = context.mesh();
        let program = context.program();
        let definition = match program.node(field.erase()) {
            Some(KernelNode::Field(definition)) => definition,
            _ => return Err(invalid_artifact("Field snapshot identity is not a Field")),
        };
        let semantic_domain = program
            .edges()
            .iter()
            .filter(|edge| edge.from() == field.erase() && edge.kind() == EdgeKind::DefinedOn)
            .filter_map(|edge| match program.node(edge.to()) {
                Some(KernelNode::Domain(_)) => edge.to().downcast::<kinds::Domain>(),
                _ => None,
            })
            .collect::<Vec<_>>();
        if semantic_domain.len() != 1 {
            return Err(invalid_artifact(
                "Field snapshot requires one exact volume Domain support",
            ));
        }
        let (realized_domain, space) = context.realized_field_space(field)?;
        if semantic_domain[0] != realized_domain {
            return Err(invalid_artifact(
                "Field snapshot Semantic and Realization Domain supports differ",
            ));
        }

        let expected_associations = match space {
            SpaceFamily::ContinuousLagrange { order } if order == NonZeroU16::MIN => {
                vec![DiscreteFieldAssociation::Vertex]
            }
            SpaceFamily::SimplexP1Bubble => vec![
                DiscreteFieldAssociation::Vertex,
                DiscreteFieldAssociation::Cell,
            ],
            _ => {
                return Err(invalid_artifact(
                    "field snapshots admit only P1 and simplex P1-bubble coefficient spaces",
                ));
            }
        };
        if blocks.len() != expected_associations.len() {
            return Err(invalid_artifact(
                "Field snapshot coefficient-block inventory differs from its Realization space",
            ));
        }
        let mut ordered = blocks;
        ordered.sort_by_key(|block| association_order(block.association()));
        if ordered
            .windows(2)
            .any(|pair| pair[0].association() == pair[1].association())
            || ordered
                .iter()
                .map(|block| block.association())
                .ne(expected_associations.iter().copied())
        {
            return Err(invalid_artifact(
                "Field snapshot coefficient blocks do not match the exact canonical space roles",
            ));
        }

        let field_shape = definition.shape().clone();
        require_portable_shape(&field_shape)?;
        let active_cells = context.active_cells(realized_domain)?;
        let active_vertices =
            support_indices_from_cells(mesh, &active_cells, DiscreteFieldAssociation::Vertex)?;
        for block in &ordered {
            if block.mesh_artifact() != mesh.digest()? {
                return Err(invalid_artifact(
                    "Field snapshot coefficient block references a stale mesh",
                ));
            }
            require_matching_shape(&field_shape, block.component_shape())?;
            block.validate_mesh_artifact(mesh)?;
            let active = match block.association() {
                DiscreteFieldAssociation::Vertex => active_vertices.as_slice(),
                DiscreteFieldAssociation::Cell => active_cells.as_slice(),
            };
            require_zero_outside_support(block, active)?;
        }

        let value = Self {
            wire: WireFieldSnapshotV2 {
                schema: FIELD_SNAPSHOT_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                model_sha256: context.model_reference().artifact().to_string(),
                semantic_revision: context.model_reference().semantic_revision().get(),
                realization_sha256: context.realization_artifact()?.to_string(),
                geometry_sha256: context.geometry_artifact()?.to_string(),
                correspondence_sha256: correspondence.digest()?.to_string(),
                mesh_sha256: mesh.digest()?.to_string(),
                field_ulid: field.ulid().to_string(),
                support_domain_ulid: realized_domain.ulid().to_string(),
                physical: WirePhysicalType {
                    unit_system: WireUnitSystem::CoherentSi,
                    dimension: WireDimension::encode(definition.dimension()),
                    value_shape: WireValueShape::encode(&field_shape),
                    frame: WireFrame::encode(definition.frame()),
                },
                representation: WireRepresentation {
                    scalar: WireScalar::F64,
                    ordering: WireOrdering::CanonicalMeshEntityMajor,
                    blocks: ordered
                        .into_iter()
                        .map(|block| {
                            Ok(WireFieldBlock {
                                association: WireAssociation::encode(block.association()),
                                discrete_field_sha256: block.digest()?.to_string(),
                            })
                        })
                        .collect::<Result<Vec<_>, Diagnostic>>()?,
                },
            },
        };
        value.validate_local(FieldDecoderLimits::default())?;
        Ok(value)
    }

    /// Decode the closed logical manifest without resolving referenced blocks.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed, oversized, noncanonical, or unsupported
    /// wire data.
    pub fn from_json(bytes: &[u8], limits: FieldDecoderLimits) -> Result<Self, Diagnostic> {
        check_json_limits(bytes, limits.json)?;
        let wire = serde_json::from_slice(bytes)
            .map_err(|error| invalid_artifact(format!("invalid Field snapshot JSON: {error}")))?;
        let value = Self { wire };
        value.validate_local(limits)?;
        Ok(value)
    }

    /// Deterministic canonical JSON bytes.
    ///
    /// # Errors
    /// Returns `EQ0901` if serialization unexpectedly fails.
    pub fn canonical_json(&self) -> Result<Vec<u8>, Diagnostic> {
        serde_json::to_vec(&self.wire)
            .map_err(|error| invalid_artifact(format!("cannot serialize Field snapshot: {error}")))
    }

    /// Domain-separated logical identity of meaning and normalized block content.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            FIELD_SNAPSHOT_SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
    }

    /// Exact Semantic Model artifact.
    #[must_use]
    pub fn model_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.model_sha256.clone())
    }

    /// Exact Realization artifact.
    #[must_use]
    pub fn realization_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.realization_sha256.clone())
    }

    /// Exact geometry revision.
    #[must_use]
    pub fn geometry_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.geometry_sha256.clone())
    }

    /// Exact geometry-to-mesh correspondence.
    #[must_use]
    pub fn correspondence_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.correspondence_sha256.clone())
    }

    /// Exact mesh revision.
    #[must_use]
    pub fn mesh_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.mesh_sha256.clone())
    }

    /// Exact Semantic Field identity.
    #[must_use]
    pub fn field(&self) -> Id<kinds::Field> {
        parse_id(&self.wire.field_ulid, "Field")
            .expect("validated Field snapshot contains one Field ULID")
    }

    /// Exact volume Domain support.
    #[must_use]
    pub fn support_domain(&self) -> Id<kinds::Domain> {
        parse_id(&self.wire.support_domain_ulid, "support Domain")
            .expect("validated Field snapshot contains one Domain ULID")
    }

    /// Physical dimension in coherent SI base units.
    #[must_use]
    pub const fn dimension(&self) -> DimExponents {
        self.wire.physical.dimension.decode()
    }

    /// Exact mathematical value shape.
    #[must_use]
    pub fn value_shape(&self) -> ValueShape {
        self.wire
            .physical
            .value_shape
            .decode()
            .expect("validated Field snapshot shape remains representable")
    }

    /// Coordinate-frame meaning of components.
    #[must_use]
    pub const fn frame(&self) -> ValueFrame {
        self.wire.physical.frame.decode()
    }

    /// Canonically ordered normalized coefficient-block identities.
    #[must_use]
    pub fn block_artifacts(&self) -> Vec<(DiscreteFieldAssociation, ArtifactDigest)> {
        self.wire
            .representation
            .blocks
            .iter()
            .map(|block| {
                (
                    block.association.decode(),
                    ArtifactDigest(block.discrete_field_sha256.clone()),
                )
            })
            .collect()
    }

    /// Rebuild and compare this logical snapshot in an exact moving context.
    ///
    /// # Errors
    /// Returns `EQ0901` for any semantic, ALE resource, block, or content
    /// drift.
    pub fn validate_against_moving<
        'a,
        M: ReplayableCanonicalModelArtifact,
        R: ReplayableFixedTopologyAleRealizationArtifact,
    >(
        &self,
        context: &ValidatedMovingSpatialContextV2<'_, M, R>,
        blocks: impl IntoIterator<Item = &'a DiscreteFieldEnvelopeV1>,
    ) -> Result<(), Diagnostic> {
        let expected = Self::new_in_context(context, self.field(), blocks)?;
        if self != &expected {
            return Err(invalid_artifact(
                "Field snapshot differs from exact moving semantic and numerical replay",
            ));
        }
        Ok(())
    }

    /// Derive the exact active mesh entities for one coefficient association.
    ///
    /// This operation revalidates the snapshot's semantic and moving-spatial
    /// lineage. Returned indices are sorted in canonical reference-mesh order;
    /// they never imply identity across mesh revisions.
    ///
    /// # Errors
    /// Returns `EQ0901` for stale snapshot lineage, an association absent from
    /// the snapshot, or invalid support incidence.
    pub fn active_entities_against_moving<
        M: ReplayableCanonicalModelArtifact,
        R: ReplayableFixedTopologyAleRealizationArtifact,
    >(
        &self,
        context: &ValidatedMovingSpatialContextV2<'_, M, R>,
        association: DiscreteFieldAssociation,
    ) -> Result<Vec<usize>, Diagnostic> {
        context.validate_snapshot(self)?;
        self.active_entities(context.correspondence(), context.mesh(), association)
    }

    fn active_entities(
        &self,
        correspondence: &GeometryMeshCorrespondenceEnvelopeV1,
        mesh: &SimplicialMeshEnvelopeV1,
        association: DiscreteFieldAssociation,
    ) -> Result<Vec<usize>, Diagnostic> {
        if !self
            .block_artifacts()
            .iter()
            .any(|(candidate, _)| *candidate == association)
        {
            return Err(invalid_artifact(
                "Field snapshot does not contain the requested coefficient association",
            ));
        }
        let cells = correspondence
            .body_cells(self.support_domain())
            .ok_or_else(|| invalid_artifact("Field snapshot Domain has no exact mesh cells"))?;
        support_indices_from_cells(mesh, &cells, association)
    }

    /// Count active entities without allocating their materialized `usize`
    /// index list.
    ///
    /// This is the preflight counterpart of
    /// [`Self::active_entities_against_moving`].
    ///
    /// # Errors
    /// Returns `EQ0901` under the same lineage and support conditions as the
    /// materializing operation.
    pub fn active_entity_count_against_moving<
        M: ReplayableCanonicalModelArtifact,
        R: ReplayableFixedTopologyAleRealizationArtifact,
    >(
        &self,
        context: &ValidatedMovingSpatialContextV2<'_, M, R>,
        association: DiscreteFieldAssociation,
    ) -> Result<usize, Diagnostic> {
        context.validate_snapshot(self)?;
        if !self
            .block_artifacts()
            .iter()
            .any(|(candidate, _)| *candidate == association)
        {
            return Err(invalid_artifact(
                "Field snapshot does not contain the requested coefficient association",
            ));
        }
        let cells = context
            .correspondence()
            .body_cell_indices(self.support_domain())
            .ok_or_else(|| invalid_artifact("Field snapshot Domain has no exact mesh cells"))?;
        match association {
            DiscreteFieldAssociation::Cell => Ok(cells.len()),
            DiscreteFieldAssociation::Vertex => {
                let mesh = context.mesh().mesh();
                let mut active = vec![false; mesh.vertices().len()];
                let mut count = 0_usize;
                for cell in cells {
                    for &vertex in &mesh.cells()[cell] {
                        if !active[vertex] {
                            active[vertex] = true;
                            count = count.checked_add(1).ok_or_else(|| {
                                invalid_artifact(
                                    "Field snapshot active Vertex count overflows usize",
                                )
                            })?;
                        }
                    }
                }
                Ok(count)
            }
        }
    }

    fn validate_local(&self, limits: FieldDecoderLimits) -> Result<(), Diagnostic> {
        if self.wire.schema != FIELD_SNAPSHOT_SCHEMA
            || self.wire.encoding != CANONICAL_ENCODING
            || self.wire.physical.unit_system != WireUnitSystem::CoherentSi
            || self.wire.representation.scalar != WireScalar::F64
            || self.wire.representation.ordering != WireOrdering::CanonicalMeshEntityMajor
        {
            return Err(invalid_artifact(
                "unsupported Field snapshot schema, encoding, unit, scalar, or ordering",
            ));
        }
        for digest in [
            &self.wire.model_sha256,
            &self.wire.realization_sha256,
            &self.wire.geometry_sha256,
            &self.wire.correspondence_sha256,
            &self.wire.mesh_sha256,
        ] {
            ArtifactDigest::from_hex(digest.clone())?;
        }
        parse_id::<kinds::Field>(&self.wire.field_ulid, "Field")?;
        parse_id::<kinds::Domain>(&self.wire.support_domain_ulid, "support Domain")?;
        let shape = self.wire.physical.value_shape.decode()?;
        require_portable_shape(&shape)?;
        let blocks = &self.wire.representation.blocks;
        if blocks.is_empty() || blocks.len() > limits.max_field_snapshot_blocks {
            return Err(invalid_artifact(
                "Field snapshot coefficient-block count is empty or exceeds the decoder limit",
            ));
        }
        for block in blocks {
            ArtifactDigest::from_hex(block.discrete_field_sha256.clone())?;
        }
        if blocks.windows(2).any(|pair| {
            association_order(pair[0].association.decode())
                >= association_order(pair[1].association.decode())
        }) {
            return Err(invalid_artifact(
                "Field snapshot coefficient blocks must be in unique canonical association order",
            ));
        }
        Ok(())
    }
}

impl<M: ReplayableCanonicalModelArtifact, R: ReplayableFixedTopologyAleRealizationArtifact>
    ValidatedFieldSnapshotContext for ValidatedMovingSpatialContextV2<'_, M, R>
{
    fn model_reference(&self) -> &ModelArtifactReference {
        ValidatedMovingSpatialContextV2::model_reference(self)
    }

    fn program(&self) -> &eqiora_sem::KernelProgram {
        ValidatedMovingSpatialContextV2::program(self)
    }

    fn realization_artifact(&self) -> Result<ArtifactDigest, Diagnostic> {
        self.realization_artifact()
    }

    fn geometry_artifact(&self) -> Result<ArtifactDigest, Diagnostic> {
        ValidatedMovingSpatialContextV2::geometry(self).digest()
    }

    fn correspondence(&self) -> &GeometryMeshCorrespondenceEnvelopeV1 {
        ValidatedMovingSpatialContextV2::correspondence(self)
    }

    fn mesh(&self) -> &SimplicialMeshEnvelopeV1 {
        ValidatedMovingSpatialContextV2::mesh(self)
    }

    fn active_cells(&self, domain: Id<kinds::Domain>) -> Result<Vec<usize>, Diagnostic> {
        self.correspondence()
            .body_cells(domain)
            .ok_or_else(|| invalid_artifact("Field snapshot Domain has no exact mesh cells"))
    }

    fn realized_field_space(
        &self,
        field: Id<kinds::Field>,
    ) -> Result<(Id<kinds::Domain>, SpaceFamily), Diagnostic> {
        ValidatedMovingSpatialContextV2::realized_field_space(self, field)
    }
}

pub(super) fn support_indices_from_cells(
    mesh: &SimplicialMeshEnvelopeV1,
    cells: &[usize],
    association: DiscreteFieldAssociation,
) -> Result<Vec<usize>, Diagnostic> {
    if association == DiscreteFieldAssociation::Cell {
        return Ok(cells.to_vec());
    }
    let dimension = mesh.dimension();
    let mut vertices = BTreeSet::new();
    for &cell in cells {
        let incidence = mesh
            .mesh()
            .incidence(MeshEntity::new(dimension, cell), 0)
            .ok_or_else(|| invalid_artifact("Field support cell has no vertex incidence"))?;
        vertices.extend(incidence.iter().map(|entry| entry.entity.index()));
    }
    Ok(vertices.into_iter().collect())
}

fn require_matching_shape(
    shape: &ValueShape,
    discrete: DiscreteFieldShape,
) -> Result<(), Diagnostic> {
    let matches = match discrete {
        DiscreteFieldShape::Scalar => shape.is_scalar(),
        DiscreteFieldShape::Vector { components } => shape.extents() == [components].as_slice(),
    };
    if matches {
        Ok(())
    } else {
        Err(invalid_artifact(
            "Field snapshot coefficient shape differs from the Semantic Field shape",
        ))
    }
}

fn require_portable_shape(shape: &ValueShape) -> Result<(), Diagnostic> {
    if shape.rank() <= 1 && shape.component_count().is_some() {
        Ok(())
    } else {
        Err(invalid_artifact(
            "field snapshots admit only scalar and rank-one vector Fields",
        ))
    }
}

fn require_zero_outside_support(
    block: &DiscreteFieldEnvelopeV1,
    active: &[usize],
) -> Result<(), Diagnostic> {
    let active = active.iter().copied().collect::<BTreeSet<_>>();
    let components = block
        .component_shape()
        .component_count()
        .map_err(|error| invalid_artifact(error.message()))?;
    for (entity, values) in block.values().chunks_exact(components).enumerate() {
        if !active.contains(&entity) && values.iter().any(|value| *value != 0.0) {
            return Err(invalid_artifact(
                "Field snapshot has a nonzero coefficient outside its exact Domain support",
            ));
        }
    }
    Ok(())
}

const fn association_order(value: DiscreteFieldAssociation) -> u8 {
    match value {
        DiscreteFieldAssociation::Vertex => 0,
        DiscreteFieldAssociation::Cell => 1,
    }
}

fn parse_id<E: eqiora_core::Entity>(value: &str, label: &str) -> Result<Id<E>, Diagnostic> {
    let parsed = Ulid::from_str(value)
        .map_err(|_| invalid_artifact(format!("{label} ULID is malformed")))?;
    if parsed.to_string() != value {
        return Err(invalid_artifact(format!(
            "{label} ULID is not in canonical spelling"
        )));
    }
    Ok(Id::from_ulid(parsed))
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireFieldSnapshotV2 {
    schema: String,
    encoding: String,
    model_sha256: String,
    semantic_revision: u64,
    realization_sha256: String,
    geometry_sha256: String,
    correspondence_sha256: String,
    mesh_sha256: String,
    field_ulid: String,
    support_domain_ulid: String,
    physical: WirePhysicalType,
    representation: WireRepresentation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WirePhysicalType {
    unit_system: WireUnitSystem,
    dimension: WireDimension,
    value_shape: WireValueShape,
    frame: WireFrame,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireRepresentation {
    scalar: WireScalar,
    ordering: WireOrdering,
    blocks: Vec<WireFieldBlock>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireFieldBlock {
    association: WireAssociation,
    discrete_field_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireUnitSystem {
    CoherentSi,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireScalar {
    F64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireOrdering {
    CanonicalMeshEntityMajor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireAssociation {
    Vertex,
    Cell,
}

impl WireAssociation {
    const fn encode(value: DiscreteFieldAssociation) -> Self {
        match value {
            DiscreteFieldAssociation::Vertex => Self::Vertex,
            DiscreteFieldAssociation::Cell => Self::Cell,
        }
    }

    const fn decode(self) -> DiscreteFieldAssociation {
        match self {
            Self::Vertex => DiscreteFieldAssociation::Vertex,
            Self::Cell => DiscreteFieldAssociation::Cell,
        }
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
            .map_err(|error| invalid_artifact(error.to_string()))
    }
}
