//! Moving spatial states over one immutable reference topology.

use std::collections::BTreeMap;
use std::num::NonZeroU16;
use std::str::FromStr;

use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, Id};
use eqiora_graph::EdgeKind;
use eqiora_meshing::DiscreteFieldAssociation;
use eqiora_realization::{RepresentedPhysicalField, SpaceFamily};
use eqiora_schema::kernel::KernelNode;
use eqiora_sem::KernelProgram;
use serde::{Deserialize, Serialize};
use ulid::Ulid;

use crate::{
    ArtifactDigest, CANONICAL_ENCODING, FieldDecoderLimits, FieldSnapshotEnvelopeV2,
    GeometryIdentityEnvelopeV1, GeometryMeshCorrespondenceEnvelopeV1, ModelArtifactReference,
    RealizationEnvelopeV6, ReplayableCanonicalModelArtifact,
    ReplayableFixedTopologyAleRealizationArtifact, ReplayableFixedTopologyGeometryStateArtifact,
    ReplayedCanonicalModel, SimplicialMeshEnvelopeV1, check_json_limits, invalid_artifact,
};

const SPATIAL_STATE_SCHEMA: &str = "eqiora.spatial-state-envelope/v2";
const MAX_EXACT_F64_INTEGER: u64 = 1_u64 << 53;

/// Replayed common resources for fixed-topology moving spatial observations.
///
/// The context is deliberately borrowed and non-serializable. It validates the
/// exact Model, ALE Realization, reference Geometry Identity, correspondence,
/// immutable mesh, and complete represented-Field inventory once. Current
/// coordinates remain properties of individual
/// [`ReplayableFixedTopologyGeometryStateArtifact`] dependencies rather than
/// mutable context state.
#[derive(Debug)]
pub struct ValidatedMovingSpatialContextV2<
    'a,
    M: ReplayableCanonicalModelArtifact,
    R: ReplayableFixedTopologyAleRealizationArtifact = RealizationEnvelopeV6,
> {
    model: &'a M,
    replayed_model: ReplayedCanonicalModel,
    realization: &'a R,
    geometry: &'a GeometryIdentityEnvelopeV1,
    correspondence: &'a GeometryMeshCorrespondenceEnvelopeV1,
    mesh: &'a SimplicialMeshEnvelopeV1,
    represented_fields: Vec<RepresentedPhysicalField>,
}

impl<'a, M: ReplayableCanonicalModelArtifact, R: ReplayableFixedTopologyAleRealizationArtifact>
    ValidatedMovingSpatialContextV2<'a, M, R>
{
    pub(crate) const fn model(&self) -> &'a M {
        self.model
    }

    /// Replay and cross-validate one exact moving-spatial common context.
    ///
    /// # Errors
    /// Returns `EQ0901` for Model, Realization, reference geometry,
    /// correspondence, mesh, or represented-Field drift.
    pub fn new(
        model: &'a M,
        realization: &'a R,
        geometry: &'a GeometryIdentityEnvelopeV1,
        correspondence: &'a GeometryMeshCorrespondenceEnvelopeV1,
        mesh: &'a SimplicialMeshEnvelopeV1,
    ) -> Result<Self, Diagnostic> {
        let replayed_model = model.replay_model()?;
        realization.validate_ale_model_artifact(model)?;
        realization.validate_ale_mesh_artifact(mesh)?;
        geometry.validate_against(model)?;
        correspondence.validate_against(geometry, model, mesh)?;
        let represented_fields = realization
            .ale_plan()?
            .coupled()
            .represented_physical_fields()
            .map_err(|error| invalid_artifact(error.to_string()))?;
        let driver = realization.ale_requirements()?.solid_displacement();
        if represented_fields
            .iter()
            .filter(|entry| entry.field() == driver)
            .count()
            != 1
        {
            return Err(invalid_artifact(
                "moving-spatial context must represent its exact geometry driver once",
            ));
        }
        Ok(Self {
            model,
            replayed_model,
            realization,
            geometry,
            correspondence,
            mesh,
            represented_fields,
        })
    }

    /// Exact selected Model identity and semantic revision.
    #[must_use]
    pub const fn model_reference(&self) -> &ModelArtifactReference {
        self.replayed_model.artifact_reference()
    }

    /// Completely replayed immutable Semantic Kernel projection.
    #[must_use]
    pub const fn program(&self) -> &KernelProgram {
        self.replayed_model.program()
    }

    /// Exact fixed-topology ALE Realization.
    #[must_use]
    pub const fn realization(&self) -> &'a R {
        self.realization
    }

    pub(crate) fn realization_artifact(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(self.realization.artifact_reference()?.artifact().clone())
    }

    /// Exact reference Geometry Identity.
    #[must_use]
    pub const fn geometry(&self) -> &'a GeometryIdentityEnvelopeV1 {
        self.geometry
    }

    /// Exact reference geometry-to-mesh correspondence.
    #[must_use]
    pub const fn correspondence(&self) -> &'a GeometryMeshCorrespondenceEnvelopeV1 {
        self.correspondence
    }

    /// Exact immutable reference mesh.
    #[must_use]
    pub const fn mesh(&self) -> &'a SimplicialMeshEnvelopeV1 {
        self.mesh
    }

    /// Canonically ordered complete physical Field inventory.
    #[must_use]
    pub fn represented_fields(&self) -> &[RepresentedPhysicalField] {
        &self.represented_fields
    }

    pub(crate) fn validate_snapshot(
        &self,
        snapshot: &FieldSnapshotEnvelopeV2,
    ) -> Result<(), Diagnostic> {
        let expected = self
            .represented_fields
            .iter()
            .find(|entry| entry.field() == snapshot.field())
            .copied()
            .ok_or_else(|| {
                invalid_artifact(
                    "moving spatial state contains a Field outside the ALE Realization inventory",
                )
            })?;
        if snapshot.model_artifact() != *self.model_reference().artifact()
            || snapshot.realization_artifact() != self.realization_artifact()?
            || snapshot.geometry_artifact() != self.geometry.digest()?
            || snapshot.correspondence_artifact() != self.correspondence.digest()?
            || snapshot.mesh_artifact() != self.mesh.digest()?
            || snapshot.support_domain() != expected.domain()
        {
            return Err(invalid_artifact(
                "moving spatial Field snapshot has stale common lineage or Domain support",
            ));
        }

        let definition = match self.program().node(snapshot.field().erase()) {
            Some(KernelNode::Field(definition)) => definition,
            _ => {
                return Err(invalid_artifact(
                    "moving spatial snapshot identity is not a Semantic Field",
                ));
            }
        };
        let supports = self
            .program()
            .edges()
            .iter()
            .filter(|edge| {
                edge.from() == snapshot.field().erase() && edge.kind() == EdgeKind::DefinedOn
            })
            .filter_map(|edge| edge.to().downcast::<kinds::Domain>())
            .collect::<Vec<_>>();
        if supports.as_slice() != [expected.domain()]
            || snapshot.dimension() != definition.dimension()
            || snapshot.value_shape() != *definition.shape()
            || snapshot.frame() != definition.frame()
        {
            return Err(invalid_artifact(
                "moving spatial Field snapshot differs from exact Semantic Field meaning",
            ));
        }

        let expected_associations = match self.realized_field_space(snapshot.field())?.1 {
            SpaceFamily::ContinuousLagrange { order } if order == NonZeroU16::MIN => {
                vec![DiscreteFieldAssociation::Vertex]
            }
            SpaceFamily::SimplexP1Bubble => vec![
                DiscreteFieldAssociation::Vertex,
                DiscreteFieldAssociation::Cell,
            ],
            _ => {
                return Err(invalid_artifact(
                    "moving spatial state v2 admits only P1 and simplex P1-bubble snapshots",
                ));
            }
        };
        let associations = snapshot
            .block_artifacts()
            .into_iter()
            .map(|(association, _)| association)
            .collect::<Vec<_>>();
        if associations != expected_associations {
            return Err(invalid_artifact(
                "moving spatial snapshot blocks differ from the exact ALE Field space",
            ));
        }
        Ok(())
    }

    pub(crate) fn realized_field_space(
        &self,
        field: Id<kinds::Field>,
    ) -> Result<(Id<kinds::Domain>, SpaceFamily), Diagnostic> {
        let plan = self.realization.ale_plan()?;
        for domain in plan.coupled().spatial().domains() {
            if let Some(binding) = domain
                .field_spaces()
                .iter()
                .find(|binding| binding.field() == field)
            {
                return Ok((domain.domain(), binding.space().family()));
            }
        }
        if let Some(eliminated) = plan
            .coupled()
            .time_step()
            .eliminated_states()
            .iter()
            .find(|state| state.pair().state() == field)
        {
            let rate = eliminated.pair().rate();
            let domain = plan
                .coupled()
                .spatial()
                .domains()
                .iter()
                .find(|domain| {
                    domain
                        .field_spaces()
                        .iter()
                        .any(|binding| binding.field() == rate)
                })
                .map(|domain| domain.domain())
                .ok_or_else(|| invalid_artifact("eliminated state rate has no realized Domain"))?;
            return Ok((domain, eliminated.state_space().family()));
        }
        Err(invalid_artifact(
            "moving spatial Field is absent from the ALE Realization spaces",
        ))
    }
}

/// One accepted complete physical state bound to one exact current geometry.
///
/// V2 is a closed moving-geometry wire, not V1 with optional coordinates. Field
/// coefficients remain reference-mesh ordered V1 snapshots; their physical
/// coordinates are supplied exclusively by the required GeometryState.
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialStateEnvelopeV2 {
    wire: WireSpatialStateV2,
}

impl SpatialStateEnvelopeV2 {
    /// Construct one moving state by replaying all immediate dependencies.
    ///
    /// Input snapshot order is normalized by exact Field identity. The
    /// GeometryState driver must be the exact solid-displacement snapshot in
    /// the complete inventory.
    ///
    /// # Errors
    /// Returns `EQ0901` for resource drift, an incomplete/reordered identity
    /// set, wrong step/time, stale GeometryState predecessor, substituted
    /// geometry driver, or snapshot meaning/space drift.
    pub fn new<'a, G: ReplayableFixedTopologyGeometryStateArtifact>(
        context: &ValidatedMovingSpatialContextV2<
            '_,
            impl ReplayableCanonicalModelArtifact,
            impl ReplayableFixedTopologyAleRealizationArtifact,
        >,
        geometry_state: &G,
        predecessor_geometry_state: Option<&G>,
        snapshots: &[FieldSnapshotEnvelopeV2],
        geometry_driver_evidence: G::DriverReplayEvidence<'a>,
    ) -> Result<Self, Diagnostic> {
        let step = geometry_state.step();
        let time_s = geometry_state.time_s();
        validate_coordinate(step, time_s)?;
        let duration = context
            .realization()
            .ale_plan()?
            .coupled()
            .time_step()
            .duration()
            .value();
        if time_s != normalize_zero((step as f64) * duration) {
            return Err(invalid_artifact(
                "moving spatial-state time differs from the exact ALE fixed-step duration",
            ));
        }

        let expected_predecessor = predecessor_geometry_state
            .map(ReplayableFixedTopologyGeometryStateArtifact::geometry_state_digest)
            .transpose()?;
        if geometry_state.model_artifact() != *context.model_reference().artifact()
            || geometry_state.semantic_revision()
                != context.model_reference().semantic_revision().get()
            || geometry_state.realization_artifact() != context.realization_artifact()?
            || geometry_state.reference_geometry_artifact() != context.geometry().digest()?
            || geometry_state.reference_correspondence_artifact()
                != context.correspondence().digest()?
            || geometry_state.reference_mesh_artifact() != context.mesh().digest()?
            || geometry_state.predecessor() != expected_predecessor
        {
            return Err(invalid_artifact(
                "GeometryState differs from the exact moving spatial context or predecessor",
            ));
        }

        if snapshots.len() != context.represented_fields().len() {
            return Err(invalid_artifact(
                "moving spatial state omits or adds a represented physical Field",
            ));
        }
        let mut ordered = snapshots.iter().collect::<Vec<_>>();
        ordered.sort_by_key(|snapshot| snapshot.field().ulid());
        if ordered
            .windows(2)
            .any(|pair| pair[0].field() == pair[1].field())
        {
            return Err(invalid_artifact(
                "moving spatial state contains a duplicate Semantic Field snapshot",
            ));
        }
        let expected = context
            .represented_fields()
            .iter()
            .map(|entry| (entry.field().ulid(), entry.domain()))
            .collect::<BTreeMap<_, _>>();
        for snapshot in &ordered {
            context.validate_snapshot(snapshot)?;
            if expected.get(&snapshot.field().ulid()) != Some(&snapshot.support_domain()) {
                return Err(invalid_artifact(
                    "moving spatial state Field inventory or support differs from its Realization",
                ));
            }
        }
        let driver_field = context
            .realization()
            .ale_requirements()?
            .solid_displacement();
        let driver = ordered
            .iter()
            .copied()
            .find(|snapshot| snapshot.field() == driver_field)
            .ok_or_else(|| invalid_artifact("moving spatial state omits its geometry driver"))?;
        if geometry_state.solid_displacement_snapshot() != driver.digest()? {
            return Err(invalid_artifact(
                "GeometryState is bound to a substituted solid-displacement snapshot",
            ));
        }
        geometry_state.validate_fixed_topology_replay(
            context,
            predecessor_geometry_state,
            driver,
            geometry_driver_evidence,
        )?;

        let value = Self {
            wire: WireSpatialStateV2 {
                schema: SPATIAL_STATE_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                reference: WireReferenceLineageV2 {
                    model_sha256: context.model_reference().artifact().to_string(),
                    semantic_revision: context.model_reference().semantic_revision().get(),
                    realization_sha256: context.realization_artifact()?.to_string(),
                    reference_geometry_sha256: context.geometry().digest()?.to_string(),
                    correspondence_sha256: context.correspondence().digest()?.to_string(),
                    reference_mesh_sha256: context.mesh().digest()?.to_string(),
                },
                accepted: WireAcceptedCoordinateV2 { step, time_s },
                geometry_state: WireGeometryStateReferenceV2 {
                    geometry_state_sha256: geometry_state.geometry_state_digest()?.to_string(),
                    predecessor_geometry_state_sha256: geometry_state
                        .predecessor()
                        .map(|digest| digest.to_string()),
                    driver_snapshot_sha256: driver.digest()?.to_string(),
                },
                fields: ordered
                    .into_iter()
                    .map(|snapshot| {
                        Ok(WireStateFieldV2 {
                            support_domain_ulid: snapshot.support_domain().ulid().to_string(),
                            field_ulid: snapshot.field().ulid().to_string(),
                            snapshot_sha256: snapshot.digest()?.to_string(),
                        })
                    })
                    .collect::<Result<Vec<_>, Diagnostic>>()?,
            },
        };
        value.validate_local(FieldDecoderLimits::default())?;
        Ok(value)
    }

    /// Decode the closed moving-state wire without resolving dependencies.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed, oversized, unknown, or noncanonical data.
    pub fn from_json(bytes: &[u8], limits: FieldDecoderLimits) -> Result<Self, Diagnostic> {
        check_json_limits(bytes, limits.json)?;
        let wire = serde_json::from_slice(bytes)
            .map_err(|error| invalid_artifact(format!("invalid spatial-state/v2 JSON: {error}")))?;
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
            invalid_artifact(format!("cannot serialize spatial state v2: {error}"))
        })
    }

    /// Domain-separated moving-state identity.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            SPATIAL_STATE_SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
    }

    /// Accepted step ordinal.
    #[must_use]
    pub const fn step(&self) -> u64 {
        self.wire.accepted.step
    }

    /// Accepted coherent-SI time in seconds.
    #[must_use]
    pub const fn time_s(&self) -> f64 {
        self.wire.accepted.time_s
    }

    /// Exact current GeometryState artifact.
    #[must_use]
    pub fn geometry_state_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.geometry_state.geometry_state_sha256.clone())
    }

    /// Exact predecessor GeometryState, absent only for the initial state.
    #[must_use]
    pub fn predecessor_geometry_state(&self) -> Option<ArtifactDigest> {
        self.wire
            .geometry_state
            .predecessor_geometry_state_sha256
            .clone()
            .map(ArtifactDigest)
    }

    /// Exact solid-displacement snapshot driving the GeometryState.
    #[must_use]
    pub fn geometry_driver_snapshot(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.geometry_state.driver_snapshot_sha256.clone())
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

    /// Exact geometry-to-mesh correspondence artifact.
    #[must_use]
    pub fn correspondence_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.reference.correspondence_sha256.clone())
    }

    /// Exact immutable reference mesh artifact.
    #[must_use]
    pub fn reference_mesh_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.reference.reference_mesh_sha256.clone())
    }

    /// Canonically ordered `(Domain, Field, snapshot)` references.
    #[must_use]
    pub fn fields(&self) -> Vec<(Id<kinds::Domain>, Id<kinds::Field>, ArtifactDigest)> {
        self.wire
            .fields
            .iter()
            .map(|entry| {
                (
                    parse_id(&entry.support_domain_ulid, "support Domain")
                        .expect("validated moving state Domain ULID"),
                    parse_id(&entry.field_ulid, "Field")
                        .expect("validated moving state Field ULID"),
                    ArtifactDigest(entry.snapshot_sha256.clone()),
                )
            })
            .collect()
    }

    /// Look up one exact Field snapshot without loading unrelated Fields.
    #[must_use]
    pub fn field_snapshot(&self, field: Id<kinds::Field>) -> Option<ArtifactDigest> {
        self.wire
            .fields
            .binary_search_by_key(&field.ulid(), |entry| {
                Ulid::from_str(&entry.field_ulid).expect("validated moving state Field ULID")
            })
            .ok()
            .map(|index| ArtifactDigest(self.wire.fields[index].snapshot_sha256.clone()))
    }

    /// Rebuild and compare this state from exact dependency objects.
    ///
    /// # Errors
    /// Returns `EQ0901` for any substituted, incomplete, stale, or reordered
    /// semantic dependency.
    pub fn validate_against<'a, G: ReplayableFixedTopologyGeometryStateArtifact>(
        &self,
        context: &ValidatedMovingSpatialContextV2<
            '_,
            impl ReplayableCanonicalModelArtifact,
            impl ReplayableFixedTopologyAleRealizationArtifact,
        >,
        geometry_state: &G,
        predecessor_geometry_state: Option<&G>,
        snapshots: &[FieldSnapshotEnvelopeV2],
        geometry_driver_evidence: G::DriverReplayEvidence<'a>,
    ) -> Result<(), Diagnostic> {
        let expected = Self::new(
            context,
            geometry_state,
            predecessor_geometry_state,
            snapshots,
            geometry_driver_evidence,
        )?;
        if self != &expected {
            return Err(invalid_artifact(
                "spatial state v2 differs from exact moving dependency replay",
            ));
        }
        Ok(())
    }

    fn validate_local(&self, limits: FieldDecoderLimits) -> Result<(), Diagnostic> {
        if self.wire.schema != SPATIAL_STATE_SCHEMA || self.wire.encoding != CANONICAL_ENCODING {
            return Err(invalid_artifact(
                "unsupported spatial-state/v2 schema or canonical encoding",
            ));
        }
        for digest in [
            &self.wire.reference.model_sha256,
            &self.wire.reference.realization_sha256,
            &self.wire.reference.reference_geometry_sha256,
            &self.wire.reference.correspondence_sha256,
            &self.wire.reference.reference_mesh_sha256,
            &self.wire.geometry_state.geometry_state_sha256,
            &self.wire.geometry_state.driver_snapshot_sha256,
        ] {
            ArtifactDigest::from_hex(digest.clone())?;
        }
        if let Some(predecessor) = &self.wire.geometry_state.predecessor_geometry_state_sha256 {
            ArtifactDigest::from_hex(predecessor.clone())?;
        }
        validate_coordinate(self.step(), self.time_s())?;
        let initial = self.predecessor_geometry_state().is_none();
        if initial != (self.step() == 0 && self.time_s() == 0.0) {
            return Err(invalid_artifact(
                "spatial-state/v2 initial coordinate and GeometryState predecessor disagree",
            ));
        }
        if self.wire.fields.is_empty() || self.wire.fields.len() > limits.max_spatial_state_fields {
            return Err(invalid_artifact(
                "spatial-state/v2 Field inventory is empty or exceeds the decoder limit",
            ));
        }
        let mut prior = None;
        let mut driver_count = 0_usize;
        for entry in &self.wire.fields {
            parse_id::<kinds::Domain>(&entry.support_domain_ulid, "support Domain")?;
            let field = parse_id::<kinds::Field>(&entry.field_ulid, "Field")?;
            ArtifactDigest::from_hex(entry.snapshot_sha256.clone())?;
            if prior.is_some_and(|prior| prior >= field.ulid()) {
                return Err(invalid_artifact(
                    "spatial-state/v2 Fields must be unique and in canonical identity order",
                ));
            }
            if entry.snapshot_sha256 == self.wire.geometry_state.driver_snapshot_sha256 {
                driver_count += 1;
            }
            prior = Some(field.ulid());
        }
        if driver_count != 1 {
            return Err(invalid_artifact(
                "spatial-state/v2 geometry driver must occur exactly once in the Field inventory",
            ));
        }
        Ok(())
    }
}

fn validate_coordinate(step: u64, time_s: f64) -> Result<(), Diagnostic> {
    if step > MAX_EXACT_F64_INTEGER
        || !time_s.is_finite()
        || time_s < 0.0
        || is_negative_zero(time_s)
    {
        return Err(invalid_artifact(
            "spatial-state/v2 coordinate must be finite, nonnegative, canonical, and exactly representable",
        ));
    }
    Ok(())
}

const fn normalize_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

fn is_negative_zero(value: f64) -> bool {
    value == 0.0 && value.is_sign_negative()
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireSpatialStateV2 {
    schema: String,
    encoding: String,
    reference: WireReferenceLineageV2,
    accepted: WireAcceptedCoordinateV2,
    geometry_state: WireGeometryStateReferenceV2,
    fields: Vec<WireStateFieldV2>,
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

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireAcceptedCoordinateV2 {
    step: u64,
    time_s: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireGeometryStateReferenceV2 {
    geometry_state_sha256: String,
    predecessor_geometry_state_sha256: Option<String>,
    driver_snapshot_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireStateFieldV2 {
    support_domain_ulid: String,
    field_ulid: String,
    snapshot_sha256: String,
}
