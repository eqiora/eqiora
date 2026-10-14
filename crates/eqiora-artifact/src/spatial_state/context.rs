//! Moving spatial states over one immutable reference topology.

use serde::{Deserialize, Serialize};
use std::num::NonZeroU16;

use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, Id};
use eqiora_graph::EdgeKind;
use eqiora_meshing::DiscreteFieldAssociation;
use eqiora_realization::{RepresentedPhysicalField, SpaceFamily};
use eqiora_schema::kernel::KernelNode;
use eqiora_sem::KernelProgram;

use crate::{
    ArtifactDigest, CanonicalRealizationArtifact, FieldSnapshotEnvelopeV2,
    GeometryIdentityEnvelopeV1, GeometryMeshCorrespondenceEnvelopeV1, ModelArtifactReference,
    RealizationEnvelope, ReplayableCanonicalModelArtifact, ReplayedCanonicalModel,
    SimplicialMeshEnvelopeV1, invalid_artifact,
};

/// Replayed common resources for fixed-topology moving spatial observations.
///
/// The context is deliberately borrowed and non-serializable. It validates the
/// exact Model, ALE Realization, reference Geometry Identity, correspondence,
/// immutable mesh, and complete represented-Field inventory once. Current
/// coordinates remain properties of individual
/// [`crate::GeometryStateEnvelope`] dependencies rather than
/// mutable context state.
#[derive(Debug)]
pub struct ValidatedMovingSpatialContext<'a, M: ReplayableCanonicalModelArtifact> {
    model: &'a M,
    replayed_model: ReplayedCanonicalModel,
    realization: &'a RealizationEnvelope,
    geometry: &'a GeometryIdentityEnvelopeV1,
    correspondence: &'a GeometryMeshCorrespondenceEnvelopeV1,
    mesh: &'a SimplicialMeshEnvelopeV1,
    represented_fields: Vec<RepresentedPhysicalField>,
    driver: Id<kinds::Field>,
    duration: f64,
}

impl<'a, M: ReplayableCanonicalModelArtifact> ValidatedMovingSpatialContext<'a, M> {
    pub(crate) const fn duration(&self) -> f64 {
        self.duration
    }

    pub(crate) const fn driver(&self) -> Id<kinds::Field> {
        self.driver
    }

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
        realization: &'a RealizationEnvelope,
        geometry: &'a GeometryIdentityEnvelopeV1,
        correspondence: &'a GeometryMeshCorrespondenceEnvelopeV1,
        mesh: &'a SimplicialMeshEnvelopeV1,
    ) -> Result<Self, Diagnostic> {
        let replayed_model = model.replay_model()?;
        realization.validate_model_artifact(model)?;
        realization.validate_mesh_artifact(model, mesh)?;
        geometry.validate_against(model)?;
        correspondence.validate_against(geometry, model, mesh)?;
        let graph = realization.graph();
        let represented_fields = graph.represented_physical_fields();
        let [
            eqiora_realization::GeometryActionNode::P1HarmonicExtension {
                drivers, duration, ..
            },
        ] = graph.geometry_actions()
        else {
            return Err(invalid_artifact(
                "moving spatial context requires one exact harmonic geometry action",
            ));
        };
        let [driver] = drivers.as_slice() else {
            return Err(invalid_artifact(
                "moving spatial context requires one exact geometry driver",
            ));
        };
        let driver = graph
            .field(*driver)
            .ok_or_else(|| invalid_artifact("geometry driver is absent from the graph"))?
            .field();
        let duration = duration.value();
        for domain in graph.domains() {
            if realization.mesh_artifact(domain.domain())? != Some(mesh.digest()?) {
                return Err(invalid_artifact(
                    "moving context requires one shared exact reference mesh",
                ));
            }
        }
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
            driver,
            duration,
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
    pub const fn realization(&self) -> &'a RealizationEnvelope {
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
                    "moving spatial state admits only P1 and simplex P1-bubble snapshots",
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
        let graph = self.realization.graph();
        let node = graph
            .fields()
            .iter()
            .find(|node| node.field() == field)
            .ok_or_else(|| {
                invalid_artifact("moving spatial Field is absent from the Realization graph")
            })?;
        let domain = graph
            .domain(node.domain())
            .ok_or_else(|| invalid_artifact("represented Field Domain is absent from the graph"))?;
        Ok((domain.domain(), node.space().family()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct MovingSpatialReference {
    pub model_sha256: String,
    pub semantic_revision: u64,
    pub geometry_sha256: String,
    pub correspondence_sha256: String,
    pub mesh_sha256: String,
    pub realization_sha256: String,
}

impl MovingSpatialReference {
    pub fn from_context<M: ReplayableCanonicalModelArtifact>(
        context: &ValidatedMovingSpatialContext<'_, M>,
    ) -> Result<Self, Diagnostic> {
        Ok(Self {
            model_sha256: context.model_reference().artifact().to_string(),
            semantic_revision: context.model_reference().semantic_revision().get(),
            geometry_sha256: context.geometry().digest()?.to_string(),
            correspondence_sha256: context.correspondence().digest()?.to_string(),
            mesh_sha256: context.mesh().digest()?.to_string(),
            realization_sha256: context.realization().digest()?.to_string(),
        })
    }
}

impl MovingSpatialReference {
    pub(crate) fn validate(&self) -> Result<(), Diagnostic> {
        for digest in [
            &self.model_sha256,
            &self.geometry_sha256,
            &self.correspondence_sha256,
            &self.mesh_sha256,
            &self.realization_sha256,
        ] {
            ArtifactDigest::from_hex(digest.clone())?;
        }
        Ok(())
    }
}
