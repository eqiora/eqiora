//! Exact Semantic Field values on one generated Cartesian Q1 mesh revision.

use std::num::NonZeroU16;
use std::str::FromStr;

use crate::dimension::WireDimension;

use eqiora_core::ValueFrame;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, Entity, Id, ValueShape};
use eqiora_graph::EdgeKind;
use eqiora_meshing::MeshTopology;
use eqiora_realization::{DiscretizationMethod, MeshPolicy, SpaceFamily};
use eqiora_schema::kernel::KernelNode;
use serde::{Deserialize, Serialize};
use ulid::Ulid;

use crate::{
    ArtifactDigest, CANONICAL_ENCODING, CartesianMeshEnvelopeV1, FieldDecoderLimits,
    GeometryIdentityEnvelopeV1, GeometryMeshCorrespondenceEnvelopeV1, RealizationEnvelope,
    ReplayableCanonicalModelArtifact, invalid_artifact,
};

const CARTESIAN_Q1_FIELD_SNAPSHOT_SCHEMA: &str = "eqiora.cartesian-q1-field-snapshot-envelope/v2";

/// One exact vertex-associated scalar or fixed-vector Field on a generated
/// Cartesian continuous-Lagrange Q1 realization.
///
/// This artifact owns normalized entity-major coefficients and their complete
/// semantic and spatial lineage. It deliberately does not imply stress
/// recovery, interpolation away from vertices, or identity across revisions.
#[derive(Debug, Clone, PartialEq)]
pub struct CartesianQ1FieldSnapshotEnvelopeV2 {
    wire: WireCartesianQ1FieldSnapshotEnvelopeV2,
}

impl CartesianQ1FieldSnapshotEnvelopeV2 {
    /// Bind finite vertex-major values to one exact generated Cartesian Q1
    /// realization. Physical metadata and support are derived from the Model;
    /// callers cannot assert them independently.
    ///
    /// Mathematical zero is normalized to positive zero before identity is
    /// computed.
    ///
    /// # Errors
    /// Returns `EQ0901` for stale lineage, a non-Q1 generated realization,
    /// unsupported Field shape, an inexact mesh, or wrong coefficient count.
    pub fn new(
        model: &impl ReplayableCanonicalModelArtifact,
        realization: &RealizationEnvelope,
        geometry: &GeometryIdentityEnvelopeV1,
        correspondence: &GeometryMeshCorrespondenceEnvelopeV1,
        mesh: &CartesianMeshEnvelopeV1,
        field: Id<kinds::Field>,
        coefficients: impl IntoIterator<Item = f64>,
    ) -> Result<Self, Diagnostic> {
        let replay = model.replay_model()?;
        let reference = replay.artifact_reference();
        realization.validate_model_artifact(model)?;
        geometry.validate_against(model)?;
        correspondence.validate_against_cartesian(geometry, model, mesh)?;

        let program = replay.program();
        let definition = match program.node(field.erase()) {
            Some(KernelNode::Field(definition)) => definition,
            _ => {
                return Err(invalid_artifact(
                    "Cartesian Q1 snapshot identity is not a Field",
                ));
            }
        };
        let supports = program
            .edges()
            .iter()
            .filter(|edge| edge.from() == field.erase() && edge.kind() == EdgeKind::DefinedOn)
            .filter_map(|edge| match program.node(edge.to()) {
                Some(KernelNode::Domain(_)) => edge.to().downcast::<kinds::Domain>(),
                _ => None,
            })
            .collect::<Vec<_>>();
        if supports.len() != 1 {
            return Err(invalid_artifact(
                "Cartesian Q1 snapshot requires one exact Domain support",
            ));
        }
        let support = supports[0];
        let body = geometry
            .bodies()
            .into_iter()
            .find(|body| body.domain() == support)
            .ok_or_else(|| {
                invalid_artifact("Cartesian Q1 snapshot support is absent from the geometry")
            })?;

        validate_realized_mesh(realization, field, support, mesh, body.bounds_m())?;
        let value_shape = WireValueShape::encode(definition.shape())?;
        let component_count = value_shape.component_count()?;
        let vertex_count = mesh
            .mesh()
            .entity_count(0)
            .ok_or_else(|| invalid_artifact("Cartesian mesh omitted its vertex stratum"))?;
        let expected_values = vertex_count
            .checked_mul(component_count)
            .ok_or_else(|| invalid_artifact("Cartesian Q1 coefficient count overflows usize"))?;
        let coefficients = coefficients
            .into_iter()
            .map(|value| if value == 0.0 { 0.0 } else { value })
            .collect::<Vec<_>>();
        if coefficients.len() != expected_values {
            return Err(invalid_artifact(format!(
                "Cartesian Q1 coefficient count {} differs from expected {expected_values}",
                coefficients.len(),
            )));
        }

        let value = Self {
            wire: WireCartesianQ1FieldSnapshotEnvelopeV2 {
                schema: CARTESIAN_Q1_FIELD_SNAPSHOT_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                model_sha256: reference.artifact().to_string(),
                model_ulid: reference.model().ulid().to_string(),
                semantic_revision: reference.semantic_revision().get(),
                realization_sha256: realization.digest()?.to_string(),
                geometry_sha256: geometry.digest()?.to_string(),
                correspondence_sha256: correspondence.digest()?.to_string(),
                mesh_sha256: mesh.digest()?.to_string(),
                field_ulid: field.ulid().to_string(),
                support_domain_ulid: support.ulid().to_string(),
                association: WireAssociation::Vertex,
                space: WireSpace::ContinuousLagrange {
                    order: NonZeroU16::MIN.get(),
                },
                scalar: WireScalar::F64,
                value_shape,
                dimension: WireDimension::encode(definition.dimension()),
                frame: WireFrame::encode(definition.frame()),
                ordering: WireOrdering::EntityMajorComponentLast,
                coefficients,
            },
        };
        value.validate_local(FieldDecoderLimits::default())?;
        Ok(value)
    }

    /// Decode bounded local data. Referenced resources remain untrusted until
    /// [`Self::validate_against`] succeeds.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed, oversized, noncanonical, or unsupported
    /// Cartesian Q1 snapshot data.
    pub fn from_json(bytes: &[u8], limits: FieldDecoderLimits) -> Result<Self, Diagnostic> {
        limits.json.check(bytes)?;
        let wire = serde_json::from_slice(bytes).map_err(|error| {
            invalid_artifact(format!("invalid Cartesian Q1 Field snapshot JSON: {error}"))
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
                "cannot serialize Cartesian Q1 Field snapshot: {error}"
            ))
        })
    }

    /// Domain-separated identity of the complete snapshot.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            CARTESIAN_Q1_FIELD_SNAPSHOT_SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
    }

    /// Exact Semantic Model artifact.
    #[must_use]
    pub fn model_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.model_sha256.clone())
    }

    /// Exact generated Realization artifact.
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

    /// Exact Cartesian mesh revision.
    #[must_use]
    pub fn mesh_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.mesh_sha256.clone())
    }

    /// Exact Semantic Field identity.
    #[must_use]
    pub fn field(&self) -> Id<kinds::Field> {
        parse_id(&self.wire.field_ulid, "Field").expect("validated Field ULID")
    }

    /// Exact Semantic Domain support.
    #[must_use]
    pub fn support_domain(&self) -> Id<kinds::Domain> {
        parse_id(&self.wire.support_domain_ulid, "support Domain")
            .expect("validated support Domain ULID")
    }

    /// Entity-major, component-last normalized coefficients.
    #[must_use]
    pub fn coefficients(&self) -> &[f64] {
        &self.wire.coefficients
    }

    /// Rebuild and compare the complete snapshot from exact dependencies.
    ///
    /// # Errors
    /// Returns `EQ0901` for any semantic, realization, spatial, or coefficient
    /// drift.
    pub fn validate_against(
        &self,
        model: &impl ReplayableCanonicalModelArtifact,
        realization: &RealizationEnvelope,
        geometry: &GeometryIdentityEnvelopeV1,
        correspondence: &GeometryMeshCorrespondenceEnvelopeV1,
        mesh: &CartesianMeshEnvelopeV1,
    ) -> Result<(), Diagnostic> {
        let expected = Self::new(
            model,
            realization,
            geometry,
            correspondence,
            mesh,
            self.field(),
            self.coefficients().iter().copied(),
        )?;
        if self != &expected {
            return Err(invalid_artifact(
                "Cartesian Q1 Field snapshot differs from exact resource replay",
            ));
        }
        Ok(())
    }

    fn validate_local(&self, limits: FieldDecoderLimits) -> Result<(), Diagnostic> {
        let wire = &self.wire;
        if wire.schema != CARTESIAN_Q1_FIELD_SNAPSHOT_SCHEMA
            || wire.encoding != CANONICAL_ENCODING
            || wire.association != WireAssociation::Vertex
            || wire.space
                != (WireSpace::ContinuousLagrange {
                    order: NonZeroU16::MIN.get(),
                })
            || wire.scalar != WireScalar::F64
            || wire.ordering != WireOrdering::EntityMajorComponentLast
        {
            return Err(invalid_artifact(
                "unsupported Cartesian Q1 snapshot schema, encoding, association, space, scalar, or ordering",
            ));
        }
        for digest in [
            &wire.model_sha256,
            &wire.realization_sha256,
            &wire.geometry_sha256,
            &wire.correspondence_sha256,
            &wire.mesh_sha256,
        ] {
            ArtifactDigest::from_hex(digest.clone())?;
        }
        Ulid::from_str(&wire.model_ulid)
            .map_err(|error| invalid_artifact(format!("invalid Model ULID: {error}")))?;
        parse_id::<kinds::Field>(&wire.field_ulid, "Field")?;
        parse_id::<kinds::Domain>(&wire.support_domain_ulid, "support Domain")?;
        let component_count = wire.value_shape.component_count()?;
        let entity_count = wire.coefficients.len() / component_count;
        if component_count > limits.max_discrete_field_components
            || entity_count > limits.max_discrete_field_entities
            || wire.coefficients.len() > limits.max_discrete_field_values
            || !wire.coefficients.len().is_multiple_of(component_count)
        {
            return Err(invalid_artifact(
                "Cartesian Q1 snapshot shape or coefficient count exceeds limits or is inconsistent",
            ));
        }
        if wire
            .coefficients
            .iter()
            .any(|value| !value.is_finite() || (value.to_bits() == (-0.0_f64).to_bits()))
        {
            return Err(invalid_artifact(
                "Cartesian Q1 coefficients must be finite with canonical positive zero",
            ));
        }
        Ok(())
    }
}

fn validate_realized_mesh(
    realization: &RealizationEnvelope,
    field: Id<kinds::Field>,
    support: Id<kinds::Domain>,
    mesh: &CartesianMeshEnvelopeV1,
    bounds: &[(f64, f64)],
) -> Result<(), Diagnostic> {
    let graph = realization.graph();
    let selected = graph
        .fields()
        .iter()
        .find(|node| node.field() == field)
        .ok_or_else(|| {
            invalid_artifact("Cartesian snapshot Field is absent from the Realization")
        })?;
    let domain = graph
        .domain(selected.domain())
        .ok_or_else(|| invalid_artifact("Cartesian snapshot Domain is absent"))?;
    if domain.domain() != support
        || bounds.len() != mesh.dimension()
        || selected.space().family()
            != (SpaceFamily::ContinuousLagrange {
                order: NonZeroU16::MIN,
            })
        || domain.discretization().method() != DiscretizationMethod::ContinuousGalerkin
        || domain.coordinates() != eqiora_realization::CoordinateTreatment::Physical
        || domain.configuration() != eqiora_realization::DomainConfiguration::FixedGeometry
    {
        return Err(invalid_artifact(
            "Cartesian Q1 snapshot requires a matching continuous-Galerkin Q1 realization",
        ));
    }
    let MeshPolicy::GeneratedUniform { cells_per_axis } = domain.discretization().mesh() else {
        return Err(invalid_artifact(
            "Cartesian Q1 snapshot requires a generated-uniform mesh policy",
        ));
    };
    let cells_per_axis = cells_per_axis.get();
    if bounds.len() != mesh.dimension()
        || (0..mesh.dimension()).any(|axis| {
            mesh.mesh().axis_cell_count(axis) != Some(cells_per_axis)
                || !axis_is_exact_generated_uniform(
                    mesh.mesh().axis_coordinates(axis),
                    bounds[axis],
                    cells_per_axis,
                )
        })
    {
        return Err(invalid_artifact(
            "Cartesian mesh differs from exact generated-uniform realization replay",
        ));
    }
    Ok(())
}

fn axis_is_exact_generated_uniform(
    coordinates: Option<&[f64]>,
    (lower, upper): (f64, f64),
    cells: usize,
) -> bool {
    let Some(coordinates) = coordinates else {
        return false;
    };
    let spacing = (upper - lower) / cells as f64;
    spacing.is_finite()
        && spacing > 0.0
        && lower + spacing > lower
        && upper - spacing < upper
        && coordinates.iter().enumerate().all(|(index, &actual)| {
            let expected = if index == cells {
                upper
            } else {
                lower + index as f64 * spacing
            };
            actual.to_bits() == expected.to_bits()
        })
}

fn parse_id<K: Entity>(value: &str, label: &str) -> Result<Id<K>, Diagnostic> {
    Ulid::from_str(value)
        .map(Id::from_ulid)
        .map_err(|error| invalid_artifact(format!("invalid {label} ULID: {error}")))
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireCartesianQ1FieldSnapshotEnvelopeV2 {
    schema: String,
    encoding: String,
    model_sha256: String,
    model_ulid: String,
    semantic_revision: u64,
    realization_sha256: String,
    geometry_sha256: String,
    correspondence_sha256: String,
    mesh_sha256: String,
    field_ulid: String,
    support_domain_ulid: String,
    association: WireAssociation,
    space: WireSpace,
    scalar: WireScalar,
    value_shape: WireValueShape,
    dimension: WireDimension,
    frame: WireFrame,
    ordering: WireOrdering,
    coefficients: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireAssociation {
    Vertex,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "family", rename_all = "kebab-case", deny_unknown_fields)]
enum WireSpace {
    ContinuousLagrange { order: u16 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireScalar {
    F64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
struct WireValueShape(Vec<u32>);

impl WireValueShape {
    fn encode(shape: &ValueShape) -> Result<Self, Diagnostic> {
        let wire = Self(shape.extents().iter().map(|extent| extent.get()).collect());
        wire.component_count()?;
        Ok(wire)
    }

    fn component_count(&self) -> Result<usize, Diagnostic> {
        if self.0.len() > 1 || self.0.contains(&0) {
            return Err(invalid_artifact(
                "Cartesian Q1 snapshot admits only scalar or fixed-vector values",
            ));
        }
        self.0.iter().try_fold(1_usize, |count, &extent| {
            count
                .checked_mul(usize::try_from(extent).map_err(|_| {
                    invalid_artifact("Cartesian Q1 value extent exceeds local usize")
                })?)
                .ok_or_else(|| invalid_artifact("Cartesian Q1 component count overflows usize"))
        })
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireOrdering {
    EntityMajorComponentLast,
}
