//! Portable, mesh-bound discrete field values.

use std::num::NonZeroU32;

use eqiora_core::Diagnostic;
use eqiora_meshing::{
    DiscreteFieldAssociation, DiscreteFieldPayload, DiscreteFieldShape, MeshTopology,
};
use serde::{Deserialize, Serialize};

use crate::{ArtifactDigest, CANONICAL_ENCODING, SimplicialMeshEnvelopeV1, invalid_artifact};

const DISCRETE_FIELD_SCHEMA: &str = "eqiora.discrete-field-envelope/v1";

/// Semantic work budgets shared by discrete Field, snapshot, storage, and state artifacts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldDecoderLimits {
    /// Common JSON syntax admission.
    pub json: crate::JsonLimits,
    /// Maximum associated entities in one discrete Field.
    pub max_discrete_field_entities: usize,
    /// Maximum components per entity in one discrete Field.
    pub max_discrete_field_components: usize,
    /// Maximum scalar values in one discrete Field.
    pub max_discrete_field_values: usize,
    /// Maximum coefficient blocks in one logical Field snapshot.
    pub max_field_snapshot_blocks: usize,
    /// Maximum raw canonical-byte chunks in one Field storage manifest.
    pub max_field_storage_chunks: usize,
    /// Maximum exact Field references in one accepted spatial state.
    pub max_spatial_state_fields: usize,
}

impl Default for FieldDecoderLimits {
    fn default() -> Self {
        Self {
            json: crate::JsonLimits::default(),
            max_discrete_field_entities: 2_000_000,
            max_discrete_field_components: 64,
            max_discrete_field_values: 16_000_000,
            max_field_snapshot_blocks: 8,
            max_field_storage_chunks: 1_000_000,
            max_spatial_state_fields: 100_000,
        }
    }
}

/// Versioned, affine-simplex-mesh-bound discrete field content.
///
/// The envelope contains only numerical content identity: exact mesh digest,
/// association, component shape, entity count, and entity-major values. Source
/// names, file locations, storage layout, units, and Semantic Model bindings
/// belong to separate typed contracts.
///
/// Decoding validates the closed wire grammar and resource limits, but does
/// not make an independently loaded mesh appear by implication. Call
/// [`Self::validate_mesh_artifact`] to obtain the accepted in-memory payload.
#[derive(Debug, Clone, PartialEq)]
pub struct DiscreteFieldEnvelopeV1 {
    wire: WireDiscreteFieldEnvelopeV1,
    shape: DiscreteFieldShape,
}

impl DiscreteFieldEnvelopeV1 {
    /// Bind an already checked payload to one exact affine-simplex mesh.
    ///
    /// The association count is independently rechecked against `mesh` before
    /// bytes are produced.
    ///
    /// # Errors
    /// Returns `EQ0901` for a mesh-linkage mismatch or a value/count that is
    /// not representable by the portable v1 wire contract.
    pub fn from_payload(
        mesh: &SimplicialMeshEnvelopeV1,
        payload: &DiscreteFieldPayload,
    ) -> Result<Self, Diagnostic> {
        recheck_payload(mesh, payload)?;
        let entity_count = u64::try_from(payload.entity_count()).map_err(|_| {
            invalid_artifact("discrete field entity count exceeds portable wire u64")
        })?;
        Ok(Self {
            wire: WireDiscreteFieldEnvelopeV1 {
                schema: DISCRETE_FIELD_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                mesh_sha256: mesh.digest()?.to_string(),
                association: WireAssociation::encode(payload.association()),
                component_shape: WireComponentShape::from(payload.component_shape()),
                entity_count,
                values: payload.values().to_vec(),
            },
            shape: payload.component_shape(),
        })
    }

    /// Decode with byte, nesting, entity, component, and scalar-value limits.
    ///
    /// This operation performs no filesystem, network, or mesh lookup.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed/unknown wire data, resource excess,
    /// invalid digest/count/shape/value data, or a non-canonical negative zero.
    pub fn from_json(bytes: &[u8], limits: FieldDecoderLimits) -> Result<Self, Diagnostic> {
        limits.json.check(bytes)?;
        let wire = serde_json::from_slice(bytes)
            .map_err(|error| invalid_artifact(format!("invalid discrete field JSON: {error}")))?;
        Self::from_wire(wire, limits)
    }

    /// Deterministic canonical JSON bytes.
    ///
    /// # Errors
    /// Returns `EQ0901` if serialization unexpectedly fails.
    pub fn canonical_json(&self) -> Result<Vec<u8>, Diagnostic> {
        serde_json::to_vec(&self.wire)
            .map_err(|error| invalid_artifact(format!("cannot serialize discrete field: {error}")))
    }

    /// Domain-separated SHA-256 identity of the complete mesh-bound field.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            DISCRETE_FIELD_SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
    }

    /// Exact affine-simplex mesh content identity named by this envelope.
    #[must_use]
    pub fn mesh_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.mesh_sha256.clone())
    }

    /// Mesh entity stratum carrying the values.
    #[must_use]
    pub const fn association(&self) -> DiscreteFieldAssociation {
        self.wire.association.decode()
    }

    /// Component shape retained in field identity.
    #[must_use]
    pub const fn component_shape(&self) -> DiscreteFieldShape {
        self.shape
    }

    /// Portable entity count after checked conversion to local `usize`.
    ///
    /// # Errors
    /// Returns `EQ0901` when the stored `u64` is not representable locally.
    pub fn entity_count(&self) -> Result<usize, Diagnostic> {
        usize::try_from(self.wire.entity_count)
            .map_err(|_| invalid_artifact("discrete field entity count exceeds local usize"))
    }

    /// Canonical entity-major scalar values.
    ///
    /// These values remain bound to [`Self::mesh_artifact`],
    /// [`Self::association`], and [`Self::component_shape`]. The projection is
    /// provided so higher-level typed artifacts can validate spatial support
    /// without inventing a second array representation.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.wire.values
    }

    /// Validate exact mesh linkage and reconstruct the accepted L2 payload.
    ///
    /// Equal entity counts are insufficient: the supplied mesh envelope must
    /// have the exact content digest named by this field.
    ///
    /// # Errors
    /// Returns `EQ0901` for a different mesh identity, association/count
    /// mismatch, or any payload invariant failure.
    pub fn validate_mesh_artifact(
        &self,
        mesh: &SimplicialMeshEnvelopeV1,
    ) -> Result<DiscreteFieldPayload, Diagnostic> {
        if mesh.digest()? != self.mesh_artifact() {
            return Err(invalid_artifact(
                "discrete field references a different simplex mesh artifact",
            ));
        }
        let payload = DiscreteFieldPayload::new(
            mesh.mesh(),
            self.association(),
            self.component_shape(),
            self.wire.values.clone(),
        )
        .map_err(|error| invalid_artifact(error.message()))?;
        if payload.entity_count() != self.entity_count()? {
            return Err(invalid_artifact(
                "discrete field entity count differs from its referenced mesh stratum",
            ));
        }
        Ok(payload)
    }

    fn from_wire(
        wire: WireDiscreteFieldEnvelopeV1,
        limits: FieldDecoderLimits,
    ) -> Result<Self, Diagnostic> {
        if wire.schema != DISCRETE_FIELD_SCHEMA || wire.encoding != CANONICAL_ENCODING {
            return Err(invalid_artifact(
                "unsupported discrete-field schema or canonical encoding",
            ));
        }
        ArtifactDigest::from_hex(wire.mesh_sha256.clone())?;
        let entity_count = usize::try_from(wire.entity_count)
            .map_err(|_| invalid_artifact("discrete field entity count exceeds local usize"))?;
        if entity_count == 0 {
            return Err(invalid_artifact(
                "discrete field entity count must be positive",
            ));
        }
        require_count(
            "discrete field entities",
            entity_count,
            limits.max_discrete_field_entities,
        )?;
        let shape = wire.component_shape.into_public();
        let component_count = shape
            .component_count()
            .map_err(|error| invalid_artifact(error.message()))?;
        require_count(
            "discrete field components",
            component_count,
            limits.max_discrete_field_components,
        )?;
        let required_values = entity_count.checked_mul(component_count).ok_or_else(|| {
            invalid_artifact("discrete field entity/component product overflows usize")
        })?;
        require_count(
            "discrete field scalar values",
            required_values,
            limits.max_discrete_field_values,
        )?;
        if wire.values.len() != required_values {
            return Err(invalid_artifact(format!(
                "discrete field requires {required_values} scalar values, received {}",
                wire.values.len(),
            )));
        }
        if wire.values.iter().any(|value| !value.is_finite()) {
            return Err(invalid_artifact("discrete field values must all be finite"));
        }
        if wire
            .values
            .iter()
            .any(|value| *value == 0.0 && value.is_sign_negative())
        {
            return Err(invalid_artifact(
                "discrete field wire values must use canonical positive zero",
            ));
        }
        Ok(Self { wire, shape })
    }
}

fn recheck_payload(
    mesh: &SimplicialMeshEnvelopeV1,
    payload: &DiscreteFieldPayload,
) -> Result<(), Diagnostic> {
    let selected_dimension = match payload.association() {
        DiscreteFieldAssociation::Vertex => 0,
        DiscreteFieldAssociation::Cell => mesh.mesh().topological_dimension(),
    };
    let expected_count = mesh
        .mesh()
        .entity_count(selected_dimension)
        .ok_or_else(|| invalid_artifact("discrete field selects a missing mesh stratum"))?;
    if expected_count != payload.entity_count() {
        return Err(invalid_artifact(format!(
            "discrete field has {} entities but the selected mesh stratum has {expected_count}",
            payload.entity_count(),
        )));
    }
    Ok(())
}

fn require_count(label: &str, actual: usize, limit: usize) -> Result<(), Diagnostic> {
    if actual > limit {
        Err(invalid_artifact(format!(
            "{label} count {actual} exceeds decoder limit {limit}",
        )))
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireDiscreteFieldEnvelopeV1 {
    schema: String,
    encoding: String,
    mesh_sha256: String,
    association: WireAssociation,
    component_shape: WireComponentShape,
    entity_count: u64,
    values: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum WireAssociation {
    Vertex,
    Cell,
}

impl WireAssociation {
    pub(crate) const fn encode(value: DiscreteFieldAssociation) -> Self {
        match value {
            DiscreteFieldAssociation::Vertex => Self::Vertex,
            DiscreteFieldAssociation::Cell => Self::Cell,
        }
    }

    pub(crate) const fn decode(self) -> DiscreteFieldAssociation {
        match self {
            Self::Vertex => DiscreteFieldAssociation::Vertex,
            Self::Cell => DiscreteFieldAssociation::Cell,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
enum WireComponentShape {
    Scalar(WireScalarShape),
    Vector(WireVectorShape),
}

impl WireComponentShape {
    const fn into_public(self) -> DiscreteFieldShape {
        match self {
            Self::Scalar(WireScalarShape {
                kind: WireScalarKind::Scalar,
            }) => DiscreteFieldShape::Scalar,
            Self::Vector(WireVectorShape {
                kind: WireVectorKind::Vector,
                components,
            }) => DiscreteFieldShape::Vector { components },
        }
    }
}

impl From<DiscreteFieldShape> for WireComponentShape {
    fn from(value: DiscreteFieldShape) -> Self {
        match value {
            DiscreteFieldShape::Scalar => Self::Scalar(WireScalarShape {
                kind: WireScalarKind::Scalar,
            }),
            DiscreteFieldShape::Vector { components } => Self::Vector(WireVectorShape {
                kind: WireVectorKind::Vector,
                components,
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireScalarKind {
    Scalar,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireVectorKind {
    Vector,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireScalarShape {
    kind: WireScalarKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireVectorShape {
    kind: WireVectorKind,
    components: NonZeroU32,
}

#[cfg(test)]
mod tests {
    use eqiora_core::diagnostic::codes;
    use eqiora_meshing::{MeshQualityGate, SimplicialMesh};

    use super::*;

    fn mesh(vertices: Vec<Vec<f64>>, cells: Vec<Vec<usize>>) -> SimplicialMeshEnvelopeV1 {
        SimplicialMeshEnvelopeV1::from_mesh(
            &SimplicialMesh::new(2, vertices, cells, MeshQualityGate::new(0.2).unwrap()).unwrap(),
        )
        .unwrap()
    }

    fn fixture_mesh() -> SimplicialMeshEnvelopeV1 {
        mesh(
            vec![
                vec![0.0, 0.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
                vec![0.0, 1.0],
            ],
            vec![vec![0, 1, 2], vec![0, 2, 3]],
        )
    }

    fn other_equal_count_mesh() -> SimplicialMeshEnvelopeV1 {
        mesh(
            vec![
                vec![0.0, 0.0],
                vec![2.0, 0.0],
                vec![2.0, 2.0],
                vec![0.0, 2.0],
            ],
            vec![vec![0, 1, 2], vec![0, 2, 3]],
        )
    }

    #[test]
    fn round_trip_preserves_mesh_bound_identity_and_reconstructs_payload() {
        let mesh = fixture_mesh();
        let payload = DiscreteFieldPayload::new(
            mesh.mesh(),
            DiscreteFieldAssociation::Vertex,
            DiscreteFieldShape::Scalar,
            vec![1.0, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let envelope = DiscreteFieldEnvelopeV1::from_payload(&mesh, &payload).unwrap();
        let bytes = envelope.canonical_json().unwrap();
        let decoded =
            DiscreteFieldEnvelopeV1::from_json(&bytes, FieldDecoderLimits::default()).unwrap();

        assert_eq!(decoded.canonical_json().unwrap(), bytes);
        assert_eq!(decoded.digest().unwrap(), envelope.digest().unwrap());
        assert_eq!(decoded.mesh_artifact(), mesh.digest().unwrap());
        assert_eq!(decoded.validate_mesh_artifact(&mesh).unwrap(), payload);
    }

    #[test]
    fn source_zero_sign_normalizes_but_negative_zero_wire_fails() {
        let mesh = fixture_mesh();
        let payload = DiscreteFieldPayload::new(
            mesh.mesh(),
            DiscreteFieldAssociation::Cell,
            DiscreteFieldShape::Scalar,
            vec![-0.0, 0.0],
        )
        .unwrap();
        let positive = DiscreteFieldEnvelopeV1::from_payload(&mesh, &payload).unwrap();
        assert_eq!(
            positive.validate_mesh_artifact(&mesh).unwrap().values(),
            &[0.0, 0.0]
        );

        let negative = positive
            .canonical_json()
            .unwrap()
            .windows(b"0.0".len())
            .position(|window| window == b"0.0")
            .map(|index| {
                let mut bytes = positive.canonical_json().unwrap();
                bytes.splice(index..index + 3, b"-0.0".iter().copied());
                bytes
            })
            .unwrap();
        let error = DiscreteFieldEnvelopeV1::from_json(&negative, FieldDecoderLimits::default())
            .unwrap_err();
        assert_eq!(error.code(), codes::INVALID_ARTIFACT);
        assert!(error.message().contains("positive zero"));
    }

    #[test]
    fn equal_count_wrong_mesh_and_count_forgery_fail_linkage() {
        let mesh = fixture_mesh();
        let payload = DiscreteFieldPayload::new(
            mesh.mesh(),
            DiscreteFieldAssociation::Cell,
            DiscreteFieldShape::Vector {
                components: NonZeroU32::new(2).unwrap(),
            },
            vec![1.0, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let envelope = DiscreteFieldEnvelopeV1::from_payload(&mesh, &payload).unwrap();
        assert!(
            envelope
                .validate_mesh_artifact(&other_equal_count_mesh())
                .unwrap_err()
                .message()
                .contains("different simplex mesh")
        );

        let mut forged: serde_json::Value =
            serde_json::from_slice(&envelope.canonical_json().unwrap()).unwrap();
        forged["entity_count"] = serde_json::json!(4);
        assert_eq!(
            DiscreteFieldEnvelopeV1::from_json(
                &serde_json::to_vec(&forged).unwrap(),
                FieldDecoderLimits::default(),
            )
            .unwrap_err()
            .code(),
            codes::INVALID_ARTIFACT,
        );
    }

    #[test]
    fn scalar_and_one_component_vector_have_distinct_identity() {
        let mesh = fixture_mesh();
        let scalar = DiscreteFieldPayload::new(
            mesh.mesh(),
            DiscreteFieldAssociation::Cell,
            DiscreteFieldShape::Scalar,
            vec![1.0, 2.0],
        )
        .unwrap();
        let vector = DiscreteFieldPayload::new(
            mesh.mesh(),
            DiscreteFieldAssociation::Cell,
            DiscreteFieldShape::Vector {
                components: NonZeroU32::MIN,
            },
            vec![1.0, 2.0],
        )
        .unwrap();
        let scalar = DiscreteFieldEnvelopeV1::from_payload(&mesh, &scalar).unwrap();
        let vector = DiscreteFieldEnvelopeV1::from_payload(&mesh, &vector).unwrap();
        assert_ne!(
            scalar.canonical_json().unwrap(),
            vector.canonical_json().unwrap()
        );
        assert_ne!(scalar.digest().unwrap(), vector.digest().unwrap());
    }

    #[test]
    fn limits_unknown_fields_and_nonfinite_values_fail_closed() {
        let mesh = fixture_mesh();
        let payload = DiscreteFieldPayload::new(
            mesh.mesh(),
            DiscreteFieldAssociation::Vertex,
            DiscreteFieldShape::Scalar,
            vec![1.0, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let envelope = DiscreteFieldEnvelopeV1::from_payload(&mesh, &payload).unwrap();
        let bytes = envelope.canonical_json().unwrap();
        let limits = FieldDecoderLimits {
            max_discrete_field_values: 3,
            ..FieldDecoderLimits::default()
        };
        assert_eq!(
            DiscreteFieldEnvelopeV1::from_json(&bytes, limits)
                .unwrap_err()
                .code(),
            codes::INVALID_ARTIFACT,
        );

        let mut unknown: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        unknown["component_shape"]["basis"] = serde_json::json!("cartesian");
        assert_eq!(
            DiscreteFieldEnvelopeV1::from_json(
                &serde_json::to_vec(&unknown).unwrap(),
                FieldDecoderLimits::default(),
            )
            .unwrap_err()
            .code(),
            codes::INVALID_ARTIFACT,
        );

        let nonfinite = bytes
            .windows(b"1.0".len())
            .position(|window| window == b"1.0")
            .map(|index| {
                let mut bytes = bytes.clone();
                bytes.splice(index..index + 3, b"1e999".iter().copied());
                bytes
            })
            .unwrap();
        assert_eq!(
            DiscreteFieldEnvelopeV1::from_json(&nonfinite, FieldDecoderLimits::default())
                .unwrap_err()
                .code(),
            codes::INVALID_ARTIFACT,
        );
    }
}
