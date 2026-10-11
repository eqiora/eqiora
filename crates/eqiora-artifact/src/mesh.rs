//! Portable, content-addressed affine-simplex mesh revisions.

use eqiora_core::Diagnostic;
use eqiora_meshing::{MeshQualityGate, MeshTopology, SimplicialMesh};
use eqiora_realization::MeshArtifactReference;
use serde::{Deserialize, Serialize};

use crate::{ArtifactDigest, CANONICAL_ENCODING, check_json_limits, invalid_artifact};

const SIMPLICIAL_MESH_SCHEMA: &str = "eqiora.simplicial-mesh-envelope/v1";

/// Semantic work budgets for the simplicial Mesh artifact family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeshDecoderLimits {
    /// Common JSON syntax admission.
    pub json: crate::JsonDecoderLimits,
    /// Maximum vertices in one Mesh artifact.
    pub max_mesh_vertices: usize,
    /// Maximum top-dimensional cells in one Mesh artifact.
    pub max_mesh_cells: usize,
    /// Maximum coordinate scalars summed across one Mesh artifact.
    pub max_mesh_coordinate_values: usize,
    /// Maximum connectivity indices summed across one Mesh artifact.
    pub max_mesh_connectivity_indices: usize,
}

impl Default for MeshDecoderLimits {
    fn default() -> Self {
        Self {
            json: crate::JsonDecoderLimits::default(),
            max_mesh_vertices: 1_000_000,
            max_mesh_cells: 2_000_000,
            max_mesh_coordinate_values: 4_000_000,
            max_mesh_connectivity_indices: 8_000_000,
        }
    }
}

/// Versioned coordinates, connectivity, acceptance policy, and recomputed
/// quality evidence for one fixed-topology affine-simplex mesh revision.
///
/// The envelope is a Realization artifact, not Semantic Model meaning. It has
/// no filesystem path, importer identity, physics field, partition, or solver
/// policy. Decoding reconstructs the same validated [`SimplicialMesh`] used by
/// numerical assembly; wire data never bypasses mesh invariants.
#[derive(Debug, Clone, PartialEq)]
pub struct SimplicialMeshEnvelopeV1 {
    wire: WireSimplicialMeshEnvelopeV1,
    mesh: SimplicialMesh,
}

impl SimplicialMeshEnvelopeV1 {
    /// Capture one already accepted affine-simplex mesh revision.
    ///
    /// # Errors
    /// Returns `EQ0901` if platform-sized connectivity cannot be represented
    /// by the portable `u64` wire contract.
    pub fn from_mesh(mesh: &SimplicialMesh) -> Result<Self, Diagnostic> {
        let cells = mesh
            .cells()
            .iter()
            .map(|cell| {
                cell.iter()
                    .map(|&vertex| {
                        u64::try_from(vertex).map_err(|_| {
                            invalid_artifact("mesh vertex index exceeds portable wire u64")
                        })
                    })
                    .collect()
            })
            .collect::<Result<Vec<Vec<u64>>, Diagnostic>>()?;
        let report = mesh.quality_report();
        Ok(Self {
            wire: WireSimplicialMeshEnvelopeV1 {
                schema: SIMPLICIAL_MESH_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                topology: WireTopology {
                    dimension: u64::try_from(mesh.topological_dimension()).map_err(|_| {
                        invalid_artifact("mesh topological dimension exceeds portable wire u64")
                    })?,
                    cell_family: WireCellFamily::Simplex,
                },
                geometry: WireGeometry {
                    coordinate_scalar: WireCoordinateScalar::F64,
                    mapping: WireGeometryMapping::Affine,
                },
                vertices: mesh.vertices().to_vec(),
                cells,
                acceptance: WireAcceptance {
                    minimum_mean_ratio: mesh.quality_gate().minimum_mean_ratio(),
                },
                evidence: WireQualityEvidence {
                    minimum_mean_ratio: report.minimum_mean_ratio(),
                    minimum_signed_measure_scale: report.minimum_signed_measure_scale(),
                },
            },
            mesh: mesh.clone(),
        })
    }

    /// Decode with byte, nesting, and mesh-specific resource limits.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed/unknown wire data, resource excess,
    /// invalid mesh topology or geometry, or quality evidence that does not
    /// exactly match a fresh reconstruction.
    pub fn from_json(bytes: &[u8], limits: MeshDecoderLimits) -> Result<Self, Diagnostic> {
        check_json_limits(bytes, limits.json)?;
        let wire: WireSimplicialMeshEnvelopeV1 = serde_json::from_slice(bytes)
            .map_err(|error| invalid_artifact(format!("invalid simplex mesh JSON: {error}")))?;
        let mesh = reconstruct(&wire, limits)?;
        Ok(Self { wire, mesh })
    }

    /// Deterministic canonical JSON bytes.
    ///
    /// # Errors
    /// Returns `EQ0901` if serialization unexpectedly fails.
    pub fn canonical_json(&self) -> Result<Vec<u8>, Diagnostic> {
        serde_json::to_vec(&self.wire)
            .map_err(|error| invalid_artifact(format!("cannot serialize simplex mesh: {error}")))
    }

    /// Domain-separated SHA-256 identity of the complete mesh revision.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            SIMPLICIAL_MESH_SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
    }

    /// Realization-layer reference to this exact content identity.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization fails.
    pub fn artifact_reference(&self) -> Result<MeshArtifactReference, Diagnostic> {
        Ok(MeshArtifactReference::from_sha256(
            self.digest()?.sha256_bytes(),
        ))
    }

    /// Reconstructed, invariant-checked mesh revision.
    #[must_use]
    pub const fn mesh(&self) -> &SimplicialMesh {
        &self.mesh
    }

    /// Topological and coordinate dimension of this full-dimensional mesh.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.mesh.topological_dimension()
    }
}

fn reconstruct(
    wire: &WireSimplicialMeshEnvelopeV1,
    limits: MeshDecoderLimits,
) -> Result<SimplicialMesh, Diagnostic> {
    if wire.schema != SIMPLICIAL_MESH_SCHEMA || wire.encoding != CANONICAL_ENCODING {
        return Err(invalid_artifact(
            "unsupported simplex-mesh schema or canonical encoding",
        ));
    }
    if wire.topology.cell_family != WireCellFamily::Simplex
        || wire.geometry.coordinate_scalar != WireCoordinateScalar::F64
        || wire.geometry.mapping != WireGeometryMapping::Affine
    {
        return Err(invalid_artifact(
            "simplex-mesh/v1 requires f64 affine simplex geometry",
        ));
    }
    require_count(
        "mesh vertices",
        wire.vertices.len(),
        limits.max_mesh_vertices,
    )?;
    require_count("mesh cells", wire.cells.len(), limits.max_mesh_cells)?;
    let coordinate_values = checked_nested_len(&wire.vertices, "mesh coordinate count")?;
    require_count(
        "mesh coordinate values",
        coordinate_values,
        limits.max_mesh_coordinate_values,
    )?;
    let connectivity_indices = checked_nested_len(&wire.cells, "mesh connectivity count")?;
    require_count(
        "mesh connectivity indices",
        connectivity_indices,
        limits.max_mesh_connectivity_indices,
    )?;

    let dimension = usize::try_from(wire.topology.dimension)
        .map_err(|_| invalid_artifact("mesh dimension exceeds local usize"))?;
    let cells = wire
        .cells
        .iter()
        .map(|cell| {
            cell.iter()
                .map(|&vertex| {
                    usize::try_from(vertex)
                        .map_err(|_| invalid_artifact("mesh vertex index exceeds local usize"))
                })
                .collect()
        })
        .collect::<Result<Vec<Vec<usize>>, Diagnostic>>()?;
    let gate = MeshQualityGate::new(wire.acceptance.minimum_mean_ratio)
        .map_err(|error| invalid_artifact(error.message()))?;
    let mesh = SimplicialMesh::new(dimension, wire.vertices.clone(), cells, gate)
        .map_err(|error| invalid_artifact(error.message()))?;
    let report = mesh.quality_report();
    if report.minimum_mean_ratio().to_bits() != wire.evidence.minimum_mean_ratio.to_bits()
        || report.minimum_signed_measure_scale().to_bits()
            != wire.evidence.minimum_signed_measure_scale.to_bits()
    {
        return Err(invalid_artifact(
            "simplex-mesh quality evidence differs from reconstructed geometry",
        ));
    }
    Ok(mesh)
}

fn checked_nested_len<T>(values: &[Vec<T>], label: &str) -> Result<usize, Diagnostic> {
    values.iter().try_fold(0_usize, |total, row| {
        total
            .checked_add(row.len())
            .ok_or_else(|| invalid_artifact(format!("{label} overflows usize")))
    })
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
struct WireSimplicialMeshEnvelopeV1 {
    schema: String,
    encoding: String,
    topology: WireTopology,
    geometry: WireGeometry,
    vertices: Vec<Vec<f64>>,
    cells: Vec<Vec<u64>>,
    acceptance: WireAcceptance,
    evidence: WireQualityEvidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireTopology {
    dimension: u64,
    cell_family: WireCellFamily,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireCellFamily {
    Simplex,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireGeometry {
    coordinate_scalar: WireCoordinateScalar,
    mapping: WireGeometryMapping,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireCoordinateScalar {
    F64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireGeometryMapping {
    Affine,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireAcceptance {
    minimum_mean_ratio: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireQualityEvidence {
    minimum_mean_ratio: f64,
    minimum_signed_measure_scale: f64,
}

#[cfg(test)]
mod tests {
    use eqiora_core::diagnostic::codes;

    use super::*;

    fn fixture() -> SimplicialMesh {
        SimplicialMesh::new(
            2,
            vec![
                vec![0.0, 0.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
                vec![0.0, 1.0],
            ],
            vec![vec![0, 1, 2], vec![0, 2, 3]],
            MeshQualityGate::new(0.25).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn canonical_round_trip_reconstructs_the_same_mesh_and_identity() {
        let envelope = SimplicialMeshEnvelopeV1::from_mesh(&fixture()).unwrap();
        let bytes = envelope.canonical_json().unwrap();
        let decoded =
            SimplicialMeshEnvelopeV1::from_json(&bytes, MeshDecoderLimits::default()).unwrap();
        assert_eq!(decoded.canonical_json().unwrap(), bytes);
        assert_eq!(decoded.mesh(), envelope.mesh());
        assert_eq!(decoded.digest().unwrap(), envelope.digest().unwrap());
        assert_eq!(
            decoded.artifact_reference().unwrap().sha256(),
            decoded.digest().unwrap().sha256_bytes(),
        );
    }

    #[test]
    fn resource_excess_unknown_fields_and_forged_evidence_fail_closed() {
        let envelope = SimplicialMeshEnvelopeV1::from_mesh(&fixture()).unwrap();
        let bytes = envelope.canonical_json().unwrap();
        let limits = MeshDecoderLimits {
            max_mesh_vertices: 3,
            ..MeshDecoderLimits::default()
        };
        assert_eq!(
            SimplicialMeshEnvelopeV1::from_json(&bytes, limits)
                .unwrap_err()
                .code(),
            codes::INVALID_ARTIFACT,
        );

        let mut unknown: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        unknown["topology"]["unknown"] = serde_json::json!(true);
        assert_eq!(
            SimplicialMeshEnvelopeV1::from_json(
                &serde_json::to_vec(&unknown).unwrap(),
                MeshDecoderLimits::default(),
            )
            .unwrap_err()
            .code(),
            codes::INVALID_ARTIFACT,
        );

        let mut forged: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        forged["evidence"]["minimum_mean_ratio"] = serde_json::json!(0.5);
        assert_eq!(
            SimplicialMeshEnvelopeV1::from_json(
                &serde_json::to_vec(&forged).unwrap(),
                MeshDecoderLimits::default(),
            )
            .unwrap_err()
            .code(),
            codes::INVALID_ARTIFACT,
        );
    }
}
