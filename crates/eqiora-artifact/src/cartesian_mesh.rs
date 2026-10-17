//! Portable, content-addressed Cartesian mesh revisions.

use eqiora_core::Diagnostic;
use eqiora_meshing::{CartesianMesh, MeshTopology};
use eqiora_realization::MeshArtifactReference;
use serde::{Deserialize, Serialize};

use crate::{ArtifactDigest, CANONICAL_ENCODING, MeshDecoderLimits, invalid_artifact};

const CARTESIAN_MESH_SCHEMA: &str = "eqiora.cartesian-mesh-envelope/v1";

/// Resource policy for decoding an axis-compressed Cartesian mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CartesianMeshDecoderLimits {
    /// Shared JSON and expanded-mesh resource limits.
    pub mesh: MeshDecoderLimits,
    /// Maximum number of reconstructed entities across every stratum.
    pub max_cartesian_entities: usize,
    /// Maximum number of vertex references across every entity closure.
    pub max_cartesian_entity_vertex_references: usize,
}

impl Default for CartesianMeshDecoderLimits {
    fn default() -> Self {
        Self {
            mesh: MeshDecoderLimits::default(),
            max_cartesian_entities: 1_000_000,
            max_cartesian_entity_vertex_references: 8_000_000,
        }
    }
}

impl CartesianMeshDecoderLimits {
    fn trusted_capture() -> Self {
        Self {
            mesh: MeshDecoderLimits::default(),
            max_cartesian_entities: usize::MAX,
            max_cartesian_entity_vertex_references: usize::MAX,
        }
    }
}

/// Versioned axes and canonical topology for one Cartesian mesh revision.
///
/// The artifact owns mesh topology and geometry, not a finite-element basis.
/// Canonical entity order is last-axis-fastest and top-cell vertex closure is
/// tensor-product order. Numerical spaces such as Q1 remain separate.
#[derive(Debug, Clone, PartialEq)]
pub struct CartesianMeshEnvelopeV1 {
    wire: WireCartesianMeshEnvelopeV1,
    mesh: CartesianMesh,
}

impl CartesianMeshEnvelopeV1 {
    /// Capture one already validated Cartesian mesh revision.
    ///
    /// # Errors
    /// Returns `EQ0901` if its axes or implied entity inventory exceed the
    /// portable decoder contract.
    pub fn from_mesh(mesh: &CartesianMesh) -> Result<Self, Diagnostic> {
        let axes = (0..mesh.topological_dimension())
            .map(|axis| {
                mesh.axis_coordinates(axis)
                    .map(<[f64]>::to_vec)
                    .ok_or_else(|| invalid_artifact("Cartesian mesh omitted a physical axis"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_axes(axes, CartesianMeshDecoderLimits::trusted_capture())
    }

    /// Decode with byte, nesting, coordinate, entity, and connectivity limits.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed, unknown, oversized, noncanonical, or
    /// invalid Cartesian mesh data.
    pub fn from_json(bytes: &[u8], mesh_limits: MeshDecoderLimits) -> Result<Self, Diagnostic> {
        Self::from_json_with_limits(
            bytes,
            CartesianMeshDecoderLimits {
                mesh: mesh_limits,
                ..CartesianMeshDecoderLimits::default()
            },
        )
    }

    /// Decode with explicit shared and Cartesian reconstruction limits.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed, unknown, oversized, noncanonical, or
    /// invalid Cartesian mesh data.
    pub fn from_json_with_limits(
        bytes: &[u8],
        limits: CartesianMeshDecoderLimits,
    ) -> Result<Self, Diagnostic> {
        limits.mesh.json.check(bytes)?;
        let wire = serde_json::from_slice(bytes)
            .map_err(|error| invalid_artifact(format!("invalid Cartesian mesh JSON: {error}")))?;
        Self::from_wire(wire, limits)
    }

    /// Deterministic canonical JSON bytes.
    ///
    /// # Errors
    /// Returns `EQ0901` if serialization unexpectedly fails.
    pub fn canonical_json(&self) -> Result<Vec<u8>, Diagnostic> {
        serde_json::to_vec(&self.wire)
            .map_err(|error| invalid_artifact(format!("cannot serialize Cartesian mesh: {error}")))
    }

    /// Domain-separated identity of the complete Cartesian mesh revision.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            CARTESIAN_MESH_SCHEMA.as_bytes(),
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

    /// Reconstructed, invariant-checked Cartesian mesh revision.
    #[must_use]
    pub const fn mesh(&self) -> &CartesianMesh {
        &self.mesh
    }

    /// Topological and coordinate dimension of this full-dimensional mesh.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.mesh.topological_dimension()
    }

    fn from_axes(
        axes: Vec<Vec<f64>>,
        limits: CartesianMeshDecoderLimits,
    ) -> Result<Self, Diagnostic> {
        let dimension = u64::try_from(axes.len())
            .map_err(|_| invalid_artifact("Cartesian mesh dimension exceeds portable u64"))?;
        Self::from_wire(
            WireCartesianMeshEnvelopeV1 {
                schema: CARTESIAN_MESH_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                dimension,
                scalar: WireScalar::F64,
                cell_family: WireCellFamily::Hypercube,
                axes,
                vertex_order: WireEntityOrder::LastAxisFastest,
                cell_order: WireEntityOrder::LastAxisFastest,
                local_node_order: WireLocalNodeOrder::TensorProductZ,
            },
            limits,
        )
    }

    fn from_wire(
        wire: WireCartesianMeshEnvelopeV1,
        limits: CartesianMeshDecoderLimits,
    ) -> Result<Self, Diagnostic> {
        if wire.schema != CARTESIAN_MESH_SCHEMA
            || wire.encoding != CANONICAL_ENCODING
            || wire.scalar != WireScalar::F64
            || wire.cell_family != WireCellFamily::Hypercube
            || wire.vertex_order != WireEntityOrder::LastAxisFastest
            || wire.cell_order != WireEntityOrder::LastAxisFastest
            || wire.local_node_order != WireLocalNodeOrder::TensorProductZ
        {
            return Err(invalid_artifact(
                "unsupported Cartesian mesh schema, encoding, scalar, topology, or ordering",
            ));
        }
        let dimension = usize::try_from(wire.dimension)
            .map_err(|_| invalid_artifact("Cartesian mesh dimension exceeds local usize"))?;
        if dimension == 0 || wire.axes.len() != dimension {
            return Err(invalid_artifact(
                "Cartesian mesh requires one nonempty axis per dimension",
            ));
        }
        let stored_coordinate_count = wire.axes.iter().try_fold(0_usize, |total, axis| {
            total
                .checked_add(axis.len())
                .ok_or_else(|| invalid_artifact("Cartesian axis coordinate count overflows usize"))
        })?;
        require_count(
            "Cartesian axis coordinate",
            stored_coordinate_count,
            limits.mesh.max_mesh_coordinate_values,
        )?;
        if wire
            .axes
            .iter()
            .flatten()
            .any(|value| value.to_bits() == (-0.0_f64).to_bits())
        {
            return Err(invalid_artifact(
                "Cartesian mesh coordinates require canonical positive zero",
            ));
        }
        if wire.axes.iter().all(|axis| axis.len() >= 2) {
            let cartesian_entity_count =
                checked_axis_state_product(&wire.axes, 2, 1, "Cartesian mesh entity")?;
            require_count(
                "Cartesian mesh entity",
                cartesian_entity_count,
                limits.max_cartesian_entities,
            )?;
            let cartesian_entity_vertex_reference_count = checked_axis_state_product(
                &wire.axes,
                3,
                2,
                "Cartesian mesh entity-closure vertex-reference",
            )?;
            require_count(
                "Cartesian mesh entity-closure vertex-reference",
                cartesian_entity_vertex_reference_count,
                limits.max_cartesian_entity_vertex_references,
            )?;
        }
        let vertex_count = wire.axes.iter().try_fold(1_usize, |count, axis| {
            count
                .checked_mul(axis.len())
                .ok_or_else(|| invalid_artifact("Cartesian vertex count overflows usize"))
        })?;
        let cell_count = wire.axes.iter().try_fold(1_usize, |count, axis| {
            count
                .checked_mul(axis.len().saturating_sub(1))
                .ok_or_else(|| invalid_artifact("Cartesian cell count overflows usize"))
        })?;
        require_count(
            "Cartesian mesh vertex",
            vertex_count,
            limits.mesh.max_mesh_vertices,
        )?;
        require_count(
            "Cartesian mesh cell",
            cell_count,
            limits.mesh.max_mesh_cells,
        )?;
        let expanded_coordinate_count = vertex_count
            .checked_mul(dimension)
            .ok_or_else(|| invalid_artifact("Cartesian mesh coordinate count overflows usize"))?;
        require_count(
            "Cartesian expanded coordinate",
            expanded_coordinate_count,
            limits.mesh.max_mesh_coordinate_values,
        )?;
        let cell_width = 1_usize
            .checked_shl(
                u32::try_from(dimension)
                    .map_err(|_| invalid_artifact("Cartesian dimension exceeds local u32"))?,
            )
            .ok_or_else(|| invalid_artifact("Cartesian cell arity overflows usize"))?;
        let connectivity_count = cell_count
            .checked_mul(cell_width)
            .ok_or_else(|| invalid_artifact("Cartesian connectivity count overflows usize"))?;
        require_count(
            "Cartesian connectivity index",
            connectivity_count,
            limits.mesh.max_mesh_connectivity_indices,
        )?;
        let mesh = CartesianMesh::from_axes(wire.axes.clone())
            .map_err(|error| invalid_artifact(error.message()))?;
        if mesh.entity_count(0) != Some(vertex_count)
            || mesh.entity_count(dimension) != Some(cell_count)
        {
            return Err(invalid_artifact(
                "Cartesian mesh reconstruction changed its implied entity inventory",
            ));
        }
        Ok(Self { wire, mesh })
    }
}

fn checked_axis_state_product(
    axes: &[Vec<f64>],
    multiplier: usize,
    offset: usize,
    label: &str,
) -> Result<usize, Diagnostic> {
    axes.iter().try_fold(1_usize, |product, axis| {
        let factor = axis
            .len()
            .checked_mul(multiplier)
            .and_then(|weighted| weighted.checked_sub(offset))
            .ok_or_else(|| invalid_artifact(format!("{label} count overflows usize")))?;
        product
            .checked_mul(factor)
            .ok_or_else(|| invalid_artifact(format!("{label} count overflows usize")))
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
struct WireCartesianMeshEnvelopeV1 {
    schema: String,
    encoding: String,
    dimension: u64,
    scalar: WireScalar,
    cell_family: WireCellFamily,
    axes: Vec<Vec<f64>>,
    vertex_order: WireEntityOrder,
    cell_order: WireEntityOrder,
    local_node_order: WireLocalNodeOrder,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireScalar {
    F64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireCellFamily {
    Hypercube,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireEntityOrder {
    LastAxisFastest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireLocalNodeOrder {
    TensorProductZ,
}
