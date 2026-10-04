//! Canonical persistence for one authenticated common Mesh occurrence.

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use eqiora_artifact::{
    ArtifactDigest, CartesianMeshEnvelopeV1, GeometryDecoderLimits,
    GeometryMeshCorrespondenceEnvelopeV1, MeshDecoderLimits, MeshProductionLineageEnvelopeV1,
    SimplicialMeshEnvelopeV1,
};
use eqiora_geometry::{CanonicalGeometryLimits, CanonicalGeometryV1};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{AuthenticatedCommonMesh, Diagnostic, NativeMeshResources, invalid};

const SCHEMA: &str = "eqiora.authenticated-common-mesh/v2";
const ENCODING: &str = "canonical-json-rfc8259-v1";
const MAX_BYTES: usize = 128 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireMeshKind {
    StructuredCartesian,
    AffineTriangle,
    AdjacentPartition,
    Gmsh4152,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WirePhysicalMesh {
    kind: WireMeshKind,
    geometry_base64: String,
    mesh_base64: String,
    correspondence_base64: String,
    production_base64: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    provider_output_base64: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireAuthenticatedMesh {
    schema: String,
    encoding: String,
    resources: WireResources,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "kebab-case", deny_unknown_fields)]
enum WireResources {
    PhysicalGeometry {
        mesh: WirePhysicalMesh,
    },
    CoordinateFactors {
        factors: super::coordinate_grid::CoordinateSource,
        mesh_base64: String,
    },
}

impl WireAuthenticatedMesh {
    fn from_resources(resources: &NativeMeshResources) -> Result<Self, Diagnostic> {
        let resources = match resources {
            NativeMeshResources::Coordinates(grid) => WireResources::CoordinateFactors {
                factors: grid.source.clone(),
                mesh_base64: encode(&grid.mesh.canonical_json()?),
            },
            _ => WireResources::PhysicalGeometry {
                mesh: WirePhysicalMesh::from_resources(resources)?,
            },
        };
        Ok(Self {
            schema: SCHEMA.to_owned(),
            encoding: ENCODING.to_owned(),
            resources,
        })
    }

    fn validate_header(&self) -> Result<(), Diagnostic> {
        if self.schema != SCHEMA || self.encoding != ENCODING {
            return Err(invalid(
                "authenticated common Mesh has an unknown schema or encoding",
            ));
        }
        if let WireResources::PhysicalGeometry { mesh } = &self.resources {
            mesh.validate()?;
        }
        Ok(())
    }

    fn decode(&self) -> Result<AuthenticatedCommonMesh, Diagnostic> {
        match &self.resources {
            WireResources::PhysicalGeometry { mesh } => mesh.decode(),
            WireResources::CoordinateFactors {
                factors,
                mesh_base64,
            } => {
                let mesh = CartesianMeshEnvelopeV1::from_json(
                    &decode(mesh_base64, "mesh")?,
                    MeshDecoderLimits::default(),
                )?;
                let grid =
                    super::coordinate_grid::CoordinateGrid::from_parts(factors.clone(), mesh)?;
                Ok(AuthenticatedCommonMesh {
                    resources: NativeMeshResources::Coordinates(grid),
                })
            }
        }
    }
}

impl AuthenticatedCommonMesh {
    /// Encode this exact authenticated occurrence as bounded canonical bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Diagnostic> {
        let wire = WireAuthenticatedMesh::from_resources(&self.resources)?;
        serde_json::to_vec(&wire)
            .map_err(|error| invalid(format!("cannot encode authenticated common Mesh: {error}")))
    }

    /// Decode, reconstruct, and reauthenticate one exact common Mesh occurrence.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Diagnostic> {
        if bytes.len() > MAX_BYTES {
            return Err(invalid(format!(
                "authenticated common Mesh has {} bytes, exceeding the {MAX_BYTES} byte limit",
                bytes.len()
            )));
        }
        let wire: WireAuthenticatedMesh = serde_json::from_slice(bytes)
            .map_err(|error| invalid(format!("invalid authenticated common Mesh JSON: {error}")))?;
        wire.validate_header()?;
        let decoded = wire.decode()?;
        if decoded.to_bytes()? != bytes {
            return Err(invalid(
                "authenticated common Mesh bytes are not the canonical encoding of their content",
            ));
        }
        Ok(decoded)
    }

    /// Domain-separated identity of the complete authenticated occurrence.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        let mut hasher = Sha256::new();
        hasher.update(SCHEMA.as_bytes());
        hasher.update([0]);
        hasher.update(self.to_bytes()?);
        Ok(ArtifactDigest::from_sha256(hasher.finalize().into()))
    }

    /// Identity of the exact Geometry or dimensioned factor projection supplying this Mesh.
    pub fn source_digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        match &self.resources {
            NativeMeshResources::Coordinates(grid) => grid.source.digest(),
            _ => Ok(ArtifactDigest::from_sha256(
                self.resources.geometry()?.digest_bytes(),
            )),
        }
    }

    /// Physical Geometry root, absent for coordinate-factor grids.
    #[must_use]
    pub fn geometry(&self) -> Option<&CanonicalGeometryV1> {
        self.resources.geometry().ok()
    }

    /// Structured Cartesian Mesh root, when this is a Cartesian occurrence.
    #[must_use]
    pub fn cartesian_mesh(&self) -> Option<&CartesianMeshEnvelopeV1> {
        match &self.resources {
            NativeMeshResources::Cartesian { mesh, .. } => Some(mesh),
            NativeMeshResources::Coordinates(grid) => Some(&grid.mesh),
            NativeMeshResources::AffineTriangleSimplicial { .. }
            | NativeMeshResources::AdjacentPartitionSimplicial { .. }
            | NativeMeshResources::GmshSimplicial { .. } => None,
        }
    }

    /// Simplicial Mesh root, when this is a simplicial occurrence.
    #[must_use]
    pub fn simplicial_mesh(&self) -> Option<&SimplicialMeshEnvelopeV1> {
        match &self.resources {
            NativeMeshResources::Cartesian { .. } | NativeMeshResources::Coordinates(_) => None,
            NativeMeshResources::AffineTriangleSimplicial { mesh, .. }
            | NativeMeshResources::AdjacentPartitionSimplicial { mesh, .. }
            | NativeMeshResources::GmshSimplicial { mesh, .. } => Some(mesh),
        }
    }

    /// Geometry-to-Mesh correspondence authenticated by this occurrence.
    #[must_use]
    pub fn correspondence(&self) -> Option<&GeometryMeshCorrespondenceEnvelopeV1> {
        match &self.resources {
            NativeMeshResources::Cartesian { correspondence, .. }
            | NativeMeshResources::AffineTriangleSimplicial { correspondence, .. }
            | NativeMeshResources::AdjacentPartitionSimplicial { correspondence, .. }
            | NativeMeshResources::GmshSimplicial { correspondence, .. } => Some(correspondence),
            NativeMeshResources::Coordinates(_) => None,
        }
    }

    /// Mesh-production lineage authenticated by this occurrence.
    #[must_use]
    pub fn production(&self) -> Option<&MeshProductionLineageEnvelopeV1> {
        match &self.resources {
            NativeMeshResources::Cartesian { production, .. }
            | NativeMeshResources::AffineTriangleSimplicial { production, .. }
            | NativeMeshResources::AdjacentPartitionSimplicial { production, .. }
            | NativeMeshResources::GmshSimplicial { production, .. } => Some(production),
            NativeMeshResources::Coordinates(_) => None,
        }
    }

    /// Exact Gmsh provider observation, when the occurrence was provider-produced.
    #[must_use]
    pub fn gmsh_provider_output(&self) -> Option<&[u8]> {
        match &self.resources {
            NativeMeshResources::GmshSimplicial {
                provider_output, ..
            } => Some(provider_output),
            NativeMeshResources::Coordinates(_)
            | NativeMeshResources::Cartesian { .. }
            | NativeMeshResources::AffineTriangleSimplicial { .. }
            | NativeMeshResources::AdjacentPartitionSimplicial { .. } => None,
        }
    }
}

impl WirePhysicalMesh {
    fn from_resources(resources: &NativeMeshResources) -> Result<Self, Diagnostic> {
        let (kind, geometry, mesh, correspondence, production, provider_output) = match resources {
            NativeMeshResources::Coordinates(_) => {
                return Err(invalid("coordinate grid has no physical mesh lineage"));
            }
            NativeMeshResources::Cartesian {
                geometry,
                mesh,
                correspondence,
                production,
            } => (
                WireMeshKind::StructuredCartesian,
                geometry.canonical_bytes(),
                mesh.canonical_json()?,
                correspondence.canonical_json()?,
                production.canonical_json()?,
                None,
            ),
            NativeMeshResources::AffineTriangleSimplicial {
                geometry,
                mesh,
                correspondence,
                production,
            } => (
                WireMeshKind::AffineTriangle,
                geometry.canonical_bytes(),
                mesh.canonical_json()?,
                correspondence.canonical_json()?,
                production.canonical_json()?,
                None,
            ),
            NativeMeshResources::AdjacentPartitionSimplicial {
                geometry,
                mesh,
                correspondence,
                production,
            } => (
                WireMeshKind::AdjacentPartition,
                geometry.canonical_bytes(),
                mesh.canonical_json()?,
                correspondence.canonical_json()?,
                production.canonical_json()?,
                None,
            ),
            NativeMeshResources::GmshSimplicial {
                geometry,
                provider_output,
                mesh,
                correspondence,
                production,
                ..
            } => (
                WireMeshKind::Gmsh4152,
                geometry.canonical_bytes(),
                mesh.canonical_json()?,
                correspondence.canonical_json()?,
                production.canonical_json()?,
                Some(provider_output.as_ref()),
            ),
        };
        Ok(Self {
            kind,
            geometry_base64: encode(geometry),
            mesh_base64: encode(&mesh),
            correspondence_base64: encode(&correspondence),
            production_base64: encode(&production),
            provider_output_base64: provider_output.map(encode),
        })
    }

    fn validate(&self) -> Result<(), Diagnostic> {
        if matches!(self.kind, WireMeshKind::Gmsh4152) != self.provider_output_base64.is_some() {
            return Err(invalid(
                "only a Gmsh authenticated common Mesh carries provider output",
            ));
        }
        Ok(())
    }

    fn decode(&self) -> Result<AuthenticatedCommonMesh, Diagnostic> {
        let geometry_bytes = decode(&self.geometry_base64, "geometry")?;
        let mesh_bytes = decode(&self.mesh_base64, "mesh")?;
        let correspondence_bytes = decode(&self.correspondence_base64, "correspondence")?;
        let production_bytes = decode(&self.production_base64, "production")?;
        let geometry = CanonicalGeometryV1::replay_canonical(
            &geometry_bytes,
            CanonicalGeometryLimits::default(),
        )?;
        let correspondence = GeometryMeshCorrespondenceEnvelopeV1::from_json(
            &correspondence_bytes,
            GeometryDecoderLimits::default(),
        )?;
        let production = MeshProductionLineageEnvelopeV1::from_json(&production_bytes)?;
        match self.kind {
            WireMeshKind::StructuredCartesian => AuthenticatedCommonMesh::structured_cartesian(
                geometry,
                CartesianMeshEnvelopeV1::from_json(&mesh_bytes, MeshDecoderLimits::default())?,
                correspondence,
                production,
            ),
            WireMeshKind::AffineTriangle => AuthenticatedCommonMesh::affine_triangle_rectangle(
                geometry,
                SimplicialMeshEnvelopeV1::from_json(&mesh_bytes, MeshDecoderLimits::default())?,
                correspondence,
                production,
            ),
            WireMeshKind::AdjacentPartition => AuthenticatedCommonMesh::adjacent_partition(
                geometry,
                SimplicialMeshEnvelopeV1::from_json(&mesh_bytes, MeshDecoderLimits::default())?,
                correspondence,
                production,
            ),
            WireMeshKind::Gmsh4152 => {
                let policy = production.gmsh_mesh_policy().ok_or_else(|| {
                    invalid("Gmsh authenticated common Mesh has a non-Gmsh production policy")
                })?;
                let output = decode(
                    self.provider_output_base64
                        .as_deref()
                        .expect("validated Gmsh provider output"),
                    "provider output",
                )?;
                AuthenticatedCommonMesh::gmsh_4152(geometry, policy, output)
            }
        }
    }
}

fn encode(bytes: &[u8]) -> String {
    BASE64_STANDARD.encode(bytes)
}

fn decode(value: &str, label: &str) -> Result<Vec<u8>, Diagnostic> {
    let bytes = BASE64_STANDARD
        .decode(value)
        .map_err(|error| invalid(format!("invalid canonical base64 {label}: {error}")))?;
    if encode(&bytes) != value {
        return Err(invalid(format!("{label} is not canonical padded base64")));
    }
    Ok(bytes)
}
