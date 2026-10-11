//! Provider-owned production lineage for one accepted common Mesh.

use eqiora_core::Diagnostic;
use eqiora_geometry::CanonicalGeometryV1;
use eqiora_meshing::{CartesianMesh, MeshQualityGate};
use serde::{Deserialize, Serialize};

use crate::{
    ArtifactDigest, CANONICAL_ENCODING, CartesianMeshEnvelopeV1,
    GeometryMeshCorrespondenceEnvelopeV1, JsonDecoderLimits, SimplicialMeshEnvelopeV1,
    check_json_limits, invalid_artifact,
};

const SCHEMA: &str = "eqiora.mesh-production-lineage-envelope/v1";
const GMSH_IDENTITY: &str = "eqiora.gmsh-cli";
const GMSH_VERSION: &str = "4.15.2";
const CARTESIAN_IDENTITY: &str = "eqiora.structured-cartesian";
const CARTESIAN_VERSION: &str = "2";
const AFFINE_TRIANGLE_IDENTITY: &str = "eqiora.affine-triangle-rectangle";
const AFFINE_TRIANGLE_VERSION: &str = "1";
const AFFINE_TRIANGLE_DIAGONAL: &str = "lower-left-to-upper-right";

/// Closed identity of a provider that currently produces a common Mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MeshProductionProvider {
    /// Exact external Gmsh CLI 4.15.2 adapter.
    Gmsh4152,
    /// Deterministic dimension-parametric structured Cartesian producer v2.
    StructuredCartesianV2,
    /// Deterministic rectangle affine-triangle producer v1.
    AffineTriangleRectangleV1,
}

impl MeshProductionProvider {
    /// Stable provider identity independent of effective numerical policy.
    #[must_use]
    pub const fn identity(self) -> &'static str {
        match self {
            Self::Gmsh4152 => GMSH_IDENTITY,
            Self::StructuredCartesianV2 => CARTESIAN_IDENTITY,
            Self::AffineTriangleRectangleV1 => AFFINE_TRIANGLE_IDENTITY,
        }
    }

    /// Exact provider implementation version.
    #[must_use]
    pub const fn version(self) -> &'static str {
        match self {
            Self::Gmsh4152 => GMSH_VERSION,
            Self::StructuredCartesianV2 => CARTESIAN_VERSION,
            Self::AffineTriangleRectangleV1 => AFFINE_TRIANGLE_VERSION,
        }
    }
}

/// Exact effective policy of the structured Cartesian provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CartesianMeshCellsV2(Box<[usize]>);

impl CartesianMeshCellsV2 {
    /// Construct positive, bounded 1D--3D per-axis cell counts.
    pub fn new(cells: impl Into<Vec<usize>>) -> Result<Self, Diagnostic> {
        let cells = cells.into();
        if cells.is_empty() || cells.len() > 3 {
            return Err(invalid_artifact(
                "Cartesian cell counts require between one and three axes",
            ));
        }
        if cells.contains(&0) {
            return Err(invalid_artifact("Cartesian cell counts must be positive"));
        }
        if cells.iter().any(|&count| count.checked_add(1).is_none()) {
            return Err(invalid_artifact(
                "Cartesian cell count overflows its axis vertex count",
            ));
        }
        CartesianMesh::validate_cell_counts(&cells)
            .map_err(|error| invalid_artifact(error.message().to_owned()))?;
        Ok(Self(cells.into_boxed_slice()))
    }

    /// Exact x/y cell counts.
    #[must_use]
    pub fn cells(&self) -> &[usize] {
        &self.0
    }
}

/// Exact effective policy of the fixed-diagonal affine-triangle provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AffineTriangleMeshCellsV1([usize; 2]);

impl AffineTriangleMeshCellsV1 {
    /// Construct positive x/y subdivision counts whose analytic sizes fit `usize`.
    pub fn new(cells: [usize; 2]) -> Result<Self, Diagnostic> {
        if cells.contains(&0) {
            return Err(invalid_artifact(
                "affine-triangle cell counts must be positive",
            ));
        }
        let [nx, ny] = cells;
        let vertices = nx
            .checked_add(1)
            .and_then(|x| ny.checked_add(1).and_then(|y| x.checked_mul(y)));
        let triangles = nx.checked_mul(ny).and_then(|cells| cells.checked_mul(2));
        let boundary_facets = nx.checked_add(ny).and_then(|sum| sum.checked_mul(2));
        if vertices.is_none() || triangles.is_none() || boundary_facets.is_none() {
            return Err(invalid_artifact(
                "affine-triangle cell counts overflow analytic mesh counts",
            ));
        }
        Ok(Self(cells))
    }

    /// Exact x/y structured-cell counts before fixed diagonal subdivision.
    #[must_use]
    pub const fn cells(self) -> [usize; 2] {
        self.0
    }

    /// Provider-owned diagonal convention; callers cannot select another one.
    #[must_use]
    pub const fn diagonal(self) -> &'static str {
        AFFINE_TRIANGLE_DIAGONAL
    }
}

/// Complete effective numerical policy of one Gmsh mesh production.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GmshMeshPolicyV1 {
    maximum_boundary_error_m: f64,
    minimum_mean_ratio: f64,
    maximum_boundary_facets: usize,
    maximum_target_size_m: f64,
    maximum_target_size_is_explicit: bool,
}

impl GmshMeshPolicyV1 {
    fn new(
        maximum_boundary_error_m: f64,
        minimum_mean_ratio: f64,
        maximum_boundary_facets: usize,
        maximum_target_size_m: f64,
        maximum_target_size_is_explicit: bool,
    ) -> Result<Self, Diagnostic> {
        if !maximum_boundary_error_m.is_finite() || maximum_boundary_error_m <= 0.0 {
            return Err(invalid_artifact(
                "maximum boundary error must be finite and positive",
            ));
        }
        MeshQualityGate::new(minimum_mean_ratio)
            .map_err(|diagnostic| invalid_artifact(diagnostic.message()))?;
        if maximum_boundary_facets < 8 {
            return Err(invalid_artifact(
                "maximum boundary facets must be at least eight",
            ));
        }
        if !maximum_target_size_m.is_finite() || maximum_target_size_m <= 0.0 {
            return Err(invalid_artifact(
                "Gmsh maximum target size must be finite and positive",
            ));
        }
        Ok(Self {
            maximum_boundary_error_m,
            minimum_mean_ratio,
            maximum_boundary_facets,
            maximum_target_size_m,
            maximum_target_size_is_explicit,
        })
    }

    /// Construct a policy whose effective target was derived by the provider.
    pub fn automatic(
        maximum_boundary_error_m: f64,
        minimum_mean_ratio: f64,
        maximum_boundary_facets: usize,
        maximum_target_size_m: f64,
    ) -> Result<Self, Diagnostic> {
        Self::new(
            maximum_boundary_error_m,
            minimum_mean_ratio,
            maximum_boundary_facets,
            maximum_target_size_m,
            false,
        )
    }

    /// Construct a policy whose effective target was supplied by the caller.
    pub fn explicit(
        maximum_boundary_error_m: f64,
        minimum_mean_ratio: f64,
        maximum_boundary_facets: usize,
        maximum_target_size_m: f64,
    ) -> Result<Self, Diagnostic> {
        Self::new(
            maximum_boundary_error_m,
            minimum_mean_ratio,
            maximum_boundary_facets,
            maximum_target_size_m,
            true,
        )
    }

    /// Effective maximum chordal boundary error in metres.
    #[must_use]
    pub const fn maximum_boundary_error_m(self) -> f64 {
        self.maximum_boundary_error_m
    }

    /// Effective affine-simplex quality gate.
    #[must_use]
    pub const fn minimum_mean_ratio(self) -> f64 {
        self.minimum_mean_ratio
    }

    /// Effective circular-boundary work limit.
    #[must_use]
    pub const fn maximum_boundary_facets(self) -> usize {
        self.maximum_boundary_facets
    }

    /// Effective global Gmsh characteristic target size in metres.
    #[must_use]
    pub const fn maximum_target_size_m(self) -> f64 {
        self.maximum_target_size_m
    }

    /// Whether the effective target was supplied by the caller.
    #[must_use]
    pub const fn maximum_target_size_is_explicit(self) -> bool {
        self.maximum_target_size_is_explicit
    }
}

/// Canonical provider occurrence that produced one exact accepted common Mesh.
///
/// This artifact is deliberately separate from provider-neutral Geometry,
/// Mesh, and correspondence identity. It binds those identities to the exact
/// provider release and every effective provider policy value.
#[derive(Debug, Clone, PartialEq)]
pub struct MeshProductionLineageEnvelopeV1 {
    wire: WireMeshProductionLineageV1,
}

impl MeshProductionLineageEnvelopeV1 {
    /// Bind one exact Gmsh 4.15.2 occurrence to accepted resources.
    ///
    /// # Errors
    /// Returns `EQ0901` if a resource digest cannot be constructed.
    pub fn from_gmsh_4152_resources(
        policy: GmshMeshPolicyV1,
        geometry: &CanonicalGeometryV1,
        mesh: &SimplicialMeshEnvelopeV1,
        correspondence: &GeometryMeshCorrespondenceEnvelopeV1,
    ) -> Result<Self, Diagnostic> {
        Self::from_resources(
            MeshProductionProvider::Gmsh4152,
            policy,
            geometry,
            mesh,
            correspondence,
        )
    }

    /// Bind one provider occurrence to accepted Geometry/Mesh resources.
    ///
    /// # Errors
    /// Returns `EQ0901` if a resource digest cannot be constructed.
    fn from_resources(
        provider: MeshProductionProvider,
        policy: GmshMeshPolicyV1,
        geometry: &CanonicalGeometryV1,
        mesh: &SimplicialMeshEnvelopeV1,
        correspondence: &GeometryMeshCorrespondenceEnvelopeV1,
    ) -> Result<Self, Diagnostic> {
        let lineage = Self {
            wire: WireMeshProductionLineageV1 {
                schema: SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                provider: WireProviderV1 {
                    identity: provider.identity().to_owned(),
                    version: provider.version().to_owned(),
                },
                effective_policy: WireEffectivePolicyV1::GmshMesh(WireGmshMeshPolicyV1::try_from(
                    policy,
                )?),
                geometry_sha256: ArtifactDigest::from_sha256(geometry.digest_bytes()).to_string(),
                mesh_sha256: mesh.digest()?.to_string(),
                correspondence_sha256: correspondence.digest()?.to_string(),
            },
        };
        lineage.validate_local()?;
        Ok(lineage)
    }

    /// Bind one dimension-parametric structured Cartesian v2 occurrence.
    pub fn from_structured_cartesian_v2_resources(
        policy: &CartesianMeshCellsV2,
        geometry: &CanonicalGeometryV1,
        mesh: &CartesianMeshEnvelopeV1,
        correspondence: &GeometryMeshCorrespondenceEnvelopeV1,
    ) -> Result<Self, Diagnostic> {
        let lineage = Self {
            wire: WireMeshProductionLineageV1 {
                schema: SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                provider: WireProviderV1 {
                    identity: CARTESIAN_IDENTITY.to_owned(),
                    version: CARTESIAN_VERSION.to_owned(),
                },
                effective_policy: WireEffectivePolicyV1::CartesianCells(
                    WireCartesianCellsV2::try_from(policy)?,
                ),
                geometry_sha256: ArtifactDigest::from_sha256(geometry.digest_bytes()).to_string(),
                mesh_sha256: mesh.digest()?.to_string(),
                correspondence_sha256: correspondence.digest()?.to_string(),
            },
        };
        lineage.validate_local()?;
        Ok(lineage)
    }

    /// Bind one fixed-diagonal affine-triangle v1 occurrence to exact rectangle resources.
    pub fn from_affine_triangle_rectangle_v1_resources(
        policy: AffineTriangleMeshCellsV1,
        geometry: &CanonicalGeometryV1,
        mesh: &SimplicialMeshEnvelopeV1,
        correspondence: &GeometryMeshCorrespondenceEnvelopeV1,
    ) -> Result<Self, Diagnostic> {
        let lineage = Self {
            wire: WireMeshProductionLineageV1 {
                schema: SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                provider: WireProviderV1 {
                    identity: AFFINE_TRIANGLE_IDENTITY.to_owned(),
                    version: AFFINE_TRIANGLE_VERSION.to_owned(),
                },
                effective_policy: WireEffectivePolicyV1::AffineTriangleCells(
                    WireAffineTriangleCellsV1::try_from(policy)?,
                ),
                geometry_sha256: ArtifactDigest::from_sha256(geometry.digest_bytes()).to_string(),
                mesh_sha256: mesh.digest()?.to_string(),
                correspondence_sha256: correspondence.digest()?.to_string(),
            },
        };
        lineage.validate_local()?;
        Ok(lineage)
    }

    /// Decode exact canonical lineage bytes.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed, unknown, noncanonical, or invalid data.
    pub fn from_json(bytes: &[u8]) -> Result<Self, Diagnostic> {
        check_json_limits(bytes, JsonDecoderLimits::default())?;
        let wire: WireMeshProductionLineageV1 = serde_json::from_slice(bytes).map_err(|error| {
            invalid_artifact(format!("invalid mesh production lineage JSON: {error}"))
        })?;
        let lineage = Self { wire };
        lineage.validate_local()?;
        if lineage.canonical_json()? != bytes {
            return Err(invalid_artifact(
                "mesh production lineage is not canonical JSON",
            ));
        }
        Ok(lineage)
    }

    /// Rebuild and compare the complete production occurrence.
    ///
    /// # Errors
    /// Returns `EQ0901` if provider, policy, or any bound resource differs.
    fn validate_against_resources(
        &self,
        provider: MeshProductionProvider,
        policy: GmshMeshPolicyV1,
        geometry: &CanonicalGeometryV1,
        mesh: &SimplicialMeshEnvelopeV1,
        correspondence: &GeometryMeshCorrespondenceEnvelopeV1,
    ) -> Result<(), Diagnostic> {
        let expected = Self::from_resources(provider, policy, geometry, mesh, correspondence)?;
        if self != &expected {
            return Err(invalid_artifact(
                "mesh production lineage differs from provider occurrence or accepted resources",
            ));
        }
        Ok(())
    }

    /// Rebuild and compare one exact Gmsh 4.15.2 occurrence.
    ///
    /// # Errors
    /// Returns `EQ0901` if policy or any bound resource differs.
    pub fn validate_against_gmsh_4152_resources(
        &self,
        policy: GmshMeshPolicyV1,
        geometry: &CanonicalGeometryV1,
        mesh: &SimplicialMeshEnvelopeV1,
        correspondence: &GeometryMeshCorrespondenceEnvelopeV1,
    ) -> Result<(), Diagnostic> {
        self.validate_against_resources(
            MeshProductionProvider::Gmsh4152,
            policy,
            geometry,
            mesh,
            correspondence,
        )
    }

    /// Rebuild and compare one exact structured Cartesian v2 occurrence.
    pub fn validate_against_structured_cartesian_v2_resources(
        &self,
        policy: &CartesianMeshCellsV2,
        geometry: &CanonicalGeometryV1,
        mesh: &CartesianMeshEnvelopeV1,
        correspondence: &GeometryMeshCorrespondenceEnvelopeV1,
    ) -> Result<(), Diagnostic> {
        let expected =
            Self::from_structured_cartesian_v2_resources(policy, geometry, mesh, correspondence)?;
        if self != &expected {
            return Err(invalid_artifact(
                "mesh production lineage differs from structured Cartesian occurrence or accepted resources",
            ));
        }
        Ok(())
    }

    /// Rebuild and compare one exact fixed-diagonal affine-triangle occurrence.
    pub fn validate_against_affine_triangle_rectangle_v1_resources(
        &self,
        policy: AffineTriangleMeshCellsV1,
        geometry: &CanonicalGeometryV1,
        mesh: &SimplicialMeshEnvelopeV1,
        correspondence: &GeometryMeshCorrespondenceEnvelopeV1,
    ) -> Result<(), Diagnostic> {
        let expected = Self::from_affine_triangle_rectangle_v1_resources(
            policy,
            geometry,
            mesh,
            correspondence,
        )?;
        if self != &expected {
            return Err(invalid_artifact(
                "mesh production lineage differs from affine-triangle rectangle occurrence or accepted resources",
            ));
        }
        Ok(())
    }

    /// Stable provider identity reconstructed from the canonical wire.
    #[must_use]
    pub fn provider_identity(&self) -> &str {
        &self.wire.provider.identity
    }

    /// Exact provider implementation version reconstructed from the wire.
    #[must_use]
    pub fn provider_version(&self) -> &str {
        &self.wire.provider.version
    }

    /// Exact effective numerical policy reconstructed from the canonical wire.
    #[must_use]
    pub fn gmsh_mesh_policy(&self) -> Option<GmshMeshPolicyV1> {
        match self.wire.effective_policy {
            WireEffectivePolicyV1::GmshMesh(policy) => policy.to_policy().ok(),
            WireEffectivePolicyV1::CartesianCells(_)
            | WireEffectivePolicyV1::AffineTriangleCells(_) => None,
        }
    }

    /// Exact Cartesian cell policy, when this is a Cartesian occurrence.
    #[must_use]
    pub fn cartesian_cells(&self) -> Option<CartesianMeshCellsV2> {
        match &self.wire.effective_policy {
            WireEffectivePolicyV1::CartesianCells(policy) => policy.to_policy().ok(),
            WireEffectivePolicyV1::GmshMesh(_) | WireEffectivePolicyV1::AffineTriangleCells(_) => {
                None
            }
        }
    }

    /// Exact affine-triangle subdivision policy, when this is that occurrence.
    #[must_use]
    pub fn affine_triangle_cells(&self) -> Option<AffineTriangleMeshCellsV1> {
        match self.wire.effective_policy {
            WireEffectivePolicyV1::AffineTriangleCells(policy) => policy.to_policy().ok(),
            WireEffectivePolicyV1::GmshMesh(_) | WireEffectivePolicyV1::CartesianCells(_) => None,
        }
    }

    /// Deterministic canonical JSON bytes.
    ///
    /// # Errors
    /// Returns `EQ0901` if serialization unexpectedly fails.
    pub fn canonical_json(&self) -> Result<Vec<u8>, Diagnostic> {
        serde_json::to_vec(&self.wire).map_err(|error| {
            invalid_artifact(format!("cannot serialize mesh production lineage: {error}"))
        })
    }

    /// Domain-separated identity of this provider occurrence.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization unexpectedly fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
    }

    fn validate_local(&self) -> Result<(), Diagnostic> {
        if self.wire.schema != SCHEMA || self.wire.encoding != CANONICAL_ENCODING {
            return Err(invalid_artifact(
                "unsupported mesh production lineage schema or encoding",
            ));
        }
        let provider = provider_from_wire(&self.wire.provider)?;
        self.wire.effective_policy.validate()?;
        let compatible = matches!(
            (provider, &self.wire.effective_policy),
            (
                MeshProductionProvider::Gmsh4152,
                WireEffectivePolicyV1::GmshMesh(_)
            ) | (
                MeshProductionProvider::StructuredCartesianV2,
                WireEffectivePolicyV1::CartesianCells(_)
            ) | (
                MeshProductionProvider::AffineTriangleRectangleV1,
                WireEffectivePolicyV1::AffineTriangleCells(_)
            )
        );
        if !compatible {
            return Err(invalid_artifact(
                "mesh production provider and effective policy kinds differ",
            ));
        }
        ArtifactDigest::from_hex(self.wire.geometry_sha256.clone())?;
        ArtifactDigest::from_hex(self.wire.mesh_sha256.clone())?;
        ArtifactDigest::from_hex(self.wire.correspondence_sha256.clone())?;
        Ok(())
    }
}

fn provider_from_wire(wire: &WireProviderV1) -> Result<MeshProductionProvider, Diagnostic> {
    match (wire.identity.as_str(), wire.version.as_str()) {
        (GMSH_IDENTITY, GMSH_VERSION) => Ok(MeshProductionProvider::Gmsh4152),
        (CARTESIAN_IDENTITY, CARTESIAN_VERSION) => {
            Ok(MeshProductionProvider::StructuredCartesianV2)
        }
        (AFFINE_TRIANGLE_IDENTITY, AFFINE_TRIANGLE_VERSION) => {
            Ok(MeshProductionProvider::AffineTriangleRectangleV1)
        }
        _ => Err(invalid_artifact(
            "mesh production lineage names an unknown provider identity or version",
        )),
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireMeshProductionLineageV1 {
    schema: String,
    encoding: String,
    provider: WireProviderV1,
    effective_policy: WireEffectivePolicyV1,
    geometry_sha256: String,
    mesh_sha256: String,
    correspondence_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireProviderV1 {
    identity: String,
    version: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireGmshMeshPolicyV1 {
    maximum_boundary_error_m: f64,
    minimum_mean_ratio: f64,
    maximum_boundary_facets: u64,
    maximum_target_size_m: f64,
    maximum_target_size_ownership: WireGmshTargetSizeOwnershipV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
enum WireEffectivePolicyV1 {
    GmshMesh(WireGmshMeshPolicyV1),
    CartesianCells(WireCartesianCellsV2),
    AffineTriangleCells(WireAffineTriangleCellsV1),
}

impl WireEffectivePolicyV1 {
    fn validate(&self) -> Result<(), Diagnostic> {
        match self {
            Self::GmshMesh(policy) => (*policy).to_policy().map(|_| ()),
            Self::CartesianCells(policy) => policy.to_policy().map(|_| ()),
            Self::AffineTriangleCells(policy) => (*policy).to_policy().map(|_| ()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireGmshTargetSizeOwnershipV1 {
    Automatic,
    Explicit,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireCartesianCellsV2 {
    cells: Vec<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireAffineTriangleCellsV1 {
    cells: [u64; 2],
    diagonal: WireAffineTriangleDiagonalV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireAffineTriangleDiagonalV1 {
    LowerLeftToUpperRight,
}

impl TryFrom<AffineTriangleMeshCellsV1> for WireAffineTriangleCellsV1 {
    type Error = Diagnostic;

    fn try_from(policy: AffineTriangleMeshCellsV1) -> Result<Self, Self::Error> {
        Ok(Self {
            cells: [
                u64::try_from(policy.0[0])
                    .map_err(|_| invalid_artifact("x cell count exceeds portable u64"))?,
                u64::try_from(policy.0[1])
                    .map_err(|_| invalid_artifact("y cell count exceeds portable u64"))?,
            ],
            diagonal: WireAffineTriangleDiagonalV1::LowerLeftToUpperRight,
        })
    }
}

impl WireAffineTriangleCellsV1 {
    fn to_policy(self) -> Result<AffineTriangleMeshCellsV1, Diagnostic> {
        AffineTriangleMeshCellsV1::new([
            usize::try_from(self.cells[0])
                .map_err(|_| invalid_artifact("x cell count exceeds local usize"))?,
            usize::try_from(self.cells[1])
                .map_err(|_| invalid_artifact("y cell count exceeds local usize"))?,
        ])
    }
}

impl TryFrom<&CartesianMeshCellsV2> for WireCartesianCellsV2 {
    type Error = Diagnostic;

    fn try_from(policy: &CartesianMeshCellsV2) -> Result<Self, Self::Error> {
        Ok(Self {
            cells: policy
                .cells()
                .iter()
                .map(|&count| {
                    u64::try_from(count)
                        .map_err(|_| invalid_artifact("Cartesian cell count exceeds portable u64"))
                })
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

impl WireCartesianCellsV2 {
    fn to_policy(&self) -> Result<CartesianMeshCellsV2, Diagnostic> {
        CartesianMeshCellsV2::new(
            self.cells
                .iter()
                .map(|&count| {
                    usize::try_from(count)
                        .map_err(|_| invalid_artifact("Cartesian cell count exceeds local usize"))
                })
                .collect::<Result<Vec<_>, _>>()?,
        )
    }
}

impl TryFrom<GmshMeshPolicyV1> for WireGmshMeshPolicyV1 {
    type Error = Diagnostic;

    fn try_from(policy: GmshMeshPolicyV1) -> Result<Self, Self::Error> {
        Ok(Self {
            maximum_boundary_error_m: policy.maximum_boundary_error_m,
            minimum_mean_ratio: policy.minimum_mean_ratio,
            maximum_boundary_facets: u64::try_from(policy.maximum_boundary_facets)
                .map_err(|_| invalid_artifact("maximum boundary facets exceeds portable u64"))?,
            maximum_target_size_m: policy.maximum_target_size_m,
            maximum_target_size_ownership: if policy.maximum_target_size_is_explicit {
                WireGmshTargetSizeOwnershipV1::Explicit
            } else {
                WireGmshTargetSizeOwnershipV1::Automatic
            },
        })
    }
}

impl WireGmshMeshPolicyV1 {
    fn to_policy(self) -> Result<GmshMeshPolicyV1, Diagnostic> {
        let maximum_boundary_facets = usize::try_from(self.maximum_boundary_facets)
            .map_err(|_| invalid_artifact("maximum boundary facets exceeds local usize"))?;
        match self.maximum_target_size_ownership {
            WireGmshTargetSizeOwnershipV1::Automatic => GmshMeshPolicyV1::automatic(
                self.maximum_boundary_error_m,
                self.minimum_mean_ratio,
                maximum_boundary_facets,
                self.maximum_target_size_m,
            ),
            WireGmshTargetSizeOwnershipV1::Explicit => GmshMeshPolicyV1::explicit(
                self.maximum_boundary_error_m,
                self.minimum_mean_ratio,
                maximum_boundary_facets,
                self.maximum_target_size_m,
            ),
        }
    }
}
