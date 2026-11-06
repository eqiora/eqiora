//! Explicit units and acceptance limits for externally supplied MSH 4.1 bytes.
use super::*;

/// Coordinate unit declared by an external source; no unit is inferred from MSH.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum MshCoordinateUnit {
    /// Coordinates are already expressed in SI metres. No conversion is performed.
    #[serde(rename = "m")]
    Metres,
}

/// Import acceptance policy, distinct from a mesh generator's effective settings.
/// The shared MSH parser additionally enforces its fixed count and decoded-work ceilings.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct MshImportPolicyV1 {
    coordinate_unit: MshCoordinateUnit,
    minimum_mean_ratio: f64,
    maximum_boundary_facets: usize,
    maximum_source_bytes: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireMshImportPolicy {
    coordinate_unit: MshCoordinateUnit,
    minimum_mean_ratio: f64,
    maximum_boundary_facets: usize,
    maximum_source_bytes: usize,
}

impl<'de> Deserialize<'de> for MshImportPolicyV1 {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = WireMshImportPolicy::deserialize(deserializer)?;
        Self::new(
            wire.coordinate_unit,
            wire.minimum_mean_ratio,
            wire.maximum_boundary_facets,
            wire.maximum_source_bytes,
        )
        .map_err(serde::de::Error::custom)
    }
}

impl MshImportPolicyV1 {
    /// Declare units and positive acceptance/work limits. This does not request generation.
    pub fn metres(
        minimum_mean_ratio: f64,
        maximum_boundary_facets: usize,
        maximum_source_bytes: usize,
    ) -> Result<Self, Diagnostic> {
        Self::new(
            MshCoordinateUnit::Metres,
            minimum_mean_ratio,
            maximum_boundary_facets,
            maximum_source_bytes,
        )
    }

    fn new(
        coordinate_unit: MshCoordinateUnit,
        minimum_mean_ratio: f64,
        maximum_boundary_facets: usize,
        maximum_source_bytes: usize,
    ) -> Result<Self, Diagnostic> {
        MeshQualityGate::new(minimum_mean_ratio)?;
        if maximum_boundary_facets == 0 || maximum_source_bytes == 0 {
            return Err(invalid_artifact(
                "external MSH import limits must be positive",
            ));
        }
        Ok(Self {
            coordinate_unit,
            minimum_mean_ratio,
            maximum_boundary_facets,
            maximum_source_bytes,
        })
    }

    pub(super) fn validate(self) -> Result<(), Diagnostic> {
        Self::new(
            self.coordinate_unit,
            self.minimum_mean_ratio,
            self.maximum_boundary_facets,
            self.maximum_source_bytes,
        )
        .map(|_| ())
    }

    /// Explicit source coordinate unit.
    #[must_use]
    pub const fn coordinate_unit(self) -> &'static str {
        match self.coordinate_unit {
            MshCoordinateUnit::Metres => "m",
        }
    }
    /// Minimum scale-independent affine-simplex quality.
    #[must_use]
    pub const fn minimum_mean_ratio(self) -> f64 {
        self.minimum_mean_ratio
    }
    /// Maximum accepted boundary facets in either dimension.
    #[must_use]
    pub const fn maximum_boundary_facets(self) -> usize {
        self.maximum_boundary_facets
    }
    /// Source byte limit checked before parsing. Fixed parser ceilings also apply.
    #[must_use]
    pub const fn maximum_source_bytes(self) -> usize {
        self.maximum_source_bytes
    }
}

impl MeshProductionLineageEnvelopeV1 {
    /// Bind an external MSH import to accepted resources without claiming CLI execution.
    /// Provider version `4.1` names the accepted MSH format profile, not an executable release.
    pub fn from_external_msh41_resources(
        policy: MshImportPolicyV1,
        geometry: &CanonicalGeometryV1,
        mesh: &SimplicialMeshEnvelopeV1,
        correspondence: &GeometryMeshCorrespondenceEnvelopeV1,
    ) -> Result<Self, Diagnostic> {
        policy.validate()?;
        let lineage = Self {
            wire: WireMeshProductionLineageV1 {
                schema: SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                provider: WireProviderV1 {
                    identity: "eqiora.external-msh".to_owned(),
                    version: "4.1".to_owned(),
                },
                effective_policy: WireEffectivePolicyV1::ExternalMsh41(policy),
                geometry_sha256: ArtifactDigest::from_sha256(geometry.digest_bytes()).to_string(),
                mesh_sha256: mesh.digest()?.to_string(),
                correspondence_sha256: correspondence.digest()?.to_string(),
            },
        };
        lineage.validate_local()?;
        Ok(lineage)
    }

    /// Compare a complete external-import policy and its accepted resource identities.
    pub fn validate_against_external_msh41_resources(
        &self,
        policy: MshImportPolicyV1,
        geometry: &CanonicalGeometryV1,
        mesh: &SimplicialMeshEnvelopeV1,
        correspondence: &GeometryMeshCorrespondenceEnvelopeV1,
    ) -> Result<(), Diagnostic> {
        let expected = Self::from_external_msh41_resources(policy, geometry, mesh, correspondence)?;
        if self != &expected {
            return Err(invalid_artifact(
                "external MSH lineage differs from policy or accepted resources",
            ));
        }
        Ok(())
    }

    /// External import policy, absent for generated meshes.
    #[must_use]
    pub fn msh_import_policy(&self) -> Option<MshImportPolicyV1> {
        match self.wire.effective_policy {
            WireEffectivePolicyV1::ExternalMsh41(policy) => Some(policy),
            _ => None,
        }
    }
}
