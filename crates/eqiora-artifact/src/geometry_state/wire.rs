//! Closed current Geometry State wire and local admission.

use super::validation::{
    is_negative_geometry_zero, validate_coordinate_payload, validate_endpoint_quality,
    validate_path_quality,
};
use super::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireGeometryState {
    pub schema: String,
    pub encoding: String,
    pub spatial_dimension: u64,
    pub reference: MovingSpatialReference,
    pub accepted: WireAccepted,
    pub driver_snapshot_sha256: String,
    pub coordinates: WireCoordinates,
    pub origin: WireOrigin,
    pub quality_evidence: WireQuality,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireAccepted {
    pub step: u64,
    pub time_s: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireCoordinates {
    pub scalar: WireScalar,
    pub unit: WireCoordinateUnit,
    pub ordering: WireCoordinateOrdering,
    pub values: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(super) enum WireOrigin {
    Initial {
        minimum_path_signed_measure_scale: f64,
    },
    Continuous {
        predecessor_geometry_state_sha256: String,
        mesh_velocity_unit: WireMeshVelocityUnit,
        mesh_velocity_m_per_s: Vec<Vec<f64>>,
        minimum_path_signed_measure_scale: f64,
    },
    Remesh {
        source_spatial_state_sha256: String,
        source_geometry_state_sha256: String,
        semantic_association_sha256: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireQuality {
    pub reference: WireEndpointQuality,
    pub current: WireEndpointQuality,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireEndpointQuality {
    pub minimum_mean_ratio: f64,
    pub minimum_signed_measure_scale: f64,
}

impl WireEndpointQuality {
    pub fn from_report(report: eqiora_meshing::MeshQualityReport) -> Self {
        Self {
            minimum_mean_ratio: report.minimum_mean_ratio(),
            minimum_signed_measure_scale: report.minimum_signed_measure_scale(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum WireScalar {
    F64,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum WireCoordinateUnit {
    Metre,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum WireMeshVelocityUnit {
    MetrePerSecond,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum WireCoordinateOrdering {
    ReferenceMeshVertex,
}

impl GeometryStateEnvelope {
    pub(super) fn validate_local(&self, limits: MeshDecoderLimits) -> Result<(), Diagnostic> {
        if self.wire.schema != GEOMETRY_STATE_SCHEMA
            || self.wire.encoding != CANONICAL_ENCODING
            || !matches!(self.wire.spatial_dimension, 2 | 3)
        {
            return Err(invalid_artifact(
                "unsupported geometry-state schema, encoding or dimension",
            ));
        }
        self.wire.reference.validate()?;
        ArtifactDigest::from_hex(self.wire.driver_snapshot_sha256.clone())?;
        if !self.time_s().is_finite()
            || self.time_s() < 0.0
            || is_negative_geometry_zero(self.time_s())
        {
            return Err(invalid_artifact(
                "geometry-state time must be finite, nonnegative and canonical",
            ));
        }
        match &self.wire.origin {
            WireOrigin::Initial {
                minimum_path_signed_measure_scale,
            } => {
                if self.step() != 0 || self.time_s() != 0.0 {
                    return Err(invalid_artifact(
                        "initial geometry-state requires step and time zero",
                    ));
                }
                validate_path_quality(*minimum_path_signed_measure_scale)?;
            }
            WireOrigin::Continuous {
                predecessor_geometry_state_sha256,
                minimum_path_signed_measure_scale,
                ..
            } => {
                ArtifactDigest::from_hex(predecessor_geometry_state_sha256.clone())?;
                if self.step() == 0 || self.time_s() == 0.0 {
                    return Err(invalid_artifact(
                        "continuous geometry-state requires positive step and time",
                    ));
                }
                validate_path_quality(*minimum_path_signed_measure_scale)?;
            }
            WireOrigin::Remesh {
                source_spatial_state_sha256,
                source_geometry_state_sha256,
                semantic_association_sha256,
            } => {
                if self.spatial_dimension() != 2 {
                    return Err(invalid_artifact(
                        "remesh geometry-state currently admits only two-dimensional transfer",
                    ));
                }
                for digest in [
                    source_spatial_state_sha256,
                    source_geometry_state_sha256,
                    semantic_association_sha256,
                ] {
                    ArtifactDigest::from_hex(digest.clone())?;
                }
            }
        }
        validate_coordinate_payload(
            self.current_coordinates_m(),
            self.mesh_velocity_m_per_s(),
            self.spatial_dimension(),
            limits,
        )?;
        for quality in [
            &self.wire.quality_evidence.reference,
            &self.wire.quality_evidence.current,
        ] {
            validate_endpoint_quality(
                quality.minimum_mean_ratio,
                quality.minimum_signed_measure_scale,
            )?;
        }
        Ok(())
    }
}
