//! Provenance-neutral canonical identity of planar circular-hole geometry v2.

use std::collections::{BTreeMap, BTreeSet};

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use serde::{Deserialize, Serialize};

use crate::canonical::{CANONICAL_ENCODING, WireEntitySet, WireLengthUnit, digest_with_schema};
use crate::region::canonical_entity_sets;
use crate::{CanonicalGeometryLimits, EDGE_DIMENSION, FACE_DIMENSION, NamedEntitySet};

pub(crate) const CIRCULAR_HOLE_SCHEMA: &str = "eqiora.planar-circular-hole-envelope/v2";
const CORNER_COUNT: usize = 4;
const OUTER_LOOP_INDEX_COUNT: usize = 4;
const BOUNDARY_COUNT: usize = 5;
const FACE_COUNT: usize = 1;

fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_ARTIFACT, message)
}

/// Closed canonical axis-aligned rectangle-minus-circle geometry v2.
///
/// This contract contains no classification tolerance or producer provenance. A CAD producer may derive
/// this content from separately admitted construction lineage, but that
/// lineage and its graph identity are not geometry content.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CircularHoleGeometry {
    bounds: [[f64; 2]; 2],
    circle_center: [f64; 2],
    circle_radius_m: f64,
    entity_sets: Vec<NamedEntitySet>,
    bytes: Vec<u8>,
    digest: [u8; 32],
}

impl CircularHoleGeometry {
    /// Construct one exact rectangle-minus-circle geometry from semantic roles.
    ///
    /// Equal boundary names group their fixed roles into one entity set. This
    /// keeps role-to-entity wiring in the exact Geometry owner rather than in a
    /// language adapter or application demo.
    ///
    /// # Errors
    /// Returns the same diagnostics as [`Self::new`] after grouping the five
    /// boundary roles and the one full-dimensional region role.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_named_roles(
        bounds: [[f64; 2]; 2],
        circle_center: [f64; 2],
        circle_radius_m: f64,
        region: &str,
        x_lower: &str,
        x_upper: &str,
        y_lower: &str,
        y_upper: &str,
        hole: &str,
    ) -> Result<Self, Diagnostic> {
        let mut boundaries = BTreeMap::<String, Vec<usize>>::new();
        for (name, entity) in [
            (x_lower, 0),
            (x_upper, 1),
            (y_lower, 2),
            (y_upper, 3),
            (hole, 4),
        ] {
            boundaries.entry(name.to_owned()).or_default().push(entity);
        }
        let mut entity_sets = boundaries
            .into_iter()
            .map(|(name, members)| NamedEntitySet::new(name, crate::EDGE_DIMENSION, members))
            .collect::<Vec<_>>();
        entity_sets.push(NamedEntitySet::new(region, crate::FACE_DIMENSION, vec![0]));
        Self::new(bounds, circle_center, circle_radius_m, entity_sets)
    }

    pub(crate) fn new(
        mut bounds: [[f64; 2]; 2],
        mut circle_center: [f64; 2],
        circle_radius_m: f64,
        entity_sets: Vec<NamedEntitySet>,
    ) -> Result<Self, Diagnostic> {
        normalize_zeros(&mut bounds, &mut circle_center);
        validate_geometry(&bounds, &circle_center, circle_radius_m)?;
        let entity_sets =
            canonical_entity_sets(entity_sets, CORNER_COUNT, BOUNDARY_COUNT, FACE_COUNT)?;
        validate_complete_result_membership(&entity_sets)?;
        let wire = WirePlanarCircularHoleGeometryV2::from_parts(
            bounds,
            circle_center,
            circle_radius_m,
            &entity_sets,
        );
        let bytes = serde_json::to_vec(&wire).map_err(|error| {
            invalid(format!(
                "cannot serialize canonical planar circular-hole geometry v2: {error}"
            ))
        })?;
        Ok(Self {
            bounds,
            circle_center,
            circle_radius_m,
            entity_sets,
            digest: digest_with_schema(CIRCULAR_HOLE_SCHEMA, &bytes),
            bytes,
        })
    }

    /// Decode one bounded byte-for-byte canonical v2 geometry.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed or unknown data, resource excess,
    /// invalid geometry or membership, or noncanonical JSON.
    pub fn decode_canonical(
        bytes: &[u8],
        limits: CanonicalGeometryLimits,
    ) -> Result<Self, Diagnostic> {
        if bytes.len() > limits.max_bytes {
            return Err(invalid(format!(
                "planar circular-hole geometry v2 has {} bytes, exceeding the {} byte decoder limit",
                bytes.len(),
                limits.max_bytes
            )));
        }
        let wire: WirePlanarCircularHoleGeometryV2 =
            serde_json::from_slice(bytes).map_err(|error| {
                invalid(format!(
                    "invalid planar circular-hole geometry v2 JSON: {error}"
                ))
            })?;
        wire.check_contract()?;
        wire.check_limits(limits)?;
        let canonical = Self::new(
            wire.bounds,
            wire.circle.center,
            wire.circle.radius_m,
            wire.entity_sets
                .into_iter()
                .map(|set| NamedEntitySet::new(set.name, set.dimension, set.members))
                .collect(),
        )?;
        if canonical.canonical_bytes() != bytes {
            return Err(invalid(
                "planar circular-hole geometry v2 JSON is not the canonical encoding of its content",
            ));
        }
        Ok(canonical)
    }

    /// Exact Cartesian bounds, ordered x then y as `[lower, upper]` metres.
    #[must_use]
    pub const fn bounds(&self) -> &[[f64; 2]; 2] {
        &self.bounds
    }

    /// Exact circle centre in metres.
    #[must_use]
    pub const fn circle_center(&self) -> [f64; 2] {
        self.circle_center
    }

    /// Exact positive circle radius in metres.
    #[must_use]
    pub const fn circle_radius_m(&self) -> f64 {
        self.circle_radius_m
    }

    /// Canonically ordered exact named entity sets.
    #[must_use]
    pub fn entity_sets(&self) -> &[NamedEntitySet] {
        &self.entity_sets
    }

    /// Exact compact canonical JSON bytes.
    #[must_use]
    pub fn canonical_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Complete domain-separated SHA-256 identity bytes.
    #[must_use]
    pub const fn digest_bytes(&self) -> [u8; 32] {
        self.digest
    }
}

fn validate_complete_result_membership(entity_sets: &[NamedEntitySet]) -> Result<(), Diagnostic> {
    let mut covered = BTreeSet::new();
    for set in entity_sets {
        if set.dimension() != EDGE_DIMENSION && set.dimension() != FACE_DIMENSION {
            return Err(invalid(
                "planar circular-hole geometry v2 names may contain only edges or faces",
            ));
        }
        for &member in set.members() {
            if !covered.insert((set.dimension(), member)) {
                return Err(invalid(
                    "planar circular-hole geometry v2 membership must be named exactly once",
                ));
            }
        }
    }
    let expected = BTreeSet::from([
        (EDGE_DIMENSION, 0),
        (EDGE_DIMENSION, 1),
        (EDGE_DIMENSION, 2),
        (EDGE_DIMENSION, 3),
        (EDGE_DIMENSION, 4),
        (FACE_DIMENSION, 0),
    ]);
    if covered != expected {
        return Err(invalid(
            "planar circular-hole geometry v2 must name complete membership exactly once",
        ));
    }
    Ok(())
}

fn normalize_zeros(bounds: &mut [[f64; 2]; 2], circle_center: &mut [f64; 2]) {
    for coordinate in bounds.iter_mut().flatten().chain(circle_center.iter_mut()) {
        if *coordinate == 0.0 {
            *coordinate = 0.0;
        }
    }
}

fn validate_geometry(
    bounds: &[[f64; 2]; 2],
    circle_center: &[f64; 2],
    circle_radius_m: f64,
) -> Result<(), Diagnostic> {
    if bounds.iter().flatten().any(|value| !value.is_finite())
        || circle_center.iter().any(|value| !value.is_finite())
    {
        return Err(invalid(
            "planar circular-hole geometry v2 coordinates and bounds must be finite metres",
        ));
    }
    if !circle_radius_m.is_finite() || circle_radius_m <= 0.0 {
        return Err(invalid(
            "planar circular-hole geometry v2 radius must be finite and positive in metres",
        ));
    }
    for axis in 0..2 {
        let [lower, upper] = bounds[axis];
        if lower >= upper {
            return Err(invalid(
                "planar circular-hole geometry v2 rectangle bounds must increase strictly",
            ));
        }
        if !(upper - lower).is_finite() {
            return Err(invalid(
                "planar circular-hole geometry v2 rectangle span must remain finite",
            ));
        }
        let lower_distance = circle_center[axis] - lower;
        let upper_distance = upper - circle_center[axis];
        if !lower_distance.is_finite() || !upper_distance.is_finite() {
            return Err(invalid(
                "planar circular-hole geometry v2 clearance arithmetic must remain finite",
            ));
        }
        if lower_distance <= circle_radius_m || upper_distance <= circle_radius_m {
            return Err(invalid(
                "closed circle must have strict positive clearance from every rectangle side",
            ));
        }
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WirePlanarCircularHoleGeometryV2 {
    schema: String,
    encoding: String,
    kind: WirePlanarCircularHoleGeometryKindV2,
    length_unit: WireLengthUnit,
    bounds: [[f64; 2]; 2],
    circle: WireCircleV2,
    entity_sets: Vec<WireEntitySet>,
}

impl WirePlanarCircularHoleGeometryV2 {
    fn from_parts(
        bounds: [[f64; 2]; 2],
        circle_center: [f64; 2],
        circle_radius_m: f64,
        entity_sets: &[NamedEntitySet],
    ) -> Self {
        Self {
            schema: CIRCULAR_HOLE_SCHEMA.to_owned(),
            encoding: CANONICAL_ENCODING.to_owned(),
            kind: WirePlanarCircularHoleGeometryKindV2::AxisAlignedRectangleWithCircularHoleV2,
            length_unit: WireLengthUnit::Metre,
            bounds,
            circle: WireCircleV2 {
                center: circle_center,
                radius_m: circle_radius_m,
            },
            entity_sets: entity_sets.iter().map(WireEntitySet::from_set).collect(),
        }
    }

    fn check_contract(&self) -> Result<(), Diagnostic> {
        if self.schema != CIRCULAR_HOLE_SCHEMA
            || self.encoding != CANONICAL_ENCODING
            || self.kind
                != WirePlanarCircularHoleGeometryKindV2::AxisAlignedRectangleWithCircularHoleV2
            || self.length_unit != WireLengthUnit::Metre
        {
            return Err(invalid(
                "unsupported planar circular-hole geometry v2 schema, encoding, kind, or unit",
            ));
        }
        Ok(())
    }

    fn check_limits(&self, limits: CanonicalGeometryLimits) -> Result<(), Diagnostic> {
        if CORNER_COUNT > limits.max_vertices {
            return Err(invalid(
                "planar circular-hole geometry v2 vertex count exceeds decoder limits",
            ));
        }
        if FACE_COUNT > limits.max_faces {
            return Err(invalid(
                "planar circular-hole geometry v2 face count exceeds decoder limits",
            ));
        }
        if OUTER_LOOP_INDEX_COUNT > limits.max_loop_indices {
            return Err(invalid(
                "planar circular-hole geometry v2 loop-index count exceeds decoder limits",
            ));
        }
        if self.entity_sets.len() > limits.max_entity_sets {
            return Err(invalid(
                "planar circular-hole geometry v2 entity-set count exceeds decoder limits",
            ));
        }
        let members = self
            .entity_sets
            .iter()
            .try_fold(0_usize, |total, set| total.checked_add(set.members.len()));
        if members.is_none_or(|count| count > limits.max_entity_set_members) {
            return Err(invalid(
                "planar circular-hole geometry v2 entity-set member count exceeds decoder limits",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WirePlanarCircularHoleGeometryKindV2 {
    AxisAlignedRectangleWithCircularHoleV2,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireCircleV2 {
    center: [f64; 2],
    radius_m: f64,
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    use crate::{EDGE_DIMENSION, FACE_DIMENSION};
    use sha2::{Digest, Sha256};

    const EXPECTED_CANONICAL: &[u8] = concat!(
        "{\"schema\":\"eqiora.planar-circular-hole-envelope/v2\"",
        ",\"encoding\":\"eqiora.canonical-json/v1\"",
        ",\"kind\":\"axis-aligned-rectangle-with-circular-hole-v2\"",
        ",\"length_unit\":\"metre\"",
        ",\"bounds\":[[0.0,2.2],[0.0,0.41]]",
        ",\"circle\":{\"center\":[0.2,0.2],\"radius_m\":0.05}",
        ",\"entity_sets\":[{\"name\":\"cylinder\",\"dimension\":1,\"members\":[4]}",
        ",{\"name\":\"inlet\",\"dimension\":1,\"members\":[0]}",
        ",{\"name\":\"outlet\",\"dimension\":1,\"members\":[1]}",
        ",{\"name\":\"walls\",\"dimension\":1,\"members\":[2,3]}",
        ",{\"name\":\"fluid\",\"dimension\":2,\"members\":[0]}]}",
    )
    .as_bytes();
    const EXPECTED_PLAIN_SHA256: &str =
        "f9a278430c0033f2b0ec148b66d4608cf1f2b559cb46aa31ac9e7259861a26f3";
    const EXPECTED_IDENTITY: &str =
        "c1226bdfc83a5539f21ecced9afe180c60c5f4ca07a952711e3f3529213dee14";

    fn lowercase_hex(bytes: impl AsRef<[u8]>) -> String {
        bytes
            .as_ref()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    fn sets() -> Vec<NamedEntitySet> {
        vec![
            NamedEntitySet::new("fluid", FACE_DIMENSION, vec![0]),
            NamedEntitySet::new("walls", EDGE_DIMENSION, vec![3, 2]),
            NamedEntitySet::new("inlet", EDGE_DIMENSION, vec![0]),
            NamedEntitySet::new("cylinder", EDGE_DIMENSION, vec![4]),
            NamedEntitySet::new("outlet", EDGE_DIMENSION, vec![1]),
        ]
    }

    #[test]
    pub(crate) fn independent_ordinary_identity_witness_is_exact() {
        let geometry =
            CircularHoleGeometry::new([[0.0, 2.2], [0.0, 0.41]], [0.2, 0.2], 0.05, sets()).unwrap();
        assert_eq!(geometry.canonical_bytes(), EXPECTED_CANONICAL);
        assert_eq!(geometry.canonical_bytes().len(), 491);
        assert_eq!(
            lowercase_hex(Sha256::digest(geometry.canonical_bytes())),
            EXPECTED_PLAIN_SHA256
        );
        assert_eq!(lowercase_hex(geometry.digest_bytes()), EXPECTED_IDENTITY);
    }

    #[test]
    pub(crate) fn retired_tolerance_bearing_wire_is_rejected() {
        let current = std::str::from_utf8(EXPECTED_CANONICAL).unwrap();
        let retired = current
            .replace(
                "planar-circular-hole-envelope/v2",
                "planar-circular-hole-envelope/v1",
            )
            .replace("circular-hole-v2", "circular-hole-v1")
            .replace("\"bounds\":", "\"tolerance_m\":1e-12,\"bounds\":");
        assert_eq!(retired.len(), 511);
        assert!(
            CircularHoleGeometry::decode_canonical(retired.as_bytes(), Default::default()).is_err()
        );
        assert!(
            crate::CanonicalGeometryV1::replay_canonical(retired.as_bytes(), Default::default())
                .is_err()
        );
    }

    #[test]
    pub(crate) fn strict_scale_independent_geometry_replays_without_tolerance() {
        for exponent in [-40, 0, 40] {
            let scale = 2.0_f64.powi(exponent);
            let geometry = CircularHoleGeometry::new(
                [[0.0, 2.2 * scale], [0.0, 0.41 * scale]],
                [0.2 * scale, 0.2 * scale],
                0.05 * scale,
                sets(),
            )
            .unwrap();
            assert!(
                !geometry
                    .canonical_bytes()
                    .windows(b"tolerance_m".len())
                    .any(|window| window == b"tolerance_m")
            );
            assert_eq!(
                CircularHoleGeometry::decode_canonical(
                    geometry.canonical_bytes(),
                    CanonicalGeometryLimits::default(),
                )
                .unwrap(),
                geometry
            );
        }
    }

    #[test]
    pub(crate) fn finite_increasing_positive_and_strict_clearance_predicates_fail_closed() {
        let valid_bounds = [[0.0, 2.2], [0.0, 0.41]];
        let valid_center = [0.2, 0.2];
        for (bounds, center, radius) in [
            ([[f64::NAN, 2.2], [0.0, 0.41]], valid_center, 0.05),
            ([[0.0, 0.0], [0.0, 0.41]], valid_center, 0.05),
            (valid_bounds, [f64::INFINITY, 0.2], 0.05),
            (valid_bounds, valid_center, 0.0),
            (valid_bounds, valid_center, f64::INFINITY),
            ([[0.0, 1.0], [0.0, 1.0]], [0.25, 0.5], 0.25),
            ([[0.0, 1.0], [0.0, 1.0]], [1.25, 0.5], 0.25),
        ] {
            assert!(CircularHoleGeometry::new(bounds, center, radius, sets()).is_err());
        }

        let mut incomplete = sets();
        incomplete.retain(|set| set.name() != "cylinder");
        assert!(CircularHoleGeometry::new(valid_bounds, valid_center, 0.05, incomplete).is_err());

        let mut duplicate = sets();
        duplicate.push(NamedEntitySet::new("second-inlet", EDGE_DIMENSION, vec![0]));
        assert!(CircularHoleGeometry::new(valid_bounds, valid_center, 0.05, duplicate).is_err());
    }

    #[test]
    pub(crate) fn closed_v2_decoder_rejects_noncanonical_and_open_wire_mutants() {
        let geometry =
            CircularHoleGeometry::new([[0.0, 2.2], [0.0, 0.41]], [0.2, 0.2], 0.05, sets()).unwrap();
        let wire = String::from_utf8(geometry.canonical_bytes().to_vec()).unwrap();
        let reordered = serde_json::to_vec(
            &serde_json::from_slice::<serde_json::Value>(geometry.canonical_bytes()).unwrap(),
        )
        .unwrap();
        for mutant in [
            wire.replacen('{', "{\"unknown\":0,", 1).into_bytes(),
            wire.replacen('{', "{\"tolerance_m\":1e-12,", 1)
                .into_bytes(),
            wire.replacen('{', "{\"classification_tolerance_m\":1e-12,", 1)
                .into_bytes(),
            wire.replacen(
                "\"schema\":",
                "\"schema\":\"eqiora.planar-circular-hole-envelope/v2\",\"schema\":",
                1,
            )
            .into_bytes(),
            wire.replacen("\"members\":[2,3]", "\"members\":[3,2]", 1)
                .into_bytes(),
            wire.replacen("\"bounds\":[[0.0,", "\"bounds\":[[0,", 1)
                .into_bytes(),
            wire.replacen("\"bounds\":[[0.0,", "\"bounds\":[[-0.0,", 1)
                .into_bytes(),
            reordered,
        ] {
            assert_ne!(mutant, geometry.canonical_bytes());
            assert!(CircularHoleGeometry::decode_canonical(&mutant, Default::default()).is_err());
        }
    }
}
