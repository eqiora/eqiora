//! Canonical identity of the admitted authored planar geometries.
//!
//! Geometry owns the content and its identity. Artifact crates may wrap this
//! value in their own reference types, but no caller can supply a digest or an
//! entity catalog independently of the validated exact content from which
//! both are derived. The accepted straight and circular wires remain separate,
//! closed replay contracts behind one opaque public owner.

mod planar_boundary;
mod polyhedra;

use crate::convex_polyhedra::ConvexPolyhedra;
use core::fmt;

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::circular_hole::CircularHoleGeometry;
use crate::{
    CanonicalCartesianBoxGeometryV1, CanonicalPlanarAdjacentRectanglePartitionV1,
    CanonicalPlanarCircularHoleGeometryV2, CanonicalPlanarRectangleGeometryV2, EDGE_DIMENSION,
    NamedEntitySet, PlanarFace, PlanarRegion,
};

const GEOMETRY_DEFINITION_SCHEMA: &str = "eqiora.geometry-definition-envelope/v1";
pub(crate) const CANONICAL_ENCODING: &str = "eqiora.canonical-json/v1";

fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_ARTIFACT, message)
}

/// Work budgets for decoding one canonical authored geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CanonicalGeometryLimits {
    /// Maximum encoded bytes accepted by the geometry decoder.
    pub max_bytes: usize,
    /// Maximum vertices in one geometry.
    pub max_vertices: usize,
    /// Maximum faces in one geometry; polyhedral decoding also bounds volumes and facet incidences.
    pub max_faces: usize,
    /// Maximum vertex indices across all outer and hole loops.
    pub max_loop_indices: usize,
    /// Maximum named entity sets.
    pub max_entity_sets: usize,
    /// Maximum members across all named entity sets.
    pub max_entity_set_members: usize,
}

impl Default for CanonicalGeometryLimits {
    fn default() -> Self {
        Self {
            max_bytes: 4 * 1024 * 1024,
            max_vertices: 1_000_000,
            max_faces: 100_000,
            // PlanarRegion currently validates segment intersections
            // quadratically. Bound the input to that stage directly rather
            // than relying on the encoded-byte cap to make the work finite.
            max_loop_indices: 4_096,
            max_entity_sets: 100_000,
            max_entity_set_members: 4_000_000,
        }
    }
}

/// Opaque canonical content and identity of one geometry revision.
///
/// This value can only be derived from one admitted exact kind or from its
/// kind-specific bounded canonical replay. It has no public kind catalogue and
/// no constructor accepting a caller-provided digest or entity-set facts.
#[derive(Clone, PartialEq)]
pub struct CanonicalGeometryV1 {
    kind: CanonicalGeometryKind,
}

#[derive(Clone, Debug, PartialEq)]
enum CanonicalGeometryKind {
    CartesianBoxV1(CanonicalCartesianBoxGeometryV1),
    ConvexPolyhedraV1(ConvexPolyhedra),
    StraightEdgedPlanarV1 {
        region: PlanarRegion,
        bytes: Vec<u8>,
        digest: [u8; 32],
    },
    CircularHolePlanarV1(CircularHoleGeometry),
    PlanarRectangleV2(CanonicalPlanarRectangleGeometryV2),
    PlanarCircularHoleV2(CanonicalPlanarCircularHoleGeometryV2),
    PlanarAdjacentRectanglePartitionV1(CanonicalPlanarAdjacentRectanglePartitionV1),
}

impl fmt::Debug for CanonicalGeometryV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CanonicalGeometryV1")
            .field("digest", &self.digest_bytes())
            .field("ambient_dimension", &self.ambient_dimension())
            .field("topological_dimension", &self.topological_dimension())
            .finish_non_exhaustive()
    }
}

impl CanonicalGeometryV1 {
    /// Decode any current canonical Geometry family through its owning decoder.
    ///
    /// The explicit child schema selects only among concurrently admitted
    /// Geometry families; it does not infer or migrate historical generations.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed, oversized, unknown, or noncanonical bytes.
    pub fn replay_canonical(
        bytes: &[u8],
        limits: CanonicalGeometryLimits,
    ) -> Result<Self, Diagnostic> {
        if bytes.len() > limits.max_bytes {
            return Err(invalid(format!(
                "geometry definition has {} bytes, exceeding the {} byte decoder limit",
                bytes.len(),
                limits.max_bytes
            )));
        }
        let value: serde_json::Value = serde_json::from_slice(bytes)
            .map_err(|error| invalid(format!("invalid geometry definition JSON: {error}")))?;
        let schema = value
            .get("schema")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| invalid("geometry definition requires one string schema"))?;
        match schema {
            GEOMETRY_DEFINITION_SCHEMA => Self::decode_canonical(bytes, limits),
            crate::convex_polyhedra::SCHEMA => {
                ConvexPolyhedra::decode(bytes, limits).map(|geometry| Self {
                    kind: CanonicalGeometryKind::ConvexPolyhedraV1(geometry),
                })
            }
            crate::circular_hole::CIRCULAR_HOLE_SCHEMA => {
                Self::decode_circular_hole_canonical(bytes, limits)
            }
            crate::planar_rectangle_v2::SCHEMA => {
                Self::decode_planar_rectangle_v2_canonical(bytes, limits)
            }
            crate::cartesian_box_v1::SCHEMA => {
                Self::decode_cartesian_box_v1_canonical(bytes, limits)
            }
            crate::circular_hole_v2::CIRCULAR_HOLE_SCHEMA_V2 => {
                Self::decode_planar_circular_hole_v2_canonical(bytes, limits)
            }
            crate::planar_adjacent_rectangle_partition_v1::SCHEMA => {
                Self::decode_planar_adjacent_rectangle_partition_v1_canonical(bytes, limits)
            }
            _ => Err(invalid("geometry definition has an unknown current schema")),
        }
    }

    /// Dimension of the physical coordinate embedding.
    #[must_use]
    pub fn ambient_dimension(&self) -> usize {
        if matches!(self.kind, CanonicalGeometryKind::ConvexPolyhedraV1(_)) {
            3
        } else {
            self.cartesian_box_bounds().map_or(2, <[_]>::len)
        }
    }

    /// Highest topological dimension represented by the geometry.
    #[must_use]
    pub fn topological_dimension(&self) -> usize {
        self.ambient_dimension()
    }

    /// Topological dimension of one exact entity-set name.
    #[must_use]
    pub fn entity_set_dimension(&self, name: &str) -> Option<usize> {
        self.entity_set(name).map(NamedEntitySet::dimension)
    }

    #[doc(hidden)]
    #[must_use]
    pub fn selection_is_boundary_of(
        &self,
        boundary: &NamedEntitySet,
        region: &NamedEntitySet,
    ) -> bool {
        if !self.owns_selection(boundary)
            || !self.owns_selection(region)
            || boundary.dimension().checked_add(1) != Some(self.topological_dimension())
            || region.dimension() != self.topological_dimension()
        {
            return false;
        }
        match &self.kind {
            CanonicalGeometryKind::StraightEdgedPlanarV1 {
                region: topology, ..
            } => boundary
                .members()
                .iter()
                .all(|edge| edge_has_selected_parent(topology, *edge, region.members())),
            CanonicalGeometryKind::CartesianBoxV1(_) => region.members() == [0],
            CanonicalGeometryKind::ConvexPolyhedraV1(geometry) => {
                geometry.selection_is_boundary_of(boundary, region)
            }
            CanonicalGeometryKind::CircularHolePlanarV1(_)
            | CanonicalGeometryKind::PlanarCircularHoleV2(_)
            | CanonicalGeometryKind::PlanarRectangleV2(_) => region.members() == [0],
            CanonicalGeometryKind::PlanarAdjacentRectanglePartitionV1(geometry) => boundary
                .members()
                .iter()
                .all(|edge| edge_has_selected_parent(geometry.region(), *edge, region.members())),
        }
    }

    #[doc(hidden)]
    #[must_use]
    pub fn selection_dimension(&self, selection: &NamedEntitySet) -> Option<usize> {
        self.owns_selection(selection)
            .then_some(selection.dimension())
    }

    fn owns_selection(&self, selection: &NamedEntitySet) -> bool {
        self.entity_sets()
            .binary_search_by(|candidate| {
                (candidate.dimension(), candidate.name())
                    .cmp(&(selection.dimension(), selection.name()))
            })
            .is_ok_and(|index| std::ptr::eq(&self.entity_sets()[index], selection))
    }

    /// Exact Cartesian embedding of one complete box side and its parent.
    ///
    /// Primitive topology or exact authored rectangle coordinates supply the axis
    /// and outward side. Grouped, incomplete, curved, nonrectangular, foreign,
    /// or stale selections return `None`.
    #[must_use]
    pub fn cartesian_boundary_embedding(
        &self,
        boundary: &NamedEntitySet,
        parent: &NamedEntitySet,
    ) -> Option<eqiora_schema::kernel::CartesianBoundaryEmbedding> {
        use eqiora_core::{DimExponents, DynQuantity};
        use eqiora_schema::kernel::{AxisBounds, BoundarySide, CartesianBoundaryEmbedding};

        if !self.selection_is_boundary_of(boundary, parent) {
            return None;
        }
        let planar = match &self.kind {
            CanonicalGeometryKind::StraightEdgedPlanarV1 { region, .. } => Some(region),
            CanonicalGeometryKind::PlanarAdjacentRectanglePartitionV1(geometry) => {
                Some(geometry.region())
            }
            _ => None,
        };
        if let Some(region) = planar {
            let ([edge], [face]) = (boundary.members(), parent.members()) else {
                return None;
            };
            let (owner, embedding) = planar_boundary::rectangle_side(region, *edge)?;
            return (owner == *face).then_some(embedding);
        }
        let bounds: &[[f64; 2]] = match &self.kind {
            CanonicalGeometryKind::CartesianBoxV1(geometry) => geometry.bounds(),
            CanonicalGeometryKind::PlanarRectangleV2(geometry) => geometry.bounds(),
            _ => return None,
        };
        let [member] = boundary.members() else {
            return None;
        };
        // These two primitive schemas define exactly two axis-major sides.
        let axis = *member / 2;
        let side = if *member % 2 == 0 {
            BoundarySide::Lower
        } else {
            BoundarySide::Upper
        };
        let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0])?;
        let axes = bounds
            .iter()
            .map(|[lower, upper]| {
                AxisBounds::new(
                    DynQuantity::new(*lower, length),
                    DynQuantity::new(*upper, length),
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .ok()?;
        CartesianBoundaryEmbedding::derive(&axes, axis, side)
    }

    /// Exact constant parent-outward normal of one supported boundary set.
    ///
    /// A classification-free rectangle derives all four axis normals from its
    /// canonical boundary topology. The curved families expose a normal only
    /// for exact single-edge x-lower and x-upper sets. A circular member,
    /// multi-side group, and unsupported geometry return `None`. Authored planar
    /// rectangles derive normals only for complete exact parent-relative sides;
    /// callers must not infer a catalogue from entity indices.
    #[must_use]
    pub fn constant_parent_outward_normal(&self, name: &str) -> Option<[f64; 2]> {
        let set = self.entity_set(name)?;
        if set.dimension() != EDGE_DIMENSION {
            return None;
        }
        let [boundary] = set.members() else {
            return None;
        };
        match &self.kind {
            CanonicalGeometryKind::PlanarRectangleV2(_) => match boundary {
                0 => Some([-1.0, 0.0]),
                1 => Some([1.0, 0.0]),
                2 => Some([0.0, -1.0]),
                3 => Some([0.0, 1.0]),
                _ => None,
            },
            CanonicalGeometryKind::CircularHolePlanarV1(_)
            | CanonicalGeometryKind::PlanarCircularHoleV2(_) => match boundary {
                0 => Some([-1.0, 0.0]),
                1 => Some([1.0, 0.0]),
                _ => None,
            },
            CanonicalGeometryKind::StraightEdgedPlanarV1 { region, .. } => {
                planar_boundary::rectangle_normal(region, *boundary)
            }
            CanonicalGeometryKind::PlanarAdjacentRectanglePartitionV1(geometry) => {
                planar_boundary::rectangle_normal(geometry.region(), *boundary)
            }
            CanonicalGeometryKind::ConvexPolyhedraV1(_)
            | CanonicalGeometryKind::CartesianBoxV1(_) => None,
        }
    }

    #[doc(hidden)]
    #[must_use]
    pub fn selections_form_opposite_parent_interface(
        &self,
        first_boundary: &str,
        first_region: &str,
        second_boundary: &str,
        second_region: &str,
    ) -> bool {
        let Some(first_boundary) = self.entity_set(first_boundary) else {
            return false;
        };
        let Some(first_region) = self.entity_set(first_region) else {
            return false;
        };
        let Some(second_boundary) = self.entity_set(second_boundary) else {
            return false;
        };
        let Some(second_region) = self.entity_set(second_region) else {
            return false;
        };
        let topology = match &self.kind {
            CanonicalGeometryKind::ConvexPolyhedraV1(geometry) => {
                return geometry.opposite_parent_interface(
                    first_boundary,
                    first_region,
                    second_boundary,
                    second_region,
                );
            }
            CanonicalGeometryKind::StraightEdgedPlanarV1 { region, .. } => region,
            CanonicalGeometryKind::PlanarAdjacentRectanglePartitionV1(geometry) => {
                geometry.region()
            }
            _ => return false,
        };
        crate::planar_interface::opposite_parent_interface(
            topology,
            first_boundary,
            first_region,
            second_boundary,
            second_region,
        )
    }
    /// Derive canonical bytes and identity from one validated region.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical JSON serialization unexpectedly fails.
    pub fn from_region(region: &PlanarRegion) -> Result<Self, Diagnostic> {
        let wire = WireGeometryDefinitionV1::from_region(region);
        let bytes = serde_json::to_vec(&wire).map_err(|error| {
            invalid(format!(
                "cannot serialize canonical geometry definition: {error}"
            ))
        })?;
        Ok(Self {
            kind: CanonicalGeometryKind::StraightEdgedPlanarV1 {
                region: region.clone(),
                digest: digest_with_schema(GEOMETRY_DEFINITION_SCHEMA, &bytes),
                bytes,
            },
        })
    }

    /// Decode one bounded, byte-for-byte canonical geometry definition.
    ///
    /// The input is not retained. It is parsed, resource-checked, revalidated
    /// through [`PlanarRegion::new`], reconstructed from that region, and
    /// accepted only when the reconstructed bytes equal the input exactly.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed or unknown wire data, resource excess,
    /// invalid geometry, or any noncanonical encoding.
    pub fn decode_canonical(
        bytes: &[u8],
        limits: CanonicalGeometryLimits,
    ) -> Result<Self, Diagnostic> {
        if bytes.len() > limits.max_bytes {
            return Err(invalid(format!(
                "geometry definition has {} bytes, exceeding the {} byte decoder limit",
                bytes.len(),
                limits.max_bytes
            )));
        }
        let wire: WireGeometryDefinitionV1 = serde_json::from_slice(bytes)
            .map_err(|error| invalid(format!("invalid geometry definition JSON: {error}")))?;
        wire.check_contract()?;
        wire.check_limits(limits)?;
        let region = wire.into_region()?;
        let canonical = Self::from_region(&region)?;
        if canonical.canonical_bytes() != bytes {
            return Err(invalid(
                "geometry definition JSON is not the canonical encoding of its content",
            ));
        }
        Ok(canonical)
    }

    /// Construct one exact axis-aligned rectangle-minus-circle geometry.
    ///
    /// This preserves the accepted circular-hole wire and digest while the
    /// public owner remains independent of that one shape family.
    ///
    /// # Errors
    /// Returns `EQ0901` for invalid exact geometry or entity-set meaning.
    pub fn from_circular_hole(
        bounds: [[f64; 2]; 2],
        circle_center: [f64; 2],
        circle_radius_m: f64,
        entity_sets: Vec<NamedEntitySet>,
        tolerance_m: f64,
    ) -> Result<Self, Diagnostic> {
        CircularHoleGeometry::new(
            bounds,
            circle_center,
            circle_radius_m,
            entity_sets,
            tolerance_m,
        )
        .map(|geometry| Self {
            kind: CanonicalGeometryKind::CircularHolePlanarV1(geometry),
        })
    }

    /// Construct one exact rectangle-minus-circle geometry from semantic roles.
    ///
    /// Equal boundary names group their fixed roles into one entity set.
    ///
    /// # Errors
    /// Returns `EQ0901` for invalid exact geometry, tolerance, or role meaning.
    #[allow(clippy::too_many_arguments)]
    pub fn from_circular_hole_named_roles(
        bounds: [[f64; 2]; 2],
        circle_center: [f64; 2],
        circle_radius_m: f64,
        tolerance_m: f64,
        region: &str,
        x_lower: &str,
        x_upper: &str,
        y_lower: &str,
        y_upper: &str,
        hole: &str,
    ) -> Result<Self, Diagnostic> {
        CircularHoleGeometry::from_named_roles(
            bounds,
            circle_center,
            circle_radius_m,
            tolerance_m,
            region,
            x_lower,
            x_upper,
            y_lower,
            y_upper,
            hole,
        )
        .map(|geometry| Self {
            kind: CanonicalGeometryKind::CircularHolePlanarV1(geometry),
        })
    }

    /// Decode the accepted circular-hole wire without widening the straight decoder.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed, unknown, excessive, invalid, or
    /// noncanonical circular-hole bytes.
    pub fn decode_circular_hole_canonical(
        bytes: &[u8],
        limits: CanonicalGeometryLimits,
    ) -> Result<Self, Diagnostic> {
        CircularHoleGeometry::decode_canonical(bytes, limits).map(|geometry| Self {
            kind: CanonicalGeometryKind::CircularHolePlanarV1(geometry),
        })
    }

    /// Validated straight-edged region content, if this kind has one.
    ///
    /// An exact curved geometry never fabricates its numerical chordal
    /// realization as exact region meaning.
    #[must_use]
    pub const fn region(&self) -> Option<&PlanarRegion> {
        match &self.kind {
            CanonicalGeometryKind::StraightEdgedPlanarV1 { region, .. } => Some(region),
            CanonicalGeometryKind::CartesianBoxV1(_)
            | CanonicalGeometryKind::ConvexPolyhedraV1(_) => None,
            CanonicalGeometryKind::CircularHolePlanarV1(_)
            | CanonicalGeometryKind::PlanarRectangleV2(_)
            | CanonicalGeometryKind::PlanarCircularHoleV2(_) => None,
            CanonicalGeometryKind::PlanarAdjacentRectanglePartitionV1(geometry) => {
                Some(geometry.region())
            }
        }
    }

    /// Exact Cartesian bounds for the classification-free rectangle kind.
    #[must_use]
    pub const fn planar_rectangle_bounds(&self) -> Option<&[[f64; 2]; 2]> {
        match &self.kind {
            CanonicalGeometryKind::PlanarRectangleV2(geometry) => Some(geometry.bounds()),
            CanonicalGeometryKind::StraightEdgedPlanarV1 { .. }
            | CanonicalGeometryKind::ConvexPolyhedraV1(_)
            | CanonicalGeometryKind::CartesianBoxV1(_)
            | CanonicalGeometryKind::CircularHolePlanarV1(_)
            | CanonicalGeometryKind::PlanarCircularHoleV2(_)
            | CanonicalGeometryKind::PlanarAdjacentRectanglePartitionV1(_) => None,
        }
    }

    /// Exact bounds for the classification-free Cartesian box kind.
    #[must_use]
    pub fn cartesian_box_bounds(&self) -> Option<&[[f64; 2]]> {
        match &self.kind {
            CanonicalGeometryKind::CartesianBoxV1(geometry) => Some(geometry.bounds()),
            _ => None,
        }
    }

    /// Exact axis-aligned bounds for the admitted circular-hole kind.
    #[must_use]
    pub const fn circular_hole_bounds(&self) -> Option<&[[f64; 2]; 2]> {
        match &self.kind {
            CanonicalGeometryKind::StraightEdgedPlanarV1 { .. }
            | CanonicalGeometryKind::ConvexPolyhedraV1(_)
            | CanonicalGeometryKind::CartesianBoxV1(_) => None,
            CanonicalGeometryKind::CircularHolePlanarV1(geometry) => Some(geometry.bounds()),
            CanonicalGeometryKind::PlanarRectangleV2(_) => None,
            CanonicalGeometryKind::PlanarAdjacentRectanglePartitionV1(_) => None,
            CanonicalGeometryKind::PlanarCircularHoleV2(geometry) => Some(geometry.bounds()),
        }
    }

    /// Exact circle centre for the admitted circular-hole kind.
    #[must_use]
    pub const fn circular_hole_center(&self) -> Option<[f64; 2]> {
        match &self.kind {
            CanonicalGeometryKind::StraightEdgedPlanarV1 { .. }
            | CanonicalGeometryKind::ConvexPolyhedraV1(_)
            | CanonicalGeometryKind::CartesianBoxV1(_) => None,
            CanonicalGeometryKind::CircularHolePlanarV1(geometry) => Some(geometry.circle_center()),
            CanonicalGeometryKind::PlanarRectangleV2(_) => None,
            CanonicalGeometryKind::PlanarAdjacentRectanglePartitionV1(_) => None,
            CanonicalGeometryKind::PlanarCircularHoleV2(geometry) => Some(geometry.circle_center()),
        }
    }

    /// Exact circle radius for the admitted circular-hole kind.
    #[must_use]
    pub const fn circular_hole_radius_m(&self) -> Option<f64> {
        match &self.kind {
            CanonicalGeometryKind::StraightEdgedPlanarV1 { .. }
            | CanonicalGeometryKind::ConvexPolyhedraV1(_)
            | CanonicalGeometryKind::CartesianBoxV1(_) => None,
            CanonicalGeometryKind::CircularHolePlanarV1(geometry) => {
                Some(geometry.circle_radius_m())
            }
            CanonicalGeometryKind::PlanarRectangleV2(_) => None,
            CanonicalGeometryKind::PlanarAdjacentRectanglePartitionV1(_) => None,
            CanonicalGeometryKind::PlanarCircularHoleV2(geometry) => {
                Some(geometry.circle_radius_m())
            }
        }
    }

    /// Producer classification precision in metres, when this kind owns one.
    ///
    /// Frozen v1 kinds own their positive classification precision. The
    /// scale-independent v2 kind has no classification policy and returns
    /// `None`.
    #[must_use]
    pub const fn classification_tolerance_m(&self) -> Option<f64> {
        match &self.kind {
            CanonicalGeometryKind::StraightEdgedPlanarV1 { region, .. } => {
                Some(region.tolerance_m())
            }
            CanonicalGeometryKind::CircularHolePlanarV1(geometry) => Some(geometry.tolerance_m()),
            CanonicalGeometryKind::ConvexPolyhedraV1(geometry) => Some(geometry.tolerance_m()),
            CanonicalGeometryKind::CartesianBoxV1(_)
            | CanonicalGeometryKind::PlanarRectangleV2(_)
            | CanonicalGeometryKind::PlanarCircularHoleV2(_)
            | CanonicalGeometryKind::PlanarAdjacentRectanglePartitionV1(_) => None,
        }
    }

    /// Canonically ordered named exact entity sets.
    #[must_use]
    pub fn entity_sets(&self) -> &[NamedEntitySet] {
        match &self.kind {
            CanonicalGeometryKind::StraightEdgedPlanarV1 { region, .. } => region.entity_sets(),
            CanonicalGeometryKind::CartesianBoxV1(geometry) => geometry.entity_sets(),
            CanonicalGeometryKind::ConvexPolyhedraV1(geometry) => geometry.entity_sets(),
            CanonicalGeometryKind::CircularHolePlanarV1(geometry) => geometry.entity_sets(),
            CanonicalGeometryKind::PlanarRectangleV2(geometry) => geometry.entity_sets(),
            CanonicalGeometryKind::PlanarCircularHoleV2(geometry) => geometry.entity_sets(),
            CanonicalGeometryKind::PlanarAdjacentRectanglePartitionV1(geometry) => {
                geometry.entity_sets()
            }
        }
    }

    /// One exact named entity set.
    #[must_use]
    pub fn entity_set(&self, name: &str) -> Option<&NamedEntitySet> {
        self.entity_sets()
            .iter()
            .find(|candidate| candidate.name() == name)
    }

    /// Exact compact canonical JSON bytes.
    #[must_use]
    pub fn canonical_bytes(&self) -> &[u8] {
        match &self.kind {
            CanonicalGeometryKind::StraightEdgedPlanarV1 { bytes, .. } => bytes,
            CanonicalGeometryKind::CartesianBoxV1(geometry) => geometry.canonical_bytes(),
            CanonicalGeometryKind::ConvexPolyhedraV1(geometry) => geometry.canonical_bytes(),
            CanonicalGeometryKind::CircularHolePlanarV1(geometry) => geometry.canonical_bytes(),
            CanonicalGeometryKind::PlanarRectangleV2(geometry) => geometry.canonical_bytes(),
            CanonicalGeometryKind::PlanarCircularHoleV2(geometry) => geometry.canonical_bytes(),
            CanonicalGeometryKind::PlanarAdjacentRectanglePartitionV1(geometry) => {
                geometry.canonical_bytes()
            }
        }
    }

    /// Complete domain-separated SHA-256 identity bytes.
    #[must_use]
    pub const fn digest_bytes(&self) -> [u8; 32] {
        match &self.kind {
            CanonicalGeometryKind::StraightEdgedPlanarV1 { digest, .. } => *digest,
            CanonicalGeometryKind::CartesianBoxV1(geometry) => geometry.digest_bytes(),
            CanonicalGeometryKind::ConvexPolyhedraV1(geometry) => geometry.digest_bytes(),
            CanonicalGeometryKind::CircularHolePlanarV1(geometry) => geometry.digest_bytes(),
            CanonicalGeometryKind::PlanarRectangleV2(geometry) => geometry.digest_bytes(),
            CanonicalGeometryKind::PlanarCircularHoleV2(geometry) => geometry.digest_bytes(),
            CanonicalGeometryKind::PlanarAdjacentRectanglePartitionV1(geometry) => {
                geometry.digest_bytes()
            }
        }
    }

    pub(crate) const fn from_planar_circular_hole_v2(
        geometry: CanonicalPlanarCircularHoleGeometryV2,
    ) -> Self {
        Self {
            kind: CanonicalGeometryKind::PlanarCircularHoleV2(geometry),
        }
    }

    pub(crate) const fn from_cartesian_box_v1(geometry: CanonicalCartesianBoxGeometryV1) -> Self {
        Self {
            kind: CanonicalGeometryKind::CartesianBoxV1(geometry),
        }
    }

    /// Decode the exact Cartesian interval/rectangle/box wire.
    pub fn decode_cartesian_box_v1_canonical(
        bytes: &[u8],
        limits: CanonicalGeometryLimits,
    ) -> Result<Self, Diagnostic> {
        CanonicalCartesianBoxGeometryV1::decode_canonical(bytes, limits)
            .map(Self::from_cartesian_box_v1)
    }

    pub(crate) const fn from_planar_rectangle_v2(
        geometry: CanonicalPlanarRectangleGeometryV2,
    ) -> Self {
        Self {
            kind: CanonicalGeometryKind::PlanarRectangleV2(geometry),
        }
    }

    pub(crate) const fn from_planar_adjacent_rectangle_partition_v1(
        geometry: CanonicalPlanarAdjacentRectanglePartitionV1,
    ) -> Self {
        Self {
            kind: CanonicalGeometryKind::PlanarAdjacentRectanglePartitionV1(geometry),
        }
    }

    /// Exact bounds and interface coordinate for the adjacent partition kind.
    #[must_use]
    pub const fn planar_adjacent_rectangle_partition(&self) -> Option<(&[[f64; 2]; 2], f64)> {
        match &self.kind {
            CanonicalGeometryKind::PlanarAdjacentRectanglePartitionV1(geometry) => {
                Some((geometry.bounds(), geometry.interface_x()))
            }
            _ => None,
        }
    }

    /// Decode the tolerance-free adjacent rectangle partition wire.
    pub fn decode_planar_adjacent_rectangle_partition_v1_canonical(
        bytes: &[u8],
        limits: CanonicalGeometryLimits,
    ) -> Result<Self, Diagnostic> {
        CanonicalPlanarAdjacentRectanglePartitionV1::decode_canonical(bytes, limits)
            .map(Self::from_planar_adjacent_rectangle_partition_v1)
    }

    /// Decode the classification-free planar rectangle wire.
    pub fn decode_planar_rectangle_v2_canonical(
        bytes: &[u8],
        limits: CanonicalGeometryLimits,
    ) -> Result<Self, Diagnostic> {
        CanonicalPlanarRectangleGeometryV2::decode_canonical(bytes, limits)
            .map(Self::from_planar_rectangle_v2)
    }

    /// Decode the accepted tolerance-free circular-hole v2 wire.
    ///
    /// The private v2 representation remains behind this common owner.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed, unknown, excessive, invalid, or
    /// noncanonical v2 bytes.
    pub fn decode_planar_circular_hole_v2_canonical(
        bytes: &[u8],
        limits: CanonicalGeometryLimits,
    ) -> Result<Self, Diagnostic> {
        CanonicalPlanarCircularHoleGeometryV2::decode_canonical(bytes, limits)
            .map(Self::from_planar_circular_hole_v2)
    }
}

fn edge_has_selected_parent(
    topology: &PlanarRegion,
    target_edge: usize,
    selected_faces: &[usize],
) -> bool {
    let mut first_edge = 0_usize;
    for (face_index, face) in topology.faces().iter().enumerate() {
        let edge_count = face.outer().len() + face.holes().iter().map(Vec::len).sum::<usize>();
        let after_face = first_edge.saturating_add(edge_count);
        if (first_edge..after_face).contains(&target_edge) {
            return selected_faces.binary_search(&face_index).is_ok();
        }
        first_edge = after_face;
    }
    false
}

pub(crate) fn digest_with_schema(schema: &str, bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(schema.as_bytes());
    hasher.update([0]);
    hasher.update(bytes);
    hasher.finalize().into()
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
struct WireGeometryDefinitionV1 {
    schema: String,
    encoding: String,
    kind: WireGeometryKind,
    length_unit: WireLengthUnit,
    tolerance_m: f64,
    vertices: Vec<[f64; 2]>,
    faces: Vec<WireFace>,
    entity_sets: Vec<WireEntitySet>,
}

impl WireGeometryDefinitionV1 {
    fn from_region(region: &PlanarRegion) -> Self {
        Self {
            schema: GEOMETRY_DEFINITION_SCHEMA.to_owned(),
            encoding: CANONICAL_ENCODING.to_owned(),
            kind: WireGeometryKind::StraightEdgedPlanarV1,
            length_unit: WireLengthUnit::Metre,
            tolerance_m: region.tolerance_m(),
            vertices: region.vertices().to_vec(),
            faces: region
                .faces()
                .iter()
                .map(|face| WireFace {
                    outer: face.outer().to_vec(),
                    holes: face.holes().to_vec(),
                })
                .collect(),
            entity_sets: region
                .entity_sets()
                .iter()
                .map(WireEntitySet::from_set)
                .collect(),
        }
    }

    fn check_contract(&self) -> Result<(), Diagnostic> {
        if self.schema != GEOMETRY_DEFINITION_SCHEMA
            || self.encoding != CANONICAL_ENCODING
            || self.kind != WireGeometryKind::StraightEdgedPlanarV1
            || self.length_unit != WireLengthUnit::Metre
        {
            return Err(invalid(
                "unsupported geometry definition schema, encoding, kind, or unit",
            ));
        }
        Ok(())
    }

    fn check_limits(&self, limits: CanonicalGeometryLimits) -> Result<(), Diagnostic> {
        if self.vertices.len() > limits.max_vertices {
            return Err(invalid("geometry vertex count exceeds decoder limits"));
        }
        if self.faces.len() > limits.max_faces {
            return Err(invalid("geometry face count exceeds decoder limits"));
        }
        let loop_indices = self.faces.iter().try_fold(0_usize, |total, face| {
            std::iter::once(&face.outer)
                .chain(face.holes.iter())
                .try_fold(total, |total, loop_indices| {
                    total.checked_add(loop_indices.len())
                })
        });
        if loop_indices.is_none_or(|count| count > limits.max_loop_indices) {
            return Err(invalid("geometry loop-index count exceeds decoder limits"));
        }
        if self.entity_sets.len() > limits.max_entity_sets {
            return Err(invalid("geometry entity-set count exceeds decoder limits"));
        }
        let entity_set_members = self
            .entity_sets
            .iter()
            .try_fold(0_usize, |total, set| total.checked_add(set.members.len()));
        if entity_set_members.is_none_or(|count| count > limits.max_entity_set_members) {
            return Err(invalid(
                "geometry entity-set member count exceeds decoder limits",
            ));
        }
        Ok(())
    }

    fn into_region(self) -> Result<PlanarRegion, Diagnostic> {
        PlanarRegion::new(
            self.vertices,
            self.faces
                .into_iter()
                .map(|face| PlanarFace::new(face.outer, face.holes))
                .collect(),
            self.entity_sets
                .into_iter()
                .map(|set| NamedEntitySet::new(set.name, set.dimension, set.members))
                .collect(),
            self.tolerance_m,
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireGeometryKind {
    StraightEdgedPlanarV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum WireLengthUnit {
    Metre,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
struct WireFace {
    outer: Vec<usize>,
    holes: Vec<Vec<usize>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub(crate) struct WireEntitySet {
    pub(crate) name: String,
    pub(crate) dimension: usize,
    pub(crate) members: Vec<usize>,
}

impl WireEntitySet {
    pub(crate) fn from_set(set: &NamedEntitySet) -> Self {
        Self {
            name: set.name().to_owned(),
            dimension: set.dimension(),
            members: set.members().to_vec(),
        }
    }
}
