//! Immutable provider-neutral authored CAD meaning.
//!
//! One opaque Rust owner replays two closed persisted operation histories:
//! the frozen rectangle-extrusion v1 wire and the bounded circular-through-cut
//! v2 wire.  The private sum is not a public feature enum or general B-rep.

pub(super) mod wire;
use wire::{WireClosedFace, WirePositiveZExtrusion, WireRectangleProfile, WireSketchPlane};

use std::f64::consts::PI;

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use serde::{Deserialize, Serialize};

use crate::cad_authored_cut::{CircularThroughCut, GRAPH_SCHEMA_V2, decode_v2, encode_v2};
use crate::cad_authored_selection::{FaceKey, GeometryFaceHandle, WireFaceSelectionV1};
use crate::canonical::{CANONICAL_ENCODING, WireLengthUnit, digest_with_schema};
use crate::{AxisAlignedBox3, CadRepairDispositionV1, ConstrainedRectangleV1};

const GRAPH_SCHEMA_V1: &str = "eqiora.cad-authored-operation-graph-envelope/v1";
const MAX_GRAPH_BYTES: usize = 4_096;

fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_ARTIFACT, message)
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct GraphCore {
    sketch: ConstrainedRectangleV1,
    extrusion_depth_m: f64,
    requested_modeling_tolerance_m: f64,
    bounds: AxisAlignedBox3,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum GraphKind {
    RectangleExtrusion,
    CircularThroughCut(CircularThroughCut),
}

/// Common immutable owner of the admitted authored-CAD operation histories.
///
/// Persisted v1 and v2 schemas remain closed private variants.  Existing v1
/// bytes and digests are reproduced exactly while callers use one feature-
/// neutral Rust surface before Python and Studio projection.
#[derive(Clone, Debug)]
pub struct GeometrySolidOperation {
    authoring_owner: u64,
    core: GraphCore,
    kind: GraphKind,
    bytes: Vec<u8>,
    digest: [u8; 32],
}

impl PartialEq for GeometrySolidOperation {
    fn eq(&self, other: &Self) -> bool {
        self.core == other.core
            && self.kind == other.kind
            && self.bytes == other.bytes
            && self.digest == other.digest
    }
}

impl GeometrySolidOperation {
    /// Construct the frozen rectangle → face → positive-z extrusion history.
    ///
    /// The requested modeling tolerance is identity-only.  It is never used
    /// as a coordinate offset, classification tolerance, or repair policy.
    ///
    /// # Errors
    /// Returns `EQ0901` for a non-positive/non-finite depth or tolerance, a
    /// non-finite derived extent, or unexpected canonical serialization.
    pub(crate) fn from_rectangle(
        authoring_owner: u64,
        sketch: ConstrainedRectangleV1,
        extrusion_depth_m: f64,
        requested_modeling_tolerance_m: f64,
    ) -> Result<Self, Diagnostic> {
        let core = validate_extrusion(sketch, extrusion_depth_m, requested_modeling_tolerance_m)?;
        let wire = WireGeometrySolidOperationV1::from_core(core);
        let bytes = serde_json::to_vec(&wire)
            .map_err(|error| invalid(format!("cannot serialize authored CAD graph: {error}")))?;
        Ok(Self::from_encoded(
            authoring_owner,
            core,
            GraphKind::RectangleExtrusion,
            GRAPH_SCHEMA_V1,
            bytes,
        ))
    }

    /// Append the one admitted circular through-all difference operation.
    ///
    /// The cut starts on the predecessor end cap and proceeds through all in
    /// negative z.  Signed inward side clearance must exceed the requested
    /// Boolean tolerance; no tolerance substitution or healing is available.
    ///
    /// # Errors
    /// Returns `EQ0901` unless the predecessor is the closed v1 history and
    /// the finite positive circle lies strictly inside every rectangle side by
    /// more than the requested Boolean tolerance.
    pub(crate) fn append_circular_through_cut(
        &self,
        center_m: [f64; 2],
        radius_m: f64,
        requested_boolean_tolerance_m: f64,
    ) -> Result<Self, Diagnostic> {
        if !matches!(self.kind, GraphKind::RectangleExtrusion) {
            return Err(invalid(
                "authored CAD v2 admits exactly one cut after the rectangle extrusion",
            ));
        }
        let circle = crate::cad_authored_cut::AdmittedCircle::new(center_m, radius_m)?;
        let cut = CircularThroughCut::new(self.core.sketch, circle, requested_boolean_tolerance_m)?;
        let bytes = encode_v2(
            self.core.sketch,
            self.core.extrusion_depth_m,
            self.core.requested_modeling_tolerance_m,
            cut,
        )?;
        Ok(Self::from_encoded(
            self.authoring_owner,
            self.core,
            GraphKind::CircularThroughCut(cut),
            GRAPH_SCHEMA_V2,
            bytes,
        ))
    }

    /// Decode either closed schema and reconstruct its one canonical byte form.
    ///
    /// Object-member order and equivalent numeric spellings are nonsemantic;
    /// duplicate/unknown members and unsupported dependencies reject.
    ///
    /// # Errors
    /// Returns `EQ0901` for excess bytes or a malformed/unsupported graph.
    pub(crate) fn decode_for_owner(authoring_owner: u64, bytes: &[u8]) -> Result<Self, Diagnostic> {
        if bytes.len() > MAX_GRAPH_BYTES {
            return Err(invalid(format!(
                "authored CAD graph has {} bytes, exceeding the {MAX_GRAPH_BYTES} byte decoder limit",
                bytes.len()
            )));
        }
        if let Ok(wire) = serde_json::from_slice::<WireGeometrySolidOperationV1>(bytes) {
            wire.check_contract()?;
            let sketch = ConstrainedRectangleV1::new(
                (wire.profile.x_bounds_m[0], wire.profile.x_bounds_m[1]),
                (wire.profile.y_bounds_m[0], wire.profile.y_bounds_m[1]),
                wire.sketch_plane.z_m,
            )?;
            return Self::from_rectangle(
                authoring_owner,
                sketch,
                wire.extrusion.depth_m,
                wire.requested_modeling_tolerance_m,
            );
        }
        if let Ok(decoded) = decode_v2(bytes) {
            let base = Self::from_rectangle(
                authoring_owner,
                decoded.sketch,
                decoded.extrusion_depth_m,
                decoded.requested_modeling_tolerance_m,
            )?;
            return base.append_circular_through_cut(
                decoded.center_m,
                decoded.radius_m,
                decoded.requested_boolean_tolerance_m,
            );
        }
        Err(invalid("unsupported or malformed authored CAD graph wire"))
    }

    fn from_encoded(
        authoring_owner: u64,
        core: GraphCore,
        kind: GraphKind,
        schema: &str,
        bytes: Vec<u8>,
    ) -> Self {
        Self {
            authoring_owner,
            core,
            kind,
            digest: digest_with_schema(schema, &bytes),
            bytes,
        }
    }

    pub(crate) const fn authoring_owner(&self) -> u64 {
        self.authoring_owner
    }

    /// Fully constrained rectangle owned by the graph.
    #[must_use]
    pub const fn sketch(&self) -> ConstrainedRectangleV1 {
        self.core.sketch
    }

    /// Positive-z extrusion depth in metres.
    #[must_use]
    pub const fn extrusion_depth_m(&self) -> f64 {
        self.core.extrusion_depth_m
    }

    /// Identity-bearing base modeling tolerance in metres.
    #[must_use]
    pub const fn requested_modeling_tolerance_m(&self) -> f64 {
        self.core.requested_modeling_tolerance_m
    }

    /// Requested Boolean tolerance, absent from the rectangle-only history.
    #[must_use]
    pub const fn requested_boolean_tolerance_m(&self) -> Option<f64> {
        match self.kind {
            GraphKind::RectangleExtrusion => None,
            GraphKind::CircularThroughCut(cut) => Some(cut.requested_tolerance_m()),
        }
    }

    /// Circular cut centre, absent from the rectangle-only history.
    #[must_use]
    pub const fn cut_center_m(&self) -> Option<[f64; 2]> {
        match self.kind {
            GraphKind::RectangleExtrusion => None,
            GraphKind::CircularThroughCut(cut) => Some(cut.center_m()),
        }
    }

    /// Circular cut radius, absent from the rectangle-only history.
    #[must_use]
    pub const fn cut_radius_m(&self) -> Option<f64> {
        match self.kind {
            GraphKind::RectangleExtrusion => None,
            GraphKind::CircularThroughCut(cut) => Some(cut.radius_m()),
        }
    }

    /// Exact outer analytic bounds; the through-cut does not change them.
    #[must_use]
    pub const fn output(&self) -> AxisAlignedBox3 {
        self.core.bounds
    }

    /// Neither admitted history performs healing or topology repair.
    #[must_use]
    pub const fn repair_disposition(&self) -> CadRepairDispositionV1 {
        CadRepairDispositionV1::None
    }

    /// Exact compact canonical JSON bytes for the graph's closed wire variant.
    #[must_use]
    pub fn canonical_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Complete domain-separated authored-graph identity.
    #[must_use]
    pub const fn digest_bytes(&self) -> [u8; 32] {
        self.digest
    }

    /// Outer rectangle-extrusion corners in lower-then-upper order.
    #[must_use]
    pub fn vertices_m(&self) -> [[f64; 3]; 8] {
        let [(x0, x1), (y0, y1), (z0, z1)] = self.core.bounds.bounds_m();
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ]
    }

    /// Exact vertex count only where the analytic representation is closed.
    #[must_use]
    pub const fn vertex_count(&self) -> Option<usize> {
        if matches!(self.kind, GraphKind::RectangleExtrusion) {
            Some(8)
        } else {
            None
        }
    }

    /// Exact edge count only where the analytic representation is closed.
    #[must_use]
    pub const fn edge_count(&self) -> Option<usize> {
        if matches!(self.kind, GraphKind::RectangleExtrusion) {
            Some(12)
        } else {
            None
        }
    }

    /// Exact face count.
    #[must_use]
    pub const fn face_count(&self) -> usize {
        if matches!(self.kind, GraphKind::RectangleExtrusion) {
            6
        } else {
            7
        }
    }

    /// Number of connected closed shells.
    #[must_use]
    pub const fn closed_shell_count(&self) -> usize {
        1
    }

    /// Number of solid bodies.
    #[must_use]
    pub const fn body_count(&self) -> usize {
        1
    }

    /// Exact body genus.
    #[must_use]
    pub const fn genus(&self) -> usize {
        if matches!(self.kind, GraphKind::RectangleExtrusion) {
            0
        } else {
            1
        }
    }

    /// Exact analytic body volume in cubic metres.
    #[must_use]
    pub fn volume_m3(&self) -> f64 {
        let [(x0, x1), (y0, y1), (z0, z1)] = self.core.bounds.bounds_m();
        let outer = (x1 - x0) * (y1 - y0) * (z1 - z0);
        match self.kind {
            GraphKind::RectangleExtrusion => outer,
            GraphKind::CircularThroughCut(cut) => {
                outer - PI * cut.radius_m().powi(2) * self.core.extrusion_depth_m
            }
        }
    }

    /// Exact analytic total surface area in square metres.
    #[must_use]
    pub fn surface_area_m2(&self) -> f64 {
        self.face_inventory()
            .iter()
            .copied()
            .map(|selection| self.face_area_for(selection))
            .sum()
    }

    fn face_inventory(&self) -> &'static [FaceKey] {
        match self.kind {
            GraphKind::RectangleExtrusion => &FaceKey::V1_ALL,
            GraphKind::CircularThroughCut(_) => &FaceKey::V2_ALL,
        }
    }

    /// Every admitted graph-bound face handle in canonical provenance order.
    ///
    /// # Errors
    /// Returns `EQ0901` only for an unexpected canonical handle serialization
    /// failure.
    pub fn face_handles(&self) -> Result<Vec<GeometryFaceHandle>, Diagnostic> {
        self.face_inventory()
            .iter()
            .copied()
            .map(|face| self.face_handle_for(face))
            .collect()
    }

    /// Bind one stable provenance key to this exact graph identity.
    ///
    /// # Errors
    /// Returns `EQ0901` for an unknown or inadmissible provenance key or an
    /// unexpected canonical handle serialization failure.
    pub fn face_handle(&self, provenance_key: &str) -> Result<GeometryFaceHandle, Diagnostic> {
        let face = FaceKey::from_provenance_key(provenance_key)
            .filter(|candidate| self.face_inventory().contains(candidate))
            .ok_or_else(|| {
                invalid(format!(
                    "authored CAD graph has no provenance selection named {provenance_key:?}",
                ))
            })?;
        self.face_handle_for(face)
    }

    pub(crate) fn face_handle_for(&self, face: FaceKey) -> Result<GeometryFaceHandle, Diagnostic> {
        if !self.face_inventory().contains(&face) {
            return Err(invalid("face is not admitted by this authored graph"));
        }
        match self.kind {
            GraphKind::RectangleExtrusion => {
                GeometryFaceHandle::bind_v1(self.authoring_owner, self.digest, face)
            }
            GraphKind::CircularThroughCut(_) => {
                GeometryFaceHandle::bind_v2(self.authoring_owner, self.digest, face)
            }
        }
    }

    /// Validate one graph-bound handle, then reveal its stable provenance key.
    ///
    /// # Errors
    /// Returns `EQ0901` before lookup for any foreign/stale graph digest or
    /// handle schema, then rejects a face outside this graph inventory.
    pub fn resolve_face(&self, handle: &GeometryFaceHandle) -> Result<&'static str, Diagnostic> {
        Ok(self.resolve_face_key(handle)?.provenance_key())
    }

    pub(crate) fn resolve_face_key(
        &self,
        handle: &GeometryFaceHandle,
    ) -> Result<FaceKey, Diagnostic> {
        let expected_v1 = matches!(self.kind, GraphKind::RectangleExtrusion);
        if handle.authoring_owner() != self.authoring_owner
            || handle.graph_digest_bytes() != self.digest
            || handle.is_v1() != expected_v1
        {
            return Err(invalid(
                "CAD face handle belongs to a foreign authored graph identity or wire variant",
            ));
        }
        let face = handle.face_key();
        if !self.face_inventory().contains(&face) {
            return Err(invalid(
                "CAD face handle selection is absent from this graph",
            ));
        }
        Ok(face)
    }

    /// Exact face area after validating the graph-bound handle.
    pub fn face_area_m2(&self, handle: &GeometryFaceHandle) -> Result<f64, Diagnostic> {
        Ok(self.face_area_for(self.resolve_face_key(handle)?))
    }

    /// Exact number of analytic boundary loops on the selected face.
    pub fn face_boundary_loop_count(
        &self,
        handle: &GeometryFaceHandle,
    ) -> Result<usize, Diagnostic> {
        let selection = self.resolve_face_key(handle)?;
        Ok(match self.kind {
            GraphKind::CircularThroughCut(_)
                if selection == FaceKey::start_cap()
                    || selection == FaceKey::end_cap()
                    || selection == FaceKey::cut_wall() =>
            {
                2
            }
            _ => 1,
        })
    }

    /// Four outward-oriented vertices for a rectangular face, when applicable.
    pub fn rectangular_face_vertices_m(
        &self,
        handle: &GeometryFaceHandle,
    ) -> Result<Option<[[f64; 3]; 4]>, Diagnostic> {
        let selection = self.resolve_face_key(handle)?;
        Ok(self.rectangular_face_cycle(selection))
    }

    /// Centroid of a rectangular selected face, when applicable.
    pub fn rectangular_face_centroid_m(
        &self,
        handle: &GeometryFaceHandle,
    ) -> Result<Option<[f64; 3]>, Diagnostic> {
        Ok(self.rectangular_face_vertices_m(handle)?.map(centroid))
    }

    /// Constant outward unit normal for a planar selected face.
    pub fn planar_face_outward_normal(
        &self,
        handle: &GeometryFaceHandle,
    ) -> Result<Option<[f64; 3]>, Diagnostic> {
        let selection = self.resolve_face_key(handle)?;
        Ok(if selection == FaceKey::cut_wall() {
            None
        } else {
            Some(planar_normal(selection))
        })
    }

    fn face_area_for(&self, selection: FaceKey) -> f64 {
        let [(x0, x1), (y0, y1), (z0, z1)] = self.core.bounds.bounds_m();
        let width = x1 - x0;
        let height = y1 - y0;
        let depth = z1 - z0;
        if selection == FaceKey::start_cap() || selection == FaceKey::end_cap() {
            let outer = width * height;
            return match self.kind {
                GraphKind::RectangleExtrusion => outer,
                GraphKind::CircularThroughCut(cut) => outer - PI * cut.radius_m().powi(2),
            };
        }
        if selection == FaceKey::profile_x_lower() || selection == FaceKey::profile_x_upper() {
            return height * depth;
        }
        if selection == FaceKey::profile_y_lower() || selection == FaceKey::profile_y_upper() {
            return width * depth;
        }
        match self.kind {
            GraphKind::CircularThroughCut(cut) => 2.0 * PI * cut.radius_m() * depth,
            GraphKind::RectangleExtrusion => unreachable!("inventory excludes cut wall"),
        }
    }

    fn rectangular_face_cycle(&self, selection: FaceKey) -> Option<[[f64; 3]; 4]> {
        if matches!(self.kind, GraphKind::CircularThroughCut(_))
            && (selection == FaceKey::start_cap()
                || selection == FaceKey::end_cap()
                || selection == FaceKey::cut_wall())
        {
            return None;
        }
        let [a, b, c, d, upper_a, upper_b, upper_c, upper_d] = self.vertices_m();
        Some(if selection == FaceKey::start_cap() {
            [a, d, c, b]
        } else if selection == FaceKey::end_cap() {
            [upper_a, upper_b, upper_c, upper_d]
        } else if selection == FaceKey::profile_x_lower() {
            [a, upper_a, upper_d, d]
        } else if selection == FaceKey::profile_x_upper() {
            [b, c, upper_c, upper_b]
        } else if selection == FaceKey::profile_y_lower() {
            [a, b, upper_b, upper_a]
        } else {
            [d, upper_d, upper_c, c]
        })
    }

    pub(crate) const fn is_cut(&self) -> bool {
        matches!(self.kind, GraphKind::CircularThroughCut(_))
    }

    pub(crate) fn planar_cut_parts(&self) -> Option<([[f64; 2]; 2], [f64; 2], f64)> {
        let GraphKind::CircularThroughCut(cut) = self.kind else {
            return None;
        };
        let (x_lower_m, x_upper_m) = self.core.sketch.x_bounds_m();
        let (y_lower_m, y_upper_m) = self.core.sketch.y_bounds_m();
        Some((
            [[x_lower_m, x_upper_m], [y_lower_m, y_upper_m]],
            cut.center_m(),
            cut.radius_m(),
        ))
    }
}

fn validate_extrusion(
    sketch: ConstrainedRectangleV1,
    extrusion_depth_m: f64,
    requested_modeling_tolerance_m: f64,
) -> Result<GraphCore, Diagnostic> {
    if !requested_modeling_tolerance_m.is_finite() || requested_modeling_tolerance_m <= 0.0 {
        return Err(invalid(
            "authored CAD modeling tolerance must be finite and positive in metres",
        ));
    }
    if !extrusion_depth_m.is_finite() || extrusion_depth_m <= 0.0 {
        return Err(invalid(
            "authored CAD extrusion depth must be finite and positive in metres",
        ));
    }
    let extrusion_depth_m = canonical_zero(extrusion_depth_m);
    let bounds = sketch.extruded_box(extrusion_depth_m)?;
    if bounds
        .bounds_m()
        .iter()
        .any(|&(lower, upper)| !(upper - lower).is_finite())
    {
        return Err(invalid(
            "authored CAD derived spans must remain finite in metres",
        ));
    }
    Ok(GraphCore {
        sketch,
        extrusion_depth_m,
        requested_modeling_tolerance_m: canonical_zero(requested_modeling_tolerance_m),
        bounds,
    })
}

fn centroid(vertices: [[f64; 3]; 4]) -> [f64; 3] {
    core::array::from_fn(|axis| {
        canonical_zero(vertices.iter().map(|vertex| vertex[axis]).sum::<f64>() / 4.0)
    })
}

fn planar_normal(selection: FaceKey) -> [f64; 3] {
    if selection == FaceKey::start_cap() {
        [0.0, 0.0, -1.0]
    } else if selection == FaceKey::end_cap() {
        [0.0, 0.0, 1.0]
    } else if selection == FaceKey::profile_x_lower() {
        [-1.0, 0.0, 0.0]
    } else if selection == FaceKey::profile_x_upper() {
        [1.0, 0.0, 0.0]
    } else if selection == FaceKey::profile_y_lower() {
        [0.0, -1.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    }
}

const fn canonical_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireGeometrySolidOperationV1 {
    schema: String,
    encoding: String,
    length_unit: WireLengthUnit,
    requested_modeling_tolerance_m: f64,
    sketch_plane: WireSketchPlane,
    profile: WireRectangleProfile,
    face: WireClosedFace,
    extrusion: WirePositiveZExtrusion,
    selections: Vec<WireFaceSelectionV1>,
}

impl WireGeometrySolidOperationV1 {
    fn from_core(core: GraphCore) -> Self {
        let (sketch_plane, profile, face, extrusion) =
            wire::rectangle_extrusion(core.sketch, core.extrusion_depth_m);
        Self {
            schema: GRAPH_SCHEMA_V1.to_owned(),
            encoding: CANONICAL_ENCODING.to_owned(),
            length_unit: WireLengthUnit::Metre,
            requested_modeling_tolerance_m: core.requested_modeling_tolerance_m,
            sketch_plane,
            profile,
            face,
            extrusion,
            selections: FaceKey::V1_ALL
                .map(|selection| {
                    WireFaceSelectionV1::try_from(selection).expect("v1 inventory is closed")
                })
                .to_vec(),
        }
    }

    fn check_contract(&self) -> Result<(), Diagnostic> {
        let expected = FaceKey::V1_ALL
            .map(|selection| WireFaceSelectionV1::try_from(selection).expect("closed v1"));
        if self.schema != GRAPH_SCHEMA_V1
            || self.encoding != CANONICAL_ENCODING
            || self.length_unit != WireLengthUnit::Metre
            || !wire::has_rectangle_extrusion_chain(
                &self.sketch_plane,
                &self.profile,
                &self.face,
                &self.extrusion,
            )
            || self.selections.as_slice() != expected.as_slice()
        {
            return Err(invalid(
                "unsupported authored CAD v1 schema, dependency chain, selection inventory, unit, or repair disposition",
            ));
        }
        Ok(())
    }
}
