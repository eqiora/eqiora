//! Private closed wire and analytic inputs for authored graph schema v2.

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use serde::{Deserialize, Serialize};

use crate::ConstrainedRectangleV1;
use crate::cad_authored_graph::wire::{
    self, EXTRUSION_ID, WireClosedFace, WireConstraint, WirePositiveZExtrusion,
    WireRectangleProfile, WireRepairDisposition, WireSketchPlane,
};
use crate::cad_authored_selection::{FaceKey, WireFaceSelectionV2};
use crate::canonical::{CANONICAL_ENCODING, WireLengthUnit};

pub(crate) const GRAPH_SCHEMA_V2: &str = "eqiora.cad-authored-operation-graph-envelope/v2";

const CUT_SKETCH_PLANE_ID: &str = "cut-sketch-plane";
const CUT_PROFILE_ID: &str = "circle-profile";
const CUT_FACE_ID: &str = "cut-profile-face";
const CUT_ID: &str = "circular-through-cut";

fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_ARTIFACT, message)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct AdmittedCircle {
    center_m: [f64; 2],
    radius_m: f64,
}

impl AdmittedCircle {
    pub(crate) fn new(mut center_m: [f64; 2], radius_m: f64) -> Result<Self, Diagnostic> {
        for coordinate in &mut center_m {
            *coordinate = canonical_zero(*coordinate);
        }
        if center_m.iter().any(|value| !value.is_finite()) {
            return Err(invalid(
                "authored CAD cut centre must contain finite metres",
            ));
        }
        if !radius_m.is_finite() || radius_m <= 0.0 {
            return Err(invalid(
                "authored CAD cut radius must be finite and positive in metres",
            ));
        }
        Ok(Self {
            center_m,
            radius_m: canonical_zero(radius_m),
        })
    }

    pub(crate) const fn center_m(self) -> [f64; 2] {
        self.center_m
    }

    pub(crate) const fn radius_m(self) -> f64 {
        self.radius_m
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CircularThroughCut {
    circle: AdmittedCircle,
    requested_tolerance_m: f64,
}

impl CircularThroughCut {
    pub(crate) fn new(
        sketch: ConstrainedRectangleV1,
        circle: AdmittedCircle,
        requested_tolerance_m: f64,
    ) -> Result<Self, Diagnostic> {
        if !requested_tolerance_m.is_finite() || requested_tolerance_m <= 0.0 {
            return Err(invalid(
                "authored CAD Boolean tolerance must be finite and positive in metres",
            ));
        }
        let center_m = circle.center_m();
        let [(x0, x1), (y0, y1)] = [sketch.x_bounds_m(), sketch.y_bounds_m()];
        let signed_side_distances = [
            center_m[0] - x0,
            x1 - center_m[0],
            center_m[1] - y0,
            y1 - center_m[1],
        ];
        if signed_side_distances.iter().any(|value| !value.is_finite()) {
            return Err(invalid(
                "authored CAD cut clearance arithmetic must remain finite",
            ));
        }
        let minimum = signed_side_distances
            .into_iter()
            .reduce(f64::min)
            .expect("four signed distances");
        let clearance = minimum - circle.radius_m();
        if !clearance.is_finite() || clearance <= requested_tolerance_m {
            return Err(invalid(
                "authored circular cut must remain inside every rectangle side by more than the requested Boolean tolerance",
            ));
        }
        Ok(Self {
            circle,
            requested_tolerance_m: canonical_zero(requested_tolerance_m),
        })
    }

    pub(crate) const fn center_m(self) -> [f64; 2] {
        self.circle.center_m()
    }

    pub(crate) const fn radius_m(self) -> f64 {
        self.circle.radius_m()
    }

    pub(crate) const fn requested_tolerance_m(self) -> f64 {
        self.requested_tolerance_m
    }
}

pub(crate) fn encode_v2(
    sketch: ConstrainedRectangleV1,
    depth_m: f64,
    requested_modeling_tolerance_m: f64,
    cut: CircularThroughCut,
) -> Result<Vec<u8>, Diagnostic> {
    serde_json::to_vec(&WireGeometrySolidOperationV2::from_parts(
        sketch,
        depth_m,
        requested_modeling_tolerance_m,
        cut,
    ))
    .map_err(|error| invalid(format!("cannot serialize authored CAD graph v2: {error}")))
}

pub(crate) struct DecodedGraphV2 {
    pub(crate) sketch: ConstrainedRectangleV1,
    pub(crate) extrusion_depth_m: f64,
    pub(crate) requested_modeling_tolerance_m: f64,
    pub(crate) center_m: [f64; 2],
    pub(crate) radius_m: f64,
    pub(crate) requested_boolean_tolerance_m: f64,
}

pub(crate) fn decode_v2(bytes: &[u8]) -> Result<DecodedGraphV2, Diagnostic> {
    let wire: WireGeometrySolidOperationV2 = serde_json::from_slice(bytes)
        .map_err(|error| invalid(format!("invalid authored CAD graph v2 JSON: {error}")))?;
    wire.check_contract()?;
    let sketch = ConstrainedRectangleV1::new(
        (wire.profile.x_bounds_m[0], wire.profile.x_bounds_m[1]),
        (wire.profile.y_bounds_m[0], wire.profile.y_bounds_m[1]),
        wire.sketch_plane.z_m,
    )?;
    Ok(DecodedGraphV2 {
        sketch,
        extrusion_depth_m: wire.extrusion.depth_m,
        requested_modeling_tolerance_m: wire.requested_modeling_tolerance_m,
        center_m: wire.cut_profile.center_m,
        radius_m: wire.cut_profile.radius_m,
        requested_boolean_tolerance_m: wire.cut.requested_tolerance_m,
    })
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireGeometrySolidOperationV2 {
    schema: String,
    encoding: String,
    length_unit: WireLengthUnit,
    requested_modeling_tolerance_m: f64,
    sketch_plane: WireSketchPlane,
    profile: WireRectangleProfile,
    face: WireClosedFace,
    extrusion: WirePositiveZExtrusion,
    cut_sketch_plane: WireCutSketchPlane,
    cut_profile: WireCircleProfile,
    cut_face: WireClosedFace,
    cut: WireCircularThroughCut,
    selections: Vec<WireFaceSelectionV2>,
}

impl WireGeometrySolidOperationV2 {
    fn from_parts(
        sketch: ConstrainedRectangleV1,
        depth_m: f64,
        requested_modeling_tolerance_m: f64,
        cut: CircularThroughCut,
    ) -> Self {
        let (sketch_plane, profile, face, extrusion) = wire::rectangle_extrusion(sketch, depth_m);
        Self {
            schema: GRAPH_SCHEMA_V2.to_owned(),
            encoding: CANONICAL_ENCODING.to_owned(),
            length_unit: WireLengthUnit::Metre,
            requested_modeling_tolerance_m,
            sketch_plane,
            profile,
            face,
            extrusion,
            cut_sketch_plane: WireCutSketchPlane {
                id: CUT_SKETCH_PLANE_ID.to_owned(),
                kind: WireCutSketchPlaneKind::OnFace,
                face: "end-cap".to_owned(),
            },
            cut_profile: WireCircleProfile {
                id: CUT_PROFILE_ID.to_owned(),
                kind: WireCircleKind::Circle,
                sketch_plane: CUT_SKETCH_PLANE_ID.to_owned(),
                constraint: WireConstraint::ClosedByConstruction,
                center_m: cut.center_m(),
                radius_m: cut.radius_m(),
            },
            cut_face: WireClosedFace::new(CUT_FACE_ID, CUT_PROFILE_ID),
            cut: WireCircularThroughCut {
                id: CUT_ID.to_owned(),
                kind: WireCutKind::DifferenceThroughAllNegativeZ,
                target: EXTRUSION_ID.to_owned(),
                tool_face: CUT_FACE_ID.to_owned(),
                requested_tolerance_m: cut.requested_tolerance_m(),
                repair: WireRepairDisposition::None,
            },
            selections: FaceKey::V2_ALL.map(WireFaceSelectionV2::from).to_vec(),
        }
    }

    fn check_contract(&self) -> Result<(), Diagnostic> {
        let expected_selections = FaceKey::V2_ALL.map(WireFaceSelectionV2::from).to_vec();
        if self.schema != GRAPH_SCHEMA_V2
            || self.encoding != CANONICAL_ENCODING
            || self.length_unit != WireLengthUnit::Metre
            || !wire::has_rectangle_extrusion_chain(
                &self.sketch_plane,
                &self.profile,
                &self.face,
                &self.extrusion,
            )
            || self.cut_sketch_plane.id != CUT_SKETCH_PLANE_ID
            || self.cut_sketch_plane.kind != WireCutSketchPlaneKind::OnFace
            || self.cut_sketch_plane.face != "end-cap"
            || self.cut_profile.id != CUT_PROFILE_ID
            || self.cut_profile.kind != WireCircleKind::Circle
            || self.cut_profile.sketch_plane != CUT_SKETCH_PLANE_ID
            || self.cut_profile.constraint != WireConstraint::ClosedByConstruction
            || !self.cut_face.matches(CUT_FACE_ID, CUT_PROFILE_ID)
            || self.cut.id != CUT_ID
            || self.cut.kind != WireCutKind::DifferenceThroughAllNegativeZ
            || self.cut.target != EXTRUSION_ID
            || self.cut.tool_face != CUT_FACE_ID
            || self.cut.repair != WireRepairDisposition::None
            || self.selections != expected_selections
        {
            return Err(invalid(
                "unsupported authored CAD v2 schema, dependency chain, selection inventory, unit, or repair disposition",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireCutSketchPlane {
    id: String,
    kind: WireCutSketchPlaneKind,
    face: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireCutSketchPlaneKind {
    OnFace,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireCircleProfile {
    id: String,
    kind: WireCircleKind,
    sketch_plane: String,
    constraint: WireConstraint,
    center_m: [f64; 2],
    radius_m: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireCircleKind {
    Circle,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireCircularThroughCut {
    id: String,
    kind: WireCutKind,
    target: String,
    tool_face: String,
    requested_tolerance_m: f64,
    repair: WireRepairDisposition,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireCutKind {
    DifferenceThroughAllNegativeZ,
}

const fn canonical_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}
