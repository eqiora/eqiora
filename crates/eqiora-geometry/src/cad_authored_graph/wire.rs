//! Shared closed rectangle-to-extrusion wire primitives for authored CAD histories.

use crate::ConstrainedRectangleV1;
use serde::{Deserialize, Serialize};

const SKETCH_PLANE_ID: &str = "sketch-plane";
const PROFILE_ID: &str = "rectangle-profile";
const FACE_ID: &str = "profile-face";
pub(crate) const EXTRUSION_ID: &str = "positive-z-extrusion";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireSketchPlane {
    id: String,
    kind: WireSketchPlaneKind,
    pub(crate) z_m: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireSketchPlaneKind {
    Xy,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireRectangleProfile {
    id: String,
    kind: WireProfileKind,
    sketch_plane: String,
    constraint: WireConstraint,
    pub(crate) x_bounds_m: [f64; 2],
    pub(crate) y_bounds_m: [f64; 2],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireProfileKind {
    AxisAlignedRectangle,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum WireConstraint {
    ClosedByConstruction,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireClosedFace {
    id: String,
    kind: WireFaceKind,
    profile: String,
    region_count: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireFaceKind {
    OneClosedLoopFace,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WirePositiveZExtrusion {
    id: String,
    kind: WireExtrusionKind,
    face: String,
    pub(crate) depth_m: f64,
    repair: WireRepairDisposition,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireExtrusionKind {
    PositiveZ,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum WireRepairDisposition {
    None,
}

impl WireClosedFace {
    pub(crate) fn new(id: &str, profile: &str) -> Self {
        Self {
            id: id.to_owned(),
            kind: WireFaceKind::OneClosedLoopFace,
            profile: profile.to_owned(),
            region_count: 1,
        }
    }

    pub(crate) fn matches(&self, id: &str, profile: &str) -> bool {
        self.id == id
            && self.kind == WireFaceKind::OneClosedLoopFace
            && self.profile == profile
            && self.region_count == 1
    }
}

pub(crate) fn rectangle_extrusion(
    sketch: ConstrainedRectangleV1,
    depth_m: f64,
) -> (
    WireSketchPlane,
    WireRectangleProfile,
    WireClosedFace,
    WirePositiveZExtrusion,
) {
    (
        WireSketchPlane {
            id: SKETCH_PLANE_ID.to_owned(),
            kind: WireSketchPlaneKind::Xy,
            z_m: sketch.plane_z_m(),
        },
        WireRectangleProfile {
            id: PROFILE_ID.to_owned(),
            kind: WireProfileKind::AxisAlignedRectangle,
            sketch_plane: SKETCH_PLANE_ID.to_owned(),
            constraint: WireConstraint::ClosedByConstruction,
            x_bounds_m: [sketch.x_bounds_m().0, sketch.x_bounds_m().1],
            y_bounds_m: [sketch.y_bounds_m().0, sketch.y_bounds_m().1],
        },
        WireClosedFace::new(FACE_ID, PROFILE_ID),
        WirePositiveZExtrusion {
            id: EXTRUSION_ID.to_owned(),
            kind: WireExtrusionKind::PositiveZ,
            face: FACE_ID.to_owned(),
            depth_m,
            repair: WireRepairDisposition::None,
        },
    )
}

pub(crate) fn has_rectangle_extrusion_chain(
    sketch_plane: &WireSketchPlane,
    profile: &WireRectangleProfile,
    face: &WireClosedFace,
    extrusion: &WirePositiveZExtrusion,
) -> bool {
    sketch_plane.id == SKETCH_PLANE_ID
        && sketch_plane.kind == WireSketchPlaneKind::Xy
        && profile.id == PROFILE_ID
        && profile.kind == WireProfileKind::AxisAlignedRectangle
        && profile.sketch_plane == SKETCH_PLANE_ID
        && profile.constraint == WireConstraint::ClosedByConstruction
        && face.matches(FACE_ID, PROFILE_ID)
        && extrusion.id == EXTRUSION_ID
        && extrusion.kind == WireExtrusionKind::PositiveZ
        && extrusion.face == FACE_ID
        && extrusion.repair == WireRepairDisposition::None
}
