//! Content-bound CAD design and build evidence.
//!
//! The wire records only Eqiora-owned normalized observations. Source and
//! kernel identities are provenance; the exact Semantic Domain and Geometry
//! Identity remain the selection and physical-boundary authorities.

use std::str::FromStr;

use eqiora_core::Diagnostic;
use eqiora_core::Id;
use eqiora_core::entity::kinds;
use eqiora_geometry::{
    AxisAlignedBox3, CadAdapterIdentityV1, CadBoxDesignV1, CadBoxObservationV1,
    CadBoxRealizationV1, CadKernelAdapter, CadRepairDispositionV1, ConstrainedRectangleV1,
    StepLengthUnitV1, StepSourceDigest,
};
use eqiora_schema::kernel::KernelNode;
use serde::{Deserialize, Serialize};
use ulid::Ulid;

use crate::{
    ArtifactDigest, CANONICAL_ENCODING, GeometryIdentityEnvelopeV1, JsonLimits, RawSourceSha256,
    ReplayableCanonicalModelArtifact, invalid_artifact, validate_text,
};

const CAD_DESIGN_SCHEMA: &str = "eqiora.cad-box-design-envelope/v1";
const CAD_BUILD_SCHEMA: &str = "eqiora.cad-box-build-evidence-envelope/v1";

/// Canonical intent for one STEP-stock/sketch-extrusion intersection.
#[derive(Clone, Debug, PartialEq)]
pub struct CadDesignEnvelopeV1 {
    wire: WireCadDesignV1,
}

impl CadDesignEnvelopeV1 {
    /// Bind one closed CAD design to the exact Model whose Cartesian body it
    /// must realize.
    ///
    /// # Errors
    /// Returns `EQ0901` unless the target names a retained three-dimensional
    /// Cartesian body with bounds exactly equal to the design result.
    pub fn new(
        model: &impl ReplayableCanonicalModelArtifact,
        design: &CadBoxDesignV1,
    ) -> Result<Self, Diagnostic> {
        validate_model_target(model, design)?;
        let replay = model.replay_model()?;
        let model_reference = replay.artifact_reference();
        let envelope = Self {
            wire: WireCadDesignV1 {
                schema: CAD_DESIGN_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                model_sha256: model_reference.artifact().to_string(),
                target_body_ulid: design.target_body().ulid().to_string(),
                step_source_sha256: design.source().to_string(),
                source_length_unit: WireLengthUnit::encode(design.source_length_unit()),
                source_uncertainty_m: design.source_uncertainty_m(),
                modeling_tolerance_m: design.modeling_tolerance_m(),
                imported_stock: design.imported_stock().into(),
                sketch: design.sketch().into(),
                extrusion: WireExtrusion {
                    direction: WireExtrusionDirection::PositiveZ,
                    depth_m: design.extrusion_depth_m(),
                },
                boolean: WireBoolean::Intersection,
                output: design.output().into(),
            },
        };
        envelope.validate_local()?;
        Ok(envelope)
    }

    /// Decode bounded canonical wire data without trusting its Model reference.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed, noncanonical, or oversized input.
    pub fn from_json(bytes: &[u8], limits: JsonLimits) -> Result<Self, Diagnostic> {
        limits.check(bytes)?;
        let wire = serde_json::from_slice(bytes)
            .map_err(|error| invalid_artifact(format!("invalid CAD design JSON: {error}")))?;
        let envelope = Self { wire };
        envelope.validate_local()?;
        Ok(envelope)
    }

    /// Rebuild the design and compare it with one exact Model artifact.
    ///
    /// # Errors
    /// Returns `EQ0901` for Model or design drift.
    pub fn validate_against(
        &self,
        model: &impl ReplayableCanonicalModelArtifact,
    ) -> Result<(), Diagnostic> {
        let design = self.design()?;
        let expected = Self::new(model, &design)?;
        if self != &expected {
            return Err(invalid_artifact(
                "CAD design differs from exact Model-bound replay",
            ));
        }
        Ok(())
    }

    /// Reconstruct the pure closed design.
    ///
    /// # Errors
    /// Returns `EQ0901` if locally held wire data cannot reconstruct the
    /// already validated contract.
    pub fn design(&self) -> Result<CadBoxDesignV1, Diagnostic> {
        CadBoxDesignV1::new(
            parse_domain(&self.wire.target_body_ulid)?,
            StepSourceDigest::from_sha256(
                RawSourceSha256::from_hex(self.wire.step_source_sha256.clone())?.sha256_bytes(),
            ),
            self.wire.source_length_unit.decode(),
            self.wire.imported_stock.decode()?,
            self.wire.sketch.decode()?,
            self.wire.extrusion.depth_m,
            self.wire.source_uncertainty_m,
            self.wire.modeling_tolerance_m,
        )
    }

    /// Deterministic canonical JSON bytes.
    ///
    /// # Errors
    /// Returns `EQ0901` if serialization unexpectedly fails.
    pub fn canonical_json(&self) -> Result<Vec<u8>, Diagnostic> {
        serde_json::to_vec(&self.wire)
            .map_err(|error| invalid_artifact(format!("cannot serialize CAD design: {error}")))
    }

    /// Domain-separated content identity of the complete design.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            CAD_DESIGN_SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
    }

    /// Exact referenced Model artifact.
    #[must_use]
    pub fn model_artifact(&self) -> ArtifactDigest {
        ArtifactDigest::from_hex(self.wire.model_sha256.clone()).expect("validated Model digest")
    }

    /// Exact raw STEP source identity.
    #[must_use]
    pub fn source_digest(&self) -> RawSourceSha256 {
        RawSourceSha256::from_hex(self.wire.step_source_sha256.clone())
            .expect("validated STEP digest")
    }

    fn validate_local(&self) -> Result<(), Diagnostic> {
        if self.wire.schema != CAD_DESIGN_SCHEMA
            || self.wire.encoding != CANONICAL_ENCODING
            || self.wire.extrusion.direction != WireExtrusionDirection::PositiveZ
            || self.wire.boolean != WireBoolean::Intersection
        {
            return Err(invalid_artifact(
                "unsupported CAD design schema, encoding, unit, extrusion, or boolean",
            ));
        }
        ArtifactDigest::from_hex(self.wire.model_sha256.clone())?;
        parse_domain(&self.wire.target_body_ulid)?;
        RawSourceSha256::from_hex(self.wire.step_source_sha256.clone())?;
        reject_noncanonical_scalars(&[
            self.wire.source_uncertainty_m,
            self.wire.modeling_tolerance_m,
            self.wire.extrusion.depth_m,
        ])?;
        let design = self.design()?;
        if design.output() != self.wire.output.decode()? {
            return Err(invalid_artifact(
                "CAD output is not the exact stock/extrusion intersection",
            ));
        }
        if WireBox::from(design.imported_stock()) != self.wire.imported_stock
            || WireSketch::from(design.sketch()) != self.wire.sketch
        {
            return Err(invalid_artifact("CAD design scalars are not canonical"));
        }
        Ok(())
    }
}

/// Exact evidence that one adapter replay produced one accepted Geometry
/// Identity without leaking kernel topology identity.
#[derive(Clone, Debug, PartialEq)]
pub struct CadBuildEvidenceEnvelopeV1 {
    wire: WireCadBuildEvidenceV1,
}

impl CadBuildEvidenceEnvelopeV1 {
    /// Bind exact design, adapter/kernel identity, normalized observations,
    /// and Geometry Identity.
    ///
    /// # Errors
    /// Returns `EQ0901` for stale resources, invalid identity text, observation
    /// drift, repair, or a result that differs from the exact Semantic body.
    pub fn new(
        model: &impl ReplayableCanonicalModelArtifact,
        design: &CadDesignEnvelopeV1,
        geometry: &GeometryIdentityEnvelopeV1,
        adapter: CadAdapterIdentityV1,
        realization: CadBoxRealizationV1,
    ) -> Result<Self, Diagnostic> {
        design.validate_against(model)?;
        geometry.validate_against(model)?;
        if design.model_artifact() != geometry.model_artifact() {
            return Err(invalid_artifact(
                "CAD design and Geometry Identity reference different Models",
            ));
        }
        validate_adapter_identity(adapter)?;
        let pure_design = design.design()?;
        validate_realization(&pure_design, realization)?;
        let body = geometry
            .bodies()
            .into_iter()
            .find(|body| body.domain() == pure_design.target_body())
            .ok_or_else(|| invalid_artifact("CAD target body is absent from Geometry Identity"))?;
        if body.bounds_m() != pure_design.output().bounds_m() {
            return Err(invalid_artifact(
                "CAD boolean result differs from the target Geometry Identity body",
            ));
        }
        let envelope = Self {
            wire: WireCadBuildEvidenceV1 {
                schema: CAD_BUILD_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                model_sha256: design.model_artifact().to_string(),
                design_sha256: design.digest()?.to_string(),
                geometry_sha256: geometry.digest()?.to_string(),
                step_source_sha256: design.source_digest().to_string(),
                adapter: WireAdapterIdentity::from(adapter),
                source_length_unit: WireLengthUnit::encode(pure_design.source_length_unit()),
                source_uncertainty_m: pure_design.source_uncertainty_m(),
                modeling_tolerance_m: pure_design.modeling_tolerance_m(),
                geometry_classification_tolerance_m: geometry.tolerance_m(),
                repair: WireRepairDisposition::None,
                imported_stock: realization.imported_stock().into(),
                extruded_tool: realization.extruded_tool().into(),
                intersection: realization.intersection().into(),
                target_body_ulid: pure_design.target_body().ulid().to_string(),
            },
        };
        envelope.validate_local()?;
        Ok(envelope)
    }

    /// Decode bounded canonical wire data without replaying external code.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed, noncanonical, or oversized input.
    pub fn from_json(bytes: &[u8], limits: JsonLimits) -> Result<Self, Diagnostic> {
        limits.check(bytes)?;
        let wire = serde_json::from_slice(bytes).map_err(|error| {
            invalid_artifact(format!("invalid CAD build evidence JSON: {error}"))
        })?;
        let envelope = Self { wire };
        envelope.validate_local()?;
        Ok(envelope)
    }

    /// Re-run the exact adapter over complete STEP bytes and compare every
    /// resource and observation with this evidence.
    ///
    /// # Errors
    /// Returns `EQ0901` for source, adapter/kernel, design, Model, Geometry, or
    /// normalized-output drift.
    pub fn validate_replay(
        &self,
        model: &impl ReplayableCanonicalModelArtifact,
        design: &CadDesignEnvelopeV1,
        geometry: &GeometryIdentityEnvelopeV1,
        adapter: &impl CadKernelAdapter,
        step_bytes: &[u8],
    ) -> Result<(), Diagnostic> {
        let pure_design = design.design()?;
        if StepSourceDigest::from_source_bytes(step_bytes) != pure_design.source() {
            return Err(invalid_artifact(
                "complete STEP bytes differ from the CAD design source digest",
            ));
        }
        let realization = adapter.realize_box_design(&pure_design, step_bytes)?;
        let expected = Self::new(model, design, geometry, adapter.identity(), realization)?;
        if self != &expected {
            return Err(invalid_artifact(
                "CAD build evidence differs from exact adapter replay",
            ));
        }
        Ok(())
    }

    /// Deterministic canonical JSON bytes.
    ///
    /// # Errors
    /// Returns `EQ0901` if serialization unexpectedly fails.
    pub fn canonical_json(&self) -> Result<Vec<u8>, Diagnostic> {
        serde_json::to_vec(&self.wire).map_err(|error| {
            invalid_artifact(format!("cannot serialize CAD build evidence: {error}"))
        })
    }

    /// Domain-separated content identity of the complete build assertion.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            CAD_BUILD_SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
    }

    /// Exact accepted Geometry Identity artifact.
    #[must_use]
    pub fn geometry_artifact(&self) -> ArtifactDigest {
        ArtifactDigest::from_hex(self.wire.geometry_sha256.clone())
            .expect("validated geometry digest")
    }

    fn validate_local(&self) -> Result<(), Diagnostic> {
        if self.wire.schema != CAD_BUILD_SCHEMA
            || self.wire.encoding != CANONICAL_ENCODING
            || self.wire.repair != WireRepairDisposition::None
        {
            return Err(invalid_artifact(
                "unsupported CAD build schema, encoding, unit, or repair disposition",
            ));
        }
        for digest in [
            &self.wire.model_sha256,
            &self.wire.design_sha256,
            &self.wire.geometry_sha256,
        ] {
            ArtifactDigest::from_hex(digest.clone())?;
        }
        RawSourceSha256::from_hex(self.wire.step_source_sha256.clone())?;
        parse_domain(&self.wire.target_body_ulid)?;
        self.wire.adapter.validate()?;
        reject_noncanonical_scalars(&[
            self.wire.source_uncertainty_m,
            self.wire.modeling_tolerance_m,
            self.wire.geometry_classification_tolerance_m,
        ])?;
        if [
            self.wire.source_uncertainty_m,
            self.wire.modeling_tolerance_m,
            self.wire.geometry_classification_tolerance_m,
        ]
        .into_iter()
        .any(|value| value <= 0.0)
        {
            return Err(invalid_artifact(
                "CAD build tolerances must be finite and positive in metres",
            ));
        }
        self.wire.imported_stock.decode()?;
        self.wire.extruded_tool.decode()?;
        self.wire.intersection.decode()?;
        Ok(())
    }
}

fn validate_model_target(
    model: &impl ReplayableCanonicalModelArtifact,
    design: &CadBoxDesignV1,
) -> Result<(), Diagnostic> {
    let replay = model.replay_model()?;
    let Some(KernelNode::Domain(_)) = replay.program().node(design.target_body().erase()) else {
        return Err(invalid_artifact(
            "CAD target does not name a retained Semantic Domain",
        ));
    };
    let bounds = replay
        .program()
        .resolved_cartesian_bounds(design.target_body())
        .map_err(|_| invalid_artifact("CAD v1 target must be a Cartesian box Domain"))?;
    if bounds.len() != 3 {
        return Err(invalid_artifact("CAD v1 target must be three-dimensional"));
    }
    let semantic_bounds = bounds
        .iter()
        .map(|axis| {
            (
                canonical_zero(axis.lower().value()),
                canonical_zero(axis.upper().value()),
            )
        })
        .collect::<Vec<_>>();
    if semantic_bounds.as_slice() != design.output().bounds_m() {
        return Err(invalid_artifact(
            "CAD design output differs from exact Semantic body bounds",
        ));
    }
    Ok(())
}

fn validate_realization(
    design: &CadBoxDesignV1,
    realization: CadBoxRealizationV1,
) -> Result<(), Diagnostic> {
    CadBoxRealizationV1::new(
        design,
        realization.imported_stock(),
        realization.extruded_tool(),
        realization.intersection(),
    )?;
    Ok(())
}

fn validate_adapter_identity(identity: CadAdapterIdentityV1) -> Result<(), Diagnostic> {
    for (label, value) in [
        ("CAD adapter ID", identity.adapter()),
        ("CAD adapter version", identity.adapter_version()),
        ("CAD kernel ID", identity.kernel()),
        ("CAD kernel version", identity.kernel_version()),
    ] {
        validate_text(label, value)?;
    }
    Ok(())
}

fn parse_domain(value: &str) -> Result<Id<kinds::Domain>, Diagnostic> {
    Ulid::from_str(value)
        .map(Id::from_ulid)
        .map_err(|_| invalid_artifact("CAD target Domain ULID is invalid"))
}

fn reject_noncanonical_scalars(values: &[f64]) -> Result<(), Diagnostic> {
    if values
        .iter()
        .any(|value| !value.is_finite() || (*value == 0.0 && value.is_sign_negative()))
    {
        return Err(invalid_artifact(
            "CAD wire scalars must be finite and erase negative zero",
        ));
    }
    Ok(())
}

const fn canonical_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireCadDesignV1 {
    schema: String,
    encoding: String,
    model_sha256: String,
    target_body_ulid: String,
    step_source_sha256: String,
    source_length_unit: WireLengthUnit,
    source_uncertainty_m: f64,
    modeling_tolerance_m: f64,
    imported_stock: WireBox,
    sketch: WireSketch,
    extrusion: WireExtrusion,
    boolean: WireBoolean,
    output: WireBox,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireCadBuildEvidenceV1 {
    schema: String,
    encoding: String,
    model_sha256: String,
    design_sha256: String,
    geometry_sha256: String,
    step_source_sha256: String,
    adapter: WireAdapterIdentity,
    source_length_unit: WireLengthUnit,
    source_uncertainty_m: f64,
    modeling_tolerance_m: f64,
    geometry_classification_tolerance_m: f64,
    repair: WireRepairDisposition,
    imported_stock: WireObservation,
    extruded_tool: WireObservation,
    intersection: WireObservation,
    target_body_ulid: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireLengthUnit {
    Metre,
    Millimetre,
}

impl WireLengthUnit {
    const fn encode(value: StepLengthUnitV1) -> Self {
        match value {
            StepLengthUnitV1::Metre => Self::Metre,
            StepLengthUnitV1::Millimetre => Self::Millimetre,
        }
    }

    const fn decode(self) -> StepLengthUnitV1 {
        match self {
            Self::Metre => StepLengthUnitV1::Metre,
            Self::Millimetre => StepLengthUnitV1::Millimetre,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireExtrusionDirection {
    PositiveZ,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireBoolean {
    Intersection,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireRepairDisposition {
    None,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireBox {
    x_lower_m: f64,
    x_upper_m: f64,
    y_lower_m: f64,
    y_upper_m: f64,
    z_lower_m: f64,
    z_upper_m: f64,
}

impl From<AxisAlignedBox3> for WireBox {
    fn from(value: AxisAlignedBox3) -> Self {
        let [x, y, z] = value.bounds_m();
        Self {
            x_lower_m: x.0,
            x_upper_m: x.1,
            y_lower_m: y.0,
            y_upper_m: y.1,
            z_lower_m: z.0,
            z_upper_m: z.1,
        }
    }
}

impl WireBox {
    fn decode(self) -> Result<AxisAlignedBox3, Diagnostic> {
        reject_noncanonical_scalars(&[
            self.x_lower_m,
            self.x_upper_m,
            self.y_lower_m,
            self.y_upper_m,
            self.z_lower_m,
            self.z_upper_m,
        ])?;
        AxisAlignedBox3::new([
            (self.x_lower_m, self.x_upper_m),
            (self.y_lower_m, self.y_upper_m),
            (self.z_lower_m, self.z_upper_m),
        ])
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireSketch {
    plane: WireSketchPlane,
    x_lower_m: f64,
    x_upper_m: f64,
    y_lower_m: f64,
    y_upper_m: f64,
    plane_z_m: f64,
    remaining_degrees_of_freedom: u64,
}

impl From<ConstrainedRectangleV1> for WireSketch {
    fn from(value: ConstrainedRectangleV1) -> Self {
        Self {
            plane: WireSketchPlane::Xy,
            x_lower_m: value.x_bounds_m().0,
            x_upper_m: value.x_bounds_m().1,
            y_lower_m: value.y_bounds_m().0,
            y_upper_m: value.y_bounds_m().1,
            plane_z_m: value.plane_z_m(),
            remaining_degrees_of_freedom: 0,
        }
    }
}

impl WireSketch {
    fn decode(self) -> Result<ConstrainedRectangleV1, Diagnostic> {
        if self.plane != WireSketchPlane::Xy || self.remaining_degrees_of_freedom != 0 {
            return Err(invalid_artifact(
                "CAD v1 sketch must be a fully constrained XY rectangle",
            ));
        }
        reject_noncanonical_scalars(&[
            self.x_lower_m,
            self.x_upper_m,
            self.y_lower_m,
            self.y_upper_m,
            self.plane_z_m,
        ])?;
        ConstrainedRectangleV1::new(
            (self.x_lower_m, self.x_upper_m),
            (self.y_lower_m, self.y_upper_m),
            self.plane_z_m,
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireSketchPlane {
    Xy,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireExtrusion {
    direction: WireExtrusionDirection,
    depth_m: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireAdapterIdentity {
    adapter: String,
    adapter_version: String,
    kernel: String,
    kernel_version: String,
}

impl From<CadAdapterIdentityV1> for WireAdapterIdentity {
    fn from(value: CadAdapterIdentityV1) -> Self {
        Self {
            adapter: value.adapter().to_owned(),
            adapter_version: value.adapter_version().to_owned(),
            kernel: value.kernel().to_owned(),
            kernel_version: value.kernel_version().to_owned(),
        }
    }
}

impl WireAdapterIdentity {
    fn validate(&self) -> Result<(), Diagnostic> {
        for (label, value) in [
            ("CAD adapter ID", self.adapter.as_str()),
            ("CAD adapter version", self.adapter_version.as_str()),
            ("CAD kernel ID", self.kernel.as_str()),
            ("CAD kernel version", self.kernel_version.as_str()),
        ] {
            validate_text(label, value)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireObservation {
    bounds: WireBox,
    solid_count: u64,
    closed_shell_count: u64,
    planar_face_count: u64,
    repair: WireRepairDisposition,
}

impl From<CadBoxObservationV1> for WireObservation {
    fn from(value: CadBoxObservationV1) -> Self {
        Self {
            bounds: value.bounds().into(),
            solid_count: u64::try_from(value.solid_count()).expect("validated CAD count"),
            closed_shell_count: u64::try_from(value.closed_shell_count())
                .expect("validated CAD count"),
            planar_face_count: u64::try_from(value.planar_face_count())
                .expect("validated CAD count"),
            repair: match value.repair() {
                CadRepairDispositionV1::None => WireRepairDisposition::None,
            },
        }
    }
}

impl WireObservation {
    fn decode(self) -> Result<CadBoxObservationV1, Diagnostic> {
        CadBoxObservationV1::new(
            self.bounds.decode()?,
            usize::try_from(self.solid_count)
                .map_err(|_| invalid_artifact("CAD solid count exceeds local usize"))?,
            usize::try_from(self.closed_shell_count)
                .map_err(|_| invalid_artifact("CAD shell count exceeds local usize"))?,
            usize::try_from(self.planar_face_count)
                .map_err(|_| invalid_artifact("CAD face count exceeds local usize"))?,
            match self.repair {
                WireRepairDisposition::None => CadRepairDispositionV1::None,
            },
        )
    }
}
