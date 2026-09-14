//! Field-aware transfer receipts for one accepted remesh seam.

use std::collections::BTreeSet;
use std::num::NonZeroUsize;

use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, DimExponents, DynQuantity, Id};
use eqiora_realization::{
    AleFsiRemeshScaleProfile2d, AleFsiRemeshTransferPlan2d, QuadraturePolicy,
};
use eqiora_solver::{
    ConvergenceReason, ExecutionReport, ExecutionTopology, LinearOperatorOrientation, SolveReport,
    SolverPlan,
};
use serde::{Deserialize, Serialize};
use ulid::Ulid;

use crate::realization_v7::wire::WireSolverPlan;
use crate::{
    ArtifactDigest, CANONICAL_ENCODING, FieldSnapshotEnvelopeV2, GeometryStateEnvelopeV2,
    MeshRevisionOverlapEnvelopeV1, SpatialStateEnvelopeV2, ValidatedMovingSpatialContextV2,
    ValidatedRemeshGeometrySourceV2, check_json_limits, invalid_artifact,
};

mod normalization;
use normalization::*;

const TRANSFER_SCHEMA: &str = "eqiora.remesh-transfer-receipt/v1";
const PROJECTION_SCHEMA: &str = "eqiora.remesh-projection-evidence/v1";
const TRANSFER_ACTION_VERSION: &str = "eqiora.ale-fsi-remesh-transfer/1";

/// Semantic work budgets shared by remesh overlap and transfer artifacts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RemeshDecoderLimits {
    /// Common JSON syntax admission.
    pub json: crate::JsonDecoderLimits,
    /// Maximum revision associations consumed by one overlap artifact.
    pub max_geometry_revision_associations: usize,
    /// Maximum positive-area cell fragments in one overlap artifact.
    pub max_mesh_overlap_cell_fragments: usize,
    /// Maximum positive-length retained-facet fragments in one overlap artifact.
    pub max_mesh_overlap_facet_fragments: usize,
    /// Maximum Field-aware entries in one transfer receipt.
    pub max_remesh_transfer_fields: usize,
    /// Maximum component solves in one typed projection evidence artifact.
    pub max_remesh_projection_solves: usize,
}

impl Default for RemeshDecoderLimits {
    fn default() -> Self {
        Self {
            json: crate::JsonDecoderLimits::default(),
            max_geometry_revision_associations: 1_000_000,
            max_mesh_overlap_cell_fragments: 16_000_000,
            max_mesh_overlap_facet_fragments: 16_000_000,
            max_remesh_transfer_fields: 100_000,
            max_remesh_projection_solves: 2,
        }
    }
}

/// Closed numerical action performed by one remesh projection solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RemeshProjectionActionV1 {
    /// One constrained solve for both velocity Fields.
    CoupledVelocity,
    /// Absolute-pressure projection without a gauge.
    AbsolutePressure,
    /// Absolute material-displacement projection.
    AbsoluteDisplacement,
}

/// Closed execution mode of one projection action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RemeshProjectionExecutionModeV1 {
    /// Boundary data prescribes every target degree of freedom exactly.
    PrescribedExactly,
    /// One or more component systems were solved and fully reported.
    Solved,
}

/// Self-contained execution evidence for one closed transfer action.
///
/// The exact overlap, action version, common transfer plan, right-hand-side
/// norm, and accepted solver report are encoded together. Consumers never
/// trust a free-standing operator or solve digest.
#[derive(Debug, Clone, PartialEq)]
pub struct RemeshProjectionEvidenceEnvelopeV1 {
    wire: WireRemeshProjectionEvidenceV1,
}

impl RemeshProjectionEvidenceEnvelopeV1 {
    /// Convert one accepted common plan and solver report into durable evidence.
    ///
    /// # Errors
    /// Returns `EQ0901` when the report did not use the exact common plan, is
    /// transposed, has an inconsistent right-hand-side norm, or is otherwise
    /// not a canonical accepted report.
    pub fn solved<'a>(
        action: RemeshProjectionActionV1,
        overlap: &MeshRevisionOverlapEnvelopeV1,
        plan: AleFsiRemeshTransferPlan2d,
        solves: impl IntoIterator<Item = (f64, &'a SolveReport)>,
        dimensionless_residual: f64,
        dimensionless_acceptance_limit: f64,
    ) -> Result<Self, Diagnostic> {
        let solves = solves
            .into_iter()
            .enumerate()
            .map(|(component, (right_hand_side_norm, report))| {
                if report.solver_plan() != plan.solver()
                    || report.orientation() != LinearOperatorOrientation::Normal
                {
                    return Err(invalid_artifact(
                        "remesh projection report must use the exact common normal-action plan",
                    ));
                }
                Ok(WireProjectionSolveV1 {
                    component: u8::try_from(component).map_err(|_| {
                        invalid_artifact("remesh projection component exceeds portable u8")
                    })?,
                    right_hand_side_norm: normalize_zero(right_hand_side_norm),
                    report: WireSolveReportV1::encode(report)?,
                })
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        let execution = match action {
            RemeshProjectionActionV1::CoupledVelocity
            | RemeshProjectionActionV1::AbsolutePressure => {
                let [solve] = solves.try_into().map_err(|_| {
                    invalid_artifact("scalar remesh projection requires exactly one solve")
                })?;
                WireProjectionExecutionV1::SolvedScalar {
                    solve: Box::new(solve),
                }
            }
            RemeshProjectionActionV1::AbsoluteDisplacement => {
                let solves = solves.try_into().map_err(|_| {
                    invalid_artifact("vector remesh projection requires exactly two solves")
                })?;
                WireProjectionExecutionV1::SolvedVector2 {
                    solves: Box::new(solves),
                }
            }
        };
        Self::finish(
            action,
            execution,
            overlap,
            plan,
            BoundedRemeshDefectV1::new(dimensionless_residual, dimensionless_acceptance_limit)?,
        )
    }

    /// Record an exactly prescribed absolute-displacement action with no solve.
    ///
    /// # Errors
    /// Returns `EQ0901` for an action other than absolute displacement.
    pub fn prescribed_exactly(
        action: RemeshProjectionActionV1,
        overlap: &MeshRevisionOverlapEnvelopeV1,
        plan: AleFsiRemeshTransferPlan2d,
    ) -> Result<Self, Diagnostic> {
        if action != RemeshProjectionActionV1::AbsoluteDisplacement {
            return Err(invalid_artifact(
                "only absolute displacement may be satisfied by exact prescription",
            ));
        }
        Self::finish(
            action,
            WireProjectionExecutionV1::PrescribedExactly,
            overlap,
            plan,
            BoundedRemeshDefectV1::new(0.0, 0.0)?,
        )
    }

    fn finish(
        action: RemeshProjectionActionV1,
        execution: WireProjectionExecutionV1,
        overlap: &MeshRevisionOverlapEnvelopeV1,
        plan: AleFsiRemeshTransferPlan2d,
        algebraic_replay: BoundedRemeshDefectV1,
    ) -> Result<Self, Diagnostic> {
        let value = Self {
            wire: WireRemeshProjectionEvidenceV1 {
                schema: PROJECTION_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                action_version: TRANSFER_ACTION_VERSION.to_owned(),
                action: WireProjectionActionV1::encode(action),
                execution,
                overlap_sha256: overlap.digest()?.to_string(),
                plan: WireRemeshTransferPlanV1::encode(plan)?,
                algebraic_replay: WireBoundedDefectV1::encode(algebraic_replay),
            },
        };
        value.validate_local(RemeshDecoderLimits::default())?;
        Ok(value)
    }

    /// Decode self-contained projection evidence.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed, unknown, or noncanonical data.
    pub fn from_json(bytes: &[u8], limits: RemeshDecoderLimits) -> Result<Self, Diagnostic> {
        check_json_limits(bytes, limits.json)?;
        let wire = serde_json::from_slice(bytes).map_err(|error| {
            invalid_artifact(format!("invalid remesh projection evidence JSON: {error}"))
        })?;
        let value = Self { wire };
        value.validate_local(limits)?;
        Ok(value)
    }

    /// Deterministic canonical JSON bytes.
    ///
    /// # Errors
    /// Returns `EQ0901` if serialization unexpectedly fails.
    pub fn canonical_json(&self) -> Result<Vec<u8>, Diagnostic> {
        serde_json::to_vec(&self.wire).map_err(|error| {
            invalid_artifact(format!(
                "cannot serialize remesh projection evidence: {error}"
            ))
        })
    }

    /// Domain-separated evidence identity.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            PROJECTION_SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
    }

    /// Closed transfer action.
    #[must_use]
    pub const fn action(&self) -> RemeshProjectionActionV1 {
        self.wire.action.decode()
    }

    /// Closed execution mode.
    #[must_use]
    pub const fn execution_mode(&self) -> RemeshProjectionExecutionModeV1 {
        self.wire.execution.mode()
    }

    /// Exact overlap consumed by the action.
    #[must_use]
    pub fn overlap_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.overlap_sha256.clone())
    }

    /// Reconstructed common transfer plan.
    ///
    /// # Errors
    /// Returns `EQ0901` if the stored closed policy is invalid.
    pub fn plan(&self) -> Result<AleFsiRemeshTransferPlan2d, Diagnostic> {
        self.wire.plan.decode()
    }

    /// Independently replayed dimensionless algebraic residual and its limit.
    #[must_use]
    pub const fn dimensionless_algebraic_replay(&self) -> BoundedRemeshDefectV1 {
        self.wire.algebraic_replay.decode()
    }
    /// Require the exact overlap rather than accepting its digest unresolved.
    ///
    /// # Errors
    /// Returns `EQ0901` for stale overlap evidence.
    pub fn validate_against_overlap(
        &self,
        overlap: &MeshRevisionOverlapEnvelopeV1,
    ) -> Result<(), Diagnostic> {
        self.validate_local(RemeshDecoderLimits::default())?;
        if self.overlap_artifact() == overlap.digest()? {
            Ok(())
        } else {
            Err(invalid_artifact(
                "remesh projection evidence references a stale overlap",
            ))
        }
    }

    fn validate_local(&self, limits: RemeshDecoderLimits) -> Result<(), Diagnostic> {
        if self.wire.schema != PROJECTION_SCHEMA
            || self.wire.encoding != CANONICAL_ENCODING
            || self.wire.action_version != TRANSFER_ACTION_VERSION
        {
            return Err(invalid_artifact(
                "unsupported remesh projection schema, encoding, or action version",
            ));
        }
        ArtifactDigest::from_hex(self.wire.overlap_sha256.clone())?;
        let plan = self.wire.plan.decode()?;
        self.wire.algebraic_replay.validate()?;
        let expected_solves = match (self.action(), self.execution_mode()) {
            (
                RemeshProjectionActionV1::CoupledVelocity
                | RemeshProjectionActionV1::AbsolutePressure,
                RemeshProjectionExecutionModeV1::Solved,
            ) => 1,
            (
                RemeshProjectionActionV1::AbsoluteDisplacement,
                RemeshProjectionExecutionModeV1::Solved,
            ) => 2,
            (
                RemeshProjectionActionV1::AbsoluteDisplacement,
                RemeshProjectionExecutionModeV1::PrescribedExactly,
            ) => 0,
            _ => {
                return Err(invalid_artifact(
                    "remesh projection action and execution mode are incompatible",
                ));
            }
        };
        if self.wire.execution.solve_count() != expected_solves
            || self.wire.execution.solve_count() > limits.max_remesh_projection_solves
        {
            return Err(invalid_artifact(
                "remesh projection has an incomplete or oversized component-solve inventory",
            ));
        }
        if self.execution_mode() == RemeshProjectionExecutionModeV1::PrescribedExactly
            && self.wire.algebraic_replay != WireBoundedDefectV1::zero()
        {
            return Err(invalid_artifact(
                "exactly prescribed remesh projection must have exact zero algebraic replay",
            ));
        }
        for component in 0..self.wire.execution.solve_count() {
            let solve = self
                .wire
                .execution
                .solve(component)
                .expect("component is below closed execution solve count");
            if usize::from(solve.component) != component {
                return Err(invalid_artifact(
                    "remesh projection component solves are not in canonical order",
                ));
            }
            solve
                .report
                .validate(plan.solver(), solve.right_hand_side_norm)?;
        }
        Ok(())
    }
}

/// Semantic role of one Field in the bounded FSI transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RemeshFieldRoleV1 {
    /// Conservative fluid velocity represented by MINI coefficients.
    FluidVelocity,
    /// Dynamic-solid P1 velocity sharing the interface trace.
    SolidVelocity,
    /// Absolute fluid pressure without an invented gauge.
    FluidPressure,
    /// Absolute material-chart solid displacement.
    SolidDisplacement,
}

/// Closed variational law used for one transferred Field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RemeshTransferLawV1 {
    /// Coupled velocity projection with trace, divergence, and momentum constraints.
    CoupledVelocityConstrainedL2,
    /// Absolute-pressure P1 L2 projection.
    AbsolutePressureL2,
    /// Absolute-displacement P1 L2 projection.
    AbsoluteDisplacementL2,
}

/// Integration chart selected by one transfer law.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RemeshIntegrationChartV1 {
    /// Undeformed material/reference coordinates.
    Material,
    /// Accepted current spatial coordinates at the remesh time.
    CurrentSpatial,
}

/// One dimensionless observed defect accepted against one explicit limit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundedRemeshDefectV1 {
    observed: f64,
    limit: f64,
}

impl BoundedRemeshDefectV1 {
    /// Admit one finite nonnegative defect no larger than its finite limit.
    ///
    /// # Errors
    /// Returns `EQ0901` for non-finite, negative, or unaccepted values.
    pub fn new(observed: f64, limit: f64) -> Result<Self, Diagnostic> {
        if !observed.is_finite()
            || !limit.is_finite()
            || observed < 0.0
            || limit < 0.0
            || observed > limit
        {
            return Err(invalid_artifact(
                "remesh transfer defect must be finite, nonnegative, and within its limit",
            ));
        }
        Ok(Self {
            observed: normalize_zero(observed),
            limit: normalize_zero(limit),
        })
    }

    /// Observed dimensionless defect.
    #[must_use]
    pub const fn observed(self) -> f64 {
        self.observed
    }

    /// Explicit accepted dimensionless limit.
    #[must_use]
    pub const fn limit(self) -> f64 {
        self.limit
    }
}

/// One Field-local projection receipt without numerical implementation types.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldTransferReceiptV1 {
    wire: WireFieldTransferReceiptV1,
}

impl FieldTransferReceiptV1 {
    /// Bind one source/target Field pair to exact typed projection evidence.
    ///
    /// `raw_projection_error_l2` remains a Field-dimensional approximation
    /// measurement and is never compared to a dimensionless limit. Algebraic
    /// replay belongs to the referenced projection action and is not repeated
    /// once per Field.
    ///
    /// # Errors
    /// Returns `EQ0901` for changed Field meaning, a role/law/chart mismatch,
    /// stale snapshots, invalid evidence, or equal source/target snapshots.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        role: RemeshFieldRoleV1,
        law: RemeshTransferLawV1,
        chart: RemeshIntegrationChartV1,
        source: &FieldSnapshotEnvelopeV2,
        target: &FieldSnapshotEnvelopeV2,
        projection: &RemeshProjectionEvidenceEnvelopeV1,
        raw_projection_error_l2: f64,
    ) -> Result<Self, Diagnostic> {
        require_role_law_chart(role, law, chart)?;
        require_role_action(role, projection.action())?;
        if source.field() != target.field()
            || source.support_domain() != target.support_domain()
            || source.dimension() != target.dimension()
            || source.value_shape() != target.value_shape()
            || source.frame() != target.frame()
            || source.model_artifact() != target.model_artifact()
            || source.digest()? == target.digest()?
        {
            return Err(invalid_artifact(
                "remesh Field receipt must retain exact Field meaning across distinct snapshots",
            ));
        }
        if !raw_projection_error_l2.is_finite() || raw_projection_error_l2 < 0.0 {
            return Err(invalid_artifact(
                "remesh Field projection error must be finite and nonnegative",
            ));
        }
        let value = Self {
            wire: WireFieldTransferReceiptV1 {
                field_ulid: source.field().ulid().to_string(),
                role: WireFieldRoleV1::encode(role),
                law: WireTransferLawV1::encode(law),
                chart: WireIntegrationChartV1::encode(chart),
                source_snapshot_sha256: source.digest()?.to_string(),
                target_snapshot_sha256: target.digest()?.to_string(),
                projection_evidence_sha256: projection.digest()?.to_string(),
                raw_projection_error_l2: normalize_zero(raw_projection_error_l2),
            },
        };
        value.validate_local()?;
        Ok(value)
    }

    /// Exact Semantic Field identity.
    #[must_use]
    pub fn field(&self) -> Id<kinds::Field> {
        parse_field(&self.wire.field_ulid).expect("validated transfer Field ULID")
    }

    /// Exact bounded FSI Field role.
    #[must_use]
    pub const fn role(&self) -> RemeshFieldRoleV1 {
        self.wire.role.decode()
    }

    /// Exact variational transfer law.
    #[must_use]
    pub const fn law(&self) -> RemeshTransferLawV1 {
        self.wire.law.decode()
    }

    /// Integration chart selected by the law.
    #[must_use]
    pub const fn chart(&self) -> RemeshIntegrationChartV1 {
        self.wire.chart.decode()
    }

    /// Exact source Field snapshot.
    #[must_use]
    pub fn source_snapshot(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.source_snapshot_sha256.clone())
    }

    /// Exact target Field snapshot.
    #[must_use]
    pub fn target_snapshot(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.target_snapshot_sha256.clone())
    }

    /// Content identity of the embedded typed projection evidence.
    #[must_use]
    pub fn projection_evidence(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.projection_evidence_sha256.clone())
    }

    /// Finite raw Field-local L2 approximation error in the Field's units.
    #[must_use]
    pub const fn raw_projection_error_l2(&self) -> f64 {
        self.wire.raw_projection_error_l2
    }

    fn validate_local(&self) -> Result<(), Diagnostic> {
        parse_field(&self.wire.field_ulid)?;
        require_role_law_chart(self.role(), self.law(), self.chart())?;
        for digest in [
            &self.wire.source_snapshot_sha256,
            &self.wire.target_snapshot_sha256,
            &self.wire.projection_evidence_sha256,
        ] {
            ArtifactDigest::from_hex(digest.clone())?;
        }
        if self.wire.source_snapshot_sha256 == self.wire.target_snapshot_sha256
            || !self.wire.raw_projection_error_l2.is_finite()
            || self.wire.raw_projection_error_l2 < 0.0
            || is_negative_zero(self.wire.raw_projection_error_l2)
        {
            return Err(invalid_artifact(
                "remesh Field receipt has invalid snapshot or projection-error evidence",
            ));
        }
        Ok(())
    }
}

/// Closed dimensional normalization used by every remesh acceptance check.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RemeshNormalizationWitnessV1 {
    scales: AleFsiRemeshScaleProfile2d,
    reference_density: f64,
    characteristic_mass: f64,
    characteristic_momentum: f64,
    characteristic_weak_divergence: f64,
    characteristic_pressure_moment: f64,
}

impl RemeshNormalizationWitnessV1 {
    /// Bind the exact typed `L`, `U`, `P` profile and `rho*` used by numerics.
    ///
    /// The intrinsic-2D characteristic quantities use unit thickness:
    /// `rho* L^2`, `rho* U L^2`, `U L`, and `P L^2`.
    ///
    /// # Errors
    /// Returns `EQ0901` unless density and every derived scale are finite and
    /// strictly positive.
    pub fn new(
        scales: AleFsiRemeshScaleProfile2d,
        reference_density: f64,
    ) -> Result<Self, Diagnostic> {
        if !reference_density.is_finite() || reference_density <= 0.0 {
            return Err(invalid_artifact(
                "remesh reference density must be finite and strictly positive",
            ));
        }
        let length = scales.length().value();
        let characteristic_mass = checked_positive_product(
            checked_positive_product(reference_density, length, "rho* L")?,
            length,
            "rho* L^2",
        )?;
        let characteristic_momentum =
            checked_positive_product(characteristic_mass, scales.velocity().value(), "rho* U L^2")?;
        let characteristic_weak_divergence =
            checked_positive_product(scales.velocity().value(), length, "U L")?;
        let characteristic_pressure_moment = checked_positive_product(
            checked_positive_product(scales.pressure().value(), length, "P L")?,
            length,
            "P L^2",
        )?;
        Ok(Self {
            scales,
            reference_density: normalize_zero(reference_density),
            characteristic_mass,
            characteristic_momentum,
            characteristic_weak_divergence,
            characteristic_pressure_moment,
        })
    }

    /// Exact typed characteristic length, velocity, and pressure.
    #[must_use]
    pub const fn scales(self) -> AleFsiRemeshScaleProfile2d {
        self.scales
    }

    /// Reference density `rho* = max(rho_fluid, rho_solid)` in coherent SI.
    #[must_use]
    pub const fn reference_density(self) -> f64 {
        self.reference_density
    }

    /// Intrinsic-2D characteristic mass `rho* L^2` under unit thickness.
    #[must_use]
    pub const fn characteristic_mass(self) -> f64 {
        self.characteristic_mass
    }

    /// Intrinsic-2D characteristic momentum `rho* U L^2`.
    #[must_use]
    pub const fn characteristic_momentum(self) -> f64 {
        self.characteristic_momentum
    }

    /// Characteristic weak-divergence functional `U L`.
    #[must_use]
    pub const fn characteristic_weak_divergence(self) -> f64 {
        self.characteristic_weak_divergence
    }

    /// Characteristic absolute-pressure zeroth moment `P L^2`.
    #[must_use]
    pub const fn characteristic_pressure_moment(self) -> f64 {
        self.characteristic_pressure_moment
    }
}

/// Coupled acceptance evidence common to the four Field transfers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RemeshTransferEvidenceV1 {
    normalization: RemeshNormalizationWitnessV1,
    momentum_before: [f64; 2],
    momentum_after: [f64; 2],
    pressure_moment_before: f64,
    pressure_moment_after: f64,
    raw_weak_divergence: f64,
    raw_shared_trace: f64,
    raw_exterior_trace: f64,
    raw_displacement_trace: f64,
    raw_harmonic_coordinate_defect: f64,
    momentum_defect: BoundedRemeshDefectV1,
    weak_divergence: BoundedRemeshDefectV1,
    shared_trace: BoundedRemeshDefectV1,
    exterior_trace: BoundedRemeshDefectV1,
    pressure_zeroth_moment: BoundedRemeshDefectV1,
    displacement_trace: BoundedRemeshDefectV1,
    harmonic_replay: BoundedRemeshDefectV1,
}

impl RemeshTransferEvidenceV1 {
    /// Normalize raw coherent-SI observables under one closed witness.
    ///
    /// Pressure moments are absolute-pressure zeroth moments, not a zero-mean
    /// gauge. No caller supplies a dimensionless observed defect: every one is
    /// recomputed here from raw evidence and the exact `L`, `U`, `P`, `rho*`
    /// witness before it is compared with the one common dimensionless limit.
    ///
    /// # Errors
    /// Returns `EQ0901` for non-finite raw evidence, a negative raw norm, an
    /// overflowing normalization, or any derived defect outside the limit.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        normalization: RemeshNormalizationWitnessV1,
        mut momentum_before: [f64; 2],
        mut momentum_after: [f64; 2],
        pressure_moment_before: f64,
        pressure_moment_after: f64,
        raw_weak_divergence: f64,
        raw_shared_trace: f64,
        raw_exterior_trace: f64,
        raw_displacement_trace: f64,
        raw_harmonic_coordinate_defect: f64,
        dimensionless_physical_acceptance_limit: f64,
    ) -> Result<Self, Diagnostic> {
        if momentum_before
            .iter()
            .chain(&momentum_after)
            .chain([pressure_moment_before, pressure_moment_after].iter())
            .any(|value| !value.is_finite())
        {
            return Err(invalid_artifact(
                "remesh momentum and pressure functionals must be finite",
            ));
        }
        let raw_nonnegative = [
            raw_weak_divergence,
            raw_shared_trace,
            raw_exterior_trace,
            raw_displacement_trace,
            raw_harmonic_coordinate_defect,
        ];
        if raw_nonnegative
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(invalid_artifact(
                "remesh raw norm evidence must be finite and nonnegative",
            ));
        }

        let momentum_delta =
            (momentum_before[0] - momentum_after[0]).hypot(momentum_before[1] - momentum_after[1]);
        let pressure_delta = (pressure_moment_before - pressure_moment_after).abs();
        let bounded = |raw: f64, characteristic: f64| {
            BoundedRemeshDefectV1::new(
                raw / characteristic,
                dimensionless_physical_acceptance_limit,
            )
        };
        let momentum_defect = bounded(momentum_delta, normalization.characteristic_momentum)?;
        let weak_divergence = bounded(
            raw_weak_divergence,
            normalization.characteristic_weak_divergence,
        )?;
        let shared_trace = bounded(raw_shared_trace, normalization.scales.velocity().value())?;
        let exterior_trace = bounded(raw_exterior_trace, normalization.scales.velocity().value())?;
        let pressure_zeroth_moment =
            bounded(pressure_delta, normalization.characteristic_pressure_moment)?;
        let displacement_trace = bounded(
            raw_displacement_trace,
            normalization.scales.length().value(),
        )?;
        let harmonic_replay = bounded(
            raw_harmonic_coordinate_defect,
            normalization.scales.length().value(),
        )?;

        normalize_array(&mut momentum_before);
        normalize_array(&mut momentum_after);
        Ok(Self {
            normalization,
            momentum_before,
            momentum_after,
            pressure_moment_before: normalize_zero(pressure_moment_before),
            pressure_moment_after: normalize_zero(pressure_moment_after),
            raw_weak_divergence: normalize_zero(raw_weak_divergence),
            raw_shared_trace: normalize_zero(raw_shared_trace),
            raw_exterior_trace: normalize_zero(raw_exterior_trace),
            raw_displacement_trace: normalize_zero(raw_displacement_trace),
            raw_harmonic_coordinate_defect: normalize_zero(raw_harmonic_coordinate_defect),
            momentum_defect,
            weak_divergence,
            shared_trace,
            exterior_trace,
            pressure_zeroth_moment,
            displacement_trace,
            harmonic_replay,
        })
    }

    /// Exact normalization witness used for every derived defect.
    #[must_use]
    pub const fn normalization(self) -> RemeshNormalizationWitnessV1 {
        self.normalization
    }

    /// Density-weighted total momentum before transfer.
    #[must_use]
    pub const fn momentum_before(self) -> [f64; 2] {
        self.momentum_before
    }

    /// Density-weighted total momentum after transfer.
    #[must_use]
    pub const fn momentum_after(self) -> [f64; 2] {
        self.momentum_after
    }

    /// Raw source absolute-pressure zeroth moment.
    #[must_use]
    pub const fn pressure_moment_before(self) -> f64 {
        self.pressure_moment_before
    }

    /// Raw target absolute-pressure zeroth moment.
    #[must_use]
    pub const fn pressure_moment_after(self) -> f64 {
        self.pressure_moment_after
    }

    /// Raw coherent-SI weak-divergence norm.
    #[must_use]
    pub const fn raw_weak_divergence(self) -> f64 {
        self.raw_weak_divergence
    }

    /// Raw coherent-SI shared velocity-trace defect.
    #[must_use]
    pub const fn raw_shared_trace(self) -> f64 {
        self.raw_shared_trace
    }

    /// Raw coherent-SI exterior velocity-trace defect.
    #[must_use]
    pub const fn raw_exterior_trace(self) -> f64 {
        self.raw_exterior_trace
    }

    /// Raw coherent-SI displacement-trace defect.
    #[must_use]
    pub const fn raw_displacement_trace(self) -> f64 {
        self.raw_displacement_trace
    }

    /// Raw coherent-SI harmonic-coordinate replay defect.
    #[must_use]
    pub const fn raw_harmonic_coordinate_defect(self) -> f64 {
        self.raw_harmonic_coordinate_defect
    }

    /// Common dimensionless physical-obligation acceptance limit.
    #[must_use]
    pub const fn dimensionless_physical_acceptance_limit(self) -> f64 {
        self.momentum_defect.limit()
    }

    /// Accepted dimensionless momentum-functional defect.
    #[must_use]
    pub const fn momentum_defect(self) -> BoundedRemeshDefectV1 {
        self.momentum_defect
    }

    /// Accepted dimensionless weak-divergence defect.
    #[must_use]
    pub const fn weak_divergence(self) -> BoundedRemeshDefectV1 {
        self.weak_divergence
    }

    /// Accepted dimensionless shared velocity-trace defect.
    #[must_use]
    pub const fn shared_trace(self) -> BoundedRemeshDefectV1 {
        self.shared_trace
    }

    /// Accepted dimensionless exterior velocity-trace defect.
    #[must_use]
    pub const fn exterior_trace(self) -> BoundedRemeshDefectV1 {
        self.exterior_trace
    }

    /// Accepted dimensionless absolute-pressure moment defect.
    #[must_use]
    pub const fn pressure_zeroth_moment(self) -> BoundedRemeshDefectV1 {
        self.pressure_zeroth_moment
    }

    /// Accepted dimensionless displacement-trace defect.
    #[must_use]
    pub const fn displacement_trace(self) -> BoundedRemeshDefectV1 {
        self.displacement_trace
    }

    /// Accepted dimensionless harmonic-coordinate replay defect.
    #[must_use]
    pub const fn harmonic_replay(self) -> BoundedRemeshDefectV1 {
        self.harmonic_replay
    }
}

/// Durable receipt for the complete bounded FSI transfer.
#[derive(Debug, Clone, PartialEq)]
pub struct RemeshTransferReceiptEnvelopeV1 {
    wire: WireRemeshTransferReceiptV1,
}

impl RemeshTransferReceiptEnvelopeV1 {
    /// Bind four exact Field transfers and their coupled obligations.
    ///
    /// # Errors
    /// Returns `EQ0901` unless the source state, overlap, target geometry,
    /// target context, Field roles, snapshot lineage, and coupled velocity
    /// operator/solve identity are complete and exact.
    pub fn new<M: crate::ReplayableCanonicalModelArtifact>(
        source: &ValidatedRemeshGeometrySourceV2<'_, M>,
        overlap: &MeshRevisionOverlapEnvelopeV1,
        target_context: &ValidatedMovingSpatialContextV2<'_, M>,
        target_geometry_state: &GeometryStateEnvelopeV2,
        fields: impl IntoIterator<Item = FieldTransferReceiptV1>,
        projections: impl IntoIterator<Item = RemeshProjectionEvidenceEnvelopeV1>,
        evidence: RemeshTransferEvidenceV1,
    ) -> Result<Self, Diagnostic> {
        let source_state = source.state();
        if overlap.source_spatial_state_artifact() != source_state.digest()?
            || overlap.target_geometry_state_artifact() != target_geometry_state.digest()?
            || source_state.model_artifact() != *target_context.model_reference().artifact()
            || target_geometry_state.model_artifact()
                != *target_context.model_reference().artifact()
            || target_geometry_state.realization_artifact()
                != target_context.realization().digest()?
            || target_geometry_state.reference_mesh_artifact() != target_context.mesh().digest()?
        {
            return Err(invalid_artifact(
                "remesh transfer receipt has stale source, overlap, geometry, or target context",
            ));
        }
        let mut fields = fields.into_iter().collect::<Vec<_>>();
        fields.sort_by_key(|entry| entry.field().ulid());
        validate_complete_fields(source_state, target_context, target_geometry_state, &fields)?;
        let mut projections = projections.into_iter().collect::<Vec<_>>();
        projections.sort_by_key(RemeshProjectionEvidenceEnvelopeV1::action);
        let source_scales = realization_remesh_scales(source.context().realization())?;
        let target_scales = realization_remesh_scales(target_context.realization())?;
        validate_normalization_closure(
            source_scales,
            target_scales,
            evidence.normalization(),
            &projections,
        )?;
        validate_complete_projections(overlap, &fields, &projections)?;
        let velocity = fields
            .iter()
            .filter(|entry| {
                matches!(
                    entry.role(),
                    RemeshFieldRoleV1::FluidVelocity | RemeshFieldRoleV1::SolidVelocity
                )
            })
            .collect::<Vec<_>>();
        if velocity.len() != 2
            || velocity[0].projection_evidence() != velocity[1].projection_evidence()
        {
            return Err(invalid_artifact(
                "coupled remesh velocity Fields must share one typed projection evidence",
            ));
        }
        let value = Self {
            wire: WireRemeshTransferReceiptV1 {
                schema: TRANSFER_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                source_spatial_state_sha256: source_state.digest()?.to_string(),
                overlap_sha256: overlap.digest()?.to_string(),
                target_geometry_state_sha256: target_geometry_state.digest()?.to_string(),
                source_realization_sha256: source_state.realization_artifact().to_string(),
                target_realization_sha256: target_context.realization().digest()?.to_string(),
                fields: fields.into_iter().map(|entry| entry.wire).collect(),
                projections: projections
                    .into_iter()
                    .map(|projection| projection.wire)
                    .collect(),
                evidence: WireTransferEvidenceV1::encode(evidence),
                target_quality: WireTargetQualityV1 {
                    minimum_mean_ratio: target_geometry_state.minimum_mean_ratio(),
                    minimum_signed_measure_scale: target_geometry_state
                        .minimum_signed_measure_scale(),
                },
            },
        };
        value.validate_local(RemeshDecoderLimits::default())?;
        Ok(value)
    }

    /// Decode bounded receipt data without resolving dependencies.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed, oversized, unknown, or noncanonical data.
    pub fn from_json(bytes: &[u8], limits: RemeshDecoderLimits) -> Result<Self, Diagnostic> {
        check_json_limits(bytes, limits.json)?;
        let wire = serde_json::from_slice(bytes).map_err(|error| {
            invalid_artifact(format!("invalid remesh transfer receipt JSON: {error}"))
        })?;
        let value = Self { wire };
        value.validate_local(limits)?;
        Ok(value)
    }

    /// Deterministic canonical JSON bytes.
    ///
    /// # Errors
    /// Returns `EQ0901` if serialization unexpectedly fails.
    pub fn canonical_json(&self) -> Result<Vec<u8>, Diagnostic> {
        serde_json::to_vec(&self.wire).map_err(|error| {
            invalid_artifact(format!("cannot serialize remesh transfer receipt: {error}"))
        })
    }

    /// Domain-separated receipt identity.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            TRANSFER_SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
    }

    /// Exact source moving state.
    #[must_use]
    pub fn source_spatial_state(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.source_spatial_state_sha256.clone())
    }

    /// Exact source/target overlap artifact.
    #[must_use]
    pub fn overlap_artifact(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.overlap_sha256.clone())
    }

    /// Exact target remesh-origin geometry state.
    #[must_use]
    pub fn target_geometry_state(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.target_geometry_state_sha256.clone())
    }

    /// Exact source Realization.
    #[must_use]
    pub fn source_realization(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.source_realization_sha256.clone())
    }

    /// Exact target Realization.
    #[must_use]
    pub fn target_realization(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.target_realization_sha256.clone())
    }

    /// Canonically ordered Field-local receipts.
    #[must_use]
    pub fn fields(&self) -> Vec<FieldTransferReceiptV1> {
        self.wire
            .fields
            .iter()
            .cloned()
            .map(|wire| FieldTransferReceiptV1 { wire })
            .collect()
    }

    /// Embedded self-contained projection evidence in closed action order.
    ///
    /// # Errors
    /// Returns `EQ0901` if any embedded evidence is invalid.
    pub fn projections(&self) -> Result<Vec<RemeshProjectionEvidenceEnvelopeV1>, Diagnostic> {
        self.projections_with_limits(RemeshDecoderLimits::default())
    }

    fn projections_with_limits(
        &self,
        limits: RemeshDecoderLimits,
    ) -> Result<Vec<RemeshProjectionEvidenceEnvelopeV1>, Diagnostic> {
        self.wire
            .projections
            .iter()
            .cloned()
            .map(|wire| {
                let value = RemeshProjectionEvidenceEnvelopeV1 { wire };
                value.validate_local(limits)?;
                Ok(value)
            })
            .collect()
    }

    /// Rebuild and compare from exact dependencies.
    ///
    /// # Errors
    /// Returns `EQ0901` for any substituted resource, Field, or evidence value.
    pub fn validate_against<M: crate::ReplayableCanonicalModelArtifact>(
        &self,
        source: &ValidatedRemeshGeometrySourceV2<'_, M>,
        overlap: &MeshRevisionOverlapEnvelopeV1,
        target_context: &ValidatedMovingSpatialContextV2<'_, M>,
        target_geometry_state: &GeometryStateEnvelopeV2,
        source_snapshots: &[FieldSnapshotEnvelopeV2],
        target_snapshots: &[FieldSnapshotEnvelopeV2],
    ) -> Result<(), Diagnostic> {
        let projections = self.projections()?;
        for projection in &projections {
            projection.validate_against_overlap(overlap)?;
        }
        let fields = replay_field_receipts(self, &projections, source_snapshots, target_snapshots)?;
        let expected = Self::new(
            source,
            overlap,
            target_context,
            target_geometry_state,
            fields,
            projections,
            self.wire.evidence.decode()?,
        )?;
        if self == &expected {
            Ok(())
        } else {
            Err(invalid_artifact(
                "remesh transfer receipt differs from exact dependency replay",
            ))
        }
    }

    fn validate_local(&self, limits: RemeshDecoderLimits) -> Result<(), Diagnostic> {
        if self.wire.schema != TRANSFER_SCHEMA || self.wire.encoding != CANONICAL_ENCODING {
            return Err(invalid_artifact(
                "unsupported remesh transfer receipt schema or encoding",
            ));
        }
        for digest in [
            &self.wire.source_spatial_state_sha256,
            &self.wire.overlap_sha256,
            &self.wire.target_geometry_state_sha256,
            &self.wire.source_realization_sha256,
            &self.wire.target_realization_sha256,
        ] {
            ArtifactDigest::from_hex(digest.clone())?;
        }
        if self.wire.source_realization_sha256 == self.wire.target_realization_sha256 {
            return Err(invalid_artifact(
                "remesh transfer receipt requires distinct mesh-bound Realizations",
            ));
        }
        if self.wire.fields.len() != 4 || self.wire.fields.len() > limits.max_remesh_transfer_fields
        {
            return Err(invalid_artifact(
                "remesh transfer receipt requires exactly four bounded FSI Fields",
            ));
        }
        let projections = self.projections_with_limits(limits)?;
        if projections.len() != 3 || projections.len() > limits.max_remesh_transfer_fields {
            return Err(invalid_artifact(
                "remesh transfer receipt requires exactly three typed projection actions",
            ));
        }
        let fields = self.fields();
        if fields
            .windows(2)
            .any(|pair| pair[0].field().ulid() >= pair[1].field().ulid())
            || fields
                .iter()
                .try_fold(BTreeSet::new(), |mut roles, entry| {
                    entry.validate_local()?;
                    if !roles.insert(entry.role()) {
                        return Err(invalid_artifact(
                            "remesh transfer Field roles must be complete and unique",
                        ));
                    }
                    Ok(roles)
                })?
                != BTreeSet::from([
                    RemeshFieldRoleV1::FluidVelocity,
                    RemeshFieldRoleV1::SolidVelocity,
                    RemeshFieldRoleV1::FluidPressure,
                    RemeshFieldRoleV1::SolidDisplacement,
                ])
        {
            return Err(invalid_artifact(
                "remesh transfer Fields are reordered, duplicated, or incomplete",
            ));
        }
        validate_complete_projections_local(&fields, &projections)?;
        let evidence = self.wire.evidence.decode()?;
        validate_evidence_projection_normalization(evidence.normalization(), &projections)?;
        let quality = self.wire.target_quality;
        if !quality.minimum_mean_ratio.is_finite()
            || quality.minimum_mean_ratio <= 0.0
            || quality.minimum_mean_ratio > 1.0
            || !quality.minimum_signed_measure_scale.is_finite()
            || quality.minimum_signed_measure_scale <= 0.0
            || is_negative_zero(quality.minimum_mean_ratio)
            || is_negative_zero(quality.minimum_signed_measure_scale)
        {
            return Err(invalid_artifact(
                "remesh target quality must be finite, canonical, and positive",
            ));
        }
        Ok(())
    }
}

fn validate_complete_fields<M: crate::ReplayableCanonicalModelArtifact>(
    source_state: &SpatialStateEnvelopeV2,
    target_context: &ValidatedMovingSpatialContextV2<'_, M>,
    target_geometry: &GeometryStateEnvelopeV2,
    fields: &[FieldTransferReceiptV1],
) -> Result<(), Diagnostic> {
    let requirements = target_context.realization().requirements()?;
    let fluid_velocity = requirements.fluid_velocity();
    let solid_displacement = requirements.solid_displacement();
    let solid_velocity = requirements
        .coupled()
        .eliminated_states()
        .iter()
        .find(|state| state.state() == requirements.solid_displacement())
        .ok_or_else(|| invalid_artifact("remesh driver lacks exact state/rate binding"))?
        .rate();
    let pressure = requirements
        .coupled()
        .domains()
        .iter()
        .find(|domain| domain.domain() == requirements.fluid_domain())
        .and_then(|domain| {
            domain
                .fields()
                .iter()
                .copied()
                .find(|field| *field != fluid_velocity)
        })
        .ok_or_else(|| invalid_artifact("target ALE requirements omit fluid pressure"))?;
    let expected = [
        (fluid_velocity, RemeshFieldRoleV1::FluidVelocity),
        (solid_velocity, RemeshFieldRoleV1::SolidVelocity),
        (pressure, RemeshFieldRoleV1::FluidPressure),
        (solid_displacement, RemeshFieldRoleV1::SolidDisplacement),
    ];
    for (field, role) in expected {
        let entry = fields
            .iter()
            .find(|entry| entry.field() == field)
            .ok_or_else(|| invalid_artifact("remesh transfer omits one exact ALE Field"))?;
        if entry.role() != role
            || source_state.field_snapshot(field) != Some(entry.source_snapshot())
            || (field == solid_displacement
                && entry.target_snapshot() != target_geometry.solid_displacement_snapshot())
        {
            return Err(invalid_artifact(
                "remesh transfer Field role or snapshot lineage differs from exact ALE requirements",
            ));
        }
    }
    Ok(())
}

fn replay_field_receipts(
    receipt: &RemeshTransferReceiptEnvelopeV1,
    projections: &[RemeshProjectionEvidenceEnvelopeV1],
    source_snapshots: &[FieldSnapshotEnvelopeV2],
    target_snapshots: &[FieldSnapshotEnvelopeV2],
) -> Result<Vec<FieldTransferReceiptV1>, Diagnostic> {
    receipt
        .fields()
        .into_iter()
        .map(|entry| {
            let source = source_snapshots
                .iter()
                .find(|snapshot| snapshot.digest().ok().as_ref() == Some(&entry.source_snapshot()))
                .ok_or_else(|| invalid_artifact("remesh receipt source snapshot is missing"))?;
            let target = target_snapshots
                .iter()
                .find(|snapshot| snapshot.digest().ok().as_ref() == Some(&entry.target_snapshot()))
                .ok_or_else(|| invalid_artifact("remesh receipt target snapshot is missing"))?;
            let projection = projections
                .iter()
                .find(|projection| {
                    projection.digest().ok().as_ref() == Some(&entry.projection_evidence())
                })
                .ok_or_else(|| {
                    invalid_artifact("remesh receipt projection evidence is unresolved")
                })?;
            FieldTransferReceiptV1::new(
                entry.role(),
                entry.law(),
                entry.chart(),
                source,
                target,
                projection,
                entry.raw_projection_error_l2(),
            )
        })
        .collect()
}

fn validate_complete_projections(
    overlap: &MeshRevisionOverlapEnvelopeV1,
    fields: &[FieldTransferReceiptV1],
    projections: &[RemeshProjectionEvidenceEnvelopeV1],
) -> Result<(), Diagnostic> {
    for projection in projections {
        projection.validate_against_overlap(overlap)?;
    }
    validate_complete_projections_local(fields, projections)
}

fn validate_complete_projections_local(
    fields: &[FieldTransferReceiptV1],
    projections: &[RemeshProjectionEvidenceEnvelopeV1],
) -> Result<(), Diagnostic> {
    let actions = projections
        .iter()
        .map(RemeshProjectionEvidenceEnvelopeV1::action)
        .collect::<BTreeSet<_>>();
    if actions
        != BTreeSet::from([
            RemeshProjectionActionV1::CoupledVelocity,
            RemeshProjectionActionV1::AbsolutePressure,
            RemeshProjectionActionV1::AbsoluteDisplacement,
        ])
        || projections
            .windows(2)
            .any(|pair| pair[0].action() >= pair[1].action())
    {
        return Err(invalid_artifact(
            "remesh projection actions must be complete, unique, and canonical",
        ));
    }
    for field in fields {
        let projection = projections
            .iter()
            .find(|projection| {
                projection.digest().ok().as_ref() == Some(&field.projection_evidence())
            })
            .ok_or_else(|| invalid_artifact("remesh Field references unresolved evidence"))?;
        require_role_action(field.role(), projection.action())?;
    }
    let plans = projections
        .iter()
        .map(RemeshProjectionEvidenceEnvelopeV1::plan)
        .collect::<Result<Vec<_>, _>>()?;
    if plans.windows(2).any(|pair| pair[0] != pair[1]) {
        return Err(invalid_artifact(
            "all remesh projections must share one exact transfer plan",
        ));
    }
    Ok(())
}

fn require_role_action(
    role: RemeshFieldRoleV1,
    action: RemeshProjectionActionV1,
) -> Result<(), Diagnostic> {
    let valid = matches!(
        (role, action),
        (
            RemeshFieldRoleV1::FluidVelocity | RemeshFieldRoleV1::SolidVelocity,
            RemeshProjectionActionV1::CoupledVelocity,
        ) | (
            RemeshFieldRoleV1::FluidPressure,
            RemeshProjectionActionV1::AbsolutePressure,
        ) | (
            RemeshFieldRoleV1::SolidDisplacement,
            RemeshProjectionActionV1::AbsoluteDisplacement,
        )
    );
    if valid {
        Ok(())
    } else {
        Err(invalid_artifact(
            "remesh Field role and typed projection action are incompatible",
        ))
    }
}

fn require_role_law_chart(
    role: RemeshFieldRoleV1,
    law: RemeshTransferLawV1,
    chart: RemeshIntegrationChartV1,
) -> Result<(), Diagnostic> {
    let valid = matches!(
        (role, law, chart),
        (
            RemeshFieldRoleV1::FluidVelocity,
            RemeshTransferLawV1::CoupledVelocityConstrainedL2,
            RemeshIntegrationChartV1::CurrentSpatial,
        ) | (
            RemeshFieldRoleV1::SolidVelocity,
            RemeshTransferLawV1::CoupledVelocityConstrainedL2,
            RemeshIntegrationChartV1::Material,
        ) | (
            RemeshFieldRoleV1::FluidPressure,
            RemeshTransferLawV1::AbsolutePressureL2,
            RemeshIntegrationChartV1::CurrentSpatial,
        ) | (
            RemeshFieldRoleV1::SolidDisplacement,
            RemeshTransferLawV1::AbsoluteDisplacementL2,
            RemeshIntegrationChartV1::Material,
        )
    );
    if valid {
        Ok(())
    } else {
        Err(invalid_artifact(
            "remesh Field role, variational law, and integration chart are incompatible",
        ))
    }
}

fn parse_field(value: &str) -> Result<Id<kinds::Field>, Diagnostic> {
    let ulid = value
        .parse::<Ulid>()
        .map_err(|_| invalid_artifact("remesh transfer Field ULID is malformed"))?;
    if ulid.to_string() != value {
        return Err(invalid_artifact(
            "remesh transfer Field ULID spelling is noncanonical",
        ));
    }
    Ok(Id::from_ulid(ulid))
}

fn normalize_array(values: &mut [f64]) {
    for value in values {
        *value = normalize_zero(*value);
    }
}

fn checked_positive_product(left: f64, right: f64, name: &'static str) -> Result<f64, Diagnostic> {
    let value = left * right;
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(invalid_artifact(format!(
            "remesh characteristic {name} must be finite and strictly positive",
        )))
    }
}

fn require_execution_identity(value: &str, label: &str) -> Result<(), Diagnostic> {
    if value.is_empty()
        || value.len() > 256
        || !value.is_ascii()
        || value.starts_with('.')
        || value.ends_with('.')
        || !value.contains('.')
        || value.bytes().any(|byte| {
            !(byte.is_ascii_lowercase() || byte.is_ascii_digit() || b".-_".contains(&byte))
        })
    {
        Err(invalid_artifact(format!(
            "remesh {label} identity must be bounded namespaced lowercase ASCII",
        )))
    } else {
        Ok(())
    }
}

fn encode_nonzero_usize(value: NonZeroUsize) -> Result<u64, Diagnostic> {
    u64::try_from(value.get())
        .map_err(|_| invalid_artifact("remesh execution count exceeds portable u64"))
}

fn decode_nonzero_usize(value: u64) -> Result<NonZeroUsize, Diagnostic> {
    usize::try_from(value)
        .ok()
        .and_then(NonZeroUsize::new)
        .ok_or_else(|| invalid_artifact("remesh execution count is zero or exceeds usize"))
}

const fn normalize_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

fn is_negative_zero(value: f64) -> bool {
    value == 0.0 && value.is_sign_negative()
}

const fn length_dimension() -> DimExponents {
    DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension")
}

const fn velocity_dimension() -> DimExponents {
    DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).expect("bounded dimension")
}

const fn pressure_dimension() -> DimExponents {
    DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).expect("bounded dimension")
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireRemeshTransferReceiptV1 {
    schema: String,
    encoding: String,
    source_spatial_state_sha256: String,
    overlap_sha256: String,
    target_geometry_state_sha256: String,
    source_realization_sha256: String,
    target_realization_sha256: String,
    fields: Vec<WireFieldTransferReceiptV1>,
    projections: Vec<WireRemeshProjectionEvidenceV1>,
    evidence: WireTransferEvidenceV1,
    target_quality: WireTargetQualityV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireFieldTransferReceiptV1 {
    field_ulid: String,
    role: WireFieldRoleV1,
    law: WireTransferLawV1,
    chart: WireIntegrationChartV1,
    source_snapshot_sha256: String,
    target_snapshot_sha256: String,
    projection_evidence_sha256: String,
    raw_projection_error_l2: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireRemeshProjectionEvidenceV1 {
    schema: String,
    encoding: String,
    action_version: String,
    action: WireProjectionActionV1,
    execution: WireProjectionExecutionV1,
    overlap_sha256: String,
    plan: WireRemeshTransferPlanV1,
    algebraic_replay: WireBoundedDefectV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireProjectionSolveV1 {
    component: u8,
    right_hand_side_norm: f64,
    report: WireSolveReportV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireProjectionActionV1 {
    CoupledVelocity,
    AbsolutePressure,
    AbsoluteDisplacement,
}

impl WireProjectionActionV1 {
    const fn encode(value: RemeshProjectionActionV1) -> Self {
        match value {
            RemeshProjectionActionV1::CoupledVelocity => Self::CoupledVelocity,
            RemeshProjectionActionV1::AbsolutePressure => Self::AbsolutePressure,
            RemeshProjectionActionV1::AbsoluteDisplacement => Self::AbsoluteDisplacement,
        }
    }

    const fn decode(self) -> RemeshProjectionActionV1 {
        match self {
            Self::CoupledVelocity => RemeshProjectionActionV1::CoupledVelocity,
            Self::AbsolutePressure => RemeshProjectionActionV1::AbsolutePressure,
            Self::AbsoluteDisplacement => RemeshProjectionActionV1::AbsoluteDisplacement,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireProjectionExecutionV1 {
    PrescribedExactly,
    SolvedScalar {
        solve: Box<WireProjectionSolveV1>,
    },
    SolvedVector2 {
        solves: Box<[WireProjectionSolveV1; 2]>,
    },
}

impl WireProjectionExecutionV1 {
    const fn mode(&self) -> RemeshProjectionExecutionModeV1 {
        match self {
            Self::PrescribedExactly => RemeshProjectionExecutionModeV1::PrescribedExactly,
            Self::SolvedScalar { .. } | Self::SolvedVector2 { .. } => {
                RemeshProjectionExecutionModeV1::Solved
            }
        }
    }

    const fn solve_count(&self) -> usize {
        match self {
            Self::PrescribedExactly => 0,
            Self::SolvedScalar { .. } => 1,
            Self::SolvedVector2 { .. } => 2,
        }
    }

    fn solve(&self, component: usize) -> Option<&WireProjectionSolveV1> {
        match (self, component) {
            (Self::SolvedScalar { solve }, 0) => Some(solve),
            (Self::SolvedVector2 { solves }, component) => solves.get(component),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireRemeshTransferPlanV1 {
    quadrature: WireTransferQuadratureV1,
    scales: WireRemeshScaleProfileV1,
    solver: WireSolverPlan,
}

impl WireRemeshTransferPlanV1 {
    fn encode(value: AleFsiRemeshTransferPlan2d) -> Result<Self, Diagnostic> {
        let quadrature = match value.quadrature() {
            QuadraturePolicy::TriangleDuffyGaussLegendre { points_per_axis }
                if points_per_axis.get() == 5 =>
            {
                WireTransferQuadratureV1::TriangleDuffyGaussLegendre5x5
            }
            _ => {
                return Err(invalid_artifact(
                    "remesh projection plan has an unsupported quadrature",
                ));
            }
        };
        Ok(Self {
            quadrature,
            scales: WireRemeshScaleProfileV1::encode(value.scales()),
            solver: WireSolverPlan::encode(value.solver())?,
        })
    }

    fn decode(self) -> Result<AleFsiRemeshTransferPlan2d, Diagnostic> {
        let quadrature = match self.quadrature {
            WireTransferQuadratureV1::TriangleDuffyGaussLegendre5x5 => {
                QuadraturePolicy::TriangleDuffyGaussLegendre {
                    points_per_axis: NonZeroUsize::new(5).expect("five is non-zero"),
                }
            }
        };
        AleFsiRemeshTransferPlan2d::new(quadrature, self.scales.decode()?, self.solver.decode()?)
            .map_err(|error| invalid_artifact(error.message()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireRemeshScaleProfileV1 {
    length_m: f64,
    velocity_m_per_s: f64,
    pressure_pa: f64,
}

impl WireRemeshScaleProfileV1 {
    fn encode(value: AleFsiRemeshScaleProfile2d) -> Self {
        Self {
            length_m: normalize_zero(value.length().value()),
            velocity_m_per_s: normalize_zero(value.velocity().value()),
            pressure_pa: normalize_zero(value.pressure().value()),
        }
    }

    fn decode(self) -> Result<AleFsiRemeshScaleProfile2d, Diagnostic> {
        let value = AleFsiRemeshScaleProfile2d::new(
            DynQuantity::new(self.length_m, length_dimension()),
            DynQuantity::new(self.velocity_m_per_s, velocity_dimension()),
            DynQuantity::new(self.pressure_pa, pressure_dimension()),
        )
        .map_err(|error| invalid_artifact(error.message()))?;
        if Self::encode(value) != self {
            return Err(invalid_artifact(
                "remesh scale profile wire is noncanonical",
            ));
        }
        Ok(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireTransferQuadratureV1 {
    TriangleDuffyGaussLegendre5x5,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireSolveReportV1 {
    backend: String,
    execution: WireExecutionReportV1,
    verification: WireExecutionReportV1,
    orientation: WireOperatorOrientationV1,
    plan: WireSolverPlan,
    reason: WireConvergenceReasonV1,
    completed_iterations: u64,
    initial_residual_norm: f64,
    reported_residual_norm: f64,
    true_residual_norm: f64,
    residual_target: f64,
}

impl WireSolveReportV1 {
    fn encode(value: &SolveReport) -> Result<Self, Diagnostic> {
        Ok(Self {
            backend: value.backend().as_str().to_owned(),
            execution: WireExecutionReportV1::encode(value.execution())?,
            verification: WireExecutionReportV1::encode(value.verification())?,
            orientation: WireOperatorOrientationV1::encode(value.orientation()),
            plan: WireSolverPlan::encode(value.solver_plan())?,
            reason: WireConvergenceReasonV1::encode(value.reason()),
            completed_iterations: u64::try_from(value.completed_iterations())
                .map_err(|_| invalid_artifact("solve-report iteration count exceeds u64"))?,
            initial_residual_norm: normalize_zero(value.initial_residual_norm()),
            reported_residual_norm: normalize_zero(value.reported_residual_norm()),
            true_residual_norm: normalize_zero(value.true_residual_norm()),
            residual_target: normalize_zero(value.residual_target()),
        })
    }

    fn validate(
        &self,
        common_plan: SolverPlan,
        right_hand_side_norm: f64,
    ) -> Result<(), Diagnostic> {
        require_execution_identity(&self.backend, "solver backend")?;
        self.execution.validate()?;
        self.verification.validate()?;
        if self.orientation != WireOperatorOrientationV1::Normal
            || self.plan.decode()? != common_plan
            || !right_hand_side_norm.is_finite()
            || right_hand_side_norm < 0.0
            || is_negative_zero(right_hand_side_norm)
        {
            return Err(invalid_artifact(
                "remesh solve report differs from the common normal-action plan",
            ));
        }
        let completed = usize::try_from(self.completed_iterations)
            .map_err(|_| invalid_artifact("solve-report iteration count exceeds usize"))?;
        let values = [
            self.initial_residual_norm,
            self.reported_residual_norm,
            self.true_residual_norm,
            self.residual_target,
        ];
        if values
            .into_iter()
            .any(|value| !value.is_finite() || value < 0.0 || is_negative_zero(value))
            || completed > common_plan.maximum_iterations().get()
            || self.residual_target
                != common_plan
                    .residual_target(right_hand_side_norm)
                    .map_err(|error| invalid_artifact(error.message()))?
            || self.true_residual_norm > self.residual_target
        {
            return Err(invalid_artifact(
                "remesh solve report is noncanonical or fails independent residual acceptance",
            ));
        }
        match (self.reason, completed) {
            (WireConvergenceReasonV1::InitialResidualSatisfied, 0)
                if self.initial_residual_norm <= self.residual_target =>
            {
                Ok(())
            }
            (WireConvergenceReasonV1::ResidualToleranceSatisfied, 1..) => Ok(()),
            _ => Err(invalid_artifact(
                "remesh solve report has inconsistent termination evidence",
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireExecutionReportV1 {
    adapter: String,
    topology: WireExecutionTopologyV1,
}

impl WireExecutionReportV1 {
    fn encode(value: ExecutionReport) -> Result<Self, Diagnostic> {
        Ok(Self {
            adapter: value.adapter().as_str().to_owned(),
            topology: WireExecutionTopologyV1::encode(value.topology())?,
        })
    }

    fn validate(&self) -> Result<(), Diagnostic> {
        require_execution_identity(&self.adapter, "execution adapter")?;
        self.topology.validate()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireExecutionTopologyV1 {
    Host {
        workers: u64,
    },
    Distributed {
        ranks: u64,
        workers_per_partition: u64,
    },
    Cuda {
        device: u16,
    },
}

impl WireExecutionTopologyV1 {
    fn encode(value: ExecutionTopology) -> Result<Self, Diagnostic> {
        Ok(match value {
            ExecutionTopology::Host { workers } => Self::Host {
                workers: encode_nonzero_usize(workers)?,
            },
            ExecutionTopology::Distributed {
                ranks,
                workers_per_partition,
            } => Self::Distributed {
                ranks: encode_nonzero_usize(ranks)?,
                workers_per_partition: encode_nonzero_usize(workers_per_partition)?,
            },
            ExecutionTopology::Cuda { device } => Self::Cuda { device },
        })
    }

    fn validate(self) -> Result<(), Diagnostic> {
        match self {
            Self::Host { workers } => decode_nonzero_usize(workers).map(drop),
            Self::Distributed {
                ranks,
                workers_per_partition,
            } => {
                decode_nonzero_usize(ranks)?;
                decode_nonzero_usize(workers_per_partition).map(drop)
            }
            Self::Cuda { .. } => Ok(()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireOperatorOrientationV1 {
    Normal,
    Transposed,
}

impl WireOperatorOrientationV1 {
    const fn encode(value: LinearOperatorOrientation) -> Self {
        match value {
            LinearOperatorOrientation::Normal => Self::Normal,
            LinearOperatorOrientation::Transposed => Self::Transposed,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireConvergenceReasonV1 {
    InitialResidualSatisfied,
    ResidualToleranceSatisfied,
}

impl WireConvergenceReasonV1 {
    const fn encode(value: ConvergenceReason) -> Self {
        match value {
            ConvergenceReason::InitialResidualSatisfied => Self::InitialResidualSatisfied,
            ConvergenceReason::ResidualToleranceSatisfied => Self::ResidualToleranceSatisfied,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireTransferEvidenceV1 {
    normalization: WireRemeshNormalizationWitnessV1,
    raw_momentum_before: [f64; 2],
    raw_momentum_after: [f64; 2],
    raw_pressure_moment_before: f64,
    raw_pressure_moment_after: f64,
    raw_weak_divergence: f64,
    raw_shared_trace: f64,
    raw_exterior_trace: f64,
    raw_displacement_trace: f64,
    raw_harmonic_coordinate_defect: f64,
    dimensionless_physical_acceptance_limit: f64,
    dimensionless_observed: WirePhysicalDefectsV1,
}

impl WireTransferEvidenceV1 {
    fn encode(value: RemeshTransferEvidenceV1) -> Self {
        Self {
            normalization: WireRemeshNormalizationWitnessV1::encode(value.normalization),
            raw_momentum_before: value.momentum_before,
            raw_momentum_after: value.momentum_after,
            raw_pressure_moment_before: value.pressure_moment_before,
            raw_pressure_moment_after: value.pressure_moment_after,
            raw_weak_divergence: value.raw_weak_divergence,
            raw_shared_trace: value.raw_shared_trace,
            raw_exterior_trace: value.raw_exterior_trace,
            raw_displacement_trace: value.raw_displacement_trace,
            raw_harmonic_coordinate_defect: value.raw_harmonic_coordinate_defect,
            dimensionless_physical_acceptance_limit: value.momentum_defect.limit,
            dimensionless_observed: WirePhysicalDefectsV1 {
                momentum: value.momentum_defect.observed,
                weak_divergence: value.weak_divergence.observed,
                shared_trace: value.shared_trace.observed,
                exterior_trace: value.exterior_trace.observed,
                pressure_zeroth_moment: value.pressure_zeroth_moment.observed,
                displacement_trace: value.displacement_trace.observed,
                harmonic_replay: value.harmonic_replay.observed,
            },
        }
    }

    fn decode(self) -> Result<RemeshTransferEvidenceV1, Diagnostic> {
        let value = RemeshTransferEvidenceV1::new(
            self.normalization.decode()?,
            self.raw_momentum_before,
            self.raw_momentum_after,
            self.raw_pressure_moment_before,
            self.raw_pressure_moment_after,
            self.raw_weak_divergence,
            self.raw_shared_trace,
            self.raw_exterior_trace,
            self.raw_displacement_trace,
            self.raw_harmonic_coordinate_defect,
            self.dimensionless_physical_acceptance_limit,
        )?;
        if Self::encode(value) != self {
            return Err(invalid_artifact(
                "remesh transfer evidence wire is noncanonical",
            ));
        }
        Ok(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireRemeshNormalizationWitnessV1 {
    scales: WireRemeshScaleProfileV1,
    reference_density_kg_per_m3: f64,
}

impl WireRemeshNormalizationWitnessV1 {
    fn encode(value: RemeshNormalizationWitnessV1) -> Self {
        Self {
            scales: WireRemeshScaleProfileV1::encode(value.scales),
            reference_density_kg_per_m3: value.reference_density,
        }
    }

    fn decode(self) -> Result<RemeshNormalizationWitnessV1, Diagnostic> {
        let value = RemeshNormalizationWitnessV1::new(
            self.scales.decode()?,
            self.reference_density_kg_per_m3,
        )?;
        if Self::encode(value) == self {
            Ok(value)
        } else {
            Err(invalid_artifact(
                "remesh normalization witness wire is noncanonical",
            ))
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WirePhysicalDefectsV1 {
    momentum: f64,
    weak_divergence: f64,
    shared_trace: f64,
    exterior_trace: f64,
    pressure_zeroth_moment: f64,
    displacement_trace: f64,
    harmonic_replay: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireBoundedDefectV1 {
    observed_dimensionless: f64,
    limit_dimensionless: f64,
}

impl WireBoundedDefectV1 {
    const fn zero() -> Self {
        Self {
            observed_dimensionless: 0.0,
            limit_dimensionless: 0.0,
        }
    }

    const fn encode(value: BoundedRemeshDefectV1) -> Self {
        Self {
            observed_dimensionless: value.observed,
            limit_dimensionless: value.limit,
        }
    }

    const fn decode(self) -> BoundedRemeshDefectV1 {
        BoundedRemeshDefectV1 {
            observed: self.observed_dimensionless,
            limit: self.limit_dimensionless,
        }
    }

    fn validate(self) -> Result<(), Diagnostic> {
        let expected =
            BoundedRemeshDefectV1::new(self.observed_dimensionless, self.limit_dimensionless)?;
        if Self::encode(expected) == self {
            Ok(())
        } else {
            Err(invalid_artifact(
                "remesh transfer defect wire is noncanonical",
            ))
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireTargetQualityV1 {
    minimum_mean_ratio: f64,
    minimum_signed_measure_scale: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireFieldRoleV1 {
    FluidVelocity,
    SolidVelocity,
    FluidPressure,
    SolidDisplacement,
}

impl WireFieldRoleV1 {
    const fn encode(value: RemeshFieldRoleV1) -> Self {
        match value {
            RemeshFieldRoleV1::FluidVelocity => Self::FluidVelocity,
            RemeshFieldRoleV1::SolidVelocity => Self::SolidVelocity,
            RemeshFieldRoleV1::FluidPressure => Self::FluidPressure,
            RemeshFieldRoleV1::SolidDisplacement => Self::SolidDisplacement,
        }
    }

    const fn decode(self) -> RemeshFieldRoleV1 {
        match self {
            Self::FluidVelocity => RemeshFieldRoleV1::FluidVelocity,
            Self::SolidVelocity => RemeshFieldRoleV1::SolidVelocity,
            Self::FluidPressure => RemeshFieldRoleV1::FluidPressure,
            Self::SolidDisplacement => RemeshFieldRoleV1::SolidDisplacement,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireTransferLawV1 {
    CoupledVelocityConstrainedL2,
    AbsolutePressureL2,
    AbsoluteDisplacementL2,
}

impl WireTransferLawV1 {
    const fn encode(value: RemeshTransferLawV1) -> Self {
        match value {
            RemeshTransferLawV1::CoupledVelocityConstrainedL2 => Self::CoupledVelocityConstrainedL2,
            RemeshTransferLawV1::AbsolutePressureL2 => Self::AbsolutePressureL2,
            RemeshTransferLawV1::AbsoluteDisplacementL2 => Self::AbsoluteDisplacementL2,
        }
    }

    const fn decode(self) -> RemeshTransferLawV1 {
        match self {
            Self::CoupledVelocityConstrainedL2 => RemeshTransferLawV1::CoupledVelocityConstrainedL2,
            Self::AbsolutePressureL2 => RemeshTransferLawV1::AbsolutePressureL2,
            Self::AbsoluteDisplacementL2 => RemeshTransferLawV1::AbsoluteDisplacementL2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireIntegrationChartV1 {
    Material,
    CurrentSpatial,
}

impl WireIntegrationChartV1 {
    const fn encode(value: RemeshIntegrationChartV1) -> Self {
        match value {
            RemeshIntegrationChartV1::Material => Self::Material,
            RemeshIntegrationChartV1::CurrentSpatial => Self::CurrentSpatial,
        }
    }

    const fn decode(self) -> RemeshIntegrationChartV1 {
        match self {
            Self::Material => RemeshIntegrationChartV1::Material,
            Self::CurrentSpatial => RemeshIntegrationChartV1::CurrentSpatial,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_solver::{
        BackendId, ExecutionReport, LinearSolver, ReductionPolicy, SERIAL_EXECUTION_PROVIDER,
        SolverProvider,
    };

    const TEST_MINRES_PROVIDER: SolverProvider = SolverProvider::new(
        BackendId::new("eqiora.test.minres"),
        env!("CARGO_PKG_VERSION"),
        &[],
    );

    fn scales(length: f64, velocity: f64, pressure: f64) -> AleFsiRemeshScaleProfile2d {
        AleFsiRemeshScaleProfile2d::new(
            DynQuantity::new(length, length_dimension()),
            DynQuantity::new(velocity, velocity_dimension()),
            DynQuantity::new(pressure, pressure_dimension()),
        )
        .unwrap()
    }

    fn plan_with_scales(scales: AleFsiRemeshScaleProfile2d) -> AleFsiRemeshTransferPlan2d {
        AleFsiRemeshTransferPlan2d::new(
            QuadraturePolicy::TriangleDuffyGaussLegendre {
                points_per_axis: NonZeroUsize::new(5).unwrap(),
            },
            scales,
            SolverPlan::new(
                LinearSolver::MinimumResidual,
                0.0,
                1.0e-12,
                NonZeroUsize::new(5).unwrap(),
            )
            .unwrap()
            .with_reduction(ReductionPolicy::Reproducible),
        )
        .unwrap()
    }

    fn plan() -> AleFsiRemeshTransferPlan2d {
        plan_with_scales(scales(2.0, 0.5, 3.0))
    }

    fn alternative_plan() -> AleFsiRemeshTransferPlan2d {
        plan_with_scales(scales(4.0, 1.0, 6.0))
    }

    fn solve(component: u8, transfer_plan: AleFsiRemeshTransferPlan2d) -> WireProjectionSolveV1 {
        let solver_plan = transfer_plan.solver();
        let report = SolveReport::accepted(
            TEST_MINRES_PROVIDER,
            SERIAL_EXECUTION_PROVIDER,
            ExecutionReport::host_serial(),
            LinearOperatorOrientation::Normal,
            solver_plan,
            ConvergenceReason::InitialResidualSatisfied,
            0,
            0.0,
            0.0,
            0.0,
            solver_plan.residual_target(0.0).unwrap(),
        )
        .unwrap();
        WireProjectionSolveV1 {
            component,
            right_hand_side_norm: 0.0,
            report: WireSolveReportV1::encode(&report).unwrap(),
        }
    }

    fn projection(execution: WireProjectionExecutionV1) -> RemeshProjectionEvidenceEnvelopeV1 {
        projection_with(
            RemeshProjectionActionV1::AbsoluteDisplacement,
            execution,
            plan(),
        )
    }

    fn projection_with(
        action: RemeshProjectionActionV1,
        execution: WireProjectionExecutionV1,
        transfer_plan: AleFsiRemeshTransferPlan2d,
    ) -> RemeshProjectionEvidenceEnvelopeV1 {
        RemeshProjectionEvidenceEnvelopeV1 {
            wire: WireRemeshProjectionEvidenceV1 {
                schema: PROJECTION_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                action_version: TRANSFER_ACTION_VERSION.to_owned(),
                action: WireProjectionActionV1::encode(action),
                execution,
                overlap_sha256: "11".repeat(32),
                plan: WireRemeshTransferPlanV1::encode(transfer_plan).unwrap(),
                algebraic_replay: WireBoundedDefectV1::zero(),
            },
        }
    }

    fn projection_for(
        action: RemeshProjectionActionV1,
        transfer_plan: AleFsiRemeshTransferPlan2d,
    ) -> RemeshProjectionEvidenceEnvelopeV1 {
        let execution = match action {
            RemeshProjectionActionV1::CoupledVelocity
            | RemeshProjectionActionV1::AbsolutePressure => {
                WireProjectionExecutionV1::SolvedScalar {
                    solve: Box::new(solve(0, transfer_plan)),
                }
            }
            RemeshProjectionActionV1::AbsoluteDisplacement => {
                WireProjectionExecutionV1::SolvedVector2 {
                    solves: Box::new([solve(0, transfer_plan), solve(1, transfer_plan)]),
                }
            }
        };
        projection_with(action, execution, transfer_plan)
    }

    fn projections(
        transfer_plan: AleFsiRemeshTransferPlan2d,
    ) -> Vec<RemeshProjectionEvidenceEnvelopeV1> {
        [
            RemeshProjectionActionV1::CoupledVelocity,
            RemeshProjectionActionV1::AbsolutePressure,
            RemeshProjectionActionV1::AbsoluteDisplacement,
        ]
        .into_iter()
        .map(|action| projection_for(action, transfer_plan))
        .collect()
    }

    fn normalization() -> RemeshNormalizationWitnessV1 {
        RemeshNormalizationWitnessV1::new(plan().scales(), 4.0).unwrap()
    }

    fn evidence_with(
        normalization: RemeshNormalizationWitnessV1,
        momentum_after: [f64; 2],
    ) -> RemeshTransferEvidenceV1 {
        RemeshTransferEvidenceV1::new(
            normalization,
            [0.0, 0.0],
            momentum_after,
            0.0,
            6.0,
            0.2,
            0.1,
            0.15,
            0.4,
            0.2,
            1.0,
        )
        .unwrap()
    }

    fn evidence() -> RemeshTransferEvidenceV1 {
        evidence_with(normalization(), [4.0, 0.0])
    }

    fn field_wire(
        ordinal: u128,
        role: RemeshFieldRoleV1,
        law: RemeshTransferLawV1,
        chart: RemeshIntegrationChartV1,
        projection: &RemeshProjectionEvidenceEnvelopeV1,
    ) -> WireFieldTransferReceiptV1 {
        WireFieldTransferReceiptV1 {
            field_ulid: Ulid::from(ordinal).to_string(),
            role: WireFieldRoleV1::encode(role),
            law: WireTransferLawV1::encode(law),
            chart: WireIntegrationChartV1::encode(chart),
            source_snapshot_sha256: "22".repeat(32),
            target_snapshot_sha256: "33".repeat(32),
            projection_evidence_sha256: projection.digest().unwrap().to_string(),
            raw_projection_error_l2: 0.0,
        }
    }

    fn receipt_with(
        evidence: RemeshTransferEvidenceV1,
        projections: Vec<RemeshProjectionEvidenceEnvelopeV1>,
    ) -> RemeshTransferReceiptEnvelopeV1 {
        let velocity = projections
            .iter()
            .find(|value| value.action() == RemeshProjectionActionV1::CoupledVelocity)
            .unwrap();
        let pressure = projections
            .iter()
            .find(|value| value.action() == RemeshProjectionActionV1::AbsolutePressure)
            .unwrap();
        let displacement = projections
            .iter()
            .find(|value| value.action() == RemeshProjectionActionV1::AbsoluteDisplacement)
            .unwrap();
        let fields = vec![
            field_wire(
                1,
                RemeshFieldRoleV1::FluidVelocity,
                RemeshTransferLawV1::CoupledVelocityConstrainedL2,
                RemeshIntegrationChartV1::CurrentSpatial,
                velocity,
            ),
            field_wire(
                2,
                RemeshFieldRoleV1::SolidVelocity,
                RemeshTransferLawV1::CoupledVelocityConstrainedL2,
                RemeshIntegrationChartV1::Material,
                velocity,
            ),
            field_wire(
                3,
                RemeshFieldRoleV1::FluidPressure,
                RemeshTransferLawV1::AbsolutePressureL2,
                RemeshIntegrationChartV1::CurrentSpatial,
                pressure,
            ),
            field_wire(
                4,
                RemeshFieldRoleV1::SolidDisplacement,
                RemeshTransferLawV1::AbsoluteDisplacementL2,
                RemeshIntegrationChartV1::Material,
                displacement,
            ),
        ];
        let mut projections = projections;
        projections.sort_by_key(RemeshProjectionEvidenceEnvelopeV1::action);
        RemeshTransferReceiptEnvelopeV1 {
            wire: WireRemeshTransferReceiptV1 {
                schema: TRANSFER_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                source_spatial_state_sha256: "44".repeat(32),
                overlap_sha256: "11".repeat(32),
                target_geometry_state_sha256: "55".repeat(32),
                source_realization_sha256: "66".repeat(32),
                target_realization_sha256: "77".repeat(32),
                fields,
                projections: projections.into_iter().map(|value| value.wire).collect(),
                evidence: WireTransferEvidenceV1::encode(evidence),
                target_quality: WireTargetQualityV1 {
                    minimum_mean_ratio: 0.5,
                    minimum_signed_measure_scale: 0.25,
                },
            },
        }
    }

    #[test]
    fn projection_wire_roundtrip_is_golden_and_self_contained() {
        let value = projection(WireProjectionExecutionV1::SolvedVector2 {
            solves: Box::new([solve(0, plan()), solve(1, plan())]),
        });
        value
            .validate_local(RemeshDecoderLimits::default())
            .unwrap();
        let bytes = value.canonical_json().unwrap();
        let decoded =
            RemeshProjectionEvidenceEnvelopeV1::from_json(&bytes, RemeshDecoderLimits::default())
                .unwrap();
        assert_eq!(decoded, value);
        assert_eq!(decoded.dimensionless_algebraic_replay().observed(), 0.0);
    }

    #[test]
    fn projection_wire_rejects_action_substitution_and_resource_excess() {
        let value = projection(WireProjectionExecutionV1::SolvedVector2 {
            solves: Box::new([solve(0, plan()), solve(1, plan())]),
        });
        let bytes = value.canonical_json().unwrap();
        let substituted = String::from_utf8(bytes.clone())
            .unwrap()
            .replace("absolute-displacement", "coupled-velocity");
        assert!(
            RemeshProjectionEvidenceEnvelopeV1::from_json(
                substituted.as_bytes(),
                RemeshDecoderLimits::default(),
            )
            .is_err()
        );

        let limits = RemeshDecoderLimits {
            max_remesh_projection_solves: 1,
            ..RemeshDecoderLimits::default()
        };
        assert!(RemeshProjectionEvidenceEnvelopeV1::from_json(&bytes, limits).is_err());
    }

    #[test]
    fn displacement_execution_is_closed_to_zero_or_two_solves() {
        let prescribed = projection(WireProjectionExecutionV1::PrescribedExactly);
        prescribed
            .validate_local(RemeshDecoderLimits::default())
            .unwrap();

        let one = projection(WireProjectionExecutionV1::SolvedScalar {
            solve: Box::new(solve(0, plan())),
        });
        assert!(one.validate_local(RemeshDecoderLimits::default()).is_err());

        let mut nonzero_prescribed = prescribed;
        nonzero_prescribed.wire.algebraic_replay = WireBoundedDefectV1 {
            observed_dimensionless: 0.0,
            limit_dimensionless: 1.0,
        };
        assert!(
            nonzero_prescribed
                .validate_local(RemeshDecoderLimits::default())
                .is_err()
        );
    }

    #[test]
    fn raw_physical_evidence_is_recomputed_under_one_normalization() {
        assert!(RemeshNormalizationWitnessV1::new(plan().scales(), 0.0).is_err());
        assert!(RemeshNormalizationWitnessV1::new(plan().scales(), f64::NAN).is_err());

        let evidence = evidence();
        let wire = WireTransferEvidenceV1::encode(evidence);
        assert_eq!(wire.decode().unwrap(), evidence);
        assert_eq!(evidence.momentum_defect().observed(), 0.5);
        assert_eq!(evidence.weak_divergence().observed(), 0.2);
        assert_eq!(evidence.shared_trace().observed(), 0.2);
        assert_eq!(evidence.exterior_trace().observed(), 0.3);
        assert_eq!(evidence.pressure_zeroth_moment().observed(), 0.5);
        assert_eq!(evidence.displacement_trace().observed(), 0.2);
        assert_eq!(evidence.harmonic_replay().observed(), 0.1);

        let mut changed_raw = wire;
        changed_raw.raw_momentum_after[0] = 2.0;
        assert!(changed_raw.decode().is_err());

        let mut changed_density = wire;
        changed_density.normalization.reference_density_kg_per_m3 = 8.0;
        assert!(changed_density.decode().is_err());

        let mut changed_scale = wire;
        changed_scale.normalization.scales.length_m = 4.0;
        assert!(changed_scale.decode().is_err());

        let original_receipt = receipt_with(evidence, projections(plan()));
        let changed_receipt = receipt_with(
            evidence_with(normalization(), [2.0, 0.0]),
            projections(plan()),
        );
        assert_ne!(
            original_receipt.digest().unwrap(),
            changed_receipt.digest().unwrap()
        );
    }

    #[test]
    fn normalization_closure_rejects_every_scale_substitution() {
        let base = plan().scales();
        let evidence = evidence();
        let projections = projections(plan());
        validate_normalization_closure(base, base, evidence.normalization(), &projections).unwrap();

        let alternative = alternative_plan().scales();
        assert!(
            validate_normalization_closure(
                base,
                alternative,
                evidence.normalization(),
                &projections,
            )
            .is_err()
        );
        assert!(
            validate_normalization_closure(
                base,
                base,
                RemeshNormalizationWitnessV1::new(alternative, 4.0).unwrap(),
                &projections,
            )
            .is_err()
        );

        let mut one_changed = projections;
        one_changed[1] = projection_for(
            RemeshProjectionActionV1::AbsolutePressure,
            alternative_plan(),
        );
        assert!(
            validate_normalization_closure(base, base, evidence.normalization(), &one_changed)
                .is_err()
        );
    }

    #[test]
    fn receipt_wire_roundtrips_and_obeys_field_budget() {
        let value = receipt_with(evidence(), projections(plan()));
        value
            .validate_local(RemeshDecoderLimits::default())
            .unwrap();
        let bytes = value.canonical_json().unwrap();
        let decoded =
            RemeshTransferReceiptEnvelopeV1::from_json(&bytes, RemeshDecoderLimits::default())
                .unwrap();
        assert_eq!(decoded, value);
        assert_eq!(decoded.canonical_json().unwrap(), bytes);

        let limits = RemeshDecoderLimits {
            max_remesh_transfer_fields: 3,
            ..RemeshDecoderLimits::default()
        };
        assert!(RemeshTransferReceiptEnvelopeV1::from_json(&bytes, limits).is_err());
    }
}
