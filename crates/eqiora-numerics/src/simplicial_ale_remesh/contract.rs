use eqiora_core::Diagnostic;
use eqiora_meshing::{FixedTopologyGeometryState2d, SimplicialRevisionOverlap2d};
use eqiora_solver::SolveReport;

use crate::canonical_fsi::AleFsiInitialPhysicalState;
use crate::simplicial_fsi::FixedReferenceFsiScale;

const COMPONENTS: usize = 2;

/// Independently replayed numerical evidence for one zero-time FSI remesh.
///
/// The two overlap values retain the exact coordinate-chart distinction:
/// solid displacement and solid velocity use the material reference chart,
/// while fluid velocity and pressure use the derived current spatial chart.
#[derive(Debug, Clone, PartialEq)]
pub struct AleFsiRemeshProjectionEvidence2d {
    solid_reference_overlap: SimplicialRevisionOverlap2d,
    fluid_current_overlap: SimplicialRevisionOverlap2d,
    target_geometry: FixedTopologyGeometryState2d,
    displacement_solve_reports: Vec<SolveReport>,
    displacement_right_hand_side_norms: Vec<f64>,
    velocity_solve_report: SolveReport,
    velocity_right_hand_side_norm: f64,
    pressure_solve_report: SolveReport,
    pressure_right_hand_side_norm: f64,
    scale: FixedReferenceFsiScale<2>,
    reference_density: f64,
    characteristic_mass: f64,
    independent_velocity_constraint_count: usize,
    displacement_l2_error: f64,
    fluid_current_density_weighted_velocity_l2_error: f64,
    solid_material_density_weighted_velocity_l2_error: f64,
    pressure_l2_error: f64,
    displacement_projection_residual_norm: f64,
    velocity_projection_residual_norm: f64,
    pressure_projection_residual_norm: f64,
    displacement_projection_acceptance_limit: f64,
    velocity_projection_acceptance_limit: f64,
    pressure_projection_acceptance_limit: f64,
    maximum_displacement_trace_defect: f64,
    maximum_shared_velocity_trace_defect: f64,
    maximum_exterior_velocity_trace_defect: f64,
    weak_incompressibility_residual_norm: f64,
    source_total_momentum: [f64; COMPONENTS],
    target_total_momentum: [f64; COMPONENTS],
    pressure_source_moment: f64,
    pressure_target_moment: f64,
    dimensionless_displacement_trace_defect: f64,
    dimensionless_shared_velocity_trace_defect: f64,
    dimensionless_exterior_velocity_trace_defect: f64,
    dimensionless_weak_incompressibility_defect: f64,
    dimensionless_momentum_defect: f64,
    dimensionless_pressure_zeroth_moment_defect: f64,
    dimensionless_physical_acceptance_limit: f64,
}

impl AleFsiRemeshProjectionEvidence2d {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        solid_reference_overlap: SimplicialRevisionOverlap2d,
        fluid_current_overlap: SimplicialRevisionOverlap2d,
        target_geometry: FixedTopologyGeometryState2d,
        displacement_solve_reports: Vec<SolveReport>,
        displacement_right_hand_side_norms: Vec<f64>,
        velocity_solve_report: SolveReport,
        velocity_right_hand_side_norm: f64,
        pressure_solve_report: SolveReport,
        pressure_right_hand_side_norm: f64,
        scale: FixedReferenceFsiScale<2>,
        reference_density: f64,
        independent_velocity_constraint_count: usize,
        displacement_l2_error: f64,
        fluid_current_density_weighted_velocity_l2_error: f64,
        solid_material_density_weighted_velocity_l2_error: f64,
        pressure_l2_error: f64,
        displacement_projection_residual_norm: f64,
        velocity_projection_residual_norm: f64,
        pressure_projection_residual_norm: f64,
        maximum_displacement_trace_defect: f64,
        maximum_shared_velocity_trace_defect: f64,
        maximum_exterior_velocity_trace_defect: f64,
        weak_incompressibility_residual_norm: f64,
        source_total_momentum: [f64; COMPONENTS],
        target_total_momentum: [f64; COMPONENTS],
        pressure_source_moment: f64,
        pressure_target_moment: f64,
    ) -> Result<Self, Diagnostic> {
        let nonnegative = [
            displacement_l2_error,
            fluid_current_density_weighted_velocity_l2_error,
            solid_material_density_weighted_velocity_l2_error,
            pressure_l2_error,
            displacement_projection_residual_norm,
            velocity_projection_residual_norm,
            pressure_projection_residual_norm,
            maximum_displacement_trace_defect,
            maximum_shared_velocity_trace_defect,
            maximum_exterior_velocity_trace_defect,
            weak_incompressibility_residual_norm,
            velocity_right_hand_side_norm,
            pressure_right_hand_side_norm,
        ];
        let displacement_solve_count = displacement_solve_reports.len();
        if independent_velocity_constraint_count == 0
            || !matches!(displacement_solve_count, 0 | COMPONENTS)
            || displacement_right_hand_side_norms.len() != displacement_solve_count
            || displacement_right_hand_side_norms
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            || nonnegative
                .iter()
                .chain(source_total_momentum.iter())
                .chain(target_total_momentum.iter())
                .chain([pressure_source_moment, pressure_target_moment].iter())
                .any(|value| !value.is_finite())
            || nonnegative.iter().any(|value| *value < 0.0)
        {
            return Err(super::invalid(
                "ALE FSI remesh evidence must contain finite non-negative errors, complete solve reports, and at least one independent velocity constraint",
            ));
        }

        let plan = velocity_solve_report.solver_plan();
        if pressure_solve_report.solver_plan() != plan
            || displacement_solve_reports
                .iter()
                .any(|report| report.solver_plan() != plan)
        {
            return Err(super::invalid(
                "ALE FSI remesh evidence must use one common solver plan",
            ));
        }
        let displacement_tolerance = displacement_solve_reports
            .iter()
            .zip(&displacement_right_hand_side_norms)
            .map(|(report, &rhs)| algebraic_replay_tolerance(report, rhs))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        let velocity_tolerance =
            algebraic_replay_tolerance(&velocity_solve_report, velocity_right_hand_side_norm)?;
        let pressure_tolerance =
            algebraic_replay_tolerance(&pressure_solve_report, pressure_right_hand_side_norm)?;
        if displacement_projection_residual_norm > displacement_tolerance
            || velocity_projection_residual_norm > velocity_tolerance
            || pressure_projection_residual_norm > pressure_tolerance
        {
            return Err(super::invalid(
                "ALE FSI remesh dimensionless algebraic replay exceeded its same-system solver target",
            ));
        }

        if !reference_density.is_finite() || reference_density <= 0.0 {
            return Err(super::invalid(
                "remesh reference density must be finite and positive",
            ));
        }
        let characteristic_mass = checked_product(
            reference_density,
            scale.length() * scale.length(),
            "characteristic mass rho* L^2",
        )?;
        let characteristic_momentum = checked_product(
            characteristic_mass,
            scale.velocity(),
            "characteristic momentum rho* U L^2",
        )?;
        let characteristic_divergence = checked_product(
            scale.velocity(),
            scale.length(),
            "characteristic weak divergence U L",
        )?;
        let characteristic_pressure_moment = checked_product(
            scale.pressure(),
            scale.length() * scale.length(),
            "characteristic pressure moment P L^2",
        )?;
        let dimensionless_displacement_trace_defect =
            maximum_displacement_trace_defect / scale.length();
        let dimensionless_shared_velocity_trace_defect =
            maximum_shared_velocity_trace_defect / scale.velocity();
        let dimensionless_exterior_velocity_trace_defect =
            maximum_exterior_velocity_trace_defect / scale.velocity();
        let dimensionless_weak_incompressibility_defect =
            weak_incompressibility_residual_norm / characteristic_divergence;
        let dimensionless_momentum_defect = (source_total_momentum[0] - target_total_momentum[0])
            .hypot(source_total_momentum[1] - target_total_momentum[1])
            / characteristic_momentum;
        let dimensionless_pressure_zeroth_moment_defect =
            (pressure_source_moment - pressure_target_moment).abs()
                / characteristic_pressure_moment;
        let dimensionless_physical_acceptance_limit =
            dimensionless_replay_tolerance(plan.residual_target(1.0)?);
        let dimensionless_physical_defects = [
            dimensionless_displacement_trace_defect,
            dimensionless_shared_velocity_trace_defect,
            dimensionless_exterior_velocity_trace_defect,
            dimensionless_weak_incompressibility_defect,
            dimensionless_momentum_defect,
            dimensionless_pressure_zeroth_moment_defect,
        ];
        if dimensionless_physical_defects
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
            || dimensionless_physical_defects
                .iter()
                .any(|value| *value > dimensionless_physical_acceptance_limit)
        {
            return Err(super::invalid(
                "ALE FSI remesh dimensionless physical obligation replay exceeded the common solver-plan criterion",
            ));
        }

        Ok(Self {
            solid_reference_overlap,
            fluid_current_overlap,
            target_geometry,
            displacement_solve_reports,
            displacement_right_hand_side_norms,
            velocity_solve_report,
            velocity_right_hand_side_norm,
            pressure_solve_report,
            pressure_right_hand_side_norm,
            scale,
            reference_density,
            characteristic_mass,
            independent_velocity_constraint_count,
            displacement_l2_error,
            fluid_current_density_weighted_velocity_l2_error,
            solid_material_density_weighted_velocity_l2_error,
            pressure_l2_error,
            displacement_projection_residual_norm,
            velocity_projection_residual_norm,
            pressure_projection_residual_norm,
            displacement_projection_acceptance_limit: displacement_tolerance,
            velocity_projection_acceptance_limit: velocity_tolerance,
            pressure_projection_acceptance_limit: pressure_tolerance,
            maximum_displacement_trace_defect,
            maximum_shared_velocity_trace_defect,
            maximum_exterior_velocity_trace_defect,
            weak_incompressibility_residual_norm,
            source_total_momentum,
            target_total_momentum,
            pressure_source_moment,
            pressure_target_moment,
            dimensionless_displacement_trace_defect,
            dimensionless_shared_velocity_trace_defect,
            dimensionless_exterior_velocity_trace_defect,
            dimensionless_weak_incompressibility_defect,
            dimensionless_momentum_defect,
            dimensionless_pressure_zeroth_moment_defect,
            dimensionless_physical_acceptance_limit,
        })
    }

    /// Exact material-chart solid overlap used by the projection.
    #[must_use]
    pub const fn solid_reference_overlap(&self) -> &SimplicialRevisionOverlap2d {
        &self.solid_reference_overlap
    }

    /// Exact current-spatial fluid overlap derived after target mesh motion.
    #[must_use]
    pub const fn fluid_current_overlap(&self) -> &SimplicialRevisionOverlap2d {
        &self.fluid_current_overlap
    }

    /// Target current geometry derived solely from transferred displacement.
    #[must_use]
    pub const fn target_geometry(&self) -> &FixedTopologyGeometryState2d {
        &self.target_geometry
    }

    /// Common-solver reports for the free displacement components.
    #[must_use]
    pub fn displacement_solve_reports(&self) -> &[SolveReport] {
        &self.displacement_solve_reports
    }

    /// Dimensionless Euclidean RHS norms for displacement reports, component order.
    #[must_use]
    pub fn dimensionless_displacement_right_hand_side_norms(&self) -> &[f64] {
        &self.displacement_right_hand_side_norms
    }

    /// Common-solver report for the coupled constrained velocity projection.
    #[must_use]
    pub const fn velocity_solve_report(&self) -> &SolveReport {
        &self.velocity_solve_report
    }

    /// Dimensionless Euclidean norm of the constrained-velocity KKT RHS.
    #[must_use]
    pub const fn dimensionless_velocity_right_hand_side_norm(&self) -> f64 {
        self.velocity_right_hand_side_norm
    }

    /// Common-solver report for absolute pressure projection.
    #[must_use]
    pub const fn pressure_solve_report(&self) -> &SolveReport {
        &self.pressure_solve_report
    }

    /// Dimensionless Euclidean norm of the absolute-pressure projection RHS.
    #[must_use]
    pub const fn dimensionless_pressure_right_hand_side_norm(&self) -> f64 {
        self.pressure_right_hand_side_norm
    }

    /// Characteristic `L`, `U`, and `P` used by the dimensionless algebra.
    #[must_use]
    pub const fn scale(&self) -> FixedReferenceFsiScale<2> {
        self.scale
    }

    /// Deterministic reference density `rho* = max(rho_fluid, rho_solid)`.
    #[must_use]
    pub const fn reference_density(&self) -> f64 {
        self.reference_density
    }

    /// Intrinsic-2D characteristic mass `rho* L^2` under unit thickness.
    #[must_use]
    pub const fn characteristic_mass(&self) -> f64 {
        self.characteristic_mass
    }

    /// Rank of the canonical independent velocity-constraint set.
    #[must_use]
    pub const fn independent_velocity_constraint_count(&self) -> usize {
        self.independent_velocity_constraint_count
    }

    /// Material-chart solid-displacement L2 transfer error.
    #[must_use]
    pub const fn displacement_l2_error(&self) -> f64 {
        self.displacement_l2_error
    }

    /// Density-weighted coupled velocity L2 transfer error.
    #[must_use]
    pub fn density_weighted_velocity_l2_error(&self) -> f64 {
        self.fluid_current_density_weighted_velocity_l2_error
            .hypot(self.solid_material_density_weighted_velocity_l2_error)
    }

    /// Fluid-current-chart density-weighted velocity L2 transfer error.
    #[must_use]
    pub const fn fluid_current_density_weighted_velocity_l2_error(&self) -> f64 {
        self.fluid_current_density_weighted_velocity_l2_error
    }

    /// Solid-material-chart density-weighted velocity L2 transfer error.
    #[must_use]
    pub const fn solid_material_density_weighted_velocity_l2_error(&self) -> f64 {
        self.solid_material_density_weighted_velocity_l2_error
    }

    /// Absolute current-fluid pressure L2 transfer error.
    #[must_use]
    pub const fn pressure_l2_error(&self) -> f64 {
        self.pressure_l2_error
    }

    /// Dimensionless independently replayed free-row displacement residual.
    #[must_use]
    pub const fn dimensionless_displacement_projection_residual_norm(&self) -> f64 {
        self.displacement_projection_residual_norm
    }

    /// Dimensionless independently replayed coupled-velocity KKT residual.
    #[must_use]
    pub const fn dimensionless_velocity_projection_residual_norm(&self) -> f64 {
        self.velocity_projection_residual_norm
    }

    /// Dimensionless independently replayed absolute-pressure residual.
    #[must_use]
    pub const fn dimensionless_pressure_projection_residual_norm(&self) -> f64 {
        self.pressure_projection_residual_norm
    }

    /// Adopted dimensionless aggregate acceptance limit for displacement.
    ///
    /// The residual combines both free component actions with the Euclidean
    /// norm, so this is the matching aggregate of their common-solver limits.
    #[must_use]
    pub const fn dimensionless_displacement_projection_acceptance_limit(&self) -> f64 {
        self.displacement_projection_acceptance_limit
    }

    /// Adopted dimensionless acceptance limit for the coupled velocity action.
    #[must_use]
    pub const fn dimensionless_velocity_projection_acceptance_limit(&self) -> f64 {
        self.velocity_projection_acceptance_limit
    }

    /// Adopted dimensionless acceptance limit for the pressure action.
    #[must_use]
    pub const fn dimensionless_pressure_projection_acceptance_limit(&self) -> f64 {
        self.pressure_projection_acceptance_limit
    }

    /// Raw coherent-SI material-boundary displacement trace defect.
    #[must_use]
    pub const fn raw_maximum_displacement_trace_defect(&self) -> f64 {
        self.maximum_displacement_trace_defect
    }

    /// Raw coherent-SI retained shared-interface velocity trace defect.
    #[must_use]
    pub const fn raw_maximum_shared_velocity_trace_defect(&self) -> f64 {
        self.maximum_shared_velocity_trace_defect
    }

    /// Raw coherent-SI homogeneous physical-exterior velocity trace defect.
    #[must_use]
    pub const fn raw_maximum_exterior_velocity_trace_defect(&self) -> f64 {
        self.maximum_exterior_velocity_trace_defect
    }

    /// Raw coherent-SI maximum across the two velocity trace obligations.
    #[must_use]
    pub const fn raw_maximum_velocity_trace_defect(&self) -> f64 {
        if self.maximum_shared_velocity_trace_defect > self.maximum_exterior_velocity_trace_defect {
            self.maximum_shared_velocity_trace_defect
        } else {
            self.maximum_exterior_velocity_trace_defect
        }
    }

    /// Raw coherent-SI target weak-divergence norm.
    #[must_use]
    pub const fn raw_weak_incompressibility_residual_norm(&self) -> f64 {
        self.weak_incompressibility_residual_norm
    }

    /// Raw coherent-SI source density-weighted total momentum.
    #[must_use]
    pub const fn raw_source_total_momentum(&self) -> [f64; COMPONENTS] {
        self.source_total_momentum
    }

    /// Raw coherent-SI target density-weighted total momentum.
    #[must_use]
    pub const fn raw_target_total_momentum(&self) -> [f64; COMPONENTS] {
        self.target_total_momentum
    }

    /// Raw coherent-SI source absolute-pressure zeroth moment.
    #[must_use]
    pub const fn raw_pressure_source_moment(&self) -> f64 {
        self.pressure_source_moment
    }

    /// Raw coherent-SI target absolute-pressure zeroth moment.
    #[must_use]
    pub const fn raw_pressure_target_moment(&self) -> f64 {
        self.pressure_target_moment
    }

    /// Dimensionless material displacement-trace defect, normalized by `L`.
    #[must_use]
    pub const fn dimensionless_displacement_trace_defect(&self) -> f64 {
        self.dimensionless_displacement_trace_defect
    }

    /// Dimensionless shared velocity-trace defect, normalized by `U`.
    #[must_use]
    pub const fn dimensionless_shared_velocity_trace_defect(&self) -> f64 {
        self.dimensionless_shared_velocity_trace_defect
    }

    /// Dimensionless exterior velocity-trace defect, normalized by `U`.
    #[must_use]
    pub const fn dimensionless_exterior_velocity_trace_defect(&self) -> f64 {
        self.dimensionless_exterior_velocity_trace_defect
    }

    /// Dimensionless weak-incompressibility defect, normalized by `U L`.
    #[must_use]
    pub const fn dimensionless_weak_incompressibility_defect(&self) -> f64 {
        self.dimensionless_weak_incompressibility_defect
    }

    /// Dimensionless momentum defect, normalized by `rho* U L^2`.
    #[must_use]
    pub const fn dimensionless_momentum_defect(&self) -> f64 {
        self.dimensionless_momentum_defect
    }

    /// Dimensionless pressure-moment defect, normalized by `P L^2`.
    #[must_use]
    pub const fn dimensionless_pressure_zeroth_moment_defect(&self) -> f64 {
        self.dimensionless_pressure_zeroth_moment_defect
    }

    /// Common dimensionless physical-obligation acceptance limit.
    #[must_use]
    pub const fn dimensionless_physical_acceptance_limit(&self) -> f64 {
        self.dimensionless_physical_acceptance_limit
    }
}

/// Complete accepted physical coefficients after one zero-time remesh.
///
/// No caller can obtain an [`AleFsiInitialPhysicalState<2>`] from this path
/// until overlap, projection, constraint, geometry, and independent replay
/// evidence have all been admitted.
#[derive(Debug, Clone, PartialEq)]
pub struct AcceptedAleFsiRemeshProjection2d {
    time: f64,
    physical: crate::simplicial_fsi::FixedReferenceFsiState<2>,
    evidence: AleFsiRemeshProjectionEvidence2d,
}

impl AcceptedAleFsiRemeshProjection2d {
    pub(super) fn new(
        time: f64,
        physical: crate::simplicial_fsi::FixedReferenceFsiState<2>,
        evidence: AleFsiRemeshProjectionEvidence2d,
    ) -> Self {
        Self {
            time,
            physical,
            evidence,
        }
    }
    /// Unchanged model time; remeshing has zero duration.
    pub const fn time(&self) -> f64 {
        self.time
    }
    /// Complete exact physical Field inventory on the accepted target mesh.
    pub const fn physical_state(&self) -> &crate::simplicial_fsi::FixedReferenceFsiState<2> {
        &self.physical
    }

    /// Accepted numerical evidence and the two exact overlap outcomes.
    #[must_use]
    pub const fn evidence(&self) -> &AleFsiRemeshProjectionEvidence2d {
        &self.evidence
    }

    /// Construct the only physical input admitted by the unchanged target ALE
    /// finalizer after this transfer has been accepted.
    ///
    /// # Errors
    /// Preserves the finalizer input's finite-value validation.
    pub fn initial_physical_state(&self) -> Result<AleFsiInitialPhysicalState<2>, Diagnostic> {
        AleFsiInitialPhysicalState::<2>::new(self.time, self.physical.clone())
    }
}

fn algebraic_replay_tolerance(report: &SolveReport, rhs_norm: f64) -> Result<f64, Diagnostic> {
    let expected_target = report.solver_plan().residual_target(rhs_norm)?;
    if report.residual_target().to_bits() != expected_target.to_bits() {
        return Err(super::invalid(
            "ALE FSI remesh solve report target does not match its dimensionless RHS norm",
        ));
    }
    Ok(8.0 * expected_target + 65_536.0 * f64::EPSILON * (1.0 + rhs_norm))
}

fn dimensionless_replay_tolerance(solver_target: f64) -> f64 {
    solver_target + 262_144.0 * f64::EPSILON
}

fn checked_product(left: f64, right: f64, name: &'static str) -> Result<f64, Diagnostic> {
    let value = left * right;
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(super::invalid(format!(
            "ALE FSI remesh {name} must be finite and strictly positive",
        )))
    }
}
