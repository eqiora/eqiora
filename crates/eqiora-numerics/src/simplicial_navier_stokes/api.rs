use std::num::NonZeroUsize;

use eqiora_assembly::AssemblyReport;
use eqiora_core::Diagnostic;
use eqiora_realization::Target;
use eqiora_solver::{LinearSolver, PreconditionerPolicy, ReductionPolicy, SolveReport, SolverPlan};

use super::{COMPONENTS, invalid, solve_failed};
use crate::simplicial_elliptic::SimplicialP1Field;
use crate::simplicial_stokes::{
    SimplicialMiniStokesPressureReference2d, SimplicialMiniStokesSolution2d,
    SimplicialMiniVelocityField2d,
};

/// Complete bounded realization controls for one fixed-domain MINI step.
///
/// The type reuses the common [`SolverPlan`] directly. It adds only
/// time-discretization and nonlinear-globalization choices that do not belong
/// to a linear backend.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MiniNavierStokesStepPlan2d {
    density: f64,
    viscosity: f64,
    time_step: f64,
    nonlinear_relative_tolerance: f64,
    nonlinear_absolute_tolerance: f64,
    maximum_newton_iterations: NonZeroUsize,
    maximum_line_search_steps: usize,
    linear_solver: SolverPlan,
    target: Target,
}

impl MiniNavierStokesStepPlan2d {
    /// Validate one host-reference backward-Euler/Newton plan.
    ///
    /// # Errors
    /// Returns `EQ0801` for non-positive physical/time data or nested linear
    /// tolerances that could accept a zero correction above the nonlinear
    /// target, and `EQ0807` when the linear policy is not serial
    /// identity-preconditioned fast-reduction sparse LU. Production
    /// preconditioning is intentionally a later claim.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        density: f64,
        viscosity: f64,
        time_step: f64,
        nonlinear_relative_tolerance: f64,
        nonlinear_absolute_tolerance: f64,
        maximum_newton_iterations: NonZeroUsize,
        maximum_line_search_steps: usize,
        linear_solver: SolverPlan,
        target: Target,
    ) -> Result<Self, Diagnostic> {
        if !density.is_finite()
            || density <= 0.0
            || !viscosity.is_finite()
            || viscosity <= 0.0
            || !time_step.is_finite()
            || time_step <= 0.0
        {
            return Err(invalid(
                "MINI Navier--Stokes density, viscosity, and time step must be finite and positive",
            ));
        }
        if !nonlinear_relative_tolerance.is_finite()
            || !nonlinear_absolute_tolerance.is_finite()
            || !(0.0..1.0).contains(&nonlinear_relative_tolerance)
            || nonlinear_absolute_tolerance < 0.0
            || (nonlinear_relative_tolerance == 0.0 && nonlinear_absolute_tolerance == 0.0)
        {
            return Err(invalid(
                "nonlinear tolerances must be finite and non-negative, relative tolerance must be below one, and both tolerances cannot be zero",
            ));
        }
        if maximum_line_search_steps > 64 {
            return Err(invalid(
                "bounded reference line search admits at most 64 halvings",
            ));
        }
        if linear_solver.relative_tolerance() >= 1.0
            || linear_solver.absolute_tolerance() > nonlinear_absolute_tolerance
        {
            return Err(invalid(
                "MINI Navier--Stokes linear tolerances must require a nonzero correction whenever the nonlinear residual exceeds its target",
            ));
        }
        if linear_solver.algorithm() != LinearSolver::SparseLu
            || linear_solver.preconditioner() != PreconditionerPolicy::Identity
            || linear_solver.reduction() != ReductionPolicy::Fast
            || target
                != (Target::HostCpu {
                    threads: NonZeroUsize::MIN,
                })
        {
            return Err(invalid(
                "bounded MINI Navier--Stokes requires serial-host identity-preconditioned fast-reduction sparse LU",
            ));
        }
        Ok(Self {
            density,
            viscosity,
            time_step,
            nonlinear_relative_tolerance,
            nonlinear_absolute_tolerance,
            maximum_newton_iterations,
            maximum_line_search_steps,
            linear_solver,
            target,
        })
    }

    /// Constant mass density.
    #[must_use]
    pub const fn density(self) -> f64 {
        self.density
    }

    /// Constant dynamic viscosity.
    #[must_use]
    pub const fn viscosity(self) -> f64 {
        self.viscosity
    }

    /// Fixed backward-Euler duration.
    #[must_use]
    pub const fn time_step(self) -> f64 {
        self.time_step
    }

    /// Relative nonlinear residual tolerance.
    #[must_use]
    pub const fn nonlinear_relative_tolerance(self) -> f64 {
        self.nonlinear_relative_tolerance
    }

    /// Absolute nonlinear residual tolerance.
    #[must_use]
    pub const fn nonlinear_absolute_tolerance(self) -> f64 {
        self.nonlinear_absolute_tolerance
    }

    /// Newton iteration bound.
    #[must_use]
    pub const fn maximum_newton_iterations(self) -> NonZeroUsize {
        self.maximum_newton_iterations
    }

    /// Backtracking-halving bound.
    #[must_use]
    pub const fn maximum_line_search_steps(self) -> usize {
        self.maximum_line_search_steps
    }

    /// Common linear solver policy used at every accepted Newton point.
    #[must_use]
    pub const fn linear_solver(self) -> SolverPlan {
        self.linear_solver
    }

    /// Exact one-worker host placement of this reference slice.
    #[must_use]
    pub const fn target(self) -> Target {
        self.target
    }

    pub(super) fn nonlinear_target(self, initial_norm: f64) -> Result<f64, Diagnostic> {
        let target = self
            .nonlinear_absolute_tolerance
            .max(self.nonlinear_relative_tolerance * initial_norm);
        if target.is_finite() {
            Ok(target)
        } else {
            Err(solve_failed("nonlinear residual target overflowed"))
        }
    }
}

/// One method-native fixed-mesh mixed state.
///
/// Velocity is the differential state. Pressure and the optional gauge
/// multiplier are algebraic coordinates retained for restart/warm-start
/// evidence; pressure is never treated as a time-integrated state.
#[derive(Debug, Clone, PartialEq)]
pub struct SimplicialMiniNavierStokesState2d {
    time: f64,
    velocity: SimplicialMiniVelocityField2d,
    pressure: SimplicialP1Field,
    pressure_reference: SimplicialMiniStokesPressureReference2d,
}

impl SimplicialMiniNavierStokesState2d {
    /// Construct one shaped fixed-mesh initial candidate.
    ///
    /// Values in this low-level reference type are method-native coordinates,
    /// never coherent-SI output from the resolved API. The selected boundary
    /// closure reconstructs its own Newton coordinates, then independently
    /// checks essential values, weak continuity, pressure mean, and gauge
    /// consistency before the first Newton iteration.
    ///
    /// # Errors
    /// Returns `EQ0801` for non-finite/negative time or mismatched meshes.
    pub fn new(
        time: f64,
        velocity: SimplicialMiniVelocityField2d,
        pressure: SimplicialP1Field,
        pressure_reference: SimplicialMiniStokesPressureReference2d,
    ) -> Result<Self, Diagnostic> {
        if !time.is_finite() || time < 0.0 {
            return Err(invalid(
                "MINI Navier--Stokes state time must be finite and non-negative",
            ));
        }
        if velocity.mesh() != pressure.mesh() {
            return Err(invalid(
                "MINI Navier--Stokes velocity and pressure must share one exact mesh",
            ));
        }
        if pressure_reference
            .gauge_multiplier()
            .is_some_and(|value| !value.is_finite())
        {
            return Err(invalid(
                "MINI Navier--Stokes pressure-reference value must be finite",
            ));
        }
        Ok(Self {
            time,
            velocity,
            pressure,
            pressure_reference,
        })
    }

    /// Admit one already accepted steady MINI solution as a shaped initial
    /// state on the same exact mesh and pressure-closure policy.
    ///
    /// # Errors
    /// Returns `EQ0801` for a non-finite/negative model time.
    pub fn from_stokes_solution(
        time: f64,
        solution: &SimplicialMiniStokesSolution2d,
    ) -> Result<Self, Diagnostic> {
        Self::new(
            time,
            solution.velocity().clone(),
            solution.pressure().clone(),
            solution.pressure_reference(),
        )
    }

    pub(super) fn accepted(
        time: f64,
        velocity: SimplicialMiniVelocityField2d,
        pressure: SimplicialP1Field,
        pressure_reference: SimplicialMiniStokesPressureReference2d,
    ) -> Self {
        Self {
            time,
            velocity,
            pressure,
            pressure_reference,
        }
    }

    /// Strictly accepted coherent model time.
    #[must_use]
    pub const fn time(&self) -> f64 {
        self.time
    }

    /// Differential MINI velocity state.
    #[must_use]
    pub const fn velocity(&self) -> &SimplicialMiniVelocityField2d {
        &self.velocity
    }

    /// Algebraic P1 pressure coordinate.
    #[must_use]
    pub const fn pressure(&self) -> &SimplicialP1Field {
        &self.pressure
    }

    /// Exact pressure-nullspace policy inherited by every step.
    #[must_use]
    pub const fn pressure_reference(&self) -> SimplicialMiniStokesPressureReference2d {
        self.pressure_reference
    }
}

/// Independently reaccepted evidence for one nonlinear implicit step.
#[derive(Debug, Clone, PartialEq)]
pub struct SimplicialMiniNavierStokesStepEvidence2d {
    nonlinear_iterations: usize,
    initial_residual_norm: f64,
    final_residual_norm: f64,
    momentum_residual_norm: f64,
    residual_target: f64,
    continuity_residual_norm: f64,
    pressure_integral: f64,
    convective_residual_norm: f64,
    convective_power: f64,
    conservative_advection_defect_norm: f64,
    named_boundary_reactions: Vec<(String, [f64; COMPONENTS])>,
    assembly_report: AssemblyReport,
    linear_solves: Vec<SolveReport>,
}

impl SimplicialMiniNavierStokesStepEvidence2d {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        nonlinear_iterations: usize,
        initial_residual_norm: f64,
        final_residual_norm: f64,
        momentum_residual_norm: f64,
        residual_target: f64,
        continuity_residual_norm: f64,
        pressure_integral: f64,
        convective_residual_norm: f64,
        convective_power: f64,
        conservative_advection_defect_norm: f64,
        named_boundary_reactions: Vec<(String, [f64; COMPONENTS])>,
        assembly_report: AssemblyReport,
        linear_solves: Vec<SolveReport>,
    ) -> Self {
        Self {
            nonlinear_iterations,
            initial_residual_norm,
            final_residual_norm,
            momentum_residual_norm,
            residual_target,
            continuity_residual_norm,
            pressure_integral,
            convective_residual_norm,
            convective_power,
            conservative_advection_defect_norm,
            named_boundary_reactions,
            assembly_report,
            linear_solves,
        }
    }

    /// Accepted Newton updates.
    #[must_use]
    pub const fn nonlinear_iterations(&self) -> usize {
        self.nonlinear_iterations
    }

    /// Residual norm at the previous-state warm start.
    #[must_use]
    pub const fn initial_residual_norm(&self) -> f64 {
        self.initial_residual_norm
    }

    /// Independently reapplied final residual norm.
    #[must_use]
    pub const fn final_residual_norm(&self) -> f64 {
        self.final_residual_norm
    }

    /// Directly reassembled nonlinear momentum-block residual norm.
    #[must_use]
    pub const fn momentum_residual_norm(&self) -> f64 {
        self.momentum_residual_norm
    }

    /// Frozen nonlinear acceptance threshold.
    #[must_use]
    pub const fn residual_target(&self) -> f64 {
        self.residual_target
    }

    /// Weak incompressibility norm with gauge contribution removed.
    #[must_use]
    pub const fn continuity_residual_norm(&self) -> f64 {
        self.continuity_residual_norm
    }

    /// Integrated algebraic pressure in the selected closure.
    #[must_use]
    pub const fn pressure_integral(&self) -> f64 {
        self.pressure_integral
    }

    /// Norm of the independently integrated convective action.
    #[must_use]
    pub const fn convective_residual_norm(&self) -> f64 {
        self.convective_residual_norm
    }

    /// Discrete self-work of the skew convective action.
    #[must_use]
    pub const fn convective_power(&self) -> f64 {
        self.convective_power
    }

    /// Norm of the explicit skew-minus-conservative weak-form defect.
    ///
    /// The vector whose norm is returned is one half of the parent-outward
    /// boundary momentum flux minus the retained
    /// `rho/2 * div(u_h) * u_h` consistency term. The boundary contribution
    /// vanishes for homogeneous trace. This is evidence for a Realization
    /// transformation, not a claim that the two discrete forms are identical.
    #[must_use]
    pub const fn conservative_advection_defect_norm(&self) -> f64 {
        self.conservative_advection_defect_norm
    }

    /// Reaction exerted on the fluid domain by each authenticated constrained surface.
    #[must_use]
    pub fn named_boundary_reaction(&self, name: &str) -> Option<[f64; COMPONENTS]> {
        self.named_boundary_reactions
            .iter()
            .find_map(|(candidate, reaction)| (candidate == name).then_some(*reaction))
    }

    pub(crate) fn named_boundary_reactions(&self) -> &[(String, [f64; COMPONENTS])] {
        &self.named_boundary_reactions
    }

    /// Accepted assembly placement and packet shape for the final point.
    #[must_use]
    pub const fn assembly_report(&self) -> &AssemblyReport {
        &self.assembly_report
    }

    /// Every common-contract Krylov report used by this Newton solve.
    #[must_use]
    pub fn linear_solves(&self) -> &[SolveReport] {
        &self.linear_solves
    }
}

/// Initial state followed by a strictly increasing accepted step sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct SimplicialMiniNavierStokesTrajectory2d {
    states: Vec<SimplicialMiniNavierStokesState2d>,
    steps: Vec<SimplicialMiniNavierStokesStepEvidence2d>,
}

impl SimplicialMiniNavierStokesTrajectory2d {
    pub(super) fn new(initial: SimplicialMiniNavierStokesState2d) -> Self {
        Self {
            states: vec![initial],
            steps: Vec::new(),
        }
    }

    pub(super) fn push(
        &mut self,
        state: SimplicialMiniNavierStokesState2d,
        evidence: SimplicialMiniNavierStokesStepEvidence2d,
    ) -> Result<(), Diagnostic> {
        let previous = self
            .states
            .last()
            .expect("trajectory owns its initial state");
        if state.time <= previous.time {
            return Err(invalid(
                "accepted MINI Navier--Stokes times must increase strictly",
            ));
        }
        self.states.push(state);
        self.steps.push(evidence);
        Ok(())
    }

    /// Initial plus accepted states in strict model-time order.
    #[must_use]
    pub fn states(&self) -> &[SimplicialMiniNavierStokesState2d] {
        &self.states
    }

    /// One evidence record per transition between adjacent states.
    #[must_use]
    pub fn steps(&self) -> &[SimplicialMiniNavierStokesStepEvidence2d] {
        &self.steps
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use eqiora_realization::Target;
    use eqiora_solver::{LinearSolver, ReductionPolicy, SolverPlan};

    use super::MiniNavierStokesStepPlan2d;

    fn plan(
        linear_relative: f64,
        linear_absolute: f64,
    ) -> Result<MiniNavierStokesStepPlan2d, eqiora_core::Diagnostic> {
        MiniNavierStokesStepPlan2d::new(
            1.0,
            1.0e-3,
            1.0e-2,
            1.0e-9,
            1.0e-11,
            NonZeroUsize::new(8).unwrap(),
            12,
            SolverPlan::new(
                LinearSolver::SparseLu,
                linear_relative,
                linear_absolute,
                NonZeroUsize::new(100).unwrap(),
            )
            .unwrap()
            .with_reduction(ReductionPolicy::Fast),
            Target::HostCpu {
                threads: NonZeroUsize::MIN,
            },
        )
    }

    #[test]
    fn nested_linear_tolerances_cannot_accept_a_zero_newton_correction_above_target() {
        assert!(plan(1.0e-6, 1.0e-11).is_ok());
        assert!(plan(1.0e-6, 1.0e-9).is_err());
        assert!(plan(1.0, 0.0).is_err());
    }
}
