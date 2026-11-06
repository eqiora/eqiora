use std::num::{NonZeroU16, NonZeroUsize};

use eqiora_core::Diagnostic;
use eqiora_solver::{LinearSolver, SolverPlan};

use crate::{
    Discretization, DiscretizationMethod, ExecutionSchedule, MeshPolicy, QuadraturePolicy, Space,
    Target, execution::validate_target_schedule,
};

/// Complete pure payload for one Realization Graph selection.
#[derive(Debug, Clone, PartialEq)]
pub struct RealizationPlan {
    space: Space,
    discretization: Discretization,
    solver: SolverPlan,
    target: Target,
    schedule: ExecutionSchedule,
}

impl RealizationPlan {
    /// Construct and cross-validate independently owned realization components.
    ///
    /// # Errors
    /// Returns `EQ0807` when a method, space, quadrature, or deployment choice
    /// contradicts another component.
    pub fn new(
        space: Space,
        discretization: Discretization,
        solver: SolverPlan,
        target: Target,
        schedule: ExecutionSchedule,
    ) -> Result<Self, Diagnostic> {
        let plan = Self {
            space,
            discretization,
            solver,
            target,
            schedule,
        };
        plan.validate()?;
        Ok(plan)
    }

    /// Discrete-space choice.
    #[must_use]
    pub const fn space(&self) -> Space {
        self.space
    }

    /// Method, mesh, and quadrature choice.
    #[must_use]
    pub const fn discretization(&self) -> Discretization {
        self.discretization
    }

    /// Solver choice.
    #[must_use]
    pub const fn solver(&self) -> &SolverPlan {
        &self.solver
    }

    /// Deployment target.
    #[must_use]
    pub const fn target(&self) -> Target {
        self.target
    }

    /// Deployment schedule.
    #[must_use]
    pub const fn schedule(&self) -> ExecutionSchedule {
        self.schedule
    }

    pub(crate) fn validate(&self) -> Result<(), Diagnostic> {
        self.discretization.validate_space(self.space)?;
        validate_target_schedule(self.target, self.schedule)
    }
}

/// Frozen definition of default policy v0: generated P1 FEM on one host CPU.
///
/// # Errors
/// Returns `EQ0807` only if the internally declared policy becomes inconsistent.
pub fn default_plan_v0() -> Result<RealizationPlan, Diagnostic> {
    let one = NonZeroUsize::MIN;
    RealizationPlan::new(
        Space::continuous_lagrange(NonZeroU16::MIN),
        Discretization::new(
            DiscretizationMethod::ContinuousGalerkin,
            MeshPolicy::GeneratedUniform {
                cells_per_axis: NonZeroUsize::new(16).expect("16 is non-zero"),
            },
            QuadraturePolicy::GaussLegendre {
                points_per_axis: NonZeroUsize::new(2).expect("2 is non-zero"),
            },
        ),
        SolverPlan::new(
            LinearSolver::ConjugateGradient,
            1.0e-13,
            1.0e-14,
            NonZeroUsize::new(512).expect("512 is non-zero"),
        )?,
        Target::HostCpu { threads: one },
        ExecutionSchedule::Offline,
    )
}
