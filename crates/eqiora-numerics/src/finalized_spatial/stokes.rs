use eqiora_assembly::AssemblyReport;
use eqiora_core::Diagnostic;
use eqiora_realization::{Target, VectorLayoutKind};
use eqiora_solver::{
    CanonicalCsrSystemView, LinearOperatorProperties, LinearProblem, LinearSolution, SolverPlan,
};

use super::FinalizedLinearCore;
use crate::discrete_block::DiscreteBlockSystem;
use crate::simplicial_stokes::SimplicialMiniStokesSolution2d;
use crate::simplicial_stokes::{FinalizedMiniStokesAssembly, FinalizedMiniStokesState};

/// Finalized algebraic handoff for one resolved simplicial MINI Stokes problem.
///
/// Canonical role recognition, scale selection, and mixed-space admission are
/// complete before this value exists. Execution adapters see only one captured
/// symmetric-indefinite CSR system and the sole [`SolverPlan`]; pressure-
/// reference, boundary-elimination, and reconstruction state remain opaque
/// until an independently accepted [`LinearSolution`] returns. The reference
/// is either an independent zero-integral constraint or the admitted
/// prescribed-traction boundary, never an implicit execution choice.
#[derive(Debug, Clone, PartialEq)]
pub struct FinalizedSimplicialMiniStokes2dProblem {
    core: FinalizedLinearCore<f64>,
    state: FinalizedMiniStokesState,
}

impl FinalizedSimplicialMiniStokes2dProblem {
    pub(crate) fn new(
        solver: SolverPlan,
        vector_layout: VectorLayoutKind,
        target: Target,
        assembly: FinalizedMiniStokesAssembly,
    ) -> Result<Self, Diagnostic> {
        let (canonical_system, state) = assembly.into_canonical()?;
        Ok(Self {
            core: FinalizedLinearCore::new(solver, vector_layout, target, canonical_system),
            state,
        })
    }

    pub(crate) fn with_block_system(
        mut self,
        block_system: &DiscreteBlockSystem,
    ) -> Result<Self, Diagnostic> {
        self.core = self
            .core
            .with_block_system(block_system, self.state.assembly_report())?;
        Ok(self)
    }

    /// Mathematical properties asserted by the resolved mixed Realization.
    #[must_use]
    pub fn operator_properties(&self) -> LinearOperatorProperties {
        self.core.operator_properties()
    }

    /// Exact backend-neutral solver policy selected by the Realization.
    #[must_use]
    pub const fn solver_plan(&self) -> SolverPlan {
        self.core.solver_plan()
    }

    /// Accepted vector layout at the algebraic handoff.
    #[must_use]
    pub const fn vector_layout(&self) -> VectorLayoutKind {
        self.core.vector_layout()
    }

    /// Borrow the single captured reduced mixed CSR system.
    #[must_use]
    pub fn canonical_csr_system_view(&self) -> &CanonicalCsrSystemView {
        self.core.canonical_csr_system_view()
    }

    /// Accepted assembly placement and packet-shape evidence.
    #[must_use]
    pub const fn assembly_report(&self) -> &AssemblyReport {
        self.state.assembly_report()
    }

    /// Borrow the finalized system through the common solver boundary.
    ///
    /// # Errors
    /// Returns `EQ0802` only if captured CSR state contradicts construction
    /// invariants.
    pub fn linear_problem(&self) -> Result<LinearProblem<'_>, Diagnostic> {
        self.core.linear_problem()
    }

    /// Reaccept and reconstruct one solution against this exact mixed system.
    ///
    /// # Errors
    /// Returns `EQ0807` for plan or topology drift and `EQ0802` when supplied
    /// values fail the finalized true-residual contract.
    pub fn finish(
        self,
        solution: LinearSolution,
    ) -> Result<SimplicialMiniStokesSolution2d, Diagnostic> {
        self.core.validate_solution(&solution)?;
        self.state
            .finish(solution, self.core.into_canonical_system())
    }
}
