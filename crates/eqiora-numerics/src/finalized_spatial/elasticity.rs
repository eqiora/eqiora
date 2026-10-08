use eqiora_assembly::AssemblyReport;
use eqiora_core::Diagnostic;
use eqiora_realization::{Target, VectorLayoutKind};
use eqiora_solver::{
    CanonicalCsrSystemView, LinearOperatorProperties, LinearProblem, LinearSolution, SolverPlan,
};

use super::FinalizedLinearCore;
use crate::cartesian_elasticity::{
    CartesianLinearElasticity2dSolution, ConformingCartesianLinearElasticityPair2dSolution,
};
use crate::cartesian_elasticity::{
    FinalizedCartesianElasticity2dAssembly, FinalizedCartesianElasticity2dState,
    FinalizedConformingCartesianElasticityPair2dAssembly,
    FinalizedConformingCartesianElasticityPair2dState,
};
use crate::discrete_block::DiscreteBlockSystem;

/// Finalized algebraic handoff for one resolved Cartesian Q1 elasticity
/// realization.
///
/// Boundary recognition and package normalization are complete before this
/// value exists. It owns one immutable reduced CSR system and keeps full-
/// system constraint/reaction state opaque until an independently accepted
/// [`LinearSolution`] is returned.
#[derive(Debug, Clone, PartialEq)]
pub struct FinalizedIsotropicElasticityCartesian2dProblem {
    core: FinalizedLinearCore<f64>,
    state: FinalizedCartesianElasticity2dState,
}

impl FinalizedIsotropicElasticityCartesian2dProblem {
    pub(crate) fn new(
        solver: SolverPlan,
        vector_layout: VectorLayoutKind,
        target: Target,
        assembly: FinalizedCartesianElasticity2dAssembly,
    ) -> Result<Self, Diagnostic> {
        let (canonical_system, state) = assembly.into_canonical()?;
        Ok(Self {
            core: FinalizedLinearCore::new(solver, vector_layout, target, canonical_system),
            state,
        })
    }

    /// Mathematical properties asserted by the accepted Q1 realization.
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

    /// Borrow the single captured reduced CSR system and right-hand side.
    #[must_use]
    pub fn canonical_csr_system_view(&self) -> &CanonicalCsrSystemView {
        self.core.canonical_csr_system_view()
    }

    /// Accepted assembly placement and packet-shape evidence.
    #[must_use]
    pub const fn assembly_report(&self) -> &AssemblyReport {
        self.state.assembly_report()
    }

    /// Borrow the finalized system through the common solver problem contract.
    ///
    /// # Errors
    /// Returns `EQ0802` only if the captured canonical view contradicts its
    /// construction invariants.
    pub fn linear_problem(&self) -> Result<LinearProblem<'_>, Diagnostic> {
        self.core.linear_problem()
    }

    /// Reaccept one solution against this exact finalized system and recover
    /// the constrained vector field plus full-system balance evidence.
    ///
    /// # Errors
    /// Returns `EQ0807` for cross-wired plan/topology evidence and `EQ0802`
    /// when the values do not satisfy the finalized system or reconstruction
    /// storage cannot be reserved.
    pub fn finish(
        self,
        solution: LinearSolution,
    ) -> Result<CartesianLinearElasticity2dSolution, Diagnostic> {
        self.core.validate_solution(&solution)?;
        self.state
            .finish(solution, self.core.into_canonical_system())
    }
}

/// Finalized monolithic algebraic handoff for two conforming elasticity bodies.
///
/// The reduced CSR system uses a Realization-owned interface vertex quotient.
/// Body-local meshes and cut systems remain opaque until solution acceptance,
/// where they reconstruct both fields and expose weak interface equilibrium.
#[derive(Debug, Clone, PartialEq)]
pub struct FinalizedConformingIsotropicElasticityCartesianPair2dProblem {
    core: FinalizedLinearCore<f64>,
    state: FinalizedConformingCartesianElasticityPair2dState,
}

impl FinalizedConformingIsotropicElasticityCartesianPair2dProblem {
    pub(crate) fn new(
        solver: SolverPlan,
        vector_layout: VectorLayoutKind,
        target: Target,
        assembly: FinalizedConformingCartesianElasticityPair2dAssembly,
        block_system: DiscreteBlockSystem,
    ) -> Result<Self, Diagnostic> {
        let (canonical_system, state) = assembly.into_canonical()?;
        let core = FinalizedLinearCore::new(solver, vector_layout, target, canonical_system)
            .with_block_system(&block_system, state.assembly_report())?;
        Ok(Self { core, state })
    }

    /// Mathematical properties asserted by the accepted monolithic Q1 realization.
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

    /// Borrow the single captured reduced quotient-space CSR system.
    #[must_use]
    pub fn canonical_csr_system_view(&self) -> &CanonicalCsrSystemView {
        self.core.canonical_csr_system_view()
    }

    /// Accepted assembly placement and packet-shape evidence.
    #[must_use]
    pub const fn assembly_report(&self) -> &AssemblyReport {
        self.state.assembly_report()
    }

    /// Borrow the finalized system through the common solver problem contract.
    ///
    /// # Errors
    /// Returns `EQ0802` only if the captured canonical view contradicts its
    /// construction invariants.
    pub fn linear_problem(&self) -> Result<LinearProblem<'_>, Diagnostic> {
        self.core.linear_problem()
    }

    /// Reaccept one solution and recover both body fields and interface balance.
    ///
    /// # Errors
    /// Returns `EQ0807` for cross-wired plan/topology evidence and `EQ0802`
    /// when the values do not satisfy the finalized quotient system.
    pub fn finish(
        self,
        solution: LinearSolution,
    ) -> Result<ConformingCartesianLinearElasticityPair2dSolution, Diagnostic> {
        self.core.validate_solution(&solution)?;
        let Self { core, state } = self;
        state.finish(solution, core.into_canonical_system())
    }
}
