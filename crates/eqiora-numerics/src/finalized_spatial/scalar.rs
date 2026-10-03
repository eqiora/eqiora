use eqiora_assembly::AssemblyReport;
use eqiora_core::Diagnostic;
use eqiora_realization::{
    DiscretizationMethod, PortableRealizationGraph, Target, VectorLayoutKind,
};
use eqiora_solver::{
    CanonicalCsrSystemView, LinearOperatorProperties, LinearProblem, LinearSolution, SolverPlan,
};

use super::FinalizedLinearCore;
use crate::canonical::ResolvedScalarEllipticCartesianSolution;
use crate::cartesian_elliptic::{
    FinalizedCartesianFemAssembly, FinalizedCartesianFemState, FinalizedCartesianFvmAssembly,
    FinalizedCartesianFvmState,
};

/// Finalized algebraic handoff for one resolved Cartesian scalar-elliptic
/// realization.
///
/// The public boundary is deliberately method-neutral: an execution adapter
/// sees the immutable sparse system, asserted operator properties, sole
/// [`SolverPlan`], and assembly evidence. FEM constraint recovery and TPFA
/// reconstruction remain opaque until an independently accepted
/// [`LinearSolution`] is returned.
#[derive(Debug, Clone, PartialEq)]
pub struct FinalizedScalarEllipticCartesianProblem {
    portable_realization: PortableRealizationGraph,
    method: DiscretizationMethod,
    core: FinalizedLinearCore,
    state: FinalizedScalarEllipticCartesianState,
}

#[derive(Debug, Clone, PartialEq)]
enum FinalizedScalarEllipticCartesianState {
    FiniteElement(FinalizedCartesianFemState),
    FiniteVolume(FinalizedCartesianFvmState),
}

impl FinalizedScalarEllipticCartesianProblem {
    pub(crate) fn finite_element_blocks(
        portable_realization: PortableRealizationGraph,
        solver: SolverPlan,
        vector_layout: VectorLayoutKind,
        target: Target,
        assembly: crate::cartesian_elliptic::linear::CartesianLinearAssembly,
    ) -> Result<Self, Diagnostic> {
        let (canonical_system, state) = assembly.into_single_field_canonical()?;
        Ok(Self {
            portable_realization,
            method: DiscretizationMethod::ContinuousGalerkin,
            core: FinalizedLinearCore::new(solver, vector_layout, target, canonical_system),
            state: FinalizedScalarEllipticCartesianState::FiniteElement(state),
        })
    }

    pub(crate) fn finite_element(
        portable_realization: PortableRealizationGraph,
        solver: SolverPlan,
        vector_layout: VectorLayoutKind,
        target: Target,
        assembly: FinalizedCartesianFemAssembly,
    ) -> Result<Self, Diagnostic> {
        let (canonical_system, state) = assembly.into_canonical()?;
        Ok(Self {
            portable_realization,
            method: DiscretizationMethod::ContinuousGalerkin,
            core: FinalizedLinearCore::new(solver, vector_layout, target, canonical_system),
            state: FinalizedScalarEllipticCartesianState::FiniteElement(state),
        })
    }

    pub(crate) fn finite_volume(
        portable_realization: PortableRealizationGraph,
        solver: SolverPlan,
        vector_layout: VectorLayoutKind,
        target: Target,
        assembly: FinalizedCartesianFvmAssembly,
    ) -> Result<Self, Diagnostic> {
        let (canonical_system, state) = assembly.into_canonical(None)?;
        Ok(Self {
            portable_realization,
            method: DiscretizationMethod::CellCenteredFiniteVolume,
            core: FinalizedLinearCore::new(solver, vector_layout, target, canonical_system),
            state: FinalizedScalarEllipticCartesianState::FiniteVolume(state),
        })
    }

    /// Numerical method whose method-native state is retained opaquely.
    #[must_use]
    pub const fn method(&self) -> DiscretizationMethod {
        self.method
    }

    /// Equation-aware portable graph independently regenerated and checked by
    /// the finalizer that owns this exact canonical system.
    #[must_use]
    pub const fn portable_realization(&self) -> &PortableRealizationGraph {
        &self.portable_realization
    }

    /// Mathematical properties asserted by this resolved realization.
    #[must_use]
    pub fn operator_properties(&self) -> LinearOperatorProperties {
        self.core.operator_properties()
    }

    /// Exact backend-neutral solver policy selected by the Realization.
    #[must_use]
    pub const fn solver_plan(&self) -> SolverPlan {
        self.core.solver_plan()
    }

    /// Replicated or explicitly distributed vector layout admitted by the
    /// resolved Realization.
    #[must_use]
    pub const fn vector_layout(&self) -> VectorLayoutKind {
        self.core.vector_layout()
    }

    /// Borrow the single captured complete-CSR mathematical source.
    ///
    /// Distributed layout derivation and its execution adapter must consume
    /// this exact view. The view also supplies the host problem used by this
    /// handoff, so raw storage, distributed shards, and residual acceptance
    /// cannot select independent operator actions.
    #[must_use]
    pub fn canonical_csr_system_view(&self) -> &CanonicalCsrSystemView {
        self.core.canonical_csr_system_view()
    }

    /// Accepted assembly placement and packet-shape evidence.
    #[must_use]
    pub const fn assembly_report(&self) -> &AssemblyReport {
        match &self.state {
            FinalizedScalarEllipticCartesianState::FiniteElement(state) => state.assembly_report(),
            FinalizedScalarEllipticCartesianState::FiniteVolume(state) => state.assembly_report(),
        }
    }

    /// Borrow the finalized system through the common solver problem contract.
    ///
    /// # Errors
    /// Returns `EQ0802` only if the already captured canonical view
    /// contradicts its construction invariants.
    pub fn linear_problem(&self) -> Result<LinearProblem<'_>, Diagnostic> {
        self.core.linear_problem()
    }

    /// Numerically reaccept and reconstruct one solution against this
    /// finalized problem.
    ///
    /// Validation is intentionally repeated at the handoff. This proves that
    /// the supplied vector satisfies these finalized arrays under the selected
    /// plan and topology; it does not prove which system originally produced
    /// the vector. A vector that satisfies two systems is admissible to both.
    /// Durable system identity is deferred to artifact persistence.
    ///
    /// # Errors
    /// Returns `EQ0807` for cross-wired plan/topology evidence and `EQ0802`
    /// when the values do not satisfy the finalized system or its bounded
    /// verification/reconstruction storage cannot be reserved.
    pub fn finish(
        self,
        solution: LinearSolution,
    ) -> Result<ResolvedScalarEllipticCartesianSolution, Diagnostic> {
        self.core.validate_solution(&solution)?;
        let Self { core, state, .. } = self;
        let canonical_system = core.into_canonical_system();
        match state {
            FinalizedScalarEllipticCartesianState::FiniteElement(state) => state
                .finish(solution, canonical_system)
                .map(ResolvedScalarEllipticCartesianSolution::FiniteElement),
            FinalizedScalarEllipticCartesianState::FiniteVolume(state) => state
                .finish(solution, canonical_system)
                .map(ResolvedScalarEllipticCartesianSolution::FiniteVolume),
        }
    }
}
