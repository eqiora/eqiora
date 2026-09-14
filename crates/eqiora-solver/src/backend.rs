use std::collections::BTreeSet;
use std::fmt::Debug;

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;

use crate::{
    CanonicalCsrSystemView, LinearOperatorOrientation, LinearOperatorProperties, LinearProblem,
    LinearSolution, LinearSolver, PreconditionerPolicy, PreparedLinearSolver, ReductionPolicy,
    ReplicatedLinearExecution, SERIAL_LINEAR_EXECUTION, ScalarType, SolverPlan, SolverProvider,
    Transposed,
};

/// One exact numerical-policy tuple implemented by a solver adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SolverCapability {
    /// Krylov or direct algorithm.
    pub algorithm: LinearSolver,
    /// Mathematical operator assertion accepted by that algorithm path.
    pub operator_properties: LinearOperatorProperties,
    /// Preconditioner implemented for this exact path.
    pub preconditioner: PreconditionerPolicy,
    /// Reduction order implemented for this exact path.
    pub reduction: ReductionPolicy,
    /// Scalar representation implemented for this exact path.
    pub scalar_type: ScalarType,
}

/// Stable Eqiora-owned identity for a concrete solver adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BackendId(&'static str);

impl BackendId {
    /// Construct a namespaced compile-time backend identity.
    #[must_use]
    pub const fn new(value: &'static str) -> Self {
        Self(value)
    }

    /// Namespaced backend identity.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        self.0
    }
}

/// Exact numerical policies admitted by one adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolverCapabilities {
    combinations: BTreeSet<SolverCapability>,
    algorithms: BTreeSet<LinearSolver>,
    preconditioners: BTreeSet<PreconditionerPolicy>,
    reductions: BTreeSet<ReductionPolicy>,
    scalar_types: BTreeSet<ScalarType>,
}

impl SolverCapabilities {
    /// Capabilities of the deterministic host-local reference oracle.
    #[must_use]
    pub fn reference() -> Self {
        let mut combinations = vec![
            SolverCapability {
                algorithm: LinearSolver::ConjugateGradient,
                operator_properties: LinearOperatorProperties::SymmetricPositiveDefinite,
                preconditioner: PreconditionerPolicy::Identity,
                reduction: ReductionPolicy::Reproducible,
                scalar_type: ScalarType::F64,
            },
            SolverCapability {
                algorithm: LinearSolver::ConjugateGradient,
                operator_properties: LinearOperatorProperties::SymmetricPositiveDefinite,
                preconditioner: PreconditionerPolicy::Jacobi,
                reduction: ReductionPolicy::Reproducible,
                scalar_type: ScalarType::F64,
            },
            SolverCapability {
                algorithm: LinearSolver::MinimumResidual,
                operator_properties: LinearOperatorProperties::SymmetricPositiveDefinite,
                preconditioner: PreconditionerPolicy::Identity,
                reduction: ReductionPolicy::Reproducible,
                scalar_type: ScalarType::F64,
            },
            SolverCapability {
                algorithm: LinearSolver::MinimumResidual,
                operator_properties: LinearOperatorProperties::SymmetricIndefinite,
                preconditioner: PreconditionerPolicy::Identity,
                reduction: ReductionPolicy::Reproducible,
                scalar_type: ScalarType::F64,
            },
        ];
        for operator_properties in [
            LinearOperatorProperties::General,
            LinearOperatorProperties::SymmetricPositiveDefinite,
            LinearOperatorProperties::SymmetricIndefinite,
        ] {
            for preconditioner in [PreconditionerPolicy::Identity, PreconditionerPolicy::Jacobi] {
                combinations.push(SolverCapability {
                    algorithm: LinearSolver::BiConjugateGradientStabilized,
                    operator_properties,
                    preconditioner,
                    reduction: ReductionPolicy::Reproducible,
                    scalar_type: ScalarType::F64,
                });
            }
        }
        Self::exact(combinations).expect("reference exact capability set is nonempty")
    }

    /// Construct a nonempty capability set when every supplied axis forms a
    /// genuinely implemented Cartesian product.
    ///
    /// # Errors
    /// Returns `EQ0807` if any capability axis is empty.
    pub fn new(
        algorithms: impl IntoIterator<Item = LinearSolver>,
        preconditioners: impl IntoIterator<Item = PreconditionerPolicy>,
        reductions: impl IntoIterator<Item = ReductionPolicy>,
        scalar_types: impl IntoIterator<Item = ScalarType>,
    ) -> Result<Self, Diagnostic> {
        let algorithms = algorithms.into_iter().collect::<BTreeSet<_>>();
        let preconditioners = preconditioners.into_iter().collect::<BTreeSet<_>>();
        let reductions = reductions.into_iter().collect::<BTreeSet<_>>();
        let scalar_types = scalar_types.into_iter().collect::<BTreeSet<_>>();
        if algorithms.is_empty()
            || preconditioners.is_empty()
            || reductions.is_empty()
            || scalar_types.is_empty()
        {
            return Err(unsupported(
                "solver capabilities require at least one value on every axis",
            ));
        }
        let combinations = algorithms
            .iter()
            .flat_map(|algorithm| {
                let properties: &[_] = match algorithm {
                    LinearSolver::ConjugateGradient => {
                        &[LinearOperatorProperties::SymmetricPositiveDefinite]
                    }
                    LinearSolver::MinimumResidual => &[
                        LinearOperatorProperties::SymmetricPositiveDefinite,
                        LinearOperatorProperties::SymmetricIndefinite,
                    ],
                    LinearSolver::BiConjugateGradientStabilized => &[
                        LinearOperatorProperties::General,
                        LinearOperatorProperties::SymmetricPositiveDefinite,
                        LinearOperatorProperties::SymmetricIndefinite,
                    ],
                    LinearSolver::SparseLu => &[
                        LinearOperatorProperties::General,
                        LinearOperatorProperties::SymmetricPositiveDefinite,
                        LinearOperatorProperties::SymmetricIndefinite,
                    ],
                };
                properties.iter().flat_map(|operator_properties| {
                    preconditioners.iter().flat_map(|preconditioner| {
                        reductions.iter().flat_map(|reduction| {
                            scalar_types.iter().map(|scalar_type| SolverCapability {
                                algorithm: *algorithm,
                                operator_properties: *operator_properties,
                                preconditioner: *preconditioner,
                                reduction: *reduction,
                                scalar_type: *scalar_type,
                            })
                        })
                    })
                })
            })
            .collect();
        Ok(Self {
            combinations,
            algorithms,
            preconditioners,
            reductions,
            scalar_types,
        })
    }

    /// Construct a nonempty set of exact supported tuples without taking a
    /// Cartesian product of independent-looking policy axes.
    ///
    /// # Errors
    /// Returns `EQ0807` when no tuple is supplied.
    pub fn exact(
        combinations: impl IntoIterator<Item = SolverCapability>,
    ) -> Result<Self, Diagnostic> {
        let combinations = combinations.into_iter().collect::<BTreeSet<_>>();
        if combinations.is_empty() {
            return Err(unsupported(
                "solver capabilities require at least one exact policy tuple",
            ));
        }
        if let Some(invalid) = combinations
            .iter()
            .find(|entry| !entry.algorithm.accepts(entry.operator_properties))
        {
            return Err(unsupported(format!(
                "solver capability has an incompatible algorithm/property pair: {invalid:?}"
            )));
        }
        Ok(Self {
            algorithms: combinations.iter().map(|entry| entry.algorithm).collect(),
            preconditioners: combinations
                .iter()
                .map(|entry| entry.preconditioner)
                .collect(),
            reductions: combinations.iter().map(|entry| entry.reduction).collect(),
            scalar_types: combinations.iter().map(|entry| entry.scalar_type).collect(),
            combinations,
        })
    }

    /// Exact implemented policy tuples.
    #[must_use]
    pub const fn combinations(&self) -> &BTreeSet<SolverCapability> {
        &self.combinations
    }

    /// Algorithms admitted by this adapter.
    #[must_use]
    pub const fn algorithms(&self) -> &BTreeSet<LinearSolver> {
        &self.algorithms
    }

    /// Preconditioners admitted by this adapter.
    #[must_use]
    pub const fn preconditioners(&self) -> &BTreeSet<PreconditionerPolicy> {
        &self.preconditioners
    }

    /// Reduction policies admitted by this adapter.
    #[must_use]
    pub const fn reductions(&self) -> &BTreeSet<ReductionPolicy> {
        &self.reductions
    }

    /// Scalar representations admitted by this adapter.
    #[must_use]
    pub const fn scalar_types(&self) -> &BTreeSet<ScalarType> {
        &self.scalar_types
    }

    /// Whether a scalar representation is admitted.
    #[must_use]
    pub fn supports_scalar(&self, scalar_type: ScalarType) -> bool {
        self.scalar_types.contains(&scalar_type)
    }

    /// Validate a plan and scalar representation without fallback.
    ///
    /// # Errors
    /// Returns `EQ0807` for any unsupported selection.
    pub fn require(&self, plan: SolverPlan, scalar_type: ScalarType) -> Result<(), Diagnostic> {
        if !self.combinations.iter().any(|entry| {
            entry.algorithm == plan.algorithm()
                && entry.preconditioner == plan.preconditioner()
                && entry.reduction == plan.reduction()
                && entry.scalar_type == scalar_type
        }) {
            return Err(unsupported(format!(
                "solver backend does not support the exact {:?}/{:?}/{:?}/{scalar_type:?} policy tuple",
                plan.algorithm(),
                plan.preconditioner(),
                plan.reduction()
            )));
        }
        Ok(())
    }

    /// Validate a complete plan, scalar type, and operator assertion.
    ///
    /// # Errors
    /// Returns `EQ0807` unless the exact tuple is implemented.
    pub fn require_problem(
        &self,
        plan: SolverPlan,
        scalar_type: ScalarType,
        operator_properties: LinearOperatorProperties,
    ) -> Result<(), Diagnostic> {
        let requested = SolverCapability {
            algorithm: plan.algorithm(),
            operator_properties,
            preconditioner: plan.preconditioner(),
            reduction: plan.reduction(),
            scalar_type,
        };
        if !self.combinations.contains(&requested) {
            return Err(unsupported(format!(
                "solver backend does not support the exact {requested:?} tuple"
            )));
        }
        Ok(())
    }
}

/// Backend-neutral solver execution boundary.
pub trait LinearSolverBackend: Debug + Sync {
    /// Stable identity and declared release/dependency inventory of this provider.
    fn provider(&self) -> SolverProvider;

    /// Stable adapter identity used in evidence and diagnostics.
    fn id(&self) -> BackendId {
        self.provider().id()
    }

    /// Exact numerical policy admitted by this adapter.
    fn capabilities(&self) -> SolverCapabilities;

    /// Prepare provider-private state for repeated run-local solves.
    ///
    /// `None` means the provider has no prepared implementation and the caller
    /// executes every candidate through `solve_with_execution`. A returned
    /// session must validate the complete structure identity and actual sparse
    /// topology before reusing provider state.
    ///
    /// # Errors
    /// Returns a structured capability or policy diagnostic.
    fn prepare_linear(
        &self,
        _plan: SolverPlan,
    ) -> Result<Option<Box<dyn PreparedLinearSolver + '_>>, Diagnostic> {
        Ok(None)
    }

    /// Solve one validated problem under the exact plan.
    ///
    /// # Errors
    /// Returns a stable diagnostic for unsupported policy, invalid operator
    /// behavior, breakdown, non-convergence, or true-residual rejection.
    fn solve(
        &self,
        problem: &LinearProblem<'_>,
        plan: SolverPlan,
    ) -> Result<LinearSolution, Diagnostic> {
        self.solve_with_execution(problem, plan, &SERIAL_LINEAR_EXECUTION)
    }

    /// Solve through one explicit replicated-vector execution.
    ///
    /// Backends must consume the execution or reject it before numerical work;
    /// they must not silently run elsewhere and rewrite provenance afterward.
    ///
    /// # Errors
    /// Returns a stable diagnostic for incompatible execution, unsupported
    /// policy, invalid operator behavior, breakdown, non-convergence, or
    /// true-residual rejection.
    fn solve_with_execution(
        &self,
        problem: &LinearProblem<'_>,
        plan: SolverPlan,
        execution: &dyn ReplicatedLinearExecution,
    ) -> Result<LinearSolution, Diagnostic>;
}

/// One resolved backend instance paired with the sole validated solver plan.
#[derive(Debug, Clone, Copy)]
pub struct LinearSolveRequest<'a> {
    backend: &'a dyn LinearSolverBackend,
    plan: SolverPlan,
}

impl<'a> LinearSolveRequest<'a> {
    /// Bind an executable adapter to a validated plan.
    #[must_use]
    pub const fn new(backend: &'a dyn LinearSolverBackend, plan: SolverPlan) -> Self {
        Self { backend, plan }
    }

    /// Execute one problem without translating the plan.
    ///
    /// # Errors
    /// Returns the backend's structured capability or numerical diagnostic.
    pub fn solve(&self, problem: &LinearProblem<'_>) -> Result<LinearSolution, Diagnostic> {
        self.backend.solve(problem, self.plan)
    }

    /// Solve one derivative right-hand side against an exact canonical CSR
    /// coefficient source in the requested orientation.
    ///
    /// The source's captured right-hand side remains its primal provenance;
    /// `right_hand_side` is the distinct right-hand side for this solve.
    ///
    /// # Errors
    /// Returns a structured shape, capability, or numerical diagnostic from
    /// problem validation or the selected backend.
    pub fn solve_canonical_oriented(
        &self,
        state_jacobian: &CanonicalCsrSystemView,
        right_hand_side: &[f64],
        orientation: LinearOperatorOrientation,
    ) -> Result<LinearSolution, Diagnostic> {
        match orientation {
            LinearOperatorOrientation::Normal => {
                let problem = LinearProblem::from_oriented_canonical(
                    state_jacobian,
                    state_jacobian,
                    right_hand_side,
                )?;
                self.solve(&problem)
            }
            LinearOperatorOrientation::Transposed => {
                let transposed = Transposed::new(state_jacobian);
                let problem = LinearProblem::from_oriented_canonical(
                    &transposed,
                    state_jacobian,
                    right_hand_side,
                )?;
                self.solve(&problem)
            }
        }
    }

    /// Resolved adapter.
    #[must_use]
    pub const fn backend(self) -> &'a dyn LinearSolverBackend {
        self.backend
    }

    /// Exact solver plan passed to the adapter.
    #[must_use]
    pub const fn plan(self) -> SolverPlan {
        self.plan
    }
}

fn unsupported(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_REALIZATION, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_capabilities_reject_mathematically_invalid_pairs() {
        for (algorithm, properties) in [
            (
                LinearSolver::ConjugateGradient,
                LinearOperatorProperties::General,
            ),
            (
                LinearSolver::MinimumResidual,
                LinearOperatorProperties::General,
            ),
        ] {
            let result = SolverCapabilities::exact([SolverCapability {
                algorithm,
                operator_properties: properties,
                preconditioner: PreconditionerPolicy::Identity,
                reduction: ReductionPolicy::Reproducible,
                scalar_type: ScalarType::F64,
            }]);
            assert_eq!(result.unwrap_err().code(), codes::INVALID_REALIZATION);
        }
    }
}
