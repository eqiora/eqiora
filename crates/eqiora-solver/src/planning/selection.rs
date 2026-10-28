//! Owned selection facts, independent from a provider's execution interface.
use super::*;

pub(super) const CUDA_PROVIDER: SolverProvider = SolverProvider::new(
    BackendId::new("eqiora.cuda.krylov"),
    env!("CARGO_PKG_VERSION"),
    &[],
);
pub(super) const CUDA_CG_ID: &str = "eqiora.cuda.cg-spd-jacobi-fast-f64";
pub(super) const CUDA_BICGSTAB_ID: &str = "eqiora.cuda.bicgstab-general-identity-fast-f64";
pub(super) const CUDA_MINRES_ID: &str = "eqiora.cuda.minres-indefinite-identity-fast-f64";

/// Declared availability supplied by an execution owner.
///
/// This value is not device authentication or permission to execute. The concrete
/// owner must authenticate its implementation and revalidate the selected plan
/// against the actual problem and deployment before numerical mutation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolverProviderAvailability {
    pub(super) provider: SolverProvider,
    pub(super) execution_provider: ExecutionProvider,
    pub(super) execution: crate::ExecutionReport,
    pub(super) capabilities: crate::SolverCapabilities,
}

impl SolverProviderAvailability {
    /// Validate the descriptors' shape and their execution identity association.
    pub fn new(
        provider: SolverProvider,
        execution_provider: ExecutionProvider,
        execution: crate::ExecutionReport,
        capabilities: crate::SolverCapabilities,
    ) -> Result<Self, Diagnostic> {
        provider.validate()?;
        execution_provider.validate()?;
        if execution.adapter() != execution_provider.id() {
            return Err(invalid_catalog("availability.execution-mismatch"));
        }
        Ok(Self {
            provider,
            execution_provider,
            execution,
            capabilities,
        })
    }
}

/// An inspectable exact solver choice, without a host backend or device handle.
/// Execution remains with the existing concrete admitted execution owner.
#[derive(Debug, Clone)]
pub struct SolverSelection {
    profile: SolverPlanningProfile,
    scalar_domain: ScalarDomain,
    scalar_type: ScalarType,
    objective: SolverPlanningObjective,
    candidate_id: &'static str,
    evidence_case: &'static str,
    plan: SolverPlan,
    provider: SolverProvider,
    reasons: Vec<(&'static str, &'static str)>,
    costs: Vec<SolverCandidateCost>,
}

impl SolverSelection {
    /// Resolve from exact declared availability through the shared catalog and ranker.
    /// Unavailable providers remain rejected candidates; execution never falls back.
    pub fn resolve<S: Scalar + num_complex::ComplexFloat<Real = f64> + Sync>(
        profile: SolverPlanningProfile,
        objective: SolverPlanningObjective,
        relative_tolerance: f64,
        absolute_tolerance: f64,
        maximum_iterations: NonZeroUsize,
        available: &[SolverProviderAvailability],
    ) -> Result<Self, Diagnostic> {
        if available
            .iter()
            .enumerate()
            .any(|(index, a)| available[..index].iter().any(|b| a.provider == b.provider))
        {
            return Err(invalid_catalog("availability.duplicate-provider"));
        }
        let candidates = profile_catalog_ids(S::DOMAIN, &profile)
            .iter()
            .map(|id| {
                let expected = expected_candidate(id);
                let source = available
                    .iter()
                    .find(|a| a.provider == expected.provider)
                    .map(CandidateProvider::Declared);
                Ok(SolverCandidate::<S> {
                    id,
                    evidence_case: expected.evidence_case,
                    plan: catalog_plan(
                        expected.algorithm,
                        expected.preconditioner,
                        expected.reduction,
                        relative_tolerance,
                        absolute_tolerance,
                        maximum_iterations,
                    )?,
                    source,
                })
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        let resolved = resolve_candidates(profile.clone(), objective, &candidates)?;
        Ok(Self {
            profile,
            scalar_domain: S::DOMAIN,
            scalar_type: S::STORAGE,
            objective,
            candidate_id: resolved.selected.id(),
            evidence_case: resolved.selected.evidence_case(),
            plan: resolved.selected.plan(),
            provider: resolved.solver_provider,
            reasons: resolved.reasons,
            costs: resolved.costs,
        })
    }

    /// Exact operator and execution facts used during selection.
    pub fn profile(&self) -> SolverPlanningProfile {
        self.profile.clone()
    }
    /// Requested ranking objective.
    pub const fn objective(&self) -> SolverPlanningObjective {
        self.objective
    }
    /// Versioned deterministic policy identity.
    pub const fn policy_id(&self) -> &'static str {
        POLICY_ID
    }
    /// Selected catalog identity.
    pub const fn selected_candidate_id(&self) -> &'static str {
        self.candidate_id
    }
    /// Registered evidence for the selection claim, not a device execution receipt.
    pub const fn selected_evidence_case(&self) -> &'static str {
        self.evidence_case
    }
    /// Exact algorithm, preconditioner, reduction and convergence controls.
    pub const fn solver_plan(&self) -> SolverPlan {
        self.plan
    }
    /// Exact selected solver provider release.
    pub const fn solver_provider(&self) -> SolverProvider {
        self.provider
    }
    /// Explicit execution provider; placement is never ranked.
    pub const fn execution_provider(&self) -> ExecutionProvider {
        self.profile.execution_provider
    }
    /// Explicit execution topology used in candidate admission.
    pub const fn execution(&self) -> crate::ExecutionReport {
        self.profile.execution
    }
    /// Deterministic admitted/rejected candidate reasons and cost assumptions.
    pub fn reasons(&self) -> impl ExactSizeIterator<Item = (&'static str, &'static str)> + '_ {
        self.reasons.iter().copied()
    }
    /// Estimates retain unknown work and allocation bounds explicitly.
    pub fn costs(&self) -> &[SolverCandidateCost] {
        &self.costs
    }
    /// Reauthenticate actual mathematical, structure and resource facts before execution.
    pub fn require_problem<S: Scalar + num_complex::ComplexFloat<Real = f64> + Sync>(
        &self,
        problem: &LinearProblem<'_, S>,
        structure: Option<&crate::AlgebraicStructure>,
    ) -> Result<(), Diagnostic> {
        if S::DOMAIN != self.scalar_domain || S::STORAGE != self.scalar_type {
            return Err(invalid_catalog("profile.scalar-mismatch"));
        }
        self.profile.require_structure(structure)?;
        self.profile.require_plan(self.plan)?;
        self.profile.require_problem(problem)?;
        self.profile
            .require_problem_resources(self.plan, self.provider, problem)
    }
}

pub(super) fn profile_catalog_ids(
    domain: ScalarDomain,
    profile: &SolverPlanningProfile,
) -> &'static [&'static str] {
    match profile.execution.topology() {
        crate::ExecutionTopology::Host { .. } => catalog_ids(domain, profile.facts.properties),
        crate::ExecutionTopology::Cuda { .. } if domain == ScalarDomain::Real => {
            match profile.facts.properties {
                LinearOperatorProperties::General => &[CUDA_BICGSTAB_ID],
                LinearOperatorProperties::SymmetricPositiveDefinite => &[CUDA_CG_ID],
                LinearOperatorProperties::SymmetricIndefinite => &[CUDA_MINRES_ID],
                _ => &[],
            }
        }
        _ => &[],
    }
}
