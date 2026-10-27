use std::num::NonZeroUsize;

use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, Scalar, ScalarDomain};

use crate::{
    BackendId, ExecutionProvider, LinearOperatorOrientation, LinearOperatorProperties,
    LinearProblem, LinearSolution, LinearSolveRequest, LinearSolver, PreconditionerPolicy,
    ProviderLibrary, ReductionPolicy, SERIAL_EXECUTION_PROVIDER, ScalarType, SolverCapability,
    SolverPlan, SolverProvider,
};

mod catalog;
mod cost;
use catalog::{expected_candidate, rejection_reason};
pub use cost::{SolverCandidateCost, SolverCostRequest, SolverPlanningSize};

const POLICY_ID: &str = "eqiora.host-serial-solver-planning/v4";

const REFERENCE_ID: &str = "eqiora.reference.bicgstab-general-jacobi-reproducible-f64";
const FAER_BICGSTAB_ID: &str = "eqiora.faer.bicgstab-general-jacobi-fast-f64";
const REFERENCE_IDENTITY_ID: &str = "eqiora.reference.bicgstab-general-identity-reproducible-f64";
const FAER_IDENTITY_ID: &str = "eqiora.faer.bicgstab-general-identity-fast-f64";
const FAER_SPARSE_LU_ID: &str = "eqiora.faer.sparse-lu-general-identity-fast-f64";

const REFERENCE_CG_ID: &str = "eqiora.reference.cg-spd-identity-reproducible-f64";
const FAER_CG_ID: &str = "eqiora.faer.cg-spd-jacobi-fast-f64";
const FAER_SPD_LU_ID: &str = "eqiora.faer.sparse-lu-spd-identity-fast-f64";
const REFERENCE_CG_JACOBI_ID: &str = "eqiora.reference.cg-spd-jacobi-reproducible-f64";
const REFERENCE_SPD_MINRES_ID: &str = "eqiora.reference.minres-spd-identity-reproducible-f64";
const REFERENCE_SPD_BICGSTAB_ID: &str = "eqiora.reference.bicgstab-spd-identity-reproducible-f64";
const REFERENCE_SPD_BICGSTAB_JACOBI_ID: &str =
    "eqiora.reference.bicgstab-spd-jacobi-reproducible-f64";
const FAER_CG_IDENTITY_ID: &str = "eqiora.faer.cg-spd-identity-fast-f64";
const FAER_SPD_BICGSTAB_ID: &str = "eqiora.faer.bicgstab-spd-identity-fast-f64";
const FAER_SPD_BICGSTAB_JACOBI_ID: &str = "eqiora.faer.bicgstab-spd-jacobi-fast-f64";
const REFERENCE_INDEFINITE_BICGSTAB_ID: &str =
    "eqiora.reference.bicgstab-indefinite-identity-reproducible-f64";
const REFERENCE_INDEFINITE_BICGSTAB_JACOBI_ID: &str =
    "eqiora.reference.bicgstab-indefinite-jacobi-reproducible-f64";
const FAER_SYMMETRIC_LU_ID: &str = "eqiora.faer.sparse-lu-symmetric-identity-fast-f64";
const REFERENCE_MINRES_ID: &str = "eqiora.reference.minres-indefinite-identity-reproducible-f64";
const FAER_INDEFINITE_LU_ID: &str = "eqiora.faer.sparse-lu-indefinite-identity-fast-f64";

const REFERENCE_COMPLEX_ID: &str =
    "eqiora.reference.bicgstab-general-identity-reproducible-complex-f64";
const FAER_COMPLEX_LU_ID: &str = "eqiora.faer.sparse-lu-general-identity-fast-complex-f64";
const FAER_COMPLEX_SYMMETRIC_LU_ID: &str =
    "eqiora.faer.sparse-lu-complex-symmetric-identity-fast-complex-f64";
const FAER_HERMITIAN_LU_ID: &str = "eqiora.faer.sparse-lu-hermitian-identity-fast-complex-f64";
const REFERENCE_HPD_ID: &str = "eqiora.reference.cg-hpd-identity-reproducible-complex-f64";
const FAER_HPD_LU_ID: &str = "eqiora.faer.sparse-lu-hpd-identity-fast-complex-f64";
const PLANNING_EVIDENCE: &str = "numerics.host-serial-solver-planning";

const REFERENCE_EVIDENCE: &str = "fluid.cartesian-advection-diffusion-fvm-2d";
const FAER_EVIDENCE: &str = "numerics.linear-backends";
const REFERENCE_IDENTITY_EVIDENCE: &str = "fluid.cell-centered-navier-stokes-fvm-2d";
const FAER_IDENTITY_EVIDENCE: &str = "electrical.parallel-dc-network";

const EMPTY_LIBRARIES: &[ProviderLibrary] = &[];
const FAER_LIBRARIES: &[ProviderLibrary] = &[ProviderLibrary::new("faer", "0.24.4")];

const REFERENCE_PROVIDER: SolverProvider = SolverProvider::new(
    BackendId::new("eqiora.reference"),
    env!("CARGO_PKG_VERSION"),
    EMPTY_LIBRARIES,
);
const FAER_PROVIDER: SolverProvider = SolverProvider::new(
    BackendId::new("eqiora.faer"),
    env!("CARGO_PKG_VERSION"),
    FAER_LIBRARIES,
);

/// Deterministic preference table used by bounded host-serial solver planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverPlanningObjective {
    /// Prefer the frozen reproducible-reduction candidate.
    Robust,
    /// Prefer Fast reduction and then the frozen direct candidate.
    Fast,
    /// Prefer catalog iterative candidates before direct factorization.
    /// This is a deterministic preference, not a memory-usage guarantee.
    LowMemory,
}

/// Structural operator facts admitted before host-serial numerical work.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostSerialSolverProfile {
    facts: PlanningProfileFacts,
    required_reduction: Option<ReductionPolicy>,
    structure: Option<crate::AlgebraicStructure>,
    size: Option<SolverPlanningSize>,
    cost_request: Option<SolverCostRequest>,
}

impl HostSerialSolverProfile {
    /// Describe a normal-orientation, complete-diagonal canonical CSR General
    /// operator without constructing or applying its numerical coefficients.
    #[must_use]
    pub const fn general_canonical_csr() -> Self {
        Self {
            facts: PlanningProfileFacts::GENERAL_CANONICAL_CSR,
            required_reduction: None,
            structure: None,
            size: None,
            cost_request: None,
        }
    }

    /// Describe a normal-orientation canonical CSR operator using exact
    /// method-owned mathematical properties and structural diagonal availability.
    /// `Some(true)` asserts a complete diagonal, `Some(false)` asserts its
    /// absence, and `None` makes no diagonal claim. Jacobi requires `Some(true)`;
    /// identity-preconditioned candidates need no diagonal assertion.
    ///
    /// Properties are assertions supplied by the mathematical admission owner.
    /// This profile does not establish positive definiteness, remove a nullspace,
    /// or authenticate a pressure gauge. `with_structure` retains method-owned
    /// Field and multiplier ownership for admission and execution authentication.
    /// A required reduction is an execution constraint, independent of ranking
    /// objective; `None` permits either policy.
    /// Constraint elimination or explicit gauge augmentation must already be complete. No matrix values
    /// are inspected while planning; execution rechecks these structural facts.
    #[must_use]
    pub const fn canonical_csr(
        properties: LinearOperatorProperties,
        complete_diagonal: Option<bool>,
        required_reduction: Option<ReductionPolicy>,
    ) -> Self {
        Self {
            facts: PlanningProfileFacts {
                properties,
                orientation: LinearOperatorOrientation::Normal,
                canonical_csr: true,
                complete_diagonal,
            },
            required_reduction,
            structure: None,
            size: None,
            cost_request: None,
        }
    }

    /// Bind method-owned Field/gauge structure before exact admission or ranking.
    ///
    /// # Errors
    /// A multiplier gauge is incompatible with a positive-definite operator assertion.
    pub fn with_structure(
        mut self,
        structure: crate::AlgebraicStructure,
    ) -> Result<Self, Diagnostic> {
        if !structure.constraints().is_empty()
            && matches!(
                self.facts.properties,
                LinearOperatorProperties::SymmetricPositiveDefinite
                    | LinearOperatorProperties::HermitianPositiveDefinite
            )
        {
            return Err(Diagnostic::error(
                codes::INVALID_REALIZATION,
                "profile.gauge-multiplier-incompatible-with-positive-definite",
            ));
        }
        self.structure = Some(structure);
        Ok(self)
    }

    /// Reauthenticate the method-owned structure before execution.
    ///
    /// # Errors
    /// Rejects changed, omitted, or unexpected Field/gauge ownership.
    pub fn require_structure(
        &self,
        structure: Option<&crate::AlgebraicStructure>,
    ) -> Result<(), Diagnostic> {
        if self.structure.as_ref() != structure {
            return Err(Diagnostic::error(
                codes::INVALID_REALIZATION,
                "profile.algebraic-structure-mismatch",
            ));
        }
        Ok(())
    }

    /// Admit an exact solver tuple against the same structural and execution
    /// requirements used before catalog ranking.
    ///
    /// # Errors
    /// Returns `EQ0807` when Jacobi lacks a complete structural diagonal or the
    /// requested reduction differs from the execution requirement.
    pub fn require_plan(&self, plan: SolverPlan) -> Result<(), Diagnostic> {
        if let Some(reason) = self.plan_rejection(plan) {
            return Err(Diagnostic::error(
                codes::INVALID_REALIZATION,
                format!("{POLICY_ID} rejected solver plan: {reason}"),
            ));
        }
        Ok(())
    }

    fn plan_rejection(&self, plan: SolverPlan) -> Option<&'static str> {
        if self.cost_request.is_some_and(|cost| {
            [cost.identity_iterations(), cost.jacobi_iterations()]
                .into_iter()
                .flatten()
                .any(|iterations| iterations > plan.maximum_iterations())
        }) {
            return Some("cost.iteration-forecast-exceeds-maximum");
        }
        if self
            .required_reduction
            .is_some_and(|required| plan.reduction() != required)
        {
            Some("profile.required-reduction-mismatch")
        } else if plan.preconditioner() == PreconditionerPolicy::Jacobi
            && self.facts.complete_diagonal != Some(true)
        {
            Some("profile.complete-diagonal-required")
        } else {
            None
        }
    }

    /// Reauthenticate every claimed fact against the actual canonical problem.
    /// An unclaimed diagonal remains unconstrained; known facts must match exactly.
    /// These structural facts are shared by real and complex coefficients; exact
    /// scalar-domain and precision admission remains the selected backend's responsibility.
    ///
    /// # Errors
    /// Returns `EQ0807` before numerical work when a claimed fact differs.
    pub fn require_problem<
        S: eqiora_core::Scalar + num_complex::ComplexFloat<Real = f64> + Sync,
    >(
        &self,
        problem: &LinearProblem<'_, S>,
    ) -> Result<(), Diagnostic> {
        let actual = PlanningProfileFacts::from_problem(problem);
        let mut claimed = self.facts;
        if claimed.complete_diagonal.is_none() {
            claimed.complete_diagonal = actual.complete_diagonal;
        }
        if claimed != actual {
            return Err(invalid_profile(claimed, actual));
        }
        Ok(())
    }
}

/// One untrusted member of the frozen host-serial solver catalog.
#[derive(Debug, Clone, Copy)]
struct HostSerialSolverCandidate<'backend, S = f64> {
    id: &'static str,
    evidence_case: &'static str,
    plan: SolverPlan,
    backend: Option<&'backend dyn crate::LinearSolverBackend<S>>,
}

impl<'backend, S: Scalar + num_complex::ComplexFloat<Real = f64> + Sync>
    HostSerialSolverCandidate<'backend, S>
{
    #[cfg(test)]
    const fn new(
        id: &'static str,
        evidence_case: &'static str,
        request: LinearSolveRequest<'backend, S>,
    ) -> Self {
        Self::planned(id, evidence_case, request.plan(), Some(request.backend()))
    }

    const fn planned(
        id: &'static str,
        evidence_case: &'static str,
        plan: SolverPlan,
        backend: Option<&'backend dyn crate::LinearSolverBackend<S>>,
    ) -> Self {
        Self {
            id,
            evidence_case,
            plan,
            backend,
        }
    }

    const fn plan(self) -> SolverPlan {
        self.plan
    }

    /// Frozen catalog identity supplied by the caller.
    #[must_use]
    const fn id(self) -> &'static str {
        self.id
    }

    /// Registered evidence identity supplied by the caller.
    #[must_use]
    const fn evidence_case(self) -> &'static str {
        self.evidence_case
    }

    /// Exact backend and solver plan supplied by the caller.
    #[must_use]
    const fn request(self) -> LinearSolveRequest<'backend, S> {
        LinearSolveRequest::new(
            self.backend
                .expect("admitted candidate has a typed backend"),
            self.plan,
        )
    }
}

/// One inspected decision bound to the exact problem against which it resolved.
#[cfg(test)]
#[derive(Debug)]
struct HostSerialSolverDecision<'problem, 'backend> {
    problem: &'problem LinearProblem<'problem>,
    objective: SolverPlanningObjective,
    selected: HostSerialSolverCandidate<'backend>,
    solver_provider: SolverProvider,
    reasons: Vec<(&'static str, &'static str)>,
}

#[cfg(test)]
impl<'problem, 'backend> HostSerialSolverDecision<'problem, 'backend> {
    /// Frozen objective used to rank the admitted candidates.
    #[must_use]
    const fn objective(&self) -> SolverPlanningObjective {
        self.objective
    }

    /// Versioned deterministic planning-policy identity.
    #[must_use]
    const fn policy_id(&self) -> &'static str {
        POLICY_ID
    }

    /// Exact selected catalog member.
    #[must_use]
    const fn selected(&self) -> HostSerialSolverCandidate<'backend> {
        self.selected
    }

    /// Exact problem borrowed during resolution.
    #[must_use]
    const fn problem(&self) -> &'problem LinearProblem<'problem> {
        self.problem
    }

    /// Exact selected solver provider release.
    #[must_use]
    const fn solver_provider(&self) -> SolverProvider {
        self.solver_provider
    }

    /// Frozen host-serial execution provider.
    #[must_use]
    const fn execution_provider(&self) -> ExecutionProvider {
        SERIAL_EXECUTION_PROVIDER
    }

    /// Stable candidate-ID/reason-code trace in ascending candidate-ID order.
    fn reasons(&self) -> impl ExactSizeIterator<Item = (&'static str, &'static str)> + '_ {
        self.reasons.iter().copied()
    }

    /// Execute exactly the selected request against the resolved problem.
    ///
    /// # Errors
    /// Returns the selected backend's capability or numerical diagnostic. No
    /// retry, fallback, plan mutation, or problem substitution is performed.
    fn solve(&self) -> Result<LinearSolution, Diagnostic> {
        self.selected.request().solve(self.problem)
    }
}

/// Exact executable plan selected from the v3 host-serial catalog using
/// admitted operator facts.
#[derive(Debug)]
pub struct ResolvedHostSerialSolverPlan<'backend, S = f64> {
    profile: HostSerialSolverProfile,
    objective: SolverPlanningObjective,
    selected: HostSerialSolverCandidate<'backend, S>,
    solver_provider: SolverProvider,
    reasons: Vec<(&'static str, &'static str)>,
    costs: Vec<SolverCandidateCost>,
}

impl<'backend, S: Scalar + num_complex::ComplexFloat<Real = f64> + Sync>
    ResolvedHostSerialSolverPlan<'backend, S>
{
    /// Plan one exact executable candidate from the frozen v3 host-serial catalog
    /// using structural operator facts and caller-owned convergence controls.
    ///
    /// # Errors
    /// Returns `EQ0807` when controls, provider identity, the exact capability
    /// tuples, or the structural profile fail admission. Planning performs no
    /// numerical operator action and executes no backend. An absent optional
    /// provider rejects its catalog members before ranking; reference candidates
    /// remain independently eligible.
    pub fn resolve(
        profile: HostSerialSolverProfile,
        objective: SolverPlanningObjective,
        relative_tolerance: f64,
        absolute_tolerance: f64,
        maximum_iterations: NonZeroUsize,
        reference_backend: &'backend dyn crate::LinearSolverBackend<S>,
        faer_backend: Option<&'backend dyn crate::LinearSolverBackend<S>>,
    ) -> Result<Self, Diagnostic> {
        let candidates = catalog_ids(S::DOMAIN, profile.facts.properties)
            .iter()
            .map(|id| {
                let expected = expected_candidate(id);
                let backend = if expected.provider == REFERENCE_PROVIDER {
                    Some(reference_backend)
                } else {
                    faer_backend
                };
                Ok(HostSerialSolverCandidate::planned(
                    id,
                    expected.evidence_case,
                    catalog_plan(
                        expected.algorithm,
                        expected.preconditioner,
                        expected.reduction,
                        relative_tolerance,
                        absolute_tolerance,
                        maximum_iterations,
                    )?,
                    backend,
                ))
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        let resolved = resolve_candidates(profile.clone(), objective, &candidates)?;
        Ok(Self {
            profile,
            objective,
            selected: resolved.selected,
            solver_provider: resolved.solver_provider,
            reasons: resolved.reasons,
            costs: resolved.costs,
        })
    }

    /// Mathematical and structural assertions admitted before execution.
    #[must_use]
    pub fn profile(&self) -> HostSerialSolverProfile {
        self.profile.clone()
    }

    /// Frozen objective used to rank admitted candidates.
    #[must_use]
    pub const fn objective(&self) -> SolverPlanningObjective {
        self.objective
    }

    /// Versioned deterministic planning-policy identity.
    #[must_use]
    pub const fn policy_id(&self) -> &'static str {
        POLICY_ID
    }

    /// Exact selected catalog-member identity.
    #[must_use]
    pub const fn selected_candidate_id(&self) -> &'static str {
        self.selected.id()
    }

    /// Registered evidence identity attached to the selected catalog member.
    #[must_use]
    pub const fn selected_evidence_case(&self) -> &'static str {
        self.selected.evidence_case()
    }

    /// Exact selected solver plan.
    #[must_use]
    pub const fn solver_plan(&self) -> SolverPlan {
        self.selected.plan()
    }

    /// Exact selected solver provider release.
    #[must_use]
    pub const fn solver_provider(&self) -> SolverProvider {
        self.solver_provider
    }

    /// Frozen host-serial execution provider.
    #[must_use]
    pub const fn execution_provider(&self) -> ExecutionProvider {
        SERIAL_EXECUTION_PROVIDER
    }

    /// Stable candidate-ID/reason-code trace in ascending candidate-ID order.
    pub fn reasons(&self) -> impl ExactSizeIterator<Item = (&'static str, &'static str)> + '_ {
        self.reasons.iter().copied()
    }

    /// Execute exactly the selected request after reauthenticating the actual
    /// problem against the structural profile used during planning.
    ///
    /// # Errors
    /// Returns a profile diagnostic before backend work, or the selected
    /// backend's capability/numerical diagnostic. No retry or fallback occurs.
    pub fn solve(
        &self,
        problem: &LinearProblem<'_, S>,
        structure: Option<&crate::AlgebraicStructure>,
    ) -> Result<LinearSolution<S>, Diagnostic> {
        self.profile.require_structure(structure)?;
        self.profile.require_plan(self.solver_plan())?;
        self.profile.require_problem(problem)?;
        self.profile.require_problem_resources(
            self.solver_plan(),
            self.solver_provider(),
            problem,
        )?;
        self.selected.request().solve(problem)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PlanningProfileFacts {
    properties: LinearOperatorProperties,
    orientation: LinearOperatorOrientation,
    canonical_csr: bool,
    complete_diagonal: Option<bool>,
}

impl PlanningProfileFacts {
    const GENERAL_CANONICAL_CSR: Self = Self {
        properties: LinearOperatorProperties::General,
        orientation: LinearOperatorOrientation::Normal,
        canonical_csr: true,
        complete_diagonal: Some(true),
    };

    fn from_problem<S: eqiora_core::Scalar + num_complex::ComplexFloat<Real = f64> + Sync>(
        problem: &LinearProblem<'_, S>,
    ) -> Self {
        let system = problem.canonical_csr_system();
        Self {
            properties: problem.properties(),
            orientation: problem.operator().orientation(),
            canonical_csr: system.is_some(),
            complete_diagonal: Some(system.is_some_and(|system| {
                has_complete_diagonal(system.rows(), system.row_offsets(), system.column_indices())
            })),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct CandidateEvaluation<'backend, S = f64> {
    candidate: HostSerialSolverCandidate<'backend, S>,
    rejection: Option<&'static str>,
    cost: Option<SolverCandidateCost>,
}

#[derive(Debug)]
struct ResolvedCandidateSet<'backend, S = f64> {
    selected: HostSerialSolverCandidate<'backend, S>,
    solver_provider: SolverProvider,
    reasons: Vec<(&'static str, &'static str)>,
    costs: Vec<SolverCandidateCost>,
}

fn catalog_plan(
    algorithm: LinearSolver,
    preconditioner: PreconditionerPolicy,
    reduction: ReductionPolicy,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    maximum_iterations: NonZeroUsize,
) -> Result<SolverPlan, Diagnostic> {
    SolverPlan::new(
        algorithm,
        relative_tolerance,
        absolute_tolerance,
        maximum_iterations,
    )
    .map(|plan| {
        plan.with_preconditioner(preconditioner)
            .with_reduction(reduction)
    })
}

/// Resolve one exact candidate from the frozen v3 host-serial catalog.
///
/// # Errors
/// Returns `EQ0807` when inventory, common controls, catalog identity, problem
/// profile, or exact backend capability admission fails. Resolution performs
/// no numerical operator action and executes no backend.
#[cfg(test)]
fn resolve_host_serial_solver_v3<'problem, 'backend>(
    problem: &'problem LinearProblem<'problem>,
    objective: SolverPlanningObjective,
    candidates: &[HostSerialSolverCandidate<'backend>],
) -> Result<HostSerialSolverDecision<'problem, 'backend>, Diagnostic> {
    let resolved = resolve_candidates(
        HostSerialSolverProfile {
            facts: PlanningProfileFacts::from_problem(problem),
            required_reduction: None,
            structure: None,
            size: None,
            cost_request: None,
        },
        objective,
        candidates,
    )?;
    Ok(HostSerialSolverDecision {
        problem,
        objective,
        selected: resolved.selected,
        solver_provider: resolved.solver_provider,
        reasons: resolved.reasons,
    })
}

fn resolve_candidates<'backend, S: Scalar + num_complex::ComplexFloat<Real = f64> + Sync>(
    profile: HostSerialSolverProfile,
    objective: SolverPlanningObjective,
    candidates: &[HostSerialSolverCandidate<'backend, S>],
) -> Result<ResolvedCandidateSet<'backend, S>, Diagnostic> {
    validate_inventory(catalog_ids(S::DOMAIN, profile.facts.properties), candidates)?;
    validate_common_controls(candidates)?;

    let mut ordered = candidates.to_vec();
    ordered.sort_by_key(|candidate| candidate.id());
    let evaluations = ordered
        .into_iter()
        .map(|candidate| {
            let rejection = rejection_reason(&profile, candidate);
            let cost = rejection.is_none().then(|| {
                profile.estimate_cost(
                    candidate.id(),
                    candidate.plan(),
                    candidate.backend.expect("admitted provider").provider(),
                    S::DOMAIN,
                    S::STORAGE,
                )
            });
            CandidateEvaluation {
                candidate,
                rejection: rejection.or_else(|| {
                    cost.and_then(|estimate| {
                        cost::resource_rejection(estimate, profile.cost_request.unwrap_or_default())
                    })
                }),
                cost,
            }
        })
        .collect::<Vec<_>>();

    let mut admitted = evaluations
        .iter()
        .filter(|evaluation| evaluation.rejection.is_none())
        .map(|evaluation| evaluation.candidate)
        .collect::<Vec<_>>();
    if admitted.is_empty() {
        let trace = evaluations
            .iter()
            .map(|evaluation| {
                (
                    evaluation.candidate.id(),
                    evaluation
                        .rejection
                        .expect("a no-admitted trace contains only rejected candidates"),
                )
            })
            .collect::<Vec<_>>();
        return Err(no_admitted(trace.as_slice()));
    }

    admitted.sort_by_key(|candidate| {
        let estimate = evaluations
            .iter()
            .find(|evaluation| evaluation.candidate.id() == candidate.id())
            .and_then(|evaluation| evaluation.cost);
        cost::rank(objective, *candidate, profile.size, estimate)
    });
    let selected = admitted[0];
    let solver_provider = selected
        .backend
        .expect("selected candidate has a typed backend")
        .provider();
    let selected_id = selected.id();
    let selected_reason = selected_reason(objective);
    let mut reasons = Vec::with_capacity(evaluations.len() * 2);
    for evaluation in &evaluations {
        if let Some(rejection) = evaluation.rejection {
            reasons.push((evaluation.candidate.id(), rejection));
        } else {
            reasons.push((evaluation.candidate.id(), "candidate.admitted"));
            cost::append_reasons(&mut reasons, &profile, evaluation.candidate);
            reasons.push((
                evaluation.candidate.id(),
                if evaluation.candidate.id() == selected_id {
                    selected_reason
                } else {
                    "candidate.not-selected"
                },
            ));
        }
    }

    Ok(ResolvedCandidateSet {
        selected,
        solver_provider,
        reasons,
        costs: evaluations
            .iter()
            .filter_map(|evaluation| evaluation.cost)
            .collect(),
    })
}

fn catalog_ids(
    domain: ScalarDomain,
    properties: LinearOperatorProperties,
) -> &'static [&'static str] {
    match (domain, properties) {
        (ScalarDomain::Real, LinearOperatorProperties::General) => &[
            REFERENCE_ID,
            REFERENCE_IDENTITY_ID,
            FAER_BICGSTAB_ID,
            FAER_IDENTITY_ID,
            FAER_SPARSE_LU_ID,
        ],
        (ScalarDomain::Real, LinearOperatorProperties::SymmetricPositiveDefinite) => &[
            REFERENCE_CG_ID,
            REFERENCE_CG_JACOBI_ID,
            REFERENCE_SPD_MINRES_ID,
            REFERENCE_SPD_BICGSTAB_ID,
            REFERENCE_SPD_BICGSTAB_JACOBI_ID,
            FAER_CG_IDENTITY_ID,
            FAER_CG_ID,
            FAER_SPD_BICGSTAB_ID,
            FAER_SPD_BICGSTAB_JACOBI_ID,
            FAER_SPD_LU_ID,
        ],
        (ScalarDomain::Real, LinearOperatorProperties::SymmetricIndefinite) => &[
            REFERENCE_MINRES_ID,
            REFERENCE_INDEFINITE_BICGSTAB_ID,
            REFERENCE_INDEFINITE_BICGSTAB_JACOBI_ID,
            FAER_INDEFINITE_LU_ID,
        ],
        (ScalarDomain::Real, LinearOperatorProperties::Symmetric) => &[FAER_SYMMETRIC_LU_ID],
        (ScalarDomain::Complex, LinearOperatorProperties::General) => {
            &[REFERENCE_COMPLEX_ID, FAER_COMPLEX_LU_ID]
        }
        (ScalarDomain::Complex, LinearOperatorProperties::ComplexSymmetric) => {
            &[FAER_COMPLEX_SYMMETRIC_LU_ID]
        }
        (ScalarDomain::Complex, LinearOperatorProperties::Hermitian) => &[FAER_HERMITIAN_LU_ID],
        (ScalarDomain::Complex, LinearOperatorProperties::HermitianPositiveDefinite) => {
            &[REFERENCE_HPD_ID, FAER_HPD_LU_ID]
        }
        _ => &[],
    }
}

fn validate_inventory<S: Scalar + num_complex::ComplexFloat<Real = f64> + Sync>(
    ids: &[&str],
    candidates: &[HostSerialSolverCandidate<'_, S>],
) -> Result<(), Diagnostic> {
    // The requested domain and mathematical class own the inventory. Candidate
    // IDs cannot reinterpret either fact, including a complete foreign catalog.
    if ids.is_empty() {
        return Err(invalid_catalog("catalog.unsupported-profile"));
    }
    if ids
        .iter()
        .any(|id| !candidates.iter().any(|candidate| candidate.id() == *id))
    {
        return Err(invalid_catalog("catalog.missing-id"));
    }
    if ids.iter().any(|id| {
        candidates
            .iter()
            .filter(|candidate| candidate.id() == *id)
            .count()
            > 1
    }) {
        return Err(invalid_catalog("catalog.duplicate-id"));
    }
    if candidates
        .iter()
        .any(|candidate| !ids.contains(&candidate.id()))
    {
        return Err(invalid_catalog("catalog.unknown-id"));
    }
    Ok(())
}

fn validate_common_controls<S: Scalar + num_complex::ComplexFloat<Real = f64> + Sync>(
    candidates: &[HostSerialSolverCandidate<'_, S>],
) -> Result<(), Diagnostic> {
    let first = candidates
        .first()
        .expect("inventory validation requires a complete nonempty catalog")
        .plan();
    let controls_match = candidates.iter().all(|candidate| {
        let plan = candidate.plan();
        plan.relative_tolerance().to_bits() == first.relative_tolerance().to_bits()
            && plan.absolute_tolerance().to_bits() == first.absolute_tolerance().to_bits()
            && plan.maximum_iterations() == first.maximum_iterations()
    });
    if !controls_match {
        return Err(invalid_catalog("catalog.control-mismatch"));
    }
    Ok(())
}

fn has_complete_diagonal(rows: usize, row_offsets: &[usize], column_indices: &[usize]) -> bool {
    (0..rows).all(|row| {
        let start = row_offsets[row];
        let end = row_offsets[row + 1];
        column_indices[start..end].contains(&row)
    })
}

fn rank_key<S: Scalar + num_complex::ComplexFloat<Real = f64> + Sync>(
    objective: SolverPlanningObjective,
    candidate: HostSerialSolverCandidate<'_, S>,
) -> (u8, u8, &'static str) {
    let plan = candidate.plan();
    match objective {
        SolverPlanningObjective::Robust => (
            u8::from(plan.reduction() != ReductionPolicy::Reproducible),
            0,
            candidate.id(),
        ),
        SolverPlanningObjective::Fast => (
            u8::from(plan.reduction() != ReductionPolicy::Fast),
            u8::from(plan.algorithm() != LinearSolver::SparseLu),
            candidate.id(),
        ),
        SolverPlanningObjective::LowMemory => (
            u8::from(plan.algorithm() == LinearSolver::SparseLu),
            0,
            candidate.id(),
        ),
    }
}

const fn selected_reason(objective: SolverPlanningObjective) -> &'static str {
    match objective {
        SolverPlanningObjective::Robust => "candidate.selected.robust-preference",
        SolverPlanningObjective::Fast => "candidate.selected.fast-preference",
        SolverPlanningObjective::LowMemory => "candidate.selected.low-memory-preference",
    }
}

fn invalid_catalog(fragment: &'static str) -> Diagnostic {
    Diagnostic::error(
        codes::INVALID_REALIZATION,
        format!("{POLICY_ID} rejected catalog: {fragment}"),
    )
}

fn invalid_profile(expected: PlanningProfileFacts, actual: PlanningProfileFacts) -> Diagnostic {
    let reason = if actual.properties != expected.properties {
        "profile.operator-properties-mismatch"
    } else if actual.orientation != expected.orientation {
        "profile.orientation-mismatch"
    } else if actual.canonical_csr != expected.canonical_csr {
        "profile.canonical-csr-mismatch"
    } else {
        "profile.diagonal-availability-mismatch"
    };
    Diagnostic::error(
        codes::INVALID_REALIZATION,
        format!("{POLICY_ID} rejected execution problem: {reason}"),
    )
}

fn no_admitted(trace: &[(&str, &str)]) -> Diagnostic {
    let rendered = trace
        .iter()
        .map(|(candidate_id, reason)| format!("{candidate_id}={reason}"))
        .collect::<Vec<_>>()
        .join(",");
    Diagnostic::error(
        codes::INVALID_REALIZATION,
        format!("{POLICY_ID} no admitted candidate; trace=[{rendered}]"),
    )
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod complex_tests;
