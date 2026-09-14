use std::num::NonZeroUsize;

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;

use crate::{
    BackendId, ExecutionProvider, LinearOperatorOrientation, LinearOperatorProperties,
    LinearProblem, LinearSolution, LinearSolveRequest, LinearSolver, PreconditionerPolicy,
    ProviderLibrary, ReductionPolicy, SERIAL_EXECUTION_PROVIDER, ScalarType, SolverCapability,
    SolverPlan, SolverProvider,
};

const POLICY_ID: &str = "eqiora.host-serial-solver-planning/v2";

const REFERENCE_ID: &str = "eqiora.reference.bicgstab-general-jacobi-reproducible-f64";
const FAER_BICGSTAB_ID: &str = "eqiora.faer.bicgstab-general-jacobi-fast-f64";
const FAER_SPARSE_LU_ID: &str = "eqiora.faer.sparse-lu-general-identity-fast-f64";

const REFERENCE_CG_ID: &str = "eqiora.reference.cg-spd-identity-reproducible-f64";
const FAER_CG_ID: &str = "eqiora.faer.cg-spd-jacobi-fast-f64";
const FAER_SPD_LU_ID: &str = "eqiora.faer.sparse-lu-spd-identity-fast-f64";
const REFERENCE_MINRES_ID: &str = "eqiora.reference.minres-indefinite-identity-reproducible-f64";
const FAER_INDEFINITE_LU_ID: &str = "eqiora.faer.sparse-lu-indefinite-identity-fast-f64";

const REFERENCE_EVIDENCE: &str = "fluid.cartesian-advection-diffusion-fvm-2d";
const FAER_EVIDENCE: &str = "numerics.linear-backends";

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
        }
    }

    /// Describe a normal-orientation canonical CSR f64 operator using exact
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
            && self.facts.properties == LinearOperatorProperties::SymmetricPositiveDefinite
        {
            return Err(Diagnostic::error(
                codes::INVALID_REALIZATION,
                "profile.gauge-multiplier-incompatible-with-spd",
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
    ///
    /// # Errors
    /// Returns `EQ0807` before numerical work when a claimed fact differs.
    pub fn require_problem(&self, problem: &LinearProblem<'_>) -> Result<(), Diagnostic> {
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
struct HostSerialSolverCandidate<'backend> {
    id: &'static str,
    evidence_case: &'static str,
    request: LinearSolveRequest<'backend>,
}

impl<'backend> HostSerialSolverCandidate<'backend> {
    /// Bind a candidate identity and evidence claim to an executable request.
    ///
    /// Construction performs no admission. The shared catalog resolver
    /// validates the complete catalog before ranking.
    #[must_use]
    const fn new(
        id: &'static str,
        evidence_case: &'static str,
        request: LinearSolveRequest<'backend>,
    ) -> Self {
        Self {
            id,
            evidence_case,
            request,
        }
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
    const fn request(self) -> LinearSolveRequest<'backend> {
        self.request
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
        self.selected.request.solve(self.problem)
    }
}

/// Exact executable plan selected from the v2 host-serial catalog before
/// numerical operator construction.
#[derive(Debug)]
pub struct ResolvedHostSerialSolverPlan<'backend> {
    profile: HostSerialSolverProfile,
    objective: SolverPlanningObjective,
    selected: HostSerialSolverCandidate<'backend>,
    solver_provider: SolverProvider,
    reasons: Vec<(&'static str, &'static str)>,
}

impl<'backend> ResolvedHostSerialSolverPlan<'backend> {
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
        self.selected.request().plan()
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
        problem: &LinearProblem<'_>,
        structure: Option<&crate::AlgebraicStructure>,
    ) -> Result<LinearSolution, Diagnostic> {
        self.profile.require_structure(structure)?;
        self.profile.require_plan(self.solver_plan())?;
        self.profile.require_problem(problem)?;
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

    fn from_problem(problem: &LinearProblem<'_>) -> Self {
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
struct CandidateEvaluation<'backend> {
    candidate: HostSerialSolverCandidate<'backend>,
    rejection: Option<&'static str>,
}

#[derive(Debug)]
struct ResolvedCandidateSet<'backend> {
    selected: HostSerialSolverCandidate<'backend>,
    solver_provider: SolverProvider,
    reasons: Vec<(&'static str, &'static str)>,
}

/// Plan one exact executable candidate from the frozen v2 host-serial catalog
/// using structural operator facts and caller-owned convergence controls.
///
/// # Errors
/// Returns `EQ0807` when controls, provider identity, the exact capability
/// tuples, or the structural profile fail admission. Planning performs no
/// numerical operator action and executes no backend.
pub fn plan_host_serial_solver_v2<'backend>(
    profile: HostSerialSolverProfile,
    objective: SolverPlanningObjective,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    maximum_iterations: NonZeroUsize,
    reference_backend: &'backend dyn crate::LinearSolverBackend,
    faer_backend: &'backend dyn crate::LinearSolverBackend,
) -> Result<ResolvedHostSerialSolverPlan<'backend>, Diagnostic> {
    let candidates = catalog_ids(profile.facts.properties)
        .iter()
        .map(|id| {
            let expected = expected_candidate(id);
            let backend = if expected.provider == REFERENCE_PROVIDER {
                reference_backend
            } else {
                faer_backend
            };
            Ok(HostSerialSolverCandidate::new(
                id,
                expected.evidence_case,
                LinearSolveRequest::new(
                    backend,
                    catalog_plan(
                        expected.algorithm,
                        expected.preconditioner,
                        expected.reduction,
                        relative_tolerance,
                        absolute_tolerance,
                        maximum_iterations,
                    )?,
                ),
            ))
        })
        .collect::<Result<Vec<_>, Diagnostic>>()?;
    let resolved = resolve_candidates(profile.clone(), objective, &candidates)?;
    Ok(ResolvedHostSerialSolverPlan {
        profile,
        objective,
        selected: resolved.selected,
        solver_provider: resolved.solver_provider,
        reasons: resolved.reasons,
    })
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

/// Resolve one exact candidate from the frozen v2 host-serial catalog.
///
/// # Errors
/// Returns `EQ0807` when inventory, common controls, catalog identity, problem
/// profile, or exact backend capability admission fails. Resolution performs
/// no numerical operator action and executes no backend.
#[cfg(test)]
fn resolve_host_serial_solver_v2<'problem, 'backend>(
    problem: &'problem LinearProblem<'problem>,
    objective: SolverPlanningObjective,
    candidates: &[HostSerialSolverCandidate<'backend>],
) -> Result<HostSerialSolverDecision<'problem, 'backend>, Diagnostic> {
    let resolved = resolve_candidates(
        HostSerialSolverProfile {
            facts: PlanningProfileFacts::from_problem(problem),
            required_reduction: None,
            structure: None,
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

fn resolve_candidates<'backend>(
    profile: HostSerialSolverProfile,
    objective: SolverPlanningObjective,
    candidates: &[HostSerialSolverCandidate<'backend>],
) -> Result<ResolvedCandidateSet<'backend>, Diagnostic> {
    validate_inventory(candidates)?;
    validate_common_controls(candidates)?;

    let mut ordered = candidates.to_vec();
    ordered.sort_by_key(|candidate| candidate.id());
    let evaluations = ordered
        .into_iter()
        .map(|candidate| CandidateEvaluation {
            candidate,
            rejection: rejection_reason(&profile, candidate),
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

    admitted.sort_by_key(|candidate| rank_key(objective, *candidate));
    let selected = admitted[0];
    let solver_provider = selected.request().backend().provider();
    let selected_id = selected.id();
    let selected_reason = selected_reason(objective);
    let mut reasons = Vec::with_capacity(evaluations.len() * 2);
    for evaluation in &evaluations {
        if let Some(rejection) = evaluation.rejection {
            reasons.push((evaluation.candidate.id(), rejection));
        } else {
            reasons.push((evaluation.candidate.id(), "candidate.admitted"));
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
    })
}

fn catalog_ids(properties: LinearOperatorProperties) -> &'static [&'static str] {
    match properties {
        LinearOperatorProperties::General => &[REFERENCE_ID, FAER_BICGSTAB_ID, FAER_SPARSE_LU_ID],
        LinearOperatorProperties::SymmetricPositiveDefinite => {
            &[REFERENCE_CG_ID, FAER_CG_ID, FAER_SPD_LU_ID]
        }
        LinearOperatorProperties::SymmetricIndefinite => {
            &[REFERENCE_MINRES_ID, FAER_INDEFINITE_LU_ID]
        }
    }
}

fn validate_inventory(candidates: &[HostSerialSolverCandidate<'_>]) -> Result<(), Diagnostic> {
    // A catalog is complete for one exact operator class. Mixed-class catalogs
    // cannot be used to reinterpret the supplied mathematical properties.
    let ids = if candidates
        .iter()
        .any(|c| catalog_ids(LinearOperatorProperties::General).contains(&c.id()))
    {
        catalog_ids(LinearOperatorProperties::General)
    } else if candidates
        .iter()
        .any(|c| catalog_ids(LinearOperatorProperties::SymmetricPositiveDefinite).contains(&c.id()))
    {
        catalog_ids(LinearOperatorProperties::SymmetricPositiveDefinite)
    } else if candidates
        .iter()
        .any(|c| catalog_ids(LinearOperatorProperties::SymmetricIndefinite).contains(&c.id()))
    {
        catalog_ids(LinearOperatorProperties::SymmetricIndefinite)
    } else {
        return Err(invalid_catalog("catalog.missing-id"));
    };
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

fn validate_common_controls(
    candidates: &[HostSerialSolverCandidate<'_>],
) -> Result<(), Diagnostic> {
    let first = candidates
        .first()
        .expect("inventory validation requires a complete nonempty catalog")
        .request()
        .plan();
    let controls_match = candidates.iter().all(|candidate| {
        let plan = candidate.request().plan();
        plan.relative_tolerance().to_bits() == first.relative_tolerance().to_bits()
            && plan.absolute_tolerance().to_bits() == first.absolute_tolerance().to_bits()
            && plan.maximum_iterations() == first.maximum_iterations()
    });
    if !controls_match {
        return Err(invalid_catalog("catalog.control-mismatch"));
    }
    Ok(())
}

fn rejection_reason(
    profile: &HostSerialSolverProfile,
    candidate: HostSerialSolverCandidate<'_>,
) -> Option<&'static str> {
    let expected = expected_candidate(candidate.id());
    if candidate.evidence_case() != expected.evidence_case {
        return Some("catalog.evidence-mismatch");
    }
    if candidate.request().backend().provider() != expected.provider {
        return Some("catalog.provider-mismatch");
    }
    if !plan_tuple_matches(candidate.request().plan(), expected) {
        return Some("catalog.plan-mismatch");
    }
    if profile.facts.properties != expected.properties {
        return Some(match expected.properties {
            LinearOperatorProperties::General => "profile.general-required",
            LinearOperatorProperties::SymmetricPositiveDefinite => "profile.spd-required",
            LinearOperatorProperties::SymmetricIndefinite => {
                "profile.symmetric-indefinite-required"
            }
        });
    }
    if profile.facts.orientation != LinearOperatorOrientation::Normal {
        return Some("profile.normal-required");
    }
    if !profile.facts.canonical_csr {
        return Some("profile.canonical-csr-required");
    }
    if let Some(reason) = profile.plan_rejection(candidate.request().plan()) {
        return Some(reason);
    }
    let required = SolverCapability {
        algorithm: expected.algorithm,
        operator_properties: expected.properties,
        preconditioner: expected.preconditioner,
        reduction: expected.reduction,
        scalar_type: ScalarType::F64,
    };
    if !candidate
        .request()
        .backend()
        .capabilities()
        .combinations()
        .contains(&required)
    {
        return Some("capability.exact-tuple-required");
    }
    None
}

#[derive(Debug, Clone, Copy)]
struct ExpectedCandidate {
    properties: LinearOperatorProperties,
    evidence_case: &'static str,
    provider: SolverProvider,
    algorithm: LinearSolver,
    preconditioner: PreconditionerPolicy,
    reduction: ReductionPolicy,
}

fn expected_candidate(id: &str) -> ExpectedCandidate {
    match id {
        REFERENCE_ID => ExpectedCandidate {
            properties: LinearOperatorProperties::General,
            evidence_case: REFERENCE_EVIDENCE,
            provider: REFERENCE_PROVIDER,
            algorithm: LinearSolver::BiConjugateGradientStabilized,
            preconditioner: PreconditionerPolicy::Jacobi,
            reduction: ReductionPolicy::Reproducible,
        },
        FAER_BICGSTAB_ID => ExpectedCandidate {
            properties: LinearOperatorProperties::General,
            evidence_case: FAER_EVIDENCE,
            provider: FAER_PROVIDER,
            algorithm: LinearSolver::BiConjugateGradientStabilized,
            preconditioner: PreconditionerPolicy::Jacobi,
            reduction: ReductionPolicy::Fast,
        },
        FAER_SPARSE_LU_ID => ExpectedCandidate {
            properties: LinearOperatorProperties::General,
            evidence_case: FAER_EVIDENCE,
            provider: FAER_PROVIDER,
            algorithm: LinearSolver::SparseLu,
            preconditioner: PreconditionerPolicy::Identity,
            reduction: ReductionPolicy::Fast,
        },
        REFERENCE_CG_ID => ExpectedCandidate {
            properties: LinearOperatorProperties::SymmetricPositiveDefinite,
            evidence_case: FAER_EVIDENCE,
            provider: REFERENCE_PROVIDER,
            algorithm: LinearSolver::ConjugateGradient,
            preconditioner: PreconditionerPolicy::Identity,
            reduction: ReductionPolicy::Reproducible,
        },
        FAER_CG_ID => ExpectedCandidate {
            properties: LinearOperatorProperties::SymmetricPositiveDefinite,
            evidence_case: FAER_EVIDENCE,
            provider: FAER_PROVIDER,
            algorithm: LinearSolver::ConjugateGradient,
            preconditioner: PreconditionerPolicy::Jacobi,
            reduction: ReductionPolicy::Fast,
        },
        FAER_SPD_LU_ID => ExpectedCandidate {
            properties: LinearOperatorProperties::SymmetricPositiveDefinite,
            evidence_case: FAER_EVIDENCE,
            provider: FAER_PROVIDER,
            algorithm: LinearSolver::SparseLu,
            preconditioner: PreconditionerPolicy::Identity,
            reduction: ReductionPolicy::Fast,
        },
        REFERENCE_MINRES_ID => ExpectedCandidate {
            properties: LinearOperatorProperties::SymmetricIndefinite,
            evidence_case: "numerics.reference-minres-w64",
            provider: REFERENCE_PROVIDER,
            algorithm: LinearSolver::MinimumResidual,
            preconditioner: PreconditionerPolicy::Identity,
            reduction: ReductionPolicy::Reproducible,
        },
        FAER_INDEFINITE_LU_ID => ExpectedCandidate {
            properties: LinearOperatorProperties::SymmetricIndefinite,
            evidence_case: FAER_EVIDENCE,
            provider: FAER_PROVIDER,
            algorithm: LinearSolver::SparseLu,
            preconditioner: PreconditionerPolicy::Identity,
            reduction: ReductionPolicy::Fast,
        },
        _ => unreachable!("inventory validation rejects unknown candidate IDs"),
    }
}

fn plan_tuple_matches(plan: SolverPlan, expected: ExpectedCandidate) -> bool {
    plan.algorithm() == expected.algorithm
        && plan.preconditioner() == expected.preconditioner
        && plan.reduction() == expected.reduction
}

fn has_complete_diagonal(rows: usize, row_offsets: &[usize], column_indices: &[usize]) -> bool {
    (0..rows).all(|row| {
        let start = row_offsets[row];
        let end = row_offsets[row + 1];
        column_indices[start..end].contains(&row)
    })
}

fn rank_key(
    objective: SolverPlanningObjective,
    candidate: HostSerialSolverCandidate<'_>,
) -> (u8, u8, &'static str) {
    let plan = candidate.request().plan();
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
