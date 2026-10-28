//! Deterministic work proxies and explicitly scoped numeric payload accounting.
//! No estimate is elapsed time, an allocator/RSS limit, or permission to reuse state.
use super::*;

/// Known dimensions of the typed square solver system after constraints/elimination.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolverPlanningSize {
    rows: usize,
    stored_slots: usize,
    exact: bool,
}

impl SolverPlanningSize {
    /// Retain an assembly-owned exact stored-slot count.
    #[must_use]
    pub const fn exact(rows: usize, stored_slots: usize) -> Self {
        Self {
            rows,
            stored_slots,
            exact: true,
        }
    }
    /// A dense square bound derived from an already admitted coordinate count.
    #[must_use]
    pub fn dense_bound(rows: usize) -> Option<Self> {
        Some(Self {
            rows,
            stored_slots: rows.checked_mul(rows)?,
            exact: false,
        })
    }
    /// Number of typed solver coordinates.
    #[must_use]
    pub const fn rows(self) -> usize {
        self.rows
    }
    /// Exact stored slots or their explicit upper bound.
    #[must_use]
    pub const fn stored_slots(self) -> usize {
        self.stored_slots
    }
    /// Whether slots are exact rather than a conservative bound.
    #[must_use]
    pub const fn is_exact(self) -> bool {
        self.exact
    }
}

/// Explicit planning assumptions; these never authorize prepared-state reuse.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SolverCostRequest {
    solves: NonZeroUsize,
    structures: NonZeroUsize,
    numeric_preparations: NonZeroUsize,
    identity_iterations: Option<NonZeroUsize>,
    jacobi_iterations: Option<NonZeroUsize>,
    numeric_payload_limit: Option<NonZeroUsize>,
    declared_horizon: bool,
}

impl Default for SolverCostRequest {
    fn default() -> Self {
        Self {
            solves: NonZeroUsize::MIN,
            structures: NonZeroUsize::MIN,
            numeric_preparations: NonZeroUsize::MIN,
            identity_iterations: None,
            jacobi_iterations: None,
            numeric_payload_limit: None,
            declared_horizon: false,
        }
    }
}

impl SolverCostRequest {
    /// Declare a solve horizon with `1 <= structures <= numeric_preparations <= solves`.
    /// Counts are planning assumptions, not historical observations or reuse authority.
    pub fn new(
        solves: NonZeroUsize,
        structures: NonZeroUsize,
        numeric_preparations: NonZeroUsize,
    ) -> Result<Self, Diagnostic> {
        if structures > numeric_preparations || numeric_preparations > solves {
            return Err(Diagnostic::error(
                codes::INVALID_REALIZATION,
                "cost horizon requires structures <= numeric preparations <= solves",
            ));
        }
        Ok(Self {
            solves,
            structures,
            numeric_preparations,
            declared_horizon: true,
            ..Self::default()
        })
    }
    /// Supply separate iteration forecasts; omission uses the requested iteration ceiling.
    #[must_use]
    pub const fn with_iteration_forecasts(
        mut self,
        identity: Option<NonZeroUsize>,
        jacobi: Option<NonZeroUsize>,
    ) -> Self {
        self.identity_iterations = identity;
        self.jacobi_iterations = jacobi;
        self
    }
    /// Bound the accounted numeric payload, not process RSS or allocator overhead.
    /// Unknown implementation bounds cannot satisfy a requested limit.
    #[must_use]
    pub const fn with_numeric_payload_limit(mut self, bytes: NonZeroUsize) -> Self {
        self.numeric_payload_limit = Some(bytes);
        self
    }
    /// Declared or default one-shot number of solves.
    #[must_use]
    pub const fn solves(self) -> NonZeroUsize {
        self.solves
    }
    /// Number of structural preparations in the forecast.
    #[must_use]
    pub const fn structures(self) -> NonZeroUsize {
        self.structures
    }
    /// Number of numeric preparations in the forecast.
    #[must_use]
    pub const fn numeric_preparations(self) -> NonZeroUsize {
        self.numeric_preparations
    }
    /// Identity-preconditioned iteration forecast, absent when using the ceiling.
    #[must_use]
    pub const fn identity_iterations(self) -> Option<NonZeroUsize> {
        self.identity_iterations
    }
    /// Jacobi-preconditioned iteration forecast, absent when using the ceiling.
    #[must_use]
    pub const fn jacobi_iterations(self) -> Option<NonZeroUsize> {
        self.jacobi_iterations
    }
    /// Requested accounted payload limit.
    #[must_use]
    pub const fn numeric_payload_limit(self) -> Option<NonZeroUsize> {
        self.numeric_payload_limit
    }
    /// Whether the horizon was explicitly declared rather than assumed one-shot.
    #[must_use]
    pub const fn declared_horizon(self) -> bool {
        self.declared_horizon
    }
}

/// Auditable algorithmic work proxy for one exact candidate, never a timing prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolverCandidateCost {
    pub(super) candidate: &'static str,
    pub(super) setup_work: Option<u128>,
    pub(super) solve_work: Option<u128>,
    pub(super) acceptance_work: Option<u128>,
    pub(super) total_work: Option<u128>,
    pub(super) numeric_payload_bytes: Option<usize>,
}

impl SolverCandidateCost {
    /// Exact candidate identity, or `manual` for an exact caller request.
    #[must_use]
    pub const fn candidate(self) -> &'static str {
        self.candidate
    }
    /// Structural and numeric setup proxy over the declared horizon.
    #[must_use]
    pub const fn setup_work(self) -> Option<u128> {
        self.setup_work
    }
    /// Repeated solve proxy over the declared horizon.
    #[must_use]
    pub const fn solve_work(self) -> Option<u128> {
        self.solve_work
    }
    /// Independent true linear-residual acceptance proxy over the declared horizon.
    /// Physical method/State assessment is outside this typed-solver proxy.
    #[must_use]
    pub const fn acceptance_work(self) -> Option<u128> {
        self.acceptance_work
    }
    /// Total proxy; absent for unavailable facts or checked overflow.
    #[must_use]
    pub const fn total_work(self) -> Option<u128> {
        self.total_work
    }
    /// Accounted input CSR/RHS payload plus bounded solver-owned numeric workspace.
    /// Excludes caller guesses, assembly/physical assessment, outputs, metadata and allocator/RSS overhead.
    #[must_use]
    pub const fn numeric_payload_bytes(self) -> Option<usize> {
        self.numeric_payload_bytes
    }
}

pub(super) fn estimate(
    profile: &HostSerialSolverProfile,
    candidate: &'static str,
    plan: SolverPlan,
    provider: SolverProvider,
    scalar_domain: ScalarDomain,
    scalar_type: ScalarType,
) -> SolverCandidateCost {
    let size = profile.size;
    let request = profile.cost_request.unwrap_or_default();
    let orientation = profile.facts.orientation;
    let work = if unmodeled_realification(plan, provider, scalar_domain) {
        None
    } else {
        size.and_then(|size| work_proxy(plan, size, request))
    };
    SolverCandidateCost {
        candidate,
        setup_work: work.map(|value| value.0),
        solve_work: work.map(|value| value.1),
        acceptance_work: work.map(|value| value.2),
        total_work: work.and_then(|value| value.0.checked_add(value.1)?.checked_add(value.2)),
        numeric_payload_bytes: size.and_then(|size| {
            if provider != REFERENCE_PROVIDER {
                return None;
            }
            let workspace = crate::reference::canonical_csr_numeric_workspace_bytes(
                plan,
                scalar_domain,
                scalar_type,
                size.rows,
                orientation,
            )?;
            // Borrowed CSR/RHS account their admitted live numeric payload, not spare capacity.
            let indices = size
                .rows
                .checked_add(1)?
                .checked_add(size.stored_slots)?
                .checked_mul(size_of::<usize>())?;
            let values = size.rows.checked_add(size.stored_slots)?.checked_mul(8)?;
            workspace.checked_add(indices)?.checked_add(values)
        }),
    }
}

fn unmodeled_realification(
    plan: SolverPlan,
    provider: SolverProvider,
    domain: ScalarDomain,
) -> bool {
    domain == ScalarDomain::Complex
        && provider == REFERENCE_PROVIDER
        && matches!(
            plan.algorithm(),
            LinearSolver::ConjugateGradient | LinearSolver::MinimumResidual
        )
}

// Versioned algorithmic work units, not measured FLOPs or seconds. Sparse LU uses a
// deliberately labelled dense envelope, not a prediction of sparse fill/pivoting.
// Host transfer/synchronization is zero. Binding and requested output costs outside
// this typed linear solve are unknown, shared terms and are not a measured optimum.
fn work_proxy(
    plan: SolverPlan,
    size: SolverPlanningSize,
    request: SolverCostRequest,
) -> Option<(u128, u128, u128)> {
    let n = size.rows as u128;
    let z = size.stored_slots as u128;
    let h = request.solves.get() as u128;
    let s = request.structures.get() as u128;
    let f = request.numeric_preparations.get() as u128;
    let iterations = match plan.preconditioner() {
        PreconditionerPolicy::Identity => request.identity_iterations,
        PreconditionerPolicy::Jacobi => request.jacobi_iterations,
    }
    .unwrap_or(plan.maximum_iterations())
    .get() as u128;
    let diagonal = if plan.preconditioner() == PreconditionerPolicy::Jacobi {
        n
    } else {
        0
    };
    let structure = n.checked_add(z)?.checked_mul(s)?;
    let (numeric, solve) = match plan.algorithm() {
        LinearSolver::SparseLu => (
            n.checked_mul(n)?.checked_mul(n)?,
            n.checked_mul(n)?
                .checked_mul(2)?
                .checked_add(n.checked_mul(3)?)?,
        ),
        LinearSolver::ConjugateGradient => (
            diagonal,
            z.checked_mul(2)?
                .checked_add(n.checked_mul(8)?)?
                .checked_add(diagonal)?
                .checked_mul(iterations)?,
        ),
        LinearSolver::BiConjugateGradientStabilized => (
            diagonal,
            z.checked_mul(4)?
                .checked_add(n.checked_mul(20)?)?
                .checked_add(diagonal.checked_mul(2)?)?
                .checked_mul(iterations)?,
        ),
        LinearSolver::MinimumResidual => {
            let k = iterations.min(n);
            // Current Reference MINRES retains and twice reorthogonalizes the basis.
            let actions = z
                .checked_mul(2)?
                .checked_add(n.checked_mul(8)?)?
                .checked_mul(k)?;
            let basis = n
                .checked_mul(k)?
                .checked_mul(k.checked_add(1)?)?
                .checked_mul(4)?;
            (0, actions.checked_add(basis)?)
        }
    };
    let setup = structure.checked_add(numeric.checked_mul(f)?)?;
    let solve = solve.checked_mul(h)?;
    let acceptance = z
        .checked_mul(2)?
        .checked_add(n.checked_mul(5)?)?
        .checked_mul(h)?;
    Some((setup, solve, acceptance))
}

pub(super) fn resource_rejection(
    estimate: SolverCandidateCost,
    request: SolverCostRequest,
) -> Option<&'static str> {
    let limit = request.numeric_payload_limit?;
    match estimate.numeric_payload_bytes {
        None => Some("resource.numeric-payload-bound-unproved"),
        Some(bytes) if bytes > limit.get() => Some("resource.numeric-payload-limit-exceeded"),
        Some(_) => None,
    }
}

impl HostSerialSolverProfile {
    /// Bind counts derived by the existing assembly/layout owner, never by a ranker probe.
    #[must_use]
    pub const fn with_size(mut self, size: SolverPlanningSize) -> Self {
        self.size = Some(size);
        self
    }
    /// Retain explicit forecast assumptions and the scoped numeric payload budget.
    #[must_use]
    pub const fn with_cost_request(mut self, request: SolverCostRequest) -> Self {
        self.cost_request = Some(request);
        self
    }
    /// Exact counts, an explicit upper bound, or unknown.
    #[must_use]
    pub const fn size(&self) -> Option<SolverPlanningSize> {
        self.size
    }
    /// Explicit cost controls; absence means the labelled one-shot/iteration-ceiling assumption.
    #[must_use]
    pub const fn cost_request(&self) -> Option<SolverCostRequest> {
        self.cost_request
    }
    /// Estimate an already mathematically/provider-admitted candidate without executing it.
    #[must_use]
    pub fn estimate_cost(
        &self,
        candidate: &'static str,
        plan: SolverPlan,
        provider: SolverProvider,
        scalar_domain: ScalarDomain,
        scalar_type: ScalarType,
    ) -> SolverCandidateCost {
        estimate(self, candidate, plan, provider, scalar_domain, scalar_type)
    }
    /// Enforce an explicitly requested numeric payload limit before solver work.
    pub fn require_resources(
        &self,
        plan: SolverPlan,
        provider: SolverProvider,
        scalar_domain: ScalarDomain,
        scalar_type: ScalarType,
    ) -> Result<(), Diagnostic> {
        let estimate = self.estimate_cost("manual", plan, provider, scalar_domain, scalar_type);
        if let Some(reason) = resource_rejection(estimate, self.cost_request.unwrap_or_default()) {
            return Err(Diagnostic::error(codes::INVALID_REALIZATION, reason));
        }
        Ok(())
    }
    /// Recheck the retained payload budget against this actual system before numerical work.
    /// Decision-time size estimates remain immutable provenance; a new slot count alone is valid.
    pub fn require_problem_resources<S: Scalar + num_complex::ComplexFloat<Real = f64> + Sync>(
        &self,
        plan: SolverPlan,
        provider: SolverProvider,
        problem: &LinearProblem<'_, S>,
    ) -> Result<(), Diagnostic> {
        if self
            .cost_request
            .unwrap_or_default()
            .numeric_payload_limit()
            .is_none()
        {
            return Ok(());
        }
        let mut current = self.clone();
        current.facts.orientation = problem.operator().orientation();
        current.size = problem
            .canonical_csr_system()
            .map(|system| SolverPlanningSize::exact(system.rows(), system.column_indices().len()));
        current.require_resources(plan, provider, S::DOMAIN, S::STORAGE)
    }
}

impl<S: Scalar + num_complex::ComplexFloat<Real = f64> + Sync> ResolvedHostSerialSolverPlan<'_, S> {
    /// Cost estimates for mathematically/provider-admitted candidates, including resource rejects.
    #[must_use]
    pub fn costs(&self) -> &[SolverCandidateCost] {
        &self.costs
    }
}

pub(super) fn rank<S: Scalar + num_complex::ComplexFloat<Real = f64> + Sync>(
    objective: SolverPlanningObjective,
    candidate: HostSerialSolverCandidate<'_, S>,
    size: Option<SolverPlanningSize>,
    estimate: Option<SolverCandidateCost>,
) -> (u8, u128, u128, &'static str) {
    if size.is_none() {
        let previous = super::rank_key(objective, candidate);
        return (previous.0, u128::from(previous.1), 0, previous.2);
    }
    let work = estimate
        .and_then(|estimate| estimate.total_work)
        .unwrap_or(u128::MAX);
    let payload = estimate
        .and_then(|estimate| estimate.numeric_payload_bytes)
        .map(|bytes| bytes as u128)
        .unwrap_or(u128::MAX);
    match objective {
        SolverPlanningObjective::Robust => (
            u8::from(candidate.plan().reduction() != ReductionPolicy::Reproducible),
            work,
            payload,
            candidate.id(),
        ),
        SolverPlanningObjective::Fast => (
            u8::from(candidate.plan().reduction() != ReductionPolicy::Fast),
            work,
            payload,
            candidate.id(),
        ),
        SolverPlanningObjective::LowMemory => (
            if payload != u128::MAX {
                0
            } else if candidate.plan().algorithm() == LinearSolver::SparseLu {
                2
            } else {
                1
            },
            payload,
            work,
            candidate.id(),
        ),
    }
}

// Keep forecast limitations next to the candidate decision that uses them.
pub(super) fn append_reasons<S: Scalar + num_complex::ComplexFloat<Real = f64> + Sync>(
    reasons: &mut Vec<(&'static str, &'static str)>,
    profile: &HostSerialSolverProfile,
    candidate: HostSerialSolverCandidate<'_, S>,
) {
    let id = candidate.id();
    let Some(size) = profile.size else {
        reasons.push((id, "cost.shape-unavailable-preference-fallback"));
        return;
    };
    if candidate.backend.is_some_and(|backend| {
        unmodeled_realification(candidate.plan(), backend.provider(), S::DOMAIN)
    }) {
        reasons.push((id, "cost.reference-complex-realification-work-unmodeled"));
    }
    reasons.push((
        id,
        if size.exact {
            "cost.work-proxy-exact-stored-slots"
        } else {
            "cost.work-proxy-dense-slot-bound"
        },
    ));
    let request = profile.cost_request.unwrap_or_default();
    reasons.push((
        id,
        if request.declared_horizon {
            "cost.declared-horizon-not-reuse-authority"
        } else {
            "cost.one-shot-horizon-assumed"
        },
    ));
    if candidate.plan().algorithm() == LinearSolver::SparseLu {
        reasons.push((id, "cost.lu-dense-envelope-fill-unknown"));
    } else {
        let forecast = match candidate.plan().preconditioner() {
            PreconditionerPolicy::Identity => request.identity_iterations,
            PreconditionerPolicy::Jacobi => request.jacobi_iterations,
        };
        reasons.push((
            id,
            if forecast.is_some() {
                "cost.declared-iterations-conditioning-unproved"
            } else {
                "cost.iteration-ceiling-conditioning-unproved"
            },
        ));
    }
    reasons.push((id, "cost.host-transfer-zero-outputs-unmodeled"));
}

#[cfg(test)]
mod realification_cost_tests {
    use super::*;

    #[test]
    fn reference_complex_lowering_work_is_unknown_without_conversion_accounting() {
        let size = SolverPlanningSize::exact(2, 4);
        for (algorithm, real_total) in [
            (LinearSolver::ConjugateGradient, 72),
            (LinearSolver::MinimumResidual, 120),
        ] {
            let plan =
                SolverPlan::new(algorithm, 1e-12, 1e-14, NonZeroUsize::new(2).unwrap()).unwrap();
            let profile = HostSerialSolverProfile::general_canonical_csr().with_size(size);
            // Independently: real n=2,z=4,k=2 gives structure6, acceptance18,
            // CG recurrence48 or MINRES recurrence48+basis48. A complex input
            // instead lowers to4real unknowns/16slots, plus packing and original
            // typed acceptance: those included terms have no modeled bound here.
            let real = profile.estimate_cost(
                "manual",
                plan,
                REFERENCE_PROVIDER,
                ScalarDomain::Real,
                ScalarType::F64,
            );
            assert_eq!(real.total_work(), Some(real_total));
            let complex = profile.estimate_cost(
                "manual",
                plan,
                REFERENCE_PROVIDER,
                ScalarDomain::Complex,
                ScalarType::F64,
            );
            assert_eq!(
                (
                    complex.setup_work(),
                    complex.solve_work(),
                    complex.acceptance_work(),
                    complex.total_work()
                ),
                (None, None, None, None)
            );
        }
        let plan = SolverPlan::new(
            LinearSolver::BiConjugateGradientStabilized,
            1e-12,
            1e-14,
            NonZeroUsize::new(2).unwrap(),
        )
        .unwrap();
        let profile = HostSerialSolverProfile::general_canonical_csr().with_size(size);
        // Native complex BiCG does not realify: 6+2*(16+40)+18=136 typed units.
        assert_eq!(
            profile
                .estimate_cost(
                    "manual",
                    plan,
                    REFERENCE_PROVIDER,
                    ScalarDomain::Complex,
                    ScalarType::F64
                )
                .total_work(),
            Some(136)
        );
    }
}
