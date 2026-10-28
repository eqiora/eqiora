//! Independent cost arithmetic and actual-system resource admission controls.
use super::*;
use eqiora::solver::{
    LinearProblem, LinearSolution, ReplicatedLinearExecution, SolverCapabilities, SolverCapability,
    SolverCostRequest, SolverPlanningSize,
};
use eqiora_core::{ScalarDomain, ScalarType};
use std::sync::atomic::{AtomicUsize, Ordering};

fn nz(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).unwrap()
}
fn forecast(h: usize, s: usize, f: usize, identity: usize, jacobi: usize) -> SolverCostRequest {
    SolverCostRequest::new(nz(h), nz(s), nz(f))
        .unwrap()
        .with_iteration_forecasts(Some(nz(identity)), Some(nz(jacobi)))
}
fn profile(
    properties: LinearOperatorProperties,
    n: usize,
    z: usize,
    request: SolverCostRequest,
) -> SolverPlanningProfile {
    SolverPlanningProfile::canonical_csr(properties, Some(true), None)
        .with_size(SolverPlanningSize::exact(n, z))
        .with_cost_request(request)
}
fn choose(
    profile: SolverPlanningProfile,
    objective: SolverPlanningObjective,
    reference: &dyn LinearSolverBackend,
    faer: Option<&dyn LinearSolverBackend>,
) -> Result<SolverSelection, eqiora::Diagnostic> {
    SolverSelection::resolve_host::<f64>(profile, objective, 1e-12, 1e-14, nz(100), reference, faer)
}
fn assert_solution(solution: &LinearSolution, expected: &[f64]) {
    for (value, expected) in solution.values().iter().zip(expected) {
        assert!((value - expected).abs() < 1e-11);
    }
    assert_eq!(solution.values().len(), expected.len());
    assert!(solution.report().true_residual_norm() <= solution.report().residual_target());
}

#[test]
fn exact_two_coordinate_work_units_and_requested_payload_are_independent_of_execution() {
    let request = forecast(2, 2, 2, 3, 3);
    let general = profile(LinearOperatorProperties::General, 2, 4, request);
    let spd = profile(
        LinearOperatorProperties::SymmetricPositiveDefinite,
        2,
        4,
        request,
    );
    // Two structural preparations inspect n+z=6 entries each. Per iteration:
    // CG: 2z+8n=24, CG/Jacobi:26, BiCGSTAB:4z+20n=56.
    // Each independent acceptance is 2z+5n=18. These are declared proxies,
    // not measured FLOPs, seconds, completed iterations or retained reuse.
    for (owner, algorithm, preconditioner, setup, solve, total, workspace) in [
        (
            &general,
            LinearSolver::BiConjugateGradientStabilized,
            PreconditionerPolicy::Identity,
            12,
            336,
            384,
            200,
        ),
        (
            &spd,
            LinearSolver::ConjugateGradient,
            PreconditionerPolicy::Identity,
            12,
            144,
            192,
            88,
        ),
        (
            &spd,
            LinearSolver::ConjugateGradient,
            PreconditionerPolicy::Jacobi,
            16,
            156,
            208,
            104,
        ),
    ] {
        let manual = plan(algorithm, preconditioner, ReductionPolicy::Reproducible);
        let cost = owner.estimate_cost(
            "manual",
            manual,
            REFERENCE_LINEAR_SOLVER.provider(),
            ScalarDomain::Real,
            ScalarType::F64,
        );
        assert_eq!(cost.candidate(), "manual");
        assert_eq!(
            (
                cost.setup_work(),
                cost.solve_work(),
                cost.acceptance_work(),
                cost.total_work()
            ),
            (Some(setup), Some(solve), Some(36), Some(total))
        );
        // CSR row offsets (3) + stored column indices (4), and four values
        // plus a two-entry RHS. Pointer-index bytes are explicitly host-sized.
        assert_eq!(
            cost.numeric_payload_bytes(),
            Some(workspace + 7 * size_of::<usize>() + 48)
        );
    }
    let jacobi = spd.clone().with_cost_request(forecast(2, 2, 2, 3, 1));
    let cg = plan(
        LinearSolver::ConjugateGradient,
        PreconditionerPolicy::Jacobi,
        ReductionPolicy::Reproducible,
    );
    assert_eq!(
        jacobi
            .estimate_cost(
                "manual",
                cg,
                REFERENCE_LINEAR_SOLVER.provider(),
                ScalarDomain::Real,
                ScalarType::F64
            )
            .total_work(),
        Some(104)
    );
    let lu = general.estimate_cost(
        "manual",
        faer_sparse_lu_plan(),
        FaerLinearSolver.provider(),
        ScalarDomain::Real,
        ScalarType::F64,
    );
    // Dense envelope: structural 12 + two numeric n^3=8 preparations;
    // two triangular-solve proxies 2n^2+3n=14; acceptance 36.
    assert_eq!(
        (
            lu.setup_work(),
            lu.solve_work(),
            lu.acceptance_work(),
            lu.total_work()
        ),
        (Some(28), Some(28), Some(36), Some(92))
    );
    assert_eq!(lu.numeric_payload_bytes(), None);
}

#[test]
fn cost_forecasts_do_not_change_exact_manual_controls_or_analytical_solutions() {
    #[derive(Debug)]
    struct Spd;
    impl CsrStorage for Spd {
        fn rows(&self) -> usize {
            2
        }
        fn columns(&self) -> usize {
            2
        }
        fn row_offsets(&self) -> &[usize] {
            &[0, 2, 4]
        }
        fn column_indices(&self) -> &[usize] {
            &[0, 1, 0, 1]
        }
        fn values(&self) -> &[f64] {
            &[4., 1., 1., 3.]
        }
    }
    let general =
        CanonicalCsrSystemView::new(&Fixture, Fixture.rhs(), LinearOperatorProperties::General)
            .unwrap();
    let spd = CanonicalCsrSystemView::new(
        &Spd,
        &[6., 7.],
        LinearOperatorProperties::SymmetricPositiveDefinite,
    )
    .unwrap();
    for (system, manual, backend) in [
        (
            &general,
            reference_identity_plan(),
            &REFERENCE_LINEAR_SOLVER as &dyn LinearSolverBackend,
        ),
        (
            &spd,
            plan(
                LinearSolver::ConjugateGradient,
                PreconditionerPolicy::Jacobi,
                ReductionPolicy::Reproducible,
            ),
            &REFERENCE_LINEAR_SOLVER as &dyn LinearSolverBackend,
        ),
        (
            &general,
            faer_sparse_lu_plan(),
            &FaerLinearSolver as &dyn LinearSolverBackend,
        ),
    ] {
        let owner = profile(system.properties(), 2, 4, forecast(2, 2, 2, 1, 1));
        owner.require_plan(manual).unwrap();
        owner
            .require_resources(
                manual,
                backend.provider(),
                ScalarDomain::Real,
                ScalarType::F64,
            )
            .unwrap();
        let _forecast = owner.estimate_cost(
            "manual",
            manual,
            backend.provider(),
            ScalarDomain::Real,
            ScalarType::F64,
        );
        let solution = LinearSolveRequest::new(backend, manual)
            .solve(&system.linear_problem().unwrap())
            .unwrap();
        // A*[1,2]=b independently in both matrices; a forecast of one iteration
        // leaves the actual 100-iteration ceiling and exact provider untouched.
        assert_solution(&solution, &[1., 2.]);
        let [x, y] = solution.values() else {
            panic!("two coordinates");
        };
        let residual = if system.properties() == LinearOperatorProperties::General {
            (6. - (4. * x + y)).hypot(8. - (2. * x + 3. * y))
        } else {
            (6. - (4. * x + y)).hypot(7. - (x + 3. * y))
        };
        assert!(residual <= solution.report().residual_target());
        assert_eq!(solution.report().solver_plan(), manual);
        assert_eq!(solution.report().solver_provider(), backend.provider());
        assert_eq!(manual.maximum_iterations().get(), 100);
    }
}

#[derive(Debug, Default)]
struct CountedReference(AtomicUsize);
impl LinearSolverBackend for CountedReference {
    fn provider(&self) -> SolverProvider {
        REFERENCE_LINEAR_SOLVER.provider()
    }
    fn capabilities(&self) -> SolverCapabilities {
        REFERENCE_LINEAR_SOLVER.capabilities()
    }
    fn solve_with_execution(
        &self,
        problem: &LinearProblem<'_>,
        plan: SolverPlan,
        execution: &dyn ReplicatedLinearExecution,
    ) -> Result<LinearSolution, eqiora::Diagnostic> {
        self.0.fetch_add(1, Ordering::SeqCst);
        REFERENCE_LINEAR_SOLVER.solve_with_execution(problem, plan, execution)
    }
}
#[derive(Debug)]
struct Diagonal;
impl CsrStorage for Diagonal {
    fn rows(&self) -> usize {
        2
    }
    fn columns(&self) -> usize {
        2
    }
    fn row_offsets(&self) -> &[usize] {
        &[0, 1, 2]
    }
    fn column_indices(&self) -> &[usize] {
        &[0, 1]
    }
    fn values(&self) -> &[f64] {
        &[4., 3.]
    }
}

#[test]
fn actual_csr_payload_is_rechecked_without_overwriting_decision_provenance() {
    let sparse =
        CanonicalCsrSystemView::new(&Diagonal, &[4., 6.], LinearOperatorProperties::General)
            .unwrap();
    let dense =
        CanonicalCsrSystemView::new(&Fixture, Fixture.rhs(), LinearOperatorProperties::General)
            .unwrap();
    let small = 232 + 5 * size_of::<usize>(); // workspace200 + values/RHS32 + five indices.
    let large = 248 + 7 * size_of::<usize>(); // workspace200 + values/RHS48 + seven indices.
    for (limit, allowed) in [(large, true), (small, false)] {
        let backend = CountedReference::default();
        let owner = profile(
            LinearOperatorProperties::General,
            2,
            2,
            forecast(1, 1, 1, 3, 3).with_numeric_payload_limit(nz(limit)),
        );
        let decision = choose(owner, SolverPlanningObjective::LowMemory, &backend, None).unwrap();
        let costs = decision.costs().to_vec();
        assert_eq!(backend.0.load(Ordering::SeqCst), 0);
        assert_eq!(
            decision.solver_plan().preconditioner(),
            PreconditionerPolicy::Identity
        );
        assert_solution(
            &decision
                .solve(&sparse.linear_problem().unwrap(), None, &backend)
                .unwrap(),
            &[1., 2.],
        );
        assert_eq!(backend.0.load(Ordering::SeqCst), 1);
        let result = decision.solve(&dense.linear_problem().unwrap(), None, &backend);
        if allowed {
            assert_solution(&result.unwrap(), &[1., 2.]);
        } else {
            assert!(
                result
                    .unwrap_err()
                    .message()
                    .contains("resource.numeric-payload-limit-exceeded")
            );
        }
        assert_eq!(
            backend.0.load(Ordering::SeqCst),
            if allowed { 2 } else { 1 }
        );
        assert_eq!(
            decision.profile().size(),
            Some(SolverPlanningSize::exact(2, 2))
        );
        assert_eq!(decision.costs(), costs);
    }
}

#[test]
fn payload_unknown_and_insufficient_limits_reject_before_any_reference_work() {
    let backend = CountedReference::default();
    let owner = profile(
        LinearOperatorProperties::General,
        2,
        4,
        forecast(1, 1, 1, 3, 3).with_numeric_payload_limit(nz(1)),
    );
    let error = choose(
        owner.clone(),
        SolverPlanningObjective::Fast,
        &backend,
        Some(&FaerLinearSolver),
    )
    .unwrap_err();
    assert!(
        error
            .message()
            .contains("resource.numeric-payload-limit-exceeded")
    );
    assert!(
        error
            .message()
            .contains("resource.numeric-payload-bound-unproved")
    );
    assert_eq!(backend.0.load(Ordering::SeqCst), 0);
    // Faer is mathematically/provider-admitted without a limit, but its unknown
    // memory implementation bound cannot be silently represented as zero.
    let plain = profile(
        LinearOperatorProperties::General,
        2,
        4,
        forecast(1, 1, 1, 3, 3),
    );
    plain.require_plan(faer_sparse_lu_plan()).unwrap();
    assert!(
        plain
            .clone()
            .with_cost_request(forecast(1, 1, 1, 3, 3).with_numeric_payload_limit(nz(1_000_000)))
            .require_resources(
                faer_sparse_lu_plan(),
                FaerLinearSolver.provider(),
                ScalarDomain::Real,
                ScalarType::F64
            )
            .unwrap_err()
            .message()
            .contains("bound-unproved")
    );
}

#[test]
fn low_memory_prefers_known_bounds_and_preserves_missing_fact_provider_reasons() {
    let decision = choose(
        profile(
            LinearOperatorProperties::General,
            2,
            4,
            forecast(1, 1, 1, 3, 3),
        ),
        SolverPlanningObjective::LowMemory,
        &REFERENCE_LINEAR_SOLVER,
        Some(&FaerLinearSolver),
    )
    .unwrap();
    assert_eq!(
        decision.solver_provider(),
        REFERENCE_LINEAR_SOLVER.provider()
    );
    assert_eq!(
        decision.solver_plan().algorithm(),
        LinearSolver::BiConjugateGradientStabilized
    );
    assert_eq!(
        decision.solver_plan().preconditioner(),
        PreconditionerPolicy::Identity
    );
    assert!(
        decision
            .costs()
            .iter()
            .filter(|cost| cost.candidate().starts_with("eqiora.faer."))
            .all(|cost| cost.numeric_payload_bytes().is_none())
    );
    let unknown =
        SolverPlanningProfile::canonical_csr(LinearOperatorProperties::General, None, None)
            .with_size(SolverPlanningSize::exact(2, 4))
            .with_cost_request(forecast(1, 1, 1, 3, 3));
    let cg = plan(
        LinearSolver::ConjugateGradient,
        PreconditionerPolicy::Identity,
        ReductionPolicy::Reproducible,
    );
    // Mathematical class admission belongs to the actual solve request;
    // profile.require_plan owns diagonal and execution/forecast facts only.
    let general =
        CanonicalCsrSystemView::new(&Fixture, Fixture.rhs(), LinearOperatorProperties::General)
            .unwrap();
    assert!(
        LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, cg)
            .solve(&general.linear_problem().unwrap())
            .is_err()
    );
    let fallback = choose(
        unknown,
        SolverPlanningObjective::Fast,
        &REFERENCE_LINEAR_SOLVER,
        None,
    )
    .unwrap();
    assert_eq!(
        fallback.solver_provider(),
        REFERENCE_LINEAR_SOLVER.provider()
    );
    assert_eq!(
        fallback.solver_plan().preconditioner(),
        PreconditionerPolicy::Identity
    );
    assert!(
        fallback
            .reasons()
            .any(|(id, reason)| id.contains("jacobi") && reason.contains("diagonal"))
    );
    assert!(
        fallback
            .reasons()
            .any(|(id, reason)| id.starts_with("eqiora.faer.")
                && id.contains("identity")
                && reason.contains("provider"))
    );
}

#[test]
fn declared_horizon_changes_the_work_proxy_ranking_without_executing_reuse() {
    // Symmetric 4/-1 tridiagonal: diagonal4 > sum(|offdiagonal|)<=2,
    // hence SPD independently of the ranker. n=100, z=3n-2=298.
    #[derive(Debug)]
    struct Tridiagonal {
        offsets: Vec<usize>,
        columns: Vec<usize>,
        values: Vec<f64>,
    }
    impl CsrStorage for Tridiagonal {
        fn rows(&self) -> usize {
            100
        }
        fn columns(&self) -> usize {
            100
        }
        fn row_offsets(&self) -> &[usize] {
            &self.offsets
        }
        fn column_indices(&self) -> &[usize] {
            &self.columns
        }
        fn values(&self) -> &[f64] {
            &self.values
        }
    }
    let mut matrix = Tridiagonal {
        offsets: vec![0],
        columns: Vec::new(),
        values: Vec::new(),
    };
    for row in 0usize..100 {
        for column in row.saturating_sub(1)..=(row + 1).min(99) {
            matrix.columns.push(column);
            matrix.values.push(if row == column { 4. } else { -1. });
        }
        matrix.offsets.push(matrix.columns.len());
    }
    let rhs = [0.; 100];
    let system = CanonicalCsrSystemView::new(
        &matrix,
        &rhs,
        LinearOperatorProperties::SymmetricPositiveDefinite,
    )
    .unwrap();
    let size = SolverPlanningSize::exact(system.rows(), system.column_indices().len());
    assert_eq!(size, SolverPlanningSize::exact(100, 298));
    let one = profile(
        system.properties(),
        size.rows(),
        size.stored_slots(),
        forecast(1, 1, 1, 100, 100),
    );
    let retained = one.clone().with_cost_request(forecast(100, 1, 1, 100, 100));
    let cold = choose(
        one,
        SolverPlanningObjective::Fast,
        &REFERENCE_LINEAR_SOLVER,
        Some(&FaerLinearSolver),
    )
    .unwrap();
    let warm = choose(
        retained,
        SolverPlanningObjective::Fast,
        &REFERENCE_LINEAR_SOLVER,
        Some(&FaerLinearSolver),
    )
    .unwrap();
    assert_eq!(
        cold.solver_plan().algorithm(),
        LinearSolver::ConjugateGradient
    );
    assert_eq!(
        cold.solver_plan().preconditioner(),
        PreconditionerPolicy::Identity
    );
    assert_eq!(warm.solver_plan().algorithm(), LinearSolver::SparseLu);
    for (decision, expected) in [
        (&cold, (398, 139600, 1096, 141094)),
        (&warm, (1000398, 2030000, 109600, 3139998)),
    ] {
        let cost = decision
            .costs()
            .iter()
            .find(|cost| cost.candidate() == decision.selected_candidate_id())
            .unwrap();
        assert_eq!(
            (
                cost.setup_work(),
                cost.solve_work(),
                cost.acceptance_work(),
                cost.total_work()
            ),
            (
                Some(expected.0),
                Some(expected.1),
                Some(expected.2),
                Some(expected.3)
            )
        );
    }
    // This checks declared policy arithmetic only. It does not execute a
    // hundred solves or claim ordinary Run retains a factorization/preparation.
}

#[test]
fn invalid_horizons_forecasts_and_arithmetic_overflow_do_not_invent_costs() {
    assert!(SolverCostRequest::new(nz(1), nz(2), nz(2)).is_err());
    assert!(SolverCostRequest::new(nz(2), nz(1), nz(3)).is_err());
    assert!(SolverPlanningSize::dense_bound(usize::MAX).is_none());
    let owner = profile(
        LinearOperatorProperties::General,
        2,
        4,
        forecast(1, 1, 1, 101, 101),
    );
    assert!(
        choose(
            owner,
            SolverPlanningObjective::Robust,
            &REFERENCE_LINEAR_SOLVER,
            None
        )
        .unwrap_err()
        .message()
        .contains("forecast-exceeds-maximum")
    );
    let huge = profile(
        LinearOperatorProperties::General,
        usize::MAX,
        usize::MAX,
        forecast(1, 1, 1, 3, 3),
    );
    let cost = huge.estimate_cost(
        "manual",
        faer_sparse_lu_plan(),
        FaerLinearSolver.provider(),
        ScalarDomain::Real,
        ScalarType::F64,
    );
    assert_eq!(
        (
            cost.setup_work(),
            cost.solve_work(),
            cost.acceptance_work(),
            cost.total_work()
        ),
        (None, None, None, None)
    );
    assert_eq!(cost.numeric_payload_bytes(), None);
}

/// Restrict an actual provider to one of its already-declared exact tuples.
#[derive(Debug)]
struct RestrictedBackend<'a> {
    backend: &'a dyn LinearSolverBackend,
    capability: SolverCapability,
}
impl LinearSolverBackend for RestrictedBackend<'_> {
    fn provider(&self) -> SolverProvider {
        self.backend.provider()
    }
    fn capabilities(&self) -> SolverCapabilities {
        assert!(
            self.backend
                .capabilities()
                .combinations()
                .contains(&self.capability)
        );
        SolverCapabilities::exact([self.capability]).unwrap()
    }
    fn solve_with_execution(
        &self,
        problem: &LinearProblem<'_>,
        plan: SolverPlan,
        execution: &dyn ReplicatedLinearExecution,
    ) -> Result<LinearSolution, eqiora::Diagnostic> {
        self.backend.solve_with_execution(problem, plan, execution)
    }
}

#[derive(Debug)]
struct MismatchedReference<'a> {
    backend: &'a CountedReference,
    provider: SolverProvider,
    execution: eqiora::solver::ExecutionReport,
}
impl LinearSolverBackend for MismatchedReference<'_> {
    fn provider(&self) -> SolverProvider {
        self.provider
    }
    fn capabilities(&self) -> SolverCapabilities {
        self.backend.capabilities()
    }
    fn default_execution(&self) -> eqiora::solver::ExecutionReport {
        self.execution
    }
    fn solve_with_execution(
        &self,
        problem: &LinearProblem<'_>,
        plan: SolverPlan,
        execution: &dyn ReplicatedLinearExecution,
    ) -> Result<LinearSolution, eqiora::Diagnostic> {
        self.backend.solve_with_execution(problem, plan, execution)
    }
}

#[test]
fn owned_selection_rejects_changed_backend_identity_before_execution() {
    let backend = CountedReference::default();
    let decision = choose(
        SolverPlanningProfile::general_canonical_csr(),
        SolverPlanningObjective::Robust,
        &backend,
        None,
    )
    .unwrap();
    let system =
        CanonicalCsrSystemView::new(&Fixture, Fixture.rhs(), LinearOperatorProperties::General)
            .unwrap();
    let problem = system.linear_problem().unwrap();
    let different_provider = MismatchedReference {
        backend: &backend,
        provider: FaerLinearSolver.provider(),
        execution: eqiora::solver::ExecutionReport::host_serial(),
    };
    let different_execution = MismatchedReference {
        backend: &backend,
        provider: backend.provider(),
        execution: eqiora::solver::ExecutionReport::host(SERIAL_EXECUTION_PROVIDER.id(), nz(2)),
    };
    let restricted = RestrictedBackend {
        backend: &backend,
        capability: SolverCapability {
            scalar_domain: ScalarDomain::Real,
            scalar_type: ScalarType::F64,
            algorithm: LinearSolver::BiConjugateGradientStabilized,
            operator_properties: LinearOperatorProperties::General,
            preconditioner: PreconditionerPolicy::Jacobi,
            reduction: ReductionPolicy::Reproducible,
        },
    };
    for changed in [
        &different_provider as &dyn LinearSolverBackend,
        &different_execution,
        &restricted,
    ] {
        let error = decision.solve(&problem, None, changed).unwrap_err();
        assert!(
            error
                .message()
                .contains("host-binding.selected-provider-mismatch")
        );
        assert_eq!(backend.0.load(Ordering::SeqCst), 0);
    }
    assert_solution(
        &decision.solve(&problem, None, &backend).unwrap(),
        &[1., 2.],
    );
    assert_eq!(backend.0.load(Ordering::SeqCst), 1);
}

#[test]
fn expanded_real_catalogs_execute_each_literal_tuple_against_analytical_systems() {
    use LinearSolver::{
        BiConjugateGradientStabilized as Bicg, ConjugateGradient as Cg, MinimumResidual as Minres,
        SparseLu as Lu,
    };
    use PreconditionerPolicy::{Identity as I, Jacobi as J};
    // Independent inventory: reference CG(I,J), MINRES(I,J), BiCGSTAB(I,J),
    // faer CG(I,J), BiCGSTAB(I,J), LU(I) for SPD; reference MINRES(I,J),
    // BiCGSTAB(I,J), faer BiCGSTAB(I,J)/LU(I) for indefinite; the same
    // MINRES/BiCGSTAB/LU combinations for plain symmetric. Zero diagonal
    // excludes Jacobi independently, without inventing a definiteness proof.
    // The booleans identify faer, and the reference tuples use Reproducible.
    let spd = [
        (
            "eqiora.reference.cg-spd-identity-reproducible-f64",
            false,
            Cg,
            I,
        ),
        (
            "eqiora.reference.cg-spd-jacobi-reproducible-f64",
            false,
            Cg,
            J,
        ),
        (
            "eqiora.reference.minres-spd-identity-reproducible-f64",
            false,
            Minres,
            I,
        ),
        (
            "eqiora.reference.minres-spd-jacobi-reproducible-f64",
            false,
            Minres,
            J,
        ),
        (
            "eqiora.reference.bicgstab-spd-identity-reproducible-f64",
            false,
            Bicg,
            I,
        ),
        (
            "eqiora.reference.bicgstab-spd-jacobi-reproducible-f64",
            false,
            Bicg,
            J,
        ),
        ("eqiora.faer.cg-spd-identity-fast-f64", true, Cg, I),
        ("eqiora.faer.cg-spd-jacobi-fast-f64", true, Cg, J),
        ("eqiora.faer.bicgstab-spd-identity-fast-f64", true, Bicg, I),
        ("eqiora.faer.bicgstab-spd-jacobi-fast-f64", true, Bicg, J),
        ("eqiora.faer.sparse-lu-spd-identity-fast-f64", true, Lu, I),
    ];
    let indefinite = [
        (
            "eqiora.faer.bicgstab-indefinite-identity-fast-f64",
            true,
            Bicg,
            I,
        ),
        (
            "eqiora.faer.bicgstab-indefinite-jacobi-fast-f64",
            true,
            Bicg,
            J,
        ),
        (
            "eqiora.reference.minres-indefinite-identity-reproducible-f64",
            false,
            Minres,
            I,
        ),
        (
            "eqiora.reference.minres-indefinite-jacobi-reproducible-f64",
            false,
            Minres,
            J,
        ),
        (
            "eqiora.reference.bicgstab-indefinite-identity-reproducible-f64",
            false,
            Bicg,
            I,
        ),
        (
            "eqiora.reference.bicgstab-indefinite-jacobi-reproducible-f64",
            false,
            Bicg,
            J,
        ),
        (
            "eqiora.faer.sparse-lu-indefinite-identity-fast-f64",
            true,
            Lu,
            I,
        ),
    ];
    let symmetric = [
        (
            "eqiora.reference.minres-symmetric-identity-reproducible-f64",
            false,
            Minres,
            I,
        ),
        (
            "eqiora.reference.minres-symmetric-jacobi-reproducible-f64",
            false,
            Minres,
            J,
        ),
        (
            "eqiora.reference.bicgstab-symmetric-identity-reproducible-f64",
            false,
            Bicg,
            I,
        ),
        (
            "eqiora.reference.bicgstab-symmetric-jacobi-reproducible-f64",
            false,
            Bicg,
            J,
        ),
        (
            "eqiora.faer.bicgstab-symmetric-identity-fast-f64",
            true,
            Bicg,
            I,
        ),
        (
            "eqiora.faer.bicgstab-symmetric-jacobi-fast-f64",
            true,
            Bicg,
            J,
        ),
        (
            "eqiora.faer.sparse-lu-symmetric-identity-fast-f64",
            true,
            Lu,
            I,
        ),
    ];
    #[derive(Debug)]
    struct DenseTwo([f64; 4]);
    impl CsrStorage for DenseTwo {
        fn rows(&self) -> usize {
            2
        }
        fn columns(&self) -> usize {
            2
        }
        fn row_offsets(&self) -> &[usize] {
            &[0, 2, 4]
        }
        fn column_indices(&self) -> &[usize] {
            &[0, 1, 0, 1]
        }
        fn values(&self) -> &[f64] {
            &self.0
        }
    }
    for (properties, values, rhs, expected, tuples) in [
        (
            LinearOperatorProperties::SymmetricPositiveDefinite,
            [4., 1., 1., 3.],
            [6., 7.],
            [1., 2.],
            spd.as_slice(),
        ),
        (
            LinearOperatorProperties::SymmetricIndefinite,
            [1., 2., 2., 1.],
            [-1., 1.],
            [1., -1.],
            indefinite.as_slice(),
        ),
        (
            LinearOperatorProperties::Symmetric,
            [1., 2., 2., 1.],
            [-1., 1.],
            [1., -1.],
            symmetric.as_slice(),
        ),
        (
            LinearOperatorProperties::Symmetric,
            [0., 1., 1., 0.],
            [2., 1.],
            [1., 2.],
            symmetric.as_slice(),
        ),
    ] {
        // SPD has leading minors4,11; indefinite has eigenvalues3,-1;
        // plain symmetric is invertible with eigenvalues1,-1 and zero diagonal.
        let system = CanonicalCsrSystemView::new(&DenseTwo(values), &rhs, properties).unwrap();
        let problem = system.linear_problem().unwrap();
        let decision = choose(
            SolverPlanningProfile::canonical_csr(properties, Some(true), None)
                .with_size(SolverPlanningSize::exact(2, 4))
                .with_cost_request(forecast(1, 1, 1, 3, 3)),
            SolverPlanningObjective::Robust,
            &REFERENCE_LINEAR_SOLVER,
            Some(&FaerLinearSolver),
        )
        .unwrap();
        let mut admitted = decision
            .costs()
            .iter()
            .map(|cost| cost.candidate())
            .collect::<Vec<_>>();
        let mut inventory = tuples.iter().map(|tuple| tuple.0).collect::<Vec<_>>();
        admitted.sort_unstable();
        inventory.sort_unstable();
        assert_eq!(admitted, inventory);
        let backend: &dyn LinearSolverBackend =
            if decision.solver_provider() == REFERENCE_LINEAR_SOLVER.provider() {
                &REFERENCE_LINEAR_SOLVER
            } else {
                &FaerLinearSolver
            };
        assert_solution(&decision.solve(&problem, None, backend).unwrap(), &expected);
        for &(id, faer, algorithm, preconditioner) in tuples {
            let backend: &dyn LinearSolverBackend = if faer {
                &FaerLinearSolver
            } else {
                &REFERENCE_LINEAR_SOLVER
            };
            let manual = plan(
                algorithm,
                preconditioner,
                if faer {
                    ReductionPolicy::Fast
                } else {
                    ReductionPolicy::Reproducible
                },
            );
            if preconditioner == J && (values[0] == 0. || values[3] == 0.) {
                // Stored zero diagonal entries are structurally present. Exact
                // capability admission succeeds; numeric Jacobi admission fails.
                backend
                    .capabilities()
                    .require_problem(manual, ScalarDomain::Real, ScalarType::F64, properties)
                    .unwrap();
                let error = LinearSolveRequest::new(backend, manual)
                    .solve(&problem)
                    .unwrap_err();
                assert!(error.message().contains("diagonal"), "{id}: {error}");
                continue;
            }
            let solution = LinearSolveRequest::new(backend, manual)
                .solve(&problem)
                .unwrap_or_else(|error| panic!("{id}: {error}"));
            assert_solution(&solution, &expected);
            let x = solution.values();
            let residual = (rhs[0] - (values[0] * x[0] + values[1] * x[1]))
                .hypot(rhs[1] - (values[2] * x[0] + values[3] * x[1]));
            assert!(
                residual <= solution.report().residual_target(),
                "{id}: {residual}"
            );
            assert_eq!(solution.report().solver_plan(), manual, "{id}");
            assert_eq!(
                solution.report().solver_provider(),
                backend.provider(),
                "{id}"
            );
            // Independently multiply the literal symmetric matrix by [-2,3].
            // The derivative RHS differs from the captured primal RHS.
            let transpose_rhs = [
                -2. * values[0] + 3. * values[2],
                -2. * values[1] + 3. * values[3],
            ];
            let transpose = LinearSolveRequest::new(backend, manual)
                .solve_canonical_oriented(
                    &system,
                    &transpose_rhs,
                    eqiora::solver::LinearOperatorOrientation::Transposed,
                )
                .unwrap_or_else(|error| panic!("{id} transpose: {error}"));
            assert_solution(&transpose, &[-2., 3.]);
            assert_eq!(transpose.report().solver_plan(), manual);
            assert_eq!(transpose.report().solver_provider(), backend.provider());
            assert_eq!(
                transpose.report().orientation(),
                eqiora::solver::LinearOperatorOrientation::Transposed
            );
            // Select this one literal candidate using existing required-reduction
            // and exact provider-capability admission, then execute that selection.
            // This binds each catalog ID to the independently listed manual tuple.
            let restricted = RestrictedBackend {
                backend,
                capability: SolverCapability {
                    scalar_domain: ScalarDomain::Real,
                    scalar_type: ScalarType::F64,
                    algorithm,
                    operator_properties: properties,
                    preconditioner,
                    reduction: manual.reduction(),
                },
            };
            let owner = SolverPlanningProfile::canonical_csr(
                properties,
                Some(true),
                Some(manual.reduction()),
            )
            .with_size(SolverPlanningSize::exact(2, 4))
            .with_cost_request(forecast(1, 1, 1, 3, 3));
            let selected = if faer {
                choose(
                    owner,
                    SolverPlanningObjective::Robust,
                    &REFERENCE_LINEAR_SOLVER,
                    Some(&restricted),
                )
            } else {
                choose(owner, SolverPlanningObjective::Robust, &restricted, None)
            }
            .unwrap_or_else(|error| panic!("{id}: {error}"));
            assert_eq!(selected.selected_candidate_id(), id);
            assert_eq!(selected.solver_plan(), manual);
            assert_eq!(selected.solver_provider(), backend.provider());
            assert_solution(
                &selected.solve(&problem, None, &restricted).unwrap(),
                &expected,
            );
        }
    }
}
