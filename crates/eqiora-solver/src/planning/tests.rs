use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;

use super::*;
use crate::csr::CanonicalCsrOperatorCallLedger;
use crate::{
    BackendId, CanonicalCsrSystemView, CsrStorage, LinearOperator, LinearOperatorOrientation,
    LinearOperatorProperties, LinearProblem, LinearSolution, LinearSolveRequest, LinearSolver,
    LinearSolverBackend, Oriented, OrientedLinearOperator, PreconditionerPolicy, ProviderLibrary,
    ReductionPolicy, ReplicatedLinearExecution, ScalarType, SolverCapabilities, SolverCapability,
    SolverPlan, SolverProvider,
};

const POLICY_ID: &str = "eqiora.host-serial-solver-planning/v5";
const REFERENCE_ID: &str = "eqiora.reference.bicgstab-general-jacobi-reproducible-f64";
const FAER_BICGSTAB_ID: &str = "eqiora.faer.bicgstab-general-jacobi-fast-f64";
const FAER_SPARSE_LU_ID: &str = "eqiora.faer.sparse-lu-general-identity-fast-f64";
const REFERENCE_EVIDENCE: &str = "fluid.cartesian-advection-diffusion-fvm-2d";
const FAER_EVIDENCE: &str = "numerics.linear-backends";

const EMPTY_LIBRARIES: &[ProviderLibrary] = &[];
const FAER_LIBRARIES: &[ProviderLibrary] = &[ProviderLibrary::new("faer", "0.24.4")];
const CHANGED_FAER_LIBRARIES: &[ProviderLibrary] = &[ProviderLibrary::new("faer", "0.24.5")];
const EXTRA_FAER_LIBRARIES: &[ProviderLibrary] = &[
    ProviderLibrary::new("faer", "0.24.4"),
    ProviderLibrary::new("rayon", "1.12.0"),
];

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

fn plan(
    algorithm: LinearSolver,
    preconditioner: PreconditionerPolicy,
    reduction: ReductionPolicy,
) -> SolverPlan {
    plan_with_controls(algorithm, preconditioner, reduction, 1.0e-12, 1.0e-14, 100)
}

fn plan_with_controls(
    algorithm: LinearSolver,
    preconditioner: PreconditionerPolicy,
    reduction: ReductionPolicy,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    maximum_iterations: usize,
) -> SolverPlan {
    SolverPlan::new(
        algorithm,
        relative_tolerance,
        absolute_tolerance,
        NonZeroUsize::new(maximum_iterations).unwrap(),
    )
    .unwrap()
    .with_preconditioner(preconditioner)
    .with_reduction(reduction)
}

fn reference_plan() -> SolverPlan {
    plan(
        LinearSolver::BiConjugateGradientStabilized,
        PreconditionerPolicy::Jacobi,
        ReductionPolicy::Reproducible,
    )
}

fn faer_bicgstab_plan() -> SolverPlan {
    plan(
        LinearSolver::BiConjugateGradientStabilized,
        PreconditionerPolicy::Jacobi,
        ReductionPolicy::Fast,
    )
}

fn faer_sparse_lu_plan() -> SolverPlan {
    plan(
        LinearSolver::SparseLu,
        PreconditionerPolicy::Identity,
        ReductionPolicy::Fast,
    )
}

fn capability(plan: SolverPlan) -> SolverCapability {
    SolverCapability {
        scalar_domain: eqiora_core::ScalarDomain::Real,
        algorithm: plan.algorithm(),
        operator_properties: LinearOperatorProperties::General,
        preconditioner: plan.preconditioner(),
        reduction: plan.reduction(),
        scalar_type: ScalarType::F64,
    }
}

#[derive(Debug)]
struct CountingBackend {
    provider: SolverProvider,
    capability: SolverCapability,
    expected_plan: SolverPlan,
    succeed: bool,
    solve_calls: AtomicUsize,
    operator_apply_calls: AtomicUsize,
    operator_diagonal_calls: AtomicUsize,
    received_problem: AtomicPtr<()>,
    received_operator: AtomicPtr<()>,
}

impl CountingBackend {
    fn new(provider: SolverProvider, expected_plan: SolverPlan) -> Self {
        Self {
            provider,
            capability: capability(expected_plan),
            expected_plan,
            succeed: false,
            solve_calls: AtomicUsize::new(0),
            operator_apply_calls: AtomicUsize::new(0),
            operator_diagonal_calls: AtomicUsize::new(0),
            received_problem: AtomicPtr::new(std::ptr::null_mut()),
            received_operator: AtomicPtr::new(std::ptr::null_mut()),
        }
    }

    fn successful(provider: SolverProvider, expected_plan: SolverPlan) -> Self {
        Self {
            provider,
            capability: capability(expected_plan),
            expected_plan,
            succeed: true,
            solve_calls: AtomicUsize::new(0),
            operator_apply_calls: AtomicUsize::new(0),
            operator_diagonal_calls: AtomicUsize::new(0),
            received_problem: AtomicPtr::new(std::ptr::null_mut()),
            received_operator: AtomicPtr::new(std::ptr::null_mut()),
        }
    }

    fn with_capability(
        provider: SolverProvider,
        expected_plan: SolverPlan,
        capability: SolverCapability,
    ) -> Self {
        Self {
            provider,
            capability,
            expected_plan,
            succeed: false,
            solve_calls: AtomicUsize::new(0),
            operator_apply_calls: AtomicUsize::new(0),
            operator_diagonal_calls: AtomicUsize::new(0),
            received_problem: AtomicPtr::new(std::ptr::null_mut()),
            received_operator: AtomicPtr::new(std::ptr::null_mut()),
        }
    }

    fn calls(&self) -> usize {
        self.solve_calls.load(Ordering::SeqCst)
    }

    fn operator_apply_calls(&self) -> usize {
        self.operator_apply_calls.load(Ordering::SeqCst)
    }

    fn operator_diagonal_calls(&self) -> usize {
        self.operator_diagonal_calls.load(Ordering::SeqCst)
    }

    fn reset_operator_ledger_before_execution(&self) {
        assert_eq!(
            self.operator_apply_calls.swap(0, Ordering::SeqCst),
            0,
            "resolution must not apply the operator through a backend"
        );
        assert_eq!(
            self.operator_diagonal_calls.swap(0, Ordering::SeqCst),
            0,
            "resolution must not request the operator diagonal through a backend"
        );
    }

    fn received_problem(&self) -> *const () {
        self.received_problem.load(Ordering::SeqCst).cast_const()
    }

    fn received_operator(&self) -> *const () {
        self.received_operator.load(Ordering::SeqCst).cast_const()
    }
}

#[derive(Debug)]
struct CountingAcceptanceExecution<'a> {
    delegate: &'a dyn ReplicatedLinearExecution,
    expected_operator: &'a dyn LinearOperator<Scalar = f64>,
    operator_apply_calls: &'a AtomicUsize,
}

impl ReplicatedLinearExecution for CountingAcceptanceExecution<'_> {
    fn provider(&self) -> crate::ExecutionProvider {
        self.delegate.provider()
    }

    fn report(&self) -> crate::ExecutionReport {
        self.delegate.report()
    }

    fn require_reduction(&self, policy: ReductionPolicy) -> Result<(), Diagnostic> {
        self.delegate.require_reduction(policy)
    }

    fn apply(
        &self,
        operator: &dyn LinearOperator<Scalar = f64>,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        assert!(
            std::ptr::eq(operator, self.expected_operator),
            "acceptance must apply the exact resolved problem operator"
        );
        self.operator_apply_calls.fetch_add(1, Ordering::SeqCst);
        self.delegate.apply(operator, input, output)
    }

    fn inner_product(&self, action: crate::FixedOrderInnerProduct<'_>) -> Result<f64, Diagnostic> {
        self.delegate.inner_product(action)
    }
}

impl LinearSolverBackend for CountingBackend {
    fn provider(&self) -> SolverProvider {
        self.provider
    }

    fn capabilities(&self) -> SolverCapabilities {
        SolverCapabilities::exact([self.capability]).unwrap()
    }

    fn solve_with_execution(
        &self,
        problem: &LinearProblem<'_>,
        selected_plan: SolverPlan,
        execution: &dyn ReplicatedLinearExecution,
    ) -> Result<LinearSolution, Diagnostic> {
        self.solve_calls.fetch_add(1, Ordering::SeqCst);
        self.received_problem.store(
            std::ptr::from_ref(problem).cast::<()>().cast_mut(),
            Ordering::SeqCst,
        );
        self.received_operator.store(
            std::ptr::from_ref(problem.operator())
                .cast::<()>()
                .cast_mut(),
            Ordering::SeqCst,
        );
        assert_eq!(
            selected_plan, self.expected_plan,
            "selected plan was mutated"
        );
        assert_eq!(
            problem.right_hand_side(),
            &[6.0, 8.0],
            "decision executed a substituted problem"
        );
        if self.succeed {
            let verifier = CountingAcceptanceExecution {
                delegate: execution,
                expected_operator: problem.operator(),
                operator_apply_calls: &self.operator_apply_calls,
            };
            crate::LinearSolution::accept_with_verifier(
                problem,
                selected_plan,
                self.provider,
                execution.provider(),
                execution.report(),
                crate::ConvergenceReason::ResidualToleranceSatisfied,
                1,
                0.0,
                vec![1.0, 2.0],
                &verifier,
            )
        } else {
            Err(Diagnostic::error(
                codes::NUMERICAL_SOLVE_FAILED,
                "selected fake backend failure",
            ))
        }
    }
}

#[derive(Debug)]
struct CountingCsrStorage {
    row_offsets: &'static [usize],
    column_indices: &'static [usize],
    values: &'static [f64],
    storage_calls: AtomicUsize,
    apply_calls: AtomicUsize,
    diagonal_calls: AtomicUsize,
}

impl CountingCsrStorage {
    fn full() -> Self {
        Self::new(&[0, 2, 4], &[0, 1, 0, 1], &[4.0, 1.0, 2.0, 3.0])
    }

    fn missing_first_diagonal() -> Self {
        Self::new(&[0, 1, 3], &[1, 0, 1], &[1.0, 2.0, 3.0])
    }

    fn missing_second_diagonal() -> Self {
        Self::new(&[0, 2, 3], &[0, 1, 0], &[4.0, 1.0, 2.0])
    }

    fn new(
        row_offsets: &'static [usize],
        column_indices: &'static [usize],
        values: &'static [f64],
    ) -> Self {
        Self {
            row_offsets,
            column_indices,
            values,
            storage_calls: AtomicUsize::new(0),
            apply_calls: AtomicUsize::new(0),
            diagonal_calls: AtomicUsize::new(0),
        }
    }

    fn reset_after_legitimate_construction(&self) {
        assert!(
            self.storage_calls.swap(0, Ordering::SeqCst) > 0,
            "canonical construction must capture the supplied storage"
        );
        self.apply_calls.store(0, Ordering::SeqCst);
        self.diagonal_calls.store(0, Ordering::SeqCst);
    }

    fn storage_calls(&self) -> usize {
        self.storage_calls.load(Ordering::SeqCst)
    }

    fn apply_calls(&self) -> usize {
        self.apply_calls.load(Ordering::SeqCst)
    }

    fn diagonal_calls(&self) -> usize {
        self.diagonal_calls.load(Ordering::SeqCst)
    }
}

impl CsrStorage for CountingCsrStorage {
    fn rows(&self) -> usize {
        self.storage_calls.fetch_add(1, Ordering::SeqCst);
        2
    }

    fn columns(&self) -> usize {
        self.storage_calls.fetch_add(1, Ordering::SeqCst);
        2
    }

    fn row_offsets(&self) -> &[usize] {
        self.storage_calls.fetch_add(1, Ordering::SeqCst);
        self.row_offsets
    }

    fn column_indices(&self) -> &[usize] {
        self.storage_calls.fetch_add(1, Ordering::SeqCst);
        self.column_indices
    }

    fn values(&self) -> &[f64] {
        self.storage_calls.fetch_add(1, Ordering::SeqCst);
        self.values
    }
}

impl CountingCsrStorage {
    pub(crate) fn rhs(&self) -> &[f64] {
        self.storage_calls.fetch_add(1, Ordering::SeqCst);
        &[6.0, 8.0]
    }
}

impl LinearOperator for CountingCsrStorage {
    type Scalar = f64;

    fn rows(&self) -> usize {
        2
    }

    fn columns(&self) -> usize {
        2
    }

    fn apply(&self, input: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        self.apply_calls.fetch_add(1, Ordering::SeqCst);
        output[0] = 4.0 * input[0] + input[1];
        output[1] = 2.0 * input[0] + 3.0 * input[1];
        Ok(())
    }

    fn diagonal(&self, output: &mut [f64]) -> Result<crate::DiagonalAvailability, Diagnostic> {
        self.diagonal_calls.fetch_add(1, Ordering::SeqCst);
        output.copy_from_slice(&[4.0, 3.0]);
        Ok(crate::DiagonalAvailability::Available)
    }
}

fn instrumented_canonical_system(
    storage: &CountingCsrStorage,
    properties: LinearOperatorProperties,
) -> (CanonicalCsrSystemView, CanonicalCsrOperatorCallLedger) {
    let mut system = CanonicalCsrSystemView::new(storage, storage.rhs(), properties).unwrap();
    let uninstrumented_clone = system.clone();
    let ledger = CanonicalCsrOperatorCallLedger::default();
    system.attach_test_operator_call_ledger(&ledger);
    assert_eq!(
        system, uninstrumented_clone,
        "test-only instrumentation must not alter canonical view equality"
    );
    assert_eq!(
        system.clone(),
        system,
        "test-only instrumentation must preserve canonical view clone equality"
    );

    {
        let problem = system.linear_problem().unwrap();
        let mut applied = [0.0; 2];
        problem.operator().apply(&[1.0, 2.0], &mut applied).unwrap();
        let mut diagonal = [0.0; 2];
        let _ = problem.operator().diagonal(&mut diagonal).unwrap();
    }
    assert_eq!(
        ledger.apply_calls(),
        1,
        "direct problem.operator().apply self-control"
    );
    assert_eq!(
        ledger.diagonal_calls(),
        1,
        "direct problem.operator().diagonal self-control"
    );
    ledger.reset();
    assert_eq!(ledger.apply_calls(), 0);
    assert_eq!(ledger.diagonal_calls(), 0);

    (system, ledger)
}

#[derive(Debug)]
struct CountingOperator {
    apply_calls: AtomicUsize,
    diagonal_calls: AtomicUsize,
    diagonal_available: bool,
}

impl Default for CountingOperator {
    fn default() -> Self {
        Self {
            apply_calls: AtomicUsize::new(0),
            diagonal_calls: AtomicUsize::new(0),
            diagonal_available: true,
        }
    }
}

impl CountingOperator {
    fn matrix_free() -> Self {
        Self {
            diagonal_available: false,
            ..Self::default()
        }
    }

    fn apply_calls(&self) -> usize {
        self.apply_calls.load(Ordering::SeqCst)
    }

    fn diagonal_calls(&self) -> usize {
        self.diagonal_calls.load(Ordering::SeqCst)
    }
}

impl LinearOperator for CountingOperator {
    type Scalar = f64;

    fn rows(&self) -> usize {
        2
    }

    fn columns(&self) -> usize {
        2
    }

    fn apply(&self, input: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        self.apply_calls.fetch_add(1, Ordering::SeqCst);
        output[0] = 4.0 * input[0] + input[1];
        output[1] = 2.0 * input[0] + 3.0 * input[1];
        Ok(())
    }

    fn diagonal(&self, output: &mut [f64]) -> Result<crate::DiagonalAvailability, Diagnostic> {
        self.diagonal_calls.fetch_add(1, Ordering::SeqCst);
        if self.diagonal_available {
            output.copy_from_slice(&[4.0, 3.0]);
            Ok(crate::DiagonalAvailability::Available)
        } else {
            Ok(crate::DiagonalAvailability::Unavailable)
        }
    }
}

impl OrientedLinearOperator for CountingOperator {
    fn supports_orientation(&self, _orientation: crate::LinearOperatorOrientation) -> bool {
        true
    }

    fn apply_oriented(
        &self,
        orientation: crate::LinearOperatorOrientation,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        if orientation == crate::LinearOperatorOrientation::Normal {
            return self.apply(input, output);
        }
        self.apply_calls.fetch_add(1, Ordering::SeqCst);
        output[0] = 4.0 * input[0] + 2.0 * input[1];
        output[1] = input[0] + 3.0 * input[1];
        Ok(())
    }
}

const OBJECTIVES: [SolverPlanningObjective; 3] = [
    SolverPlanningObjective::Robust,
    SolverPlanningObjective::Fast,
    SolverPlanningObjective::LowMemory,
];
const REFERENCE_IDENTITY_ID: &str = "eqiora.reference.bicgstab-general-identity-reproducible-f64";
const FAER_IDENTITY_ID: &str = "eqiora.faer.bicgstab-general-identity-fast-f64";
const ORACLE: &str =
    include_str!("../../../../verify/numerics/host-serial-solver-planning/expected/policy-v5.json");

fn reference_identity_plan() -> SolverPlan {
    reference_plan().with_preconditioner(PreconditionerPolicy::Identity)
}
fn faer_identity_plan() -> SolverPlan {
    faer_bicgstab_plan().with_preconditioner(PreconditionerPolicy::Identity)
}

#[derive(Clone, Copy)]
struct Spec {
    id: &'static str,
    evidence: &'static str,
    provider: SolverProvider,
    plan: fn() -> SolverPlan,
}
// Literal tuple inventory is independent of the production catalog builder.
const SPECS: [Spec; 5] = [
    Spec {
        id: FAER_IDENTITY_ID,
        evidence: "electrical.parallel-dc-network",
        provider: FAER_PROVIDER,
        plan: faer_identity_plan,
    },
    Spec {
        id: FAER_BICGSTAB_ID,
        evidence: FAER_EVIDENCE,
        provider: FAER_PROVIDER,
        plan: faer_bicgstab_plan,
    },
    Spec {
        id: FAER_SPARSE_LU_ID,
        evidence: FAER_EVIDENCE,
        provider: FAER_PROVIDER,
        plan: faer_sparse_lu_plan,
    },
    Spec {
        id: REFERENCE_IDENTITY_ID,
        evidence: "fluid.cell-centered-navier-stokes-fvm-2d",
        provider: REFERENCE_PROVIDER,
        plan: reference_identity_plan,
    },
    Spec {
        id: REFERENCE_ID,
        evidence: REFERENCE_EVIDENCE,
        provider: REFERENCE_PROVIDER,
        plan: reference_plan,
    },
];
const ALL: u8 = 0b11111;
fn backends(success: bool) -> [CountingBackend; 5] {
    SPECS.map(|spec| {
        if success {
            CountingBackend::successful(spec.provider, (spec.plan)())
        } else {
            CountingBackend::new(spec.provider, (spec.plan)())
        }
    })
}
fn exact_catalog(backends: &[CountingBackend; 5]) -> [HostSerialSolverCandidate<'_>; 5] {
    std::array::from_fn(|i| {
        HostSerialSolverCandidate::new(
            SPECS[i].id,
            SPECS[i].evidence,
            LinearSolveRequest::new(&backends[i], (SPECS[i].plan)()),
        )
    })
}
fn oracle() -> &'static serde_json::Value {
    static VALUE: std::sync::OnceLock<serde_json::Value> = std::sync::OnceLock::new();
    VALUE.get_or_init(|| serde_json::from_str(ORACLE).unwrap())
}
fn objective_name(objective: SolverPlanningObjective) -> &'static str {
    match objective {
        SolverPlanningObjective::Robust => "Robust",
        SolverPlanningObjective::Fast => "Fast",
        SolverPlanningObjective::LowMemory => "LowMemory",
    }
}
fn expected_record(mask: u8, objective: SolverPlanningObjective) -> &'static serde_json::Value {
    let ids = SPECS
        .iter()
        .enumerate()
        .filter(|(i, _)| mask & (1 << i) != 0)
        .map(|(_, spec)| spec.id)
        .collect::<Vec<_>>();
    oracle()["admitted_subsets"]
        .as_array()
        .unwrap()
        .iter()
        .find(|subset| {
            subset["admitted_candidate_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|id| id.as_str().unwrap())
                .collect::<Vec<_>>()
                == ids
        })
        .unwrap()["objectives"]
        .as_array()
        .unwrap()
        .iter()
        .find(|record| record["objective"] == objective_name(objective))
        .unwrap()
}
fn expected_selection(mask: u8, objective: SolverPlanningObjective) -> &'static str {
    expected_record(mask, objective)["selected_candidate_id"]
        .as_str()
        .unwrap()
}
fn expected_trace(
    rejections: &[(usize, &'static str)],
    objective: SolverPlanningObjective,
) -> Vec<(&'static str, &'static str)> {
    let mask = rejections
        .iter()
        .fold(ALL, |mask, (index, _)| mask & !(1 << index));
    expected_record(mask, objective)["ordered_reasons"]
        .as_array()
        .unwrap()
        .iter()
        .map(|pair| {
            let id = pair[0].as_str().unwrap();
            let reason = rejections
                .iter()
                .find(|(i, _)| SPECS[*i].id == id)
                .map_or_else(|| pair[1].as_str().unwrap(), |(_, reason)| *reason);
            (id, reason)
        })
        .collect()
}
fn assert_zero_work(
    storage: &CountingCsrStorage,
    ledger: &CanonicalCsrOperatorCallLedger,
    backends: &[CountingBackend; 5],
) {
    assert_eq!(storage.storage_calls(), 0);
    assert_eq!(storage.apply_calls(), 0);
    assert_eq!(storage.diagonal_calls(), 0);
    assert_eq!(ledger.apply_calls(), 0);
    assert_eq!(ledger.diagonal_calls(), 0);
    for backend in backends {
        assert_eq!(backend.calls(), 0);
        assert_eq!(backend.operator_apply_calls(), 0);
        assert_eq!(backend.operator_diagonal_calls(), 0);
        assert!(backend.received_problem().is_null());
        assert!(backend.received_operator().is_null());
    }
}
fn assert_rejections(
    problem: &LinearProblem<'_>,
    candidates: &[HostSerialSolverCandidate<'_>],
    rejected: &[(usize, &'static str)],
) {
    for objective in OBJECTIVES {
        let decision = resolve_host_serial_solver_v5(problem, objective, candidates).unwrap();
        let mask = rejected.iter().fold(ALL, |mask, (i, _)| mask & !(1 << i));
        assert_eq!(
            decision.selected().id(),
            expected_selection(mask, objective)
        );
        assert_eq!(
            decision.reasons().collect::<Vec<_>>(),
            expected_trace(rejected, objective)
        );
    }
}
fn assert_all_rejected(error: &Diagnostic, reasons: &[(usize, &'static str)]) {
    let trace = reasons
        .iter()
        .map(|(i, reason)| format!("{}={reason}", SPECS[*i].id))
        .collect::<Vec<_>>()
        .join(",");
    assert_eq!(error.code(), codes::INVALID_REALIZATION);
    assert_eq!(
        error.message(),
        format!("{POLICY_ID} no admitted candidate; trace=[{trace}]")
    );
}
fn permutations<T: Copy>(values: &mut [T], start: usize, visit: &mut impl FnMut(&[T])) {
    if start == values.len() {
        visit(values);
        return;
    }
    for i in start..values.len() {
        values.swap(start, i);
        permutations(values, start + 1, visit);
        values.swap(start, i);
    }
}

#[test]
fn exact_catalog_decisions_and_reasons_are_permutation_invariant() {
    let storage = CountingCsrStorage::full();
    let (system, ledger) =
        instrumented_canonical_system(&storage, LinearOperatorProperties::General);
    storage.reset_after_legitimate_construction();
    let problem = system.linear_problem().unwrap();
    let backends = backends(false);
    let mut catalog = exact_catalog(&backends);
    let mut count = 0;
    permutations(&mut catalog, 0, &mut |catalog| {
        count += 1;
        for objective in OBJECTIVES {
            let decision = resolve_host_serial_solver_v5(&problem, objective, catalog).unwrap();
            assert_eq!(decision.objective(), objective);
            assert_eq!(decision.policy_id(), POLICY_ID);
            assert_eq!(decision.selected().id(), expected_selection(ALL, objective));
            assert_eq!(
                decision.reasons().collect::<Vec<_>>(),
                expected_trace(&[], objective)
            );
            assert!(std::ptr::eq(decision.problem(), &problem));
            assert!(std::ptr::eq(
                decision.problem().operator(),
                problem.operator()
            ));
            assert_eq!(
                decision.execution_provider(),
                crate::SERIAL_EXECUTION_PROVIDER
            );
            assert_eq!(
                decision.solver_provider(),
                decision.selected().request().backend().provider()
            );
            assert_zero_work(&storage, &ledger, &backends);
        }
    });
    assert_eq!(count, 120);
}

#[test]
fn every_observable_admitted_subset_has_exact_reranking_and_trace() {
    let storage = CountingCsrStorage::full();
    let (system, ledger) =
        instrumented_canonical_system(&storage, LinearOperatorProperties::General);
    storage.reset_after_legitimate_construction();
    let problem = system.linear_problem().unwrap();
    let backends = backends(false);
    let catalog = exact_catalog(&backends);
    for mask in 1..=ALL {
        let rejected = SPECS
            .iter()
            .enumerate()
            .filter(|(i, _)| mask & (1 << i) == 0)
            .map(|(i, _)| (i, "catalog.evidence-mismatch"))
            .collect::<Vec<_>>();
        let candidates = std::array::from_fn::<_, 5, _>(|i| {
            HostSerialSolverCandidate::new(
                catalog[i].id(),
                if mask & (1 << i) == 0 {
                    "stale"
                } else {
                    catalog[i].evidence_case()
                },
                catalog[i].request(),
            )
        });
        assert_rejections(&problem, &candidates, &rejected);
        assert_zero_work(&storage, &ledger, &backends);
    }
}

#[test]
fn malformed_inventory_rejects_before_profile_or_numerical_work() {
    let storage = CountingCsrStorage::full();
    let (system, ledger) =
        instrumented_canonical_system(&storage, LinearOperatorProperties::General);
    storage.reset_after_legitimate_construction();
    let problem = system.linear_problem().unwrap();
    let backends = backends(false);
    let catalog = exact_catalog(&backends);
    let missing = catalog[..4].to_vec();
    let mut duplicate = catalog.to_vec();
    duplicate.push(catalog[0]);
    let mut unknown = catalog.to_vec();
    unknown.push(HostSerialSolverCandidate::new(
        "unknown",
        FAER_EVIDENCE,
        catalog[0].request(),
    ));
    for (candidates, reason) in [
        (missing, "catalog.missing-id"),
        (duplicate, "catalog.duplicate-id"),
        (unknown, "catalog.unknown-id"),
    ] {
        let error =
            resolve_host_serial_solver_v5(&problem, SolverPlanningObjective::Robust, &candidates)
                .unwrap_err();
        assert!(error.message().contains(reason), "{error}");
        assert_zero_work(&storage, &ledger, &backends);
    }
}

#[test]
fn common_controls_are_compared_by_bits_and_exact_iteration_count() {
    assert_eq!(1.0e-12_f64.to_bits(), 0x3d71_9799_812d_ea11);
    assert_eq!(1.0e-14_f64.to_bits(), 0x3d06_849b_86a1_2b9b);
    let storage = CountingCsrStorage::full();
    let (system, ledger) =
        instrumented_canonical_system(&storage, LinearOperatorProperties::General);
    storage.reset_after_legitimate_construction();
    let problem = system.linear_problem().unwrap();
    let backends = backends(false);
    let catalog = exact_catalog(&backends);
    for i in 0..SPECS.len() {
        let original = (SPECS[i].plan)();
        for (relative, absolute, iterations) in [
            (f64::from_bits(1e-12_f64.to_bits() + 1), 1e-14, 100),
            (1e-12, f64::from_bits(1e-14_f64.to_bits() + 1), 100),
            (1e-12, 1e-14, 99),
        ] {
            let changed = plan_with_controls(
                original.algorithm(),
                original.preconditioner(),
                original.reduction(),
                relative,
                absolute,
                iterations,
            );
            let mut candidates = catalog;
            candidates[i] = HostSerialSolverCandidate::new(
                SPECS[i].id,
                SPECS[i].evidence,
                LinearSolveRequest::new(&backends[i], changed),
            );
            let error =
                resolve_host_serial_solver_v5(&problem, SolverPlanningObjective::Fast, &candidates)
                    .unwrap_err();
            assert!(error.message().contains("catalog.control-mismatch"));
            assert_zero_work(&storage, &ledger, &backends);
        }
    }
}

#[test]
fn catalog_validation_precedence_and_exact_plan_tuple_are_frozen() {
    let storage = CountingCsrStorage::full();
    let (system, ledger) =
        instrumented_canonical_system(&storage, LinearOperatorProperties::General);
    storage.reset_after_legitimate_construction();
    let problem = system.linear_problem().unwrap();
    let backends = backends(false);
    let catalog = exact_catalog(&backends);
    for i in 0..SPECS.len() {
        let spec = SPECS[i];
        let original = (spec.plan)();
        let bad_providers = [
            SolverProvider::new(spec.provider.id(), "stale", spec.provider.libraries()),
            SolverProvider::new(
                spec.provider.id(),
                env!("CARGO_PKG_VERSION"),
                CHANGED_FAER_LIBRARIES,
            ),
            SolverProvider::new(
                spec.provider.id(),
                env!("CARGO_PKG_VERSION"),
                EXTRA_FAER_LIBRARIES,
            ),
        ];
        for provider in bad_providers.into_iter().chain([SolverProvider::new(
            spec.provider.id(),
            env!("CARGO_PKG_VERSION"),
            EMPTY_LIBRARIES,
        )]) {
            if provider == spec.provider {
                continue;
            }
            let bad = CountingBackend::new(provider, original);
            let mut candidates = catalog;
            candidates[i] = HostSerialSolverCandidate::new(
                spec.id,
                "stale",
                LinearSolveRequest::new(&bad, original),
            );
            assert_rejections(&problem, &candidates, &[(i, "catalog.evidence-mismatch")]);
            candidates[i] = HostSerialSolverCandidate::new(
                spec.id,
                spec.evidence,
                LinearSolveRequest::new(
                    &bad,
                    original.with_preconditioner(
                        if original.preconditioner() == PreconditionerPolicy::Identity {
                            PreconditionerPolicy::Jacobi
                        } else {
                            PreconditionerPolicy::Identity
                        },
                    ),
                ),
            );
            assert_rejections(&problem, &candidates, &[(i, "catalog.provider-mismatch")]);
            assert_eq!(bad.calls(), 0);
        }
        let changed = [
            original.with_preconditioner(
                if original.preconditioner() == PreconditionerPolicy::Identity {
                    PreconditionerPolicy::Jacobi
                } else {
                    PreconditionerPolicy::Identity
                },
            ),
            original.with_reduction(if original.reduction() == ReductionPolicy::Fast {
                ReductionPolicy::Reproducible
            } else {
                ReductionPolicy::Fast
            }),
            plan(
                LinearSolver::ConjugateGradient,
                original.preconditioner(),
                original.reduction(),
            ),
        ];
        for changed in changed {
            let mut candidates = catalog;
            candidates[i] = HostSerialSolverCandidate::new(
                spec.id,
                spec.evidence,
                LinearSolveRequest::new(&backends[i], changed),
            );
            assert_rejections(&problem, &candidates, &[(i, "catalog.plan-mismatch")]);
        }
        assert_zero_work(&storage, &ledger, &backends);
    }
}

#[test]
fn profile_and_capability_rejections_cannot_be_ranked() {
    let storage = CountingCsrStorage::full();
    let (system, ledger) =
        instrumented_canonical_system(&storage, LinearOperatorProperties::General);
    storage.reset_after_legitimate_construction();
    let problem = system.linear_problem().unwrap();
    let backends = backends(false);
    let catalog = exact_catalog(&backends);
    // The leading principal minors are 4 and 11, independently proving SPD.
    let symmetric_storage = CountingCsrStorage::new(&[0, 2, 4], &[0, 1, 0, 1], &[4., 1., 1., 3.]);
    let (symmetric, symmetric_ledger) = instrumented_canonical_system(
        &symmetric_storage,
        LinearOperatorProperties::SymmetricPositiveDefinite,
    );
    symmetric_storage.reset_after_legitimate_construction();
    let error = resolve_host_serial_solver_v5(
        &symmetric.linear_problem().unwrap(),
        SolverPlanningObjective::Robust,
        &catalog,
    )
    .unwrap_err();
    assert!(error.message().contains("catalog.missing-id"));
    assert_zero_work(&symmetric_storage, &symmetric_ledger, &backends);
    for i in 0..SPECS.len() {
        let spec = SPECS[i];
        let original = (spec.plan)();
        let exact = capability(original);
        for changed in [
            SolverCapability {
                algorithm: if exact.algorithm == LinearSolver::SparseLu {
                    LinearSolver::BiConjugateGradientStabilized
                } else {
                    LinearSolver::SparseLu
                },
                ..exact
            },
            SolverCapability {
                operator_properties: LinearOperatorProperties::SymmetricPositiveDefinite,
                ..exact
            },
            SolverCapability {
                preconditioner: if exact.preconditioner == PreconditionerPolicy::Jacobi {
                    PreconditionerPolicy::Identity
                } else {
                    PreconditionerPolicy::Jacobi
                },
                ..exact
            },
            SolverCapability {
                reduction: if exact.reduction == ReductionPolicy::Fast {
                    ReductionPolicy::Reproducible
                } else {
                    ReductionPolicy::Fast
                },
                ..exact
            },
            SolverCapability {
                scalar_type: ScalarType::F32,
                ..exact
            },
        ] {
            let bad = CountingBackend::with_capability(spec.provider, original, changed);
            let mut candidates = catalog;
            candidates[i] = HostSerialSolverCandidate::new(
                spec.id,
                spec.evidence,
                LinearSolveRequest::new(&bad, original),
            );
            assert_rejections(
                &problem,
                &candidates,
                &[(i, "capability.exact-tuple-required")],
            );
            assert_eq!(bad.calls(), 0);
        }
        assert_zero_work(&storage, &ledger, &backends);
    }
}

#[test]
fn unsupported_profiles_reject_with_zero_numerical_calls() {
    let storage = CountingCsrStorage::full();
    let (system, ledger) =
        instrumented_canonical_system(&storage, LinearOperatorProperties::General);
    storage.reset_after_legitimate_construction();
    let backends = backends(false);
    let catalog = exact_catalog(&backends);
    let operator = CountingOperator::default();
    let hand_built = LinearProblem::new(
        &operator,
        FixtureRhs::rhs(),
        LinearOperatorProperties::General,
    )
    .unwrap();
    let matrix_free = CountingOperator::matrix_free();
    let matrix_free_problem = LinearProblem::new(
        &matrix_free,
        FixtureRhs::rhs(),
        LinearOperatorProperties::General,
    )
    .unwrap();
    let oriented = Oriented::new(&system, LinearOperatorOrientation::Transposed).unwrap();
    let transposed = LinearProblem::new(
        &oriented,
        FixtureRhs::rhs(),
        LinearOperatorProperties::General,
    )
    .unwrap();
    for problem in [&hand_built, &matrix_free_problem, &transposed] {
        let error =
            resolve_host_serial_solver_v5(problem, SolverPlanningObjective::Robust, &catalog)
                .unwrap_err();
        assert!(
            error.message().contains("profile.canonical-csr-required")
                || error.message().contains("profile.normal-required")
        );
    }
    assert_eq!(operator.apply_calls(), 0);
    assert_eq!(operator.diagonal_calls(), 0);
    assert_eq!(matrix_free.apply_calls(), 0);
    assert_eq!(matrix_free.diagonal_calls(), 0);
    for missing in [
        CountingCsrStorage::missing_first_diagonal(),
        CountingCsrStorage::missing_second_diagonal(),
    ] {
        let (system, actual) =
            instrumented_canonical_system(&missing, LinearOperatorProperties::General);
        missing.reset_after_legitimate_construction();
        assert_rejections(
            &system.linear_problem().unwrap(),
            &catalog,
            &[
                (1, "profile.complete-diagonal-required"),
                (4, "profile.complete-diagonal-required"),
            ],
        );
        assert_zero_work(&missing, &actual, &backends);
    }
    assert_zero_work(&storage, &ledger, &backends);
}
struct FixtureRhs;
impl FixtureRhs {
    fn rhs() -> &'static [f64] {
        &[6., 8.]
    }
}

#[test]
fn no_admitted_diagnostic_freezes_the_complete_ordered_trace() {
    let storage = CountingCsrStorage::full();
    let (system, ledger) =
        instrumented_canonical_system(&storage, LinearOperatorProperties::General);
    storage.reset_after_legitimate_construction();
    let problem = system.linear_problem().unwrap();
    let backends = backends(false);
    let catalog = exact_catalog(&backends)
        .map(|c| HostSerialSolverCandidate::new(c.id(), "stale", c.request()));
    for objective in OBJECTIVES {
        let error = resolve_host_serial_solver_v5(&problem, objective, &catalog).unwrap_err();
        assert_all_rejected(
            &error,
            &SPECS
                .iter()
                .enumerate()
                .map(|(i, _)| (i, "catalog.evidence-mismatch"))
                .collect::<Vec<_>>(),
        );
    }
    assert_zero_work(&storage, &ledger, &backends);
}

fn execute_selected(succeed: bool) {
    for objective in OBJECTIVES {
        let storage = CountingCsrStorage::full();
        let (system, ledger) =
            instrumented_canonical_system(&storage, LinearOperatorProperties::General);
        storage.reset_after_legitimate_construction();
        let problem = system.linear_problem().unwrap();
        let backends = backends(succeed);
        let catalog = exact_catalog(&backends);
        let decision = resolve_host_serial_solver_v5(&problem, objective, &catalog).unwrap();
        assert_zero_work(&storage, &ledger, &backends);
        for backend in &backends {
            backend.reset_operator_ledger_before_execution();
        }
        let result = decision.solve();
        if succeed {
            assert_eq!(result.unwrap().values(), &[1., 2.]);
        } else {
            assert_eq!(result.unwrap_err().code(), codes::NUMERICAL_SOLVE_FAILED);
        }
        for (i, backend) in backends.iter().enumerate() {
            let selected = SPECS[i].id == expected_selection(ALL, objective);
            assert_eq!(backend.calls(), usize::from(selected));
            assert_eq!(
                backend.operator_apply_calls(),
                if selected && succeed { 2 } else { 0 }
            );
            assert_eq!(backend.operator_diagonal_calls(), 0);
            if selected {
                assert!(std::ptr::eq(
                    backend.received_problem(),
                    std::ptr::from_ref(&problem).cast()
                ));
                assert!(std::ptr::eq(
                    backend.received_operator(),
                    std::ptr::from_ref(problem.operator()).cast()
                ));
            } else {
                assert!(backend.received_problem().is_null());
                assert!(backend.received_operator().is_null());
            }
        }
        assert_eq!(ledger.apply_calls(), if succeed { 2 } else { 0 });
        assert_eq!(ledger.diagonal_calls(), 0);
    }
}
#[test]
fn selected_failure_executes_once_without_retry_plan_mutation_or_problem_substitution() {
    execute_selected(false);
}
#[test]
fn selected_success_executes_once_and_applies_exact_problem_for_both_true_residuals() {
    execute_selected(true);
}

#[test]
fn registered_host_serial_planning_oracle_executes_all_private_falsifiers() {
    super::complex_tests::typed_capabilities_and_requested_inventory_precede_ranking();
    super::complex_tests::typed_provider_identity_and_operator_class_are_independent();
    let checks: [fn(); 10] = [
        exact_catalog_decisions_and_reasons_are_permutation_invariant,
        every_observable_admitted_subset_has_exact_reranking_and_trace,
        malformed_inventory_rejects_before_profile_or_numerical_work,
        common_controls_are_compared_by_bits_and_exact_iteration_count,
        catalog_validation_precedence_and_exact_plan_tuple_are_frozen,
        profile_and_capability_rejections_cannot_be_ranked,
        unsupported_profiles_reject_with_zero_numerical_calls,
        no_admitted_diagnostic_freezes_the_complete_ordered_trace,
        selected_failure_executes_once_without_retry_plan_mutation_or_problem_substitution,
        selected_success_executes_once_and_applies_exact_problem_for_both_true_residuals,
    ];
    for check in checks {
        check();
    }
}

#[test]
fn unknown_diagonal_uses_identity_and_preserves_provider_unavailability_reasons() {
    let reference = CountingBackend::new(REFERENCE_PROVIDER, reference_identity_plan());
    let faer = CountingBackend::new(FAER_PROVIDER, faer_identity_plan());
    for objective in OBJECTIVES {
        let profile =
            HostSerialSolverProfile::canonical_csr(LinearOperatorProperties::General, None, None);
        let decision = ResolvedHostSerialSolverPlan::resolve(
            profile,
            objective,
            1e-12,
            1e-14,
            NonZeroUsize::new(100).unwrap(),
            &reference,
            Some(&faer),
        )
        .unwrap();
        assert_eq!(
            decision.selected_candidate_id(),
            if objective == SolverPlanningObjective::Robust {
                REFERENCE_IDENTITY_ID
            } else {
                FAER_IDENTITY_ID
            }
        );
        assert!(
            decision
                .reasons()
                .any(|reason| reason == (REFERENCE_ID, "profile.complete-diagonal-required"))
        );
        assert!(
            decision
                .reasons()
                .any(|reason| reason == (FAER_SPARSE_LU_ID, "capability.exact-tuple-required"))
        );
    }
    let decision = ResolvedHostSerialSolverPlan::resolve(
        HostSerialSolverProfile::canonical_csr(LinearOperatorProperties::General, None, None),
        SolverPlanningObjective::Fast,
        1e-12,
        1e-14,
        NonZeroUsize::new(100).unwrap(),
        &reference,
        Some(&reference),
    )
    .unwrap();
    assert_eq!(decision.selected_candidate_id(), REFERENCE_IDENTITY_ID);
    assert!(
        decision
            .reasons()
            .any(|reason| reason == (FAER_IDENTITY_ID, "catalog.provider-mismatch"))
    );
    assert_eq!(reference.calls(), 0);
    assert_eq!(faer.calls(), 0);
}

#[test]
fn symmetric_profiles_require_their_exact_capability_before_any_backend_work() {
    for (properties, algorithm, id) in [
        (
            LinearOperatorProperties::SymmetricPositiveDefinite,
            LinearSolver::ConjugateGradient,
            "eqiora.reference.cg-spd-identity-reproducible-f64",
        ),
        (
            LinearOperatorProperties::SymmetricIndefinite,
            LinearSolver::MinimumResidual,
            "eqiora.reference.minres-indefinite-identity-reproducible-f64",
        ),
        (
            LinearOperatorProperties::Symmetric,
            LinearSolver::MinimumResidual,
            "eqiora.reference.minres-symmetric-identity-reproducible-f64",
        ),
    ] {
        let exact = plan(
            algorithm,
            PreconditionerPolicy::Identity,
            ReductionPolicy::Reproducible,
        );
        let mut reference = CountingBackend::new(REFERENCE_PROVIDER, exact);
        reference.capability = capability(reference_plan());
        let mut faer = CountingBackend::new(FAER_PROVIDER, faer_sparse_lu_plan());
        // These descriptors advertise only General capabilities. The correct
        // algorithm alone cannot admit the requested symmetric operator class.
        let profile = HostSerialSolverProfile::canonical_csr(properties, Some(false), None);
        let rejected = ResolvedHostSerialSolverPlan::resolve(
            profile.clone(),
            SolverPlanningObjective::Robust,
            1e-12,
            1e-14,
            NonZeroUsize::new(100).unwrap(),
            &reference,
            Some(&faer),
        )
        .unwrap_err();
        assert!(
            rejected
                .message()
                .contains("capability.exact-tuple-required")
        );
        assert_eq!(reference.solve_calls.load(Ordering::SeqCst), 0);
        assert_eq!(faer.solve_calls.load(Ordering::SeqCst), 0);
        // Sparse LU accepts all three classes, but the provider must advertise
        // the exact class. Change only that field before admitting the direct candidate.
        faer.capability.operator_properties = properties;
        let direct = ResolvedHostSerialSolverPlan::resolve(
            profile.clone(),
            SolverPlanningObjective::Robust,
            1e-12,
            1e-14,
            NonZeroUsize::new(100).unwrap(),
            &reference,
            Some(&faer),
        )
        .unwrap();
        assert_eq!(direct.solver_plan().algorithm(), LinearSolver::SparseLu);
        assert_eq!(direct.solver_provider(), FAER_PROVIDER);
        reference.capability = SolverCapability {
            operator_properties: properties,
            ..capability(exact)
        };
        let accepted = ResolvedHostSerialSolverPlan::resolve(
            profile.clone(),
            SolverPlanningObjective::Robust,
            1e-12,
            1e-14,
            NonZeroUsize::new(100).unwrap(),
            &reference,
            Some(&faer),
        )
        .unwrap();
        assert_eq!(accepted.selected_candidate_id(), id);
        assert_eq!(accepted.solver_plan(), exact);
        assert_eq!(reference.solve_calls.load(Ordering::SeqCst), 0);
        assert_eq!(faer.solve_calls.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn required_reduction_filters_before_objective_ranking_and_exact_admission() {
    let exact = plan(
        LinearSolver::MinimumResidual,
        PreconditionerPolicy::Identity,
        ReductionPolicy::Reproducible,
    );
    let mut reference = CountingBackend::new(REFERENCE_PROVIDER, exact);
    reference.capability = SolverCapability {
        operator_properties: LinearOperatorProperties::SymmetricIndefinite,
        ..capability(exact)
    };
    let mut faer = CountingBackend::new(FAER_PROVIDER, faer_sparse_lu_plan());
    faer.capability.operator_properties = LinearOperatorProperties::SymmetricIndefinite;
    let profile = HostSerialSolverProfile::canonical_csr(
        LinearOperatorProperties::SymmetricIndefinite,
        None,
        Some(ReductionPolicy::Reproducible),
    );
    profile.require_plan(exact).unwrap();
    let error = profile.require_plan(faer_sparse_lu_plan()).unwrap_err();
    assert!(
        error
            .message()
            .contains("profile.required-reduction-mismatch")
    );
    for objective in [
        SolverPlanningObjective::Robust,
        SolverPlanningObjective::Fast,
        SolverPlanningObjective::LowMemory,
    ] {
        let selected = ResolvedHostSerialSolverPlan::resolve(
            profile.clone(),
            objective,
            1e-12,
            1e-14,
            NonZeroUsize::new(100).unwrap(),
            &reference,
            Some(&faer),
        )
        .unwrap();
        assert_eq!(selected.solver_plan(), exact);
        assert_eq!(selected.solver_provider(), REFERENCE_PROVIDER);
        assert!(
            selected
                .reasons()
                .any(|reason| reason
                    == (FAER_INDEFINITE_LU_ID, "profile.required-reduction-mismatch"))
        );
    }
    // Change only the execution requirement: Fast now admits LU and excludes MINRES,
    // even when the caller's preference is Robust. No algorithm is forced by a name.
    let fast_profile = HostSerialSolverProfile::canonical_csr(
        LinearOperatorProperties::SymmetricIndefinite,
        None,
        Some(ReductionPolicy::Fast),
    );
    let fast = ResolvedHostSerialSolverPlan::resolve(
        fast_profile,
        SolverPlanningObjective::Robust,
        1e-12,
        1e-14,
        NonZeroUsize::new(100).unwrap(),
        &reference,
        Some(&faer),
    )
    .unwrap();
    assert_eq!(fast.solver_plan().algorithm(), LinearSolver::SparseLu);
    assert!(
        fast.reasons()
            .any(|reason| reason == (REFERENCE_MINRES_ID, "profile.required-reduction-mismatch"))
    );
    assert_eq!(reference.solve_calls.load(Ordering::SeqCst), 0);
    assert_eq!(faer.solve_calls.load(Ordering::SeqCst), 0);
}
