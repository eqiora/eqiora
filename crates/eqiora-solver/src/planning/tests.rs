use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;

use super::*;
use crate::csr::CanonicalCsrOperatorCallLedger;
use crate::{
    BackendId, CanonicalCsrSystemView, CompleteCsrStorage, LinearOperator,
    LinearOperatorOrientation, LinearOperatorProperties, LinearProblem, LinearSolution,
    LinearSolveRequest, LinearSolver, LinearSolverBackend, PreconditionerPolicy, ProviderLibrary,
    ReductionPolicy, ReplicatedLinearExecution, ScalarType, SolverCapabilities, SolverCapability,
    SolverPlan, SolverProvider, TransposeLinearOperator, Transposed,
};

const POLICY_ID: &str = "eqiora.host-serial-solver-planning/v2";
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
    expected_operator: &'a dyn LinearOperator,
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
        operator: &dyn LinearOperator,
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
            crate::accept_linear_solution_with_verifier(
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

fn backends() -> (CountingBackend, CountingBackend, CountingBackend) {
    (
        CountingBackend::new(REFERENCE_PROVIDER, reference_plan()),
        CountingBackend::new(FAER_PROVIDER, faer_bicgstab_plan()),
        CountingBackend::new(FAER_PROVIDER, faer_sparse_lu_plan()),
    )
}

fn exact_catalog<'a>(
    reference: &'a CountingBackend,
    faer_bicgstab: &'a CountingBackend,
    faer_sparse_lu: &'a CountingBackend,
) -> [HostSerialSolverCandidate<'a>; 3] {
    // Deliberately not candidate-ID order.
    [
        HostSerialSolverCandidate::new(
            REFERENCE_ID,
            REFERENCE_EVIDENCE,
            LinearSolveRequest::new(reference, reference_plan()),
        ),
        HostSerialSolverCandidate::new(
            FAER_SPARSE_LU_ID,
            FAER_EVIDENCE,
            LinearSolveRequest::new(faer_sparse_lu, faer_sparse_lu_plan()),
        ),
        HostSerialSolverCandidate::new(
            FAER_BICGSTAB_ID,
            FAER_EVIDENCE,
            LinearSolveRequest::new(faer_bicgstab, faer_bicgstab_plan()),
        ),
    ]
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

impl CompleteCsrStorage for CountingCsrStorage {
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

    fn right_hand_side(&self) -> &[f64] {
        self.storage_calls.fetch_add(1, Ordering::SeqCst);
        &[6.0, 8.0]
    }
}

impl LinearOperator for CountingCsrStorage {
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
    storage: &dyn CompleteCsrStorage,
    properties: LinearOperatorProperties,
) -> (CanonicalCsrSystemView, CanonicalCsrOperatorCallLedger) {
    let mut system = CanonicalCsrSystemView::new(storage, properties).unwrap();
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
    assert_actual_canonical_operator_ledger(&ledger, 0, 0);

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

impl TransposeLinearOperator for CountingOperator {
    fn apply_transpose(&self, input: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
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

const FAER_BICGSTAB_BIT: u8 = 0b001;
const FAER_SPARSE_LU_BIT: u8 = 0b010;
const REFERENCE_BIT: u8 = 0b100;

fn assert_zero_backend_solves(
    reference: &CountingBackend,
    faer_bicgstab: &CountingBackend,
    faer_sparse_lu: &CountingBackend,
) {
    for (name, backend) in [
        ("reference", reference),
        ("faer BiCGSTAB", faer_bicgstab),
        ("faer SparseLU", faer_sparse_lu),
    ] {
        assert_eq!(backend.calls(), 0, "{name} backend solve ledger");
        assert_eq!(
            backend.operator_apply_calls(),
            0,
            "{name} backend operator apply ledger"
        );
        assert_eq!(
            backend.operator_diagonal_calls(),
            0,
            "{name} backend operator diagonal ledger"
        );
        assert_eq!(
            backend.received_problem(),
            std::ptr::null(),
            "{name} backend problem ledger"
        );
        assert_eq!(
            backend.received_operator(),
            std::ptr::null(),
            "{name} backend operator identity ledger"
        );
    }
}

fn reset_backend_operator_ledgers_before_execution(
    reference: &CountingBackend,
    faer_bicgstab: &CountingBackend,
    faer_sparse_lu: &CountingBackend,
) {
    for backend in [reference, faer_bicgstab, faer_sparse_lu] {
        backend.reset_operator_ledger_before_execution();
    }
}

fn assert_zero_operator_ledger(operator: &CountingOperator) {
    assert_eq!(operator.apply_calls(), 0, "operator apply ledger");
    assert_eq!(operator.diagonal_calls(), 0, "operator diagonal ledger");
}

fn assert_zero_canonical_source_ledger(storage: &CountingCsrStorage) {
    assert_eq!(
        storage.storage_calls(),
        0,
        "resolution must use only the captured canonical CSR owner"
    );
    assert_eq!(storage.apply_calls(), 0, "source operator apply ledger");
    assert_eq!(
        storage.diagonal_calls(),
        0,
        "source operator diagonal ledger"
    );
}

fn assert_zero_canonical_ledger(
    storage: &CountingCsrStorage,
    ledger: &CanonicalCsrOperatorCallLedger,
) {
    assert_zero_canonical_source_ledger(storage);
    assert_actual_canonical_operator_ledger(ledger, 0, 0);
}

fn assert_actual_canonical_operator_ledger(
    ledger: &CanonicalCsrOperatorCallLedger,
    expected_apply: usize,
    expected_diagonal: usize,
) {
    assert_eq!(
        ledger.apply_calls(),
        expected_apply,
        "total exact owned canonical operator apply ledger"
    );
    assert_eq!(
        ledger.diagonal_calls(),
        expected_diagonal,
        "total exact owned canonical operator diagonal ledger"
    );
}

fn selected_reason(objective: SolverPlanningObjective) -> &'static str {
    match objective {
        SolverPlanningObjective::Robust => "candidate.selected.robust-preference",
        SolverPlanningObjective::Fast => "candidate.selected.fast-preference",
        SolverPlanningObjective::LowMemory => "candidate.selected.low-memory-preference",
    }
}

fn candidate_bit(candidate_id: &str) -> u8 {
    match candidate_id {
        FAER_BICGSTAB_ID => FAER_BICGSTAB_BIT,
        FAER_SPARSE_LU_ID => FAER_SPARSE_LU_BIT,
        REFERENCE_ID => REFERENCE_BIT,
        other => panic!("unexpected frozen candidate {other}"),
    }
}

fn expected_subset_selection(
    admitted_mask: u8,
    objective: SolverPlanningObjective,
) -> &'static str {
    match (admitted_mask, objective) {
        (FAER_BICGSTAB_BIT, _) => FAER_BICGSTAB_ID,
        (FAER_SPARSE_LU_BIT, _) => FAER_SPARSE_LU_ID,
        (REFERENCE_BIT, _) => REFERENCE_ID,
        (0b011, SolverPlanningObjective::Robust) => FAER_BICGSTAB_ID,
        (0b011, SolverPlanningObjective::Fast) => FAER_SPARSE_LU_ID,
        (0b011, SolverPlanningObjective::LowMemory) => FAER_BICGSTAB_ID,
        (0b101, SolverPlanningObjective::Robust) => REFERENCE_ID,
        (0b101, SolverPlanningObjective::Fast | SolverPlanningObjective::LowMemory) => {
            FAER_BICGSTAB_ID
        }
        (0b110, SolverPlanningObjective::Robust | SolverPlanningObjective::LowMemory) => {
            REFERENCE_ID
        }
        (0b110, SolverPlanningObjective::Fast) => FAER_SPARSE_LU_ID,
        (0b111, objective) => expected_selection(objective),
        _ => panic!("unexpected nonempty admitted mask {admitted_mask:#05b}"),
    }
}

fn expected_subset_trace(
    admitted_mask: u8,
    objective: SolverPlanningObjective,
    rejection_reason: &'static str,
) -> Vec<(&'static str, &'static str)> {
    let selected = expected_subset_selection(admitted_mask, objective);
    let mut trace = Vec::new();
    for candidate_id in [FAER_BICGSTAB_ID, FAER_SPARSE_LU_ID, REFERENCE_ID] {
        if admitted_mask & candidate_bit(candidate_id) == 0 {
            trace.push((candidate_id, rejection_reason));
        } else {
            trace.push((candidate_id, "candidate.admitted"));
            trace.push((
                candidate_id,
                if candidate_id == selected {
                    selected_reason(objective)
                } else {
                    "candidate.not-selected"
                },
            ));
        }
    }
    trace
}

fn expected_trace(objective: SolverPlanningObjective) -> Vec<(&'static str, &'static str)> {
    match objective {
        SolverPlanningObjective::Robust => vec![
            (FAER_BICGSTAB_ID, "candidate.admitted"),
            (FAER_BICGSTAB_ID, "candidate.not-selected"),
            (FAER_SPARSE_LU_ID, "candidate.admitted"),
            (FAER_SPARSE_LU_ID, "candidate.not-selected"),
            (REFERENCE_ID, "candidate.admitted"),
            (REFERENCE_ID, "candidate.selected.robust-preference"),
        ],
        SolverPlanningObjective::Fast => vec![
            (FAER_BICGSTAB_ID, "candidate.admitted"),
            (FAER_BICGSTAB_ID, "candidate.not-selected"),
            (FAER_SPARSE_LU_ID, "candidate.admitted"),
            (FAER_SPARSE_LU_ID, "candidate.selected.fast-preference"),
            (REFERENCE_ID, "candidate.admitted"),
            (REFERENCE_ID, "candidate.not-selected"),
        ],
        SolverPlanningObjective::LowMemory => vec![
            (FAER_BICGSTAB_ID, "candidate.admitted"),
            (FAER_BICGSTAB_ID, "candidate.selected.low-memory-preference"),
            (FAER_SPARSE_LU_ID, "candidate.admitted"),
            (FAER_SPARSE_LU_ID, "candidate.not-selected"),
            (REFERENCE_ID, "candidate.admitted"),
            (REFERENCE_ID, "candidate.not-selected"),
        ],
    }
}

fn expected_selection(objective: SolverPlanningObjective) -> &'static str {
    match objective {
        SolverPlanningObjective::Robust => REFERENCE_ID,
        SolverPlanningObjective::Fast => FAER_SPARSE_LU_ID,
        SolverPlanningObjective::LowMemory => FAER_BICGSTAB_ID,
    }
}

#[test]
fn exact_catalog_decisions_and_reasons_are_permutation_invariant() {
    let storage = CountingCsrStorage::full();
    let (system, actual_operator_ledger) =
        instrumented_canonical_system(&storage, LinearOperatorProperties::General);
    storage.reset_after_legitimate_construction();
    let problem = system.linear_problem().unwrap();
    let (reference, faer_bicgstab, faer_sparse_lu) = backends();
    let catalog = exact_catalog(&reference, &faer_bicgstab, &faer_sparse_lu);
    let permutations = [
        [catalog[0], catalog[1], catalog[2]],
        [catalog[0], catalog[2], catalog[1]],
        [catalog[1], catalog[0], catalog[2]],
        [catalog[1], catalog[2], catalog[0]],
        [catalog[2], catalog[0], catalog[1]],
        [catalog[2], catalog[1], catalog[0]],
    ];

    for candidates in permutations {
        for objective in OBJECTIVES {
            let decision = resolve_host_serial_solver_v2(&problem, objective, &candidates).unwrap();
            assert_eq!(decision.objective(), objective);
            assert_eq!(decision.policy_id(), POLICY_ID);
            assert_eq!(decision.selected().id(), expected_selection(objective));
            assert_eq!(
                decision.problem().operator().orientation(),
                LinearOperatorOrientation::Normal
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
            assert_eq!(
                decision.reasons().collect::<Vec<_>>(),
                expected_trace(objective)
            );
            assert_zero_canonical_ledger(&storage, &actual_operator_ledger);
            assert_zero_backend_solves(&reference, &faer_bicgstab, &faer_sparse_lu);
        }
    }

    assert_eq!(FAER_PROVIDER.libraries().len(), 1);
    assert_eq!(FAER_PROVIDER.libraries()[0].name(), "faer");
    assert_eq!(FAER_PROVIDER.libraries()[0].version(), "0.24.4");
}

#[test]
fn every_observable_admitted_subset_has_exact_reranking_and_trace() {
    let storage = CountingCsrStorage::full();
    let (system, actual_operator_ledger) =
        instrumented_canonical_system(&storage, LinearOperatorProperties::General);
    storage.reset_after_legitimate_construction();
    let problem = system.linear_problem().unwrap();
    let (reference, faer_bicgstab, faer_sparse_lu) = backends();
    let catalog = exact_catalog(&reference, &faer_bicgstab, &faer_sparse_lu);

    for admitted_mask in 1..=0b111 {
        let candidates = catalog.map(|candidate| {
            HostSerialSolverCandidate::new(
                candidate.id(),
                if admitted_mask & candidate_bit(candidate.id()) == 0 {
                    "numerics.stale-evidence"
                } else {
                    candidate.evidence_case()
                },
                candidate.request(),
            )
        });
        for objective in OBJECTIVES {
            let decision = resolve_host_serial_solver_v2(&problem, objective, &candidates).unwrap();
            assert_eq!(
                decision.selected().id(),
                expected_subset_selection(admitted_mask, objective)
            );
            assert_eq!(
                decision.reasons().collect::<Vec<_>>(),
                expected_subset_trace(admitted_mask, objective, "catalog.evidence-mismatch")
            );
            assert_zero_canonical_ledger(&storage, &actual_operator_ledger);
            assert_zero_backend_solves(&reference, &faer_bicgstab, &faer_sparse_lu);
        }
    }
}

fn assert_inventory_failure(error: &Diagnostic, expected: &str) {
    assert_eq!(error.code(), codes::INVALID_REALIZATION);
    let fragments = [
        "catalog.missing-id",
        "catalog.duplicate-id",
        "catalog.unknown-id",
        "catalog.control-mismatch",
    ];
    assert!(error.message().contains(expected));
    assert_eq!(
        fragments
            .iter()
            .filter(|fragment| error.message().contains(**fragment))
            .count(),
        1,
        "inventory diagnostics contain exactly one frozen failure fragment"
    );
}

#[test]
fn malformed_inventory_rejects_before_profile_or_numerical_work() {
    let storage = CountingCsrStorage::full();
    let (system, actual_operator_ledger) =
        instrumented_canonical_system(&storage, LinearOperatorProperties::General);
    storage.reset_after_legitimate_construction();
    let problem = system.linear_problem().unwrap();
    assert!(std::ptr::addr_eq(
        problem.operator(),
        &system as &dyn LinearOperator
    ));
    let (reference, faer_bicgstab, faer_sparse_lu) = backends();
    let catalog = exact_catalog(&reference, &faer_bicgstab, &faer_sparse_lu);
    let missing = [catalog[0], catalog[1]];
    let duplicate = [catalog[0], catalog[1], catalog[1], catalog[2]];
    let unknown = [
        catalog[0],
        catalog[1],
        catalog[2],
        HostSerialSolverCandidate::new(
            "eqiora.unknown",
            FAER_EVIDENCE,
            LinearSolveRequest::new(&faer_bicgstab, faer_bicgstab_plan()),
        ),
    ];

    for (candidates, fragment) in [
        (&missing[..], "catalog.missing-id"),
        (&duplicate[..], "catalog.duplicate-id"),
        (&unknown[..], "catalog.unknown-id"),
    ] {
        actual_operator_ledger.reset();
        let error =
            resolve_host_serial_solver_v2(&problem, SolverPlanningObjective::Robust, candidates)
                .unwrap_err();
        assert_inventory_failure(&error, fragment);
        assert_zero_canonical_ledger(&storage, &actual_operator_ledger);
        assert_zero_backend_solves(&reference, &faer_bicgstab, &faer_sparse_lu);
    }
}

#[test]
fn common_controls_are_compared_by_bits_and_exact_iteration_count() {
    assert_eq!(1.0e-12_f64.to_bits(), 0x3d71_9799_812d_ea11);
    assert_eq!(1.0e-14_f64.to_bits(), 0x3d06_849b_86a1_2b9b);
    let storage = CountingCsrStorage::full();
    let (system, actual_operator_ledger) =
        instrumented_canonical_system(&storage, LinearOperatorProperties::General);
    storage.reset_after_legitimate_construction();
    let problem = system.linear_problem().unwrap();
    assert!(std::ptr::addr_eq(
        problem.operator(),
        &system as &dyn LinearOperator
    ));
    let (reference, faer_bicgstab, faer_sparse_lu) = backends();
    let catalog = exact_catalog(&reference, &faer_bicgstab, &faer_sparse_lu);
    let mutations = [
        plan_with_controls(
            LinearSolver::SparseLu,
            PreconditionerPolicy::Identity,
            ReductionPolicy::Fast,
            f64::from_bits(1.0e-12_f64.to_bits() + 1),
            1.0e-14,
            100,
        ),
        plan_with_controls(
            LinearSolver::SparseLu,
            PreconditionerPolicy::Identity,
            ReductionPolicy::Fast,
            1.0e-12,
            f64::from_bits(1.0e-14_f64.to_bits() + 1),
            100,
        ),
        plan_with_controls(
            LinearSolver::SparseLu,
            PreconditionerPolicy::Identity,
            ReductionPolicy::Fast,
            1.0e-12,
            1.0e-14,
            99,
        ),
    ];
    for mutated in mutations {
        actual_operator_ledger.reset();
        let candidates = [
            catalog[0],
            HostSerialSolverCandidate::new(
                FAER_SPARSE_LU_ID,
                FAER_EVIDENCE,
                LinearSolveRequest::new(&faer_sparse_lu, mutated),
            ),
            catalog[2],
        ];
        let error =
            resolve_host_serial_solver_v2(&problem, SolverPlanningObjective::Fast, &candidates)
                .unwrap_err();
        assert_inventory_failure(&error, "catalog.control-mismatch");
        assert_zero_canonical_ledger(&storage, &actual_operator_ledger);
        assert_zero_backend_solves(&reference, &faer_bicgstab, &faer_sparse_lu);
    }
}

#[test]
fn catalog_validation_precedence_and_exact_plan_tuple_are_frozen() {
    let storage = CountingCsrStorage::full();
    let (system, actual_operator_ledger) =
        instrumented_canonical_system(&storage, LinearOperatorProperties::General);
    storage.reset_after_legitimate_construction();
    let problem = system.linear_problem().unwrap();
    let (reference, faer_bicgstab, faer_sparse_lu) = backends();
    let catalog = exact_catalog(&reference, &faer_bicgstab, &faer_sparse_lu);

    let stale_evidence = [
        catalog[0],
        catalog[1],
        HostSerialSolverCandidate::new(
            FAER_BICGSTAB_ID,
            "numerics.stale-evidence",
            LinearSolveRequest::new(&faer_bicgstab, faer_sparse_lu_plan()),
        ),
    ];
    assert_faer_bicgstab_rejection(&problem, &stale_evidence, "catalog.evidence-mismatch");
    assert_zero_canonical_ledger(&storage, &actual_operator_ledger);
    assert_zero_backend_solves(&reference, &faer_bicgstab, &faer_sparse_lu);

    let stale_evidence_and_provider = CountingBackend::new(
        SolverProvider::new(
            BackendId::new("eqiora.faer"),
            "0.1.0-alpha.1",
            FAER_LIBRARIES,
        ),
        faer_bicgstab_plan(),
    );
    let simultaneous_evidence_provider_failure = [
        catalog[0],
        catalog[1],
        HostSerialSolverCandidate::new(
            FAER_BICGSTAB_ID,
            "numerics.stale-evidence",
            LinearSolveRequest::new(&stale_evidence_and_provider, faer_bicgstab_plan()),
        ),
    ];
    actual_operator_ledger.reset();
    assert_faer_bicgstab_rejection(
        &problem,
        &simultaneous_evidence_provider_failure,
        "catalog.evidence-mismatch",
    );
    assert_zero_canonical_ledger(&storage, &actual_operator_ledger);
    assert_zero_backend_solves(&reference, &faer_bicgstab, &faer_sparse_lu);
    assert_eq!(
        stale_evidence_and_provider.calls(),
        0,
        "simultaneously stale evidence/provider backend solve ledger"
    );
    assert_eq!(stale_evidence_and_provider.operator_apply_calls(), 0);
    assert_eq!(stale_evidence_and_provider.operator_diagonal_calls(), 0);
    assert_eq!(
        stale_evidence_and_provider.received_problem(),
        std::ptr::null()
    );
    assert_eq!(
        stale_evidence_and_provider.received_operator(),
        std::ptr::null()
    );

    for bad_provider in [
        SolverProvider::new(
            BackendId::new("eqiora.faer"),
            "0.1.0-alpha.1",
            FAER_LIBRARIES,
        ),
        SolverProvider::new(
            BackendId::new("eqiora.faer"),
            env!("CARGO_PKG_VERSION"),
            CHANGED_FAER_LIBRARIES,
        ),
        SolverProvider::new(
            BackendId::new("eqiora.faer"),
            env!("CARGO_PKG_VERSION"),
            EMPTY_LIBRARIES,
        ),
        SolverProvider::new(
            BackendId::new("eqiora.faer"),
            env!("CARGO_PKG_VERSION"),
            EXTRA_FAER_LIBRARIES,
        ),
    ] {
        let backend = CountingBackend::new(bad_provider, faer_bicgstab_plan());
        let candidates = [
            catalog[0],
            catalog[1],
            HostSerialSolverCandidate::new(
                FAER_BICGSTAB_ID,
                FAER_EVIDENCE,
                LinearSolveRequest::new(&backend, faer_sparse_lu_plan()),
            ),
        ];
        assert_faer_bicgstab_rejection(&problem, &candidates, "catalog.provider-mismatch");
        assert_zero_canonical_ledger(&storage, &actual_operator_ledger);
        assert_zero_backend_solves(&reference, &faer_bicgstab, &faer_sparse_lu);
        assert_eq!(backend.calls(), 0, "rejected provider backend solve ledger");
    }

    for bad_plan in [
        plan(
            LinearSolver::ConjugateGradient,
            PreconditionerPolicy::Jacobi,
            ReductionPolicy::Fast,
        ),
        plan(
            LinearSolver::BiConjugateGradientStabilized,
            PreconditionerPolicy::Identity,
            ReductionPolicy::Fast,
        ),
        plan(
            LinearSolver::BiConjugateGradientStabilized,
            PreconditionerPolicy::Jacobi,
            ReductionPolicy::Reproducible,
        ),
    ] {
        let candidates = [
            catalog[0],
            catalog[1],
            HostSerialSolverCandidate::new(
                FAER_BICGSTAB_ID,
                FAER_EVIDENCE,
                LinearSolveRequest::new(&faer_bicgstab, bad_plan),
            ),
        ];
        assert_faer_bicgstab_rejection(&problem, &candidates, "catalog.plan-mismatch");
        assert_zero_canonical_ledger(&storage, &actual_operator_ledger);
        assert_zero_backend_solves(&reference, &faer_bicgstab, &faer_sparse_lu);
    }

    let nongeneral_storage = CountingCsrStorage::full();
    let (nongeneral, nongeneral_operator_ledger) = instrumented_canonical_system(
        &nongeneral_storage,
        LinearOperatorProperties::SymmetricPositiveDefinite,
    );
    nongeneral_storage.reset_after_legitimate_construction();
    let plan_before_profile = [
        catalog[0],
        catalog[1],
        HostSerialSolverCandidate::new(
            FAER_BICGSTAB_ID,
            FAER_EVIDENCE,
            LinearSolveRequest::new(
                &faer_bicgstab,
                plan(
                    LinearSolver::ConjugateGradient,
                    PreconditionerPolicy::Jacobi,
                    ReductionPolicy::Fast,
                ),
            ),
        ),
    ];
    let error = resolve_host_serial_solver_v2(
        &nongeneral.linear_problem().unwrap(),
        SolverPlanningObjective::LowMemory,
        &plan_before_profile,
    )
    .unwrap_err();
    assert_no_admitted_trace(
        &error,
        &[
            (FAER_BICGSTAB_ID, "catalog.plan-mismatch"),
            (FAER_SPARSE_LU_ID, "profile.general-required"),
            (REFERENCE_ID, "profile.general-required"),
        ],
    );
    assert_zero_canonical_ledger(&nongeneral_storage, &nongeneral_operator_ledger);
    assert_zero_backend_solves(&reference, &faer_bicgstab, &faer_sparse_lu);
}

#[test]
fn profile_and_capability_rejections_cannot_be_ranked() {
    let storage = CountingCsrStorage::full();
    let (system, actual_operator_ledger) =
        instrumented_canonical_system(&storage, LinearOperatorProperties::General);
    storage.reset_after_legitimate_construction();
    let problem = system.linear_problem().unwrap();
    let (reference, faer_bicgstab, faer_sparse_lu) = backends();
    let catalog = exact_catalog(&reference, &faer_bicgstab, &faer_sparse_lu);
    let exact = capability(faer_bicgstab_plan());
    let capability_mutants = [
        SolverCapability {
            algorithm: LinearSolver::SparseLu,
            ..exact
        },
        SolverCapability {
            operator_properties: LinearOperatorProperties::SymmetricPositiveDefinite,
            ..exact
        },
        SolverCapability {
            preconditioner: PreconditionerPolicy::Identity,
            ..exact
        },
        SolverCapability {
            reduction: ReductionPolicy::Reproducible,
            ..exact
        },
        SolverCapability {
            scalar_type: ScalarType::F32,
            ..exact
        },
    ];
    for mutated_capability in capability_mutants {
        let omitted = CountingBackend::with_capability(
            FAER_PROVIDER,
            faer_bicgstab_plan(),
            mutated_capability,
        );
        let candidates = [
            catalog[0],
            catalog[1],
            HostSerialSolverCandidate::new(
                FAER_BICGSTAB_ID,
                FAER_EVIDENCE,
                LinearSolveRequest::new(&omitted, faer_bicgstab_plan()),
            ),
        ];
        assert_faer_bicgstab_rejection(&problem, &candidates, "capability.exact-tuple-required");
        assert_zero_canonical_ledger(&storage, &actual_operator_ledger);
        assert_zero_backend_solves(&reference, &faer_bicgstab, &faer_sparse_lu);
        assert_eq!(
            omitted.calls(),
            0,
            "rejected capability backend solve ledger"
        );
    }

    let omitted = CountingBackend::with_capability(
        FAER_PROVIDER,
        faer_bicgstab_plan(),
        SolverCapability {
            scalar_type: ScalarType::F32,
            ..exact
        },
    );
    let candidates = [
        catalog[0],
        catalog[1],
        HostSerialSolverCandidate::new(
            FAER_BICGSTAB_ID,
            FAER_EVIDENCE,
            LinearSolveRequest::new(&omitted, faer_bicgstab_plan()),
        ),
    ];
    let nongeneral_storage = CountingCsrStorage::full();
    let (nongeneral, nongeneral_operator_ledger) = instrumented_canonical_system(
        &nongeneral_storage,
        LinearOperatorProperties::SymmetricPositiveDefinite,
    );
    nongeneral_storage.reset_after_legitimate_construction();
    let error = resolve_host_serial_solver_v2(
        &nongeneral.linear_problem().unwrap(),
        SolverPlanningObjective::LowMemory,
        &candidates,
    )
    .unwrap_err();
    assert_all_rejected(&error, "profile.general-required");
    assert_zero_canonical_ledger(&nongeneral_storage, &nongeneral_operator_ledger);
    assert_zero_backend_solves(&reference, &faer_bicgstab, &faer_sparse_lu);
    assert_eq!(omitted.calls(), 0, "profile-before-capability solve ledger");
}

fn assert_faer_bicgstab_rejection(
    problem: &LinearProblem<'_>,
    candidates: &[HostSerialSolverCandidate<'_>],
    reason: &'static str,
) {
    for objective in OBJECTIVES {
        let decision = resolve_host_serial_solver_v2(problem, objective, candidates).unwrap();
        assert_eq!(
            decision.selected().id(),
            expected_subset_selection(FAER_SPARSE_LU_BIT | REFERENCE_BIT, objective)
        );
        assert_eq!(
            decision.reasons().collect::<Vec<_>>(),
            expected_subset_trace(FAER_SPARSE_LU_BIT | REFERENCE_BIT, objective, reason,)
        );
    }
}

fn no_admitted_message(trace: &[(&str, &str)]) -> String {
    let rendered = trace
        .iter()
        .map(|(candidate_id, reason)| format!("{candidate_id}={reason}"))
        .collect::<Vec<_>>()
        .join(",");
    format!("{POLICY_ID} no admitted candidate; trace=[{rendered}]")
}

fn assert_no_admitted_trace(error: &Diagnostic, trace: &[(&str, &str)]) {
    assert_eq!(error.code(), codes::INVALID_REALIZATION);
    assert_eq!(error.message(), no_admitted_message(trace));
}

fn assert_all_rejected(error: &Diagnostic, reason: &'static str) {
    assert_no_admitted_trace(
        error,
        &[
            (FAER_BICGSTAB_ID, reason),
            (FAER_SPARSE_LU_ID, reason),
            (REFERENCE_ID, reason),
        ],
    );
}

#[test]
fn unsupported_profiles_reject_with_zero_numerical_calls() {
    let (reference, faer_bicgstab, faer_sparse_lu) = backends();
    let catalog = exact_catalog(&reference, &faer_bicgstab, &faer_sparse_lu);

    let nongeneral_storage = CountingCsrStorage::full();
    let (nongeneral, nongeneral_operator_ledger) = instrumented_canonical_system(
        &nongeneral_storage,
        LinearOperatorProperties::SymmetricPositiveDefinite,
    );
    nongeneral_storage.reset_after_legitimate_construction();
    assert_all_rejected(
        &resolve_host_serial_solver_v2(
            &nongeneral.linear_problem().unwrap(),
            SolverPlanningObjective::Robust,
            &catalog,
        )
        .unwrap_err(),
        "profile.general-required",
    );
    assert_zero_canonical_ledger(&nongeneral_storage, &nongeneral_operator_ledger);
    assert_zero_backend_solves(&reference, &faer_bicgstab, &faer_sparse_lu);

    let hand_built_operator = CountingOperator::default();
    let hand_built = LinearProblem::new(
        &hand_built_operator,
        &[6.0, 8.0],
        LinearOperatorProperties::General,
    )
    .unwrap();
    assert_all_rejected(
        &resolve_host_serial_solver_v2(&hand_built, SolverPlanningObjective::Robust, &catalog)
            .unwrap_err(),
        "profile.canonical-csr-required",
    );
    assert_zero_operator_ledger(&hand_built_operator);
    assert_zero_backend_solves(&reference, &faer_bicgstab, &faer_sparse_lu);

    let matrix_free_operator = CountingOperator::matrix_free();
    let matrix_free = LinearProblem::new(
        &matrix_free_operator,
        &[6.0, 8.0],
        LinearOperatorProperties::General,
    )
    .unwrap();
    assert_all_rejected(
        &resolve_host_serial_solver_v2(&matrix_free, SolverPlanningObjective::Fast, &catalog)
            .unwrap_err(),
        "profile.canonical-csr-required",
    );
    assert_zero_operator_ledger(&matrix_free_operator);
    assert_zero_backend_solves(&reference, &faer_bicgstab, &faer_sparse_lu);

    let transposed_source = CountingOperator::default();
    let transposed_operator = Transposed::new(&transposed_source);
    let transposed = LinearProblem::new(
        &transposed_operator,
        &[6.0, 8.0],
        LinearOperatorProperties::General,
    )
    .unwrap();
    let transposed_error =
        resolve_host_serial_solver_v2(&transposed, SolverPlanningObjective::Robust, &catalog)
            .unwrap_err();
    let normal_trace = [
        (FAER_BICGSTAB_ID, "profile.normal-required"),
        (FAER_SPARSE_LU_ID, "profile.normal-required"),
        (REFERENCE_ID, "profile.normal-required"),
    ];
    let canonical_trace = [
        (FAER_BICGSTAB_ID, "profile.canonical-csr-required"),
        (FAER_SPARSE_LU_ID, "profile.canonical-csr-required"),
        (REFERENCE_ID, "profile.canonical-csr-required"),
    ];
    assert_eq!(transposed_error.code(), codes::INVALID_REALIZATION);
    assert!(
        [
            no_admitted_message(&normal_trace),
            no_admitted_message(&canonical_trace),
        ]
        .iter()
        .any(|expected| transposed_error.message() == expected),
        "transposed hand-built input must reject at one of the unordered profile sub-gates: {}",
        transposed_error.message()
    );
    assert_zero_operator_ledger(&transposed_source);
    assert_zero_backend_solves(&reference, &faer_bicgstab, &faer_sparse_lu);

    for missing_storage in [
        CountingCsrStorage::missing_first_diagonal(),
        CountingCsrStorage::missing_second_diagonal(),
    ] {
        let (missing, missing_operator_ledger) =
            instrumented_canonical_system(&missing_storage, LinearOperatorProperties::General);
        missing_storage.reset_after_legitimate_construction();
        let problem = missing.linear_problem().unwrap();
        let decision =
            resolve_host_serial_solver_v2(&problem, SolverPlanningObjective::Robust, &catalog)
                .unwrap();
        assert_eq!(decision.selected().id(), FAER_SPARSE_LU_ID);
        assert_eq!(
            decision.reasons().collect::<Vec<_>>(),
            vec![
                (FAER_BICGSTAB_ID, "profile.complete-diagonal-required"),
                (FAER_SPARSE_LU_ID, "candidate.admitted"),
                (FAER_SPARSE_LU_ID, "candidate.selected.robust-preference"),
                (REFERENCE_ID, "profile.complete-diagonal-required"),
            ]
        );
        assert_zero_canonical_ledger(&missing_storage, &missing_operator_ledger);
        assert_zero_backend_solves(&reference, &faer_bicgstab, &faer_sparse_lu);
    }
}

#[test]
fn no_admitted_diagnostic_freezes_the_complete_ordered_trace() {
    let storage = CountingCsrStorage::full();
    let (system, actual_operator_ledger) =
        instrumented_canonical_system(&storage, LinearOperatorProperties::General);
    storage.reset_after_legitimate_construction();
    let problem = system.linear_problem().unwrap();
    let (reference, faer_bicgstab, faer_sparse_lu) = backends();
    let catalog = exact_catalog(&reference, &faer_bicgstab, &faer_sparse_lu);
    let candidates = catalog.map(|candidate| {
        HostSerialSolverCandidate::new(
            candidate.id(),
            "numerics.stale-evidence",
            candidate.request(),
        )
    });
    for objective in OBJECTIVES {
        let error = resolve_host_serial_solver_v2(&problem, objective, &candidates).unwrap_err();
        assert_all_rejected(&error, "catalog.evidence-mismatch");
        assert_zero_canonical_ledger(&storage, &actual_operator_ledger);
        assert_zero_backend_solves(&reference, &faer_bicgstab, &faer_sparse_lu);
    }
}

fn assert_execution_ledger(
    objective: SolverPlanningObjective,
    problem: &LinearProblem<'_>,
    reference: &CountingBackend,
    faer_bicgstab: &CountingBackend,
    faer_sparse_lu: &CountingBackend,
    selected_operator_apply_calls: usize,
) {
    let expected_selected = expected_selection(objective);
    let expected_problem = std::ptr::from_ref(problem).cast::<()>();
    let expected_operator = std::ptr::from_ref(problem.operator()).cast::<()>();
    for (candidate_id, backend) in [
        (REFERENCE_ID, reference),
        (FAER_BICGSTAB_ID, faer_bicgstab),
        (FAER_SPARSE_LU_ID, faer_sparse_lu),
    ] {
        if candidate_id == expected_selected {
            assert_eq!(backend.calls(), 1, "selected backend exact call ledger");
            assert!(
                std::ptr::eq(backend.received_problem(), expected_problem),
                "selected backend must receive the exact problem borrowed by resolution"
            );
            assert!(
                std::ptr::eq(backend.received_operator(), expected_operator),
                "selected backend must receive the exact owned canonical operator"
            );
            assert_eq!(
                backend.operator_apply_calls(),
                selected_operator_apply_calls,
                "selected backend operator apply ledger"
            );
        } else {
            assert_eq!(backend.calls(), 0, "unselected backend call ledger");
            assert_eq!(
                backend.received_problem(),
                std::ptr::null(),
                "an unselected backend must receive no problem"
            );
            assert_eq!(
                backend.received_operator(),
                std::ptr::null(),
                "an unselected backend must receive no operator"
            );
            assert_eq!(
                backend.operator_apply_calls(),
                0,
                "unselected backend operator apply ledger"
            );
        }
        assert_eq!(
            backend.operator_diagonal_calls(),
            0,
            "backend operator diagonal ledger"
        );
    }
}

#[test]
fn selected_failure_executes_once_without_retry_plan_mutation_or_problem_substitution() {
    for objective in OBJECTIVES {
        let storage = CountingCsrStorage::full();
        let (system, actual_operator_ledger) =
            instrumented_canonical_system(&storage, LinearOperatorProperties::General);
        storage.reset_after_legitimate_construction();
        let problem = system.linear_problem().unwrap();
        let (reference, faer_bicgstab, faer_sparse_lu) = backends();
        let catalog = exact_catalog(&reference, &faer_bicgstab, &faer_sparse_lu);
        let decision = resolve_host_serial_solver_v2(&problem, objective, &catalog).unwrap();
        assert_eq!(decision.selected().id(), expected_selection(objective));
        assert!(std::ptr::eq(decision.problem(), &problem));
        assert!(std::ptr::eq(
            decision.problem().operator(),
            problem.operator()
        ));
        assert_zero_canonical_ledger(&storage, &actual_operator_ledger);
        assert_zero_backend_solves(&reference, &faer_bicgstab, &faer_sparse_lu);
        reset_backend_operator_ledgers_before_execution(
            &reference,
            &faer_bicgstab,
            &faer_sparse_lu,
        );

        let error = decision.solve().unwrap_err();
        assert_eq!(error.code(), codes::NUMERICAL_SOLVE_FAILED);
        assert_execution_ledger(
            objective,
            &problem,
            &reference,
            &faer_bicgstab,
            &faer_sparse_lu,
            0,
        );
        assert_zero_canonical_ledger(&storage, &actual_operator_ledger);
    }
}

#[test]
fn selected_success_executes_once_and_applies_exact_problem_for_both_true_residuals() {
    for objective in OBJECTIVES {
        let storage = CountingCsrStorage::full();
        let (system, actual_operator_ledger) =
            instrumented_canonical_system(&storage, LinearOperatorProperties::General);
        storage.reset_after_legitimate_construction();
        let problem = system.linear_problem().unwrap();
        let reference = CountingBackend::successful(REFERENCE_PROVIDER, reference_plan());
        let faer_bicgstab = CountingBackend::successful(FAER_PROVIDER, faer_bicgstab_plan());
        let faer_sparse_lu = CountingBackend::successful(FAER_PROVIDER, faer_sparse_lu_plan());
        let catalog = exact_catalog(&reference, &faer_bicgstab, &faer_sparse_lu);
        let decision = resolve_host_serial_solver_v2(&problem, objective, &catalog).unwrap();
        assert_eq!(decision.selected().id(), expected_selection(objective));
        assert!(std::ptr::eq(decision.problem(), &problem));
        assert!(std::ptr::eq(
            decision.problem().operator(),
            problem.operator()
        ));
        assert_zero_canonical_ledger(&storage, &actual_operator_ledger);
        assert_zero_backend_solves(&reference, &faer_bicgstab, &faer_sparse_lu);
        reset_backend_operator_ledgers_before_execution(
            &reference,
            &faer_bicgstab,
            &faer_sparse_lu,
        );

        let solution = decision.solve().unwrap();
        assert_eq!(solution.values(), &[1.0, 2.0]);
        assert_execution_ledger(
            objective,
            &problem,
            &reference,
            &faer_bicgstab,
            &faer_sparse_lu,
            2,
        );
        assert_zero_canonical_source_ledger(&storage);
        assert_actual_canonical_operator_ledger(&actual_operator_ledger, 2, 0);
    }
}

#[test]
fn registered_host_serial_planning_oracle_executes_all_private_falsifiers() {
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
    assert_eq!(checks.len(), 10, "the frozen private oracle inventory");
    for check in checks {
        check();
    }
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
            &faer,
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
            &faer,
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
            &faer,
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
            &faer,
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
        &faer,
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
