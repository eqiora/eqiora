use eqiora_core::ScalarType;
use std::collections::BTreeSet;
use std::f64::consts::PI;
use std::num::NonZeroUsize;

use eqiora::Diagnostic;
use eqiora::compiler::compile;
use eqiora::graph::{GraphStore, InMemoryGraphStore};
use eqiora::realization::{
    DefaultPolicyVersion, DiscretizationMethod, ExecutionSchedule, RealizationCapabilities,
    RealizationPlan, RealizationRequest, RealizationRequirements, SemanticRevision,
    SpatialDimensionSupport, Target, TargetCapabilities, VectorLayoutKind, default_plan_v0,
    resolve,
};
use eqiora::sem::KernelProgram;
use eqiora::solver::{
    CanonicalCsrSystemView, CompleteCsrStorage, ConvergenceReason, DiagonalAvailability,
    LinearOperator, LinearOperatorProperties, LinearProblem, LinearSolver, LinearSolverBackend,
    PreconditionerPolicy, REFERENCE_LINEAR_SOLVER, ReductionPolicy, SERIAL_EXECUTION_PROVIDER,
    SolverCapabilities, SolverCapability, SolverPlan, Transposed,
};
use eqiora_backend_faer::{
    FAER_ADAPTER_VERSION, FAER_SOLVER_PROVIDER, FAER_VERSION, FaerLinearSolver,
};
use eqiora_execution::{AdmittedExecution, DeploymentBinding, HostExecutorDescriptor};
use eqiora_numerics::{
    scalar::ResolvedScalarEllipticCartesianSolution, scalar::ResolvedScalarEllipticSolution1d,
    scalar::finalize_resolved_scalar_elliptic_cartesian, scalar::solve_resolved_scalar_elliptic_1d,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};

const SOURCE: &str = include_str!("../../../verify/numerics/poisson-fem-fvm/models/poisson.eqi");
const SPARSE_LU_CONTRACT: &str =
    include_str!("../../../verify/numerics/linear-backends/expected/sparse-lu-contract.json");
const SPARSE_LU_CONTRACT_SHA256: &str =
    "666309634cca3d6be5d16d8e90e6ad01d0b92694cbb70fd03acce38ef8e98780";

#[test]
fn faer_sparse_lu_source_keeps_serial_and_residual_handoffs() {
    let source = format!(
        "{}\n{}",
        include_str!("../../eqiora-backend-faer/src/sparse_lu.rs"),
        include_str!("../../eqiora-backend-faer/src/sparse_lu_factor.rs"),
    );
    let normalized_source = source.split_whitespace().collect::<Vec<_>>().join(" ");
    for process_global_wrapper in [
        ".sp_lu(",
        ".sp_qr(",
        ".sp_cholesky(",
        ".sp_solve_lower_triangular_in_place(",
        ".sp_solve_upper_triangular_in_place(",
        ".sp_solve_unit_lower_triangular_in_place(",
        ".sp_solve_unit_upper_triangular_in_place(",
    ] {
        assert!(
            !source.contains(process_global_wrapper),
            "sparse LU must not call {process_global_wrapper}"
        );
    }
    assert!(source.contains("factorize_numeric_lu("));
    assert!(source.contains("solve_in_place_with_conj("));
    assert!(source.contains("let parallelism = Par::Seq;"));
    assert!(!source.contains("Par::Rayon"));
    assert!(!source.contains("Par::rayon"));
    assert!(normalized_source.contains(
        "column_matrix.as_ref(), parallelism, MemStack::new(&mut buffer), Default::default(),"
    ));
    assert!(normalized_source.contains(
        "LuRef::new_unchecked(&symbolic.factor, &numeric.factor).solve_in_place_with_conj( Conj::No, output.as_mut(), parallelism, MemStack::new(&mut buffer), );"
    ));
    assert!(
        normalized_source
            .contains("let reported_residual_norm = fixed_residual_norm(problem, &values)?;")
    );
    assert!(normalized_source.contains(
        "ConvergenceReason::ResidualToleranceSatisfied, 1, reported_residual_norm, values,"
    ));
}

#[test]
fn canonical_poisson_agrees_between_reference_and_faer_backends() {
    let program = compile_program();
    let requirements = RealizationRequirements::new(
        NonZeroUsize::MIN,
        ScalarType::F64,
        VectorLayoutKind::Replicated,
    );
    let semantic_revision = SemanticRevision::new(program.revision().0);

    let reference = resolve(
        &RealizationRequest::default(program.model(), semantic_revision, DefaultPolicyVersion::V0),
        requirements,
        &RealizationCapabilities::scalar_elliptic_reference(),
    )
    .unwrap();
    let (_, reference_solution) =
        solve_resolved_scalar_elliptic_1d(&program, &reference, &REFERENCE_LINEAR_SOLVER).unwrap();

    let default = default_plan_v0().unwrap();
    let faer_plan = RealizationPlan::new(
        default.space(),
        default.discretization(),
        default.solver().with_reduction(ReductionPolicy::Fast),
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
        ExecutionSchedule::Offline,
    )
    .unwrap();
    let faer_capabilities = RealizationCapabilities::cartesian_product(
        [DiscretizationMethod::ContinuousGalerkin],
        [(
            eqiora::realization::MeshKind::GeneratedCartesian,
            SpatialDimensionSupport::exact(NonZeroUsize::MIN),
        )],
        [VectorLayoutKind::Replicated],
        FaerLinearSolver.capabilities(),
        TargetCapabilities::none().with_host_cpu(NonZeroUsize::MIN),
    )
    .unwrap();
    let faer = resolve(
        &RealizationRequest::explicit(
            program.model(),
            semantic_revision,
            eqiora::realization::RealizationRevision::new(1),
            faer_plan,
        ),
        requirements,
        &faer_capabilities,
    )
    .unwrap();
    let (_, finalized) = finalize_resolved_scalar_elliptic_cartesian(&program, &faer).unwrap();
    let binding = DeploymentBinding::bind_host(
        finalized.portable_realization(),
        HostExecutorDescriptor::new(
            FAER_SOLVER_PROVIDER,
            SERIAL_EXECUTION_PROVIDER,
            NonZeroUsize::MIN,
            FaerLinearSolver.capabilities(),
        ),
    )
    .unwrap();
    let admitted = AdmittedExecution::admit_host_linear(
        finalized.portable_realization(),
        finalized.canonical_csr_system_view(),
        binding,
    )
    .unwrap();
    let produced = FaerLinearSolver
        .solve(
            &finalized.linear_problem().unwrap(),
            finalized.solver_plan(),
        )
        .unwrap();
    let accepted = admitted.accept(produced).unwrap();
    let (linear_solution, receipt) = accepted.into_parts();
    let faer_solution = finalized.finish(linear_solution).unwrap();
    assert_eq!(receipt.solver_provider(), FAER_SOLVER_PROVIDER);
    assert_eq!(receipt.execution_provider(), SERIAL_EXECUTION_PROVIDER);
    assert_eq!(
        receipt.report().solver_provider(),
        receipt.binding().solver_provider()
    );
    assert_eq!(
        receipt.report().execution_provider(),
        receipt.binding().execution_provider()
    );
    assert_eq!(
        receipt.solver_provider().implementation_version(),
        FAER_ADAPTER_VERSION
    );
    assert_eq!(receipt.solver_provider().libraries().len(), 1);
    assert_eq!(receipt.solver_provider().libraries()[0].name(), "faer");
    assert_eq!(
        receipt.solver_provider().libraries()[0].version(),
        FAER_VERSION
    );

    let ResolvedScalarEllipticSolution1d::FiniteElement(reference) = reference_solution else {
        panic!("the reference plan selects finite elements");
    };
    let ResolvedScalarEllipticCartesianSolution::FiniteElement(faer) = faer_solution else {
        panic!("both plans select the same finite-element realization");
    };
    for (reference, faer) in reference
        .field()
        .values()
        .iter()
        .zip(faer.field().vertex_values())
    {
        assert!((reference - faer).abs() < 5.0e-13);
    }
    assert!(reference.residual_norm() < 1.0e-11);
    assert!(faer.solve_report().true_residual_norm() < 1.0e-11);
    let midpoint = faer.field().vertex_values()[faer.field().vertex_values().len() / 2];
    assert!(((PI * 0.5).sin() - midpoint).abs() < 4.0e-3);
    assert_eq!(
        SolverCapabilities::reference().reductions(),
        REFERENCE_LINEAR_SOLVER.capabilities().reductions()
    );
}

#[test]
fn faer_bicgstab_jacobi_and_exact_capability_boundary_are_registered() {
    let oracle = sparse_lu_oracle();
    let capabilities = FaerLinearSolver.capabilities();
    let mut expected = BTreeSet::from([
        SolverCapability {
            algorithm: LinearSolver::ConjugateGradient,
            operator_properties: LinearOperatorProperties::SymmetricPositiveDefinite,
            preconditioner: PreconditionerPolicy::Identity,
            reduction: ReductionPolicy::Fast,
            scalar_type: ScalarType::F64,
        },
        SolverCapability {
            algorithm: LinearSolver::ConjugateGradient,
            operator_properties: LinearOperatorProperties::SymmetricPositiveDefinite,
            preconditioner: PreconditionerPolicy::Jacobi,
            reduction: ReductionPolicy::Fast,
            scalar_type: ScalarType::F64,
        },
        SolverCapability {
            algorithm: LinearSolver::BiConjugateGradientStabilized,
            operator_properties: LinearOperatorProperties::General,
            preconditioner: PreconditionerPolicy::Identity,
            reduction: ReductionPolicy::Fast,
            scalar_type: ScalarType::F64,
        },
        SolverCapability {
            algorithm: LinearSolver::BiConjugateGradientStabilized,
            operator_properties: LinearOperatorProperties::General,
            preconditioner: PreconditionerPolicy::Jacobi,
            reduction: ReductionPolicy::Fast,
            scalar_type: ScalarType::F64,
        },
    ]);
    expected.extend(
        oracle
            .contract_expectations
            .capabilities
            .positive
            .iter()
            .map(capability_from_fixture),
    );
    assert_eq!(capabilities.combinations(), &expected);
    for unsupported in &oracle.contract_expectations.capabilities.negative {
        let tuple = capability_from_fixture(unsupported);
        let plan = sparse_lu_plan(&oracle)
            .with_preconditioner(tuple.preconditioner)
            .with_reduction(tuple.reduction);
        capabilities
            .require_problem(plan, tuple.scalar_type, tuple.operator_properties)
            .unwrap_err();
    }

    let operator = DenseOperator {
        entries: [[4.0, 1.0], [2.0, 3.0]],
    };
    let problem =
        LinearProblem::new(&operator, &[6.0, 8.0], LinearOperatorProperties::General).unwrap();
    let plan = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1.0e-12,
        1.0e-14,
        NonZeroUsize::new(100).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Jacobi)
    .with_reduction(ReductionPolicy::Fast);
    let solution = FaerLinearSolver.solve(&problem, plan).unwrap();
    assert!((solution.values()[0] - 1.0).abs() < 1.0e-12);
    assert!((solution.values()[1] - 2.0).abs() < 1.0e-12);
    assert!(solution.report().true_residual_norm() <= solution.report().residual_target());

    for (unsupported, properties) in [
        (
            plan.with_preconditioner(PreconditionerPolicy::Identity),
            LinearOperatorProperties::SymmetricPositiveDefinite,
        ),
        (
            plan.with_reduction(ReductionPolicy::Reproducible),
            LinearOperatorProperties::General,
        ),
    ] {
        let error = capabilities
            .require_problem(unsupported, ScalarType::F64, properties)
            .unwrap_err();
        assert!(error.message().contains("exact"));
    }
}

#[test]
fn faer_sparse_lu_matches_the_precommitted_exact_rational_oracle() {
    run_sparse_lu_oracle();
    let oracle = sparse_lu_oracle();
    let fixture_digest = Sha256::digest(SPARSE_LU_CONTRACT.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    assert_eq!(fixture_digest, SPARSE_LU_CONTRACT_SHA256);
    let plan = sparse_lu_plan(&oracle);
    let expected = rational_vector(&oracle.mathematics.principal.solution);
    let principal = fixture_storage(&oracle.mathematics.principal);
    let system =
        CanonicalCsrSystemView::new(&principal, LinearOperatorProperties::General).unwrap();

    let solution = FaerLinearSolver
        .solve(&system.linear_problem().unwrap(), plan)
        .unwrap();
    let positive_case = oracle
        .contract_expectations
        .test_plan
        .cases
        .iter()
        .find(|case| case.id == "principal-positive-solve")
        .unwrap();
    assert!(!positive_case.expected_early_exit);
    assert_eq!(solution.report().algorithm(), LinearSolver::SparseLu);
    assert_eq!(
        solution.report().reason(),
        ConvergenceReason::ResidualToleranceSatisfied
    );
    assert_eq!(solution.report().completed_iterations(), 1);
    assert_eq!(
        solution.report().reported_residual_norm(),
        solution.report().true_residual_norm()
    );
    assert_solution_matches_case(solution.values(), &expected, positive_case);
    assert!(
        solution.report().true_residual_norm().powi(2)
            <= rational(positive_case.expected_residual_squared_at_most)
    );

    let initial = rational_vector(
        &oracle
            .mathematics
            .principal
            .initial_guesses
            .already_satisfied
            .vector,
    );
    let early_problem = system
        .linear_problem()
        .unwrap()
        .with_initial_guess(&initial)
        .unwrap();
    let early_case = oracle
        .contract_expectations
        .test_plan
        .cases
        .iter()
        .find(|case| case.id == "early-initial-guess-accepted")
        .unwrap();
    assert!(early_case.expected_early_exit);
    let early = FaerLinearSolver.solve(&early_problem, plan).unwrap();
    assert_eq!(
        early.report().reason(),
        ConvergenceReason::InitialResidualSatisfied
    );
    assert_eq!(early.report().completed_iterations(), 0);
    assert_eq!(early.values(), initial);

    let not_satisfied = rational_vector(
        &oracle
            .mathematics
            .principal
            .initial_guesses
            .not_satisfied
            .vector,
    );
    let not_satisfied_problem = system
        .linear_problem()
        .unwrap()
        .with_initial_guess(&not_satisfied)
        .unwrap();
    let not_satisfied_solution = FaerLinearSolver
        .solve(&not_satisfied_problem, plan)
        .unwrap();
    let not_satisfied_case = oracle
        .contract_expectations
        .test_plan
        .cases
        .iter()
        .find(|case| case.id == "initial-guess-not-accepted-early")
        .unwrap();
    assert!(!not_satisfied_case.expected_early_exit);
    assert_eq!(
        not_satisfied_solution.report().reason(),
        ConvergenceReason::ResidualToleranceSatisfied
    );
    assert_eq!(not_satisfied_solution.report().completed_iterations(), 1);
    assert_eq!(
        not_satisfied_solution.report().reported_residual_norm(),
        not_satisfied_solution.report().true_residual_norm()
    );
    assert_solution_matches_case(
        not_satisfied_solution.values(),
        &expected,
        not_satisfied_case,
    );
    assert!(
        not_satisfied_solution.report().true_residual_norm().powi(2)
            <= rational(not_satisfied_case.expected_residual_squared_at_most)
    );

    let hand_built = LinearProblem::new(
        &system,
        system.right_hand_side(),
        LinearOperatorProperties::General,
    )
    .unwrap();
    let missing_capture = FaerLinearSolver.solve(&hand_built, plan).unwrap_err();
    assert_eq!(
        missing_capture.code(),
        eqiora::diagnostic::codes::INVALID_REALIZATION
    );

    let matrix_free_operator = DenseOperator {
        entries: [[4.0, 1.0], [2.0, 3.0]],
    };
    let matrix_free = LinearProblem::new(
        &matrix_free_operator,
        &[6.0, 8.0],
        LinearOperatorProperties::General,
    )
    .unwrap();
    let matrix_free_error = FaerLinearSolver.solve(&matrix_free, plan).unwrap_err();
    assert_eq!(
        matrix_free_error.code(),
        eqiora::diagnostic::codes::INVALID_REALIZATION
    );

    let transposed_operator = Transposed::new(&system);
    let transposed = LinearProblem::new(
        &transposed_operator,
        system.right_hand_side(),
        LinearOperatorProperties::General,
    )
    .unwrap();
    let transposed_error = FaerLinearSolver.solve(&transposed, plan).unwrap_err();
    assert_eq!(
        transposed_error.code(),
        eqiora::diagnostic::codes::INVALID_REALIZATION
    );
    assert!(transposed_error.message().contains("normal-orientation"));

    let rank_deficient = fixture_storage(&oracle.mathematics.rank_deficient);
    let rank_deficient =
        CanonicalCsrSystemView::new(&rank_deficient, LinearOperatorProperties::General).unwrap();
    let rank_deficient_case = oracle
        .contract_expectations
        .test_plan
        .cases
        .iter()
        .find(|case| case.id == "rank-deficient-fail-closed")
        .unwrap();
    assert!(!rank_deficient_case.expected_early_exit);
    let rejected = FaerLinearSolver
        .solve(&rank_deficient.linear_problem().unwrap(), plan)
        .unwrap_err();
    assert_eq!(
        rejected.code(),
        eqiora::diagnostic::codes::NUMERICAL_SOLVE_FAILED
    );
}

#[derive(Debug)]
struct DenseOperator {
    entries: [[f64; 2]; 2],
}

impl LinearOperator for DenseOperator {
    fn rows(&self) -> usize {
        2
    }

    fn columns(&self) -> usize {
        2
    }

    fn apply(&self, input: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        for (row, target) in self.entries.iter().zip(output) {
            *target = row
                .iter()
                .zip(input)
                .map(|(entry, value)| entry * value)
                .sum();
        }
        Ok(())
    }

    fn diagonal(&self, output: &mut [f64]) -> Result<DiagonalAvailability, Diagnostic> {
        output.copy_from_slice(&[self.entries[0][0], self.entries[1][1]]);
        Ok(DiagonalAvailability::Available)
    }
}

fn compile_program() -> KernelProgram {
    let mut compiled =
        compile("verify/numerics/poisson-fem-fvm/models/poisson.eqi", SOURCE).unwrap();
    let (transaction, model, _) = compiled.remove(0).into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    KernelProgram::from_snapshot(&store.snapshot(), model).unwrap()
}

fn sparse_lu_oracle() -> SparseLuOracle {
    serde_json::from_str(SPARSE_LU_CONTRACT).expect("the frozen sparse-LU oracle is valid JSON")
}

fn run_sparse_lu_oracle() {
    let repository = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .expect("eqiora crate belongs to the workspace root");
    let python = std::env::var_os("PYTHON").unwrap_or_else(|| "python3".into());
    let output = std::process::Command::new(python)
        .current_dir(repository)
        .arg("verify/numerics/linear-backends/oracle/sparse_lu_oracle.py")
        .arg("--expect-digest")
        .arg(SPARSE_LU_CONTRACT_SHA256)
        .arg("--summary")
        .output()
        .expect("the registered sparse-LU case requires Python 3");
    assert!(
        output.status.success(),
        "sparse-LU oracle failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn sparse_lu_plan(oracle: &SparseLuOracle) -> SolverPlan {
    let plan = &oracle.contract_expectations.test_plan.plan;
    assert_eq!(plan.solver, "SparseLu");
    assert_eq!(plan.operator_property, "General");
    assert_eq!(plan.preconditioner, "Identity");
    assert_eq!(plan.reduction, "Fast");
    assert_eq!(plan.scalar, "F64");
    SolverPlan::new(
        LinearSolver::SparseLu,
        rational(plan.relative_tolerance),
        rational(plan.absolute_tolerance),
        NonZeroUsize::new(plan.maximum_iterations).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Identity)
    .with_reduction(ReductionPolicy::Fast)
}

fn capability_from_fixture(value: &FrozenCapability) -> SolverCapability {
    assert_eq!(value.solver, "SparseLu");
    assert_eq!(value.scalar, "F64");
    SolverCapability {
        algorithm: LinearSolver::SparseLu,
        operator_properties: match value.operator_property.as_str() {
            "General" => LinearOperatorProperties::General,
            "SymmetricPositiveDefinite" => LinearOperatorProperties::SymmetricPositiveDefinite,
            "SymmetricIndefinite" => LinearOperatorProperties::SymmetricIndefinite,
            other => panic!("unexpected frozen operator property {other}"),
        },
        preconditioner: match value.preconditioner.as_str() {
            "Identity" => PreconditionerPolicy::Identity,
            "Jacobi" => PreconditionerPolicy::Jacobi,
            other => panic!("unexpected frozen preconditioner {other}"),
        },
        reduction: match value.reduction.as_str() {
            "Fast" => ReductionPolicy::Fast,
            "Reproducible" => ReductionPolicy::Reproducible,
            other => panic!("unexpected frozen reduction {other}"),
        },
        scalar_type: ScalarType::F64,
    }
}

fn fixture_storage(system: &FrozenSystem) -> FixtureStorage {
    assert_eq!(system.n + 1, system.csr.row_ptr.len());
    FixtureStorage {
        n: system.n,
        row_offsets: system.csr.row_ptr.clone(),
        column_indices: system.csr.col_idx.clone(),
        values: rational_vector(&system.csr.values),
        right_hand_side: rational_vector(&system.rhs),
    }
}

fn assert_solution_matches_case(actual: &[f64], expected: &[f64], case: &FrozenCase) {
    let ceiling = rational(
        case.componentwise_solution_error_ceiling
            .expect("accepted case freezes a componentwise ceiling"),
    );
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() <= ceiling);
    }
}

fn rational_vector(values: &[FrozenRational]) -> Vec<f64> {
    values.iter().copied().map(rational).collect()
}

fn rational(value: FrozenRational) -> f64 {
    let converted = value.num as f64 / value.den as f64;
    assert_eq!(converted * value.den as f64, value.num as f64);
    converted
}

#[derive(Debug)]
struct FixtureStorage {
    n: usize,
    row_offsets: Vec<usize>,
    column_indices: Vec<usize>,
    values: Vec<f64>,
    right_hand_side: Vec<f64>,
}

impl CompleteCsrStorage for FixtureStorage {
    fn rows(&self) -> usize {
        self.n
    }

    fn columns(&self) -> usize {
        self.n
    }

    fn row_offsets(&self) -> &[usize] {
        &self.row_offsets
    }

    fn column_indices(&self) -> &[usize] {
        &self.column_indices
    }

    fn values(&self) -> &[f64] {
        &self.values
    }

    fn right_hand_side(&self) -> &[f64] {
        &self.right_hand_side
    }
}

#[derive(Debug, Deserialize)]
struct SparseLuOracle {
    mathematics: FrozenMathematics,
    contract_expectations: FrozenContractExpectations,
}

#[derive(Debug, Deserialize)]
struct FrozenMathematics {
    principal: FrozenPrincipal,
    rank_deficient: FrozenSystem,
}

#[derive(Debug, Deserialize)]
struct FrozenPrincipal {
    #[serde(flatten)]
    system: FrozenSystem,
    solution: Vec<FrozenRational>,
    initial_guesses: FrozenInitialGuesses,
}

impl std::ops::Deref for FrozenPrincipal {
    type Target = FrozenSystem;

    fn deref(&self) -> &Self::Target {
        &self.system
    }
}

#[derive(Debug, Deserialize)]
struct FrozenSystem {
    n: usize,
    csr: FrozenCsr,
    rhs: Vec<FrozenRational>,
}

#[derive(Debug, Deserialize)]
struct FrozenCsr {
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    values: Vec<FrozenRational>,
}

#[derive(Debug, Deserialize)]
struct FrozenInitialGuesses {
    already_satisfied: FrozenInitialGuess,
    not_satisfied: FrozenInitialGuess,
}

#[derive(Debug, Deserialize)]
struct FrozenInitialGuess {
    vector: Vec<FrozenRational>,
}

#[derive(Debug, Deserialize)]
struct FrozenContractExpectations {
    capabilities: FrozenCapabilities,
    test_plan: FrozenTestPlan,
}

#[derive(Debug, Deserialize)]
struct FrozenCapabilities {
    positive: Vec<FrozenCapability>,
    negative: Vec<FrozenCapability>,
}

#[derive(Debug, Deserialize)]
struct FrozenCapability {
    solver: String,
    operator_property: String,
    preconditioner: String,
    reduction: String,
    scalar: String,
}

#[derive(Debug, Deserialize)]
struct FrozenTestPlan {
    plan: FrozenPlan,
    cases: Vec<FrozenCase>,
}

#[derive(Debug, Deserialize)]
struct FrozenPlan {
    solver: String,
    operator_property: String,
    preconditioner: String,
    reduction: String,
    scalar: String,
    relative_tolerance: FrozenRational,
    absolute_tolerance: FrozenRational,
    maximum_iterations: usize,
}

#[derive(Debug, Deserialize)]
struct FrozenCase {
    id: String,
    expected_residual_squared_at_most: FrozenRational,
    componentwise_solution_error_ceiling: Option<FrozenRational>,
    expected_early_exit: bool,
}

#[derive(Debug, Clone, Copy, Deserialize)]
struct FrozenRational {
    num: i64,
    den: u64,
}
