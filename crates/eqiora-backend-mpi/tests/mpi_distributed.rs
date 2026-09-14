#![cfg(feature = "mpi-runtime")]

use std::collections::BTreeSet;
use std::env;
use std::io::Read;
use std::num::NonZeroUsize;
use std::process::{Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use eqiora_backend_mpi::{
    CollectivePhaseV1, MPI_DISTRIBUTED_KRYLOV_BACKEND, MPI_EXECUTION, MpiExecutionGroup,
    MpiRankLocalCsrAction, MpiThreadSupport, RankLocalDeviceV1,
};
use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, ScalarType};
use eqiora_distributed::{
    DistributedLinearSystem, GlobalVectorSpace, LocalCsrShard, Partition, PartitionId,
};
use eqiora_solver::{
    CanonicalCsrSystemView, CompleteCsrStorage, ExecutionReport, LinearOperatorProperties,
    LinearSolver, PreconditionerPolicy, ReductionPolicy, SolverCapability, SolverPlan,
};
use mpi::Threading;
use mpi::traits::CommunicatorCollectives;

const CHILD_ENV: &str = "EQIORA_MPI_TEST_CHILD";
#[cfg(feature = "mpi-test-hooks")]
const FAULT_ENV: &str = "EQIORA_MPI_TEST_FAULT";
#[cfg(feature = "mpi-test-hooks")]
const FAULT_SOLVER_ENV: &str = "EQIORA_MPI_TEST_FAULT_SOLVER";
const MIN_PHYSICAL_NODES_ENV: &str = "EQIORA_MPI_MIN_PHYSICAL_NODES";
const PROCESSOR_NAME_BYTES: usize = 512;
const CHILD_TIMEOUT: Duration = Duration::from_secs(30);

#[test]
fn mpi_one_two_four_rank_protocols_match_the_global_oracle() {
    if env::var_os(CHILD_ENV).is_some() {
        return;
    }
    for ranks in [1, 2, 4] {
        let output = run_mpi_child(ranks, "mpi_child_executes_admitted_run", &[]);
        assert_success(ranks, "normal", output);
    }
}

#[test]
fn mpi_unverified_solver_tuples_are_rejected_before_distributed_execution() {
    if env::var_os(CHILD_ENV).is_some() {
        return;
    }
    let output = run_mpi_child(
        1,
        "mpi_child_rejects_unverified_solver_tuples_at_preflight",
        &[],
    );
    assert_success(1, "capability preflight", output);
}

#[cfg(feature = "mpi-test-hooks")]
#[test]
fn post_admission_faults_are_collective_and_do_not_hang() {
    if env::var_os(CHILD_ENV).is_some() {
        return;
    }
    let solver_faults: [(&str, &[&str]); 2] = [
        (
            "cg",
            &[
                "local-action",
                "jacobi",
                "plan",
                "producer",
                "gather",
                "host-verifier",
            ],
        ),
        ("minres", &["local-action"]),
    ];
    for ranks in [1, 2, 4] {
        for (solver, points) in solver_faults {
            for point in points {
                let selected_rank = usize::from(ranks > 1);
                let fault = format!("{point}:{selected_rank}");
                let output = run_mpi_child(
                    ranks,
                    "mpi_child_fault_is_collective",
                    &[(FAULT_ENV, &fault), (FAULT_SOLVER_ENV, solver)],
                );
                assert_success(ranks, &format!("{solver} {point}"), output);
            }
        }
    }
}

#[cfg(feature = "mpi-test-hooks")]
#[test]
fn same_id_provider_version_and_library_drift_are_collectively_rejected() {
    if env::var_os(CHILD_ENV).is_some() {
        return;
    }
    for ranks in [2, 4] {
        for point in ["provider-version", "provider-library"] {
            let fault = format!("{point}:1");
            let output = run_mpi_child(
                ranks,
                "mpi_child_fault_is_collective",
                &[(FAULT_ENV, &fault), (FAULT_SOLVER_ENV, "cg")],
            );
            assert_success(ranks, point, output);
        }
    }
}

#[test]
fn mpi_child_executes_admitted_run() {
    if env::var_os(CHILD_ENV).is_none() {
        return;
    }
    let (universe, provided) = mpi::initialize_with_threading(Threading::Funneled)
        .expect("the application initializes MPI exactly once");
    let world = universe.world();
    let mut group =
        MpiExecutionGroup::duplicate(&world, provided, MpiThreadSupport::Funneled).unwrap();
    assert_minimum_physical_nodes(&world);

    let mut physical_identity = [0_u8; 16];
    physical_identity[15] = u8::try_from(group.partition().index() + 1).unwrap();
    let topology = group
        .agree_rank_device_topology(RankLocalDeviceV1::new(0, physical_identity))
        .unwrap();
    assert_eq!(topology.devices().len(), group.partitions().get());
    assert_eq!(
        topology.devices()[group.partition().index()].physical_identity(),
        physical_identity
    );
    let mut topology_fingerprints = vec![0_u8; 32 * group.partitions().get()];
    world.all_gather_into(&topology.fingerprint(), &mut topology_fingerprints[..]);
    assert!(
        topology_fingerprints
            .as_chunks::<32>()
            .0
            .iter()
            .map(|candidate| candidate.as_slice())
            .all(|candidate| candidate == topology.fingerprint())
    );

    let ordinal = u16::from(group.partition().index() == usize::from(group.partitions().get() > 1));
    let error = group
        .agree_rank_device_topology(RankLocalDeviceV1::new(ordinal, physical_identity))
        .unwrap_err();
    assert_common_diagnostic(&world, &error);
    if group.partitions().get() > 1 {
        let error = group
            .agree_rank_device_topology(RankLocalDeviceV1::new(0, [7_u8; 16]))
            .unwrap_err();
        assert_common_diagnostic(&world, &error);
    }
    group.agree_composed_local_readiness(&Ok(())).unwrap();
    let selected_rank = usize::from(group.partitions().get() > 1);
    let preparation = if group.partition().index() == selected_rank {
        Err(Diagnostic::error(
            codes::INVALID_REALIZATION,
            "injected composed action preparation failure",
        ))
    } else {
        Ok(())
    };
    let error = group
        .agree_composed_local_readiness(&preparation)
        .unwrap_err();
    assert_common_diagnostic(&world, &error);
    group.agree_composed_execution_summary([9_u8; 32]).unwrap();
    let mut drifted_summary = [9_u8; 32];
    drifted_summary[0] ^= u8::try_from(group.partition().index()).unwrap();
    if group.partitions().get() > 1 {
        let error = group
            .agree_composed_execution_summary(drifted_summary)
            .unwrap_err();
        assert_eq!(error.code(), codes::NUMERICAL_SOLVE_FAILED);
        assert_common_diagnostic(&world, &error);
    }

    let dimension = 12;
    let expected_solution = (0..dimension)
        .map(|index| index as f64 + 0.25)
        .collect::<Vec<_>>();
    let (cg_complete, cg_system) = admitted_case(&group, dimension, &expected_solution);
    for reduction in [ReductionPolicy::Reproducible, ReductionPolicy::Fast] {
        let plan = solver_plan(reduction);
        let admitted = group.admit(&cg_system, &cg_complete, plan).unwrap();
        assert_eq!(admitted.plan(), plan);
        assert_eq!(admitted.system(), &cg_system);
        let fingerprint = admitted.admission_fingerprint();
        let result = admitted.solve_and_replicate_with_trace().unwrap();
        let trace = result.trace();
        assert_eq!(trace.admission_fingerprint(), fingerprint);
        assert_eq!(
            trace.steps().first().unwrap().phase(),
            CollectivePhaseV1::Admission
        );
        assert!(
            trace
                .steps()
                .iter()
                .enumerate()
                .all(|(ordinal, step)| step.ordinal() == ordinal)
        );
        assert_eq!(
            trace.completed_iterations(),
            result.solution().report().completed_iterations()
        );
        let (solution, _) = result.into_parts();
        for (&actual, &expected) in solution.values().iter().zip(&expected_solution) {
            assert!((actual - expected).abs() <= 1.0e-10);
        }
        assert_eq!(
            solution.report().execution(),
            ExecutionReport::distributed(MPI_EXECUTION, group.partitions())
        );
        assert_eq!(
            solution.report().verification(),
            ExecutionReport::host_serial()
        );
        assert_eq!(solution.report().reduction(), reduction);
        assert_eq!(solution.report().backend(), MPI_DISTRIBUTED_KRYLOV_BACKEND);
    }

    let (complete, system) = admitted_minres_case(&group, dimension, &expected_solution);
    let plan = minres_plan();
    let mut local_action = DelegatingLocalAction {
        expected_partition: group.partition(),
        calls: 0,
        reject: false,
    };
    let result = group
        .admit(&system, &complete, plan)
        .unwrap()
        .solve_and_replicate_with_local_action(&mut local_action)
        .unwrap();
    assert!(local_action.calls > 0);
    assert_eq!(
        result.trace().completed_iterations(),
        result.solution().report().completed_iterations()
    );
    for (&actual, &expected) in result.solution().values().iter().zip(&expected_solution) {
        assert!((actual - expected).abs() <= 1.0e-10);
    }
    assert_eq!(
        result.solution().report().backend(),
        MPI_DISTRIBUTED_KRYLOV_BACKEND
    );
    assert_eq!(
        result.solution().report().algorithm(),
        LinearSolver::MinimumResidual
    );
    assert_eq!(
        result.solution().report().preconditioner(),
        PreconditionerPolicy::Identity
    );
    assert_eq!(
        result.solution().report().reduction(),
        ReductionPolicy::Reproducible
    );

    let mut rejecting_action = DelegatingLocalAction {
        expected_partition: group.partition(),
        calls: 0,
        reject: group.partition().index() == usize::from(group.partitions().get() > 1),
    };
    let error = group
        .admit(&system, &complete, plan)
        .unwrap()
        .solve_and_replicate_with_local_action(&mut rejecting_action)
        .unwrap_err();
    assert_eq!(error.code(), codes::NUMERICAL_SOLVE_FAILED);
    assert!(error.message().contains("distributed phase LocalAction"));
    assert_common_diagnostic(&world, &error);

    // A complete-view cross-wire on only one rank is rejected before the
    // admitted execution token exists.
    let changed = changed_complete(&cg_complete, group.partition());
    let error = group
        .admit(
            &cg_system,
            &changed,
            solver_plan(ReductionPolicy::Reproducible),
        )
        .unwrap_err();
    assert_common_diagnostic(&world, &error);

    // Every local plan is independently valid, but rank-local plan drift must
    // fail at the plan-inclusive admission fingerprint before execution.
    if group.partitions().get() > 1 {
        let drifted_plan = if group.partition().index() == 0 {
            solver_plan(ReductionPolicy::Fast)
        } else {
            solver_plan(ReductionPolicy::Reproducible)
        };
        let error = group
            .admit(&cg_system, &cg_complete, drifted_plan)
            .unwrap_err();
        assert_common_diagnostic(&world, &error);
    }
    world.barrier();
}

struct DelegatingLocalAction {
    expected_partition: PartitionId,
    calls: usize,
    reject: bool,
}

impl MpiRankLocalCsrAction for DelegatingLocalAction {
    fn apply_owned_rows(
        &mut self,
        shard: LocalCsrShard<'_>,
        owned_input: &[f64],
        ghosts: &[f64],
        owned_output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        self.calls += 1;
        if shard.layout().partition() != self.expected_partition {
            return Err(Diagnostic::error(
                codes::INVALID_REALIZATION,
                "injected local action received the wrong admitted shard",
            ));
        }
        if self.reject {
            return Err(Diagnostic::error(
                codes::NUMERICAL_SOLVE_FAILED,
                "injected composed local action failure",
            ));
        }
        shard.apply(owned_input, ghosts, owned_output)
    }
}

#[test]
fn mpi_child_rejects_unverified_solver_tuples_at_preflight() {
    if env::var_os(CHILD_ENV).is_none() {
        return;
    }
    let (universe, provided) = mpi::initialize_with_threading(Threading::Funneled)
        .expect("the application initializes MPI exactly once");
    let world = universe.world();
    let group = MpiExecutionGroup::duplicate(&world, provided, MpiThreadSupport::Funneled).unwrap();
    assert_eq!(
        group.solver_capabilities().combinations(),
        &BTreeSet::from([
            SolverCapability {
                algorithm: LinearSolver::ConjugateGradient,
                operator_properties: LinearOperatorProperties::SymmetricPositiveDefinite,
                preconditioner: PreconditionerPolicy::Jacobi,
                reduction: ReductionPolicy::Reproducible,
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
                algorithm: LinearSolver::MinimumResidual,
                operator_properties: LinearOperatorProperties::SymmetricIndefinite,
                preconditioner: PreconditionerPolicy::Identity,
                reduction: ReductionPolicy::Reproducible,
                scalar_type: ScalarType::F64,
            },
        ])
    );
    group
        .solver_capabilities()
        .require_problem(
            minres_plan(),
            ScalarType::F64,
            LinearOperatorProperties::SymmetricIndefinite,
        )
        .unwrap();

    for unsupported in [
        minres_plan().with_reduction(ReductionPolicy::Fast),
        minres_plan().with_preconditioner(PreconditionerPolicy::Jacobi),
    ] {
        let error = group
            .solver_capabilities()
            .require_problem(
                unsupported,
                ScalarType::F64,
                LinearOperatorProperties::SymmetricIndefinite,
            )
            .unwrap_err();
        assert_eq!(error.code(), codes::INVALID_REALIZATION);
        assert!(error.message().contains("exact"));
    }

    for reduction in [ReductionPolicy::Reproducible, ReductionPolicy::Fast] {
        let unverified_cross_product = SolverPlan::new(
            LinearSolver::ConjugateGradient,
            1.0e-12,
            1.0e-13,
            NonZeroUsize::new(100).unwrap(),
        )
        .unwrap()
        .with_preconditioner(PreconditionerPolicy::Identity)
        .with_reduction(reduction);
        let error = group
            .solver_capabilities()
            .require_problem(
                unverified_cross_product,
                ScalarType::F64,
                LinearOperatorProperties::SymmetricPositiveDefinite,
            )
            .unwrap_err();
        assert_eq!(error.code(), codes::INVALID_REALIZATION);
        assert!(error.message().contains("Identity"));
        assert!(error.message().contains("exact"));
    }
}

/// Stable exact-test alias retained by the recorded physical two-node command.
#[test]
fn mpi_child_executes_halo_and_collectives() {
    mpi_child_executes_admitted_run();
}

#[cfg(feature = "mpi-test-hooks")]
#[test]
fn mpi_child_fault_is_collective() {
    if env::var_os(CHILD_ENV).is_none() {
        return;
    }
    let (universe, provided) = mpi::initialize_with_threading(Threading::Funneled)
        .expect("the application initializes MPI exactly once");
    let world = universe.world();
    let mut group =
        MpiExecutionGroup::duplicate(&world, provided, MpiThreadSupport::Funneled).unwrap();
    let dimension = 12;
    let expected_solution = (0..dimension)
        .map(|index| index as f64 + 0.25)
        .collect::<Vec<_>>();
    let solver = env::var(FAULT_SOLVER_ENV).unwrap_or_else(|_| "cg".to_owned());
    let (complete, system, plan) = match solver.as_str() {
        "cg" => {
            let (complete, system) = admitted_case(&group, dimension, &expected_solution);
            (complete, system, solver_plan(ReductionPolicy::Reproducible))
        }
        "minres" => {
            let (complete, system) = admitted_minres_case(&group, dimension, &expected_solution);
            (complete, system, minres_plan())
        }
        _ => panic!("unknown MPI fault-test solver {solver}"),
    };

    // Fault hooks are consumed only after this collective admission succeeds.
    let admitted = group.admit(&system, &complete, plan).unwrap();
    let error = admitted.solve_and_replicate().unwrap_err();
    if env::var(FAULT_ENV).is_ok_and(|fault| fault.starts_with("local-action:")) {
        assert_eq!(error.code(), codes::NUMERICAL_SOLVE_FAILED);
        assert!(error.message().contains("distributed phase LocalAction"));
        assert!(error.message().contains(&format!(
            "rejected partition {}",
            usize::from(group.partitions().get() > 1)
        )));
    }
    assert_common_diagnostic(&world, &error);
    world.barrier();
}

fn admitted_case(
    group: &MpiExecutionGroup,
    dimension: usize,
    expected_solution: &[f64],
) -> (CanonicalCsrSystemView, DistributedLinearSystem) {
    let (offsets, columns, values) = tridiagonal(dimension);
    let rhs = global_apply(&offsets, &columns, &values, expected_solution);
    let complete = CanonicalCsrSystemView::new(
        &TestStorage {
            dimension,
            offsets,
            columns,
            values,
            rhs,
        },
        LinearOperatorProperties::SymmetricPositiveDefinite,
    )
    .unwrap();
    let owners = (0..dimension)
        .map(|global| PartitionId::new((global + 1) % group.partitions().get()))
        .collect();
    let partition = Partition::new(
        GlobalVectorSpace::new(NonZeroUsize::new(dimension).unwrap(), ScalarType::F64),
        group.partitions(),
        owners,
    )
    .unwrap();
    let system = DistributedLinearSystem::from_complete(&complete, partition).unwrap();
    (complete, system)
}

fn admitted_minres_case(
    group: &MpiExecutionGroup,
    dimension: usize,
    expected_solution: &[f64],
) -> (CanonicalCsrSystemView, DistributedLinearSystem) {
    let (offsets, columns, values) = symmetric_indefinite_pairs(dimension);
    let rhs = global_apply(&offsets, &columns, &values, expected_solution);
    let complete = CanonicalCsrSystemView::new(
        &TestStorage {
            dimension,
            offsets,
            columns,
            values,
            rhs,
        },
        LinearOperatorProperties::SymmetricIndefinite,
    )
    .unwrap();
    let owners = (0..dimension)
        .map(|global| PartitionId::new((global + 1) % group.partitions().get()))
        .collect();
    let partition = Partition::new(
        GlobalVectorSpace::new(NonZeroUsize::new(dimension).unwrap(), ScalarType::F64),
        group.partitions(),
        owners,
    )
    .unwrap();
    let system = DistributedLinearSystem::from_complete(&complete, partition).unwrap();
    (complete, system)
}

fn solver_plan(reduction: ReductionPolicy) -> SolverPlan {
    SolverPlan::new(
        LinearSolver::ConjugateGradient,
        1.0e-12,
        1.0e-13,
        NonZeroUsize::new(100).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Jacobi)
    .with_reduction(reduction)
}

fn minres_plan() -> SolverPlan {
    SolverPlan::new(
        LinearSolver::MinimumResidual,
        1.0e-12,
        1.0e-13,
        NonZeroUsize::new(100).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Identity)
    .with_reduction(ReductionPolicy::Reproducible)
}

fn changed_complete(
    complete: &CanonicalCsrSystemView,
    partition: PartitionId,
) -> CanonicalCsrSystemView {
    let mut rhs = complete.right_hand_side().to_vec();
    if partition.index() == 0 {
        rhs[0] += 1.0;
    }
    CanonicalCsrSystemView::new(
        &TestStorage {
            dimension: complete.rows(),
            offsets: complete.row_offsets().to_vec(),
            columns: complete.column_indices().to_vec(),
            values: complete.values().to_vec(),
            rhs,
        },
        complete.properties(),
    )
    .unwrap()
}

struct ChildOutput {
    status: ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

fn run_mpi_child(ranks: usize, test: &str, extra_environment: &[(&str, &str)]) -> ChildOutput {
    let executable = env::current_exe().expect("integration-test executable is available");
    let launcher = env::var_os("EQIORA_MPI_LAUNCHER").unwrap_or_else(|| "mpirun".into());
    let mut command = Command::new(&launcher);
    if launcher_accepts_oversubscribe(&launcher) {
        command.arg("--oversubscribe");
    }
    command
        .args(["-n", &ranks.to_string()])
        .arg(executable)
        .args(["--exact", test, "--nocapture"])
        .env(CHILD_ENV, "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (name, value) in extra_environment {
        command.env(name, value);
    }
    let mut child = command
        .spawn()
        .expect("CI MPI evidence requires mpirun on PATH");
    let started = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().expect("MPI child status is readable") {
            break status;
        }
        if started.elapsed() >= CHILD_TIMEOUT {
            child.kill().expect("timed-out MPI launcher can be killed");
            let _ = child.wait();
            panic!("{ranks}-rank MPI child {test} exceeded {CHILD_TIMEOUT:?}");
        }
        thread::sleep(Duration::from_millis(20));
    };
    let mut stdout = Vec::new();
    child
        .stdout
        .take()
        .unwrap()
        .read_to_end(&mut stdout)
        .unwrap();
    let mut stderr = Vec::new();
    child
        .stderr
        .take()
        .unwrap()
        .read_to_end(&mut stderr)
        .unwrap();
    ChildOutput {
        status,
        stdout,
        stderr,
    }
}

fn launcher_accepts_oversubscribe(launcher: &std::ffi::OsStr) -> bool {
    Command::new(launcher)
        .args(["--oversubscribe", "--version"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

fn assert_success(ranks: usize, case: &str, output: ChildOutput) {
    assert!(
        output.status.success(),
        "{ranks}-rank MPI {case} child failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

fn assert_common_diagnostic(world: &impl CommunicatorCollectives, error: &eqiora_core::Diagnostic) {
    const DIAGNOSTIC_BYTES: usize = 256;
    let rendered = format!("{}:{}", error.code(), error.message());
    assert!(rendered.len() <= DIAGNOSTIC_BYTES);
    let ranks = usize::try_from(world.size()).unwrap();
    let mut diagnostics = vec![0_u8; ranks * DIAGNOSTIC_BYTES];
    let mut local = [0_u8; DIAGNOSTIC_BYTES];
    local[..rendered.len()].copy_from_slice(rendered.as_bytes());
    world.all_gather_into(&local[..], &mut diagnostics[..]);
    assert!(
        diagnostics
            .as_chunks::<DIAGNOSTIC_BYTES>()
            .0
            .iter()
            .map(|diagnostic| diagnostic.as_slice())
            .all(|diagnostic| diagnostic == local)
    );
}

#[derive(Debug)]
struct TestStorage {
    dimension: usize,
    offsets: Vec<usize>,
    columns: Vec<usize>,
    values: Vec<f64>,
    rhs: Vec<f64>,
}

impl CompleteCsrStorage for TestStorage {
    fn rows(&self) -> usize {
        self.dimension
    }

    fn columns(&self) -> usize {
        self.dimension
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

    fn right_hand_side(&self) -> &[f64] {
        &self.rhs
    }
}

fn assert_minimum_physical_nodes(world: &impl CommunicatorCollectives) {
    let Ok(required) = env::var(MIN_PHYSICAL_NODES_ENV) else {
        return;
    };
    let required = required
        .parse::<NonZeroUsize>()
        .expect("EQIORA_MPI_MIN_PHYSICAL_NODES must be a positive integer");
    let ranks = usize::try_from(world.size()).expect("MPI communicator size fits usize");
    assert!(required.get() <= ranks);
    let processor = mpi::environment::processor_name().expect("MPI processor name is UTF-8");
    assert!(processor.len() <= PROCESSOR_NAME_BYTES);
    let mut local = [0_u8; PROCESSOR_NAME_BYTES];
    local[..processor.len()].copy_from_slice(processor.as_bytes());
    let mut gathered = vec![0_u8; PROCESSOR_NAME_BYTES * ranks];
    world.all_gather_into(&local[..], &mut gathered[..]);
    let processors = gathered
        .as_chunks::<PROCESSOR_NAME_BYTES>()
        .0
        .iter()
        .map(|name| {
            let length = name
                .iter()
                .position(|byte| *byte == 0)
                .unwrap_or(name.len());
            &name[..length]
        })
        .collect::<BTreeSet<_>>();
    assert!(processors.len() >= required.get());
}

fn tridiagonal(dimension: usize) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut offsets = Vec::with_capacity(dimension + 1);
    let mut columns = Vec::new();
    let mut values = Vec::new();
    offsets.push(0);
    for row in 0..dimension {
        if row > 0 {
            columns.push(row - 1);
            values.push(-1.0);
        }
        columns.push(row);
        values.push(2.0);
        if row + 1 < dimension {
            columns.push(row + 1);
            values.push(-1.0);
        }
        offsets.push(columns.len());
    }
    (offsets, columns, values)
}

fn symmetric_indefinite_pairs(dimension: usize) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    assert_eq!(dimension % 2, 0);
    let offsets = (0..=dimension).collect();
    let columns = (0..dimension).map(|row| row ^ 1).collect();
    let values = vec![1.0; dimension];
    (offsets, columns, values)
}

fn global_apply(offsets: &[usize], columns: &[usize], values: &[f64], input: &[f64]) -> Vec<f64> {
    (0..input.len())
        .map(|row| {
            (offsets[row]..offsets[row + 1])
                .map(|entry| values[entry] * input[columns[entry]])
                .sum()
        })
        .collect()
}
