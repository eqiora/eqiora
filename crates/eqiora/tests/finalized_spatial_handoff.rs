use eqiora_core::ScalarType;
use std::num::{NonZeroU16, NonZeroUsize};

use eqiora::Diagnostic;
use eqiora::compiler::compile;
use eqiora::diagnostic::codes;
use eqiora::distributed::{DistributedLinearSystem, GlobalVectorSpace, Partition};
use eqiora::graph::{GraphStore, InMemoryGraphStore};
use eqiora::realization::{
    Discretization, DiscretizationMethod, ExecutionSchedule, MeshKind, MeshPolicy,
    QuadraturePolicy, RealizationCapabilities, RealizationPlan, RealizationRequest,
    RealizationRequirements, RealizationRevision, SemanticRevision, Space, SpatialDimensionSupport,
    Target, TargetCapabilities, VectorLayoutKind, resolve,
};
use eqiora::sem::KernelProgram;
use eqiora::solver::{
    BackendId, ConvergenceReason, ExecutionId, ExecutionProvider, ExecutionReport,
    ExecutionTopology, FixedOrderInnerProduct, LinearOperator, LinearOperatorProperties,
    LinearSolution, LinearSolver, LinearSolverBackend, REFERENCE_LINEAR_SOLVER, ReductionPolicy,
    ReplicatedLinearExecution, SERIAL_LINEAR_EXECUTION, SolverCapabilities, SolverCapability,
    SolverPlan, SolverProvider, accept_linear_solution_with_verifier,
};
use eqiora_numerics::{
    scalar::ResolvedScalarEllipticCartesianSolution,
    scalar::finalize_resolved_scalar_elliptic_cartesian,
    scalar::solve_resolved_scalar_elliptic_cartesian,
};

const SOURCE: &str =
    include_str!("../../../verify/numerics/finalized-spatial-handoff/models/poisson.eqi");

#[test]
fn finalized_spatial_handoff_is_symmetric_numerically_reaccepted_and_fail_closed() {
    let program = compile_program();
    let solver = solver_plan(256);
    let fem = resolve_plan(
        &program,
        DiscretizationMethod::ContinuousGalerkin,
        4,
        solver,
        1,
    );
    let fvm = resolve_plan(
        &program,
        DiscretizationMethod::CellCenteredFiniteVolume,
        3,
        solver,
        2,
    );

    let (_, fem_problem) = finalize_resolved_scalar_elliptic_cartesian(&program, &fem).unwrap();
    let (_, fvm_problem) = finalize_resolved_scalar_elliptic_cartesian(&program, &fvm).unwrap();
    assert_eq!(
        fem_problem.method(),
        DiscretizationMethod::ContinuousGalerkin
    );
    assert_eq!(
        fvm_problem.method(),
        DiscretizationMethod::CellCenteredFiniteVolume
    );
    assert_eq!(fem_problem.canonical_csr_system_view().rows(), 9);
    assert_eq!(fvm_problem.canonical_csr_system_view().rows(), 9);
    assert_eq!(
        fem_problem.operator_properties(),
        LinearOperatorProperties::SymmetricPositiveDefinite
    );
    assert_eq!(fem_problem.solver_plan(), solver);
    assert_eq!(fvm_problem.solver_plan(), solver);
    assert_eq!(fem_problem.vector_layout(), VectorLayoutKind::Replicated);
    assert_eq!(fvm_problem.vector_layout(), VectorLayoutKind::Replicated);
    assert_eq!(fem_problem.assembly_report().target_count(), 2);
    assert_eq!(fvm_problem.assembly_report().target_count(), 1);

    let fem_linear = REFERENCE_LINEAR_SOLVER
        .solve(&fem_problem.linear_problem().unwrap(), solver)
        .unwrap();
    let fvm_linear = REFERENCE_LINEAR_SOLVER
        .solve(&fvm_problem.linear_problem().unwrap(), solver)
        .unwrap();

    assert_eq!(
        fem_linear.report().residual_target().to_bits(),
        fvm_linear.report().residual_target().to_bits()
    );
    let cross_wire = fvm_problem.clone().finish(fem_linear.clone()).unwrap_err();
    assert_eq!(cross_wire.code(), codes::NUMERICAL_SOLVE_FAILED);
    assert!(
        cross_wire
            .message()
            .contains("exceeds this finalized spatial problem target"),
        "{}",
        cross_wire.message()
    );

    let different_plan = solver_plan(257);
    let different_plan_solution = REFERENCE_LINEAR_SOLVER
        .solve(&fem_problem.linear_problem().unwrap(), different_plan)
        .unwrap();
    let plan_mismatch = fem_problem
        .clone()
        .finish(different_plan_solution)
        .unwrap_err();
    assert!(plan_mismatch.message().contains("different SolverPlan"));

    let cuda_claim = accept_linear_solution_with_verifier(
        &fem_problem.linear_problem().unwrap(),
        solver,
        test_solver_provider(BackendId::new("eqiora.test.cross-wire")),
        test_execution_provider(ExecutionId::new("eqiora.cuda.test")),
        ExecutionReport::cuda(ExecutionId::new("eqiora.cuda.test"), 0),
        ConvergenceReason::ResidualToleranceSatisfied,
        fem_linear.report().completed_iterations(),
        fem_linear.report().reported_residual_norm(),
        fem_linear.values().to_vec(),
        &SERIAL_LINEAR_EXECUTION,
    )
    .unwrap();
    let topology_mismatch = fem_problem.clone().finish(cuda_claim).unwrap_err();
    assert!(
        topology_mismatch
            .message()
            .contains("replicated host realization")
    );

    let distributed_claim = distributed_candidate(&fem_problem, 2);
    let layout_mismatch = fem_problem.clone().finish(distributed_claim).unwrap_err();
    assert!(
        layout_mismatch
            .message()
            .contains("replicated host realization")
    );

    let invalid_verifier = ClaimedVerifier {
        report: ExecutionReport::cuda(ExecutionId::new("eqiora.test.invalid-verifier"), 0),
    };
    let invalid_verification_claim = accept_linear_solution_with_verifier(
        &fem_problem.linear_problem().unwrap(),
        solver,
        test_solver_provider(BackendId::new("eqiora.test.invalid-verification")),
        fem_linear.report().execution_provider(),
        fem_linear.report().execution(),
        fem_linear.report().reason(),
        fem_linear.report().completed_iterations(),
        fem_linear.report().reported_residual_norm(),
        fem_linear.values().to_vec(),
        &invalid_verifier,
    )
    .unwrap();
    let verification_mismatch = fem_problem
        .clone()
        .finish(invalid_verification_claim)
        .unwrap_err();
    assert!(
        verification_mismatch
            .message()
            .contains("host verification")
    );

    let fem_finished = fem_problem.finish(fem_linear).unwrap();
    let fvm_finished = fvm_problem.finish(fvm_linear).unwrap();
    let (_, fem_direct) =
        solve_resolved_scalar_elliptic_cartesian(&program, &fem, &REFERENCE_LINEAR_SOLVER).unwrap();
    let (_, fvm_direct) =
        solve_resolved_scalar_elliptic_cartesian(&program, &fvm, &REFERENCE_LINEAR_SOLVER).unwrap();
    assert_eq!(fem_finished, fem_direct);
    assert_eq!(fvm_finished, fvm_direct);
    assert!(matches!(
        fem_finished,
        ResolvedScalarEllipticCartesianSolution::FiniteElement(_)
    ));
    assert!(matches!(
        fvm_finished,
        ResolvedScalarEllipticCartesianSolution::FiniteVolume(_)
    ));
}

#[test]
fn distributed_finalized_handoff_uses_one_canonical_view_and_exact_topology() {
    let program = compile_program();
    let solver = solver_plan(256);
    let fem = resolve_plan_with_placement(
        &program,
        DiscretizationMethod::ContinuousGalerkin,
        4,
        solver,
        11,
        VectorLayoutKind::Distributed,
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
    );
    let fvm = resolve_plan_with_placement(
        &program,
        DiscretizationMethod::CellCenteredFiniteVolume,
        3,
        solver,
        12,
        VectorLayoutKind::Distributed,
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
    );
    let (_, fem_problem) = finalize_resolved_scalar_elliptic_cartesian(&program, &fem).unwrap();
    let (_, fvm_problem) = finalize_resolved_scalar_elliptic_cartesian(&program, &fvm).unwrap();

    assert_eq!(fem_problem.vector_layout(), VectorLayoutKind::Distributed);
    let complete = fem_problem.canonical_csr_system_view();
    assert_eq!(complete.rows(), complete.right_hand_side().len());
    let partition = Partition::balanced_contiguous(
        GlobalVectorSpace::new(NonZeroUsize::new(complete.rows()).unwrap(), ScalarType::F64),
        NonZeroUsize::new(2).unwrap(),
    )
    .unwrap();
    let distributed = DistributedLinearSystem::from_complete(complete, partition).unwrap();
    assert!(distributed.matches_complete(complete));
    assert_eq!(distributed.operator().layouts().len(), 2);

    let candidate = distributed_candidate(&fem_problem, 2);
    assert_eq!(
        candidate.report().execution().topology(),
        ExecutionTopology::Distributed {
            ranks: NonZeroUsize::new(2).unwrap(),
            workers_per_partition: NonZeroUsize::MIN,
        }
    );
    assert_eq!(
        candidate.report().verification().topology(),
        ExecutionTopology::Host {
            workers: NonZeroUsize::MIN,
        }
    );

    let host_candidate = REFERENCE_LINEAR_SOLVER
        .solve(&fem_problem.linear_problem().unwrap(), solver)
        .unwrap();
    let topology_mismatch = fem_problem.clone().finish(host_candidate).unwrap_err();
    assert!(
        topology_mismatch
            .message()
            .contains("distributed host realization")
    );

    let wrong_method = fvm_problem.clone().finish(candidate.clone()).unwrap_err();
    assert_eq!(wrong_method.code(), codes::NUMERICAL_SOLVE_FAILED);
    assert!(
        wrong_method
            .message()
            .contains("exceeds this finalized spatial problem target")
    );

    let finished = fem_problem.finish(candidate).unwrap();
    assert!(matches!(
        finished,
        ResolvedScalarEllipticCartesianSolution::FiniteElement(_)
    ));
}

#[test]
fn distributed_finalization_rejects_unadmitted_placement_combinations() {
    let program = compile_program();
    let solver = solver_plan(256);
    let threaded = resolve_plan_with_placement(
        &program,
        DiscretizationMethod::ContinuousGalerkin,
        4,
        solver,
        21,
        VectorLayoutKind::Distributed,
        Target::HostCpu {
            threads: NonZeroUsize::new(2).unwrap(),
        },
    );
    let threaded_error =
        finalize_resolved_scalar_elliptic_cartesian(&program, &threaded).unwrap_err();
    assert!(threaded_error.message().contains("one host worker"));

    let cuda = resolve_plan_with_placement(
        &program,
        DiscretizationMethod::ContinuousGalerkin,
        4,
        solver,
        22,
        VectorLayoutKind::Distributed,
        Target::CudaGpu { device: 3 },
    );
    let cuda_error = finalize_resolved_scalar_elliptic_cartesian(&program, &cuda).unwrap_err();
    assert!(cuda_error.message().contains("distributed CUDA"));
}

fn distributed_candidate(
    problem: &eqiora_numerics::scalar::FinalizedScalarEllipticCartesianProblem,
    ranks: usize,
) -> LinearSolution {
    let plan = problem.solver_plan();
    let host = REFERENCE_LINEAR_SOLVER
        .solve(&problem.linear_problem().unwrap(), plan)
        .unwrap();
    accept_linear_solution_with_verifier(
        &problem.linear_problem().unwrap(),
        plan,
        test_solver_provider(BackendId::new("eqiora.test.distributed-loopback")),
        test_execution_provider(ExecutionId::new("eqiora.distributed.loopback")),
        ExecutionReport::distributed(
            ExecutionId::new("eqiora.distributed.loopback"),
            NonZeroUsize::new(ranks).unwrap(),
        ),
        host.report().reason(),
        host.report().completed_iterations(),
        host.report().reported_residual_norm(),
        host.values().to_vec(),
        &SERIAL_LINEAR_EXECUTION,
    )
    .unwrap()
}

#[derive(Debug, Clone, Copy)]
struct ClaimedVerifier {
    report: ExecutionReport,
}

impl ReplicatedLinearExecution for ClaimedVerifier {
    fn provider(&self) -> ExecutionProvider {
        test_execution_provider(self.report.adapter())
    }

    fn report(&self) -> ExecutionReport {
        self.report
    }

    fn require_reduction(&self, policy: ReductionPolicy) -> Result<(), Diagnostic> {
        SERIAL_LINEAR_EXECUTION.require_reduction(policy)
    }

    fn apply(
        &self,
        operator: &dyn LinearOperator,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        SERIAL_LINEAR_EXECUTION.apply(operator, input, output)
    }

    fn inner_product(&self, action: FixedOrderInnerProduct<'_>) -> Result<f64, Diagnostic> {
        SERIAL_LINEAR_EXECUTION.inner_product(action)
    }
}

const fn test_solver_provider(id: BackendId) -> SolverProvider {
    SolverProvider::new(id, env!("CARGO_PKG_VERSION"), &[])
}

const fn test_execution_provider(id: ExecutionId) -> ExecutionProvider {
    ExecutionProvider::new(id, env!("CARGO_PKG_VERSION"), &[])
}

fn solver_plan(maximum_iterations: usize) -> SolverPlan {
    SolverPlan::new(
        LinearSolver::ConjugateGradient,
        0.0,
        1.0e-12,
        NonZeroUsize::new(maximum_iterations).unwrap(),
    )
    .unwrap()
}

fn resolve_plan(
    program: &KernelProgram,
    method: DiscretizationMethod,
    cells: usize,
    solver: SolverPlan,
    revision: u64,
) -> eqiora::realization::ResolvedRealization {
    resolve_plan_with_placement(
        program,
        method,
        cells,
        solver,
        revision,
        VectorLayoutKind::Replicated,
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn resolve_plan_with_placement(
    program: &KernelProgram,
    method: DiscretizationMethod,
    cells: usize,
    solver: SolverPlan,
    revision: u64,
    vector_layout: VectorLayoutKind,
    target: Target,
) -> eqiora::realization::ResolvedRealization {
    let (space, quadrature) = match method {
        DiscretizationMethod::ContinuousGalerkin => (
            Space::continuous_lagrange(NonZeroU16::MIN),
            QuadraturePolicy::GaussLegendre {
                points_per_axis: NonZeroUsize::new(2).unwrap(),
            },
        ),
        DiscretizationMethod::CellCenteredFiniteVolume => {
            (Space::cell_constant(), QuadraturePolicy::CellCentroid)
        }
    };
    let plan = RealizationPlan::new(
        space,
        Discretization::new(
            method,
            MeshPolicy::GeneratedUniform {
                cells_per_axis: NonZeroUsize::new(cells).unwrap(),
            },
            quadrature,
        ),
        solver,
        target,
        ExecutionSchedule::Offline,
    )
    .unwrap();
    let targets = match target {
        Target::HostCpu { threads } => TargetCapabilities::none().with_host_cpu(threads),
        Target::CudaGpu { device } => TargetCapabilities::none().with_cuda_device(device),
    };
    let solver_capabilities = SolverCapabilities::exact([SolverCapability {
        algorithm: solver.algorithm(),
        operator_properties: LinearOperatorProperties::SymmetricPositiveDefinite,
        preconditioner: solver.preconditioner(),
        reduction: solver.reduction(),
        scalar_type: ScalarType::F64,
    }])
    .unwrap();
    let capabilities = RealizationCapabilities::cartesian_product(
        [
            DiscretizationMethod::ContinuousGalerkin,
            DiscretizationMethod::CellCenteredFiniteVolume,
        ],
        [(
            MeshKind::GeneratedCartesian,
            SpatialDimensionSupport::inclusive(NonZeroUsize::MIN, NonZeroUsize::new(3).unwrap())
                .unwrap(),
        )],
        [vector_layout],
        solver_capabilities,
        targets,
    )
    .unwrap();
    resolve(
        &RealizationRequest::explicit(
            program.model(),
            SemanticRevision::new(program.revision().0),
            RealizationRevision::new(revision),
            plan,
        ),
        RealizationRequirements::new(
            NonZeroUsize::new(2).unwrap(),
            ScalarType::F64,
            vector_layout,
        ),
        &capabilities,
    )
    .unwrap()
}

fn compile_program() -> KernelProgram {
    let mut compiled = compile(
        "verify/numerics/finalized-spatial-handoff/models/poisson.eqi",
        SOURCE,
    )
    .unwrap();
    let (transaction, model, _) = compiled.remove(0).into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    KernelProgram::from_snapshot(&store.snapshot(), model).unwrap()
}
