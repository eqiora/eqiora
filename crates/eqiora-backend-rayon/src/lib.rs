//! Isolated Rayon host execution adapter.
//!
//! The adapter owns a bounded Rayon pool and schedules indexed local assembly
//! packets, operators that expose disjoint output rows, and Eqiora's
//! fixed-order scalar-reduction partials. Assembly order, solver policy, and
//! floating-point policy remain in their L2 contracts.

use std::fmt;
use std::num::NonZeroUsize;
use std::sync::Mutex;

use eqiora_assembly::{
    AssemblyAccumulator, AssemblyBackend, AssemblyPlan, AssemblyResult, AssemblyWork,
};
use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, ScalarType};
use eqiora_execution::DeploymentBinding;
use eqiora_realization::{Target, TargetCapabilities};
use eqiora_solver::{
    ExecutionId, ExecutionProvider, ExecutionReport, ExecutionTopology, FixedOrderInnerProduct,
    LinearOperator, LinearProblem, LinearSolution, LinearSolverBackend, ProviderLibrary,
    ReductionPolicy, ReplicatedLinearExecution, SERIAL_LINEAR_EXECUTION, SolverCapabilities,
    SolverPlan, SolverProvider,
};
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

/// Stable execution identity recorded independently from the solver backend.
pub const RAYON_EXECUTION: ExecutionId = ExecutionId::new("eqiora.rayon");

/// Exact Eqiora Rayon execution-adapter package version compiled into this binary.
pub const RAYON_ADAPTER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Exact Rayon dependency version compiled into this adapter.
pub const RAYON_VERSION: &str = "1.12.0";

/// Complete identity of the Rayon execution provider compiled into this binary.
pub const RAYON_EXECUTION_PROVIDER: ExecutionProvider = ExecutionProvider::new(
    RAYON_EXECUTION,
    RAYON_ADAPTER_VERSION,
    &[ProviderLibrary::new("rayon", RAYON_VERSION)],
);

/// Number of consecutive packets evaluated before ordered scatter resumes.
///
/// This is memory/scheduling policy only. Packet and floating-point order do
/// not depend on the batch length.
pub const ORDERED_ASSEMBLY_BATCH_LENGTH: usize = 256;

/// One run-owned bounded host thread pool.
pub struct CpuThreadPool {
    pool: ThreadPool,
    workers: NonZeroUsize,
}

impl CpuThreadPool {
    /// Materialize the exact Rayon placement admitted by a deployment binding.
    ///
    /// The binding is deliberately created before this call, so insufficient
    /// executor capacity cannot allocate worker threads as a side effect.
    ///
    /// # Errors
    /// Returns `EQ0807` unless the binding selected the Rayon adapter and a
    /// host topology, or if the operating system cannot create the pool.
    pub fn from_deployment(binding: &DeploymentBinding) -> Result<Self, Diagnostic> {
        let executor = binding.host_executor().ok_or_else(|| {
            invalid_realization("a CPU thread pool requires a host deployment binding")
        })?;
        if executor.adapter() != RAYON_EXECUTION {
            return Err(invalid_realization(
                "a CPU thread pool requires a deployment bound to the Rayon adapter",
            ));
        }
        let ExecutionTopology::Host { workers } = binding.execution().topology() else {
            return Err(invalid_realization(
                "a CPU thread pool requires a host deployment topology",
            ));
        };
        Self::new(workers)
    }

    /// Build an isolated pool without modifying Rayon's global pool.
    ///
    /// # Errors
    /// Returns `EQ0807` if the operating system cannot create the requested
    /// worker pool.
    pub fn new(workers: NonZeroUsize) -> Result<Self, Diagnostic> {
        let pool = ThreadPoolBuilder::new()
            .num_threads(workers.get())
            .thread_name(|index| format!("eqiora-rayon-{index}"))
            .build()
            .map_err(|error| {
                invalid_realization(format!(
                    "cannot build an isolated {}-worker CPU pool: {error}",
                    workers
                ))
            })?;
        Ok(Self { pool, workers })
    }

    /// Exact number of worker threads owned by this pool.
    #[must_use]
    pub const fn workers(&self) -> NonZeroUsize {
        self.workers
    }

    /// Host placement capability admitted by this concrete pool.
    #[must_use]
    pub fn target_capabilities(&self) -> TargetCapabilities {
        TargetCapabilities::none().with_host_cpu(self.workers)
    }

    /// Compose this exact pool with one resolved host target and solver.
    ///
    /// # Errors
    /// Returns `EQ0807` unless the resolved target requests exactly the worker
    /// count owned by this pool. A pool is not silently widened or narrowed.
    pub fn solver<'a>(
        &'a self,
        target: Target,
        backend: &'a dyn LinearSolverBackend,
    ) -> Result<ThreadedLinearSolver<'a>, Diagnostic> {
        self.require_target(target)?;
        if !backend
            .capabilities()
            .reductions()
            .contains(&ReductionPolicy::Reproducible)
        {
            return Err(invalid_realization(format!(
                "solver backend {} cannot be placed on the first Rayon execution because it does not admit reproducible reductions",
                backend.id().as_str()
            )));
        }
        Ok(ThreadedLinearSolver {
            pool: self,
            backend,
        })
    }

    /// Bind ordered assembly to this exact run-owned pool and resolved target.
    ///
    /// # Errors
    /// Returns `EQ0807` unless the target is host CPU execution with exactly
    /// the worker count owned by this pool.
    pub fn assembler(&self, target: Target) -> Result<RayonAssemblyBackend<'_>, Diagnostic> {
        self.require_target(target)?;
        Ok(RayonAssemblyBackend { pool: self })
    }

    fn require_target(&self, target: Target) -> Result<(), Diagnostic> {
        let Target::HostCpu { threads } = target else {
            return Err(invalid_realization(
                "a CPU thread pool can execute only a HostCpu target",
            ));
        };
        if threads != self.workers {
            return Err(invalid_realization(format!(
                "resolved target requests {threads} worker(s), but this pool owns exactly {}",
                self.workers
            )));
        }
        Ok(())
    }
}

impl fmt::Debug for CpuThreadPool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CpuThreadPool")
            .field("workers", &self.workers)
            .finish_non_exhaustive()
    }
}

/// Ordered assembly adapter backed by one run-owned Rayon pool.
#[derive(Clone, Copy)]
pub struct RayonAssemblyBackend<'a> {
    pool: &'a CpuThreadPool,
}

impl fmt::Debug for RayonAssemblyBackend<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RayonAssemblyBackend")
            .field("workers", &self.pool.workers)
            .finish()
    }
}

impl AssemblyBackend for RayonAssemblyBackend<'_> {
    fn assemble(
        &self,
        plan: &AssemblyPlan,
        work: &dyn AssemblyWork,
    ) -> Result<AssemblyResult, Diagnostic> {
        let packet_count = work.packet_count();
        if packet_count == 0 {
            return Err(assembly_failed(
                "assembly work requires at least one logical packet",
            ));
        }
        let mut accumulator = AssemblyAccumulator::new(plan)?;
        for batch_start in (0..packet_count).step_by(ORDERED_ASSEMBLY_BATCH_LENGTH) {
            let batch_end = (batch_start + ORDERED_ASSEMBLY_BATCH_LENGTH).min(packet_count);
            let projected_packets = self.pool.pool.install(|| {
                let plan = accumulator.plan();
                (batch_start..batch_end)
                    .into_par_iter()
                    .map(|packet_index| {
                        (
                            packet_index,
                            work.evaluate(packet_index)
                                .and_then(|packet| packet.project(plan)),
                        )
                    })
                    .collect::<Vec<_>>()
            });
            for (packet_index, projected) in projected_packets {
                accumulator = accumulator.scatter_projected(packet_index, &projected?)?;
            }
        }
        accumulator.finish(ExecutionReport::host(RAYON_EXECUTION, self.pool.workers))
    }
}

/// Solver decorator that changes operator placement but not solver policy.
#[derive(Clone, Copy)]
pub struct ThreadedLinearSolver<'a> {
    pool: &'a CpuThreadPool,
    backend: &'a dyn LinearSolverBackend,
}

impl fmt::Debug for ThreadedLinearSolver<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ThreadedLinearSolver")
            .field("workers", &self.pool.workers)
            .field("backend", &self.backend.id())
            .finish()
    }
}

impl LinearSolverBackend for ThreadedLinearSolver<'_> {
    fn provider(&self) -> SolverProvider {
        self.backend.provider()
    }

    fn capabilities(&self) -> SolverCapabilities {
        let backend = self.backend.capabilities();
        SolverCapabilities::exact(
            backend
                .combinations()
                .iter()
                .filter(|entry| entry.reduction == ReductionPolicy::Reproducible)
                .copied(),
        )
        .expect("construction admitted at least one reproducible backend tuple")
    }

    fn solve_with_execution(
        &self,
        problem: &LinearProblem<'_>,
        plan: SolverPlan,
        execution: &dyn ReplicatedLinearExecution,
    ) -> Result<LinearSolution, Diagnostic> {
        if execution.report() != SERIAL_LINEAR_EXECUTION.report() {
            return Err(invalid_realization(
                "a Rayon solver cannot be nested inside another replicated execution",
            ));
        }
        self.capabilities()
            .require_problem(plan, ScalarType::F64, problem.properties())?;
        let execution = RayonLinearExecution { pool: self.pool };
        self.backend.solve_with_execution(problem, plan, &execution)
    }
}

#[derive(Clone, Copy)]
struct RayonLinearExecution<'a> {
    pool: &'a CpuThreadPool,
}

impl fmt::Debug for RayonLinearExecution<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RayonLinearExecution")
            .field("workers", &self.pool.workers)
            .finish()
    }
}

impl ReplicatedLinearExecution for RayonLinearExecution<'_> {
    fn provider(&self) -> ExecutionProvider {
        RAYON_EXECUTION_PROVIDER
    }

    fn report(&self) -> ExecutionReport {
        ExecutionReport::host(RAYON_EXECUTION, self.pool.workers)
    }

    fn require_reduction(&self, policy: ReductionPolicy) -> Result<(), Diagnostic> {
        if policy == ReductionPolicy::Reproducible {
            Ok(())
        } else {
            Err(invalid_realization(
                "the first Rayon execution admits only reproducible reductions",
            ))
        }
    }

    fn apply(
        &self,
        operator: &dyn LinearOperator,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        if input.len() != operator.columns() || output.len() != operator.rows() {
            return Err(solve_failed(format!(
                "threaded {}x{} operator received input/output sizes {}/{}",
                operator.rows(),
                operator.columns(),
                input.len(),
                output.len()
            )));
        }
        if self.pool.workers == NonZeroUsize::MIN {
            return operator.apply(input, output);
        }
        if output.is_empty() {
            return operator.apply(input, output);
        }
        let row_action = operator.row_action().ok_or_else(|| {
            invalid_realization(
                "multi-worker CPU execution requires an operator with disjoint row actions",
            )
        })?;
        let task_count = self.pool.workers.get().min(output.len());
        let chunk_size = output.len().div_ceil(task_count);
        let failure = Mutex::new(None::<(usize, Diagnostic)>);
        self.pool.pool.install(|| {
            output
                .par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(chunk_index, chunk)| {
                    let start = chunk_index * chunk_size;
                    let end = start + chunk.len();
                    if let Err(diagnostic) = row_action.apply_rows(start..end, input, chunk) {
                        let mut first = failure
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        if first
                            .as_ref()
                            .is_none_or(|(current_index, _)| chunk_index < *current_index)
                        {
                            *first = Some((chunk_index, diagnostic));
                        }
                    }
                });
        });
        if let Some((_, diagnostic)) = failure
            .into_inner()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
        {
            return Err(diagnostic);
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err(solve_failed(
                "threaded operator action produced a non-finite value",
            ));
        }
        Ok(())
    }

    fn inner_product(&self, action: FixedOrderInnerProduct<'_>) -> Result<f64, Diagnostic> {
        let partials = self.pool.pool.install(|| {
            (0..action.partial_count())
                .into_par_iter()
                .map(|index| action.evaluate_partial(index))
                .collect::<Vec<_>>()
        });
        let partials = partials.into_iter().collect::<Result<Vec<_>, _>>()?;
        action.finish(&partials)
    }
}

fn invalid_realization(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_REALIZATION, message)
}

fn solve_failed(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::NUMERICAL_SOLVE_FAILED, message)
}

fn assembly_failed(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::ASSEMBLY_FAILED, message)
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use eqiora_assembly::{
        AssemblyMap, AssemblyPacket, AssemblyTarget, DofId, IndexedAssemblyWork, LinearSystem,
        LocalContribution, LocalUnknown, REFERENCE_ASSEMBLY_BACKEND, TargetAssemblyMap,
    };
    use eqiora_solver::{
        DiagonalAvailability, LinearOperatorProperties, LinearSolver, PreconditionerPolicy,
        REFERENCE_LINEAR_SOLVER, REPRODUCIBLE_INNER_PRODUCT_CHUNK_LENGTH, ReductionPolicy,
        RowLinearAction,
    };

    use super::*;

    fn system_bits(system: &LinearSystem) -> Vec<u8> {
        system
            .matrix()
            .values()
            .iter()
            .chain(system.rhs())
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect()
    }

    #[derive(Debug)]
    struct DenseRows {
        values: [[f64; 4]; 4],
    }

    impl RowLinearAction for DenseRows {
        fn apply_rows(
            &self,
            rows: Range<usize>,
            input: &[f64],
            output: &mut [f64],
        ) -> Result<(), Diagnostic> {
            if input.len() != 4 || rows.end > 4 || output.len() != rows.len() {
                return Err(solve_failed("test row action shape mismatch"));
            }
            for (row, result) in rows.zip(output) {
                *result = self.values[row]
                    .iter()
                    .zip(input)
                    .map(|(left, right)| left * right)
                    .sum();
            }
            Ok(())
        }
    }

    impl LinearOperator for DenseRows {
        fn rows(&self) -> usize {
            4
        }

        fn columns(&self) -> usize {
            4
        }

        fn apply(&self, input: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
            self.apply_rows(0..4, input, output)
        }

        fn row_action(&self) -> Option<&dyn RowLinearAction> {
            Some(self)
        }

        fn diagonal(&self, output: &mut [f64]) -> Result<DiagonalAvailability, Diagnostic> {
            if output.len() != 4 {
                return Err(solve_failed("test diagonal shape mismatch"));
            }
            for (index, value) in output.iter_mut().enumerate() {
                *value = self.values[index][index];
            }
            Ok(DiagonalAvailability::Available)
        }
    }

    #[derive(Debug)]
    struct Unpartitioned;

    impl LinearOperator for Unpartitioned {
        fn rows(&self) -> usize {
            1
        }

        fn columns(&self) -> usize {
            1
        }

        fn apply(&self, input: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
            output[0] = input[0];
            Ok(())
        }
    }

    #[derive(Debug)]
    struct FastOnlyBackend;

    impl LinearSolverBackend for FastOnlyBackend {
        fn provider(&self) -> SolverProvider {
            SolverProvider::new(
                eqiora_solver::BackendId::new("test.fast-only"),
                "0.0.0-test",
                &[],
            )
        }

        fn capabilities(&self) -> SolverCapabilities {
            SolverCapabilities::new(
                [LinearSolver::ConjugateGradient],
                [PreconditionerPolicy::Identity],
                [ReductionPolicy::Fast],
                [ScalarType::F64],
            )
            .unwrap()
        }

        fn solve_with_execution(
            &self,
            _problem: &LinearProblem<'_>,
            _plan: SolverPlan,
            _execution: &dyn ReplicatedLinearExecution,
        ) -> Result<LinearSolution, Diagnostic> {
            unreachable!("construction must reject this backend")
        }
    }

    fn plan() -> SolverPlan {
        SolverPlan::new(
            LinearSolver::ConjugateGradient,
            1.0e-13,
            1.0e-14,
            NonZeroUsize::new(100).unwrap(),
        )
        .unwrap()
        .with_preconditioner(PreconditionerPolicy::Jacobi)
        .with_reduction(ReductionPolicy::Reproducible)
    }

    #[test]
    fn rayon_provider_names_the_exact_adapter_and_library_release() {
        RAYON_EXECUTION_PROVIDER.validate().unwrap();
        assert_eq!(RAYON_EXECUTION_PROVIDER.id(), RAYON_EXECUTION);
        assert_eq!(
            RAYON_EXECUTION_PROVIDER.implementation_version(),
            RAYON_ADAPTER_VERSION
        );
        assert_eq!(RAYON_EXECUTION_PROVIDER.libraries().len(), 1);
        assert_eq!(RAYON_EXECUTION_PROVIDER.libraries()[0].name(), "rayon");
        assert_eq!(
            RAYON_EXECUTION_PROVIDER.libraries()[0].version(),
            RAYON_VERSION
        );
    }

    #[test]
    fn owned_pool_executes_partitioned_operator_with_exact_evidence() {
        let pool = CpuThreadPool::new(NonZeroUsize::new(4).unwrap()).unwrap();
        assert_eq!(pool.pool.install(rayon::current_num_threads), 4);
        assert!(
            pool.pool
                .broadcast(|_| std::thread::current()
                    .name()
                    .unwrap_or_default()
                    .starts_with("eqiora-rayon-"))
                .into_iter()
                .all(|named| named)
        );
        let operator = DenseRows {
            values: [
                [4.0, 1.0, 0.0, 0.0],
                [1.0, 4.0, 1.0, 0.0],
                [0.0, 1.0, 4.0, 1.0],
                [0.0, 0.0, 1.0, 3.0],
            ],
        };
        let right_hand_side = [1.0, 2.0, 3.0, 4.0];
        let problem = LinearProblem::new(
            &operator,
            &right_hand_side,
            LinearOperatorProperties::SymmetricPositiveDefinite,
        )
        .unwrap();
        let direct = REFERENCE_LINEAR_SOLVER.solve(&problem, plan()).unwrap();
        let threaded_solver = pool
            .solver(
                Target::HostCpu {
                    threads: NonZeroUsize::new(4).unwrap(),
                },
                &REFERENCE_LINEAR_SOLVER,
            )
            .unwrap();
        assert_eq!(
            threaded_solver.provider(),
            REFERENCE_LINEAR_SOLVER.provider()
        );
        let threaded = threaded_solver.solve(&problem, plan()).unwrap();
        assert_eq!(threaded.values(), direct.values());
        assert_eq!(threaded.report().backend(), direct.report().backend());
        assert_eq!(
            threaded.report().solver_provider(),
            REFERENCE_LINEAR_SOLVER.provider()
        );
        assert_eq!(
            threaded.report().execution_provider(),
            RAYON_EXECUTION_PROVIDER
        );
        assert_eq!(
            threaded.report().verification_provider(),
            RAYON_EXECUTION_PROVIDER
        );
        assert_eq!(
            threaded.report().execution(),
            ExecutionReport::host(RAYON_EXECUTION, NonZeroUsize::new(4).unwrap())
        );
        assert_eq!(
            threaded.report().verification(),
            ExecutionReport::host(RAYON_EXECUTION, NonZeroUsize::new(4).unwrap())
        );
        assert_eq!(threaded.report().reduction(), ReductionPolicy::Reproducible);
        assert_numerical_report_eq(threaded.report(), direct.report());
    }

    #[test]
    fn fixed_order_reduction_is_bit_identical_for_one_and_four_workers() {
        let mut right = vec![1.0; 2 * REPRODUCIBLE_INNER_PRODUCT_CHUNK_LENGTH + 1];
        right[0] = 1.0e16;
        right[REPRODUCIBLE_INNER_PRODUCT_CHUNK_LENGTH] = -1.0e16;
        let left = vec![1.0; right.len()];
        let action = FixedOrderInnerProduct::new(&left, &right).unwrap();
        let serial = SERIAL_LINEAR_EXECUTION.inner_product(action).unwrap();

        for workers in [NonZeroUsize::MIN, NonZeroUsize::new(4).unwrap()] {
            let pool = CpuThreadPool::new(workers).unwrap();
            let execution = RayonLinearExecution { pool: &pool };
            assert_eq!(execution.inner_product(action).unwrap(), serial);
            assert_eq!(
                execution.report(),
                ExecutionReport::host(RAYON_EXECUTION, workers)
            );
        }
    }

    #[test]
    fn ordered_assembly_is_bit_identical_at_every_worker_count() {
        let plan = AssemblyPlan::new(vec![AssemblyTarget::new(1).unwrap()]).unwrap();
        let target = plan.target_id(0).unwrap();
        let packet_count = ORDERED_ASSEMBLY_BATCH_LENGTH + 3;
        let work = IndexedAssemblyWork::new(packet_count, |index| {
            let rhs = match index {
                0 => 1.0e16,
                1 => 1.0,
                2 => -1.0e16,
                _ => 1.0,
            };
            let dof = DofId::new(0);
            AssemblyPacket::new(
                LocalContribution::new(1, 1, vec![1.0], vec![rhs])?,
                vec![TargetAssemblyMap::new(
                    target,
                    AssemblyMap::new(vec![Some(dof)], vec![LocalUnknown::Free(dof)])?,
                )],
            )
        });
        let reference = REFERENCE_ASSEMBLY_BACKEND.assemble(&plan, &work).unwrap();
        let reference_bits = system_bits(reference.system(target).unwrap());

        for worker_count in [1, 4, 16, 64] {
            let workers = NonZeroUsize::new(worker_count).unwrap();
            let pool = CpuThreadPool::new(workers).unwrap();
            let threaded = pool
                .assembler(Target::HostCpu { threads: workers })
                .unwrap()
                .assemble(&plan, &work)
                .unwrap();

            assert_eq!(
                system_bits(threaded.system(target).unwrap()),
                reference_bits
            );
            assert_eq!(threaded.report().packet_count(), packet_count);
            assert_eq!(threaded.report().target_count(), 1);
            assert_eq!(
                threaded.report().execution(),
                ExecutionReport::host(RAYON_EXECUTION, workers)
            );
        }
    }

    #[test]
    fn ordered_assembly_reports_the_lowest_failing_index() {
        let plan = AssemblyPlan::new(vec![AssemblyTarget::new(1).unwrap()]).unwrap();
        let target = plan.target_id(0).unwrap();
        let work = IndexedAssemblyWork::new(12, |index| {
            if index == 3 || index == 7 {
                return Err(assembly_failed(format!("packet {index} failed")));
            }
            let dof = DofId::new(0);
            AssemblyPacket::new(
                LocalContribution::new(1, 1, vec![1.0], vec![0.0])?,
                vec![TargetAssemblyMap::new(
                    target,
                    AssemblyMap::new(vec![Some(dof)], vec![LocalUnknown::Free(dof)])?,
                )],
            )
        });
        let pool = CpuThreadPool::new(NonZeroUsize::new(4).unwrap()).unwrap();
        let diagnostic = pool
            .assembler(Target::HostCpu {
                threads: NonZeroUsize::new(4).unwrap(),
            })
            .unwrap()
            .assemble(&plan, &work)
            .unwrap_err();
        assert!(diagnostic.message().contains("packet 3 failed"));
    }

    #[test]
    fn ordered_assembly_reports_lowest_projection_failure_at_every_worker_count() {
        let plan = AssemblyPlan::new(vec![AssemblyTarget::new(1).unwrap()]).unwrap();
        let valid_target = plan.target_id(0).unwrap();
        let foreign_plan = AssemblyPlan::new(vec![AssemblyTarget::new(1).unwrap(); 8]).unwrap();
        let work = IndexedAssemblyWork::new(12, |index| {
            let target = match index {
                2 | 7 => foreign_plan.target_id(index).unwrap(),
                _ => valid_target,
            };
            let dof = DofId::new(0);
            AssemblyPacket::new(
                LocalContribution::new(1, 1, vec![1.0], vec![0.0])?,
                vec![TargetAssemblyMap::new(
                    target,
                    AssemblyMap::new(vec![Some(dof)], vec![LocalUnknown::Free(dof)])?,
                )],
            )
        });

        for worker_count in [1, 4, 16, 64] {
            let workers = NonZeroUsize::new(worker_count).unwrap();
            let pool = CpuThreadPool::new(workers).unwrap();
            let diagnostic = pool
                .assembler(Target::HostCpu { threads: workers })
                .unwrap()
                .assemble(&plan, &work)
                .unwrap_err();
            assert_eq!(diagnostic.code(), codes::ASSEMBLY_FAILED);
            assert!(
                diagnostic
                    .message()
                    .contains("references target 2 outside plan count 1")
            );
        }
    }

    #[test]
    fn rayon_execution_rejects_fast_reduction_and_nesting() {
        let pool = CpuThreadPool::new(NonZeroUsize::new(2).unwrap()).unwrap();
        let execution = RayonLinearExecution { pool: &pool };
        assert_eq!(
            execution
                .require_reduction(ReductionPolicy::Fast)
                .unwrap_err()
                .code(),
            codes::INVALID_REALIZATION
        );
        assert_eq!(
            pool.solver(
                Target::HostCpu {
                    threads: NonZeroUsize::new(2).unwrap(),
                },
                &FastOnlyBackend,
            )
            .unwrap_err()
            .code(),
            codes::INVALID_REALIZATION
        );

        let operator = DenseRows {
            values: [
                [4.0, 1.0, 0.0, 0.0],
                [1.0, 4.0, 1.0, 0.0],
                [0.0, 1.0, 4.0, 1.0],
                [0.0, 0.0, 1.0, 3.0],
            ],
        };
        let problem = LinearProblem::new(
            &operator,
            &[1.0, 2.0, 3.0, 4.0],
            LinearOperatorProperties::SymmetricPositiveDefinite,
        )
        .unwrap();
        let threaded = pool
            .solver(
                Target::HostCpu {
                    threads: NonZeroUsize::new(2).unwrap(),
                },
                &REFERENCE_LINEAR_SOLVER,
            )
            .unwrap();
        assert_eq!(
            threaded
                .solve_with_execution(&problem, plan(), &execution)
                .unwrap_err()
                .code(),
            codes::INVALID_REALIZATION
        );
    }

    #[test]
    fn multi_worker_execution_fails_closed_without_row_capability() {
        let pool = CpuThreadPool::new(NonZeroUsize::new(2).unwrap()).unwrap();
        assert_eq!(
            pool.solver(
                Target::HostCpu {
                    threads: NonZeroUsize::new(1).unwrap(),
                },
                &REFERENCE_LINEAR_SOLVER,
            )
            .unwrap_err()
            .code(),
            codes::INVALID_REALIZATION
        );
        let problem = LinearProblem::new(
            &Unpartitioned,
            &[1.0],
            LinearOperatorProperties::SymmetricPositiveDefinite,
        )
        .unwrap();
        assert_eq!(
            pool.solver(
                Target::HostCpu {
                    threads: NonZeroUsize::new(2).unwrap(),
                },
                &REFERENCE_LINEAR_SOLVER,
            )
            .unwrap()
            .solve(&problem, plan())
            .unwrap_err()
            .code(),
            codes::INVALID_REALIZATION
        );
    }

    fn assert_numerical_report_eq(
        left: &eqiora_solver::SolveReport,
        right: &eqiora_solver::SolveReport,
    ) {
        assert_eq!(left.backend(), right.backend());
        assert_eq!(left.orientation(), right.orientation());
        assert_eq!(left.algorithm(), right.algorithm());
        assert_eq!(left.preconditioner(), right.preconditioner());
        assert_eq!(left.reduction(), right.reduction());
        assert_eq!(left.reason(), right.reason());
        assert_eq!(left.completed_iterations(), right.completed_iterations());
        assert_eq!(left.initial_residual_norm(), right.initial_residual_norm());
        assert_eq!(
            left.reported_residual_norm(),
            right.reported_residual_norm()
        );
        assert_eq!(left.true_residual_norm(), right.true_residual_norm());
        assert_eq!(left.residual_target(), right.residual_target());
    }
}
