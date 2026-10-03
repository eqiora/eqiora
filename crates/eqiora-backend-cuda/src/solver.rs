use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaEvent, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use eqiora_core::{Diagnostic, ScalarType};
use eqiora_device::{
    Completion, DeviceBufferDescriptor, DeviceDescriptor, Fence, QueueSlot, QueueTimeline,
    TransferEvidence, WaitedCompletion,
};
use eqiora_execution::{
    AcceptedLinearExecution, AdmittedExecution, CUDA_LINEAR_DEVICE_CAPABILITIES,
    CsrDeviceTransferEvidence, CudaLinearExecutionTrace, DeviceValueGeneration,
};
use eqiora_solver::{
    BackendId, CanonicalCsrSystemView, ConvergenceReason, ExecutionId, ExecutionProvider,
    ExecutionReport, LinearSolution, LinearSolver, ProviderLibrary, SERIAL_LINEAR_EXECUTION,
    SolverPlan, SolverProvider, accept_linear_solution_with_verifier,
};

use crate::blas::{BlasError, BlasHandle};
use crate::ffi::{CusparseHandle, SpmvPlan};
use crate::runtime::{
    CudaComputeCapability, CudaDeviceUuid, CudaLibraryVersions, convert_indices, descriptor,
    discover_cuda_devices, driver_failed, materialize_queue, solve_failed, transfer_to_device,
    transfer_to_host, unsupported,
};
use crate::{CUDA_ADAPTER_VERSION, CUDA_BINDING_TOOLKIT, CUDARC_VERSION};

mod capabilities;
mod inverse_diagonal;

use inverse_diagonal::inverse_diagonal;

/// Stable CUDA Krylov backend identity.
pub const CUDA_LINEAR_SOLVER_BACKEND: BackendId = BackendId::new("eqiora.cuda.krylov");

/// Stable CUDA execution identity, separate from the solver algorithm.
pub const CUDA_LINEAR_EXECUTION: ExecutionId = ExecutionId::new("eqiora.cuda.single-device");

const CUDA_EXECUTION_LIBRARIES: &[ProviderLibrary] = &[
    ProviderLibrary::new("cuda-binding-toolkit", CUDA_BINDING_TOOLKIT),
    ProviderLibrary::new("cudarc", CUDARC_VERSION),
];

/// Exact declared CUDA Krylov solver release.
pub const CUDA_LINEAR_SOLVER_PROVIDER: SolverProvider =
    SolverProvider::new(CUDA_LINEAR_SOLVER_BACKEND, CUDA_ADAPTER_VERSION, &[]);

/// Exact declared CUDA execution release and binding versions.
pub const CUDA_LINEAR_EXECUTION_PROVIDER: ExecutionProvider = ExecutionProvider::new(
    CUDA_LINEAR_EXECUTION,
    CUDA_ADAPTER_VERSION,
    CUDA_EXECUTION_LIBRARIES,
);

/// Explicit matrix, problem-vector, preconditioner, and result transfers for
/// one CUDA linear solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CudaLinearTransferEvidence {
    row_offsets: TransferEvidence<i64>,
    column_indices: TransferEvidence<i64>,
    values: TransferEvidence<f64>,
    right_hand_side: TransferEvidence<f64>,
    initial_guess: TransferEvidence<f64>,
    inverse_diagonal: Option<TransferEvidence<f64>>,
    solution: TransferEvidence<f64>,
}

impl CudaLinearTransferEvidence {
    /// Finalized CSR row offsets uploaded once.
    #[must_use]
    pub const fn row_offsets(self) -> TransferEvidence<i64> {
        self.row_offsets
    }

    /// Finalized CSR column indices uploaded once.
    #[must_use]
    pub const fn column_indices(self) -> TransferEvidence<i64> {
        self.column_indices
    }

    /// Finalized CSR values uploaded once.
    #[must_use]
    pub const fn values(self) -> TransferEvidence<f64> {
        self.values
    }

    /// Right-hand side uploaded once.
    #[must_use]
    pub const fn right_hand_side(self) -> TransferEvidence<f64> {
        self.right_hand_side
    }

    /// Explicit or implicit-zero initial guess uploaded once.
    #[must_use]
    pub const fn initial_guess(self) -> TransferEvidence<f64> {
        self.initial_guess
    }

    /// Jacobi inverse diagonal upload, absent for identity preconditioning.
    #[must_use]
    pub const fn inverse_diagonal(self) -> Option<TransferEvidence<f64>> {
        self.inverse_diagonal
    }

    /// Accepted candidate copied back for independent host verification.
    #[must_use]
    pub const fn solution(self) -> TransferEvidence<f64> {
        self.solution
    }

    /// Total explicit host-to-device bytes.
    #[must_use]
    pub fn host_to_device_bytes(self) -> usize {
        let required = [
            self.row_offsets.plan().bytes().expect("validated transfer"),
            self.column_indices
                .plan()
                .bytes()
                .expect("validated transfer"),
            self.values.plan().bytes().expect("validated transfer"),
            self.right_hand_side
                .plan()
                .bytes()
                .expect("validated transfer"),
            self.initial_guess
                .plan()
                .bytes()
                .expect("validated transfer"),
        ]
        .into_iter()
        .sum::<usize>();
        required
            + self.inverse_diagonal.map_or(0, |transfer| {
                transfer.plan().bytes().expect("validated transfer")
            })
    }

    /// Accepted solution bytes copied back to the host.
    #[must_use]
    pub fn device_to_host_bytes(self) -> usize {
        self.solution.plan().bytes().expect("validated transfer")
    }
}

/// Auditable device evidence surrounding the backend-neutral solve report.
#[derive(Debug, Clone, PartialEq)]
pub struct CudaLinearSolveEvidence {
    device: DeviceDescriptor,
    physical_uuid: CudaDeviceUuid,
    compute_capability: CudaComputeCapability,
    versions: CudaLibraryVersions,
    transfers: CudaLinearTransferEvidence,
    inputs_ready: WaitedCompletion,
    solve_visible: WaitedCompletion,
    solution_visible: WaitedCompletion,
    workspace_bytes: usize,
    timings: eqiora_device::DeviceExecutionTimings,
}

impl CudaLinearSolveEvidence {
    /// Exact selected CUDA device and admitted capabilities.
    #[must_use]
    pub const fn device(&self) -> &DeviceDescriptor {
        &self.device
    }

    /// Physical UUID observed beside the selected runtime-local device.
    #[must_use]
    pub const fn physical_uuid(&self) -> CudaDeviceUuid {
        self.physical_uuid
    }

    /// Compute capability reported for the exact selected device.
    #[must_use]
    pub const fn compute_capability(&self) -> CudaComputeCapability {
        self.compute_capability
    }

    /// CUDA driver, cuSPARSE, cuBLAS, binding, and adapter versions.
    #[must_use]
    pub const fn versions(&self) -> CudaLibraryVersions {
        self.versions
    }

    /// Explicit host/device movement evidence.
    #[must_use]
    pub const fn transfers(&self) -> CudaLinearTransferEvidence {
        self.transfers
    }

    /// Successful wait after every canonical input upload.
    #[must_use]
    pub const fn inputs_ready(&self) -> WaitedCompletion {
        self.inputs_ready
    }

    /// Completion of the accepted device solve phase.
    #[must_use]
    pub const fn solve_completion(&self) -> Completion {
        self.solve_visible.completion()
    }

    /// Successful wait for the device solve fence.
    #[must_use]
    pub const fn solve_visible(&self) -> WaitedCompletion {
        self.solve_visible
    }

    /// Successful wait for the complete D2H output fence.
    #[must_use]
    pub const fn solution_visible(&self) -> WaitedCompletion {
        self.solution_visible
    }

    /// Retained external cuSPARSE workspace size.
    #[must_use]
    pub const fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }

    /// Separately observed setup, movement, solve, and verification phases.
    #[must_use]
    pub const fn timings(&self) -> eqiora_device::DeviceExecutionTimings {
        self.timings
    }

    fn execution_trace(&self) -> Result<CudaLinearExecutionTrace, Diagnostic> {
        let solution = self.transfers.solution().plan().source();
        let eqiora_device::MemoryRegion::Device(solution) = solution else {
            return Err(unsupported("CUDA solution transfer lost its device source"));
        };
        let initial = DeviceValueGeneration::new(solution.id(), std::num::NonZeroU64::MIN);
        let solved = DeviceValueGeneration::new(
            solution.id(),
            std::num::NonZeroU64::new(2).expect("two is non-zero"),
        );
        CudaLinearExecutionTrace::new(
            CsrDeviceTransferEvidence::new(
                self.transfers.row_offsets(),
                self.transfers.column_indices(),
                self.transfers.values(),
                self.transfers.right_hand_side(),
                self.transfers.initial_guess(),
                self.transfers.inverse_diagonal(),
                self.transfers.solution(),
            ),
            self.inputs_ready,
            self.solve_visible,
            self.solution_visible,
            initial,
            solved,
            solved,
            self.workspace_bytes,
        )
    }
}

/// Independently accepted solution and CUDA-specific execution evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct CudaLinearSolveResult {
    solution: LinearSolution,
    evidence: CudaLinearSolveEvidence,
}

/// Graph-bound accepted execution paired with CUDA-specific runtime evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct AcceptedCudaLinearSolveResult {
    accepted: AcceptedLinearExecution,
    evidence: CudaLinearSolveEvidence,
}

impl AcceptedCudaLinearSolveResult {
    /// Backend-neutral accepted solution and execution receipt.
    #[must_use]
    pub const fn accepted(&self) -> &AcceptedLinearExecution {
        &self.accepted
    }

    /// CUDA runtime, library, transfer, fence, workspace, and timing evidence.
    #[must_use]
    pub const fn evidence(&self) -> &CudaLinearSolveEvidence {
        &self.evidence
    }

    /// Consume without copying either accepted output or adapter evidence.
    #[must_use]
    pub fn into_parts(self) -> (AcceptedLinearExecution, CudaLinearSolveEvidence) {
        (self.accepted, self.evidence)
    }
}

impl CudaLinearSolveResult {
    /// Backend-neutral accepted values and convergence report.
    #[must_use]
    pub const fn solution(&self) -> &LinearSolution {
        &self.solution
    }

    /// CUDA versions, transfers, completion, and timings.
    #[must_use]
    pub const fn evidence(&self) -> &CudaLinearSolveEvidence {
        &self.evidence
    }

    /// Consume the result into independently owned solution and evidence.
    #[must_use]
    pub fn into_parts(self) -> (LinearSolution, CudaLinearSolveEvidence) {
        (self.solution, self.evidence)
    }
}

/// One CUDA device selected for CSR Krylov execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaLinearSolver {
    device_ordinal: u16,
}

impl CudaLinearSolver {
    /// Select a runtime-visible device without loading CUDA yet.
    #[must_use]
    pub const fn new(device_ordinal: u16) -> Self {
        Self { device_ordinal }
    }

    /// Admit one already discovered device for this solver slice without
    /// creating a context, queue, allocation, or vendor-library handle.
    ///
    /// # Errors
    /// Returns `EQ0807` for a descriptor from another runtime, a descriptor
    /// with a different deployment-visible ordinal, or the first missing
    /// device capability, in that order.
    pub fn admit_device(self, device: &DeviceDescriptor) -> Result<(), Diagnostic> {
        if device.id().runtime() != crate::CUDA_RUNTIME_ID {
            return Err(unsupported(format!(
                "CUDA solver requires runtime `{}`, received `{}`",
                crate::CUDA_RUNTIME_ID.as_str(),
                device.id().runtime().as_str(),
            )));
        }
        if device.id().ordinal() != self.device_ordinal {
            return Err(unsupported(format!(
                "CUDA solver selected device {}, received descriptor for device {}",
                self.device_ordinal,
                device.id().ordinal(),
            )));
        }
        device.require(CUDA_LINEAR_DEVICE_CAPABILITIES)
    }

    /// Solve one finalized assembled system with a single resident CSR and
    /// resident Krylov vectors, then accept it through the independent serial
    /// host operator and fixed-order reduction oracle.
    ///
    /// # Errors
    /// Returns a stable diagnostic for unsupported policy/property, invalid
    /// CSR or preconditioner, unavailable CUDA capability, library failure,
    /// numerical breakdown/non-convergence, or host true-residual rejection.
    pub fn solve(
        self,
        system: &CanonicalCsrSystemView,
        initial_guess: Option<&[f64]>,
        plan: SolverPlan,
    ) -> Result<CudaLinearSolveResult, Diagnostic> {
        solve(
            self.device_ordinal,
            None,
            None,
            None,
            system,
            initial_guess,
            plan,
        )
    }
}

/// L3 adapter seam that consumes an exact L2 CUDA execution admission.
///
/// This trait is intentionally not re-exported by the `eqiora` facade: its
/// argument is an internal execution-layer token used to keep runtime adapters
/// honest while the eventual public run API remains small.
pub trait CudaAdmittedExecutionAdapter {
    /// Execute one graph-bound implicit-zero solve without reselecting its
    /// provider, device, queue, system, or solver policy.
    ///
    /// # Errors
    /// Returns a stable diagnostic for any binding substitution, runtime drift,
    /// CUDA failure, numerical rejection, transfer/fence contradiction, or
    /// receipt replay failure.
    fn execute_admitted(
        self,
        admitted: AdmittedExecution<'_>,
    ) -> Result<AcceptedCudaLinearSolveResult, Diagnostic>;
}

impl CudaAdmittedExecutionAdapter for CudaLinearSolver {
    fn execute_admitted(
        self,
        admitted: AdmittedExecution<'_>,
    ) -> Result<AcceptedCudaLinearSolveResult, Diagnostic> {
        let executor = admitted
            .binding()
            .cuda_executor()
            .ok_or_else(|| unsupported("CUDA adapter requires a device deployment binding"))?;
        if executor.solver_provider() != CUDA_LINEAR_SOLVER_PROVIDER {
            return Err(unsupported(
                "CUDA deployment selected a different linear-solver backend",
            ));
        }
        if executor.execution_provider() != CUDA_LINEAR_EXECUTION_PROVIDER {
            return Err(unsupported(
                "CUDA deployment selected a different execution adapter",
            ));
        }
        if executor.device().id().runtime() != crate::CUDA_RUNTIME_ID {
            return Err(unsupported(
                "CUDA deployment selected a device from another runtime",
            ));
        }
        if executor.device().id().ordinal() != self.device_ordinal {
            return Err(unsupported(
                "CUDA solver ordinal differs from the admitted device",
            ));
        }
        if executor.queue().ordinal() != 0 {
            return Err(unsupported(
                "single-device CUDA execution v1 admits only queue ordinal zero",
            ));
        }
        let device = executor.device().clone();
        let queue = executor.queue();
        let minimum_device_payload_bytes =
            admitted.minimum_device_payload_bytes().ok_or_else(|| {
                unsupported(
                    "CUDA adapter received an admission without a device-memory lower bound",
                )
            })?;
        let result = solve(
            self.device_ordinal,
            Some(&device),
            Some(queue),
            Some(minimum_device_payload_bytes),
            admitted.system(),
            None,
            admitted.solver_plan(),
        )?;
        let (solution, evidence) = result.into_parts();
        let trace = evidence.execution_trace()?;
        let accepted = admitted.accept_cuda(solution, trace)?;
        Ok(AcceptedCudaLinearSolveResult { accepted, evidence })
    }
}

fn solve(
    device_ordinal: u16,
    expected_device: Option<&DeviceDescriptor>,
    expected_queue: Option<QueueSlot>,
    admitted_device_payload_bytes: Option<usize>,
    system: &CanonicalCsrSystemView,
    initial_guess: Option<&[f64]>,
    plan: SolverPlan,
) -> Result<CudaLinearSolveResult, Diagnostic> {
    let total_started = Instant::now();
    CudaLinearSolver::capabilities().require_problem(
        plan,
        eqiora_core::ScalarDomain::Real,
        ScalarType::F64,
        system.properties(),
    )?;
    let mut problem = system.linear_problem()?;
    if let Some(initial_guess) = initial_guess {
        problem = problem.with_initial_guess(initial_guess)?;
    }
    validate_canonical_system(system, initial_guess)?;
    let dimension = i32::try_from(system.rows())
        .map_err(|_| unsupported("CUDA Krylov dimension exceeds signed 32-bit cuBLAS size"))?;
    let rows = i64::from(dimension);
    let columns = i64::try_from(system.columns())
        .map_err(|_| unsupported("CSR column count exceeds signed 64-bit CUDA size"))?;
    let nonzeros = i64::try_from(system.values().len())
        .map_err(|_| unsupported("CSR nonzero count exceeds signed 64-bit CUDA size"))?;
    let row_offsets = convert_indices(system.row_offsets(), "row offset")?;
    let column_indices = convert_indices(system.column_indices(), "column index")?;
    let inverse_diagonal = inverse_diagonal(system, plan)?;
    let initial_values = initial_guess.map_or_else(|| vec![0.0; system.columns()], <[f64]>::to_vec);

    let discovered = discover_cuda_devices(crate::runtime::cuda_device_count()?)?
        .into_iter()
        .find(|candidate| candidate.descriptor.id().ordinal() == device_ordinal)
        .ok_or_else(|| unsupported(format!("CUDA device {device_ordinal} is not visible")))?;
    if expected_device.is_some_and(|expected| expected != &discovered.descriptor) {
        return Err(unsupported(
            "CUDA device capability snapshot changed after deployment binding",
        ));
    }
    let device = discovered.descriptor;
    CudaLinearSolver::new(device_ordinal).admit_device(&device)?;
    let queue_slot = expected_queue.unwrap_or_else(|| QueueSlot::new(device.id(), 0));
    if queue_slot.device() != device.id() {
        return Err(unsupported(
            "CUDA queue binding does not belong to the selected device",
        ));
    }

    let setup_started = Instant::now();
    let context = CudaContext::new(usize::from(device_ordinal)).map_err(driver_failed)?;
    let stream = context.new_stream().map_err(driver_failed)?;
    let queue = materialize_queue(queue_slot)?;
    let mut timeline = QueueTimeline::new(queue);
    let sparse = CusparseHandle::new(stream.cu_stream()).map_err(cusparse_failed)?;
    let blas = BlasHandle::new(stream.cu_stream()).map_err(blas_failed)?;

    let mut device_rows = stream
        .alloc_zeros::<i64>(row_offsets.len())
        .map_err(driver_failed)?;
    let mut device_columns = stream
        .alloc_zeros::<i64>(column_indices.len())
        .map_err(driver_failed)?;
    let mut device_values = stream
        .alloc_zeros::<f64>(system.values().len())
        .map_err(driver_failed)?;
    let mut device_rhs = stream
        .alloc_zeros::<f64>(system.right_hand_side().len())
        .map_err(driver_failed)?;
    let mut device_solution = stream
        .alloc_zeros::<f64>(initial_values.len())
        .map_err(driver_failed)?;
    let mut device_residual = stream
        .alloc_zeros::<f64>(initial_values.len())
        .map_err(driver_failed)?;
    let mut device_inverse_diagonal = inverse_diagonal
        .as_ref()
        .map(|values| {
            stream
                .alloc_zeros::<f64>(values.len())
                .map_err(driver_failed)
        })
        .transpose()?;
    let mut vectors = KrylovWorkspace::new(&stream, plan.algorithm(), initial_values.len())?;

    let (row_pointer, row_guard) = device_rows.device_ptr(&stream);
    let (column_pointer, column_guard) = device_columns.device_ptr(&stream);
    let (value_pointer, value_guard) = device_values.device_ptr(&stream);
    let (solution_pointer, solution_guard) = device_solution.device_ptr(&stream);
    let (residual_pointer, residual_guard) = device_residual.device_ptr_mut(&stream);
    let spmv = SpmvPlan::new(
        &sparse,
        rows,
        columns,
        nonzeros,
        row_pointer,
        column_pointer,
        value_pointer,
        solution_pointer,
        residual_pointer,
        false,
    )
    .map_err(cusparse_failed)?;
    drop(residual_guard);
    drop(solution_guard);
    drop(value_guard);
    drop(column_guard);
    drop(row_guard);
    if let Some(known_payload) = admitted_device_payload_bytes {
        let observed_lower_bound = known_payload
            .checked_add(spmv.workspace_bytes().max(1))
            .ok_or_else(|| unsupported("CUDA device-memory lower bound overflowed"))?;
        let device_total = usize::try_from(device.total_memory_bytes().get()).unwrap_or(usize::MAX);
        if observed_lower_bound > device_total {
            return Err(unsupported(format!(
                "known device payload plus cuSPARSE workspace requires at least {observed_lower_bound} bytes, selected device reports {device_total} total bytes",
            )));
        }
    }
    let mut device_sparse_workspace = stream
        .alloc_zeros::<u8>(spmv.workspace_bytes().max(1))
        .map_err(driver_failed)?;
    stream.synchronize().map_err(driver_failed)?;
    let setup = setup_started.elapsed();

    let descriptors = LinearBufferDescriptors::new(
        device.id(),
        row_offsets.len(),
        column_indices.len(),
        system.values().len(),
        system.right_hand_side().len(),
        initial_values.len(),
        inverse_diagonal.as_ref().map(Vec::len),
    )?;
    let host_to_device_started = Instant::now();
    stream
        .memcpy_htod(&row_offsets, &mut device_rows)
        .map_err(driver_failed)?;
    let row_completion = Completion::new(timeline.next_submission()?);
    stream
        .memcpy_htod(&column_indices, &mut device_columns)
        .map_err(driver_failed)?;
    let column_completion = Completion::new(timeline.next_submission()?);
    stream
        .memcpy_htod(system.values(), &mut device_values)
        .map_err(driver_failed)?;
    let values_completion = Completion::new(timeline.next_submission()?);
    stream
        .memcpy_htod(system.right_hand_side(), &mut device_rhs)
        .map_err(driver_failed)?;
    let right_hand_side_completion = Completion::new(timeline.next_submission()?);
    stream
        .memcpy_htod(&initial_values, &mut device_solution)
        .map_err(driver_failed)?;
    let initial_guess_completion = Completion::new(timeline.next_submission()?);
    let inverse_diagonal_completion = match (&inverse_diagonal, &mut device_inverse_diagonal) {
        (Some(host), Some(device)) => {
            stream.memcpy_htod(host, device).map_err(driver_failed)?;
            Some(Completion::new(timeline.next_submission()?))
        }
        (None, None) => None,
        _ => unreachable!("host and device Jacobi storage are constructed together"),
    };
    let inputs_fence = CudaEventFence::record(&stream, &mut timeline)?;
    let inputs_ready = WaitedCompletion::wait(&inputs_fence)?;
    let host_to_device = host_to_device_started.elapsed();

    let solve_started = Instant::now();
    let outcome = match &mut vectors {
        KrylovWorkspace::ConjugateGradient(workspace) => solve_cg(
            &stream,
            &sparse,
            &spmv,
            &blas,
            dimension,
            plan,
            &device_rhs,
            &mut device_solution,
            &mut device_residual,
            device_inverse_diagonal.as_ref(),
            &mut device_sparse_workspace,
            workspace,
        )?,
        KrylovWorkspace::BiConjugateGradientStabilized(workspace) => solve_bicgstab(
            &stream,
            &sparse,
            &spmv,
            &blas,
            dimension,
            plan,
            &device_rhs,
            &mut device_solution,
            &mut device_residual,
            device_inverse_diagonal.as_ref(),
            &mut device_sparse_workspace,
            workspace,
        )?,
        KrylovWorkspace::MinimumResidual(workspace) => solve_minres(
            &stream,
            &sparse,
            &spmv,
            &blas,
            dimension,
            plan,
            &device_rhs,
            &mut device_solution,
            &mut device_residual,
            &mut device_sparse_workspace,
            workspace,
        )?,
    };
    let solve_fence = CudaEventFence::record(&stream, &mut timeline)?;
    let solve_visible = WaitedCompletion::wait(&solve_fence)?;
    let solve_elapsed = solve_started.elapsed();

    let device_to_host_started = Instant::now();
    let values = stream.clone_dtoh(&device_solution).map_err(driver_failed)?;
    let solution_transfer_completion = Completion::new(timeline.next_submission()?);
    let solution_fence = CudaEventFence::record(&stream, &mut timeline)?;
    let solution_visible = WaitedCompletion::wait(&solution_fence)?;
    let device_to_host = device_to_host_started.elapsed();

    let verification_started = Instant::now();
    let solution = accept_linear_solution_with_verifier(
        &problem,
        plan,
        CUDA_LINEAR_SOLVER_PROVIDER,
        CUDA_LINEAR_EXECUTION_PROVIDER,
        ExecutionReport::cuda(CUDA_LINEAR_EXECUTION, device_ordinal),
        outcome.reason,
        outcome.iterations,
        outcome.reported_residual_norm,
        values,
        &SERIAL_LINEAR_EXECUTION,
    )?;
    let verification = verification_started.elapsed();
    let total = total_started.elapsed();

    let versions = CudaLibraryVersions {
        driver: crate::ffi::driver_version().map_err(driver_failed)?,
        cusparse: sparse.version().map_err(cusparse_failed)?,
        cublas: Some(blas.version().map_err(blas_failed)?),
        cudarc: CUDARC_VERSION,
        binding_toolkit: CUDA_BINDING_TOOLKIT,
    };
    let transfers = CudaLinearTransferEvidence {
        row_offsets: transfer_to_device(descriptors.rows, row_completion)?,
        column_indices: transfer_to_device(descriptors.columns, column_completion)?,
        values: transfer_to_device(descriptors.values, values_completion)?,
        right_hand_side: transfer_to_device(
            descriptors.right_hand_side,
            right_hand_side_completion,
        )?,
        initial_guess: transfer_to_device(descriptors.solution, initial_guess_completion)?,
        inverse_diagonal: match (descriptors.inverse_diagonal, inverse_diagonal_completion) {
            (Some(descriptor), Some(completion)) => {
                Some(transfer_to_device(descriptor, completion)?)
            }
            (None, None) => None,
            _ => unreachable!("Jacobi descriptor and completion are constructed together"),
        },
        solution: transfer_to_host(descriptors.solution, solution_transfer_completion)?,
    };
    let timings = eqiora_device::DeviceExecutionTimings::new(
        setup,
        host_to_device,
        solve_elapsed,
        device_to_host,
        verification,
        total,
    )?;
    Ok(CudaLinearSolveResult {
        solution,
        evidence: CudaLinearSolveEvidence {
            device,
            physical_uuid: discovered.physical_uuid,
            compute_capability: discovered.compute_capability,
            versions,
            transfers,
            inputs_ready,
            solve_visible,
            solution_visible,
            workspace_bytes: spmv.workspace_bytes(),
            timings,
        },
    })
}

#[derive(Debug)]
struct CudaEventFence {
    event: CudaEvent,
    completion: Completion,
}

impl CudaEventFence {
    fn record(stream: &CudaStream, timeline: &mut QueueTimeline) -> Result<Self, Diagnostic> {
        let event = stream.record_event(None).map_err(driver_failed)?;
        let completion = Completion::new(timeline.next_submission()?);
        Ok(Self { event, completion })
    }
}

impl Fence for CudaEventFence {
    fn completion(&self) -> Completion {
        self.completion
    }

    fn wait(&self) -> Result<(), Diagnostic> {
        self.event.synchronize().map_err(driver_failed)
    }
}

#[derive(Debug, Clone, Copy)]
struct SolverOutcome {
    reason: ConvergenceReason,
    iterations: usize,
    reported_residual_norm: f64,
}

struct CgWorkspace {
    preconditioned: CudaSlice<f64>,
    direction: CudaSlice<f64>,
    action: CudaSlice<f64>,
}

struct BicgstabWorkspace {
    shadow: CudaSlice<f64>,
    direction: CudaSlice<f64>,
    action: CudaSlice<f64>,
    preconditioned_direction: CudaSlice<f64>,
    intermediate: CudaSlice<f64>,
    preconditioned_intermediate: CudaSlice<f64>,
    intermediate_action: CudaSlice<f64>,
}

struct MinresWorkspace {
    previous_residual: CudaSlice<f64>,
    basis: CudaSlice<f64>,
    applied: CudaSlice<f64>,
    direction: CudaSlice<f64>,
    previous_direction: CudaSlice<f64>,
    older_direction: CudaSlice<f64>,
}

enum KrylovWorkspace {
    ConjugateGradient(CgWorkspace),
    BiConjugateGradientStabilized(Box<BicgstabWorkspace>),
    MinimumResidual(Box<MinresWorkspace>),
}

impl KrylovWorkspace {
    fn new(
        stream: &Arc<CudaStream>,
        algorithm: LinearSolver,
        dimension: usize,
    ) -> Result<Self, Diagnostic> {
        let allocate = || stream.alloc_zeros::<f64>(dimension).map_err(driver_failed);
        match algorithm {
            LinearSolver::ConjugateGradient => Ok(Self::ConjugateGradient(CgWorkspace {
                preconditioned: allocate()?,
                direction: allocate()?,
                action: allocate()?,
            })),
            LinearSolver::BiConjugateGradientStabilized => Ok(Self::BiConjugateGradientStabilized(
                Box::new(BicgstabWorkspace {
                    shadow: allocate()?,
                    direction: allocate()?,
                    action: allocate()?,
                    preconditioned_direction: allocate()?,
                    intermediate: allocate()?,
                    preconditioned_intermediate: allocate()?,
                    intermediate_action: allocate()?,
                }),
            )),
            LinearSolver::MinimumResidual => Ok(Self::MinimumResidual(Box::new(MinresWorkspace {
                previous_residual: allocate()?,
                basis: allocate()?,
                applied: allocate()?,
                direction: allocate()?,
                previous_direction: allocate()?,
                older_direction: allocate()?,
            }))),
            LinearSolver::SparseLu => {
                Err(unsupported("the CUDA backend does not implement sparse LU"))
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn solve_minres(
    stream: &CudaStream,
    sparse: &CusparseHandle,
    spmv: &SpmvPlan,
    blas: &BlasHandle,
    dimension: i32,
    plan: SolverPlan,
    rhs: &CudaSlice<f64>,
    solution: &mut CudaSlice<f64>,
    residual: &mut CudaSlice<f64>,
    sparse_workspace: &mut CudaSlice<u8>,
    vectors: &mut MinresWorkspace,
) -> Result<SolverOutcome, Diagnostic> {
    recompute_residual(
        stream,
        sparse,
        spmv,
        blas,
        dimension,
        rhs,
        solution,
        residual,
        sparse_workspace,
    )?;
    let right_hand_side_norm = norm(stream, blas, dimension, rhs)?;
    let target = plan.residual_target(right_hand_side_norm)?;
    let initial_residual_norm = norm(stream, blas, dimension, residual)?;
    if initial_residual_norm <= target {
        return Ok(SolverOutcome {
            reason: ConvergenceReason::InitialResidualSatisfied,
            iterations: 0,
            reported_residual_norm: initial_residual_norm,
        });
    }

    copy(
        stream,
        blas,
        dimension,
        residual,
        &mut vectors.previous_residual,
    )?;
    let mut beta = initial_residual_norm;
    let mut previous_beta = 0.0;
    let mut diagonal_bar = 0.0;
    let mut epsilon = 0.0;
    let mut residual_projection = initial_residual_norm;
    let mut cosine = -1.0;
    let mut sine = 0.0;
    let mut reported_residual_norm = initial_residual_norm;

    for iteration in 1..=plan.maximum_iterations().get() {
        require_positive(beta, "MINRES Lanczos normalization")?;
        copy(stream, blas, dimension, residual, &mut vectors.basis)?;
        scale(
            stream,
            blas,
            dimension,
            finite_scalar(1.0 / beta, "MINRES basis scale")?,
            &mut vectors.basis,
        )?;
        spmv_action(
            stream,
            sparse,
            spmv,
            &vectors.basis,
            &mut vectors.applied,
            sparse_workspace,
            1.0,
            0.0,
        )?;
        if iteration >= 2 {
            let recurrence = finite_scalar(beta / previous_beta, "MINRES Lanczos recurrence")?;
            axpy(
                stream,
                blas,
                dimension,
                -recurrence,
                &vectors.previous_residual,
                &mut vectors.applied,
            )?;
        }
        let diagonal = dot(stream, blas, dimension, &vectors.basis, &vectors.applied)?;
        let recurrence = finite_scalar(diagonal / beta, "MINRES diagonal recurrence")?;
        axpy(
            stream,
            blas,
            dimension,
            -recurrence,
            residual,
            &mut vectors.applied,
        )?;

        std::mem::swap(&mut vectors.previous_residual, residual);
        std::mem::swap(residual, &mut vectors.applied);
        previous_beta = beta;
        beta = norm(stream, blas, dimension, residual)?;

        let previous_epsilon = epsilon;
        let delta = finite_scalar(
            cosine * diagonal_bar + sine * diagonal,
            "MINRES rotated off-diagonal",
        )?;
        let diagonal_rotated = finite_scalar(
            sine * diagonal_bar - cosine * diagonal,
            "MINRES rotated diagonal",
        )?;
        epsilon = finite_scalar(sine * beta, "MINRES epsilon")?;
        diagonal_bar = finite_scalar(-cosine * beta, "MINRES diagonal bar")?;
        let rotation_norm = finite_scalar(diagonal_rotated.hypot(beta), "MINRES rotation norm")?;
        if rotation_norm <= f64::MIN_POSITIVE {
            return Err(solve_failed(
                "CUDA MINRES orthogonal rotation broke down before convergence",
            ));
        }
        cosine = finite_scalar(diagonal_rotated / rotation_norm, "MINRES rotation cosine")?;
        sine = finite_scalar(beta / rotation_norm, "MINRES rotation sine")?;
        let step_projection =
            finite_scalar(cosine * residual_projection, "MINRES step projection")?;
        residual_projection =
            finite_scalar(residual_projection * sine, "MINRES residual projection")?;

        std::mem::swap(
            &mut vectors.older_direction,
            &mut vectors.previous_direction,
        );
        std::mem::swap(&mut vectors.previous_direction, &mut vectors.direction);
        copy(
            stream,
            blas,
            dimension,
            &vectors.basis,
            &mut vectors.direction,
        )?;
        axpy(
            stream,
            blas,
            dimension,
            -previous_epsilon,
            &vectors.older_direction,
            &mut vectors.direction,
        )?;
        axpy(
            stream,
            blas,
            dimension,
            -delta,
            &vectors.previous_direction,
            &mut vectors.direction,
        )?;
        scale(
            stream,
            blas,
            dimension,
            finite_scalar(1.0 / rotation_norm, "MINRES direction scale")?,
            &mut vectors.direction,
        )?;
        axpy(
            stream,
            blas,
            dimension,
            step_projection,
            &vectors.direction,
            solution,
        )?;

        reported_residual_norm = residual_projection.abs();
        if reported_residual_norm <= target {
            recompute_residual(
                stream,
                sparse,
                spmv,
                blas,
                dimension,
                rhs,
                solution,
                &mut vectors.applied,
                sparse_workspace,
            )?;
            if norm(stream, blas, dimension, &vectors.applied)? <= target {
                return Ok(SolverOutcome {
                    reason: ConvergenceReason::ResidualToleranceSatisfied,
                    iterations: iteration,
                    reported_residual_norm,
                });
            }
        }
        if beta == 0.0 {
            recompute_residual(
                stream,
                sparse,
                spmv,
                blas,
                dimension,
                rhs,
                solution,
                &mut vectors.applied,
                sparse_workspace,
            )?;
            let true_residual_norm = norm(stream, blas, dimension, &vectors.applied)?;
            return Err(solve_failed(format!(
                "CUDA MINRES Lanczos space closed with true residual {true_residual_norm:e} above target {target:e}"
            )));
        }
    }

    recompute_residual(
        stream,
        sparse,
        spmv,
        blas,
        dimension,
        rhs,
        solution,
        &mut vectors.applied,
        sparse_workspace,
    )?;
    let true_residual_norm = norm(stream, blas, dimension, &vectors.applied)?;
    Err(solve_failed(format!(
        "CUDA MINRES reached {} iterations: reported residual {reported_residual_norm:e}, true residual {true_residual_norm:e}, target {target:e}",
        plan.maximum_iterations()
    )))
}

#[allow(clippy::too_many_arguments)]
fn solve_cg(
    stream: &CudaStream,
    sparse: &CusparseHandle,
    spmv: &SpmvPlan,
    blas: &BlasHandle,
    dimension: i32,
    plan: SolverPlan,
    rhs: &CudaSlice<f64>,
    solution: &mut CudaSlice<f64>,
    residual: &mut CudaSlice<f64>,
    inverse_diagonal: Option<&CudaSlice<f64>>,
    sparse_workspace: &mut CudaSlice<u8>,
    vectors: &mut CgWorkspace,
) -> Result<SolverOutcome, Diagnostic> {
    recompute_residual(
        stream,
        sparse,
        spmv,
        blas,
        dimension,
        rhs,
        solution,
        residual,
        sparse_workspace,
    )?;
    let right_hand_side_norm = norm(stream, blas, dimension, rhs)?;
    let target = plan.residual_target(right_hand_side_norm)?;
    let initial_residual_norm = norm(stream, blas, dimension, residual)?;
    if initial_residual_norm <= target {
        return Ok(SolverOutcome {
            reason: ConvergenceReason::InitialResidualSatisfied,
            iterations: 0,
            reported_residual_norm: initial_residual_norm,
        });
    }

    precondition(
        stream,
        blas,
        dimension,
        inverse_diagonal,
        residual,
        &mut vectors.preconditioned,
    )?;
    copy(
        stream,
        blas,
        dimension,
        &vectors.preconditioned,
        &mut vectors.direction,
    )?;
    let mut residual_product = dot(stream, blas, dimension, residual, &vectors.preconditioned)?;
    require_positive(residual_product, "CG preconditioned residual curvature")?;

    let mut reported_residual_norm = initial_residual_norm;
    for iteration in 1..=plan.maximum_iterations().get() {
        spmv_action(
            stream,
            sparse,
            spmv,
            &vectors.direction,
            &mut vectors.action,
            sparse_workspace,
            1.0,
            0.0,
        )?;
        let curvature = dot(stream, blas, dimension, &vectors.direction, &vectors.action)?;
        require_positive(curvature, "CG operator curvature")?;
        let step = finite_scalar(residual_product / curvature, "CG step")?;
        axpy(stream, blas, dimension, step, &vectors.direction, solution)?;
        axpy(stream, blas, dimension, -step, &vectors.action, residual)?;
        reported_residual_norm = norm(stream, blas, dimension, residual)?;
        if reported_residual_norm <= target {
            recompute_residual(
                stream,
                sparse,
                spmv,
                blas,
                dimension,
                rhs,
                solution,
                residual,
                sparse_workspace,
            )?;
            if norm(stream, blas, dimension, residual)? <= target {
                return Ok(SolverOutcome {
                    reason: ConvergenceReason::ResidualToleranceSatisfied,
                    iterations: iteration,
                    reported_residual_norm,
                });
            }
            precondition(
                stream,
                blas,
                dimension,
                inverse_diagonal,
                residual,
                &mut vectors.preconditioned,
            )?;
            residual_product = dot(stream, blas, dimension, residual, &vectors.preconditioned)?;
            require_positive(residual_product, "CG restarted residual curvature")?;
            copy(
                stream,
                blas,
                dimension,
                &vectors.preconditioned,
                &mut vectors.direction,
            )?;
            continue;
        }

        precondition(
            stream,
            blas,
            dimension,
            inverse_diagonal,
            residual,
            &mut vectors.preconditioned,
        )?;
        let next_product = dot(stream, blas, dimension, residual, &vectors.preconditioned)?;
        require_positive(next_product, "CG next residual curvature")?;
        let beta = finite_scalar(next_product / residual_product, "CG direction scale")?;
        scale(stream, blas, dimension, beta, &mut vectors.direction)?;
        axpy(
            stream,
            blas,
            dimension,
            1.0,
            &vectors.preconditioned,
            &mut vectors.direction,
        )?;
        residual_product = next_product;
    }
    Err(solve_failed(format!(
        "CUDA CG reached {} iterations with reported residual {reported_residual_norm:e} and target {target:e}",
        plan.maximum_iterations()
    )))
}

#[allow(clippy::too_many_arguments)]
fn solve_bicgstab(
    stream: &CudaStream,
    sparse: &CusparseHandle,
    spmv: &SpmvPlan,
    blas: &BlasHandle,
    dimension: i32,
    plan: SolverPlan,
    rhs: &CudaSlice<f64>,
    solution: &mut CudaSlice<f64>,
    residual: &mut CudaSlice<f64>,
    inverse_diagonal: Option<&CudaSlice<f64>>,
    sparse_workspace: &mut CudaSlice<u8>,
    vectors: &mut BicgstabWorkspace,
) -> Result<SolverOutcome, Diagnostic> {
    recompute_residual(
        stream,
        sparse,
        spmv,
        blas,
        dimension,
        rhs,
        solution,
        residual,
        sparse_workspace,
    )?;
    let right_hand_side_norm = norm(stream, blas, dimension, rhs)?;
    let target = plan.residual_target(right_hand_side_norm)?;
    let initial_residual_norm = norm(stream, blas, dimension, residual)?;
    if initial_residual_norm <= target {
        return Ok(SolverOutcome {
            reason: ConvergenceReason::InitialResidualSatisfied,
            iterations: 0,
            reported_residual_norm: initial_residual_norm,
        });
    }
    copy(stream, blas, dimension, residual, &mut vectors.shadow)?;

    let mut rho_previous = 1.0;
    let mut alpha = 1.0;
    let mut omega = 1.0;
    let mut fresh_start = true;
    let mut reported_residual_norm = initial_residual_norm;
    for iteration in 1..=plan.maximum_iterations().get() {
        let rho = dot(stream, blas, dimension, &vectors.shadow, residual)?;
        require_nonzero(rho, "BiCGSTAB shadow residual product")?;
        if fresh_start {
            copy(stream, blas, dimension, residual, &mut vectors.direction)?;
            fresh_start = false;
        } else {
            require_nonzero(omega, "BiCGSTAB omega")?;
            let beta = finite_scalar(
                (rho / rho_previous) * (alpha / omega),
                "BiCGSTAB direction scale",
            )?;
            axpy(
                stream,
                blas,
                dimension,
                -omega,
                &vectors.action,
                &mut vectors.direction,
            )?;
            scale(stream, blas, dimension, beta, &mut vectors.direction)?;
            axpy(
                stream,
                blas,
                dimension,
                1.0,
                residual,
                &mut vectors.direction,
            )?;
        }

        precondition(
            stream,
            blas,
            dimension,
            inverse_diagonal,
            &vectors.direction,
            &mut vectors.preconditioned_direction,
        )?;
        spmv_action(
            stream,
            sparse,
            spmv,
            &vectors.preconditioned_direction,
            &mut vectors.action,
            sparse_workspace,
            1.0,
            0.0,
        )?;
        let denominator = dot(stream, blas, dimension, &vectors.shadow, &vectors.action)?;
        require_nonzero(denominator, "BiCGSTAB alpha denominator")?;
        alpha = finite_scalar(rho / denominator, "BiCGSTAB alpha")?;
        copy(stream, blas, dimension, residual, &mut vectors.intermediate)?;
        axpy(
            stream,
            blas,
            dimension,
            -alpha,
            &vectors.action,
            &mut vectors.intermediate,
        )?;
        let intermediate_norm = norm(stream, blas, dimension, &vectors.intermediate)?;
        if intermediate_norm <= target {
            axpy(
                stream,
                blas,
                dimension,
                alpha,
                &vectors.preconditioned_direction,
                solution,
            )?;
            if accept_or_restart_bicgstab(
                stream,
                sparse,
                spmv,
                blas,
                dimension,
                rhs,
                solution,
                residual,
                sparse_workspace,
                target,
                &mut vectors.shadow,
                &mut vectors.direction,
                &mut vectors.action,
            )? {
                return Ok(SolverOutcome {
                    reason: ConvergenceReason::ResidualToleranceSatisfied,
                    iterations: iteration,
                    reported_residual_norm: intermediate_norm,
                });
            }
            fresh_start = true;
            rho_previous = 1.0;
            alpha = 1.0;
            omega = 1.0;
            continue;
        }

        precondition(
            stream,
            blas,
            dimension,
            inverse_diagonal,
            &vectors.intermediate,
            &mut vectors.preconditioned_intermediate,
        )?;
        spmv_action(
            stream,
            sparse,
            spmv,
            &vectors.preconditioned_intermediate,
            &mut vectors.intermediate_action,
            sparse_workspace,
            1.0,
            0.0,
        )?;
        let action_norm_squared = dot(
            stream,
            blas,
            dimension,
            &vectors.intermediate_action,
            &vectors.intermediate_action,
        )?;
        require_positive(action_norm_squared, "BiCGSTAB intermediate action norm")?;
        omega = finite_scalar(
            dot(
                stream,
                blas,
                dimension,
                &vectors.intermediate_action,
                &vectors.intermediate,
            )? / action_norm_squared,
            "BiCGSTAB omega",
        )?;
        require_nonzero(omega, "BiCGSTAB omega")?;
        axpy(
            stream,
            blas,
            dimension,
            alpha,
            &vectors.preconditioned_direction,
            solution,
        )?;
        axpy(
            stream,
            blas,
            dimension,
            omega,
            &vectors.preconditioned_intermediate,
            solution,
        )?;
        copy(stream, blas, dimension, &vectors.intermediate, residual)?;
        axpy(
            stream,
            blas,
            dimension,
            -omega,
            &vectors.intermediate_action,
            residual,
        )?;
        reported_residual_norm = norm(stream, blas, dimension, residual)?;
        if reported_residual_norm <= target {
            if accept_or_restart_bicgstab(
                stream,
                sparse,
                spmv,
                blas,
                dimension,
                rhs,
                solution,
                residual,
                sparse_workspace,
                target,
                &mut vectors.shadow,
                &mut vectors.direction,
                &mut vectors.action,
            )? {
                return Ok(SolverOutcome {
                    reason: ConvergenceReason::ResidualToleranceSatisfied,
                    iterations: iteration,
                    reported_residual_norm,
                });
            }
            fresh_start = true;
            rho_previous = 1.0;
            alpha = 1.0;
            omega = 1.0;
            continue;
        }
        rho_previous = rho;
    }
    Err(solve_failed(format!(
        "CUDA BiCGSTAB reached {} iterations with reported residual {reported_residual_norm:e} and target {target:e}",
        plan.maximum_iterations()
    )))
}

#[allow(clippy::too_many_arguments)]
fn accept_or_restart_bicgstab(
    stream: &CudaStream,
    sparse: &CusparseHandle,
    spmv: &SpmvPlan,
    blas: &BlasHandle,
    dimension: i32,
    rhs: &CudaSlice<f64>,
    solution: &CudaSlice<f64>,
    residual: &mut CudaSlice<f64>,
    sparse_workspace: &mut CudaSlice<u8>,
    target: f64,
    shadow: &mut CudaSlice<f64>,
    direction: &mut CudaSlice<f64>,
    action: &mut CudaSlice<f64>,
) -> Result<bool, Diagnostic> {
    recompute_residual(
        stream,
        sparse,
        spmv,
        blas,
        dimension,
        rhs,
        solution,
        residual,
        sparse_workspace,
    )?;
    if norm(stream, blas, dimension, residual)? <= target {
        return Ok(true);
    }
    copy(stream, blas, dimension, residual, shadow)?;
    scale(stream, blas, dimension, 0.0, direction)?;
    scale(stream, blas, dimension, 0.0, action)?;
    Ok(false)
}

#[allow(clippy::too_many_arguments)]
fn recompute_residual(
    stream: &CudaStream,
    sparse: &CusparseHandle,
    spmv: &SpmvPlan,
    blas: &BlasHandle,
    dimension: i32,
    rhs: &CudaSlice<f64>,
    solution: &CudaSlice<f64>,
    residual: &mut CudaSlice<f64>,
    sparse_workspace: &mut CudaSlice<u8>,
) -> Result<(), Diagnostic> {
    copy(stream, blas, dimension, rhs, residual)?;
    spmv_action(
        stream,
        sparse,
        spmv,
        solution,
        residual,
        sparse_workspace,
        -1.0,
        1.0,
    )
}

fn precondition(
    stream: &CudaStream,
    blas: &BlasHandle,
    dimension: i32,
    inverse_diagonal: Option<&CudaSlice<f64>>,
    input: &CudaSlice<f64>,
    output: &mut CudaSlice<f64>,
) -> Result<(), Diagnostic> {
    match inverse_diagonal {
        Some(diagonal) => diagonal_multiply(stream, blas, dimension, diagonal, input, output),
        None => copy(stream, blas, dimension, input, output),
    }
}

#[allow(clippy::too_many_arguments)]
fn spmv_action(
    stream: &CudaStream,
    handle: &CusparseHandle,
    plan: &SpmvPlan,
    input: &CudaSlice<f64>,
    output: &mut CudaSlice<f64>,
    workspace: &mut CudaSlice<u8>,
    alpha: f64,
    beta: f64,
) -> Result<(), Diagnostic> {
    let (input_pointer, input_guard) = input.device_ptr(stream);
    let (output_pointer, output_guard) = output.device_ptr_mut(stream);
    let (workspace_pointer, workspace_guard) = workspace.device_ptr_mut(stream);
    let result = plan
        .apply(
            handle,
            workspace_pointer,
            input_pointer,
            output_pointer,
            alpha,
            beta,
        )
        .map_err(cusparse_failed);
    drop(workspace_guard);
    drop(output_guard);
    drop(input_guard);
    result
}

fn copy(
    stream: &CudaStream,
    handle: &BlasHandle,
    dimension: i32,
    input: &CudaSlice<f64>,
    output: &mut CudaSlice<f64>,
) -> Result<(), Diagnostic> {
    let (input_pointer, input_guard) = input.device_ptr(stream);
    let (output_pointer, output_guard) = output.device_ptr_mut(stream);
    let result = handle
        .copy(dimension, input_pointer, output_pointer)
        .map_err(blas_failed);
    drop(output_guard);
    drop(input_guard);
    result
}

fn dot(
    stream: &CudaStream,
    handle: &BlasHandle,
    dimension: i32,
    left: &CudaSlice<f64>,
    right: &CudaSlice<f64>,
) -> Result<f64, Diagnostic> {
    let (left_pointer, left_guard) = left.device_ptr(stream);
    let (right_pointer, right_guard) = right.device_ptr(stream);
    let result = handle
        .dot(dimension, left_pointer, right_pointer)
        .map_err(blas_failed)
        .and_then(|value| finite_scalar(value, "cuBLAS dot product"));
    drop(right_guard);
    drop(left_guard);
    result
}

fn norm(
    stream: &CudaStream,
    handle: &BlasHandle,
    dimension: i32,
    values: &CudaSlice<f64>,
) -> Result<f64, Diagnostic> {
    let (pointer, guard) = values.device_ptr(stream);
    let result = handle
        .norm(dimension, pointer)
        .map_err(blas_failed)
        .and_then(|value| finite_scalar(value, "cuBLAS Euclidean norm"));
    drop(guard);
    result
}

fn axpy(
    stream: &CudaStream,
    handle: &BlasHandle,
    dimension: i32,
    alpha: f64,
    input: &CudaSlice<f64>,
    output: &mut CudaSlice<f64>,
) -> Result<(), Diagnostic> {
    finite_scalar(alpha, "cuBLAS vector-update scale")?;
    let (input_pointer, input_guard) = input.device_ptr(stream);
    let (output_pointer, output_guard) = output.device_ptr_mut(stream);
    let result = handle
        .axpy(dimension, alpha, input_pointer, output_pointer)
        .map_err(blas_failed);
    drop(output_guard);
    drop(input_guard);
    result
}

fn scale(
    stream: &CudaStream,
    handle: &BlasHandle,
    dimension: i32,
    alpha: f64,
    values: &mut CudaSlice<f64>,
) -> Result<(), Diagnostic> {
    finite_scalar(alpha, "cuBLAS vector scale")?;
    let (pointer, guard) = values.device_ptr_mut(stream);
    let result = handle.scale(dimension, alpha, pointer).map_err(blas_failed);
    drop(guard);
    result
}

fn diagonal_multiply(
    stream: &CudaStream,
    handle: &BlasHandle,
    dimension: i32,
    diagonal: &CudaSlice<f64>,
    input: &CudaSlice<f64>,
    output: &mut CudaSlice<f64>,
) -> Result<(), Diagnostic> {
    let (diagonal_pointer, diagonal_guard) = diagonal.device_ptr(stream);
    let (input_pointer, input_guard) = input.device_ptr(stream);
    let (output_pointer, output_guard) = output.device_ptr_mut(stream);
    let result = handle
        .diagonal_multiply(dimension, diagonal_pointer, input_pointer, output_pointer)
        .map_err(blas_failed);
    drop(output_guard);
    drop(input_guard);
    drop(diagonal_guard);
    result
}

fn validate_canonical_system(
    system: &CanonicalCsrSystemView,
    initial_guess: Option<&[f64]>,
) -> Result<(), Diagnostic> {
    if system.values().is_empty() {
        return Err(unsupported(
            "CUDA CSR solve requires at least one structural nonzero",
        ));
    }
    if let Some(initial_guess) = initial_guess
        && (initial_guess.len() != system.columns()
            || initial_guess.iter().any(|value| !value.is_finite()))
    {
        return Err(solve_failed(format!(
            "CUDA initial guess requires {} finite values, received {}",
            system.columns(),
            initial_guess.len()
        )));
    }
    Ok(())
}

fn finite_scalar(value: f64, label: &str) -> Result<f64, Diagnostic> {
    value
        .is_finite()
        .then_some(value)
        .ok_or_else(|| solve_failed(format!("{label} is non-finite")))
}

fn require_positive(value: f64, label: &str) -> Result<(), Diagnostic> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(solve_failed(format!(
            "{label} must be positive, got {value:e}"
        )))
    }
}

fn require_nonzero(value: f64, label: &str) -> Result<(), Diagnostic> {
    if value.is_finite() && value != 0.0 {
        Ok(())
    } else {
        Err(solve_failed(format!("{label} broke down at {value:e}")))
    }
}

fn blas_failed(error: BlasError) -> Diagnostic {
    solve_failed(format!("cuBLAS operation failed: {error:?}"))
}

fn cusparse_failed(error: crate::ffi::FfiError) -> Diagnostic {
    solve_failed(format!("cuSPARSE operation failed: {error:?}"))
}

struct LinearBufferDescriptors {
    rows: DeviceBufferDescriptor<i64>,
    columns: DeviceBufferDescriptor<i64>,
    values: DeviceBufferDescriptor<f64>,
    right_hand_side: DeviceBufferDescriptor<f64>,
    inverse_diagonal: Option<DeviceBufferDescriptor<f64>>,
    solution: DeviceBufferDescriptor<f64>,
}

impl LinearBufferDescriptors {
    #[allow(clippy::too_many_arguments)]
    fn new(
        device: eqiora_device::DeviceId,
        row_offsets: usize,
        column_indices: usize,
        values: usize,
        right_hand_side: usize,
        initial_guess: usize,
        inverse_diagonal: Option<usize>,
    ) -> Result<Self, Diagnostic> {
        Ok(Self {
            rows: descriptor(device, row_offsets)?,
            columns: descriptor(device, column_indices)?,
            values: descriptor(device, values)?,
            right_hand_side: descriptor(device, right_hand_side)?,
            solution: descriptor(device, initial_guess)?,
            inverse_diagonal: inverse_diagonal
                .map(|elements| descriptor(device, elements))
                .transpose()?,
        })
    }
}
