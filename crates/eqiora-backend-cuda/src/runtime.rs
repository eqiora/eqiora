use std::borrow::Cow;
use std::num::{NonZeroU64, NonZeroUsize};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaEvent, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use eqiora_assembly::CsrMatrix;
use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_device::{
    BufferId, Completion, DeviceBufferDescriptor, DeviceCapability, DeviceDescriptor, DeviceId,
    DeviceRuntime, Fence, HostBufferDescriptor, MemoryRegion, QueueId, QueueSlot, QueueTimeline,
    RuntimeId, SparseActionPolicy, SparseActionTolerance, TransferEvidence, TransferPlan,
    WaitedCompletion,
};
use eqiora_execution::DeviceValueGeneration;

use crate::blas;
use crate::ffi::{self, CusparseHandle, SpmvPlan};
use crate::{CUDA_BINDING_TOOLKIT, CUDA_RUNTIME_ID, CUDARC_VERSION};

/// CUDA driver, bindings, and sparse-library versions used by one action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaLibraryVersions {
    pub(crate) driver: i32,
    pub(crate) cusparse: i32,
    pub(crate) cublas: Option<i32>,
    pub(crate) cudarc: &'static str,
    pub(crate) binding_toolkit: &'static str,
}

/// CUDA compute capability reported by the selected physical device.
///
/// This Eqiora-owned value crosses the adapter boundary; CUDA driver structs
/// and integer attribute identifiers remain private to the adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CudaComputeCapability {
    major: u16,
    minor: u16,
}

/// Stable 16-byte identity reported by the CUDA driver for one physical device.
///
/// The value is deployment evidence. It is deliberately separate from the
/// runtime-local [`DeviceId`], whose ordinal may be remapped independently in
/// each process by `CUDA_VISIBLE_DEVICES`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CudaDeviceUuid([u8; 16]);

impl CudaDeviceUuid {
    fn from_driver(bytes: [u8; 16]) -> Result<Self, Diagnostic> {
        if bytes == [0; 16] {
            return Err(unsupported(
                "CUDA reported an all-zero physical device UUID",
            ));
        }
        Ok(Self(bytes))
    }

    /// Raw fixed-width bytes suitable for bounded topology agreement.
    #[must_use]
    pub const fn as_bytes(self) -> [u8; 16] {
        self.0
    }
}

impl std::fmt::Debug for CudaDeviceUuid {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("CudaDeviceUuid(")?;
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        formatter.write_str(")")
    }
}

/// Live CUDA discovery record retaining physical identity beside L2 capacity.
///
/// `DeviceDescriptor` remains the backend-neutral capability snapshot. This
/// adapter-owned record adds only CUDA observations needed to prove that two
/// process-local ordinal-zero devices are physically distinct.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaDeviceObservation {
    pub(crate) descriptor: DeviceDescriptor,
    pub(crate) physical_uuid: CudaDeviceUuid,
    pub(crate) compute_capability: CudaComputeCapability,
}

impl CudaDeviceObservation {
    /// Backend-neutral descriptor selected by deployment binding.
    #[must_use]
    pub const fn descriptor(&self) -> &DeviceDescriptor {
        &self.descriptor
    }

    /// Physical device UUID reported by the live CUDA driver.
    #[must_use]
    pub const fn physical_uuid(&self) -> CudaDeviceUuid {
        self.physical_uuid
    }

    /// Compute capability reported by the same live discovery operation.
    #[must_use]
    pub const fn compute_capability(&self) -> CudaComputeCapability {
        self.compute_capability
    }
}

impl CudaComputeCapability {
    fn from_driver(major: i32, minor: i32) -> Result<Self, Diagnostic> {
        let major = u16::try_from(major)
            .map_err(|_| unsupported("CUDA reported an invalid compute-capability major value"))?;
        let minor = u16::try_from(minor)
            .map_err(|_| unsupported("CUDA reported an invalid compute-capability minor value"))?;
        if major == 0 {
            return Err(unsupported(
                "CUDA reported a zero compute-capability major value",
            ));
        }
        Ok(Self { major, minor })
    }

    /// Major component reported by the CUDA driver.
    #[must_use]
    pub const fn major(self) -> u16 {
        self.major
    }

    /// Minor component reported by the CUDA driver.
    #[must_use]
    pub const fn minor(self) -> u16 {
        self.minor
    }
}

impl CudaLibraryVersions {
    /// CUDA driver integer version (for example, `12000`).
    #[must_use]
    pub const fn driver(self) -> i32 {
        self.driver
    }

    /// cuSPARSE integer version reported by its live handle.
    #[must_use]
    pub const fn cusparse(self) -> i32 {
        self.cusparse
    }

    /// cuBLAS integer version when dense vector actions participated.
    #[must_use]
    pub const fn cublas(self) -> Option<i32> {
        self.cublas
    }

    /// Exact Rust adapter dependency version.
    #[must_use]
    pub const fn cudarc(self) -> &'static str {
        self.cudarc
    }

    /// CUDA toolkit ABI selected for generated bindings.
    #[must_use]
    pub const fn binding_toolkit(self) -> &'static str {
        self.binding_toolkit
    }
}

/// Explicit matrix, input, and output transfer evidence for one CSR action.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CudaCsrTransferEvidence {
    row_offsets: TransferEvidence<i64>,
    column_indices: TransferEvidence<i64>,
    values: TransferEvidence<f64>,
    input: TransferEvidence<f64>,
    output: TransferEvidence<f64>,
}

impl CudaCsrTransferEvidence {
    /// Matrix row-offset transfer.
    #[must_use]
    pub const fn row_offsets(self) -> TransferEvidence<i64> {
        self.row_offsets
    }

    /// Matrix column-index transfer.
    #[must_use]
    pub const fn column_indices(self) -> TransferEvidence<i64> {
        self.column_indices
    }

    /// Matrix nonzero-value transfer.
    #[must_use]
    pub const fn values(self) -> TransferEvidence<f64> {
        self.values
    }

    /// Dense input-vector transfer.
    #[must_use]
    pub const fn input(self) -> TransferEvidence<f64> {
        self.input
    }

    /// Dense output-vector transfer.
    #[must_use]
    pub const fn output(self) -> TransferEvidence<f64> {
        self.output
    }

    /// Total host-to-device bytes, including the finalized CSR and input.
    #[must_use]
    pub fn host_to_device_bytes(self) -> usize {
        self.row_offsets.plan().bytes().expect("validated transfer")
            + self
                .column_indices
                .plan()
                .bytes()
                .expect("validated transfer")
            + self.values.plan().bytes().expect("validated transfer")
            + self.input.plan().bytes().expect("validated transfer")
    }

    /// Dense output bytes copied back to the host.
    #[must_use]
    pub fn device_to_host_bytes(self) -> usize {
        self.output.plan().bytes().expect("validated transfer")
    }
}

/// Auditable evidence for one CUDA CSR action accepted against a host
/// reference vector.
#[derive(Debug, Clone, PartialEq)]
pub struct CudaCsrActionEvidence {
    device: DeviceDescriptor,
    physical_uuid: CudaDeviceUuid,
    compute_capability: CudaComputeCapability,
    versions: CudaLibraryVersions,
    policy: SparseActionPolicy,
    transfers: CudaCsrTransferEvidence,
    action_completion: Completion,
    workspace_bytes: usize,
    maximum_absolute_error: f64,
    maximum_scaled_error: f64,
    timings: eqiora_device::DeviceExecutionTimings,
}

impl CudaCsrActionEvidence {
    /// Exact selected device and admitted features.
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

    /// Runtime/library versions queried from live handles.
    #[must_use]
    pub const fn versions(&self) -> CudaLibraryVersions {
        self.versions
    }

    /// Sparse-action ordering policy.
    #[must_use]
    pub const fn policy(&self) -> SparseActionPolicy {
        self.policy
    }

    /// Exact transfer plans and completion identities.
    #[must_use]
    pub const fn transfers(&self) -> CudaCsrTransferEvidence {
        self.transfers
    }

    /// Completion of the cuSPARSE action itself.
    #[must_use]
    pub const fn action_completion(&self) -> Completion {
        self.action_completion
    }

    /// Retained external cuSPARSE workspace size.
    #[must_use]
    pub const fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }

    /// Largest absolute difference from the independent host action.
    #[must_use]
    pub const fn maximum_absolute_error(&self) -> f64 {
        self.maximum_absolute_error
    }

    /// Largest `error / (absolute + relative * |reference|)` ratio.
    #[must_use]
    pub const fn maximum_scaled_error(&self) -> f64 {
        self.maximum_scaled_error
    }

    /// Separately observed execution phases.
    #[must_use]
    pub const fn timings(&self) -> eqiora_device::DeviceExecutionTimings {
        self.timings
    }
}

/// CUDA values plus complete action evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct CudaCsrActionResult {
    values: Vec<f64>,
    evidence: CudaCsrActionEvidence,
}

impl CudaCsrActionResult {
    /// Device-computed values accepted against the call's host reference.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Action, transfer, version, comparison, and timing evidence.
    #[must_use]
    pub const fn evidence(&self) -> &CudaCsrActionEvidence {
        &self.evidence
    }
}

/// Immutable setup evidence for one run-owned resident rectangular CSR action.
///
/// Matrix transfers occur only while constructing the owning session. Every
/// subsequent action reuses these exact device allocations and cuSPARSE
/// descriptors.
#[derive(Debug, Clone, PartialEq)]
pub struct CudaResidentCsrSetupEvidence {
    device: DeviceDescriptor,
    physical_uuid: CudaDeviceUuid,
    compute_capability: CudaComputeCapability,
    versions: CudaLibraryVersions,
    policy: SparseActionPolicy,
    queue: QueueId,
    rows: usize,
    columns: usize,
    nonzeros: usize,
    row_offsets: TransferEvidence<i64>,
    column_indices: TransferEvidence<i64>,
    values: TransferEvidence<f64>,
    matrix_ready: WaitedCompletion,
    known_payload_bytes: usize,
    workspace_bytes: usize,
}

impl CudaResidentCsrSetupEvidence {
    /// Exact selected runtime-local device descriptor.
    #[must_use]
    pub const fn device(&self) -> &DeviceDescriptor {
        &self.device
    }

    /// Live physical UUID observed for the selected device.
    #[must_use]
    pub const fn physical_uuid(&self) -> CudaDeviceUuid {
        self.physical_uuid
    }

    /// Live compute capability observed for the selected device.
    #[must_use]
    pub const fn compute_capability(&self) -> CudaComputeCapability {
        self.compute_capability
    }

    /// Driver, cuSPARSE, binding, and adapter versions used by the session.
    #[must_use]
    pub const fn versions(&self) -> CudaLibraryVersions {
        self.versions
    }

    /// Exact sparse-action policy fixed for the session.
    #[must_use]
    pub const fn policy(&self) -> SparseActionPolicy {
        self.policy
    }

    /// Materialized process-local queue used by every transfer and action.
    #[must_use]
    pub const fn queue(&self) -> QueueId {
        self.queue
    }

    /// Rectangular row count.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Rectangular column count.
    #[must_use]
    pub const fn columns(&self) -> usize {
        self.columns
    }

    /// Exact nonzero count.
    #[must_use]
    pub const fn nonzeros(&self) -> usize {
        self.nonzeros
    }

    /// Sole row-offset upload for the resident matrix.
    #[must_use]
    pub const fn row_offsets(&self) -> TransferEvidence<i64> {
        self.row_offsets
    }

    /// Sole column-index upload for the resident matrix.
    #[must_use]
    pub const fn column_indices(&self) -> TransferEvidence<i64> {
        self.column_indices
    }

    /// Sole coefficient upload for the resident matrix.
    #[must_use]
    pub const fn values(&self) -> TransferEvidence<f64> {
        self.values
    }

    /// Successful wait that made all three matrix transfers visible.
    #[must_use]
    pub const fn matrix_ready(&self) -> WaitedCompletion {
        self.matrix_ready
    }

    /// Known resident CSR plus input/output payload, excluding vendor workspace.
    #[must_use]
    pub const fn known_payload_bytes(&self) -> usize {
        self.known_payload_bytes
    }

    /// Retained external cuSPARSE workspace size.
    #[must_use]
    pub const fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }
}

/// One successful repeated action over a resident CUDA CSR matrix.
///
/// The evidence is constant-size. Matrix transfers live only in
/// [`CudaResidentCsrSetupEvidence`], so an action cannot pretend to have
/// re-uploaded or replaced the resident operator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CudaResidentCsrActionEvidence {
    ordinal: NonZeroU64,
    input: TransferEvidence<f64>,
    input_ready: WaitedCompletion,
    action_completion: Completion,
    action_visible: WaitedCompletion,
    output: TransferEvidence<f64>,
    output_visible: WaitedCompletion,
    input_generation: DeviceValueGeneration,
    output_generation: DeviceValueGeneration,
}

impl CudaResidentCsrActionEvidence {
    /// Dense one-based action ordinal within this session.
    #[must_use]
    pub const fn ordinal(self) -> NonZeroU64 {
        self.ordinal
    }

    /// Host-to-device input transfer for this action.
    #[must_use]
    pub const fn input(self) -> TransferEvidence<f64> {
        self.input
    }

    /// Successful wait after this action's input upload.
    #[must_use]
    pub const fn input_ready(self) -> WaitedCompletion {
        self.input_ready
    }

    /// Submission identity assigned to the cuSPARSE action.
    #[must_use]
    pub const fn action_completion(self) -> Completion {
        self.action_completion
    }

    /// Successful wait after the cuSPARSE action.
    #[must_use]
    pub const fn action_visible(self) -> WaitedCompletion {
        self.action_visible
    }

    /// Device-to-host owned-row output transfer for this action.
    #[must_use]
    pub const fn output(self) -> TransferEvidence<f64> {
        self.output
    }

    /// Successful wait after the owned-row output became host-visible.
    #[must_use]
    pub const fn output_visible(self) -> WaitedCompletion {
        self.output_visible
    }

    /// Logical generation written into the reused input allocation.
    #[must_use]
    pub const fn input_generation(self) -> DeviceValueGeneration {
        self.input_generation
    }

    /// Logical generation written into and downloaded from the output allocation.
    #[must_use]
    pub const fn output_generation(self) -> DeviceValueGeneration {
        self.output_generation
    }
}

/// Run-owned resident rectangular CSR action on one selected CUDA device.
///
/// The matrix and two dense buffers have fixed shape and allocation identity
/// for the session lifetime. `apply` accepts distinct safe Rust input/output
/// borrows, uploads only the dense input, performs one deterministic SpMV, and
/// downloads only the dense output.
#[derive(Debug)]
pub struct CudaResidentCsrActionSession {
    // Drop cuSPARSE descriptors before the allocations whose pointers they retain.
    plan: SpmvPlan,
    handle: CusparseHandle,
    workspace: CudaSlice<u8>,
    device_rows: CudaSlice<i64>,
    device_columns: CudaSlice<i64>,
    device_values: CudaSlice<f64>,
    device_input: CudaSlice<f64>,
    device_output: CudaSlice<f64>,
    stream: Arc<CudaStream>,
    timeline: QueueTimeline,
    descriptors: BufferDescriptors,
    setup: CudaResidentCsrSetupEvidence,
    action_count: u64,
    poisoned: bool,
}

impl Drop for CudaResidentCsrActionSession {
    fn drop(&mut self) {
        // `apply` can return after an asynchronous CUDA error before an event
        // wait produced evidence. Quiesce the owned stream while every
        // descriptor and allocation is still live; fields are dropped only
        // after this method returns. CUDA reports prior asynchronous failures
        // through this synchronization, which cannot be surfaced from Drop.
        let _ = self.stream.synchronize();
    }
}

impl CudaResidentCsrActionSession {
    /// Materialize one validated rectangular CSR exactly once on CUDA.
    ///
    /// The supplied observation must still exactly match live discovery. The
    /// queue must belong to that device and this first resident slice accepts
    /// only [`SparseActionPolicy::Deterministic`]. All validation precedes
    /// context creation and numerical allocation where possible.
    ///
    /// # Errors
    /// Returns a stable diagnostic for policy, shape, binding, capability,
    /// live-discovery, allocation, transfer, descriptor, workspace, or fence
    /// failure. It never falls back to a host action.
    pub fn new(
        matrix: &CsrMatrix,
        selected: &CudaDeviceObservation,
        queue: QueueSlot,
        policy: SparseActionPolicy,
    ) -> Result<Self, Diagnostic> {
        validate_resident_request(matrix, selected, queue, policy)?;
        let row_offsets = convert_indices(matrix.row_offsets(), "row offset")?;
        let column_indices = convert_indices(matrix.column_indices(), "column index")?;
        let known_payload_bytes = resident_payload_bytes(matrix)?;
        let available =
            usize::try_from(selected.descriptor.total_memory_bytes().get()).map_err(|_| {
                unsupported("CUDA device memory size exceeds the host addressable range")
            })?;
        if known_payload_bytes > available {
            return Err(unsupported(format!(
                "resident CUDA CSR action requires {known_payload_bytes} known bytes, selected device reports {available} total bytes"
            )));
        }

        let live = discover_cuda_devices(cuda_device_count()?)?
            .into_iter()
            .find(|candidate| candidate.descriptor.id() == selected.descriptor.id())
            .ok_or_else(|| {
                unsupported(format!(
                    "selected CUDA device {} is no longer visible",
                    selected.descriptor.id().ordinal()
                ))
            })?;
        if &live != selected {
            return Err(unsupported(
                "CUDA device descriptor, UUID, or compute capability changed after discovery",
            ));
        }

        let context = CudaContext::new(usize::from(selected.descriptor.id().ordinal()))
            .map_err(driver_failed)?;
        let stream = context.new_stream().map_err(driver_failed)?;
        let queue = materialize_queue(queue)?;
        let mut timeline = QueueTimeline::new(queue);
        let handle = CusparseHandle::new(stream.cu_stream()).map_err(cusparse_failed)?;

        let mut device_rows = stream
            .alloc_zeros::<i64>(row_offsets.len())
            .map_err(driver_failed)?;
        let mut device_columns = stream
            .alloc_zeros::<i64>(column_indices.len())
            .map_err(driver_failed)?;
        let mut device_values = stream
            .alloc_zeros::<f64>(matrix.values().len())
            .map_err(driver_failed)?;
        let device_input = stream
            .alloc_zeros::<f64>(matrix.columns())
            .map_err(driver_failed)?;
        let mut device_output = stream
            .alloc_zeros::<f64>(matrix.rows())
            .map_err(driver_failed)?;
        let descriptors = BufferDescriptors::new(
            selected.descriptor.id(),
            row_offsets.len(),
            column_indices.len(),
            matrix.values().len(),
            matrix.columns(),
            matrix.rows(),
        )?;
        validate_resident_descriptors(&descriptors)?;
        stream.synchronize().map_err(driver_failed)?;

        stream
            .memcpy_htod(&row_offsets, &mut device_rows)
            .map_err(driver_failed)?;
        let row_completion = Completion::new(timeline.next_submission()?);
        stream
            .memcpy_htod(&column_indices, &mut device_columns)
            .map_err(driver_failed)?;
        let column_completion = Completion::new(timeline.next_submission()?);
        stream
            .memcpy_htod(matrix.values(), &mut device_values)
            .map_err(driver_failed)?;
        let values_completion = Completion::new(timeline.next_submission()?);
        let matrix_ready = CudaEventFence::record(&stream, &mut timeline)?.waited()?;

        let rows = i64::try_from(matrix.rows())
            .map_err(|_| unsupported("CSR row count exceeds CUDA's signed 64-bit descriptor"))?;
        let columns = i64::try_from(matrix.columns())
            .map_err(|_| unsupported("CSR column count exceeds CUDA's signed 64-bit descriptor"))?;
        let nonzeros = i64::try_from(matrix.values().len()).map_err(|_| {
            unsupported("CSR nonzero count exceeds CUDA's signed 64-bit descriptor")
        })?;
        let (row_pointer, row_guard) = device_rows.device_ptr(&stream);
        let (column_pointer, column_guard) = device_columns.device_ptr(&stream);
        let (value_pointer, value_guard) = device_values.device_ptr(&stream);
        let (input_pointer, input_guard) = device_input.device_ptr(&stream);
        let (output_pointer, output_guard) = device_output.device_ptr_mut(&stream);
        let plan = SpmvPlan::new(
            &handle,
            rows,
            columns,
            nonzeros,
            row_pointer,
            column_pointer,
            value_pointer,
            input_pointer,
            output_pointer,
            true,
        )
        .map_err(cusparse_failed)?;
        drop(output_guard);
        drop(input_guard);
        drop(value_guard);
        drop(column_guard);
        drop(row_guard);
        let allocated_workspace_bytes = plan.workspace_bytes().max(1);
        let workspace = stream
            .alloc_zeros::<u8>(allocated_workspace_bytes)
            .map_err(driver_failed)?;
        stream.synchronize().map_err(driver_failed)?;
        let observed_payload_bytes = known_payload_bytes
            .checked_add(allocated_workspace_bytes)
            .ok_or_else(|| unsupported("resident CUDA CSR payload size overflowed"))?;
        if observed_payload_bytes > available {
            return Err(unsupported(format!(
                "resident CUDA CSR action plus workspace requires {observed_payload_bytes} bytes, selected device reports {available} total bytes"
            )));
        }

        let setup = CudaResidentCsrSetupEvidence {
            device: selected.descriptor.clone(),
            physical_uuid: selected.physical_uuid,
            compute_capability: selected.compute_capability,
            versions: CudaLibraryVersions {
                driver: ffi::driver_version().map_err(driver_failed)?,
                cusparse: handle.version().map_err(cusparse_failed)?,
                cublas: None,
                cudarc: CUDARC_VERSION,
                binding_toolkit: CUDA_BINDING_TOOLKIT,
            },
            policy,
            queue,
            rows: matrix.rows(),
            columns: matrix.columns(),
            nonzeros: matrix.values().len(),
            row_offsets: transfer_to_device(descriptors.rows, row_completion)?,
            column_indices: transfer_to_device(descriptors.columns, column_completion)?,
            values: transfer_to_device(descriptors.values, values_completion)?,
            matrix_ready,
            known_payload_bytes,
            workspace_bytes: plan.workspace_bytes(),
        };
        validate_resident_setup_evidence(&setup)?;
        Ok(Self {
            plan,
            handle,
            workspace,
            device_rows,
            device_columns,
            device_values,
            device_input,
            device_output,
            stream,
            timeline,
            descriptors,
            setup,
            action_count: 0,
            poisoned: false,
        })
    }

    /// Immutable matrix, binding, transfer, UUID, and library evidence.
    #[must_use]
    pub const fn setup_evidence(&self) -> &CudaResidentCsrSetupEvidence {
        &self.setup
    }

    /// Number of successfully completed actions.
    #[must_use]
    pub const fn action_count(&self) -> u64 {
        self.action_count
    }

    /// Apply the resident rectangular matrix into distinct caller-owned output.
    ///
    /// Matrix storage is never transferred here. A shape or finite-input
    /// failure happens before device submission and leaves the session usable.
    /// Any CUDA submission or synchronization failure poisons the session so a
    /// caller cannot continue from an unprovable queue state.
    ///
    /// # Errors
    /// Returns a stable diagnostic for input/output shape, non-finite input or
    /// output, exhausted generation identity, CUDA action, transfer, or fence
    /// failure.
    pub fn apply(
        &mut self,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<CudaResidentCsrActionEvidence, Diagnostic> {
        validate_resident_action(self.setup.columns, self.setup.rows, input, output)?;
        if self.poisoned {
            return Err(solve_failed(
                "resident CUDA CSR action session is poisoned by an earlier runtime failure",
            ));
        }
        let ordinal = self
            .action_count
            .checked_add(1)
            .and_then(NonZeroU64::new)
            .ok_or_else(|| unsupported("resident CUDA CSR action generation is exhausted"))?;
        match self.apply_inner(input, output, ordinal) {
            Ok(evidence) => {
                self.action_count = ordinal.get();
                Ok(evidence)
            }
            Err(error) => {
                self.poisoned = true;
                Err(error)
            }
        }
    }

    fn apply_inner(
        &mut self,
        input: &[f64],
        output: &mut [f64],
        ordinal: NonZeroU64,
    ) -> Result<CudaResidentCsrActionEvidence, Diagnostic> {
        self.stream
            .memcpy_htod(input, &mut self.device_input)
            .map_err(driver_failed)?;
        let input_completion = Completion::new(self.timeline.next_submission()?);
        let input_ready = CudaEventFence::record(&self.stream, &mut self.timeline)?.waited()?;

        let (_, row_guard) = self.device_rows.device_ptr(&self.stream);
        let (_, column_guard) = self.device_columns.device_ptr(&self.stream);
        let (_, value_guard) = self.device_values.device_ptr(&self.stream);
        let (input_pointer, input_guard) = self.device_input.device_ptr(&self.stream);
        let (output_pointer, output_guard) = self.device_output.device_ptr_mut(&self.stream);
        let (workspace_pointer, workspace_guard) = self.workspace.device_ptr_mut(&self.stream);
        self.plan
            .apply(
                &self.handle,
                workspace_pointer,
                input_pointer,
                output_pointer,
                1.0,
                0.0,
            )
            .map_err(cusparse_failed)?;
        let action_completion = Completion::new(self.timeline.next_submission()?);
        drop(workspace_guard);
        drop(output_guard);
        drop(input_guard);
        drop(value_guard);
        drop(column_guard);
        drop(row_guard);
        let action_visible = CudaEventFence::record(&self.stream, &mut self.timeline)?.waited()?;

        self.stream
            .memcpy_dtoh(&self.device_output, output)
            .map_err(driver_failed)?;
        let output_completion = Completion::new(self.timeline.next_submission()?);
        let output_visible = CudaEventFence::record(&self.stream, &mut self.timeline)?.waited()?;
        if output.iter().any(|value| !value.is_finite()) {
            return Err(solve_failed(
                "resident CUDA CSR action produced a non-finite owned-row value",
            ));
        }

        let input_generation = DeviceValueGeneration::new(self.descriptors.input.id(), ordinal);
        let output_generation = DeviceValueGeneration::new(self.descriptors.output.id(), ordinal);
        let evidence = CudaResidentCsrActionEvidence {
            ordinal,
            input: transfer_to_device(self.descriptors.input, input_completion)?,
            input_ready,
            action_completion,
            action_visible,
            output: transfer_to_host(self.descriptors.output, output_completion)?,
            output_visible,
            input_generation,
            output_generation,
        };
        validate_resident_action_evidence(&self.setup, evidence)?;
        Ok(evidence)
    }
}

/// Dynamically loaded cudarc runtime adapter.
#[derive(Debug, Clone, Copy, Default)]
pub struct CudaRuntime;

impl CudaRuntime {
    /// Snapshot every visible CUDA device without exposing the trait import
    /// through language facades.
    ///
    /// # Errors
    /// Returns a stable diagnostic when the driver or required sparse library
    /// cannot be loaded, queried, or represented by Eqiora-owned evidence.
    pub fn discover(self) -> Result<Vec<DeviceDescriptor>, Diagnostic> {
        <Self as DeviceRuntime>::discover(&self)
    }

    /// Observe every visible CUDA device with its physical UUID.
    ///
    /// Unlike a runtime-local ordinal, the UUID remains suitable for proving
    /// physical-device distinctness across separately masked processes. It is
    /// execution provenance and never Semantic Model identity.
    ///
    /// # Errors
    /// Returns a stable diagnostic when the CUDA driver or required sparse
    /// library cannot be loaded, or a live record cannot be represented.
    pub fn observe(self) -> Result<Vec<CudaDeviceObservation>, Diagnostic> {
        discover_cuda_devices(cuda_device_count()?)
    }
}

impl DeviceRuntime for CudaRuntime {
    fn id(&self) -> RuntimeId {
        CUDA_RUNTIME_ID
    }

    fn discover(&self) -> Result<Vec<DeviceDescriptor>, Diagnostic> {
        Ok(discover_cuda_devices(cuda_device_count()?)?
            .into_iter()
            .map(|device| device.descriptor)
            .collect())
    }
}

pub(crate) fn discover_cuda_devices(count: i32) -> Result<Vec<CudaDeviceObservation>, Diagnostic> {
    let count = u16::try_from(count)
        .map_err(|_| unsupported("CUDA reported an invalid or unsupported device count"))?;
    let mut devices = Vec::with_capacity(usize::from(count));
    ffi::probe_cusparse().map_err(cusparse_failed)?;
    let has_dense_vector_level_1 = blas::probe_cublas().is_ok();
    for ordinal in 0..count {
        let properties = ffi::device_properties(ordinal).map_err(driver_failed)?;
        let compute_capability = CudaComputeCapability::from_driver(
            properties.compute_capability.0,
            properties.compute_capability.1,
        )?;
        let physical_uuid = CudaDeviceUuid::from_driver(properties.uuid)?;
        let memory =
            NonZeroU64::new(u64::try_from(properties.total_memory_bytes).map_err(|_| {
                unsupported("CUDA device memory size does not fit the Eqiora evidence type")
            })?)
            .ok_or_else(|| unsupported("CUDA reported a device with zero visible memory"))?;
        let mut capabilities = vec![
            DeviceCapability::Float32,
            DeviceCapability::CsrMatrixVectorProduct,
            DeviceCapability::AsynchronousQueue,
        ];
        if has_dense_vector_level_1 {
            capabilities.push(DeviceCapability::DenseVectorLevel1);
        }
        if compute_capability.major() > 1
            || (compute_capability.major(), compute_capability.minor()) == (1, 3)
        {
            capabilities.push(DeviceCapability::Float64);
        }
        devices.push(CudaDeviceObservation {
            descriptor: DeviceDescriptor::new(
                DeviceId::new(CUDA_RUNTIME_ID, ordinal),
                properties.name,
                memory,
                capabilities,
            )?,
            physical_uuid,
            compute_capability,
        });
    }
    Ok(devices)
}

/// Execute one finalized `f64` CSR action on CUDA and compare it with the
/// independent host operator under an explicit tolerance.
///
/// # Errors
/// Returns a stable diagnostic for unavailable runtime/library/device,
/// unsupported shape/index conversion, transfer or cuSPARSE failure,
/// non-finite data, or disagreement with the host oracle.
pub fn verify_csr_action(
    matrix: &CsrMatrix,
    input: &[f64],
    device_ordinal: u16,
    policy: SparseActionPolicy,
    tolerance: SparseActionTolerance,
) -> Result<CudaCsrActionResult, Diagnostic> {
    verify_csr_action_inner(matrix, input, None, device_ordinal, policy, tolerance)
}

/// Execute one finalized `f64` CSR action on CUDA and compare it with an
/// explicitly supplied independent reference vector.
///
/// This boundary is useful when the reference action is produced outside the
/// CUDA call, for example so evidence tooling can order and time the host and
/// device actions independently. The caller remains responsible for the
/// provenance of `reference`; Eqiora still rejects the result unless every
/// device value agrees under `tolerance`.
///
/// # Errors
/// Returns a stable diagnostic for an invalid reference, unavailable
/// runtime/library/device, unsupported shape/index conversion, transfer or
/// cuSPARSE failure, non-finite data, or disagreement with the reference.
pub fn verify_csr_action_against(
    matrix: &CsrMatrix,
    input: &[f64],
    reference: &[f64],
    device_ordinal: u16,
    policy: SparseActionPolicy,
    tolerance: SparseActionTolerance,
) -> Result<CudaCsrActionResult, Diagnostic> {
    validate_reference(matrix, reference)?;
    verify_csr_action_inner(
        matrix,
        input,
        Some(reference),
        device_ordinal,
        policy,
        tolerance,
    )
}

fn validate_reference(matrix: &CsrMatrix, reference: &[f64]) -> Result<(), Diagnostic> {
    if reference.len() != matrix.rows() {
        return Err(solve_failed(format!(
            "CUDA CSR reference has {} values but the matrix has {} rows",
            reference.len(),
            matrix.rows()
        )));
    }
    if reference.iter().any(|value| !value.is_finite()) {
        return Err(solve_failed("CUDA CSR reference requires finite values"));
    }
    Ok(())
}

fn verify_csr_action_inner(
    matrix: &CsrMatrix,
    input: &[f64],
    reference: Option<&[f64]>,
    device_ordinal: u16,
    policy: SparseActionPolicy,
    tolerance: SparseActionTolerance,
) -> Result<CudaCsrActionResult, Diagnostic> {
    let total_started = Instant::now();
    validate_action(matrix, input)?;
    let setup_started = Instant::now();
    let rows = i64::try_from(matrix.rows())
        .map_err(|_| unsupported("CSR row count exceeds CUDA's signed 64-bit descriptor"))?;
    let columns = i64::try_from(matrix.columns())
        .map_err(|_| unsupported("CSR column count exceeds CUDA's signed 64-bit descriptor"))?;
    let nonzeros = i64::try_from(matrix.values().len())
        .map_err(|_| unsupported("CSR nonzero count exceeds CUDA's signed 64-bit descriptor"))?;
    let row_offsets = convert_indices(matrix.row_offsets(), "row offset")?;
    let column_indices = convert_indices(matrix.column_indices(), "column index")?;

    let discovered = discover_cuda_devices(cuda_device_count()?)?
        .into_iter()
        .find(|candidate| candidate.descriptor.id().ordinal() == device_ordinal)
        .ok_or_else(|| unsupported(format!("CUDA device {device_ordinal} is not visible")))?;
    let device = discovered.descriptor;
    device.require([
        DeviceCapability::Float64,
        DeviceCapability::CsrMatrixVectorProduct,
        DeviceCapability::AsynchronousQueue,
    ])?;
    let context = CudaContext::new(usize::from(device_ordinal)).map_err(driver_failed)?;
    let stream = context.new_stream().map_err(driver_failed)?;
    let queue = materialize_queue(QueueSlot::new(device.id(), 0))?;
    let mut timeline = QueueTimeline::new(queue);
    let handle = CusparseHandle::new(stream.cu_stream()).map_err(cusparse_failed)?;

    let mut device_rows = stream
        .alloc_zeros::<i64>(row_offsets.len())
        .map_err(driver_failed)?;
    let mut device_columns = stream
        .alloc_zeros::<i64>(column_indices.len())
        .map_err(driver_failed)?;
    let mut device_values = stream
        .alloc_zeros::<f64>(matrix.values().len())
        .map_err(driver_failed)?;
    let mut device_input = stream
        .alloc_zeros::<f64>(input.len())
        .map_err(driver_failed)?;
    let mut device_output = stream
        .alloc_zeros::<f64>(matrix.rows())
        .map_err(driver_failed)?;
    let descriptors = BufferDescriptors::new(
        device.id(),
        row_offsets.len(),
        column_indices.len(),
        matrix.values().len(),
        input.len(),
        matrix.rows(),
    )?;
    stream.synchronize().map_err(driver_failed)?;
    let mut setup = setup_started.elapsed();

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
        .memcpy_htod(matrix.values(), &mut device_values)
        .map_err(driver_failed)?;
    let value_completion = Completion::new(timeline.next_submission()?);
    stream
        .memcpy_htod(input, &mut device_input)
        .map_err(driver_failed)?;
    let input_completion = Completion::new(timeline.next_submission()?);
    stream.synchronize().map_err(driver_failed)?;
    let host_to_device = host_to_device_started.elapsed();

    let descriptor_setup_started = Instant::now();
    let (row_pointer, _row_access) = device_rows.device_ptr(&stream);
    let (column_pointer, _column_access) = device_columns.device_ptr(&stream);
    let (value_pointer, _value_access) = device_values.device_ptr(&stream);
    let (input_pointer, _input_access) = device_input.device_ptr(&stream);
    let (output_pointer, _output_access) = device_output.device_ptr_mut(&stream);
    let plan = SpmvPlan::new(
        &handle,
        rows,
        columns,
        nonzeros,
        row_pointer,
        column_pointer,
        value_pointer,
        input_pointer,
        output_pointer,
        policy == SparseActionPolicy::Deterministic,
    )
    .map_err(cusparse_failed)?;
    drop(_output_access);
    drop(_input_access);
    drop(_value_access);
    drop(_column_access);
    drop(_row_access);
    let mut workspace = stream
        .alloc_zeros::<u8>(plan.workspace_bytes().max(1))
        .map_err(driver_failed)?;
    stream.synchronize().map_err(driver_failed)?;
    setup += descriptor_setup_started.elapsed();

    let action_started = Instant::now();
    let action_completion;
    {
        let (_, _row_access) = device_rows.device_ptr(&stream);
        let (_, _column_access) = device_columns.device_ptr(&stream);
        let (_, _value_access) = device_values.device_ptr(&stream);
        let (input_pointer, _input_access) = device_input.device_ptr(&stream);
        let (output_pointer, _output_access) = device_output.device_ptr_mut(&stream);
        let (workspace_pointer, _workspace_access) = workspace.device_ptr_mut(&stream);
        plan.apply(
            &handle,
            workspace_pointer,
            input_pointer,
            output_pointer,
            1.0,
            0.0,
        )
        .map_err(cusparse_failed)?;
        action_completion = Completion::new(timeline.next_submission()?);
        stream.synchronize().map_err(driver_failed)?;
    }
    let action = action_started.elapsed();

    let device_to_host_started = Instant::now();
    let values = stream.clone_dtoh(&device_output).map_err(driver_failed)?;
    let output_completion = Completion::new(timeline.next_submission()?);
    stream.synchronize().map_err(driver_failed)?;
    let device_to_host = device_to_host_started.elapsed();

    let verification_started = Instant::now();
    let expected = match reference {
        Some(reference) => Cow::Borrowed(reference),
        None => Cow::Owned(matrix.multiply(input)?),
    };
    let (maximum_absolute_error, maximum_scaled_error) =
        compare_action(&expected, &values, tolerance)?;
    let verification = verification_started.elapsed();
    let total = total_started.elapsed();
    let timings = eqiora_device::DeviceExecutionTimings::new(
        setup,
        host_to_device,
        action,
        device_to_host,
        verification,
        total,
    )?;
    let versions = CudaLibraryVersions {
        driver: ffi::driver_version().map_err(driver_failed)?,
        cusparse: handle.version().map_err(cusparse_failed)?,
        cublas: None,
        cudarc: CUDARC_VERSION,
        binding_toolkit: CUDA_BINDING_TOOLKIT,
    };
    let transfers = CudaCsrTransferEvidence {
        row_offsets: transfer_to_device(descriptors.rows, row_completion)?,
        column_indices: transfer_to_device(descriptors.columns, column_completion)?,
        values: transfer_to_device(descriptors.values, value_completion)?,
        input: transfer_to_device(descriptors.input, input_completion)?,
        output: transfer_to_host(descriptors.output, output_completion)?,
    };
    Ok(CudaCsrActionResult {
        values,
        evidence: CudaCsrActionEvidence {
            device,
            physical_uuid: discovered.physical_uuid,
            compute_capability: discovered.compute_capability,
            versions,
            policy,
            transfers,
            action_completion,
            workspace_bytes: plan.workspace_bytes(),
            maximum_absolute_error,
            maximum_scaled_error,
            timings,
        },
    })
}

#[derive(Debug)]
struct BufferDescriptors {
    rows: DeviceBufferDescriptor<i64>,
    columns: DeviceBufferDescriptor<i64>,
    values: DeviceBufferDescriptor<f64>,
    input: DeviceBufferDescriptor<f64>,
    output: DeviceBufferDescriptor<f64>,
}

impl BufferDescriptors {
    fn new(
        device: DeviceId,
        row_count: usize,
        column_count: usize,
        value_count: usize,
        input_count: usize,
        output_count: usize,
    ) -> Result<Self, Diagnostic> {
        Ok(Self {
            rows: descriptor(device, row_count)?,
            columns: descriptor(device, column_count)?,
            values: descriptor(device, value_count)?,
            input: descriptor(device, input_count)?,
            output: descriptor(device, output_count)?,
        })
    }
}

fn validate_resident_request(
    matrix: &CsrMatrix,
    selected: &CudaDeviceObservation,
    queue: QueueSlot,
    policy: SparseActionPolicy,
) -> Result<(), Diagnostic> {
    validate_matrix(matrix)?;
    if policy != SparseActionPolicy::Deterministic {
        return Err(unsupported(
            "resident CUDA CSR action v1 requires deterministic sparse action policy",
        ));
    }
    if selected.descriptor.id().runtime() != CUDA_RUNTIME_ID {
        return Err(unsupported(format!(
            "resident CUDA CSR action requires runtime `{}`, received `{}`",
            CUDA_RUNTIME_ID.as_str(),
            selected.descriptor.id().runtime().as_str()
        )));
    }
    if queue.device() != selected.descriptor.id() {
        return Err(unsupported(
            "resident CUDA CSR action queue does not belong to the selected device",
        ));
    }
    selected.descriptor.require([
        DeviceCapability::Float64,
        DeviceCapability::CsrMatrixVectorProduct,
        DeviceCapability::AsynchronousQueue,
    ])
}

fn resident_payload_bytes(matrix: &CsrMatrix) -> Result<usize, Diagnostic> {
    let row_offsets = checked_bytes::<i64>(matrix.row_offsets().len(), "CSR row offsets")?;
    let column_indices = checked_bytes::<i64>(matrix.column_indices().len(), "CSR column indices")?;
    let values = checked_bytes::<f64>(matrix.values().len(), "CSR values")?;
    let input = checked_bytes::<f64>(matrix.columns(), "CSR input")?;
    let output = checked_bytes::<f64>(matrix.rows(), "CSR output")?;
    [row_offsets, column_indices, values, input, output]
        .into_iter()
        .try_fold(0_usize, |total, bytes| {
            total
                .checked_add(bytes)
                .ok_or_else(|| unsupported("resident CUDA CSR payload size overflowed"))
        })
}

fn checked_bytes<T>(elements: usize, label: &str) -> Result<usize, Diagnostic> {
    elements
        .checked_mul(size_of::<T>())
        .ok_or_else(|| unsupported(format!("resident CUDA {label} byte size overflowed")))
}

fn validate_resident_descriptors(descriptors: &BufferDescriptors) -> Result<(), Diagnostic> {
    let ids = [
        descriptors.rows.id(),
        descriptors.columns.id(),
        descriptors.values.id(),
        descriptors.input.id(),
        descriptors.output.id(),
    ];
    let device = ids[0].device();
    if ids.iter().any(|id| id.device() != device) {
        return Err(unsupported(
            "resident CUDA CSR allocations do not belong to one selected device",
        ));
    }
    for (index, id) in ids.iter().enumerate() {
        if ids[..index].contains(id) {
            return Err(unsupported(
                "resident CUDA CSR matrix, input, and output allocations must not alias",
            ));
        }
    }
    Ok(())
}

fn validate_resident_action(
    columns: usize,
    rows: usize,
    input: &[f64],
    output: &[f64],
) -> Result<(), Diagnostic> {
    if input.len() != columns || output.len() != rows {
        return Err(solve_failed(format!(
            "resident CUDA CSR action is {rows}x{columns} but input/output have {}/{} values",
            input.len(),
            output.len()
        )));
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err(solve_failed(
            "resident CUDA CSR action requires finite input values",
        ));
    }
    Ok(())
}

fn validate_resident_setup_evidence(
    evidence: &CudaResidentCsrSetupEvidence,
) -> Result<(), Diagnostic> {
    let matrix_ready = evidence.matrix_ready.completion();
    for transfer in [
        evidence.row_offsets.completion(),
        evidence.column_indices.completion(),
        evidence.values.completion(),
    ] {
        if !transfer.happens_before(matrix_ready)? {
            return Err(unsupported(
                "resident CUDA CSR matrix transfer was not visible before session admission",
            ));
        }
    }
    if evidence.policy != SparseActionPolicy::Deterministic
        || evidence.queue.slot().device() != evidence.device.id()
        || evidence.rows == 0
        || evidence.columns == 0
        || evidence.nonzeros == 0
        || evidence.known_payload_bytes == 0
    {
        return Err(unsupported(
            "resident CUDA CSR setup evidence is incomplete or contradictory",
        ));
    }
    Ok(())
}

fn validate_resident_action_evidence(
    setup: &CudaResidentCsrSetupEvidence,
    evidence: CudaResidentCsrActionEvidence,
) -> Result<(), Diagnostic> {
    let input_completion = evidence.input.completion();
    let input_ready = evidence.input_ready.completion();
    let action_visible = evidence.action_visible.completion();
    let output_completion = evidence.output.completion();
    let output_visible = evidence.output_visible.completion();
    for (earlier, later) in [
        (setup.matrix_ready.completion(), input_completion),
        (input_completion, input_ready),
        (input_ready, evidence.action_completion),
        (evidence.action_completion, action_visible),
        (action_visible, output_completion),
        (output_completion, output_visible),
    ] {
        if !earlier.happens_before(later)? {
            return Err(unsupported(
                "resident CUDA CSR action completion order is not strict and queue-local",
            ));
        }
    }
    let MemoryRegion::Device(input) = evidence.input.plan().destination() else {
        return Err(unsupported(
            "resident CUDA CSR input transfer lost its device destination",
        ));
    };
    let MemoryRegion::Device(output) = evidence.output.plan().source() else {
        return Err(unsupported(
            "resident CUDA CSR output transfer lost its device source",
        ));
    };
    if input.id() != evidence.input_generation.buffer()
        || output.id() != evidence.output_generation.buffer()
        || input.id() == output.id()
        || evidence.input_generation.generation() != evidence.ordinal
        || evidence.output_generation.generation() != evidence.ordinal
    {
        return Err(unsupported(
            "resident CUDA CSR buffer residency or value generation drifted",
        ));
    }
    Ok(())
}

#[derive(Debug)]
struct CudaEventFence {
    event: CudaEvent,
    completion: Completion,
}

impl CudaEventFence {
    fn record(stream: &Arc<CudaStream>, timeline: &mut QueueTimeline) -> Result<Self, Diagnostic> {
        let event = stream.record_event(None).map_err(driver_failed)?;
        let completion = Completion::new(timeline.next_submission()?);
        Ok(Self { event, completion })
    }

    fn waited(&self) -> Result<WaitedCompletion, Diagnostic> {
        WaitedCompletion::wait(self)
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

static NEXT_CUDA_ALLOCATION: AtomicU64 = AtomicU64::new(1);
static NEXT_CUDA_QUEUE_MATERIALIZATION: AtomicU64 = AtomicU64::new(1);

pub(crate) fn materialize_queue(slot: QueueSlot) -> Result<QueueId, Diagnostic> {
    Ok(QueueId::new(
        slot,
        next_process_identity(
            &NEXT_CUDA_QUEUE_MATERIALIZATION,
            "CUDA queue materialization identity",
        )?,
    ))
}

pub(crate) fn descriptor<T: eqiora_device::DeviceElement>(
    device: DeviceId,
    elements: usize,
) -> Result<DeviceBufferDescriptor<T>, Diagnostic> {
    let allocation = next_process_identity(&NEXT_CUDA_ALLOCATION, "CUDA allocation identity")?;
    Ok(DeviceBufferDescriptor::new(
        BufferId::new(device, allocation),
        NonZeroUsize::new(elements).expect("validated CUDA buffers are nonempty"),
    ))
}

fn next_process_identity(timeline: &AtomicU64, label: &str) -> Result<NonZeroU64, Diagnostic> {
    #[allow(deprecated, reason = "try_update requires Rust 1.95; MSRV is 1.89")]
    let identity = timeline
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |allocation| {
            allocation.checked_add(1)
        })
        .map_err(|_| unsupported(format!("{label} space is exhausted")))?;
    Ok(NonZeroU64::new(identity).expect("process identity timelines begin at one"))
}

pub(crate) fn transfer_to_device<T: eqiora_device::DeviceElement>(
    device: DeviceBufferDescriptor<T>,
    completion: Completion,
) -> Result<TransferEvidence<T>, Diagnostic> {
    let host = HostBufferDescriptor::new(device.elements());
    let plan = TransferPlan::new(MemoryRegion::Host(host), MemoryRegion::Device(device))?;
    TransferEvidence::new(plan, completion)
}

pub(crate) fn transfer_to_host<T: eqiora_device::DeviceElement>(
    device: DeviceBufferDescriptor<T>,
    completion: Completion,
) -> Result<TransferEvidence<T>, Diagnostic> {
    let host = HostBufferDescriptor::new(device.elements());
    let plan = TransferPlan::new(MemoryRegion::Device(device), MemoryRegion::Host(host))?;
    TransferEvidence::new(plan, completion)
}

pub(crate) fn validate_action(matrix: &CsrMatrix, input: &[f64]) -> Result<(), Diagnostic> {
    validate_matrix(matrix)?;
    if input.len() != matrix.columns() {
        return Err(solve_failed(format!(
            "CUDA CSR matrix has {} columns but input has {} values",
            matrix.columns(),
            input.len()
        )));
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err(solve_failed(
            "CUDA CSR action requires finite matrix and input values",
        ));
    }
    Ok(())
}

fn validate_matrix(matrix: &CsrMatrix) -> Result<(), Diagnostic> {
    if matrix.rows() == 0 || matrix.columns() == 0 || matrix.values().is_empty() {
        return Err(unsupported(
            "CUDA CSR action requires a nonempty matrix with at least one nonzero",
        ));
    }
    if matrix.values().iter().any(|value| !value.is_finite()) {
        return Err(solve_failed(
            "CUDA CSR action requires finite matrix values",
        ));
    }
    let offsets = matrix.row_offsets();
    if offsets.len() != matrix.rows() + 1
        || offsets.first() != Some(&0)
        || offsets.last() != Some(&matrix.values().len())
        || offsets.windows(2).any(|pair| pair[0] > pair[1])
        || matrix.column_indices().len() != matrix.values().len()
        || matrix
            .column_indices()
            .iter()
            .any(|column| *column >= matrix.columns())
    {
        return Err(solve_failed(
            "CUDA CSR action received an invalid finalized CSR structure",
        ));
    }
    Ok(())
}

pub(crate) fn convert_indices(values: &[usize], label: &str) -> Result<Vec<i64>, Diagnostic> {
    let mut converted = Vec::new();
    converted
        .try_reserve_exact(values.len())
        .map_err(|_| unsupported(format!("could not reserve converted CSR {label}")))?;
    for value in values {
        converted.push(
            i64::try_from(*value).map_err(|_| {
                unsupported(format!("CSR {label} exceeds signed 64-bit CUDA index"))
            })?,
        );
    }
    Ok(converted)
}

fn compare_action(
    expected: &[f64],
    actual: &[f64],
    tolerance: SparseActionTolerance,
) -> Result<(f64, f64), Diagnostic> {
    if expected.len() != actual.len()
        || expected
            .iter()
            .chain(actual)
            .any(|value| !value.is_finite())
    {
        return Err(solve_failed(
            "CUDA CSR reference and result must have the same shape and finite values",
        ));
    }
    let mut maximum_absolute = 0.0_f64;
    let mut maximum_scaled = 0.0_f64;
    for (index, (reference, candidate)) in expected.iter().zip(actual).enumerate() {
        let error = (reference - candidate).abs();
        let threshold = tolerance.threshold(*reference);
        let scaled = if threshold == 0.0 {
            if error == 0.0 { 0.0 } else { f64::INFINITY }
        } else {
            error / threshold
        };
        maximum_absolute = maximum_absolute.max(error);
        maximum_scaled = maximum_scaled.max(scaled);
        if error > threshold {
            return Err(solve_failed(format!(
                "CUDA CSR action differs from the host oracle at row {index}: error {error:e} exceeds {threshold:e}"
            )));
        }
    }
    Ok((maximum_absolute, maximum_scaled))
}

pub(crate) fn cuda_device_count() -> Result<i32, Diagnostic> {
    catch_unwind(AssertUnwindSafe(CudaContext::device_count))
        .map_err(|_| unsupported("CUDA discovery could not load a compatible driver library"))?
        .map_err(driver_failed)
}

pub(crate) fn driver_failed(error: cudarc::driver::DriverError) -> Diagnostic {
    unsupported(format!("CUDA driver operation failed: {error:?}"))
}

pub(crate) fn cusparse_failed(error: ffi::FfiError) -> Diagnostic {
    solve_failed(format!("cuSPARSE operation failed: {error:?}"))
}

pub(crate) fn unsupported(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_REALIZATION, message)
}

pub(crate) fn solve_failed(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::NUMERICAL_SOLVE_FAILED, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rectangular_matrix() -> CsrMatrix {
        CsrMatrix::from_sorted_csr(
            2,
            3,
            vec![0, 2, 4],
            vec![0, 1, 1, 2],
            vec![1.0, 2.0, -1.0, 3.0],
        )
        .unwrap()
    }

    fn observed_device(ordinal: u16) -> CudaDeviceObservation {
        CudaDeviceObservation {
            descriptor: DeviceDescriptor::new(
                DeviceId::new(CUDA_RUNTIME_ID, ordinal),
                "synthetic CUDA device",
                NonZeroU64::new(1 << 20).unwrap(),
                [
                    DeviceCapability::Float64,
                    DeviceCapability::CsrMatrixVectorProduct,
                    DeviceCapability::AsynchronousQueue,
                ],
            )
            .unwrap(),
            physical_uuid: CudaDeviceUuid::from_driver([ordinal as u8 + 1; 16]).unwrap(),
            compute_capability: CudaComputeCapability::from_driver(8, 6).unwrap(),
        }
    }

    #[test]
    fn allocation_identity_is_monotone_across_descriptor_groups() {
        let device = DeviceId::new(CUDA_RUNTIME_ID, 0);
        let first = BufferDescriptors::new(device, 2, 1, 1, 1, 1).unwrap();
        let second = BufferDescriptors::new(device, 2, 1, 1, 1, 1).unwrap();

        assert!(first.output.id().allocation() < second.rows.id().allocation());
    }

    #[test]
    fn separate_materializations_of_one_slot_have_distinct_queue_identity() {
        let slot = QueueSlot::new(DeviceId::new(CUDA_RUNTIME_ID, 0), 0);
        let first = materialize_queue(slot).unwrap();
        let second = materialize_queue(slot).unwrap();

        assert_eq!(first.slot(), second.slot());
        assert!(first.materialization() < second.materialization());
        let mut first_timeline = QueueTimeline::new(first);
        let mut second_timeline = QueueTimeline::new(second);
        let first_completion = Completion::new(first_timeline.next_submission().unwrap());
        let second_completion = Completion::new(second_timeline.next_submission().unwrap());
        assert!(first_completion.happens_before(second_completion).is_err());
    }

    #[test]
    fn physical_uuid_is_fixed_width_and_rejects_nil() {
        let bytes = [0xabu8; 16];
        assert_eq!(
            CudaDeviceUuid::from_driver(bytes).unwrap().as_bytes(),
            bytes
        );
        assert!(CudaDeviceUuid::from_driver([0; 16]).is_err());
    }

    #[test]
    fn resident_request_rejects_policy_and_queue_ownership_drift_before_runtime_work() {
        let matrix = rectangular_matrix();
        let selected = observed_device(0);
        let exact_queue = QueueSlot::new(selected.descriptor.id(), 0);
        let policy = validate_resident_request(
            &matrix,
            &selected,
            exact_queue,
            SparseActionPolicy::BackendNative,
        )
        .unwrap_err();
        assert!(policy.message().contains("deterministic"));

        let foreign_queue = QueueSlot::new(DeviceId::new(CUDA_RUNTIME_ID, 1), 0);
        let queue = validate_resident_request(
            &matrix,
            &selected,
            foreign_queue,
            SparseActionPolicy::Deterministic,
        )
        .unwrap_err();
        assert!(queue.message().contains("does not belong"));

        validate_resident_request(
            &matrix,
            &selected,
            QueueSlot::new(selected.descriptor.id(), 1),
            SparseActionPolicy::Deterministic,
        )
        .unwrap();
    }

    #[test]
    fn resident_action_shape_and_finite_input_fail_without_submission() {
        let output = [0.0; 2];
        let shape = validate_resident_action(3, 2, &[1.0, 2.0], &output).unwrap_err();
        assert!(shape.message().contains("input/output have 2/2"));
        let finite = validate_resident_action(3, 2, &[1.0, f64::NAN, 3.0], &output).unwrap_err();
        assert!(finite.message().contains("finite input"));
        validate_resident_action(3, 2, &[1.0, 2.0, 3.0], &output).unwrap();
    }

    #[test]
    fn resident_descriptors_reject_any_allocation_alias() {
        let device = DeviceId::new(CUDA_RUNTIME_ID, 0);
        let shared = BufferId::new(device, NonZeroU64::MIN);
        let descriptors = BufferDescriptors {
            rows: DeviceBufferDescriptor::new(shared, NonZeroUsize::new(3).unwrap()),
            columns: DeviceBufferDescriptor::new(shared, NonZeroUsize::new(4).unwrap()),
            values: DeviceBufferDescriptor::new(
                BufferId::new(device, NonZeroU64::new(2).unwrap()),
                NonZeroUsize::new(4).unwrap(),
            ),
            input: DeviceBufferDescriptor::new(
                BufferId::new(device, NonZeroU64::new(3).unwrap()),
                NonZeroUsize::new(3).unwrap(),
            ),
            output: DeviceBufferDescriptor::new(
                BufferId::new(device, NonZeroU64::new(4).unwrap()),
                NonZeroUsize::new(2).unwrap(),
            ),
        };
        let error = validate_resident_descriptors(&descriptors).unwrap_err();
        assert!(error.message().contains("must not alias"));
    }
}
