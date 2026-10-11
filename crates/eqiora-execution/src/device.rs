use std::num::NonZeroU64;

use eqiora_core::Diagnostic;
use eqiora_device::{
    BufferId, Completion, DeviceBufferDescriptor, DeviceElement, DeviceId, MemoryRegion, QueueId,
    QueueSlot, TransferDirection, TransferEvidence, WaitedCompletion,
};
use eqiora_solver::{CanonicalCsrSystemView, PreconditionerPolicy, SolverPlan};

use crate::binding::invalid;

/// Explicit typed movements for one finalized CSR device solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CsrDeviceTransferEvidence {
    row_offsets: TransferEvidence<i64>,
    column_indices: TransferEvidence<i64>,
    values: TransferEvidence<f64>,
    right_hand_side: TransferEvidence<f64>,
    zero_initial_solution: TransferEvidence<f64>,
    inverse_diagonal: Option<TransferEvidence<f64>>,
    complete_solution: TransferEvidence<f64>,
}

impl CsrDeviceTransferEvidence {
    /// Retain every transfer slot of the bounded implicit-zero CSR solve.
    #[must_use]
    pub const fn new(
        row_offsets: TransferEvidence<i64>,
        column_indices: TransferEvidence<i64>,
        values: TransferEvidence<f64>,
        right_hand_side: TransferEvidence<f64>,
        zero_initial_solution: TransferEvidence<f64>,
        inverse_diagonal: Option<TransferEvidence<f64>>,
        complete_solution: TransferEvidence<f64>,
    ) -> Self {
        Self {
            row_offsets,
            column_indices,
            values,
            right_hand_side,
            zero_initial_solution,
            inverse_diagonal,
            complete_solution,
        }
    }

    /// CSR row-offset upload.
    #[must_use]
    pub const fn row_offsets(self) -> TransferEvidence<i64> {
        self.row_offsets
    }

    /// CSR column-index upload.
    #[must_use]
    pub const fn column_indices(self) -> TransferEvidence<i64> {
        self.column_indices
    }

    /// CSR coefficient upload.
    #[must_use]
    pub const fn values(self) -> TransferEvidence<f64> {
        self.values
    }

    /// Complete right-hand-side upload.
    #[must_use]
    pub const fn right_hand_side(self) -> TransferEvidence<f64> {
        self.right_hand_side
    }

    /// Sole admitted implicit-zero initial-solution upload.
    #[must_use]
    pub const fn zero_initial_solution(self) -> TransferEvidence<f64> {
        self.zero_initial_solution
    }

    /// Jacobi inverse-diagonal upload, absent for identity preconditioning.
    #[must_use]
    pub const fn inverse_diagonal(self) -> Option<TransferEvidence<f64>> {
        self.inverse_diagonal
    }

    /// Complete solved vector copied to host.
    #[must_use]
    pub const fn complete_solution(self) -> TransferEvidence<f64> {
        self.complete_solution
    }
}

/// One logical value generation in a reused device allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceValueGeneration {
    buffer: BufferId,
    generation: NonZeroU64,
}

impl DeviceValueGeneration {
    /// Name a logical generation of one runtime-local allocation.
    #[must_use]
    pub const fn new(buffer: BufferId, generation: NonZeroU64) -> Self {
        Self { buffer, generation }
    }

    /// Reused device allocation.
    #[must_use]
    pub const fn buffer(self) -> BufferId {
        self.buffer
    }

    /// Adapter-owned monotone logical generation.
    #[must_use]
    pub const fn generation(self) -> NonZeroU64 {
        self.generation
    }
}

/// Closed trace of the first one-device CSR Krylov execution.
///
/// Vendor events and pointers remain private to the adapter. The trace retains
/// exact typed transfers, queue completions, successful fence waits, reused
/// solution generations, and the external sparse workspace actually reported
/// by the adapter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CudaLinearExecutionTrace {
    transfers: CsrDeviceTransferEvidence,
    inputs_ready: WaitedCompletion,
    solve_visible: WaitedCompletion,
    solution_visible: WaitedCompletion,
    initial_solution: DeviceValueGeneration,
    solved_solution: DeviceValueGeneration,
    downloaded_solution: DeviceValueGeneration,
    external_sparse_workspace_bytes: usize,
}

impl CudaLinearExecutionTrace {
    /// Construct one ordered trace after the adapter has waited for the solve
    /// and complete-output visibility fences.
    ///
    /// # Errors
    /// Returns `EQ0807` for cross-queue or non-monotone phases, mismatched
    /// transfer completions, or a substituted solution generation.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        transfers: CsrDeviceTransferEvidence,
        inputs_ready: WaitedCompletion,
        solve_visible: WaitedCompletion,
        solution_visible: WaitedCompletion,
        initial_solution: DeviceValueGeneration,
        solved_solution: DeviceValueGeneration,
        downloaded_solution: DeviceValueGeneration,
        external_sparse_workspace_bytes: usize,
    ) -> Result<Self, Diagnostic> {
        let uploads = upload_completions(transfers);
        for (index, completion) in uploads.iter().enumerate() {
            let Some(completion) = completion else {
                continue;
            };
            if uploads[..index].contains(&Some(*completion)) {
                return Err(invalid(
                    "each canonical device-input transfer must have a distinct submission identity",
                ));
            }
            if !completion.happens_before(inputs_ready.completion())? {
                return Err(invalid(
                    "every device-input transfer must precede the waited inputs-ready fence",
                ));
            }
        }
        let solve_completion = solve_visible.completion();
        let output_transfer = transfers.complete_solution().completion();
        let output_visible = solution_visible.completion();
        if !inputs_ready.completion().happens_before(solve_completion)?
            || !solve_completion.happens_before(output_transfer)?
            || !output_transfer.happens_before(output_visible)?
        {
            return Err(invalid(
                "input fence, solve fence, output transfer, and host-visibility fence must be strictly ordered",
            ));
        }
        if initial_solution.buffer != solved_solution.buffer
            || downloaded_solution != solved_solution
            || initial_solution
                .generation
                .get()
                .checked_add(1)
                .and_then(NonZeroU64::new)
                != Some(solved_solution.generation)
        {
            return Err(invalid(
                "device solution must advance exactly one generation in one reused allocation before download",
            ));
        }
        Ok(Self {
            transfers,
            inputs_ready,
            solve_visible,
            solution_visible,
            initial_solution,
            solved_solution,
            downloaded_solution,
            external_sparse_workspace_bytes,
        })
    }

    /// Complete typed transfer set.
    #[must_use]
    pub const fn transfers(self) -> CsrDeviceTransferEvidence {
        self.transfers
    }

    /// Queue completion recorded after all inputs were uploaded.
    #[must_use]
    pub const fn inputs_ready(self) -> WaitedCompletion {
        self.inputs_ready
    }

    /// Successfully waited device-solve completion.
    #[must_use]
    pub const fn solve_visible(self) -> WaitedCompletion {
        self.solve_visible
    }

    /// Successfully waited complete host-output visibility.
    #[must_use]
    pub const fn solution_visible(self) -> WaitedCompletion {
        self.solution_visible
    }

    /// Exact runtime queue materialized for this trace.
    #[must_use]
    pub const fn queue(self) -> QueueId {
        self.inputs_ready.completion().submission().queue()
    }

    /// Initial zero value resident in the reused solution allocation.
    #[must_use]
    pub const fn initial_solution(self) -> DeviceValueGeneration {
        self.initial_solution
    }

    /// Solve-produced value in the reused solution allocation.
    #[must_use]
    pub const fn solved_solution(self) -> DeviceValueGeneration {
        self.solved_solution
    }

    /// Exact solved generation consumed by the D2H transfer.
    #[must_use]
    pub const fn downloaded_solution(self) -> DeviceValueGeneration {
        self.downloaded_solution
    }

    /// External sparse-action workspace reported by the adapter.
    ///
    /// This is not total resident device memory.
    #[must_use]
    pub const fn external_sparse_workspace_bytes(self) -> usize {
        self.external_sparse_workspace_bytes
    }

    pub(crate) fn validate_against(
        self,
        system: &CanonicalCsrSystemView,
        plan: SolverPlan,
        device: DeviceId,
        queue: QueueSlot,
    ) -> Result<(), Diagnostic> {
        if queue.device() != device {
            return Err(invalid(
                "admitted device queue does not belong to the selected device",
            ));
        }
        let materialized_queue = self.queue();
        if materialized_queue.slot() != queue {
            return Err(invalid(
                "device execution trace substituted the admitted logical queue slot",
            ));
        }
        for completion in trace_completions(self).into_iter().flatten() {
            if completion.submission().queue() != materialized_queue {
                return Err(invalid(
                    "device execution trace mixed materialized command queues",
                ));
            }
        }

        let rows = require_upload(
            self.transfers.row_offsets,
            system.row_offsets().len(),
            device,
            "row offsets",
        )?;
        let columns = require_upload(
            self.transfers.column_indices,
            system.column_indices().len(),
            device,
            "column indices",
        )?;
        let values = require_upload(
            self.transfers.values,
            system.values().len(),
            device,
            "matrix values",
        )?;
        let right = require_upload(
            self.transfers.right_hand_side,
            system.right_hand_side().len(),
            device,
            "right-hand side",
        )?;
        let initial = require_upload(
            self.transfers.zero_initial_solution,
            system.columns(),
            device,
            "zero initial solution",
        )?;
        let diagonal = match (plan.preconditioner(), self.transfers.inverse_diagonal) {
            (PreconditionerPolicy::Identity, None) => None,
            (PreconditionerPolicy::Jacobi, Some(transfer)) => Some(require_upload(
                transfer,
                system.columns(),
                device,
                "Jacobi inverse diagonal",
            )?),
            (PreconditionerPolicy::Identity, Some(_)) => {
                return Err(invalid(
                    "identity preconditioning cannot carry a diagonal transfer",
                ));
            }
            (PreconditionerPolicy::Jacobi, None) => {
                return Err(invalid(
                    "Jacobi preconditioning requires an inverse-diagonal transfer",
                ));
            }
        };
        let output = require_download(
            self.transfers.complete_solution,
            system.columns(),
            device,
            "complete solution",
        )?;

        if initial.id() != output.id()
            || self.initial_solution.buffer != initial.id()
            || self.solved_solution.buffer != output.id()
            || self.downloaded_solution != self.solved_solution
        {
            return Err(invalid(
                "D2H transfer did not consume the solved generation of the admitted solution allocation",
            ));
        }
        let allocations = [
            Some(rows.id()),
            Some(columns.id()),
            Some(values.id()),
            Some(right.id()),
            Some(output.id()),
            diagonal.map(DeviceBufferDescriptor::id),
        ];
        for (index, buffer) in allocations.iter().enumerate() {
            if buffer.is_some() && allocations[..index].contains(buffer) {
                return Err(invalid(
                    "distinct canonical device slots cannot alias one allocation",
                ));
            }
        }
        Ok(())
    }
}

fn upload_completions(transfers: CsrDeviceTransferEvidence) -> [Option<Completion>; 6] {
    [
        Some(transfers.row_offsets.completion()),
        Some(transfers.column_indices.completion()),
        Some(transfers.values.completion()),
        Some(transfers.right_hand_side.completion()),
        Some(transfers.zero_initial_solution.completion()),
        transfers.inverse_diagonal.map(TransferEvidence::completion),
    ]
}

fn trace_completions(trace: CudaLinearExecutionTrace) -> [Option<Completion>; 10] {
    let uploads = upload_completions(trace.transfers);
    [
        uploads[0],
        uploads[1],
        uploads[2],
        uploads[3],
        uploads[4],
        uploads[5],
        Some(trace.inputs_ready.completion()),
        Some(trace.solve_visible.completion()),
        Some(trace.transfers.complete_solution.completion()),
        Some(trace.solution_visible.completion()),
    ]
}

fn require_upload<T: DeviceElement>(
    transfer: TransferEvidence<T>,
    elements: usize,
    device: DeviceId,
    label: &str,
) -> Result<DeviceBufferDescriptor<T>, Diagnostic> {
    let plan = transfer.plan();
    if plan.direction() != TransferDirection::HostToDevice {
        return Err(invalid(format!("{label} must move from host to device")));
    }
    let (MemoryRegion::Host(host), MemoryRegion::Device(buffer)) =
        (plan.source(), plan.destination())
    else {
        return Err(invalid(format!(
            "{label} transfer endpoints contradict its direction"
        )));
    };
    if host.elements().get() != elements
        || buffer.elements().get() != elements
        || buffer.id().device() != device
    {
        return Err(invalid(format!(
            "{label} transfer extent or device differs from the admitted canonical slot"
        )));
    }
    Ok(buffer)
}

fn require_download<T: DeviceElement>(
    transfer: TransferEvidence<T>,
    elements: usize,
    device: DeviceId,
    label: &str,
) -> Result<DeviceBufferDescriptor<T>, Diagnostic> {
    let plan = transfer.plan();
    if plan.direction() != TransferDirection::DeviceToHost {
        return Err(invalid(format!("{label} must move from device to host")));
    }
    let (MemoryRegion::Device(buffer), MemoryRegion::Host(host)) =
        (plan.source(), plan.destination())
    else {
        return Err(invalid(format!(
            "{label} transfer endpoints contradict its direction"
        )));
    };
    if host.elements().get() != elements
        || buffer.elements().get() != elements
        || buffer.id().device() != device
    {
        return Err(invalid(format!(
            "{label} transfer extent or device differs from the admitted canonical slot"
        )));
    }
    Ok(buffer)
}
