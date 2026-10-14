use std::num::NonZeroUsize;

use eqiora::distributed::DistributedAdmissionFingerprintV1;
use eqiora_execution::{DistributedLinearExecutionTrace, ExecutionReceipt};

pub(crate) fn assert_trace(
    receipt: &ExecutionReceipt,
    admission: DistributedAdmissionFingerprintV1,
    partitions: usize,
    workers_per_partition: NonZeroUsize,
) -> &DistributedLinearExecutionTrace {
    let trace = receipt
        .distributed_trace()
        .expect("the graph-bound MPI run retains its distributed trace");
    assert_eq!(trace.system(), receipt.operator());
    assert_eq!(trace.admission(), admission);
    assert_eq!(trace.owner_gather_dimension(), receipt.dimension());
    assert_eq!(trace.partitions().get(), partitions);
    assert_eq!(trace.workers_per_partition(), workers_per_partition);
    assert!(!trace.steps().is_empty());
    assert!(trace.steps().len() <= trace.trace_capacity());
    assert!(
        trace
            .steps()
            .iter()
            .enumerate()
            .all(|(ordinal, step)| step.ordinal() == ordinal)
    );
    trace
}
