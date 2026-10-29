use super::*;
use eqiora_execution::ProcessGroupSlot;
use eqiora_solver::{REFERENCE_SOLVER_PROVIDER, SERIAL_EXECUTION_PROVIDER};

#[test]
fn selected_native_descriptor_requires_actual_shape_and_exact_snapshot() {
    let ranks = NonZeroUsize::new(4).unwrap();
    let caps = native_capabilities();
    let descriptor = |solver, execution, slot, count, workers, capabilities| {
        DistributedExecutorDescriptor::new(
            solver,
            execution,
            ProcessGroupSlot::new(slot),
            count,
            workers,
            capabilities,
        )
    };
    let exact = descriptor(
        MPI_DISTRIBUTED_KRYLOV_SOLVER_PROVIDER,
        MPI_EXECUTION_PROVIDER,
        0,
        ranks,
        NonZeroUsize::MIN,
        caps.clone(),
    );
    require_selected_descriptor(&exact, ranks, &caps).unwrap();
    let subset = SolverCapabilities::exact([*caps.combinations().iter().next().unwrap()]).unwrap();
    for changed in [
        descriptor(
            REFERENCE_SOLVER_PROVIDER,
            MPI_EXECUTION_PROVIDER,
            0,
            ranks,
            NonZeroUsize::MIN,
            caps.clone(),
        ),
        descriptor(
            MPI_DISTRIBUTED_KRYLOV_SOLVER_PROVIDER,
            SERIAL_EXECUTION_PROVIDER,
            0,
            ranks,
            NonZeroUsize::MIN,
            caps.clone(),
        ),
        descriptor(
            MPI_DISTRIBUTED_KRYLOV_SOLVER_PROVIDER,
            MPI_EXECUTION_PROVIDER,
            1,
            ranks,
            NonZeroUsize::MIN,
            caps.clone(),
        ),
        descriptor(
            MPI_DISTRIBUTED_KRYLOV_SOLVER_PROVIDER,
            MPI_EXECUTION_PROVIDER,
            0,
            NonZeroUsize::MIN,
            NonZeroUsize::MIN,
            caps.clone(),
        ),
        descriptor(
            MPI_DISTRIBUTED_KRYLOV_SOLVER_PROVIDER,
            MPI_EXECUTION_PROVIDER,
            0,
            ranks,
            NonZeroUsize::new(2).unwrap(),
            caps.clone(),
        ),
        descriptor(
            MPI_DISTRIBUTED_KRYLOV_SOLVER_PROVIDER,
            MPI_EXECUTION_PROVIDER,
            0,
            ranks,
            NonZeroUsize::MIN,
            subset,
        ),
    ] {
        assert!(
            require_selected_descriptor(&changed, ranks, &caps)
                .unwrap_err()
                .message()
                .contains("actual providers, slot zero, ranks, one worker or exact capabilities")
        );
    }
}
