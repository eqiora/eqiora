use super::*;

#[test]
fn cuda_trace_rejects_queue_generation_and_visibility_substitution() {
    let graph = portable_cuda_graph();
    let system = system([1.0, 0.0]);
    let device = cuda_device(1 << 20);
    let queue = materialized_queue(device.id(), 0, 3);
    let binding = cuda_binding(&graph, device);
    let plan = binding.solver_plan().clone();
    let (trace, initial) = cuda_trace(queue.device(), queue);
    let transfers = trace.transfers();
    let duplicated_column = TransferEvidence::new(
        transfers.column_indices().plan(),
        transfers.row_offsets().completion(),
    )
    .unwrap();
    let duplicate_transfer = CudaLinearExecutionTrace::new(
        CsrDeviceTransferEvidence::new(
            transfers.row_offsets(),
            duplicated_column,
            transfers.values(),
            transfers.right_hand_side(),
            transfers.zero_initial_solution(),
            transfers.inverse_diagonal(),
            transfers.complete_solution(),
        ),
        trace.inputs_ready(),
        trace.solve_visible(),
        trace.solution_visible(),
        trace.initial_solution(),
        trace.solved_solution(),
        trace.downloaded_solution(),
        trace.external_sparse_workspace_bytes(),
    )
    .unwrap_err();
    assert!(duplicate_transfer.message().contains("distinct submission"));

    let skipped = DeviceValueGeneration::new(
        trace.initial_solution().buffer(),
        NonZeroU64::new(3).unwrap(),
    );
    let skipped_generation = CudaLinearExecutionTrace::new(
        trace.transfers(),
        trace.inputs_ready(),
        trace.solve_visible(),
        trace.solution_visible(),
        trace.initial_solution(),
        skipped,
        skipped,
        trace.external_sparse_workspace_bytes(),
    )
    .unwrap_err();
    assert!(
        skipped_generation
            .message()
            .contains("exactly one generation")
    );

    let wrong_queue = materialized_queue(queue.device(), 1, 4);
    let (wrong_trace, _) = cuda_trace(queue.device(), wrong_queue);
    let rejected = AdmittedExecution::admit_cuda_linear(&graph, &system, binding)
        .unwrap()
        .accept_cuda(cuda_solution(&system, plan.clone()), wrong_trace)
        .unwrap_err();
    assert!(rejected.message().contains("logical queue"));

    let mut timeline = QueueTimeline::new(queue);
    let inputs = Completion::new(timeline.next_submission().unwrap());
    let solve = Completion::new(timeline.next_submission().unwrap());
    let output = Completion::new(timeline.next_submission().unwrap());
    let solution = device_buffer::<f64>(queue.device(), 5, 2);
    let stale = CudaLinearExecutionTrace::new(
        trace.transfers(),
        WaitedCompletion::wait(&TestFence {
            completion: inputs,
            succeeds: true,
        })
        .unwrap(),
        WaitedCompletion::wait(&TestFence {
            completion: solve,
            succeeds: true,
        })
        .unwrap(),
        WaitedCompletion::wait(&TestFence {
            completion: output,
            succeeds: true,
        })
        .unwrap(),
        initial,
        DeviceValueGeneration::new(solution.id(), NonZeroU64::new(2).unwrap()),
        initial,
        128,
    );
    assert!(stale.is_err());

    let failed_visibility = WaitedCompletion::wait(&TestFence {
        completion: output,
        succeeds: false,
    });
    assert!(failed_visibility.is_err());
}

#[test]
fn solver_report_validation_still_rejects_impossible_bounds() {
    let invalid = SolverPlan::new(
        eqiora_solver::LinearSolver::ConjugateGradient,
        f64::NAN,
        0.0,
        NonZeroUsize::MIN,
    )
    .unwrap_err();
    assert_eq!(invalid.code(), codes::INVALID_REALIZATION);
    assert_eq!(NonZeroU64::MIN.get(), 1);
}
