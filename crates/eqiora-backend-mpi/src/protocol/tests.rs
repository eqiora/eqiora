use super::*;

use eqiora_distributed::GlobalVectorSpace;
use eqiora_solver::{
    BackendId, ExecutionId, ExecutionProvider, ExecutionReport, LinearOperatorOrientation,
    SolverPlan, SolverProvider,
};

#[test]
fn phase_status_selects_lowest_rejected_partition() {
    let partitions = NonZeroUsize::new(3).unwrap();
    let step = CollectiveStepV1::new(CollectivePhaseV1::LocalAction, 4, 2);
    let records = [
        PhaseStatusV1::ready(step, PartitionId::new(0)).unwrap(),
        PhaseStatusV1::rejected(
            step,
            PartitionId::new(1),
            DistributedProtocolFailureV1::NumericalFailure,
        )
        .unwrap(),
        PhaseStatusV1::rejected(
            step,
            PartitionId::new(2),
            DistributedProtocolFailureV1::InvalidRealization,
        )
        .unwrap(),
    ];
    let decoded = records
        .into_iter()
        .map(PhaseStatusV1::encode)
        .map(PhaseStatusV1::decode)
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let error = evaluate_phase_statuses(&decoded, partitions, step).unwrap_err();
    assert_eq!(error.code(), codes::NUMERICAL_SOLVE_FAILED);
    assert!(error.message().contains("partition 1"));

    let mut wrong_iteration = decoded;
    wrong_iteration[2].iteration = 5;
    assert_eq!(
        evaluate_phase_statuses(&wrong_iteration, partitions, step)
            .unwrap_err()
            .code(),
        codes::INVALID_REALIZATION
    );
    let mut wrong_ordinal = wrong_iteration;
    wrong_ordinal[2].iteration = 4;
    wrong_ordinal[2].ordinal = 3;
    assert_eq!(
        evaluate_phase_statuses(&wrong_ordinal, partitions, step)
            .unwrap_err()
            .code(),
        codes::INVALID_REALIZATION
    );
}

#[test]
fn phase_status_wire_golden_includes_the_exact_collective_step() {
    let encoded = PhaseStatusV1::rejected(
        CollectiveStepV1::new(CollectivePhaseV1::Reduction, 0x0102, 0x0304),
        PartitionId::new(5),
        DistributedProtocolFailureV1::NumericalFailure,
    )
    .unwrap()
    .encode();
    assert_eq!(
        encoded,
        [
            1, 7, 2, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 5,
        ]
    );
    assert_eq!(PhaseStatusV1::decode(encoded).unwrap().encode(), encoded);
}

#[test]
fn accepted_collective_trace_requires_dense_admission_and_terminal_suffix() {
    let completed_iterations = 4;
    let phases = [
        (CollectivePhaseV1::Admission, 0),
        (CollectivePhaseV1::LocalAction, 0),
        (CollectivePhaseV1::ProducerReport, completed_iterations),
        (CollectivePhaseV1::ProducerReport, completed_iterations),
        (CollectivePhaseV1::ProducerReport, completed_iterations),
        (CollectivePhaseV1::GatherPreparation, 0),
        (CollectivePhaseV1::GatherValidation, 0),
        (CollectivePhaseV1::HostAcceptance, completed_iterations),
        (CollectivePhaseV1::ResultAgreement, 0),
        (CollectivePhaseV1::ResultAgreement, 0),
    ];
    let steps = phases
        .into_iter()
        .enumerate()
        .map(|(ordinal, (phase, iteration))| CollectiveStepV1::new(phase, iteration, ordinal))
        .collect::<Vec<_>>();
    validate_successful_collective_trace(&steps, completed_iterations).unwrap();

    let mut sparse = steps.clone();
    sparse[1] = CollectiveStepV1::new(CollectivePhaseV1::LocalAction, 0, 2);
    assert!(validate_successful_collective_trace(&sparse, completed_iterations).is_err());

    let mut repeated_admission = steps.clone();
    repeated_admission[1] = CollectiveStepV1::new(CollectivePhaseV1::Admission, 0, 1);
    assert!(
        validate_successful_collective_trace(&repeated_admission, completed_iterations).is_err()
    );

    let mut reordered = steps.clone();
    reordered.swap(5, 6);
    reordered[5] = CollectiveStepV1::new(reordered[5].phase(), reordered[5].iteration(), 5);
    reordered[6] = CollectiveStepV1::new(reordered[6].phase(), reordered[6].iteration(), 6);
    assert!(validate_successful_collective_trace(&reordered, completed_iterations).is_err());

    assert!(validate_successful_collective_trace(&steps, completed_iterations + 1).is_err());
}

#[test]
fn admission_requires_exact_rank_order_and_fingerprint() {
    let partitions = NonZeroUsize::new(2).unwrap();
    let record = |rank: u64, fingerprint: u8| {
        let mut bytes = [0_u8; ADMISSION_RECORD_BYTES];
        bytes[0] = ADMISSION_PROTOCOL_VERSION;
        bytes[2..10].copy_from_slice(&2_u64.to_be_bytes());
        bytes[10..18].copy_from_slice(&rank.to_be_bytes());
        bytes[18..50].fill(fingerprint);
        AdmissionRecordV1::decode(bytes).unwrap()
    };
    assert!(evaluate_admission(&[record(0, 7), record(1, 7)], partitions).is_ok());
    assert_eq!(
        evaluate_admission(&[record(0, 7), record(1, 8)], partitions)
            .unwrap_err()
            .code(),
        codes::INVALID_REALIZATION
    );
    assert_eq!(
        evaluate_admission(&[record(1, 7), record(0, 7)], partitions)
            .unwrap_err()
            .code(),
        codes::INVALID_REALIZATION
    );
}

#[test]
fn owner_gather_uses_explicit_indices_for_noncontiguous_ownership() {
    let partition = arbitrary_partition();
    let plan = OwnedGatherPlanV1::new(&partition).unwrap();
    assert_eq!(plan.counts(), &[2, 1, 2]);
    assert_eq!(plan.displacements(), &[0, 2, 3]);

    // Rank-order blocks: rank 0 owns [1, 4], rank 1 owns [2], rank 2 owns
    // [0, 3]. Neither counts nor displacements imply these indices.
    let indices = [1_u64, 4, 2, 0, 3];
    let values = [11.0, 44.0, 22.0, 0.0, 33.0];
    assert_eq!(
        plan.reconstruct(&indices, &values).unwrap(),
        [0.0, 11.0, 22.0, 33.0, 44.0]
    );

    let mut wrong_owner = indices;
    wrong_owner[0] = 0;
    assert!(plan.reconstruct(&wrong_owner, &values).is_err());
    let mut nonfinite = values;
    nonfinite[3] = f64::NAN;
    assert!(plan.reconstruct(&indices, &nonfinite).is_err());
}

#[test]
fn mpi_v0_count_conversion_is_checked() {
    assert_eq!(
        checked_mpi_count(i32::MAX as usize, "test").unwrap(),
        i32::MAX
    );
    assert!(checked_mpi_count(i32::MAX as usize + 1, "test").is_err());
}

#[test]
fn producer_summary_covers_every_admitted_report_axis() {
    const CHANGED_LIBRARIES: &[ProviderLibrary] =
        &[ProviderLibrary::new("eqiora-test-provider", "9.9.9")];
    let plan = SolverPlan::new(
        LinearSolver::ConjugateGradient,
        1.0e-12,
        1.0e-14,
        NonZeroUsize::new(10).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Jacobi);
    let solver_provider = SolverProvider::new(BackendId::new("eqiora.mpi.krylov"), "test", &[]);
    let execution_provider = ExecutionProvider::new(ExecutionId::new("eqiora.mpi"), "test", &[]);
    let report = SolveReport::accepted(
        solver_provider,
        execution_provider,
        ExecutionReport::distributed(
            ExecutionId::new("eqiora.mpi"),
            NonZeroUsize::new(3).unwrap(),
        ),
        LinearOperatorOrientation::Normal,
        plan,
        ConvergenceReason::ResidualToleranceSatisfied,
        2,
        1.0,
        1.0e-15,
        1.0e-15,
        1.0e-12,
    )
    .unwrap();
    let summary = ProducerReportSummaryV3::from_report(&report).unwrap();
    let baseline = ProducerReportRecordV3::from_report(&report);
    assert_eq!(baseline.summarize().unwrap(), summary);

    // Summary agreement retains full structured configuration even though the
    // current MPI execution capabilities reject this family before solving.
    let groups = [
        vec![eqiora_solver::AlgebraicBlock::Field(eqiora_core::Id::new())],
        vec![eqiora_solver::AlgebraicBlock::Field(eqiora_core::Id::new())],
    ];
    let structured = |provider: &str, entries| {
        let choice =
            eqiora_solver::FixedFactorizationChoice::new(LinearSolver::SparseLu, provider).unwrap();
        PreconditionerPolicy::UpperDiagonalSchur(
            eqiora_solver::UpperDiagonalSchurConfig::new(
                groups.clone(),
                choice.clone(),
                choice,
                entries,
                64,
            )
            .unwrap(),
        )
    };
    let structured_baseline = ProducerReportRecordV3 {
        preconditioner: structured("eqiora.faer", 32),
        ..baseline.clone()
    };
    for policy in [
        structured("eqiora.faer", 33),
        structured("other.provider", 32),
    ] {
        assert_ne!(
            structured_baseline.summarize().unwrap(),
            ProducerReportRecordV3 {
                preconditioner: policy,
                ..baseline.clone()
            }
            .summarize()
            .unwrap()
        );
    }

    let cases = [
        (
            "solver provider ID",
            ProducerReportRecordV3 {
                solver_provider: SolverProvider::new(
                    BackendId::new("eqiora.other.cg"),
                    baseline.solver_provider.implementation_version(),
                    baseline.solver_provider.libraries(),
                ),
                ..baseline.clone()
            },
        ),
        (
            "solver provider implementation version",
            ProducerReportRecordV3 {
                solver_provider: SolverProvider::new(
                    baseline.solver_provider.id(),
                    "9.9.9",
                    baseline.solver_provider.libraries(),
                ),
                ..baseline.clone()
            },
        ),
        (
            "solver provider libraries",
            ProducerReportRecordV3 {
                solver_provider: SolverProvider::new(
                    baseline.solver_provider.id(),
                    baseline.solver_provider.implementation_version(),
                    CHANGED_LIBRARIES,
                ),
                ..baseline.clone()
            },
        ),
        (
            "execution provider ID",
            ProducerReportRecordV3 {
                execution_provider: ExecutionProvider::new(
                    ExecutionId::new("eqiora.other.mpi"),
                    baseline.execution_provider.implementation_version(),
                    baseline.execution_provider.libraries(),
                ),
                ..baseline.clone()
            },
        ),
        (
            "execution provider implementation version",
            ProducerReportRecordV3 {
                execution_provider: ExecutionProvider::new(
                    baseline.execution_provider.id(),
                    "9.9.9",
                    baseline.execution_provider.libraries(),
                ),
                ..baseline.clone()
            },
        ),
        (
            "execution provider libraries",
            ProducerReportRecordV3 {
                execution_provider: ExecutionProvider::new(
                    baseline.execution_provider.id(),
                    baseline.execution_provider.implementation_version(),
                    CHANGED_LIBRARIES,
                ),
                ..baseline.clone()
            },
        ),
        (
            "verification provider ID",
            ProducerReportRecordV3 {
                verification_provider: ExecutionProvider::new(
                    ExecutionId::new("eqiora.other.verifier"),
                    baseline.verification_provider.implementation_version(),
                    baseline.verification_provider.libraries(),
                ),
                ..baseline.clone()
            },
        ),
        (
            "verification provider implementation version",
            ProducerReportRecordV3 {
                verification_provider: ExecutionProvider::new(
                    baseline.verification_provider.id(),
                    "9.9.9",
                    baseline.verification_provider.libraries(),
                ),
                ..baseline.clone()
            },
        ),
        (
            "verification provider libraries",
            ProducerReportRecordV3 {
                verification_provider: ExecutionProvider::new(
                    baseline.verification_provider.id(),
                    baseline.verification_provider.implementation_version(),
                    CHANGED_LIBRARIES,
                ),
                ..baseline.clone()
            },
        ),
        (
            "backend compatibility projection",
            ProducerReportRecordV3 {
                backend: "eqiora.other.cg",
                ..baseline.clone()
            },
        ),
        (
            "execution adapter compatibility projection",
            ProducerReportRecordV3 {
                execution_adapter: "eqiora.other.mpi",
                ..baseline.clone()
            },
        ),
        (
            "distributed ranks",
            ProducerReportRecordV3 {
                topology: ExecutionTopology::Distributed {
                    ranks: NonZeroUsize::new(4).unwrap(),
                    workers_per_partition: NonZeroUsize::MIN,
                },
                ..baseline.clone()
            },
        ),
        (
            "workers per partition",
            ProducerReportRecordV3 {
                topology: ExecutionTopology::Distributed {
                    ranks: NonZeroUsize::new(3).unwrap(),
                    workers_per_partition: NonZeroUsize::new(2).unwrap(),
                },
                ..baseline.clone()
            },
        ),
        (
            "orientation",
            ProducerReportRecordV3 {
                orientation: LinearOperatorOrientation::Transposed,
                ..baseline.clone()
            },
        ),
        (
            "reason",
            ProducerReportRecordV3 {
                reason: ConvergenceReason::InitialResidualSatisfied,
                ..baseline.clone()
            },
        ),
        (
            "iterations",
            ProducerReportRecordV3 {
                completed_iterations: 3,
                ..baseline.clone()
            },
        ),
        (
            "initial residual bits",
            ProducerReportRecordV3 {
                initial_residual_norm: f64::from_bits(baseline.initial_residual_norm.to_bits() + 1),
                ..baseline.clone()
            },
        ),
        (
            "reported residual bits",
            ProducerReportRecordV3 {
                reported_residual_norm: f64::from_bits(
                    baseline.reported_residual_norm.to_bits() + 1,
                ),
                ..baseline.clone()
            },
        ),
        (
            "true residual bits",
            ProducerReportRecordV3 {
                true_residual_norm: f64::from_bits(baseline.true_residual_norm.to_bits() + 1),
                ..baseline.clone()
            },
        ),
        (
            "target bits",
            ProducerReportRecordV3 {
                residual_target: f64::from_bits(baseline.residual_target.to_bits() + 1),
                ..baseline.clone()
            },
        ),
        (
            "algorithm",
            ProducerReportRecordV3 {
                algorithm: LinearSolver::BiConjugateGradientStabilized,
                ..baseline.clone()
            },
        ),
        (
            "preconditioner",
            ProducerReportRecordV3 {
                preconditioner: PreconditionerPolicy::Identity,
                ..baseline.clone()
            },
        ),
        (
            "reduction",
            ProducerReportRecordV3 {
                reduction: ReductionPolicy::Fast,
                ..baseline.clone()
            },
        ),
        (
            "relative tolerance bits",
            ProducerReportRecordV3 {
                relative_tolerance: f64::from_bits(baseline.relative_tolerance.to_bits() + 1),
                ..baseline.clone()
            },
        ),
        (
            "absolute tolerance bits",
            ProducerReportRecordV3 {
                absolute_tolerance: f64::from_bits(baseline.absolute_tolerance.to_bits() + 1),
                ..baseline.clone()
            },
        ),
        (
            "maximum iterations",
            ProducerReportRecordV3 {
                maximum_iterations: NonZeroUsize::new(11).unwrap(),
                ..baseline.clone()
            },
        ),
    ];
    for (axis, changed) in cases {
        assert_ne!(changed.summarize().unwrap(), summary, "missing {axis} axis");
    }

    let minres = ProducerReportRecordV3 {
        algorithm: LinearSolver::MinimumResidual,
        preconditioner: PreconditionerPolicy::Identity,
        ..baseline.clone()
    }
    .summarize()
    .expect("stable producer-report tag 2 represents MINRES");
    assert_ne!(minres, summary);
}

fn arbitrary_partition() -> Partition {
    Partition::new(
        GlobalVectorSpace::new(NonZeroUsize::new(5).unwrap(), eqiora_core::ScalarType::F64),
        NonZeroUsize::new(3).unwrap(),
        vec![
            PartitionId::new(2),
            PartitionId::new(0),
            PartitionId::new(1),
            PartitionId::new(2),
            PartitionId::new(0),
        ],
    )
    .unwrap()
}
