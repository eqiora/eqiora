#![cfg(feature = "mpi-cuda")]

use eqiora_core::ScalarType;
use std::cell::RefCell;
use std::collections::HashSet;
use std::env;
use std::io::Read;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::process::{Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use eqiora::Diagnostic;
use eqiora::assembly::{AssemblyBackend, AssemblyPlan, AssemblyResult, AssemblyWork, LinearSystem};
use eqiora::backends::cuda::CudaRuntime;
use eqiora::backends::mpi::{
    MPI_DISTRIBUTED_KRYLOV_BACKEND, MPI_DISTRIBUTED_KRYLOV_SOLVER_PROVIDER, MPI_EXECUTION,
    MPI_EXECUTION_PROVIDER, MpiAdmittedExecutionAdapter, MpiExecutionGroup,
    MpiSpatialAssemblyBackend, MpiThreadSupport,
};
use eqiora::backends::mpi_cuda::{
    MPI_CUDA_LOCAL_ACTION_EXECUTION_PROVIDER, MpiCudaAdmittedExecutionAdapter,
    MpiCudaLinearExecutionEvidence,
};
use eqiora::device::{QueueSlot, SparseActionPolicy, TransferDirection};
use eqiora::meshing::MeshEntity;
use eqiora::realization::{
    CoupledFieldwiseRealizationRequest, DiscretizationMethod, MeshKind, RealizationCapabilities,
    SpatialDimensionSupport, TargetCapabilities, VectorLayoutKind, resolve_coupled_fieldwise,
};
use eqiora::solver::{
    AlgebraicBlock, ExecutionReport, LinearOperatorOrientation, LinearOperatorProperties,
    LinearSolver, PreconditionerPolicy, REFERENCE_LINEAR_SOLVER, ReductionPolicy,
    SolverCapabilities, SolverCapability,
};
use eqiora_execution::{
    AdmittedExecution, CudaPartitionPlacement, DeploymentBinding, DistributedDeviceTransport,
    DistributedExecutorDescriptor, ExecutionReceipt, ExecutionStepKind, ProcessGroupSlot,
};
use eqiora_numerics::{
    fsi::FixedReferenceFsiScaleProfile2d, fsi::ResolvedFixedReferenceFsiSolution2d,
    fsi::lower_fixed_reference_fsi_cartesian_2d,
};
use eqiora_numerics::{
    fsi::finalize_resolved_fixed_reference_fsi_step_2d_with_assembly,
    fsi::fixed_reference_fsi_distributed_cuda_plan_2d,
    fsi::fixed_reference_fsi_requirements_2d_for_layout,
};
use eqiora_spatial_distribution::{
    CellOwnershipClaim, DistributedAssemblyEvidence, DistributedMeshLayout, MeshRevisionIdentityV1,
};
use mpi::Threading;
use mpi::traits::CommunicatorCollectives;
use support::fixed_reference_fsi::{
    direct_document, execution_context, prestrained_state, spatial_context,
};

mod support;

const CHILD_ENV: &str = "EQIORA_FIXED_REFERENCE_FSI_MPI_CUDA_CHILD";
const CHILD_TEST: &str = "fixed_reference_fsi_distributed_cuda_solve_mpi_2d_child";
const SELECTORS_ENV: &str = "EQIORA_MPI_CUDA_DEVICE_SELECTORS";
const CHILD_TIMEOUT: Duration = Duration::from_secs(300);
const CHILD_OUTPUT_LIMIT: usize = 64 * 1024;
const CPU_MPI_CUDA_ABSOLUTE: f64 = 2.0e-10;
const CPU_MPI_CUDA_RELATIVE: f64 = 2.0e-10;

#[test]
#[ignore = "requires an explicitly selected physical MPI-CUDA topology"]
fn fixed_reference_fsi_distributed_cuda_solve_mpi_2d_runs_on_one_two_and_four_ranks() {
    if env::var_os(CHILD_ENV).is_some() {
        return;
    }
    let selectors = required_device_selectors();
    for ranks in [1, 2, 4] {
        assert_success(ranks, run_mpi_child(ranks, &selectors));
    }
}

#[test]
fn fixed_reference_fsi_distributed_cuda_solve_mpi_2d_child() {
    if env::var_os(CHILD_ENV).is_none() {
        return;
    }

    assert_eq!(
        env::var("EQIORA_CUDA_DEVICE").as_deref(),
        Ok("0"),
        "the verification launcher fixes the common Realization-local ordinal"
    );
    let visible = CudaRuntime
        .observe()
        .expect("the isolated CUDA runtime is available");
    assert_eq!(visible.len(), 1, "each MPI process sees exactly one device");
    let local_device = visible[0].descriptor().clone();
    assert_eq!(local_device.id().ordinal(), 0);

    let (universe, provided) = mpi::initialize_with_threading(Threading::Funneled)
        .expect("the child application initializes MPI exactly once");
    let world = universe.world();
    let mut group =
        MpiExecutionGroup::duplicate(&world, provided, MpiThreadSupport::Funneled).unwrap();
    let partitions = group.partitions();
    assert!(matches!(partitions.get(), 1 | 2 | 4));

    let document = direct_document();
    let canonical = lower_fixed_reference_fsi_cartesian_2d(document.program())
        .expect("fixed-reference FSI semantics lower");
    let spatial = spatial_context(document.program(), &canonical);
    let host = execution_context(document.program(), &canonical, &spatial);
    let previous = prestrained_state(document.program(), &spatial, &host);
    let mesh_sha256 = spatial
        .mesh_artifact
        .digest()
        .expect("authenticated mesh digest")
        .sha256_bytes();
    assert_eq!(mesh_sha256, host.mesh_reference.sha256());

    let reference_capture =
        CapturingAssemblyBackend::new(&eqiora::assembly::REFERENCE_ASSEMBLY_BACKEND);
    let reference_finalized = finalize_resolved_fixed_reference_fsi_step_2d_with_assembly(
        &canonical,
        &host.resolved,
        host.mesh_reference,
        &spatial.mesh,
        &spatial.partition,
        &previous,
        &reference_capture,
    )
    .expect("complete CPU reference assembly finalizes");
    let reference_systems = reference_capture
        .take()
        .expect("reference assembly exposes both targets")
        .systems()
        .to_vec();
    let reference_operator = reference_finalized.linear_system().agreement_fingerprint();
    let reference_solution = reference_finalized
        .solve(&REFERENCE_LINEAR_SOLVER)
        .expect("independent CPU MINRES and FSI acceptance succeed");

    let distributed_resolved = resolve_distributed_cuda(&canonical, &host, &group);
    let layout = mesh_layout(&spatial.mesh, partitions, mesh_sha256);
    let (distributed_finalized, assembly_evidence, candidate_systems) = {
        let backend = MpiSpatialAssemblyBackend::new(&mut group, layout)
            .expect("the mesh layout matches the physical execution group");
        let capture = CapturingAssemblyBackend::new(&backend);
        let finalized = finalize_resolved_fixed_reference_fsi_step_2d_with_assembly(
            &canonical,
            &distributed_resolved,
            host.mesh_reference,
            &spatial.mesh,
            &spatial.partition,
            &previous,
            &capture,
        )
        .expect("physical owner-routed distributed CUDA FSI assembly finalizes");
        let candidate = capture
            .take()
            .expect("MPI assembly exposes both reconstructed targets");
        let evidence = backend
            .accepted_evidence()
            .expect("accepted MPI evidence remains readable")
            .expect("successful assembly publishes evidence");
        (finalized, evidence, candidate.systems().to_vec())
    };

    assert_eq!(candidate_systems.len(), reference_systems.len());
    for (candidate, reference) in candidate_systems.iter().zip(&reference_systems) {
        assert_system_bits(candidate, reference);
    }
    assert_assembly_evidence(&assembly_evidence, partitions, mesh_sha256);

    let prepared = distributed_finalized
        .bind_distributed_assembly(&assembly_evidence)
        .expect("accepted reduced/full lineage binds the FSI handoff");
    assert_eq!(
        prepared.complete_system().agreement_fingerprint(),
        reference_operator,
        "CPU, MPI, and rank-local CUDA consume one finalized operator"
    );
    assert!(
        prepared
            .distributed_system()
            .matches_complete(prepared.complete_system())
    );
    assert_eq!(
        prepared.distributed_system().partition().owners(),
        assembly_evidence.target_partitions()[0].owners()
    );
    if partitions.get() > 1 {
        assert!(
            !prepared
                .distributed_system()
                .operator()
                .halo()
                .exchanges()
                .is_empty()
        );
    }

    let fallback_token = AdmittedExecution::admit_distributed_cuda_linear(
        prepared.realization_graph(),
        prepared.distributed_system(),
        prepared.complete_system(),
        distributed_cuda_binding(prepared.realization_graph(), &group, local_device.clone()),
    )
    .expect("the composite fallback falsifier admits the exact same system");
    let fallback = group
        .execute_admitted(fallback_token)
        .expect_err("ordinary MPI execution must not replace the delegated CUDA action");
    assert_eq!(
        fallback.code(),
        eqiora::diagnostic::codes::INVALID_REALIZATION
    );
    assert!(fallback.message().contains("Admission"));

    let binding = distributed_cuda_binding(prepared.realization_graph(), &group, local_device);
    let expected_admission = prepared
        .distributed_system()
        .admission_fingerprint(prepared.solver_plan())
        .unwrap();
    let admitted = AdmittedExecution::admit_distributed_cuda_linear(
        prepared.realization_graph(),
        prepared.distributed_system(),
        prepared.complete_system(),
        binding,
    )
    .expect("the exact assembly-derived distributed CUDA system admits");
    assert_eq!(admitted.distributed_admission(), Some(expected_admission));

    let executed = group
        .execute_admitted_mpi_cuda(admitted)
        .expect("host-staged MPI plus CUDA MINRES produces one accepted result");
    assert_composite_evidence(&world, executed.evidence(), partitions);
    assert_execution_receipt(
        &world,
        executed.accepted().receipt(),
        expected_admission,
        prepared.assembly_receipt().identity().as_bytes(),
        prepared.reduced_system_identity().as_bytes(),
        prepared.full_system_identity().as_bytes(),
        mesh_sha256,
    );
    assert_solve_report(
        executed.accepted().solution().report(),
        prepared.solver_plan(),
        partitions,
    );

    let (accepted, _rank_local_evidence) = executed.into_parts();
    let distributed = prepared
        .finish(accepted)
        .expect("the unchanged FSI finish accepts the MPI-CUDA result");
    assert_eq!(distributed.assembly_receipt(), assembly_evidence.receipt());
    assert_eq!(
        distributed.reduced_system_identity(),
        assembly_evidence.system_identities()[0]
    );
    assert_eq!(
        distributed.full_system_identity(),
        assembly_evidence.system_identities()[1]
    );
    assert_physics_acceptance(distributed.solution());
    assert_normalized_solution_conformance(&reference_solution, distributed.solution());

    world.barrier();
    drop(group);
}

fn resolve_distributed_cuda(
    canonical: &eqiora_numerics::fsi::FixedReferenceFsiCartesianModel2d,
    host: &support::fixed_reference_fsi::ExecutionContext,
    group: &MpiExecutionGroup,
) -> eqiora::realization::ResolvedCoupledFieldwiseRealization {
    let host_plan = host.resolved.plan();
    let velocity = canonical
        .fluid()
        .velocity()
        .downcast::<eqiora::kinds::Field>()
        .expect("fluid velocity is a Field");
    let pressure = canonical
        .fluid()
        .pressure()
        .downcast::<eqiora::kinds::Field>()
        .expect("fluid pressure is a Field");
    let plan = fixed_reference_fsi_distributed_cuda_plan_2d(
        canonical,
        host.mesh_reference,
        host_plan.time_step().duration(),
        FixedReferenceFsiScaleProfile2d::new(
            host_plan.spatial().coordinate_length_scale().quantity(),
            plan_field_scale(host_plan, velocity),
            plan_field_scale(host_plan, pressure),
        )
        .expect("the host Realization owns coherent FSI scales"),
        host_plan.solver(),
        0,
    )
    .expect("the exact distributed CUDA FSI plan is valid");
    resolve_coupled_fieldwise(
        &CoupledFieldwiseRealizationRequest::explicit(
            host.resolved.model(),
            host.resolved.semantic_revision(),
            host.resolved.realization_revision(),
            plan,
        ),
        fixed_reference_fsi_requirements_2d_for_layout(canonical, VectorLayoutKind::Distributed),
        &distributed_cuda_capabilities(group.solver_capabilities()),
    )
    .expect("the exact MPI MINRES and CUDA placement resolve")
}

fn distributed_cuda_capabilities(
    solver: eqiora::solver::SolverCapabilities,
) -> RealizationCapabilities {
    let selected = SolverCapability {
        algorithm: LinearSolver::MinimumResidual,
        operator_properties: LinearOperatorProperties::SymmetricIndefinite,
        preconditioner: PreconditionerPolicy::Identity,
        reduction: ReductionPolicy::Reproducible,
        scalar_type: ScalarType::F64,
    };
    solver
        .require_problem(
            eqiora::solver::SolverPlan::new(selected.algorithm, 1.0e-8, 1.0e-12, NonZeroUsize::MIN)
                .expect("the admission probe is valid"),
            selected.scalar_type,
            selected.operator_properties,
        )
        .expect("the MPI+CUDA group implements the exact FSI solver tuple");
    let solver = SolverCapabilities::exact([selected]).expect("the FSI solver tuple is exact");
    RealizationCapabilities::cartesian_product(
        [DiscretizationMethod::ContinuousGalerkin],
        [(
            MeshKind::ImportedAffineSimplicial,
            SpatialDimensionSupport::exact(NonZeroUsize::new(2).unwrap()),
        )],
        [VectorLayoutKind::Distributed],
        solver,
        TargetCapabilities::none().with_cuda_device(0),
    )
    .expect("the distributed CUDA capability axes are exact and nonempty")
}

fn distributed_cuda_binding(
    graph: &eqiora::realization::PortableRealizationGraph,
    group: &MpiExecutionGroup,
    device: eqiora::device::DeviceDescriptor,
) -> DeploymentBinding {
    let queue = QueueSlot::new(device.id(), 0);
    DeploymentBinding::bind_distributed_cuda(
        graph,
        DistributedExecutorDescriptor::new(
            MPI_DISTRIBUTED_KRYLOV_SOLVER_PROVIDER,
            MPI_EXECUTION_PROVIDER,
            ProcessGroupSlot::new(0),
            group.partitions(),
            NonZeroUsize::MIN,
            group.solver_capabilities(),
        ),
        CudaPartitionPlacement::new(
            MPI_CUDA_LOCAL_ACTION_EXECUTION_PROVIDER,
            device,
            queue,
            SparseActionPolicy::Deterministic,
        )
        .expect("the local queue belongs to the selected device"),
        DistributedDeviceTransport::HostStaged,
    )
    .expect("the process group and local CUDA action satisfy the exact graph")
}

fn assert_composite_evidence(
    world: &impl CommunicatorCollectives,
    evidence: &MpiCudaLinearExecutionEvidence,
    partitions: NonZeroUsize,
) {
    assert_eq!(evidence.transport(), DistributedDeviceTransport::HostStaged);
    assert_eq!(evidence.topology().devices().len(), partitions.get());
    let mut identities = HashSet::new();
    for device in evidence.topology().devices() {
        assert_eq!(device.ordinal(), 0);
        assert_ne!(device.physical_identity(), [0; 16]);
        assert!(identities.insert(device.physical_identity()));
    }
    assert_eq!(identities.len(), partitions.get());
    assert_rank_bytes_agree(world, &evidence.topology().fingerprint());
    assert_rank_bytes_agree(world, &evidence.common_summary());

    let setup = evidence.setup();
    assert_eq!(setup.policy(), SparseActionPolicy::Deterministic);
    assert_eq!(setup.device().id().ordinal(), 0);
    assert_eq!(
        setup.physical_uuid().as_bytes(),
        evidence.topology().devices()[evidence.partition().index()].physical_identity()
    );
    assert!(setup.rows() > 0);
    assert!(setup.columns() >= setup.rows());
    assert!(setup.nonzeros() >= setup.rows());
    assert!(setup.known_payload_bytes() > 0);
    for transfer in [
        setup.row_offsets().plan().direction(),
        setup.column_indices().plan().direction(),
        setup.values().plan().direction(),
    ] {
        assert_eq!(transfer, TransferDirection::HostToDevice);
    }
    let matrix_ready = setup.matrix_ready().completion();
    for completion in [
        setup.row_offsets().completion(),
        setup.column_indices().completion(),
        setup.values().completion(),
    ] {
        assert!(completion.happens_before(matrix_ready).unwrap());
    }

    assert!(!evidence.actions().is_empty());
    let input_buffer = evidence.actions()[0].input_generation().buffer();
    let output_buffer = evidence.actions()[0].output_generation().buffer();
    assert_ne!(input_buffer, output_buffer);
    for (index, action) in evidence.actions().iter().copied().enumerate() {
        let generation = u64::try_from(index + 1).unwrap();
        assert_eq!(action.ordinal().get(), generation);
        assert_eq!(action.input_generation().generation().get(), generation);
        assert_eq!(action.output_generation().generation().get(), generation);
        assert_eq!(action.input_generation().buffer(), input_buffer);
        assert_eq!(action.output_generation().buffer(), output_buffer);
        assert_eq!(
            action.input().plan().direction(),
            TransferDirection::HostToDevice
        );
        assert_eq!(
            action.output().plan().direction(),
            TransferDirection::DeviceToHost
        );
        assert!(
            action
                .input()
                .completion()
                .happens_before(action.input_ready().completion())
                .unwrap()
        );
        assert!(
            action
                .input_ready()
                .completion()
                .happens_before(action.action_completion())
                .unwrap()
        );
        assert!(
            action
                .action_completion()
                .happens_before(action.action_visible().completion())
                .unwrap()
        );
        assert!(
            action
                .action_visible()
                .completion()
                .happens_before(action.output().completion())
                .unwrap()
        );
        assert!(
            action
                .output()
                .completion()
                .happens_before(action.output_visible().completion())
                .unwrap()
        );
    }
}

fn assert_execution_receipt(
    world: &impl CommunicatorCollectives,
    receipt: &ExecutionReceipt,
    admission: eqiora::distributed::DistributedAdmissionFingerprintV1,
    assembly: [u8; 32],
    reduced: [u8; 32],
    full: [u8; 32],
    mesh: [u8; 32],
) {
    let trace = receipt
        .distributed_trace()
        .expect("the composition retains its distributed trace");
    assert_eq!(trace.system(), receipt.operator());
    assert_eq!(trace.admission(), admission);
    assert_eq!(trace.owner_gather_dimension(), receipt.dimension());
    assert_eq!(
        trace.partitions().get(),
        usize::try_from(world.size()).unwrap()
    );
    assert_eq!(trace.workers_per_partition(), NonZeroUsize::MIN);
    assert!(!trace.steps().is_empty());
    assert!(trace.steps().len() <= trace.trace_capacity());
    assert_eq!(
        receipt.dag().steps(),
        &[
            ExecutionStepKind::AgreeDistributedAdmission,
            ExecutionStepKind::SolveDistributedKrylov,
            ExecutionStepKind::AgreeDistributedProducerReport,
            ExecutionStepKind::GatherDistributedOwnedCandidate,
            ExecutionStepKind::AcceptWithNativeHostVerification,
            ExecutionStepKind::AgreeDistributedAcceptedResult,
            ExecutionStepKind::ReplayTrueResidualOnHost,
            ExecutionStepKind::AgreeDistributedReceipt,
            ExecutionStepKind::AcceptHostComplete,
        ]
    );
    assert_rank_bytes_agree(
        world,
        &[
            assembly,
            reduced,
            full,
            mesh,
            receipt.operator().as_bytes(),
            receipt.output().as_bytes(),
            trace.partition().as_bytes(),
            trace.layout().as_bytes(),
            trace.admission().as_bytes(),
        ]
        .concat(),
    );
}

fn assert_solve_report(
    report: &eqiora::solver::SolveReport,
    plan: eqiora::solver::SolverPlan,
    partitions: NonZeroUsize,
) {
    assert_eq!(report.backend(), MPI_DISTRIBUTED_KRYLOV_BACKEND);
    assert_eq!(
        report.execution(),
        ExecutionReport::distributed(MPI_EXECUTION, partitions)
    );
    assert_eq!(report.verification(), ExecutionReport::host_serial());
    assert_eq!(report.orientation(), LinearOperatorOrientation::Normal);
    assert_eq!(report.solver_plan(), plan);
    assert_eq!(report.algorithm(), LinearSolver::MinimumResidual);
    assert_eq!(report.preconditioner(), PreconditionerPolicy::Identity);
    assert_eq!(report.reduction(), ReductionPolicy::Reproducible);
}

fn assert_assembly_evidence(
    evidence: &DistributedAssemblyEvidence,
    partitions: NonZeroUsize,
    mesh_sha256: [u8; 32],
) {
    let receipt = evidence.receipt();
    assert_eq!(receipt.mesh_revision().as_bytes(), mesh_sha256);
    assert_eq!(receipt.packet_count(), 8);
    assert_eq!(receipt.target_count(), 2);
    assert_eq!(receipt.partition_count(), partitions);
    assert_eq!(evidence.target_partitions().len(), 2);
    assert_eq!(evidence.shards().len(), 2);
    for (target, (partition, shards)) in evidence
        .target_partitions()
        .iter()
        .zip(evidence.shards())
        .enumerate()
    {
        assert_eq!(partition.partition_count(), partitions);
        let mut rows = shards
            .iter()
            .enumerate()
            .flat_map(|(partition_index, shard)| {
                assert_eq!(shard.target().index(), target);
                assert_eq!(shard.partition().index(), partition_index);
                shard.rows().iter().map(|row| row.index())
            })
            .collect::<Vec<_>>();
        rows.sort_unstable();
        assert_eq!(rows, (0..partition.global_size().get()).collect::<Vec<_>>());
    }
}

fn assert_physics_acceptance(solution: &ResolvedFixedReferenceFsiSolution2d) {
    let numerical = solution.numerical_evidence();
    assert!(numerical.residual_norm() < 1.0e-9);
    assert!(numerical.continuity_residual_norm() < 1.0e-9);
    assert!(numerical.kinematic_residual_norm() < 1.0e-14);
    assert_eq!(numerical.interface_velocity_jump_norm(), 0.0);
    assert!(numerical.interface_action_imbalance_norm() < 1.0e-9);
    assert!(numerical.energy_balance().defect().abs() < 1.0e-9);
}

fn assert_normalized_solution_conformance(
    reference: &ResolvedFixedReferenceFsiSolution2d,
    candidate: &ResolvedFixedReferenceFsiSolution2d,
) {
    assert_eq!(candidate.model(), reference.model());
    assert_eq!(candidate.semantic_revision(), reference.semantic_revision());
    assert_eq!(candidate.mesh_artifact(), reference.mesh_artifact());
    assert_eq!(candidate.fields(), reference.fields());
    assert_eq!(
        candidate.fluid_velocity_vertices(),
        reference.fluid_velocity_vertices()
    );
    assert_eq!(
        candidate.fluid_velocity_cells(),
        reference.fluid_velocity_cells()
    );
    assert_eq!(
        candidate.fluid_pressure_vertices(),
        reference.fluid_pressure_vertices()
    );
    assert_eq!(
        candidate.solid_velocity_vertices(),
        reference.solid_velocity_vertices()
    );
    assert_eq!(candidate.solid_cells(), reference.solid_cells());
    assert_eq!(candidate.interface_facets(), reference.interface_facets());

    assert_close_slices(
        "dimensionless algebraic coefficient",
        reference.numerical_evidence().algebraic_values(),
        candidate.numerical_evidence().algebraic_values(),
        1.0,
    );
    let velocity_scale = field_scale(reference, reference.fields().fluid_velocity());
    assert_eq!(
        velocity_scale,
        field_scale(reference, reference.fields().solid_velocity())
    );
    assert_close_vectors(
        "velocity/U",
        reference.vertex_velocity_coefficients(),
        candidate.vertex_velocity_coefficients(),
        velocity_scale,
    );
    assert_close_vectors(
        "fluid bubble velocity/U",
        &reference
            .fluid_velocity_bubble_coefficients()
            .values()
            .copied()
            .collect::<Vec<_>>(),
        &candidate
            .fluid_velocity_bubble_coefficients()
            .values()
            .copied()
            .collect::<Vec<_>>(),
        velocity_scale,
    );
    assert_close_slices(
        "pressure/P",
        reference.fluid_pressure_coefficients(),
        candidate.fluid_pressure_coefficients(),
        field_scale(reference, reference.fields().fluid_pressure()),
    );
    assert_close_vectors(
        "displacement/L",
        reference.solid_displacement_coefficients(),
        candidate.solid_displacement_coefficients(),
        reference
            .realization_plan()
            .time_step()
            .eliminated_states()
            .iter()
            .find(|state| state.pair().state() == reference.fields().solid_displacement())
            .unwrap()
            .state_scale()
            .quantity()
            .value(),
    );
}

fn plan_field_scale(
    plan: &eqiora::realization::CoupledFieldwiseRealizationPlan,
    field: eqiora::Id<eqiora::kinds::Field>,
) -> eqiora::DynQuantity {
    plan.scaling()
        .block_scales()
        .iter()
        .find_map(|entry| {
            (entry.block() == AlgebraicBlock::Field(field)).then(|| entry.scale().quantity())
        })
        .expect("every represented Field has one scale")
}

fn field_scale(
    solution: &ResolvedFixedReferenceFsiSolution2d,
    field: eqiora::Id<eqiora::kinds::Field>,
) -> f64 {
    plan_field_scale(solution.realization_plan(), field).value()
}

fn assert_close_vectors(label: &str, reference: &[[f64; 2]], candidate: &[[f64; 2]], scale: f64) {
    assert_eq!(candidate.len(), reference.len());
    for (index, (reference, candidate)) in reference
        .iter()
        .flatten()
        .zip(candidate.iter().flatten())
        .enumerate()
    {
        assert_close(label, index, reference / scale, candidate / scale);
    }
}

fn assert_close_slices(label: &str, reference: &[f64], candidate: &[f64], scale: f64) {
    assert_eq!(candidate.len(), reference.len());
    for (index, (&reference, &candidate)) in reference.iter().zip(candidate).enumerate() {
        assert_close(label, index, reference / scale, candidate / scale);
    }
}

fn assert_close(label: &str, index: usize, reference: f64, candidate: f64) {
    assert!(reference.is_finite() && candidate.is_finite());
    let difference = (candidate - reference).abs();
    let tolerance =
        CPU_MPI_CUDA_ABSOLUTE + CPU_MPI_CUDA_RELATIVE * reference.abs().max(candidate.abs());
    assert!(
        difference <= tolerance,
        "{label} {index} differs: reference={reference:.17e}, candidate={candidate:.17e}, difference={difference:.3e}, tolerance={tolerance:.3e}"
    );
}

fn mesh_layout(
    mesh: &eqiora::meshing::SimplicialMesh,
    partitions: NonZeroUsize,
    mesh_sha256: [u8; 32],
) -> DistributedMeshLayout {
    DistributedMeshLayout::derive(
        MeshRevisionIdentityV1::from_sha256(mesh_sha256),
        mesh,
        partitions,
        (0..8)
            .map(|cell| {
                let owner = match partitions.get() {
                    1 => 0,
                    2 => usize::from(cell < 4),
                    4 => cell % 4,
                    _ => unreachable!("the fixture admits one/two/four ranks"),
                };
                CellOwnershipClaim::new(
                    MeshEntity::new(2, cell),
                    eqiora::distributed::PartitionId::new(owner),
                )
            })
            .collect(),
    )
    .expect("cell ownership derives one complete mesh layout")
}

fn assert_system_bits(candidate: &LinearSystem, reference: &LinearSystem) {
    assert_eq!(
        candidate.matrix().row_offsets(),
        reference.matrix().row_offsets()
    );
    assert_eq!(
        candidate.matrix().column_indices(),
        reference.matrix().column_indices()
    );
    assert_eq!(
        candidate
            .matrix()
            .values()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        reference
            .matrix()
            .values()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        candidate
            .rhs()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        reference
            .rhs()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}

fn assert_rank_bytes_agree(world: &impl CommunicatorCollectives, local: &[u8]) {
    let ranks = usize::try_from(world.size()).expect("MPI size fits usize");
    let mut gathered = vec![0_u8; ranks * local.len()];
    world.all_gather_into(local, &mut gathered[..]);
    assert!(
        gathered
            .chunks_exact(local.len())
            .all(|candidate| candidate == local)
    );
}

#[derive(Debug)]
struct CapturingAssemblyBackend<'a> {
    inner: &'a dyn AssemblyBackend,
    accepted: RefCell<Option<AssemblyResult>>,
}

impl<'a> CapturingAssemblyBackend<'a> {
    fn new(inner: &'a dyn AssemblyBackend) -> Self {
        Self {
            inner,
            accepted: RefCell::new(None),
        }
    }

    fn take(&self) -> Option<AssemblyResult> {
        self.accepted.borrow_mut().take()
    }
}

impl AssemblyBackend for CapturingAssemblyBackend<'_> {
    fn assemble(
        &self,
        plan: &AssemblyPlan,
        work: &dyn AssemblyWork,
    ) -> Result<AssemblyResult, Diagnostic> {
        let result = self.inner.assemble(plan, work)?;
        *self.accepted.borrow_mut() = Some(result.clone());
        Ok(result)
    }
}

struct ChildOutput {
    status: ExitStatus,
    stdout: BoundedOutput,
    stderr: BoundedOutput,
}

struct BoundedOutput {
    bytes: Vec<u8>,
    truncated: bool,
}

fn required_device_selectors() -> String {
    let value = env::var(SELECTORS_ENV)
        .unwrap_or_else(|_| panic!("registered evidence requires {SELECTORS_ENV}"));
    let selectors = value.split(',').collect::<Vec<_>>();
    assert_eq!(
        selectors.len(),
        4,
        "{SELECTORS_ENV} must contain exactly four entries"
    );
    assert!(selectors.iter().all(|selector| !selector.is_empty()));
    assert_eq!(
        selectors.iter().copied().collect::<HashSet<_>>().len(),
        selectors.len(),
        "{SELECTORS_ENV} selectors must be distinct"
    );
    value
}

fn run_mpi_child(ranks: usize, selectors: &str) -> ChildOutput {
    let executable = env::current_exe().expect("integration-test executable is available");
    let launcher = env::var_os("EQIORA_MPI_LAUNCHER").unwrap_or_else(|| "mpirun".into());
    let wrapper = case_root().join("launch-rank.sh");
    let mut command = Command::new(&launcher);
    if launcher_accepts_oversubscribe(&launcher) {
        command.arg("--oversubscribe");
    }
    let mut child = command
        .args(["-n", &ranks.to_string()])
        .arg(wrapper)
        .arg(executable)
        .args(["--exact", CHILD_TEST, "--nocapture"])
        .env(CHILD_ENV, "1")
        .env(SELECTORS_ENV, selectors)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("registered MPI-CUDA evidence requires mpirun on PATH");
    let stdout = child.stdout.take().expect("child stdout is captured");
    let stderr = child.stderr.take().expect("child stderr is captured");
    let stdout_reader = thread::spawn(move || drain_bounded(stdout, CHILD_OUTPUT_LIMIT));
    let stderr_reader = thread::spawn(move || drain_bounded(stderr, CHILD_OUTPUT_LIMIT));
    let started = Instant::now();
    let (status, timed_out) = loop {
        if let Some(status) = child.try_wait().expect("child status is readable") {
            break (status, false);
        }
        if started.elapsed() >= CHILD_TIMEOUT {
            child.kill().expect("timed-out MPI launcher can be killed");
            break (child.wait().expect("killed MPI launcher is reaped"), true);
        }
        thread::sleep(Duration::from_millis(20));
    };
    let stdout = stdout_reader
        .join()
        .unwrap()
        .expect("stdout remains readable");
    let stderr = stderr_reader
        .join()
        .unwrap()
        .expect("stderr remains readable");
    if timed_out {
        panic!(
            "{ranks}-rank MPI-CUDA child exceeded {CHILD_TIMEOUT:?}\nstdout{}:\n{}\nstderr{}:\n{}",
            truncation_marker(&stdout),
            String::from_utf8_lossy(&stdout.bytes),
            truncation_marker(&stderr),
            String::from_utf8_lossy(&stderr.bytes),
        );
    }
    ChildOutput {
        status,
        stdout,
        stderr,
    }
}

fn case_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("verify/fsi/fixed-reference-distributed-cuda-solve-mpi-2d")
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

fn assert_success(ranks: usize, output: ChildOutput) {
    assert!(
        output.status.success(),
        "{ranks}-rank MPI-CUDA child failed\nstdout{}:\n{}\nstderr{}:\n{}",
        truncation_marker(&output.stdout),
        String::from_utf8_lossy(&output.stdout.bytes),
        truncation_marker(&output.stderr),
        String::from_utf8_lossy(&output.stderr.bytes),
    );
}

fn drain_bounded(mut reader: impl Read, maximum: usize) -> std::io::Result<BoundedOutput> {
    let mut bytes = Vec::with_capacity(maximum);
    let mut truncated = false;
    let mut chunk = [0_u8; 8 * 1024];
    loop {
        let count = reader.read(&mut chunk)?;
        if count == 0 {
            break;
        }
        let retained = count.min(maximum.saturating_sub(bytes.len()));
        bytes.extend_from_slice(&chunk[..retained]);
        truncated |= retained != count;
    }
    Ok(BoundedOutput { bytes, truncated })
}

fn truncation_marker(output: &BoundedOutput) -> &'static str {
    if output.truncated { " (truncated)" } else { "" }
}
