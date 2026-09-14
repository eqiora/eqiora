#![cfg(feature = "mpi")]

use eqiora_core::ScalarType;
use std::cell::RefCell;
use std::env;
use std::io::Read;
use std::num::NonZeroUsize;
use std::process::{Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use eqiora::Diagnostic;
use eqiora::artifact::{
    DistributedTransportV1, ExecutionProvenanceV1, ExecutionTopologyV1, MpiThreadSupportV1,
};
use eqiora::assembly::{AssemblyBackend, AssemblyPlan, AssemblyResult, AssemblyWork, LinearSystem};
use eqiora::backends::mpi::{
    MPI_ADAPTER_VERSION, MPI_DISTRIBUTED_KRYLOV_BACKEND, MPI_DISTRIBUTED_KRYLOV_SOLVER_PROVIDER,
    MPI_EXECUTION, MPI_EXECUTION_PROVIDER, MPI_RS_VERSION, MpiAdmittedExecutionAdapter,
    MpiExecutionGroup, MpiSpatialAssemblyBackend, MpiThreadSupport,
};
use eqiora::distributed::{DistributedLinearSystem, Partition, PartitionId};
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
    AdmittedExecution, DeploymentBinding, DistributedExecutorDescriptor, ExecutionReceipt,
    ExecutionStepKind, ProcessGroupSlot,
};
use eqiora_numerics::fsi::lower_fixed_reference_fsi_cartesian_2d;
use eqiora_numerics::{
    fsi::ResolvedFixedReferenceFsiSolution2d,
    fsi::finalize_resolved_fixed_reference_fsi_step_2d_with_assembly,
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

const CHILD_ENV: &str = "EQIORA_FIXED_REFERENCE_FSI_SOLVE_MPI_CHILD";
const CHILD_TEST: &str = "fixed_reference_fsi_distributed_solve_mpi_2d_child";
const CHILD_TIMEOUT: Duration = Duration::from_secs(180);
const CHILD_OUTPUT_LIMIT: usize = 64 * 1024;
const CPU_MPI_ABSOLUTE: f64 = 2.0e-10;
const CPU_MPI_RELATIVE: f64 = 2.0e-10;

#[test]
fn fixed_reference_fsi_distributed_solve_mpi_2d_runs_on_one_two_and_four_ranks() {
    if env::var_os(CHILD_ENV).is_some() {
        return;
    }

    for ranks in [1, 2, 4] {
        assert_success(ranks, run_mpi_child(ranks));
    }
}

#[test]
fn fixed_reference_fsi_distributed_solve_mpi_2d_child() {
    if env::var_os(CHILD_ENV).is_none() {
        return;
    }

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
    let reference_execution = execution_context(document.program(), &canonical, &spatial);
    let previous = prestrained_state(&spatial);
    let mesh_sha256 = spatial
        .mesh_artifact
        .digest()
        .expect("authenticated mesh digest")
        .sha256_bytes();
    assert_eq!(mesh_sha256, reference_execution.mesh_reference.sha256());

    let reference_capture =
        CapturingAssemblyBackend::new(&eqiora::assembly::REFERENCE_ASSEMBLY_BACKEND);
    let reference_finalized = finalize_resolved_fixed_reference_fsi_step_2d_with_assembly(
        &canonical,
        &reference_execution.resolved,
        reference_execution.mesh_reference,
        &spatial.mesh,
        &spatial.partition,
        &previous,
        &reference_capture,
    )
    .expect("independent complete CPU reference assembly finalizes");
    let reference_systems = reference_capture
        .take()
        .expect("reference backend exposes both accepted targets")
        .systems()
        .to_vec();
    assert_eq!(reference_systems.len(), 2);
    let reference_solution = reference_finalized
        .solve(&REFERENCE_LINEAR_SOLVER)
        .expect("independent CPU MINRES and FSI acceptance succeed");

    let distributed_resolved = resolve_coupled_fieldwise(
        &CoupledFieldwiseRealizationRequest::explicit(
            reference_execution.resolved.model(),
            reference_execution.resolved.semantic_revision(),
            reference_execution.resolved.realization_revision(),
            reference_execution.resolved.plan().clone(),
        ),
        fixed_reference_fsi_requirements_2d_for_layout(&canonical, VectorLayoutKind::Distributed),
        &distributed_capabilities(group.solver_capabilities()),
    )
    .expect("the exact MPI MINRES capability resolves the distributed FSI plan");

    let layout = mesh_layout(&spatial.mesh, partitions, mesh_sha256);
    let (distributed_finalized, assembly_evidence, candidate_systems) = {
        let backend = MpiSpatialAssemblyBackend::new(&mut group, layout)
            .expect("the authenticated mesh layout matches the physical execution group");
        let capture = CapturingAssemblyBackend::new(&backend);
        let finalized = finalize_resolved_fixed_reference_fsi_step_2d_with_assembly(
            &canonical,
            &distributed_resolved,
            reference_execution.mesh_reference,
            &spatial.mesh,
            &spatial.partition,
            &previous,
            &capture,
        )
        .expect("physical MPI owner-routed FSI assembly finalizes");
        let candidate = capture
            .take()
            .expect("MPI assembly exposes both reconstructed verification targets");
        let evidence = backend
            .accepted_evidence()
            .expect("accepted MPI evidence remains readable")
            .expect("successful physical assembly publishes evidence");
        (finalized, evidence, candidate.systems().to_vec())
    };

    assert_eq!(candidate_systems.len(), reference_systems.len());
    for (candidate, reference) in candidate_systems.iter().zip(&reference_systems) {
        assert_system_bits(candidate, reference);
    }
    assert_assembly_evidence(&assembly_evidence, partitions, mesh_sha256);

    let prepared = distributed_finalized
        .bind_distributed_assembly(&assembly_evidence)
        .expect("accepted reduced/full assembly lineage binds the distributed FSI handoff");
    assert_eq!(prepared.assembly_receipt(), assembly_evidence.receipt());
    assert_eq!(
        prepared.reduced_system_identity(),
        assembly_evidence.system_identities()[0]
    );
    assert_eq!(
        prepared.full_system_identity(),
        assembly_evidence.system_identities()[1]
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
        assert_forged_owner_map_fails_collectively(&world, &mut group, &prepared);
    }

    let binding = distributed_binding(prepared.realization_graph(), &group);
    let expected_admission = prepared
        .distributed_system()
        .admission_fingerprint(prepared.solver_plan())
        .unwrap();
    let admitted = AdmittedExecution::admit_distributed_linear(
        prepared.realization_graph(),
        prepared.distributed_system(),
        prepared.complete_system(),
        binding,
    )
    .expect("generic execution admission accepts only the assembly-derived layout");
    assert_eq!(admitted.distributed_admission(), Some(expected_admission));
    let accepted = group
        .execute_admitted(admitted)
        .expect("MPI MINRES produces one commonly accepted complete result");
    let execution_provenance = observed_execution(&group, accepted.receipt());
    assert_runtime_provenance(&execution_provenance, &group);
    assert_execution_provenance_agreement(&world, &execution_provenance);
    assert_execution_receipt(
        &world,
        accepted.receipt(),
        expected_admission,
        prepared.assembly_receipt().identity().as_bytes(),
        prepared.reduced_system_identity().as_bytes(),
        prepared.full_system_identity().as_bytes(),
        mesh_sha256,
    );
    assert_solve_report(
        accepted.solution().report(),
        prepared.solver_plan(),
        partitions,
    );

    let distributed = prepared
        .finish(accepted)
        .expect("the unchanged FSI finish accepts the MPI result");
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

fn distributed_capabilities(solver: eqiora::solver::SolverCapabilities) -> RealizationCapabilities {
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
        .expect("the MPI group implements the exact FSI solver tuple");
    let solver = SolverCapabilities::exact([selected]).expect("the FSI solver tuple is exact");
    RealizationCapabilities::cartesian_product(
        [DiscretizationMethod::ContinuousGalerkin],
        [(
            MeshKind::ImportedAffineSimplicial,
            SpatialDimensionSupport::exact(NonZeroUsize::new(2).expect("two is nonzero")),
        )],
        [VectorLayoutKind::Distributed],
        solver,
        TargetCapabilities::none().with_host_cpu(NonZeroUsize::MIN),
    )
    .expect("the distributed FSI capability axes are exact and nonempty")
}

fn distributed_binding(
    graph: &eqiora::realization::PortableRealizationGraph,
    group: &MpiExecutionGroup,
) -> DeploymentBinding {
    DeploymentBinding::bind_distributed(
        graph,
        DistributedExecutorDescriptor::new(
            MPI_DISTRIBUTED_KRYLOV_SOLVER_PROVIDER,
            MPI_EXECUTION_PROVIDER,
            ProcessGroupSlot::new(0),
            group.partitions(),
            NonZeroUsize::MIN,
            group.solver_capabilities(),
        ),
    )
    .expect("the process group supplies the exact graph-selected MINRES tuple")
}

fn assert_forged_owner_map_fails_collectively(
    world: &impl CommunicatorCollectives,
    group: &mut MpiExecutionGroup,
    prepared: &eqiora_numerics::fsi::PreparedDistributedFixedReferenceFsiStep2d,
) {
    let exact = prepared.distributed_system().partition();
    let partition = if group.partition().index() == group.partitions().get() - 1 {
        let owners = exact
            .owners()
            .iter()
            .map(|owner| PartitionId::new((owner.index() + 1) % exact.count().get()))
            .collect();
        Partition::new(exact.space(), exact.count(), owners)
            .expect("owner-label rotation preserves one nonempty owner set per partition")
    } else {
        exact.clone()
    };
    let forged = DistributedLinearSystem::from_complete(prepared.complete_system(), partition)
        .expect("the falsifier is locally coherent but not assembly-authoritative");
    let binding = distributed_binding(prepared.realization_graph(), group);
    let admitted = AdmittedExecution::admit_distributed_linear(
        prepared.realization_graph(),
        &forged,
        prepared.complete_system(),
        binding,
    )
    .expect("the rank-local token is structurally valid before collective agreement");
    let error = group
        .execute_admitted(admitted)
        .expect_err("one forged row-owner authority must fail collective admission");
    assert_eq!(error.code(), eqiora::diagnostic::codes::INVALID_REALIZATION);
    assert_common_diagnostic(world, &error);
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
    assert_eq!(evidence.system_identities().len(), 2);
    for (target, (partition, shards)) in evidence
        .target_partitions()
        .iter()
        .zip(evidence.shards())
        .enumerate()
    {
        assert_eq!(partition.partition_count(), partitions);
        assert_eq!(shards.len(), partitions.get());
        let dimension = partition.global_size().get();
        let mut rows = shards
            .iter()
            .enumerate()
            .flat_map(|(partition_index, shard)| {
                assert_eq!(shard.target().index(), target);
                assert_eq!(shard.partition(), PartitionId::new(partition_index));
                shard.rows().iter().map(|row| row.index())
            })
            .collect::<Vec<_>>();
        rows.sort_unstable();
        assert_eq!(rows, (0..dimension).collect::<Vec<_>>());
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
        .expect("the graph-bound MPI run retains its distributed trace");
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
    assert!(
        trace
            .steps()
            .iter()
            .enumerate()
            .all(|(ordinal, step)| step.ordinal() == ordinal)
    );
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
    let local = [
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
    .concat();
    assert_rank_bytes_agree(world, &local);
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
    assert_eq!(
        candidate.realization_revision(),
        reference.realization_revision()
    );
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
    let pressure_scale = field_scale(reference, reference.fields().fluid_pressure());
    let displacement_scale = reference
        .realization_plan()
        .time_step()
        .eliminated_state()
        .state_scale()
        .quantity()
        .value();
    assert_close_vectors(
        "velocity coefficient divided by U",
        reference.vertex_velocity_coefficients(),
        candidate.vertex_velocity_coefficients(),
        velocity_scale,
    );
    assert_close_vectors(
        "fluid bubble velocity coefficient divided by U",
        reference.fluid_velocity_bubble_coefficients(),
        candidate.fluid_velocity_bubble_coefficients(),
        velocity_scale,
    );
    assert_close_slices(
        "fluid pressure coefficient divided by P",
        reference.fluid_pressure_coefficients(),
        candidate.fluid_pressure_coefficients(),
        pressure_scale,
    );
    assert_close_vectors(
        "solid displacement coefficient divided by L",
        reference.solid_displacement_coefficients(),
        candidate.solid_displacement_coefficients(),
        displacement_scale,
    );
}

fn field_scale(
    solution: &ResolvedFixedReferenceFsiSolution2d,
    field: eqiora::Id<eqiora::kinds::Field>,
) -> f64 {
    solution
        .realization_plan()
        .scaling()
        .block_scales()
        .iter()
        .find_map(|entry| {
            (entry.block() == AlgebraicBlock::Field(field))
                .then(|| entry.scale().quantity().value())
        })
        .expect("every represented physical Field has one exact Realization scale")
}

fn assert_close_vectors(label: &str, reference: &[[f64; 2]], candidate: &[[f64; 2]], scale: f64) {
    assert_eq!(candidate.len(), reference.len());
    let reference = reference.iter().flat_map(|value| value.iter().copied());
    let candidate = candidate.iter().flat_map(|value| value.iter().copied());
    for (index, (reference, candidate)) in reference.zip(candidate).enumerate() {
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
    let tolerance = CPU_MPI_ABSOLUTE + CPU_MPI_RELATIVE * reference.abs().max(candidate.abs());
    assert!(
        difference <= tolerance,
        "{label} {index} differs: reference={reference:.17e}, candidate={candidate:.17e}, difference={difference:.3e}, tolerance={tolerance:.3e}"
    );
}

fn observed_execution(
    group: &MpiExecutionGroup,
    receipt: &ExecutionReceipt,
) -> ExecutionProvenanceV1 {
    let raw_library = mpi::environment::library_version()
        .expect("MPI implementation reports a UTF-8 library version");
    let implementation = mpi_implementation(&raw_library);
    let version = normalize_mpi_library_version(&raw_library);
    assert!(!version.is_empty());
    let (standard_major, standard_minor) = mpi::environment::version();
    ExecutionProvenanceV1::from_provider_releases(
        receipt.solver_provider(),
        receipt.execution_provider(),
        ExecutionTopologyV1::Distributed {
            partitions: group.partitions(),
            workers_per_partition: NonZeroUsize::MIN,
            transport: DistributedTransportV1::Mpi {
                implementation: implementation.to_owned(),
                version,
                thread_support: artifact_thread_support(group.thread_support()),
            },
        },
        ReductionPolicy::Reproducible,
        [("mpi-standard", format!("{standard_major}.{standard_minor}"))],
    )
    .unwrap()
}

fn assert_runtime_provenance(execution: &ExecutionProvenanceV1, group: &MpiExecutionGroup) {
    assert_eq!(execution.adapter(), MPI_EXECUTION.as_str());
    assert_eq!(execution.adapter_version(), MPI_ADAPTER_VERSION);
    assert_eq!(
        execution.solver_backend(),
        MPI_DISTRIBUTED_KRYLOV_BACKEND.as_str()
    );
    assert_eq!(execution.solver_backend_version(), MPI_ADAPTER_VERSION);
    assert_eq!(execution.reduction(), ReductionPolicy::Reproducible);
    assert_eq!(
        execution.libraries().get("mpi-rs").map(String::as_str),
        Some(MPI_RS_VERSION)
    );
    assert!(execution.libraries().contains_key("mpi-standard"));
    let ExecutionTopologyV1::Distributed {
        partitions,
        workers_per_partition,
        transport,
    } = execution.topology().unwrap()
    else {
        panic!("observed MPI provenance must retain distributed topology");
    };
    assert_eq!(partitions, group.partitions());
    assert_eq!(workers_per_partition, NonZeroUsize::MIN);
    let DistributedTransportV1::Mpi {
        implementation,
        version,
        thread_support,
    } = transport
    else {
        panic!("observed MPI provenance must retain its physical MPI transport");
    };
    assert!(!implementation.is_empty());
    assert!(!version.is_empty());
    assert_eq!(
        thread_support,
        artifact_thread_support(group.thread_support())
    );
}

fn assert_execution_provenance_agreement(
    world: &impl CommunicatorCollectives,
    execution: &ExecutionProvenanceV1,
) {
    let fingerprint = execution
        .agreement_fingerprint()
        .expect("validated runtime observation has one stable agreement identity");
    assert_rank_bytes_agree(world, &fingerprint.as_bytes());
}

fn normalize_mpi_library_version(value: &str) -> String {
    value
        .split(|character: char| character.is_whitespace() || character.is_control())
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn mpi_implementation(library_version: &str) -> &'static str {
    let lower = library_version.to_ascii_lowercase();
    if lower.contains("open mpi") || lower.contains("open-mpi") {
        "openmpi"
    } else if lower.contains("mpich") {
        "mpich"
    } else {
        "system-mpi"
    }
}

const fn artifact_thread_support(value: MpiThreadSupport) -> MpiThreadSupportV1 {
    match value {
        MpiThreadSupport::Single => MpiThreadSupportV1::Single,
        MpiThreadSupport::Funneled => MpiThreadSupportV1::Funneled,
        MpiThreadSupport::Serialized => MpiThreadSupportV1::Serialized,
        MpiThreadSupport::Multiple => MpiThreadSupportV1::Multiple,
    }
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
                    _ => unreachable!("the registered fixture admits only one/two/four ranks"),
                };
                CellOwnershipClaim::new(MeshEntity::new(2, cell), PartitionId::new(owner))
            })
            .collect(),
    )
    .expect("exact cell ownership derives one complete distributed mesh layout")
}

fn assert_system_bits(candidate: &LinearSystem, reference: &LinearSystem) {
    assert_eq!(candidate.matrix().rows(), reference.matrix().rows());
    assert_eq!(candidate.matrix().columns(), reference.matrix().columns());
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

fn assert_common_diagnostic(world: &impl CommunicatorCollectives, error: &Diagnostic) {
    const DIAGNOSTIC_BYTES: usize = 256;
    let rendered = format!("{}:{}", error.code(), error.message());
    assert!(rendered.len() <= DIAGNOSTIC_BYTES);
    let ranks = usize::try_from(world.size()).unwrap();
    let mut diagnostics = vec![0_u8; ranks * DIAGNOSTIC_BYTES];
    let mut local = [0_u8; DIAGNOSTIC_BYTES];
    local[..rendered.len()].copy_from_slice(rendered.as_bytes());
    world.all_gather_into(&local[..], &mut diagnostics[..]);
    assert!(
        diagnostics
            .as_chunks::<DIAGNOSTIC_BYTES>()
            .0
            .iter()
            .map(|diagnostic| diagnostic.as_slice())
            .all(|diagnostic| diagnostic == local)
    );
}

fn assert_rank_bytes_agree(world: &impl CommunicatorCollectives, local: &[u8]) {
    let ranks = usize::try_from(world.size()).expect("MPI size is a nonnegative usize");
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

fn run_mpi_child(ranks: usize) -> ChildOutput {
    let executable = env::current_exe().expect("integration-test executable is available");
    let launcher = env::var_os("EQIORA_MPI_LAUNCHER").unwrap_or_else(|| "mpirun".into());
    let mut command = Command::new(&launcher);
    if launcher_accepts_oversubscribe(&launcher) {
        command.arg("--oversubscribe");
    }
    let mut child = command
        .args(["-n", &ranks.to_string()])
        .arg(executable)
        .args(["--exact", CHILD_TEST, "--nocapture"])
        .env(CHILD_ENV, "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("registered MPI evidence requires mpirun on PATH");
    let stdout = child.stdout.take().expect("MPI child stdout is captured");
    let stderr = child.stderr.take().expect("MPI child stderr is captured");
    let stdout_reader = thread::spawn(move || drain_bounded(stdout, CHILD_OUTPUT_LIMIT));
    let stderr_reader = thread::spawn(move || drain_bounded(stderr, CHILD_OUTPUT_LIMIT));
    let started = Instant::now();
    let (status, timed_out) = loop {
        if let Some(status) = child.try_wait().expect("MPI child status is readable") {
            break (status, false);
        }
        if started.elapsed() >= CHILD_TIMEOUT {
            child.kill().expect("timed-out MPI launcher can be killed");
            break (
                child.wait().expect("the killed MPI launcher is reaped"),
                true,
            );
        }
        thread::sleep(Duration::from_millis(20));
    };
    let stdout = stdout_reader
        .join()
        .expect("MPI stdout reader does not panic")
        .expect("MPI child stdout remains readable");
    let stderr = stderr_reader
        .join()
        .expect("MPI stderr reader does not panic")
        .expect("MPI child stderr remains readable");
    if timed_out {
        panic!(
            "{ranks}-rank fixed-reference FSI solve MPI child exceeded {CHILD_TIMEOUT:?}\nstdout{}:\n{}\nstderr{}:\n{}",
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
        "{ranks}-rank fixed-reference FSI solve MPI child failed\nstdout{}:\n{}\nstderr{}:\n{}",
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
