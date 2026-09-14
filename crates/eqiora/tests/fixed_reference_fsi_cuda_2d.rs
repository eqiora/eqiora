#![cfg(feature = "cuda")]

use eqiora_core::ScalarType;
use std::num::{NonZeroU64, NonZeroUsize};

use eqiora::backends::cuda::{
    CUDA_LINEAR_EXECUTION, CUDA_LINEAR_EXECUTION_PROVIDER, CUDA_LINEAR_SOLVER_BACKEND,
    CUDA_LINEAR_SOLVER_PROVIDER, CUDA_RUNTIME_ID, CudaLinearSolver, CudaRuntime,
};
use eqiora::device::{DeviceDescriptor, DeviceId, QueueSlot};
use eqiora::realization::{
    CoupledFieldwiseRealizationPlan, CoupledFieldwiseRealizationRequest, DiscretizationMethod,
    MeshKind, RealizationCapabilities, RealizationRevision, ResolvedCoupledFieldwiseRealization,
    SemanticRevision, SpatialDimensionSupport, TargetCapabilities, VectorLayoutKind,
    resolve_coupled_fieldwise,
};
use eqiora::solver::{
    AlgebraicBlock, ExecutionReport, LinearOperatorOrientation, LinearOperatorProperties,
    LinearSolver, LinearSolverBackend, PreconditionerPolicy, REFERENCE_LINEAR_SOLVER,
    ReductionPolicy, SolverCapabilities, SolverCapability, SolverPlan,
};
use eqiora_backend_cuda::CudaAdmittedExecutionAdapter;
use eqiora_execution::{
    CUDA_LINEAR_DEVICE_CAPABILITIES, CudaExecutorDescriptor, DeploymentBinding, ExecutionStepKind,
};
use eqiora_numerics::{
    fsi::FixedReferenceFsiCartesianModel2d, fsi::FixedReferenceFsiScaleProfile2d,
    fsi::ResolvedFixedReferenceFsiSolution2d, fsi::finalize_resolved_fixed_reference_fsi_step_2d,
    fsi::lower_fixed_reference_fsi_cartesian_2d,
};
use eqiora_numerics::{
    fsi::fixed_reference_fsi_cuda_plan_2d, fsi::fixed_reference_fsi_requirements_2d,
};
use support::fixed_reference_fsi::{
    direct_document, execution_context, prestrained_state, spatial_context,
};

mod support;

const CPU_CUDA_ABSOLUTE: f64 = 2.0e-10;
const CPU_CUDA_RELATIVE: f64 = 2.0e-10;

#[test]
fn cpu_and_cuda_realizations_finalize_one_exact_fsi_operator() {
    let document = direct_document();
    let canonical = lower_fixed_reference_fsi_cartesian_2d(document.program())
        .expect("fixed-reference FSI semantics lower");
    let spatial = spatial_context(document.program(), &canonical);
    let host = execution_context(document.program(), &canonical, &spatial);
    let previous = prestrained_state(&spatial);

    let host_finalized = finalize_resolved_fixed_reference_fsi_step_2d(
        &canonical,
        &host.resolved,
        host.mesh_reference,
        &spatial.mesh,
        &spatial.partition,
        &previous,
    )
    .expect("the host reference Realization finalizes");
    let cuda_resolved = resolve_cuda(&canonical, &host, 0);
    let cuda_finalized = finalize_resolved_fixed_reference_fsi_step_2d(
        &canonical,
        &cuda_resolved,
        host.mesh_reference,
        &spatial.mesh,
        &spatial.partition,
        &previous,
    )
    .expect("the one-device Realization finalizes");

    assert_eq!(
        cuda_finalized.linear_system().agreement_fingerprint(),
        host_finalized.linear_system().agreement_fingerprint(),
        "placement and reduction must not select a second FSI operator"
    );

    let admitted = cuda_finalized
        .admit_cuda(cuda_binding(
            cuda_finalized.realization_graph(),
            synthetic_cuda_device(0),
        ))
        .expect("allocation-free admission accepts the exact device ordinal");
    assert_eq!(
        admitted.system().agreement_fingerprint(),
        host_finalized.linear_system().agreement_fingerprint()
    );
    assert_eq!(admitted.solver_plan(), cuda_finalized.solver_plan());
    assert!(admitted.minimum_device_payload_bytes().is_some());

    let error = cuda_finalized
        .admit_cuda(cuda_binding(
            cuda_finalized.realization_graph(),
            synthetic_cuda_device(1),
        ))
        .expect_err("a deployment-local device substitution must fail before runtime work");
    assert_eq!(error.code(), eqiora::diagnostic::codes::INVALID_REALIZATION);
    assert!(error.message().contains("device differs"));
}

#[test]
#[ignore = "requires EQIORA_CUDA_DEVICE and an explicitly selected physical CUDA device"]
fn fixed_reference_fsi_runs_through_the_exact_cuda_execution_handoff() {
    let device_ordinal = std::env::var("EQIORA_CUDA_DEVICE")
        .expect("set EQIORA_CUDA_DEVICE for the explicit hardware gate")
        .parse::<u16>()
        .expect("EQIORA_CUDA_DEVICE must be a u16 ordinal");
    let device = CudaRuntime
        .discover()
        .expect("the selected CUDA runtime loads")
        .into_iter()
        .find(|candidate| candidate.id().ordinal() == device_ordinal)
        .expect("the selected CUDA device is visible");
    CudaLinearSolver::new(device_ordinal)
        .admit_device(&device)
        .expect("the selected device has the exact CSR Krylov capabilities");

    let document = direct_document();
    let canonical = lower_fixed_reference_fsi_cartesian_2d(document.program())
        .expect("fixed-reference FSI semantics lower");
    let spatial = spatial_context(document.program(), &canonical);
    let host = execution_context(document.program(), &canonical, &spatial);
    let previous = prestrained_state(&spatial);

    let host_finalized = finalize_resolved_fixed_reference_fsi_step_2d(
        &canonical,
        &host.resolved,
        host.mesh_reference,
        &spatial.mesh,
        &spatial.partition,
        &previous,
    )
    .expect("the independent host reference operator finalizes");
    let host_operator = host_finalized.linear_system().agreement_fingerprint();
    let host_values = REFERENCE_LINEAR_SOLVER
        .solve(
            &host_finalized
                .linear_system()
                .linear_problem()
                .expect("the finalized host CSR is a valid linear problem"),
            host_finalized.solver_plan(),
        )
        .expect("the independent CPU MINRES oracle converges");
    let host_solution = host_finalized
        .finish(host_values)
        .expect("the sole FSI finish accepts the CPU oracle");

    let cuda_resolved = resolve_cuda(&canonical, &host, device_ordinal);
    let cuda_finalized = finalize_resolved_fixed_reference_fsi_step_2d(
        &canonical,
        &cuda_resolved,
        host.mesh_reference,
        &spatial.mesh,
        &spatial.partition,
        &previous,
    )
    .expect("the exact CUDA Realization finalizes the same FSI operator");
    let operator = cuda_finalized.linear_system().agreement_fingerprint();
    assert_eq!(
        operator, host_operator,
        "the CPU oracle and CUDA handoff must consume one finalized CSR/RHS"
    );
    let cuda_plan = cuda_finalized.solver_plan();
    let dimension = cuda_finalized.linear_system().columns();
    let binding = cuda_binding(cuda_finalized.realization_graph(), device.clone());
    let admitted = cuda_finalized
        .admit_cuda(binding)
        .expect("the finalized FSI subject admits the selected device");
    let minimum_device_payload = admitted
        .minimum_device_payload_bytes()
        .expect("CUDA admission retains its known resident-payload lower bound");
    let executed = CudaLinearSolver::new(device_ordinal)
        .execute_admitted(admitted)
        .expect("physical CUDA MINRES produces an independently accepted host result");
    let (accepted, cuda_evidence) = executed.into_parts();
    assert_cuda_receipt(
        accepted.receipt(),
        operator,
        cuda_plan,
        dimension,
        minimum_device_payload,
        &cuda_evidence,
    );
    let expected_receipt = accepted.receipt().clone();
    let (cuda_solution, receipt) = cuda_finalized
        .finish_cuda(accepted)
        .expect("the unchanged FSI finish accepts the CUDA result and retains its receipt");

    assert_eq!(receipt, expected_receipt);
    assert_physics_acceptance(&cuda_solution);
    assert_normalized_solution_conformance(&host_solution, &cuda_solution);
}

fn resolve_cuda(
    canonical: &FixedReferenceFsiCartesianModel2d,
    host: &support::fixed_reference_fsi::ExecutionContext,
    device: u16,
) -> ResolvedCoupledFieldwiseRealization {
    let host_plan = host.resolved.plan();
    let velocity = canonical
        .fluid()
        .velocity()
        .downcast::<eqiora::kinds::Field>()
        .expect("canonical fluid velocity is a Field");
    let pressure = canonical
        .fluid()
        .pressure()
        .downcast::<eqiora::kinds::Field>()
        .expect("canonical fluid pressure is a Field");
    let plan = fixed_reference_fsi_cuda_plan_2d(
        canonical,
        host.mesh_reference,
        host_plan.time_step().duration(),
        FixedReferenceFsiScaleProfile2d::new(
            host_plan.spatial().coordinate_length_scale().quantity(),
            plan_field_scale(host_plan, velocity),
            plan_field_scale(host_plan, pressure),
        )
        .expect("the host Realization owns positive coherent-SI FSI scales"),
        host_plan.solver().with_reduction(ReductionPolicy::Fast),
        device,
    )
    .expect("the exact one-device FSI plan is valid");
    let solver_plan = plan.solver();
    resolve_coupled_fieldwise(
        &CoupledFieldwiseRealizationRequest::explicit(
            host.resolved.model(),
            SemanticRevision::new(canonical.semantic_revision()),
            RealizationRevision::new(1),
            plan,
        ),
        fixed_reference_fsi_requirements_2d(canonical),
        &cuda_capabilities(device, solver_plan),
    )
    .expect("the exact one-device MINRES capability resolves the FSI plan")
}

fn cuda_capabilities(device: u16, plan: SolverPlan) -> RealizationCapabilities {
    CudaLinearSolver::capabilities()
        .require_problem(
            plan,
            ScalarType::F64,
            LinearOperatorProperties::SymmetricIndefinite,
        )
        .expect("the CUDA provider implements the exact selected FSI solver tuple");
    let solver = SolverCapabilities::exact([SolverCapability {
        algorithm: plan.algorithm(),
        operator_properties: LinearOperatorProperties::SymmetricIndefinite,
        preconditioner: plan.preconditioner(),
        reduction: plan.reduction(),
        scalar_type: ScalarType::F64,
    }])
    .expect("the CUDA FSI solver tuple is exact");
    RealizationCapabilities::cartesian_product(
        [DiscretizationMethod::ContinuousGalerkin],
        [(
            MeshKind::ImportedAffineSimplicial,
            SpatialDimensionSupport::exact(NonZeroUsize::new(2).expect("two is non-zero")),
        )],
        [VectorLayoutKind::Replicated],
        solver,
        TargetCapabilities::none().with_cuda_device(device),
    )
    .expect("the exact CUDA FSI capability axes are nonempty")
}

fn plan_field_scale(
    plan: &CoupledFieldwiseRealizationPlan,
    field: eqiora::Id<eqiora::kinds::Field>,
) -> eqiora::DynQuantity {
    plan.scaling()
        .block_scales()
        .iter()
        .find_map(|entry| {
            (entry.block() == AlgebraicBlock::Field(field)).then(|| entry.scale().quantity())
        })
        .expect("the host Realization scales every represented physical Field exactly once")
}

fn synthetic_cuda_device(ordinal: u16) -> DeviceDescriptor {
    DeviceDescriptor::new(
        DeviceId::new(CUDA_RUNTIME_ID, ordinal),
        format!("synthetic CUDA device {ordinal}"),
        NonZeroU64::new(1 << 30).expect("one GiB is non-zero"),
        CUDA_LINEAR_DEVICE_CAPABILITIES,
    )
    .expect("the allocation-free device snapshot is complete")
}

fn cuda_binding(
    graph: &eqiora::realization::PortableRealizationGraph,
    device: DeviceDescriptor,
) -> DeploymentBinding {
    DeploymentBinding::bind_cuda(
        graph,
        CudaExecutorDescriptor::new(
            CUDA_LINEAR_SOLVER_PROVIDER,
            CUDA_LINEAR_EXECUTION_PROVIDER,
            device.clone(),
            QueueSlot::new(device.id(), 0),
            CudaLinearSolver::capabilities(),
        )
        .expect("the logical queue belongs to the selected device"),
    )
    .expect("the selected executor supplies the graph's exact CUDA solver tuple")
}

fn assert_cuda_receipt(
    receipt: &eqiora_execution::ExecutionReceipt,
    operator: eqiora::solver::CanonicalCsrAgreementFingerprintV1,
    plan: SolverPlan,
    dimension: usize,
    minimum_device_payload: usize,
    evidence: &eqiora::backends::cuda::CudaLinearSolveEvidence,
) {
    assert_eq!(receipt.operator(), operator);
    assert_eq!(receipt.solver_plan(), plan);
    assert_eq!(receipt.dimension(), dimension);
    assert_eq!(
        receipt.minimum_device_payload_bytes(),
        Some(minimum_device_payload)
    );
    assert_eq!(receipt.report().backend(), CUDA_LINEAR_SOLVER_BACKEND);
    assert_eq!(
        receipt.report().execution(),
        ExecutionReport::cuda(CUDA_LINEAR_EXECUTION, evidence.device().id().ordinal())
    );
    assert_eq!(
        receipt.report().verification(),
        ExecutionReport::host_serial()
    );
    assert_eq!(
        receipt.acceptance_verification(),
        ExecutionReport::host_serial()
    );
    assert_eq!(
        receipt.report().orientation(),
        LinearOperatorOrientation::Normal
    );
    assert_eq!(receipt.report().algorithm(), LinearSolver::MinimumResidual);
    assert_eq!(
        receipt.report().preconditioner(),
        PreconditionerPolicy::Identity
    );
    assert_eq!(receipt.report().reduction(), ReductionPolicy::Fast);
    assert_eq!(
        receipt.dag().steps(),
        &[
            ExecutionStepKind::TransferInputsToCuda,
            ExecutionStepKind::AwaitCudaInputsReady,
            ExecutionStepKind::SolveOnCuda,
            ExecutionStepKind::AwaitCudaSolveCompletion,
            ExecutionStepKind::TransferCandidateToHost,
            ExecutionStepKind::AwaitHostVisibility,
            ExecutionStepKind::AcceptWithNativeHostVerification,
            ExecutionStepKind::ReplayTrueResidualOnHost,
            ExecutionStepKind::AcceptHostComplete,
        ]
    );

    let trace = receipt
        .cuda_trace()
        .expect("the accepted device execution retains its typed trace");
    assert_eq!(trace.inputs_ready(), evidence.inputs_ready());
    assert_eq!(trace.solve_visible(), evidence.solve_visible());
    assert_eq!(trace.solution_visible(), evidence.solution_visible());
    assert_eq!(
        trace.external_sparse_workspace_bytes(),
        evidence.workspace_bytes()
    );
    assert!(
        trace
            .inputs_ready()
            .completion()
            .happens_before(trace.solve_visible().completion())
            .expect("the trace uses one ordered queue")
    );
    assert!(
        trace
            .solve_visible()
            .completion()
            .happens_before(trace.transfers().complete_solution().completion())
            .expect("the trace uses one ordered queue")
    );
    assert!(
        trace
            .transfers()
            .complete_solution()
            .completion()
            .happens_before(trace.solution_visible().completion())
            .expect("the trace uses one ordered queue")
    );
    assert!(trace.transfers().inverse_diagonal().is_none());
    assert_eq!(
        trace.initial_solution().buffer(),
        trace.solved_solution().buffer()
    );
    assert_eq!(
        trace.solved_solution().generation().get(),
        trace.initial_solution().generation().get() + 1
    );
    assert_eq!(trace.downloaded_solution(), trace.solved_solution());
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
    assert_eq!(
        candidate.solid_displacement_vertices(),
        reference.solid_displacement_vertices()
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
        field_scale(reference, reference.fields().fluid_pressure()),
    );
    assert_close_vectors(
        "solid displacement coefficient divided by L",
        reference.solid_displacement_coefficients(),
        candidate.solid_displacement_coefficients(),
        reference
            .realization_plan()
            .time_step()
            .eliminated_state()
            .state_scale()
            .quantity()
            .value(),
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
    let tolerance = CPU_CUDA_ABSOLUTE + CPU_CUDA_RELATIVE * reference.abs().max(candidate.abs());
    assert!(
        difference <= tolerance,
        "{label} {index} differs: reference={reference:.17e}, candidate={candidate:.17e}, difference={difference:.3e}, tolerance={tolerance:.3e}"
    );
}
