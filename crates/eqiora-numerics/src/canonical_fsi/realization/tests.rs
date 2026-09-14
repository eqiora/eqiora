use std::num::{NonZeroU16, NonZeroUsize};

use eqiora_compiler::compile;
use eqiora_core::{Diagnostic, DynQuantity, ScalarType, diagnostic::codes};
use eqiora_distributed::PartitionId;
use eqiora_execution::{
    AcceptedLinearExecution, AdmittedExecution, DeploymentBinding, ExecutionReceipt,
};
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_meshing::{CellId, FacetId, MeshEntity, MeshQualityGate, MeshTopology};
use eqiora_realization::{
    CoupledFieldwiseRealizationPlan, CoupledFieldwiseRealizationRequest,
    CoupledFieldwiseSpatialDiscretization, Discretization, DiscretizationMethod,
    MeshArtifactReference, MeshKind, MeshPolicy, PlacementRequirementNode, QuadraturePolicy,
    RealizationCapabilities, RealizationRevision, SemanticRevision, SolveRoot, SpaceFamily,
    SpatialDimensionSupport, Target, TargetCapabilities, VectorLayoutKind,
    resolve_coupled_fieldwise,
};
use eqiora_sem::KernelProgram;
use eqiora_solver::{
    LinearOperatorProperties, LinearSolver, PreconditionerPolicy, REFERENCE_LINEAR_SOLVER,
    ReductionPolicy, SolverCapabilities, SolverCapability, SolverPlan,
};
use eqiora_spatial_distribution::{
    CellOwnershipClaim, DistributedAssemblyEvidence, DistributedMeshLayout,
    LoopbackSpatialAssemblyBackend, MeshRevisionIdentityV1,
};

use super::*;

const SOURCE: &str =
    include_str!("../../../../../verify/fsi/fixed-reference-monolithic-step-2d/models/direct.eqi");

#[test]
fn requirements_keep_semantics_while_vector_layout_is_explicit() {
    let program = compile_program(SOURCE);
    let model = super::super::lower_fixed_reference_fsi_cartesian_2d(&program)
        .expect("canonical FSI meaning");
    let replicated = fixed_reference_fsi_requirements_2d(&model);
    let distributed =
        fixed_reference_fsi_requirements_2d_for_layout(&model, VectorLayoutKind::Distributed);

    assert_eq!(
        replicated.execution().vector_layout(),
        VectorLayoutKind::Replicated
    );
    assert_eq!(
        distributed.execution().vector_layout(),
        VectorLayoutKind::Distributed
    );
    assert_eq!(replicated.domains(), distributed.domains());
    assert_eq!(replicated.trace_quotients(), distributed.trace_quotients());
    assert_eq!(
        replicated.eliminated_state(),
        distributed.eliminated_state()
    );
    assert_eq!(
        replicated.execution().spatial_dimension(),
        distributed.execution().spatial_dimension()
    );
    assert_eq!(
        replicated.execution().scalar_type(),
        distributed.execution().scalar_type()
    );
}

#[test]
fn plan_is_the_exact_gauge_free_monolithic_selection() {
    let program = compile_program(SOURCE);
    let model = super::super::lower_fixed_reference_fsi_cartesian_2d(&program)
        .expect("canonical FSI meaning");
    let plan = fixed_reference_fsi_plan_2d(
        &model,
        mesh_reference(),
        DynQuantity::new(0.1, TIME),
        scales(),
        reference_solver(),
    )
    .expect("exact FSI plan");

    assert_eq!(
        plan.operator_properties(),
        LinearOperatorProperties::SymmetricIndefinite
    );
    assert_eq!(plan.spatial().domains().len(), 2);
    assert!(
        plan.spatial()
            .domains()
            .iter()
            .all(|domain| domain.constraints().is_empty())
    );
    let families = plan
        .spatial()
        .domains()
        .iter()
        .flat_map(|domain| domain.field_spaces())
        .map(|binding| binding.space().family())
        .collect::<Vec<_>>();
    assert!(families.contains(&SpaceFamily::SimplexP1Bubble));
    assert_eq!(
        families
            .iter()
            .filter(|family| {
                **family
                    == (SpaceFamily::ContinuousLagrange {
                        order: NonZeroU16::MIN,
                    })
            })
            .count(),
        2
    );
    assert_eq!(
        plan.time_step().eliminated_state().pair(),
        state_pair(&model)
    );
    assert_eq!(
        plan.scaling().weak_functional_scale().quantity(),
        scales().weak_functional()
    );
    assert_eq!(
        fixed_reference_fsi_requirements_2d(&model).trace_quotients(),
        plan.spatial().trace_quotients()
    );
}

#[test]
fn cuda_plan_changes_only_the_exact_execution_tuple() {
    let fixture = Fixture::new(SOURCE);
    let host = fixed_reference_fsi_plan_2d(
        &fixture.model,
        mesh_reference(),
        DynQuantity::new(0.1, TIME),
        scales(),
        reference_solver(),
    )
    .expect("exact host FSI plan");
    let cuda = fixed_reference_fsi_cuda_plan_2d(
        &fixture.model,
        mesh_reference(),
        DynQuantity::new(0.1, TIME),
        scales(),
        fast_solver(),
        3,
    )
    .expect("exact CUDA FSI plan");

    assert_eq!(cuda.spatial(), host.spatial());
    assert_eq!(cuda.time_step(), host.time_step());
    assert_eq!(cuda.scaling(), host.scaling());
    assert_eq!(cuda.operator_properties(), host.operator_properties());
    assert_eq!(cuda.target(), Target::CudaGpu { device: 3 });
    assert_eq!(cuda.solver().algorithm(), LinearSolver::MinimumResidual);
    assert_eq!(
        cuda.solver().preconditioner(),
        PreconditionerPolicy::Identity
    );
    assert_eq!(cuda.solver().reduction(), ReductionPolicy::Fast);

    let host_finalized = finalize_resolved_fixed_reference_fsi_step_2d(
        &fixture.model,
        &fixture.resolve(host),
        mesh_reference(),
        &fixture.mesh,
        &fixture.partition,
        &fixture.previous,
    )
    .expect("exact host plan finalizes the reference FSI operator");
    let finalized = finalize_resolved_fixed_reference_fsi_step_2d(
        &fixture.model,
        &fixture.resolve(cuda),
        mesh_reference(),
        &fixture.mesh,
        &fixture.partition,
        &fixture.previous,
    )
    .expect("exact CUDA plan finalizes the unchanged FSI operator");
    assert_eq!(
        finalized.linear_system().properties(),
        LinearOperatorProperties::SymmetricIndefinite
    );
    assert_eq!(
        finalized.linear_system().agreement_fingerprint(),
        host_finalized.linear_system().agreement_fingerprint(),
        "execution placement and reduction must not create a second FSI operator"
    );
    assert_eq!(
        finalized.realization_plan().target(),
        Target::CudaGpu { device: 3 }
    );
    let graph = finalized.realization_graph();
    let SolveRoot::Linear(root) = graph.root() else {
        panic!("FSI root must be linear");
    };
    let placement = graph
        .linear_solve(root)
        .and_then(|solve| graph.placement(solve.placement()))
        .expect("linear CUDA placement exists");
    assert_eq!(
        placement,
        PlacementRequirementNode::CudaDevices {
            devices_per_partition: NonZeroUsize::MIN,
        }
    );
}

#[test]
fn host_and_cuda_plans_reject_each_others_reduction_policy() {
    let fixture = Fixture::new(SOURCE);
    let host_error = fixed_reference_fsi_plan_2d(
        &fixture.model,
        mesh_reference(),
        DynQuantity::new(0.1, TIME),
        scales(),
        fast_solver(),
    )
    .expect_err("host execution must not silently accept the CUDA reduction policy");
    assert_eq!(host_error.code(), codes::INVALID_REALIZATION);
    assert!(host_error.message().contains("host execution"));

    let cuda_error = fixed_reference_fsi_cuda_plan_2d(
        &fixture.model,
        mesh_reference(),
        DynQuantity::new(0.1, TIME),
        scales(),
        reference_solver(),
        0,
    )
    .expect_err("CUDA execution must not silently accept the host reduction policy");
    assert_eq!(cuda_error.code(), codes::INVALID_REALIZATION);
    assert!(cuda_error.message().contains("CUDA execution"));

    let distributed_cuda_error = fixed_reference_fsi_distributed_cuda_plan_2d(
        &fixture.model,
        mesh_reference(),
        DynQuantity::new(0.1, TIME),
        scales(),
        fast_solver(),
        0,
    )
    .expect_err("distributed CUDA must retain the MPI parent's reproducible reductions");
    assert_eq!(distributed_cuda_error.code(), codes::INVALID_REALIZATION);
    assert!(
        distributed_cuda_error
            .message()
            .contains("distributed CUDA")
    );
}

#[test]
fn rejects_non_reference_solver_before_a_plan_exists() {
    let program = compile_program(SOURCE);
    let model = super::super::lower_fixed_reference_fsi_cartesian_2d(&program)
        .expect("canonical FSI meaning");
    let solver = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1.0e-11,
        1.0e-13,
        NonZeroUsize::new(20_000).unwrap(),
    )
    .unwrap();
    let error = fixed_reference_fsi_plan_2d(
        &model,
        mesh_reference(),
        DynQuantity::new(0.1, TIME),
        scales(),
        solver,
    )
    .expect_err("a non-MINRES plan must fail closed");
    assert_eq!(error.code(), codes::INVALID_REALIZATION);
}

#[test]
fn finalization_replays_quadrature_and_mesh_identity_exactly() {
    let fixture = Fixture::new(SOURCE);
    let plan = fixed_reference_fsi_plan_2d(
        &fixture.model,
        mesh_reference(),
        DynQuantity::new(0.1, TIME),
        scales(),
        reference_solver(),
    )
    .unwrap();
    let wrong_quadrature = CoupledFieldwiseRealizationPlan::new(
        CoupledFieldwiseSpatialDiscretization::new(
            plan.spatial().coordinate_length_scale(),
            plan.spatial().domains().iter().cloned(),
            plan.spatial().trace_quotients(),
            Discretization::new(
                DiscretizationMethod::ContinuousGalerkin,
                MeshPolicy::ImportedSimplicial {
                    artifact: mesh_reference(),
                },
                QuadraturePolicy::TriangleDuffyGaussLegendre {
                    points_per_axis: NonZeroUsize::new(3).unwrap(),
                },
            ),
        )
        .unwrap(),
        plan.time_step(),
        plan.scaling().clone(),
        plan.operator_properties(),
        plan.solver(),
        plan.target(),
        plan.schedule(),
    )
    .unwrap();
    let wrong_resolved = fixture.resolve(wrong_quadrature);
    assert_eq!(
        finalize_resolved_fixed_reference_fsi_step_2d(
            &fixture.model,
            &wrong_resolved,
            mesh_reference(),
            &fixture.mesh,
            &fixture.partition,
            &fixture.previous,
        )
        .expect_err("degree-four rather than degree-six quadrature must fail closed")
        .code(),
        codes::INVALID_REALIZATION
    );

    let exact = fixture.resolve(plan);
    assert_eq!(
        finalize_resolved_fixed_reference_fsi_step_2d(
            &fixture.model,
            &exact,
            MeshArtifactReference::from_sha256([99; 32]),
            &fixture.mesh,
            &fixture.partition,
            &fixture.previous,
        )
        .expect_err("a stale mesh reference must fail before assembly")
        .code(),
        codes::INVALID_REALIZATION
    );
}

#[test]
fn finalization_accepts_the_complete_exact_plan() {
    let fixture = Fixture::new(SOURCE);
    let plan = fixed_reference_fsi_plan_2d(
        &fixture.model,
        mesh_reference(),
        DynQuantity::new(0.1, TIME),
        scales(),
        reference_solver(),
    )
    .unwrap();
    let finalized = finalize_resolved_fixed_reference_fsi_step_2d(
        &fixture.model,
        &fixture.resolve(plan),
        mesh_reference(),
        &fixture.mesh,
        &fixture.partition,
        &fixture.previous,
    )
    .expect("exact canonical/Realization/mesh bridge finalizes");
    assert_eq!(
        finalized.linear_system().properties(),
        LinearOperatorProperties::SymmetricIndefinite
    );
    assert!(finalized.linear_system().rows() > 0);
    assert_eq!(
        finalized.linear_system().rows(),
        finalized.linear_system().columns()
    );
}

#[test]
fn block_projection_rejects_a_plan_pair_without_its_model_kinematics() {
    let mut fixture = Fixture::new(SOURCE);
    let plan = fixed_reference_fsi_plan_2d(
        &fixture.model,
        mesh_reference(),
        DynQuantity::new(0.1, TIME),
        scales(),
        reference_solver(),
    )
    .unwrap();
    let resolved = fixture.resolve(plan);
    let kinematic = fixture
        .model
        .equation_roles
        .relations
        .values_mut()
        .find(|entry| {
            matches!(
                entry.kind,
                crate::form_compiler::equation_roles::Role::Kinematic { .. }
            )
        })
        .unwrap();
    let crate::form_compiler::equation_roles::Role::Kinematic { state, rate } = kinematic.kind
    else {
        unreachable!()
    };
    kinematic.kind = crate::form_compiler::equation_roles::Role::Kinematic {
        state: rate,
        rate: state,
    };
    let error = super::block::fixed_reference_fsi_block_system(
        &fixture.model,
        &resolved,
        mesh_reference(),
        &fixture.mesh,
        &fixture.partition,
    )
    .unwrap_err();
    assert_eq!(error.message(), "Plan role has no matching Model equation");
}

#[test]
fn finalized_cuda_admission_owns_the_exact_system_subject() {
    let _admit: for<'a> fn(
        &'a FinalizedResolvedFixedReferenceFsiStep2d,
        DeploymentBinding,
    ) -> Result<AdmittedExecution<'a>, Diagnostic> =
        FinalizedResolvedFixedReferenceFsiStep2d::admit_cuda;
}

#[test]
fn cuda_finish_requires_the_opaque_solution_receipt_pair() {
    let _finish: fn(
        FinalizedResolvedFixedReferenceFsiStep2d,
        AcceptedLinearExecution,
    )
        -> Result<(ResolvedFixedReferenceFsiSolution2d, ExecutionReceipt), Diagnostic> =
        FinalizedResolvedFixedReferenceFsiStep2d::finish_cuda;
}

#[test]
fn finalization_admits_distributed_cuda_without_changing_the_operator() {
    let fixture = Fixture::new(SOURCE);
    let plan = fixed_reference_fsi_distributed_cuda_plan_2d(
        &fixture.model,
        mesh_reference(),
        DynQuantity::new(0.1, TIME),
        scales(),
        reference_solver(),
        0,
    )
    .unwrap();
    let resolved = fixture.resolve_for_layout(plan, VectorLayoutKind::Distributed);
    let finalized = finalize_resolved_fixed_reference_fsi_step_2d(
        &fixture.model,
        &resolved,
        mesh_reference(),
        &fixture.mesh,
        &fixture.partition,
        &fixture.previous,
    )
    .expect("distributed CUDA changes execution composition, not FSI finalization");
    assert_eq!(finalized.vector_layout(), VectorLayoutKind::Distributed);
    assert_eq!(
        finalized.realization_plan().target(),
        Target::CudaGpu { device: 0 }
    );
    assert_eq!(
        finalized.solver_plan().reduction(),
        ReductionPolicy::Reproducible
    );

    let rejected = fixed_reference_fsi_distributed_cuda_plan_2d(
        &fixture.model,
        mesh_reference(),
        DynQuantity::new(0.1, TIME),
        scales(),
        reference_solver(),
        1,
    )
    .expect_err("every launcher-isolated MPI rank must resolve local CUDA ordinal zero");
    assert!(rejected.message().contains("ordinal zero"));
}

#[test]
fn finalized_core_inherits_distributed_layout_from_resolved_requirements() {
    let fixture = Fixture::new(SOURCE);
    let plan = fixed_reference_fsi_plan_2d(
        &fixture.model,
        mesh_reference(),
        DynQuantity::new(0.1, TIME),
        scales(),
        reference_solver(),
    )
    .unwrap();
    let resolved = fixture.resolve_for_layout(plan, VectorLayoutKind::Distributed);
    let finalized = finalize_resolved_fixed_reference_fsi_step_2d(
        &fixture.model,
        &resolved,
        mesh_reference(),
        &fixture.mesh,
        &fixture.partition,
        &fixture.previous,
    )
    .expect("distributed ownership changes execution, not canonical FSI finalization");

    assert_eq!(finalized.vector_layout(), VectorLayoutKind::Distributed);
    assert_eq!(
        finalized.realization_graph().systems()[0].partition(),
        VectorLayoutKind::Distributed
    );
}

#[test]
fn distributed_assembly_binding_seals_both_targets_and_exact_owner_layout() {
    let fixture = Fixture::new(SOURCE);
    let (finalized, evidence) =
        finalize_with_loopback(&fixture, VectorLayoutKind::Distributed, &fixture.previous);
    let prepared = finalized
        .bind_distributed_assembly(&evidence)
        .expect("exact reduced and full assembly evidence binds");

    assert_eq!(
        prepared.realization_graph().systems()[0].partition(),
        VectorLayoutKind::Distributed
    );
    assert!(
        prepared
            .distributed_system()
            .matches_complete(prepared.complete_system())
    );
    assert_eq!(
        prepared.distributed_system().partition().count(),
        NonZeroUsize::new(2).unwrap()
    );
    assert_eq!(prepared.assembly_receipt(), evidence.receipt());
    assert_eq!(
        prepared.reduced_system_identity(),
        evidence.system_identities()[0]
    );
    assert_eq!(
        prepared.full_system_identity(),
        evidence.system_identities()[1]
    );
}

#[test]
fn distributed_finish_requires_the_opaque_solution_receipt_pair() {
    let _finish: fn(
        PreparedDistributedFixedReferenceFsiStep2d,
        AcceptedLinearExecution,
    ) -> Result<AcceptedDistributedFixedReferenceFsiStep2d, Diagnostic> =
        PreparedDistributedFixedReferenceFsiStep2d::finish;
}

#[test]
fn distributed_assembly_binding_rejects_replicated_realization() {
    let fixture = Fixture::new(SOURCE);
    let (finalized, evidence) =
        finalize_with_loopback(&fixture, VectorLayoutKind::Replicated, &fixture.previous);
    let error = finalized
        .bind_distributed_assembly(&evidence)
        .expect_err("distributed evidence cannot widen replicated Realization");
    assert_eq!(error.code(), codes::INVALID_REALIZATION);
    assert!(error.message().contains("explicitly distributed"));
}

#[test]
fn distributed_assembly_binding_rejects_foreign_operator_evidence() {
    let fixture = Fixture::new(SOURCE);
    let (finalized, _) =
        finalize_with_loopback(&fixture, VectorLayoutKind::Distributed, &fixture.previous);
    let zero_previous = FixedReferenceFsiState::<2>::new(
        &fixture.mesh,
        &fixture.partition,
        vec![[0.0; 2]; fixture.mesh.vertices().len()],
        fixture
            .partition
            .fluid_cells()
            .iter()
            .copied()
            .map(|cell| (cell, [0.0; 2]))
            .collect(),
        vec![[0.0; 2]; fixture.mesh.vertices().len()],
    )
    .unwrap();
    let (_, foreign_evidence) =
        finalize_with_loopback(&fixture, VectorLayoutKind::Distributed, &zero_previous);

    let error = finalized
        .bind_distributed_assembly(&foreign_evidence)
        .expect_err("same-shape evidence for another finalized RHS must fail closed");
    assert_eq!(error.code(), codes::ASSEMBLY_FAILED);
    assert!(
        error
            .message()
            .contains("differs from the accepted assembly target")
    );
}

#[test]
fn distributed_assembly_binding_rejects_foreign_mesh_revision() {
    let fixture = Fixture::new(SOURCE);
    let (finalized, _) =
        finalize_with_loopback(&fixture, VectorLayoutKind::Distributed, &fixture.previous);
    let mut foreign_sha256 = mesh_reference().sha256();
    foreign_sha256[0] ^= 1;
    let (_, foreign_evidence) = finalize_with_loopback_for_mesh(
        &fixture,
        VectorLayoutKind::Distributed,
        &fixture.previous,
        MeshArtifactReference::from_sha256(foreign_sha256),
    );

    let error = finalized
        .bind_distributed_assembly(&foreign_evidence)
        .expect_err("identical algebra over another authenticated mesh must fail closed");
    assert_eq!(error.code(), codes::INVALID_REALIZATION);
    assert!(error.message().contains("mesh revision"));
}

#[test]
fn accepted_solution_reconstructs_exact_field_ids_and_supports() {
    let fixture = Fixture::new(SOURCE);
    let plan = fixed_reference_fsi_plan_2d(
        &fixture.model,
        mesh_reference(),
        DynQuantity::new(0.1, TIME),
        scales(),
        reference_solver(),
    )
    .unwrap();
    let resolved = fixture.resolve(plan);
    let expected_semantic_revision = resolved.semantic_revision();
    let expected_realization_revision = resolved.realization_revision();
    let solution = finalize_resolved_fixed_reference_fsi_step_2d(
        &fixture.model,
        &resolved,
        mesh_reference(),
        &fixture.mesh,
        &fixture.partition,
        &fixture.previous,
    )
    .unwrap()
    .solve(&REFERENCE_LINEAR_SOLVER)
    .expect("reference MINRES solution passes all FSI acceptance gates");

    assert_eq!(solution.model(), fixture.model.model());
    assert_eq!(solution.semantic_revision(), expected_semantic_revision);
    assert_eq!(
        solution.realization_revision(),
        expected_realization_revision
    );
    assert_eq!(solution.fields(), field_identities(&fixture.model));
    assert_eq!(
        solution.fluid_velocity_cells().len(),
        solution.fluid_velocity_bubble_coefficients().len()
    );
    assert_eq!(
        solution.fluid_pressure_vertices().len(),
        solution.fluid_pressure_coefficients().len()
    );
    let interface_vertex = fixture.partition.interface_vertices()[1];
    assert_eq!(
        solution.fluid_velocity_coefficient(interface_vertex),
        solution.solid_velocity_coefficient(interface_vertex)
    );
    assert!(
        solution
            .solid_displacement_coefficient(interface_vertex)
            .is_some()
    );
}

struct Fixture {
    program: KernelProgram,
    model: FixedReferenceFsiCartesianModel2d,
    mesh: SimplicialMesh,
    partition: FixedReferenceFsiPartition<2>,
    previous: FixedReferenceFsiState<2>,
}

impl Fixture {
    fn new(source: &str) -> Self {
        let program = compile_program(source);
        let model = super::super::lower_fixed_reference_fsi_cartesian_2d(&program)
            .expect("canonical FSI meaning");
        let mesh = physical_mesh();
        let (fluid, solid, interface) = inventories(&mesh);
        let partition =
            FixedReferenceFsiPartition::<2>::new(&mesh, fluid, solid, interface).unwrap();
        let mut displacement = vec![[0.0; 2]; mesh.vertices().len()];
        let free_interface = mesh
            .vertices()
            .iter()
            .position(|point| point.as_slice() == [1.0, 0.5])
            .unwrap();
        displacement[free_interface] = [0.02, 0.0];
        let previous = FixedReferenceFsiState::<2>::new(
            &mesh,
            &partition,
            vec![[0.0; 2]; mesh.vertices().len()],
            partition
                .fluid_cells()
                .iter()
                .copied()
                .map(|cell| (cell, [0.0; 2]))
                .collect(),
            displacement,
        )
        .unwrap();
        Self {
            program,
            model,
            mesh,
            partition,
            previous,
        }
    }

    fn resolve(
        &self,
        plan: CoupledFieldwiseRealizationPlan,
    ) -> ResolvedCoupledFieldwiseRealization {
        self.resolve_for_layout(plan, VectorLayoutKind::Replicated)
    }

    fn resolve_for_layout(
        &self,
        plan: CoupledFieldwiseRealizationPlan,
        vector_layout: VectorLayoutKind,
    ) -> ResolvedCoupledFieldwiseRealization {
        let capabilities = fsi_capabilities(vector_layout, plan.target());
        resolve_coupled_fieldwise(
            &CoupledFieldwiseRealizationRequest::explicit(
                self.program.model(),
                SemanticRevision::new(self.program.revision().0),
                RealizationRevision::new(1),
                plan,
            ),
            fixed_reference_fsi_requirements_2d_for_layout(&self.model, vector_layout),
            &capabilities,
        )
        .expect("coupled reference capability resolves")
    }
}

fn fsi_capabilities(vector_layout: VectorLayoutKind, target: Target) -> RealizationCapabilities {
    let reduction = match (vector_layout, target) {
        (_, Target::HostCpu { .. }) => ReductionPolicy::Reproducible,
        (VectorLayoutKind::Distributed, Target::CudaGpu { .. }) => ReductionPolicy::Reproducible,
        (VectorLayoutKind::Replicated, Target::CudaGpu { .. }) => ReductionPolicy::Fast,
    };
    let solver = fsi_solver_capabilities(reduction);
    let targets = match target {
        Target::HostCpu { threads } => TargetCapabilities::none().with_host_cpu(threads),
        Target::CudaGpu { device } => TargetCapabilities::none().with_cuda_device(device),
    };
    RealizationCapabilities::cartesian_product(
        [DiscretizationMethod::ContinuousGalerkin],
        [(
            MeshKind::ImportedAffineSimplicial,
            SpatialDimensionSupport::exact(NonZeroUsize::new(2).unwrap()),
        )],
        [vector_layout],
        solver,
        targets,
    )
    .expect("the exact FSI capability axes are nonempty")
}

fn fsi_solver_capabilities(reduction: ReductionPolicy) -> SolverCapabilities {
    SolverCapabilities::exact([SolverCapability {
        algorithm: LinearSolver::MinimumResidual,
        operator_properties: LinearOperatorProperties::SymmetricIndefinite,
        preconditioner: PreconditionerPolicy::Identity,
        reduction,
        scalar_type: ScalarType::F64,
    }])
    .expect("the fixed-reference FSI solver tuple is exact")
}

fn finalize_with_loopback(
    fixture: &Fixture,
    vector_layout: VectorLayoutKind,
    previous: &FixedReferenceFsiState<2>,
) -> (
    FinalizedResolvedFixedReferenceFsiStep2d,
    DistributedAssemblyEvidence,
) {
    finalize_with_loopback_for_mesh(fixture, vector_layout, previous, mesh_reference())
}

fn finalize_with_loopback_for_mesh(
    fixture: &Fixture,
    vector_layout: VectorLayoutKind,
    previous: &FixedReferenceFsiState<2>,
    mesh_reference: MeshArtifactReference,
) -> (
    FinalizedResolvedFixedReferenceFsiStep2d,
    DistributedAssemblyEvidence,
) {
    let plan = fixed_reference_fsi_plan_2d(
        &fixture.model,
        mesh_reference,
        DynQuantity::new(0.1, TIME),
        scales(),
        reference_solver(),
    )
    .unwrap();
    let resolved = fixture.resolve_for_layout(plan, vector_layout);
    let partition_count = NonZeroUsize::new(2).unwrap();
    let claims = (0..fixture.mesh.cells().len())
        .map(|cell| {
            CellOwnershipClaim::new(
                MeshEntity::new(2, cell),
                PartitionId::new(cell % partition_count.get()),
            )
        })
        .collect();
    let layout = DistributedMeshLayout::derive(
        MeshRevisionIdentityV1::from_sha256(mesh_reference.sha256()),
        &fixture.mesh,
        partition_count,
        claims,
    )
    .unwrap();
    let backend = LoopbackSpatialAssemblyBackend::new(layout);
    let finalized = finalize_resolved_fixed_reference_fsi_step_2d_with_assembly(
        &fixture.model,
        &resolved,
        mesh_reference,
        &fixture.mesh,
        &fixture.partition,
        previous,
        &backend,
    )
    .expect("loopback distributed assembly finalizes exact FSI targets");
    let evidence = backend
        .accepted_evidence()
        .expect("loopback evidence lock remains healthy")
        .expect("successful assembly publishes evidence");
    (finalized, evidence)
}

fn compile_program(source: &str) -> KernelProgram {
    let mut compiled = compile("fixed-reference-fsi.eqi", source).expect("source compiles");
    let (transaction, model, _) = compiled.remove(0).into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).expect("transaction commits");
    KernelProgram::from_snapshot(&store.snapshot(), model).expect("model validates")
}

fn scales() -> FixedReferenceFsiScaleProfile2d {
    FixedReferenceFsiScaleProfile2d::new(
        DynQuantity::new(2.0, LENGTH),
        DynQuantity::new(0.5, VELOCITY),
        DynQuantity::new(4.0, PRESSURE),
    )
    .unwrap()
}

fn reference_solver() -> SolverPlan {
    SolverPlan::new(
        LinearSolver::MinimumResidual,
        1.0e-11,
        1.0e-13,
        NonZeroUsize::new(20_000).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Identity)
    .with_reduction(ReductionPolicy::Reproducible)
}

fn fast_solver() -> SolverPlan {
    reference_solver().with_reduction(ReductionPolicy::Fast)
}

fn mesh_reference() -> MeshArtifactReference {
    MeshArtifactReference::from_sha256([7; 32])
}

fn physical_mesh() -> SimplicialMesh {
    SimplicialMesh::new(
        2,
        vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 0.5],
            vec![1.0, 0.5],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, 0.0],
            vec![2.0, 0.5],
            vec![2.0, 1.0],
        ],
        vec![
            vec![0, 1, 3],
            vec![0, 3, 2],
            vec![2, 3, 5],
            vec![2, 5, 4],
            vec![1, 6, 7],
            vec![1, 7, 3],
            vec![3, 7, 8],
            vec![3, 8, 5],
        ],
        MeshQualityGate::new(0.05).unwrap(),
    )
    .unwrap()
}

fn inventories(mesh: &SimplicialMesh) -> (Vec<CellId>, Vec<CellId>, Vec<FacetId>) {
    let mut fluid = Vec::new();
    let mut solid = Vec::new();
    for (index, cell) in mesh.cells().iter().enumerate() {
        let centroid_x = cell
            .iter()
            .map(|vertex| mesh.vertices()[*vertex][0])
            .sum::<f64>()
            / 3.0;
        if centroid_x < 1.0 {
            fluid.push(CellId::new(index));
        } else {
            solid.push(CellId::new(index));
        }
    }
    let interface = (0..mesh.entity_count(1).unwrap())
        .filter(|&facet| {
            mesh.entity_vertices(MeshEntity::new(1, facet))
                .unwrap()
                .iter()
                .all(|vertex| mesh.vertices()[vertex.index()][0] == 1.0)
        })
        .map(FacetId::new)
        .collect();
    (fluid, solid, interface)
}

#[path = "tests/inventory.rs"]
mod inventory;
