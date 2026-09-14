use eqiora_core::ScalarType;
use std::num::{NonZeroU16, NonZeroUsize};

use eqiora::compiler::compile;
use eqiora::graph::{GraphStore, InMemoryGraphStore};
use eqiora::realization::{
    AlgebraicBlockScale, DefaultPolicyVersion, Discretization, DiscretizationMethod,
    ExecutionSchedule, FieldSpaceBinding, FieldwiseRealizationPlan, FieldwiseRealizationRequest,
    FieldwiseRealizationRequirements, FieldwiseSpatialDiscretization, MeshArtifactReference,
    MeshKind, MeshPolicy, PlacementRequirementNode, PositivePhysicalScale, QuadraturePolicy,
    RealizationCapabilities, RealizationCapability, RealizationCapabilityContext, RealizationPlan,
    RealizationRequest, RealizationRequirements, RealizationRevision, ScheduleCapability,
    SemanticRevision, SolveRoot, Space, SpatialCapability, SpatialDimensionSupport,
    SymmetricCongruenceScaling, Target, TargetCapability, VectorLayoutKind, default_plan_v0,
    resolve, resolve_fieldwise,
};
use eqiora::sem::KernelProgram;
use eqiora::solver::{
    AlgebraicBlock, ExecutionReport, LinearOperatorProperties, LinearSolver, PreconditionerPolicy,
    REFERENCE_LINEAR_SOLVER, ReductionPolicy, SolveReport, SolverCapabilities, SolverCapability,
    SolverPlan,
};
use eqiora_backend_rayon::{CpuThreadPool, RAYON_EXECUTION};
use eqiora_numerics::{
    scalar::ResolvedScalarEllipticCartesianSolution, scalar::ResolvedScalarEllipticSolution1d,
    scalar::lower_scalar_elliptic_1d, scalar::solve_resolved_scalar_elliptic_1d,
    scalar::solve_resolved_scalar_elliptic_1d_with_assembly,
    scalar::solve_resolved_scalar_elliptic_cartesian,
    scalar::solve_resolved_scalar_elliptic_cartesian_with_assembly,
};

const SOURCE: &str = include_str!("../../../verify/numerics/poisson-fem-fvm/models/poisson.eqi");
const SOURCE_2D: &str =
    include_str!("../../../verify/numerics/cartesian-poisson-fem-fvm/models/poisson.eqi");
const SOURCE_3D: &str =
    include_str!("../../../verify/numerics/cartesian-poisson-3d-fem-fvm/models/poisson.eqi");

#[test]
fn realization_admission_rejects_unverified_axis_recombination() {
    let dimension = NonZeroUsize::new(2).unwrap();
    let common_context = |method, mesh_kind| {
        RealizationCapabilityContext::new(
            SpatialCapability::new(method, mesh_kind, SpatialDimensionSupport::exact(dimension)),
            VectorLayoutKind::Replicated,
            TargetCapability::HostCpu {
                maximum_threads: NonZeroUsize::MIN,
            },
            ScheduleCapability::Offline,
        )
    };
    let fem_solver = SolverCapability {
        algorithm: LinearSolver::ConjugateGradient,
        operator_properties: LinearOperatorProperties::SymmetricPositiveDefinite,
        preconditioner: PreconditionerPolicy::Identity,
        reduction: ReductionPolicy::Reproducible,
        scalar_type: ScalarType::F64,
    };
    let fvm_solver = SolverCapability {
        algorithm: LinearSolver::BiConjugateGradientStabilized,
        operator_properties: LinearOperatorProperties::General,
        preconditioner: PreconditionerPolicy::Jacobi,
        reduction: ReductionPolicy::Reproducible,
        scalar_type: ScalarType::F64,
    };
    let capabilities = RealizationCapabilities::exact([
        RealizationCapability::new(
            common_context(
                DiscretizationMethod::ContinuousGalerkin,
                MeshKind::ImportedAffineSimplicial,
            ),
            fem_solver,
        )
        .unwrap(),
        RealizationCapability::new(
            common_context(
                DiscretizationMethod::CellCenteredFiniteVolume,
                MeshKind::GeneratedCartesian,
            ),
            fvm_solver,
        )
        .unwrap(),
    ])
    .unwrap();
    let domain = eqiora::Id::new();
    let field = eqiora::Id::new();
    let requirements = FieldwiseRealizationRequirements::new(
        domain,
        [field],
        RealizationRequirements::new(dimension, ScalarType::F64, VectorLayoutKind::Replicated),
    )
    .unwrap();

    let fem = fieldwise_request(domain, field, FieldwisePath::Fem, fem_solver);
    let fvm = fieldwise_request(domain, field, FieldwisePath::Fvm, fvm_solver);
    assert!(
        resolve_fieldwise(&fem, requirements.clone(), &capabilities).is_ok(),
        "the explicitly admitted FEM path must resolve"
    );
    assert!(
        resolve_fieldwise(&fvm, requirements.clone(), &capabilities).is_ok(),
        "the explicitly admitted FVM path must resolve"
    );

    let recombined = fieldwise_request(domain, field, FieldwisePath::Fem, fvm_solver);
    let error = resolve_fieldwise(&recombined, requirements, &capabilities).unwrap_err();
    assert_eq!(error.code(), eqiora::diagnostic::codes::INVALID_REALIZATION);
    assert!(error.message().contains("no exact tuple"));

    let program = compile_cartesian_program(
        "verify/numerics/cartesian-poisson-fem-fvm/models/poisson.eqi",
        SOURCE_2D,
    );
    let generic_plan = RealizationPlan::new(
        Space::cell_constant(),
        Discretization::new(
            DiscretizationMethod::CellCenteredFiniteVolume,
            MeshPolicy::GeneratedUniform {
                cells_per_axis: NonZeroUsize::new(2).unwrap(),
            },
            QuadraturePolicy::CellCentroid,
        ),
        SolverPlan::new(
            fvm_solver.algorithm,
            1.0e-12,
            1.0e-14,
            NonZeroUsize::new(32).unwrap(),
        )
        .unwrap()
        .with_preconditioner(fvm_solver.preconditioner),
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
        ExecutionSchedule::Offline,
    )
    .unwrap();
    let resolved = resolve(
        &RealizationRequest::explicit(
            program.model(),
            SemanticRevision::new(program.revision().0),
            RealizationRevision::new(2),
            generic_plan,
        ),
        RealizationRequirements::new(dimension, ScalarType::F64, VectorLayoutKind::Replicated),
        &capabilities,
    )
    .expect("generic compatibility resolution retains its admitted property candidate");
    let error = resolved
        .portable_graph(
            domain,
            field,
            LinearOperatorProperties::SymmetricPositiveDefinite,
        )
        .expect_err("an equation claim outside the retained exact tuple must fail closed");
    assert_eq!(error.code(), eqiora::diagnostic::codes::INVALID_REALIZATION);
    assert!(
        error
            .message()
            .contains("not admitted for operator properties")
    );
}

#[derive(Clone, Copy)]
enum FieldwisePath {
    Fem,
    Fvm,
}

fn reference_spd_solver_capabilities() -> SolverCapabilities {
    SolverCapabilities::exact([SolverCapability {
        algorithm: LinearSolver::ConjugateGradient,
        operator_properties: LinearOperatorProperties::SymmetricPositiveDefinite,
        preconditioner: PreconditionerPolicy::Identity,
        reduction: ReductionPolicy::Reproducible,
        scalar_type: ScalarType::F64,
    }])
    .expect("the reference Poisson solver tuple is exact")
}

fn fieldwise_request(
    domain: eqiora::Id<eqiora::kinds::Domain>,
    field: eqiora::Id<eqiora::kinds::Field>,
    path: FieldwisePath,
    solver: SolverCapability,
) -> FieldwiseRealizationRequest {
    let (space, discretization) = match path {
        FieldwisePath::Fem => (
            Space::continuous_lagrange(NonZeroU16::MIN),
            Discretization::new(
                DiscretizationMethod::ContinuousGalerkin,
                MeshPolicy::ImportedSimplicial {
                    artifact: MeshArtifactReference::from_sha256([0x6a; 32]),
                },
                QuadraturePolicy::TriangleDuffyGaussLegendre {
                    points_per_axis: NonZeroUsize::new(3).unwrap(),
                },
            ),
        ),
        FieldwisePath::Fvm => (
            Space::cell_constant(),
            Discretization::new(
                DiscretizationMethod::CellCenteredFiniteVolume,
                MeshPolicy::GeneratedUniform {
                    cells_per_axis: NonZeroUsize::new(4).unwrap(),
                },
                QuadraturePolicy::CellCentroid,
            ),
        ),
    };
    let length_scale = PositivePhysicalScale::new(eqiora::DynQuantity::new(
        1.0,
        eqiora::DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension"),
    ))
    .unwrap();
    let unit_scale = PositivePhysicalScale::new(eqiora::DynQuantity::new(
        1.0,
        eqiora::DimExponents::DIMENSIONLESS,
    ))
    .unwrap();
    let spatial = FieldwiseSpatialDiscretization::new(
        domain,
        length_scale,
        [FieldSpaceBinding::new(field, space)],
        [],
        discretization,
    )
    .unwrap();
    let scaling = SymmetricCongruenceScaling::new(
        [AlgebraicBlockScale::new(
            AlgebraicBlock::Field(field),
            unit_scale,
        )],
        unit_scale,
    )
    .unwrap();
    let plan = FieldwiseRealizationPlan::new(
        spatial,
        scaling,
        solver.operator_properties,
        SolverPlan::new(
            solver.algorithm,
            1.0e-10,
            1.0e-12,
            NonZeroUsize::new(500).unwrap(),
        )
        .unwrap()
        .with_preconditioner(solver.preconditioner)
        .with_reduction(solver.reduction),
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
        ExecutionSchedule::Offline,
    )
    .unwrap();
    FieldwiseRealizationRequest::explicit(
        eqiora::ontology::OntologyId::new(),
        SemanticRevision::new(1),
        RealizationRevision::new(1),
        plan,
    )
}

#[test]
fn canonical_poisson_has_worker_independent_reproducible_cpu_evidence() {
    let program = compile_program();
    let requirements = RealizationRequirements::new(
        NonZeroUsize::MIN,
        ScalarType::F64,
        VectorLayoutKind::Replicated,
    );
    let semantic_revision = SemanticRevision::new(program.revision().0);
    let serial = resolve(
        &RealizationRequest::default(program.model(), semantic_revision, DefaultPolicyVersion::V0),
        requirements,
        &RealizationCapabilities::scalar_elliptic_reference(),
    )
    .unwrap();
    let lowered = lower_scalar_elliptic_1d(&program).unwrap();
    let serial_graph = serial
        .portable_graph(
            lowered.domain_id(),
            lowered.field_id(),
            LinearOperatorProperties::SymmetricPositiveDefinite,
        )
        .unwrap();
    assert!(matches!(serial_graph.root(), SolveRoot::Linear(_)));
    assert_eq!(serial_graph.domains()[0].domain(), lowered.domain_id());
    assert_eq!(serial_graph.fields()[0].field(), lowered.field_id());
    let (_, serial_solution) =
        solve_resolved_scalar_elliptic_1d(&program, &serial, &REFERENCE_LINEAR_SOLVER).unwrap();

    let workers = NonZeroUsize::new(4).unwrap();
    let pool = CpuThreadPool::new(workers).unwrap();
    let base = default_plan_v0().unwrap();
    let threaded_plan = RealizationPlan::new(
        base.space(),
        base.discretization(),
        base.solver(),
        Target::HostCpu { threads: workers },
        ExecutionSchedule::Offline,
    )
    .unwrap();
    let capabilities = RealizationCapabilities::cartesian_product(
        [
            DiscretizationMethod::ContinuousGalerkin,
            DiscretizationMethod::CellCenteredFiniteVolume,
        ],
        [(
            eqiora::realization::MeshKind::GeneratedCartesian,
            SpatialDimensionSupport::exact(NonZeroUsize::MIN),
        )],
        [VectorLayoutKind::Replicated],
        reference_spd_solver_capabilities(),
        pool.target_capabilities(),
    )
    .unwrap();
    let threaded = resolve(
        &RealizationRequest::explicit(
            program.model(),
            semantic_revision,
            RealizationRevision::new(1),
            threaded_plan,
        ),
        requirements,
        &capabilities,
    )
    .unwrap();
    let threaded_graph = threaded
        .portable_graph(
            lowered.domain_id(),
            lowered.field_id(),
            LinearOperatorProperties::SymmetricPositiveDefinite,
        )
        .unwrap();
    assert_eq!(threaded_graph.domains(), serial_graph.domains());
    assert_eq!(threaded_graph.fields(), serial_graph.fields());
    assert_eq!(
        threaded_graph.placements(),
        [PlacementRequirementNode::HostWorkers {
            workers_per_partition: workers,
        }]
    );
    let threaded_backend = pool
        .solver(threaded.plan().target(), &REFERENCE_LINEAR_SOLVER)
        .unwrap();
    let threaded_assembly = pool.assembler(threaded.plan().target()).unwrap();
    let (_, threaded_solution) = solve_resolved_scalar_elliptic_1d_with_assembly(
        &program,
        &threaded,
        &threaded_assembly,
        &threaded_backend,
    )
    .unwrap();

    let (
        ResolvedScalarEllipticSolution1d::FiniteElement(serial),
        ResolvedScalarEllipticSolution1d::FiniteElement(threaded),
    ) = (serial_solution, threaded_solution)
    else {
        panic!("the default plan selects the P1 finite-element realization");
    };
    assert_eq!(threaded.field().values(), serial.field().values());
    assert_eq!(threaded.cell_gradients(), serial.cell_gradients());
    assert_eq!(threaded.endpoint_reactions(), serial.endpoint_reactions());
    assert_eq!(threaded.residual_norm(), serial.residual_norm());
    assert_eq!(
        serial.assembly_report().execution(),
        ExecutionReport::host_serial()
    );
    assert_eq!(
        threaded.assembly_report().execution(),
        ExecutionReport::host(RAYON_EXECUTION, workers)
    );
    assert_eq!(
        threaded.assembly_report().packet_count(),
        serial.assembly_report().packet_count()
    );
    assert_eq!(
        threaded.assembly_report().target_count(),
        serial.assembly_report().target_count()
    );
    assert_eq!(
        serial.solve_report().execution(),
        ExecutionReport::host_serial()
    );
    assert_eq!(
        threaded.solve_report().execution(),
        ExecutionReport::host(RAYON_EXECUTION, workers)
    );
    assert_eq!(
        threaded.solve_report().backend(),
        serial.solve_report().backend()
    );
    assert_eq!(
        threaded.solve_report().orientation(),
        serial.solve_report().orientation()
    );
    assert_eq!(
        threaded.solve_report().algorithm(),
        serial.solve_report().algorithm()
    );
    assert_eq!(
        threaded.solve_report().preconditioner(),
        serial.solve_report().preconditioner()
    );
    assert_eq!(
        threaded.solve_report().reduction(),
        serial.solve_report().reduction()
    );
    assert_eq!(
        threaded.solve_report().reason(),
        serial.solve_report().reason()
    );
    assert_eq!(
        threaded.solve_report().completed_iterations(),
        serial.solve_report().completed_iterations()
    );
    assert_eq!(
        threaded.solve_report().initial_residual_norm(),
        serial.solve_report().initial_residual_norm()
    );
    assert_eq!(
        threaded.solve_report().reported_residual_norm(),
        serial.solve_report().reported_residual_norm()
    );
    assert_eq!(
        threaded.solve_report().true_residual_norm(),
        serial.solve_report().true_residual_norm()
    );
    assert_eq!(
        threaded.solve_report().residual_target(),
        serial.solve_report().residual_target()
    );

    let unsupported_workers = NonZeroUsize::new(5).unwrap();
    let unsupported_plan = RealizationPlan::new(
        base.space(),
        base.discretization(),
        base.solver(),
        Target::HostCpu {
            threads: unsupported_workers,
        },
        ExecutionSchedule::Offline,
    )
    .unwrap();
    assert!(
        resolve(
            &RealizationRequest::explicit(
                program.model(),
                semantic_revision,
                RealizationRevision::new(2),
                unsupported_plan,
            ),
            requirements,
            &capabilities,
        )
        .is_err()
    );
}

#[test]
fn cartesian_fem_and_fvm_have_exact_worker_independent_2d_and_3d_evidence() {
    let workers = NonZeroUsize::new(4).unwrap();
    let pool = CpuThreadPool::new(workers).unwrap();

    for (file, source, dimension, cells) in [
        (
            "verify/numerics/cartesian-poisson-fem-fvm/models/poisson.eqi",
            SOURCE_2D,
            2_usize,
            4_usize,
        ),
        (
            "verify/numerics/cartesian-poisson-3d-fem-fvm/models/poisson.eqi",
            SOURCE_3D,
            3_usize,
            3_usize,
        ),
    ] {
        let program = compile_cartesian_program(file, source);
        let dimension = NonZeroUsize::new(dimension).unwrap();
        let requirements =
            RealizationRequirements::new(dimension, ScalarType::F64, VectorLayoutKind::Replicated);
        let semantic_revision = SemanticRevision::new(program.revision().0);
        let threaded_capabilities = RealizationCapabilities::cartesian_product(
            [
                DiscretizationMethod::ContinuousGalerkin,
                DiscretizationMethod::CellCenteredFiniteVolume,
            ],
            [(
                eqiora::realization::MeshKind::GeneratedCartesian,
                SpatialDimensionSupport::exact(dimension),
            )],
            [VectorLayoutKind::Replicated],
            reference_spd_solver_capabilities(),
            pool.target_capabilities(),
        )
        .unwrap();

        for (revision, method) in [
            (0_u64, DiscretizationMethod::ContinuousGalerkin),
            (1_u64, DiscretizationMethod::CellCenteredFiniteVolume),
        ] {
            let serial = resolve(
                &RealizationRequest::explicit(
                    program.model(),
                    semantic_revision,
                    RealizationRevision::new(revision),
                    cartesian_plan(cells, dimension.get(), method, NonZeroUsize::MIN),
                ),
                requirements,
                &RealizationCapabilities::scalar_elliptic_reference(),
            )
            .unwrap();
            let threaded = resolve(
                &RealizationRequest::explicit(
                    program.model(),
                    semantic_revision,
                    RealizationRevision::new(revision + 2),
                    cartesian_plan(cells, dimension.get(), method, workers),
                ),
                requirements,
                &threaded_capabilities,
            )
            .unwrap();
            let (_, serial_solution) = solve_resolved_scalar_elliptic_cartesian(
                &program,
                &serial,
                &REFERENCE_LINEAR_SOLVER,
            )
            .unwrap();
            let threaded_solver = pool
                .solver(threaded.plan().target(), &REFERENCE_LINEAR_SOLVER)
                .unwrap();
            let threaded_assembly = pool.assembler(threaded.plan().target()).unwrap();
            let (_, threaded_solution) = solve_resolved_scalar_elliptic_cartesian_with_assembly(
                &program,
                &threaded,
                &threaded_assembly,
                &threaded_solver,
            )
            .unwrap();

            let cell_count = cells.pow(u32::try_from(dimension.get()).unwrap());
            match (serial_solution, threaded_solution) {
                (
                    ResolvedScalarEllipticCartesianSolution::FiniteElement(serial),
                    ResolvedScalarEllipticCartesianSolution::FiniteElement(threaded),
                ) => {
                    assert_eq!(threaded.field(), serial.field());
                    assert_eq!(threaded.algebraic_values(), serial.algebraic_values());
                    assert_eq!(
                        threaded.boundary_reaction_sum(),
                        serial.boundary_reaction_sum()
                    );
                    assert_eq!(threaded.integrated_source(), serial.integrated_source());
                    assert_assembly_evidence(
                        serial.assembly_report(),
                        threaded.assembly_report(),
                        workers,
                        cell_count,
                        2,
                    );
                    assert_solve_evidence(serial.solve_report(), threaded.solve_report(), workers);
                }
                (
                    ResolvedScalarEllipticCartesianSolution::FiniteVolume(serial),
                    ResolvedScalarEllipticCartesianSolution::FiniteVolume(threaded),
                ) => {
                    assert_eq!(threaded.mesh(), serial.mesh());
                    assert_eq!(threaded.cell_centers(), serial.cell_centers());
                    assert_eq!(threaded.cell_values(), serial.cell_values());
                    assert_eq!(threaded.reconstruction(), serial.reconstruction());
                    assert_eq!(threaded.boundary_flux_sum(), serial.boundary_flux_sum());
                    assert_eq!(threaded.integrated_source(), serial.integrated_source());
                    let facet_count = dimension
                        .get()
                        .checked_mul(cells + 1)
                        .and_then(|count| {
                            count
                                .checked_mul(cells.pow(u32::try_from(dimension.get() - 1).unwrap()))
                        })
                        .unwrap();
                    assert_assembly_evidence(
                        serial.assembly_report(),
                        threaded.assembly_report(),
                        workers,
                        cell_count + facet_count,
                        1,
                    );
                    assert_solve_evidence(serial.solve_report(), threaded.solve_report(), workers);
                }
                _ => panic!("serial and threaded plans selected different methods"),
            }
        }
    }
}

fn cartesian_plan(
    cells: usize,
    dimension: usize,
    method: DiscretizationMethod,
    threads: NonZeroUsize,
) -> RealizationPlan {
    let (space, quadrature) = match method {
        DiscretizationMethod::ContinuousGalerkin => (
            Space::continuous_lagrange(NonZeroU16::MIN),
            QuadraturePolicy::GaussLegendre {
                points_per_axis: NonZeroUsize::new(2).unwrap(),
            },
        ),
        DiscretizationMethod::CellCenteredFiniteVolume => {
            (Space::cell_constant(), QuadraturePolicy::CellCentroid)
        }
    };
    let unknown_scale = cells.pow(u32::try_from(dimension).unwrap());
    RealizationPlan::new(
        space,
        Discretization::new(
            method,
            MeshPolicy::GeneratedUniform {
                cells_per_axis: NonZeroUsize::new(cells).unwrap(),
            },
            quadrature,
        ),
        SolverPlan::new(
            LinearSolver::ConjugateGradient,
            1.0e-12,
            1.0e-14,
            NonZeroUsize::new(unknown_scale * 8).unwrap(),
        )
        .unwrap(),
        Target::HostCpu { threads },
        ExecutionSchedule::Offline,
    )
    .unwrap()
}

fn assert_assembly_evidence(
    serial: &eqiora::assembly::AssemblyReport,
    threaded: &eqiora::assembly::AssemblyReport,
    workers: NonZeroUsize,
    packet_count: usize,
    target_count: usize,
) {
    assert_eq!(serial.execution(), ExecutionReport::host_serial());
    assert_eq!(
        threaded.execution(),
        ExecutionReport::host(RAYON_EXECUTION, workers)
    );
    assert_eq!(serial.packet_count(), packet_count);
    assert_eq!(threaded.packet_count(), packet_count);
    assert_eq!(serial.target_count(), target_count);
    assert_eq!(threaded.target_count(), target_count);
}

fn assert_solve_evidence(serial: &SolveReport, threaded: &SolveReport, workers: NonZeroUsize) {
    assert_eq!(serial.execution(), ExecutionReport::host_serial());
    assert_eq!(
        threaded.execution(),
        ExecutionReport::host(RAYON_EXECUTION, workers)
    );
    assert_eq!(threaded.backend(), serial.backend());
    assert_eq!(threaded.orientation(), serial.orientation());
    assert_eq!(threaded.algorithm(), serial.algorithm());
    assert_eq!(threaded.preconditioner(), serial.preconditioner());
    assert_eq!(threaded.reduction(), serial.reduction());
    assert_eq!(threaded.reason(), serial.reason());
    assert_eq!(
        threaded.completed_iterations(),
        serial.completed_iterations()
    );
    assert_eq!(
        threaded.initial_residual_norm(),
        serial.initial_residual_norm()
    );
    assert_eq!(
        threaded.reported_residual_norm(),
        serial.reported_residual_norm()
    );
    assert_eq!(threaded.true_residual_norm(), serial.true_residual_norm());
    assert_eq!(threaded.residual_target(), serial.residual_target());
}

fn compile_program() -> KernelProgram {
    let mut compiled =
        compile("verify/numerics/poisson-fem-fvm/models/poisson.eqi", SOURCE).unwrap();
    let (transaction, model, _) = compiled.remove(0).into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    KernelProgram::from_snapshot(&store.snapshot(), model).unwrap()
}

fn compile_cartesian_program(file: &str, source: &str) -> KernelProgram {
    let mut compiled = compile(file, source).unwrap();
    let (transaction, model, _) = compiled.remove(0).into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    KernelProgram::from_snapshot(&store.snapshot(), model).unwrap()
}
