mod remesh_policy;

use crate::simplicial_fsi::test_model::mesh_3d::{inventories_3d, mesh_3d, mesh_3d_with_upper_z};
use std::num::{NonZeroU16, NonZeroUsize};

use eqiora_compiler::compile;
use eqiora_core::{Diagnostic, DimExponents, DynQuantity};
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_meshing::{FacetId, MeshQualityGate};
use eqiora_realization::{
    AleGeometryQualityGate, AlgebraicBlockScale, BackwardEulerRelationStep,
    BackwardEulerStateBinding, BackwardEulerStep, CoupledFieldwiseRealizationPlan,
    CoupledFieldwiseSpatialDiscretization, Discretization, DomainFieldDiscretization,
    FieldSpaceBinding, FixedTopologyAleCoupledRealizationPlan,
    FixedTopologyAleCoupledRealizationRequest, GclCompatibleAlePullback, MeshKind,
    NonlinearSolvePlan, P1HarmonicMeshMotionPolicy, PositivePhysicalScale, RealizationCapabilities,
    RealizationRevision, SemanticRevision, Space, SpatialDimensionSupport,
    SymmetricCongruenceScaling, TargetCapabilities, resolve_fixed_topology_ale_coupled,
};
use eqiora_sem::KernelProgram;
use eqiora_solver::{
    BackendId, LinearProblem, LinearSolution, LinearSolver, PreconditionerPolicy,
    REFERENCE_LINEAR_SOLVER, ReductionPolicy, ReplicatedLinearExecution, SolverCapabilities,
    SolverCapability, SolverPlan, SolverProvider,
};

use super::*;
use crate::canonical_fsi::AleFsiCartesianModel;
use crate::simplicial_fsi::FixedReferenceFsiBoundary;
use crate::simplicial_fsi::FixedReferenceFsiPartition;

const BASE_SOURCE: &str =
    include_str!("../../../../../verify/fsi/fixed-reference-monolithic-step-2d/models/direct.eqi");
const TIME: DimExponents =
    DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).expect("bounded dimension");
#[test]
fn exact_resolved_plan_finalizes_and_accepts_a_trajectory() {
    let fixture = Fixture::new();
    let finalized = fixture.finalize(fixture.initial()).unwrap();
    assert_eq!(finalized.model(), fixture.model.model());
    assert_eq!(finalized.mesh_artifact(), mesh_reference());
    assert_eq!(finalized.fields(), field_identities(&fixture.model));
    assert_eq!(
        finalized.initial_state().geometry().coordinates(),
        fixture.mesh.vertices()
    );
    assert!(!finalized.motion().influence_solve_reports().is_empty());

    let trajectory = finalized
        .solve(
            NonZeroStepCount::new(NonZeroUsize::new(2).unwrap()),
            &NoSolveGeneralBackend,
        )
        .expect("zero equilibrium accepts without invoking a Newton linear solve");
    assert_eq!(trajectory.states().len(), 3);
    assert_eq!(trajectory.steps().len(), 2);
    assert!(
        trajectory
            .steps()
            .iter()
            .all(|step| step.nonlinear_iterations() == 0)
    );
}

#[test]
fn reversed_momentum_equations_preserve_the_nonzero_ale_trajectory() {
    let source = ale_source();
    let mut reversed = source.clone();
    replace_exactly(
        &mut reversed,
        "relation solid_momentum on solid {\n    solid_density",
        "relation solid_momentum on solid {\n    -(solid_density",
        1,
    );
    replace_exactly(
        &mut reversed,
        ") - grad(solid_load_potential) = 0;",
        ") - grad(solid_load_potential)) = 0;",
        1,
    );
    let solve = |source: &str, fluid_orientation, solid_orientation| {
        let mut fixture = Fixture::from_source(source);
        // Resolve well below the comparison tolerance: changing semantic IDs can
        // permute the linear system and change an iterative solver's stopping point.
        let original = fixture.resolved.plan();
        let coupled = original.coupled();
        let linear = SolverPlan::new(
            LinearSolver::BiConjugateGradientStabilized,
            1e-13,
            1e-15,
            NonZeroUsize::new(500).unwrap(),
        )
        .unwrap()
        .with_preconditioner(PreconditionerPolicy::Identity)
        .with_reduction(ReductionPolicy::Reproducible);
        let plan = FixedTopologyAleCoupledRealizationPlan::new(
            CoupledFieldwiseRealizationPlan::new(
                coupled.spatial().clone(),
                coupled.time_step().clone(),
                coupled.scaling().clone(),
                coupled.operator_properties(),
                linear,
                coupled.target(),
                coupled.schedule(),
            )
            .unwrap(),
            original.fluid_time_step(),
            original.solid_kinematic_relation(),
            original.mesh_motion().clone(),
            original.pullback(),
            NonlinearSolvePlan::new(1e-12, 1e-14, NonZeroUsize::new(20).unwrap(), 16).unwrap(),
        )
        .unwrap();
        fixture.resolved = fixture.resolve_with(
            plan,
            fixed_topology_ale_fsi_requirements(&fixture.model),
            SemanticRevision::new(fixture.model.semantic_revision()),
        );
        assert_eq!(
            fixture.model.solid().momentum_orientation(),
            solid_orientation
        );
        assert_eq!(
            fixture.model.fluid().momentum_orientation(),
            fluid_orientation
        );
        let vertex = fixture
            .mesh
            .vertices()
            .iter()
            .position(|point| point.as_slice() == [1.0, 0.5])
            .unwrap();
        let mut initial = exact_initial(
            &fixture.program,
            &fixture.model,
            fixture.resolved.coupled().plan(),
            &fixture.mesh,
            &fixture.partition,
            Some((vertex, [0.01, -0.005])),
        );
        for (key, value) in &mut initial
            .physical
            .fields
            .get_mut(&solid_displacement(&fixture.model).erase())
            .unwrap()
            .coefficients
        {
            if key.entity == MeshEntity::new(0, vertex) {
                *value = [0.001, -0.0005][key.component];
            }
        }
        let trajectory = fixture
            .finalize(initial)
            .unwrap()
            .solve(
                NonZeroStepCount::new(NonZeroUsize::MIN),
                &REFERENCE_LINEAR_SOLVER,
            )
            .unwrap();
        assert!(trajectory.steps()[0].nonlinear_iterations() > 0);
        let accepted = trajectory.states().last().unwrap().physical_state();
        [
            fluid_velocity(&fixture.model),
            fluid_pressure(&fixture.model),
            solid_velocity(&fixture.model),
            solid_displacement(&fixture.model),
        ]
        .into_iter()
        .flat_map(|field| {
            accepted.fields[&field.erase()]
                .coefficients
                .values()
                .copied()
        })
        .collect::<Vec<_>>()
    };
    // Multiplying one residual equation by -1 preserves the physical problem.
    // Compare role/entity coefficients, independent of semantic IDs and solver ordering.
    let forward = solve(&source, 1.0, 1.0);
    let mut reversed_fluid = source.clone();
    replace_exactly(
        &mut reversed_fluid,
        "relation fluid_momentum on fluid {\n    fluid_density",
        "relation fluid_momentum on fluid {\n    -(fluid_density",
        1,
    );
    replace_exactly(
        &mut reversed_fluid,
        ") - grad(fluid_load_potential) = 0;",
        ") - grad(fluid_load_potential)) = 0;",
        1,
    );
    for values in [
        solve(&reversed, 1.0, -1.0),
        solve(&reversed_fluid, -1.0, 1.0),
    ] {
        assert_eq!(forward.len(), values.len());
        for (left, right) in forward.iter().zip(&values) {
            assert!((left - right).abs() <= 1e-10, "{left} differs from {right}");
        }
    }
}

#[test]
fn direct_tetrahedral_model_reaches_the_same_finalized_newton_boundary() {
    let program = compile_program(&ale_source_3d());
    let model = super::super::lower_ale_fsi_cartesian::<3>(&program).unwrap();
    let mesh = mesh_3d(MeshQualityGate::new(0.1).unwrap());
    let (fluid, solid, _) = inventories_3d(&mesh);
    let partition = exact_partition(&model, &mesh, fluid.clone(), solid.clone()).unwrap();
    let boundary = FixedReferenceFsiBoundary::<3>::homogeneous_exterior(&mesh).unwrap();
    assert!(FixedReferenceFsiBoundary::<2>::homogeneous_exterior(&mesh).is_err());
    assert!(exact_partition(&model, &mesh, fluid, solid[..solid.len() - 1].to_vec()).is_err());
    let requirements = fixed_topology_ale_fsi_requirements(&model);
    assert_eq!(
        requirements.coupled().execution().spatial_dimension().get(),
        3
    );
    let plan = build_plan(&model, fluid_pressure(&model), 0.1);
    assert_eq!(
        plan.coupled().spatial().discretization().quadrature(),
        required_quadrature_policy::<3>().unwrap()
    );
    let resolved = resolve(
        &program,
        plan,
        requirements,
        SemanticRevision::new(model.semantic_revision()),
        3,
    );

    let fixed = boundary.fixed_zero_velocity_vertices()[0];
    let invalid = exact_initial(
        &program,
        &model,
        resolved.coupled().plan(),
        &mesh,
        &partition,
        Some((fixed.index(), [0.0, 0.0, 1.0])),
    );
    assert!(
        finalize_resolved_fixed_topology_ale_fsi_3d(
            &model,
            &resolved,
            mesh_reference(),
            &mesh,
            &partition,
            &boundary,
            invalid,
            &REFERENCE_LINEAR_SOLVER,
        )
        .unwrap_err()
        .message()
        .contains("homogeneous exterior closure")
    );

    let short_mesh = mesh_3d_with_upper_z(0.9, MeshQualityGate::new(0.1).unwrap());
    let (fluid, solid, _) = inventories_3d(&short_mesh);
    let short_partition = exact_partition(&model, &short_mesh, fluid, solid).unwrap();
    let short_boundary = FixedReferenceFsiBoundary::<3>::homogeneous_exterior(&short_mesh).unwrap();
    assert!(
        finalize_resolved_fixed_topology_ale_fsi_3d(
            &model,
            &resolved,
            mesh_reference(),
            &short_mesh,
            &short_partition,
            &short_boundary,
            exact_initial(
                &program,
                &model,
                resolved.coupled().plan(),
                &short_mesh,
                &short_partition,
                None,
            ),
            &REFERENCE_LINEAR_SOLVER,
        )
        .unwrap_err()
        .message()
        .contains("mesh exterior does not lie on an exact semantic side")
    );

    let finalized: FinalizedResolvedFixedTopologyAleFsi<3> =
        finalize_resolved_fixed_topology_ale_fsi_3d(
            &model,
            &resolved,
            mesh_reference(),
            &mesh,
            &partition,
            &boundary,
            exact_initial(
                &program,
                &model,
                resolved.coupled().plan(),
                &mesh,
                &partition,
                None,
            ),
            &REFERENCE_LINEAR_SOLVER,
        )
        .unwrap();
    let expected_fields: AleFsiFieldIdentities<3> = field_identities(&model);
    assert_eq!(finalized.fields(), expected_fields);
    assert_eq!(finalized.step_plan().scale().power(), 4.0);
    assert_eq!(
        finalized.initial_state().geometry().coordinates(),
        mesh.vertices()
    );
    let trajectory = finalized
        .solve(
            NonZeroStepCount::new(NonZeroUsize::MIN),
            &NoSolveGeneralBackend,
        )
        .unwrap();
    assert_eq!(trajectory.states().len(), 2);
    assert_eq!(trajectory.steps().len(), 1);
    assert_eq!(trajectory.steps()[0].nonlinear_iterations(), 0);
}

#[test]
fn finalization_rejects_stale_model_revision_mesh_and_role() {
    let fixture = Fixture::new();
    let mut foreign_source = ale_source();
    replace_exactly(
        &mut foreign_source,
        "parameter fluid_density: kg / m ^ 3 = 2;",
        "parameter fluid_density: kg / m ^ 3 = 2.5;",
        1,
    );
    let foreign = Fixture::from_source(&foreign_source);
    assert!(
        finalize_resolved_fixed_topology_ale_fsi_2d(
            &fixture.model,
            &foreign.resolved,
            mesh_reference(),
            &fixture.mesh,
            &fixture.partition,
            &fixture.boundary,
            fixture.initial(),
            &REFERENCE_LINEAR_SOLVER,
        )
        .unwrap_err()
        .message()
        .contains("exact lowered Semantic Model revision")
    );

    let stale_revision = fixture.resolve_with(
        fixture.plan(fluid_pressure(&fixture.model)),
        fixed_topology_ale_fsi_requirements(&fixture.model),
        SemanticRevision::new(fixture.model.semantic_revision() + 1),
    );
    assert!(
        fixture
            .finalize_resolved(&stale_revision, mesh_reference(), fixture.initial())
            .unwrap_err()
            .message()
            .contains("exact lowered Semantic Model revision")
    );

    let mut stale = mesh_reference().sha256();
    stale[0] ^= 1;
    assert!(
        fixture
            .finalize_resolved(
                &fixture.resolved,
                MeshArtifactReference::from_sha256(stale),
                fixture.initial(),
            )
            .unwrap_err()
            .message()
            .contains("authenticated mesh digest")
    );

    let foreign_pressure = Id::new();
    let role_plan = fixture.plan(foreign_pressure);
    let role_requirements = requirements_with_pressure(&fixture.model, foreign_pressure);
    let role_resolved = fixture.resolve_with(
        role_plan,
        role_requirements,
        SemanticRevision::new(fixture.model.semantic_revision()),
    );
    assert!(
        fixture
            .finalize_resolved(&role_resolved, mesh_reference(), fixture.initial())
            .unwrap_err()
            .message()
            .contains("exact canonical Domain")
    );
}

#[test]
fn finalization_rejects_quality_and_initial_boundary_drift() {
    let fixture = Fixture::new();
    let lower_quality_mesh = mesh(MeshQualityGate::new(0.2).unwrap());
    let (fluid, solid, _) = inventories(&lower_quality_mesh);
    let partition = exact_partition(&fixture.model, &lower_quality_mesh, fluid, solid).unwrap();
    let boundary =
        FixedReferenceFsiBoundary::<2>::homogeneous_exterior(&lower_quality_mesh).unwrap();
    assert!(
        finalize_resolved_fixed_topology_ale_fsi_2d(
            &fixture.model,
            &fixture.resolved,
            mesh_reference(),
            &lower_quality_mesh,
            &partition,
            &boundary,
            exact_initial(
                &fixture.program,
                &fixture.model,
                fixture.resolved.coupled().plan(),
                &lower_quality_mesh,
                &partition,
                None,
            ),
            &REFERENCE_LINEAR_SOLVER,
        )
        .unwrap_err()
        .message()
        .contains("exact resolved geometry-quality gate")
    );

    let initial = exact_initial(
        &fixture.program,
        &fixture.model,
        fixture.resolved.coupled().plan(),
        &fixture.mesh,
        &fixture.partition,
        Some((0, [1.0, 0.0])),
    );
    assert!(
        fixture
            .finalize(initial)
            .unwrap_err()
            .message()
            .contains("homogeneous exterior closure")
    );
}

struct Fixture {
    program: KernelProgram,
    model: AleFsiCartesianModel<2>,
    mesh: SimplicialMesh,
    partition: FixedReferenceFsiPartition<2>,
    boundary: FixedReferenceFsiBoundary<2>,
    resolved: ResolvedFixedTopologyAleCoupledRealization,
}

impl Fixture {
    fn new() -> Self {
        Self::from_source(&ale_source())
    }

    fn from_source(source: &str) -> Self {
        let program = compile_program(source);
        let model = super::super::lower_ale_fsi_cartesian::<2>(&program).unwrap();
        let mesh = mesh(MeshQualityGate::new(0.3).unwrap());
        let (fluid, solid, _) = inventories(&mesh);
        let partition = exact_partition(&model, &mesh, fluid, solid).unwrap();
        let boundary = FixedReferenceFsiBoundary::<2>::homogeneous_exterior(&mesh).unwrap();
        let plan = Self::build_plan(&model, fluid_pressure(&model), 0.3);
        let resolved = resolve(
            &program,
            plan,
            fixed_topology_ale_fsi_requirements(&model),
            SemanticRevision::new(model.semantic_revision()),
            2,
        );
        Self {
            program,
            model,
            mesh,
            partition,
            boundary,
            resolved,
        }
    }

    fn initial(&self) -> AleFsiInitialPhysicalState<2> {
        exact_initial(
            &self.program,
            &self.model,
            self.resolved.coupled().plan(),
            &self.mesh,
            &self.partition,
            None,
        )
    }

    fn plan(&self, pressure: Id<kinds::Field>) -> FixedTopologyAleCoupledRealizationPlan {
        Self::build_plan(&self.model, pressure, 0.3)
    }

    fn build_plan(
        model: &AleFsiCartesianModel<2>,
        pressure: Id<kinds::Field>,
        minimum_mean_ratio: f64,
    ) -> FixedTopologyAleCoupledRealizationPlan {
        build_plan(model, pressure, minimum_mean_ratio)
    }

    fn resolve_with(
        &self,
        plan: FixedTopologyAleCoupledRealizationPlan,
        requirements: FixedTopologyAleCoupledRealizationRequirements,
        semantic_revision: SemanticRevision,
    ) -> ResolvedFixedTopologyAleCoupledRealization {
        resolve(&self.program, plan, requirements, semantic_revision, 2)
    }

    fn finalize(
        &self,
        initial: AleFsiInitialPhysicalState<2>,
    ) -> Result<FinalizedResolvedFixedTopologyAleFsi<2>, Diagnostic> {
        self.finalize_resolved(&self.resolved, mesh_reference(), initial)
    }

    fn finalize_resolved(
        &self,
        resolved: &ResolvedFixedTopologyAleCoupledRealization,
        mesh_artifact: MeshArtifactReference,
        initial: AleFsiInitialPhysicalState<2>,
    ) -> Result<FinalizedResolvedFixedTopologyAleFsi<2>, Diagnostic> {
        finalize_resolved_fixed_topology_ale_fsi_2d(
            &self.model,
            resolved,
            mesh_artifact,
            &self.mesh,
            &self.partition,
            &self.boundary,
            initial,
            &REFERENCE_LINEAR_SOLVER,
        )
    }
}

fn build_plan<const D: usize>(
    model: &AleFsiCartesianModel<D>,
    pressure: Id<kinds::Field>,
    minimum_mean_ratio: f64,
) -> FixedTopologyAleCoupledRealizationPlan {
    let p1 = Space::continuous_lagrange(NonZeroU16::MIN);
    let length = physical_scale(2.0, LENGTH);
    let velocity = physical_scale(1.0, VELOCITY);
    let pressure_scale = physical_scale(1.0, PRESSURE);
    let coupled = CoupledFieldwiseRealizationPlan::new(
        CoupledFieldwiseSpatialDiscretization::new(
            length,
            [
                DomainFieldDiscretization::new(
                    fluid_domain(model),
                    [
                        FieldSpaceBinding::new(fluid_velocity(model), Space::simplex_p1_bubble()),
                        FieldSpaceBinding::new(pressure, p1),
                    ],
                    [],
                )
                .unwrap(),
                DomainFieldDiscretization::new(
                    solid_domain(model),
                    [FieldSpaceBinding::new(solid_velocity(model), p1)],
                    [],
                )
                .unwrap(),
            ],
            [trace_quotient(model)],
            Discretization::new(
                DiscretizationMethod::ContinuousGalerkin,
                MeshPolicy::ImportedSimplicial {
                    artifact: mesh_reference(),
                },
                required_quadrature_policy::<D>().unwrap(),
            ),
        )
        .unwrap(),
        BackwardEulerStep::new(
            DynQuantity::new(0.02, TIME),
            [BackwardEulerStateBinding::new(
                state_pair(model),
                p1,
                length,
            )],
        )
        .unwrap(),
        SymmetricCongruenceScaling::new(
            [
                AlgebraicBlockScale::new(AlgebraicBlock::Field(fluid_velocity(model)), velocity),
                AlgebraicBlockScale::new(AlgebraicBlock::Field(pressure), pressure_scale),
                AlgebraicBlockScale::new(AlgebraicBlock::Field(solid_velocity(model)), velocity),
            ],
            PositivePhysicalScale::new(
                weak_functional_power::<D>(
                    pressure_scale.quantity(),
                    velocity.quantity(),
                    length.quantity(),
                )
                .unwrap(),
            )
            .unwrap(),
        )
        .unwrap(),
        LinearOperatorProperties::General,
        nonlinear_solver(),
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
        ExecutionSchedule::Offline,
    )
    .unwrap();
    let duration = DynQuantity::new(0.02, TIME);
    FixedTopologyAleCoupledRealizationPlan::new(
        coupled,
        BackwardEulerRelationStep::new(fluid_relation(model), fluid_velocity(model), duration)
            .unwrap(),
        solid_kinematic_relation(model),
        P1HarmonicMeshMotionPolicy::new(
            [fluid_domain(model)],
            [(solid_domain(model), solid_displacement(model))],
            [connection(model)],
            AleGeometryQualityGate::new(minimum_mean_ratio).unwrap(),
            harmonic_solver(),
        )
        .unwrap(),
        GclCompatibleAlePullback::new(fluid_relation(model), fluid_velocity(model)),
        NonlinearSolvePlan::new(1.0e-7, 1.0e-10, NonZeroUsize::new(20).unwrap(), 16).unwrap(),
    )
    .unwrap()
}

fn resolve(
    program: &KernelProgram,
    plan: FixedTopologyAleCoupledRealizationPlan,
    requirements: FixedTopologyAleCoupledRealizationRequirements,
    semantic_revision: SemanticRevision,
    dimension: usize,
) -> ResolvedFixedTopologyAleCoupledRealization {
    resolve_fixed_topology_ale_coupled(
        &FixedTopologyAleCoupledRealizationRequest::explicit(
            program.model(),
            semantic_revision,
            RealizationRevision::new(3),
            plan,
        ),
        requirements,
        &capabilities(dimension),
    )
    .unwrap()
}

fn requirements_with_pressure(
    model: &AleFsiCartesianModel<2>,
    pressure: Id<kinds::Field>,
) -> FixedTopologyAleCoupledRealizationRequirements {
    let requirements = fixed_topology_ale_fsi_requirements(model);
    let coupled = CoupledFieldwiseRealizationRequirements::new(
        [
            DomainFieldInventory::new(fluid_domain(model), [fluid_velocity(model), pressure])
                .unwrap(),
            DomainFieldInventory::new(
                solid_domain(model),
                [solid_velocity(model), solid_displacement(model)],
            )
            .unwrap(),
        ],
        requirements.coupled().trace_quotients(),
        requirements.coupled().eliminated_states().iter().copied(),
        requirements.coupled().execution(),
    )
    .unwrap();
    FixedTopologyAleCoupledRealizationRequirements::new(
        coupled,
        fluid_domain(model),
        solid_domain(model),
        fluid_relation(model),
        solid_kinematic_relation(model),
        fluid_velocity(model),
        solid_displacement(model),
    )
    .unwrap()
}

fn capabilities(dimension: usize) -> RealizationCapabilities {
    RealizationCapabilities::cartesian_product(
        [DiscretizationMethod::ContinuousGalerkin],
        [(
            MeshKind::ImportedAffineSimplicial,
            SpatialDimensionSupport::exact(NonZeroUsize::new(dimension).unwrap()),
        )],
        [VectorLayoutKind::Replicated],
        SolverCapabilities::exact([
            SolverCapability {
                scalar_domain: eqiora_core::ScalarDomain::Real,
                algorithm: LinearSolver::BiConjugateGradientStabilized,
                operator_properties: LinearOperatorProperties::General,
                preconditioner: PreconditionerPolicy::Identity,
                reduction: ReductionPolicy::Reproducible,
                scalar_type: ScalarType::F64,
            },
            SolverCapability {
                scalar_domain: eqiora_core::ScalarDomain::Real,
                algorithm: LinearSolver::ConjugateGradient,
                operator_properties: LinearOperatorProperties::SymmetricPositiveDefinite,
                preconditioner: PreconditionerPolicy::Identity,
                reduction: ReductionPolicy::Reproducible,
                scalar_type: ScalarType::F64,
            },
        ])
        .unwrap(),
        TargetCapabilities::none().with_host_cpu(NonZeroUsize::MIN),
    )
    .unwrap()
}

fn exact_partition<const D: usize>(
    model: &AleFsiCartesianModel<D>,
    mesh: &SimplicialMesh,
    fluid_cells: Vec<CellId>,
    solid_cells: Vec<CellId>,
) -> Result<FixedReferenceFsiPartition<D>, Diagnostic> {
    FixedReferenceFsiPartition::new(
        mesh,
        [
            (fluid_domain(model), fluid_cells),
            (solid_domain(model), solid_cells),
        ],
        &[trace_quotient(model)],
    )
}

fn exact_initial<const D: usize>(
    program: &KernelProgram,
    model: &AleFsiCartesianModel<D>,
    plan: &CoupledFieldwiseRealizationPlan,
    mesh: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<D>,
    velocity_override: Option<(usize, [f64; D])>,
) -> AleFsiInitialPhysicalState<D> {
    let fluid_domain = fluid_domain(model);
    let solid_domain = solid_domain(model);
    let physical = crate::simplicial_fsi::FixedReferenceFsiState::new(
        program,
        plan,
        mesh,
        partition,
        [
            (
                fluid_velocity(model),
                vector_coefficients(partition, fluid_domain, true, velocity_override),
            ),
            (
                fluid_pressure(model),
                partition
                    .domain_vertices(fluid_domain)
                    .unwrap()
                    .iter()
                    .map(|vertex| (MeshEntity::new(0, vertex.index()), 0, 0, 0.0))
                    .collect(),
            ),
            (
                solid_velocity(model),
                vector_coefficients(partition, solid_domain, false, velocity_override),
            ),
            (
                solid_displacement(model),
                vector_coefficients(partition, solid_domain, false, None),
            ),
        ],
    )
    .unwrap();
    AleFsiInitialPhysicalState::new(0.0, physical).unwrap()
}

fn vector_coefficients<const D: usize>(
    partition: &FixedReferenceFsiPartition<D>,
    domain: Id<kinds::Domain>,
    bubbles: bool,
    override_vertex: Option<(usize, [f64; D])>,
) -> Vec<(MeshEntity, usize, usize, f64)> {
    let mut values = partition
        .domain_vertices(domain)
        .unwrap()
        .iter()
        .flat_map(|vertex| {
            let vector = override_vertex
                .filter(|(index, _)| *index == vertex.index())
                .map_or([0.0; D], |(_, value)| value);
            vector
                .into_iter()
                .enumerate()
                .map(move |(component, value)| {
                    (MeshEntity::new(0, vertex.index()), 0, component, value)
                })
        })
        .collect::<Vec<_>>();
    if bubbles {
        values.extend(
            partition
                .domain_cells(domain)
                .unwrap()
                .iter()
                .flat_map(|cell| {
                    (0..D)
                        .map(move |component| (MeshEntity::new(D, cell.index()), 0, component, 0.0))
                }),
        );
    }
    values
}

fn mesh(quality: MeshQualityGate) -> SimplicialMesh {
    let xs = [0.0, 0.5, 1.0, 1.5, 2.0];
    let mut vertices = Vec::new();
    for y in [0.0, 0.5, 1.0] {
        for x in xs {
            vertices.push(vec![x, y]);
        }
    }
    let width = xs.len();
    let mut cells = Vec::new();
    for row in 0..2 {
        for column in 0..width - 1 {
            let lower_left = row * width + column;
            let lower_right = lower_left + 1;
            let upper_left = lower_left + width;
            let upper_right = upper_left + 1;
            cells.push(vec![lower_left, lower_right, upper_right]);
            cells.push(vec![lower_left, upper_right, upper_left]);
        }
    }
    SimplicialMesh::new(2, vertices, cells, quality).unwrap()
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

fn compile_program(source: &str) -> KernelProgram {
    let mut compiled = compile("ale-fsi.eqi", source).unwrap();
    let (transaction, model, _) = compiled.remove(0).into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    KernelProgram::from_snapshot(&store.snapshot(), model).unwrap()
}

fn ale_source() -> String {
    format!(
            "public operator outer_product(input left: spatial[1], input right: spatial[1]): spatial[2]\n  = component(left, 0) * component(right, 1);\n{}",
            BASE_SOURCE.replace(
                "fluid_density * derivative(fluid_velocity)\n      - div(",
                "fluid_density * derivative(fluid_velocity)\n      + div(fluid_density * outer_product(left = fluid_velocity, right = fluid_velocity))\n      - div(",
            )
        )
}

fn ale_source_3d() -> String {
    let mut source = ale_source()
        .replace("vector<m / s, 2>", "vector<m / s, 3>")
        .replace("vector<m, 2>", "vector<m, 3>");
    replace_exactly(
        &mut source,
        "ambient_dimension = 2",
        "ambient_dimension = 3",
        3,
    );
    for (from, to) in [
        (
            "domain fluid = box(0, 1, 0, 1);",
            "domain fluid = box(0, 1, 0, 1, 0, 1);",
        ),
        (
            "domain solid = box(1, 2, 0, 1);",
            "domain solid = box(1, 2, 0, 1, 0, 1);",
        ),
        (
            "  domain fluid_y_upper = boundary(fluid, axis = 1, side = upper);",
            "  domain fluid_y_upper = boundary(fluid, axis = 1, side = upper);\n  domain fluid_z_lower = boundary(fluid, axis = 2, side = lower);\n  domain fluid_z_upper = boundary(fluid, axis = 2, side = upper);",
        ),
        (
            "  domain solid_y_upper = boundary(solid, axis = 1, side = upper);",
            "  domain solid_y_upper = boundary(solid, axis = 1, side = upper);\n  domain solid_z_lower = boundary(solid, axis = 2, side = lower);\n  domain solid_z_upper = boundary(solid, axis = 2, side = upper);",
        ),
        (
            "      fluid_x_lower, fluid_x_upper, fluid_y_lower, fluid_y_upper\n",
            "      fluid_x_lower, fluid_x_upper, fluid_y_lower, fluid_y_upper,\n      fluid_z_lower, fluid_z_upper\n",
        ),
        (
            "      solid_x_lower, solid_x_upper, solid_y_lower, solid_y_upper\n",
            "      solid_x_lower, solid_x_upper, solid_y_lower, solid_y_upper,\n      solid_z_lower, solid_z_upper\n",
        ),
        (
            "  instance fluid_y_upper_zero: ZeroVelocity2d(\n    body = fluid, face = fluid_y_upper\n  );",
            "  instance fluid_y_upper_zero: ZeroVelocity2d(\n    body = fluid, face = fluid_y_upper\n  );\n  instance fluid_z_lower_zero: ZeroVelocity2d(\n    body = fluid, face = fluid_z_lower\n  );\n  instance fluid_z_upper_zero: ZeroVelocity2d(\n    body = fluid, face = fluid_z_upper\n  );",
        ),
        (
            "  instance solid_y_upper_zero: ZeroVelocity2d(\n    body = solid, face = solid_y_upper\n  );",
            "  instance solid_y_upper_zero: ZeroVelocity2d(\n    body = solid, face = solid_y_upper\n  );\n  instance solid_z_lower_zero: ZeroVelocity2d(\n    body = solid, face = solid_z_lower\n  );\n  instance solid_z_upper_zero: ZeroVelocity2d(\n    body = solid, face = solid_z_upper\n  );",
        ),
        (
            "  connect fluid_boundary.mechanical[boundary = fluid_y_upper],\n    fluid_y_upper_zero.mechanical;",
            "  connect fluid_boundary.mechanical[boundary = fluid_y_upper],\n    fluid_y_upper_zero.mechanical;\n  connect fluid_boundary.mechanical[boundary = fluid_z_lower],\n    fluid_z_lower_zero.mechanical;\n  connect fluid_boundary.mechanical[boundary = fluid_z_upper],\n    fluid_z_upper_zero.mechanical;",
        ),
        (
            "  connect solid_boundary.mechanical[boundary = solid_y_upper],\n    solid_y_upper_zero.mechanical;",
            "  connect solid_boundary.mechanical[boundary = solid_y_upper],\n    solid_y_upper_zero.mechanical;\n  connect solid_boundary.mechanical[boundary = solid_z_lower],\n    solid_z_lower_zero.mechanical;\n  connect solid_boundary.mechanical[boundary = solid_z_upper],\n    solid_z_upper_zero.mechanical;",
        ),
    ] {
        replace_exactly(&mut source, from, to, 1);
    }
    for (from, to, expected) in [
        ("ZeroVelocity2d", "ZeroVelocity3d", 11),
        (
            "NewtonianMechanicalInterface2d",
            "NewtonianMechanicalInterface3d",
            2,
        ),
        (
            "ElastodynamicMechanicalInterface2d",
            "ElastodynamicMechanicalInterface3d",
            2,
        ),
    ] {
        replace_exactly(&mut source, from, to, expected);
    }
    source
}

fn replace_exactly(source: &mut String, from: &str, to: &str, expected: usize) {
    assert_eq!(
        source.match_indices(from).count(),
        expected,
        "source lift drifted"
    );
    *source = source.replace(from, to);
}

fn mesh_reference() -> MeshArtifactReference {
    MeshArtifactReference::from_sha256([154; 32])
}

fn physical_scale(value: f64, dimension: DimExponents) -> PositivePhysicalScale {
    PositivePhysicalScale::new(DynQuantity::new(value, dimension)).unwrap()
}

fn harmonic_solver() -> eqiora_solver::SolverPlan {
    SolverPlan::new(
        LinearSolver::ConjugateGradient,
        1.0e-12,
        1.0e-14,
        NonZeroUsize::new(500).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Identity)
    .with_reduction(ReductionPolicy::Reproducible)
}

fn nonlinear_solver() -> eqiora_solver::SolverPlan {
    SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1.0e-9,
        1.0e-11,
        NonZeroUsize::new(500).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Identity)
    .with_reduction(ReductionPolicy::Reproducible)
}

#[derive(Debug)]
struct NoSolveGeneralBackend;

impl LinearSolverBackend for NoSolveGeneralBackend {
    fn provider(&self) -> SolverProvider {
        SolverProvider::new(
            BackendId::new("eqiora.test.no-solve-general"),
            env!("CARGO_PKG_VERSION"),
            &[],
        )
    }

    fn capabilities(&self) -> SolverCapabilities {
        SolverCapabilities::exact([SolverCapability {
            scalar_domain: eqiora_core::ScalarDomain::Real,
            algorithm: LinearSolver::BiConjugateGradientStabilized,
            operator_properties: LinearOperatorProperties::General,
            preconditioner: PreconditionerPolicy::Identity,
            reduction: ReductionPolicy::Reproducible,
            scalar_type: ScalarType::F64,
        }])
        .unwrap()
    }

    fn solve_with_execution(
        &self,
        _problem: &LinearProblem<'_>,
        _plan: SolverPlan,
        _execution: &dyn ReplicatedLinearExecution,
    ) -> Result<LinearSolution, Diagnostic> {
        Err(Diagnostic::error(
            codes::NUMERICAL_SOLVE_FAILED,
            "zero-equilibrium bridge test must not invoke its linear backend",
        ))
    }
}
