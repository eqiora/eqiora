mod mesh_3d;

use mesh_3d::{inventories_3d, mesh_3d, mesh_3d_with_upper_z};
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
use crate::simplicial_ale_fsi::AleFsiBoundary;
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
fn direct_tetrahedral_model_reaches_the_same_finalized_newton_boundary() {
    let program = compile_program(&ale_source_3d());
    let model = super::super::lower_ale_fsi_cartesian_3d(&program).unwrap();
    let mesh = mesh_3d(MeshQualityGate::new(0.1).unwrap());
    let (fluid, solid, interface) = inventories_3d(&mesh);
    let partition = FixedReferenceFsiPartition::<3>::new(
        &mesh,
        fluid.clone(),
        solid.clone(),
        interface.clone(),
    )
    .unwrap();
    let boundary = AleFsiBoundary::<3>::homogeneous_exterior(&mesh).unwrap();
    assert!(AleFsiBoundary::<2>::homogeneous_exterior(&mesh).is_err());
    assert!(
        FixedReferenceFsiPartition::<3>::new(
            &mesh,
            fluid,
            solid,
            interface[..interface.len() - 1].to_vec(),
        )
        .is_err()
    );
    let requirements = fixed_topology_ale_fsi_requirements_3d(&model);
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

    let mut fixed_velocity = vec![[0.0; 3]; mesh.vertices().len()];
    let fixed = boundary.fixed_zero_velocity_vertices()[0];
    fixed_velocity[fixed.index()][2] = 1.0;
    let invalid = AleFsiInitialPhysicalState::<3>::new(
        0.0,
        fixed_velocity,
        partition
            .fluid_cells()
            .iter()
            .copied()
            .map(|cell| (cell, [0.0; 3]))
            .collect(),
        vec![0.0; partition.fluid_vertices().len()],
        vec![[0.0; 3]; mesh.vertices().len()],
    )
    .unwrap();
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
    let (fluid, solid, interface) = inventories_3d(&short_mesh);
    let short_partition =
        FixedReferenceFsiPartition::<3>::new(&short_mesh, fluid, solid, interface).unwrap();
    let short_boundary = AleFsiBoundary::<3>::homogeneous_exterior(&short_mesh).unwrap();
    assert!(
        finalize_resolved_fixed_topology_ale_fsi_3d(
            &model,
            &resolved,
            mesh_reference(),
            &short_mesh,
            &short_partition,
            &short_boundary,
            initial_for_3d(&short_mesh, &short_partition),
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
            initial_for_3d(&mesh, &partition),
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
        fixed_topology_ale_fsi_requirements_2d(&fixture.model),
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
    let (fluid, solid, interface) = inventories(&lower_quality_mesh);
    let partition =
        FixedReferenceFsiPartition::<2>::new(&lower_quality_mesh, fluid, solid, interface).unwrap();
    let boundary = AleFsiBoundary::<2>::homogeneous_exterior(&lower_quality_mesh).unwrap();
    assert!(
        finalize_resolved_fixed_topology_ale_fsi_2d(
            &fixture.model,
            &fixture.resolved,
            mesh_reference(),
            &lower_quality_mesh,
            &partition,
            &boundary,
            initial_for(&lower_quality_mesh, &partition),
            &REFERENCE_LINEAR_SOLVER,
        )
        .unwrap_err()
        .message()
        .contains("exact resolved geometry-quality gate")
    );

    let mut velocity = vec![[0.0; 2]; fixture.mesh.vertices().len()];
    velocity[0] = [1.0, 0.0];
    let initial = AleFsiInitialPhysicalState::<2>::new(
        0.0,
        velocity,
        fixture
            .partition
            .fluid_cells()
            .iter()
            .copied()
            .map(|cell| (cell, [0.0; 2]))
            .collect(),
        vec![0.0; fixture.partition.fluid_vertices().len()],
        vec![[0.0; 2]; fixture.mesh.vertices().len()],
    )
    .unwrap();
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
    boundary: AleFsiBoundary<2>,
    resolved: ResolvedFixedTopologyAleCoupledRealization,
}

impl Fixture {
    fn new() -> Self {
        Self::from_source(&ale_source())
    }

    fn from_source(source: &str) -> Self {
        let program = compile_program(source);
        let model = super::super::lower_ale_fsi_cartesian_2d(&program).unwrap();
        let mesh = mesh(MeshQualityGate::new(0.3).unwrap());
        let (fluid, solid, interface) = inventories(&mesh);
        let partition =
            FixedReferenceFsiPartition::<2>::new(&mesh, fluid, solid, interface).unwrap();
        let boundary = AleFsiBoundary::<2>::homogeneous_exterior(&mesh).unwrap();
        let plan = Self::build_plan(&model, fluid_pressure(&model), 0.3);
        let resolved = resolve(
            &program,
            plan,
            fixed_topology_ale_fsi_requirements_2d(&model),
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
        initial_for(&self.mesh, &self.partition)
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
            fluid_domain(model),
            solid_domain(model),
            solid_displacement(model),
            connection(model),
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
    let requirements = fixed_topology_ale_fsi_requirements_2d(model);
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
                algorithm: LinearSolver::BiConjugateGradientStabilized,
                operator_properties: LinearOperatorProperties::General,
                preconditioner: PreconditionerPolicy::Identity,
                reduction: ReductionPolicy::Reproducible,
                scalar_type: ScalarType::F64,
            },
            SolverCapability {
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

fn initial_for(
    mesh: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<2>,
) -> AleFsiInitialPhysicalState<2> {
    AleFsiInitialPhysicalState::<2>::new(
        0.0,
        vec![[0.0; 2]; mesh.vertices().len()],
        partition
            .fluid_cells()
            .iter()
            .copied()
            .map(|cell| (cell, [0.0; 2]))
            .collect(),
        vec![0.0; partition.fluid_vertices().len()],
        vec![[0.0; 2]; mesh.vertices().len()],
    )
    .unwrap()
}

fn initial_for_3d(
    mesh: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<3>,
) -> AleFsiInitialPhysicalState<3> {
    AleFsiInitialPhysicalState::<3>::new(
        0.0,
        vec![[0.0; 3]; mesh.vertices().len()],
        partition
            .fluid_cells()
            .iter()
            .copied()
            .map(|cell| (cell, [0.0; 3]))
            .collect(),
        vec![0.0; partition.fluid_vertices().len()],
        vec![[0.0; 3]; mesh.vertices().len()],
    )
    .unwrap()
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
