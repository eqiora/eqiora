#![cfg(feature = "faer")]

use eqiora_core::ScalarType;
use std::collections::BTreeMap;
use std::num::{NonZeroU16, NonZeroU32, NonZeroUsize};

use eqiora::api::ModelDocument;
use eqiora::artifact::{
    DiscreteFieldEnvelopeV1, FieldSnapshotEnvelopeV2, GeometryIdentityEnvelopeV1,
    GeometryMeshCorrespondenceEnvelopeV1, GeometryStateEnvelopeV1, LayoutArtifacts, ModelEnvelope,
    RealizationEnvelopeV6, SimplicialMeshEnvelopeV1, SpatialStateEnvelopeV2,
    SpatialTrajectoryEnvelopeV2, SpatialTrajectorySegmentEnvelopeV2,
    ValidatedMovingSpatialContextV2,
};
use eqiora::backends::faer::FaerLinearSolver;
use eqiora::meshing::{
    CellId, DiscreteFieldAssociation, DiscreteFieldPayload, DiscreteFieldShape, FacetId,
    FixedTopologyGeometryAction2d, MeshEntity, MeshQualityGate, MeshTopology, SimplicialMesh,
    VertexId,
};
use eqiora::realization::{
    AleGeometryQualityGate, AlgebraicBlockScale, BackwardEulerRelationStep,
    BackwardEulerStateBinding, BackwardEulerStatePair, BackwardEulerStep, ConformingTraceQuotient,
    CoupledFieldwiseRealizationPlan, CoupledFieldwiseSpatialDiscretization, Discretization,
    DiscretizationMethod, DomainFieldDiscretization, ExecutionSchedule, FieldSpaceBinding,
    FixedTopologyAleCoupledRealizationPlan, FixedTopologyAleCoupledRealizationRequest,
    GclCompatibleAlePullback, MeshArtifactReference, MeshKind, MeshPolicy, NonlinearSolvePlan,
    P1HarmonicMeshMotionPolicy, PositivePhysicalScale, QuadraturePolicy, RealizationCapabilities,
    RealizationRevision, ResolvedFixedTopologyAleCoupledRealization, SemanticRevision, SolveRoot,
    Space, SpatialDimensionSupport, SymmetricCongruenceScaling, Target, TargetCapabilities,
    TraceFieldEndpoint, VectorLayoutKind, resolve_fixed_topology_ale_coupled,
};
use eqiora::solver::{
    AlgebraicBlock, LinearOperatorProperties, LinearSolver, PreconditionerPolicy, ReductionPolicy,
    SolverCapabilities, SolverCapability, SolverPlan,
};
use eqiora::{DimExponents, DynQuantity, Id, kinds};
use eqiora_numerics::{
    ale::AleFsiBoundary, ale::AleFsiCartesianModel, ale::AleFsiInitialPhysicalState,
    ale::AleFsiState, ale::AleFsiTrajectory, ale::P1HarmonicMeshMotionAction,
    ale::finalize_resolved_fixed_topology_ale_fsi_2d, ale::fixed_topology_ale_fsi_requirements_2d,
    ale::lower_ale_fsi_cartesian_2d, common::NonZeroStepCount, fsi::FixedReferenceFsiPartition,
    fsi::FixedReferenceFsiState,
};

const COMPONENTS: usize = 2;
const FINAL_TIME: f64 = 0.02;
const LENGTH: DimExponents =
    DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension");
const TIME: DimExponents =
    DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).expect("bounded dimension");
const VELOCITY: DimExponents =
    DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).expect("bounded dimension");
const PRESSURE: DimExponents =
    DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).expect("bounded dimension");
const WEAK_FUNCTIONAL: DimExponents =
    DimExponents::from_integers([1, 1, -3, 0, 0, 0, 0]).expect("bounded dimension");
const DIRECT_SOURCE: &str =
    include_str!("../../../verify/fsi/fixed-topology-ale-monolithic-2d/models/direct.eqi");

#[test]
fn faer_closes_moving_fsi_evidence_and_first_order_refinement() {
    let semantic =
        eqiora::api::ModelDocument::compile("fixed-topology-ale-fsi-direct.eqi", DIRECT_SOURCE)
            .unwrap();
    let canonical = lower_ale_fsi_cartesian_2d(semantic.program()).unwrap();
    assert_eq!(canonical.fluid().bounds(), &[[0.0, 1.0], [0.0, 1.0]]);
    assert_eq!(
        canonical.solid().continuum().bounds(),
        &[[1.0, 2.0], [0.0, 1.0]]
    );
    assert_eq!(canonical.fluid().mass_density(), 1.0);
    assert_eq!(canonical.fluid().dynamic_viscosity(), 0.2);
    assert_eq!(canonical.solid().mass_density(), 1.0);
    assert_eq!(canonical.solid().continuum().shear_modulus(), 2.0);
    assert_eq!(canonical.solid().continuum().first_lame_parameter(), 1.0);

    let fixture = Fixture::new(semantic, canonical);
    let model_reference = fixture.document.artifact_reference().unwrap();
    assert_eq!(model_reference.model(), fixture.canonical.model());
    assert_eq!(
        model_reference.semantic_revision().get(),
        fixture.canonical.semantic_revision()
    );
    assert_eq!(
        fixture.mesh_artifact.artifact_reference().unwrap(),
        fixture.mesh_reference
    );
    let coarse = fixture.advance(0.02, 1);
    let medium = fixture.advance(0.01, 2);
    let fine = fixture.advance(0.005, 4);
    let trajectory = &medium.trajectory;
    let fluid_interior = fixture.vertex([0.5, 0.5]);
    assert_eq!(
        medium.motion.fluid_interior_vertices(),
        &[VertexId::new(fluid_interior)]
    );
    assert_ne!(
        trajectory.initial_state().geometry().coordinates()[fluid_interior],
        fixture.mesh.vertices()[fluid_interior]
    );
    assert!(!medium.motion.driver_vertices().is_empty());
    assert!(!medium.motion.influence_solve_reports().is_empty());
    for report in medium.motion.influence_solve_reports() {
        assert_eq!(
            report.backend(),
            eqiora::solver::LinearSolverBackend::id(&FaerLinearSolver)
        );
        assert_eq!(report.algorithm(), LinearSolver::ConjugateGradient);
        assert!(report.true_residual_norm() <= report.residual_target());
    }

    assert_eq!(trajectory.states().len(), 3);
    assert_eq!(trajectory.steps().len(), 2);
    assert_eq!(trajectory.final_state().time(), FINAL_TIME);
    assert_ne!(
        vector_field(
            trajectory.initial_state(),
            solid_displacement(&fixture.canonical)
        ),
        vector_field(
            trajectory.final_state(),
            solid_displacement(&fixture.canonical)
        )
    );

    assert_harmonic_geometry_replays(&fixture, &medium.motion, trajectory);
    assert_consecutive_geometry_and_evidence(&fixture, trajectory, 0.01);
    assert_moving_artifact_dag_replays(&fixture, trajectory, 0.01);
    assert_static_geometry_falsifier(&fixture, &medium.motion);
    assert_eq!(coarse.trajectory.final_state().time(), FINAL_TIME);
    assert_eq!(trajectory.final_state().time(), FINAL_TIME);
    assert_eq!(fine.trajectory.final_state().time(), FINAL_TIME);

    let coarse_medium = solid_displacement_mass_distance(
        &fixture.mesh,
        &fixture.partition,
        solid_domain(&fixture.canonical),
        solid_displacement(&fixture.canonical),
        coarse.trajectory.final_state(),
        trajectory.final_state(),
    );
    let medium_fine = solid_displacement_mass_distance(
        &fixture.mesh,
        &fixture.partition,
        solid_domain(&fixture.canonical),
        solid_displacement(&fixture.canonical),
        trajectory.final_state(),
        fine.trajectory.final_state(),
    );
    let observed_order = (coarse_medium / medium_fine).log2();
    assert!(coarse_medium > medium_fine);
    assert!(
        observed_order > 0.75,
        "expected first-order refinement, observed p={observed_order:e} from {coarse_medium:e}/{medium_fine:e}"
    );

    let stale = MeshArtifactReference::from_sha256([155; 32]);
    let error = finalize_resolved_fixed_topology_ale_fsi_2d(
        &fixture.canonical,
        &fixture.resolve(0.01),
        stale,
        &fixture.mesh,
        &fixture.partition,
        &fixture.boundary,
        fixture.initial_physical(0.01),
        &FaerLinearSolver,
    )
    .expect_err("a foreign authenticated mesh digest must fail before execution");
    assert_eq!(error.code(), eqiora::diagnostic::codes::INVALID_REALIZATION);
    assert!(error.message().contains("authenticated mesh digest"));
}

struct Fixture {
    document: ModelDocument,
    canonical: AleFsiCartesianModel<2>,
    mesh_artifact: SimplicialMeshEnvelopeV1,
    mesh_reference: MeshArtifactReference,
    mesh: SimplicialMesh,
    partition: FixedReferenceFsiPartition<2>,
    boundary: AleFsiBoundary<2>,
}

impl Fixture {
    fn new(document: ModelDocument, canonical: AleFsiCartesianModel<2>) -> Self {
        let mesh = two_domain_mesh();
        let mesh_artifact = SimplicialMeshEnvelopeV1::from_mesh(&mesh).unwrap();
        let mesh_reference = mesh_artifact.artifact_reference().unwrap();
        assert_eq!(mesh_artifact.mesh(), &mesh);
        assert_eq!(document.program().model(), canonical.model());
        let (fluid, solid, interface) = inventories(&mesh);
        let _ = interface;
        let partition = FixedReferenceFsiPartition::<2>::new(
            &mesh,
            [
                (fluid_domain(&canonical), fluid),
                (solid_domain(&canonical), solid),
            ],
            &[trace_quotient(&canonical)],
        )
        .unwrap();
        let boundary = AleFsiBoundary::<2>::homogeneous_exterior(&mesh).unwrap();
        Self {
            document,
            canonical,
            mesh_artifact,
            mesh_reference,
            mesh,
            partition,
            boundary,
        }
    }

    fn initial_physical(&self, time_step: f64) -> AleFsiInitialPhysicalState<2> {
        AleFsiInitialPhysicalState::<2>::new(0.0, self.physical_state(time_step, [0.0015, 0.0005]))
            .unwrap()
    }

    fn physical_state(
        &self,
        time_step: f64,
        interface_displacement: [f64; COMPONENTS],
    ) -> FixedReferenceFsiState<2> {
        let interface_vertex = find_vertex(&self.mesh, [1.0, 0.5]);
        let vector = |domain, value: [f64; COMPONENTS]| {
            self.partition
                .domain_vertices(domain)
                .unwrap()
                .iter()
                .flat_map(move |vertex| {
                    value
                        .into_iter()
                        .enumerate()
                        .map(move |(component, value)| {
                            (MeshEntity::new(0, vertex.index()), 0, component, value)
                        })
                })
                .collect::<Vec<_>>()
        };
        let mut fluid_values = vector(fluid_domain(&self.canonical), [0.0; COMPONENTS]);
        for cell in self
            .partition
            .domain_cells(fluid_domain(&self.canonical))
            .unwrap()
        {
            for component in 0..COMPONENTS {
                fluid_values.push((MeshEntity::new(2, cell.index()), 0, component, 0.0));
            }
        }
        let mut displacement = vector(solid_domain(&self.canonical), [0.0; COMPONENTS]);
        for (entity, _, component, value) in &mut displacement {
            if entity.index() == interface_vertex {
                *value = interface_displacement[*component];
            }
        }
        FixedReferenceFsiState::new(
            self.document.program(),
            realization_plan(&self.canonical, self.mesh_reference, time_step).coupled(),
            &self.mesh,
            &self.partition,
            [
                (fluid_velocity(&self.canonical), fluid_values),
                (
                    fluid_pressure(&self.canonical),
                    self.partition
                        .domain_vertices(fluid_domain(&self.canonical))
                        .unwrap()
                        .iter()
                        .map(|vertex| (MeshEntity::new(0, vertex.index()), 0, 0, 0.0))
                        .collect(),
                ),
                (
                    solid_velocity(&self.canonical),
                    vector(solid_domain(&self.canonical), [0.0; COMPONENTS]),
                ),
                (solid_displacement(&self.canonical), displacement),
            ],
        )
        .unwrap()
    }

    fn vertex(&self, coordinate: [f64; COMPONENTS]) -> usize {
        find_vertex(&self.mesh, coordinate)
    }

    fn resolve(&self, time_step: f64) -> ResolvedFixedTopologyAleCoupledRealization {
        resolve_fixed_topology_ale_coupled(
            &FixedTopologyAleCoupledRealizationRequest::explicit(
                self.canonical.model(),
                SemanticRevision::new(self.canonical.semantic_revision()),
                RealizationRevision::new(1),
                realization_plan(&self.canonical, self.mesh_reference, time_step),
            ),
            fixed_topology_ale_fsi_requirements_2d(&self.canonical),
            &capabilities(),
        )
        .unwrap()
    }

    fn advance(&self, time_step: f64, steps: usize) -> ExecutedTrajectory {
        let resolved = self.resolve(time_step);
        let finalized = finalize_resolved_fixed_topology_ale_fsi_2d(
            &self.canonical,
            &resolved,
            self.mesh_reference,
            &self.mesh,
            &self.partition,
            &self.boundary,
            self.initial_physical(time_step),
            &FaerLinearSolver,
        )
        .unwrap();
        assert_eq!(finalized.model(), self.canonical.model());
        assert_eq!(
            finalized.semantic_revision().get(),
            self.canonical.semantic_revision()
        );
        assert_eq!(finalized.mesh_artifact(), self.mesh_reference);
        assert!(matches!(
            finalized.realization_graph().root(),
            SolveRoot::Nonlinear(_)
        ));
        assert_eq!(finalized.realization_graph().geometry_actions().len(), 1);
        let fields = finalized.fields();
        assert_eq!(fields.fluid_velocity(), fluid_velocity(&self.canonical));
        assert_eq!(fields.fluid_pressure(), fluid_pressure(&self.canonical));
        assert_eq!(fields.solid_velocity(), solid_velocity(&self.canonical));
        assert_eq!(
            fields.solid_displacement(),
            solid_displacement(&self.canonical)
        );
        let motion = finalized.motion().clone();
        let trajectory = finalized
            .solve(
                NonZeroStepCount::new(NonZeroUsize::new(steps).unwrap()),
                &FaerLinearSolver,
            )
            .unwrap();
        ExecutedTrajectory { motion, trajectory }
    }
}

struct ExecutedTrajectory {
    motion: P1HarmonicMeshMotionAction<2>,
    trajectory: AleFsiTrajectory<2>,
}

struct MovingSnapshotSet {
    snapshots: Vec<FieldSnapshotEnvelopeV2>,
    blocks: Vec<(Id<kinds::Field>, Vec<DiscreteFieldEnvelopeV1>)>,
}

impl MovingSnapshotSet {
    fn snapshot(&self, field: Id<kinds::Field>) -> &FieldSnapshotEnvelopeV2 {
        self.snapshots
            .iter()
            .find(|snapshot| snapshot.field() == field)
            .expect("the complete ALE Field inventory contains this Field")
    }

    fn blocks(&self, field: Id<kinds::Field>) -> &[DiscreteFieldEnvelopeV1] {
        self.blocks
            .iter()
            .find_map(|(candidate, blocks)| (*candidate == field).then_some(blocks.as_slice()))
            .expect("the complete ALE Field inventory retains its normalized blocks")
    }
}

fn assert_moving_artifact_dag_replays(
    fixture: &Fixture,
    trajectory: &AleFsiTrajectory<2>,
    time_step: f64,
) {
    let model = ModelEnvelope::from_program(fixture.document.program()).unwrap();
    assert_eq!(
        model.canonical_json().unwrap(),
        fixture.document.canonical_json().unwrap()
    );
    let geometry = GeometryIdentityEnvelopeV1::new(
        &model,
        [
            fluid_domain(&fixture.canonical),
            solid_domain(&fixture.canonical),
        ],
        1.0e-12,
    )
    .unwrap();
    let correspondence =
        GeometryMeshCorrespondenceEnvelopeV1::new(&geometry, &model, &fixture.mesh_artifact)
            .unwrap();
    let resolved = fixture.resolve(time_step);
    let realization =
        RealizationEnvelopeV6::from_resolved(&model, &resolved, LayoutArtifacts::Replicated)
            .unwrap();
    let context = ValidatedMovingSpatialContextV2::new(
        &model,
        &realization,
        &geometry,
        &correspondence,
        &fixture.mesh_artifact,
    )
    .unwrap();

    let snapshots = trajectory
        .states()
        .iter()
        .map(|state| moving_snapshots(fixture, &context, state))
        .collect::<Vec<_>>();
    for snapshot_set in &snapshots {
        for snapshot in &snapshot_set.snapshots {
            snapshot
                .validate_against_moving(&context, snapshot_set.blocks(snapshot.field()))
                .unwrap();
        }
    }

    let mut geometry_states = Vec::with_capacity(trajectory.states().len());
    let mut spatial_states = Vec::with_capacity(trajectory.states().len());
    for (step, (state, snapshot_set)) in trajectory.states().iter().zip(&snapshots).enumerate() {
        let predecessor = geometry_states.last();
        let geometry_state = GeometryStateEnvelopeV1::new(
            &model,
            &geometry,
            &correspondence,
            &fixture.mesh_artifact,
            &realization,
            step as u64,
            state.time(),
            predecessor,
            snapshot_set.snapshot(solid_displacement(&fixture.canonical)),
            state
                .geometry()
                .coordinates()
                .iter()
                .map(|coordinate| coordinate.to_vec())
                .collect(),
        )
        .unwrap();
        let spatial_state = SpatialStateEnvelopeV2::new(
            &context,
            &geometry_state,
            predecessor,
            &snapshot_set.snapshots,
            (),
        )
        .unwrap();
        spatial_state
            .validate_against(
                &context,
                &geometry_state,
                predecessor,
                &snapshot_set.snapshots,
                (),
            )
            .unwrap();
        geometry_states.push(geometry_state);
        spatial_states.push(spatial_state);
    }

    assert!(
        SpatialStateEnvelopeV2::new(
            &context,
            &geometry_states[0],
            None,
            &snapshots[1].snapshots,
            (),
        )
        .is_err(),
        "a GeometryState cannot be paired with a substituted numerical snapshot inventory"
    );

    let initial_segment =
        SpatialTrajectorySegmentEnvelopeV2::new(&context, &spatial_states[..1]).unwrap();
    let continuation_segment =
        SpatialTrajectorySegmentEnvelopeV2::new(&context, &spatial_states[1..]).unwrap();
    initial_segment
        .validate_against(&context, &spatial_states[..1])
        .unwrap();
    continuation_segment
        .validate_against(&context, &spatial_states[1..])
        .unwrap();

    let initial_root = SpatialTrajectoryEnvelopeV2::start(&context, &initial_segment).unwrap();
    let final_root =
        SpatialTrajectoryEnvelopeV2::extend(&context, &initial_root, &continuation_segment)
            .unwrap();
    initial_root
        .validate_against(&context, None, std::slice::from_ref(&initial_segment))
        .unwrap();
    final_root
        .validate_against(
            &context,
            Some(&initial_root),
            &[initial_segment.clone(), continuation_segment.clone()],
        )
        .unwrap();
    assert_eq!(final_root.generation(), 1);
    assert_eq!(
        final_root.previous_root(),
        Some(initial_root.digest().unwrap())
    );
    assert_eq!(final_root.last_step(), 2);
    assert_eq!(
        final_root.last_geometry_state(),
        geometry_states[2].digest().unwrap()
    );

    let bytes = final_root.canonical_json().unwrap();
    let decoded = SpatialTrajectoryEnvelopeV2::from_json(&bytes, Default::default()).unwrap();
    assert_eq!(decoded.canonical_json().unwrap(), bytes);
    assert_eq!(decoded.digest().unwrap(), final_root.digest().unwrap());
    decoded
        .validate_segments(&context, &[initial_segment, continuation_segment])
        .unwrap();
}

fn moving_snapshots(
    fixture: &Fixture,
    context: &ValidatedMovingSpatialContextV2<'_, ModelEnvelope>,
    state: &AleFsiState<2>,
) -> MovingSnapshotSet {
    let vector = DiscreteFieldShape::Vector {
        components: NonZeroU32::new(COMPONENTS as u32).unwrap(),
    };
    let mut blocks = Vec::new();

    let mut fluid_vertex_velocity = vec![[0.0; COMPONENTS]; fixture.mesh.vertices().len()];
    let fluid_velocity_values = vector_field(state, fluid_velocity(&fixture.canonical));
    for vertex in fixture
        .partition
        .domain_vertices(fluid_domain(&fixture.canonical))
        .unwrap()
    {
        fluid_vertex_velocity[vertex.index()] =
            fluid_velocity_values[&MeshEntity::new(0, vertex.index())];
    }
    let mut fluid_cell_velocity = vec![[0.0; COMPONENTS]; fixture.mesh.cells().len()];
    for (entity, value) in &fluid_velocity_values {
        if entity.dimension() == 2 {
            fluid_cell_velocity[entity.index()] = *value;
        }
    }
    blocks.push((
        fluid_velocity(&fixture.canonical),
        vec![
            discrete_block(
                &fixture.mesh_artifact,
                DiscreteFieldAssociation::Vertex,
                vector,
                flatten_vectors(&fluid_vertex_velocity),
            ),
            discrete_block(
                &fixture.mesh_artifact,
                DiscreteFieldAssociation::Cell,
                vector,
                flatten_vectors(&fluid_cell_velocity),
            ),
        ],
    ));

    let mut pressure = vec![0.0; fixture.mesh.vertices().len()];
    for (entity, value) in scalar_field(state, fluid_pressure(&fixture.canonical)) {
        pressure[entity.index()] = value;
    }
    blocks.push((
        fluid_pressure(&fixture.canonical),
        vec![discrete_block(
            &fixture.mesh_artifact,
            DiscreteFieldAssociation::Vertex,
            DiscreteFieldShape::Scalar,
            pressure,
        )],
    ));

    let mut solid_velocity_values = vec![[0.0; COMPONENTS]; fixture.mesh.vertices().len()];
    let mut solid_displacement_values = vec![[0.0; COMPONENTS]; fixture.mesh.vertices().len()];
    let velocity = vector_field(state, solid_velocity(&fixture.canonical));
    let displacement = vector_field(state, solid_displacement(&fixture.canonical));
    for vertex in fixture
        .partition
        .domain_vertices(solid_domain(&fixture.canonical))
        .unwrap()
    {
        let entity = MeshEntity::new(0, vertex.index());
        solid_velocity_values[vertex.index()] = velocity[&entity];
        solid_displacement_values[vertex.index()] = displacement[&entity];
    }
    blocks.push((
        solid_velocity(&fixture.canonical),
        vec![discrete_block(
            &fixture.mesh_artifact,
            DiscreteFieldAssociation::Vertex,
            vector,
            flatten_vectors(&solid_velocity_values),
        )],
    ));
    blocks.push((
        solid_displacement(&fixture.canonical),
        vec![discrete_block(
            &fixture.mesh_artifact,
            DiscreteFieldAssociation::Vertex,
            vector,
            flatten_vectors(&solid_displacement_values),
        )],
    ));

    let snapshots = blocks
        .iter()
        .map(|(field, field_blocks)| {
            FieldSnapshotEnvelopeV2::new_moving(context, *field, field_blocks).unwrap()
        })
        .collect();
    MovingSnapshotSet { snapshots, blocks }
}

fn discrete_block(
    mesh: &SimplicialMeshEnvelopeV1,
    association: DiscreteFieldAssociation,
    shape: DiscreteFieldShape,
    values: Vec<f64>,
) -> DiscreteFieldEnvelopeV1 {
    let payload = DiscreteFieldPayload::new(mesh.mesh(), association, shape, values).unwrap();
    DiscreteFieldEnvelopeV1::from_payload(mesh, &payload).unwrap()
}

fn flatten_vectors(values: &[[f64; COMPONENTS]]) -> Vec<f64> {
    values
        .iter()
        .flat_map(|value| value.iter().copied())
        .collect()
}

fn assert_harmonic_geometry_replays(
    fixture: &Fixture,
    motion: &P1HarmonicMeshMotionAction<2>,
    trajectory: &AleFsiTrajectory<2>,
) {
    for state in trajectory.states() {
        state
            .validate_against(&fixture.mesh, &fixture.partition, motion)
            .unwrap();
        let field = solid_displacement(&fixture.canonical);
        let displacement = motion.apply(field, &vector_vertices(state, field)).unwrap();
        for ((reference, displacement), current) in fixture
            .mesh
            .vertices()
            .iter()
            .zip(displacement)
            .zip(state.geometry().coordinates())
        {
            for component in 0..COMPONENTS {
                assert_eq!(
                    current[component].to_bits(),
                    (reference[component] + displacement[component]).to_bits()
                );
            }
        }
    }
}

fn assert_consecutive_geometry_and_evidence(
    fixture: &Fixture,
    trajectory: &AleFsiTrajectory<2>,
    time_step: f64,
) {
    let finalized = finalize_resolved_fixed_topology_ale_fsi_2d(
        &fixture.canonical,
        &fixture.resolve(time_step),
        fixture.mesh_reference,
        &fixture.mesh,
        &fixture.partition,
        &fixture.boundary,
        fixture.initial_physical(time_step),
        &FaerLinearSolver,
    )
    .unwrap();
    for (states, evidence) in trajectory.states().windows(2).zip(trajectory.steps()) {
        let action = FixedTopologyGeometryAction2d::new(
            &fixture.mesh,
            states[0].geometry(),
            states[1].geometry(),
            time_step,
        )
        .unwrap();
        assert_eq!(evidence.accepted_time(), states[1].time());
        assert!(evidence.final_residual_norm() <= evidence.residual_target());
        assert!(evidence.continuity_residual_norm() <= evidence.residual_target() + 1.0e-8);
        assert!(evidence.kinematic_residual_norm() < 1.0e-12);
        assert_eq!(evidence.interface_velocity_jump_norm(), 0.0);
        assert!(evidence.interface_action_imbalance_norm() < 1.0e-6);
        assert!(evidence.interface_power_imbalance() < 1.0e-6);
        assert!(evidence.maximum_affine_metric_identity_defect() < 1.0e-10);
        assert!(evidence.minimum_current_mean_ratio() > 0.3);
        assert!(evidence.minimum_current_signed_jacobian() > 0.0);
        assert!(evidence.minimum_path_signed_jacobian() > 0.0);
        let (column_count, color_count, singleton_count, assembly_count, maximum_error) = finalized
            .verify_accepted_jacobian(&states[0], &states[1])
            .unwrap();
        assert!(maximum_error < 1.0e-3);
        assert!(column_count > color_count);
        assert_eq!(assembly_count, 2 * color_count);
        assert_eq!(singleton_count, 2);
        assert!(assembly_count < 2 * column_count);
        assert!(evidence.probed_moving_fluid_cell_count() > 0);
        assert!(evidence.gcl_active_moving_fluid_cell_count() > 0);
        assert!(evidence.compatible_constant_free_stream_residual_norm() < 1.0e-12);
        assert!(evidence.omitted_gcl_witness_norm() > 1.0e-8);
        assert_eq!(
            evidence.nonlinear_linear_solves().len(),
            evidence.nonlinear_iterations()
        );
        for report in evidence.nonlinear_linear_solves() {
            assert_eq!(
                report.backend(),
                eqiora::solver::LinearSolverBackend::id(&FaerLinearSolver)
            );
            assert_eq!(
                report.algorithm(),
                LinearSolver::BiConjugateGradientStabilized
            );
            assert!(report.true_residual_norm() <= report.residual_target());
        }
        let solid_velocity_values = vector_field(&states[1], solid_velocity(&fixture.canonical));
        for vertex in fixture
            .partition
            .domain_vertices(solid_domain(&fixture.canonical))
            .unwrap()
        {
            let exact_velocity = solid_velocity_values[&MeshEntity::new(0, vertex.index())];
            for (component, exact_component) in exact_velocity.iter().enumerate() {
                let quotient = (states[1].geometry().coordinates()[vertex.index()][component]
                    - states[0].geometry().coordinates()[vertex.index()][component])
                    / time_step;
                let velocity = action.vertex_velocities()[vertex.index()][component];
                assert_eq!(velocity.to_bits(), quotient.to_bits());
                let scale = 1.0_f64.max(velocity.abs());
                assert!((velocity - exact_component).abs() < 2.0e-13 * scale);
            }
        }
        for cell in action.cells() {
            let scale = cell
                .endpoint_metric_rate()
                .abs()
                .max(cell.current_map().measure_scale())
                .max(1.0);
            assert!(cell.metric_identity_defect().abs() < 1.0e-12 * scale);
            assert!(cell.minimum_path_signed_measure_scale() > 0.0);
        }
    }
}

fn assert_static_geometry_falsifier(fixture: &Fixture, motion: &P1HarmonicMeshMotionAction<2>) {
    let zero = vec![[0.0; COMPONENTS]; fixture.mesh.vertices().len()];
    let physical = fixture.physical_state(0.02, [0.0; COMPONENTS]);
    let displacement = physical_vector_vertices(&physical, solid_displacement(&fixture.canonical));
    assert_eq!(
        motion
            .apply(solid_displacement(&fixture.canonical), &displacement)
            .unwrap(),
        zero
    );
    let state =
        AleFsiState::<2>::new(0.0, &fixture.mesh, &fixture.partition, motion, physical).unwrap();
    assert_eq!(state.geometry().coordinates(), fixture.mesh.vertices());
    let action =
        FixedTopologyGeometryAction2d::new(&fixture.mesh, state.geometry(), state.geometry(), 0.02)
            .unwrap();
    assert!(
        action
            .vertex_velocities()
            .iter()
            .flatten()
            .all(|value| *value == 0.0)
    );
    for cell in action.cells() {
        assert_eq!(cell.previous_map(), cell.current_map());
        assert_eq!(cell.current_velocity_divergence(), 0.0);
        assert_eq!(cell.skew_gcl_correction(), 0.0);
        assert_eq!(cell.endpoint_metric_rate(), 0.0);
    }
}

fn solid_displacement_mass_distance(
    mesh: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<2>,
    domain: Id<kinds::Domain>,
    field: Id<kinds::Field>,
    left: &AleFsiState<2>,
    right: &AleFsiState<2>,
) -> f64 {
    let mut squared = 0.0;
    let left_displacement = vector_field(left, field);
    let right_displacement = vector_field(right, field);
    for cell in partition.domain_cells(domain).unwrap() {
        let vertices = mesh
            .entity_vertices(MeshEntity::new(2, cell.index()))
            .unwrap();
        let coordinates = vertices
            .iter()
            .map(|vertex| &mesh.vertices()[vertex.index()])
            .collect::<Vec<_>>();
        let twice_area = (coordinates[1][0] - coordinates[0][0])
            * (coordinates[2][1] - coordinates[0][1])
            - (coordinates[2][0] - coordinates[0][0]) * (coordinates[1][1] - coordinates[0][1]);
        let area = 0.5 * twice_area.abs();
        for row in 0..3 {
            for column in 0..3 {
                let mass = area / 12.0 * if row == column { 2.0 } else { 1.0 };
                let row_vertex = vertices[row].index();
                let column_vertex = vertices[column].index();
                for component in 0..COMPONENTS {
                    let row = MeshEntity::new(0, row_vertex);
                    let column = MeshEntity::new(0, column_vertex);
                    let row_difference =
                        left_displacement[&row][component] - right_displacement[&row][component];
                    let column_difference = left_displacement[&column][component]
                        - right_displacement[&column][component];
                    squared += mass * row_difference * column_difference;
                }
            }
        }
    }
    assert!(squared.is_finite() && squared > 0.0);
    squared.sqrt()
}

fn harmonic_solver_plan() -> SolverPlan {
    SolverPlan::new(
        LinearSolver::ConjugateGradient,
        1.0e-12,
        1.0e-14,
        NonZeroUsize::new(500).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Identity)
    .with_reduction(ReductionPolicy::Fast)
}

fn nonlinear_solver_plan() -> SolverPlan {
    SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1.0e-9,
        1.0e-11,
        NonZeroUsize::new(2_000).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Identity)
    .with_reduction(ReductionPolicy::Fast)
}

fn realization_plan(
    model: &AleFsiCartesianModel<2>,
    mesh_artifact: MeshArtifactReference,
    time_step: f64,
) -> FixedTopologyAleCoupledRealizationPlan {
    let p1 = Space::continuous_lagrange(NonZeroU16::MIN);
    let length = physical_scale(2.0, LENGTH);
    let velocity = physical_scale(1.0, VELOCITY);
    let pressure = physical_scale(1.0, PRESSURE);
    let duration = DynQuantity::new(time_step, TIME);
    let coupled = CoupledFieldwiseRealizationPlan::new(
        CoupledFieldwiseSpatialDiscretization::new(
            length,
            [
                DomainFieldDiscretization::new(
                    fluid_domain(model),
                    [
                        FieldSpaceBinding::new(fluid_velocity(model), Space::simplex_p1_bubble()),
                        FieldSpaceBinding::new(fluid_pressure(model), p1),
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
                    artifact: mesh_artifact,
                },
                QuadraturePolicy::TriangleDuffyGaussLegendre {
                    points_per_axis: NonZeroUsize::new(5).unwrap(),
                },
            ),
        )
        .unwrap(),
        BackwardEulerStep::new(
            duration,
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
                AlgebraicBlockScale::new(AlgebraicBlock::Field(fluid_pressure(model)), pressure),
                AlgebraicBlockScale::new(AlgebraicBlock::Field(solid_velocity(model)), velocity),
            ],
            physical_scale(2.0, WEAK_FUNCTIONAL),
        )
        .unwrap(),
        LinearOperatorProperties::General,
        nonlinear_solver_plan(),
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
        ExecutionSchedule::Offline,
    )
    .unwrap();
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
            AleGeometryQualityGate::new(0.3).unwrap(),
            harmonic_solver_plan(),
        )
        .unwrap(),
        GclCompatibleAlePullback::new(fluid_relation(model), fluid_velocity(model)),
        NonlinearSolvePlan::new(1.0e-7, 1.0e-10, NonZeroUsize::new(20).unwrap(), 16).unwrap(),
    )
    .unwrap()
}

fn capabilities() -> RealizationCapabilities {
    RealizationCapabilities::cartesian_product(
        [DiscretizationMethod::ContinuousGalerkin],
        [(
            MeshKind::ImportedAffineSimplicial,
            SpatialDimensionSupport::exact(NonZeroUsize::new(2).unwrap()),
        )],
        [VectorLayoutKind::Replicated],
        SolverCapabilities::exact([
            SolverCapability {
                algorithm: LinearSolver::BiConjugateGradientStabilized,
                operator_properties: LinearOperatorProperties::General,
                preconditioner: PreconditionerPolicy::Identity,
                reduction: ReductionPolicy::Fast,
                scalar_type: ScalarType::F64,
            },
            SolverCapability {
                algorithm: LinearSolver::ConjugateGradient,
                operator_properties: LinearOperatorProperties::SymmetricPositiveDefinite,
                preconditioner: PreconditionerPolicy::Identity,
                reduction: ReductionPolicy::Fast,
                scalar_type: ScalarType::F64,
            },
        ])
        .unwrap(),
        TargetCapabilities::none().with_host_cpu(NonZeroUsize::MIN),
    )
    .unwrap()
}

fn physical_scale(value: f64, dimension: DimExponents) -> PositivePhysicalScale {
    PositivePhysicalScale::new(DynQuantity::new(value, dimension)).unwrap()
}

fn fluid_domain(model: &AleFsiCartesianModel<2>) -> Id<kinds::Domain> {
    model.fluid().domain().downcast().unwrap()
}

fn solid_domain(model: &AleFsiCartesianModel<2>) -> Id<kinds::Domain> {
    model.solid().continuum().domain().downcast().unwrap()
}

fn fluid_velocity(model: &AleFsiCartesianModel<2>) -> Id<kinds::Field> {
    model.fluid().velocity().downcast().unwrap()
}

fn fluid_pressure(model: &AleFsiCartesianModel<2>) -> Id<kinds::Field> {
    model.fluid().pressure().downcast().unwrap()
}

fn solid_velocity(model: &AleFsiCartesianModel<2>) -> Id<kinds::Field> {
    model.solid().velocity().downcast().unwrap()
}

fn solid_displacement(model: &AleFsiCartesianModel<2>) -> Id<kinds::Field> {
    model.solid().continuum().displacement().downcast().unwrap()
}

fn fluid_relation(model: &AleFsiCartesianModel<2>) -> Id<kinds::Relation> {
    model.fluid().momentum_relation().downcast().unwrap()
}

fn solid_kinematic_relation(model: &AleFsiCartesianModel<2>) -> Id<kinds::Relation> {
    fixed_topology_ale_fsi_requirements_2d(model).solid_kinematic_relation()
}

fn connection(model: &AleFsiCartesianModel<2>) -> Id<kinds::Connection> {
    model.interface().connection().downcast().unwrap()
}

fn vector_field(
    state: &AleFsiState<2>,
    field: Id<kinds::Field>,
) -> BTreeMap<MeshEntity, [f64; COMPONENTS]> {
    vector_physical_field(state.physical_state(), field)
}

fn vector_physical_field(
    state: &FixedReferenceFsiState<2>,
    field: Id<kinds::Field>,
) -> BTreeMap<MeshEntity, [f64; COMPONENTS]> {
    let mut values = BTreeMap::<_, [Option<f64>; COMPONENTS]>::new();
    for (entity, slot, component, value) in state.coefficients(field).unwrap() {
        assert_eq!(slot, 0);
        values.entry(entity).or_insert([None; COMPONENTS])[component] = Some(value);
    }
    values
        .into_iter()
        .map(|(entity, values)| (entity, values.map(Option::unwrap)))
        .collect()
}

fn vector_vertices(
    state: &AleFsiState<2>,
    field: Id<kinds::Field>,
) -> BTreeMap<VertexId, [f64; COMPONENTS]> {
    physical_vector_vertices(state.physical_state(), field)
}

fn physical_vector_vertices(
    state: &FixedReferenceFsiState<2>,
    field: Id<kinds::Field>,
) -> BTreeMap<VertexId, [f64; COMPONENTS]> {
    vector_physical_field(state, field)
        .into_iter()
        .filter_map(|(entity, value)| {
            (entity.dimension() == 0).then_some((VertexId::new(entity.index()), value))
        })
        .collect()
}

fn scalar_field(state: &AleFsiState<2>, field: Id<kinds::Field>) -> BTreeMap<MeshEntity, f64> {
    state
        .physical_state()
        .coefficients(field)
        .unwrap()
        .map(|(entity, slot, component, value)| {
            assert_eq!((slot, component), (0, 0));
            (entity, value)
        })
        .collect()
}

fn trace_quotient(model: &AleFsiCartesianModel<2>) -> ConformingTraceQuotient {
    ConformingTraceQuotient::new(
        connection(model),
        TraceFieldEndpoint::new(fluid_domain(model), fluid_velocity(model)),
        TraceFieldEndpoint::new(solid_domain(model), solid_velocity(model)),
    )
    .unwrap()
}

fn state_pair(model: &AleFsiCartesianModel<2>) -> BackwardEulerStatePair {
    BackwardEulerStatePair::new(
        fixed_topology_ale_fsi_requirements_2d(model).solid_kinematic_relation(),
        solid_displacement(model),
        solid_velocity(model),
    )
    .unwrap()
}

fn two_domain_mesh() -> SimplicialMesh {
    let x_coordinates = [0.0, 0.5, 1.0, 1.5, 2.0];
    let mut vertices = Vec::new();
    for y in [0.0, 0.5, 1.0] {
        for x in x_coordinates {
            vertices.push(vec![x, y]);
        }
    }
    let width = x_coordinates.len();
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
    SimplicialMesh::new(2, vertices, cells, MeshQualityGate::new(0.3).unwrap()).unwrap()
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

fn find_vertex(mesh: &SimplicialMesh, target: [f64; COMPONENTS]) -> usize {
    mesh.vertices()
        .iter()
        .position(|coordinates| coordinates.as_slice() == target)
        .unwrap()
}
