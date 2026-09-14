#![cfg(feature = "faer")]

// The immutable-release target remains independently runnable, while the
// registered FSI case also executes its exact-package falsifiers.
#[path = "fsi_3d_package_releases.rs"]
mod package_authoring_evidence;

use eqiora_core::ScalarType;
use std::collections::BTreeMap;
use std::num::{NonZeroU16, NonZeroU32, NonZeroUsize};

use eqiora::api::ModelDocument;
use eqiora::artifact::{
    DiscreteFieldEnvelopeV1, ExecutionProvenanceV1, ExecutionTopologyV1, FieldSnapshotEnvelopeV2,
    GeometryIdentityEnvelopeV1, GeometryMeshCorrespondenceEnvelopeV1, GeometryStateEnvelopeV3,
    LayoutArtifacts, ModelEnvelope, RealizationEnvelopeV6, RunManifestV2, SimplicialMeshEnvelopeV1,
    SpatialStateEnvelopeV2, SpatialTrajectoryEnvelopeV2, SpatialTrajectorySegmentEnvelopeV2,
    ValidatedMovingSpatialContextV2,
};
use eqiora::backends::faer::FaerLinearSolver;
use eqiora::meshing::{
    CellId, DiscreteFieldAssociation, DiscreteFieldPayload, DiscreteFieldShape, FacetId,
    FixedTopologyGeometryAction, FixedTopologyGeometryState, MeshEntity, MeshQualityGate,
    MeshTopology, SimplicialMesh,
};
use eqiora::realization::{
    AleGeometryQualityGate, AlgebraicBlockScale, BackwardEulerRelationStep,
    BackwardEulerStateBinding, BackwardEulerStatePair, BackwardEulerStep, ConformingTraceQuotient,
    CoupledFieldwiseRealizationPlan, CoupledFieldwiseSpatialDiscretization, Discretization,
    DiscretizationMethod, DomainFieldDiscretization, ExecutionSchedule, FieldSpaceBinding,
    FixedTopologyAleCoupledRealizationPlan, FixedTopologyAleCoupledRealizationRequest,
    GclCompatibleAlePullback, MeshArtifactReference, MeshKind, MeshPolicy, NonlinearSolvePlan,
    P1HarmonicMeshMotionPolicy, PositivePhysicalScale, QuadraturePolicy, RealizationCapabilities,
    RealizationRevision, ResolvedFixedTopologyAleCoupledRealization, SemanticRevision, Space,
    SpatialDimensionSupport, SymmetricCongruenceScaling, Target, TargetCapabilities,
    TraceFieldEndpoint, VectorLayoutKind, resolve_fixed_topology_ale_coupled,
};
use eqiora::solver::{
    AlgebraicBlock, LinearOperatorProperties, LinearSolver, PreconditionerPolicy, ReductionPolicy,
    SolverCapabilities, SolverCapability, SolverPlan,
};
use eqiora::{DimExponents, DynQuantity, Id, kinds};
use eqiora_numerics::{
    ale::AleFsiBoundary, ale::AleFsiCartesianModel, ale::AleFsiInitialPhysicalState,
    ale::AleFsiState, ale::AleFsiTrajectory, ale::finalize_resolved_fixed_topology_ale_fsi_3d,
    ale::fixed_topology_ale_fsi_requirements_3d, ale::lower_ale_fsi_cartesian_3d,
    common::NonZeroStepCount, fsi::FixedReferenceFsiPartition, fsi::FixedReferenceFsiState,
};

const D: usize = 3;
const FINAL_TIME: f64 = 0.02;
const LENGTH: DimExponents =
    DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension");
const TIME: DimExponents =
    DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).expect("bounded dimension");
const VELOCITY: DimExponents =
    DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).expect("bounded dimension");
const PRESSURE: DimExponents =
    DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).expect("bounded dimension");
const WEAK_FUNCTIONAL_3D: DimExponents =
    DimExponents::from_integers([1, 2, -3, 0, 0, 0, 0]).expect("bounded dimension");
const DIRECT_SOURCE: &str =
    include_str!("../../../verify/fsi/fixed-topology-ale-monolithic-3d/models/direct.eqi");

#[test]
fn faer_closes_tetrahedral_trajectory_and_first_order_refinement() {
    let document =
        eqiora::api::ModelDocument::compile("fixed-topology-ale-fsi-direct-3d.eqi", DIRECT_SOURCE)
            .unwrap();
    let canonical = lower_ale_fsi_cartesian_3d(document.program()).unwrap();
    let fixture = Fixture::new(document, canonical);

    let coarse = fixture.advance(0.02, 1);
    let medium = fixture.advance(0.01, 2);
    let fine = fixture.advance(0.005, 4);
    assert_eq!(coarse.states().len(), 2);
    assert_eq!(medium.states().len(), 3);
    assert_eq!(fine.states().len(), 5);
    assert_eq!(coarse.final_state().time(), FINAL_TIME);
    assert_eq!(medium.final_state().time(), FINAL_TIME);
    assert_eq!(fine.final_state().time(), FINAL_TIME);

    for trajectory in [&coarse, &medium, &fine] {
        let finalized = finalize_resolved_fixed_topology_ale_fsi_3d(
            &fixture.canonical,
            &fixture.resolve(trajectory.states()[1].time() - trajectory.states()[0].time()),
            fixture.mesh_reference,
            &fixture.mesh,
            &fixture.partition,
            &fixture.boundary,
            fixture.initial(trajectory.states()[1].time() - trajectory.states()[0].time()),
            &FaerLinearSolver,
        )
        .unwrap();
        let mut previous_color_count = None;
        for (states, step) in trajectory.states().windows(2).zip(trajectory.steps()) {
            assert!(step.final_residual_norm() <= step.residual_target());
            assert!(step.continuity_residual_norm() <= step.residual_target() + 1.0e-8);
            assert!(step.kinematic_residual_norm() < 1.0e-12);
            assert_eq!(step.interface_velocity_jump_norm(), 0.0);
            assert!(step.maximum_affine_metric_identity_defect() < 1.0e-11);
            assert!(step.minimum_current_mean_ratio() > 0.0);
            assert!(step.minimum_current_signed_jacobian() > 0.0);
            assert!(step.minimum_path_signed_jacobian() > 0.0);
            let (column_count, color_count, singleton_count, assembly_count, maximum_error) =
                finalized
                    .verify_accepted_jacobian(&states[0], &states[1])
                    .unwrap();
            assert!(maximum_error < 1.0e-3);
            assert!(column_count > color_count);
            assert_eq!(assembly_count, 2 * color_count);
            assert_eq!(singleton_count, 3);
            assert!(assembly_count < 2 * column_count);
            if let Some(previous) = previous_color_count.replace(color_count) {
                assert_eq!(previous, color_count);
            }
            assert!(step.probed_moving_fluid_cell_count() > 0);
            assert!(step.gcl_active_moving_fluid_cell_count() > 0);
            assert!(step.compatible_constant_free_stream_residual_norm() < 1.0e-12);
            assert!(step.omitted_gcl_witness_norm() > 1.0e-10);
            assert!(step.interface_action_imbalance_norm() < 1.0e-9);
            assert!(step.interface_power_imbalance().abs() < 1.0e-9);
        }
    }

    let coarse_medium = solid_displacement_mass_distance(
        &fixture.mesh,
        &fixture.partition,
        solid_domain(&fixture.canonical),
        solid_displacement(&fixture.canonical),
        coarse.final_state(),
        medium.final_state(),
    );
    let medium_fine = solid_displacement_mass_distance(
        &fixture.mesh,
        &fixture.partition,
        solid_domain(&fixture.canonical),
        solid_displacement(&fixture.canonical),
        medium.final_state(),
        fine.final_state(),
    );
    let observed_order = (coarse_medium / medium_fine).log2();
    assert!(coarse_medium > medium_fine);
    assert!(
        observed_order > 0.70,
        "expected bounded first-order refinement, observed p={observed_order:e}"
    );

    publish_moving_artifact_dag(&fixture, &medium, 0.01);
}

struct Fixture {
    document: ModelDocument,
    canonical: AleFsiCartesianModel<3>,
    mesh_artifact: SimplicialMeshEnvelopeV1,
    mesh_reference: MeshArtifactReference,
    mesh: SimplicialMesh,
    partition: FixedReferenceFsiPartition<3>,
    boundary: AleFsiBoundary<3>,
}

impl Fixture {
    fn new(document: ModelDocument, canonical: AleFsiCartesianModel<3>) -> Self {
        let mesh = two_domain_tetrahedral_mesh();
        let mesh_artifact = eqiora::artifact::SimplicialMeshEnvelopeV1::from_mesh(&mesh).unwrap();
        let mesh_reference = mesh_artifact.artifact_reference().unwrap();
        let (fluid, solid, interface) = inventories(&mesh);
        let _ = interface;
        let partition = FixedReferenceFsiPartition::<3>::new(
            &mesh,
            [
                (fluid_domain(&canonical), fluid),
                (solid_domain(&canonical), solid),
            ],
            &[trace_quotient(&canonical)],
        )
        .unwrap();
        let boundary = AleFsiBoundary::<3>::homogeneous_exterior(&mesh).unwrap();
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

    fn initial(&self, time_step: f64) -> AleFsiInitialPhysicalState<3> {
        let vector = |domain, value: [f64; D]| {
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
        let mut displacement = vector(solid_domain(&self.canonical), [0.0; D]);
        let interface_center = find_vertex(&self.mesh, [1.0, 0.5, 0.5]);
        for (entity, _, component, value) in &mut displacement {
            if entity.index() == interface_center {
                *value = [0.0015, 0.0005, 0.00025][*component];
            }
        }
        let mut fluid_velocity_values = vector(fluid_domain(&self.canonical), [0.0; D]);
        for cell in self
            .partition
            .domain_cells(fluid_domain(&self.canonical))
            .unwrap()
        {
            for component in 0..D {
                fluid_velocity_values.push((MeshEntity::new(3, cell.index()), 0, component, 0.0));
            }
        }
        let physical = FixedReferenceFsiState::new(
            self.document.program(),
            realization_plan(&self.canonical, self.mesh_reference, time_step).coupled(),
            &self.mesh,
            &self.partition,
            [
                (fluid_velocity(&self.canonical), fluid_velocity_values),
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
                    vector(solid_domain(&self.canonical), [0.0; D]),
                ),
                (solid_displacement(&self.canonical), displacement),
            ],
        )
        .unwrap();
        AleFsiInitialPhysicalState::<3>::new(0.0, physical).unwrap()
    }

    fn resolve(&self, time_step: f64) -> ResolvedFixedTopologyAleCoupledRealization {
        resolve_fixed_topology_ale_coupled(
            &FixedTopologyAleCoupledRealizationRequest::explicit(
                self.canonical.model(),
                SemanticRevision::new(self.canonical.semantic_revision()),
                RealizationRevision::new(1),
                realization_plan(&self.canonical, self.mesh_reference, time_step),
            ),
            fixed_topology_ale_fsi_requirements_3d(&self.canonical),
            &capabilities(),
        )
        .unwrap()
    }

    fn advance(&self, time_step: f64, steps: usize) -> AleFsiTrajectory<3> {
        finalize_resolved_fixed_topology_ale_fsi_3d(
            &self.canonical,
            &self.resolve(time_step),
            self.mesh_reference,
            &self.mesh,
            &self.partition,
            &self.boundary,
            self.initial(time_step),
            &FaerLinearSolver,
        )
        .unwrap()
        .solve(
            NonZeroStepCount::new(NonZeroUsize::new(steps).unwrap()),
            &FaerLinearSolver,
        )
        .unwrap()
    }
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
            .expect("complete 3D ALE Field inventory")
    }

    fn blocks(&self, field: Id<kinds::Field>) -> &[DiscreteFieldEnvelopeV1] {
        self.blocks
            .iter()
            .find_map(|(candidate, blocks)| (*candidate == field).then_some(blocks.as_slice()))
            .expect("normalized 3D ALE Field blocks")
    }
}

fn publish_moving_artifact_dag(
    fixture: &Fixture,
    trajectory: &AleFsiTrajectory<3>,
    time_step: f64,
) {
    let model = ModelEnvelope::from_program(fixture.document.program()).unwrap();
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
    let realization = RealizationEnvelopeV6::from_resolved(
        &model,
        &fixture.resolve(time_step),
        LayoutArtifacts::Replicated,
    )
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
        let driver = snapshot_set.snapshot(solid_displacement(&fixture.canonical));
        let driver_blocks = snapshot_set.blocks(solid_displacement(&fixture.canonical));
        let geometry_state = GeometryStateEnvelopeV3::new(
            &context,
            step as u64,
            state.time(),
            predecessor,
            driver,
            driver_blocks,
            state
                .geometry()
                .coordinates()
                .iter()
                .map(|coordinate| coordinate.to_vec())
                .collect(),
        )
        .unwrap();
        let bytes = geometry_state.canonical_json().unwrap();
        let decoded = GeometryStateEnvelopeV3::from_json(&bytes, Default::default()).unwrap();
        assert_eq!(decoded.canonical_json().unwrap(), bytes);
        decoded
            .validate_against(&context, predecessor, driver, driver_blocks)
            .unwrap();
        let spatial_state = SpatialStateEnvelopeV2::new(
            &context,
            &geometry_state,
            predecessor,
            &snapshot_set.snapshots,
            driver_blocks,
        )
        .unwrap();
        spatial_state
            .validate_against(
                &context,
                &geometry_state,
                predecessor,
                &snapshot_set.snapshots,
                driver_blocks,
            )
            .unwrap();
        geometry_states.push(geometry_state);
        spatial_states.push(spatial_state);
    }

    assert_eq!(geometry_states.len(), 3);
    assert_eq!(spatial_states.len(), 3);
    assert_geometry_state_v3_replay_falsifiers(fixture, &context, &snapshots, &geometry_states);

    let mut positive_but_nondriven = trajectory.states()[1]
        .geometry()
        .coordinates()
        .iter()
        .map(|coordinate| coordinate.to_vec())
        .collect::<Vec<_>>();
    positive_but_nondriven[12][0] += 1.0e-4;
    assert!(
        FixedTopologyGeometryState::<3>::new(
            fixture.mesh_artifact.mesh(),
            positive_but_nondriven.clone(),
        )
        .is_ok(),
        "the falsifier must retain an admissible positively oriented endpoint",
    );
    let driver_1 = snapshots[1].snapshot(solid_displacement(&fixture.canonical));
    let driver_blocks_1 = snapshots[1].blocks(solid_displacement(&fixture.canonical));
    assert!(
        GeometryStateEnvelopeV3::new(
            &context,
            1,
            trajectory.states()[1].time(),
            Some(&geometry_states[0]),
            driver_1,
            driver_blocks_1,
            positive_but_nondriven,
        )
        .is_err(),
        "a positive coordinate state that is not the harmonic driver projection must fail",
    );
    assert!(
        GeometryStateEnvelopeV3::new(
            &context,
            0,
            trajectory.states()[0].time(),
            None,
            driver_1,
            driver_blocks_1,
            trajectory.states()[0]
                .geometry()
                .coordinates()
                .iter()
                .map(|coordinate| coordinate.to_vec())
                .collect(),
        )
        .is_err(),
        "a self-consistent displacement snapshot cannot drive unrelated coordinates",
    );
    assert!(
        SpatialStateEnvelopeV2::new(
            &context,
            &geometry_states[0],
            None,
            &snapshots[1].snapshots,
            driver_blocks_1,
        )
        .is_err(),
        "accepted geometry cannot be cross-wired to another numerical state",
    );

    let initial_segment =
        SpatialTrajectorySegmentEnvelopeV2::new(&context, &spatial_states[..1]).unwrap();
    let continuation_segment =
        SpatialTrajectorySegmentEnvelopeV2::new(&context, &spatial_states[1..]).unwrap();
    let initial_root = SpatialTrajectoryEnvelopeV2::start(&context, &initial_segment).unwrap();
    let final_root =
        SpatialTrajectoryEnvelopeV2::extend(&context, &initial_root, &continuation_segment)
            .unwrap();
    final_root
        .validate_against(
            &context,
            Some(&initial_root),
            &[initial_segment.clone(), continuation_segment.clone()],
        )
        .unwrap();
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

    let execution = ExecutionProvenanceV1::new(
        "eqiora.reference.fixed-topology-ale-fsi",
        env!("CARGO_PKG_VERSION"),
        "eqiora-backend-faer",
        env!("CARGO_PKG_VERSION"),
        ExecutionTopologyV1::Host {
            workers: NonZeroUsize::MIN,
        },
        ReductionPolicy::Fast,
    )
    .unwrap();
    let run = RunManifestV2::new(&realization, execution)
        .unwrap()
        .with_output(final_root.digest().unwrap());
    run.validate_against(&realization).unwrap();
}

fn assert_geometry_state_v3_replay_falsifiers(
    fixture: &Fixture,
    context: &ValidatedMovingSpatialContextV2<'_, ModelEnvelope, RealizationEnvelopeV6>,
    snapshots: &[MovingSnapshotSet],
    geometry_states: &[GeometryStateEnvelopeV3],
) {
    let initial = &geometry_states[0];
    let next = &geometry_states[1];
    let driver = snapshots[1].snapshot(solid_displacement(&fixture.canonical));
    let driver_blocks = snapshots[1].blocks(solid_displacement(&fixture.canonical));
    let bytes = next.canonical_json().unwrap();

    let mut wrong_dimension: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    wrong_dimension["spatial_dimension"] = serde_json::json!(2);
    assert!(
        GeometryStateEnvelopeV3::from_json(
            &serde_json::to_vec(&wrong_dimension).unwrap(),
            Default::default(),
        )
        .is_err(),
        "geometry-state/v3 must reject a substituted spatial dimension",
    );

    let mut authored_topology: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    authored_topology["cells"] = serde_json::json!([[0, 1, 2, 3]]);
    assert!(
        GeometryStateEnvelopeV3::from_json(
            &serde_json::to_vec(&authored_topology).unwrap(),
            Default::default(),
        )
        .is_err(),
        "geometry-state/v3 must reject caller-authored topology",
    );

    for pointer in [
        "/action_evidence/mesh_velocity_m_per_s/0/0",
        "/quality_evidence/reference/minimum_mean_ratio",
        "/quality_evidence/current/minimum_mean_ratio",
        "/action_evidence/minimum_path_signed_measure_scale",
    ] {
        let mut drifted: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        *drifted.pointer_mut(pointer).unwrap() = serde_json::json!(0.123);
        let drifted = GeometryStateEnvelopeV3::from_json(
            &serde_json::to_vec(&drifted).unwrap(),
            Default::default(),
        )
        .unwrap();
        assert!(
            drifted
                .validate_against(context, Some(initial), driver, driver_blocks)
                .is_err(),
            "geometry-state/v3 replay must reject derived-evidence drift at {pointer}",
        );
    }

    for pointer in [
        "/reference/model_sha256",
        "/reference/geometry_sha256",
        "/reference/correspondence_sha256",
        "/reference/mesh_sha256",
        "/reference/realization_sha256",
        "/solid_displacement_snapshot_sha256",
    ] {
        let mut stale: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        *stale.pointer_mut(pointer).unwrap() = serde_json::json!("22".repeat(32));
        let stale = GeometryStateEnvelopeV3::from_json(
            &serde_json::to_vec(&stale).unwrap(),
            Default::default(),
        )
        .unwrap();
        assert!(
            stale
                .validate_against(context, Some(initial), driver, driver_blocks)
                .is_err(),
            "geometry-state/v3 replay must reject stale lineage at {pointer}",
        );
    }

    let rotated = fixture
        .mesh
        .vertices()
        .iter()
        .map(|vertex| vec![-vertex[0], -vertex[1], vertex[2]])
        .collect::<Vec<_>>();
    let reference = FixedTopologyGeometryState::<D>::reference(&fixture.mesh).unwrap();
    let rotated_state =
        FixedTopologyGeometryState::<D>::new(&fixture.mesh, rotated.clone()).unwrap();
    let path =
        FixedTopologyGeometryAction::<D>::new(&fixture.mesh, &reference, &rotated_state, 1.0);
    assert!(
        path.is_err(),
        "the positive endpoint must still lose orientation along the reference path",
    );
    let initial_driver = snapshots[0].snapshot(solid_displacement(&fixture.canonical));
    let initial_driver_blocks = snapshots[0].blocks(solid_displacement(&fixture.canonical));
    assert!(
        GeometryStateEnvelopeV3::new(
            context,
            0,
            0.0,
            None,
            initial_driver,
            initial_driver_blocks,
            rotated,
        )
        .is_err(),
        "geometry-state/v3 must reject a positive endpoint whose reference path degenerates",
    );
}

fn moving_snapshots(
    fixture: &Fixture,
    context: &ValidatedMovingSpatialContextV2<'_, ModelEnvelope, RealizationEnvelopeV6>,
    state: &AleFsiState<3>,
) -> MovingSnapshotSet {
    let vector = DiscreteFieldShape::Vector {
        components: NonZeroU32::new(D as u32).unwrap(),
    };
    let mut blocks = Vec::new();

    let mut fluid_vertex_velocity = vec![[0.0; D]; fixture.mesh.vertices().len()];
    let fluid_velocity_values = vector_field(state, fluid_velocity(&fixture.canonical));
    for vertex in fixture
        .partition
        .domain_vertices(fluid_domain(&fixture.canonical))
        .unwrap()
    {
        fluid_vertex_velocity[vertex.index()] =
            fluid_velocity_values[&MeshEntity::new(0, vertex.index())];
    }
    let mut fluid_cell_velocity = vec![[0.0; D]; fixture.mesh.cells().len()];
    for (entity, value) in &fluid_velocity_values {
        if entity.dimension() == D {
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

    let mut solid_velocity_values = vec![[0.0; D]; fixture.mesh.vertices().len()];
    let mut solid_displacement_values = vec![[0.0; D]; fixture.mesh.vertices().len()];
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

fn flatten_vectors(values: &[[f64; D]]) -> Vec<f64> {
    values.iter().flatten().copied().collect()
}

fn realization_plan(
    model: &AleFsiCartesianModel<3>,
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
                QuadraturePolicy::SimplexDuffyGaussLegendre {
                    spatial_dimension: NonZeroUsize::new(D).unwrap(),
                    points_per_axis: NonZeroUsize::new(7).unwrap(),
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
            physical_scale(4.0, WEAK_FUNCTIONAL_3D),
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
        fixed_topology_ale_fsi_requirements_3d(model).solid_kinematic_relation(),
        P1HarmonicMeshMotionPolicy::new(
            fluid_domain(model),
            solid_domain(model),
            solid_displacement(model),
            connection(model),
            AleGeometryQualityGate::new(0.1).unwrap(),
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
            SpatialDimensionSupport::exact(NonZeroUsize::new(D).unwrap()),
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
        NonZeroUsize::new(4_000).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Identity)
    .with_reduction(ReductionPolicy::Fast)
}

fn physical_scale(value: f64, dimension: DimExponents) -> PositivePhysicalScale {
    PositivePhysicalScale::new(DynQuantity::new(value, dimension)).unwrap()
}

fn fluid_domain(model: &AleFsiCartesianModel<3>) -> Id<kinds::Domain> {
    model.fluid().domain().downcast().unwrap()
}

fn solid_domain(model: &AleFsiCartesianModel<3>) -> Id<kinds::Domain> {
    model.solid().continuum().domain().downcast().unwrap()
}

fn fluid_velocity(model: &AleFsiCartesianModel<3>) -> Id<kinds::Field> {
    model.fluid().velocity().downcast().unwrap()
}

fn fluid_pressure(model: &AleFsiCartesianModel<3>) -> Id<kinds::Field> {
    model.fluid().pressure().downcast().unwrap()
}

fn solid_velocity(model: &AleFsiCartesianModel<3>) -> Id<kinds::Field> {
    model.solid().velocity().downcast().unwrap()
}

fn solid_displacement(model: &AleFsiCartesianModel<3>) -> Id<kinds::Field> {
    model.solid().continuum().displacement().downcast().unwrap()
}

fn fluid_relation(model: &AleFsiCartesianModel<3>) -> Id<kinds::Relation> {
    model.fluid().momentum_relation().downcast().unwrap()
}

fn connection(model: &AleFsiCartesianModel<3>) -> Id<kinds::Connection> {
    model.interface().connection().downcast().unwrap()
}

fn trace_quotient(model: &AleFsiCartesianModel<3>) -> ConformingTraceQuotient {
    ConformingTraceQuotient::new(
        connection(model),
        TraceFieldEndpoint::new(fluid_domain(model), fluid_velocity(model)),
        TraceFieldEndpoint::new(solid_domain(model), solid_velocity(model)),
    )
    .unwrap()
}

fn vector_field(state: &AleFsiState<3>, field: Id<kinds::Field>) -> BTreeMap<MeshEntity, [f64; D]> {
    let mut values = BTreeMap::<_, [Option<f64>; D]>::new();
    for (entity, slot, component, value) in state
        .physical_state()
        .coefficients(field)
        .expect("accepted exact Field")
    {
        assert_eq!(slot, 0);
        values.entry(entity).or_insert([None; D])[component] = Some(value);
    }
    values
        .into_iter()
        .map(|(entity, values)| (entity, values.map(Option::unwrap)))
        .collect()
}

fn scalar_field(state: &AleFsiState<3>, field: Id<kinds::Field>) -> BTreeMap<MeshEntity, f64> {
    state
        .physical_state()
        .coefficients(field)
        .expect("accepted exact Field")
        .map(|(entity, slot, component, value)| {
            assert_eq!((slot, component), (0, 0));
            (entity, value)
        })
        .collect()
}

fn state_pair(model: &AleFsiCartesianModel<3>) -> BackwardEulerStatePair {
    BackwardEulerStatePair::new(
        fixed_topology_ale_fsi_requirements_3d(model).solid_kinematic_relation(),
        solid_displacement(model),
        solid_velocity(model),
    )
    .unwrap()
}

fn two_domain_tetrahedral_mesh() -> SimplicialMesh {
    let vertices = vec![
        vec![0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 1.0, 1.0],
        vec![0.0, 0.0, 1.0],
        vec![1.0, 0.0, 0.0],
        vec![1.0, 1.0, 0.0],
        vec![1.0, 1.0, 1.0],
        vec![1.0, 0.0, 1.0],
        vec![2.0, 0.0, 0.0],
        vec![2.0, 1.0, 0.0],
        vec![2.0, 1.0, 1.0],
        vec![2.0, 0.0, 1.0],
        vec![0.5, 0.5, 0.5],
        vec![1.0, 0.5, 0.5],
        vec![1.5, 0.5, 0.5],
    ];
    let interface = [[4, 5, 13], [5, 6, 13], [6, 7, 13], [7, 4, 13]];
    let fluid_surface = [
        [0, 3, 2],
        [0, 2, 1],
        [0, 4, 7],
        [0, 7, 3],
        [1, 2, 6],
        [1, 6, 5],
        [0, 1, 5],
        [0, 5, 4],
        [3, 7, 6],
        [3, 6, 2],
    ];
    let solid_surface = [
        [8, 9, 10],
        [8, 10, 11],
        [4, 8, 11],
        [4, 11, 7],
        [5, 6, 10],
        [5, 10, 9],
        [4, 5, 9],
        [4, 9, 8],
        [7, 11, 10],
        [7, 10, 6],
    ];
    let mut cells = fluid_surface
        .into_iter()
        .chain(interface)
        .map(|face| vec![12, face[0], face[1], face[2]])
        .chain(
            solid_surface
                .into_iter()
                .chain(interface)
                .map(|face| vec![14, face[0], face[1], face[2]]),
        )
        .collect::<Vec<_>>();
    for cell in &mut cells {
        if signed_tetrahedron_measure(&vertices, cell) < 0.0 {
            cell.swap(1, 2);
        }
    }
    SimplicialMesh::new(3, vertices, cells, MeshQualityGate::new(0.1).unwrap()).unwrap()
}

fn inventories(mesh: &SimplicialMesh) -> (Vec<CellId>, Vec<CellId>, Vec<FacetId>) {
    let fluid = (0..14).map(CellId::new).collect();
    let solid = (14..28).map(CellId::new).collect();
    let interface = (0..mesh.entity_count(2).unwrap())
        .filter(|&facet| {
            mesh.entity_vertices(MeshEntity::new(2, facet))
                .unwrap()
                .iter()
                .all(|vertex| mesh.vertices()[vertex.index()][0] == 1.0)
        })
        .map(FacetId::new)
        .collect();
    (fluid, solid, interface)
}

fn signed_tetrahedron_measure(vertices: &[Vec<f64>], cell: &[usize]) -> f64 {
    let origin = &vertices[cell[0]];
    let column = |vertex: usize, axis: usize| vertices[cell[vertex]][axis] - origin[axis];
    column(1, 0) * (column(2, 1) * column(3, 2) - column(3, 1) * column(2, 2))
        - column(2, 0) * (column(1, 1) * column(3, 2) - column(3, 1) * column(1, 2))
        + column(3, 0) * (column(1, 1) * column(2, 2) - column(2, 1) * column(1, 2))
}

fn solid_displacement_mass_distance(
    mesh: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<3>,
    domain: Id<kinds::Domain>,
    field: Id<kinds::Field>,
    left: &AleFsiState<3>,
    right: &AleFsiState<3>,
) -> f64 {
    let mut squared = 0.0;
    let left_displacement = vector_field(left, field);
    let right_displacement = vector_field(right, field);
    for cell in partition.domain_cells(domain).unwrap() {
        let vertices = mesh
            .entity_vertices(MeshEntity::new(3, cell.index()))
            .unwrap();
        let volume =
            signed_tetrahedron_measure(mesh.vertices(), &mesh.cells()[cell.index()]).abs() / 6.0;
        for row in 0..4 {
            for column in 0..4 {
                let mass = volume / 20.0 * if row == column { 2.0 } else { 1.0 };
                let row_vertex = vertices[row].index();
                let column_vertex = vertices[column].index();
                for component in 0..D {
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

fn find_vertex(mesh: &SimplicialMesh, target: [f64; D]) -> usize {
    mesh.vertices()
        .iter()
        .position(|coordinates| coordinates.as_slice() == target)
        .unwrap()
}
