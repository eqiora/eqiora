use std::num::NonZeroUsize;

use eqiora_assembly::{AssemblyPlan, AssemblyResult, AssemblyTarget, CsrMatrix, LinearSystem};
use eqiora_meshing::{CellId, FacetId, MeshEntity, MeshQualityGate, MeshTopology, SimplicialMesh};
use eqiora_realization::{NonlinearSolvePlan, Target};
use eqiora_solver::{
    BackendId, ConvergenceReason, ExecutionReport, LinearOperatorOrientation, LinearSolveRequest,
    LinearSolver, PreconditionerPolicy, REFERENCE_LINEAR_SOLVER, ReductionPolicy,
    SERIAL_EXECUTION_PROVIDER, SolverPlan, SolverProvider,
};

use super::super::{AleFsiState, AleFsiStepPlan, P1HarmonicMeshMotionAction};
use super::*;
use crate::simplicial_fsi::{
    FixedReferenceFsiLoad, FixedReferenceFsiMaterial, FixedReferenceFsiPartition,
    FixedReferenceFsiScale,
};

const COMPONENTS: usize = 2;
const COMPONENTS_3D: usize = 3;
const TEST_ALE_FSI_SOLVER_PROVIDER: SolverProvider = SolverProvider::new(
    BackendId::new("eqiora.test.ale-fsi"),
    env!("CARGO_PKG_VERSION"),
    &[],
);

#[test]
fn interface_action_exposes_finite_balance_and_power_helpers() {
    let action =
        AleFsiInterfaceAction::<2>::new(VertexId::new(3), [2.0, -1.0], [-2.0, 1.0]).unwrap();
    assert_eq!(action.vertex(), VertexId::new(3));
    assert_eq!(action.fluid(), [2.0, -1.0]);
    assert_eq!(action.solid(), [-2.0, 1.0]);
    assert_eq!(action.imbalance(), [0.0, 0.0]);
    assert_eq!(action.imbalance_norm(), 0.0);
    assert_eq!(action.fluid_power([3.0, 4.0]).unwrap(), 2.0);
    assert_eq!(action.solid_power([3.0, 4.0]).unwrap(), -2.0);
    assert_eq!(action.power_imbalance([3.0, 4.0]).unwrap(), 0.0);
    assert!(action.power_imbalance([f64::NAN, 0.0]).is_err());
    assert!(
        AleFsiInterfaceAction::<2>::new(VertexId::new(3), [f64::INFINITY, 0.0], [0.0; 2]).is_err()
    );
}

#[test]
fn three_dimensional_interface_action_is_typed_and_fails_closed() {
    let action =
        AleFsiInterfaceAction::<3>::new(VertexId::new(7), [2.0, -1.0, 0.5], [-2.0, 1.0, -0.5])
            .unwrap();
    assert_eq!(action.vertex(), VertexId::new(7));
    assert_eq!(action.imbalance(), [0.0; 3]);
    assert_eq!(action.fluid_power([3.0, 4.0, 2.0]).unwrap(), 3.0);
    assert_eq!(action.solid_power([3.0, 4.0, 2.0]).unwrap(), -3.0);
    assert_eq!(action.power_imbalance([3.0, 4.0, 2.0]).unwrap(), 0.0);
    assert!(action.fluid_power([0.0, f64::NAN, 0.0]).is_err());
    assert!(
        AleFsiInterfaceAction::<3>::new(VertexId::new(7), [f64::INFINITY, 0.0, 0.0], [0.0; 3],)
            .is_err()
    );
    assert!(AleFsiInterfaceAction::<1>::new(VertexId::new(0), [0.0], [0.0]).is_err());

    let ordered = [
        AleFsiInterfaceAction::<3>::new(VertexId::new(2), [0.0; 3], [0.0; 3]).unwrap(),
        AleFsiInterfaceAction::<3>::new(VertexId::new(4), [0.0; 3], [0.0; 3]).unwrap(),
    ];
    assert!(validate_interface_order(&ordered).is_ok());
    assert!(validate_interface_order(&ordered.into_iter().rev().collect::<Vec<_>>()).is_err());
}

#[test]
fn three_dimensional_evidence_and_trajectory_are_exercised_and_fail_closed() {
    let fixture = fixture_3d();
    let plan = step_plan_3d();
    let interface_vertex = fixture.partition.interface_vertices()[0];
    let initial = state_3d(0.0, &fixture, interface_vertex);
    let current = state_3d(plan.time_step(), &fixture, interface_vertex);
    let geometry = plan
        .geometry_action(
            &fixture.mesh,
            &fixture.partition,
            &fixture.motion,
            &initial,
            &current,
        )
        .unwrap();
    let evidence = AleFsiStepEvidence::<3>::new(
        plan,
        &geometry,
        &current,
        accepted_input_3d(plan, interface_vertex),
    )
    .unwrap();
    assert_eq!(evidence.interface_actions()[0].imbalance(), [0.0; 3]);
    assert_eq!(evidence.interface_power_imbalance(), 0.0);

    let mut power_overflow = accepted_input_3d(plan, interface_vertex);
    power_overflow.interface_actions = vec![
        AleFsiInterfaceAction::<3>::new(interface_vertex, [1.0, -2.0, f64::MAX], [-1.0, 2.0, -3.0])
            .unwrap(),
    ];
    let error = AleFsiStepEvidence::<3>::new(plan, &geometry, &current, power_overflow)
        .expect_err("third-component interface-power overflow must fail closed");
    assert!(
        error
            .message()
            .contains("interface-power evaluation overflowed")
    );

    let mut trajectory = AleFsiTrajectory::<3>::new(initial);
    trajectory.push(current, evidence.clone()).unwrap();
    assert_eq!(trajectory.states().len(), 2);
    assert_eq!(trajectory.steps(), std::slice::from_ref(&evidence));

    let later = state_3d(2.0 * plan.time_step(), &fixture, interface_vertex);
    let state_count = trajectory.states().len();
    let step_count = trajectory.steps().len();
    let error = trajectory
        .push(later, evidence)
        .expect_err("trajectory evidence time must bind the appended state");
    assert!(error.message().contains("match their evidence time"));
    assert_eq!(trajectory.states().len(), state_count);
    assert_eq!(trajectory.steps().len(), step_count);
}

#[test]
fn evidence_derives_geometry_interface_and_nonlinear_acceptance() {
    let fixture = fixture();
    let plan = step_plan();
    let previous = state(0.0, &fixture);
    let current = state(plan.time_step(), &fixture);
    let geometry = plan
        .geometry_action(
            &fixture.mesh,
            &fixture.partition,
            &fixture.motion,
            &previous,
            &current,
        )
        .unwrap();
    let interface_vertex = fixture.partition.interface_vertices()[0];
    let evidence = AleFsiStepEvidence::<2>::new(
        plan,
        &geometry,
        &current,
        accepted_input(plan, interface_vertex),
    )
    .unwrap();

    assert_eq!(evidence.accepted_time(), current.time());
    assert_eq!(evidence.nonlinear_iterations(), 1);
    assert_eq!(evidence.initial_residual_norm(), 1.0);
    assert_eq!(evidence.residual_target(), 1.0e-9);
    assert_eq!(evidence.final_residual_norm(), 1.0e-10);
    assert_eq!(evidence.continuity_residual_norm(), 1.0e-11);
    assert_eq!(evidence.kinematic_residual_norm(), 1.0e-12);
    assert_eq!(evidence.interface_velocity_jump_norm(), 0.0);
    assert_eq!(evidence.interface_actions().len(), 1);
    assert_eq!(evidence.interface_action_imbalance_norm(), 0.0);
    assert_eq!(evidence.interface_power_imbalance(), 0.0);
    assert!(evidence.maximum_affine_metric_identity_defect() >= 0.0);
    assert!(evidence.minimum_current_mean_ratio() > 0.0);
    assert!(evidence.minimum_current_signed_jacobian() > 0.0);
    assert!(evidence.minimum_path_signed_jacobian() > 0.0);
    assert_eq!(evidence.probed_moving_fluid_cell_count(), 0);
    assert_eq!(evidence.gcl_active_moving_fluid_cell_count(), 0);
    assert_eq!(
        evidence.compatible_constant_free_stream_residual_norm(),
        0.0
    );
    assert_eq!(evidence.omitted_gcl_witness_norm(), 0.0);
    assert_eq!(evidence.assembly_report().packet_count(), 1);
    assert_eq!(evidence.nonlinear_linear_solves().len(), 1);
}

#[test]
fn evidence_rejects_iteration_mismatch_and_stale_geometry() {
    let fixture = fixture();
    let plan = step_plan();
    let previous = state(0.0, &fixture);
    let current = state(plan.time_step(), &fixture);
    let geometry = plan
        .geometry_action(
            &fixture.mesh,
            &fixture.partition,
            &fixture.motion,
            &previous,
            &current,
        )
        .unwrap();
    let interface_vertex = fixture.partition.interface_vertices()[0];

    let mut mismatched = accepted_input(plan, interface_vertex);
    mismatched.nonlinear_linear_solves.clear();
    assert!(AleFsiStepEvidence::<2>::new(plan, &geometry, &current, mismatched).is_err());

    let mut missing_gcl_witness = accepted_input(plan, interface_vertex);
    missing_gcl_witness.probed_moving_fluid_cell_count = 1;
    missing_gcl_witness.gcl_active_moving_fluid_cell_count = 1;
    assert!(AleFsiStepEvidence::<2>::new(plan, &geometry, &current, missing_gcl_witness).is_err());

    let mut nonzero_static_probe = accepted_input(plan, interface_vertex);
    nonzero_static_probe.compatible_constant_free_stream_residual_norm = 1.0e-6;
    assert!(AleFsiStepEvidence::<2>::new(plan, &geometry, &current, nonzero_static_probe).is_err());

    let mut displacement = vec![[0.0; COMPONENTS]; fixture.mesh.vertices().len()];
    for vertex in fixture.partition.solid_vertices() {
        displacement[vertex.index()] = [0.002, 0.0];
    }
    let moved = AleFsiState::<2>::new(
        current.time(),
        &fixture.mesh,
        &fixture.partition,
        &fixture.motion,
        vec![[0.0; COMPONENTS]; fixture.mesh.vertices().len()],
        fixture
            .partition
            .fluid_cells()
            .iter()
            .copied()
            .map(|cell| (cell, [0.0; COMPONENTS]))
            .collect(),
        vec![0.0; fixture.partition.fluid_vertices().len()],
        displacement,
    )
    .unwrap();
    assert!(
        AleFsiStepEvidence::<2>::new(
            plan,
            &geometry,
            &moved,
            accepted_input(plan, interface_vertex),
        )
        .is_err()
    );
}

#[test]
fn trajectory_push_is_atomic_and_time_bound() {
    let fixture = fixture();
    let plan = step_plan();
    let initial = state(0.0, &fixture);
    let current = state(plan.time_step(), &fixture);
    let geometry = plan
        .geometry_action(
            &fixture.mesh,
            &fixture.partition,
            &fixture.motion,
            &initial,
            &current,
        )
        .unwrap();
    let evidence = AleFsiStepEvidence::<2>::new(
        plan,
        &geometry,
        &current,
        accepted_input(plan, fixture.partition.interface_vertices()[0]),
    )
    .unwrap();
    let mut trajectory = AleFsiTrajectory::<2>::new(initial);
    trajectory.push(current.clone(), evidence.clone()).unwrap();
    assert_eq!(trajectory.states().len(), 2);
    assert_eq!(trajectory.steps().len(), 1);
    assert_eq!(trajectory.initial_state().time(), 0.0);
    assert_eq!(trajectory.final_state().time(), plan.time_step());

    let state_count = trajectory.states().len();
    let step_count = trajectory.steps().len();
    assert!(trajectory.push(current, evidence).is_err());
    assert_eq!(trajectory.states().len(), state_count);
    assert_eq!(trajectory.steps().len(), step_count);
}

struct Fixture {
    mesh: SimplicialMesh,
    partition: FixedReferenceFsiPartition<2>,
    motion: P1HarmonicMeshMotionAction<2>,
}

struct Fixture3d {
    mesh: SimplicialMesh,
    partition: FixedReferenceFsiPartition<3>,
    motion: P1HarmonicMeshMotionAction<3>,
}

fn fixture() -> Fixture {
    let mesh = two_domain_mesh();
    let (fluid, solid, interface) = inventories(&mesh);
    let partition = FixedReferenceFsiPartition::<2>::new(&mesh, fluid, solid, interface).unwrap();
    let motion =
        P1HarmonicMeshMotionAction::<2>::new(&mesh, &partition, harmonic_solver()).unwrap();
    Fixture {
        mesh,
        partition,
        motion,
    }
}

fn fixture_3d() -> Fixture3d {
    let (mesh, partition) =
        partitioned_block_3d(&[0.0, 0.5, 1.0, 2.0], &[0.0, 0.5, 1.0], &[0.0, 0.5, 1.0]);
    let motion =
        P1HarmonicMeshMotionAction::<3>::new(&mesh, &partition, harmonic_solver()).unwrap();
    Fixture3d {
        mesh,
        partition,
        motion,
    }
}

fn state(time: f64, fixture: &Fixture) -> AleFsiState<2> {
    AleFsiState::<2>::new(
        time,
        &fixture.mesh,
        &fixture.partition,
        &fixture.motion,
        vec![[0.0; COMPONENTS]; fixture.mesh.vertices().len()],
        fixture
            .partition
            .fluid_cells()
            .iter()
            .copied()
            .map(|cell| (cell, [0.0; COMPONENTS]))
            .collect(),
        vec![0.0; fixture.partition.fluid_vertices().len()],
        vec![[0.0; COMPONENTS]; fixture.mesh.vertices().len()],
    )
    .unwrap()
}

fn state_3d(time: f64, fixture: &Fixture3d, interface_vertex: VertexId) -> AleFsiState<3> {
    let mut vertex_velocity = vec![[0.0; COMPONENTS_3D]; fixture.mesh.vertices().len()];
    vertex_velocity[interface_vertex.index()] = [0.0, 0.0, 2.0];
    AleFsiState::<3>::new(
        time,
        &fixture.mesh,
        &fixture.partition,
        &fixture.motion,
        vertex_velocity,
        fixture
            .partition
            .fluid_cells()
            .iter()
            .copied()
            .map(|cell| (cell, [0.0; COMPONENTS_3D]))
            .collect(),
        vec![0.0; fixture.partition.fluid_vertices().len()],
        vec![[0.0; COMPONENTS_3D]; fixture.mesh.vertices().len()],
    )
    .unwrap()
}

fn accepted_input(
    plan: AleFsiStepPlan<2>,
    interface_vertex: VertexId,
) -> AleFsiStepEvidenceInput2d {
    AleFsiStepEvidenceInput2d {
        nonlinear_iterations: 1,
        initial_residual_norm: 1.0,
        final_residual_norm: 1.0e-10,
        continuity_residual_norm: 1.0e-11,
        kinematic_residual_norm: 1.0e-12,
        interface_velocity_jump_norm: 0.0,
        interface_actions: vec![
            AleFsiInterfaceAction::<2>::new(interface_vertex, [1.0, -2.0], [-1.0, 2.0]).unwrap(),
        ],
        probed_moving_fluid_cell_count: 0,
        gcl_active_moving_fluid_cell_count: 0,
        compatible_constant_free_stream_residual_norm: 0.0,
        omitted_gcl_witness_norm: 0.0,
        assembly_report: assembly_report(),
        nonlinear_linear_solves: vec![linear_report(plan)],
    }
}

fn accepted_input_3d(
    plan: AleFsiStepPlan<3>,
    interface_vertex: VertexId,
) -> AleFsiStepEvidenceInput<3> {
    AleFsiStepEvidenceInput {
        nonlinear_iterations: 1,
        initial_residual_norm: 1.0,
        final_residual_norm: 1.0e-10,
        continuity_residual_norm: 1.0e-11,
        kinematic_residual_norm: 1.0e-12,
        interface_velocity_jump_norm: 0.0,
        interface_actions: vec![
            AleFsiInterfaceAction::<3>::new(interface_vertex, [1.0, -2.0, 3.0], [-1.0, 2.0, -3.0])
                .unwrap(),
        ],
        probed_moving_fluid_cell_count: 0,
        gcl_active_moving_fluid_cell_count: 0,
        compatible_constant_free_stream_residual_norm: 0.0,
        omitted_gcl_witness_norm: 0.0,
        assembly_report: assembly_report(),
        nonlinear_linear_solves: vec![linear_report(plan)],
    }
}

fn linear_report<const D: usize>(plan: AleFsiStepPlan<D>) -> SolveReport {
    SolveReport::accepted(
        TEST_ALE_FSI_SOLVER_PROVIDER,
        SERIAL_EXECUTION_PROVIDER,
        ExecutionReport::host_serial(),
        LinearOperatorOrientation::Normal,
        plan.linear_solver(),
        ConvergenceReason::ResidualToleranceSatisfied,
        1,
        1.0,
        1.0e-12,
        1.0e-12,
        1.0e-10,
    )
    .unwrap()
}

fn assembly_report() -> AssemblyReport {
    let plan = AssemblyPlan::new(vec![AssemblyTarget::new(1).unwrap()]).unwrap();
    let matrix = CsrMatrix::from_sorted_csr(1, 1, vec![0, 1], vec![0], vec![1.0]).unwrap();
    let system = LinearSystem::new(matrix, vec![0.0]).unwrap();
    *AssemblyResult::from_complete_systems(&plan, vec![system], 1, ExecutionReport::host_serial())
        .unwrap()
        .report()
}

fn step_plan() -> AleFsiStepPlan<2> {
    let nonlinear =
        NonlinearSolvePlan::new(1.0e-9, 1.0e-12, NonZeroUsize::new(20).unwrap(), 12).unwrap();
    let linear = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1.0e-10,
        1.0e-12,
        NonZeroUsize::new(500).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Identity)
    .with_reduction(ReductionPolicy::Fast);
    AleFsiStepPlan::<2>::new(
        0.05,
        FixedReferenceFsiMaterial::<2>::new(1.0, 0.1, 1.0, 2.0, 1.0).unwrap(),
        FixedReferenceFsiScale::<2>::new(2.0, 1.0, 1.0).unwrap(),
        FixedReferenceFsiLoad::Zero,
        nonlinear,
        linear,
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
    )
    .unwrap()
}

fn step_plan_3d() -> AleFsiStepPlan<3> {
    let nonlinear =
        NonlinearSolvePlan::new(1.0e-9, 1.0e-12, NonZeroUsize::new(20).unwrap(), 12).unwrap();
    let linear = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1.0e-10,
        1.0e-12,
        NonZeroUsize::new(500).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Identity)
    .with_reduction(ReductionPolicy::Fast);
    AleFsiStepPlan::<3>::new(
        0.05,
        FixedReferenceFsiMaterial::<3>::new(1.0, 0.1, 1.0, 2.0, 1.0).unwrap(),
        FixedReferenceFsiScale::<3>::new(2.0, 1.0, 1.0).unwrap(),
        FixedReferenceFsiLoad::Zero,
        nonlinear,
        linear,
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
    )
    .unwrap()
}

fn harmonic_solver() -> LinearSolveRequest<'static> {
    let plan = SolverPlan::new(
        LinearSolver::ConjugateGradient,
        1.0e-12,
        1.0e-14,
        NonZeroUsize::new(500).unwrap(),
    )
    .unwrap();
    LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, plan)
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

fn partitioned_block_3d(
    x_coordinates: &[f64],
    y_coordinates: &[f64],
    z_coordinates: &[f64],
) -> (SimplicialMesh, FixedReferenceFsiPartition<3>) {
    let nx = x_coordinates.len();
    let ny = y_coordinates.len();
    let vertex = |x: usize, y: usize, z: usize| z * ny * nx + y * nx + x;
    let vertices = z_coordinates
        .iter()
        .flat_map(|&z| {
            y_coordinates
                .iter()
                .flat_map(move |&y| x_coordinates.iter().map(move |&x| vec![x, y, z]))
        })
        .collect::<Vec<_>>();
    let permutations = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let mut cells = Vec::new();
    let mut fluid_cells = Vec::new();
    let mut solid_cells = Vec::new();
    for x in 0..nx - 1 {
        for y in 0..ny - 1 {
            for z in 0..z_coordinates.len() - 1 {
                for permutation in permutations {
                    let mut offset = [0, 0, 0];
                    let mut tetrahedron = vec![vertex(x, y, z)];
                    for axis in permutation {
                        offset[axis] = 1;
                        tetrahedron.push(vertex(x + offset[0], y + offset[1], z + offset[2]));
                    }
                    if signed_tetrahedron_measure(&vertices, &tetrahedron) < 0.0 {
                        tetrahedron.swap(1, 2);
                    }
                    let id = CellId::new(cells.len());
                    if x_coordinates[x + 1] <= 1.0 {
                        fluid_cells.push(id);
                    } else {
                        solid_cells.push(id);
                    }
                    cells.push(tetrahedron);
                }
            }
        }
    }
    let mesh =
        SimplicialMesh::new(3, vertices, cells, MeshQualityGate::new(0.02).unwrap()).unwrap();
    let interface_facets = (0..mesh.entity_count(2).unwrap())
        .filter_map(|facet| {
            let vertices = mesh.entity_vertices(MeshEntity::new(2, facet)).unwrap();
            vertices
                .iter()
                .all(|vertex| mesh.vertices()[vertex.index()][0] == 1.0)
                .then_some(FacetId::new(facet))
        })
        .collect::<Vec<_>>();
    let partition =
        FixedReferenceFsiPartition::<3>::new(&mesh, fluid_cells, solid_cells, interface_facets)
            .unwrap();
    (mesh, partition)
}

fn signed_tetrahedron_measure(vertices: &[Vec<f64>], cell: &[usize]) -> f64 {
    let origin = &vertices[cell[0]];
    let column = |vertex: usize, axis: usize| vertices[cell[vertex]][axis] - origin[axis];
    column(1, 0) * (column(2, 1) * column(3, 2) - column(3, 1) * column(2, 2))
        - column(2, 0) * (column(1, 1) * column(3, 2) - column(3, 1) * column(1, 2))
        + column(3, 0) * (column(1, 1) * column(2, 2) - column(2, 1) * column(1, 2))
}
