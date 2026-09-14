use std::num::NonZeroUsize;

use eqiora_meshing::{
    CellId, FacetId, FixedTopologyGeometryState2d, MeshEntity, MeshQualityGate, MeshTopology,
};
use eqiora_solver::{
    LinearSolveRequest, PreconditionerPolicy, REFERENCE_LINEAR_SOLVER, ReductionPolicy,
};

use super::*;
use crate::simplicial_ale_fsi::P1HarmonicMeshMotionAction;
use crate::simplicial_fsi::{
    FixedReferenceFsiBoundary, FixedReferenceFsiLoad, FixedReferenceFsiMaterial,
    FixedReferenceFsiPartition, FixedReferenceFsiScale,
};

const COMPONENTS: usize = 2;

#[test]
fn state_derives_geometry_from_the_only_admitted_driver() {
    let fixture = fixture();
    let solid_displacement = moving_solid_displacement(&fixture);
    let state = AleFsiState::<2>::new(
        0.25,
        &fixture.mesh,
        &fixture.partition,
        &fixture.motion,
        zero_vertex_vectors(&fixture.mesh),
        fixture
            .partition
            .fluid_cells()
            .iter()
            .copied()
            .map(|cell| (cell, [0.0; COMPONENTS]))
            .collect(),
        (0..fixture.partition.fluid_vertices().len())
            .map(|index| index as f64)
            .collect(),
        solid_displacement.clone(),
    )
    .unwrap();

    let all_displacement = fixture.motion.apply(&solid_displacement).unwrap();
    let expected = current_coordinates(&fixture.mesh, &all_displacement).unwrap();
    assert_eq!(state.geometry().coordinates(), expected);
    assert_eq!(state.time(), 0.25);
    assert_eq!(state.vertex_velocity().len(), fixture.mesh.vertices().len());
    assert_eq!(
        state.fluid_cell_bubble_velocity().len(),
        fixture.partition.fluid_cells().len()
    );
    assert_eq!(
        state.fluid_pressure().len(),
        fixture.partition.fluid_vertices().len()
    );
    assert_eq!(state.solid_displacement(), solid_displacement);
    let fixed = state
        .to_fixed_reference_state(&fixture.mesh, &fixture.partition)
        .unwrap();
    assert_eq!(fixed.solid_displacement(), state.solid_displacement());
    state
        .validate_against(&fixture.mesh, &fixture.partition, &fixture.motion)
        .unwrap();
}

#[test]
fn state_rejects_wrong_pressure_support_and_nonfinite_fields() {
    let fixture = fixture();
    let valid_pressure = vec![0.0; fixture.partition.fluid_vertices().len()];
    let error = AleFsiState::<2>::new(
        0.0,
        &fixture.mesh,
        &fixture.partition,
        &fixture.motion,
        zero_vertex_vectors(&fixture.mesh),
        fixture
            .partition
            .fluid_cells()
            .iter()
            .copied()
            .map(|cell| (cell, [0.0; COMPONENTS]))
            .collect(),
        valid_pressure[..valid_pressure.len() - 1].to_vec(),
        zero_vertex_vectors(&fixture.mesh),
    )
    .unwrap_err();
    assert!(error.message().contains("canonical fluid-vertex order"));

    let fluid_only = fixture
        .partition
        .fluid_vertices()
        .iter()
        .find(|vertex| {
            fixture
                .partition
                .solid_vertices()
                .binary_search(vertex)
                .is_err()
        })
        .unwrap();
    let mut unsupported = zero_vertex_vectors(&fixture.mesh);
    unsupported[fluid_only.index()] = [0.01, 0.0];
    assert!(
        AleFsiState::<2>::new(
            0.0,
            &fixture.mesh,
            &fixture.partition,
            &fixture.motion,
            zero_vertex_vectors(&fixture.mesh),
            fixture
                .partition
                .fluid_cells()
                .iter()
                .copied()
                .map(|cell| (cell, [0.0; COMPONENTS]))
                .collect(),
            valid_pressure.clone(),
            unsupported,
        )
        .is_err()
    );

    let mut velocity = zero_vertex_vectors(&fixture.mesh);
    velocity[0][0] = f64::NAN;
    assert!(
        AleFsiState::<2>::new(
            0.0,
            &fixture.mesh,
            &fixture.partition,
            &fixture.motion,
            velocity,
            fixture
                .partition
                .fluid_cells()
                .iter()
                .copied()
                .map(|cell| (cell, [0.0; COMPONENTS]))
                .collect(),
            valid_pressure,
            zero_vertex_vectors(&fixture.mesh),
        )
        .is_err()
    );
}

#[test]
fn restart_replay_rejects_a_substituted_derived_geometry() {
    let fixture = fixture();
    let mut state = zero_state(0.0, &fixture);
    let mut substituted = fixture.mesh.vertices().to_vec();
    substituted[0][0] += 0.01;
    state.geometry = FixedTopologyGeometryState2d::new(&fixture.mesh, substituted).unwrap();
    let error = state
        .validate_against(&fixture.mesh, &fixture.partition, &fixture.motion)
        .unwrap_err();
    assert!(
        error
            .message()
            .contains("replayed absolute harmonic motion")
    );
}

#[test]
fn step_plan_closes_general_solver_serial_target_and_exact_time() {
    let fixture = fixture();
    let plan = valid_plan();
    assert_eq!(
        plan.operator_properties(),
        LinearOperatorProperties::General
    );
    assert_eq!(
        plan.linear_solver().algorithm(),
        LinearSolver::BiConjugateGradientStabilized
    );
    assert_eq!(plan.material(), material());
    assert_eq!(
        plan.scale(),
        FixedReferenceFsiScale::<2>::new(2.0, 1.0, 1.0).unwrap()
    );
    assert_eq!(plan.load(), FixedReferenceFsiLoad::Zero);
    assert_eq!(plan.nonlinear(), nonlinear());
    assert_eq!(
        plan.target(),
        Target::HostCpu {
            threads: NonZeroUsize::MIN
        }
    );
    assert_eq!(plan.fixed_reference_config().time_step(), plan.time_step());
    let _boundary: AleFsiBoundary<2> =
        FixedReferenceFsiBoundary::<2>::homogeneous_exterior(&fixture.mesh).unwrap();
    let previous = zero_state(0.0, &fixture);
    let current = zero_state(plan.time_step(), &fixture);
    let action = plan
        .geometry_action(
            &fixture.mesh,
            &fixture.partition,
            &fixture.motion,
            &previous,
            &current,
        )
        .unwrap();
    assert_eq!(action.time_step(), plan.time_step());

    let wrong_time = zero_state(2.0 * plan.time_step(), &fixture);
    assert!(
        plan.geometry_action(
            &fixture.mesh,
            &fixture.partition,
            &fixture.motion,
            &previous,
            &wrong_time,
        )
        .is_err()
    );

    let rounded_previous = zero_state(f64::MAX, &fixture);
    let rounded_current = zero_state(f64::MAX, &fixture);
    assert!(
        plan.geometry_action(
            &fixture.mesh,
            &fixture.partition,
            &fixture.motion,
            &rounded_previous,
            &rounded_current,
        )
        .is_err()
    );
}

#[test]
fn step_plan_rejects_symmetric_solver_and_parallel_target() {
    let material = material();
    let scale = FixedReferenceFsiScale::<2>::new(2.0, 1.0, 1.0).unwrap();
    let nonlinear = nonlinear();
    let minres = SolverPlan::new(
        LinearSolver::MinimumResidual,
        1.0e-10,
        1.0e-12,
        NonZeroUsize::new(500).unwrap(),
    )
    .unwrap();
    let serial = Target::HostCpu {
        threads: NonZeroUsize::MIN,
    };
    assert!(
        AleFsiStepPlan::<2>::new(
            0.05,
            material,
            scale,
            FixedReferenceFsiLoad::Zero,
            nonlinear,
            minres,
            serial,
        )
        .is_err()
    );

    let general = general_solver();
    assert!(
        AleFsiStepPlan::<2>::new(
            0.05,
            material,
            scale,
            FixedReferenceFsiLoad::Zero,
            nonlinear,
            general,
            Target::HostCpu {
                threads: NonZeroUsize::new(2).unwrap(),
            },
        )
        .is_err()
    );
}

#[test]
fn tetrahedral_state_replays_the_only_geometry_driver_and_rejects_wrong_shape() {
    let fixture = fixture_3d();
    let solid_displacement = moving_solid_displacement_3d(&fixture);
    let state = AleFsiState::<3>::new(
        0.25,
        &fixture.mesh,
        &fixture.partition,
        &fixture.motion,
        zero_vertex_vectors_3d(&fixture.mesh),
        fixture
            .partition
            .fluid_cells()
            .iter()
            .copied()
            .map(|cell| (cell, [0.0; 3]))
            .collect(),
        vec![0.0; fixture.partition.fluid_vertices().len()],
        solid_displacement.clone(),
    )
    .expect("tetrahedral ALE state is admitted");
    let all_displacement = fixture
        .motion
        .apply(&solid_displacement)
        .expect("sealed motion derives the complete displacement");
    let expected = current_coordinates::<3>(&fixture.mesh, &all_displacement)
        .expect("current coordinates are finite");
    assert_eq!(state.geometry().coordinates(), expected);
    assert_eq!(state.solid_displacement(), solid_displacement);
    state
        .validate_against(&fixture.mesh, &fixture.partition, &fixture.motion)
        .expect("exact tetrahedral state replays");
    let fixed = state
        .to_fixed_reference_state(&fixture.mesh, &fixture.partition)
        .expect("fixed-reference state bridge remains exact");
    assert_eq!(fixed.vertex_velocity(), state.vertex_velocity());
    assert_eq!(fixed.solid_displacement(), state.solid_displacement());

    assert!(
        AleFsiState::<3>::new(
            0.0,
            &fixture.mesh,
            &fixture.partition,
            &fixture.motion,
            zero_vertex_vectors_3d(&fixture.mesh),
            fixture
                .partition
                .fluid_cells()
                .iter()
                .copied()
                .map(|cell| (cell, [0.0; 3]))
                .collect(),
            vec![0.0; fixture.partition.fluid_vertices().len() - 1],
            zero_vertex_vectors_3d(&fixture.mesh),
        )
        .is_err()
    );
    assert!(
        AleFsiState::<3>::new(
            0.0,
            &fixture.mesh,
            &fixture.partition,
            &fixture.motion,
            zero_vertex_vectors_3d(&fixture.mesh)[1..].to_vec(),
            fixture
                .partition
                .fluid_cells()
                .iter()
                .copied()
                .map(|cell| (cell, [0.0; 3]))
                .collect(),
            vec![0.0; fixture.partition.fluid_vertices().len()],
            zero_vertex_vectors_3d(&fixture.mesh),
        )
        .is_err()
    );
}

#[test]
fn tetrahedral_restart_and_step_action_fail_closed_against_substituted_geometry() {
    let fixture = fixture_3d();
    let plan = valid_plan_3d();
    let previous = zero_state_3d(0.0, &fixture);
    let current = AleFsiState::<3>::new(
        plan.time_step(),
        &fixture.mesh,
        &fixture.partition,
        &fixture.motion,
        zero_vertex_vectors_3d(&fixture.mesh),
        fixture
            .partition
            .fluid_cells()
            .iter()
            .copied()
            .map(|cell| (cell, [0.0; 3]))
            .collect(),
        vec![0.0; fixture.partition.fluid_vertices().len()],
        moving_solid_displacement_3d(&fixture),
    )
    .expect("moving tetrahedral state is admitted");
    let action = plan
        .geometry_action(
            &fixture.mesh,
            &fixture.partition,
            &fixture.motion,
            &previous,
            &current,
        )
        .expect("consecutive states derive one geometry action");
    assert_eq!(action.time_step(), plan.time_step());
    assert_eq!(
        action.vertex_velocities().len(),
        fixture.mesh.vertices().len()
    );
    let _boundary: AleFsiBoundary<3> =
        FixedReferenceFsiBoundary::<3>::homogeneous_exterior(&fixture.mesh)
            .expect("tetrahedral exterior boundary closes");

    let mut substituted = current.clone();
    let mut coordinates = substituted.geometry.coordinates().to_vec();
    coordinates[0][0] += 1.0e-3;
    substituted.geometry = FixedTopologyGeometryState::<3>::new(&fixture.mesh, coordinates)
        .expect("substituted geometry remains individually admissible");
    assert!(
        substituted
            .validate_against(&fixture.mesh, &fixture.partition, &fixture.motion)
            .is_err()
    );
    assert!(
        plan.geometry_action(
            &fixture.mesh,
            &fixture.partition,
            &fixture.motion,
            &previous,
            &substituted,
        )
        .is_err()
    );

    let wrong_time = zero_state_3d(2.0 * plan.time_step(), &fixture);
    assert!(
        plan.geometry_action(
            &fixture.mesh,
            &fixture.partition,
            &fixture.motion,
            &previous,
            &wrong_time,
        )
        .is_err()
    );
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
    let mesh = two_domain_mesh_with_fluid_interior();
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
    let mesh = two_domain_tetrahedral_mesh_with_fluid_interior();
    let (fluid, solid, interface) = inventories_3d(&mesh);
    let partition = FixedReferenceFsiPartition::<3>::new(&mesh, fluid, solid, interface)
        .expect("exact tetrahedral material partition");
    let motion = P1HarmonicMeshMotionAction::<3>::new(&mesh, &partition, harmonic_solver())
        .expect("tetrahedral harmonic motion seals");
    Fixture3d {
        mesh,
        partition,
        motion,
    }
}

fn two_domain_mesh_with_fluid_interior() -> SimplicialMesh {
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

fn two_domain_tetrahedral_mesh_with_fluid_interior() -> SimplicialMesh {
    let x_coordinates = [0.0, 0.5, 1.0, 2.0];
    let y_coordinates = [0.0, 0.5, 1.0];
    let z_coordinates = [0.0, 0.5, 1.0];
    let nx = x_coordinates.len();
    let ny = y_coordinates.len();
    let vertex = |x: usize, y: usize, z: usize| z * ny * nx + y * nx + x;
    let vertices = z_coordinates
        .iter()
        .flat_map(|&z| {
            y_coordinates
                .iter()
                .flat_map(move |&y| x_coordinates.into_iter().map(move |x| vec![x, y, z]))
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
                    cells.push(tetrahedron);
                }
            }
        }
    }
    SimplicialMesh::new(
        3,
        vertices,
        cells,
        MeshQualityGate::new(0.02).expect("valid tetrahedral quality gate"),
    )
    .expect("valid conforming tetrahedral test mesh")
}

fn signed_tetrahedron_measure(vertices: &[Vec<f64>], cell: &[usize]) -> f64 {
    let origin = &vertices[cell[0]];
    let column = |vertex: usize, axis: usize| vertices[cell[vertex]][axis] - origin[axis];
    column(1, 0) * (column(2, 1) * column(3, 2) - column(3, 1) * column(2, 2))
        - column(2, 0) * (column(1, 1) * column(3, 2) - column(3, 1) * column(1, 2))
        + column(3, 0) * (column(1, 1) * column(2, 2) - column(2, 1) * column(1, 2))
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

fn inventories_3d(mesh: &SimplicialMesh) -> (Vec<CellId>, Vec<CellId>, Vec<FacetId>) {
    let mut fluid = Vec::new();
    let mut solid = Vec::new();
    for (index, cell) in mesh.cells().iter().enumerate() {
        let centroid_x = cell
            .iter()
            .map(|vertex| mesh.vertices()[*vertex][0])
            .sum::<f64>()
            / 4.0;
        if centroid_x < 1.0 {
            fluid.push(CellId::new(index));
        } else {
            solid.push(CellId::new(index));
        }
    }
    let interface = (0..mesh.entity_count(2).expect("3D mesh owns facets"))
        .filter(|&facet| {
            mesh.entity_vertices(MeshEntity::new(2, facet))
                .expect("test facet owns vertices")
                .iter()
                .all(|vertex| mesh.vertices()[vertex.index()][0] == 1.0)
        })
        .map(FacetId::new)
        .collect();
    (fluid, solid, interface)
}

fn moving_solid_displacement(fixture: &Fixture) -> Vec<[f64; COMPONENTS]> {
    let mut displacement = zero_vertex_vectors(&fixture.mesh);
    for vertex in fixture.partition.solid_vertices() {
        let point = &fixture.mesh.vertices()[vertex.index()];
        displacement[vertex.index()] = [0.01 * point[1], 0.005 * point[0]];
    }
    displacement
}

fn moving_solid_displacement_3d(fixture: &Fixture3d) -> Vec<[f64; 3]> {
    let mut displacement = zero_vertex_vectors_3d(&fixture.mesh);
    for vertex in fixture.partition.solid_vertices() {
        let point = &fixture.mesh.vertices()[vertex.index()];
        displacement[vertex.index()] = [
            0.005 * point[1],
            0.003 * point[0] - 0.002 * point[2],
            0.004 * point[1],
        ];
    }
    displacement
}

fn zero_state(time: f64, fixture: &Fixture) -> AleFsiState<2> {
    AleFsiState::<2>::new(
        time,
        &fixture.mesh,
        &fixture.partition,
        &fixture.motion,
        zero_vertex_vectors(&fixture.mesh),
        fixture
            .partition
            .fluid_cells()
            .iter()
            .copied()
            .map(|cell| (cell, [0.0; COMPONENTS]))
            .collect(),
        vec![0.0; fixture.partition.fluid_vertices().len()],
        zero_vertex_vectors(&fixture.mesh),
    )
    .unwrap()
}

fn zero_state_3d(time: f64, fixture: &Fixture3d) -> AleFsiState<3> {
    AleFsiState::<3>::new(
        time,
        &fixture.mesh,
        &fixture.partition,
        &fixture.motion,
        zero_vertex_vectors_3d(&fixture.mesh),
        fixture
            .partition
            .fluid_cells()
            .iter()
            .copied()
            .map(|cell| (cell, [0.0; 3]))
            .collect(),
        vec![0.0; fixture.partition.fluid_vertices().len()],
        zero_vertex_vectors_3d(&fixture.mesh),
    )
    .expect("valid zero tetrahedral state")
}

fn zero_vertex_vectors(mesh: &SimplicialMesh) -> Vec<[f64; COMPONENTS]> {
    vec![[0.0; COMPONENTS]; mesh.vertices().len()]
}

fn zero_vertex_vectors_3d(mesh: &SimplicialMesh) -> Vec<[f64; 3]> {
    vec![[0.0; 3]; mesh.vertices().len()]
}

fn valid_plan() -> AleFsiStepPlan<2> {
    AleFsiStepPlan::<2>::new(
        0.05,
        material(),
        FixedReferenceFsiScale::<2>::new(2.0, 1.0, 1.0).unwrap(),
        FixedReferenceFsiLoad::Zero,
        nonlinear(),
        general_solver(),
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
    )
    .unwrap()
}

fn valid_plan_3d() -> AleFsiStepPlan<3> {
    AleFsiStepPlan::<3>::new(
        0.05,
        FixedReferenceFsiMaterial::<3>::new(1.0, 0.1, 1.0, 2.0, 1.0)
            .expect("coercive tetrahedral material"),
        FixedReferenceFsiScale::<3>::new(2.0, 1.0, 1.0).expect("finite tetrahedral scales"),
        FixedReferenceFsiLoad::Zero,
        nonlinear(),
        general_solver(),
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
    )
    .expect("valid tetrahedral ALE plan")
}

fn material() -> FixedReferenceFsiMaterial<2> {
    FixedReferenceFsiMaterial::<2>::new(1.0, 0.1, 1.0, 2.0, 1.0).unwrap()
}

fn nonlinear() -> NonlinearSolvePlan {
    NonlinearSolvePlan::new(1.0e-9, 1.0e-12, NonZeroUsize::new(20).unwrap(), 12).unwrap()
}

fn general_solver() -> SolverPlan {
    SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1.0e-10,
        1.0e-12,
        NonZeroUsize::new(500).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Identity)
    .with_reduction(ReductionPolicy::Fast)
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
