use std::num::NonZeroUsize;

use eqiora_meshing::{CellId, FixedTopologyGeometryState2d, MeshQualityGate, SimplicialMesh};
use eqiora_realization::{NonlinearSolvePlan, Target};
use eqiora_solver::{
    LinearOperatorProperties, LinearSolveRequest, LinearSolver, PreconditionerPolicy,
    REFERENCE_LINEAR_SOLVER, ReductionPolicy, SolverPlan,
};

use super::*;
use crate::simplicial_ale_fsi::test_support::{
    fluid_domain, material as exact_material, motion, partition, physical_state,
    solid_displacement, solid_domain,
};
use crate::simplicial_fsi::{
    FixedReferenceFsiLoad, FixedReferenceFsiPartition, FixedReferenceFsiScale,
};

#[test]
fn state_derives_geometry_from_the_exact_displacement_field() {
    let fixture = fixture();
    let physical = physical_state(&fixture.partition, |vertex| {
        let point = &fixture.mesh.vertices()[vertex.index()];
        [0.01 * point[1], 0.005 * point[0]]
    });
    let state = AleFsiState::new(
        0.25,
        &fixture.mesh,
        &fixture.partition,
        &fixture.motion,
        physical,
    )
    .unwrap();

    let driver = state
        .physical_state()
        .vector_vertices(solid_displacement())
        .unwrap();
    let displacement = fixture.motion.apply(solid_displacement(), &driver).unwrap();
    for (index, (reference, current)) in fixture
        .mesh
        .vertices()
        .iter()
        .zip(state.geometry().coordinates())
        .enumerate()
    {
        for component in 0..2 {
            assert_eq!(
                current[component],
                reference[component] + displacement[index][component]
            );
        }
    }
    assert_eq!(state.time(), 0.25);
    assert_eq!(state.physical_state().fields().count(), 4);
    state
        .validate_against(&fixture.mesh, &fixture.partition, &fixture.motion)
        .unwrap();
}

#[test]
fn state_rejects_nonfinite_exact_driver_and_substituted_geometry() {
    let fixture = fixture();
    let first = fixture.partition.domain_vertices(solid_domain()).unwrap()[0];
    let invalid = physical_state(&fixture.partition, |vertex| {
        if vertex == first {
            [f64::NAN, 0.0]
        } else {
            [0.0; 2]
        }
    });
    assert!(
        AleFsiState::new(
            0.0,
            &fixture.mesh,
            &fixture.partition,
            &fixture.motion,
            invalid,
        )
        .is_err()
    );

    let mut state = zero_state(0.0, &fixture);
    let mut coordinates = fixture.mesh.vertices().to_vec();
    coordinates[0][0] += 0.01;
    state.geometry = FixedTopologyGeometryState2d::new(&fixture.mesh, coordinates).unwrap();
    assert!(
        state
            .validate_against(&fixture.mesh, &fixture.partition, &fixture.motion)
            .is_err()
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
    assert_eq!(
        plan.target(),
        Target::HostCpu {
            threads: NonZeroUsize::MIN
        }
    );

    let previous = zero_state(0.0, &fixture);
    let current = zero_state(plan.time_step(), &fixture);
    assert_eq!(
        plan.geometry_action(
            &fixture.mesh,
            &fixture.partition,
            &fixture.motion,
            &previous,
            &current,
        )
        .unwrap()
        .time_step(),
        plan.time_step()
    );
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
}

#[test]
fn step_plan_rejects_incompatible_solver_or_target() {
    let symmetric = SolverPlan::new(
        LinearSolver::MinimumResidual,
        1.0e-10,
        1.0e-12,
        NonZeroUsize::new(500).unwrap(),
    )
    .unwrap();
    for (solver, target) in [
        (
            symmetric,
            Target::HostCpu {
                threads: NonZeroUsize::MIN,
            },
        ),
        (
            general_solver(),
            Target::HostCpu {
                threads: NonZeroUsize::new(2).unwrap(),
            },
        ),
    ] {
        assert!(
            AleFsiStepPlan::<2>::new(
                0.05,
                exact_material(),
                FixedReferenceFsiScale::new(2.0, 1.0, 1.0).unwrap(),
                FixedReferenceFsiLoad::Zero,
                nonlinear(),
                solver,
                target,
            )
            .is_err()
        );
    }
}

struct Fixture {
    mesh: SimplicialMesh,
    partition: FixedReferenceFsiPartition<2>,
    motion: P1HarmonicMeshMotionAction<2>,
}

fn fixture() -> Fixture {
    let mesh = two_domain_mesh();
    let (fluid, solid) = inventories(&mesh);
    let partition = partition(&mesh, fluid, solid);
    let motion = motion(&mesh, &partition, harmonic_solver());
    assert!(partition.domain_cells(fluid_domain()).is_some());
    Fixture {
        mesh,
        partition,
        motion,
    }
}

fn two_domain_mesh() -> SimplicialMesh {
    let xs = [0.0, 0.5, 1.0, 1.5, 2.0];
    let vertices = [0.0, 0.5, 1.0]
        .into_iter()
        .flat_map(|y| xs.into_iter().map(move |x| vec![x, y]))
        .collect::<Vec<_>>();
    let mut cells = Vec::new();
    for row in 0..2 {
        for column in 0..xs.len() - 1 {
            let lower_left = row * xs.len() + column;
            let lower_right = lower_left + 1;
            let upper_left = lower_left + xs.len();
            let upper_right = upper_left + 1;
            cells.push(vec![lower_left, lower_right, upper_right]);
            cells.push(vec![lower_left, upper_right, upper_left]);
        }
    }
    SimplicialMesh::new(2, vertices, cells, MeshQualityGate::new(0.3).unwrap()).unwrap()
}

fn inventories(mesh: &SimplicialMesh) -> (Vec<CellId>, Vec<CellId>) {
    (0..mesh.cells().len()).map(CellId::new).partition(|cell| {
        mesh.cells()[cell.index()]
            .iter()
            .map(|&vertex| mesh.vertices()[vertex][0])
            .sum::<f64>()
            / 3.0
            < 1.0
    })
}

fn zero_state(time: f64, fixture: &Fixture) -> AleFsiState<2> {
    AleFsiState::new(
        time,
        &fixture.mesh,
        &fixture.partition,
        &fixture.motion,
        physical_state(&fixture.partition, |_vertex| [0.0; 2]),
    )
    .unwrap()
}

fn valid_plan() -> AleFsiStepPlan<2> {
    AleFsiStepPlan::new(
        0.05,
        exact_material(),
        FixedReferenceFsiScale::new(2.0, 1.0, 1.0).unwrap(),
        FixedReferenceFsiLoad::Zero,
        nonlinear(),
        general_solver(),
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
    )
    .unwrap()
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
