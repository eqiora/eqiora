use eqiora_core::ScalarType;
use std::f64::consts::PI;
use std::num::NonZeroUsize;

use eqiora::artifact::CartesianMeshEnvelopeV1;
use eqiora::meshing::{MeshEntity, MeshGeometry, MeshTopology};
use eqiora::realization::{
    DiscretizationMethod, MeshKind, MeshPolicy, NonlinearSolvePlan, RealizationCapabilities,
    RealizationRevision, SemanticRevision, SpatialDimensionSupport, TargetCapabilities,
    TransientCellCenteredIncompressibleFlowCapabilities,
    TransientCellCenteredIncompressibleFlowRealizationRequest, VectorLayoutKind,
    resolve_transient_cell_centered_incompressible_flow,
};
use eqiora::solver::{
    LinearOperatorProperties, LinearSolver, PreconditionerPolicy, REFERENCE_LINEAR_SOLVER,
    ReductionPolicy, SolverCapabilities, SolverCapability, SolverPlan,
};
use eqiora::{DimExponents, DynQuantity};
use eqiora_meshing::CartesianMesh;
use eqiora_numerics::{
    common::NonZeroStepCount, fluid::CellCenteredNavierStokesInitialState2d,
    fluid::CellCenteredPressureField2d, fluid::CellCenteredVelocityField2d,
    fluid::IncompressibleFlowScaleProfile2d, fluid::ResolvedCellCenteredNavierStokesTrajectory2d,
    fluid::TransientNavierStokesRun2d,
    fluid::advance_resolved_transient_navier_stokes_cell_centered_2d,
    fluid::lower_transient_incompressible_navier_stokes_cartesian_2d,
    fluid::transient_navier_stokes_cell_centered_plan_2d,
    fluid::transient_navier_stokes_cell_centered_requirements_2d,
};

const SOURCE: &str =
    include_str!("../../../verify/fluid/cell-centered-navier-stokes-fvm-2d/models/direct.eqi");

const LENGTH: DimExponents =
    DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension");
const VELOCITY: DimExponents =
    DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).expect("bounded dimension");
const PRESSURE: DimExponents =
    DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).expect("bounded dimension");
const TIME: DimExponents =
    DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).expect("bounded dimension");

#[derive(Clone, Copy)]
enum InitialTransform {
    Identity,
    Negated,
    ReflectedX,
}

#[test]
fn one_canonical_model_advances_through_collocated_fvm_without_checkerboard_nullspace() {
    let trajectory = run_case([1.0, 1.0, 1.0], 0.01, 2, InitialTransform::Identity);

    assert_eq!(trajectory.states().len(), 3);
    assert_eq!(trajectory.steps().len(), 2);
    assert_eq!(trajectory.solver_backend().as_str(), "eqiora.reference");
    assert!(trajectory.states()[0].velocity().volume_l2_norm().unwrap() > 0.0);
    assert_accepted(&trajectory);
}

#[test]
fn physical_result_is_scale_invariant_and_oriented_under_reflection_and_reversal() {
    let baseline = run_case([1.0, 1.0, 1.0], 0.01, 2, InitialTransform::Identity);
    let nonunit = run_case([2.5, 0.4, 3.0], 0.01, 2, InitialTransform::Identity);
    let reflected = run_case([1.0, 1.0, 1.0], 0.01, 2, InitialTransform::ReflectedX);
    let reversed = run_case([1.0, 1.0, 1.0], 0.01, 2, InitialTransform::Negated);

    assert!(velocity_difference(final_state(&baseline), final_state(&nonunit)) < 2.0e-8);
    assert!(pressure_difference(final_state(&baseline), final_state(&nonunit)) < 2.0e-8);
    assert_reflected_x(final_state(&baseline), final_state(&reflected), 2.0e-8);
    assert_eq!(
        baseline.states()[0].velocity().volume_l2_norm().unwrap(),
        reversed.states()[0].velocity().volume_l2_norm().unwrap()
    );
    assert!(
        reversed
            .states()
            .last()
            .unwrap()
            .velocity()
            .volume_l2_norm()
            .unwrap()
            > 0.0
    );
    assert_accepted(&nonunit);
    assert_accepted(&reflected);
    assert_accepted(&reversed);
}

#[test]
fn backward_euler_step_refinement_is_first_order_on_one_fixed_fvm_mesh() {
    let coarse = run_case([1.0, 1.0, 1.0], 0.0025, 2, InitialTransform::Identity);
    let medium = run_case([1.0, 1.0, 1.0], 0.00125, 4, InitialTransform::Identity);
    let fine = run_case([1.0, 1.0, 1.0], 0.000625, 8, InitialTransform::Identity);
    let coarse_medium = velocity_difference(final_state(&coarse), final_state(&medium));
    let medium_fine = velocity_difference(final_state(&medium), final_state(&fine));
    let ratio = coarse_medium / medium_fine;
    assert!(
        (1.6..=2.4).contains(&ratio),
        "backward-Euler step-doubling ratio was {ratio:e} ({coarse_medium:e}/{medium_fine:e})"
    );
}

fn run_case(
    scale_values: [f64; 3],
    time_step: f64,
    steps: usize,
    transform: InitialTransform,
) -> ResolvedCellCenteredNavierStokesTrajectory2d {
    let document = eqiora::api::ModelDocument::compile(
        "verify/fluid/cell-centered-navier-stokes-fvm-2d/models/direct.eqi",
        SOURCE,
    )
    .unwrap();
    let replay = eqiora::api::ModelDocument::replay(&document.canonical_json().unwrap()).unwrap();
    assert_eq!(replay.digest().unwrap(), document.digest().unwrap());
    assert_eq!(replay.program().model(), document.program().model());
    assert_eq!(replay.program().revision(), document.program().revision());
    let model =
        lower_transient_incompressible_navier_stokes_cartesian_2d(document.program()).unwrap();
    let scales = IncompressibleFlowScaleProfile2d::new(
        DynQuantity::new(scale_values[0], LENGTH),
        DynQuantity::new(scale_values[1], VELOCITY),
        DynQuantity::new(scale_values[2], PRESSURE),
    )
    .unwrap();
    let linear = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1.0e-11,
        1.0e-13,
        NonZeroUsize::new(4_000).unwrap(),
    )
    .unwrap()
    .with_reduction(ReductionPolicy::Reproducible);
    let mesh = CartesianMesh::uniform(model.bounds(), &[4, 4]).unwrap();
    let mesh_envelope = CartesianMeshEnvelopeV1::from_mesh(&mesh).unwrap();
    let plan = transient_navier_stokes_cell_centered_plan_2d(
        &model,
        MeshPolicy::SuppliedCartesian {
            artifact: mesh_envelope.artifact_reference().unwrap(),
            cells: [NonZeroUsize::new(4).unwrap(); 2],
        },
        scales,
        DynQuantity::new(time_step, TIME),
        NonlinearSolvePlan::new(1.0e-9, 1.0e-11, NonZeroUsize::new(16).unwrap(), 12).unwrap(),
        linear,
    )
    .unwrap();
    let request = TransientCellCenteredIncompressibleFlowRealizationRequest::explicit(
        document.program().model(),
        SemanticRevision::new(document.program().revision().0),
        RealizationRevision::new(1),
        plan,
    );
    let resolved = resolve_transient_cell_centered_incompressible_flow(
        &request,
        transient_navier_stokes_cell_centered_requirements_2d(&model),
        &capabilities(),
    )
    .unwrap();
    let streamfunction = (0..mesh.entity_count(2).unwrap())
        .map(|cell| {
            let center = mesh.entity_center(MeshEntity::new(2, cell)).unwrap();
            0.02 * (PI * center[0]).sin().powi(2) * (PI * center[1]).sin().powi(2)
        })
        .collect::<Vec<_>>();
    let base_velocity = (0..mesh.entity_count(2).unwrap())
        .map(|cell| {
            let indices = mesh.cell_multi_index(MeshEntity::new(2, cell)).unwrap();
            [
                boundary_closed_difference(&mesh, &streamfunction, indices, 1),
                -boundary_closed_difference(&mesh, &streamfunction, indices, 0),
            ]
        })
        .collect::<Vec<_>>();
    let velocity_values = transformed_velocity(&mesh, &base_velocity, transform);
    let history = physical_face_volume_fluxes(&mesh, &velocity_values);
    let velocity = CellCenteredVelocityField2d::new(mesh.clone(), velocity_values).unwrap();
    let pressure = CellCenteredPressureField2d::new(mesh, vec![0.0; 16]).unwrap();
    assert!(
        CellCenteredNavierStokesInitialState2d::new(
            &model,
            DynQuantity::new(0.0, TIME),
            velocity.clone(),
            pressure.clone(),
            0.0,
            Vec::new(),
        )
        .is_err(),
        "missing method history was admitted",
    );
    let mut corrupt_history = history.clone();
    corrupt_history[0] = f64::NAN;
    assert!(
        CellCenteredNavierStokesInitialState2d::new(
            &model,
            DynQuantity::new(0.0, TIME),
            velocity.clone(),
            pressure.clone(),
            0.0,
            corrupt_history,
        )
        .is_err(),
        "non-finite method history was admitted",
    );
    let initial = CellCenteredNavierStokesInitialState2d::new(
        &model,
        DynQuantity::new(0.0, TIME),
        velocity,
        pressure,
        0.0,
        history,
    )
    .unwrap();
    let trajectory = advance_resolved_transient_navier_stokes_cell_centered_2d(
        document.program(),
        &resolved,
        &mesh_envelope,
        initial,
        TransientNavierStokesRun2d::new(NonZeroStepCount::new(NonZeroUsize::new(steps).unwrap())),
        &REFERENCE_LINEAR_SOLVER,
    )
    .unwrap();
    assert_eq!(trajectory.model(), &model);
    assert_eq!(trajectory.realization(), &resolved);
    trajectory
}

fn physical_face_volume_fluxes(mesh: &CartesianMesh, velocity: &[[f64; 2]]) -> Vec<f64> {
    (0..mesh.entity_count(1).unwrap())
        .map(|index| {
            let facet = MeshEntity::new(1, index);
            let free_axes = mesh.entity_free_axes(facet).unwrap();
            let normal_axis = (0..2)
                .find(|axis| free_axes.binary_search(axis).is_err())
                .unwrap();
            let adjacent = mesh.incidence(facet, 2).unwrap();
            let measure = 2.0 * mesh.geometry_map(facet).unwrap().measure_scale();
            match adjacent.as_slice() {
                [first, second] => {
                    measure
                        * 0.5
                        * (velocity[first.entity.index()][normal_axis]
                            + velocity[second.entity.index()][normal_axis])
                }
                [_] => 0.0,
                _ => unreachable!("Cartesian facet has one or two adjacent cells"),
            }
        })
        .collect()
}

fn assert_accepted(trajectory: &ResolvedCellCenteredNavierStokesTrajectory2d) {
    for (index, step) in trajectory.steps().iter().enumerate() {
        assert!(step.iterations() > 0);
        if index == 0 {
            assert!(step.initial_continuity_norm() < 1.0e-14);
        } else {
            assert!(step.initial_continuity_norm().is_finite());
        }
        assert!(step.momentum_residual_norm() <= step.momentum_residual_target());
        assert!(step.continuity_residual_norm() <= step.continuity_residual_target());
        assert!(step.gauge_residual().abs() <= step.gauge_residual_target());
        assert!(step.maximum_momentum_replay_defect() <= step.replay_tolerance());
        assert!(step.maximum_continuity_replay_defect() <= step.replay_tolerance());
        assert!(step.maximum_centered_jvp_defect() < 2.0e-7);
        assert!(step.maximum_face_cancellation_defect() <= step.replay_tolerance());
        assert!(step.maximum_flux_reuse_defect() <= step.replay_tolerance());
        assert!(step.global_mass_defect() <= step.replay_tolerance());
        assert!(step.maximum_affine_pressure_correction() < 1.0e-12);
        assert!(step.minimum_checkerboard_correction_norm() > 1.0e-12);
    }
}

fn transformed_velocity(
    mesh: &CartesianMesh,
    values: &[[f64; 2]],
    transform: InitialTransform,
) -> Vec<[f64; 2]> {
    match transform {
        InitialTransform::Identity => values.to_vec(),
        InitialTransform::Negated => values.iter().map(|value| [-value[0], -value[1]]).collect(),
        InitialTransform::ReflectedX => (0..values.len())
            .map(|cell| {
                let indices = mesh.cell_multi_index(MeshEntity::new(2, cell)).unwrap();
                let source = mesh
                    .cell_at(&[
                        mesh.axis_cell_count(0).unwrap() - 1 - indices[0],
                        indices[1],
                    ])
                    .unwrap()
                    .index();
                [-values[source][0], values[source][1]]
            })
            .collect(),
    }
}

fn final_state(
    trajectory: &ResolvedCellCenteredNavierStokesTrajectory2d,
) -> &eqiora_numerics::fluid::ResolvedCellCenteredNavierStokesState2d {
    trajectory.states().last().unwrap()
}

fn velocity_difference(
    left: &eqiora_numerics::fluid::ResolvedCellCenteredNavierStokesState2d,
    right: &eqiora_numerics::fluid::ResolvedCellCenteredNavierStokesState2d,
) -> f64 {
    let values = left
        .velocity()
        .values()
        .iter()
        .zip(right.velocity().values())
        .map(|(left, right)| [left[0] - right[0], left[1] - right[1]])
        .collect();
    CellCenteredVelocityField2d::new(left.velocity().mesh().clone(), values)
        .unwrap()
        .volume_l2_norm()
        .unwrap()
}

fn pressure_difference(
    left: &eqiora_numerics::fluid::ResolvedCellCenteredNavierStokesState2d,
    right: &eqiora_numerics::fluid::ResolvedCellCenteredNavierStokesState2d,
) -> f64 {
    left.pressure()
        .values()
        .iter()
        .zip(right.pressure().values())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max)
}

fn assert_reflected_x(
    original: &eqiora_numerics::fluid::ResolvedCellCenteredNavierStokesState2d,
    reflected: &eqiora_numerics::fluid::ResolvedCellCenteredNavierStokesState2d,
    tolerance: f64,
) {
    let mesh = original.velocity().mesh();
    let reflected_velocity = transformed_velocity(
        mesh,
        original.velocity().values(),
        InitialTransform::ReflectedX,
    );
    let maximum_velocity_defect = reflected
        .velocity()
        .values()
        .iter()
        .zip(reflected_velocity)
        .flat_map(|(actual, expected)| {
            [
                (actual[0] - expected[0]).abs(),
                (actual[1] - expected[1]).abs(),
            ]
        })
        .fold(0.0, f64::max);
    let x_cells = mesh.axis_cell_count(0).unwrap();
    let maximum_pressure_defect = (0..original.pressure().values().len())
        .map(|cell| {
            let indices = mesh.cell_multi_index(MeshEntity::new(2, cell)).unwrap();
            let source = mesh
                .cell_at(&[x_cells - 1 - indices[0], indices[1]])
                .unwrap()
                .index();
            (reflected.pressure().values()[cell] - original.pressure().values()[source]).abs()
        })
        .fold(0.0, f64::max);
    assert!(maximum_velocity_defect <= tolerance);
    assert!(maximum_pressure_defect <= tolerance);
}

fn boundary_closed_difference(
    mesh: &CartesianMesh,
    values: &[f64],
    indices: &[usize],
    axis: usize,
) -> f64 {
    let count = mesh.axis_cell_count(axis).unwrap();
    let value_at = |axis_index| {
        let mut neighbor = indices.to_vec();
        neighbor[axis] = axis_index;
        values[mesh.cell_at(&neighbor).unwrap().index()]
    };
    let difference = if indices[axis] == 0 {
        0.5 * (value_at(0) + value_at(1))
    } else if indices[axis] + 1 == count {
        -0.5 * (value_at(count - 2) + value_at(count - 1))
    } else {
        0.5 * (value_at(indices[axis] + 1) - value_at(indices[axis] - 1))
    };
    let coordinates = mesh.axis_coordinates(axis).unwrap();
    difference / (coordinates[1] - coordinates[0])
}

fn capabilities() -> TransientCellCenteredIncompressibleFlowCapabilities {
    let solver = SolverCapabilities::exact([SolverCapability {
        algorithm: LinearSolver::BiConjugateGradientStabilized,
        operator_properties: LinearOperatorProperties::General,
        preconditioner: PreconditionerPolicy::Identity,
        reduction: ReductionPolicy::Reproducible,
        scalar_type: ScalarType::F64,
    }])
    .unwrap();
    TransientCellCenteredIncompressibleFlowCapabilities::new(
        RealizationCapabilities::cartesian_product(
            [DiscretizationMethod::CellCenteredFiniteVolume],
            [(
                MeshKind::SuppliedCartesian,
                SpatialDimensionSupport::exact(NonZeroUsize::new(2).unwrap()),
            )],
            [VectorLayoutKind::Replicated],
            solver,
            TargetCapabilities::none().with_host_cpu(NonZeroUsize::MIN),
        )
        .unwrap(),
    )
}
