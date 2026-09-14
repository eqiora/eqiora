#![cfg(feature = "faer")]

use eqiora::backends::faer::FaerLinearSolver;
use eqiora::meshing::{OverlapCoordinateChart2d, SimplicialRevisionOverlap2d};
use eqiora::solver::REFERENCE_LINEAR_SOLVER;
use eqiora_numerics::{
    ale::finalize_resolved_fixed_topology_ale_fsi_2d,
    ale::remesh_resolved_fixed_topology_ale_fsi_2d,
};

#[path = "support/remeshing_transfer_2d_artifact.rs"]
mod artifact;
#[path = "support/remeshing_transfer_2d_case.rs"]
mod case;

use artifact::assert_artifact_vertical_slice;
use case::{
    Case, TIME_STEP, assert_exact_interface_bisection, assert_genuine_many_to_many,
    assert_interface_witness, assert_numerical_falsifiers, assert_scale_invariant_projection,
    assert_strong_source_witness, one_step, state_fields, transfer_plan,
};

#[test]
fn cpu_reference_closes_remesh_target_continuation_and_v3_trajectory_replay() {
    let case = Case::new();
    let source_resolved = case.resolve(case.source_reference, 1, TIME_STEP, None);
    let source_finalized = finalize_resolved_fixed_topology_ale_fsi_2d(
        &case.canonical,
        &source_resolved,
        case.source_reference,
        &case.source_mesh,
        &case.source_partition,
        &case.source_boundary,
        case.initial_physical(&source_resolved),
        &FaerLinearSolver,
    )
    .unwrap();
    let source_trajectory = source_finalized
        .clone()
        .solve(one_step(), &FaerLinearSolver)
        .unwrap();
    let source_state = source_trajectory.final_state();
    assert_eq!(source_state.time(), TIME_STEP);
    assert_strong_source_witness(&case, source_state);

    let target_resolved = case.resolve(case.target_reference, 2, TIME_STEP, None);
    let material_probe = SimplicialRevisionOverlap2d::new(
        OverlapCoordinateChart2d::Material,
        &case.source_mesh,
        case.source_partition
            .domain_cells(case.fields.solid_domain)
            .unwrap(),
        &case.target_mesh,
        case.target_partition
            .domain_cells(case.fields.solid_domain)
            .unwrap(),
    )
    .unwrap();
    assert_genuine_many_to_many(&material_probe, "material solid");
    assert_numerical_falsifiers(&case, &source_finalized, source_state, &target_resolved);
    let transfer_plan = transfer_plan();
    let accepted = remesh_resolved_fixed_topology_ale_fsi_2d(
        &case.canonical,
        &source_finalized,
        source_state,
        &target_resolved,
        case.target_reference,
        &case.target_mesh,
        &case.target_partition,
        &case.target_boundary,
        transfer_plan,
        &FaerLinearSolver,
        &REFERENCE_LINEAR_SOLVER,
    )
    .unwrap();
    let projection = accepted.projection().clone();
    assert_eq!(projection.time(), source_state.time());
    assert_genuine_many_to_many(
        projection.evidence().solid_reference_overlap(),
        "material solid",
    );
    assert_genuine_many_to_many(
        projection.evidence().fluid_current_overlap(),
        "current-spatial fluid",
    );
    let projected_fields = state_fields(
        &case,
        projection.physical_state(),
        &case.target_mesh,
        &case.target_partition,
    );
    assert_interface_witness(
        &case.target_mesh,
        &projected_fields.vertex_velocity,
        &projected_fields.solid_displacement,
        "target",
    );
    let source_fields = state_fields(
        &case,
        source_state.physical_state(),
        &case.source_mesh,
        &case.source_partition,
    );
    assert_exact_interface_bisection(
        &case.source_mesh,
        &source_fields.solid_displacement,
        &case.target_mesh,
        projection.evidence().target_geometry().coordinates(),
    );
    assert_scale_invariant_projection(
        &case,
        &source_finalized,
        source_state,
        &accepted,
        &projection,
        transfer_plan,
    );
    let target_trajectory = accepted
        .target()
        .clone()
        .solve(one_step(), &FaerLinearSolver)
        .unwrap();
    assert_eq!(target_trajectory.initial_state().time(), TIME_STEP);
    assert_eq!(target_trajectory.final_state().time(), 2.0 * TIME_STEP);
    assert_eq!(target_trajectory.steps().len(), 1);
    assert!(
        target_trajectory.steps()[0].final_residual_norm()
            <= target_trajectory.steps()[0].residual_target()
    );

    assert_artifact_vertical_slice(
        &case,
        &source_resolved,
        &target_resolved,
        &source_trajectory,
        &target_trajectory,
        &projection,
        transfer_plan,
    );
}
