use eqiora_numerics::CommonTransientFlowPlan;
use std::collections::BTreeMap;
use std::num::NonZeroUsize;

use eqiora_artifact::{
    CartesianMeshCellsV2, GeometryMeshCorrespondenceEnvelopeV1, MeshProductionLineageEnvelopeV1,
    ModelEnvelope,
};
use eqiora_geometry::{GeometryGraph, PlanarTopologyHandle};
use eqiora_numerics::{
    AuthenticatedCommonMesh, CommonBackwardEuler, CommonSolvePolicy, CommonSpatialPolicy,
    IncompressibleScalingRequest2d, resolve_common_plan,
};
use eqiora_realization::NonlinearSolvePlan;
use eqiora_solver::REFERENCE_LINEAR_SOLVER;

use super::*;
use crate::ModelDocument;

const SOURCE: &str =
    include_str!("../../../../verify/fluid/cell-centered-navier-stokes-fvm-2d/models/direct.eqi");
const TIME_BACKEND: TimeBackendIdentity = TimeBackendIdentity::new("eqiora.test.time", "1");

#[test]
fn exact_common_transient_request_round_trips_and_normalizes_schedule_spelling() {
    let plan = transient_plan();
    let state = plan.zero_state(0.0).unwrap();
    let by_steps = RunRequest::from_steps(
        ResolvedCommonPlan::TransientFlow(Box::new(plan.clone())),
        state.clone(),
        2,
        vec![1, 2],
    )
    .unwrap();
    let by_times = RunRequest::from_times(
        ResolvedCommonPlan::TransientFlow(Box::new(plan)),
        state,
        0.02,
        vec![0.01, 0.02],
    )
    .unwrap();

    assert_eq!(by_steps.identity(), by_times.identity());
    assert_eq!(by_steps.to_bytes().unwrap(), by_times.to_bytes().unwrap());

    let bytes = by_steps.to_bytes().unwrap();
    let replayed = RunRequest::from_bytes(&bytes, &REFERENCE_LINEAR_SOLVER, TIME_BACKEND).unwrap();
    assert_eq!(replayed, by_steps);
    assert_eq!(replayed.to_bytes().unwrap(), bytes);
    assert_eq!(replayed.native().identity(), by_steps.identity());
    assert_eq!(replayed.into_native().identity(), by_steps.identity());
}

#[test]
fn replay_rejects_false_identity_noncanonical_and_invalid_schedule_content() {
    let request = request();
    let bytes = request.to_bytes().unwrap();

    let mut false_identity: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    false_identity["identity"] = serde_json::Value::String("0".repeat(64));
    assert!(decode_wire(&false_identity).is_err());

    let mut duplicate_output: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    duplicate_output["output_steps"] = serde_json::json!([1, 1]);
    assert!(decode_wire(&duplicate_output).is_err());

    let mut out_of_range: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    out_of_range["output_steps"] = serde_json::json!([1, 3]);
    assert!(decode_wire(&out_of_range).is_err());

    let mut empty_output: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    empty_output["output_steps"] = serde_json::json!([]);
    assert!(decode_wire(&empty_output).is_err());

    let mut unknown: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    unknown["unknown"] = serde_json::json!(true);
    assert!(decode_wire(&unknown).is_err());

    let mut trailing = bytes;
    trailing.push(b'\n');
    assert!(RunRequest::from_bytes(&trailing, &REFERENCE_LINEAR_SOLVER, TIME_BACKEND).is_err());
}

#[test]
fn replay_rejects_cross_wired_roots_bad_base64_and_oversized_input() {
    let request = request();
    let bytes = request.to_bytes().unwrap();

    let mut foreign_plan: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    mutate_embedded_json(&mut foreign_plan, "plan_base64", |plan| {
        plan["family"] = serde_json::json!("scalar");
    });
    assert!(decode_wire(&foreign_plan).is_err());

    let mut foreign_state: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    mutate_embedded_json(&mut foreign_state, "state_base64", |state| {
        state["state_space_identity"] = serde_json::Value::String("0".repeat(64));
    });
    assert!(decode_wire(&foreign_state).is_err());

    let mut bad_base64: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    bad_base64["state_base64"] = serde_json::json!("!");
    assert!(decode_wire(&bad_base64).is_err());

    assert!(
        RunRequest::from_bytes_with_limit(b"{}", &REFERENCE_LINEAR_SOLVER, TIME_BACKEND, 1,)
            .is_err()
    );
}

#[test]
fn preparation_rejects_nonfinite_time_and_invalid_step_schedules() {
    let plan = transient_plan();
    let state = plan.zero_state(0.0).unwrap();
    assert!(
        RunRequest::from_times(
            ResolvedCommonPlan::TransientFlow(Box::new(plan.clone())),
            state.clone(),
            f64::NAN,
            vec![0.01]
        )
        .is_err()
    );
    assert!(
        RunRequest::from_steps(
            ResolvedCommonPlan::TransientFlow(Box::new(plan.clone())),
            state.clone(),
            0,
            vec![1]
        )
        .is_err()
    );
    assert!(
        RunRequest::from_steps(
            ResolvedCommonPlan::TransientFlow(Box::new(plan.clone())),
            state.clone(),
            2,
            vec![]
        )
        .is_err()
    );
    assert!(
        RunRequest::from_steps(
            ResolvedCommonPlan::TransientFlow(Box::new(plan)),
            state,
            2,
            vec![2, 1]
        )
        .is_err()
    );
}

fn request() -> RunRequest {
    let plan = transient_plan();
    let state = plan.zero_state(0.0).unwrap();
    RunRequest::from_steps(
        ResolvedCommonPlan::TransientFlow(Box::new(plan)),
        state,
        2,
        vec![1, 2],
    )
    .unwrap()
}

fn transient_plan() -> CommonTransientFlowPlan {
    let document = ModelDocument::compile("transient-direct.eqi", SOURCE).unwrap();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let geometry_graph = GeometryGraph::new();
    let rectangle = geometry_graph.rectangle([0.0, 1.0], [0.0, 1.0]).unwrap();
    let edges = rectangle.boundaries();
    let geometry = geometry_graph
        .build(
            &rectangle,
            &BTreeMap::from([
                ("body".to_owned(), vec![rectangle.region().into()]),
                (
                    "x_lower".to_owned(),
                    vec![PlanarTopologyHandle::from(edges[0])],
                ),
                (
                    "x_upper".to_owned(),
                    vec![PlanarTopologyHandle::from(edges[1])],
                ),
                (
                    "y_lower".to_owned(),
                    vec![PlanarTopologyHandle::from(edges[2])],
                ),
                (
                    "y_upper".to_owned(),
                    vec![PlanarTopologyHandle::from(edges[3])],
                ),
            ]),
        )
        .unwrap();
    let cells = CartesianMeshCellsV2::new([2, 3]).unwrap();
    let (mesh, correspondence) =
        GeometryMeshCorrespondenceEnvelopeV1::from_planar_rectangle_v2_cartesian(
            &geometry,
            cells.cells().try_into().unwrap(),
        )
        .unwrap();
    let production = MeshProductionLineageEnvelopeV1::from_structured_cartesian_v2_resources(
        &cells,
        &geometry,
        &mesh,
        &correspondence,
    )
    .unwrap();
    let owner =
        AuthenticatedCommonMesh::structured_cartesian(geometry, mesh, correspondence, production)
            .unwrap();
    let nonlinear =
        NonlinearSolvePlan::new(1.0e-9, 1.0e-11, NonZeroUsize::new(16).unwrap(), 12).unwrap();
    let solve = CommonSolvePolicy::Newton {
        nonlinear,
        linear: eqiora_numerics::CommonLinearRequest::exact(
            eqiora_solver::SolverPlan::new(
                eqiora_solver::LinearSolver::BiConjugateGradientStabilized,
                1e-10,
                1e-12,
                NonZeroUsize::new(2_000).unwrap(),
            )
            .unwrap()
            .with_preconditioner(eqiora_solver::PreconditionerPolicy::Identity)
            .with_reduction(eqiora_solver::ReductionPolicy::Reproducible),
            eqiora_solver::REFERENCE_SOLVER_PROVIDER,
        )
        .unwrap(),
    };
    let scaling = IncompressibleScalingRequest2d::from_si(Some(1.0), Some(2.0), Some(3.0)).unwrap();
    let temporal = CommonBackwardEuler::from_seconds(0.01).unwrap();
    resolve_common_plan(
        &model,
        owner,
        CommonSpatialPolicy::CellCentered,
        solve,
        Some(scaling),
        Some(temporal),
        &REFERENCE_LINEAR_SOLVER,
        None,
    )
    .unwrap()
    .as_transient_flow()
    .cloned()
    .expect("fixture retains its admitted transient_flow Plan")
}

fn decode_wire(wire: &serde_json::Value) -> Result<RunRequest, Diagnostic> {
    RunRequest::from_bytes(
        &serde_json::to_vec(wire).unwrap(),
        &REFERENCE_LINEAR_SOLVER,
        TIME_BACKEND,
    )
}

fn mutate_embedded_json(
    outer: &mut serde_json::Value,
    key: &str,
    mutate: impl FnOnce(&mut serde_json::Value),
) {
    let encoded = outer[key].as_str().unwrap();
    let bytes = BASE64_STANDARD.decode(encoded).unwrap();
    let mut inner: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    mutate(&mut inner);
    outer[key] =
        serde_json::Value::String(BASE64_STANDARD.encode(serde_json::to_vec(&inner).unwrap()));
}
