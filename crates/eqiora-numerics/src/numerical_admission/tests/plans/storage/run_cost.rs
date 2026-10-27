//! Schedule identity and proof falsifiers through the existing storage fixture owners.
use super::*;
use crate::{CommonResult, CommonTrajectory};

const SOURCE: &str = r#"
public component Heat(support body:volume(ambient_dimension=1),
 support left:boundary(parent=body),support right:boundary(parent=body)) {
 state u:1 on body in h1; initial {u=2;}
 relation balance on body {3[s/m^2]*derivative(u)-div(grad(u))=4[1/m^2];}
 relation left_value on left {trace(u)=2;}
 relation right_value on right {trace(u)=2;}
}
"#;

fn fixture(
    source: &str,
    cells: usize,
    cost: Option<eqiora_solver::SolverCostRequest>,
) -> ResolvedCommonPlan {
    try_fixture(source, cells, cost, CommonSpatialPolicy::Q1).unwrap()
}

fn try_fixture(
    source: &str,
    cells: usize,
    cost: Option<eqiora_solver::SolverCostRequest>,
    spatial: CommonSpatialPolicy,
) -> Result<ResolvedCommonPlan, Diagnostic> {
    let geometry = cartesian_interval();
    let body = geometry.entity_set("body").unwrap();
    let model = compile_model(
        "run-cost.eqi",
        source,
        &geometry,
        "Heat",
        &[
            ("body", body, None),
            (
                "left",
                geometry.entity_set("left").unwrap(),
                Some(("body", body)),
            ),
            (
                "right",
                geometry.entity_set("right").unwrap(),
                Some(("body", body)),
            ),
        ],
        &[],
    );
    let mut linear = exact_reference_linear(
        LinearSolver::BiConjugateGradientStabilized,
        1e-12,
        1e-14,
        NonZeroUsize::new(100).unwrap(),
    );
    if let Some(cost) = cost {
        linear = linear.with_cost(cost);
    }
    ResolvedCommonPlan::resolve(
        &model,
        cartesian_box_resources(&geometry, &[cells]),
        spatial,
        CommonSolvePolicy::Linear(linear),
        None,
        Some(CommonTemporalPolicy::backward_euler(0.25).unwrap()),
        &REFERENCE_LINEAR_SOLVER,
        None,
        None,
    )
}

#[test]
fn run_cost_numeric_cap_keeps_its_existing_scope_and_dense_shape_bound() {
    // One free real coordinate: 12 BiCG recurrence scalars + 1 norm partial,
    // two matrix/RHS scalars, and three host CSR indices. This is not RSS.
    let bound = 120 + 3 * size_of::<usize>();
    let cost = eqiora_solver::SolverCostRequest::default()
        .with_numeric_payload_limit(NonZeroUsize::new(bound).unwrap());
    let original = fixture(SOURCE, 2, Some(cost));
    let size = original.solver_planning_size().unwrap();
    assert_eq!(
        (size.rows(), size.stored_slots(), size.is_exact()),
        (1, 1, false)
    );
    let planned = original
        .for_run(3, vec![1, 3], &REFERENCE_LINEAR_SOLVER)
        .unwrap();
    let evidence = planned.solver_planning_costs()[0];
    assert_eq!(evidence.numeric_payload_bytes(), Some(bound));
    assert_eq!(evidence.retained_numeric_bytes(), Some(288));
    assert_eq!(
        planned
            .as_spatial()
            .unwrap()
            .admission
            .linear
            .planning_profile
            .as_ref()
            .unwrap()
            .cost_request()
            .unwrap()
            .numeric_payload_limit(),
        cost.numeric_payload_limit()
    );
    let below = cost.with_numeric_payload_limit(NonZeroUsize::new(bound - 1).unwrap());
    assert!(
        try_fixture(SOURCE, 2, Some(below), CommonSpatialPolicy::Q1)
            .unwrap_err()
            .message()
            .contains("numeric-payload-limit-exceeded")
    );
    // The adjacent cell-constant transient route remains executable, but this
    // new Q1-only dimension owner makes no payload assertion for that route.
    assert!(try_fixture(SOURCE, 2, None, CommonSpatialPolicy::CellCenteredTpfa).is_ok());
    assert!(
        try_fixture(SOURCE, 2, Some(cost), CommonSpatialPolicy::CellCenteredTpfa)
            .unwrap_err()
            .message()
            .contains("numeric-payload-bound-unproved")
    );
}

#[test]
fn run_cost_plan_is_immutable_manual_exact_and_standalone_rederived() {
    let cost = eqiora_solver::SolverCostRequest::new(
        NonZeroUsize::new(2).unwrap(),
        NonZeroUsize::new(2).unwrap(),
        NonZeroUsize::new(2).unwrap(),
    )
    .unwrap()
    .with_iteration_forecasts(Some(NonZeroUsize::new(5).unwrap()), None);
    let original = fixture(SOURCE, 2, Some(cost));
    let before = original.to_bytes().unwrap();
    let planned = original
        .for_run(3, vec![1, 3], &REFERENCE_LINEAR_SOLVER)
        .unwrap();
    assert_eq!(original.to_bytes().unwrap(), before);
    assert_ne!(original.identity(), planned.identity());
    assert_eq!(original.effective_solver(), planned.effective_solver());
    assert_eq!(
        original.linear_solver_provider(),
        planned.linear_solver_provider()
    );
    let replay =
        ResolvedCommonPlan::from_bytes(&planned.to_bytes().unwrap(), &REFERENCE_LINEAR_SOLVER, &[])
            .unwrap();
    assert_eq!(replay, planned);
    let profile = replay
        .as_spatial()
        .unwrap()
        .admission
        .linear
        .planning_profile
        .as_ref()
        .unwrap();
    let derived = profile.cost_request().unwrap();
    assert_eq!(
        (
            derived.solves().get(),
            derived.structures().get(),
            derived.numeric_preparations().get()
        ),
        (3, 3, 3)
    );
    assert_eq!(derived.identity_iterations(), cost.identity_iterations());
    assert_eq!(derived.jacobi_iterations(), cost.jacobi_iterations());
    let wire: serde_json::Value = serde_json::from_slice(&planned.to_bytes().unwrap()).unwrap();
    // Q1 interval: two cells, two local coordinates, two Gauss samples.
    // Ordinary complete Run: 2H+O=8 assemblies, local samples2*2*(4+2)=24.
    assert_eq!(wire["run_planning"]["assembly_passes"], 8);
    assert_eq!(wire["run_planning"]["local_quadrature_work"], 24);
    assert_eq!(wire["run_planning"]["common_work"], 320);
    assert_eq!(wire["run_planning"]["retained_numeric_bytes"], 288);
    // Even if manual execution keeps the same tuple, the complete original
    // forecast remains exact requested provenance for the Run Plan identity.
    let canonical = String::from_utf8(planned.to_bytes().unwrap()).unwrap();
    assert_eq!(canonical.matches("\"identity_iterations\":5").count(), 1);
    let altered_assumption =
        canonical.replace("\"identity_iterations\":5", "\"identity_iterations\":6");
    assert!(
        ResolvedCommonPlan::from_bytes(
            altered_assumption.as_bytes(),
            &REFERENCE_LINEAR_SOLVER,
            &[]
        )
        .is_err(),
        "caller forecast tampering"
    );
    for (key, value) in [
        ("steps", 3),
        ("assembly_passes", 8),
        ("common_work", 320),
        ("retained_numeric_bytes", 288),
    ] {
        let original = format!("\"{key}\":{value}");
        assert_eq!(canonical.matches(&original).count(), 1);
        let tampered = canonical.replace(&original, &format!("\"{key}\":17"));
        assert!(
            ResolvedCommonPlan::from_bytes(tampered.as_bytes(), &REFERENCE_LINEAR_SOLVER, &[])
                .is_err(),
            "{key}"
        );
    }
}

#[test]
fn run_cost_schedule_rejects_before_work_and_values_retain_the_existing_recurrence() {
    let planned = fixture(SOURCE, 2, None)
        .for_run(3, vec![1, 3], &REFERENCE_LINEAR_SOLVER)
        .unwrap();
    let state = planned
        .as_spatial()
        .unwrap()
        .initial_state(Some(0.), Vec::new())
        .unwrap();
    let bytes = state.to_bytes().unwrap();
    for (steps, outputs) in [(4, vec![1, 3]), (3, vec![3]), (3, vec![3, 1]), (0, vec![1])] {
        assert!(
            CommonTransientRunRequest::from_steps(planned.clone(), state.clone(), steps, outputs)
                .is_err()
        );
    }
    assert_eq!(state.to_bytes().unwrap(), bytes);
    let request =
        CommonTransientRunRequest::from_steps(planned.clone(), state, 3, vec![1, 3]).unwrap();
    let std::ops::ControlFlow::Continue(outputs) = request
        .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
        .unwrap()
    else {
        panic!("accepted Run")
    };
    // Existing storage.rs recurrence: e_next=1/2*e+1/4 for c=3,dt=1/4.
    for ((step, accepted), expected) in outputs.iter().zip([2.25, 2.4375]) {
        assert!([1, 3].contains(step));
        assert!((accepted.state().spatial_values().unwrap()[1] - expected).abs() <= 1e-12);
        assert!(accepted.state().spatial_retained_values_bytes().unwrap() <= 48);
    }
    let retained = request.state().spatial_retained_values_bytes().unwrap()
        + outputs
            .iter()
            .map(|(_, accepted)| {
                accepted
                    .previous_state()
                    .spatial_retained_values_bytes()
                    .unwrap()
                    + accepted.state().spatial_retained_values_bytes().unwrap()
            })
            .sum::<usize>();
    // Retained request/output coordinates and a live current State fit the
    // independently derived envelope; indices/factors/scratch/RSS are excluded.
    assert!(
        retained + 48
            <= planned.solver_planning_costs()[0]
                .retained_numeric_bytes()
                .unwrap()
    );
    let trajectory = CommonTrajectory::accept_spatial_transient(request, outputs).unwrap();
    let result = CommonResult::accept_trajectory(0., trajectory).unwrap();
    assert_eq!(
        CommonResult::from_bytes(&result.to_bytes().unwrap(), &planned)
            .unwrap()
            .to_bytes()
            .unwrap(),
        result.to_bytes().unwrap()
    );
}

#[test]
fn run_cost_unknown_natural_facet_work_remains_unknown() {
    let natural = SOURCE
        .replace("in h1", "in smooth")
        .replace("state u:1", "variable g:1 on body in smooth; relation profile on body {g=2+coordinate(0)/1[m];} state u:1")
        .replace("initial {u=2;}", "initial {u=g;}")
        .replace("relation left_value on left {trace(u)=2;}", "relation left_value on left {normal(grad(u))=-1[1/m];}")
        .replace("relation right_value on right {trace(u)=2;}", "relation right_value on right {normal(grad(u))=1[1/m];}")
        .replace("4[1/m^2]", "0[1/m^2]");
    let planned = fixture(&natural, 2, None)
        .for_run(2, vec![2], &REFERENCE_LINEAR_SOLVER)
        .unwrap();
    let cost = planned.solver_planning_costs()[0];
    assert_eq!(cost.run_work(), None);
    assert_eq!(cost.total_work(), None);
    let initial = planned
        .as_spatial()
        .unwrap()
        .initial_state(Some(0.), Vec::new())
        .unwrap();
    let request = CommonTransientRunRequest::from_steps(planned, initial, 2, vec![2]).unwrap();
    let std::ops::ControlFlow::Continue(outputs) = request
        .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
        .unwrap()
    else {
        panic!("accepted Neumann storage")
    };
    // The affine field 2+x is an exact equilibrium with these outward traces.
    for (value, expected) in outputs[0]
        .1
        .state()
        .spatial_values()
        .unwrap()
        .iter()
        .zip([2., 2.5, 3.])
    {
        assert!((*value - expected).abs() <= 1e-12);
    }
}

#[test]
fn run_cost_source_time_inventory_rejects_hidden_capacity_boundary_and_rhs() {
    for source in [
        SOURCE.replace("3[s/m^2]*derivative(u)", "(3+time()/1[s])*1[s/m^2]*derivative(u)"),
        SOURCE.replace("trace(u)=2", "trace(u)=2+time()/1[s]"),
        SOURCE.replace("4[1/m^2]", "(4+time()/1[s])*1[1/m^2]"),
        SOURCE.replace("state u:1", "variable c:s/m^2 on body in smooth; relation capacity on body {c=(3+time()/1[s])*1[s/m^2];} state u:1").replace("3[s/m^2]*derivative(u)", "c*derivative(u)"),
    ] {
        let plan = fixture(&source, 2, None);
        assert!(plan.for_run(3, vec![1, 3], &REFERENCE_LINEAR_SOLVER).unwrap_err().message().contains("autonomous"));
    }
    let unrelated = SOURCE.replace("state u:1", "observable clock:s=time(); state u:1");
    assert!(
        fixture(&unrelated, 2, None)
            .for_run(3, vec![1, 3], &REFERENCE_LINEAR_SOLVER)
            .is_ok()
    );
}

#[test]
fn run_cost_coordinate_coefficients_and_changed_mesh_rederive_identity() {
    let coordinate = SOURCE.replace("state u:1", "variable log_k:1 on body in smooth; relation coefficient on body {log_k=coordinate(0)/1[m];} state u:1")
        .replace("-div(grad(u))", "-div(math.exp(log_k)*grad(u))").replace("4[1/m^2]", "0[1/m^2]");
    let a = fixture(&coordinate, 2, None)
        .for_run(2, vec![2], &REFERENCE_LINEAR_SOLVER)
        .unwrap();
    let b = fixture(&coordinate, 3, None)
        .for_run(2, vec![2], &REFERENCE_LINEAR_SOLVER)
        .unwrap();
    assert_ne!(a.identity(), b.identity());
    assert_ne!(a.solver_planning_size(), b.solver_planning_size());
    let initial = a
        .as_spatial()
        .unwrap()
        .initial_state(Some(0.), Vec::new())
        .unwrap();
    assert!(CommonTransientRunRequest::from_steps(b, initial.clone(), 2, vec![2]).is_err());
    let request = CommonTransientRunRequest::from_steps(a, initial, 2, vec![2]).unwrap();
    let std::ops::ControlFlow::Continue(outputs) = request
        .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
        .unwrap()
    else {
        panic!("accepted equilibrium")
    };
    // Constant2 is an exact equilibrium for every positive coordinate diffusion.
    for value in outputs[0].1.state().spatial_values().unwrap() {
        assert!((*value - 2.).abs() <= 1e-12);
    }
}
