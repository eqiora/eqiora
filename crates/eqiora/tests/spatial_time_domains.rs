//! Typed spatial history uses the same ordinary temporal lifecycle.
use eqiora::api::ModelDocument;
use eqiora::artifact::{
    CartesianMeshCellsV2, GeometryMeshCorrespondenceEnvelopeV1, MeshProductionLineageEnvelopeV1,
    ModelEnvelope,
};
use eqiora::compiler::StaticBindingValue;
use eqiora::geometry::GeometryGraph;
use eqiora::solver::{LinearSolver, REFERENCE_LINEAR_SOLVER, ReductionPolicy, SolverPlan};
use eqiora_numerics::{
    AuthenticatedCommonMesh, CommonBackwardEuler, CommonLinearRequest, CommonResult,
    CommonSolvePolicy, CommonSpatialPolicy, CommonState, CommonTrajectory,
    CommonTransientRunRequest, ResolvedCommonPlan,
};
use std::{collections::BTreeMap, num::NonZeroUsize};

fn probe(complex: bool, law: bool) -> Result<Vec<f64>, String> {
    let kind = if complex { "complex<1>" } else { "1" };
    let datum = if complex { "math.complex(2,4)" } else { "2" };
    let source = if complex {
        "math.complex(4,8)/1[m^2]"
    } else {
        "4[1/m^2]"
    };
    let balance = if law {
        format!("law balance on body {{ storage 3[s/m^2]*u; flux -grad(u); source {source}; }}")
    } else {
        format!("relation balance on body {{ 3[s/m^2]*derivative(u)-div(grad(u))={source}; }}")
    };
    let source = format!(
        r#"public component Heat(
      support body:volume(ambient_dimension=1),
      support left:boundary(parent=body),support right:boundary(parent=body)){{
      state u:{kind} on body in h1;
      initial {{u={datum};}}
      {balance}
      relation left_value on left {{trace(u)={datum};}}
      relation right_value on right {{trace(u)={datum};}}
    }}"#
    );
    let plan = resolve_source(&source, 0.25, CommonSolvePolicy::Linear(linear()))?;
    let plan = replay_plan(plan);
    let spatial = plan.as_spatial().unwrap().clone();
    let initial = spatial
        .initial_state(Some(0.), Vec::new())
        .map_err(|e| format!("initial State: {e:?}"))?;
    let before = initial.to_bytes().unwrap();
    let initial = CommonState::from_bytes(&before, &plan).unwrap();
    let run =
        CommonTransientRunRequest::from_steps(plan.clone(), initial.clone(), 3, vec![1, 2, 3])
            .unwrap();
    let std::ops::ControlFlow::Continue(outputs) = run
        .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
        .map_err(|e| format!("Run: {e:?}"))?
    else {
        panic!("unexpected interruption")
    };
    let mut values = Vec::new();
    // Two Q1 elements with fixed endpoint datum: e_next=e/2+1/4.
    for ((_, step), expected) in outputs.iter().zip([2.25, 2.375, 2.4375]) {
        let fields = spatial
            .state_fields(step.state())
            .unwrap()
            .collect::<Vec<_>>();
        let coordinates = fields[0].3;
        let stride = if complex { 2 } else { 1 };
        assert!((coordinates[stride] - expected).abs() < 1e-12);
        if complex {
            assert!((coordinates[stride + 1] - 2. * expected).abs() < 1e-12);
        }
        values.extend_from_slice(coordinates);
    }
    assert_eq!(initial.to_bytes().unwrap(), before);
    let std::ops::ControlFlow::Break((1, restart)) = run
        .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |step, _| step == 1)
        .unwrap()
    else {
        panic!("stop after first accepted step");
    };
    let restart = CommonState::from_bytes(&restart.to_bytes().unwrap(), &plan).unwrap();
    let resumed = CommonTransientRunRequest::from_steps(plan.clone(), restart, 2, vec![2]).unwrap();
    let std::ops::ControlFlow::Continue(resumed) = resumed
        .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
        .unwrap()
    else {
        panic!("resume");
    };
    assert_eq!(resumed[0].1.state(), outputs.last().unwrap().1.state());
    let result = CommonResult::accept_trajectory(
        0.,
        CommonTrajectory::accept_spatial_transient(run, outputs).unwrap(),
    )
    .unwrap();
    assert_eq!(
        CommonResult::from_bytes(&result.to_bytes().unwrap(), &plan).unwrap(),
        result
    );
    Ok(values)
}
#[test]
fn real_law_time_execution() {
    probe(false, true).unwrap();
}
#[test]
fn complex_law_time_execution() {
    probe(true, true).unwrap();
}
#[test]
fn real_relation_time_execution() {
    probe(false, false).unwrap();
}
#[test]
fn complex_relation_time_execution() {
    probe(true, false).unwrap();
}

fn resolve_source(
    source: &str,
    step: f64,
    solve: CommonSolvePolicy,
) -> Result<ResolvedCommonPlan, String> {
    let graph = GeometryGraph::new();
    let interval = graph.interval([0., 1.]).unwrap();
    let sides = interval.boundaries();
    let geometry = graph
        .build(
            &interval,
            &BTreeMap::from([
                ("body".into(), vec![interval.region().into()]),
                ("left".into(), vec![sides[0].into()]),
                ("right".into(), vec![sides[1].into()]),
            ]),
        )
        .unwrap();
    let body = geometry.entity_set("body").unwrap();
    let document = ModelDocument::compile_selected(
        "complex-time-probe.eqi",
        source,
        "Heat",
        &[
            (
                "body",
                StaticBindingValue::GeometrySupport {
                    geometry: &geometry,
                    selection: body,
                    parent: None,
                },
            ),
            (
                "left",
                StaticBindingValue::GeometrySupport {
                    geometry: &geometry,
                    selection: geometry.entity_set("left").unwrap(),
                    parent: Some(body),
                },
            ),
            (
                "right",
                StaticBindingValue::GeometrySupport {
                    geometry: &geometry,
                    selection: geometry.entity_set("right").unwrap(),
                    parent: Some(body),
                },
            ),
        ],
    )
    .map_err(|e| format!("compile: {e:?}"))?;
    let cells = CartesianMeshCellsV2::new(vec![2]).unwrap();
    let (mesh, correspondence) =
        GeometryMeshCorrespondenceEnvelopeV1::from_cartesian_box_v1(&geometry, cells.cells())
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
    ResolvedCommonPlan::resolve(
        &ModelEnvelope::from_program(document.program()).unwrap(),
        owner,
        CommonSpatialPolicy::Q1,
        solve,
        None,
        Some(CommonBackwardEuler::from_seconds(step).unwrap()),
        &REFERENCE_LINEAR_SOLVER,
        None,
    )
    .map_err(|e| format!("resolve: {e:?}"))
}
fn linear() -> CommonLinearRequest {
    let solver = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1e-12,
        1e-14,
        NonZeroUsize::new(100).unwrap(),
    )
    .unwrap()
    .with_reduction(ReductionPolicy::Reproducible);
    CommonLinearRequest::exact(solver, REFERENCE_LINEAR_SOLVER.provider()).unwrap()
}
fn replay_plan(plan: ResolvedCommonPlan) -> ResolvedCommonPlan {
    let restored = ResolvedCommonPlan::from_bytes(
        &plan.to_bytes().unwrap(),
        &REFERENCE_LINEAR_SOLVER,
        eqiora::time::TimeBackendCapabilities::new(
            eqiora::time::TimeBackendIdentity::new("eqiora.test.time", "1"),
            &[eqiora::ScalarDomain::Real, eqiora::ScalarDomain::Complex],
            &[eqiora::ScalarType::F64],
        ),
    )
    .unwrap();
    assert_eq!(restored, plan);
    restored
}
const NONLINEAR: &str = r#"public component Heat(
 support body:volume(ambient_dimension=1),support left:boundary(parent=body),support right:boundary(parent=body)) {
 state u:complex<1> on body in smooth;
 initial {u=math.complex(1,2);}
 relation balance on body { 1[s/m^2]*derivative(u)-div(grad(u))+0.1[1/m^2]*math.abs2(u)*u=0; }
 relation left_flux on left { normal(grad(u))=0; }
 relation right_flux on right { normal(grad(u))=0; }
}"#;
fn newton(maximum: usize) -> CommonSolvePolicy {
    CommonSolvePolicy::Newton {
        linear: linear(),
        nonlinear: eqiora::realization::NonlinearSolvePlan::new(
            0.,
            1e-12,
            NonZeroUsize::new(maximum).unwrap(),
            16,
        )
        .unwrap(),
    }
}
#[test]
fn nonholomorphic_newton_time_history_refines_and_replays() {
    for law in [false, true] {
        let source = if law {
            NONLINEAR.replace(
                "relation balance on body { 1[s/m^2]*derivative(u)-div(grad(u))+0.1[1/m^2]*math.abs2(u)*u=0; }",
                "law balance on body { storage 1[s/m^2]*u; flux -grad(u); source -0.1[1/m^2]*math.abs2(u)*u; }",
            )
        } else {
            NONLINEAR.to_owned()
        };
        let mut errors = Vec::new();
        for steps in [4, 8, 16] {
            let dt = 1. / f64::from(steps);
            let plan = replay_plan(resolve_source(&source, dt, newton(20)).unwrap());
            let spatial = plan.as_spatial().unwrap();
            let initial = spatial.initial_state(Some(0.), Vec::new()).unwrap();
            let request = CommonTransientRunRequest::from_steps(
                plan.clone(),
                initial,
                steps as usize,
                (1..=steps as usize).collect(),
            )
            .unwrap();
            let std::ops::ControlFlow::Continue(outputs) = request
                .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
                .unwrap()
            else {
                panic!("complete nonlinear history");
            };
            let mut amplitude = 1.;
            for (_, accepted) in &outputs {
                // Uniform u=(1+2i)*a obeys a'=-(1/2)*a³.
                // Backward Euler is the unique positive root x+dt*x³/2=a_previous.
                let (mut lower, mut upper) = (0., amplitude);
                for _ in 0..80 {
                    let middle = (lower + upper) / 2.;
                    if middle + dt * middle * middle * middle / 2. > amplitude {
                        upper = middle;
                    } else {
                        lower = middle;
                    }
                }
                amplitude = (lower + upper) / 2.;
                for (_, _, _, values) in spatial.state_fields(accepted.state()).unwrap() {
                    let (pairs, remainder) = values.as_chunks::<2>();
                    assert!(remainder.is_empty());
                    for [real, imaginary] in pairs {
                        assert!((*real - amplitude).abs() < 1e-10);
                        assert!((*imaginary - 2. * amplitude).abs() < 1e-10);
                    }
                }
            }
            // The continuous solution at t=1 is a=1/sqrt(2); backward Euler is first order.
            let last = spatial
                .state_fields(outputs.last().unwrap().1.state())
                .unwrap()
                .next()
                .unwrap()
                .3[0];
            errors.push((last - 1. / 2_f64.sqrt()).abs());
            let result = CommonResult::accept_trajectory(
                0.,
                CommonTrajectory::accept_spatial_transient(request, outputs).unwrap(),
            )
            .unwrap();
            assert_eq!(
                CommonResult::from_bytes(&result.to_bytes().unwrap(), &plan).unwrap(),
                result
            );
        }
        for pair in errors.windows(2) {
            assert!((1.7..2.2).contains(&(pair[0] / pair[1])));
        }
    }
}
#[test]
fn failed_nonlinear_step_keeps_accepted_state_unchanged() {
    let plan = resolve_source(NONLINEAR, 0.25, newton(1)).unwrap();
    let initial = plan
        .as_spatial()
        .unwrap()
        .initial_state(Some(0.), Vec::new())
        .unwrap();
    let before = initial.to_bytes().unwrap();
    let request = CommonTransientRunRequest::from_steps(plan, initial.clone(), 1, vec![1]).unwrap();
    assert!(
        request
            .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
            .is_err()
    );
    assert_eq!(initial.to_bytes().unwrap(), before);
}

fn nonlinear_storage_history(law: bool) {
    let balance = if law {
        "law balance on body { storage 1[s/m^2]*u*u; flux -grad(u); source -u*u/1[m^2]; }"
    } else {
        "relation balance on body { 1[s/m^2]*derivative(u*u)-div(grad(u))+u*u/1[m^2]=0; }"
    };
    let source = format!(
        "public component Heat(
        support body:volume(ambient_dimension=1),
        support left:boundary(parent=body),support right:boundary(parent=body)) {{
        state u:1 on body in smooth;
        initial {{u=1;}}
        {balance}
        relation left_flux on left {{normal(grad(u))=0;}}
        relation right_flux on right {{normal(grad(u))=0;}}
        }}"
    );
    let mut errors = Vec::new();
    for steps in [4, 8, 16] {
        let dt = 1. / f64::from(steps);
        let plan = replay_plan(resolve_source(&source, dt, newton(20)).unwrap());
        let initial = plan
            .as_spatial()
            .unwrap()
            .initial_state(Some(0.), Vec::new())
            .unwrap();
        let before = initial.to_bytes().unwrap();
        let request = CommonTransientRunRequest::from_steps(
            plan.clone(),
            initial.clone(),
            steps as usize,
            (1..=steps as usize).collect(),
        )
        .unwrap();
        let std::ops::ControlFlow::Continue(outputs) = request
            .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
            .unwrap()
        else {
            panic!("complete nonlinear-storage history");
        };
        // d(u²)/dt=-u² gives 2u*u_dot=-u² and u(t)=exp(-t/2) on u>0.
        // Existing State-coordinate Backward Euler sets u_dot=(u-u_old)/dt:
        // its nonzero root is u=2*u_old/(2+dt), not u_old/sqrt(1+dt).
        let ratio = 2. / (2. + dt);
        for (step, (_, accepted)) in (1..=steps).zip(&outputs) {
            let expected = ratio.powi(step);
            for (_, _, _, values) in plan
                .as_spatial()
                .unwrap()
                .state_fields(accepted.state())
                .unwrap()
            {
                for value in values {
                    assert!((*value - expected).abs() < 1e-10);
                }
            }
        }
        let last = plan
            .as_spatial()
            .unwrap()
            .state_fields(outputs.last().unwrap().1.state())
            .unwrap()
            .next()
            .unwrap()
            .3[0];
        errors.push((last - (-0.5_f64).exp()).abs());
        let restart =
            CommonState::from_bytes(&outputs[0].1.state().to_bytes().unwrap(), &plan).unwrap();
        let resumed = CommonTransientRunRequest::from_steps(
            plan.clone(),
            restart,
            steps as usize - 1,
            vec![steps as usize - 1],
        )
        .unwrap();
        let std::ops::ControlFlow::Continue(resumed) = resumed
            .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
            .unwrap()
        else {
            panic!("complete nonlinear-storage restart");
        };
        assert_eq!(
            resumed.last().unwrap().1.state(),
            outputs.last().unwrap().1.state()
        );
        let result = CommonResult::accept_trajectory(
            0.,
            CommonTrajectory::accept_spatial_transient(request, outputs).unwrap(),
        )
        .unwrap();
        assert_eq!(
            CommonResult::from_bytes(&result.to_bytes().unwrap(), &plan).unwrap(),
            result
        );
        assert_eq!(initial.to_bytes().unwrap(), before);
    }
    for pair in errors.windows(2) {
        assert!((1.7..2.2).contains(&(pair[0] / pair[1])));
    }
}

#[test]
fn nonlinear_law_storage_uses_state_coordinate_backward_euler() {
    nonlinear_storage_history(true);
}

#[test]
fn nonlinear_relation_storage_uses_state_coordinate_backward_euler() {
    nonlinear_storage_history(false);
}
