//! The ordinary native Run owns physical outputs independently of its stage clock.
use super::*;
use eqiora_time::TimeMethod;

#[test]
fn spatial_midpoint_runs_preserve_sampling_solver_intent_and_restart() {
    let h = 1. / 16.;
    let until = 4. * h;
    for complex in [false, true] {
        for spatial in [
            CommonSpatialPolicy::Q1,
            CommonSpatialPolicy::CellCenteredTpfa,
        ] {
            for newton in [false, true] {
                let base = resolve(
                    complex,
                    spatial,
                    h,
                    "1[s/m^2]*derivative(u)-div(grad(u))=0;",
                    Some("0"),
                );
                let requested = policy::policy(&base, TimeMethod::ImplicitMidpoint, h);
                let solver = match base.canonical_solve_request().unwrap() {
                    CommonSolvePolicy::Newton { linear, .. } if !newton => {
                        CommonSolvePolicy::Linear(linear)
                    }
                    policy => policy,
                };
                let resolved = ResolvedCommonPlan::resolve(
                    base.model_artifact(),
                    base.authenticated_mesh().unwrap(),
                    spatial,
                    solver,
                    None,
                    Some(requested.into()),
                    &REFERENCE_LINEAR_SOLVER,
                    Some(eqiora_time::ImplicitMidpointTimeBackend::CAPABILITIES),
                    None,
                )
                .unwrap();
                let plan = resolved.as_spatial().unwrap();
                let (field, ty) = plan.fields().next().unwrap();
                let one = C::new(1., if complex { 2. } else { 0. });
                let q1 = spatial == CommonSpatialPolicy::Q1;
                let values = CommonInitialValues::new(
                    ty.shape().clone(),
                    if q1 {
                        vec![C::new(0., 0.), one, C::new(0., 0.)]
                    } else {
                        vec![one; 2]
                    },
                )
                .unwrap();
                let initial = plan
                    .initial_state(
                        Some(0.),
                        vec![
                            CommonInitialField::new(
                                resolved.model_artifact().digest().unwrap(),
                                field,
                                q1.then(|| values.clone()),
                                (!q1).then_some(values),
                            )
                            .unwrap(),
                        ],
                    )
                    .unwrap();
                let before = initial.to_bytes().unwrap();
                let sparse = CommonTransientRunRequest::from_times(
                    resolved.clone(),
                    initial.clone(),
                    until,
                    vec![until],
                )
                .unwrap()
                .run(&REFERENCE_LINEAR_SOLVER, |_, _| {
                    panic!("midpoint must use the production implicit stage solver")
                })
                .unwrap();
                let outputs = vec![h / 2., 2. * h, until];
                let request = CommonTransientRunRequest::from_times(
                    resolved.clone(),
                    initial.clone(),
                    until,
                    outputs.clone(),
                )
                .unwrap();
                assert!(request.output_steps().is_err());
                assert!(request.accepted_steps().is_err());
                let dense = request
                    .run(&REFERENCE_LINEAR_SOLVER, |_, _| {
                        panic!("midpoint must use the production implicit stage solver")
                    })
                    .unwrap();
                assert_eq!(sparse.spatial_time_history(), dense.spatial_time_history());
                assert_eq!(dense.spatial_time_history().unwrap().steps().len(), 4);
                let interior_only = CommonTransientRunRequest::from_times(
                    resolved.clone(),
                    initial.clone(),
                    until,
                    vec![h / 2.],
                )
                .unwrap()
                .run(&REFERENCE_LINEAR_SOLVER, |_, _| {
                    panic!("midpoint must use the production implicit stage solver")
                })
                .unwrap();
                assert_eq!(
                    interior_only.spatial_time_history(),
                    dense.spatial_time_history()
                );
                assert_eq!(interior_only.spatial_state(0), dense.spatial_state(0));
                assert!(interior_only.spatial_state(1).is_none());
                let factor: f64 = if q1 { 5. / 11. } else { 3. / 5. };
                for (index, amplitude) in [
                    (0, (1. + factor) / 2.),
                    (1, factor.powi(2)),
                    (2, factor.powi(4)),
                ] {
                    let state = dense.spatial_state(index).unwrap();
                    assert_eq!(state.time_s(), Some(outputs[index]));
                    let coefficients = plan.state_fields(state).unwrap().next().unwrap().3;
                    let expected = if q1 {
                        vec![C::new(0., 0.), one * amplitude, C::new(0., 0.)]
                    } else {
                        vec![one * amplitude; 2]
                    };
                    close(coefficients, &coordinates(&expected, complex));
                    let bytes = state.to_bytes().unwrap();
                    assert_eq!(
                        CommonState::from_bytes(&bytes, &resolved)
                            .unwrap()
                            .to_bytes()
                            .unwrap(),
                        bytes
                    );
                }
                for stage in 0..4 {
                    let solves = dense.spatial_stage_solves(stage).unwrap();
                    assert!(!solves.is_empty());
                    for solve in solves {
                        assert_eq!(
                            solve.solver().id(),
                            plan.admission.linear.provider.id().as_str()
                        );
                    }
                }
                let resumed = CommonTransientRunRequest::from_times(
                    resolved.clone(),
                    dense.spatial_state(1).unwrap().clone(),
                    until,
                    vec![until],
                )
                .unwrap()
                .run(&REFERENCE_LINEAR_SOLVER, |_, _| {
                    panic!("midpoint must use the production implicit stage solver")
                })
                .unwrap();
                close(
                    resumed.spatial_state(0).unwrap().spatial_values().unwrap(),
                    dense.spatial_state(2).unwrap().spatial_values().unwrap(),
                );
                assert_eq!(initial.to_bytes().unwrap(), before);
                let bytes = dense.to_bytes().unwrap();
                assert_eq!(
                    crate::CommonTrajectory::from_bytes(&bytes, &resolved)
                        .unwrap()
                        .to_bytes()
                        .unwrap(),
                    bytes
                );
                if newton && !complex && q1 {
                    reject_false_correction(&bytes, &resolved);
                }
                let result = crate::CommonResult::accept_trajectory(0., dense).unwrap();
                let bytes = result.to_bytes().unwrap();
                assert_eq!(
                    crate::CommonResult::from_bytes(&bytes, &resolved)
                        .unwrap()
                        .to_bytes()
                        .unwrap(),
                    bytes
                );
                assert_eq!(
                    result
                        .trajectory()
                        .unwrap()
                        .spatial_state(2)
                        .unwrap()
                        .time_s(),
                    Some(until)
                );
            }
        }
    }
}

fn reject_false_correction(bytes: &[u8], plan: &ResolvedCommonPlan) {
    let mut false_correction: serde_json::Value = serde_json::from_slice(bytes).unwrap();
    false_correction["payload"]["execution"]["stages"][0]["convergence"] =
        serde_json::json!({"kind":"correction","weighted_norm":0.,"point":[0.],"direction":[1.]});
    let error =
        crate::CommonTrajectory::from_bytes(&serde_json::to_vec(&false_correction).unwrap(), plan)
            .unwrap_err();
    assert!(
        error.message().contains("small full Newton correction"),
        "{error}"
    );
    // A zero correction at a wrong endpoint is small but does not solve the
    // original equation. Keep clocks and the linear stencil valid so rejection
    // must reach the independently reassembled stage equation.
    let mut false_equation: serde_json::Value = serde_json::from_slice(bytes).unwrap();
    let old = false_equation["payload"]["history"][3]["start_state"][0]
        .as_f64()
        .unwrap();
    let wrong = false_equation["payload"]["history"][3]["end_state"][0]
        .as_f64()
        .unwrap()
        + 0.01;
    false_equation["payload"]["history"][3]["end_state"][0] = serde_json::json!(wrong);
    false_equation["payload"]["history"][3]["dense_output"]["states"][1][0] =
        serde_json::json!(wrong);
    false_equation["payload"]["history"][3]["midpoint_state"][0] =
        serde_json::json!(0.5 * old + 0.5 * wrong);
    false_equation["payload"]["execution"]["stages"][3]["convergence"] = serde_json::json!({"kind":"correction","weighted_norm":0.,"point":[wrong],"direction":[0.]});
    let error =
        crate::CommonTrajectory::from_bytes(&serde_json::to_vec(&false_equation).unwrap(), plan)
            .unwrap_err();
    assert!(
        error
            .message()
            .contains("accepted linear solution residual"),
        "{error}"
    );
}

#[test]
fn nonlinear_storage_runs_replay_the_original_state_dependent_stage() {
    for complex in [false, true] {
        for spatial in [
            CommonSpatialPolicy::Q1,
            CommonSpatialPolicy::CellCenteredTpfa,
        ] {
            let base = resolve(complex, spatial, 0.25, "derivative(u*u)=-u*u/1[s];", None);
            let plan = policy::with_policy(
                &base,
                policy::policy(&base, TimeMethod::ImplicitMidpoint, 0.25),
            )
            .unwrap();
            let spatial_plan = plan.as_spatial().unwrap();
            let (field, ty) = spatial_plan.fields().next().unwrap();
            let q1 = spatial == CommonSpatialPolicy::Q1;
            let initial_values =
                vec![C::new(1., if complex { 2. } else { 0. }); if q1 { 3 } else { 2 }];
            let values =
                CommonInitialValues::new(ty.shape().clone(), initial_values.clone()).unwrap();
            let initial = spatial_plan
                .initial_state(
                    Some(0.),
                    vec![
                        CommonInitialField::new(
                            plan.model_artifact().digest().unwrap(),
                            field,
                            q1.then(|| values.clone()),
                            (!q1).then_some(values),
                        )
                        .unwrap(),
                    ],
                )
                .unwrap();
            let before = initial.to_bytes().unwrap();
            let trajectory = CommonTransientRunRequest::from_times(
                plan.clone(),
                initial.clone(),
                1.,
                vec![0.25, 0.5, 0.75, 1.],
            )
            .unwrap()
            .run(&REFERENCE_LINEAR_SOLVER, |_, _| {
                panic!("midpoint must use the production implicit stage solver")
            })
            .unwrap();
            // 2*u*u'=-u² on the nonzero branch gives u'=-u/2.
            // At h=1/4 the midpoint factor is (1-1/16)/(1+1/16)=15/17.
            for index in 0..4 {
                let expected = initial_values
                    .iter()
                    .map(|value| value * (15_f64 / 17.).powi(index as i32 + 1))
                    .collect::<Vec<_>>();
                close(
                    trajectory
                        .spatial_state(index)
                        .unwrap()
                        .spatial_values()
                        .unwrap(),
                    &coordinates(&expected, complex),
                );
                assert!(trajectory.spatial_stage_solves(index).unwrap().len() > 1);
            }
            let result = crate::CommonResult::accept_trajectory(0., trajectory).unwrap();
            let bytes = result.to_bytes().unwrap();
            assert_eq!(
                crate::CommonResult::from_bytes(&bytes, &plan)
                    .unwrap()
                    .to_bytes()
                    .unwrap(),
                bytes
            );
            assert_eq!(initial.to_bytes().unwrap(), before);
        }
    }
}
