//! Residual-native BDF through ordinary spatial Plan, State, Run and replay.
use super::*;
use eqiora::TimeStateCoordinate;
use eqiora::backends::sundials::IdaTimeBackend;
use eqiora::time::TimeMethod;
use eqiora_numerics::CommonOdePolicy;

#[test]
fn residual_native_spatial_constraints_keep_the_physical_branch_and_replay() {
    for spatial in [
        CommonSpatialPolicy::Q1,
        CommonSpatialPolicy::CellCenteredTpfa,
    ] {
        let source = r#"public component Heat(
            support body:volume(ambient_dimension=1),
            support left:boundary(parent=body),support right:boundary(parent=body)) {
            state u:1 on body in smooth;
            variable z:1 on body in smooth;
            initial {u=1;z=1;}
            relation evolution on body {(1+u*u)*(1[s]*derivative(u)-1)=0;}
            relation constraint on body {z-u*u-time()/1[s]=0;}
        }"#;
        let (document, base) = resolve_source_document(source, 0.1, newton(32)).unwrap();
        let u = document.field_ref("definition.u").unwrap().id();
        let temporal = CommonOdePolicy::new(
            TimeMethod::Bdf,
            1e-4,
            1e-9,
            base.as_spatial()
                .unwrap()
                .fields()
                .map(|(field, _)| (TimeStateCoordinate::new(field, 0, 0, false), 1e-11))
                .collect(),
        )
        .unwrap();
        let plan = ResolvedCommonPlan::resolve(
            base.model_artifact(),
            base.authenticated_mesh().unwrap(),
            spatial,
            newton(32),
            None,
            Some(temporal.into()),
            &REFERENCE_LINEAR_SOLVER,
            Some(IdaTimeBackend::CAPABILITIES),
            None,
        )
        .unwrap();
        let bytes = plan.to_bytes().unwrap();
        let plan = ResolvedCommonPlan::from_bytes(
            &bytes,
            &REFERENCE_LINEAR_SOLVER,
            &[IdaTimeBackend::CAPABILITIES],
        )
        .unwrap();
        assert_eq!(plan.to_bytes().unwrap(), bytes);
        let initial = plan
            .as_spatial()
            .unwrap()
            .initial_state(Some(0.), Vec::new())
            .unwrap();
        let initial_bytes = initial.to_bytes().unwrap();
        let run = |times| {
            CommonTransientRunRequest::from_times(plan.clone(), initial.clone(), 0.5, times)
                .unwrap()
                .run(
                    &REFERENCE_LINEAR_SOLVER,
                    |_, _| panic!("residual-native policy must consume the original residual"),
                    |problem, time| IdaTimeBackend::new().solve(problem, time),
                )
                .unwrap()
        };
        let sparse = run(vec![0.5]);
        let dense = run(vec![0.07, 0.25, 0.5]);
        assert_eq!(sparse.spatial_time_history(), dense.spatial_time_history());
        // The positive factor 1+u² never vanishes: u'=1, u(0)=1.
        // Thus u=1+t and z=(1+t)²+t on every constant spatial mode.
        for (i, time) in [0.07_f64, 0.25, 0.5].into_iter().enumerate() {
            let state = dense.spatial_state(i).unwrap();
            for (field, _, _, values) in plan.as_spatial().unwrap().state_fields(state).unwrap() {
                let expected = if field == u {
                    1. + time
                } else {
                    (1. + time).powi(2) + time
                };
                assert!(values.iter().all(|value| (value - expected).abs() < 1e-6));
            }
            assert_eq!(
                CommonState::from_bytes(&state.to_bytes().unwrap(), &plan).unwrap(),
                *state
            );
        }
        assert_eq!(initial.to_bytes().unwrap(), initial_bytes);
        let bytes = dense.to_bytes().unwrap();
        assert_eq!(CommonTrajectory::from_bytes(&bytes, &plan).unwrap(), dense);
        let original: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let reject = |forged: serde_json::Value, diagnostic: &str| {
            let error = CommonTrajectory::from_bytes(&serde_json::to_vec(&forged).unwrap(), &plan)
                .unwrap_err();
            assert!(format!("{error:?}").contains(diagnostic), "{error:?}");
        };
        let mut missing = original.clone();
        missing["payload"]["execution"]["outputs"] = serde_json::json!([]);
        reject(missing, "incomplete output recovery inventory");
        let mut predictor = original.clone();
        let residual =
            &mut predictor["payload"]["execution"]["outputs"][0]["solve"]["initial_residual_norm"];
        *residual = serde_json::json!(residual.as_f64().unwrap() + 1.);
        reject(
            predictor,
            "predictor residual differs from original equations",
        );
        let mut retired = original;
        retired["schema"] = serde_json::json!("eqiora.common-trajectory/v13");
        reject(retired, "unknown schema or encoding");
        let result = CommonResult::accept_trajectory(0., dense).unwrap();
        assert_eq!(
            CommonResult::from_bytes(&result.to_bytes().unwrap(), &plan).unwrap(),
            result
        );
    }
}

#[test]
fn residual_native_complex_constraints_follow_the_half_angle_branch() {
    for spatial in [
        CommonSpatialPolicy::Q1,
        CommonSpatialPolicy::CellCenteredTpfa,
    ] {
        let source = r#"public component Heat(
            support body:volume(ambient_dimension=1),
            support left:boundary(parent=body),support right:boundary(parent=body)) {
            state u:complex<1> on body in smooth;
            variable z:complex<1> on body in smooth;
            initial {u=math.complex(1,0);z=math.complex(1,0);}
            relation evolution on body {1[s]*derivative(u)-math.complex(0,1)*u=0;}
            relation constraint on body {z*z-u=0;}
        }"#;
        let (document, base) = resolve_source_document(source, 0.1, newton(32)).unwrap();
        let u = document.field_ref("definition.u").unwrap().id();
        let temporal = CommonOdePolicy::new(
            TimeMethod::Bdf,
            1e-4,
            1e-9,
            base.as_spatial()
                .unwrap()
                .fields()
                .flat_map(|(field, _)| {
                    [false, true].map(move |imaginary| {
                        (TimeStateCoordinate::new(field, 0, 0, imaginary), 1e-11)
                    })
                })
                .collect(),
        )
        .unwrap();
        let plan = ResolvedCommonPlan::resolve(
            base.model_artifact(),
            base.authenticated_mesh().unwrap(),
            spatial,
            newton(32),
            None,
            Some(temporal.into()),
            &REFERENCE_LINEAR_SOLVER,
            Some(IdaTimeBackend::CAPABILITIES),
            None,
        )
        .unwrap();
        let initial = plan
            .as_spatial()
            .unwrap()
            .initial_state(Some(0.), Vec::new())
            .unwrap();
        let end = std::f64::consts::TAU;
        let times = [0.7, 3.5, end];
        let run = |outputs| {
            CommonTransientRunRequest::from_times(plan.clone(), initial.clone(), end, outputs)
                .unwrap()
                .run(
                    &REFERENCE_LINEAR_SOLVER,
                    |_, _| panic!("BDF consumes the original residual"),
                    |problem, time| IdaTimeBackend::new().solve(problem, time),
                )
                .unwrap()
        };
        let sparse = run(vec![end]);
        let dense = run(times.to_vec());
        assert_eq!(sparse.spatial_time_history(), dense.spatial_time_history());
        // u'=i*u gives u=exp(i*t). Continuity from z(0)=1 selects
        // z=exp(i*t/2), including z(2*pi)=-1 rather than the principal root +1.
        for (index, time) in times.into_iter().enumerate() {
            for (field, _, _, values) in plan
                .as_spatial()
                .unwrap()
                .state_fields(dense.spatial_state(index).unwrap())
                .unwrap()
            {
                let angle = if field == u { time } else { time / 2. };
                let (pairs, remainder) = values.as_chunks::<2>();
                assert!(remainder.is_empty());
                for [real, imaginary] in pairs {
                    assert!((real - angle.cos()).abs() < 1e-6);
                    assert!((imaginary - angle.sin()).abs() < 1e-6);
                }
            }
        }
        assert_eq!(
            CommonTrajectory::from_bytes(&dense.to_bytes().unwrap(), &plan).unwrap(),
            dense
        );
    }
}

#[test]
fn residual_native_prescribed_acceleration_completes_algebraic_initial_rates() {
    let source = r#"public component Heat(
        support body:volume(ambient_dimension=1),
        support left:boundary(parent=body),support right:boundary(parent=body)) {
        state u:1 on body in smooth;
        variable z:1 on body in smooth;
        initial {u=0;z=0;}
        relation evolution on body {1[s]*derivative(u)-2*time()/1[s]=0;}
        relation constraint on body {z-1[s]*derivative(u)=0;}
        relation left_value on left {trace(u)=(time()/1[s])^2;}
        relation right_value on right {trace(u)=(time()/1[s])^2;}
        form mixed for evolution,constraint {
            test wu:1 for u zero_on left,right; test wz:1 for z;
            evolution: integrate(body,inner(wu,1[s]*derivative(u)-2*time()/1[s]))=0;
            constraint: integrate(body,inner(wz,z-1[s]*derivative(u)))=0;
        }
    }"#;
    let (document, base) = resolve_source_document(source, 0.1, newton(32)).unwrap();
    let u = document.field_ref("definition.u").unwrap().id();
    let policy = CommonOdePolicy::new(
        TimeMethod::Bdf,
        1e-4,
        1e-9,
        base.as_spatial()
            .unwrap()
            .fields()
            .map(|(field, _)| (TimeStateCoordinate::new(field, 0, 0, false), 1e-11))
            .collect(),
    )
    .unwrap();
    let plan = ResolvedCommonPlan::resolve(
        base.model_artifact(),
        base.authenticated_mesh().unwrap(),
        CommonSpatialPolicy::Q1,
        newton(32),
        None,
        Some(policy.into()),
        &REFERENCE_LINEAR_SOLVER,
        Some(IdaTimeBackend::CAPABILITIES),
        document.authored_formulation_projection().unwrap(),
    )
    .unwrap();
    let initial = plan
        .as_spatial()
        .unwrap()
        .initial_state(Some(0.), Vec::new())
        .unwrap();
    let before = initial.to_bytes().unwrap();
    let trajectory =
        CommonTransientRunRequest::from_times(plan.clone(), initial.clone(), 0.5, vec![0.07, 0.5])
            .unwrap()
            .run(
                &REFERENCE_LINEAR_SOLVER,
                |_, _| panic!("BDF consumes the original residual"),
                |problem, time| {
                    for (kind, rate) in problem
                        .variable_kinds()
                        .iter()
                        .zip(problem.initial_derivative())
                    {
                        let expected = match kind {
                            eqiora::time::DaeVariableKind::Differential => 0.,
                            eqiora::time::DaeVariableKind::Algebraic => 2.,
                        };
                        assert!(
                            (rate - expected).abs() < 1e-11,
                            "initial rate {rate} != {expected}"
                        );
                    }
                    IdaTimeBackend::new().solve(problem, time)
                },
            )
            .unwrap();
    // All spatial nodes obey u=t² and z=u'=2t. Boundary mass columns
    // lie outside the free mass range, so differentiating the constraint
    // requires the prescribed acceleration u''=2 and physical z'=2.
    for (index, time) in [0.07_f64, 0.5].into_iter().enumerate() {
        for (field, _, _, values) in plan
            .as_spatial()
            .unwrap()
            .state_fields(trajectory.spatial_state(index).unwrap())
            .unwrap()
        {
            let expected = if field == u { time * time } else { 2. * time };
            assert!(
                values.iter().all(|value| (value - expected).abs() < 1e-6),
                "{values:?} != {expected}"
            );
        }
    }
    assert_eq!(initial.to_bytes().unwrap(), before);
    assert_eq!(
        CommonTrajectory::from_bytes(&trajectory.to_bytes().unwrap(), &plan).unwrap(),
        trajectory
    );
}
