//! Native adaptive execution consumes the same spatial residuals and accepted history.
use super::*;
use eqiora::TimeStateCoordinate;
use eqiora::backends::diffsol::DiffsolTimeBackend;
use eqiora::time::{TimeMethod, TimeSolution};
use eqiora_numerics::{CommonInitialField, CommonInitialValues, CommonOdePolicy};
use num_complex::Complex64 as C;

fn fixture(
    complex: bool,
    spatial: CommonSpatialPolicy,
    balance: &str,
    boundary: bool,
) -> (ResolvedCommonPlan, CommonState) {
    let kind = if complex { "complex<1>" } else { "1" };
    let traces = if boundary {
        "relation l on left {trace(u)=0;} relation r on right {trace(u)=0;}"
    } else {
        ""
    };
    let source = format!(
        "public component Heat(support body:volume(ambient_dimension=1), support left:boundary(parent=body), support right:boundary(parent=body)){{ state u:{kind} on body in smooth; relation balance on body {{{balance}}} {traces} }}"
    );
    let solve = if boundary {
        CommonSolvePolicy::Linear(linear())
    } else {
        newton(64)
    };
    let base = resolve_source(&source, 1. / 16., solve).unwrap();
    let (field, ty) = base.as_spatial().unwrap().fields().next().unwrap();
    let temporal = CommonOdePolicy::new(
        TimeMethod::Tsitouras45,
        1. / 16.,
        1e-10,
        [false, true]
            .into_iter()
            .take(if complex { 2 } else { 1 })
            .map(|imaginary| (TimeStateCoordinate::new(field, 0, 0, imaginary), 1e-12))
            .collect(),
    )
    .unwrap();
    let plan = ResolvedCommonPlan::resolve(
        base.model_artifact(),
        base.authenticated_mesh().unwrap(),
        spatial,
        solve,
        None,
        Some(temporal.into()),
        &REFERENCE_LINEAR_SOLVER,
        Some(DiffsolTimeBackend::CAPABILITIES),
        None,
    )
    .unwrap();
    let plan = ResolvedCommonPlan::from_bytes(
        &plan.to_bytes().unwrap(),
        &REFERENCE_LINEAR_SOLVER,
        &[DiffsolTimeBackend::CAPABILITIES],
    )
    .unwrap();
    let q1 = spatial == CommonSpatialPolicy::Q1;
    let one = C::new(1., if complex { 2. } else { 0. });
    let mut values = vec![one; if q1 { 3 } else { 2 }];
    if boundary && q1 {
        values[0] = C::new(0., 0.);
        values[2] = C::new(0., 0.);
    }
    let values = CommonInitialValues::new(ty.shape().clone(), values).unwrap();
    let initial = plan
        .as_spatial()
        .unwrap()
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
    (plan, initial)
}

fn request(
    plan: &ResolvedCommonPlan,
    initial: &CommonState,
    times: Vec<f64>,
) -> CommonTransientRunRequest {
    CommonTransientRunRequest::from_times(plan.clone(), initial.clone(), 0.25, times).unwrap()
}

#[test]
fn spatial_adaptive_heat_retains_exact_sampling_history_and_replays_outputs() {
    for complex in [false, true] {
        for spatial in [
            CommonSpatialPolicy::Q1,
            CommonSpatialPolicy::CellCenteredTpfa,
        ] {
            let (plan, initial) = fixture(
                complex,
                spatial,
                "1[s/m^2]*derivative(u)-div(grad(u))=0;",
                true,
            );
            let before = initial.to_bytes().unwrap();
            let run = |times| {
                request(&plan, &initial, times)
                    .run(
                        &REFERENCE_LINEAR_SOLVER,
                        |problem, time| DiffsolTimeBackend::new().solve(problem, time),
                        |_, _| panic!("unexpected residual-native backend"),
                    )
                    .unwrap()
            };
            let sparse = run(vec![0.25]);
            let dense = run(vec![0.03125, 0.125, 0.25]);
            assert_eq!(sparse.spatial_time_history(), dense.spatial_time_history());
            assert!(dense.spatial_stage_solves(0).is_none());
            // Two cells: Q1 M_ii=1/3, K_ii=4; TPFA M=I/2 and
            // K=[[6,-2],[-2,6]], so the symmetric mode decays at 12 or 8 / s.
            let q1 = spatial == CommonSpatialPolicy::Q1;
            let rate: f64 = if q1 { 12. } else { 8. };
            for (index, time) in [0.03125, 0.125, 0.25].into_iter().enumerate() {
                let state = dense.spatial_state(index).unwrap();
                let values = plan
                    .as_spatial()
                    .unwrap()
                    .state_fields(state)
                    .unwrap()
                    .next()
                    .unwrap()
                    .3;
                let stride = if complex { 2 } else { 1 };
                let offset = if q1 { stride } else { 0 };
                assert!((values[offset] - (-rate * time).exp()).abs() < 2e-9);
                if complex {
                    assert!((values[offset + 1] - 2. * (-rate * time).exp()).abs() < 2e-9);
                }
            }
            let restart = CommonState::from_bytes(
                &dense.spatial_state(1).unwrap().to_bytes().unwrap(),
                &plan,
            )
            .unwrap();
            let resumed =
                CommonTransientRunRequest::from_times(plan.clone(), restart, 0.25, vec![0.25])
                    .unwrap()
                    .run(
                        &REFERENCE_LINEAR_SOLVER,
                        |problem, time| DiffsolTimeBackend::new().solve(problem, time),
                        |_, _| panic!("unexpected residual-native backend"),
                    )
                    .unwrap();
            let resumed_values = plan
                .as_spatial()
                .unwrap()
                .state_fields(resumed.spatial_state(0).unwrap())
                .unwrap()
                .next()
                .unwrap()
                .3;
            let uninterrupted_values = plan
                .as_spatial()
                .unwrap()
                .state_fields(dense.spatial_state(2).unwrap())
                .unwrap()
                .next()
                .unwrap()
                .3;
            for (actual, reference) in resumed_values.iter().zip(uninterrupted_values) {
                assert!((actual - reference).abs() < 2e-9);
            }
            let bytes = dense.to_bytes().unwrap();
            assert_eq!(CommonTrajectory::from_bytes(&bytes, &plan).unwrap(), dense);
            let mut forged: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            forged["payload"]["execution"]["report"]["backend_version"] =
                serde_json::json!("foreign");
            assert!(
                CommonTrajectory::from_bytes(&serde_json::to_vec(&forged).unwrap(), &plan).is_err()
            );
            let result = CommonResult::accept_trajectory(0., dense).unwrap();
            assert_eq!(
                CommonResult::from_bytes(&result.to_bytes().unwrap(), &plan).unwrap(),
                result
            );
            let rejected = request(&plan, &initial, vec![0.125, 0.25])
                .run(
                    &REFERENCE_LINEAR_SOLVER,
                    |problem, time| {
                        let solution = DiffsolTimeBackend::new().solve(problem, time)?;
                        let mut values = (0..solution.times().len())
                            .flat_map(|index| solution.state(index).unwrap().iter().copied())
                            .collect::<Vec<_>>();
                        values[0] += 1.;
                        TimeSolution::accepted_with_history(
                            solution.dimension(),
                            solution.times().to_vec(),
                            values,
                            solution.report(),
                            solution.history().unwrap().clone(),
                        )
                    },
                    |_, _| panic!("unexpected residual-native backend"),
                )
                .unwrap_err();
            assert!(
                rejected.message().contains("accepted time history"),
                "{rejected}"
            );
            assert_eq!(initial.to_bytes().unwrap(), before);
        }
    }
}

#[test]
fn adaptive_spatial_rates_keep_nonlinear_and_conjugate_storage() {
    for spatial in [
        CommonSpatialPolicy::Q1,
        CommonSpatialPolicy::CellCenteredTpfa,
    ] {
        for (complex, balance) in [
            (false, "derivative(u*u)=-u*u/1[s];"),
            (true, "derivative(math.conj(u))=-u/1[s];"),
        ] {
            let (plan, initial) = fixture(complex, spatial, balance, false);
            let trajectory = request(&plan, &initial, vec![0.25])
                .run(
                    &REFERENCE_LINEAR_SOLVER,
                    |problem, time| DiffsolTimeBackend::new().solve(problem, time),
                    |_, _| panic!("unexpected residual-native backend"),
                )
                .unwrap();
            let values = plan
                .as_spatial()
                .unwrap()
                .state_fields(trajectory.spatial_state(0).unwrap())
                .unwrap()
                .next()
                .unwrap()
                .3;
            // 2*u*u'=-u^2 gives u=exp(-t/2). Conjugation gives x'=-x, y'=y.
            let expected = if complex {
                vec![(-0.25_f64).exp(), 2. * 0.25_f64.exp()]
            } else {
                vec![(-0.125_f64).exp()]
            };
            for point in values.chunks_exact(expected.len()) {
                for (actual, expected) in point.iter().zip(&expected) {
                    assert!((actual - expected).abs() < 2e-9);
                }
            }
            assert_eq!(
                CommonTrajectory::from_bytes(&trajectory.to_bytes().unwrap(), &plan).unwrap(),
                trajectory
            );
        }
    }
}
