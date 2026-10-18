//! Retained occurrence queries share the static physical reconstruction.
use super::*;
use eqiora_schema::kernel::KernelNode;

#[test]
fn planar_storage_observations_select_exact_sparse_outputs() {
    let geometry = geometry(false);
    let body = geometry.entity_set("body").unwrap();
    let model = compile_model(
        "retained-observations.eqi",
        r#"public component Heat(
            support body:volume(ambient_dimension=2),
            support outer:boundary(parent=body)
        ) {
            state u:1 on body in h1;
            initial {u=2;}
            law balance on body {storage 3[s/m^2]*u; flux -grad(u); source 3[1/m^2];}
            relation fixed on outer {trace(u)=2;}
            observable profile:1 on body=u;
            observable volume:m^2=integral(u,measure(body));
            observable boundary:m=integral(trace(u),measure(outer));
            observable clock:s=time();
            observable weighted:m^2*s=integral(time()*u,measure(body));
        }"#,
        &geometry,
        "Heat",
        &[
            ("body", body, None),
            (
                "outer",
                geometry.entity_set("outer").unwrap(),
                Some(("body", body)),
            ),
        ],
        &[],
    );
    for permuted in [false, true] {
        let plan = ResolvedCommonPlan::resolve(
            &model,
            resources(&geometry, permuted),
            CommonSpatialPolicy::P1,
            CommonSolvePolicy::Linear(exact_reference_linear(
                LinearSolver::BiConjugateGradientStabilized,
                1e-12,
                1e-14,
                NonZeroUsize::new(100).unwrap(),
            )),
            None,
            Some(CommonBackwardEuler::from_seconds(0.25).unwrap()),
            &REFERENCE_LINEAR_SOLVER,
            None,
        )
        .unwrap();
        let spatial = plan.as_spatial().unwrap();
        let initial = spatial.initial_state(Some(0.), Vec::new()).unwrap();
        let run =
            CommonTransientRunRequest::from_steps(plan.clone(), initial, 3, vec![1, 3]).unwrap();
        let std::ops::ControlFlow::Continue(outputs) = run
            .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
            .unwrap()
        else {
            panic!("complete run required")
        };
        let trajectory = crate::CommonTrajectory::accept_spatial_transient(run, outputs).unwrap();
        let result = crate::CommonResult::accept_trajectory(0., trajectory).unwrap();
        let bytes = result.to_bytes().unwrap();
        let replay = crate::CommonResult::from_bytes(&bytes, &plan).unwrap();
        for result in [&result, &replay] {
            // Four area-1/4 triangles give M_cc=1/6, K_cc=4, F_c=1.
            // With c=3 and dt=1/4, 6*e_next=2*e+1, so e_1=1/6,
            // e_3=13/54. Integral(phi_c)=1/3, perimeter=4.
            for (index, (time, excess)) in
                [(0.25, 1. / 6.), (0.75, 13. / 54.)].into_iter().enumerate()
            {
                let state = result.trajectory().unwrap().spatial_steps().unwrap()[index]
                    .1
                    .state();
                for definition in spatial.kernel().nodes().filter_map(|node| match node {
                    KernelNode::Observable(value) => Some(value),
                    _ => None,
                }) {
                    let unit = definition.value_type().dimension();
                    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
                    let seconds = DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).unwrap();
                    let (observation, expected) = if unit == DimExponents::DIMENSIONLESS {
                        (
                            result
                                .observe_at(
                                    &model,
                                    definition.id(),
                                    &[DynQuantity::new(0.5, length); 2],
                                    &std::collections::HashMap::new(),
                                    Some(index),
                                )
                                .unwrap(),
                            2. + excess,
                        )
                    } else if unit == seconds {
                        (
                            result
                                .observe(
                                    &model,
                                    definition.id(),
                                    &std::collections::HashMap::new(),
                                    Some(index),
                                )
                                .unwrap(),
                            time,
                        )
                    } else {
                        let rules = result
                            .plan()
                            .observation_quadratures(&[definition.id()], 2)
                            .unwrap();
                        let expected = if unit == length {
                            8.
                        } else if unit
                            == DimExponents::from_integers([0, 2, 1, 0, 0, 0, 0]).unwrap()
                        {
                            time * (2. + excess / 3.)
                        } else {
                            2. + excess / 3.
                        };
                        let observation = result
                            .observe(&model, definition.id(), &rules, Some(index))
                            .unwrap();
                        assert_eq!(observation.quadratures(), &rules);
                        assert!(
                            result
                                .observe(&model, definition.id(), &rules, None)
                                .unwrap_err()
                                .message()
                                .contains("explicit output index")
                        );
                        assert!(
                            result
                                .observe(&model, definition.id(), &rules, Some(2))
                                .unwrap_err()
                                .message()
                                .contains("outside the retained schedule")
                        );
                        (observation, expected)
                    };
                    let actual = observation.value().real_scalar_value().unwrap();
                    assert!(
                        (actual.value() - expected).abs() < 1e-11,
                        "{actual:?}, expected {expected}"
                    );
                    assert_eq!(actual.dim(), unit);
                    assert_eq!(observation.result_identity(), result.identity());
                    assert_eq!(observation.output(), Some((index, time, state.identity())));
                }
            }
            assert_eq!(result.to_bytes().unwrap(), bytes);
        }
    }
}

#[test]
fn planar_time_integrals_require_covered_accepted_intervals() {
    let geometry = geometry(false);
    let body = geometry.entity_set("body").unwrap();
    let model = compile_model(
        "spatial-time.eqi",
        r#"public component Ramp(support body:volume(ambient_dimension=2), support outer:boundary(parent=body)) {
        state u:1 on body in smooth;
        initial {u=0;}
        law balance on body {storage 1[s/m^2]*u; flux -grad(u); source 1[1/m^2];}
        relation insulated on outer {normal(grad(u))=0;}
        observable amount:m^2=integral(u*u,measure(body));
        observable profile:1 on body=u;
    }"#,
        &geometry,
        "Ramp",
        &[
            ("body", body, None),
            (
                "outer",
                geometry.entity_set("outer").unwrap(),
                Some(("body", body)),
            ),
        ],
        &[],
    );
    for permuted in [false, true] {
        let plan = ResolvedCommonPlan::resolve(
            &model,
            resources(&geometry, permuted),
            CommonSpatialPolicy::P1,
            CommonSolvePolicy::Linear(exact_reference_linear(
                LinearSolver::BiConjugateGradientStabilized,
                1e-12,
                1e-14,
                NonZeroUsize::new(100).unwrap(),
            )),
            None,
            Some(CommonBackwardEuler::from_seconds(0.25).unwrap()),
            &REFERENCE_LINEAR_SOLVER,
            None,
        )
        .unwrap();
        let spatial = plan.as_spatial().unwrap();
        let definitions = spatial
            .kernel()
            .nodes()
            .filter_map(|node| {
                if let KernelNode::Observable(value) = node {
                    Some(value)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let amount = definitions
            .iter()
            .find(|d| d.value_type().dimension() != DimExponents::DIMENSIONLESS)
            .unwrap()
            .id();
        let profile = definitions
            .iter()
            .find(|d| d.value_type().dimension() == DimExponents::DIMENSIONLESS)
            .unwrap()
            .id();
        let run = |outputs| {
            let initial = spatial.initial_state(Some(0.), Vec::new()).unwrap();
            let request =
                CommonTransientRunRequest::from_steps(plan.clone(), initial, 4, outputs).unwrap();
            let std::ops::ControlFlow::Continue(steps) = request
                .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
                .unwrap()
            else {
                panic!("complete run")
            };
            crate::CommonResult::accept_trajectory(
                0.,
                crate::CommonTrajectory::accept_spatial_transient(request, steps).unwrap(),
            )
            .unwrap()
        };
        let dense = run(vec![1, 2, 3, 4]);
        let sparse = run(vec![1, 3]);
        let rules = dense.plan().observation_quadratures(&[amount], 2).unwrap();
        let rule = crate::TimeObservationInterpolation::EndpointLinear;
        // The constant-in-space solution u=t is exact under backward Euler.
        // Integrating the piecewise-linear Observable t^2 at h=1/4 gives
        // 1/3+h^2/6=11/32; clipping to [1/8,7/8] gives 59/256.
        for result in [&dense, &sparse] {
            let bytes = result.to_bytes().unwrap();
            let replay = crate::CommonResult::from_bytes(&bytes, &plan).unwrap();
            for candidate in [result, &replay] {
                let trajectory = candidate.trajectory().unwrap();
                for (window, expected) in [([0., 0.25], 1. / 128.), ([0.5, 0.75], 13. / 128.)] {
                    let value = trajectory
                        .observe_time_integral(&model, amount, rule, Some(window), &rules)
                        .unwrap();
                    assert!(
                        (value.value().real_scalar_value().unwrap().value() - expected).abs()
                            < 1e-11
                    );
                    assert_eq!(
                        value.value().value_type().dimension(),
                        DimExponents::from_integers([0, 2, 1, 0, 0, 0, 0]).unwrap()
                    );
                    let minimum = trajectory
                        .observe_time_minimum(&model, amount, rule, Some(window), &rules)
                        .unwrap();
                    let maximum = trajectory
                        .observe_time_maximum(&model, amount, rule, Some(window), &rules)
                        .unwrap();
                    for (extreme, time) in [(minimum, window[0]), (maximum, window[1])] {
                        assert!(
                            (extreme.value().real_scalar_value().unwrap().value() - time * time)
                                .abs()
                                < 1e-11
                        );
                        assert_eq!(extreme.extremum_time_s(), Some(time));
                        assert_eq!(extreme.spatial_quadratures(), &rules);
                        assert_eq!(
                            extreme.value().value_type().dimension(),
                            DimExponents::from_integers([0, 2, 0, 0, 0, 0, 0]).unwrap()
                        );
                    }
                    assert_eq!(value.interval_s(), window);
                    assert_eq!(value.spatial_quadratures(), &rules);
                    assert_eq!(value.trajectory_identity(), trajectory.identity());
                }
                if trajectory.spatial_steps().unwrap().len() == 4 {
                    for (window, expected) in
                        [(None, 11. / 32.), (Some([0.125, 0.875]), 59. / 256.)]
                    {
                        let value = trajectory
                            .observe_time_integral(&model, amount, rule, window, &rules)
                            .unwrap();
                        assert!(
                            (value.value().real_scalar_value().unwrap().value() - expected).abs()
                                < 1e-11
                        );
                    }
                    let min = trajectory
                        .observe_time_minimum(&model, amount, rule, Some([0.125, 0.875]), &rules)
                        .unwrap();
                    let max = trajectory
                        .observe_time_maximum(&model, amount, rule, Some([0.125, 0.875]), &rules)
                        .unwrap();
                    assert!(
                        (min.value().real_scalar_value().unwrap().value() - 1. / 32.).abs() < 1e-11
                    );
                    assert!(
                        (max.value().real_scalar_value().unwrap().value() - 25. / 32.).abs()
                            < 1e-11
                    );
                    assert_eq!(min.extremum_time_s(), Some(0.125));
                    assert_eq!(max.extremum_time_s(), Some(0.875));
                    assert!(
                        trajectory
                            .observe_time_integral(
                                &model,
                                amount,
                                crate::TimeObservationInterpolation::NativeMidpointQuadratic,
                                None,
                                &rules
                            )
                            .unwrap_err()
                            .message()
                            .contains("no native midpoint")
                    );
                    assert!(
                        trajectory
                            .observe_time_integral(&model, amount, rule, None, &Default::default())
                            .is_err()
                    );
                    assert!(
                        trajectory
                            .observe_time_integral(&model, profile, rule, None, &Default::default())
                            .unwrap_err()
                            .message()
                            .contains("finite Observable")
                    );
                } else {
                    for observable in [amount, profile] {
                        for window in [None, Some([0., 0.5]), Some([0.25, 0.5]), Some([0.5, 1.])] {
                            assert!(
                                trajectory
                                    .observe_time_minimum(&model, observable, rule, window, &rules)
                                    .unwrap_err()
                                    .message()
                                    .contains("not covered by retained accepted steps")
                            );
                            assert!(
                                trajectory
                                    .observe_time_integral(&model, observable, rule, window, &rules)
                                    .unwrap_err()
                                    .message()
                                    .contains("not covered by retained accepted steps")
                            );
                        }
                    }
                }
                assert_eq!(candidate.to_bytes().unwrap(), bytes);
            }
        }
    }
}
