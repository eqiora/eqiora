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
        let initial = spatial.initial_state(0., Vec::new()).unwrap();
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
                        let rules = result.observation_quadratures(definition.id(), 2).unwrap();
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
