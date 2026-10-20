use super::*;

#[test]
fn stationary_material_flux_matches_nonzero_natural_data_without_double_counting() {
    for upper in [None, Some("4"), Some("1.99")] {
        let geometry = cartesian_interval();
        let source = POISSON_INTERVAL
            .replace("on body in h1", "on body in smooth")
            .replace(
                "-div(grad(potential)) - source_scale",
                "-div((1+potential^2)*grad(potential))+2*(1+coordinate(0)/1[m])/1[m^2]",
            )
            .replace(
                "left { trace(potential) = 0; }",
                "left { trace(potential) = 1; }",
            )
            .replace(
                "right { trace(potential) = 0; }",
                "right { normal((1+potential^2)*grad(potential)) = 5[1/m]; }",
            );
        let source = if let Some(upper) = upper {
            guarded_natural_source(&source, upper)
        } else {
            source
        };
        let model = scalar_box_model(&geometry, &source, "PoissonInterval", &["left", "right"]);
        let plan = ResolvedCommonPlan::resolve(
            &model,
            cartesian_box_resources(&geometry, &[2]),
            CommonSpatialPolicy::Q1,
            CommonSolvePolicy::Newton {
                linear: exact_reference_linear(
                    LinearSolver::BiConjugateGradientStabilized,
                    1e-13,
                    1e-14,
                    NonZeroUsize::new(100).unwrap(),
                ),
                nonlinear: NonlinearSolvePlan::new(0., 1e-11, NonZeroUsize::new(16).unwrap(), 16)
                    .unwrap(),
            },
            None,
            None,
            &REFERENCE_LINEAR_SOLVER,
            None,
        )
        .unwrap();
        let plan = replay_plan(plan, &REFERENCE_LINEAR_SOLVER);
        let spatial = plan.as_spatial().unwrap();
        let field = spatial.fields().next().unwrap().0;
        let initial = spatial
            .initial_state(
                None,
                vec![
                    CommonInitialField::new(
                        model.digest().unwrap(),
                        field,
                        Some(
                            CommonInitialValues::new(
                                eqiora_core::ValueShape::scalar(),
                                if upper == Some("1.99") {
                                    vec![1., 1.5, 2.]
                                } else {
                                    vec![1.; 3]
                                },
                            )
                            .unwrap(),
                        ),
                        None,
                    )
                    .unwrap(),
                ],
            )
            .unwrap();
        let result = spatial.run_result(Some(&initial), &REFERENCE_LINEAR_SOLVER);
        if upper == Some("1.99") {
            // On the last half-cell, even the largest volume abscissa has
            // u=7/4+1/(4*sqrt(3))<1.99. Only the natural face has u=2.
            assert_eq!(
                result.err().unwrap().code(),
                eqiora_core::diagnostic::codes::EXPRESSION_DOMAIN_VIOLATION
            );
            continue;
        }
        let result = result.unwrap();
        assert!(result.nonlinear_iterations(None).unwrap() > 1);
        // On [0,1], u=1+x gives q=(1+u²)u', q'=2(1+x), q(1)=5.
        // Both the volume load and the outward datum are nonzero; adding q again
        // on the natural facet cannot satisfy these exact nodal values.
        for (actual, expected) in result
            .field_block(0, 0)
            .unwrap()
            .1
            .iter()
            .zip([1., 1.5, 2.])
        {
            assert!((actual - expected).abs() < 1e-11, "{actual} != {expected}");
        }
        assert_eq!(
            crate::CommonResult::from_bytes(&result.to_bytes().unwrap(), &plan).unwrap(),
            result
        );
        assert!(
            crate::CommonResult::from_bytes(
                &result_replay::reseal_values(&result, 0, &[1., 1.5, 2.25]),
                &plan,
            )
            .is_err()
        );
    }
}

fn guarded_natural_source(source: &str, upper: &str) -> String {
    let equation = source
        .replace(
            "public component PoissonInterval(",
            "component Equation(property response:Response,",
        )
        .replace("(1+potential^2)", "response(value=potential)");
    format!(
        "property contract Response(input value:1):1 {{derivatives first_partials;}}
property release Material:Response {{
 analytic {{value=1+value^2;source_unit:1=1;}}
 validity value>=0 and value<={upper};outside reject;branch single;
 citation org.example.manufactured;license spdx.CC0_1_0;
}}
{equation}
public component PoissonInterval(
 support body:volume(ambient_dimension=1),
 support left:boundary(parent=body),support right:boundary(parent=body),
 parameter source_scale:1/m^2
) {{
 instance physics:Equation(body=body,left=left,right=right,
   source_scale=source_scale,response=Material);
}}"
    )
}

#[test]
fn stationary_material_reaction_and_flux_close_the_public_newton_path() {
    let geometry = cartesian_interval();
    for flux in [false, true] {
        let diffusion = if flux {
            "-div((1+potential^2)*grad(potential))"
        } else {
            "-div(grad(potential))"
        };
        let source = POISSON_INTERVAL
            .replace(
                "-div(grad(potential)) - source_scale",
                &format!("{diffusion}+potential^2/1[m^2]-4*source_scale"),
            )
            .replace("trace(potential) = 0", "trace(potential) = 2");
        let model = scalar_box_model(&geometry, &source, "PoissonInterval", &["left", "right"]);
        let plan = ResolvedCommonPlan::resolve(
            &model,
            cartesian_box_resources(&geometry, &[2]),
            CommonSpatialPolicy::Q1,
            CommonSolvePolicy::Newton {
                linear: exact_reference_linear(
                    LinearSolver::BiConjugateGradientStabilized,
                    1e-13,
                    1e-14,
                    NonZeroUsize::new(100).unwrap(),
                ),
                nonlinear: NonlinearSolvePlan::new(0., 1e-11, NonZeroUsize::new(16).unwrap(), 16)
                    .unwrap(),
            },
            None,
            None,
            &REFERENCE_LINEAR_SOLVER,
            None,
        )
        .unwrap();
        let plan = replay_plan(plan, &REFERENCE_LINEAR_SOLVER);
        let spatial = plan.as_spatial().unwrap();
        let field = spatial.fields().next().unwrap().0;
        let initial = spatial
            .initial_state(
                None,
                vec![
                    CommonInitialField::new(
                        model.digest().unwrap(),
                        field,
                        Some(
                            CommonInitialValues::new(
                                eqiora_core::ValueShape::scalar(),
                                vec![2., 0., 2.],
                            )
                            .unwrap(),
                        ),
                        None,
                    )
                    .unwrap(),
                ],
            )
            .unwrap();
        let result = spatial
            .run_result(Some(&initial), &REFERENCE_LINEAR_SOLVER)
            .unwrap();
        assert!(result.nonlinear_iterations(None).unwrap() > 1);
        // u=2 makes both the complete nonlinear source residual and gradient zero.
        for value in result.field_block(0, 0).unwrap().1 {
            assert!((value - 2.).abs() < 1e-11);
        }
        assert_eq!(
            crate::CommonResult::from_bytes(&result.to_bytes().unwrap(), &plan).unwrap(),
            result
        );
        assert!(
            crate::CommonResult::from_bytes(
                &result_replay::reseal_values(&result, 0, &[2., 2.25, 2.]),
                &plan
            )
            .is_err()
        );
    }
}

#[test]
fn stationary_newton_retains_seed_history_and_replays_original_equations() {
    let geometry = cartesian_interval();
    let model = scalar_box_model(
        &geometry,
        POISSON_INTERVAL,
        "PoissonInterval",
        &["left", "right"],
    );
    let plan = ResolvedCommonPlan::resolve(
        &model,
        cartesian_box_resources(&geometry, &[4]),
        CommonSpatialPolicy::Q1,
        CommonSolvePolicy::Newton {
            linear: exact_reference_linear(
                LinearSolver::BiConjugateGradientStabilized,
                1e-13,
                1e-14,
                NonZeroUsize::new(100).unwrap(),
            ),
            nonlinear: NonlinearSolvePlan::new(0., 1e-11, NonZeroUsize::new(8).unwrap(), 8)
                .unwrap(),
        },
        None,
        None,
        &REFERENCE_LINEAR_SOLVER,
        None,
    )
    .unwrap();
    let plan = replay_plan(plan, &REFERENCE_LINEAR_SOLVER);
    let spatial = plan.as_spatial().unwrap();
    let field = spatial.fields().next().unwrap().0;
    let state = |values| {
        spatial
            .initial_state(
                None,
                vec![
                    CommonInitialField::new(
                        model.digest().unwrap(),
                        field,
                        Some(
                            CommonInitialValues::new(eqiora_core::ValueShape::scalar(), values)
                                .unwrap(),
                        ),
                        None,
                    )
                    .unwrap(),
                ],
            )
            .unwrap()
    };
    let initial = state(vec![0., 4., -2., 7., 0.]);
    let initial_bytes = initial.to_bytes().unwrap();
    assert!(spatial.run_result(None, &REFERENCE_LINEAR_SOLVER).is_err());
    let result = spatial
        .run_result(Some(&initial), &REFERENCE_LINEAR_SOLVER)
        .unwrap();
    assert_eq!(initial.to_bytes().unwrap(), initial_bytes);
    assert_eq!(result.nonlinear_iterations(None), Some(1));
    assert!(result.solve_evidence(None, None).is_none());
    assert!(result.solve_evidence(None, Some(0)).is_some());
    assert!(result.solve_evidence(None, Some(1)).is_none());
    assert!(
        result.nonlinear_true_residual_norm(None).unwrap()
            <= result.nonlinear_residual_target(None).unwrap()
    );
    let (_, actual, _) = result.field_block(0, 0).unwrap();
    // -u''=1 on [0,1], u(0)=u(1)=0: Q1 gives exact nodal x(1-x)/2.
    let exact = [0., 3. / 32., 1. / 8., 3. / 32., 0.];
    for (actual, exact) in actual.iter().zip(exact) {
        assert!((actual - exact).abs() < 1e-11);
    }
    let bytes = result.to_bytes().unwrap();
    assert_eq!(
        crate::CommonResult::from_bytes(&bytes, &plan).unwrap(),
        result
    );
    assert!(
        crate::CommonResult::from_bytes(&result_replay::reseal_values(&result, 0, &[0.; 5]), &plan)
            .unwrap_err()
            .message()
            .contains("residual")
    );
    let zero = spatial
        .run_result(Some(&state(exact.to_vec())), &REFERENCE_LINEAR_SOLVER)
        .unwrap();
    assert_eq!(zero.nonlinear_iterations(None), Some(0));
    assert!(zero.solve_evidence(None, Some(0)).is_none());
    assert_eq!(
        crate::CommonResult::from_bytes(&zero.to_bytes().unwrap(), &plan).unwrap(),
        zero
    );
    let obsolete = String::from_utf8(bytes)
        .unwrap()
        .replace("common-result/v25", "common-result/v24");
    assert!(crate::CommonResult::from_bytes(obsolete.as_bytes(), &plan).is_err());
}

#[test]
fn stationary_state_owns_complete_fields_without_model_time() {
    let geometry = cartesian_interval();
    let model = scalar_box_model(
        &geometry,
        POISSON_INTERVAL,
        "PoissonInterval",
        &["left", "right"],
    );
    let plan = ResolvedCommonPlan::Spatial(Box::new(resolve_scalar_box(
        &model,
        cartesian_box_resources(&geometry, &[4]),
        CommonSpatialPolicy::Q1,
    )));
    let spatial = plan.as_spatial().unwrap();
    let field = spatial.fields().next().unwrap().0;
    let assignment = |values| {
        CommonInitialField::new(
            model.digest().unwrap(),
            field,
            Some(CommonInitialValues::new(eqiora_core::ValueShape::scalar(), values).unwrap()),
            None,
        )
        .unwrap()
    };
    // A seed is not an accepted solution: these interior values deliberately
    // violate Poisson, while obeying the exact nodal support and boundary trace.
    let values = vec![0., 4., -2., 7., 0.];
    let state = spatial
        .initial_state(None, vec![assignment(values.clone())])
        .unwrap();
    assert_eq!(state.time_s(), None);
    assert_eq!(state.spatial_values(), Some(values.as_slice()));
    let bytes = state.to_bytes().unwrap();
    assert_eq!(CommonState::from_bytes(&bytes, &plan).unwrap(), state);
    let projected = spatial.state_fields(&state).unwrap().collect::<Vec<_>>();
    assert_eq!(projected.len(), 1);
    assert_eq!(projected[0].0, field);
    assert_eq!(projected[0].3, values);

    assert!(
        spatial
            .initial_state(Some(0.), vec![assignment(values)])
            .is_err()
    );
    assert!(spatial.initial_state(None, vec![]).is_err());
    assert!(
        spatial
            .initial_state(None, vec![assignment(vec![0.; 4])])
            .is_err()
    );
    assert!(
        spatial
            .initial_state(None, vec![assignment(vec![1.; 5])])
            .is_err()
    );
    assert!(
        CommonTransientRunRequest::from_steps(plan.clone(), state.clone(), 1, vec![1]).is_err()
    );
    assert!(CommonAcceptedSpatialStep::new(&state, state.clone()).is_err());

    let other = ResolvedCommonPlan::Spatial(Box::new(resolve_scalar_box(
        &model,
        cartesian_box_resources(&geometry, &[8]),
        CommonSpatialPolicy::Q1,
    )));
    assert!(CommonState::from_bytes(&bytes, &other).is_err());
    let obsolete = String::from_utf8(bytes)
        .unwrap()
        .replace("common-spatial-state/v6", "common-spatial-state/v5");
    assert!(CommonState::from_bytes(obsolete.as_bytes(), &plan).is_err());
}
