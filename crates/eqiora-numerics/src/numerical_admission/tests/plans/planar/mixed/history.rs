use super::*;

#[test]
fn mini_history_preserves_uniform_throughflow_and_zero_bubble_coefficients() {
    let source = FLOW
        .replace("variable u:", "state u:")
        .replace("variable p:", "state p:")
        .replace(
            "{-div(2[kg/(m*s)]",
            "{1[kg/m^3]*derivative(u)-div(2[kg/(m*s)]",
        )
        .replace(
            "relation constraint on body",
            "initial {u=grad(g); p=0[kg/(m*s^2)];} relation constraint on body",
        );
    let geometry = geometry(true);
    let body = geometry.entity_set("body").unwrap();
    let mut supports = vec![("body", body, None)];
    supports.extend(["left", "right", "bottom", "top"].map(|name| {
        (
            name,
            geometry.entity_set(name).unwrap(),
            Some(("body", body)),
        )
    }));
    let model = compile_model(
        "mini-history.eqi",
        &source,
        &geometry,
        "Throughflow",
        &supports,
        &[],
    );
    for algorithm in [
        LinearSolver::BiConjugateGradientStabilized,
        LinearSolver::MinimumResidual,
    ] {
        for permuted in [false, true] {
            let resolved = ResolvedCommonPlan::resolve(
                &model,
                open_resources(&geometry, permuted),
                CommonMethodRequest::Exact {
                    spatial: CommonSpatialPolicy::MiniP1,
                    formulation: FormulationKind::MixedGalerkin,
                },
                CommonSolvePolicy::Linear(exact_reference_linear(
                    algorithm,
                    // An absolute target bounds every algebraic subvector by the
                    // complete residual norm; storage RHS scaling cannot weaken it.
                    0.0,
                    1e-12,
                    NonZeroUsize::new(1000).unwrap(),
                )),
                None,
                Some(CommonBackwardEuler::from_seconds(0.25).unwrap().into()),
                &REFERENCE_LINEAR_SOLVER,
                None,
                None,
            )
            .unwrap();
            let resolved = replay_plan(resolved, &REFERENCE_LINEAR_SOLVER);
            let plan = resolved.as_spatial().unwrap();
            let expected = plan
                .fields()
                .flat_map(|(_, ty)| {
                    if ty.shape().is_scalar() {
                        vec![0.0; 6]
                    } else {
                        let mut values = [1.0, 0.0].repeat(6);
                        values.extend([0.0; 10]);
                        values
                    }
                })
                .collect::<Vec<_>>();
            let supplied = plan
                .fields()
                .map(|(field, ty)| {
                    let vertices = if ty.shape().is_scalar() {
                        vec![0.0; 6]
                    } else {
                        [1.0, 0.0].repeat(6)
                    };
                    let cells = (!ty.shape().is_scalar()).then(|| {
                        CommonInitialValues::new(ty.shape().clone(), vec![0.0; 10]).unwrap()
                    });
                    CommonInitialField::new(
                        model.digest().unwrap(),
                        field,
                        Some(CommonInitialValues::new(ty.shape().clone(), vertices).unwrap()),
                        cells,
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();
            let supplied_state = plan.initial_state(Some(0.25), supplied.clone()).unwrap();
            assert_eq!(supplied_state.spatial_values().unwrap(), expected);
            let velocity = supplied
                .iter()
                .position(|field| field.cell().is_some())
                .unwrap();
            let pressure = 1 - velocity;
            // Constant throughflow has zero viscous residual and divergence.
            // With fixed inlet/wall rates, consistency requires
            // M*u_dot + B^T*p = 0 and B*u_dot = 0. M is positive definite
            // and this open-outlet MINI pair has full-row-rank B, so
            // p^T*B*M^-1*B^T*p = 0 requires p=0. An arbitrary nonzero p
            // satisfies the original divergence row but has no consistent rate.
            let mut incompatible_rate = supplied.clone();
            incompatible_rate[pressure] = CommonInitialField::new(
                model.digest().unwrap(),
                supplied[pressure].field(),
                Some(
                    CommonInitialValues::new(eqiora_core::ValueShape::scalar(), vec![1.0; 6])
                        .unwrap(),
                ),
                None,
            )
            .unwrap();
            let Err(error) = plan.initial_state(Some(0.25), incompatible_rate) else {
                panic!("nonzero pressure has no compatible initial velocity rate");
            };
            assert!(error.message().contains("rate compatibility"), "{error:?}");
            for cell_values in [
                None,
                Some(
                    CommonInitialValues::new(
                        eqiora_core::ValueShape::new([2]).unwrap(),
                        vec![0.0; 8],
                    )
                    .unwrap(),
                ),
            ] {
                let mut bad = supplied.clone();
                bad[velocity] = CommonInitialField::new(
                    model.digest().unwrap(),
                    supplied[velocity].field(),
                    supplied[velocity].vertex().cloned(),
                    cell_values,
                )
                .unwrap();
                assert!(plan.initial_state(Some(0.25), bad).is_err());
            }
            let mut bad = supplied.clone();
            bad[pressure] = CommonInitialField::new(
                model.digest().unwrap(),
                supplied[pressure].field(),
                supplied[pressure].vertex().cloned(),
                Some(
                    CommonInitialValues::new(eqiora_core::ValueShape::scalar(), vec![0.0; 5])
                        .unwrap(),
                ),
            )
            .unwrap();
            assert!(plan.initial_state(Some(0.25), bad).is_err());
            let initial = plan.initial_state(Some(0.0), Vec::new()).unwrap();
            let initial = CommonState::from_bytes(&initial.to_bytes().unwrap(), &resolved).unwrap();
            for (actual, expected) in initial.spatial_values().unwrap().iter().zip(&expected) {
                assert!((actual - expected).abs() < 1e-13);
            }
            let retained = initial.to_bytes().unwrap();
            let run = CommonTransientRunRequest::from_steps(
                resolved.clone(),
                initial.clone(),
                3,
                vec![1, 2, 3],
            )
            .unwrap();
            let std::ops::ControlFlow::Continue(outputs) = run
                .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
                .unwrap()
            else {
                panic!("complete run")
            };
            assert_eq!(retained, initial.to_bytes().unwrap());
            for (_, state) in &outputs {
                let state = state.state();
                let state = CommonState::from_bytes(&state.to_bytes().unwrap(), &resolved).unwrap();
                assert_eq!(state.spatial_values().unwrap().len(), expected.len());
                for (actual, expected) in state.spatial_values().unwrap().iter().zip(&expected) {
                    assert!((actual - expected).abs() < 1e-10);
                }
            }
            let restart =
                CommonState::from_bytes(&outputs[0].1.state().to_bytes().unwrap(), &resolved)
                    .unwrap();
            let run = CommonTransientRunRequest::from_steps(resolved, restart, 2, vec![2]).unwrap();
            let std::ops::ControlFlow::Continue(restarted) = run
                .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
                .unwrap()
            else {
                panic!("complete restart")
            };
            assert_eq!(restarted.last().unwrap().1, outputs[2].1);
        }
    }
}
