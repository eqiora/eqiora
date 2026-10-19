use super::*;

#[test]
fn steady_dyadic_transport_uses_newton_without_backward_euler() {
    let mut source = source(true, 0.);
    for index in 0..3 {
        for field in ["u", "v"] {
            let storage = format!("rho{index}*derivative({field}{index})\n                    +");
            assert!(source.contains(&storage));
            source = source.replace(&storage, "");
        }
    }
    let plan = resolve_with_solve(
        &source,
        CommonSolvePolicy::Newton {
            linear: exact_reference_linear(
                LinearSolver::BiConjugateGradientStabilized,
                1e-13,
                1e-14,
                NonZeroUsize::new(1000).unwrap(),
            ),
            nonlinear: NonlinearSolvePlan::new(0., 1e-10, NonZeroUsize::new(12).unwrap(), 12)
                .unwrap(),
        },
        None,
    )
    .unwrap();
    let plan = replay_plan(plan, &REFERENCE_LINEAR_SOLVER);
    assert!(plan.backward_euler().is_none());
    let spatial = plan.as_spatial().unwrap();
    let assignments = spatial
        .fields()
        .map(|(field, ty)| {
            let count = spatial.field_coefficient_entities(field).unwrap().len();
            let mut values = [1., 2.].repeat(count);
            // The centre of each 3x3 Region has no prescribed boundary value.
            // A perturbation demands genuine nonlinear updates while retaining all
            // exterior and conforming interface traces.
            assert_eq!(count, 9);
            values[8] += 0.125;
            CommonInitialField::new(
                plan.model_artifact().digest().unwrap(),
                field,
                Some(CommonInitialValues::new(ty.shape().clone(), values).unwrap()),
                None,
            )
            .unwrap()
        })
        .collect();
    let initial = spatial.initial_state(None, assignments).unwrap();
    let before = initial.to_bytes().unwrap();
    let result = spatial
        .run_result(Some(&initial), &REFERENCE_LINEAR_SOLVER)
        .unwrap();
    assert_eq!(before, initial.to_bytes().unwrap());
    assert!(result.nonlinear_iterations(None).unwrap() > 0);
    // Independently, each open Region has zero diffusion, feedback and
    // divergence at u=v=(1,2). Distinct densities require both interface fluxes.
    for field in 0..result.field_count() {
        let (_, values, _) = result.field_block(field, 0).unwrap();
        for pair in values.as_chunks::<2>().0 {
            assert!((pair[0] - 1.).abs() < 1e-9);
            assert!((pair[1] - 2.).abs() < 1e-9);
        }
    }
    assert!(result.nonlinear_true_residual_norm(None).unwrap() <= 1e-10);
    let bytes = result.to_bytes().unwrap();
    assert_eq!(
        crate::CommonResult::from_bytes(&bytes, &plan).unwrap(),
        result
    );
    let (_, values, _) = result.field_block(0, 0).unwrap();
    let mut changed = values.to_vec();
    changed[8] += 0.25;
    let error =
        crate::CommonResult::from_bytes(&result_replay::reseal_values(&result, 0, &changed), &plan)
            .unwrap_err();
    assert!(error.message().contains("residual"), "{error:?}");
}
