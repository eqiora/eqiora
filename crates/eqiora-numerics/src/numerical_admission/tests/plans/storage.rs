use super::*;
mod algebraic;
mod algebraic_kinematic;
mod coupled;
mod displacement_boundary;
mod initial;
mod kinematic;
mod regions;
mod vector;

#[test]
fn scalar_region_run_preserves_consistent_mass_and_nonzero_boundary_history() {
    let geometry = cartesian_interval();
    let body = geometry.entity_set("body").unwrap();
    let source = r#"
public component Heat(
    support body: volume(ambient_dimension = 1),
    support left: boundary(parent = body),
    support right: boundary(parent = body),
    parameter c: s / m^2
) {
    state u: 1 on body in h1;
    initial { u = 2; }
    law balance on body { storage c * u; flux -grad(u); source 4 [1 / m^2]; }
    relation left_value on left { trace(u) = 2; }
    relation right_value on right { trace(u) = 2; }
}
"#;
    // Two length-1/2 elements: M_ii=c/3, K_ii=4, F_i=2.
    // With fixed endpoints 2 and dt=1/4, the interior excess follows
    // e_next = c/(c+3)*e + (3/2)/(c+3), independently of assembly output.
    for (capacity, expected) in [
        (3., [2.25, 2.375, 2.4375]),
        (6., [2. + 1. / 6., 2. + 5. / 18., 2. + 19. / 54.]),
    ] {
        let model = compile_model(
            "scalar-history.eqi",
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
            &[(
                "c",
                DynQuantity::new(
                    capacity,
                    DimExponents::from_integers([0, -2, 1, 0, 0, 0, 0]).unwrap(),
                ),
            )],
        );
        let resolved = ResolvedCommonPlan::resolve(
            &model,
            cartesian_box_resources(&geometry, &[2]),
            CommonSpatialPolicy::Q1,
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
        let initial = resolved
            .as_linear()
            .unwrap()
            .initial_state(0.0, Vec::new())
            .unwrap();
        assert_eq!(initial.linear_values().unwrap(), &[2.; 3]);
        for (values, message) in [
            (vec![2.; 2], "exact mapped coefficient inventory"),
            (vec![2., f64::NAN, 2.], "history is nonfinite"),
        ] {
            let mut invalid = initial.clone();
            invalid.kind = CommonStateKind::Linear(values.into_boxed_slice());
            let error = resolved
                .as_linear()
                .unwrap()
                .advance_scalar(&invalid, &REFERENCE_LINEAR_SOLVER, 0.25)
                .unwrap_err();
            assert!(error.message().contains(message), "{}", error.message());
        }
        let run =
            CommonTransientRunRequest::from_steps(resolved, initial, 3, vec![1, 2, 3]).unwrap();
        let std::ops::ControlFlow::Continue(outputs) = run
            .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
            .unwrap()
        else {
            panic!("three accepted steps must complete");
        };
        assert_eq!(outputs.len(), 3);
        for ((_, state), expected) in outputs.iter().zip(expected) {
            let values = state.linear_values().unwrap();
            assert_eq!(values[0], 2.);
            assert_eq!(values[2], 2.);
            assert!((values[1] - expected).abs() < 1e-12);
        }
    }
}
