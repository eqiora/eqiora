use super::*;

#[test]
fn planar_storage_preserves_mass_history_and_restart() {
    let geometry = geometry(false);
    let body = geometry.entity_set("body").unwrap();
    let source = r#"
public component Heat(
    support body: volume(ambient_dimension = 2),
    support outer: boundary(parent = body),
    parameter c: s / m^2
) {
    state u: 1 on body in h1;
    initial { u = 2; }
    law balance on body { storage c * u; flux -grad(u); source 3 [1 / m^2]; }
    relation prescribed on outer { trace(u) = 2; }
}
"#;
    // Four area-1/4 triangles: integral(phi_c^2)=1/6,
    // integral(|grad(phi_c)|^2)=4, integral(3*phi_c)=1.
    // Fixed boundary 2 and dt=1/4 give (4+2*c/3)*e_next=(2*c/3)*e+1.
    for (capacity, expected) in [
        (3., [1. / 6., 2. / 9., 13. / 54.]),
        (6., [1. / 8., 3. / 16., 7. / 32.]),
    ] {
        let model = compile_model(
            "planar-storage.eqi",
            source,
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
            &[(
                "c",
                DynQuantity::new(
                    capacity,
                    DimExponents::from_integers([0, -2, 1, 0, 0, 0, 0]).unwrap(),
                ),
            )],
        );
        for permuted in [false, true] {
            let resolved = ResolvedCommonPlan::resolve(
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
            let time = eqiora_time::TimeBackendCapabilities::new(
                eqiora_time::TimeBackendIdentity::new("eqiora.test.time", "1"),
                &[
                    eqiora_core::ScalarDomain::Real,
                    eqiora_core::ScalarDomain::Complex,
                ],
                &[eqiora_core::ScalarType::F64],
            );
            let bytes = resolved.to_bytes().unwrap();
            let replay =
                ResolvedCommonPlan::from_bytes(&bytes, &REFERENCE_LINEAR_SOLVER, time).unwrap();
            assert_eq!(replay.to_bytes().unwrap(), bytes);
            let initial = replay.as_linear().unwrap().initial_state().unwrap();
            assert_eq!(initial.scalar_values().unwrap(), &[2.; 5]);
            let plan = replay.as_linear().unwrap();
            assert!(
                plan.scalar_state(0., vec![2.; 4])
                    .unwrap_err()
                    .message()
                    .contains("complete finite nodal")
            );
            assert!(
                plan.scalar_state(0., vec![f64::NAN; 5])
                    .unwrap_err()
                    .message()
                    .contains("complete finite nodal")
            );
            assert!(
                plan.scalar_state(0., vec![0.; 5])
                    .unwrap_err()
                    .message()
                    .contains("prescribed boundary")
            );
            let run =
                CommonTransientRunRequest::from_steps(replay.clone(), initial, 3, vec![1, 2, 3])
                    .unwrap();
            let std::ops::ControlFlow::Continue(outputs) = run
                .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
                .unwrap()
            else {
                panic!("three accepted steps must complete");
            };
            assert_eq!(outputs.len(), 3);
            for ((_, state), excess) in outputs.iter().zip(expected) {
                let values = state.scalar_values().unwrap();
                assert_eq!(&values[..4], &[2.; 4]);
                assert!((values[4] - (2. + excess)).abs() < 1e-12);
                let bytes = state.to_bytes().unwrap();
                assert_eq!(CommonState::from_bytes(&bytes, &replay).unwrap(), *state);
            }
            let restart =
                CommonState::from_bytes(&outputs[0].1.to_bytes().unwrap(), &replay).unwrap();
            let run =
                CommonTransientRunRequest::from_steps(replay, restart, 2, vec![1, 2]).unwrap();
            let std::ops::ControlFlow::Continue(restarted) = run
                .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
                .unwrap()
            else {
                panic!("restart must complete");
            };
            assert_eq!(restarted[1].1, outputs[2].1);
        }
    }
}

#[test]
fn planar_storage_initializes_affine_coordinates_and_keeps_equilibrium() {
    let geometry = geometry(false);
    let body = geometry.entity_set("body").unwrap();
    let model = compile_model(
        "planar-affine-storage.eqi",
        r#"
public component Heat(
    support body: volume(ambient_dimension = 2),
    support outer: boundary(parent = body),
    parameter c: s / m^2
) {
    coordinate x: m on body from body[0];
    state u: m on body in h1;
    initial { u = x; }
    law balance on body { storage c * u; flux -grad(u); source 0 [1 / m]; }
    relation prescribed on outer { trace(u) = coordinate(0); }
}
"#,
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
        &[(
            "c",
            DynQuantity::new(
                3.,
                DimExponents::from_integers([0, -2, 1, 0, 0, 0, 0]).unwrap(),
            ),
        )],
    );
    let owner = resources(&geometry, false);
    let expected = owner
        .simplicial_mesh()
        .unwrap()
        .mesh()
        .vertices()
        .iter()
        .map(|p| p[0])
        .collect::<Vec<_>>();
    let resolved = ResolvedCommonPlan::resolve(
        &model,
        owner,
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
    let initial = resolved.as_linear().unwrap().initial_state().unwrap();
    assert_eq!(initial.scalar_values().unwrap(), expected);
    let run = CommonTransientRunRequest::from_steps(resolved, initial, 2, vec![2]).unwrap();
    let std::ops::ControlFlow::Continue(outputs) = run
        .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
        .unwrap()
    else {
        panic!("affine equilibrium must complete");
    };
    for (actual, expected) in outputs[0].1.scalar_values().unwrap().iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-12);
    }
}
