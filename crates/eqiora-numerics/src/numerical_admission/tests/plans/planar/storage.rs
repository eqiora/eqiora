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
            let initial = replay
                .as_linear()
                .unwrap()
                .initial_state(0.0, Vec::new())
                .unwrap();
            assert_eq!(initial.linear_values().unwrap(), &[2.; 5]);
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
                let state = state.state();
                let values = state.linear_values().unwrap();
                assert_eq!(&values[..4], &[2.; 4]);
                assert!((values[4] - (2. + excess)).abs() < 1e-12);
                let bytes = state.to_bytes().unwrap();
                assert_eq!(CommonState::from_bytes(&bytes, &replay).unwrap(), *state);
            }
            let trajectory =
                crate::CommonTrajectory::accept_spatial_transient(run.clone(), outputs.clone())
                    .unwrap();
            // u=2+e*phi_c on the unit square, with integral(phi_c)=1/3
            // and integral(phi_c^2)=1/6. These integrals independently fix
            // each volume term; no boundary-work/conservation claim is made.
            use crate::form_compiler::region::balance::BalanceRole;
            let first = &trajectory.spatial_steps().unwrap()[0].1;
            let balances = first.balance.as_ref().unwrap().as_ref().unwrap();
            assert!(
                balances
                    .defects()
                    .unwrap()
                    .iter()
                    .all(|(_, defect)| defect.value().abs() < 1e-11)
            );
            // The prescribed boundary supplies the independently derived residual
            // work. For e=1/6,c=3 this is -7/6; for e=1/8,c=6 it is -1.
            let boundary_work = balances
                .boundary
                .iter()
                .map(|term| term.work.value())
                .sum::<f64>();
            let e = expected[0];
            let expected_boundary = capacity * (2. * e / 3. + e * e / 6.) + e * e - 1.5 - e / 4.;
            assert!((boundary_work - expected_boundary).abs() < 1e-11);
            let balances = &balances.cells;
            assert_eq!(balances.len(), 4);
            let terms = balances
                .iter()
                .flat_map(|(_, cell)| &cell.contributions)
                .collect::<Vec<_>>();
            let sum = |role,
                       select: fn(
                &crate::form_compiler::region::balance::BalanceContribution,
            ) -> f64| {
                terms
                    .iter()
                    .filter(|term| term.role == role)
                    .map(|term| select(term))
                    .sum::<f64>()
            };
            let excess = expected[0];
            assert!(
                (sum(BalanceRole::Storage, |term| term.previous) - 2. * capacity).abs() < 1e-11
            );
            assert!(
                (sum(BalanceRole::Storage, |term| term.current)
                    - capacity * (2. + 2. * excess / 3. + excess * excess / 12.))
                    .abs()
                    < 1e-11
            );
            assert!(
                (sum(BalanceRole::Storage, |term| term.increment)
                    - capacity * excess * excess / 12.)
                    .abs()
                    < 1e-11
            );
            assert!(
                (sum(BalanceRole::Dissipation, |term| term.work) - excess * excess).abs() < 1e-11
            );
            assert!(
                (sum(BalanceRole::AppliedVolumeWork, |term| term.work) - (1.5 + excess / 4.)).abs()
                    < 1e-11
            );
            let bytes = trajectory.to_bytes().unwrap();
            let restored = crate::CommonTrajectory::from_bytes(&bytes, &replay).unwrap();
            assert_eq!(
                restored.spatial_steps().unwrap()[0].1.execution,
                first.execution
            );
            let result = crate::CommonResult::accept_trajectory(0., restored.clone()).unwrap();
            assert_eq!(
                result.solve_algorithm(Some(0)),
                Some(plan.linear().algorithm())
            );
            assert_eq!(
                result.solve_true_residual_norm(Some(0)),
                first.residual.map(|value| value.0)
            );
            let mut wrong_plan_metadata = outputs.clone();
            wrong_plan_metadata[0]
                .1
                .execution
                .as_mut()
                .unwrap()
                .solve
                .relative_tolerance *= 2.;
            assert!(
                crate::CommonTrajectory::accept_spatial_transient(run.clone(), wrong_plan_metadata)
                    .unwrap_err()
                    .message()
                    .contains("metadata differs")
            );
            let mut wrong_assembly = outputs.clone();
            wrong_assembly[0]
                .1
                .execution
                .as_mut()
                .unwrap()
                .assembly
                .packet_count += 1;
            assert!(
                crate::CommonTrajectory::accept_spatial_transient(run.clone(), wrong_assembly)
                    .unwrap_err()
                    .message()
                    .contains("assembly metadata")
            );

            assert_eq!(
                restored.spatial_steps().unwrap()[0].1.balance,
                first.balance
            );

            assert_eq!(
                crate::CommonTrajectory::from_bytes(&bytes, &replay)
                    .unwrap()
                    .to_bytes()
                    .unwrap(),
                bytes
            );
            assert_eq!(
                outputs[0].1.previous_state().identity(),
                run.state().identity()
            );
            assert_eq!(
                outputs[2].1.previous_state().identity(),
                outputs[1].1.state().identity()
            );
            // A checkpoint can satisfy all boundary values and have the right
            // timestamp while still violating the driven step equation.
            let one_output = CommonTransientRunRequest::from_steps(
                replay.clone(),
                run.state().clone(),
                3,
                vec![1],
            )
            .unwrap();
            let false_solution = CommonAcceptedSpatialStep::new(
                run.state(),
                plan.scalar_state(0.25, vec![2.; 5]).unwrap(),
            );
            let error = crate::CommonTrajectory::accept_spatial_transient(
                one_output,
                vec![(1, false_solution)],
            )
            .unwrap_err();
            assert!(error.message().contains("residual"));
            let mut forged = outputs.clone();
            let inconsistent = plan.scalar_state(0.25, vec![2.; 5]).unwrap();
            forged[1].1 =
                CommonAcceptedSpatialStep::new(&inconsistent, outputs[1].1.state().clone());
            assert!(
                crate::CommonTrajectory::accept_spatial_transient(run.clone(), forged)
                    .unwrap_err()
                    .message()
                    .contains("predecessor checkpoint")
            );
            let sparse_run = CommonTransientRunRequest::from_steps(
                replay.clone(),
                run.state().clone(),
                3,
                vec![3],
            )
            .unwrap();
            let std::ops::ControlFlow::Continue(sparse) = sparse_run
                .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
                .unwrap()
            else {
                panic!("complete sparse run");
            };
            assert_eq!(
                sparse[0].1.previous_state().identity(),
                outputs[1].1.state().identity()
            );
            let sparse =
                crate::CommonTrajectory::accept_spatial_transient(sparse_run, sparse).unwrap();
            let bytes = sparse.to_bytes().unwrap();
            assert_eq!(
                crate::CommonTrajectory::from_bytes(&bytes, &replay)
                    .unwrap()
                    .to_bytes()
                    .unwrap(),
                bytes
            );
            let restart =
                CommonState::from_bytes(&outputs[0].1.state().to_bytes().unwrap(), &replay)
                    .unwrap();
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
    let initial = resolved
        .as_linear()
        .unwrap()
        .initial_state(0.0, Vec::new())
        .unwrap();
    assert_eq!(initial.linear_values().unwrap(), expected);
    let run = CommonTransientRunRequest::from_steps(resolved, initial, 2, vec![2]).unwrap();
    let std::ops::ControlFlow::Continue(outputs) = run
        .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
        .unwrap()
    else {
        panic!("affine equilibrium must complete");
    };
    for (actual, expected) in outputs[0]
        .1
        .state()
        .linear_values()
        .unwrap()
        .iter()
        .zip(expected)
    {
        assert!((actual - expected).abs() < 1e-12);
    }
}
