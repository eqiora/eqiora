//! Prescribed Time follows the ordinary spatial State/history lifecycle.
use super::*;

fn resolve(
    spatial: CommonSpatialPolicy,
    balance: &str,
    boundary: &str,
    temporal: Option<CommonTemporalPolicy>,
    nonlinear: bool,
) -> Result<ResolvedCommonPlan, Diagnostic> {
    let geometry = cartesian_interval();
    let body = geometry.entity_set("body").unwrap();
    let source = format!(
        "public component Heat(
        support body:volume(ambient_dimension=1),
        support left:boundary(parent=body),support right:boundary(parent=body)) {{
        state u:1 on body in smooth; initial {{u=0;}}
        {balance}
        relation left_value on left {{trace(u)={boundary};}}
        relation right_value on right {{trace(u)={boundary};}}
    }}"
    );
    let model = compile_model(
        "time-data.eqi",
        &source,
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
        &[],
    );
    let linear = exact_reference_linear(
        LinearSolver::BiConjugateGradientStabilized,
        1e-12,
        1e-14,
        NonZeroUsize::new(100).unwrap(),
    );
    let solve = if nonlinear {
        CommonSolvePolicy::Newton {
            linear,
            nonlinear: NonlinearSolvePlan::new(0., 1e-12, NonZeroUsize::new(16).unwrap(), 16)
                .unwrap(),
        }
    } else {
        CommonSolvePolicy::Linear(linear)
    };
    ResolvedCommonPlan::resolve(
        &model,
        cartesian_box_resources(&geometry, &[2]),
        spatial,
        solve,
        None,
        temporal,
        &REFERENCE_LINEAR_SOLVER,
        None,
        None,
    )
}

#[test]
fn timed_volume_and_trace_use_endpoint_history_through_replay_and_restart() {
    for balance in [
        "law balance on body {storage 1[s/m^2]*u; flux -grad(u); source 2*time()/1[s*m^2];}",
        "relation balance on body {1[s/m^2]*derivative(u)-div(grad(u))=2*time()/1[s*m^2];}",
    ] {
        for nonlinear in [false, true] {
            let resolved = resolve(
                CommonSpatialPolicy::Q1,
                balance,
                "time()*time()/1[s^2]",
                Some(CommonTemporalPolicy::backward_euler(0.25).unwrap()),
                nonlinear,
            )
            .unwrap();
            let resolved = replay_plan(resolved, &REFERENCE_LINEAR_SOLVER);
            let plan = resolved.as_spatial().unwrap();
            let initial = plan.initial_state(Some(0.), Vec::new()).unwrap();
            let before = initial.to_bytes().unwrap();
            let mut state = initial.clone();
            // Q1 integrals: M_ii=1/3, each M_if=1/12, K_ii=4,
            // each K_if=-2, F_i=t. The boundary mass uses endpoint secants.
            // At h=1/4 the independent one-row BE recurrence gives:
            for (n, midpoint) in [11. / 128., 143. / 512., 1215. / 2048.]
                .into_iter()
                .enumerate()
            {
                let t = (n + 1) as f64 / 4.;
                state = plan
                    .advance_scalar(&state, &REFERENCE_LINEAR_SOLVER)
                    .unwrap()
                    .state()
                    .clone();
                for (&actual, expected) in
                    state
                        .spatial_values()
                        .unwrap()
                        .iter()
                        .zip([t * t, midpoint, t * t])
                {
                    assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
                }
                state = CommonState::from_bytes(&state.to_bytes().unwrap(), &resolved).unwrap();
            }
            assert_eq!(initial.to_bytes().unwrap(), before);
            assert!(
                resolve(
                    CommonSpatialPolicy::Q1,
                    balance,
                    "time()*time()/1[s^2]",
                    None,
                    nonlinear
                )
                .unwrap_err()
                .message()
                .contains("coefficient Time requires an explicit finite evaluation point")
            );
        }
    }
}

#[test]
fn timed_capacity_is_certified_at_the_accepted_state_time() {
    let resolved = resolve(
        CommonSpatialPolicy::Q1,
        "relation balance on body {(1-time()/1[s])*1[s/m^2]*derivative(u)-div(grad(u))=0;}",
        "0",
        Some(CommonTemporalPolicy::backward_euler(1.).unwrap()),
        false,
    )
    .unwrap();
    let plan = resolved.as_spatial().unwrap();
    let initial = plan.initial_state(Some(0.), Vec::new()).unwrap();
    // R=0 and the BE stiffness is regular at t=1, but the original mass is
    // zero. A certificate sampled at t=0 would incorrectly accept this State.
    let error = plan
        .advance_scalar(&initial, &REFERENCE_LINEAR_SOLVER)
        .unwrap_err();
    assert!(
        error
            .message()
            .contains("Jacobian is singular at its consistent rate"),
        "{error:?}"
    );
}

#[test]
fn explicit_time_in_a_singular_algebraic_block_requires_its_constraint_tangent() {
    // With homogeneous Neumann traces, constants are the nullspace of K.
    // M*u_dot + K*u + M*v = 0 and K*v + M*(u-t) = 0 therefore
    // require u=t and v=-1. Differentiating the second row supplies -M*1;
    // omitting this explicit Time partial incorrectly accepts v=0 instead.
    let source = r#"public model DrivenConstraint(
        support body:volume(ambient_dimension=1),
        support left:boundary(parent=body),support right:boundary(parent=body)) {
        state u:1 on body in smooth;
        state v:1 on body in smooth;
        state w:1 on body in smooth;
        initial {u=0;v=-1;w=0;}
        law evolution on body {storage 1[s/m^2]*u; flux -grad(u); source -v*1[1/m^2];}
        law constraint on body {flux -grad(v); source (time()/1[s]-u)*1[1/m^2];}
        law regular on body {flux -grad(w); source -w*1[1/m^2];}
        relation left_flux on left {normal(grad(u))=0;normal(grad(v))=0;normal(grad(w))=0;}
        relation right_flux on right {normal(grad(u))=0;normal(grad(v))=0;normal(grad(w))=0;}
    }"#;
    let (resolved, fields) = coupled::resolve(source, "DrivenConstraint").unwrap();
    let plan = resolved.as_spatial().unwrap();
    let initial = plan.initial_state(Some(0.), Vec::new()).unwrap();
    let at = |time: f64, multiplier: f64| {
        plan.fields()
            .flat_map(|(field, _)| {
                let value = if field.erase() == fields[0] {
                    time
                } else if field.erase() == fields[1] {
                    multiplier
                } else {
                    0.
                };
                [value; 3]
            })
            .collect::<Vec<_>>()
    };
    for time in [0., 1.] {
        plan.scalar_state(Some(time), at(time, -1.)).unwrap();
        let error = plan.scalar_state(Some(time), at(time, 0.)).unwrap_err();
        assert!(error.message().contains("rate compatibility"), "{error:?}");
    }
    let mut state = initial;
    for index in 1..=3 {
        let time = index as f64 * 0.25;
        let step = plan
            .advance_scalar(&state, &REFERENCE_LINEAR_SOLVER)
            .unwrap();
        for (&actual, expected) in step
            .state()
            .spatial_values()
            .unwrap()
            .iter()
            .zip(at(time, -1.))
        {
            assert!((actual - expected).abs() < 1e-10, "{actual} != {expected}");
        }
        state = CommonState::from_bytes(&step.state().to_bytes().unwrap(), &resolved).unwrap();
    }
}

#[test]
fn spatial_checkpoint_rejects_a_forged_step_time_even_at_equilibrium() {
    let resolved = resolve(
        CommonSpatialPolicy::Q1,
        "law balance on body {storage 1[s/m^2]*u; flux -grad(u); source 0[1/m^2];}",
        "0",
        Some(CommonTemporalPolicy::backward_euler(0.25).unwrap()),
        false,
    )
    .unwrap();
    let plan = resolved.as_spatial().unwrap();
    let initial = plan.initial_state(Some(0.), Vec::new()).unwrap();
    let accepted = plan
        .advance_scalar(&initial, &REFERENCE_LINEAR_SOLVER)
        .unwrap();
    let CommonStateKind::Spatial {
        values,
        previous: Some(previous),
    } = &accepted.state().kind
    else {
        unreachable!();
    };
    let mut changed = previous.clone();
    changed.grid.origin_s = 0.125;
    assert!(
        plan.spatial_state(Some(0.25), values.to_vec(), Some(changed))
            .is_err(),
        "zero residual cannot authenticate a predecessor at the wrong Plan step Time"
    );
    let mut changed = previous.clone();
    changed.grid.index = 0;
    assert!(
        plan.spatial_state(Some(0.25), values.to_vec(), Some(changed))
            .unwrap_err()
            .message()
            .contains("preceding Time grid point")
    );
    let mut changed = previous.clone();
    changed.values[1] = 1.; // Interior change preserves every essential trace.
    assert!(
        plan.spatial_state(Some(0.25), values.to_vec(), Some(changed))
            .unwrap_err()
            .message()
            .contains("residual")
    );
    let mut changed = accepted.state().clone();
    let CommonStateKind::Spatial {
        previous: Some(history),
        ..
    } = &mut changed.kind
    else {
        unreachable!();
    };
    history.grid.origin_s = 0.125;
    let error = CommonState::from_bytes(&changed.to_bytes().unwrap(), &resolved).unwrap_err();
    assert!(
        error
            .message()
            .contains("Time differs from its exact Plan time grid"),
        "{error:?}"
    );
}

#[test]
fn spatial_checkpoint_preserves_the_original_decimal_time_grid_on_restart() {
    let resolved = resolve(
        CommonSpatialPolicy::Q1,
        "law balance on body {storage 1[s/m^2]*u; flux -grad(u); source 0[1/m^2];}",
        "0",
        Some(CommonTemporalPolicy::backward_euler(0.1).unwrap()),
        false,
    )
    .unwrap();
    let plan = resolved.as_spatial().unwrap();
    let initial = plan.initial_state(Some(0.), Vec::new()).unwrap();
    let run =
        CommonTransientRunRequest::from_steps(resolved.clone(), initial, 10, (1..=10).collect())
            .unwrap();
    let std::ops::ControlFlow::Continue(outputs) = run
        .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
        .unwrap()
    else {
        panic!("complete run");
    };
    let restart =
        CommonState::from_bytes(&outputs[2].1.state().to_bytes().unwrap(), &resolved).unwrap();
    let resumed = CommonTransientRunRequest::from_steps(
        resolved.clone(),
        restart.clone(),
        7,
        (1..=7).collect(),
    )
    .unwrap();
    let by_times = CommonTransientRunRequest::from_times(
        resolved.clone(),
        restart,
        1.,
        (4..=10).map(|index| index as f64 * 0.1).collect(),
    )
    .unwrap();
    assert_eq!(resumed.identity(), by_times.identity());
    let std::ops::ControlFlow::Continue(repeated) = resumed
        .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
        .unwrap()
    else {
        panic!("complete restart");
    };
    for ((_, expected), (_, actual)) in outputs[3..].iter().zip(&repeated) {
        assert_eq!(actual.state().time_s(), expected.state().time_s());
        assert_eq!(
            actual.state().to_bytes().unwrap(),
            expected.state().to_bytes().unwrap()
        );
    }
    let trajectory = crate::CommonTrajectory::accept_spatial_transient(resumed, repeated).unwrap();
    let bytes = trajectory.to_bytes().unwrap();
    assert_eq!(
        crate::CommonTrajectory::from_bytes(&bytes, &resolved)
            .unwrap()
            .to_bytes()
            .unwrap(),
        bytes
    );
}

#[test]
fn tpfa_time_inputs_use_native_faces_and_common_state_history() {
    for balance in [
        "law balance on body {storage 1[s/m^2]*u; flux -grad(u); source 2*time()/1[s*m^2];}",
        "relation balance on body {1[s/m^2]*derivative(u)-div(grad(u))=2*time()/1[s*m^2];}",
    ] {
        for nonlinear in [false, true] {
            let resolved = resolve(
                CommonSpatialPolicy::CellCenteredTpfa,
                balance,
                "time()*time()/1[s^2]",
                Some(CommonTemporalPolicy::backward_euler(0.25).unwrap()),
                nonlinear,
            )
            .unwrap();
            let resolved = replay_plan(resolved, &REFERENCE_LINEAR_SOLVER);
            let plan = resolved.as_spatial().unwrap();
            let mut state = plan.initial_state(Some(0.), Vec::new()).unwrap();
            // Each cell has mass 1/2 and a half-cell boundary distance 1/4.
            // Equal cell values obey 6*u_n=2*u_old+4*t_n²+t_n.
            for expected in [1. / 12., 5. / 18., 16. / 27.] {
                let accepted = plan
                    .advance_scalar(&state, &REFERENCE_LINEAR_SOLVER)
                    .unwrap();
                let values = accepted.state().spatial_values().unwrap();
                assert_eq!(values.len(), 2);
                for actual in values {
                    assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
                }
                state = CommonState::from_bytes(&accepted.state().to_bytes().unwrap(), &resolved)
                    .unwrap();
            }
        }
    }
}

#[test]
fn tpfa_polynomial_volume_uses_the_declared_quadrature() {
    let resolved = resolve(
        CommonSpatialPolicy::CellCenteredTpfa,
        "law balance on body {storage 1[s/m^2]*u; flux -grad(u); source coordinate(0)^2/1[m^4];}",
        "0",
        Some(CommonTemporalPolicy::backward_euler(0.25).unwrap()),
        false,
    )
    .unwrap();
    let resolved = replay_plan(resolved, &REFERENCE_LINEAR_SOLVER);
    let plan = resolved.as_spatial().unwrap();
    assert_eq!(
        plan.portable_realization().domains()[0]
            .discretization()
            .quadrature(),
        eqiora_realization::QuadraturePolicy::GaussLegendre {
            points_per_axis: NonZeroUsize::new(2).unwrap()
        }
    );
    let initial = plan.initial_state(Some(0.), Vec::new()).unwrap();
    let step = plan
        .advance_scalar(&initial, &REFERENCE_LINEAR_SOLVER)
        .unwrap();
    // Each cell has M/dt=2; native faces give K=[[6,-2],[-2,6]].
    // Integrating x² gives loads (1/24,7/24), hence (11,29)/720.
    let values = step.state().spatial_values().unwrap();
    assert_eq!(values.len(), 2);
    for (&actual, expected) in values.iter().zip([11. / 720., 29. / 720.]) {
        assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
    }
    CommonState::from_bytes(&step.state().to_bytes().unwrap(), &resolved).unwrap();
}

#[test]
fn stationary_tpfa_newton_uses_cell_initial_state_and_common_result_replay() {
    let resolved = resolve(
        CommonSpatialPolicy::CellCenteredTpfa,
        "relation balance on body {-div(grad(u))+(u+u*u*u-2)/1[m^2]=0;}",
        "1",
        None,
        true,
    )
    .unwrap();
    let resolved = replay_plan(resolved, &REFERENCE_LINEAR_SOLVER);
    let plan = resolved.as_spatial().unwrap();
    let assignments = plan
        .fields()
        .map(|(field, ty)| {
            CommonInitialField::new(
                resolved.model_artifact().digest().unwrap(),
                field,
                None,
                Some(CommonInitialValues::new(ty.shape().clone(), vec![0.; 2]).unwrap()),
            )
            .unwrap()
        })
        .collect();
    let initial = plan.initial_state(None, assignments).unwrap();
    let before = initial.to_bytes().unwrap();
    let result = plan
        .run_result(Some(&initial), &REFERENCE_LINEAR_SOLVER)
        .unwrap();
    assert!(result.nonlinear_iterations(None).unwrap() > 0);
    assert_eq!(before, initial.to_bytes().unwrap());
    let (_, values, _) = result.field_block(0, 0).unwrap();
    assert_eq!(values.len(), 2);
    // The constant u=1 has zero face flux and u+u³-2=0.
    for &value in values {
        assert!((value - 1.).abs() < 1e-12);
    }
    let bytes = result.to_bytes().unwrap();
    assert_eq!(
        crate::CommonResult::from_bytes(&bytes, &resolved).unwrap(),
        result
    );
}
