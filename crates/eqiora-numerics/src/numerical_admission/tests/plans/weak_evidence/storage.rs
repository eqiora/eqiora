use super::*;

const SOURCE: &str = r#"
public component Wave(
    support body: volume(ambient_dimension = 1),
    support left: boundary(parent = body),
    support right: boundary(parent = body)
) {
    state u: 1 on body in h1;
    initial { u = 2; }
    law balance on body { storage 3[s/m^2] * u; flux -grad(u); source 4[1/m^2]; }
    relation left_value on left { trace(u) = 2; }
    relation right_value on right { trace(u) = 2; }
    form weak_storage for balance {
        test w: 1 for u zero_on left, right;
        integrate(body, w * 3[s/m^2] * derivative(u))
            + integrate(body, dot(grad(w), grad(u)))
            = integrate(body, w * 4[1/m^2]);
    }
}
"#;

fn storage_plan(source: &str) -> Result<ResolvedCommonPlan, Diagnostic> {
    let geometry = cartesian_interval();
    let (program, projection) = compile(source, &geometry)?;
    let model = ModelEnvelope::from_program(&program).unwrap();
    ResolvedCommonPlan::resolve(
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
        Some(&projection),
    )
}

#[test]
fn authored_storage_pairing_reaches_accepted_scalar_steps() {
    let resolved = replay_plan(storage_plan(SOURCE).unwrap(), &REFERENCE_LINEAR_SOLVER);
    let initial = resolved
        .as_linear()
        .unwrap()
        .initial_state(0.0, Vec::new())
        .unwrap();
    let run =
        CommonTransientRunRequest::from_steps(resolved.clone(), initial, 3, vec![1, 2, 3]).unwrap();
    let std::ops::ControlFlow::Continue(outputs) = run
        .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
        .unwrap()
    else {
        panic!("authored storage must reach all accepted steps");
    };
    // Two half-length elements: M_ii=1, K_ii=4, F_i=2.
    // dt=1/4 and boundary 2 give e_next=e/2+1/4.
    for ((_, state), expected) in outputs.iter().zip([2.25, 2.375, 2.4375]) {
        assert_eq!(
            CommonState::from_bytes(&state.to_bytes().unwrap(), &resolved).unwrap(),
            *state
        );
        let values = state.linear_values().unwrap();
        assert_eq!(values[0], 2.);
        assert_eq!(values[2], 2.);
        assert!((values[1] - expected).abs() < 1e-12);
    }
    let restart = CommonState::from_bytes(&outputs[0].1.to_bytes().unwrap(), &resolved).unwrap();
    let run = CommonTransientRunRequest::from_steps(resolved, restart, 2, vec![2]).unwrap();
    let std::ops::ControlFlow::Continue(restarted) = run
        .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
        .unwrap()
    else {
        panic!("authored storage restart must complete");
    };
    assert_eq!(restarted[0].1, outputs[2].1);
}

#[test]
fn authored_storage_rejects_missing_rate_wrong_capacity_and_state_substitution() {
    for replacement in [
        "0[1/m^2]",
        "w * 6[s/m^2] * derivative(u)",
        "w * 3[s/m^2] * (u / 1[s])",
    ] {
        let source = SOURCE.replace("w * 3[s/m^2] * derivative(u)", replacement);
        let error = storage_plan(&source).unwrap_err();
        assert!(
            error.message().contains("left bilinear term"),
            "{}",
            error.message()
        );
    }
    let error = storage_plan(&SOURCE.replace("derivative(u)", "derivative(w)")).unwrap_err();
    assert!(
        error.message().contains("scalar State Field"),
        "{}",
        error.message()
    );
}

#[test]
fn authored_storage_projection_replays_only_the_current_epoch() {
    let (_, projection) = compile(SOURCE, &cartesian_interval()).unwrap();
    assert_eq!(
        AuthoredFormulationProjection::decode(projection.canonical_bytes()).unwrap(),
        projection
    );
    let old = String::from_utf8(projection.canonical_bytes().to_vec())
        .unwrap()
        .replace("eqiora.authored-form/v16", "eqiora.authored-form/v15");
    assert!(AuthoredFormulationProjection::decode(old.as_bytes()).is_err());
}
