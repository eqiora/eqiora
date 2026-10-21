//! Candidate-dependent rate regularity through ordinary State acceptance.
use super::*;

fn source(kind: &str, initial: &str, equation: &str) -> String {
    format!(
        "public model Stored(
        support body:volume(ambient_dimension=1),
        support left:boundary(parent=body),support right:boundary(parent=body)) {{
        state u:{kind} on body in smooth;
        state v:1 on body in smooth; state w:1 on body in smooth;
        initial {{u={initial};v=0;w=0;}}
        relation evolution on body {{{equation}}}
        relation first on body {{v*v+v=0;}}
        relation second on body {{w*w+w=0;}}
    }}"
    )
}

fn resolve(source: &str, step: f64) -> (ResolvedCommonPlan, [eqiora_core::RawId; 3]) {
    coupled::resolve_with_policy(
        source,
        "Stored",
        step,
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
    )
    .unwrap()
}

fn require_singular<T>(result: Result<T, Diagnostic>) {
    let Err(error) = result else {
        panic!("singular source-rate coordinate must reject");
    };
    assert!(
        error
            .message()
            .contains("Jacobian is singular at its consistent rate"),
        "{error:?}"
    );
}

#[test]
fn nonlinear_storage_regularity_covers_initial_accepted_zero_update_and_replay() {
    for (kind, initial_literal, equation, singular_literal, singular_values) in [
        ("1", "1", "1[s]*derivative(u*u)+u=0;", "0", vec![0.]),
        (
            "complex<1>",
            "math.complex(1,1)",
            "1[s]*derivative(math.complex(math.real(u)*math.real(u),math.imag(u)*math.imag(u)))+u=0;",
            "math.complex(0,0)",
            vec![0., 0.],
        ),
        (
            "complex<1>",
            "math.complex(1,1)",
            "1[s]*derivative(math.complex(math.real(u)*math.real(u),math.imag(u)*math.imag(u)))+math.complex(0,math.imag(u))=0;",
            "math.complex(1,0)",
            vec![1., 0.],
        ),
    ] {
        let model = source(kind, initial_literal, equation);
        let (resolved, fields) = resolve(&model, 0.25);
        let resolved = replay_plan(resolved, &REFERENCE_LINEAR_SOLVER);
        let plan = resolved.as_spatial().unwrap();
        let initial = plan.initial_state(Some(0.), Vec::new()).unwrap();
        let before = initial.to_bytes().unwrap();
        let step = plan
            .advance_scalar(&initial, &REFERENCE_LINEAR_SOLVER, 0.25)
            .unwrap();
        let start = plan
            .fields()
            .take_while(|(id, _)| id.erase() != fields[0])
            .map(|(_, ty)| {
                3 * if ty.scalar_domain() == eqiora_core::ScalarDomain::Complex {
                    2
                } else {
                    1
                }
            })
            .sum::<usize>();
        for state in [&initial, step.state()] {
            assert_eq!(
                CommonState::from_bytes(&state.to_bytes().unwrap(), &resolved).unwrap(),
                *state
            );
            let mut singular = state.clone();
            let CommonStateKind::Spatial(values) = &mut singular.kind else {
                unreachable!();
            };
            for node in 0..3 {
                values[start + node * singular_values.len()
                    ..start + (node + 1) * singular_values.len()]
                    .copy_from_slice(&singular_values);
            }
            let values = values.to_vec();
            require_singular(plan.scalar_state(state.time_s(), values));
            require_singular(CommonState::from_bytes(
                &singular.to_bytes().unwrap(),
                &resolved,
            ));
            // R=0 and the BE tangent is nonzero here. Newton's zero-update
            // success must still pass the original source-rate State gate.
            require_singular(plan.advance_scalar(
                &singular,
                &REFERENCE_LINEAR_SOLVER,
                state.time_s().unwrap() + 0.25,
            ));
        }
        assert_eq!(initial.to_bytes().unwrap(), before);
        let (resolved, _) = resolve(&source(kind, singular_literal, equation), 0.25);
        require_singular(
            resolved
                .as_spatial()
                .unwrap()
                .initial_state(Some(0.), Vec::new()),
        );
    }
}

#[test]
fn regular_backward_euler_tangent_does_not_certify_a_singular_state() {
    // At u=1, 2u*u_dot+u(2-u)=0 has the consistent rate -1/2.
    // For dt=2, the BE residual is u(u-1)+u(2-u)=u. Its exact root
    // 0 has a regular step Jacobian, but its mass 2u is zero. Test that
    // candidate directly: a floating-point solve need not land exactly on 0,
    // and exact rank is deliberately not a near-singularity tolerance.
    let model = source("1", "1", "1[s]*derivative(u*u)+u*(2-u)=0;");
    let (resolved, _) = resolve(&model, 2.);
    let plan = resolved.as_spatial().unwrap();
    let initial = plan.initial_state(Some(0.), Vec::new()).unwrap();
    let before = initial.to_bytes().unwrap();
    require_singular(
        plan.scalar_state(Some(2.), vec![0.; initial.spatial_values().unwrap().len()]),
    );
    assert_eq!(initial.to_bytes().unwrap(), before);
}

#[test]
fn candidate_dependent_mass_uses_the_consistent_rate_for_algebraic_partials() {
    // u_dot=1 and v*u_dot-laplacian(v)=0 with Neumann traces give
    // (u,v)=(t,0). At the actual rate 1 the algebraic block is K+M,
    // which is positive definite. At rate zero it would be the singular K.
    // Diffusion anchors the v test space without altering this constant mode.
    let model = source("1", "0", "1[s]*derivative(u)=1;")
        .replace("v*v+v=0;", "1[s]*v*derivative(u)-1[m^2]*div(grad(v))=0;")
        .replace("relation second", "relation left_flux on left {normal(grad(v))=0;} relation right_flux on right {normal(grad(v))=0;} relation second");
    let (resolved, fields) = resolve(&model, 0.25);
    let plan = resolved.as_spatial().unwrap();
    let initial = plan.initial_state(Some(0.), Vec::new()).unwrap();
    let step = plan
        .advance_scalar(&initial, &REFERENCE_LINEAR_SOLVER, 0.25)
        .unwrap();
    for (field, expected) in fields.into_iter().zip([0.25, 0., 0.]) {
        for value in plan
            .state_fields(step.state())
            .unwrap()
            .find(|(id, ..)| id.erase() == field)
            .unwrap()
            .3
        {
            assert!((value - expected).abs() < 1e-11);
        }
    }
    assert_eq!(
        CommonState::from_bytes(&step.state().to_bytes().unwrap(), &resolved).unwrap(),
        *step.state()
    );
}

#[test]
fn large_differential_forcing_cannot_hide_an_algebraic_state_violation() {
    let model = source("1", "1", "1[s]*derivative(u*u)=1e20;");
    let (resolved, _) = resolve(&model, 0.25);
    resolved
        .as_spatial()
        .unwrap()
        .initial_state(Some(0.), Vec::new())
        .unwrap();
    let (invalid, _) = resolve(&model.replace("u=1;v=0;w=0;", "u=1;v=1e-6;w=0;"), 0.25);
    let Err(error) = invalid
        .as_spatial()
        .unwrap()
        .initial_state(Some(0.), Vec::new())
    else {
        panic!("the algebraic equation must retain its own residual scale");
    };
    assert!(
        error.message().contains("equation-local residual"),
        "{error:?}"
    );
}

#[test]
fn one_complex_field_can_have_differential_real_and_algebraic_imaginary_parts() {
    let model = source(
        "complex<1>",
        "math.complex(1,0)",
        "1[s]*derivative(math.complex(math.real(u),0))+math.complex(0,math.imag(u))=0;",
    );
    let (resolved, _) = resolve(&model, 0.25);
    let plan = resolved.as_spatial().unwrap();
    let initial = plan.initial_state(Some(0.), Vec::new()).unwrap();
    let step = plan
        .advance_scalar(&initial, &REFERENCE_LINEAR_SOLVER, 0.25)
        .unwrap();
    assert_eq!(initial.spatial_values(), step.state().spatial_values());
    let (resolved, _) = resolve(
        &model.replace("u=math.complex(1,0)", "u=math.complex(1,1)"),
        0.25,
    );
    let Err(error) = resolved
        .as_spatial()
        .unwrap()
        .initial_state(Some(0.), Vec::new())
    else {
        panic!("imaginary algebraic constraint must reject independently of its real rate");
    };
    assert!(error.message().contains("rate compatibility"), "{error:?}");
}
