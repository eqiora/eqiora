use super::*;

mod box_case;

const SOURCE: &str = r#"
public component Wave(
    support body: volume(ambient_dimension = 1),
    support left: boundary(parent = body),
    support right: boundary(parent = body)
) {
    coordinate x: m on body from body[0];
    state u: 1 on body in h1;
    initial { u = 1 + x / 1[m]; }
    law balance on body {
        storage 3[s/m^2] * u;
        flux -grad(u) + 2[1/m] * u * grad(coordinate(0));
        source 2[1/m^2];
    }
    relation left_value on left { trace(u) = 1; }
    relation right_value on right { trace(u) = 2; }
    form weak_transport for balance {
        test w: 1 for u zero_on left, right;
        integrate(body, w * 3[s/m^2] * derivative(u))
            + integrate(body, dot(grad(w), grad(u)))
            - integrate(body, dot(grad(w), 2[1/m] * u * grad(coordinate(0))))
            = integrate(body, w * 2[1/m^2]);
    }
}
"#;

fn plan(source: &str) -> Result<ResolvedCommonPlan, Diagnostic> {
    let geometry = cartesian_interval();
    let (program, projection) = compile(source, &geometry)?;
    let model = ModelEnvelope::from_program(&program).unwrap();
    ResolvedCommonPlan::resolve(
        &model,
        cartesian_box_resources(&geometry, &[3]),
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

fn first_values(source: &str) -> Vec<f64> {
    let resolved = replay_plan(plan(source).unwrap(), &REFERENCE_LINEAR_SOLVER);
    let initial = resolved.as_linear().unwrap().initial_state().unwrap();
    let run =
        CommonTransientRunRequest::from_steps(resolved.clone(), initial, 2, vec![1, 2]).unwrap();
    let std::ops::ControlFlow::Continue(outputs) = run
        .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
        .unwrap()
    else {
        panic!("transport step must complete");
    };
    assert_eq!(outputs.len(), 2);
    let state = &outputs[0].1;
    assert_eq!(
        CommonState::from_bytes(&state.to_bytes().unwrap(), &resolved).unwrap(),
        *state
    );
    let restart = CommonState::from_bytes(&state.to_bytes().unwrap(), &resolved).unwrap();
    let resumed = CommonTransientRunRequest::from_steps(resolved, restart, 1, vec![1]).unwrap();
    let std::ops::ControlFlow::Continue(restarted) = resumed
        .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
        .unwrap()
    else {
        panic!("transport restart must complete");
    };
    assert_eq!(restarted[0].1, outputs[1].1);
    state.scalar_values().unwrap().to_vec()
}

#[test]
fn conservative_transport_retains_direction_and_replays_accepted_steps() {
    // u=1+x, k=1, b=2 gives flux=1+2x and source=2 independently.
    // For three cells and c=3, dt=1/4, the free matrix is
    // [[26/3, -7/3+b/2],[-7/3-b/2,26/3]]. Reversing b alone gives
    // forcing defect [4/3,4/3], hence first-step excess [12/53,10/53].
    for (velocity, expected) in [
        ("2[1/m]", [4. / 3., 5. / 3.]),
        ("-2[1/m]", [4. / 3. + 12. / 53., 5. / 3. + 10. / 53.]),
        ("0[1/m]", [4. / 3. + 2. / 19., 5. / 3. + 2. / 19.]),
    ] {
        let values = first_values(&SOURCE.replace("2[1/m]", velocity));
        assert_eq!(values.len(), 4);
        assert_eq!([values[0], values[3]], [1., 2.]);
        for (actual, expected) in values[1..3].iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-11, "{values:?}");
        }
    }
}

#[test]
fn varying_transport_velocity_retains_the_density_divergence() {
    // b=2x and u=1+x: div(b*u-grad(u))=2+4x, not merely b*grad(u)=2x.
    let source = SOURCE
        .replace(
            "source 2[1/m^2];",
            "source 2[1/m^2] + 4[1/m^3] * coordinate(0);",
        )
        .replace("w * 2[1/m^2]", "w * (2[1/m^2] + 4[1/m^3] * coordinate(0))")
        .replace("2[1/m]", "2[1/m^2] * coordinate(0)");
    let values = first_values(&source);
    for (actual, expected) in values.iter().zip([1., 4. / 3., 5. / 3., 2.]) {
        assert!((actual - expected).abs() < 1e-11, "{values:?}");
    }
}

#[test]
fn authored_transport_rejects_a_missing_or_reversed_velocity() {
    for velocity in ["0[1/m]", "-2[1/m]"] {
        let source = SOURCE.replace("dot(grad(w), 2[1/m]", &format!("dot(grad(w), {velocity}"));
        let error = plan(&source).unwrap_err();
        assert!(
            error.message().contains("left bilinear term"),
            "{}",
            error.message()
        );
    }
}

#[test]
fn diffusion_only_callback_rejects_steady_transport() {
    let source = SOURCE
        .replace("    initial { u = 1 + x / 1[m]; }\n", "")
        .replace("        storage 3[s/m^2] * u;\n", "")
        .replace(
            "integrate(body, w * 3[s/m^2] * derivative(u))\n            + ",
            "",
        );
    let (program, _) = compile(&source, &cartesian_interval()).unwrap();
    let domain = program
        .nodes()
        .find_map(|node| match node {
            eqiora_schema::kernel::KernelNode::Domain(value)
                if matches!(
                    value.kind(),
                    eqiora_schema::kernel::DomainKind::GeometryRegion { .. }
                ) =>
            {
                Some(value.id().erase())
            }
            _ => None,
        })
        .unwrap();
    let form = crate::form_compiler::derive_candidate_with_dimension(&program, domain, 1)
        .unwrap()
        .unwrap();
    let error = form
        .admit_quadrature(&eqiora_meshing::QuadratureRule::gauss_legendre(2).unwrap())
        .err()
        .unwrap();
    assert!(
        error.message().contains("diffusion/load callback adapter"),
        "{}",
        error.message()
    );
}
