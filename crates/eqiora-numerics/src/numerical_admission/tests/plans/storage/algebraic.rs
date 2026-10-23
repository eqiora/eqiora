use super::*;

const SOURCE: &str = r#"
public model Constrained(
    support body: volume(ambient_dimension = 1),
    support left: boundary(parent = body),
    support right: boundary(parent = body)
) {
    coordinate x: m on body from body[0];
    state u: 1 on body in h1;
    state v: 1 on body in h1;
    state w: 1 on body in h1;
    initial {
        u = 4 * x * (1[m] - x) / 1[m^2];
        v = (13/168) * 4 * x * (1[m] - x) / 1[m^2];
        w = (1/168) * 4 * x * (1[m] - x) / 1[m^2];
    }
    law evolution on body { storage 1[s/m^2]*u; flux -grad(u); source -v * 1[1/m^2]; }
    law constraint_v on body { flux -grad(v); source (u+w-v) * 1[1/m^2]; }
    law constraint_w on body { flux -grad(w); source (v-w) * 1[1/m^2]; }
    relation left_values on left { trace(u)=0; trace(v)=0; trace(w)=0; }
    relation right_values on right { trace(u)=0; trace(v)=0; trace(w)=0; }
}
"#;

fn complex_source(source: &str) -> String {
    source
        .replace(": 1 on body", ": complex<1> on body")
        .replace("4 * x", "math.complex(1,2) * 4 * x")
}

#[test]
fn mixed_storage_preserves_algebraic_feedback_and_restart() {
    // Two half-length elements give M_ii=1/3, K_ii=4.
    // Algebraic rows: 13v=u+w, 13w=v, hence v=13u/168 and w=u/168.
    // Evolution: u' = -(12+13/168)u = -2029u/168.
    // Backward Euler with dt=1/4 multiplies u by 672/2701 each step.
    for source in [
        SOURCE.to_owned(),
        SOURCE.replace("Constrained", "Renamed"),
        complex_source(SOURCE),
    ] {
        let parts = if source.contains("complex<1>") { 2 } else { 1 };
        let entry = if source.contains("model Renamed") {
            "Renamed"
        } else {
            "Constrained"
        };
        let (resolved, fields) = coupled::resolve(&source, entry).unwrap();
        let replay = replay_plan(resolved.clone(), &REFERENCE_LINEAR_SOLVER);
        let plan = resolved.as_spatial().unwrap();
        let interior = fields.map(|field| {
            parts
                * (3 * plan
                    .fields()
                    .position(|(id, _)| id.erase() == field)
                    .unwrap()
                    + 1)
        });
        let initial = plan.initial_state(Some(0.0), Vec::new()).unwrap();
        for part in 0..parts {
            let mut invalid = initial.clone();
            let CommonStateKind::Spatial { values, .. } = &mut invalid.kind else {
                unreachable!()
            };
            values[interior[1] + part] += 0.001;
            let error = CommonState::from_bytes(&invalid.to_bytes().unwrap(), &replay).unwrap_err();
            assert!(error.message().contains("algebraic"), "{}", error.message());
        }
        let before = initial.to_bytes().unwrap();
        let run =
            CommonTransientRunRequest::from_steps(resolved, initial.clone(), 3, vec![1, 2, 3])
                .unwrap();
        let std::ops::ControlFlow::Continue(outputs) = run
            .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
            .unwrap()
        else {
            panic!("complete run")
        };
        for (step, (_, state)) in (1..=3).zip(&outputs) {
            let state = state.state();
            let u = (672.0_f64 / 2701.0).powi(step);
            for (index, expected) in interior.into_iter().zip([u, 13.0 * u / 168.0, u / 168.0]) {
                for part in 0..parts {
                    let component = if part == 0 { expected } else { 2. * expected };
                    assert!(
                        (state.spatial_values().unwrap()[index + part] - component).abs() < 1e-11
                    );
                }
            }
        }
        assert_eq!(before, initial.to_bytes().unwrap());
        let restarted =
            CommonState::from_bytes(&outputs[0].1.state().to_bytes().unwrap(), &replay).unwrap();
        let run = CommonTransientRunRequest::from_steps(replay, restarted, 2, vec![2]).unwrap();
        let std::ops::ControlFlow::Continue(restarted) = run
            .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
            .unwrap()
        else {
            panic!("complete restart")
        };
        assert_eq!(restarted.last().unwrap().1, outputs[2].1);
    }
}

#[test]
fn mixed_storage_rejects_inconsistent_algebraic_initial_state() {
    let imaginary_error = complex_source(SOURCE).replace(
        "v = (13/168) * math.complex(1,2)",
        "v = math.complex(13/168,28/168)",
    );
    for source in [
        SOURCE.replace("v = (13/168)", "v = (14/168)"),
        imaginary_error,
    ] {
        for step in [0.25, 1e-12] {
            let (resolved, _) = coupled::resolve_with_step(&source, "Constrained", step).unwrap();
            let error = resolved
                .as_spatial()
                .unwrap()
                .initial_state(Some(0.0), Vec::new())
                .unwrap_err();
            assert!(error.message().contains("algebraic"), "{}", error.message());
        }
    }
}

#[test]
fn vector_storage_and_algebraic_field_preserve_affine_equilibrium() {
    let mut source = String::from(
        "model VectorConstraint() {
        domain body=box(0,1,0,1,0,1);
        variable g:m on body in smooth;
        relation potential on body {g=2*coordinate(0)+3*coordinate(1)+5*coordinate(2);}
        state u:vector<1,3> on body in smooth;
        state v:vector<1,3> on body in smooth;
        initial {u=grad(g); v=grad(g);}
        relation evolution on body {1[s/m^2]*derivative(u)=div(grad(u))+(v-u)*1[1/m^2];}
        relation constraint on body {-div(grad(v))+(v-u)*1[1/m^2]=0;}",
    );
    for (axis, label) in ["x", "y", "z"].into_iter().enumerate() {
        for side in ["lower", "upper"] {
            source += &format!("domain {label}_{side}=boundary(body,axis={axis},side={side});
                relation fixed_{label}_{side} on {label}_{side} {{trace(u)=trace(grad(g)); trace(v)=trace(grad(g));}}");
        }
    }
    source += "}";
    let (transaction, model, _) = eqiora_compiler::compile("vector-constraint.eqi", &source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let resolved = ResolvedCommonPlan::resolve(
        &ModelEnvelope::from_program(&program).unwrap(),
        cartesian_box_resources(&cartesian_box_3d(), &[2, 2, 2]),
        CommonSpatialPolicy::Q1,
        CommonSolvePolicy::Linear(exact_reference_linear(
            LinearSolver::BiConjugateGradientStabilized,
            1e-12,
            1e-14,
            NonZeroUsize::new(100).unwrap(),
        )),
        None,
        Some(CommonTemporalPolicy::backward_euler(0.25).unwrap()),
        &REFERENCE_LINEAR_SOLVER,
        None,
        None,
    )
    .unwrap();
    let resolved = replay_plan(resolved, &REFERENCE_LINEAR_SOLVER);
    let initial = resolved
        .as_spatial()
        .unwrap()
        .initial_state(Some(0.), Vec::new())
        .unwrap();
    // Constant vectors have zero Laplacian and u=v cancels both reactions.
    // Every node/component therefore retains the independently prescribed (2,3,5).
    let expected = (0..54).flat_map(|_| [2., 3., 5.]).collect::<Vec<_>>();
    assert_eq!(initial.spatial_values().unwrap(), expected);
    let run =
        CommonTransientRunRequest::from_steps(resolved.clone(), initial, 3, vec![1, 2, 3]).unwrap();
    let std::ops::ControlFlow::Continue(outputs) = run
        .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
        .unwrap()
    else {
        panic!("complete vector run")
    };
    for (_, state) in outputs {
        let state = state.state();
        for (actual, expected) in state.spatial_values().unwrap().iter().zip(&expected) {
            assert!((actual - expected).abs() < 1e-11);
        }
        assert_eq!(
            CommonState::from_bytes(&state.to_bytes().unwrap(), &resolved).unwrap(),
            *state
        );
    }
}

#[test]
fn algebraic_snapshot_allows_zero_state_partial_of_differential_equations() {
    // u_dot=0 has a zero State partial. The two algebraic equations still
    // give v=13u/168 and w=u/168 on this two-element constrained mesh.
    let source = SOURCE
        .replace(
            "law evolution on body { storage 1[s/m^2]*u; flux -grad(u); source -v * 1[1/m^2]; }",
            "relation evolution on body { 1[s/m^2]*derivative(u)=0; }",
        )
        .replace("trace(u)=0; ", "");
    // The pointwise evolution has no spatial flux, so only v and w require
    // boundary laws. u's endpoint values are initial data, not extra traces.
    for source in [source.clone(), complex_source(&source)] {
        let (resolved, _) = coupled::resolve(&source, "Constrained").unwrap();
        let resolved = replay_plan(resolved, &REFERENCE_LINEAR_SOLVER);
        let initial = resolved
            .as_spatial()
            .unwrap()
            .initial_state(Some(0.), Vec::new())
            .unwrap();
        let expected = initial.spatial_values().unwrap().to_vec();
        let initial = CommonState::from_bytes(&initial.to_bytes().unwrap(), &resolved).unwrap();
        let run = CommonTransientRunRequest::from_steps(resolved.clone(), initial, 2, vec![1, 2])
            .unwrap();
        let std::ops::ControlFlow::Continue(outputs) = run
            .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
            .unwrap()
        else {
            panic!("complete run");
        };
        for (_, step) in outputs {
            for (actual, expected) in step.state().spatial_values().unwrap().iter().zip(&expected) {
                assert!((actual - expected).abs() < 1e-11);
            }
            assert_eq!(
                CommonState::from_bytes(&step.state().to_bytes().unwrap(), &resolved).unwrap(),
                *step.state()
            );
        }
    }
}

#[test]
fn initial_state_requires_regular_algebraic_partition_even_at_zero_residual() {
    for regular in [true, false] {
        // At v=w=0 the algebraic Jacobian is diag(alpha*M, M), where
        // M is the positive two-element Q1 mass matrix. A zero residual
        // therefore represents an index-one point only when alpha != 0.
        let alpha = if regular { 1 } else { 0 };
        let source = format!(
            "public model Constrained(
            support body:volume(ambient_dimension=1),
            support left:boundary(parent=body),support right:boundary(parent=body)) {{
            state u:1 on body in h1; state v:1 on body in h1; state w:1 on body in h1;
            initial {{u=1;v=0;w=0;}}
            relation evolution on body {{1[s]*derivative(u)=0;}}
            relation first on body {{v*v+{alpha}*v=0;}}
            relation second on body {{w*w+w=0;}}
        }}"
        );
        let (resolved, _) = resolve_nonlinear_constraint(&source);
        let initial = resolved
            .as_spatial()
            .unwrap()
            .initial_state(Some(0.), Vec::new());
        if regular {
            initial.unwrap();
        } else {
            let error = initial.expect_err("singular algebraic point must reject");
            assert!(error.message().contains("regularity"), "{error:?}");
        }
    }
}

fn resolve_nonlinear_constraint(source: &str) -> (ResolvedCommonPlan, [eqiora_core::RawId; 3]) {
    coupled::resolve_with_policy(
        source,
        "Constrained",
        0.25,
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

#[test]
fn algebraic_regularity_covers_complex_parts_accepted_states_and_replay() {
    // f(x)=x²(x-1) has roots 0 and 1, but f'(0)=0 and f'(1)=1.
    // The second case keeps Re(v) regular and makes only Im(v) singular.
    // The last case has algebraic blocks [[1,1],[1,2]] at (-1,1) and
    // [[1,1],[1,1]] at (0,0), tensor M. Every singular row is nonempty.
    for (scalar, initial_literal, constraint, singular_literal, singular_values) in [
        ("1", "1", "v*v*(v-1)", "0", [0., 0.]),
        (
            "complex<1>",
            "math.complex(1,1)",
            "math.complex(math.real(v)*math.real(v)*(math.real(v)-1),math.imag(v)*math.imag(v)*(math.imag(v)-1))",
            "math.complex(1,0)",
            [1., 0.],
        ),
        (
            "complex<1>",
            "math.complex(-1,1)",
            "math.complex(math.real(v)+math.imag(v),math.real(v)+math.imag(v)+math.imag(v)*math.imag(v)*(math.imag(v)-1))",
            "0",
            [0., 0.],
        ),
    ] {
        let source = format!(
            "public model Constrained(
                support body:volume(ambient_dimension=1),
                support left:boundary(parent=body),support right:boundary(parent=body)) {{
                state u:{scalar} on body in h1;
                state v:{scalar} on body in h1;
                state w:{scalar} on body in h1;
                initial {{u=1;v={initial_literal};w=0;}}
                relation evolution on body {{1[s]*derivative(u)=0;}}
                relation first on body {{{constraint}=0;}}
                relation second on body {{w*w+w=0;}}
            }}"
        );
        let (resolved, fields) = resolve_nonlinear_constraint(&source);
        let resolved = replay_plan(resolved, &REFERENCE_LINEAR_SOLVER);
        let plan = resolved.as_spatial().unwrap();
        let parts = if scalar == "1" { 1 } else { 2 };
        let start = 3
            * parts
            * plan
                .fields()
                .position(|(id, _)| id.erase() == fields[1])
                .unwrap();
        let initial = plan.initial_state(Some(0.), Vec::new()).unwrap();
        let before = initial.to_bytes().unwrap();
        let expected = initial.spatial_values().unwrap().to_vec();
        let run =
            CommonTransientRunRequest::from_steps(resolved.clone(), initial.clone(), 2, vec![1, 2])
                .unwrap();
        let std::ops::ControlFlow::Continue(outputs) = run
            .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
            .unwrap()
        else {
            panic!("complete constant run");
        };
        assert_eq!(initial.to_bytes().unwrap(), before);
        for state in std::iter::once(&initial).chain(outputs.iter().map(|(_, step)| step.state())) {
            for (actual, expected) in state.spatial_values().unwrap().iter().zip(&expected) {
                assert!((actual - expected).abs() < 1e-12);
            }
            assert_eq!(
                CommonState::from_bytes(&state.to_bytes().unwrap(), &resolved).unwrap(),
                *state
            );
            let mut singular = state.clone();
            let CommonStateKind::Spatial { values, .. } = &mut singular.kind else {
                unreachable!()
            };
            for node in 0..3 {
                values[start + node * parts..start + (node + 1) * parts]
                    .copy_from_slice(&singular_values[..parts]);
            }
            let singular_coordinates = values.to_vec();
            let error = CommonState::from_bytes(&singular.to_bytes().unwrap(), &resolved)
                .expect_err("zero-residual singular replay must reject");
            assert!(
                error.message().contains("algebraic regularity")
                    || error.message().contains("has no nonzero entries"),
                "{error:?}"
            );
            // This is also the constructor used when accepting a new step.
            let error = plan
                .scalar_state(state.time_s(), singular_coordinates)
                .expect_err("singular accepted point must reject");
            assert!(
                error.message().contains("algebraic regularity"),
                "{error:?}"
            );
        }
        let restart =
            CommonState::from_bytes(&outputs[0].1.state().to_bytes().unwrap(), &resolved).unwrap();
        let run =
            CommonTransientRunRequest::from_steps(resolved.clone(), restart, 1, vec![1]).unwrap();
        let std::ops::ControlFlow::Continue(restarted) = run
            .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
            .unwrap()
        else {
            panic!("complete restart");
        };
        assert_eq!(restarted.last().unwrap().1, outputs.last().unwrap().1);
        let source = source.replace(
            &format!("v={initial_literal}"),
            &format!("v={singular_literal}"),
        );
        let (resolved, _) = resolve_nonlinear_constraint(&source);
        let error = resolved
            .as_spatial()
            .unwrap()
            .initial_state(Some(0.), Vec::new())
            .expect_err("singular initial constraint must reject");
        assert!(
            error.message().contains("algebraic regularity"),
            "{error:?}"
        );
    }
}

#[test]
fn algebraic_initialization_only_demands_unknown_state_partials() {
    for initial in [1, 0] {
        // Differential u is prescribed, so the initial unknowns are its rate
        // and the algebraic values v,w. At (u,v,w)=(0,0,0), u_dot=0 gives
        // R=0 and the selected Jacobian is diag(M,M,M). The unused derivative
        // of sqrt(u) with respect to the fixed u is undefined at this point.
        let source = format!(
            "public model Constrained(
                support body:volume(ambient_dimension=1),
                support left:boundary(parent=body),support right:boundary(parent=body)) {{
                state u:1 on body in h1; state v:1 on body in h1; state w:1 on body in h1;
                initial {{u={initial};v=0;w=0;}}
                relation evolution on body {{1[s]*derivative(u)=math.sqrt(u);}}
                relation first on body {{v*v+v=0;}}
                relation second on body {{w*w+w=0;}}
            }}"
        );
        let (resolved, _) = resolve_nonlinear_constraint(&source);
        resolved
            .as_spatial()
            .unwrap()
            .initial_state(Some(0.), Vec::new())
            .expect("initialization must not demand the fixed differential State partial");
    }
}
