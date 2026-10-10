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

#[test]
fn mixed_storage_preserves_algebraic_feedback_and_restart() {
    // Two half-length elements give M_ii=1/3, K_ii=4.
    // Algebraic rows: 13v=u+w, 13w=v, hence v=13u/168 and w=u/168.
    // Evolution: u' = -(12+13/168)u = -2029u/168.
    // Backward Euler with dt=1/4 multiplies u by 672/2701 each step.
    for source in [SOURCE.to_owned(), SOURCE.replace("Constrained", "Renamed")] {
        let entry = if source.contains("model Renamed") {
            "Renamed"
        } else {
            "Constrained"
        };
        let (resolved, fields) = coupled::resolve(&source, entry).unwrap();
        let replay = replay_plan(resolved.clone(), &REFERENCE_LINEAR_SOLVER);
        let plan = resolved.as_linear().unwrap();
        let interior = fields.map(|field| {
            3 * plan
                .fields()
                .position(|(id, _)| id.erase() == field)
                .unwrap()
                + 1
        });
        let initial = plan.initial_state(0.0, Vec::new()).unwrap();
        let mut invalid = initial.clone();
        let CommonStateKind::Linear(values) = &mut invalid.kind else {
            unreachable!()
        };
        values[interior[1]] += 0.001;
        let error = CommonState::from_bytes(&invalid.to_bytes().unwrap(), &replay).unwrap_err();
        assert!(error.message().contains("algebraic"), "{}", error.message());
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
            let u = (672.0_f64 / 2701.0).powi(step);
            for (index, expected) in interior.into_iter().zip([u, 13.0 * u / 168.0, u / 168.0]) {
                assert!((state.linear_values().unwrap()[index] - expected).abs() < 1e-11);
            }
        }
        assert_eq!(before, initial.to_bytes().unwrap());
        let restarted =
            CommonState::from_bytes(&outputs[0].1.to_bytes().unwrap(), &replay).unwrap();
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
    let source = SOURCE.replace("v = (13/168)", "v = (14/168)");
    for step in [0.25, 1e-12] {
        let (resolved, _) = coupled::resolve_with_step(&source, "Constrained", step).unwrap();
        let error = resolved
            .as_linear()
            .unwrap()
            .initial_state(0.0, Vec::new())
            .unwrap_err();
        assert!(error.message().contains("algebraic"), "{}", error.message());
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
        Some(CommonBackwardEuler::from_seconds(0.25).unwrap()),
        &REFERENCE_LINEAR_SOLVER,
        None,
    )
    .unwrap();
    let resolved = replay_plan(resolved, &REFERENCE_LINEAR_SOLVER);
    let initial = resolved
        .as_linear()
        .unwrap()
        .initial_state(0., Vec::new())
        .unwrap();
    // Constant vectors have zero Laplacian and u=v cancels both reactions.
    // Every node/component therefore retains the independently prescribed (2,3,5).
    let expected = (0..54).flat_map(|_| [2., 3., 5.]).collect::<Vec<_>>();
    assert_eq!(initial.linear_values().unwrap(), expected);
    let run =
        CommonTransientRunRequest::from_steps(resolved.clone(), initial, 3, vec![1, 2, 3]).unwrap();
    let std::ops::ControlFlow::Continue(outputs) = run
        .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
        .unwrap()
    else {
        panic!("complete vector run")
    };
    for (_, state) in outputs {
        for (actual, expected) in state.linear_values().unwrap().iter().zip(&expected) {
            assert!((actual - expected).abs() < 1e-11);
        }
        assert_eq!(
            CommonState::from_bytes(&state.to_bytes().unwrap(), &resolved).unwrap(),
            state
        );
    }
}
