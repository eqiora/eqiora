use super::*;
use eqiora_core::RawId;
const SOURCE: &str = r#"
public model Coupled(
    support body: volume(ambient_dimension = 1),
    support left: boundary(parent = body),
    support right: boundary(parent = body)
) {
    coordinate x: m on body from body[0];
    state u: 1 on body in h1;
    state v: 1 on body in h1;
    state w: 1 on body in h1;
    initial {
        u = 12 * x * (1[m] - x) / 1[m^2];
        v = 4 * x * (1[m] - x) / 1[m^2];
        w = 0;
    }
    law first on body { storage 1[s/m^2] * u; flux -grad(u); source -(u-v) * 1[1/m^2]; }
    law second on body { storage 1[s/m^2] * v; flux -grad(v); source -(2*v-u-w) * 1[1/m^2]; }
    law third on body { storage 1[s/m^2] * w; flux -grad(w); source -(w-v) * 1[1/m^2]; }
    relation left_values on left { trace(u) = 0; trace(v) = 0; trace(w) = 0; }
    relation right_values on right { trace(u) = 0; trace(v) = 0; trace(w) = 0; }
}
"#;

fn resolve(source: &str, entry: &str) -> Result<(ResolvedCommonPlan, [RawId; 3]), Diagnostic> {
    let geometry = cartesian_interval();
    let body = geometry.entity_set("body").unwrap();
    let bindings = [
        (
            "body",
            StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: body,
                parent: None,
            },
        ),
        (
            "left",
            StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set("left").unwrap(),
                parent: Some(body),
            },
        ),
        (
            "right",
            StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set("right").unwrap(),
                parent: Some(body),
            },
        ),
    ];
    let compiled =
        CompiledModel::compile_selected("coupled-storage.eqi", source, entry, &bindings).unwrap();
    let (transaction, model, symbols) = compiled.into_parts();
    let fields = ["u", "v", "w"].map(|name| symbols.get(name).unwrap());
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program =
        KernelProgram::from_snapshot_with_geometry(&store.snapshot(), model, &[&geometry]).unwrap();
    let model = ModelEnvelope::from_program(&program).unwrap();
    let resolved = ResolvedCommonPlan::resolve(
        &model,
        cartesian_box_resources(&geometry, &[2]),
        CommonSpatialPolicy::Q1,
        CommonSolvePolicy::Linear(exact_reference_linear(
            LinearSolver::BiConjugateGradientStabilized,
            1e-13,
            1e-14,
            NonZeroUsize::new(100).unwrap(),
        )),
        None,
        Some(CommonBackwardEuler::from_seconds(0.25).unwrap()),
        &REFERENCE_LINEAR_SOLVER,
        None,
    )?;
    Ok((resolved, fields))
}

#[test]
fn three_stored_fields_share_one_coupled_backward_euler_solve() {
    let hierarchical = hierarchical_source();
    let renamed = hierarchical.replace("Coupled", "Network").replace("Exchange", "Cell")
        .replace("first:", "alpha:").replace("second:", "beta:").replace("third:", "gamma:")
        .replace("    state u: 1 on body in h1;\n    state v: 1 on body in h1;\n    state w: 1 on body in h1;",
                 "    state w: 1 on body in h1;\n    state u: 1 on body in h1;\n    state v: 1 on body in h1;")
        .replace("        w = 0;", "    }\n    initial { w = 0;");
    // Add the stationary vector (1,2,3): L times that vector is (-1,0,1).
    // Its affine source balances reaction exactly, including distinct boundary data.
    let shifted = SOURCE
        .replace("u = 12", "u = 1 + 12")
        .replace("v = 4", "v = 2 + 4")
        .replace("w = 0;", "w = 3;")
        .replace("source -(u-v)", "source (-(u-v)-1)")
        .replace("source -(w-v)", "source (-(w-v)+1)")
        .replace("trace(u) = 0", "trace(u) = 1")
        .replace("trace(v) = 0", "trace(v) = 2")
        .replace("trace(w) = 0", "trace(w) = 3");
    for (source, entry, offset) in [
        (SOURCE, "Coupled", [0.; 3]),
        (hierarchical.as_str(), "Coupled", [0.; 3]),
        (renamed.as_str(), "Network", [0.; 3]),
        (shifted.as_str(), "Coupled", [1., 2., 3.]),
    ] {
        let (resolved, fields) = resolve(source, entry).unwrap();
        let replay = replay_plan(resolved.clone(), &REFERENCE_LINEAR_SOLVER);
        let plan = resolved.as_linear().unwrap();
        let initial = plan.initial_state().unwrap();
        assert_eq!(initial.linear_values().unwrap().len(), 9);
        // The public canonical Field inventory owns the Field-major nodal ordering.
        let interior = fields.map(|field| {
            3 * plan
                .fields()
                .position(|(id, _)| id.erase() == field)
                .unwrap()
                + 1
        });
        for ((index, expected), offset) in interior.into_iter().zip([3.0, 1.0, 0.0]).zip(offset) {
            assert_eq!(initial.linear_values().unwrap()[index], expected + offset);
        }
        for (values, message) in [
            (vec![0.; 8], "exact mapped coefficient inventory"),
            (vec![f64::NAN; 9], "history is nonfinite"),
        ] {
            let mut invalid = initial.clone();
            invalid.kind = CommonStateKind::Linear(values.into_boxed_slice());
            let error = plan
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
            panic!("all steps must complete")
        };
        assert_eq!(outputs.len(), 3);
        // Two h=1/2 elements give M_ii=1/3 and K_ii=4, hence rate 12.
        // L=[[1,-1,0],[-1,2,-1],[0,-1,1]] has modes (1,1,1),
        // (1,0,-1), (1,-2,1), eigenvalues 0,1,3. Initial (3,1,0)
        // decomposes as 4/3, 3/2, 1/6 times those modes. For dt=1/4,
        // backward Euler attenuates them by 1/4, 4/17, 4/19 per step.
        for (step, (_, state)) in (1..=3).zip(&outputs) {
            let a = (4.0 / 3.0) * 0.25_f64.powi(step);
            let b = 1.5 * (4.0_f64 / 17.0).powi(step);
            let c = (1.0 / 6.0) * (4.0_f64 / 19.0).powi(step);
            for ((index, expected), offset) in interior
                .into_iter()
                .zip([a + b + c, a - 2.0 * c, a - b + c])
                .zip(offset)
            {
                let values = state.linear_values().unwrap();
                assert!((values[index] - (expected + offset)).abs() < 1e-11);
                assert_eq!(values[index - 1], offset);
                assert_eq!(values[index + 1], offset);
            }
        }
        let restart = CommonState::from_bytes(&outputs[0].1.to_bytes().unwrap(), &replay).unwrap();
        let run = CommonTransientRunRequest::from_steps(replay, restart, 2, vec![1, 2]).unwrap();
        let std::ops::ControlFlow::Continue(restarted) = run
            .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
            .unwrap()
        else {
            panic!("restart must complete")
        };
        assert_eq!(restarted[1].1, outputs[2].1);
    }
}

fn hierarchical_source() -> String {
    let start = SOURCE.find("    law first").unwrap();
    let end = SOURCE.find("    relation left_values").unwrap();
    let component = r#"
component Exchange(
    support body: volume(ambient_dimension = 1),
    state q: 1 on body,
    variable a: 1 on body,
    variable b: 1 on body,
    parameter capacity: s/m^2,
    parameter ra: 1/s,
    parameter rb: 1/s
) {
    law local on body {
        storage capacity*q;
        flux -(capacity/1[s/m^2])*grad(q);
        source capacity*(ra*(a-q)+rb*(b-q));
    }
}
"#;
    format!(
        "{component}{}{}{}",
        &SOURCE[..start],
        r#"
    instance first: Exchange(body=body, q=u, a=v, b=w, capacity=1[s/m^2], ra=1[1/s], rb=0[1/s]);
    instance second: Exchange(body=body, q=v, a=u, b=w, capacity=2[s/m^2], ra=1[1/s], rb=1[1/s]);
    instance third: Exchange(body=body, q=w, a=v, b=u, capacity=3[s/m^2], ra=1[1/s], rb=0[1/s]);
"#,
        &SOURCE[end..]
    )
}

#[test]
fn coupled_storage_rejects_changed_physics_and_incomplete_conditions() {
    for (from, to, message) in [
        (
            "source -(u-v) * 1[1/m^2]",
            "source u*v * 1[1/m^2]",
            "nonlinear product",
        ),
        (
            "source -(u-v) * 1[1/m^2]",
            "source derivative(u) * 1[s/m^2]",
            "source requires prescribed data or linear reaction terms",
        ),
        (
            "source -(u-v) * 1[1/m^2]",
            "source div(grad(u))",
            "source requires prescribed data or linear reaction terms",
        ),
        (
            "storage 1[s/m^2] * u",
            "storage 1[s/m^2] * v",
            "storage requires one coefficient times its exact Field",
        ),
        (
            "storage 1[s/m^2] * u",
            "storage -1[s/m^2] * u",
            "strictly positive constant capacity",
        ),
        (
            "storage 1[s/m^2] * u;",
            "",
            "storage for every unknown Field",
        ),
        (
            "trace(u) = 0; trace(v) = 0; trace(w) = 0;",
            "trace(u) = 0; trace(u) = 0; trace(w) = 0;",
            "duplicate Field boundary law",
        ),
        (
            "trace(u) = 0; trace(v) = 0; trace(w) = 0;",
            "trace(u) = 0; trace(v) = 0;",
            "complete boundary law coverage",
        ),
    ] {
        let source = SOURCE.replace(from, to);
        assert_ne!(source, SOURCE);
        let error = resolve(&source, "Coupled").unwrap_err();
        assert!(
            error.message().contains(message),
            "{to}: {}",
            error.message()
        );
    }
}
