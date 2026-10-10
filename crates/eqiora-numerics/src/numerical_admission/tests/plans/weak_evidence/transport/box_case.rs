use super::*;

const SOURCE: &str = r#"
public component Wave(
    support body: volume(ambient_dimension = 2),
    support left: boundary(parent = body),
    support right: boundary(parent = body),
    support bottom: boundary(parent = body),
    support top: boundary(parent = body)
) {
    coordinate x: m on body from body[0];
    coordinate y: m on body from body[1];
    state u: 1 on body in h1;
    initial { u = 1 + x / 1[m] + 2 * y / 1[m]; }
    law balance on body {
        storage 3[s/m^2] * u;
        flux -grad(u) + u * (2[1/m] * grad(coordinate(0)) - 1[1/m] * grad(coordinate(1)));
        source 0[1/m^2];
    }
    relation left_value on left { trace(u) = 1 + 1[1/m] * coordinate(0) + 2[1/m] * coordinate(1); }
    relation right_value on right { trace(u) = 1 + 1[1/m] * coordinate(0) + 2[1/m] * coordinate(1); }
    relation bottom_value on bottom { trace(u) = 1 + 1[1/m] * coordinate(0) + 2[1/m] * coordinate(1); }
    relation top_value on top { trace(u) = 1 + 1[1/m] * coordinate(0) + 2[1/m] * coordinate(1); }
    form weak_transport for balance {
        test w: 1 for u zero_on left, right, bottom, top;
        integrate(body, w * 3[s/m^2] * derivative(u))
            + integrate(body, dot(grad(w), grad(u)))
            - integrate(body, dot(grad(w), u * (2[1/m] * grad(coordinate(0)) - 1[1/m] * grad(coordinate(1)))))
            = integrate(body, w * 0[1/m^2]);
    }
}
"#;

fn resolve(source: &str) -> Result<ResolvedCommonPlan, Diagnostic> {
    let geometry = CanonicalGeometryV1::decode_cartesian_box_v1_canonical(
        br#"{"schema":"eqiora.cartesian-box-envelope/v1","encoding":"eqiora.canonical-json/v1","length_unit":"metre","bounds":[[0.0,1.0],[0.0,1.0]],"entity_sets":[{"name":"bottom","dimension":1,"members":[2]},{"name":"left","dimension":1,"members":[0]},{"name":"right","dimension":1,"members":[1]},{"name":"top","dimension":1,"members":[3]},{"name":"region","dimension":2,"members":[0]}]}"#,
        eqiora_geometry::CanonicalGeometryLimits::default(),
    ).unwrap();
    let body = geometry.entity_set("region").unwrap();
    let bindings = ["body", "left", "right", "bottom", "top"].map(|name| {
        (
            name,
            StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: if name == "body" {
                    body
                } else {
                    geometry.entity_set(name).unwrap()
                },
                parent: (name != "body").then_some(body),
            },
        )
    });
    let compiled = CompiledModel::compile_selected("transport-box.eqi", source, "Wave", &bindings)
        .map_err(|errors| errors.into_iter().next().unwrap())?;
    let projection = compiled
        .authored_formulations()
        .next()
        .unwrap()
        .projection()
        .clone();
    let (transaction, model, _) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program =
        KernelProgram::from_snapshot_with_geometry(&store.snapshot(), model, &[&geometry]).unwrap();
    let model = ModelEnvelope::from_program(&program).unwrap();
    ResolvedCommonPlan::resolve(
        &model,
        cartesian_box_resources(&geometry, &[2, 2]),
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
fn box_transport_retains_each_prescribed_axis() {
    // u=1+x+2y and beta=(2,-1) give div(beta*u-grad(u))=0.
    // Swapping beta's axes gives div=3. The single interior Q1 basis has
    // integral 1/4, M=1/3, K=8/3, dt=1/4, hence excess -3/16.
    for (source, expected) in [
        (SOURCE.to_owned(), 2.5),
        (
            SOURCE.replace(
                "2[1/m] * grad(coordinate(0)) - 1[1/m] * grad(coordinate(1))",
                "2[1/m] * grad(coordinate(1)) - 1[1/m] * grad(coordinate(0))",
            ),
            2.5 - 3. / 16.,
        ),
    ] {
        let resolved = replay_plan(resolve(&source).unwrap(), &REFERENCE_LINEAR_SOLVER);
        let initial = resolved
            .as_linear()
            .unwrap()
            .initial_state(0.0, Vec::new())
            .unwrap();
        let initial_values = initial.linear_values().unwrap().to_vec();
        let run = CommonTransientRunRequest::from_steps(resolved, initial, 1, vec![1]).unwrap();
        let std::ops::ControlFlow::Continue(outputs) = run
            .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
            .unwrap()
        else {
            panic!("box step must complete");
        };
        assert_eq!(outputs.len(), 1);
        let values = outputs[0].1.linear_values().unwrap();
        assert_eq!(values.len(), 9);
        for (index, value) in values.iter().enumerate() {
            if index == 4 {
                assert!((value - expected).abs() < 1e-11, "{values:?}");
            } else {
                assert_eq!(*value, initial_values[index]);
            }
        }
    }
    let (law, weak) = SOURCE.split_once("    form weak_transport").unwrap();
    let wrong = format!(
        "{law}    form weak_transport{}",
        weak.replace("grad(coordinate(0))", "grad(coordinate(1))")
    );
    let error = resolve(&wrong).unwrap_err();
    assert!(
        error.message().contains("left bilinear term"),
        "{}",
        error.message()
    );
}
