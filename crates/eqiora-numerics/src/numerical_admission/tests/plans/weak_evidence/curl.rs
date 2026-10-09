use super::*;

#[test]
fn planar_scalar_curl_pairing_reaches_the_existing_q1_plan_and_result() {
    check_planar_curl_pairing("-div(grad(u))", false);
}

#[test]
fn retained_planar_scalar_curl_curl_reaches_q1_correspondence_and_result() {
    for complex in [false, true] {
        check_planar_curl_pairing("curl(curl(u))", complex);
    }
}

fn check_planar_curl_pairing(strong_operator: &str, complex: bool) {
    let geometry = CanonicalGeometryV1::decode_cartesian_box_v1_canonical(
        br#"{"schema":"eqiora.cartesian-box-envelope/v1","encoding":"eqiora.canonical-json/v1","length_unit":"metre","bounds":[[0.0,1.0],[0.0,1.0]],"entity_sets":[{"name":"bottom","dimension":1,"members":[2]},{"name":"left","dimension":1,"members":[0]},{"name":"right","dimension":1,"members":[1]},{"name":"top","dimension":1,"members":[3]},{"name":"body","dimension":2,"members":[0]}]}"#,
        eqiora_geometry::CanonicalGeometryLimits::default(),
    ).unwrap();
    let names = ["left", "right", "bottom", "top"];
    let members = names.map(|name| geometry.entity_set(name).unwrap());
    let bindings = [
        (
            "body",
            StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set("body").unwrap(),
                parent: None,
            },
        ),
        (
            "surface",
            StaticBindingValue::CompleteExterior {
                geometry: &geometry,
                members: &members,
                parent: geometry.entity_set("body").unwrap(),
            },
        ),
    ];
    let source = r#"public component Poisson(
        support body:volume(ambient_dimension=2), support surface:complete_exterior(parent=body)
    ) {
        variable u:1 on body;
        relation law on body { -div(grad(u))=1[1/m^2]; }
        relation fixed[face in surface] on face { trace(u)=0; }
        form weak for law {
            test eta:1 for u zero_on surface;
            integrate(body,dot(curl(eta),curl(u)))=integrate(body,eta*1[1/m^2]);
        }
    }"#
    .replace("-div(grad(u))", strong_operator);
    let source = if complex {
        source
            .replace("variable u:1", "variable u:complex<1>")
            .replace("=1[1/m^2]", "=math.complex(1[1/m^2],2[1/m^2])")
            .replace("dot(curl(eta),curl(u))", "inner(curl(eta),curl(u))")
            .replace("eta*1[1/m^2]", "inner(eta,math.complex(1[1/m^2],2[1/m^2]))")
    } else {
        source
    };
    let compile = |source: &str| {
        let compiled =
            CompiledModel::compile_selected("planar-curl.eqi", source, "Poisson", &bindings)
                .unwrap();
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
            KernelProgram::from_snapshot_with_geometry(&store.snapshot(), model, &[&geometry])
                .unwrap();
        (ModelEnvelope::from_program(&program).unwrap(), projection)
    };
    let (model, projection) = compile(&source);
    let resolve = |model: &ModelEnvelope, projection: &AuthoredFormulationProjection| {
        ResolvedCommonPlan::resolve(
            model,
            cartesian_box_resources(&geometry, &[2, 2]),
            CommonSpatialPolicy::Q1,
            CommonSolvePolicy::Linear(exact_reference_linear(
                LinearSolver::BiConjugateGradientStabilized,
                1e-12,
                1e-14,
                NonZeroUsize::new(64).unwrap(),
            )),
            None,
            None,
            &REFERENCE_LINEAR_SOLVER,
            Some(projection),
        )
    };
    let plan = replay_plan(
        resolve(&model, &projection).unwrap(),
        &REFERENCE_LINEAR_SOLVER,
    );
    let formulation = plan.as_scalar().unwrap().formulation().unwrap();
    let rules = formulation.rule_ids();
    let expected_rule = if strong_operator == "curl(curl(u))" {
        "fem.derive.v1.planar-scalar-curl-curl-by-parts"
    } else {
        "fem.derive.v1.divergence-by-parts"
    };
    assert!(rules.contains(&expected_rule), "{rules:?}");
    let result = plan
        .as_scalar()
        .unwrap()
        .run_result(&REFERENCE_LINEAR_SOLVER)
        .unwrap();
    let (_, values, shape) = result.field_block(0, 0).unwrap();
    assert_eq!(shape, &[3, 3]);
    // On the four unit-square Q1 cells, the sole interior basis has
    // integral 1/4 and stiffness 8/3, hence u_center=3/32. All edges are zero.
    let channels = if complex { 2 } else { 1 };
    assert_eq!(values.len(), 9 * channels);
    for (i, value) in values.iter().enumerate() {
        // The complex load 1+2i scales the same real Q1 solution by 1+2i.
        let expected = if i / channels == 4 {
            (i % channels + 1) as f64 * 3.0 / 32.0
        } else {
            0.0
        };
        assert!((value - expected).abs() < 1e-12, "coefficient {i}: {value}");
    }
    let bytes = result.to_bytes().unwrap();
    assert_eq!(
        crate::CommonResult::from_bytes(&bytes, &plan)
            .unwrap()
            .to_bytes()
            .unwrap(),
        bytes
    );
    let (_, wrong) = compile(&source.replace("curl(eta),curl(u)", "-curl(eta),curl(u)"));
    assert_eq!(wrong.trial_ulids(), projection.trial_ulids());
    if strong_operator == "curl(curl(u))" {
        let (wrong_model, wrong_form) = compile(&source.replace("curl(curl(u))", "-curl(curl(u))"));
        let error = resolve(&wrong_model, &wrong_form).unwrap_err();
        assert!(
            error.message().contains("left bilinear term")
                || error.message().contains("authored weak residual differs"),
            "{error:?}"
        );
    }
    let error = resolve(&model, &wrong).unwrap_err();
    assert!(
        error.message().contains("left bilinear term")
            || error.message().contains("authored weak residual differs"),
        "{error:?}"
    );
}
