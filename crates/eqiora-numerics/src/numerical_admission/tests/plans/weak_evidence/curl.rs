use super::*;

#[test]
fn planar_scalar_curl_pairing_reaches_the_existing_q1_plan_and_result() {
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
    }"#;
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
    let (model, projection) = compile(source);
    let resolve = |projection: &AuthoredFormulationProjection| {
        ResolvedCommonPlan::resolve(
            &model,
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
    let plan = replay_plan(resolve(&projection).unwrap(), &REFERENCE_LINEAR_SOLVER);
    let result = plan
        .as_scalar()
        .unwrap()
        .run_result(&REFERENCE_LINEAR_SOLVER)
        .unwrap();
    let (_, values, shape) = result.field_block(0, 0).unwrap();
    assert_eq!(shape, &[3, 3]);
    // On the four unit-square Q1 cells, the sole interior basis has
    // integral 1/4 and stiffness 8/3, hence u_center=3/32. All edges are zero.
    for (i, value) in values.iter().enumerate() {
        let expected = if i == 4 { 3.0 / 32.0 } else { 0.0 };
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
    let (_, wrong) = compile(&source.replace("dot(curl(eta),curl(u))", "-dot(curl(eta),curl(u))"));
    assert_eq!(wrong.trial_ulids(), projection.trial_ulids());
    let error = resolve(&wrong).unwrap_err();
    assert!(
        error.message().contains("left bilinear term")
            || error.message().contains("authored weak residual differs"),
        "{error:?}"
    );
}
