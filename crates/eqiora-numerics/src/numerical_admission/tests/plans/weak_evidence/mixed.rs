use super::*;

fn cyclic_source(derived: bool) -> String {
    let mut source = String::from(
        "public component Wave(support body:volume(ambient_dimension=1),support left:boundary(parent=body),support right:boundary(parent=body)) {\n",
    );
    for field in ["a", "b", "c"] {
        source.push_str(&format!("variable {field}:1 on body in smooth;\n"));
    }
    let rows = [("a", "b", "c", 1), ("b", "c", "a", 2), ("c", "a", "b", 3)];
    for (field, first, second, value) in rows {
        source.push_str(&format!("relation r{field} on body {{ {field}-div(1[m^2]*(grad({first})+grad({second})))={value}; }}\n"));
        for side in ["left", "right"] {
            source.push_str(&format!("relation {field}_{side} on {side} {{normal(1[m^2]*(grad({first})+grad({second})))=0;}}\n"));
        }
    }
    source
        .push_str("form explicit for ra,rb,rc {test wc:1 for c;test wb:1 for b;test wa:1 for a;\n");
    for (field, first, second, value) in rows {
        if derived {
            source.push_str(&format!("derive r{field} using w{field};\n"));
        } else {
            source.push_str(&format!("r{field}: integrate(body,w{field}*{field}+1[m^2]*dot(grad(w{field}),grad({first})+grad({second})))=integrate(body,w{field}*{value});\n"));
        }
    }
    source.push_str("}}\n");
    source
}

#[test]
fn explicit_cyclic_rows_resolve_execute_and_replay_without_principal_inference() {
    let source = cyclic_source(false);
    let geometry = geometry(false);
    let (program, projection) = compile(&source, &geometry).unwrap();
    let model = ModelEnvelope::from_program(&program).unwrap();
    let error = RecognizedNativeAdmission::recognize(
        &model,
        cartesian_box_resources(&geometry, &[2]),
        &CommonSpatialRequest::Uniform(CommonSpatialPolicy::Q1),
        None,
        &BTreeMap::new(),
    )
    .unwrap_err();
    assert!(error.message().contains("principal"), "{error:?}");
    let plan = replay_plan(resolve(&source).unwrap(), &REFERENCE_LINEAR_SOLVER);
    let result = plan
        .as_spatial()
        .unwrap()
        .run_result(None, &REFERENCE_LINEAR_SOLVER)
        .unwrap();
    assert_eq!(result.field_count(), 3);
    let expected = projection
        .test_restrictions()
        .iter()
        .map(|(test, field, _, _, _)| {
            (
                field.as_str(),
                match test.as_str() {
                    "wa" => 1.,
                    "wb" => 2.,
                    "wc" => 3.,
                    _ => panic!("unknown test"),
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    for index in 0..3 {
        let value = expected[result.field(index).unwrap().0];
        let (_, values, shape) = result.field_block(index, 0).unwrap();
        assert_eq!(shape, &[3]);
        assert!(values.iter().all(|actual| (actual - value).abs() < 1e-10));
    }
}

const MIXED: &str = r#"public component Wave(
    support body:volume(ambient_dimension=1),
    support left:boundary(parent=body), support right:boundary(parent=body)) {
    state u:1 on body in smooth;
    variable v:1 on body in smooth;
    initial { u=1; v=0; }
    relation evolution on body { derivative(u)-1[m^2/s]*div(grad(v))=0; }
    relation auxiliary on body { v+1[m^2]*div(grad(u))=0; }
    relation ul on left { normal(grad(u))=0; }
    relation ur on right { normal(grad(u))=0; }
    relation vl on left { normal(grad(v))=0; }
    relation vr on right { normal(grad(v))=0; }
    form mixed for evolution, auxiliary {
        test wu:1 for u;
        test wv:1 for v;
        evolution: integrate(body,wu*derivative(u)+1[m^2/s]*dot(grad(wu),grad(v)))=0;
        auxiliary: integrate(body,wv*v-1[m^2]*dot(grad(wv),grad(u)))=0;
    }
}"#;

#[test]
fn explicit_cross_field_flux_and_storage_lower_through_shared_region_actions() {
    let geometry = geometry(false);
    let (program, projection) = compile(MIXED, &geometry).unwrap();
    crate::form_compiler::mixed::check(&program, &projection).unwrap();
    let tests = projection.equation_test_fields().unwrap();
    let domain = eqiora_core::Id::<eqiora_core::entity::kinds::Domain>::from_ulid(
        projection.domain_ulid().unwrap().parse().unwrap(),
    )
    .erase();
    let form = crate::form_compiler::spatial::CompiledSpatialForm::<f64>::derive_with_tests(
        &program,
        domain,
        1,
        &BTreeSet::new(),
        Some(0.),
        &tests,
    )
    .unwrap();
    assert_eq!(form.fields().len(), 2);
    assert!(form.is_transient());
    assert!(form.boundary_laws().values().all(|laws| laws.len() == 2));
    form.volume().unwrap();
}

#[test]
fn explicit_mixed_rows_reject_wrong_volume_terms_and_boundary_signs() {
    let geometry = geometry(false);
    let wrong = MIXED.replace(
        "wu*derivative(u)+1[m^2/s]*dot",
        "wu*derivative(u)+2[m^2/s]*dot",
    );
    let (program, projection) = compile(&wrong, &geometry).unwrap();
    assert!(
        crate::form_compiler::mixed::check(&program, &projection)
            .unwrap_err()
            .message()
            .contains("authored row differs")
    );

    let prescribed = MIXED
        .replace("normal(grad(v))=0", "normal(1[m^2/s]*grad(v))=2[m/s]")
        .replace("integrate(body,wu*derivative(u)+1[m^2/s]*dot(grad(wu),grad(v)))=0", "integrate(body,wu*derivative(u)+1[m^2/s]*dot(grad(wu),grad(v)))-integrate(left,trace(wu,on=left)*2[m/s])-integrate(right,trace(wu,on=right)*2[m/s])=0");
    let (program, projection) = compile(&prescribed, &geometry).unwrap();
    crate::form_compiler::mixed::check(&program, &projection).unwrap();
    let wrong = prescribed.replace("-integrate(left", "+integrate(left");
    let (program, projection) = compile(&wrong, &geometry).unwrap();
    assert!(crate::form_compiler::mixed::check(&program, &projection).is_err());
}

#[test]
fn pairing_only_request_derives_and_executes_the_shared_cyclic_form() {
    let geometry = geometry(false);
    let (program, authored) = compile(&cyclic_source(false), &geometry).unwrap();
    let (_, request) = compile(&cyclic_source(true), &geometry).unwrap();
    assert!(request.requests_weak_derivation());
    assert!(request.equations().is_empty());
    let effective = crate::form_compiler::mixed::resolve(&program, &request).unwrap();
    assert_eq!(
        effective.equation_test_fields().unwrap(),
        authored.equation_test_fields().unwrap()
    );
    crate::form_compiler::mixed::check(&program, &effective).unwrap();
    let mut changed = effective.equations().to_vec();
    changed[0].1 = eqiora_compiler::AuthoredFormExpressionV1::Mul {
        left: Box::new(eqiora_compiler::AuthoredFormExpressionV1::Number { value: 2. }),
        right: Box::new(changed[0].1.clone()),
    };
    let forged = effective.with_derived_equations(changed).unwrap();
    assert!(crate::form_compiler::mixed::resolve(&program, &forged).is_err());
    let model = ModelEnvelope::from_program(&program).unwrap();
    let plan = ResolvedCommonPlan::resolve(
        &model,
        cartesian_box_resources(&geometry, &[2]),
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
        None,
        Some(&request),
    )
    .unwrap();
    let plan = replay_plan(plan, &REFERENCE_LINEAR_SOLVER);
    let result = plan
        .as_spatial()
        .unwrap()
        .run_result(None, &REFERENCE_LINEAR_SOLVER)
        .unwrap();
    let expected = authored
        .test_restrictions()
        .iter()
        .map(|(name, field, _, _, _)| {
            (
                field.as_str(),
                match name.as_str() {
                    "wa" => 1.,
                    "wb" => 2.,
                    "wc" => 3.,
                    _ => unreachable!(),
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    for i in 0..3 {
        let value = expected[result.field(i).unwrap().0];
        assert!(
            result
                .field_block(i, 0)
                .unwrap()
                .1
                .iter()
                .all(|actual| (actual - value).abs() < 1e-10)
        );
    }
}
