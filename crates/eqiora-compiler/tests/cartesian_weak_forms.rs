//! Weak-form dimensions come from the exact Model Domain, independent of meshing.
use eqiora_compiler::{AuthoredFormulationProjection, CompiledModel};

fn source(dimensions: usize) -> String {
    let bounds = vec!["0,1"; dimensions].join(",");
    format!(
        r#"model M() {{
        domain body=box({bounds});
        domain face=boundary(body,axis=0,side=lower);
        variable u:vector<1,{dimensions}> on body in h1;
        relation law on body {{ -div(grad(u))+u*1[1/m^2]=u*0[1/m^2]; }}
        form weak for law {{
            test v:1 for u zero_on face;
            integrate(body,frobenius(grad(v),grad(u))+dot(v,u)*1[1/m^2])=0;
        }}
    }}"#
    )
}

#[test]
fn one_vector_trial_keeps_exact_cartesian_support_and_form_replay() {
    for dimensions in [2, 3] {
        let compiled =
            CompiledModel::compile_selected("cartesian-form.eqi", &source(dimensions), "M", &[])
                .unwrap();
        let form = compiled.authored_formulations().next().unwrap();
        assert!(form.domain().is_some());
        assert_eq!(form.projection().test_restrictions().len(), 1);
        assert_eq!(form.projection().test_restrictions()[0].2.len(), 1);
        assert_eq!(
            &AuthoredFormulationProjection::decode(form.projection().canonical_bytes()).unwrap(),
            form.projection()
        );
    }
}

#[test]
fn explicitly_framed_vector_coefficient_keeps_weak_form_and_replay() {
    let text = source(3)
        .replace(
            "variable u:vector<1,3>",
            "parameter forcing:vector<1,3> = tensor_value(frame=body,components=[1,2,3]); variable u:vector<1,3>",
        )
        .replace("u*0[1/m^2]", "forcing*1[1/m^2]")
        .replace(
            "dot(v,u)*1[1/m^2])=0;",
            "dot(v,u)*1[1/m^2])=integrate(body,dot(v,forcing)*1[1/m^2]);",
        );
    let compiled = CompiledModel::compile_selected("vector-coefficient.eqi", &text, "M", &[])
        .unwrap_or_else(|errors| panic!("{errors:?}"));
    let form = compiled.authored_formulations().next().unwrap();
    assert!(form.domain().is_some());
    assert_eq!(
        &AuthoredFormulationProjection::decode(form.projection().canonical_bytes()).unwrap(),
        form.projection()
    );
}

fn closed_coefficient_source(declaration: &str, law: &str, weak: &str) -> String {
    format!(
        r#"public component C(support body:volume(ambient_dimension=2),support face:boundary(parent=body)) {{
        {declaration}
        variable u:vector<1,2> on body in h1;
        relation law on body {{ {law} }}
        form weak for law {{
            test v:1 for u zero_on face;
            {weak}
        }}
    }}"#
    )
}

fn compile_closed_coefficient(text: &str) -> CompiledModel {
    use eqiora_compiler::StaticBindingValue;
    use eqiora_geometry::GeometryGraph;
    use std::collections::BTreeMap;
    let graph = GeometryGraph::new();
    let region = graph.rectangle([0., 1.], [0., 1.]).unwrap();
    let geometry = graph
        .build(
            &region,
            &BTreeMap::from([
                ("body".into(), vec![region.region().into()]),
                ("face".into(), vec![region.boundaries()[0].into()]),
                ("right".into(), vec![region.boundaries()[1].into()]),
                ("bottom".into(), vec![region.boundaries()[2].into()]),
                ("top".into(), vec![region.boundaries()[3].into()]),
            ]),
        )
        .unwrap();
    CompiledModel::compile_selected(
        "closed-coefficient.eqi",
        text,
        "C",
        &[
            (
                "body",
                StaticBindingValue::GeometrySupport {
                    geometry: &geometry,
                    selection: geometry.entity_set("body").unwrap(),
                    parent: None,
                },
            ),
            (
                "face",
                StaticBindingValue::GeometrySupport {
                    geometry: &geometry,
                    selection: geometry.entity_set("face").unwrap(),
                    parent: Some(geometry.entity_set("body").unwrap()),
                },
            ),
        ],
    )
    .unwrap_or_else(|errors| panic!("{errors:?}"))
}

#[test]
fn component_tensor_defaults_preserve_dimensions_and_row_major_values() {
    let text = closed_coefficient_source(
        "parameter stress:tensor<Pa,2,2>=tensor_value(frame=body,components=[[2,3],[5,7]]);",
        "-div(grad(u))*1[Pa*m]=u*1[Pa/m];",
        "integrate(body,frobenius(grad(v),stress))=integrate(body,dot(v,u)*1[Pa/m]);",
    );
    let compiled = compile_closed_coefficient(&text);
    let form = compiled
        .authored_formulations()
        .next()
        .unwrap()
        .projection();
    let wire = std::str::from_utf8(form.canonical_bytes()).unwrap();
    assert!(wire.contains("\"kind\":\"spatial-tensor\""), "{wire}");
    assert!(wire.contains("\"shape\":[2,2]"), "{wire}");
    assert!(
        wire.contains("\"dimension\":[[1,1],[-1,1],[-2,1],[0,1],[0,1],[0,1],[0,1]]"),
        "{wire}"
    );
    assert!(
        wire.contains("\"values\":[[2.0,0.0],[3.0,0.0],[5.0,0.0],[7.0,0.0]]"),
        "{wire}"
    );
    assert_eq!(
        &AuthoredFormulationProjection::decode(form.canonical_bytes()).unwrap(),
        form
    );
    assert!(
        AuthoredFormulationProjection::decode(
            wire.replace("eqiora.authored-form/v20", "eqiora.authored-form/v19")
                .as_bytes()
        )
        .is_err()
    );
}

#[test]
fn component_vector_defaults_preserve_complex_components() {
    let text = closed_coefficient_source(
        "parameter forcing:vector<complex<1>,2>=tensor_value(frame=body,components=[math.complex(2,3),math.complex(5,7)]);",
        "-div(grad(u))=forcing*1[1/m^2];",
        "integrate(body,inner(grad(v),grad(u)))=integrate(body,inner(v,forcing)*1[1/m^2]);",
    ).replace("variable u:vector<1,2>", "variable u:vector<complex<1>,2>");
    let compiled = compile_closed_coefficient(&text);
    let form = compiled
        .authored_formulations()
        .next()
        .unwrap()
        .projection();
    let wire = std::str::from_utf8(form.canonical_bytes()).unwrap();
    assert!(wire.contains("\"complex\":true"), "{wire}");
    assert!(wire.contains("\"values\":[[2.0,3.0],[5.0,7.0]]"), "{wire}");
    assert_eq!(
        &AuthoredFormulationProjection::decode(form.canonical_bytes()).unwrap(),
        form
    );
}

#[test]
fn cartesian_form_rejects_foreign_parent_and_channel_trials() {
    for invalid in [
        source(3).replace(
            "domain face=boundary(body",
            "domain other=box(0,1,0,1,0,1); domain face=boundary(other",
        ),
        source(3).replace("vector<1,3>", "array<1,3>"),
    ] {
        assert!(
            CompiledModel::compile_selected("invalid-cartesian-form.eqi", &invalid, "M", &[])
                .is_err()
        );
    }
}

fn oriented_source(dimensions: usize, expression: &str) -> String {
    source(dimensions).replace(
        "integrate(body,frobenius(grad(v),grad(u))+dot(v,u)*1[1/m^2])",
        expression,
    )
}

#[test]
fn oriented_forms_retain_closed_operators_and_replay() {
    for (dimensions, expression, operator) in [
        (3, "integrate(body,dot(curl(v),curl(u)))", "curl"),
        (2, "integrate(body,curl(v)*curl(u))", "curl"),
        (
            3,
            "integrate(face,dot(tangential_trace(v),trace(u)))",
            "tangential_trace",
        ),
        (
            2,
            "integrate(face,tangential_trace(v)*tangential_trace(u))",
            "tangential_trace",
        ),
        (3, "integrate(body,dot(cross(v,u),u))", "cross"),
    ] {
        let compiled = CompiledModel::compile_selected(
            "oriented-form.eqi",
            &oriented_source(dimensions, expression),
            "M",
            &[],
        )
        .unwrap_or_else(|errors| panic!("{expression}: {errors:?}"));
        let projection = compiled
            .authored_formulations()
            .next()
            .unwrap()
            .projection();
        let bytes = projection.canonical_bytes();
        let wire = std::str::from_utf8(bytes).unwrap();
        assert!(wire.contains(&operator.replace('_', "-")), "{wire}");
        assert_eq!(
            &AuthoredFormulationProjection::decode(bytes).unwrap(),
            projection
        );
        assert!(
            AuthoredFormulationProjection::decode(
                wire.replace("eqiora.authored-form/v20", "eqiora.authored-form/v12")
                    .as_bytes(),
            )
            .is_err()
        );
    }
}

#[test]
fn oriented_forms_reject_wrong_shapes_and_boundary_scope() {
    for (dimensions, expression, message) in [
        (2, "integrate(body,dot(curl(v),curl(u)))", "contraction"),
        (2, "integrate(body,dot(cross(v,u),u))", "exact type rule"),
        (3, "integrate(body,dot(tangential_trace(v),u))", "boundary"),
        (
            3,
            "integrate(face,dot(curl(trace(v)),trace(u)))",
            "gradient",
        ),
    ] {
        let errors = CompiledModel::compile_selected(
            "invalid-oriented-form.eqi",
            &oriented_source(dimensions, expression),
            "M",
            &[],
        )
        .unwrap_err();
        assert!(
            errors
                .iter()
                .any(|e| e.message().to_lowercase().contains(message)),
            "{expression}: {errors:?}"
        );
    }
}

#[test]
fn complex_oriented_forms_preserve_sesquilinear_dependence() {
    for expression in [
        "integrate(body,inner(curl(v),curl(u)))",
        "integrate(face,inner(tangential_trace(v),trace(u)))",
        "integrate(body,inner(cross(v,b),u))",
    ] {
        let text = oriented_source(3, expression).replace(
            "variable u:vector<1,3>",
            "variable b:vector<1,3> on body; variable u:vector<complex<1>,3>",
        );
        let compiled = CompiledModel::compile_selected("complex-oriented.eqi", &text, "M", &[])
            .unwrap_or_else(|errors| panic!("{expression}: {errors:?}"));
        let form = compiled
            .authored_formulations()
            .next()
            .unwrap()
            .projection();
        assert_eq!(
            &AuthoredFormulationProjection::decode(form.canonical_bytes()).unwrap(),
            form
        );
        let invalid = text.replace("inner(", "dot(");
        let errors = CompiledModel::compile_selected("invalid-oriented.eqi", &invalid, "M", &[])
            .unwrap_err();
        assert!(
            errors
                .iter()
                .any(|e| e.message().contains("conjugate-linear test dependence")),
            "{errors:?}"
        );
    }
}
