//! Weak-form dimensions come from the exact Model Domain, independent of meshing.
use eqiora_compiler::{AuthoredFormulationProjection, CompiledModel};

fn source(dimensions: usize) -> String {
    let bounds = vec!["0,1"; dimensions].join(",");
    format!(
        r#"model M() {{
        domain body=box({bounds});
        domain face=boundary(body,axis=0,side=lower);
        variable u:vector<1,{dimensions}> on body;
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
                wire.replace("eqiora.authored-form/v14", "eqiora.authored-form/v12")
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
