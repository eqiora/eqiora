use eqiora_compiler::{AuthoredFormulationProjection, CompiledModel};

fn source(space: &str, integrand: &str) -> String {
    format!(
        r#"model M() {{
        domain body=box(0,1,0,1,0,1);
        domain face=boundary(body,axis=0,side=lower);
        variable u:vector<1,3> on body;
        relation law on body {{ curl(curl(u))=u*0[1/m^2]; }}
        form weak for law {{ test v:1 for u {space}; integrate(body,{integrand})=0; }}
    }}"#
    )
}

#[test]
fn continuum_test_spaces_retain_their_declared_derivatives_and_exact_identity() {
    let mut identities = Vec::new();
    for (space, expression) in [
        ("h1", "dot(curl(v),curl(u))"),
        ("hcurl", "dot(curl(v),curl(u))"),
        ("hdiv", "div(v)*div(u)"),
        ("l2", "dot(v,u)"),
    ] {
        let model = CompiledModel::compile_selected(
            "space.eqi",
            &source(&format!("in {space}"), expression),
            "M",
            &[],
        )
        .unwrap();
        let form = model.authored_formulations().next().unwrap().projection();
        assert_eq!(form.test_restrictions()[0].4.as_deref(), Some(space));
        assert_eq!(
            &AuthoredFormulationProjection::decode(form.canonical_bytes()).unwrap(),
            form
        );
        identities.push(form.source_identity().to_owned());
        let old = String::from_utf8(form.canonical_bytes().to_vec())
            .unwrap()
            .replace("eqiora.authored-form/v15", "eqiora.authored-form/v13");
        assert!(AuthoredFormulationProjection::decode(old.as_bytes()).is_err());
    }
    // The first two have identical equations but different conditional spaces.
    assert_ne!(identities[0], identities[1]);
    for (space, expression, diagnostic) in [
        (
            "in hcurl",
            "frobenius(grad(v),grad(u))",
            "declared regularity",
        ),
        ("in hdiv", "dot(curl(v),curl(u))", "declared regularity"),
        ("in l2", "div(v)*div(u)", "declared regularity"),
        (
            "in hcurl zero_on face",
            "dot(curl(v),curl(u))",
            "full zero_on",
        ),
        ("in smooth", "dot(v,u)", "spatial test regularity"),
    ] {
        let errors =
            CompiledModel::compile_selected("invalid.eqi", &source(space, expression), "M", &[])
                .unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.message().contains(diagnostic)),
            "{errors:?}"
        );
    }
    // Changing only the declared space cannot make an unsupported derivative legal
    // when the projection is decoded independently of its source compilation.
    let model = CompiledModel::compile_selected(
        "h1.eqi",
        &source("in h1", "frobenius(grad(v),grad(u))"),
        "M",
        &[],
    )
    .unwrap();
    let form = model.authored_formulations().next().unwrap().projection();
    let forged = String::from_utf8(form.canonical_bytes().to_vec())
        .unwrap()
        .replace("\"h1\"", "\"hcurl\"");
    assert!(
        AuthoredFormulationProjection::decode(forged.as_bytes())
            .unwrap_err()
            .message()
            .contains("declared regularity")
    );
}
