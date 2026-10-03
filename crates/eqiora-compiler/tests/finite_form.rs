//! Finite authored equations retain global coordinates without fabricated Geometry.
use eqiora_compiler::{AuthoredFormulationProjection, CompiledModel, StaticBindingValue};

const SOURCE: &str = r#"public component Network(parameter g:1, parameter i1:1, parameter i2:1, parameter offset:1) {
    variable v1:1; variable v2:1;
    relation first { g*(v1-v2)=i1; }
    relation second { g*(v2-v1)=i2; }
    form floating for first, second {
        finite voltage(v1,v2);
        gauge voltage { reference v1=offset; compatibility i1+i2=0; }
        g*(v1-v2)=i1;
        g*(v2-v1)=i2;
    }
}"#;
fn compile(source: &str) -> Result<CompiledModel, Vec<eqiora_core::Diagnostic>> {
    let values = eqiora_lang::parse(
        "values.eqi",
        "model V(){parameter g:1=2;parameter i1:1=6;parameter i2:1=-6;parameter offset:1=4;}",
    )
    .into_document()
    .unwrap();
    let bindings = values.models()[0]
        .items()
        .iter()
        .filter_map(|item| {
            if let eqiora_lang::Item::Parameter(p) = item {
                Some((p.name(), StaticBindingValue::Expression(p.value())))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    CompiledModel::compile_selected("network.eqi", source, "Network", &bindings)
}
#[test]
fn finite_reference_preserves_model_and_ordered_global_projection() {
    let compiled = compile(SOURCE).unwrap();
    let plain = compile(&format!(
        "{} }}",
        &SOURCE[..SOURCE.find("    form floating").unwrap()]
    ));
    let plain = plain.unwrap();
    assert_eq!(compiled.transaction().ops(), plain.transaction().ops());
    let form = compiled.authored_formulations().next().unwrap();
    assert_eq!(form.domain(), None);
    assert_eq!(form.projection().domain_ulid(), None);
    assert_eq!(form.projection().finite_space(), Some("voltage"));
    assert_eq!(
        form.projection().gauge_field_ulids(),
        Some(form.projection().trial_ulids())
    );
    assert_eq!(
        form.projection(),
        &AuthoredFormulationProjection::decode(form.projection().canonical_bytes()).unwrap()
    );
    let formatted = eqiora_lang::format(
        &eqiora_lang::parse("network.eqi", SOURCE)
            .into_document()
            .unwrap(),
    );
    let again = compile(&formatted).unwrap();
    assert_eq!(
        form.projection(),
        again.authored_formulations().next().unwrap().projection()
    );
}
#[test]
fn wrong_finite_binder_or_original_equation_is_rejected() {
    let with_foreign = SOURCE
        .replace("variable v2:1;", "variable v2:1; variable other:1;")
        .replace("reference v1=offset", "reference other=offset");
    assert!(compile(&with_foreign).is_err());
    for (from, to) in [
        ("finite voltage(v1,v2)", "finite voltage(v1,v1)"),
        ("gauge voltage", "gauge v1"),
        ("finite voltage(v1,v2)", "finite v1(v1,v2)"),
        ("        g*(v2-v1)=i2;", "        g*(v2-v1)=i1;"),
        ("reference v1=offset", "reference integrate(v1,v1)=offset"),
    ] {
        assert!(compile(&SOURCE.replace(from, to)).is_err(), "{to}");
    }
}
