//! Model declarations retain the same authored-form owner as Components.
use eqiora_compiler::{AuthoredFormulationProjection, CompiledModel};

const SOURCE: &str = r#"
space Spin=orthonormal(up,down);
model Wave(parameter h:1=2) {
 variable u:coordinates<complex<1>,Spin>;
 variable lambda:1;
 relation states {h*u=lambda*u;}
 form weak for states {
  test eta:1 for u;
  inner(eta,h*u)=inner(eta,lambda*u);
 }
}
"#;

#[test]
fn model_form_compiles_without_changing_original_model_meaning() {
    let original = format!("{} }}", SOURCE.split_once(" form weak").unwrap().0);
    let original =
        CompiledModel::compile_selected("model-form.eqi", &original, "Wave", &[]).unwrap();
    let model = CompiledModel::compile_selected("model-form.eqi", SOURCE, "Wave", &[]).unwrap();
    assert_eq!(model.transaction().ops(), original.transaction().ops());
    assert_eq!(model.authored_formulations().len(), 1);
    let form = model.authored_formulations().next().unwrap().projection();
    assert_eq!(
        form,
        &AuthoredFormulationProjection::decode(form.canonical_bytes()).unwrap()
    );
    let formatted = eqiora_lang::format(
        &eqiora_lang::parse("model-form.eqi", SOURCE)
            .into_document()
            .unwrap(),
    );
    let replayed =
        CompiledModel::compile_selected("model-form.eqi", &formatted, "Wave", &[]).unwrap();
    assert_eq!(
        form,
        replayed
            .authored_formulations()
            .next()
            .unwrap()
            .projection()
    );
}

#[test]
fn model_form_checks_the_same_conjugated_test_role() {
    let wrong = SOURCE.replace("inner(eta,h*u)", "inner(h*u,eta)");
    let errors = CompiledModel::compile_selected("model-form.eqi", &wrong, "Wave", &[])
        .expect_err("wrong conjugation must reject");
    assert!(format!("{errors:?}").contains("conjugate-linear test dependence"));
}

#[test]
fn selected_model_parameter_binding_retains_its_form_and_live_meaning() {
    let value = eqiora_lang::SourceAstFactory::expression(
        eqiora_lang::ExprKind::Number(eqiora_lang::DecimalLiteral::parse("3").unwrap()),
        eqiora_lang::TextRange::default(),
    )
    .unwrap();
    let bindings = [("h", eqiora_compiler::StaticBindingValue::Expression(&value))];
    let bound =
        CompiledModel::compile_selected("model-form.eqi", SOURCE, "Wave", &bindings).unwrap();
    let default = CompiledModel::compile_selected("model-form.eqi", SOURCE, "Wave", &[]).unwrap();
    let original = format!("{} }}", SOURCE.split_once(" form weak").unwrap().0);
    let original =
        CompiledModel::compile_selected("model-form.eqi", &original, "Wave", &bindings).unwrap();
    assert_eq!(bound.transaction().ops(), original.transaction().ops());
    assert_ne!(bound.transaction().ops(), default.transaction().ops());
    assert_eq!(bound.authored_formulations().len(), 1);
}
