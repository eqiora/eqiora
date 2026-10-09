//! Global weak tests retain finite coordinate types without a fabricated Domain.
use eqiora_compiler::{AuthoredFormulationProjection, CompiledModel};

const SOURCE: &str = r#"
space Spin=orthonormal(up,down);
public component Wave() {
 parameter h:1=2;
 variable u:coordinates<complex<1>,Spin>;
 variable lambda:1;
 relation states { h*u=lambda*u; }
 form weak for states {
  test eta:1 for u;
  inner(eta,h*u)=inner(eta,lambda*u);
 }
}
"#;

fn compile(source: &str) -> Result<CompiledModel, Vec<eqiora_core::Diagnostic>> {
    CompiledModel::compile_selected("global-weak.eqi", source, "Wave", &[])
}

#[test]
fn global_coordinate_weak_form_preserves_model_test_type_and_canonical_replay() {
    let compiled = compile(SOURCE).unwrap();
    let plain = compile(&format!(
        "{} }}",
        &SOURCE[..SOURCE.find(" form weak").unwrap()]
    ))
    .unwrap();
    assert_eq!(compiled.transaction().ops(), plain.transaction().ops());
    let form = compiled.authored_formulations().next().unwrap();
    assert_eq!(form.domain(), None);
    let projection = form.projection();
    assert_eq!(projection.domain_ulid(), None);
    assert_eq!(projection.finite_space(), None);
    assert_eq!(projection.test_restrictions().len(), 1);
    assert!(projection.test_restrictions()[0].2.is_empty());
    assert_eq!(
        projection.test_restrictions()[0].1,
        projection.trial_ulids()[0]
    );
    assert_eq!(
        projection,
        &AuthoredFormulationProjection::decode(projection.canonical_bytes()).unwrap()
    );
    let formatted = eqiora_lang::format(
        &eqiora_lang::parse("global-weak.eqi", SOURCE)
            .into_document()
            .unwrap(),
    );
    let again = compile(&formatted).unwrap();
    assert_eq!(
        projection,
        again.authored_formulations().next().unwrap().projection()
    );
    let retired = String::from_utf8(projection.canonical_bytes().to_vec())
        .unwrap()
        .replace("eqiora.authored-form/v13", "eqiora.authored-form/v11");
    assert!(AuthoredFormulationProjection::decode(retired.as_bytes()).is_err());
}

#[test]
fn global_weak_typing_rejects_wrong_conjugation_basis_and_spatial_operations() {
    for source in [
        SOURCE.replace("inner(eta,h*u)", "inner(h*u,eta)"),
        SOURCE.replace("inner(eta,h*u)", "inner(eta,math.conj(h*u))"),
        SOURCE.replace("test eta:1 for u;", "test eta:1 for u zero_on boundary;"),
        SOURCE.replace("inner(eta,h*u)", "inner(grad(eta),grad(h*u))"),
        SOURCE
            .replace(
                "space Spin=orthonormal(up,down);",
                "space Spin=orthonormal(up,down); space Other=orthonormal(first,second);",
            )
            .replace(
                "variable lambda:1;",
                "variable lambda:1; variable v:coordinates<complex<1>,Other>;",
            )
            .replace("inner(eta,h*u)", "inner(eta,v)"),
    ] {
        assert!(compile(&source).is_err(), "{source}");
    }
}

#[test]
fn finite_coordinate_scaling_uses_shared_value_type_rules() {
    for source in [
        SOURCE.replace("h*u", "u*h"),
        SOURCE.replace("h*u", "u/h"),
        SOURCE.replace(
            "parameter h:1=2;",
            "parameter h:complex<1>=math.complex(2,3);",
        ),
        SOURCE.replace("coordinates<complex<1>,Spin>", "coordinates<1,Spin>"),
    ] {
        compile(&source).unwrap_or_else(|errors| panic!("{source}: {errors:?}"));
    }
}

#[test]
fn finite_map_application_retains_complex_coefficients_and_nominal_endpoints() {
    let source = SOURCE.replace("parameter h:1=2;",
        "parameter h:map<complex<1>,Spin,Spin>=linear_map(Spin,Spin,[[2,math.complex(0,-1)],[math.complex(0,1),2]]);")
        .replace("h*u", "apply(h,u)");
    let compiled = compile(&source).unwrap();
    let form = compiled
        .authored_formulations()
        .next()
        .unwrap()
        .projection();
    assert_eq!(
        form,
        &AuthoredFormulationProjection::decode(form.canonical_bytes()).unwrap()
    );
    assert!(
        String::from_utf8(form.canonical_bytes().to_vec())
            .unwrap()
            .contains("apply")
    );
    for wrong in [
        source.replace("inner(eta,apply(h,u))", "inner(apply(h,u),eta)"),
        source.replace("inner(eta,apply(h,u))", "inner(eta,apply(h,math.conj(u)))"),
        source.replace("inner(eta,apply(h,u))", "inner(eta,apply(u,h))"),
        source
            .replace(
                "space Spin=orthonormal(up,down);",
                "space Spin=orthonormal(up,down); space Other=orthonormal(first,second);",
            )
            .replace(
                "variable lambda:1;",
                "variable lambda:1; variable v:coordinates<complex<1>,Other>;",
            )
            .replace("inner(eta,apply(h,u))", "inner(eta,apply(h,v))"),
    ] {
        assert!(compile(&wrong).is_err(), "{wrong}");
    }
}
