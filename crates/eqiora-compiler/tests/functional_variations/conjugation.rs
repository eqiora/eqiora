use super::*;
use eqiora_compiler::{AuthoredFormExpressionV1 as E, AuthoredFormulationProjection};

fn complex_form(expression: &str) -> Result<CompiledModel, Vec<eqiora_core::Diagnostic>> {
    let source = format!(
        r#"
public component Energy(
    support body:volume(ambient_dimension=1),
    support left:boundary(parent=body),
    support right:boundary(parent=body),
    parameter bulk:J/m,
    parameter gradient:J*m,
    parameter phase:complex<1>=math.complex(0,1)
) {{
    variable c:complex<1> on body;
    relation stationarity on body {{ bulk*phase*c-div(gradient*grad(c))=0; }}
    relation left_value on left {{ trace(c)=0; }}
    relation right_value on right {{ trace(c)=0; }}
    form stationary for stationarity {{
        test eta:1 for c zero_on left,right;
        {expression}=0;
    }}
}}
"#
    );
    compile_source_with_values(
        &source,
        &geometry(),
        "model Values(){parameter bulk:1=2;parameter gradient:1=3;parameter phase:complex<1>=math.complex(0,1);}",
    )
}

#[test]
fn complex_scalar_domains_survive_authored_operators() {
    let compiled =
        complex_form("integrate(body,bulk*inner(eta,phase*c)+gradient*inner(grad(eta),grad(c)))")
            .unwrap_or_else(|errors| panic!("{errors:?}"));
    let form = compiled
        .authored_formulations()
        .next()
        .unwrap()
        .projection();
    assert_eq!(
        AuthoredFormulationProjection::decode(form.canonical_bytes()).unwrap(),
        *form
    );
    // math.complex requires real inputs. Equal operands isolate scalar-domain
    // rejection from dimension, shape and support mismatch.
    for expression in [
        "c*2",
        "2*c",
        "c/2",
        "2/c",
        "c+2",
        "2-c",
        "c^2",
        "math.sin(c)",
        "math.conj(c)",
        "inner(eta,c)",
        "phase",
        "inner(grad(eta),grad(c))",
        "math.complex(1,2)",
    ] {
        let source = format!("integrate(body,math.complex({expression},{expression}))");
        let errors = complex_form(&source).unwrap_err();
        assert!(
            errors.iter().any(|error| error
                .message()
                .contains("math.complex requires two equally dimensioned real scalars")),
            "{expression}: {errors:?}"
        );
    }
    let integral = "integrate(body,inner(eta,c))";
    assert!(
        complex_form(&format!("math.complex({integral},{integral})"))
            .unwrap_err()
            .iter()
            .any(|error| error
                .message()
                .contains("math.complex requires two equally dimensioned real scalars"))
    );
}

#[test]
fn authored_pairing_retains_conjugation_and_argument_order_in_current_wire() {
    let geometry = geometry();
    let plain = compile(
        "integrate(body,bulk*eta*c+gradient*dot(grad(eta),grad(c)))",
        &geometry,
    );
    let inner = compile(
        "integrate(body,bulk*inner(eta,c)+gradient*inner(grad(eta),grad(c)))",
        &geometry,
    );
    let conjugate = compile(
        "integrate(body,bulk*math.conj(eta)*c+gradient*inner(grad(eta),grad(c)))",
        &geometry,
    );
    assert_eq!(plain.transaction().ops(), inner.transaction().ops());
    assert_eq!(plain.transaction().ops(), conjugate.transaction().ops());
    let projection = |model: &CompiledModel| {
        model
            .authored_formulations()
            .next()
            .unwrap()
            .projection()
            .clone()
    };
    let (plain, inner, conjugate) = (
        projection(&plain),
        projection(&inner),
        projection(&conjugate),
    );
    assert_ne!(plain.canonical_bytes(), inner.canonical_bytes());
    assert_ne!(inner.canonical_bytes(), conjugate.canonical_bytes());
    let [(_, E::Integrate { integrand, .. }, _)] = inner.equations() else {
        panic!("one retained integral");
    };
    let E::Add { left, .. } = integrand.as_ref() else {
        panic!("mass and gradient terms");
    };
    let E::Mul { right, .. } = left.as_ref() else {
        panic!("coefficient times inner product");
    };
    let E::Inner { left, right } = right.as_ref() else {
        panic!("explicit inner product");
    };
    let (E::Test { field_ulid }, E::Field { ulid }) = (left.as_ref(), right.as_ref()) else {
        panic!("the conjugated first argument is the test, second is the trial");
    };
    assert_eq!(field_ulid, ulid);
    for form in [&inner, &conjugate] {
        assert_eq!(
            AuthoredFormulationProjection::decode(form.canonical_bytes()).unwrap(),
            *form
        );
        let text = std::str::from_utf8(form.canonical_bytes()).unwrap();
        assert!(text.contains("\"inner\""));
        assert!(
            AuthoredFormulationProjection::decode(
                text.replace("eqiora.authored-form/v10", "eqiora.authored-form/v9")
                    .as_bytes()
            )
            .is_err()
        );
    }
    assert!(
        std::str::from_utf8(conjugate.canonical_bytes())
            .unwrap()
            .contains("\"conjugate\"")
    );
    assert!(
        try_compile_profile(
            "integrate(body,bulk*inner(eta,grad(c)))",
            &geometry,
            "1",
            "J/m",
            "J*m",
            "1",
            ""
        )
        .is_err()
    );
}
