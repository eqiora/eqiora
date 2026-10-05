use super::*;
use eqiora_compiler::{AuthoredFormExpressionV1 as E, AuthoredFormulationProjection};

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
        "integrate(body,bulk*conj(eta)*c+gradient*inner(grad(eta),grad(c)))",
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
