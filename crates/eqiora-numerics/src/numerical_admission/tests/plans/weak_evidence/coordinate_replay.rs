use super::*;
use eqiora_compiler::AuthoredFormExpressionV1 as E;

#[test]
fn zero_terms_cannot_hide_foreign_coordinate_support_or_factor() {
    let source = SOURCE
        .replace(
            "inner(eta,q*u)",
            "inner(eta,q*u)+0*inner(eta,s*coordinate(0)*u)",
        )
        .replace(
            "inner(trace(eta),g)",
            "inner(trace(eta),g+0*q*coordinate(0))",
        );
    resolve(&source).unwrap();
    let geometry = geometry(false);
    let (program, projection) = compile(&source, &geometry).unwrap();
    let domain = eqiora_core::Id::<eqiora_core::entity::kinds::Domain>::from_ulid(
        projection.domain_ulid().unwrap().parse().unwrap(),
    )
    .erase();
    let derived = crate::form_compiler::derive_candidate_with_dimension(&program, domain, 1)
        .unwrap()
        .unwrap();
    crate::form_compiler::admit_authored_scalar_primal_form(&projection, &program, &derived)
        .unwrap();
    let volume_term = &projection.equations()[0].1;
    let E::Add {
        right: boundary_term,
        ..
    } = &projection.equations()[0].2
    else {
        panic!("boundary sum");
    };
    let E::Integrate {
        domain_ulid: boundary,
        ..
    } = boundary_term.as_ref()
    else {
        panic!("boundary integral");
    };
    let volume = projection.domain_ulid().unwrap();
    // All replacements name live Domains; existence alone is insufficient.
    for (original, key, replacement) in [
        (volume_term, "support_ulid", boundary.as_str()),
        (volume_term, "factor_ulid", boundary.as_str()),
        (boundary_term.as_ref(), "support_ulid", volume),
        (boundary_term.as_ref(), "factor_ulid", boundary.as_str()),
    ] {
        let mut value = serde_json::to_value(original).unwrap();
        assert_eq!(forge(&mut value, key, replacement), 1);
        let changed: E = serde_json::from_value(value).unwrap();
        let text = std::str::from_utf8(projection.canonical_bytes())
            .unwrap()
            .replacen(
                &serde_json::to_string(original).unwrap(),
                &serde_json::to_string(&changed).unwrap(),
                1,
            );
        let forged = AuthoredFormulationProjection::decode(text.as_bytes()).unwrap();
        assert!(
            crate::form_compiler::admit_authored_scalar_primal_form(&forged, &program, &derived)
                .unwrap_err()
                .message()
                .contains("coordinate support"),
            "numeric cancellation erased foreign coordinate {key}"
        );
    }
}

fn forge(value: &mut serde_json::Value, key: &str, replacement: &str) -> usize {
    match value {
        serde_json::Value::Object(object)
            if object.get("kind").and_then(serde_json::Value::as_str) == Some("coordinate") =>
        {
            object.insert(key.into(), serde_json::Value::String(replacement.into()));
            1
        }
        serde_json::Value::Object(object) => object
            .values_mut()
            .map(|value| forge(value, key, replacement))
            .sum(),
        serde_json::Value::Array(values) => values
            .iter_mut()
            .map(|value| forge(value, key, replacement))
            .sum(),
        _ => 0,
    }
}
