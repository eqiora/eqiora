use eqiora_core::entity::kinds;
use eqiora_core::{DimExponents, Id, ScalarDomain, ValueType};

use super::*;

fn projection() -> AuthoredFormulationProjection {
    let id = |value: &str| value.parse::<Ulid>().expect("fixed ULID");
    let expression = AuthoredFormExpression {
        kind: AuthoredFormExpressionKind::Number(1.0),
        value_type: ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS).unwrap(),
        support: None,
    };
    AuthoredFormulationProjection::encode_weak(
        "a".repeat(64),
        "weak".into(),
        Some(Id::<kinds::Domain>::from_ulid(id("01ARZ3NDEKTSV4RRFFQ69G5FAW")).erase()),
        vec![(
            "w".into(),
            "01ARZ3NDEKTSV4RRFFQ69G5FAX".into(),
            vec!["01ARZ3NDEKTSV4RRFFQ69G5FAY".into()],
            DimExponents::DIMENSIONLESS.exponents(),
            Some("h1".into()),
        )],
        vec![(
            "01ARZ3NDEKTSV4RRFFQ69G5FAV".into(),
            super::expression(&expression),
            super::expression(&expression),
        )],
    )
    .unwrap()
}

#[test]
fn derivation_request_preserves_choices_and_rejects_changed_effective_rows() {
    let mut wire = projection().wire;
    let relation = wire.equations[0].0.clone();
    let field = wire.trial_ulids[0].clone();
    let WireBinding::WeakTests { derivations, .. } = &mut wire.binding else {
        unreachable!()
    };
    derivations.push((relation.clone(), field.clone()));
    wire.equations.clear();
    let request =
        AuthoredFormulationProjection::decode(&serde_json::to_vec(&wire).unwrap()).unwrap();
    assert!(request.requests_weak_derivation());
    assert!(request.equations().is_empty());
    assert_eq!(request.equation_test_fields().unwrap().len(), 1);
    let equations = vec![(
        relation.clone(),
        AuthoredFormExpressionV1::Test { field_ulid: field },
        AuthoredFormExpressionV1::Number { value: 0. },
    )];
    let effective = request.with_derived_equations(equations.clone()).unwrap();
    assert!(effective.requests_weak_derivation());
    assert_eq!(effective.equations(), equations);
    assert_eq!(
        effective.equation_test_fields().unwrap(),
        request.equation_test_fields().unwrap()
    );
    let mut wrong = equations;
    wrong[0].1 = AuthoredFormExpressionV1::Number { value: 1. };
    assert!(request.with_derived_equations(wrong).is_err());
    let WireBinding::WeakTests { derivations, .. } = &mut wire.binding else {
        unreachable!()
    };
    derivations.push(derivations[0].clone());
    assert!(AuthoredFormulationProjection::decode(&serde_json::to_vec(&wire).unwrap()).is_err());
}

#[test]
fn test_dimension_round_trip_rejects_noncanonical_exponents() {
    let mut wire = projection().wire;
    let length = [(0, 1), (1, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)];
    if let WireBinding::WeakTests { tests, .. } = &mut wire.binding {
        tests[0].3 = length;
    }
    let admitted =
        AuthoredFormulationProjection::decode(&serde_json::to_vec(&wire).unwrap()).unwrap();
    assert_eq!(admitted.test_restrictions()[0].3, length);
    for exponent in [(2, 2), (1, 0), (-1, -1)] {
        if let WireBinding::WeakTests { tests, .. } = &mut wire.binding {
            tests[0].3[1] = exponent;
        }
        assert!(
            AuthoredFormulationProjection::decode(&serde_json::to_vec(&wire).unwrap())
                .unwrap_err()
                .message()
                .contains("canonical rational SI")
        );
    }
}

#[test]
fn conditional_implication_rejects_changed_or_missing_hypotheses() {
    let projection = projection();
    let text = String::from_utf8(projection.canonical_bytes().to_vec()).unwrap();
    for changed in [
        text.replace("strong-implies-weak", "weak-implies-strong"),
        text.replace("fixed-domain", "moving-domain"),
    ] {
        assert!(
            AuthoredFormulationProjection::decode(changed.as_bytes())
                .unwrap_err()
                .message()
                .contains("hypotheses")
        );
    }
}

#[test]
fn plural_codec_rejects_duplicate_test_names_and_old_schema() {
    let mut wire = projection().wire;
    wire.trial_ulids.push("01ARZ3NDEKTSV4RRFFQ69G5FAZ".into());
    wire.equations.push((
        "01ARZ3NDEKTSV4RRFFQ69G5FB0".into(),
        AuthoredFormExpressionV1::Number { value: 0.0 },
        AuthoredFormExpressionV1::Number { value: 0.0 },
    ));
    if let WireBinding::WeakTests { tests, .. } = &mut wire.binding {
        tests.push((
            "q".into(),
            "01ARZ3NDEKTSV4RRFFQ69G5FAZ".into(),
            vec![],
            DimExponents::DIMENSIONLESS.exponents(),
            Some("h1".into()),
        ));
    }
    wire.assumptions = AuthoredFormulationProjection::required_assumptions()
        .iter()
        .map(|s| (*s).into())
        .collect();
    assert!(AuthoredFormulationProjection::decode(&serde_json::to_vec(&wire).unwrap()).is_ok());
    if let WireBinding::WeakTests { tests, .. } = &mut wire.binding {
        tests[1].0 = tests[0].0.clone();
    }
    assert!(
        AuthoredFormulationProjection::decode(&serde_json::to_vec(&wire).unwrap())
            .unwrap_err()
            .message()
            .contains("unique")
    );
    let bytes = projection().canonical_bytes().to_vec();
    for schema in [
        "eqiora.authored-form/v18",
        "eqiora.authored-form/v17",
        "eqiora.authored-scalar-form/v3",
    ] {
        let old = String::from_utf8(bytes.clone())
            .unwrap()
            .replace("eqiora.authored-form/v19", schema);
        assert!(AuthoredFormulationProjection::decode(old.as_bytes()).is_err());
    }
}

#[test]
fn one_codec_owns_canonical_round_trip_and_fail_closed_decode() {
    let projection = projection();
    let bytes = projection.canonical_bytes();
    assert_eq!(
        AuthoredFormulationProjection::decode(bytes).unwrap(),
        projection
    );

    let mut trailing = bytes.to_vec();
    trailing.push(b' ');
    assert!(AuthoredFormulationProjection::decode(&trailing).is_err());

    let unknown = String::from_utf8(bytes.to_vec())
        .unwrap()
        .replace("\"equations\"", "\"unknown\":0,\"equations\"");
    assert!(AuthoredFormulationProjection::decode(unknown.as_bytes()).is_err());

    let malformed_identity = String::from_utf8(bytes.to_vec())
        .unwrap()
        .replace("01ARZ3NDEKTSV4RRFFQ69G5FAV", "not-a-canonical-ulid-value");
    assert!(AuthoredFormulationProjection::decode(malformed_identity.as_bytes()).is_err());

    assert!(AuthoredFormulationProjection::decode(&vec![b' '; MAX_BYTES + 1]).is_err());
}
