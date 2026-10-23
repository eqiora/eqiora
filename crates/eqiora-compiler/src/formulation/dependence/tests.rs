use super::*;
use AuthoredFormExpressionV1 as E;

fn id(value: u128) -> String {
    ulid::Ulid::from(value).to_string()
}

fn form(count: u128, reversed: bool, repeated: bool) -> AuthoredFormulationProjection {
    let mut tests = (1..=count)
        .map(|i| {
            (
                format!("w{i}"),
                id(i),
                Vec::new(),
                DimExponents::DIMENSIONLESS.exponents(),
                Some("h1".into()),
            )
        })
        .collect::<Vec<_>>();
    let mut equations = (1..=count)
        .map(|i| {
            (
                id(100 + i),
                E::Mul {
                    left: Box::new(E::Test {
                        field_ulid: id(if repeated { 1 } else { i }),
                    }),
                    right: Box::new(E::Field {
                        ulid: id(i % count + 1),
                    }),
                },
                E::Number { value: 0. },
            )
        })
        .collect::<Vec<_>>();
    if reversed {
        tests.reverse();
        equations.rotate_left(1);
    }
    AuthoredFormulationProjection::encode_weak(
        "0".repeat(64),
        "mixed".into(),
        Some(Id::<kinds::Domain>::from_ulid(ulid::Ulid::from(100)).erase()),
        tests,
        equations,
    )
    .unwrap()
}

#[test]
fn equation_tests_follow_arguments_independently_of_order_and_cross_field_values() {
    let first = form(3, false, false).equation_test_fields().unwrap();
    let reordered = form(3, true, false).equation_test_fields().unwrap();
    assert_eq!(first, reordered);
    for i in 1..=3 {
        let relation = Id::<kinds::Relation>::from_ulid(ulid::Ulid::from(100 + i)).erase();
        assert_eq!(first[&relation], field_id(&id(i)).unwrap());
    }
}

#[test]
fn repeated_test_rows_reject_instead_of_using_declaration_order() {
    let error = form(3, false, true).equation_test_fields().unwrap_err();
    assert!(error.message().contains("repeat a test Field"));
}

#[test]
fn row_count_is_bounded_by_the_shared_projection_budget_not_a_physics_profile() {
    let projection = form(12, true, false);
    let decoded = AuthoredFormulationProjection::decode(projection.canonical_bytes()).unwrap();
    assert_eq!(decoded.equation_test_fields().unwrap().len(), 12);
    assert_eq!(
        decoded.assumptions(),
        AuthoredFormulationProjection::required_assumptions()
    );
}
