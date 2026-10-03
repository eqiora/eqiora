use super::identity;

#[test]
fn complete_let_type_annotations_change_source_identity() {
    let real = "model m() { let x: m = 0; }";
    let complex = "model m() { let x: complex<m> = 0; }";
    let array = "model m() { let x: array<complex<m>, 3> = 0; }";
    assert_ne!(identity(real), identity(complex));
    assert_ne!(identity(complex), identity(array));
}

#[test]
fn source_structure_has_exact_identity() {
    let base = "model m() { parameter p: m = 2; let k: 1 / m = math.pi / p; }";
    let reformatted = "model m() {\n parameter p: m = 2;\n let k: 1/m = math.pi/p;\n}";
    let renamed = "model m() { parameter p: m = 2; let wave: 1 / m = math.pi / p; }";
    let changed = "model m() { parameter p: m = 2; let k: 1 / m = 2 / p; }";

    assert_eq!(identity(base), identity(reformatted));
    assert_ne!(identity(base), identity(renamed));
    assert_ne!(identity(base), identity(changed));
}

#[test]
fn omitted_dimension_has_distinct_deterministic_identity() {
    let annotated = "model m() { parameter p: m = 2; let k: 1 / m = math.pi / p; }";
    let inferred = "model m() { parameter p: m = 2; let k = math.pi / p; }";
    let reformatted = "model m() {\n parameter p: m = 2;\n let k=math.pi/p;\n}";

    assert_ne!(identity(annotated), identity(inferred));
    assert_eq!(identity(inferred), identity(reformatted));
}

#[test]
fn support_assertions_change_identity_in_both_containers() {
    for container in ["model M()", "component C()"] {
        let omitted = format!("{container} {{ let q: m = value; }}");
        let asserted = format!("{container} {{ let q: m on body = value; }}");
        let other = format!("{container} {{ let q: m on other = value; }}");
        assert_ne!(identity(&omitted), identity(&asserted));
        assert_ne!(identity(&asserted), identity(&other));
        let document = eqiora_lang::parse("asserted.eqi", &asserted)
            .into_document()
            .unwrap();
        assert_eq!(
            identity(&asserted),
            identity(&eqiora_lang::format(&document))
        );
    }
}

#[test]
fn omitted_activation_keeps_the_existing_record_contract() {
    use crate::source_identity::{
        Budget, Encoder, LocalSourceIdentityLimits, compile_time, encode_expression, encode_name,
        value_type,
    };
    for (annotation, support) in [("", ""), (": m", ""), ("", " on body"), (": m", " on body")] {
        let source = format!("model M() {{ let q{annotation}{support} = value; }}");
        let document = eqiora_lang::parse("omitted.eqi", &source)
            .into_document()
            .unwrap();
        let eqiora_lang::Item::Let(alias) = &document.models()[0].items()[0] else {
            panic!("let")
        };
        let limits = LocalSourceIdentityLimits::default();
        let mut actual = Encoder::new(limits.max_canonical_bytes);
        compile_time::encode_let(&mut actual, alias, &mut Budget::new(limits)).unwrap();
        // Existing alias record: name=1, optional type=2, expression=3, optional support=4.
        // Absence of the new assertion must emit no tag or sentinel.
        let mut expected = Encoder::new(limits.max_canonical_bytes);
        let mut budget = Budget::new(limits);
        expected
            .field(1, |encoder| encode_name(encoder, "q", &mut budget))
            .unwrap();
        if let Some(value_type) = alias.value_type() {
            expected
                .field(2, |encoder| {
                    value_type::encode_value_type(encoder, value_type, &mut budget, 1)
                })
                .unwrap();
        }
        expected
            .field(3, |encoder| {
                encode_expression(encoder, alias.value(), &mut budget, 1)
            })
            .unwrap();
        if !support.is_empty() {
            expected
                .field(4, |encoder| encode_name(encoder, "body", &mut budget))
                .unwrap();
        }
        assert_eq!(actual.finish().unwrap(), expected.finish().unwrap());
    }
}

#[test]
fn activation_assertions_change_identity_in_both_containers() {
    for container in ["model M()", "component C()"] {
        for support in ["", " on body"] {
            let omitted = format!("{container} {{ let q: m{support} = value; }}");
            let asserted = format!("{container} {{ let q: m{support} at sample = value; }}");
            let other = format!("{container} {{ let q: m{support} at other = value; }}");
            assert_ne!(identity(&omitted), identity(&asserted));
            assert_ne!(identity(&asserted), identity(&other));
            let document = eqiora_lang::parse("asserted.eqi", &asserted)
                .into_document()
                .unwrap();
            assert_eq!(
                identity(&asserted),
                identity(&eqiora_lang::format(&document))
            );
        }
    }
}

#[test]
fn arrays_and_indices_retain_order_shape_and_selection_identity() {
    for (left, right) in [
        ("[1, 2]", "[2, 1]"),
        ("[1, 2]", "[[1, 2]]"),
        ("samples[0]", "samples[1]"),
        ("samples[0]", "other[0]"),
        ("samples[0][1]", "samples[1][0]"),
        ("10[ms]", "(10)[ms]"),
    ] {
        let left = format!("model M() {{ let x = {left}; }}");
        let right = format!("model M() {{ let x = {right}; }}");
        assert_ne!(identity(&left), identity(&right));
        let document = eqiora_lang::parse("array.eqi", &left)
            .into_document()
            .unwrap();
        assert_eq!(identity(&left), identity(&eqiora_lang::format(&document)));
    }
}

#[test]
fn parameter_expression_identity_matches_native_factory_and_preserves_signed_literals() {
    use eqiora_lang::{ExprKind, Item, SourceAstFactory, TextRange, VisibilitySyntax};
    for initializer in ["[1, 2]", "math.complex(1, 2)", "-2", "-2[V]"] {
        let source = format!("model M() {{ parameter p: V = {initializer}; }}");
        let document = eqiora_lang::parse("parameter.eqi", &source)
            .into_document()
            .unwrap();
        let Item::Parameter(parameter) = &document.models()[0].items()[0] else {
            panic!("parameter")
        };
        let range = TextRange::new(0, 0);
        let value = if initializer == "-2" {
            // Old native signed-literal representation retains the same identity.
            SourceAstFactory::expression(
                ExprKind::Number(
                    eqiora_lang::DecimalLiteral::parse("-2.0").expect("exact literal"),
                ),
                range,
            )
            .unwrap()
        } else if initializer == "-2[V]" {
            let unit = SourceAstFactory::expression(ExprKind::Name("V".into()), range).unwrap();
            SourceAstFactory::expression(
                ExprKind::Quantity {
                    value: eqiora_lang::DecimalLiteral::parse("-2").unwrap(),
                    unit: Box::new(unit),
                },
                range,
            )
            .unwrap()
        } else {
            parameter.value().clone()
        };
        let parameter =
            SourceAstFactory::parameter("p", parameter.value_type().clone(), value, range).unwrap();
        let model = SourceAstFactory::model(
            VisibilitySyntax::Private,
            "M",
            Vec::new(),
            vec![Item::Parameter(parameter)],
            range,
        )
        .unwrap();
        let native = SourceAstFactory::document(Vec::new(), vec![], vec![], vec![model]).unwrap();
        assert_eq!(
            crate::source_identity::LocalSourceIdentity::from_document(&native).unwrap(),
            identity(&source)
        );
    }
}

#[test]
fn quantity_identity_preserves_exact_decimals_before_numerical_rounding() {
    let source = |literal: &str| format!("model M() {{ parameter p: m = {literal}[nm]; }}");
    assert_eq!(identity(&source("0.1")), identity(&source("10e-2")));
    // These distinct exact decimals round to the same unscaled binary64.
    assert_ne!(
        identity(&source("0.1")),
        identity(&source("0.100000000000000001")),
    );
}

#[test]
fn negative_dimensioned_constructor_values_match_native_and_formatted_identity() {
    use eqiora_core::{DimExponents, ScalarDomain, ValueLiteral, ValueType};
    use eqiora_lang::{Item, SourceAstFactory, TextRange, VisibilitySyntax};
    let dimension = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let complex =
        ValueType::scalar(ScalarDomain::Complex, dimension).expect("admitted numeric scalar type");
    for (value, initializer, annotation) in [
        (
            ValueLiteral::new(complex.clone(), [(1.0, -2.0)]).unwrap(),
            "math.complex(1[m], -2[m])",
            "complex<m>",
        ),
        (
            ValueLiteral::new(complex.array(2).unwrap(), [(1.0, -2.0), (-3.0, 4.0)]).unwrap(),
            "[math.complex(1[m], -2[m]), math.complex(-3[m], 4[m])]",
            "array<complex<m>, 2>",
        ),
    ] {
        let source = format!("model M() {{ parameter p: {annotation} = {initializer}; }}");
        let parsed = eqiora_lang::parse("signed.eqi", &source)
            .into_document()
            .unwrap();
        let Item::Parameter(parameter) = &parsed.models()[0].items()[0] else {
            panic!("parameter")
        };
        let range = TextRange::new(0, 0);
        let expression =
            SourceAstFactory::value_literal(&value, None, range, |_| None, |_| None, |_| None)
                .unwrap();
        let parameter =
            SourceAstFactory::parameter("p", parameter.value_type().clone(), expression, range)
                .unwrap();
        let model = SourceAstFactory::model(
            VisibilitySyntax::Private,
            "M",
            Vec::new(),
            vec![Item::Parameter(parameter)],
            range,
        )
        .unwrap();
        let native = SourceAstFactory::document(Vec::new(), vec![], vec![], vec![model]).unwrap();
        let native_identity =
            crate::source_identity::LocalSourceIdentity::from_document(&native).unwrap();
        assert_eq!(native_identity, identity(&source));
        assert_eq!(native_identity, identity(&eqiora_lang::format(&native)));
    }
    // Negation of an expression remains structural; this is not algebraic folding.
    assert_ne!(
        identity("model M() { let x=-(1[m]+2[m]); }"),
        identity("model M() { let x=-3[m]; }")
    );
}

#[test]
fn exact_numeric_identity_retains_adjacent_large_decimals_and_literal_sign() {
    assert_ne!(
        identity("model M() { let n=9007199254740993; }"),
        identity("model M() { let n=9007199254740992; }")
    );
    assert_eq!(
        identity("model M() { let n=9007199254740993; }"),
        identity("model M() { let n=9007199254740993.0; }")
    );
    assert_eq!(
        identity("model M() { let n=-0; }"),
        identity("model M() { let n=0; }")
    );
}

#[test]
fn predicates_and_equation_sides_have_unambiguous_source_identity() {
    let specimens = [
        "true", "false", "0", "1", "a == b", "a != b", "a < b", "a <= b", "a > b", "a >= b",
        "a and b", "a or b", "not a", "-a",
    ];
    let mut identities = std::collections::BTreeSet::new();
    for expression in specimens {
        let source = format!("model M() {{ let p = {expression}; }}");
        assert!(identities.insert(identity(&source)), "{expression}");
        let document = eqiora_lang::parse("predicate.eqi", &source)
            .into_document()
            .unwrap();
        assert_eq!(identity(&source), identity(&eqiora_lang::format(&document)));
    }
    assert_ne!(
        identity("model M() { relation r { a=b; } }"),
        identity("model M() { relation r { a==b=true; } }")
    );
    assert_ne!(
        identity("model M() { relation r { a=b; } }"),
        identity("model M() { relation r { b=a; } }")
    );
    // Not is not the literal-sign canonicalization used by unary minus.
    assert_ne!(
        identity("model M() { let p=not 1[s]; }"),
        identity("model M() { let p=-1[s]; }")
    );
}
