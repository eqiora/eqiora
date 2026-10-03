use eqiora_lang::{BinaryOp, DraftExpression, ExprKind, Item, UnaryOp, format, parse};

#[test]
fn predicates_preserve_precedence_and_equation_delimiter() {
    let source = "model M(parameter enabled: bool = true) { let p = not 1 < 2 and false or true; relation r { enabled = 1 == 2; } }";
    let document = parse("bool.eqi", source).into_document().unwrap();
    let Item::Let(value) = &document.models()[0].items()[0] else {
        panic!("let")
    };
    let ExprKind::Binary {
        op: BinaryOp::Or,
        left,
        ..
    } = value.value().kind()
    else {
        panic!("or")
    };
    let ExprKind::Binary {
        op: BinaryOp::And,
        left,
        ..
    } = left.kind()
    else {
        panic!("and")
    };
    assert!(matches!(
        left.kind(),
        ExprKind::Unary {
            op: UnaryOp::Not,
            ..
        }
    ));
    let rendered = format(&document);
    assert_eq!(
        format(&parse("again.eqi", &rendered).into_document().unwrap()),
        rendered
    );
    assert!(
        parse("missing.eqi", "model M() { relation r { true == false; } }")
            .into_document()
            .is_err()
    );
    assert!(
        parse("chain.eqi", "model M() { let p = 1 < 2 < 3; }")
            .into_document()
            .is_err()
    );
}

#[test]
fn tight_nominal_type_assignment_and_comparisons_coexist() {
    let source = "model M() { indexset Rows=range(3); parameter i:index<Rows>=index(Rows,2); let p = 2>=1 and 1<=2 and 1!=2; let q = (1 < 2) == true; }";
    let document = parse("tight.eqi", source).into_document().unwrap();
    let rendered = format(&document);
    assert_eq!(
        format(&parse("again.eqi", &rendered).into_document().unwrap()),
        rendered
    );
}

#[test]
fn native_boolean_is_not_a_numeric_literal() {
    let expression = DraftExpression::boolean(true)
        .logical_and(DraftExpression::boolean(false).logical_not())
        .source_ast(|_| None, |_| None)
        .unwrap();
    assert!(matches!(
        expression.kind(),
        ExprKind::Binary {
            op: BinaryOp::And,
            ..
        }
    ));
    let literal = eqiora_core::ValueLiteral::boolean(false);
    let source = eqiora_lang::SourceAstFactory::value_literal(
        &literal,
        None,
        eqiora_lang::TextRange::new(0, 0),
        |_| None,
        |_| None,
        |_| None,
    )
    .unwrap();
    assert!(matches!(source.kind(), ExprKind::Boolean(false)));
}

#[test]
fn low_precedence_negation_requires_explicit_operand_grouping() {
    assert!(
        parse("bad.eqi", "model M() { let p = true == not false; }")
            .into_document()
            .is_err()
    );
    assert!(
        parse("good.eqi", "model M() { let p = true == (not false); }")
            .into_document()
            .is_ok()
    );
}
