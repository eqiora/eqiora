use eqiora_lang::{ActivationSyntax, BinaryOp, Expr, ExprKind, Item, format, parse};

fn arithmetic_tree(expression: &Expr) -> String {
    match expression.kind() {
        ExprKind::Name(name) => name.clone(),
        ExprKind::Number(value) => value.canonical_text(),
        ExprKind::Unary { value, .. } => format!("neg({})", arithmetic_tree(value)),
        ExprKind::Binary {
            op: BinaryOp::Pow,
            left,
            right,
        } => format!("pow({},{})", arithmetic_tree(left), arithmetic_tree(right)),
        other => panic!("outside fixed precedence corpus: {other:?}"),
    }
}

#[test]
fn unary_minus_and_right_associative_power_keep_the_authored_arithmetic() {
    for (expression, expected) in [
        ("-x^2", "neg(pow(x,2))"),
        ("(-x)^2", "pow(neg(x),2)"),
        ("x^-2", "pow(x,neg(2))"),
        ("(x^y)^z", "pow(pow(x,y),z)"),
        ("x^y^z", "pow(x,pow(y,z))"),
        ("-x^-2", "neg(pow(x,neg(2)))"),
    ] {
        let source = format!("model M() {{ relation r {{ {expression} = 0; }} }}");
        let document = parse("precedence.eqi", &source).into_document().unwrap();
        let Item::Relation(relation) = &document.models()[0].items()[0] else {
            panic!("relation")
        };
        assert_eq!(
            arithmetic_tree(relation.conditions().unwrap()[0].left()),
            expected
        );
        let formatted = format(&document);
        let reparsed = parse("precedence.eqi", &formatted).into_document().unwrap();
        let Item::Relation(relation) = &reparsed.models()[0].items()[0] else {
            panic!("relation")
        };
        assert_eq!(
            arithmetic_tree(relation.conditions().unwrap()[0].left()),
            expected,
            "{formatted}"
        );
        assert_eq!(format(&reparsed), formatted);
    }
}

#[test]
fn every_equality_keeps_both_sides_and_utf8_ranges_without_zero_escape() {
    for right in ["0", "(0)", "-0", "(-0)", "-(-0)", "y", "y - z", "0 * y"] {
        let source = format!("// α\r\nmodel M() {{ relation balance {{ x = {right}; }} }}");
        let document = parse("ordered.eqi", &source).into_document().unwrap();
        let Item::Relation(relation) = &document.models()[0].items()[0] else {
            panic!("relation")
        };
        let equation = &relation.conditions().unwrap()[0];
        let slice =
            |range: eqiora_lang::TextRange| &source[range.start() as usize..range.end() as usize];
        assert_eq!(slice(equation.range()), format!("x = {right}"));
        assert_eq!(slice(equation.left().range()), "x");
        assert_eq!(slice(equation.right().range()), right);
        assert!(matches!(equation.left().kind(), ExprKind::Name(name) if name == "x"));
        let formatted = format(&document);
        let reparsed = parse("ordered.eqi", &formatted).into_document().unwrap();
        assert_eq!(format(&reparsed), formatted);
        if matches!(right, "0" | "(0)") {
            assert!(formatted.contains("x = 0;"));
            assert!(!formatted.contains("x = (0);"));
        }
    }
}

#[test]
fn support_and_activation_are_independent_optional_header_axes() {
    for (header, domain, clock) in [
        ("relation r", None, None),
        ("relation r on body", Some("body"), None),
        ("relation r at tick", None, Some("tick")),
        ("relation r on body at tick", Some("body"), Some("tick")),
    ] {
        let source = format!("model M() {{ {header} {{ x = y; }} }}");
        let document = parse("headers.eqi", &source).into_document().unwrap();
        let Item::Relation(relation) = &document.models()[0].items()[0] else {
            panic!("relation")
        };
        assert_eq!(relation.domain(), domain);
        assert_eq!(
            relation.activation(),
            &clock.map_or(ActivationSyntax::Continuous, |name| {
                ActivationSyntax::Named(name.to_owned())
            })
        );
        assert!(format(&document).contains(header));
    }
    for header in [
        "relation r continuous",
        "relation r periodic(tick)",
        "relation r: tick",
        "relation r at tick on body",
        "relation r on body continuous",
    ] {
        assert!(
            parse(
                "retired.eqi",
                &format!("model M() {{ {header} {{ x = y; }} }}")
            )
            .into_document()
            .is_err(),
            "{header}"
        );
    }
    assert!(
        parse("empty.eqi", "model M() { relation empty {} }")
            .into_document()
            .is_err()
    );
}

#[test]
fn numeric_syntax_preserves_zero_and_nonzero_before_domain_admission() {
    // Decimal syntax has no binary64 range gate: contextual lowering owns it.
    for (value, zero) in [
        ("0", true),
        ("-0", true),
        ("0e-999", true),
        ("-0e-999", true),
        ("5e-324", false),
        ("-5e-324", false),
        ("1e-324", false),
        ("-1e-324", false),
        ("(1e-324)", false),
        ("1e999", false),
    ] {
        let document = parse(
            "number.eqi",
            &format!("model M() {{ relation r {{ x = {value}; }} }}"),
        )
        .into_document()
        .unwrap();
        let Item::Relation(relation) = &document.models()[0].items()[0] else {
            panic!("relation")
        };
        let mut expression = relation.conditions().unwrap()[0].right();
        while let ExprKind::Unary { value, .. } = expression.kind() {
            expression = value;
        }
        let ExprKind::Number(literal) = expression.kind() else {
            panic!("number")
        };
        assert_eq!(literal.is_zero(), zero, "{value}");
        let formatted = format(&document);
        assert_eq!(
            format(&parse("replay.eqi", &formatted).into_document().unwrap()),
            formatted
        );
    }
}

#[test]
fn expression_recursion_and_constructed_depth_are_bounded_before_allocation() {
    let additive = |count: usize| {
        std::iter::repeat_n("x", count)
            .collect::<Vec<_>>()
            .join(" + ")
    };
    for expression in [
        format!("{}x", "-".repeat(256)),
        format!("{}x{}", "(".repeat(256), ")".repeat(256)),
        std::iter::repeat_n("x", 257)
            .collect::<Vec<_>>()
            .join(" ^ "),
        additive(257),
        format!("{}x{}", "f(".repeat(256), ")".repeat(256)),
    ] {
        let source = format!("model M() {{ relation r {{ {expression} = 0; }} }}");
        let parsed = parse("bounded.eqi", &source);
        assert!(
            parsed
                .diagnostics()
                .iter()
                .any(|diagnostic| diagnostic.message().contains("256-level limit")),
            "{expression}"
        );
    }
    let source = format!("model M() {{ relation r {{ {} = 0; }} }}", additive(256));
    let document = parse("limit.eqi", &source).into_document().unwrap();
    let formatted = format(&document);
    let reparsed = parse("limit.eqi", &formatted).into_document().unwrap();
    assert_eq!(format(&reparsed), formatted);
}
