use eqiora_lang::{
    CallArguments, ExprKind, NamePath, PureValueClassSyntax, SourceAstFactory as F, TextRange,
    VisibilitySyntax, format, parse,
};

#[test]
fn typed_polynomial_and_qualified_composition_retain_named_order_and_comments() {
    let source = r#"
operator polynomial(input x: m, input k: N / m, input c: N / m^3): N = k*x+c*x*x*x;
operator composed(input x: m, input k: N / m, input c: N / m^3): N =
  Material.polynomial(c = c, // coefficient
    x = x, k = k);
"#;
    let document = parse("operator.eqi", source).into_document().unwrap();
    let first = &document.pure_operators()[0];
    assert!(matches!(
        first.formals()[0].value_class(),
        PureValueClassSyntax::Typed(_)
    ));
    assert!(matches!(first.result(), PureValueClassSyntax::Typed(_)));
    let ExprKind::Call { callee, arguments } = document.pure_operators()[1].body().kind() else {
        panic!("composition")
    };
    assert_eq!(callee.as_str(), "Material.polynomial");
    let bindings = arguments.named().unwrap();
    assert_eq!(
        bindings
            .iter()
            .map(|value| value.name())
            .collect::<Vec<_>>(),
        ["c", "x", "k"]
    );
    for binding in bindings {
        let range = binding.range();
        assert_eq!(
            &source[range.start() as usize..range.end() as usize],
            format!("{} = {}", binding.name(), binding.name())
        );
    }
    let formatted = format(&document);
    assert!(formatted.contains("// coefficient"));
    assert!(!formatted.contains("pure operator"));
    assert_eq!(
        format(&parse("again.eqi", &formatted).into_document().unwrap()),
        formatted
    );
}

#[test]
fn named_arguments_reject_duplicates_positional_suffix_and_missing_parts() {
    for call in [
        "f(x=1,x=2)",
        "f(x=1,2)",
        "f(x=)",
        "f(x=1,)",
        "f(=1)",
        "f(x=1 y=2)",
    ] {
        let source = format!("operator f(input x: 1): 1 = {call};");
        assert!(
            parse("bad.eqi", &source).into_document().is_err(),
            "{source}"
        );
    }
}

#[test]
fn native_named_call_rewrites_values_without_renaming_formal_keys() {
    let range = TextRange::new(0, 0);
    let value = F::expression(ExprKind::Name("x".into()), range).unwrap();
    let binding = F::named_binding("x", value, range).unwrap();
    let callee = NamePath::from_segments(["Material", "polynomial"], range).unwrap();
    let expression = F::expression(
        ExprKind::Call {
            callee: callee.clone(),
            arguments: CallArguments::Named(vec![binding.clone()]),
        },
        range,
    )
    .unwrap();
    let rewritten = expression.rewrite_name_paths(|path| {
        (path.as_str() == "x").then(|| NamePath::from_segments(["owner", "x"], range).unwrap())
    });
    let ExprKind::Call { arguments, .. } = rewritten.kind() else {
        panic!("call")
    };
    let binding = &arguments.named().unwrap()[0];
    assert_eq!(binding.name(), "x");
    assert!(matches!(binding.value().kind(), ExprKind::Path(path) if path.as_str() == "owner.x"));
    assert!(arguments.positional().is_none());
    assert!(
        F::expression(
            ExprKind::Call {
                callee,
                arguments: CallArguments::Named(vec![binding.clone(), binding.clone()])
            },
            range
        )
        .is_err()
    );
}

#[test]
fn native_typed_operator_uses_the_shared_exact_body() {
    let range = TextRange::new(0, 0);
    let dimension = F::expression(ExprKind::Name("m".into()), range).unwrap();
    let value_type = F::value_type(
        eqiora_lang::ValueTypeSyntaxKind::Scalar {
            domain: eqiora_core::ScalarDomain::Real,
            dimension,
        },
        range,
    )
    .unwrap();
    let formal =
        F::pure_operator_formal("x", PureValueClassSyntax::Typed(value_type.clone()), range)
            .unwrap();
    let body = F::expression(ExprKind::Name("x".into()), range).unwrap();
    let operator = F::pure_operator(
        VisibilitySyntax::Public,
        "identity",
        vec![formal],
        PureValueClassSyntax::Typed(value_type),
        body,
        range,
    )
    .unwrap();
    let document =
        F::document_with_pure_operators(vec![], vec![], vec![], vec![operator], vec![]).unwrap();
    assert_eq!(
        format(&document),
        "public operator identity(input x: m): m = x;\n"
    );
    assert!(
        parse("native.eqi", &format(&document))
            .into_document()
            .is_ok()
    );
}

#[test]
fn exact_operator_tokens_and_named_calls_keep_shared_depth_bounds() {
    let source = "operator exact(input x: 1): 1 = 9007199254740993 + 1e-400;";
    let document = parse("exact.eqi", source).into_document().unwrap();
    let ExprKind::Binary { left, right, .. } = document.pure_operators()[0].body().kind() else {
        panic!("sum")
    };
    assert!(
        matches!(left.kind(), ExprKind::Number(value) if value.to_i64().unwrap() == 9_007_199_254_740_993)
    );
    assert!(
        matches!(right.kind(), ExprKind::Number(value) if !value.is_zero() && value.to_f64().is_err())
    );
    for (depth, accepted) in [(255, true), (256, false)] {
        let body = format!("{}x{}", "f(x = ".repeat(depth), ")".repeat(depth));
        let source = format!("operator bounded(input x: 1): 1 = {body};");
        assert_eq!(
            parse("depth.eqi", &source).into_document().is_ok(),
            accepted
        );
    }
}
