use eqiora_lang::{format, parse};

#[test]
fn tensor_axis_options_preserve_tuples_and_operand_order_on_round_trip() {
    for expression in [
        "contract(a, b, axes = ((2, 0), (3, 1)))",
        "component(a, indices = (1,))",
        "permute_axes(a, order = (1, 0))",
        "contract(a, b, axes = ())",
    ] {
        let source = format!("model M() {{ observable result:1 = {expression}; }}");
        let document = parse("tensor.eqi", &source).into_document().unwrap();
        let rendered = format(&document);
        assert!(rendered.contains(expression), "{rendered}");
        assert_eq!(
            format(&parse("tensor.eqi", &rendered).into_document().unwrap()),
            rendered
        );
    }
}

#[test]
fn named_options_cannot_be_followed_by_positional_operands_or_repeated() {
    for expression in [
        "contract(a, axes = ((0, 0),), b)",
        "contract(a, b, axes = (), axes = ())",
        "tensor_value(a, frame = body, components = [1, 2])",
    ] {
        let source = format!("model M() {{ observable result:1 = {expression}; }}");
        assert!(
            parse("tensor.eqi", &source).into_document().is_err(),
            "{source}"
        );
    }
}
