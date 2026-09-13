use eqiora_lang::{TokenKind, format, lex, parse};

const COMMENTED_EXAMPLES: &[(&str, &str)] = &[
    (
        "algebraic-split.eqi",
        include_str!("../../../examples/algebraic-split.eqi"),
    ),
    ("decay.eqi", include_str!("../../../examples/decay.eqi")),
    (
        "fixed-reference-fsi.eqi",
        include_str!("../../../examples/fixed-reference-fsi.eqi"),
    ),
    (
        "mixed-boundary-elasticity.eqi",
        include_str!("../../../examples/mixed-boundary-elasticity.eqi"),
    ),
    (
        "steady-flow-past-cylinder.eqi",
        include_str!("../../../examples/steady-flow-past-cylinder.eqi"),
    ),
];

#[test]
fn formatting_canonicalizes_examples_without_losing_authored_comments() {
    for (filename, source) in COMMENTED_EXAMPLES {
        let document = parse(*filename, source)
            .into_document()
            .expect("commented example parses");
        let formatted = format(&document);
        let comments = |text: &str| {
            lex(*filename, text)
                .0
                .iter()
                .filter(|token| {
                    matches!(token.kind(), TokenKind::LineComment | TokenKind::DocComment)
                })
                .map(|token| token.text().to_owned())
                .collect::<Vec<_>>()
        };
        assert_eq!(comments(&formatted), comments(source), "{filename}");

        let reparsed = parse(*filename, &formatted)
            .into_document()
            .expect("preserved example reparses");
        assert_eq!(format(&reparsed), formatted, "{filename}");
    }
}

#[test]
fn formatting_preserves_leading_trailing_and_body_comments() {
    let source = "// leading\nmodel M() { // trailing\n  // body\n  variable x: 1;\n}\n";
    let document = parse("comments.eqi", source)
        .into_document()
        .expect("comment positions parse");

    assert_eq!(format(&document), source);
}

#[test]
fn formatting_still_canonicalizes_comment_free_source() {
    let document = parse(
        "plain.eqi",
        "model plain() {variable x: 1;relation r{x=0;}}",
    )
    .into_document()
    .expect("plain source parses");
    let formatted = format(&document);

    assert_eq!(
        formatted,
        "model plain() {\n  variable x: 1;\n  relation r {\n    x = 0;\n  }\n}\n"
    );
    let reparsed = parse("plain.eqi", &formatted)
        .into_document()
        .expect("formatted plain source reparses");
    assert_eq!(format(&reparsed), formatted);
}
