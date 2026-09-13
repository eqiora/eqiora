use eqiora_lang::{
    Item, Notation, NotationAccent, NotationAtom, NotationMark, NotationNode, NotationStyle,
    SourceAstFactory, format, parse,
};

fn round_trip(island: &str, canonical: &str) -> Notation {
    let value = Notation::parse(island).unwrap();
    assert_eq!(value.canonical(), canonical);
    let again = Notation::parse(canonical).unwrap();
    assert_eq!(again.root(), value.root());
    value
}

#[test]
fn complete_atom_table_is_case_sensitive_and_canonical() {
    for letter in b'a'..=b'z' {
        for letter in [char::from(letter), char::from(letter).to_ascii_uppercase()] {
            let spelling = format!("@{{{letter}}}");
            assert_eq!(
                round_trip(&spelling, &spelling).root(),
                &NotationNode::Atom(NotationAtom::Latin(letter))
            );
        }
    }
    for name in [
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
        "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon", "phi",
        "chi", "psi", "omega",
    ] {
        let uppercase = format!("{}{}", name[..1].to_ascii_uppercase(), &name[1..]);
        let lower = round_trip(&format!("@{{\\{name}}}"), &format!("@{{\\{name}}}"));
        let upper = round_trip(
            &format!("@{{\\{uppercase}}}"),
            &format!("@{{\\{uppercase}}}"),
        );
        assert_ne!(lower.root(), upper.root());
    }
    for name in [
        "varepsilon",
        "vartheta",
        "varkappa",
        "varpi",
        "varrho",
        "varsigma",
        "varphi",
        "hbar",
        "ell",
        "aleph",
    ] {
        let spelling = format!("@{{\\{name}}}");
        round_trip(&spelling, &spelling);
    }
}

#[test]
fn styles_accents_tensor_and_intrinsic_scripts_are_typed_not_expressions() {
    for name in [
        "mathrm", "mathit", "mathbf", "mathsf", "mathtt", "mathcal", "mathbb", "hat", "tilde",
        "bar", "vec", "dot", "ddot",
    ] {
        let spelling = format!("@{{\\{name}{{x}}}}");
        round_trip(&spelling, &spelling);
    }
    assert_eq!(
        round_trip(r"@{\hat{x}}", r"@{\hat{x}}").root(),
        &NotationNode::Accented(
            NotationAccent::Hat,
            Box::new(NotationNode::Atom(NotationAtom::Latin('x')))
        )
    );
    assert_eq!(
        round_trip(r"@{\mathbb{R}}", r"@{\mathbb{R}}").root(),
        &NotationNode::Styled(
            NotationStyle::BlackboardBold,
            Box::new(NotationNode::Atom(NotationAtom::Latin('R')))
        )
    );
    for (input, canonical) in [
        (r"@{\sigma_{ij}}", r"@{\sigma_{i j}}"),
        (r"@{x^2_i}", r"@{x_{i}^{2}}"),
        (r"@{x''}", r"@{x^{\prime \prime}}"),
        (r"@{x^{*+-}}", r"@{x^{\star \plus \minus}}"),
        (r"@{x_{\alpha i}}", r"@{x_{\alpha i}}"),
        (r"@{{\vec{v}}}", r"@{\vec{v}}"),
        (r"@{x_{i^{2}}}", r"@{x_{i^{2}}}"),
    ] {
        round_trip(input, canonical);
    }
    for (name, mark) in [
        ("prime", NotationMark::Prime),
        ("star", NotationMark::Star),
        ("top", NotationMark::Transpose),
        ("dagger", NotationMark::Dagger),
        ("plus", NotationMark::Plus),
        ("minus", NotationMark::Minus),
        ("pm", NotationMark::PlusMinus),
        ("mp", NotationMark::MinusPlus),
    ] {
        let spelling = format!("@{{x^{{\\{name}}}}}");
        let notation = round_trip(&spelling, &spelling);
        let NotationNode::Scripted { superscript, .. } = notation.root() else {
            panic!("script");
        };
        assert_eq!(superscript, &[NotationNode::Mark(mark)]);
    }
}

#[test]
fn excluded_tex_classes_and_unsupported_aliases_fail_at_the_notation_range() {
    // Each denial starts from the admitted symbol island; no earlier source gate.
    round_trip(r"@{\mu}", r"@{\mu}");
    for inner in [
        r"\input{x}",
        r"\include{x}",
        r"\write18{x}",
        r"\def{x}",
        r"\newcommand{x}",
        r"\begin{matrix}",
        r"\end{matrix}",
        r"\frac{x}{y}",
        r"\sqrt{x}",
        r"\text{arbitrary}",
        r"\unknown",
        r"\bf{x}",
        r"\boldsymbol{x}",
        r"\overline{x}",
        r"\left(x\right)",
        r"\hspace{1}",
        r"\kern1",
        r"\color{red}",
        r"\href{x}{y}",
        r"\(x\)",
        r"\[x\]",
        "$x$",
        "$$x$$",
        "x&y",
        r"x\\y",
        "x%comment",
        "x+y",
        "words",
        "λ",
        "x_i_j",
        "x^i^j",
        "x^{}",
        "x^i'",
        "{x_i}_j",
        "",
    ] {
        let island = format!("@{{{inner}}}");
        assert!(Notation::parse(&island).is_err(), "{island}");
        let source = format!("// UTF-8 🧪\nmodel M() {{ variable x {island}: 1; }}");
        let parsed = parse("symbols.eqi", &source);
        assert!(!parsed.diagnostics().is_empty(), "{island}");
        let start = source.find("@{").unwrap() as u32;
        assert!(
            parsed.diagnostics().iter().any(|diagnostic| {
                diagnostic.source_span().is_some_and(|span| {
                    span.start >= start && span.end <= start + island.len() as u32
                })
            }),
            "{:?}",
            parsed.diagnostics()
        );
    }
}

#[test]
fn bytes_depth_scripts_and_lexer_allocation_are_bounded() {
    let exact = format!("@{{x{}}}", " ".repeat(1020));
    assert_eq!(exact.len(), 1024);
    assert!(Notation::parse(&exact).is_ok());
    assert!(
        Notation::parse(&format!("@{{x{}}}", " ".repeat(1021)))
            .unwrap_err()
            .message()
            .contains("1024-byte")
    );
    let depth = |n| format!("@{{{}x{}}}", "\\hat{".repeat(n), "}".repeat(n));
    round_trip(&depth(7), &depth(7));
    assert!(
        Notation::parse(&depth(8))
            .unwrap_err()
            .message()
            .contains("depth")
    );
    let script = |n| format!("@{{x_{{{}}}}}", "i".repeat(n));
    assert!(Notation::parse(&script(32)).is_ok());
    assert!(
        Notation::parse(&script(33))
            .unwrap_err()
            .message()
            .contains("32-script")
    );
    let long = format!(
        "model M() {{ variable x @{{{} }}: 1; }}",
        "x".repeat(50_000)
    );
    let (tokens, diagnostics) = eqiora_lang::lex("long.eqi", &long);
    assert!(!diagnostics.is_empty());
    assert!(
        tokens
            .iter()
            .filter(|token| token.kind() == eqiora_lang::TokenKind::Notation)
            .all(|token| token.text().len() <= 1024)
    );
    assert_eq!(
        tokens.iter().map(|token| token.text()).collect::<String>(),
        long
    );
}

#[test]
fn declaration_metadata_survives_native_clone_reordering_and_comment_formatting() {
    let source = r"/// Model summary.
model M @{\mathcal{M}}() {
  /// Stress.
  variable stress @{\sigma_{ij}}: 1; // tail
  state estimate @{\hat{x}}: 1;
  parameter viscosity @{\mu}: 1 = 1;
  relation law @{L} { stress = estimate; }
}";
    let document = parse("notation.eqi", source).into_document().unwrap();
    assert_eq!(document.notations().len(), 5);
    for (_, notation) in document.notations() {
        assert!(
            source[notation.range().start() as usize..notation.range().end() as usize]
                .starts_with("@{")
        );
    }
    let formatted = format(&document);
    assert_eq!(
        format(&parse("again.eqi", &formatted).into_document().unwrap()),
        formatted
    );
    let model = &document.models()[0];
    let reordered = SourceAstFactory::model(
        model.visibility(),
        model.name(),
        vec![],
        vec![model.items()[1].clone(), model.items()[0].clone()],
        model.range(),
    )
    .unwrap()
    .with_notation(Notation::parse(r"@{\mathbf{N}}").unwrap());
    let reconstructed = SourceAstFactory::flat_document(vec![reordered]).unwrap();
    let emitted = format(&reconstructed);
    assert!(emitted.contains(r"model M @{\mathbf{N}}()"));
    assert!(emitted.find("estimate").unwrap() < emitted.find("stress").unwrap());
    assert!(emitted.contains(r"stress @{\sigma_{i j}}: 1; // tail"));
    let Item::Field(field) = &reconstructed.models()[0].items()[0] else {
        panic!("field");
    };
    assert_eq!(field.notation().unwrap().canonical(), r"@{\hat{x}}");
    let original_notation = field.notation().unwrap().clone();
    let renamed = SourceAstFactory::field(
        "renamed_estimate",
        field.domain().map(str::to_owned),
        field.role(),
        field.activation().clone(),
        field.value_type().clone(),
        field.range(),
    )
    .unwrap()
    .with_notation(original_notation.clone());
    assert_eq!(renamed.notation(), Some(&original_notation));
    let renamed_model = SourceAstFactory::model(
        model.visibility(),
        model.name(),
        vec![],
        vec![Item::Field(renamed)],
        model.range(),
    )
    .unwrap();
    let renamed_source = format(&SourceAstFactory::flat_document(vec![renamed_model]).unwrap());
    assert!(renamed_source.contains(r"state renamed_estimate @{\hat{x}}: 1;"));
}

#[test]
fn notation_is_only_a_declaration_header_not_a_reference_or_expression() {
    let material = "material composition M { property p = Release; } model Main() {}";
    assert!(parse("material.eqi", material).diagnostics().is_empty());
    assert!(
        !parse(
            "material-notation.eqi",
            &material.replace("p =", "p @{x} =")
        )
        .diagnostics()
        .is_empty()
    );
    for source in [
        "model M() { variable x: 1; relation r { x @{y} = 0; } }",
        "model M() { variable x: 1 @{y}; }",
        "model M() { variable x @{y} @{z}: 1; }",
        "model M() { variable x: 1; initial x @{y} = 0; }",
    ] {
        assert!(
            !parse("bad-position.eqi", source).diagnostics().is_empty(),
            "{source}"
        );
    }
    let broken = parse(
        "recovery.eqi",
        "model M() { variable broken @{x}: ; variable retained: 1; }",
    );
    assert!(!broken.diagnostics().is_empty());
    assert_eq!(broken.document().unwrap().notations().len(), 0);
    let multiple = format!(
        "model M @{{M}}() {{ {} variable retained @{{r}}: 1; }}",
        "variable broken @{x}: ;".repeat(200)
    );
    let recovered = parse("multiple-recovery.eqi", &multiple);
    assert!(!recovered.diagnostics().is_empty());
    let notations: Vec<_> = recovered
        .document()
        .unwrap()
        .notations()
        .map(|(_, notation)| notation.canonical())
        .collect();
    assert_eq!(notations, ["@{M}", "@{r}"]);
}

#[test]
fn notation_uses_every_existing_named_header_and_never_moves_to_a_reference() {
    let source = r"
dimension Scalar @{S} = 1;
public enum Mode @{\mathbb{M}} { Off, On }
space Basis @{B} = orthonormal(e0, e1);
connector Pin @{P} { across voltage: 1; through current: 1; }
public operator scale @{f}(input value @{v}: scalar): scalar = component(value);
component Part @{C}(
  parameter rate @{r}: 1,
  support body @{\Omega}: volume(ambient_dimension = 2),
  input signal @{s}: 1,
  output result @{y}: 1,
  clock sample @{k}: periodic,
  port pin @{p}: Pin
) {
  indexset Items @{I} = range(2);
  variable value @{x}: 1;
  let alias @{a} = value;
  clock tick @{t} = periodic(1 [s]);
  event crossing_event @{e} = crossing(value, direction = rising);
  relation law @{L}[i in Items] { value = 0; }
}
model Main @{N}() {
  domain body @{D} = box(0, 1, 0, 1);
  instance part @{q}: Part(rate = 1, body = body, signal = 0, sample = tick, pin = pin);
}
";
    let parsed = parse("all-headers.eqi", source).into_document().unwrap();
    assert_eq!(parsed.notations().len(), source.matches("@{").count());
    let formatted = format(&parsed);
    assert_eq!(
        format(&parse("formatted.eqi", &formatted).into_document().unwrap()),
        formatted
    );
    assert!(formatted.contains("relation law @{L}[i in Items]"));
    assert!(formatted.contains("alias @{a} = value;"));
    assert!(formatted.contains("sample = tick"));
}
