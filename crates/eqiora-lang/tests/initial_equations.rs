use eqiora_lang::{
    ActivationSyntax, ExprKind, FieldRoleSyntax, Item, SourceAstFactory, TextRange, format, parse,
};

#[test]
fn native_initial_equations_share_source_ast_without_field_literals() {
    use eqiora_core::{DimExponents, ScalarDomain, ValueType};
    use eqiora_lang::{DraftDeclaration, DraftExpression, DraftField, Module};
    let state = DraftField::new(
        "x",
        ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS).unwrap(),
        FieldRoleSyntax::State,
    );
    let condition = (
        state.expression(),
        DraftExpression::constant(eqiora_lang::DecimalLiteral::parse("1.0").unwrap()),
    );
    let draft = Module::new(
        "Decay",
        [state.into(), DraftDeclaration::Initial(vec![condition])],
    )
    .unwrap();
    let native = &draft;
    let Item::Initial(initial) = &native.model().items()[1] else {
        panic!("initial equations keep their own owner");
    };
    assert_eq!(initial.equations().len(), 1);
    assert_eq!(
        native.graph_path(initial.range()).unwrap().to_string(),
        "Decay.initial"
    );
    let document =
        SourceAstFactory::document(Vec::new(), vec![], vec![], vec![native.model().clone()])
            .unwrap();
    let source = format(&document);
    assert!(source.contains("state x: 1;"));
    assert!(source.contains("initial {"));
    let parsed = parse("native.eqi", &source).into_document().unwrap();
    assert_eq!(format(&parsed), source);
}

#[test]
fn native_initial_conditions_reject_empty_nonfinite_and_foreign_symbols() {
    use eqiora_core::{DimExponents, ScalarDomain, ValueType};
    use eqiora_lang::{DraftDeclaration, DraftExpression, DraftField, Module};
    let value_type = ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS).unwrap();
    let included = DraftField::new("x", value_type.clone(), FieldRoleSyntax::State);
    let foreign = DraftField::new("x", value_type, FieldRoleSyntax::State);
    for (equations, expected) in [
        (vec![], "at least one equation"),
        (
            vec![(
                DraftExpression::complex(f64::NAN, 0.0),
                DraftExpression::constant(eqiora_lang::DecimalLiteral::parse("0").unwrap()),
            )],
            "non-finite",
        ),
        (
            vec![(
                DraftExpression::complex(f64::INFINITY, 0.0),
                DraftExpression::constant(eqiora_lang::DecimalLiteral::parse("0").unwrap()),
            )],
            "non-finite",
        ),
        (
            vec![(
                foreign.expression(),
                DraftExpression::constant(eqiora_lang::DecimalLiteral::parse("0").unwrap()),
            )],
            "foreign or omitted Field",
        ),
    ] {
        let diagnostics = Module::new(
            "M",
            [
                included.clone().into(),
                DraftDeclaration::Initial(equations),
            ],
        )
        .unwrap_err();
        assert!(
            diagnostics.iter().any(|d| d.message().contains(expected)),
            "{diagnostics:?}"
        );
        assert!(
            diagnostics
                .iter()
                .all(|d| d.graph_path().unwrap().to_string() == "M.initial")
        );
    }
}

#[test]
fn initial_equation_units_and_sign_roundtrip_without_erasure() {
    let source = "model M() { state distance: complex<m>; initial { distance = -2500[mm]; } }";
    let document = parse("field-unit.eqi", source).into_document().unwrap();
    let Item::Initial(group) = &document.models()[0].items()[1] else {
        panic!("expected initial equations");
    };
    let initial = group.equations()[0].right();
    assert_eq!(
        &source[initial.range().start() as usize..initial.range().end() as usize],
        "-2500[mm]"
    );
    let ExprKind::Unary {
        op: eqiora_lang::UnaryOp::Neg,
        value,
    } = initial.kind()
    else {
        panic!("equation expressions preserve unary negation");
    };
    assert!(matches!(value.kind(), ExprKind::Quantity { value, unit }
        if value.canonical_text() == "2500" && matches!(unit.kind(), ExprKind::Name(name) if name == "mm")));
    let formatted = format(&document);
    assert!(formatted.contains("= -2500 [mm];"));
    let reparsed = parse("formatted.eqi", &formatted).into_document().unwrap();
    assert_eq!(format(&reparsed), formatted);
}

#[test]
fn scalar_field_absence_roundtrips_through_source() {
    let document = parse(
        "uninitialized-field.eqi",
        "model M() { variable pressure: Pa; }",
    )
    .into_document()
    .expect("uninitialized scalar Field parses");
    let Item::Field(field) = &document.models()[0].items()[0] else {
        panic!("fixture contains one Field");
    };
    assert_eq!(field.role(), FieldRoleSyntax::Variable);

    let formatted = format(&document);
    assert_eq!(formatted, "model M() {\n  variable pressure: Pa;\n}\n");
    assert!(parse("formatted.eqi", &formatted).into_document().is_ok());
}

#[test]
fn factory_constructs_an_uninitialized_scalar_field() {
    let range = TextRange::new(0, 0);
    let dimension = SourceAstFactory::expression(
        ExprKind::Number(eqiora_lang::DecimalLiteral::parse("1.0").expect("exact literal")),
        range,
    )
    .expect("dimension");
    let value_type = SourceAstFactory::value_type(
        eqiora_lang::ValueTypeSyntaxKind::Scalar {
            domain: eqiora_core::ScalarDomain::Real,
            dimension,
        },
        range,
    )
    .unwrap();
    let field = SourceAstFactory::field(
        "pressure",
        None,
        FieldRoleSyntax::Variable,
        eqiora_schema::kernel::SpatialRegularity::Unspecified,
        ActivationSyntax::Continuous,
        value_type,
        range,
    )
    .expect("uninitialized scalar Field");
    let model = SourceAstFactory::model(
        eqiora_lang::VisibilitySyntax::Private,
        "flow",
        vec![],
        vec![Item::Field(field)],
        range,
    )
    .expect("model");
    let document = SourceAstFactory::document(Vec::new(), Vec::new(), Vec::new(), vec![model])
        .expect("document");

    let source = format(&document);
    assert_eq!(source, "model flow() {\n  variable pressure: 1;\n}\n");
    assert!(parse("factory-field.eqi", &source).into_document().is_ok());
}
