use eqiora_core::{Id, ValueLiteral, entity::kinds};
use eqiora_lang::{
    DecimalLiteral, DraftDeclaration, DraftParameter, Item, Module, SourceAstFactory,
    ValueTypeSyntaxKind, format, parse,
};
use eqiora_schema::kernel::FiniteSpaceDef;

#[test]
fn nominal_declarations_family_members_and_connections_round_trip() {
    let source = r#"
space Species = orthonormal(A, B);
component Cell(parameter value: integer, output y: integer) { relation r { y = value; } }
model M(parameter n: integer = 2, output y: integer) {
  // Ordered, bounded occurrences.
  indexset Stages = range(n);
  parameter population: counts<Species> = counts(Species, [2, 9007199254740993]);
  instance cell[i in Stages]: Cell(value = ordinal(i));
  connect cell[index(Stages, 0)].y -> y;
}
"#;
    let document = parse("nominal.eqi", source).into_document().unwrap();
    assert!(
        matches!(document.finite_spaces()[0].value().kind(), eqiora_lang::ExprKind::Call { callee, arguments } if callee.as_str() == "orthonormal" && arguments.expressions().len() == 2)
    );
    assert!(
        matches!(&document.models()[0].items()[0], Item::IndexSet(set) if set.name() == "Stages")
    );
    let rendered = format(&document);
    assert_eq!(
        format(&parse("rendered.eqi", &rendered).into_document().unwrap()),
        rendered
    );
    assert!(rendered.contains("9007199254740993"));
    assert!(rendered.contains("cell[index(Stages, 0)].y"));
}

#[test]
fn native_nominal_projection_requires_registered_exact_declaration_identity() {
    let definition = FiniteSpaceDef::new(
        Id::<kinds::FiniteSpace>::new(),
        ["A".to_owned(), "B".to_owned()],
    )
    .unwrap();
    let value = ValueLiteral::integer(definition.counts(), [2, 9007199254740993]).unwrap();
    let parameter = DraftParameter::new("population", value.clone());
    assert!(Module::new("M", [parameter.clone().into()]).is_err());
    let draft = Module::new(
        "M",
        [
            DraftDeclaration::FiniteSpace {
                name: "Species".into(),
                definition: definition.clone(),
            },
            parameter.into(),
        ],
    )
    .unwrap();
    let native = &draft;
    assert_eq!(
        native.nominal_identity("Species"),
        Some(definition.id().erase())
    );
    assert_eq!(native.document().finite_spaces()[0].name(), "Species");
    let Item::Parameter(parameter) = &native.model().items()[0] else {
        panic!("parameter")
    };
    assert!(
        matches!(parameter.value_type().kind(), ValueTypeSyntaxKind::Counts(name) if name.as_str() == "Species")
    );
    assert_eq!(
        parameter.value_type().resolved_nominal(),
        Some(value.value_type())
    );
    assert!(format(native.document()).contains("9007199254740993"));
    assert!(
        SourceAstFactory::expression(
            eqiora_lang::ExprKind::Number(DecimalLiteral::parse("1").unwrap()),
            eqiora_lang::TextRange::new(0, 0)
        )
        .is_ok()
    );
}

#[test]
fn shared_named_definitions_keep_nominal_constructor_and_assertion_boundaries() {
    use eqiora_lang::{TextRange, VisibilitySyntax};
    let range = TextRange::new(0, 0);
    let value = SourceAstFactory::expression(
        eqiora_lang::ExprKind::Number(DecimalLiteral::parse("2").unwrap()),
        range,
    )
    .unwrap();
    let alias =
        SourceAstFactory::let_alias("Rows", None, Some("body".into()), None, value, range).unwrap();
    assert!(
        SourceAstFactory::model(
            VisibilitySyntax::Private,
            "M",
            vec![],
            vec![Item::IndexSet(alias.clone())],
            range
        )
        .is_err()
    );
    let document = parse("empty.eqi", "model M() {}").into_document().unwrap();
    assert!(SourceAstFactory::with_finite_space(document, alias).is_err());
}

#[test]
fn nominal_resolution_metadata_preserves_authored_expression_and_rejects_foreign_rebinding() {
    let mut document = parse("binding.eqi", "space Species = orthonormal(A,B); model M() { parameter population: counts<Species> = counts(Species,[2,3]); }")
        .into_document().unwrap();
    let before = format(&document);
    let first = FiniteSpaceDef::new(
        Id::<kinds::FiniteSpace>::new(),
        ["A".to_owned(), "B".to_owned()],
    )
    .unwrap();
    let foreign = FiniteSpaceDef::new(
        Id::<kinds::FiniteSpace>::new(),
        ["A".to_owned(), "B".to_owned()],
    )
    .unwrap();
    let name = eqiora_lang::NamePath::from_segments(["Species"], eqiora_lang::TextRange::new(0, 0))
        .unwrap();
    let mut bound = 0;
    SourceAstFactory::visit_expressions(&mut document, |_, expression| {
        if matches!(expression.kind(), eqiora_lang::ExprKind::Call { callee, .. } if callee.as_str() == "counts")
        {
            SourceAstFactory::bind_nominal_expression(
                expression,
                std::slice::from_ref(&name),
                first.counts(),
            )
            .unwrap();
            assert_eq!(expression.resolved_nominal(), Some(&first.counts()));
            assert!(
                SourceAstFactory::bind_nominal_expression(
                    expression,
                    std::slice::from_ref(&name),
                    foreign.counts()
                )
                .is_err()
            );
            bound += 1;
        }
    });
    assert_eq!(bound, 1);
    assert_eq!(format(&document), before);
}

#[test]
fn native_finite_numeric_literals_project_complete_basis_and_scalar_types() {
    use eqiora_core::{DimExponents, ScalarDomain, ValueType};
    let spin = FiniteSpaceDef::new(Id::new(), ["up".into(), "down".into()]).unwrap();
    let control = FiniteSpaceDef::new(Id::new(), ["u0".into(), "u1".into()]).unwrap();
    for value_type in [
        ValueType::coordinates(
            spin.basis(),
            ScalarDomain::Complex,
            DimExponents::DIMENSIONLESS,
        )
        .unwrap(),
        ValueType::coordinates(
            spin.basis().dual(),
            ScalarDomain::Complex,
            DimExponents::DIMENSIONLESS,
        )
        .unwrap(),
        ValueType::linear_map(
            spin.basis(),
            control.basis(),
            ScalarDomain::Complex,
            DimExponents::DIMENSIONLESS,
        )
        .unwrap(),
        ValueType::linear_map(
            control.basis().dual(),
            spin.basis().dual(),
            ScalarDomain::Complex,
            DimExponents::DIMENSIONLESS,
        )
        .unwrap(),
    ] {
        let count = value_type.shape().component_count().unwrap();
        let value = ValueLiteral::new(
            value_type.clone(),
            (0..count).map(|i| (i as f64, -(i as f64))),
        )
        .unwrap();
        let draft = Module::new(
            "M",
            [
                DraftDeclaration::FiniteSpace {
                    name: "Spin".into(),
                    definition: spin.clone(),
                },
                DraftDeclaration::FiniteSpace {
                    name: "Control".into(),
                    definition: control.clone(),
                },
                DraftParameter::new("value", value).into(),
            ],
        )
        .unwrap();
        let Item::Parameter(parameter) = &draft.model().items()[0] else {
            panic!("parameter");
        };
        assert_eq!(parameter.value_type().resolved_nominal(), Some(&value_type));
        assert_eq!(parameter.value().resolved_nominal(), Some(&value_type));
        let rendered = format(draft.document());
        let replay = parse("replay.eqi", &rendered).into_document().unwrap();
        assert_eq!(format(&replay), rendered);
        assert!(rendered.contains("math.complex"));
        if value_type.finite_bases().any(|basis| basis.is_dual()) {
            assert!(rendered.contains("dual("));
        }
    }
}
