use eqiora_lang::{
    ActivationSyntax, BinaryOp, ComponentItem, ExprKind, Item, NamePath, SourceAstFactory as F,
    TextRange, VisibilitySyntax, format, parse,
};
use eqiora_schema::kernel::EventDirection;

#[test]
fn model_and_component_events_keep_guards_direction_resets_aliases_and_comments() {
    for container in ["model", "component"] {
        let source = format!(
            r#"{container} Ball() {{
  // impact guard
  event impact = crossing(height - 0[m], direction = falling);
  let reflected at impact = -restitution * pre(velocity);
  relation reset at impact {{ next(velocity) = reflected; next(height) = 0[m]; }}
}}"#
        );
        let document = parse("event.eqi", &source).into_document().unwrap();
        let (event, alias, relation) = if container == "model" {
            let items = document.models()[0].items();
            let (Item::Event(event), Item::Let(alias), Item::Relation(relation)) =
                (&items[0], &items[1], &items[2])
            else {
                panic!("model event items")
            };
            (event, alias, relation)
        } else {
            let items = document.components()[0].items();
            let (
                ComponentItem::Event(event),
                ComponentItem::Let(alias),
                ComponentItem::Relation(relation),
            ) = (&items[0], &items[1], &items[2])
            else {
                panic!("component event items")
            };
            (event, alias, relation)
        };
        assert_eq!(event.name(), "impact");
        assert_eq!(event.direction(), EventDirection::Falling);
        assert!(matches!(
            event.guard().kind(),
            ExprKind::Binary {
                op: BinaryOp::Sub,
                ..
            }
        ));
        assert_eq!(
            &source[event.range().start() as usize..event.range().end() as usize],
            "event impact = crossing(height - 0[m], direction = falling);"
        );
        assert_eq!(alias.activation(), Some("impact"));
        assert_eq!(
            relation.activation(),
            &ActivationSyntax::Named("impact".into())
        );
        assert_eq!(relation.conditions().unwrap().len(), 2);
        assert!(
            matches!(relation.conditions().unwrap()[0].left().kind(), ExprKind::Call { callee, .. } if callee.as_str() == "next")
        );
        let formatted = format(&document);
        assert!(formatted.contains("// impact guard"));
        assert_eq!(
            format(&parse("again.eqi", &formatted).into_document().unwrap()),
            formatted
        );
    }
}

#[test]
fn explicit_directions_have_source_and_native_parity() {
    let range = TextRange::new(0, 0);
    for (direction, spelling) in [
        (EventDirection::Any, "any"),
        (EventDirection::Rising, "rising"),
        (EventDirection::Falling, "falling"),
    ] {
        let guard = F::expression(ExprKind::Name("height".into()), range).unwrap();
        let event = F::event("impact", guard, direction, range).unwrap();
        let component = F::component(
            VisibilitySyntax::Private,
            "Ball",
            vec![],
            vec![ComponentItem::Event(event)],
            range,
        )
        .unwrap();
        let document = F::document(vec![], vec![], vec![component], vec![]).unwrap();
        let formatted = format(&document);
        assert!(formatted.contains(&format!(
            "event impact = crossing(height, direction = {spelling});"
        )));
        let reparsed = parse("native.eqi", &formatted).into_document().unwrap();
        let ComponentItem::Event(event) = &reparsed.components()[0].items()[0] else {
            panic!("event")
        };
        assert_eq!(event.direction(), direction);
        assert!(matches!(event.guard().kind(), ExprKind::Name(name) if name == "height"));
    }
}

#[test]
fn events_reject_missing_or_noncanonical_parts_and_public_signatures() {
    for declaration in [
        "event impact = crossing(height);",
        "event impact = crossing(height, direction = sideways);",
        "event impact = crossing(height, falling);",
        "event impact = crossing(height, direction = falling, direction = rising);",
        "event impact = crossing(direction = falling, height);",
        "event impact = crossing(, direction = falling);",
        "event impact: m = crossing(height, direction = falling);",
        "event impact = crossing(height, direction = falling)",
        "public event impact = crossing(height, direction = falling);",
    ] {
        for container in ["model", "component"] {
            let source = format!("{container} Ball() {{ {declaration} }}");
            assert!(
                parse("bad.eqi", &source).into_document().is_err(),
                "{source}"
            );
        }
    }
    assert!(
        parse("signature.eqi", "component Ball(event impact) {}")
            .into_document()
            .is_err()
    );
    assert!(
        parse(
            "top.eqi",
            "event impact = crossing(height, direction = falling);"
        )
        .into_document()
        .is_err()
    );
}

#[test]
fn event_guards_participate_in_the_common_scoped_expression_walk() {
    let mut document = parse("visit.eqi", "component Ball() { event impact = crossing(height, direction = falling); } model M() { event warm = crossing(temperature, direction = rising); }").into_document().unwrap();
    let mut seen = Vec::new();
    F::visit_expressions(&mut document, |scope, expression| {
        if let ExprKind::Name(name) = expression.kind() {
            seen.push((scope.unwrap().to_owned(), name.clone()));
        }
        *expression = expression.rewrite_name_paths(|path| {
            NamePath::from_segments(["owned", path.as_str()], path.range()).ok()
        });
    });
    assert_eq!(
        seen,
        [
            ("Ball".to_owned(), "height".to_owned()),
            ("M".to_owned(), "temperature".to_owned())
        ]
    );
    let Item::Event(event) = &document.models()[0].items()[0] else {
        panic!("event")
    };
    assert!(
        matches!(event.guard().kind(), ExprKind::Path(path) if path.as_str() == "owned.temperature")
    );
}

#[test]
fn event_guard_syntax_retains_shared_bounds_and_defers_complete_type_admission() {
    // Boolean and shaped guards are syntactically retained, so compiler type admission
    // must reject them at the event owner instead of a misleading parse gate.
    for guard in ["true", "[1,2]", "if enabled then height else offset"] {
        let source =
            format!("model M() {{ event impact = crossing({guard}, direction = falling); }}");
        assert!(parse("types.eqi", &source).into_document().is_ok());
    }
    for (depth, accepted) in [(255, true), (256, false)] {
        let guard = format!("{}height{}", "f(x = ".repeat(depth), ")".repeat(depth));
        let source =
            format!("model M() {{ event impact = crossing({guard}, direction = falling); }}");
        assert_eq!(
            parse("depth.eqi", &source).into_document().is_ok(),
            accepted
        );
    }
    let range = TextRange::new(0, 0);
    let guard = F::expression(ExprKind::Name("height".into()), range).unwrap();
    assert!(F::event("bad.name", guard.clone(), EventDirection::Falling, range).is_err());
    assert!(
        F::event(
            "impact",
            guard,
            EventDirection::Falling,
            TextRange::new(2, 1)
        )
        .is_err()
    );
}
