use eqiora_lang::{ComponentItem, Item, SignatureItem, SourceAstFactory, TextRange, parse};

fn exact(source: &str, name: &str) -> Option<TextRange> {
    let start = source.find(name).unwrap() as u32;
    Some(TextRange::new(start, start + name.len() as u32))
}

#[test]
fn authored_activation_names_have_exact_ranges_without_changing_activation_values() {
    let source = "// 🧪\r\ncomponent C(input incoming:1 at input_clock, output outgoing:1 at output_clock, port mechanical:Connector){state memory:1 at // at decoy\r\n field_clock;port exposed:signal output 1 at port_clock;relation update at relation_clock{memory=memory;}let previous at alias_clock=memory;variable continuous:1;} model M(){state x:1 at model_field;port exposed:signal output 1 at model_port;relation update at model_relation{x=x;}let previous at model_alias=x;}";
    let document = parse("ranges.eqi", source).into_document().unwrap();
    let component = &document.components()[0];
    for (item, expected) in
        component
            .signature()
            .iter()
            .zip([Some("input_clock"), Some("output_clock"), None])
    {
        let actual = match item {
            SignatureItem::Input(value) | SignatureItem::Output(value) => {
                value.activation_name_range()
            }
            SignatureItem::Port(value) => value.activation_name_range(),
            _ => panic!("signature fixture"),
        };
        assert_eq!(actual, expected.and_then(|name| exact(source, name)));
    }
    for item in component.items() {
        let (actual, expected, generated) = match item {
            ComponentItem::Field(value) if value.name() == "continuous" => {
                assert_eq!(value.activation_name_range(), None);
                continue;
            }
            ComponentItem::Field(value) => {
                let generated = SourceAstFactory::field(
                    value.name(),
                    value.domain().map(str::to_owned),
                    value.role(),
                    value.activation().clone(),
                    value.value_type().clone(),
                    value.range(),
                )
                .unwrap();
                assert_eq!(generated.activation(), value.activation());
                (
                    value.activation_name_range(),
                    "field_clock",
                    generated.activation_name_range(),
                )
            }
            ComponentItem::Port(value) => {
                let generated = SourceAstFactory::component_port(
                    value.visibility(),
                    value.name(),
                    value.syntax().clone(),
                    value.range(),
                )
                .unwrap();
                assert_eq!(generated.syntax(), value.syntax());
                (
                    value.activation_name_range(),
                    "port_clock",
                    generated.activation_name_range(),
                )
            }
            ComponentItem::Relation(value) => {
                let generated = SourceAstFactory::relation(
                    value.name(),
                    value.activation().clone(),
                    value.domain().map(str::to_owned),
                    value.conditions().unwrap().to_vec(),
                    value.range(),
                )
                .unwrap();
                assert_eq!(generated.activation(), value.activation());
                (
                    value.activation_name_range(),
                    "relation_clock",
                    generated.activation_name_range(),
                )
            }
            ComponentItem::Let(value) => {
                let generated = SourceAstFactory::let_alias(
                    value.name(),
                    value.value_type().cloned(),
                    value.domain().map(str::to_owned),
                    value.activation().map(str::to_owned),
                    value.value().clone(),
                    value.range(),
                )
                .unwrap();
                assert_eq!(generated.activation(), value.activation());
                (
                    value.activation_name_range(),
                    "alias_clock",
                    generated.activation_name_range(),
                )
            }
            _ => panic!("Component fixture"),
        };
        assert_eq!(actual, exact(source, expected));
        assert_eq!(
            generated, None,
            "source-free factories cannot infer an authored token"
        );
    }
    for item in document.models()[0].items() {
        let (actual, expected) = match item {
            Item::Field(value) => (value.activation_name_range(), "model_field"),
            Item::Port(value) => {
                let generated =
                    SourceAstFactory::port(value.name(), value.syntax().clone(), value.range())
                        .unwrap();
                assert_eq!(generated.activation_name_range(), None);
                assert_eq!(generated.syntax(), value.syntax());
                (value.activation_name_range(), "model_port")
            }
            Item::Relation(value) => (value.activation_name_range(), "model_relation"),
            Item::Let(value) => (value.activation_name_range(), "model_alias"),
            _ => panic!("Model fixture"),
        };
        assert_eq!(actual, exact(source, expected));
    }
    let formatted = eqiora_lang::format(&document);
    let reparsed = parse("formatted.eqi", &formatted).into_document().unwrap();
    let ComponentItem::Field(field) = &reparsed.components()[0].items()[0] else {
        panic!("field")
    };
    assert_eq!(
        field.activation_name_range(),
        exact(&formatted, "field_clock")
    );
    assert_eq!(eqiora_lang::format(&reparsed), formatted);
}
