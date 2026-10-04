//! Canonical tags and payloads for private Component items and requirements.

use super::*;

pub(super) fn encode_component_item(
    item: &ComponentItem,
    budget: &mut Budget,
) -> Result<Vec<u8>, Diagnostic> {
    let mut encoder = Encoder::new(budget.limits.max_canonical_bytes);
    match item {
        ComponentItem::Domain(declaration) => {
            encoder.u16(31)?;
            encode_domain(&mut encoder, declaration, budget)?;
        }
        ComponentItem::IndexSet(declaration) => {
            encoder.u16(18)?;
            encode_let(&mut encoder, declaration, budget)?;
        }
        ComponentItem::Let(declaration) => {
            encoder.u16(17)?;
            encode_let(&mut encoder, declaration, budget)?;
        }
        ComponentItem::Observable(declaration) => {
            encoder.u16(30)?;
            compile_time::encode_observable(&mut encoder, declaration, budget)?;
        }
        ComponentItem::Parameter(declaration) => {
            encoder.u16(1)?;
            encode_component_parameter(&mut encoder, declaration, budget)?;
        }
        ComponentItem::Port(declaration) => {
            encoder.u16(2)?;
            encode_component_port(&mut encoder, declaration, budget)?;
        }
        ComponentItem::PortFamily(declaration) => {
            encoder.u16(COMPONENT_PORT_FAMILY_ITEM_TAG)?;
            encode_component_port_family(&mut encoder, declaration, budget)?;
        }
        ComponentItem::Initial(declaration) => {
            encoder.u16(15)?;
            encode_initial(&mut encoder, declaration, budget)?;
        }
        ComponentItem::Field(declaration) => {
            encoder.u16(3)?;
            encode_field(&mut encoder, declaration, budget)?;
        }
        ComponentItem::Event(declaration) => {
            encoder.u16(16)?;
            declarations::encode_event(&mut encoder, declaration, budget)?;
        }
        ComponentItem::Clock(declaration) => {
            encoder.u16(4)?;
            declarations::encode_clock(&mut encoder, declaration, budget)?;
        }
        ComponentItem::Relation(declaration) => {
            encoder.u16(5)?;
            encode_relation(&mut encoder, declaration, budget)?;
        }
        ComponentItem::RelationFamily(declaration) => {
            encoder.u16(COMPONENT_RELATION_FAMILY_ITEM_TAG)?;
            encode_relation_family(&mut encoder, declaration, budget)?;
        }
        ComponentItem::Connection(declaration) => {
            encoder.u16(COMPONENT_CONNECTION_ITEM_TAG)?;
            encode_connection(&mut encoder, declaration, budget)?;
        }
        ComponentItem::BoundaryConnection(declaration) => {
            encoder.u16(match declaration.syntax() {
                ConnectionSyntax::Conserving => COMPONENT_BOUNDARY_CONNECTION_ITEM_TAG,
                ConnectionSyntax::SpatialPeriodic => COMPONENT_SPATIAL_PERIODIC_CONNECTION_ITEM_TAG,
                ConnectionSyntax::Signal => {
                    return Err(source_identity_error(
                        "boundary Connection cannot use signal semantics",
                    ));
                }
            })?;
            encode_boundary_connection(&mut encoder, declaration, budget)?;
        }
        ComponentItem::Instance(declaration) => {
            encoder.u16(7)?;
            encode_instance(&mut encoder, declaration, budget)?;
        }
        _ => {
            return Err(source_identity_error(
                "component item is newer than source identity v1",
            ));
        }
    }
    encoder.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_lang::{SourceAstFactory, TextRange, VisibilitySyntax};

    fn identity(source: &str) -> LocalSourceIdentity {
        let document = eqiora_lang::parse("identity.eqi", source)
            .into_document()
            .unwrap();
        LocalSourceIdentity::from_document(&document).unwrap()
    }

    #[test]
    fn alias_item_tags_separate_containers_and_declaration_roles() {
        let doc = eqiora_lang::parse(
            "tags.eqi",
            "component C() { let a = 1; } model M() { let a = 1; }",
        )
        .into_document()
        .unwrap();
        let mut budget = Budget::new(LocalSourceIdentityLimits::default());
        let component =
            encode_component_item(&doc.components()[0].items()[0], &mut budget).unwrap();
        let model = encode_model_item(&doc.models()[0].items()[0], &mut budget).unwrap();
        assert_eq!(&component[..2], &17u16.to_be_bytes());
        assert_eq!(&model[..2], &13u16.to_be_bytes());
        assert_eq!(&component[2..], &model[2..]);
        assert_ne!(
            identity("component C() { let a: 1 = 1; }"),
            identity("component C() { parameter a: 1 = 1; }")
        );
        assert_ne!(
            identity("component C() { let a = 1; }"),
            identity("component C() { let a = 2; }")
        );
        assert_ne!(
            identity("component C() { let a = 1; }"),
            identity("component C() { let a: 1 = 1; }")
        );
    }

    #[test]
    fn alias_identity_ignores_order_comments_ranges_and_factory_route() {
        let source = "component C() { let a = 1; let b = a; }";
        let document = eqiora_lang::parse("identity.eqi", source)
            .into_document()
            .unwrap();
        let expected = LocalSourceIdentity::from_document(&document).unwrap();
        assert_eq!(
            expected,
            identity("component C() { // comment\n let b = a; let a = 1; }")
        );
        assert_eq!(expected, identity(&eqiora_lang::format(&document)));
        let range = TextRange::new(0, 0);
        let aliases = [
            (
                "a",
                ExprKind::Number(eqiora_lang::DecimalLiteral::parse("1.0").expect("exact literal")),
            ),
            ("b", ExprKind::Name("a".to_owned())),
        ]
        .into_iter()
        .map(|(name, kind)| {
            ComponentItem::Let(
                SourceAstFactory::let_alias(
                    name,
                    None,
                    None,
                    None,
                    SourceAstFactory::expression(kind, range).unwrap(),
                    range,
                )
                .unwrap(),
            )
        })
        .collect();
        let component =
            SourceAstFactory::component(VisibilitySyntax::Private, "C", Vec::new(), aliases, range)
                .unwrap();
        let factory =
            SourceAstFactory::document(Vec::new(), vec![], vec![component], vec![]).unwrap();
        assert_eq!(
            expected,
            LocalSourceIdentity::from_document(&factory).unwrap()
        );
    }
}
