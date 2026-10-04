//! Private Component body declaration dispatch.

use super::*;

impl Parser<'_> {
    pub(super) fn parse_component_item(&mut self) -> Option<ParsedComponentItem> {
        if self.at_keyword("public") {
            self.error_here(
                "public requirements belong in the signature; body declarations are private",
            );
            self.bump();
            self.parse_private_component_item()?;
            return Some(ParsedComponentItem::Discarded);
        }
        self.parse_private_component_item()
    }

    fn parse_private_component_item(&mut self) -> Option<ParsedComponentItem> {
        let start = self.current().range().start();
        let visibility = VisibilitySyntax::Private;

        if self.at_keyword("parameter") {
            return self
                .parse_component_parameter(start, visibility, true)
                .map(ComponentItem::Parameter)
                .map(Box::new)
                .map(ParsedComponentItem::Retained);
        }
        if self.at_keyword("port") {
            return self
                .parse_component_port(start, visibility, true)
                .map(|port| {
                    ParsedComponentItem::Retained(Box::new(match port {
                        ParsedComponentPort::Ordinary(port) => ComponentItem::Port(port),
                        ParsedComponentPort::Family(port) => ComponentItem::PortFamily(port),
                    }))
                });
        }
        let item = if self.at_keyword("support") {
            self.parse_product_support().map(ComponentItem::Domain)
        } else if self.at_keyword("indexset") {
            self.parse_index_set().map(ComponentItem::IndexSet)
        } else if self.at_keyword("coordinate") {
            self.parse_coordinate().map(ComponentItem::Coordinate)
        } else if self.at_keyword("let") {
            self.parse_let().map(ComponentItem::Let)
        } else if self.at_keyword("observable") {
            self.parse_observable().map(ComponentItem::Observable)
        } else if self.at_keyword("variable") || self.at_keyword("state") {
            self.parse_field(true).map(ComponentItem::Field)
        } else if self.at_keyword("initial") {
            self.parse_initial().map(ComponentItem::Initial)
        } else if self.at_keyword("event") {
            self.parse_event().map(ComponentItem::Event)
        } else if self.at_keyword("clock") {
            self.parse_clock().map(ComponentItem::Clock)
        } else if self.at_keyword("law") {
            self.parse_law().map(ComponentItem::Relation)
        } else if self.at_keyword("relation") {
            self.parse_component_relation()
                .map(|relation| match relation {
                    ParsedRelation::Ordinary(relation) => ComponentItem::Relation(relation),
                    ParsedRelation::Family(relation) => ComponentItem::RelationFamily(relation),
                })
        } else if self.at_keyword("connect") {
            self.parse_connection(true)
                .map(|connection| match connection {
                    ParsedConnection::Ordinary(connection) => ComponentItem::Connection(connection),
                    ParsedConnection::Boundary(connection) => {
                        ComponentItem::BoundaryConnection(connection)
                    }
                })
        } else if self.at_keyword("instance") {
            self.parse_instance().map(ComponentItem::Instance)
        } else {
            self.error_here(
                "expected parameter, port, variable, state, initial, clock, relation, law, connect, or instance in component",
            );
            None
        }?;
        Some(ParsedComponentItem::Retained(Box::new(item)))
    }
}
