use crate::ast::{NamedDefinitionDecl, ParameterDecl, TextRange};
use crate::lexer::TokenKind;

use super::Parser;

impl Parser<'_> {
    pub(super) fn parse_parameter(&mut self) -> Option<ParameterDecl> {
        let start = self.expect_keyword("parameter")?.range().start();
        let name = self.declaration_name("declaration name")?.text().to_owned();
        self.expect(TokenKind::Colon, "`:` before dimension")?;
        let value_type = self.parse_value_type()?;
        self.expect(TokenKind::Equal, "`=` before value")?;
        let value = self.parse_expression(0)?;
        let end = self
            .expect(TokenKind::Semicolon, "`;` after declaration")?
            .range()
            .end();
        Some(ParameterDecl {
            comments: Default::default(),
            name,
            value_type,
            value,
            range: TextRange::new(start, end),
        })
    }

    pub(super) fn parse_observable(&mut self) -> Option<crate::ObservableDecl> {
        let start = self.expect_keyword("observable")?.range().start();
        let name = self.declaration_name("declaration name")?.text().to_owned();
        self.expect(TokenKind::Colon, "`:` before dimension")?;
        let value_type = self.parse_value_type()?;
        self.expect(TokenKind::Equal, "`=` before value")?;
        let value = self.parse_expression(0)?;
        let end = self
            .expect(TokenKind::Semicolon, "`;` after declaration")?
            .range()
            .end();
        Some(crate::ObservableDecl {
            comments: Default::default(),
            name,
            value_type,
            value,
            range: TextRange::new(start, end),
        })
    }

    pub(super) fn parse_coordinate(&mut self) -> Option<NamedDefinitionDecl> {
        let start = self.expect_keyword("coordinate")?.range().start();
        let name = self.declaration_name("coordinate name")?.text().to_owned();
        self.expect(TokenKind::Colon, "`:` before coordinate dimension")?;
        let value_type = self.parse_value_type()?;
        self.expect_keyword("on")?;
        let domain = self
            .expect_identifier("coordinate support")?
            .text()
            .to_owned();
        self.expect_keyword("from")?;
        let factor = self.parse_expression(0)?;
        let end = self
            .expect(TokenKind::Semicolon, "`;` after coordinate declaration")?
            .range()
            .end();
        match crate::SourceAstFactory::coordinate(
            name,
            value_type,
            domain,
            factor,
            TextRange::new(start, end),
        ) {
            Ok(value) => Some(value),
            Err(error) => {
                self.error_previous(error.to_string());
                None
            }
        }
    }

    pub(super) fn parse_let(&mut self) -> Option<NamedDefinitionDecl> {
        let start = self.expect_keyword("let")?.range().start();
        let name = self.declaration_name("alias name")?.text().to_owned();
        let value_type = if self.at(TokenKind::Colon) {
            self.bump();
            Some(self.parse_value_type()?)
        } else {
            None
        };
        let domain = if self.at_keyword("on") {
            self.bump();
            Some(
                self.expect_identifier("let support assertion")?
                    .text()
                    .to_owned(),
            )
        } else {
            None
        };
        let (activation, activation_name_range) = if self.at_keyword("at") {
            self.bump();
            let token = self.expect_identifier("let activation assertion")?;
            (Some(token.text().to_owned()), Some(token.range()))
        } else {
            (None, None)
        };
        self.expect(TokenKind::Equal, "`=` before alias expression")?;
        let value = self.parse_expression(0)?;
        let end = self
            .expect(TokenKind::Semicolon, "`;` after declaration")?
            .range()
            .end();
        Some(NamedDefinitionDecl {
            activation_name_range,
            visibility: crate::VisibilitySyntax::Private,
            comments: Default::default(),
            name,
            value_type,
            domain,
            activation,
            value,
            range: TextRange::new(start, end),
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::ast::Item;
    use crate::{format, parse};

    #[test]
    fn coordinate_declarations_round_trip_in_models_and_components() {
        for owner in ["model", "component"] {
            let source = format!(
                "{owner} M() {{ coordinate x:m on phase from position[0]; coordinate v:m/s on phase from velocity; }}"
            );
            let document = parse("coordinates.eqi", &source).into_document().unwrap();
            let formatted = format(&document);
            assert!(formatted.contains("coordinate x: m on phase from position[0];"));
            assert!(formatted.contains("coordinate v: m / s on phase from velocity;"));
            assert_eq!(
                format(&parse("again.eqi", &formatted).into_document().unwrap()),
                formatted
            );
        }
    }

    #[test]
    fn coordinate_declarations_require_exact_selectors_without_initializers_or_activation() {
        for declaration in [
            "coordinate x on phase from position;",
            "coordinate x:m from position;",
            "coordinate x:m on phase from position at continuous;",
            "coordinate x:m on phase from position=1[m];",
            "coordinate x:m on phase from position+velocity;",
            "coordinate x:m on phase from position[-1];",
            "coordinate x:m on phase from position[0.5];",
        ] {
            let source = format!("model M() {{ {declaration} }}");
            assert!(
                parse("invalid-coordinate.eqi", &source)
                    .into_document()
                    .is_err(),
                "{declaration}"
            );
        }
    }

    #[test]
    fn complete_let_type_annotations_round_trip() {
        let source = "model M() { let z: complex<m> = 0; let channels: array<complex<m>, 3> = 0; }";
        let document = parse("typed-let.eqi", source).into_document().unwrap();
        let formatted = format(&document);
        assert!(formatted.contains("let z: complex<m> = 0;"));
        assert!(formatted.contains("let channels: array<complex<m>, 3> = 0;"));
        assert_eq!(
            format(&parse("again.eqi", &formatted).into_document().unwrap()),
            formatted
        );
    }

    #[test]
    fn parser_and_formatter_retain_annotated_and_inferred_let_aliases() {
        let source =
            "model M() { let wave_number = math.pi / length; let checked: 1 / m = wave_number; }";
        let document = parse("let.eqi", source)
            .into_document()
            .expect("let aliases parse");
        let Item::Let(declaration) = &document.models()[0].items()[0] else {
            panic!("model item is a let alias");
        };
        assert_eq!(declaration.name(), "wave_number");
        assert!(declaration.value_type().is_none());
        let Item::Let(checked) = &document.models()[0].items()[1] else {
            panic!("second model item is a let alias");
        };
        assert!(checked.value_type().is_some());
        let formatted = format(&document);
        assert_eq!(
            formatted,
            "model M() {\n  let wave_number = math.pi / length;\n  let checked: 1 / m = wave_number;\n}\n"
        );
        let reparsed = parse("formatted-let.eqi", &formatted)
            .into_document()
            .expect("formatted let aliases parse");
        assert_eq!(format(&reparsed), formatted);
    }
}
