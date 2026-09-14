//! Method-neutral fixed-domain conservation Law declarations.

use super::Parser;
use crate::ast::{ActivationSyntax, ConservationSyntax, RelationBody, RelationDecl, TextRange};
use crate::lexer::TokenKind;

impl Parser<'_> {
    pub(super) fn parse_law(&mut self) -> Option<RelationDecl> {
        let start = self.expect_keyword("law")?.range().start();
        let name = self.declaration_name("Law name")?.text().to_owned();
        self.expect_keyword("on")?;
        let domain = self.expect_identifier("Law support")?.text().to_owned();
        self.expect(TokenKind::LeftBrace, "`{` before Law terms")?;
        let mut storage = None;
        let mut flux = None;
        let mut source = None;
        while !self.at(TokenKind::RightBrace) && !self.at(TokenKind::Eof) {
            let term = self
                .expect_identifier("storage, flux, or source")?
                .text()
                .to_owned();
            let expression = self.parse_expression(0)?;
            self.expect(TokenKind::Semicolon, "`;` after Law term")?;
            let slot = match term.as_str() {
                "storage" => &mut storage,
                "flux" => &mut flux,
                "source" => &mut source,
                _ => {
                    self.error_here("Law terms are storage, flux, and source");
                    return None;
                }
            };
            if slot.replace(expression).is_some() {
                self.error_here("a Law term must be declared exactly once");
                return None;
            }
        }
        let end = self
            .expect(TokenKind::RightBrace, "`}` after Law")?
            .range()
            .end();
        let (Some(flux), Some(source)) = (flux, source) else {
            self.error_here("Law requires explicit flux and source; only storage may be omitted for steady balance");
            return None;
        };
        Some(RelationDecl {
            comments: Default::default(),
            name,
            activation: ActivationSyntax::Continuous,
            domain: Some(domain),
            body: RelationBody::Conservation(Box::new(ConservationSyntax {
                storage,
                flux,
                source,
            })),
            range: TextRange::new(start, end),
        })
    }
}
