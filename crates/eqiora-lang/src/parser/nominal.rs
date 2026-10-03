//! Parsing for finite nominal declarations and bounded family binders.

use super::*;
use crate::ast::{FamilyBinderSyntax, NamedDefinitionDecl};
use std::collections::BTreeSet;

impl Parser<'_> {
    pub(super) fn parse_finite_space(
        &mut self,
        start: u32,
        visibility: VisibilitySyntax,
    ) -> Option<NamedDefinitionDecl> {
        self.expect_keyword("space")?;
        let name = self
            .declaration_name("finite space name")?
            .text()
            .to_owned();
        self.expect(TokenKind::Equal, "`=` before finite space definition")?;
        if self.at_keyword("product") {
            self.bump();
            self.expect(TokenKind::LeftParen, "`(` before product factors")?;
            let left = self.parse_expression(0)?;
            self.expect(TokenKind::Comma, "`,` between product factors")?;
            let right = self.parse_expression(0)?;
            self.expect(TokenKind::RightParen, "`)` after product factors")?;
            let end = self
                .expect(TokenKind::Semicolon, "`;` after finite product")?
                .range()
                .end();
            return Some(NamedDefinitionDecl::plain(
                name,
                crate::ast::nominal::definition_call(
                    "product",
                    vec![left, right],
                    TextRange::new(start, end),
                ),
                TextRange::new(start, end),
                visibility,
            ));
        }
        self.expect_keyword("orthonormal")?;
        self.expect(TokenKind::LeftParen, "`(` before finite space labels")?;
        let mut labels = Vec::new();
        let mut distinct = BTreeSet::new();
        loop {
            let label = self.expect_identifier("finite space basis label")?;
            if !distinct.insert(label.text().to_owned()) {
                self.error_token(&label, "finite space basis labels must be distinct");
                return None;
            }
            labels.push(label.text().to_owned());
            if !self.at(TokenKind::Comma) {
                break;
            }
            self.bump();
        }
        self.expect(TokenKind::RightParen, "`)` after finite space labels")?;
        let end = self
            .expect(TokenKind::Semicolon, "`;` after finite space")?
            .range()
            .end();
        Some(NamedDefinitionDecl::plain(
            name,
            crate::ast::nominal::definition_call(
                "orthonormal",
                labels
                    .into_iter()
                    .map(|name| Expr {
                        resolved_enum: None,
                        resolved_nominal: None,
                        kind: crate::ExprKind::Name(name),
                        range: TextRange::new(start, end),
                    })
                    .collect(),
                TextRange::new(start, end),
            ),
            TextRange::new(start, end),
            visibility,
        ))
    }

    pub(super) fn parse_index_set(&mut self) -> Option<NamedDefinitionDecl> {
        let start = self.expect_keyword("indexset")?.range().start();
        let name = self.declaration_name("index set name")?.text().to_owned();
        self.expect(TokenKind::Equal, "`=` before index set definition")?;
        self.expect_keyword("range")?;
        self.expect(TokenKind::LeftParen, "`(` before index set extent")?;
        let extent = self.parse_expression(0)?;
        self.expect(TokenKind::RightParen, "`)` after index set extent")?;
        let end = self
            .expect(TokenKind::Semicolon, "`;` after index set")?
            .range()
            .end();
        Some(NamedDefinitionDecl::plain(
            name,
            crate::ast::nominal::definition_call("range", vec![extent], TextRange::new(start, end)),
            TextRange::new(start, end),
            VisibilitySyntax::Private,
        ))
    }

    pub(super) fn parse_index_family_binder(&mut self) -> Option<FamilyBinderSyntax> {
        let start = self
            .expect(TokenKind::LeftBracket, "`[` before index family binder")?
            .range()
            .start();
        let binder = self
            .expect_identifier("index family binder")?
            .text()
            .to_owned();
        self.expect_keyword("in")?;
        let set = self.parse_name_path("index family set")?;
        let end = self
            .expect(TokenKind::RightBracket, "`]` after index family binder")?
            .range()
            .end();
        Some(FamilyBinderSyntax {
            member: binder,
            set,
            range: TextRange::new(start, end),
        })
    }
}
