//! Relation declaration and natural-equation parsing.

use crate::ast::{
    ActivationSyntax, InitialDecl, RelationCondition, RelationDecl, RelationFamilyDecl, TextRange,
};
use crate::lexer::TokenKind;

use super::Parser;

pub(super) enum ParsedRelation {
    Ordinary(RelationDecl),
    Family(RelationFamilyDecl),
}

impl Parser<'_> {
    pub(super) fn parse_initial(&mut self) -> Option<InitialDecl> {
        let start = self.expect_keyword("initial")?.range().start();
        self.expect(TokenKind::LeftBrace, "`{` before initialization equations")?;
        let mut equations = Vec::new();
        while !self.at(TokenKind::RightBrace) && !self.at(TokenKind::Eof) {
            let condition = self.parse_relation_statement()?;
            if condition.kind() != eqiora_schema::kernel::RelationConditionKind::Equality {
                self.error_here("initial blocks admit only equality conditions");
                return None;
            }
            equations.push(condition);
        }
        if equations.is_empty() {
            self.error_here("initial requires at least one equation");
        }
        let end = self
            .expect(TokenKind::RightBrace, "`}` after initialization")?
            .range()
            .end();
        Some(InitialDecl {
            comments: Default::default(),
            equations,
            range: TextRange::new(start, end),
        })
    }

    pub(super) fn parse_component_relation(&mut self) -> Option<ParsedRelation> {
        let start = self.expect_keyword("relation")?.range().start();
        let name = self.declaration_name("Relation name")?.text().to_owned();
        let binder = if self.at(TokenKind::LeftBracket) {
            Some(self.parse_index_family_binder()?)
        } else {
            None
        };
        let domain = if self.at_keyword("on") {
            self.bump();
            Some(self.expect_identifier("Relation Domain")?.text().to_owned())
        } else {
            None
        };
        let (activation, activation_name_range) = if self.at_keyword("at") {
            self.bump();
            let token = self.expect_identifier("Relation activation")?;
            (
                ActivationSyntax::Named(token.text().to_owned()),
                Some(token.range()),
            )
        } else {
            (ActivationSyntax::Continuous, None)
        };
        self.expect(TokenKind::LeftBrace, "`{` before equations")?;
        let mut equations = Vec::new();
        while !self.at(TokenKind::RightBrace) && !self.at(TokenKind::Eof) {
            equations.push(self.parse_relation_statement()?);
        }
        if equations.is_empty() {
            self.error_here("Relation requires at least one equation");
        }
        let end = self
            .expect(TokenKind::RightBrace, "`}` after Relation")?
            .range()
            .end();
        let relation = RelationDecl {
            activation_name_range,
            comments: Default::default(),
            name,
            activation,
            domain,
            body: crate::ast::RelationBody::Conditions(equations),
            range: TextRange::new(start, end),
        };
        let Some(binder) = binder else {
            return Some(ParsedRelation::Ordinary(relation));
        };
        Some(ParsedRelation::Family(RelationFamilyDecl {
            relation,
            binder,
        }))
    }

    fn parse_relation_statement(&mut self) -> Option<RelationCondition> {
        if self.at_keyword("inequality") || self.at_keyword("complementarity") {
            return self.parse_constraint_statement();
        }
        if self.at_keyword("inclusion") {
            self.error_here("differential inclusions require an explicitly supported mathematical contract; no penalty substitution is available");
            return None;
        }
        let left = self.parse_expression(0)?;
        self.expect(TokenKind::Equal, "`=` after Relation left-hand expression")?;
        let right = self.parse_expression(0)?;
        self.expect(TokenKind::Semicolon, "`;` after equation")?;
        let range = TextRange::new(left.range().start(), right.range().end());
        Some(RelationCondition {
            kind: eqiora_schema::kernel::RelationConditionKind::Equality,
            left,
            right,
            range,
        })
    }
}
