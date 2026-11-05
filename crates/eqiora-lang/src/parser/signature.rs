//! Shared Component and Model external interface parsing.
use super::*;
use crate::ast::{ClockRequirementDecl, SignatureItem};

impl Parser<'_> {
    pub(super) fn parse_signature(&mut self) -> Option<Vec<SignatureItem>> {
        self.expect(
            TokenKind::LeftParen,
            "`(` before signature (use `()` when empty)",
        )?;
        let mut requirements = Vec::new();
        while !self.at(TokenKind::RightParen) && !self.at(TokenKind::Eof) {
            let start = self.current().range().start();
            let requirement = if self.at_keyword("support") {
                SignatureItem::Support(self.parse_support_slot(start, VisibilitySyntax::Public)?)
            } else if self.at_keyword("expression") {
                self.bump();
                SignatureItem::Expression(self.parse_field_head(
                    start,
                    crate::FieldRoleSyntax::Variable,
                    false,
                )?)
            } else if self.at_keyword("variable") || self.at_keyword("state") {
                SignatureItem::Field(self.parse_field(false)?)
            } else if self.at_keyword("input") {
                SignatureItem::Input(self.parse_field(false)?)
            } else if self.at_keyword("output") {
                SignatureItem::Output(self.parse_field(false)?)
            } else if self.at_keyword("parameter") {
                SignatureItem::Parameter(self.parse_component_parameter(
                    start,
                    VisibilitySyntax::Public,
                    false,
                )?)
            } else if self.at_keyword("port") {
                match self.parse_component_port(start, VisibilitySyntax::Public, false)? {
                    ParsedComponentPort::Ordinary(port) => {
                        if matches!(port.syntax(), PortSyntax::Signal { .. }) {
                            self.error_here("causal interfaces use input/output signature entries");
                            return None;
                        }
                        SignatureItem::Port(port)
                    }
                    ParsedComponentPort::Family(port) => SignatureItem::PortFamily(port),
                }
            } else if self.at_keyword("property") {
                self.bump();
                let name = self
                    .declaration_name("property requirement name")?
                    .text()
                    .to_owned();
                self.expect(TokenKind::Colon, "`:` before property contract")?;
                let contract = self.parse_name_path("property contract name")?;
                let range = TextRange::new(start, contract.range().end());
                SignatureItem::Property(crate::ast_property::ComponentPropertyDecl {
                    comments: Default::default(),
                    name,
                    contract,
                    range,
                })
            } else if self.at_keyword("clock") {
                self.bump();
                let name = self
                    .declaration_name("required clock name")?
                    .text()
                    .to_owned();
                self.expect(TokenKind::Colon, "`:` before periodic clock contract")?;
                let end = self.expect_keyword("periodic")?.range().end();
                SignatureItem::Clock(ClockRequirementDecl {
                    comments: Default::default(),
                    name,
                    range: TextRange::new(start, end),
                })
            } else {
                self.error_here("expected parameter, expression, support, variable, state, clock, input, output, port, or property signature entry");
                return None;
            };
            requirements.push(requirement);
            if !self.at(TokenKind::Comma) {
                break;
            }
            self.bump();
        }
        self.expect(TokenKind::RightParen, "`)` after signature")?;
        Some(requirements)
    }
}
