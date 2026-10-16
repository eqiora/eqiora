//! Explicitly normalized probability laws; no source sampling operation.
use super::Parser;
use crate::ast::{ActivationSyntax, RelationBody, RelationDecl, TextRange};
use crate::lexer::TokenKind;

impl Parser<'_> {
    pub(super) fn parse_distribution(&mut self) -> Option<RelationDecl> {
        let start = self.expect_keyword("distribution")?.range().start();
        let name = self
            .declaration_name("probability law name")?
            .text()
            .to_owned();
        self.expect(TokenKind::Colon, "`:` before probability law")?;
        let constructor = self
            .expect_identifier("normalize_mass or normalize_density")?
            .text()
            .to_owned();
        self.expect(TokenKind::LeftParen, "`(` before probability law inputs")?;
        let body = match constructor.as_str() {
            "normalize_mass" => {
                let outcomes = self.parse_value_type()?;
                self.expect(TokenKind::Comma, "`,` before mass weights")?;
                self.expect_keyword("weights")?;
                self.expect(TokenKind::Equal, "`=` before mass weights")?;
                RelationBody::ProbabilityMass {
                    outcomes,
                    weights: Box::new(self.parse_expression(0)?),
                }
            }
            "normalize_density" => {
                let measure = self
                    .expect_identifier("measure or spherical_measure")?
                    .text()
                    .to_owned();
                let measure = match measure.as_str() {
                    "measure" => None,
                    "spherical_measure" => {
                        Some(eqiora_schema::kernel::ObservableMeasure::SphericalVolume)
                    }
                    _ => {
                        self.error_here("density requires an explicit measure(domain)");
                        return None;
                    }
                };
                self.expect(TokenKind::LeftParen, "`(` before measure Domain")?;
                let domain = self.expect_identifier("measure Domain")?.text().to_owned();
                self.expect(TokenKind::RightParen, "`)` after measure Domain")?;
                self.expect(TokenKind::Comma, "`,` before density")?;
                self.expect_keyword("density")?;
                self.expect(TokenKind::Equal, "`=` before density")?;
                RelationBody::ProbabilityDensity {
                    domain,
                    measure,
                    density: Box::new(self.parse_expression(0)?),
                }
            }
            _ => {
                self.error_here(
                    "probability law must explicitly normalize_mass or normalize_density",
                );
                return None;
            }
        };
        self.expect(TokenKind::RightParen, "`)` after probability law inputs")?;
        let end = self
            .expect(TokenKind::Semicolon, "`;` after probability law")?
            .range()
            .end();
        Some(RelationDecl {
            activation_name_range: None,
            comments: Default::default(),
            name,
            activation: ActivationSyntax::Continuous,
            domain: None,
            body,
            range: TextRange::new(start, end),
        })
    }
}
