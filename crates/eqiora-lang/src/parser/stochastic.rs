//! Closed noise and stochastic Law syntax; numerical choices are not source meaning.
use super::Parser;
use crate::ast::{ActivationSyntax, RelationBody, RelationDecl, TextRange};
use crate::lexer::TokenKind;
use eqiora_schema::kernel::{NoiseProcess, StochasticCalculus};

impl Parser<'_> {
    pub(super) fn parse_noise(&mut self) -> Option<RelationDecl> {
        let start = self.expect_keyword("noise")?.range().start();
        let name = self.declaration_name("noise group name")?.text().to_owned();
        self.expect(TokenKind::Colon, "`:` before process law")?;
        let process = self
            .expect_identifier("wiener or poisson")?
            .text()
            .to_owned();
        let (process, parameter_name) = match process.as_str() {
            "wiener" => (NoiseProcess::Wiener, "covariance"),
            "poisson" => (NoiseProcess::Poisson, "intensity"),
            _ => {
                self.error_here("noise process must be wiener or poisson");
                return None;
            }
        };
        self.expect(TokenKind::LeftParen, "`(` before process parameters")?;
        self.expect_keyword(parameter_name)?;
        self.expect(TokenKind::Equal, "`=` before noise parameter")?;
        let parameter = self.parse_expression(0)?;
        let independent = if self.at(TokenKind::Comma) {
            self.bump();
            self.expect_keyword("independent")?;
            self.expect(TokenKind::Equal, "`=` before independence assumption")?;
            if self.at_keyword("true") {
                self.bump();
                true
            } else if self.at_keyword("false") {
                self.bump();
                false
            } else {
                self.error_here("independence must be an explicit true or false");
                return None;
            }
        } else {
            false
        };
        self.expect(TokenKind::RightParen, "`)` after process parameters")?;
        let end = self
            .expect(TokenKind::Semicolon, "`;` after noise declaration")?
            .range()
            .end();
        Some(RelationDecl {
            activation_name_range: None,
            comments: Default::default(),
            name,
            activation: ActivationSyntax::Continuous,
            domain: None,
            body: RelationBody::Noise {
                process,
                parameter: Box::new(parameter),
                independent,
            },
            range: TextRange::new(start, end),
        })
    }

    pub(super) fn parse_stochastic_law(
        &mut self,
        start: u32,
        name: String,
    ) -> Option<RelationDecl> {
        self.expect_keyword("for")?;
        let target = self.expect_identifier("owned State")?.text().to_owned();
        self.expect(
            TokenKind::LeftBrace,
            "`{` before stochastic Law coefficients",
        )?;
        let mut calculus = None;
        let mut drift = None;
        let mut diffusions = Vec::new();
        let mut jumps = Vec::new();
        let mut groups = std::collections::BTreeSet::new();
        while !self.at(TokenKind::RightBrace) && !self.at(TokenKind::Eof) {
            let term = self
                .expect_identifier("calculus, drift, diffusion or jump")?
                .text()
                .to_owned();
            match term.as_str() {
                "calculus" => {
                    let name = self
                        .expect_identifier("ito or stratonovich")?
                        .text()
                        .to_owned();
                    let value = match name.as_str() {
                        "ito" => StochasticCalculus::Ito,
                        "stratonovich" => StochasticCalculus::Stratonovich,
                        _ => {
                            self.error_here("stochastic calculus must be ito or stratonovich");
                            return None;
                        }
                    };
                    if calculus.replace(value).is_some() {
                        self.error_here("stochastic calculus must occur exactly once");
                        return None;
                    }
                }
                "drift" => {
                    if drift.replace(self.parse_expression(0)?).is_some() {
                        self.error_here("stochastic drift must occur exactly once");
                        return None;
                    }
                }
                "diffusion" | "jump" => {
                    let group = self
                        .expect_identifier("exact noise group")?
                        .text()
                        .to_owned();
                    if !groups.insert(group.clone()) {
                        self.error_here("noise group is bound more than once");
                        return None;
                    }
                    self.expect(TokenKind::Equal, "`=` before increment coefficient")?;
                    let coefficient = self.parse_expression(0)?;
                    if term == "diffusion" {
                        diffusions.push((group, coefficient));
                    } else {
                        jumps.push((group, coefficient));
                    }
                }
                _ => {
                    self.error_here("stochastic Law terms are calculus, drift, diffusion and jump");
                    return None;
                }
            }
            self.expect(TokenKind::Semicolon, "`;` after stochastic Law term")?;
        }
        let end = self
            .expect(TokenKind::RightBrace, "`}` after stochastic Law")?
            .range()
            .end();
        if (diffusions.is_empty() && jumps.is_empty())
            || diffusions.is_empty() != calculus.is_none()
            || calculus.is_some() != drift.is_some()
        {
            self.error_here(
                "continuous noise requires calculus and drift; a pure jump Law has neither",
            );
            return None;
        }
        Some(RelationDecl {
            activation_name_range: None,
            comments: Default::default(),
            name,
            activation: ActivationSyntax::Continuous,
            domain: None,
            body: RelationBody::Stochastic {
                target,
                calculus,
                drift: drift.map(Box::new),
                diffusions,
                jumps,
            },
            range: TextRange::new(start, end),
        })
    }
}
