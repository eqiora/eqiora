//! Private crossing events; guard typing belongs to the common compiler.
use super::{Parser, TokenKind};
use crate::{EventDecl, TextRange};
use eqiora_schema::kernel::EventDirection;

impl Parser<'_> {
    pub(super) fn parse_event(&mut self) -> Option<EventDecl> {
        let start = self.expect_keyword("event")?.range().start();
        let name = self.declaration_name("event name")?.text().to_owned();
        self.expect(TokenKind::Equal, "`=` before event definition")?;
        self.expect_keyword("crossing")?;
        self.expect(TokenKind::LeftParen, "`(` after crossing")?;
        let guard = self.parse_expression(0)?;
        self.expect(TokenKind::Comma, "`,` before crossing direction")?;
        self.expect_keyword("direction")?;
        self.expect(TokenKind::Equal, "`=` before crossing direction")?;
        let token = self.expect_identifier("crossing direction")?;
        let direction = match token.text() {
            "any" => EventDirection::Any,
            "rising" => EventDirection::Rising,
            "falling" => EventDirection::Falling,
            _ => {
                self.error_token(
                    &token,
                    "expected crossing direction `any`, `rising`, or `falling`",
                );
                return None;
            }
        };
        let priority = if self.at(TokenKind::Comma) {
            self.bump();
            self.expect_keyword("priority")?;
            self.expect(TokenKind::Equal, "`=` before event priority")?;
            let negative = self.at(TokenKind::Minus);
            if negative {
                self.bump();
            }
            let token = self.expect(TokenKind::Number, "signed integer event priority")?;
            let spelling = if negative {
                format!("-{}", token.text())
            } else {
                token.text().to_owned()
            };
            match spelling.parse::<i64>() {
                Ok(value) => value,
                Err(_) => {
                    self.error_token(
                        &token,
                        "event priority must be a signed 64-bit integer literal",
                    );
                    return None;
                }
            }
        } else {
            0
        };
        self.expect(TokenKind::RightParen, "`)` after crossing direction")?;
        let end = self
            .expect(TokenKind::Semicolon, "`;` after event")?
            .range()
            .end();
        Some(EventDecl {
            comments: Default::default(),
            name,
            guard,
            direction,
            priority,
            range: TextRange::new(start, end),
        })
    }
}
