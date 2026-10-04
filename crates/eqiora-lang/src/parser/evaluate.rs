//! Point bindings name exact coordinates rather than runtime keyword arguments.
use super::*;

impl Parser<'_> {
    pub(super) fn parse_evaluate(&mut self, path: NamePath) -> Option<(Expr, usize)> {
        self.expect(TokenKind::LeftParen, "`(` after evaluate")?;
        let (value, mut depth) = self.parse_expression_with_depth(0)?;
        self.expect(TokenKind::Comma, "`,` before evaluation point")?;
        self.expect_keyword("at")?;
        self.expect(TokenKind::Equal, "`=` after at")?;
        self.expect(TokenKind::LeftParen, "`(` before coordinate bindings")?;
        let mut at = Vec::new();
        loop {
            let coordinate = self.parse_name_path("evaluation coordinate")?;
            if at
                .iter()
                .any(|(name, _): &(NamePath, Expr)| name.as_str() == coordinate.as_str())
            {
                self.error_here("point evaluation repeats a coordinate binding");
                return None;
            }
            self.expect(TokenKind::Equal, "`=` after evaluation coordinate")?;
            let (point, point_depth) = self.parse_expression_with_depth(0)?;
            depth = depth.max(point_depth);
            at.push((coordinate, point));
            if !self.at(TokenKind::Comma) {
                break;
            }
            self.bump();
        }
        self.expect(TokenKind::RightParen, "`)` after coordinate bindings")?;
        let side = if self.at(TokenKind::Comma) {
            self.bump();
            self.expect_keyword("side")?;
            self.expect(TokenKind::Equal, "`=` after side")?;
            if self.at_keyword("lower") {
                self.bump();
                Some(BoundarySideSyntax::Lower)
            } else {
                self.expect_keyword("upper")?;
                Some(BoundarySideSyntax::Upper)
            }
        } else {
            None
        };
        let end = self
            .expect(TokenKind::RightParen, "`)` after evaluate")?
            .range()
            .end();
        Some((
            Expr {
                kind: ExprKind::Evaluate {
                    value: Box::new(value),
                    at,
                    side,
                },
                range: TextRange::new(path.range().start(), end),
                resolved_enum: None,
                resolved_nominal: None,
            },
            self.parent_depth(depth)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::parse;

    #[test]
    fn point_bindings_roundtrip_and_reject_ambiguous_syntax() {
        let source = "model M(){observable probe:1=evaluate(f,at=(x=0.25[m]),side=lower);}";
        let document = parse("point.eqi", source).into_document().unwrap();
        let text = crate::format(&document);
        assert!(text.contains("evaluate(f, at = (x = "));
        assert!(text.contains("), side = lower)"));
        assert!(parse("point.eqi", &text).diagnostics().is_empty());
        for point in ["()", "(x=0,x=1)", "(x)", "(x=0,)"] {
            assert!(
                !parse(
                    "invalid.eqi",
                    &format!("model M(){{observable p:1=evaluate(f,at={point});}}")
                )
                .diagnostics()
                .is_empty()
            );
        }
    }
}
