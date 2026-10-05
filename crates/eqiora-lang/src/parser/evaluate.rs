//! Point bindings name exact coordinates rather than runtime keyword arguments.
use super::*;

impl Parser<'_> {
    pub(super) fn parse_coordinate_binding(&mut self, path: NamePath) -> Option<(Expr, usize)> {
        use eqiora_schema::kernel::CoordinateMapFactor;
        let factor = match path.as_str() {
            "jacobian_determinant" => Some(CoordinateMapFactor::SignedJacobian),
            "volume_jacobian" => Some(CoordinateMapFactor::VolumeScale),
            "map_orientation" => Some(CoordinateMapFactor::Orientation),
            _ => None,
        };
        let pullback = path.as_str() == "pullback";
        let mapping = pullback || factor.is_some();
        self.expect(TokenKind::LeftParen, "`(` after coordinate operator")?;
        let (value, mut depth) = if factor.is_some() {
            (None, 0)
        } else {
            let (value, depth) = self.parse_expression_with_depth(0)?;
            self.expect(TokenKind::Comma, "`,` before coordinate bindings")?;
            (Some(value), depth)
        };
        let mut source = Vec::new();
        if mapping {
            self.expect_keyword("from")?;
            self.expect(TokenKind::Equal, "`=` after from")?;
            self.expect(TokenKind::LeftParen, "`(` before source coordinates")?;
            loop {
                let coordinate = self.parse_name_path("source coordinate")?;
                if source
                    .iter()
                    .any(|name: &NamePath| name.as_str() == coordinate.as_str())
                {
                    self.error_here("pullback repeats a source coordinate");
                    return None;
                }
                source.push(coordinate);
                if !self.at(TokenKind::Comma) {
                    break;
                }
                self.bump();
            }
            self.expect(TokenKind::RightParen, "`)` after source coordinates")?;
            self.expect(TokenKind::Comma, "`,` before target bindings")?;
        }
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
        let side = if !mapping && self.at(TokenKind::Comma) {
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
                kind: if let Some(factor) = factor {
                    ExprKind::CoordinateMapFactor { factor, source, at }
                } else if pullback {
                    ExprKind::Pullback {
                        value: Box::new(value.expect("value operator")),
                        source,
                        at,
                    }
                } else {
                    ExprKind::Evaluate {
                        value: Box::new(value.expect("value operator")),
                        at,
                        side,
                    }
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
    fn differential_factors_roundtrip_and_require_complete_binder_syntax() {
        for operator in ["jacobian_determinant", "volume_jacobian", "map_orientation"] {
            let source = format!(
                "model M(){{observable p:1={operator}(from=(xi,eta),at=(x=2*xi+eta,y=3*eta));}}"
            );
            let document = parse("factor.eqi", &source).into_document().unwrap();
            let formatted = crate::format(&document);
            assert!(formatted.contains(&format!("{operator}(from = (xi, eta), at = (")));
            assert_eq!(
                crate::format(&parse("factor.eqi", &formatted).into_document().unwrap()),
                formatted
            );
            for arguments in [
                "from=(),at=(x=xi)",
                "from=(xi,xi),at=(x=xi)",
                "from=(xi),at=()",
                "from=(xi),at=(x=xi,x=xi)",
                "from=(xi),at=(x=xi),side=lower",
                "f,from=(xi),at=(x=xi)",
            ] {
                assert!(
                    !parse(
                        "invalid.eqi",
                        &format!("model M(){{observable p:1={operator}({arguments});}}")
                    )
                    .diagnostics()
                    .is_empty()
                );
            }
        }
    }

    #[test]
    fn pullback_binders_roundtrip_without_losing_source_or_target_coordinates() {
        let source = "model M(){observable p:m=pullback(f,from=(ref.x,ref.y),at=(body.x=2*ref.x+ref.y,body.y=3*ref.y));}";
        let document = parse("map.eqi", source).into_document().unwrap();
        let formatted = crate::format(&document);
        assert!(formatted.contains("pullback(f, from = (ref.x, ref.y), at = (body.x = "));
        assert!(parse("map.eqi", &formatted).diagnostics().is_empty());
        assert_eq!(
            crate::format(&parse("map.eqi", &formatted).into_document().unwrap()),
            formatted
        );
        for expression in [
            "pullback(f,from=(),at=(x=xi))",
            "pullback(f,from=(xi,xi),at=(x=xi))",
            "pullback(f,from=(xi),at=())",
            "pullback(f,from=(xi),at=(x=xi,x=2*xi))",
            "pullback(f,at=(x=xi))",
            "pullback(f,from=(xi),at=(x=xi),side=upper)",
        ] {
            let source = format!("model M(){{observable p:m={expression};}}");
            assert!(
                !parse("invalid.eqi", &source).diagnostics().is_empty(),
                "{expression}"
            );
        }
    }

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
