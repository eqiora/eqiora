use eqiora_core::ScalarDomain;

use crate::ast::{TextRange, ValueTypeSyntax, ValueTypeSyntaxKind};
use crate::lexer::TokenKind;

use super::Parser;

impl Parser<'_> {
    pub(super) fn parse_value_type(&mut self) -> Option<ValueTypeSyntax> {
        self.parse_value_type_at_depth(0)
    }

    fn parse_value_type_at_depth(&mut self, depth: usize) -> Option<ValueTypeSyntax> {
        if depth >= 256 {
            self.error_here("source resource limit exceeded: type nesting exceeds 256");
            return None;
        }
        let start = self.current().range().start();
        if ["coordinates", "map", "counts", "index"]
            .iter()
            .any(|name| self.at_keyword(name))
        {
            return self.parse_nominal_value_type(depth, start);
        }
        if self.at_keyword("integer") || self.at_keyword("bool") {
            let token = self.bump();
            return Some(ValueTypeSyntax {
                resolved_nominal: None,
                kind: Box::new(ValueTypeSyntaxKind::Scalar {
                    domain: if token.text() == "bool" {
                        ScalarDomain::Boolean
                    } else {
                        ScalarDomain::Integer
                    },
                    dimension: crate::Expr {
                        resolved_enum: None,
                        resolved_nominal: None,
                        kind: crate::ExprKind::Number(
                            crate::DecimalLiteral::parse("1").expect("one"),
                        ),
                        range: token.range(),
                    },
                }),
                range: token.range(),
            });
        }
        if self.at_keyword("complex") {
            self.bump();
            self.expect(TokenKind::LeftAngle, "`<` after complex")?;
            let dimension = self.parse_dimension_expression()?;
            let end = self
                .expect(TokenKind::RightAngle, "`>` after complex dimension")?
                .range()
                .end();
            return Some(ValueTypeSyntax {
                resolved_nominal: None,
                kind: Box::new(ValueTypeSyntaxKind::Scalar {
                    domain: ScalarDomain::Complex,
                    dimension,
                }),
                range: TextRange::new(start, end),
            });
        }
        if !["vector", "tensor", "array"]
            .iter()
            .any(|name| self.at_keyword(name))
        {
            let dimension = self.parse_dimension_expression()?;
            let path = match dimension.kind() {
                crate::ExprKind::Name(name) => {
                    Some(crate::NamePath::single(name.clone(), dimension.range()))
                }
                crate::ExprKind::Path(path) => Some(path.clone()),
                _ => None,
            };
            return Some(if let Some(path) = path {
                ValueTypeSyntax {
                    kind: Box::new(ValueTypeSyntaxKind::Named(path)),
                    range: dimension.range(),
                    resolved_nominal: None,
                }
            } else {
                ValueTypeSyntax::real(dimension)
            });
        }
        let constructor = self.bump().text().to_owned();
        self.expect(TokenKind::LeftAngle, "`<` after type constructor")?;
        let element = self.parse_value_type_at_depth(depth + 1)?;
        self.expect(TokenKind::Comma, "`,` before type extent")?;
        if constructor == "array" {
            // Comparisons have lower precedence than static arithmetic; the
            // enclosing `>` terminates this expression rather than comparing it.
            let extent = self.parse_expression(9)?;
            let end = self
                .expect(TokenKind::RightAngle, "`>` after array extent")?
                .range()
                .end();
            return match crate::SourceAstFactory::value_type(
                ValueTypeSyntaxKind::Array {
                    element: Box::new(element),
                    extent,
                },
                TextRange::new(start, end),
            ) {
                Ok(value) => Some(value),
                Err(error) => {
                    self.error_previous(error.message());
                    None
                }
            };
        }
        let mut extents = vec![self.parse_type_extent()?];
        while self.at(TokenKind::Comma) {
            if constructor != "tensor" {
                self.error_here("vector and array require exactly one extent");
                return None;
            }
            if extents.len() >= 4 {
                self.error_here("source resource limit exceeded: spatial tensor rank exceeds 4");
                return None;
            }
            self.bump();
            extents.push(self.parse_type_extent()?);
        }
        let end = self
            .expect(TokenKind::RightAngle, "`>` after type extents")?
            .range()
            .end();
        let kind = match constructor.as_str() {
            "vector" => ValueTypeSyntaxKind::Vector {
                scalar: Box::new(element),
                extent: extents[0],
            },
            "tensor" => ValueTypeSyntaxKind::Tensor {
                scalar: Box::new(element),
                extents,
            },
            _ => unreachable!("closed type constructors"),
        };
        match crate::SourceAstFactory::value_type(kind, TextRange::new(start, end)) {
            Ok(value) => Some(value),
            Err(error) => {
                self.error_previous(error.message());
                None
            }
        }
    }

    fn parse_nominal_value_type(&mut self, depth: usize, start: u32) -> Option<ValueTypeSyntax> {
        let constructor = self.bump().text().to_owned();
        self.expect(TokenKind::LeftAngle, "`<` before nominal type declaration")?;
        let kind = if matches!(constructor.as_str(), "coordinates" | "map") {
            if [
                "coordinates",
                "map",
                "counts",
                "index",
                "array",
                "vector",
                "tensor",
            ]
            .iter()
            .any(|name| self.at_keyword(name))
            {
                self.error_here("finite component types require a scalar component type");
                return None;
            }
            let scalar = Box::new(self.parse_value_type_at_depth(depth + 1)?);
            self.expect(TokenKind::Comma, "`,` before finite basis")?;
            let source = self.parse_finite_basis()?;
            if constructor == "map" {
                self.expect(TokenKind::Comma, "`,` before output basis")?;
                let target = self.parse_finite_basis()?;
                ValueTypeSyntaxKind::LinearMap {
                    scalar,
                    source,
                    target,
                }
            } else {
                ValueTypeSyntaxKind::Coordinates {
                    scalar,
                    basis: source,
                }
            }
        } else {
            let name = self.parse_name_path("nominal type declaration")?;
            if constructor == "counts" {
                ValueTypeSyntaxKind::Counts(name)
            } else {
                ValueTypeSyntaxKind::Index(name)
            }
        };
        let end = self
            .expect(TokenKind::RightAngle, "`>` after nominal type")?
            .range()
            .end();
        match crate::SourceAstFactory::value_type(kind, TextRange::new(start, end)) {
            Ok(value) => Some(value),
            Err(error) => {
                self.error_previous(error.message());
                None
            }
        }
    }

    fn parse_finite_basis(&mut self) -> Option<crate::FiniteBasisSyntax> {
        let dual = self.at_keyword("dual");
        if dual {
            self.bump();
            self.expect(TokenKind::LeftAngle, "`<` after dual")?;
        }
        let name = self.parse_name_path("finite space declaration")?;
        if dual {
            self.expect(TokenKind::RightAngle, "`>` after dual basis")?;
        }
        Some(crate::FiniteBasisSyntax { name, dual })
    }

    fn parse_type_extent(&mut self) -> Option<u32> {
        let extent = self.parse_u64("positive integer type extent")?;
        let Some(extent) = u32::try_from(extent).ok().filter(|extent| *extent > 0) else {
            self.error_previous("type extent must be a positive u32 integer");
            return None;
        };
        Some(extent)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Item, ValueTypeSyntaxKind, format, parse};

    #[test]
    fn mathematical_type_constructors_have_one_round_trip_spelling() {
        for value_type in [
            "V",
            "m ^ (-1 / 2)",
            "complex<V>",
            "vector<complex<V>, 3>",
            "tensor<Pa, 2, 2>",
            "array<array<V, 2>, 3>",
            "array<vector<complex<V>, 2>, 3>",
            "coordinates<complex<1>, Spin>",
            "coordinates<V, dual<Control>>",
            "map<complex<1>, Spin, Spin>",
            "map<1 / s, dual<Control>, dual<Spin>>",
        ] {
            let source = format!("model Types() {{ parameter value: {value_type} = 0; }}");
            let parsed = parse("types.eqi", &source);
            let document = parsed.document().expect(value_type);
            let canonical = format(document);
            assert!(canonical.contains(value_type), "{canonical}");
            let replayed = parse("formatted.eqi", &canonical).into_document().unwrap();
            assert_eq!(format(&replayed), canonical);
        }
    }

    #[test]
    fn arrays_retain_their_element_type_instead_of_becoming_spatial_axes() {
        let document = parse(
            "array.eqi",
            "model M() { parameter channels: array<vector<complex<V>, 2>, 3> = 0; }",
        )
        .into_document()
        .unwrap();
        let Item::Parameter(parameter) = &document.models()[0].items()[0] else {
            panic!("parameter");
        };
        let ValueTypeSyntaxKind::Array { element, extent } = parameter.value_type().kind() else {
            panic!("array");
        };
        assert!(matches!(extent.kind(), crate::ExprKind::Number(n) if n.to_i64().ok() == Some(3)));
        let ValueTypeSyntaxKind::Vector { scalar, extent } = element.kind() else {
            panic!("vector");
        };
        assert_eq!(*extent, 2);
        assert_eq!(
            scalar.scalar_domain(),
            Some(eqiora_core::ScalarDomain::Complex)
        );
    }

    #[test]
    fn malformed_or_unbounded_types_reject_at_the_source_type() {
        for value_type in [
            "real<V>",
            "complex<complex<V>>",
            "vector<array<V, 2>, 3>",
            "tensor<vector<V, 2>, 2, 2>",
            "vector<V, 0>",
            "array<V, 1.5>",
            "vector<V, 2, 2>",
            "tensor<V, 2>",
            "tensor<V, 2, 2, 2, 2, 2>",
            "array<V, 65537>",
            "array<array<V, 256>, 257>",
            "coordinates<array<1, 2>, Spin>",
            "map<vector<1, 2>, Spin, Spin>",
            "map<1, Spin>",
            "coordinates<1, dual<dual<Spin>>>",
            "array<coordinates<1, Spin>, 1>",
        ] {
            let source = format!("model M() {{ parameter invalid: {value_type} = 0; }}");
            assert!(
                parse("invalid.eqi", &source).into_document().is_err(),
                "{value_type}"
            );
        }
        let limit = "model M() { parameter values: array<V, 65536> = 0; }";
        assert!(parse("limit.eqi", limit).into_document().is_ok());
        // Unary expressions are syntactically valid; exact static admission in
        // the compiler rejects negative extents after expression evaluation.
        assert!(
            parse("negative.eqi", "model M(){variable x:array<V,-1>;}")
                .into_document()
                .is_ok()
        );
        let nested = format!(
            "model M() {{ parameter values: {}V{} = 0; }}",
            "array<".repeat(256),
            ", 1>".repeat(256)
        );
        assert!(parse("nested.eqi", &nested).into_document().is_err());
    }
}
