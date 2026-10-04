use crate::cartesian::CartesianCoordinateSyntax;

use super::*;

impl Parser<'_> {
    pub(super) fn parse_product_support(&mut self) -> Option<DomainDecl> {
        let start = self.expect_keyword("support")?.range().start();
        let name = self
            .declaration_name("product support name")?
            .text()
            .to_owned();
        self.expect(TokenKind::Colon, "`:` before product support")?;
        self.expect_keyword("product")?;
        self.expect(TokenKind::LeftParen, "`(` before coordinate factors")?;
        let mut factors = vec![
            self.expect_identifier("coordinate factor support")?
                .text()
                .to_owned(),
        ];
        while self.at(TokenKind::Comma) {
            self.bump();
            factors.push(
                self.expect_identifier("coordinate factor support")?
                    .text()
                    .to_owned(),
            );
        }
        self.expect(TokenKind::RightParen, "`)` after coordinate factors")?;
        let end = self
            .expect(TokenKind::Semicolon, "`;` after product support")?
            .range()
            .end();
        Some(DomainDecl {
            comments: Default::default(),
            name,
            syntax: DomainSyntax::Product { factors },
            range: TextRange::new(start, end),
        })
    }

    pub(super) fn parse_domain(&mut self) -> Option<DomainDecl> {
        let start = self.expect_keyword("domain")?.range().start();
        let name = self.declaration_name("Domain name")?.text().to_owned();
        self.expect(TokenKind::Equal, "`=` before Domain geometry")?;
        let syntax = if self.at_keyword("box") {
            self.bump();
            self.expect(TokenKind::LeftParen, "`(` after `box`")?;
            let mut coordinates = vec![self.parse_cartesian_coordinate()?];
            while self.at(TokenKind::Comma) {
                self.bump();
                coordinates.push(self.parse_cartesian_coordinate()?);
            }
            self.expect(TokenKind::RightParen, "`)` after Cartesian bounds")?;
            if coordinates.len() < 2 || coordinates.len() % 2 != 0 {
                self.error_previous("box requires one lower/upper coordinate pair per axis");
                return None;
            }
            DomainSyntax::CartesianBox(
                coordinates
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|pair| (pair[0].clone(), pair[1].clone()))
                    .collect(),
            )
        } else if self.at_keyword("boundary") {
            self.bump();
            self.expect(TokenKind::LeftParen, "`(` after `boundary`")?;
            let parent = self
                .expect_identifier("parent Domain name")?
                .text()
                .to_owned();
            self.expect(TokenKind::Comma, "`,` before boundary axis")?;
            self.expect_keyword("axis")?;
            self.expect(TokenKind::Equal, "`=` after boundary axis")?;
            let axis_u64 = self.parse_u64("zero-based boundary axis")?;
            let axis = usize::try_from(axis_u64).ok().or_else(|| {
                self.error_previous("boundary axis exceeds this platform's usize range");
                None
            })?;
            self.expect(TokenKind::Comma, "`,` before boundary side")?;
            self.expect_keyword("side")?;
            self.expect(TokenKind::Equal, "`=` after boundary side")?;
            let side = if self.at_keyword("lower") {
                self.bump();
                BoundarySideSyntax::Lower
            } else if self.at_keyword("upper") {
                self.bump();
                BoundarySideSyntax::Upper
            } else {
                self.error_here("expected `lower` or `upper` boundary side");
                return None;
            };
            self.expect(TokenKind::RightParen, "`)` after boundary selector")?;
            DomainSyntax::Boundary { parent, axis, side }
        } else if self.at_keyword("scalar_physical") {
            let (across_name, across_type, through_name, through_type) =
                self.parse_scalar_physical_types()?;
            DomainSyntax::ScalarPhysical {
                across_name,
                across_type,
                through_name,
                through_type,
            }
        } else {
            self.error_here(
                "expected `box(...)`, `boundary(...)`, or `scalar_physical(...)` Domain contract",
            );
            return None;
        };
        let end = self
            .expect(TokenKind::Semicolon, "`;` after Domain")?
            .range()
            .end();
        Some(DomainDecl {
            comments: Default::default(),
            name,
            syntax,
            range: TextRange::new(start, end),
        })
    }

    fn parse_cartesian_coordinate(&mut self) -> Option<CartesianCoordinateSyntax> {
        if self.at(TokenKind::Identifier) {
            let token = self.bump();
            return Some(CartesianCoordinateSyntax::Parameter {
                name: token.text().to_owned(),
                range: token.range(),
            });
        }
        let start = self.current().range().start();
        let value = self.parse_signed_number()?;
        Some(CartesianCoordinateSyntax::Fixed {
            value,
            range: TextRange::new(start, self.previous_significant_range().end()),
        })
    }
}
