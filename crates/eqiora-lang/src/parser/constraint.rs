//! Mathematical inequalities and complementarity, distinct from Boolean expressions.
use super::Parser;
use crate::ast::{BinaryOp, ExprKind, RelationCondition, TextRange};
use crate::lexer::TokenKind;
use eqiora_schema::kernel::RelationConditionKind;

impl Parser<'_> {
    pub(super) fn parse_constraint_statement(&mut self) -> Option<RelationCondition> {
        let complementarity = self.at_keyword("complementarity");
        let start = self.bump().range().start();
        self.expect(TokenKind::LeftParen, "`(` before constraint operands")?;
        let left = self.parse_expression(0)?;
        let (kind, left, right) = if complementarity {
            self.expect(
                TokenKind::Comma,
                "`,` between explicit nonnegativity predicates",
            )?;
            let right = self.parse_expression(0)?;
            (RelationConditionKind::Complementarity, left, right)
        } else {
            match left.kind() {
                ExprKind::Binary {
                    op: BinaryOp::GreaterEqual,
                    left,
                    right,
                } => (
                    RelationConditionKind::Inequality,
                    left.as_ref().clone(),
                    right.as_ref().clone(),
                ),
                ExprKind::Binary {
                    op: BinaryOp::LessEqual,
                    left,
                    right,
                } => (
                    RelationConditionKind::Inequality,
                    right.as_ref().clone(),
                    left.as_ref().clone(),
                ),
                _ => {
                    self.error_here("inequality requires an explicit non-strict real ordering");
                    return None;
                }
            }
        };
        self.expect(TokenKind::RightParen, "`)` after constraint operands")?;
        let end = self
            .expect(TokenKind::Semicolon, "`;` after mathematical constraint")?
            .range()
            .end();
        Some(RelationCondition {
            kind,
            left,
            right,
            range: TextRange::new(start, end),
        })
    }
}
