//! Resolve each point selector as a coordinate before retaining its binding.
use super::*;

impl ExpressionChecker<'_, '_, '_> {
    pub(super) fn evaluate_point(
        &mut self,
        expression: &Expr,
        value: &Expr,
        at: &[(eqiora_lang::NamePath, Expr)],
        side: bool,
    ) -> Result<ExpressionType<String>, Diagnostic> {
        let value = self.check(value)?;
        let mut names = std::collections::BTreeSet::new();
        let mut points = Vec::new();
        for (coordinate, point) in at {
            let SymbolContract::Coordinate(selected) = self.scope.resolve_symbol(coordinate)?
            else {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.scope.file,
                    coordinate.range(),
                    "evaluation binding must name a declared coordinate",
                ));
            };
            if !names.insert(coordinate.as_str()) {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.scope.file,
                    coordinate.range(),
                    "evaluation repeats a coordinate binding",
                ));
            }
            points.push((selected, self.check(point)?));
        }
        crate::lower::point_result_type(&value, &points, side).map_err(|message| {
            source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.scope.file,
                expression.range(),
                message,
            )
        })
    }
}
