//! Resolve each point selector as a coordinate before retaining its binding.
use super::*;

impl ExpressionChecker<'_, '_, '_> {
    pub(super) fn coordinate_map_factor(
        &mut self,
        expression: &Expr,
        factor: eqiora_schema::kernel::CoordinateMapFactor,
        source: &[eqiora_lang::NamePath],
        at: &[(eqiora_lang::NamePath, Expr)],
    ) -> Result<ExpressionType<String>, Diagnostic> {
        let coordinate = |name: &eqiora_lang::NamePath| {
            let SymbolContract::Coordinate(selected) = self.scope.resolve_symbol(name)? else {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.scope.file,
                    name.range(),
                    "pullback selector must name a declared coordinate",
                ));
            };
            Ok(selected)
        };
        let source = source
            .iter()
            .map(coordinate)
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        let selectors = at
            .iter()
            .map(|(name, _)| coordinate(name))
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        let points = selectors
            .into_iter()
            .zip(at)
            .map(|(selected, (_, mapped))| Ok((selected, self.check(mapped)?)))
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        eqiora_schema::kernel::typing::coordinate_map_factor(factor, &source, &points).map_err(
            |error| {
                source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.scope.file,
                    expression.range(),
                    error.to_string(),
                )
            },
        )
    }
    pub(super) fn pullback(
        &mut self,
        expression: &Expr,
        value: &Expr,
        source: &[eqiora_lang::NamePath],
        at: &[(eqiora_lang::NamePath, Expr)],
    ) -> Result<ExpressionType<String>, Diagnostic> {
        let coordinate = |name: &eqiora_lang::NamePath| {
            let SymbolContract::Coordinate(selected) = self.scope.resolve_symbol(name)? else {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.scope.file,
                    name.range(),
                    "pullback selector must name a declared coordinate",
                ));
            };
            Ok(selected)
        };
        let source = source
            .iter()
            .map(coordinate)
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        let selectors = at
            .iter()
            .map(|(name, _)| coordinate(name))
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        let points = selectors
            .into_iter()
            .zip(at)
            .map(|(selected, (_, mapped))| Ok((selected, self.check(mapped)?)))
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        let value = self.check(value)?;
        eqiora_schema::kernel::typing::coordinate_pullback(&value, &source, &points).map_err(
            |error| {
                source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.scope.file,
                    expression.range(),
                    error.to_string(),
                )
            },
        )
    }
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
