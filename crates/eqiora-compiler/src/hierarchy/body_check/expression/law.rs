//! Definition-body typing of exact physical conservation terms.
use super::*;

impl ExpressionChecker<'_, '_, '_> {
    pub(super) fn check_law(
        &mut self,
        terms: &eqiora_lang::ConservationSyntax,
    ) -> Result<(), Diagnostic> {
        if !matches!(self.relation_support, Some(SpatialSupport::Volume { .. })) {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.scope.file,
                terms.flux().range(),
                "fixed-domain Law requires volume support",
            ));
        }
        let flux = self.check(terms.flux())?;
        let balance = typing::divergence(&flux)
            .map_err(|error| type_error(self.scope.file, terms.flux(), error))?;
        let source = self.check(terms.source())?;
        if !crate::lower::equality::is_contextual_zero(terms.source()) {
            typing::additive(&balance, &source)
                .map_err(|error| type_error(self.scope.file, terms.source(), error))?;
        }
        for (expression, value) in [(terms.source(), &source), (terms.flux(), &flux)] {
            if value.value_type.array_rank() != 0 {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.scope.file,
                    expression.range(),
                    "scalar conservation Law requires non-array physical terms",
                ));
            }
        }
        if !balance.shape().is_scalar() {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.scope.file,
                terms.flux().range(),
                "scalar conservation Law requires physical vector flux",
            ));
        }
        typing::residual(&balance, self.relation_support.as_ref())
            .map_err(|error| type_error(self.scope.file, terms.flux(), error))?;
        Ok(())
    }
}
