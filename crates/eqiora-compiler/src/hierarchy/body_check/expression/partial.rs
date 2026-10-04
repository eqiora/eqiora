//! Exact independent binding admission for explicit partials.
use super::*;

impl ExpressionChecker<'_, '_, '_> {
    pub(super) fn jvp(
        &mut self,
        expression: &Expr,
        arguments: &eqiora_lang::CallArguments,
    ) -> Result<ExpressionType<String>, Diagnostic> {
        let file = self.scope.file;
        let (value, selectors, directions) =
            crate::pure_operator::actions::arguments(file, expression, arguments)?;
        let output = self.check(value)?;
        let mut seen = std::collections::BTreeSet::new();
        for (selector, direction) in selectors.iter().zip(directions) {
            let binding = crate::pure_operator::actions::binding(file, selector)?;
            let selected = self.partial_binding(&binding)?;
            crate::lower::partial_result_type(&output, &selected).map_err(|message| {
                source_error(codes::LANGUAGE_TYPE_ERROR, file, selector.range(), message)
            })?;
            if !seen.insert(binding.as_str().to_owned()) || self.check(direction)? != selected {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    direction.range(),
                    "jvp requires distinct independent inputs and directions with their exact types",
                ));
            }
        }
        Ok(output)
    }

    pub(super) fn vjp(
        &mut self,
        expression: &Expr,
        arguments: &eqiora_lang::CallArguments,
    ) -> Result<ExpressionType<String>, Diagnostic> {
        let file = self.scope.file;
        let (value, selector, cotangent) =
            crate::pure_operator::actions::vjp_arguments(file, expression, arguments)?;
        let output = self.check(value)?;
        let binding = crate::pure_operator::actions::binding(file, selector)?;
        let selected = self.partial_binding(&binding)?;
        crate::lower::partial_result_type(&output, &selected).map_err(|message| {
            source_error(codes::LANGUAGE_TYPE_ERROR, file, selector.range(), message)
        })?;
        let dual = output.dimension().pow(-1, 1).ok_or_else(|| {
            source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                expression.range(),
                "vjp dual dimension overflows",
            )
        })?;
        if self.check(cotangent)? != ExpressionType::scalar(dual, output.support.clone()) {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                cotangent.range(),
                "vjp cotangent must have the exact real dual output type",
            ));
        }
        let expanded = crate::pure_operator::actions::expand_vjp(file, expression, arguments)?;
        self.check(&expanded)
    }

    pub(super) fn partial_binding(
        &self,
        binding: &eqiora_lang::NamePath,
    ) -> Result<ExpressionType<String>, Diagnostic> {
        match self.scope.resolve_symbol(binding)? {
            SymbolContract::Coordinate(ty)
                if matches!(
                    ty.support,
                    Some(eqiora_schema::kernel::typing::SpatialSupport::Boundary { .. })
                ) =>
            {
                Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.scope.file,
                    binding.range(),
                    "coordinate partials on boundaries require an admitted intrinsic chart",
                ))
            }
            SymbolContract::Coordinate(ty)
            | SymbolContract::Parameter(ty)
            | SymbolContract::Field(
                ty,
                eqiora_lang::FieldRoleSyntax::State,
                ActivationSyntax::Continuous,
            ) => Ok(ty),
            _ => Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.scope.file,
                binding.range(),
                "partial binding must name a declared coordinate, independent Parameter or continuous state Field; aliases and algebraic solutions are not independent",
            )),
        }
    }
}
