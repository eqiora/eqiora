use super::*;

impl DefinitionScope<'_, '_> {
    pub(in crate::hierarchy::body_check) fn bind_coordinate(
        &mut self,
        declaration: &eqiora_lang::NamedDefinitionDecl,
    ) -> Result<(), Diagnostic> {
        let invalid = |message: String| {
            source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.file,
                declaration.range(),
                message,
            )
        };
        let (factor, axis) = crate::hierarchy::coordinates::selector(self.file, declaration)?;
        let support = declaration
            .domain()
            .and_then(|name| self.spatial_support(name))
            .ok_or_else(|| invalid("coordinate requires an exact declared support".into()))?;
        let factor = self
            .spatial_support(factor)
            .ok_or_else(|| invalid("coordinate factor is not a declared support".into()))?;
        if matches!(declaration.value().kind(), eqiora_lang::ExprKind::Name(_))
            && factor.intrinsic_dimensions() != 1
        {
            return Err(invalid(
                "coordinate from a multi-axis factor requires an explicit axis".into(),
            ));
        }
        let inferred =
            eqiora_schema::kernel::typing::coordinate(factor.domain(), axis, Some(&support))
                .map_err(|error| invalid(error.to_string()))?;
        let declared = crate::value_types::lower_scalar_type(
            self.file,
            declaration
                .value_type()
                .expect("validated coordinate annotation"),
        )?;
        if declared != inferred.value_type {
            return Err(invalid(
                "coordinate annotation differs from the exact factor axis type".into(),
            ));
        }
        self.symbols.insert(
            declaration.name().to_owned(),
            SymbolContract::Coordinate(inferred),
        );
        Ok(())
    }
}
