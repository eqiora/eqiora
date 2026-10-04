use super::{
    AstConstructionError, Expr, NamedBindingDecl, NamedDefinitionDecl, ParameterDecl,
    SourceAstFactory, TextRange, checked_identifier, checked_range, validate_expression,
    validate_identifier,
};

pub(super) fn validate_named_binding(
    binding: &NamedBindingDecl,
) -> Result<(), AstConstructionError> {
    validate_identifier(binding.name(), "Parameter binding")?;
    checked_range(binding.range())?;
    validate_expression(binding.value())
}

impl SourceAstFactory {
    /// Construct one compilation-unit structural dimension alias.
    ///
    /// # Errors
    /// Returns an error for a malformed name, expression, or range.
    pub fn dimension_alias(
        visibility: crate::VisibilitySyntax,
        name: impl Into<String>,
        expression: Expr,
        range: TextRange,
    ) -> Result<NamedDefinitionDecl, AstConstructionError> {
        validate_expression(&expression)?;
        Ok(NamedDefinitionDecl::plain(
            checked_identifier(name, "dimension alias")?,
            expression,
            checked_range(range)?,
            visibility,
        ))
    }

    /// Construct a model-level typed Parameter declaration.
    ///
    /// # Errors
    /// Returns an error for a non-finite value or malformed source shape.
    pub fn parameter(
        name: impl Into<String>,
        value_type: crate::ValueTypeSyntax,
        value: Expr,
        range: TextRange,
    ) -> Result<ParameterDecl, AstConstructionError> {
        validate_expression(&value)?;
        Ok(ParameterDecl {
            comments: Default::default(),
            name: checked_identifier(name, "Parameter")?,
            value_type,
            value,
            range: checked_range(range)?,
        })
    }

    /// Construct a model-level typed Observable declaration.
    ///
    /// # Errors
    /// Returns an error for a non-finite value or malformed source shape.
    pub fn observable(
        name: impl Into<String>,
        value_type: crate::ValueTypeSyntax,
        value: Expr,
        range: TextRange,
    ) -> Result<crate::ObservableDecl, AstConstructionError> {
        validate_expression(&value)?;
        Ok(crate::ObservableDecl {
            comments: Default::default(),
            name: checked_identifier(name, "Observable")?,
            value_type,
            value,
            range: checked_range(range)?,
        })
    }

    /// Construct an exact coordinate projection on a declared support.
    /// The selector is a factor name, optionally indexed by a literal axis.
    /// # Errors
    /// Rejects malformed names, ranges, or selectors that are not coordinate projections.
    pub fn coordinate(
        name: impl Into<String>,
        value_type: crate::ValueTypeSyntax,
        domain: impl Into<String>,
        factor: Expr,
        range: TextRange,
    ) -> Result<NamedDefinitionDecl, AstConstructionError> {
        let declaration = Self::let_alias(
            name,
            Some(value_type),
            Some(domain.into()),
            None,
            factor,
            range,
        )?;
        validate_coordinate(&declaration)?;
        Ok(declaration)
    }

    /// Construct a reusable immutable local expression alias.
    ///
    /// # Errors
    /// Returns an error for malformed source expressions, names, or ranges.
    pub fn let_alias(
        name: impl Into<String>,
        value_type: Option<crate::ValueTypeSyntax>,
        domain: Option<String>,
        activation: Option<String>,
        value: Expr,
        range: TextRange,
    ) -> Result<NamedDefinitionDecl, AstConstructionError> {
        validate_expression(&value)?;
        Ok(NamedDefinitionDecl {
            activation_name_range: None,
            visibility: crate::VisibilitySyntax::Private,
            comments: Default::default(),
            name: checked_identifier(name, "let alias")?,
            value_type,
            domain: domain
                .map(|name| checked_identifier(name, "let support assertion"))
                .transpose()?,
            activation: activation
                .map(|name| checked_identifier(name, "let activation assertion"))
                .transpose()?,
            value,
            range: checked_range(range)?,
        })
    }
}

pub(super) fn validate_coordinate(value: &NamedDefinitionDecl) -> Result<(), AstConstructionError> {
    let factor = match value.value().kind() {
        crate::ExprKind::Name(name) => Some(name),
        crate::ExprKind::Index { value, index } => match (value.kind(), index.kind()) {
            (crate::ExprKind::Name(name), crate::ExprKind::Number(axis))
                if axis.to_i64().is_ok_and(|axis| axis >= 0) =>
            {
                Some(name)
            }
            _ => None,
        },
        _ => None,
    };
    let Some(factor) = factor else {
        return Err(AstConstructionError::new(
            "coordinate requires one factor name with an optional nonnegative literal axis",
        ));
    };
    validate_identifier(factor, "coordinate factor")?;
    if value.value_type().is_none() || value.domain().is_none() || value.activation().is_some() {
        return Err(AstConstructionError::new(
            "coordinate requires a dimension and support, without temporal activation",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::{ExprKind, SourceAstFactory, TextRange, format, parse};

    #[test]
    fn checked_factory_constructs_a_formattable_dimension_prefix() {
        let range = TextRange::new(0, 0);
        let expression = SourceAstFactory::expression(ExprKind::Name("m".to_owned()), range)
            .expect("dimension expression");
        let model = parse("model.eqi", "model M() { variable x: Length; }")
            .into_document()
            .expect("model source")
            .models()[0]
            .clone();
        let document = SourceAstFactory::document_with_dimensions(
            Vec::new(),
            vec![
                SourceAstFactory::dimension_alias(
                    crate::VisibilitySyntax::Private,
                    "Length",
                    expression,
                    range,
                )
                .unwrap(),
            ],
            Vec::new(),
            Vec::new(),
            Vec::new(),
            vec![model],
        )
        .expect("dimension-bearing document");

        assert_eq!(
            format(&document),
            "dimension Length = m;\n\nmodel M() {\n  variable x: Length;\n}\n"
        );
    }

    #[test]
    fn checked_factory_retains_optional_let_dimension_assertions() {
        let range = TextRange::new(0, 1);
        let value = SourceAstFactory::expression(
            ExprKind::Number(crate::DecimalLiteral::parse("1.0").expect("exact literal")),
            range,
        )
        .expect("value");
        let dimension =
            SourceAstFactory::expression(ExprKind::Name("m".to_owned()), range).expect("dimension");

        let inferred =
            SourceAstFactory::let_alias("inferred", None, None, None, value.clone(), range)
                .expect("inferred alias");
        let annotated = SourceAstFactory::let_alias(
            "annotated",
            Some(crate::ValueTypeSyntax::real(dimension)),
            None,
            None,
            value,
            range,
        )
        .expect("annotated alias");

        assert!(inferred.value_type().is_none());
        assert!(annotated.value_type().is_some());
    }
}
