use super::*;

impl ExpressionContext<'_> {
    pub(super) fn compile_contraction(
        &mut self,
        expression: &Expr,
        name: &str,
        left: &Expr,
        right: &Expr,
    ) -> Result<AuthoredFormExpression, Diagnostic> {
        let left = self.compile(left)?;
        let right = self.compile(right)?;
        let invalid = || {
            error(
                self.file,
                expression.range(),
                "contraction requires identical component roles, bases and shapes",
            )
        };
        let normalized = |value: &ValueType, other: &ValueType| {
            value
                .clone()
                .with_common_scalar_domain(other)
                .ok_or_else(invalid)?
                .with_dimension(DimExponents::DIMENSIONLESS)
                .map_err(|_| invalid())
        };
        if normalized(&left.value_type, &right.value_type)?
            != normalized(&right.value_type, &left.value_type)?
            || (name == "dot" && left.value_type.shape().rank() != 1)
            || (name == "frobenius" && left.value_type.shape().rank() != 2)
        {
            return Err(invalid());
        }
        let dimension = left
            .value_type
            .dimension()
            .mul(right.value_type.dimension())
            .ok_or_else(|| {
                error(
                    self.file,
                    expression.range(),
                    "contraction dimension overflows",
                )
            })?;
        let domain = left
            .value_type
            .scalar_domain()
            .common(right.value_type.scalar_domain())
            .ok_or_else(invalid)?;
        let value_type = ValueType::scalar(domain, dimension).map_err(|_| invalid())?;
        let support = merge_support(self.file, expression.range(), left.support, right.support)?;
        let (left, right) = (Box::new(left), Box::new(right));
        let kind = match name {
            "inner" => AuthoredFormExpressionKind::Inner(left, right),
            "dot" => AuthoredFormExpressionKind::Dot(left, right),
            "frobenius" => AuthoredFormExpressionKind::Frobenius(left, right),
            _ => unreachable!("caller selects the contraction vocabulary"),
        };
        Ok(typed(kind, value_type, support))
    }
}
