//! Authored forms use the same oriented typing rules as Model expressions.
use super::*;
use eqiora_schema::kernel::typing::{ExpressionType, SpatialSupport};

impl ExpressionContext<'_> {
    fn physical_support(&self, domain: Id<kinds::Domain>) -> SpatialSupport<RawId> {
        let domain = domain.erase();
        match self.index.boundary_of.get(&domain) {
            Some(parent) => SpatialSupport::Boundary {
                domain,
                parent: *parent,
                dimensions: self.ambient_dimension,
            },
            None => SpatialSupport::Volume {
                domain,
                dimensions: self.ambient_dimension,
            },
        }
    }

    fn oriented_type(&self, value: &AuthoredFormExpression) -> ExpressionType<RawId> {
        ExpressionType::new(
            value.value_type.clone(),
            value.support.map(|domain| self.physical_support(domain)),
        )
    }

    pub(super) fn compile_oriented(
        &mut self,
        expression: &Expr,
        name: &str,
        argument: &Expr,
    ) -> Result<AuthoredFormExpression, Diagnostic> {
        use crate::math::oriented::Operation;
        let operation = Operation::named(name).expect("closed oriented vocabulary");
        let argument = self.compile(argument)?;
        let context = self
            .integration_domain
            .map(|domain| self.physical_support(domain));
        let result = operation
            .result_type(&self.oriented_type(&argument), context.as_ref())
            .map_err(|message| error(self.file, expression.range(), message))?;
        let support = result
            .support
            .as_ref()
            .and_then(|value| value.domain().downcast());
        let kind = match operation {
            Operation::Curl => AuthoredFormExpressionKind::Curl(Box::new(argument)),
            Operation::TangentialTrace => {
                AuthoredFormExpressionKind::TangentialTrace(Box::new(argument))
            }
        };
        Ok(typed(kind, result.value_type, support))
    }

    pub(super) fn compile_cross(
        &mut self,
        expression: &Expr,
        left: &Expr,
        right: &Expr,
    ) -> Result<AuthoredFormExpression, Diagnostic> {
        let left = self.compile(left)?;
        let right = self.compile(right)?;
        let arguments = [self.oriented_type(&left), self.oriented_type(&right)];
        let definition = crate::math::tensor::Operation::Cross
            .definition(&arguments)
            .map_err(|e| error(self.file, expression.range(), e.to_string()))?;
        let result = definition
            .instantiate(&arguments)
            .map_err(|e| error(self.file, expression.range(), e.to_string()))?;
        let result = result.result_type();
        let support = result
            .support
            .as_ref()
            .and_then(|value| value.domain().downcast());
        Ok(typed(
            AuthoredFormExpressionKind::Cross(Box::new(left), Box::new(right)),
            result.value_type.clone(),
            support,
        ))
    }
}
