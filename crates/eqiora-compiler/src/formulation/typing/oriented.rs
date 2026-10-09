//! Authored forms use the same oriented typing rules as Model expressions.
use super::*;
use eqiora_schema::kernel::typing::{ExpressionType, SpatialSupport};

impl ExpressionContext<'_> {
    pub(in crate::formulation) fn physical_support(
        &self,
        domain: Id<kinds::Domain>,
    ) -> SpatialSupport<RawId> {
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

    pub(super) fn compile_boundary(
        &mut self,
        expression: &Expr,
        name: &str,
        arguments: &eqiora_lang::CallArguments,
    ) -> Result<AuthoredFormExpression, Diagnostic> {
        use crate::math::boundary::Operation;
        let selected = crate::math::boundary::source(arguments)
            .map_err(|message| error(self.file, expression.range(), message))?;
        let resolve = |name: &str| {
            let raw = resolve_symbol(self.file, expression.range(), name, self.symbols)?;
            raw.downcast::<kinds::Domain>()
                .filter(|_| matches!(self.index.nodes.get(&raw), Some(KernelNode::Domain(_))))
                .ok_or_else(|| {
                    error(
                        self.file,
                        expression.range(),
                        "boundary selector must name an exact Domain",
                    )
                })
        };
        let target = selected
            .on
            .map(resolve)
            .transpose()?
            .or(self.integration_domain)
            .filter(|domain| {
                self.index.boundary_of.get(&domain.erase()).copied()
                    == self.relation_domain.map(Id::erase)
            })
            .ok_or_else(|| {
                error(
                    self.file,
                    expression.range(),
                    "trace requires an exact boundary of the Relation Domain",
                )
            })?;
        let from = selected
            .from
            .map(resolve)
            .transpose()?
            .map(|domain| self.physical_support(domain));
        let target_type = self.physical_support(target);
        let argument = self.compile(selected.value)?;
        let operation = Operation::named(name).expect("boundary vocabulary");
        let result = operation
            .result_type(
                &self.oriented_type(&argument),
                Some(&target_type),
                from.as_ref(),
            )
            .map_err(|message| error(self.file, expression.range(), message))?;
        let kind = match operation {
            Operation::Trace => AuthoredFormExpressionKind::Trace(Box::new(argument)),
            Operation::Tangential => {
                AuthoredFormExpressionKind::TangentialTrace(Box::new(argument))
            }
            Operation::Normal => AuthoredFormExpressionKind::NormalTrace(Box::new(argument)),
        };
        Ok(typed(kind, result.value_type, Some(target)))
    }

    pub(super) fn compile_curl(
        &mut self,
        expression: &Expr,
        argument: &Expr,
    ) -> Result<AuthoredFormExpression, Diagnostic> {
        use crate::math::oriented::Operation;
        let operation = Operation::Curl;
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
        let kind = AuthoredFormExpressionKind::Curl(Box::new(argument));
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
