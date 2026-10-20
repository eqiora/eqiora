//! Closed pure applications use the same component lowering as live expressions.
use super::*;
use eqiora_core::{ScalarDomain, ValueLiteral};
use eqiora_schema::kernel::{ExprDagBuilder, typing::RootContract};

impl ComponentScalarization {
    pub(crate) fn lower_scalar_definition(
        definition: &PureOperatorDefinition,
        typed: &TypedResidual<()>,
        arguments: &[ExprId],
        expected: &ExpressionType<()>,
        work: &mut usize,
    ) -> Result<(ScalarInputIrBuilder, ScalarInputValueId), Diagnostic> {
        let mut lowering = ComponentDagLowering::new(
            typed.expression(),
            typed.node_types(),
            work,
            SpatialContext::Pointwise,
        );
        let mut inputs = Vec::with_capacity(arguments.len());
        for argument in arguments {
            let ty = typed
                .node_type(*argument)
                .ok_or_else(|| invalid_component_ir("pure argument has no type"))?;
            if ty.value_type.scalar_domain() != ScalarDomain::Real || !ty.shape().is_scalar() {
                return Err(invalid_component_ir(
                    "numeric scalar projection requires real scalar arguments",
                ));
            }
            inputs.push(lowering.builder.scoped_input(ty.dimension())?);
        }
        let root = lowering.lower_pure_definition_with_inputs(
            definition,
            arguments,
            &[],
            expected,
            ScalarPart::Real,
            Some(inputs),
        )?;
        Ok((lowering.builder, root))
    }

    /// Evaluate a closed typed pure application through ordinary component IR.
    /// Scalar, tensor, real and complex values retain the same definition and
    /// ordered formal-calculus lowering, including local derivative identities.
    ///
    /// # Errors
    /// Rejects invalid argument types, unsupported component operations, existing
    /// lowering resource bounds, and demanded domain or representability failures.
    pub fn evaluate_pure_operator(
        definition: &PureOperatorDefinition,
        arguments: &[&ValueLiteral],
    ) -> Result<ValueLiteral, Diagnostic> {
        let types = arguments
            .iter()
            .map(|value| ExpressionType::<()>::new(value.value_type().clone(), None))
            .collect::<Vec<_>>();
        let instance = definition
            .instantiate(&types)
            .map_err(|error| invalid_component_ir(error.to_string()))?;
        let result_type = instance.result_type().value_type.clone();
        let complex = result_type.scalar_domain() == ScalarDomain::Complex;
        let mut builder = ExprDagBuilder::new();
        let arguments = arguments
            .iter()
            .map(|value| builder.constant((*value).clone()))
            .collect::<Result<Vec<_>, _>>()?;
        let root = builder.pure_operator(definition, arguments)?;
        let typed = TypedResidual::<()>::infer(
            builder.finish([root])?,
            None,
            |_| None,
            RootContract::ComponentwiseResidual,
            |_| -> Result<ExpressionType<()>, Diagnostic> {
                Err(invalid_component_ir(
                    "closed pure application contains a symbol",
                ))
            },
        )
        .map_err(|error| {
            invalid_component_ir(format!("pure application typing failed: {error:?}"))
        })?;
        let values = Self::lower(&typed)?.evaluate(|_| None)?;
        let components = values
            .chunks_exact(if complex { 2 } else { 1 })
            .map(|value| (value[0], if complex { value[1] } else { 0. }));
        ValueLiteral::new(result_type, components).map_err(|error| {
            invalid_component_ir(format!("pure application result is invalid: {error}"))
        })
    }
}
