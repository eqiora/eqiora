//! Closed finite Parameter algebra uses the same typed component IR as execution.
use super::*;
use eqiora_schema::kernel::{
    ExprDagBuilder,
    typing::{ExpressionType, RootContract, TypedResidual},
};

pub(super) fn evaluate(
    file: &str,
    expression: &Expr,
    context: ExpressionContext<'_>,
    resolve: &mut impl FnMut(&str, TextRange) -> Result<SymbolicParameterValue, Diagnostic>,
    (resolve_clock, resolve_frame): StaticContexts<'_>,
    (name, arguments): (&str, &eqiora_lang::CallArguments),
    evaluate_values: bool,
) -> Result<EvaluatedParameter, Diagnostic> {
    let error = |message: String| {
        source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            expression.range(),
            message,
        )
    };
    let arguments = arguments
        .positional()
        .ok_or_else(|| error("finite operation requires positional arguments".into()))?;
    let (operation, arguments) =
        crate::math::finite::Operation::source(name, arguments).map_err(|e| error(e.into()))?;
    let operands = arguments
        .iter()
        .map(|argument| {
            evaluate_mode(
                file,
                argument,
                context,
                resolve,
                (&mut *resolve_clock, &mut *resolve_frame),
                None,
                evaluate_values,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let types = operands
        .iter()
        .map(|operand| ExpressionType::<()>::new(operand.value_type.value_type().clone(), None))
        .collect::<Vec<_>>();
    let ty = operation
        .result_type(&types)
        .map_err(|e| error(e.to_string()))?
        .value_type;
    crate::typed_values::check_type(&ty).map_err(error)?;
    let value = if evaluate_values {
        operands
            .iter()
            .map(|o| o.value.as_ref())
            .collect::<Option<Vec<_>>>()
            .map(|values| {
                let mut builder = ExprDagBuilder::new();
                let inputs = values
                    .into_iter()
                    .map(|v| builder.constant(v.clone()))
                    .collect::<Result<Vec<_>, _>>()?;
                let root = operation.emit(&mut builder, &inputs)?;
                let typed = TypedResidual::<()>::infer(
                    builder.finish([root])?,
                    None,
                    |_| None,
                    RootContract::ComponentwiseResidual,
                    |_| Err::<ExpressionType<()>, ()>(()),
                )
                .map_err(|e| error(format!("finite Parameter IR typing failed: {e:?}")))?;
                let components =
                    eqiora_ir::ComponentScalarization::lower(&typed)?.evaluate(|_| None)?;
                let complex = ty.scalar_domain() == ScalarDomain::Complex;
                ValueLiteral::new(
                    ty.clone(),
                    components
                        .chunks_exact(if complex { 2 } else { 1 })
                        .map(|c| (c[0], if complex { c[1] } else { 0. })),
                )
                .map_err(|e| error(e.to_string()))
            })
            .transpose()?
    } else {
        None
    };
    let lowered = operands
        .iter()
        .map(|o| o.expression.clone())
        .collect::<Option<Vec<_>>>()
        .map(|arguments| LoweringExpression::finite(operation, arguments, expression.range()));
    Ok(EvaluatedParameter {
        value,
        value_type: EvaluatedType::Known(ty),
        bare_literal: false,
        expression: lowered,
        lineage: Some(ParameterLineage::Derived),
    })
}
