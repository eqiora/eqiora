//! Retained pure applications use the shared scalar or component IR projection.
use super::*;
use eqiora_schema::kernel::{
    ExprDagBuilder,
    pure_operator::PureOperatorDefinition,
    typing::{ExpressionType, RootContract, TypedResidual},
};

pub(super) fn evaluate(
    owner: RawId,
    definition: &PureOperatorDefinition,
    arguments: &[&ValueLiteral],
    component_work: &mut usize,
) -> Result<ValueLiteral, Diagnostic> {
    let types = arguments
        .iter()
        .map(|value| ExpressionType::<()>::new(value.value_type().clone(), None))
        .collect::<Vec<_>>();
    let instance = definition
        .instantiate(&types)
        .map_err(|error| Diagnostic::error(codes::NOT_IMPLEMENTED, error.to_string()))?;
    let result_type = instance.result_type().value_type.clone();
    let complex = result_type.scalar_domain() == ScalarDomain::Complex;
    let count = result_type
        .shape()
        .component_count()
        .ok_or_else(component_budget_error)?;
    let cost = count
        .checked_mul(definition.nodes().len())
        .and_then(|cost| cost.checked_mul(if complex { 4 } else { 1 }))
        .and_then(|cost| cost.checked_add(arguments.len()))
        .ok_or_else(component_budget_error)?;
    check_component_work(*component_work, cost)?;
    *component_work += cost;
    let mut builder = ExprDagBuilder::new();
    let arguments = arguments
        .iter()
        .map(|value| builder.constant((*value).clone()))
        .collect::<Result<Vec<_>, _>>()?;
    if !complex && types.iter().all(|ty| ty.shape().is_scalar()) && result_type.shape().is_scalar()
    {
        let root = builder.project_scalar_operator(&instance, &arguments, 1_000_000)?;
        let dag = builder.finish([root])?;
        return evaluate_selected(owner, &dag, &[root], &mut |_| None)?
            .pop()
            .ok_or_else(|| {
                Diagnostic::error(
                    codes::INVALID_EXPRESSION_DAG,
                    "pure operator result is absent",
                )
            });
    }
    let root = builder.pure_operator(definition, arguments)?;
    let typed = TypedResidual::<()>::infer(
        builder.finish([root])?,
        None,
        RootContract::ComponentwiseResidual,
        |_| -> Result<ExpressionType<()>, Diagnostic> {
            Err(Diagnostic::error(
                codes::INVALID_EXPRESSION_DAG,
                "closed pure application contains a symbol",
            ))
        },
    )
    .map_err(|error| {
        Diagnostic::error(
            codes::INVALID_EXPRESSION_DAG,
            format!("pure application typing failed: {error:?}"),
        )
    })?;
    let values = eqiora_ir::ComponentScalarization::lower(&typed)?.evaluate(|_| None)?;
    let components = values
        .chunks_exact(if complex { 2 } else { 1 })
        .map(|value| (value[0], if complex { value[1] } else { 0.0 }));
    ValueLiteral::new(result_type, components).map_err(|error| {
        Diagnostic::error(
            codes::INVALID_EXPRESSION_DAG,
            format!("pure application result is invalid: {error}"),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::{DimExponents, Id, ValueFrame, ValueShape, entity::kinds};

    #[test]
    fn retained_tensor_application_evaluates_real_and_complex_values() {
        let owner: RawId = Id::<kinds::Field>::new().into();
        let mut builder = ExprDagBuilder::new();
        let matrix_type = ValueType::shaped(
            ScalarDomain::Real,
            DimExponents::DIMENSIONLESS,
            ValueShape::new([2, 2]).unwrap(),
            ValueFrame::SpatialCartesian,
        )
        .unwrap();
        let vector_type = ValueType::shaped(
            ScalarDomain::Complex,
            DimExponents::DIMENSIONLESS,
            ValueShape::new([2]).unwrap(),
            ValueFrame::SpatialCartesian,
        )
        .unwrap();
        let matrix = builder
            .constant(
                ValueLiteral::new(
                    matrix_type,
                    [(2.0, 0.0), (3.0, 0.0), (5.0, 0.0), (7.0, 0.0)],
                )
                .unwrap(),
            )
            .unwrap();
        let vector = builder
            .constant(ValueLiteral::new(vector_type, [(11.0, 1.0), (13.0, -2.0)]).unwrap())
            .unwrap();
        let product = builder
            .pure_operator(
                &PureOperatorDefinition::contract(2, 2, 1, &[(1, 0)]).unwrap(),
                [matrix, vector],
            )
            .unwrap();
        let component = builder
            .pure_operator(
                &PureOperatorDefinition::tensor_component(2, &[1]).unwrap(),
                [product],
            )
            .unwrap();
        let dag = builder.finish([product, component]).unwrap();
        let result = evaluate_selected(owner, &dag, &[product, component], &mut |_| None).unwrap();
        assert_eq!(result[0].component(0), Some((61.0, -4.0)));
        assert_eq!(result[0].component(1), Some((146.0, -9.0)));
        assert_eq!(result[1].component(0), Some((146.0, -9.0)));
    }
}
