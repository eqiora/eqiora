//! One admitted real unary profile and one action shared by JVP and VJP.
use super::*;
use eqiora_core::{DimExponents, DynQuantity, ValueLiteral};
use eqiora_schema::kernel::UnaryMathFunction;

pub(super) fn admit(function: UnaryMathFunction) -> Result<(), Diagnostic> {
    match function {
        UnaryMathFunction::Sin
        | UnaryMathFunction::Cos
        | UnaryMathFunction::Exp
        | UnaryMathFunction::Log
        | UnaryMathFunction::Sqrt => Ok(()),
        _ => Err(ir_builder_error(
            "unadmitted real scalar mathematical function",
        )),
    }
}

pub(super) fn value(function: UnaryMathFunction, value: f64) -> Result<f64, Diagnostic> {
    let value = ValueLiteral::try_from(DynQuantity::new(value, DimExponents::DIMENSIONLESS))
        .map_err(|error| ir_builder_error(error.to_string()))?;
    // Units were checked before numerical lowering. Domain and finite-value
    // decisions still belong to the canonical typed mathematical operation.
    Ok(function
        .evaluate(&value)?
        .real_scalar_value()
        .expect("real scalar function")
        .value())
}

pub(super) fn action(
    function: UnaryMathFunction,
    value: f64,
    direction: f64,
) -> Result<f64, Diagnostic> {
    use UnaryMathFunction::*;
    Ok(match function {
        Sin => direction * value.cos(),
        Cos => -direction * value.sin(),
        Exp => direction * value.exp(),
        Log if value > 0.0 => direction / value,
        Sqrt if value > 0.0 => direction / (2.0 * value.sqrt()),
        Sqrt => {
            return Err(ir_builder_error(
                "square-root derivative is undefined at zero",
            ));
        }
        _ => {
            return Err(ir_builder_error(
                "unadmitted real scalar mathematical derivative",
            ));
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ComponentScalarization;
    use eqiora_core::{Id, ScalarDomain, ValueType};
    use eqiora_schema::kernel::{
        ExprDagBuilder,
        typing::{ExpressionType, RootContract, TypedResidual},
    };

    fn programs(function: UnaryMathFunction) -> (ScalarOperatorIr, ComponentScalarization) {
        let mut builder = ExprDagBuilder::new();
        let input = builder.symbol(SymbolRef::Parameter(Id::new())).unwrap();
        let output = builder.unary_math(function, input).unwrap();
        let typed = TypedResidual::<()>::infer(
            builder.finish([output]).unwrap(),
            None,
            |_| None,
            RootContract::InitialResiduals,
            |_| {
                Ok::<_, ()>(ExpressionType::new(
                    ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS).unwrap(),
                    None,
                ))
            },
        )
        .unwrap();
        (
            ScalarOperatorIr::lower_typed_scalar(&typed).unwrap(),
            ComponentScalarization::lower(&typed).unwrap(),
        )
    }

    #[test]
    fn elementary_values_and_actions_agree_across_scalar_and_component_owners() {
        use UnaryMathFunction::*;
        for (function, input, expected, derivative) in [
            (Sin, 0.0, 0.0, 1.0),
            (Cos, std::f64::consts::FRAC_PI_2, 0.0, -1.0),
            (Exp, 0.0, 1.0, 1.0),
            (Log, 1.0, 0.0, 1.0),
            (Sqrt, 4.0, 2.0, 0.25),
        ] {
            let (scalar, component) = programs(function);
            let close = |actual: f64, reference: f64| {
                assert!(
                    (actual - reference).abs() <= 4.0 * f64::EPSILON,
                    "{function:?}: {actual} != {reference}"
                )
            };
            close(scalar.evaluate(&[input]).unwrap()[0], expected);
            let scalar = scalar
                .linearize(&[input], &[DifferentiationRole::Parameter])
                .unwrap();
            let component = component
                .linearize(|_| Some((input, DifferentiationRole::Parameter)))
                .unwrap();
            for action in [&scalar as &dyn LinearizedRelation<f64>, &component] {
                let mut value = [0.0];
                action.primal(&mut value).unwrap();
                close(value[0], expected);
                action
                    .jvp(RelationTangent::Parameter(&[2.0]), &mut value)
                    .unwrap();
                close(value[0], 2.0 * derivative);
                action
                    .vjp(&[3.0], RelationCotangent::Parameter(&mut value))
                    .unwrap();
                close(value[0], 3.0 * derivative);
            }
        }
    }

    #[test]
    fn logarithm_actions_keep_representable_ratios_and_domains_reject() {
        let (logarithm, component) = programs(UnaryMathFunction::Log);
        let tiny = f64::from_bits(1);
        for input in [tiny, 1.0, f64::MAX] {
            let scalar = logarithm
                .linearize(&[input], &[DifferentiationRole::Parameter])
                .unwrap();
            let component = component
                .linearize(|_| Some((input, DifferentiationRole::Parameter)))
                .unwrap();
            for action in [&scalar as &dyn LinearizedRelation<f64>, &component] {
                let mut value = [0.0];
                action
                    .jvp(RelationTangent::Parameter(&[input]), &mut value)
                    .unwrap();
                assert_eq!(value, [1.0]);
                action
                    .vjp(&[input], RelationCotangent::Parameter(&mut value))
                    .unwrap();
                assert_eq!(value, [1.0]);
            }
        }
        for input in [-1.0, 0.0] {
            assert!(logarithm.evaluate(&[input]).is_err());
        }
        let (root, _) = programs(UnaryMathFunction::Sqrt);
        assert!(root.evaluate(&[-1.0]).is_err());
        assert!(
            root.linearize(&[0.0], &[DifferentiationRole::Parameter])
                .unwrap()
                .jvp(RelationTangent::Parameter(&[0.0]), &mut [0.0])
                .is_err()
        );
        assert!(logarithm.require_regular_density(|_| true).is_err());
        logarithm.require_regular_density(|_| false).unwrap();
        programs(UnaryMathFunction::Cos)
            .0
            .require_regular_density(|_| true)
            .unwrap();
    }
}
