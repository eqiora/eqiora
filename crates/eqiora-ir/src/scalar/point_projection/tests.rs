//! Numerical primitive relocation keeps the same derivative program after binding.
use super::*;
use eqiora_core::{DimExponents, DynQuantity, Id, ScalarDomain, ValueType};

#[test]
fn unused_complex_values_do_not_change_exact_discrete_control_admission() {
    let unit = DimExponents::DIMENSIONLESS;
    let mut builder = ExprDagBuilder::new();
    builder
        .constant(
            ValueLiteral::new(
                ValueType::scalar(ScalarDomain::Complex, unit).unwrap(),
                [(1., 2.)],
            )
            .unwrap(),
        )
        .unwrap();
    let integer = ValueType::scalar(ScalarDomain::Integer, unit).unwrap();
    let a = builder
        .constant(ValueLiteral::integer(integer.clone(), [9_007_199_254_740_992]).unwrap())
        .unwrap();
    let b = builder
        .constant(ValueLiteral::integer(integer, [9_007_199_254_740_993]).unwrap())
        .unwrap();
    let condition = builder
        .compare(eqiora_schema::kernel::ComparisonOp::Less, a, b)
        .unwrap();
    let yes = builder.constant(DynQuantity::new(7., unit)).unwrap();
    let no = builder.constant(DynQuantity::new(9., unit)).unwrap();
    let root = builder.select(condition, yes, no).unwrap();
    let typed = TypedResidual::<()>::infer(
        builder.finish([root]).unwrap(),
        None,
        |_| None,
        RootContract::InitialResiduals,
        |_| -> Result<ExpressionType<()>, ()> { unreachable!() },
    )
    .unwrap();
    assert_eq!(
        ScalarOperatorIr::lower_typed_scalar(&typed)
            .unwrap()
            .evaluate(&[])
            .unwrap(),
        [7.]
    );
}

#[test]
fn real_scalar_projection_keeps_complex_intermediates_actions_and_lazy_domains() {
    use eqiora_schema::kernel::UnaryMathFunction;
    use eqiora_schema::kernel::typing::SpatialSupport;
    let unit = DimExponents::DIMENSIONLESS;
    let ty = ValueType::scalar(ScalarDomain::Real, unit).unwrap();
    let symbol = SymbolRef::Parameter(Id::new());
    let mut builder = ExprDagBuilder::new();
    let x = builder.symbol(symbol).unwrap();
    let one = builder.constant(DynQuantity::new(1., unit)).unwrap();
    let zero = builder.constant(DynQuantity::new(0., unit)).unwrap();
    let negative = builder.constant(DynQuantity::new(-1., unit)).unwrap();
    let invalid = builder
        .unary_math(UnaryMathFunction::Sqrt, negative)
        .unwrap();
    let denominator = builder.complex(one, x).unwrap();
    let square = builder.mul(x, x).unwrap();
    let value = builder.div(square, denominator).unwrap();
    let real = builder.unary_math(UnaryMathFunction::Real, value).unwrap();
    let imaginary = builder.unary_math(UnaryMathFunction::Imag, value).unwrap();
    let positive = builder
        .compare(eqiora_schema::kernel::ComparisonOp::Greater, x, zero)
        .unwrap();
    let real = builder.select(positive, real, invalid).unwrap();
    let typed = TypedResidual::infer(
        builder.finish([real, imaginary]).unwrap(),
        None::<SpatialSupport<()>>,
        |_| None,
        RootContract::InitialResiduals,
        |_| Ok::<_, ()>(ExpressionType::new(ty.clone(), None)),
    )
    .unwrap();
    let operator = ScalarOperatorIr::lower_typed_scalar(&typed).unwrap();
    assert_eq!(operator.symbols(), &[symbol]);
    let point = [ValueLiteral::from_real(ty, 2.).unwrap()];
    let values = operator
        .evaluate_typed(&[real, imaginary], &mut |_| Some(point[0].clone()))
        .unwrap();
    for (actual, expected) in values.iter().zip([0.8, -1.6]) {
        assert!((actual.real_scalar_value().unwrap().value() - expected).abs() < 1e-14);
    }
    assert!(
        operator
            .evaluate_typed(&[value], &mut |_| Some(point[0].clone()))
            .is_err()
    );
    // g=x²/(1+ix). At x=2, g'=.16-1.12i and g''=-.176+.032i.
    let linear = operator
        .linearize_typed(&point, &[DifferentiationRole::Unknown])
        .unwrap();
    let mut action = [0.; 2];
    linear
        .jvp(RelationTangent::Unknown(&[1.]), &mut action)
        .unwrap();
    for (actual, expected) in action.into_iter().zip([0.16, -1.12]) {
        assert!((actual - expected).abs() < 1e-13);
    }
    linear
        .mixed_jvp(
            RelationTangent::Unknown(&[1.]),
            RelationTangent::Unknown(&[1.]),
            RelationTangent::Unknown(&[0.]),
            &mut action,
        )
        .unwrap();
    for (actual, expected) in action.into_iter().zip([-0.176, 0.032]) {
        assert!((actual - expected).abs() < 1e-13);
    }
    // Initial tangent compatibility decomposes these same roots into opaque
    // additive terms. Their admitted input types and complex-division primitive
    // must survive that projection without a round trip through source syntax.
    use num_traits::ToPrimitive;
    let mut reconstructed = [0.; 2];
    for (atom, coefficients) in operator.additive_terms(&point, symbol).unwrap() {
        let linear = atom
            .linearize_typed(&point, &[DifferentiationRole::Unknown])
            .unwrap();
        let mut derivative = [0.];
        linear
            .jvp(RelationTangent::Unknown(&[1.]), &mut derivative)
            .unwrap();
        for (value, coefficient) in reconstructed.iter_mut().zip(coefficients) {
            *value += coefficient.to_f64().unwrap() * derivative[0];
        }
    }
    for (actual, expected) in reconstructed.into_iter().zip([0.16, -1.12]) {
        assert!((actual - expected).abs() < 1e-13);
    }
    assert!(operator.evaluate(&[-1.]).is_err());
}

#[test]
fn quotient_partial_survives_frozen_inputs_and_array_component_selection() {
    let dimension = DimExponents::DIMENSIONLESS;
    let ty = ValueType::scalar(ScalarDomain::Real, dimension).unwrap();
    let symbols = vec![
        SymbolRef::Parameter(Id::new()),
        SymbolRef::Parameter(Id::new()),
    ];
    for array in [false, true] {
        let mut builder = ScalarInputIrBuilder::new();
        let values = symbols
            .iter()
            .enumerate()
            .map(|(index, symbol)| {
                let coordinate = ScalarSymbolCoordinate::for_value(*symbol, &ty)
                    .unwrap()
                    .remove(0);
                builder
                    .input(ScalarInputSlot::new(index as u32, coordinate, dimension))
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let quotient = builder.div(values[0], values[1]).unwrap();
        let one = builder.constant(DynQuantity::new(1., dimension)).unwrap();
        let partial = builder
            .formal_directional(quotient, &[None, Some(one)], dimension, |_| Ok(()))
            .unwrap();
        let mut root = partial.0;
        let mut array_operands = Vec::new();
        if array {
            array_operands.extend([partial.0, one.0]);
            let values = builder
                .push(Instruction::Array { start: 0, len: 2 })
                .unwrap();
            root = builder.push(Instruction::Index(values.0, 0)).unwrap().0;
        }
        assert!(builder.primal_checks.is_empty());
        let operator = ScalarOperatorIr {
            typed_inputs: Some(vec![ty.clone(); 2]),
            source_values: Vec::new(),
            symbols: symbols.clone(),
            instructions: builder.instructions,
            typed_constants: builder.typed_constants,
            array_operands,
            definitions: Vec::new(),
            roots: vec![root],
        };
        for (coordinate, direction, expected_value, expected_action) in
            [(2., 1., -0.5, 0.5), (1e-200, 1e-200, -1e200, 2e200)]
        {
            let point = [coordinate, coordinate]
                .map(|value| ValueLiteral::try_from(DynQuantity::new(value, dimension)).unwrap());
            let (values, _) = operator
                .evaluate_value_trace(&operator.roots, &mut |symbol| {
                    symbols
                        .iter()
                        .position(|candidate| *candidate == symbol)
                        .map(|index| point[index].clone())
                })
                .unwrap();
            let value = values[0].real_scalar_value().unwrap().value();
            assert!((value / expected_value - 1.).abs() <= 16. * f64::EPSILON);
            let linear = operator
                .linearize_typed(
                    &point,
                    &[DifferentiationRole::Frozen, DifferentiationRole::Unknown],
                )
                .unwrap();
            // partial_v (u/v)=-u/v². With u fixed, its differential is 2u*dv/v³.
            // At tiny u=v, the coefficient alone overflows but these actions fit.
            let mut action = [0.];
            linear
                .jvp(RelationTangent::Unknown(&[direction]), &mut action)
                .unwrap();
            assert!((action[0] / expected_action - 1.).abs() <= 16. * f64::EPSILON);
            let mut adjoint = [0.];
            linear
                .vjp(&[direction], RelationCotangent::Unknown(&mut adjoint))
                .unwrap();
            assert!((adjoint[0] / expected_action - 1.).abs() <= 16. * f64::EPSILON);
        }
        let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
        let wrong_type = [
            ValueLiteral::try_from(DynQuantity::new(2., length)).unwrap(),
            ValueLiteral::try_from(DynQuantity::new(2., dimension)).unwrap(),
        ];
        assert!(
            operator
                .linearize_typed(
                    &wrong_type,
                    &[DifferentiationRole::Frozen, DifferentiationRole::Unknown]
                )
                .is_err()
        );
        let invalid = [2., 0.]
            .map(|value| ValueLiteral::try_from(DynQuantity::new(value, dimension)).unwrap());
        assert!(
            operator
                .linearize_typed(
                    &invalid,
                    &[DifferentiationRole::Frozen, DifferentiationRole::Unknown]
                )
                .is_err()
        );
    }
}
