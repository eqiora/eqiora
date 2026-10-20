//! Numerical primitive relocation keeps the same derivative program after binding.
use super::*;
use eqiora_core::{DimExponents, DynQuantity, Id, ScalarDomain, ValueType};

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
