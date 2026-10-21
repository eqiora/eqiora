//! Held-fixed inputs keep their complete mathematical coordinates in local scopes.
use super::*;
use eqiora_core::{ScalarDomain, ValueLiteral, ValueType};
use eqiora_schema::kernel::UnaryMathFunction;
use eqiora_schema::kernel::pure_operator::{CalculusBuilder, ComponentIndex, PureValueClass};

#[test]
fn real_partials_hold_typed_inputs_fixed_through_nested_scopes() {
    let real = ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS).unwrap();
    let x = ValueLiteral::from_real(real.clone(), 3.).unwrap();
    for (shaped, complex) in [(false, false), (false, true), (true, false), (true, true)] {
        let scalar = ValueType::scalar(
            if complex {
                ScalarDomain::Complex
            } else {
                ScalarDomain::Real
            },
            DimExponents::DIMENSIONLESS,
        )
        .unwrap();
        let ty = if shaped {
            scalar.array(2).unwrap()
        } else {
            scalar
        };
        let points = [
            (2., if complex { 5. } else { 0. }),
            (7., if complex { -4. } else { 0. }),
        ];
        let count = if shaped { 2 } else { 1 };
        let z = ValueLiteral::new(ty.clone(), points[..count].iter().copied()).unwrap();
        for order in [1, 2] {
            let class = PureValueClass::from_value_type(ty.clone()).unwrap();
            let mut builder = CalculusBuilder::new(
                [
                    PureValueClass::from_value_type(real.clone()).unwrap(),
                    class.clone(),
                ],
                class,
            )
            .unwrap();
            let independent = builder
                .push(CalculusNode::FormalComponent {
                    formal: 0,
                    axes: Box::new([]),
                })
                .unwrap();
            let held = builder
                .push(CalculusNode::FormalComponent {
                    formal: 1,
                    axes: if shaped {
                        vec![ComponentIndex::Result(0)].into()
                    } else {
                        Box::new([])
                    },
                })
                .unwrap();
            let conjugate = builder
                .push(CalculusNode::UnaryMath(UnaryMathFunction::Conj, held))
                .unwrap();
            let square = builder
                .push(CalculusNode::Mul(independent, independent))
                .unwrap();
            let mut root = builder.push(CalculusNode::Mul(square, conjugate)).unwrap();
            if shaped || complex {
                // A held typed input does not grant a scalar partial with respect
                // to that array or complex formal. Such products need their own
                // coordinate/action contract; failure must leave this builder usable.
                assert!(builder.partial(root, 1).is_err());
            }
            for _ in 0..order {
                root = builder.partial(root, 0).unwrap();
            }
            let definition = builder.finish(root).unwrap();
            let actual =
                ComponentScalarization::evaluate_pure_operator(&definition, &[&x, &z]).unwrap();
            // f(x,z)=x²*conj(z), holding all of z fixed: f_x=2x*conj(z),
            // f_xx=2*conj(z). At x=3 these exact binary64 values need no tolerance.
            let expected = if order == 1 {
                [
                    (12., if complex { -30. } else { 0. }),
                    (42., if complex { 24. } else { 0. }),
                ]
            } else {
                [
                    (4., if complex { -10. } else { 0. }),
                    (14., if complex { 8. } else { 0. }),
                ]
            };
            assert_eq!(actual.value_type(), &ty);
            assert_eq!(
                actual.components().unwrap().collect::<Vec<_>>(),
                expected[..count]
            );
        }
    }
}
