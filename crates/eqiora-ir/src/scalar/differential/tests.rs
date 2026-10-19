use super::*;
use eqiora_core::{DimExponents, DynQuantity, Id, ScalarDomain, ValueType};

fn reject_finite_map_work(_: usize) -> Result<(), Diagnostic> {
    Err(ir_builder_error(
        "quotient tests do not contain finite maps",
    ))
}

#[test]
fn emitted_quotient_retains_affine_and_polynomial_coefficient_bindings() {
    let dimension = DimExponents::DIMENSIONLESS;
    let ty = ValueType::scalar(ScalarDomain::Real, dimension).unwrap();
    for exponent in [-700, 0, 700] {
        let scale = 2f64.powi(exponent);
        let mut builder = ScalarInputIrBuilder::new();
        let coordinates = (0..3)
            .map(|_| {
                ScalarSymbolCoordinate::for_value(SymbolRef::Parameter(Id::new()), &ty)
                    .unwrap()
                    .remove(0)
            })
            .collect::<Vec<_>>();
        let inputs = coordinates
            .iter()
            .enumerate()
            .map(|(index, coordinate)| {
                builder
                    .input(ScalarInputSlot::new(
                        index as u32,
                        coordinate.clone(),
                        dimension,
                    ))
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let numerator = builder.mul(inputs[0], inputs[1]).unwrap();
        let root = builder.div(numerator, inputs[2]).unwrap();
        let step = builder
            .constant(DynQuantity::new(scale, dimension))
            .unwrap();
        let derivative = builder
            .directional(root, &[None, None, Some(step)], reject_finite_map_work)
            .unwrap()
            .unwrap();
        let ir = builder.finish([derivative]).unwrap();
        // D_p(u*lambda/p)[s] = -s*u*lambda/p². At p=s this is
        // -u*lambda/s; its next p-direction s gives 2*u*lambda/s.
        let (affine, tangent) = ir
            .bind_affine_direction(
                &coordinates[..1],
                &[
                    (coordinates[1].clone(), 3., None),
                    (coordinates[2].clone(), scale, Some(scale)),
                ],
            )
            .unwrap();
        assert_eq!(affine.coefficients(), [-3. / scale]);
        assert_eq!(tangent.coefficients(), [6. / scale]);
        assert_eq!(affine.offsets(), [0.]);
        assert_eq!(tangent.offsets(), [0.]);
        let pencil = ir
            .bind_polynomial_pencil(
                &coordinates[..1],
                &coordinates[1..2],
                1,
                1000,
                &[(coordinates[2].clone(), scale)],
            )
            .unwrap();
        assert_eq!(pencil.len(), 1);
        assert_eq!(pencil[&vec![1]].coefficients(), [-1. / scale]);
        assert_eq!(pencil[&vec![1]].offsets(), [0.]);
    }
}

#[test]
fn quotient_actions_preserve_unbalanced_scales_and_cancellation() {
    // For da=0, D(a/b)[0,db]=-a*db/b². These powers of two
    // keep the final answers exact, even when a/b or db/b cannot be
    // represented as an intermediate binary64 value.
    for (a, b, da, db, expected) in [
        (
            2f64.powi(-700),
            2f64.powi(500),
            0.,
            2f64.powi(700),
            -2f64.powi(-1000),
        ),
        (
            2f64.powi(-700),
            2f64.powi(-500),
            0.,
            2f64.powi(700),
            -2f64.powi(1000),
        ),
        // Scaling numerator and denominator equally leaves their ratio fixed.
        // The two terms da/b and a*db/b² overflow individually, but cancel.
        (
            2f64.powi(-500),
            2f64.powi(-500),
            2f64.powi(700),
            2f64.powi(700),
            0.,
        ),
    ] {
        let actual = quotient_direction(
            &mut Numerical,
            Scaled::new(a),
            Scaled::new(b),
            Some(Scaled::new(da)),
            Some(Scaled::new(db)),
        )
        .unwrap()
        .unwrap()
        .value();
        assert_eq!(actual, expected, "point ({a}, {b}), direction ({da}, {db})");
    }
}

#[test]
fn guarded_unitful_quotient_coefficients_keep_forward_reverse_and_mixed_actions() {
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let time = DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).unwrap();
    for exponent in [-700, 0, 700] {
        let scale = 2f64.powi(exponent);
        let mut builder = ScalarInputIrBuilder::new();
        let mut inputs = Vec::new();
        for (ordinal, dimension) in [length, time].into_iter().enumerate() {
            let ty = ValueType::scalar(ScalarDomain::Real, dimension).unwrap();
            let coordinate =
                ScalarSymbolCoordinate::for_value(SymbolRef::Parameter(Id::new()), &ty)
                    .unwrap()
                    .remove(0);
            inputs.push(
                builder
                    .input(ScalarInputSlot::new(ordinal as u32, coordinate, dimension))
                    .unwrap(),
            );
        }
        let root = builder.div(inputs[0], inputs[1]).unwrap();
        let zero = builder.constant(DynQuantity::new(0., length)).unwrap();
        let positive = builder
            .compare(
                eqiora_schema::kernel::ComparisonOp::Greater,
                inputs[0],
                zero,
            )
            .unwrap();
        let root = builder.require(positive, root).unwrap();
        let da = builder
            .constant(DynQuantity::new(scale, DimExponents::DIMENSIONLESS))
            .unwrap();
        let db = builder
            .constant(DynQuantity::new(2. * scale, time.div(length).unwrap()))
            .unwrap();
        let derivative = builder
            .directional(root, &[Some(da), Some(db)], reject_finite_map_work)
            .unwrap()
            .unwrap();
        let ir = builder.finish([derivative]).unwrap();
        // At a=b=s, g(a,b)=s/b-2sa/b²=-1. Its directional first and
        // mixed derivatives for s*e_a and s*e_b are -2 and 4.
        assert_eq!(ir.evaluate(&[scale, scale]).unwrap(), [-1.]);
        let action = ir
            .linearize(&[scale, scale], &[DifferentiationRole::Unknown; 2])
            .unwrap();
        let mut result = [0.];
        action
            .jvp(RelationTangent::Unknown(&[scale, 0.]), &mut result)
            .unwrap();
        assert_eq!(result, [-2.]);
        let mut reverse = [0.; 2];
        action
            .vjp(&[scale], RelationCotangent::Unknown(&mut reverse))
            .unwrap();
        assert_eq!(reverse, [-2., 3.]);
        action
            .mixed_jvp(
                RelationTangent::Unknown(&[scale, 0.]),
                RelationTangent::Unknown(&[0., scale]),
                RelationTangent::Unknown(&[0.; 2]),
                &mut result,
            )
            .unwrap();
        assert_eq!(result, [4.]);
        assert!(ir.evaluate(&[-scale, scale]).is_err());
        assert!(ir.evaluate(&[scale, 0.]).is_err());
    }
}

#[test]
fn quotient_products_retain_representable_actions_across_binary64_scales() {
    let instructions = [
        Instruction::Read(SymbolSlot(0)),
        Instruction::Read(SymbolSlot(1)),
        Instruction::Div(ValueId(0), ValueId(1)),
    ];
    let roots = [ValueId(2)];
    for exponent in [-1020, -700, 0, 700, 1020] {
        let scale = 2f64.powi(exponent);
        let action = ScalarLinearization::bind(
            &instructions,
            &roots,
            2,
            &[scale, scale],
            &[DifferentiationRole::Unknown; 2],
        )
        .unwrap();
        // f(s(1+t), s(1+u))=(1+t)/(1+u): f_t=1, f_u=-1,
        // f_tu=-1 and f_uu=2, independently of the power-of-two scale s.
        let mut output = [0.];
        for (direction, expected) in [
            ([scale, 0.], 1.),
            ([0., scale], -1.),
            ([scale, scale], 0.),
            ([-scale, scale], -2.),
        ] {
            action
                .jvp(RelationTangent::Unknown(&direction), &mut output)
                .unwrap();
            assert_eq!(output, [expected]);
        }
        for weight in [1., scale] {
            let mut reverse = [0.; 2];
            action
                .vjp(&[weight], RelationCotangent::Unknown(&mut reverse))
                .unwrap();
            assert_eq!(reverse, [weight / scale, -weight / scale]);
        }
        for (left, expected) in [([scale, 0.], -1.), ([0., scale], 2.)] {
            action
                .mixed_jvp(
                    RelationTangent::Unknown(&left),
                    RelationTangent::Unknown(&[0., scale]),
                    RelationTangent::Unknown(&[0.; 2]),
                    &mut output,
                )
                .unwrap();
            assert_eq!(output, [expected]);
        }
    }
    let invalid = ScalarLinearization::bind(
        &instructions,
        &roots,
        2,
        &[1., 0.],
        &[DifferentiationRole::Unknown; 2],
    )
    .unwrap();
    assert!(
        invalid
            .jvp(RelationTangent::Unknown(&[0.; 2]), &mut [0.])
            .is_err()
    );
    assert!(
        invalid
            .vjp(&[0.], RelationCotangent::Unknown(&mut [0.; 2]))
            .is_err()
    );
}

#[test]
fn emitted_and_affine_quotient_actions_use_the_same_range_safe_rule() {
    let dimension = DimExponents::DIMENSIONLESS;
    let ty = ValueType::scalar(ScalarDomain::Real, dimension).unwrap();
    for exponent in [-1020, -700, 0, 700, 1020] {
        let scale = 2f64.powi(exponent);
        let mut builder = ScalarInputIrBuilder::new();
        let mut inputs = Vec::new();
        for ordinal in 0..2 {
            let symbol = SymbolRef::Parameter(Id::new());
            let coordinate = ScalarSymbolCoordinate::for_value(symbol, &ty)
                .unwrap()
                .remove(0);
            inputs.push(
                builder
                    .input(ScalarInputSlot::new(ordinal, coordinate, dimension))
                    .unwrap(),
            );
        }
        let root = builder.div(inputs[0], inputs[1]).unwrap();
        let da = builder
            .constant(DynQuantity::new(scale, dimension))
            .unwrap();
        let db = builder
            .constant(DynQuantity::new(2. * scale, dimension))
            .unwrap();
        let seeds = [Some(da), Some(db)];
        let first = builder
            .directional(root, &seeds, reject_finite_map_work)
            .unwrap()
            .unwrap();
        let second = builder
            .directional(first, &seeds, reject_finite_map_work)
            .unwrap()
            .unwrap();
        let ir = builder.finish([root, first, second]).unwrap();
        // f(x)=(1+x)/(1+2x) has f(0)=1, f'(0)=-1, f''(0)=4.
        assert_eq!(ir.evaluate(&[scale, scale]).unwrap(), [1., -1., 4.]);

        let numerator = AffineValue {
            value: scale,
            tangent: Some(Scaled::new(scale)),
        };
        let denominator = AffineValue {
            value: scale,
            tangent: Some(Scaled::new(2. * scale)),
        };
        assert_eq!(
            numerator.divide(denominator),
            AffineValue {
                value: 1.,
                tangent: Some(Scaled::new(-1.))
            }
        );
        assert_eq!(
            AffineValue::constant(scale)
                .divide(AffineValue::constant(scale))
                .tangent,
            None
        );
    }
}
