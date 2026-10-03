//! Independent native property values and point-bound active derivatives.
use eqiora_core::entity::kinds;
use eqiora_core::{DimExponents, Id, ScalarDomain, ValueLiteral, ValueType};
use eqiora_ir::{
    CalculusBuilder, CalculusNode, DifferentiationRole, ExactRational, LinearizedRelation,
    PureValueClass, RelationCotangent, RelationTangent, ScalarOperatorIr,
};
use eqiora_schema::kernel::{ComparisonOp, ExprDagBuilder, ExprId, SymbolRef, UnaryMathFunction};

fn voltage() -> DimExponents {
    DimExponents::from_integers([1, 2, -3, -1, 0, 0, 0]).unwrap()
}
fn current() -> DimExponents {
    DimExponents::from_integers([0, 0, 0, 1, 0, 0, 0]).unwrap()
}
fn real(d: DimExponents, x: f64) -> ValueLiteral {
    ValueLiteral::from_real(
        ValueType::scalar(ScalarDomain::Real, d).expect("numeric scalar type"),
        x,
    )
    .unwrap()
}
fn value(value: &ValueLiteral) -> f64 {
    value.real_scalar_value().unwrap().value()
}
fn class(d: DimExponents) -> PureValueClass {
    PureValueClass::invariant_scalar()
        .with_dimension(d)
        .with_scalar_domain(ScalarDomain::Real)
        .unwrap()
}
fn formal(builder: &mut CalculusBuilder, formal: u16) -> eqiora_ir::CalculusNodeId {
    builder
        .push(CalculusNode::FormalComponent {
            formal,
            axes: Box::new([]),
        })
        .unwrap()
}

fn diode() -> (ScalarOperatorIr, ExprId) {
    let g = current().div(voltage()).unwrap();
    let mut b =
        CalculusBuilder::new([class(voltage()), class(g), class(g)], class(current())).unwrap();
    let x = formal(&mut b, 0);
    let forward = formal(&mut b, 1);
    let reverse = formal(&mut b, 2);
    let zero = b
        .push(CalculusNode::Rational {
            value: ExactRational::integer(0),
            dimension: voltage(),
        })
        .unwrap();
    let condition = b
        .push(CalculusNode::Compare(ComparisonOp::Greater, x, zero))
        .unwrap();
    let then_value = b.push(CalculusNode::Mul(forward, x)).unwrap();
    let else_value = b.push(CalculusNode::Mul(reverse, x)).unwrap();
    let root = b
        .push(CalculusNode::Select {
            condition,
            then_value,
            else_value,
        })
        .unwrap();
    let definition = b.finish(root).unwrap();
    let mut b = ExprDagBuilder::new();
    let args = [0, 1, 2].map(|_| {
        b.symbol(SymbolRef::Parameter(Id::<kinds::Parameter>::new()))
            .unwrap()
    });
    let root = b.pure_operator(&definition, args).unwrap();
    let dag = b.finish([root]).unwrap();
    (ScalarOperatorIr::lower(&dag).unwrap(), root)
}

#[test]
fn retained_piecewise_batch_and_derivatives_preserve_live_parameter_dependencies() {
    let (ir, root) = diode();
    let g = current().div(voltage()).unwrap();
    let rows = [-4., 0., 3.].map(|x| vec![real(voltage(), x), real(g, 2.), real(g, 0.5)]);
    let result = ir.evaluate_typed_batch(&[root], &rows).unwrap();
    assert_eq!(
        result.iter().map(|row| value(&row[0])).collect::<Vec<_>>(),
        [-2., 0., 6.]
    );
    assert!(
        result
            .iter()
            .all(|row| row[0].value_type().dimension() == current())
    );
    let roles = [
        DifferentiationRole::Unknown,
        DifferentiationRole::Parameter,
        DifferentiationRole::Parameter,
    ];
    assert!(ir.linearize_typed(&rows[1], &roles).is_err());
    for (row, slope, parameter) in [(&rows[0], 0.5, [0., -4.]), (&rows[2], 2., [3., 0.])] {
        let linear = ir.linearize_typed(row, &roles).unwrap();
        let mut tangent = [0.];
        linear
            .jvp(RelationTangent::Unknown(&[1.]), &mut tangent)
            .unwrap();
        assert_eq!(tangent, [slope]);
        linear
            .jvp(RelationTangent::Parameter(&[1., 1.]), &mut tangent)
            .unwrap();
        assert_eq!(tangent, [parameter.iter().sum::<f64>()]);
        let mut unknown = [0.];
        let mut params = [0., 0.];
        linear
            .vjp(
                &[1.],
                RelationCotangent::Both {
                    unknown: &mut unknown,
                    parameter: &mut params,
                },
            )
            .unwrap();
        assert_eq!(unknown, [slope]);
        assert_eq!(params, parameter);
    }
    let changed = vec![real(voltage(), 3.), real(g, 4.), real(g, 0.5)];
    assert_eq!(
        value(&ir.evaluate_typed_batch(&[root], &[changed]).unwrap()[0][0]),
        12.
    );
}

#[test]
fn guarded_sqrt_skips_negative_arm_and_rejects_only_demanded_derivative_boundaries() {
    let square = voltage().pow(2, 1).unwrap();
    let mut b = ExprDagBuilder::new();
    let x = b.symbol(SymbolRef::Parameter(Id::new())).unwrap();
    let zero = b.constant(real(square, 0.)).unwrap();
    let condition = b.compare(ComparisonOp::Greater, x, zero).unwrap();
    let positive = b.unary_math(UnaryMathFunction::Sqrt, x).unwrap();
    let fallback = b.constant(real(voltage(), 0.)).unwrap();
    let root = b.select(condition, positive, fallback).unwrap();
    let dag = b.finish([root]).unwrap();
    let ir = ScalarOperatorIr::lower(&dag).unwrap();
    let rows = [4., -1., 0.].map(|x| vec![real(square, x)]);
    let result = ir.evaluate_typed_batch(&[root], &rows).unwrap();
    assert_eq!(
        result.iter().map(|row| value(&row[0])).collect::<Vec<_>>(),
        [2., 0., 0.]
    );
    for (row, expected) in [(&rows[0], 0.25), (&rows[1], 0.)] {
        let linear = ir
            .linearize_typed(row, &[DifferentiationRole::Unknown])
            .unwrap();
        let mut derivative = [0.];
        linear
            .jvp(RelationTangent::Unknown(&[1.]), &mut derivative)
            .unwrap();
        assert_eq!(derivative, [expected]);
    }
    assert!(
        ir.linearize_typed(&rows[2], &[DifferentiationRole::Unknown])
            .is_err()
    );
    assert!(
        ir.evaluate_typed_batch(&[root], &[vec![real(voltage(), 4.)]])
            .is_err(),
        "inactive branch typing remains checked"
    );
}

#[test]
fn require_enforces_clamp_domain_without_invalid_math_sentinels() {
    let mut b = ExprDagBuilder::new();
    let args = [0, 1, 2].map(|_| b.symbol(SymbolRef::Parameter(Id::new())).unwrap());
    let [x, lo, hi] = args;
    let valid = b.compare(ComparisonOp::LessEqual, lo, hi).unwrap();
    let low = b.compare(ComparisonOp::Less, x, lo).unwrap();
    let high = b.compare(ComparisonOp::Greater, x, hi).unwrap();
    let inner = b.select(high, hi, x).unwrap();
    let clamped = b.select(low, lo, inner).unwrap();
    let root = b.require(valid, clamped).unwrap();
    let dag = b.finish([root]).unwrap();
    let ir = ScalarOperatorIr::lower(&dag).unwrap();
    let row = |x, lo, hi| vec![real(voltage(), x), real(voltage(), lo), real(voltage(), hi)];
    let roles = [
        DifferentiationRole::Unknown,
        DifferentiationRole::Parameter,
        DifferentiationRole::Parameter,
    ];
    assert!(
        ir.evaluate_typed_batch(&[root], &[row(1., 3., 2.)])
            .is_err()
    );
    assert_eq!(
        value(
            &ir.evaluate_typed_batch(&[root], &[row(3., 2., 2.)])
                .unwrap()[0][0]
        ),
        2.
    );
    assert!(ir.linearize_typed(&row(3., 2., 2.), &roles).is_err());
    assert!(ir.linearize_typed(&row(0., 0., 2.), &roles).is_err());
    assert!(ir.linearize_typed(&row(2., 0., 2.), &roles).is_err());
    let linear = ir.linearize_typed(&row(1., 0., 2.), &roles).unwrap();
    let mut tangent = [0.];
    linear
        .jvp(RelationTangent::Unknown(&[1.]), &mut tangent)
        .unwrap();
    assert_eq!(tangent, [1.]);
}

#[test]
fn extrema_use_point_bound_selection_derivatives_and_reject_ties() {
    let mut builder = ExprDagBuilder::new();
    let x = builder
        .symbol(SymbolRef::Field(Id::<kinds::Field>::new()))
        .unwrap();
    let cap = builder.constant(real(voltage(), 2.)).unwrap();
    let minimum = builder.min(x, cap).unwrap();
    let maximum = builder.max(x, cap).unwrap();
    let ir = ScalarOperatorIr::lower(&builder.finish([minimum, maximum]).unwrap()).unwrap();
    for (point, expected, slopes) in [(1., [1., 2.], [1., 0.]), (3., [2., 3.], [0., 1.])] {
        let inputs = [real(voltage(), point)];
        let values = ir
            .evaluate_typed(&[minimum, maximum], &mut |_| Some(inputs[0].clone()))
            .unwrap();
        assert_eq!(values.iter().map(value).collect::<Vec<_>>(), expected);
        let linear = ir
            .linearize_typed(&inputs, &[DifferentiationRole::Unknown])
            .unwrap();
        let mut tangent = [0.; 2];
        linear
            .jvp(RelationTangent::Unknown(&[1.]), &mut tangent)
            .unwrap();
        assert_eq!(tangent, slopes);
    }
    assert!(
        ir.linearize_typed(&[real(voltage(), 2.)], &[DifferentiationRole::Unknown])
            .is_err()
    );
}

#[test]
fn sine_direct_and_retained_operator_share_values_and_first_derivatives() {
    let d = DimExponents::DIMENSIONLESS;
    for retained in [false, true] {
        let mut b = ExprDagBuilder::new();
        let x = b.symbol(SymbolRef::Parameter(Id::new())).unwrap();
        let root = if retained {
            let mut c = CalculusBuilder::new([class(d)], class(d)).unwrap();
            let input = formal(&mut c, 0);
            let root = c
                .push(CalculusNode::UnaryMath(UnaryMathFunction::Sin, input))
                .unwrap();
            b.pure_operator(&c.finish(root).unwrap(), [x]).unwrap()
        } else {
            b.unary_math(UnaryMathFunction::Sin, x).unwrap()
        };
        let ir = ScalarOperatorIr::lower(&b.finish([root]).unwrap()).unwrap();
        // Taylor series at zero establishes these exact values and slopes;
        // pi/2 has binary64 argument error, bounded here by four ulps at one.
        for (point, expected, slope) in [
            (0., 0., 1.),
            (std::f64::consts::FRAC_PI_2, 1., 0.),
            (-std::f64::consts::FRAC_PI_2, -1., 0.),
        ] {
            let inputs = [real(d, point)];
            let got = ir
                .evaluate_typed(&[root], &mut |_| Some(inputs[0].clone()))
                .unwrap();
            assert!((value(&got[0]) - expected).abs() <= 4. * f64::EPSILON);
            let linear = ir
                .linearize_typed(&inputs, &[DifferentiationRole::Unknown])
                .unwrap();
            let mut tangent = [0.];
            linear
                .jvp(RelationTangent::Unknown(&[1.]), &mut tangent)
                .unwrap();
            assert!((tangent[0] - slope).abs() <= 4. * f64::EPSILON);
            let mut adjoint = [0.];
            linear
                .vjp(&[1.], RelationCotangent::Unknown(&mut adjoint))
                .unwrap();
            assert!((adjoint[0] - slope).abs() <= 4. * f64::EPSILON);
        }
        assert!(
            ir.evaluate_typed(&[root], &mut |_| Some(real(voltage(), 1.)))
                .is_err()
        );
    }
}

#[test]
fn frozen_typed_subgraphs_have_zero_tangent_without_evaluating_inactive_domains() {
    let d = DimExponents::DIMENSIONLESS;
    let mut b = ExprDagBuilder::new();
    let x = b.symbol(SymbolRef::Field(Id::new())).unwrap();
    let integer = |n| {
        ValueLiteral::integer(ValueType::scalar(ScalarDomain::Integer, d).unwrap(), [n]).unwrap()
    };
    let n = b.constant(integer(7)).unwrap();
    let two = b.constant(integer(2)).unwrap();
    let quotient = b.quotient(n, two).unwrap();
    let coefficient = b.to_real(quotient).unwrap();
    let active = b.mul(coefficient, x).unwrap();
    let zero = b.constant(real(d, 0.)).unwrap();
    let sqrt_zero = b.unary_math(UnaryMathFunction::Sqrt, zero).unwrap();
    let minus_one = b.constant(real(d, -1.)).unwrap();
    let invalid = b.unary_math(UnaryMathFunction::Sqrt, minus_one).unwrap();
    let yes = b.constant(ValueLiteral::boolean(true)).unwrap();
    let chosen = b.select(yes, active, invalid).unwrap();
    let root = b.add(chosen, sqrt_zero).unwrap();
    let ir = ScalarOperatorIr::lower(&b.finish([root]).unwrap()).unwrap();
    let linear = ir
        .linearize_typed(&[real(d, 2.)], &[DifferentiationRole::Unknown])
        .unwrap();
    let mut tangent = [0.];
    linear
        .jvp(RelationTangent::Unknown(&[1.]), &mut tangent)
        .unwrap();
    // Integer quotient 7/2 is 3; hence d(3*x + sqrt(0))/dx = 3.
    assert_eq!(tangent, [3.]);
    let mut adjoint = [0.];
    linear
        .vjp(&[1.], RelationCotangent::Unknown(&mut adjoint))
        .unwrap();
    assert_eq!(adjoint, [3.]);
}

#[test]
fn selected_array_derivatives_preserve_eager_value_errors_and_only_selected_boundaries() {
    let d = DimExponents::DIMENSIONLESS;
    let mut b = ExprDagBuilder::new();
    let x = b.symbol(SymbolRef::Field(Id::new())).unwrap();
    let zero = b.constant(real(d, 0.)).unwrap();
    let twice = b.add(x, x).unwrap();
    let root = b.unary_math(UnaryMathFunction::Sqrt, x).unwrap();
    let array = b.array([twice, root, zero]).unwrap();
    let first = b.index(array, 0).unwrap();
    let last = b.index(array, 2).unwrap();
    let ir = ScalarOperatorIr::lower(&b.finish([first, last]).unwrap()).unwrap();
    let linear = ir
        .linearize_typed(&[real(d, 0.)], &[DifferentiationRole::Unknown])
        .unwrap();
    let mut tangent = [0.; 2];
    linear
        .jvp(RelationTangent::Unknown(&[1.]), &mut tangent)
        .unwrap();
    assert_eq!(tangent, [2., 0.]);
    // sqrt(0) is a valid eagerly evaluated value, but its derivative is not
    // demanded. sqrt(-1) is an invalid value even in the nonselected component.
    assert!(
        ir.linearize_typed(&[real(d, -1.)], &[DifferentiationRole::Unknown])
            .is_err()
    );
}

#[test]
fn canonical_abs_preserves_real_values_slopes_and_zero_boundary_rejection() {
    let dimension = voltage();
    for retained in [false, true] {
        let mut b = ExprDagBuilder::new();
        let x = b.symbol(SymbolRef::Parameter(Id::new())).unwrap();
        let root = if retained {
            let mut c = CalculusBuilder::new([class(dimension)], class(dimension)).unwrap();
            let input = formal(&mut c, 0);
            let root = c
                .push(CalculusNode::UnaryMath(UnaryMathFunction::Abs, input))
                .unwrap();
            b.pure_operator(&c.finish(root).unwrap(), [x]).unwrap()
        } else {
            b.unary_math(UnaryMathFunction::Abs, x).unwrap()
        };
        let ir = ScalarOperatorIr::lower(&b.finish([root]).unwrap()).unwrap();
        for (point, expected, slope) in [(-3., 3., -1.), (2., 2., 1.)] {
            let input = real(dimension, point);
            let result = ir
                .evaluate_typed(&[root], &mut |_| Some(input.clone()))
                .unwrap();
            assert_eq!(value(&result[0]), expected);
            assert_eq!(result[0].value_type().dimension(), dimension);
            let linear = ir
                .linearize_typed(&[input], &[DifferentiationRole::Unknown])
                .unwrap();
            let mut tangent = [0.];
            linear
                .jvp(RelationTangent::Unknown(&[1.]), &mut tangent)
                .unwrap();
            assert_eq!(tangent, [slope]);
            let mut adjoint = [0.];
            linear
                .vjp(&[1.], RelationCotangent::Unknown(&mut adjoint))
                .unwrap();
            assert_eq!(adjoint, [slope]);
        }
        let zero = real(dimension, 0.);
        assert_eq!(
            value(
                &ir.evaluate_typed(&[root], &mut |_| Some(zero.clone()))
                    .unwrap()[0]
            ),
            0.
        );
        assert!(
            ir.linearize_typed(&[zero], &[DifferentiationRole::Unknown])
                .is_err()
        );
    }
}
