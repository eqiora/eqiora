//! Quotient values and derivatives derived in independent real coordinates.
use super::*;
use crate::{DifferentiationRole, LinearizedRelation, RelationCotangent, RelationTangent};
use eqiora_core::{DimExponents, Id, ValueType, entity::kinds};
use eqiora_schema::kernel::{
    ExprDagBuilder,
    pure_operator::{CalculusBuilder, CalculusNode, PureValueClass},
    typing::RootContract,
};

#[test]
fn complex_quotient_primal_and_real_pairing_products() {
    check_quotient(false);
    check_quotient(true);
}

#[test]
fn real_closed_quotient_actions_survive_denominator_square_underflow() {
    let numerator = SymbolRef::Field(Id::<kinds::Field>::new());
    let denominator = SymbolRef::Parameter(Id::<kinds::Parameter>::new());
    let mut dag = ExprDagBuilder::new();
    let a = dag.symbol(numerator).unwrap();
    let b = dag.symbol(denominator).unwrap();
    let root = dag.pure_operator(&quotient_definition(), [a, b]).unwrap();
    let typed = TypedResidual::infer(
        dag.finish([root]).unwrap(),
        None,
        |_| None,
        RootContract::ComponentwiseResidual,
        |_| {
            Ok::<_, ()>(ExpressionType::<()>::new(
                ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS).unwrap(),
                None,
            ))
        },
    )
    .unwrap();
    let lowered = ComponentScalarization::lower(&typed).unwrap();
    let role = |symbol| {
        if symbol == numerator {
            DifferentiationRole::Unknown
        } else {
            DifferentiationRole::Parameter
        }
    };
    let linearized = lowered
        .linearize(|c| Some((1e-200, role(c.symbol()))))
        .unwrap();
    let mut value = [0.];
    linearized.primal(&mut value).unwrap();
    assert_eq!(value, [1.]);
    // d(u/v)/dv = -u/v² = -1e200 at u=v=1e-200. The result is
    // representable even though evaluating v² in binary64 gives zero.
    linearized
        .jvp(RelationTangent::Parameter(&[1.]), &mut value)
        .unwrap();
    assert!((value[0] / -1e200 - 1.).abs() <= 16. * f64::EPSILON);
    linearized
        .jvp(
            RelationTangent::Both {
                unknown: &[1.],
                parameter: &[1.],
            },
            &mut value,
        )
        .unwrap();
    assert_eq!(value, [0.]);
    let mut du = [0.];
    let mut dp = [0.];
    linearized
        .vjp(
            &[1e-200],
            RelationCotangent::Both {
                unknown: &mut du,
                parameter: &mut dp,
            },
        )
        .unwrap();
    assert!((du[0] - 1.).abs() <= 16. * f64::EPSILON);
    assert!((dp[0] + 1.).abs() <= 16. * f64::EPSILON);
    let zero = lowered.linearize(|c| Some((0., role(c.symbol())))).unwrap();
    assert!(zero.primal(&mut value).is_err());
}

fn check_quotient(closed_operator: bool) {
    let numerator = SymbolRef::Field(Id::<kinds::Field>::new());
    let denominator = SymbolRef::Parameter(Id::<kinds::Parameter>::new());
    let mut dag = ExprDagBuilder::new();
    let a = dag.symbol(numerator).unwrap();
    let b = dag.symbol(denominator).unwrap();
    let q = if closed_operator {
        dag.pure_operator(&quotient_definition(), [a, b]).unwrap()
    } else {
        dag.div(a, b).unwrap()
    };
    let typed = TypedResidual::infer(
        dag.finish([q]).unwrap(),
        None,
        |_| None,
        RootContract::ComponentwiseResidual,
        |_| {
            Ok::<_, ()>(ExpressionType::<()>::new(
                ValueType::scalar(ScalarDomain::Complex, DimExponents::DIMENSIONLESS).unwrap(),
                None,
            ))
        },
    )
    .unwrap();
    let lowered = ComponentScalarization::lower(&typed).unwrap();
    let close = |actual: f64, expected: f64| {
        assert!(
            (actual - expected).abs() < 16. * f64::EPSILON,
            "{actual} != {expected}"
        );
    };
    // (x+iy)/(u+iv) = ((xu+yv)+i(yu-xv))/(u²+v²).
    // Differentiating these two real expressions at (2,3,4,-1) gives
    // [4/17,-1/17,-6/289,61/289] and [1/17,4/17,-61/289,-6/289].
    let linearized = lowered
        .linearize(|c| {
            Some(if c.symbol() == numerator {
                (
                    [2., 3.][usize::from(c.is_imaginary())],
                    DifferentiationRole::Unknown,
                )
            } else {
                (
                    [4., -1.][usize::from(c.is_imaginary())],
                    DifferentiationRole::Parameter,
                )
            })
        })
        .unwrap();
    let mut value = [0.; 2];
    linearized.primal(&mut value).unwrap();
    for (actual, expected) in value.into_iter().zip([5. / 17., 14. / 17.]) {
        close(actual, expected);
    }
    let direction = |coordinates: &[crate::ScalarSymbolCoordinate], values: [f64; 2]| {
        coordinates
            .iter()
            .map(|c| values[usize::from(c.is_imaginary())])
            .collect::<Vec<_>>()
    };
    let unknown = direction(linearized.unknown_coordinates(), [1., -2.]);
    let parameter = direction(linearized.parameter_coordinates(), [3., -4.]);
    linearized
        .jvp(
            RelationTangent::Both {
                unknown: &unknown,
                parameter: &parameter,
            },
            &mut value,
        )
        .unwrap();
    for (actual, expected) in value.into_iter().zip([-160. / 289., -278. / 289.]) {
        close(actual, expected);
    }
    close(2. * value[0] - 3. * value[1], 514. / 289.);
    let mut du = [0.; 2];
    let mut dp = [0.; 2];
    linearized
        .vjp(
            &[2., -3.],
            RelationCotangent::Both {
                unknown: &mut du,
                parameter: &mut dp,
            },
        )
        .unwrap();
    for (c, actual) in linearized.unknown_coordinates().iter().zip(du) {
        close(
            actual,
            [5. / 17., -14. / 17.][usize::from(c.is_imaginary())],
        );
    }
    for (c, actual) in linearized.parameter_coordinates().iter().zip(dp) {
        close(
            actual,
            [171. / 289., 140. / 289.][usize::from(c.is_imaginary())],
        );
    }
    // At a=b=tiny, q=1. Scaling da by tiny gives dq=1, while scaling
    // both da and db equally gives dq=0. These derivatives are representable
    // although the unscaled coordinate Jacobian 1/tiny overflows binary64.
    let tiny = f64::from_bits(1);
    let scaled = lowered
        .linearize(|c| {
            Some((
                if c.is_imaginary() { 0. } else { tiny },
                if c.symbol() == numerator {
                    DifferentiationRole::Unknown
                } else {
                    DifferentiationRole::Parameter
                },
            ))
        })
        .unwrap();
    let unknown = direction(scaled.unknown_coordinates(), [tiny, 0.]);
    let parameter = direction(scaled.parameter_coordinates(), [tiny, 0.]);
    scaled
        .jvp(RelationTangent::Unknown(&unknown), &mut value)
        .unwrap();
    assert_eq!(value, [1., 0.]);
    scaled
        .jvp(
            RelationTangent::Both {
                unknown: &unknown,
                parameter: &parameter,
            },
            &mut value,
        )
        .unwrap();
    assert_eq!(value, [0., 0.]);
    scaled
        .vjp(
            &[tiny, 0.],
            RelationCotangent::Both {
                unknown: &mut du,
                parameter: &mut dp,
            },
        )
        .unwrap();
    for (c, actual) in scaled.unknown_coordinates().iter().zip(du) {
        assert_eq!(actual, if c.is_imaginary() { 0. } else { 1. });
    }
    for (c, actual) in scaled.parameter_coordinates().iter().zip(dp) {
        assert_eq!(actual, if c.is_imaginary() { 0. } else { -1. });
    }
    let invalid = lowered.linearize(|c| {
        Some((
            if c.symbol() == denominator { 0. } else { 1. },
            DifferentiationRole::Unknown,
        ))
    });
    assert!(invalid.unwrap().primal(&mut [0.; 2]).is_err());
}

fn quotient_definition() -> eqiora_schema::kernel::pure_operator::PureOperatorDefinition {
    let generic = PureValueClass::invariant_scalar();
    let mut builder = CalculusBuilder::new(vec![generic.clone(); 2], generic).unwrap();
    let left = builder
        .push(CalculusNode::FormalComponent {
            formal: 0,
            axes: Box::new([]),
        })
        .unwrap();
    let right = builder
        .push(CalculusNode::FormalComponent {
            formal: 1,
            axes: Box::new([]),
        })
        .unwrap();
    let root = builder.push(CalculusNode::Div(left, right)).unwrap();
    builder.finish(root).unwrap()
}
