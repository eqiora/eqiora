//! Projection preserves the authored complete value's evaluation obligations.
use super::*;
use crate::ScalarOperatorIr;
use eqiora_core::{DimExponents, ValueLiteral};
use eqiora_schema::kernel::{ExprDagBuilder, UnaryMathFunction, typing::RootContract};

fn projected_value(argument: f64, complex: bool, inactive: bool) -> TypedResidual<()> {
    let mut b = ExprDagBuilder::new();
    let one = b
        .constant(DynQuantity::new(1., DimExponents::DIMENSIONLESS))
        .unwrap();
    let x = b
        .constant(DynQuantity::new(argument, DimExponents::DIMENSIONLESS))
        .unwrap();
    let other = b.unary_math(UnaryMathFunction::Log, x).unwrap();
    let root = if complex {
        let value = b.complex(one, other).unwrap();
        b.unary_math(UnaryMathFunction::Real, value).unwrap()
    } else {
        let value = b.array([one, other]).unwrap();
        b.index(value, 0).unwrap()
    };
    let root = if inactive {
        let condition = b.constant(ValueLiteral::boolean(false)).unwrap();
        b.select(condition, root, one).unwrap()
    } else {
        root
    };
    TypedResidual::infer(
        b.finish([root]).unwrap(),
        None,
        |_| None,
        RootContract::ComponentwiseResidual,
        |_| -> Result<ExpressionType<()>, ()> { unreachable!() },
    )
    .unwrap()
}

#[test]
fn projection_preserves_complete_array_and_complex_domains() {
    for complex in [false, true] {
        for argument in [1., -1.] {
            let typed = projected_value(argument, complex, false);
            let dag = typed.expression();
            // The ordinary scalar evaluator directly represents authored arrays.
            // Complex construction has a separate component lowering boundary.
            let ordinary = (!complex).then(|| {
                ScalarOperatorIr::lower(dag)
                    .unwrap()
                    .evaluate_typed(dag.roots(), &mut |_| None)
            });
            let components = ComponentScalarization::lower(&typed)
                .unwrap()
                .evaluate(|_| None);
            if argument > 0. {
                if let Some(ordinary) = ordinary {
                    assert_eq!(
                        ordinary.unwrap()[0].real_scalar_value().unwrap().value(),
                        1.
                    );
                }
                assert_eq!(components.unwrap(), [1.]);
            } else {
                if let Some(ordinary) = ordinary {
                    assert_eq!(
                        ordinary.unwrap_err().code(),
                        codes::EXPRESSION_DOMAIN_VIOLATION
                    );
                }
                assert_eq!(
                    components.unwrap_err().code(),
                    codes::EXPRESSION_DOMAIN_VIOLATION,
                    "complex={complex}"
                );
            }
        }
    }
}

#[test]
fn complete_value_demand_stays_inside_its_selected_branch() {
    for complex in [false, true] {
        let typed = projected_value(-1., complex, true);
        assert_eq!(
            ComponentScalarization::lower(&typed)
                .unwrap()
                .evaluate(|_| None)
                .unwrap(),
            [1.]
        );
    }
}

#[test]
fn demanded_values_retain_domains_without_creating_derivative_contributions() {
    use crate::{DifferentiationRole, LinearizedRelation, RelationCotangent, RelationTangent};
    use eqiora_core::{Id, ScalarDomain, ValueType};
    let x = SymbolRef::Parameter(Id::new());
    let mut b = ExprDagBuilder::new();
    let input = b.symbol(x).unwrap();
    let one = b
        .constant(DynQuantity::new(1., DimExponents::DIMENSIONLESS))
        .unwrap();
    let log = b.unary_math(UnaryMathFunction::Log, input).unwrap();
    let array = b.array([one, log]).unwrap();
    let root = b.index(array, 0).unwrap();
    let typed = TypedResidual::infer(
        b.finish([root]).unwrap(),
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
    let components = ComponentScalarization::lower(&typed).unwrap();
    let action = components
        .linearize(|_| Some((2., DifferentiationRole::Parameter)))
        .unwrap();
    assert_eq!(action.parameter_coordinates().len(), 1);
    let mut output = [0.];
    action.primal(&mut output).unwrap();
    assert_eq!(output, [1.]);
    action
        .jvp(RelationTangent::Parameter(&[3.]), &mut output)
        .unwrap();
    assert_eq!(output, [0.]);
    action
        .vjp(&[7.], RelationCotangent::Parameter(&mut output))
        .unwrap();
    assert_eq!(output, [0.]);
    for invalid in [0., -1.] {
        assert!(
            components
                .linearize(|_| Some((invalid, DifferentiationRole::Parameter)))
                .is_err()
        );
        assert!(
            components.rows()[0]
                .bind_affine(&[], &[(components.rows()[0].symbols()[0].clone(), invalid)])
                .is_err()
        );
    }
}

#[test]
fn complete_value_demand_uses_the_current_coordinate_substitution() {
    use eqiora_core::{Id, entity::kinds};
    use eqiora_schema::kernel::typing::SpatialSupport;
    let domain = Id::<kinds::Domain>::new();
    let symbol = SymbolRef::Coordinate {
        support: domain,
        factor: domain,
        axis: 0,
    };
    let mut b = ExprDagBuilder::new();
    let x = b.symbol(symbol).unwrap();
    let unit = b
        .constant(DynQuantity::new(
            1.,
            DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap(),
        ))
        .unwrap();
    let normalized = b.div(x, unit).unwrap();
    let log = b.unary_math(UnaryMathFunction::Log, normalized).unwrap();
    let one = b
        .constant(DynQuantity::new(1., DimExponents::DIMENSIONLESS))
        .unwrap();
    let array = b.array([one, log]).unwrap();
    let selected = b.index(array, 0).unwrap();
    let negative = b.neg(x).unwrap();
    let pulled = b.pullback(selected, vec![x], vec![(x, negative)]).unwrap();
    let root = b.add(selected, pulled).unwrap();
    let typed = TypedResidual::infer(
        b.finish([root]).unwrap(),
        None,
        |_| None,
        RootContract::ValueRoots,
        |_| {
            ExpressionType::coordinate(
                &domain.erase(),
                0,
                Some(&SpatialSupport::Volume {
                    domain: domain.erase(),
                    dimensions: 1,
                }),
            )
        },
    )
    .unwrap();
    let lowered = ComponentScalarization::lower(&typed).unwrap();
    // One occurrence requires x>0 and the self-pullback requires -x>0.
    // Reusing the first occurrence's demand graph would incorrectly return 2.
    assert_eq!(
        lowered.evaluate(|_| Some(1.)).unwrap_err().code(),
        codes::EXPRESSION_DOMAIN_VIOLATION
    );
}
