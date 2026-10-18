//! Unitful guarded components share scalar point-domain and derivative decisions.
use super::*;
use crate::{DifferentiationRole, LinearizedRelation, RelationCotangent, RelationTangent};
use eqiora_core::{DimExponents, Id, ScalarDomain, ValueType};
use eqiora_schema::kernel::{ComparisonOp, ExprDagBuilder, typing::RootContract};

#[test]
fn guarded_vector_keeps_units_and_rejects_active_boundaries() {
    let x = SymbolRef::Field(Id::new());
    let p = SymbolRef::Parameter(Id::new());
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let mut builder = ExprDagBuilder::new();
    let field = builder.symbol(x).unwrap();
    let gain = builder.symbol(p).unwrap();
    let lower = builder.constant(DynQuantity::new(0.0, length)).unwrap();
    let upper = builder.constant(DynQuantity::new(2.0, length)).unwrap();
    let above = builder
        .compare(ComparisonOp::GreaterEqual, field, lower)
        .unwrap();
    let below = builder
        .compare(ComparisonOp::LessEqual, field, upper)
        .unwrap();
    let condition = builder.and(above, below).unwrap();
    let product = builder.mul(field, gain).unwrap();
    let double = builder.add(product, product).unwrap();
    let vector = builder.array([product, double]).unwrap();
    let root = builder.require(condition, vector).unwrap();
    let typed = TypedResidual::<()>::infer(
        builder.finish([root]).unwrap(),
        None,
        |_| None,
        RootContract::InitialResiduals,
        |symbol| {
            Ok::<_, ()>(ExpressionType::new(
                ValueType::scalar(
                    ScalarDomain::Real,
                    if symbol == x {
                        length
                    } else {
                        DimExponents::DIMENSIONLESS
                    },
                )
                .unwrap(),
                None,
            ))
        },
    )
    .unwrap();
    let program = ComponentScalarization::lower(&typed).unwrap();
    let point = |value, role| {
        program.linearize(|coordinate| {
            Some(if coordinate.symbol() == x {
                (value, role)
            } else {
                (3.0, DifferentiationRole::Parameter)
            })
        })
    };
    let action = point(1.0, DifferentiationRole::Unknown).unwrap();
    let mut primal = [0.0; 2];
    action.primal(&mut primal).unwrap();
    assert_eq!(primal, [3.0, 6.0]);
    let mut tangent = [0.0; 2];
    action
        .jvp(
            RelationTangent::Both {
                unknown: &[1.0],
                parameter: &[2.0],
            },
            &mut tangent,
        )
        .unwrap();
    assert_eq!(tangent, [5.0, 10.0]);
    let (mut dx, mut dp) = ([0.0], [0.0]);
    action
        .vjp(
            &[2.0, -0.5],
            RelationCotangent::Both {
                unknown: &mut dx,
                parameter: &mut dp,
            },
        )
        .unwrap();
    assert_eq!(dx, [3.0]);
    assert_eq!(dp, [1.0]);
    for outside in [-1.0, 3.0] {
        assert!(point(outside, DifferentiationRole::Unknown).is_err());
        assert!(point(outside, DifferentiationRole::Frozen).is_err());
    }
    for boundary in [0.0, 2.0] {
        assert!(point(boundary, DifferentiationRole::Unknown).is_err());
        point(boundary, DifferentiationRole::Frozen).unwrap();
    }
}

fn guarded_map(inverse: bool) -> (ComponentScalarization, SymbolRef, SymbolRef) {
    use eqiora_core::FiniteBasis;
    use eqiora_schema::kernel::FiniteUnaryOperation;
    let matrix = SymbolRef::Field(Id::new());
    let gate = SymbolRef::Parameter(Id::new());
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let basis = FiniteBasis::new(Id::new(), 2).unwrap();
    let map = ValueType::linear_map(basis, basis, ScalarDomain::Real, length).unwrap();
    let mut builder = ExprDagBuilder::new();
    let a = builder.symbol(matrix).unwrap();
    let p = builder.symbol(gate).unwrap();
    let zero = builder
        .constant(DynQuantity::new(0., DimExponents::DIMENSIONLESS))
        .unwrap();
    let condition = builder
        .compare(ComparisonOp::GreaterEqual, p, zero)
        .unwrap();
    // Guard each map coordinate before gathering the invariant's operands.
    let a = builder.require(condition, a).unwrap();
    let det = builder
        .finite_unary(FiniteUnaryOperation::Determinant, a)
        .unwrap();
    let mut roots = vec![det];
    if inverse {
        roots.push(
            builder
                .finite_unary(FiniteUnaryOperation::Inverse, a)
                .unwrap(),
        );
    }
    let typed = TypedResidual::<()>::infer(
        builder.finish(roots).unwrap(),
        None,
        |_| None,
        RootContract::ComponentwiseResidual,
        |symbol| {
            Ok::<_, ()>(ExpressionType::new(
                if symbol == matrix {
                    map.clone()
                } else {
                    ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS).unwrap()
                },
                None,
            ))
        },
    )
    .unwrap();
    (ComponentScalarization::lower(&typed).unwrap(), matrix, gate)
}

#[test]
fn guarded_unitful_maps_share_values_forward_and_reverse_actions() {
    let (program, matrix, _) = guarded_map(true);
    let flat =
        |c: &ScalarSymbolCoordinate| (2 * c.component_index()[0] + c.component_index()[1]) as usize;
    let values = [2., 1., 3., 4.];
    // det(A)=ad-bc=5 m²; inverse(A)=[[d,-b],[-c,a]]/det(A), in 1/m.
    let expected = [5., 4. / 5., -1. / 5., -3. / 5., 2. / 5.];
    let actual = program
        .evaluate(|c| {
            Some(if c.symbol() == matrix {
                values[flat(c)]
            } else {
                0.
            })
        })
        .unwrap();
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-13);
    }
    let action = program
        .linearize(|c| {
            Some(if c.symbol() == matrix {
                (values[flat(c)], DifferentiationRole::Unknown)
            } else {
                (0., DifferentiationRole::Frozen)
            })
        })
        .unwrap();
    let h = [1., 2., -1., 3.];
    let direction = action
        .unknown_coordinates()
        .iter()
        .map(|c| h[flat(c)])
        .collect::<Vec<_>>();
    let mut tangent = [0.; 5];
    action
        .jvp(RelationTangent::Unknown(&direction), &mut tangent)
        .unwrap();
    // Differentiate the explicit two-by-two polynomial/quotients above.
    for (actual, expected) in tangent
        .into_iter()
        .zip([5., -1. / 5., -1. / 5., 4. / 5., -1. / 5.])
    {
        assert!((actual - expected).abs() < 1e-12);
    }
    let mut reverse = vec![0.; direction.len()];
    action
        .vjp(
            &[2., 3., -1., 4., -2.],
            RelationCotangent::Unknown(&mut reverse),
        )
        .unwrap();
    // Gradient of 2(ad-bc)+(3d+b-4c-2a)/(ad-bc).
    let expected = [202. / 25., -154. / 25., -73. / 25., 121. / 25.];
    for (c, actual) in action.unknown_coordinates().iter().zip(reverse) {
        assert!((actual - expected[flat(c)]).abs() < 1e-12);
    }
    // A demanded boundary is regular only when the condition is held fixed.
    assert!(
        program
            .linearize(|c| Some((
                if c.symbol() == matrix {
                    values[flat(c)]
                } else {
                    0.
                },
                DifferentiationRole::Unknown
            )))
            .is_err()
    );
    assert!(
        program
            .evaluate(|c| Some(if c.symbol() == matrix {
                values[flat(c)]
            } else {
                -1.
            }))
            .is_err()
    );
}

#[test]
fn guarded_map_domains_keep_singular_determinants_and_reject_invalid_inverses() {
    let (program, matrix, _) = guarded_map(false);
    let flat =
        |c: &ScalarSymbolCoordinate| (2 * c.component_index()[0] + c.component_index()[1]) as usize;
    let values = [1., 2., 2., 4.];
    let action = program
        .linearize(|c| {
            Some(if c.symbol() == matrix {
                (values[flat(c)], DifferentiationRole::Unknown)
            } else {
                (1., DifferentiationRole::Frozen)
            })
        })
        .unwrap();
    let h = [1., 2., -1., 3.];
    let direction = action
        .unknown_coordinates()
        .iter()
        .map(|c| h[flat(c)])
        .collect::<Vec<_>>();
    let mut value = [0.];
    action.primal(&mut value).unwrap();
    assert_eq!(value, [0.]);
    action
        .jvp(RelationTangent::Unknown(&direction), &mut value)
        .unwrap();
    assert!((value[0] - 5.).abs() < 1e-12);
    let (program, matrix, _) = guarded_map(true);
    for values in [values, [1., 0., 0., 1e-18]] {
        for role in [DifferentiationRole::Unknown, DifferentiationRole::Frozen] {
            assert!(
                program
                    .linearize(|c| Some((
                        if c.symbol() == matrix {
                            values[flat(c)]
                        } else {
                            1.
                        },
                        role
                    )))
                    .is_err()
            );
        }
    }
}
