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
