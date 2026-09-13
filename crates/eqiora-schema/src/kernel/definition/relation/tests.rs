use super::*;
use crate::kernel::typing::{ExpressionType, RootContract, SpatialSupport, TypedResidual};
use crate::kernel::{ExprDagBuilder, SymbolRef};
use eqiora_core::{DimExponents, ScalarDomain, ValueType};

fn dimension(values: [i32; 7]) -> DimExponents {
    DimExponents::from_integers(values).unwrap()
}

fn check(
    kind: RelationConditionKind,
    left: ExpressionType<u32>,
    right: ExpressionType<u32>,
    support: Option<SpatialSupport<u32>>,
) -> bool {
    let first = Id::new();
    let second = Id::new();
    let mut builder = ExprDagBuilder::new();
    let a = builder.symbol(SymbolRef::Field(first)).unwrap();
    let b = builder.symbol(SymbolRef::Field(second)).unwrap();
    let dag = builder.finish([a, b]).unwrap();
    let relation = RelationDef::with_conditions(Id::new(), dag.clone(), vec![kind]).unwrap();
    let typed = TypedResidual::infer(
        dag,
        support.clone(),
        RootContract::RelationOperands,
        |symbol| {
            if symbol == SymbolRef::Field(first) {
                Ok::<_, ()>(left.clone())
            } else {
                Ok(right.clone())
            }
        },
    )
    .unwrap();
    relation
        .validate_conditions(&typed, support.as_ref())
        .is_ok()
}

#[test]
fn complementarity_orders_gap_and_force_without_equating_units() {
    let gap = ExpressionType::new(
        ValueType::scalar(ScalarDomain::Real, dimension([0, 1, 0, 0, 0, 0, 0])).unwrap(),
        None,
    );
    let force = ExpressionType::new(
        ValueType::scalar(ScalarDomain::Real, dimension([1, 1, -2, 0, 0, 0, 0])).unwrap(),
        None,
    );
    assert!(check(
        RelationConditionKind::Complementarity,
        gap.clone(),
        force.clone(),
        None
    ));
    assert!(!check(
        RelationConditionKind::Equality,
        gap.clone(),
        force.clone(),
        None
    ));
    assert!(!check(
        RelationConditionKind::Inequality,
        gap.clone(),
        force,
        None
    ));
    assert!(check(
        RelationConditionKind::Inequality,
        gap.clone(),
        gap,
        None
    ));
}

#[test]
fn order_is_neither_boolean_nor_complex_nor_a_vector() {
    let real = ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS).unwrap();
    for unsupported in [
        ValueType::boolean(),
        ValueType::scalar(ScalarDomain::Complex, DimExponents::DIMENSIONLESS).unwrap(),
        real.clone().array(2).unwrap(),
    ] {
        assert!(!check(
            RelationConditionKind::Complementarity,
            ExpressionType::new(unsupported, None),
            ExpressionType::new(real.clone(), None),
            None
        ));
    }
}

#[test]
fn complementarity_preserves_exact_support_not_equal_shape() {
    let real = ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS).unwrap();
    let first = SpatialSupport::Volume {
        domain: 7,
        dimensions: 1,
    };
    let second = SpatialSupport::Volume {
        domain: 8,
        dimensions: 1,
    };
    assert!(check(
        RelationConditionKind::Complementarity,
        ExpressionType::new(real.clone(), Some(first.clone())),
        ExpressionType::new(real.clone(), Some(first.clone())),
        Some(first.clone())
    ));
    assert!(!check(
        RelationConditionKind::Complementarity,
        ExpressionType::new(real.clone(), Some(first.clone())),
        ExpressionType::new(real, Some(second)),
        Some(first)
    ));
}

#[test]
fn descriptor_count_cannot_drop_a_nonnegativity_condition() {
    let mut builder = ExprDagBuilder::new();
    let a = builder.symbol(SymbolRef::Field(Id::new())).unwrap();
    let expression = builder.finish([a, a]).unwrap();
    assert!(RelationDef::with_conditions(Id::new(), expression, vec![]).is_err());
}
