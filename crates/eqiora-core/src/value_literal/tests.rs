use super::*;
use crate::{DimExponents, ValueFrame, ValueShape};

fn real() -> ValueType {
    ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS).expect("checked scalar type")
}
fn complex() -> ValueType {
    ValueType::scalar(ScalarDomain::Complex, DimExponents::DIMENSIONLESS)
        .expect("checked scalar type")
}

#[test]
fn complex_and_spatial_channel_values_retain_every_ordered_component() {
    let scalar = ValueLiteral::new(complex(), [(2.0, -3.0)]).unwrap();
    assert_eq!(scalar.component(0), Some((2.0, -3.0)));
    assert_eq!(scalar.real_scalar_value(), None);
    let element = ValueType::shaped(
        ScalarDomain::Complex,
        DimExponents::DIMENSIONLESS,
        ValueShape::new([2]).unwrap(),
        ValueFrame::SpatialCartesian,
    )
    .unwrap();
    let ty = element.array(3).unwrap();
    // Three ordered channels, each with two Cartesian components.
    let expected = [
        (1.0, 2.0),
        (3.0, 4.0),
        (5.0, 6.0),
        (7.0, 8.0),
        (9.0, 10.0),
        (11.0, 12.0),
    ];
    let value = ValueLiteral::new(ty.clone(), expected).unwrap();
    assert_eq!(value.value_type(), &ty);
    assert_eq!(value.component_count(), 6);
    assert_eq!(value.components().unwrap().collect::<Vec<_>>(), expected);
    assert_eq!(value.components().unwrap().next_back(), Some((11.0, 12.0)));
    assert_eq!(value.component(6), None);
    assert_eq!(value.real_scalar_value(), None);
    assert!(!value.is_zero());
}

#[test]
fn exact_real_scalar_extraction_does_not_erase_shape_or_domain() {
    let value = ValueLiteral::try_from(DynQuantity::new(7.0, real().dimension())).unwrap();
    assert_eq!(
        value.real_scalar_value(),
        Some(DynQuantity::new(7.0, real().dimension()))
    );
    for ty in [real().array(1).unwrap(), complex()] {
        let value = ValueLiteral::new(ty, [(7.0, 0.0)]).unwrap();
        assert_eq!(value.real_scalar_value(), None);
    }
    assert_eq!(
        ValueLiteral::from_real(complex(), 5.0)
            .unwrap()
            .component(0),
        Some((5.0, 0.0))
    );
}

#[test]
fn zero_is_canonical_compact_and_does_not_broadcast_nonzero_scalars() {
    let ty = complex().array(u32::MAX).unwrap();
    let zero = ValueLiteral::from_real(ty, -0.0).unwrap();
    assert!(matches!(zero.payload, Payload::Zero));
    assert_eq!(zero.component_count(), u32::MAX as usize);
    assert_eq!(zero.component(u32::MAX as usize - 1), Some((0.0, 0.0)));
    assert_eq!(zero.component(u32::MAX as usize), None);
    assert_eq!(zero.components().unwrap().len(), u32::MAX as usize);
    let ty = real().array(3).unwrap();
    let explicit = ValueLiteral::new(ty.clone(), [(-0.0, -0.0); 3]).unwrap();
    assert_eq!(explicit, ValueLiteral::from_real(ty.clone(), 0.0).unwrap());
    assert!(explicit.is_zero());
    for (r, i) in explicit.components().unwrap() {
        assert_eq!(r.to_bits(), 0.0_f64.to_bits());
        assert_eq!(i.to_bits(), 0.0_f64.to_bits());
    }
    let mixed = ValueLiteral::new(ty.clone(), [(-0.0, -0.0), (1.0, -0.0), (-0.0, 0.0)]).unwrap();
    assert_eq!(mixed.component(2).unwrap().0.to_bits(), 0.0_f64.to_bits());
    assert_eq!(mixed.component(1).unwrap().1.to_bits(), 0.0_f64.to_bits());
    assert_eq!(
        ValueLiteral::from_real(ty, 1.0),
        Err(InvalidValueLiteral::NonzeroShape)
    );
}

#[test]
fn component_cardinality_is_checked_without_unbounded_collection() {
    let ty = real().array(2).unwrap();
    for input in [vec![], vec![(1.0, 0.0)], vec![(1.0, 0.0); 3]] {
        assert_eq!(
            ValueLiteral::new(ty.clone(), input),
            Err(InvalidValueLiteral::ComponentCount)
        );
    }
    let mut reads = 0;
    let input = std::iter::from_fn(|| {
        reads += 1;
        Some((0.0, 0.0))
    });
    assert_eq!(
        ValueLiteral::new(ty, input),
        Err(InvalidValueLiteral::ComponentCount)
    );
    assert_eq!(reads, 3);
    let huge = real().array(u32::MAX).unwrap();
    assert_eq!(
        ValueLiteral::new(huge, [(1.0, 0.0)]),
        Err(InvalidValueLiteral::ComponentCount)
    );
}

#[test]
fn invalid_components_cannot_enter_the_value() {
    assert_eq!(
        ValueLiteral::new(real(), [(0.0, 1.0)]),
        Err(InvalidValueLiteral::ImaginaryInReal)
    );
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for pair in [(invalid, 0.0), (0.0, invalid)] {
            assert_eq!(
                ValueLiteral::new(complex(), [pair]),
                Err(InvalidValueLiteral::NonFinite)
            );
        }
        assert_eq!(
            ValueLiteral::from_real(real(), invalid),
            Err(InvalidValueLiteral::NonFinite)
        );
    }
}

#[test]
fn exact_integer_boundaries_arithmetic_and_explicit_conversion() {
    let int = |n| {
        ValueLiteral::from_integer(
            ValueType::scalar(ScalarDomain::Integer, DimExponents::DIMENSIONLESS)
                .expect("checked scalar type"),
            n,
        )
        .unwrap()
    };
    let n = int(9_007_199_254_740_993);
    assert_eq!(n.integer_scalar_value(), Some(9_007_199_254_740_993));
    assert_ne!(n, int(9_007_199_254_740_992));
    assert!(n.component(0).is_none());
    assert!(n.components().is_none());
    assert!(n.real_scalar_value().is_none());
    assert_eq!(n.checked_add(&int(1)).unwrap(), int(9_007_199_254_740_994));
    assert_eq!(int(-7).checked_quotient(&int(3)).unwrap(), int(-2));
    assert_eq!(int(-7).checked_remainder(&int(3)).unwrap(), int(-1));
    assert_eq!(int(7).checked_remainder(&int(-3)).unwrap(), int(1));
    assert_eq!(
        int(i64::MAX).checked_add(&int(1)),
        Err(InvalidValueLiteral::IntegerOverflow)
    );
    assert_eq!(
        int(i64::MIN).checked_sub(&int(1)),
        Err(InvalidValueLiteral::IntegerOverflow)
    );
    assert_eq!(
        int(i64::MAX).checked_mul(&int(2)),
        Err(InvalidValueLiteral::IntegerOverflow)
    );
    assert_eq!(
        int(i64::MIN).checked_neg(),
        Err(InvalidValueLiteral::IntegerOverflow)
    );
    assert_eq!(
        int(i64::MIN).checked_quotient(&int(-1)),
        Err(InvalidValueLiteral::IntegerOverflow)
    );
    assert_eq!(
        int(i64::MIN).checked_remainder(&int(-1)),
        Err(InvalidValueLiteral::IntegerOverflow)
    );
    assert_eq!(
        int(1).checked_quotient(&int(0)),
        Err(InvalidValueLiteral::ZeroDivisor)
    );
    assert_eq!(
        n.to_real().unwrap().real_scalar_value().unwrap().value(),
        9_007_199_254_740_992.0
    );
    assert_eq!(
        int(9_007_199_254_740_995)
            .to_real()
            .unwrap()
            .real_scalar_value()
            .unwrap()
            .value(),
        9_007_199_254_740_996.0
    );
    let real = |n| {
        ValueLiteral::from_real(
            ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
                .expect("checked scalar type"),
            n,
        )
        .unwrap()
    };
    assert_eq!(
        real(-9223372036854775808.0).to_integer().unwrap(),
        int(i64::MIN)
    );
    assert_eq!(
        real(9223372036854775808.0).to_integer(),
        Err(InvalidValueLiteral::IntegerConversion)
    );
    assert_eq!(
        real(1.5).to_integer(),
        Err(InvalidValueLiteral::IntegerConversion)
    );
    assert_eq!(real(-0.0).to_integer().unwrap(), int(0));
    assert!(real(1.0).checked_add(&int(1)).is_err());
    assert!(real(0.0).checked_add(&real(0.0)).is_err());
    assert!(real(0.0).checked_neg().is_err());
}

#[test]
fn integer_arrays_keep_order_cardinality_and_compact_zero() {
    let ty = ValueType::scalar(ScalarDomain::Integer, DimExponents::DIMENSIONLESS)
        .expect("checked scalar type")
        .array(3)
        .unwrap();
    let value =
        ValueLiteral::integer(ty.clone(), [i64::MIN, 9_007_199_254_740_993, i64::MAX]).unwrap();
    assert_eq!(
        value.integer_components().unwrap().collect::<Vec<_>>(),
        [i64::MIN, 9_007_199_254_740_993, i64::MAX]
    );
    assert!(value.integer_scalar_value().is_none());
    assert!(value.integer_component(3).is_none());
    assert!(ValueLiteral::integer(ty.clone(), [1, 2]).is_err());
    assert!(ValueLiteral::integer(ty.clone(), [1, 2, 3, 4]).is_err());
    assert!(ValueLiteral::from_integer(ty.clone(), 1).is_err());
    assert!(ValueLiteral::from_real(ty.clone(), 0.0).is_err());
    assert!(ValueLiteral::new(ty, [(0.0, 0.0); 3]).is_err());
    let huge = ValueType::scalar(ScalarDomain::Integer, DimExponents::DIMENSIONLESS)
        .expect("checked scalar type")
        .array(u32::MAX)
        .unwrap();
    let zero = ValueLiteral::from_integer(huge, 0).unwrap();
    assert!(zero.is_zero());
    assert!(matches!(zero.payload, super::Payload::Zero));
    assert_eq!(zero.integer_component(u32::MAX as usize - 1), Some(0));
    assert!(zero.component(0).is_none());
}

#[test]
fn nominal_counts_and_indexes_do_not_inherit_integer_coercions() {
    use crate::{Id, entity::kinds};
    let species = Id::<kinds::FiniteSpace>::new();
    let foreign = Id::<kinds::FiniteSpace>::new();
    let counts_type = ValueType::counts(species, 2).unwrap();
    let changes_type = ValueType::coordinates(
        crate::FiniteBasis::new(species, 2).unwrap(),
        crate::ScalarDomain::Integer,
        crate::DimExponents::DIMENSIONLESS,
    )
    .unwrap();
    let counts = ValueLiteral::integer(counts_type.clone(), [2, 9_007_199_254_740_993]).unwrap();
    let changes = ValueLiteral::integer(changes_type.clone(), [-1, 1]).unwrap();
    let updated = counts.checked_add(&changes).unwrap();
    assert_eq!(
        updated.integer_components().unwrap().collect::<Vec<_>>(),
        [1, 9_007_199_254_740_994]
    );
    assert_eq!(updated.value_type(), &counts_type);
    assert!(ValueLiteral::integer(counts_type.clone(), [-1, 0]).is_err());
    assert!(
        counts
            .checked_add(&ValueLiteral::integer(changes_type, [-3, 0]).unwrap())
            .is_err()
    );
    assert!(
        counts
            .checked_add(
                &ValueLiteral::integer(
                    ValueType::coordinates(
                        crate::FiniteBasis::new(foreign, 2).unwrap(),
                        crate::ScalarDomain::Integer,
                        crate::DimExponents::DIMENSIONLESS
                    )
                    .unwrap(),
                    [-1, 1]
                )
                .unwrap()
            )
            .is_err()
    );
    assert!(counts.checked_add(&counts).is_err());
    assert!(changes.checked_add(&counts).is_err());
    assert!(counts.checked_neg().is_err());
    assert!(counts.checked_sub(&counts).is_err());
    assert!(counts.checked_mul(&counts).is_err());
    assert!(counts_type.array(2).is_err());
    let set = Id::<kinds::IndexSet>::new();
    let index_type = ValueType::index(set, 3).unwrap();
    let index = ValueLiteral::from_integer(index_type.clone(), 2).unwrap();
    assert!(index.integer_scalar_value().is_none());
    assert_eq!(index.ordinal().unwrap().integer_scalar_value(), Some(2));
    assert!(index.to_real().is_err());
    assert!(index.checked_add(&index).is_err());
    assert!(index.checked_neg().is_err());
    assert!(ValueLiteral::from_integer(index_type.clone(), 3).is_err());
    assert!(ValueLiteral::from_integer(index_type, -1).is_err());
    assert_ne!(index.value_type(), &ValueType::index(Id::new(), 3).unwrap());
    assert!(ValueType::index(set, 0).is_err());
}

#[test]
fn booleans_never_coerce_to_numeric_values_or_zero() {
    let integer = ValueLiteral::from_integer(
        ValueType::scalar(ScalarDomain::Integer, DimExponents::DIMENSIONLESS)
            .expect("checked scalar type"),
        1,
    )
    .unwrap();
    let real = ValueLiteral::from_real(
        ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
            .expect("checked scalar type"),
        1.0,
    )
    .unwrap();
    assert_eq!(integer.as_bool(), None);
    assert_eq!(real.as_bool(), None);
    for truth in [false, true] {
        let value = ValueLiteral::boolean(truth);
        assert_eq!(value.value_type(), &ValueType::boolean());
        assert_eq!(value.as_bool(), Some(truth));
        assert!(!value.is_zero());
        assert_eq!(value.component_count(), 1);
        assert!(value.component(0).is_none());
        assert!(value.components().is_none());
        assert!(value.integer_component(0).is_none());
        assert!(value.integer_components().is_none());
        assert!(value.integer_scalar_value().is_none());
        assert!(value.real_scalar_value().is_none());
        assert!(value.to_real().is_err());
        assert!(value.to_integer().is_err());
        assert!(value.ordinal().is_err());
        assert!(value.checked_neg().is_err());
        for other in [&value, &integer, &real] {
            assert!(value.checked_add(other).is_err());
            assert!(value.checked_sub(other).is_err());
            assert!(value.checked_mul(other).is_err());
            assert!(value.checked_quotient(other).is_err());
            assert!(value.checked_remainder(other).is_err());
            assert!(other.checked_add(&value).is_err());
        }
        assert_ne!(value, integer);
        assert_ne!(value, real);
    }
    assert_ne!(ValueLiteral::boolean(false), ValueLiteral::boolean(true));
    for (number, real_number) in [(0, 0.0), (1, 1.0)] {
        assert!(ValueLiteral::from_integer(ValueType::boolean(), number).is_err());
        assert!(ValueLiteral::integer(ValueType::boolean(), [number]).is_err());
        assert!(ValueLiteral::from_real(ValueType::boolean(), real_number).is_err());
        assert!(ValueLiteral::new(ValueType::boolean(), [(real_number, 0.0)]).is_err());
    }
}

#[test]
fn enum_values_preserve_nominality_without_numeric_coercion() {
    let id = crate::Id::new();
    let ty = ValueType::enumeration(id, 2).unwrap();
    let first = ValueLiteral::enum_value(ty.clone(), 0).unwrap();
    let second = ValueLiteral::enum_value(ty.clone(), 1).unwrap();
    assert_eq!(first.enum_tag(), Some(0));
    assert!(!first.is_zero());
    assert!(!first.checked_equal(&second).unwrap());
    assert!(first.checked_equal(&first).unwrap());
    assert!(
        first
            .checked_equal(
                &ValueLiteral::enum_value(ValueType::enumeration(crate::Id::new(), 2).unwrap(), 0)
                    .unwrap()
            )
            .is_err()
    );
    assert!(ValueType::enumeration(id, 0).is_err());
    assert!(ValueLiteral::enum_value(ty.clone(), 2).is_err());
    assert!(ValueLiteral::enum_value(ty.clone(), u32::MAX).is_err());
    assert!(ty.clone().array(2).is_err());
    assert!(ValueLiteral::from_integer(ty.clone(), 0).is_err());
    assert!(ValueLiteral::from_real(ty.clone(), 0.).is_err());
    assert!(
        ty.with_dimension(DimExponents::from_integers([1, 0, 0, 0, 0, 0, 0]).unwrap())
            .is_err()
    );
    assert!(ValueType::scalar(ScalarDomain::Enum, DimExponents::DIMENSIONLESS).is_err());
    assert!(first.component(0).is_none());
    assert!(first.components().is_none());
    assert!(first.integer_component(0).is_none());
    assert!(first.integer_scalar_value().is_none());
    assert!(first.as_bool().is_none());
    assert!(first.checked_order(&second).is_err());
    assert!(first.checked_add(&second).is_err());
    assert!(first.checked_neg().is_err());
    assert!(first.to_real().is_err());
    assert!(first.to_integer().is_err());
}
