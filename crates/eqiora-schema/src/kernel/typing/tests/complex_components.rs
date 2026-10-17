use super::*;
use eqiora_core::{FiniteBasis, ScalarDomain, ValueType};

#[test]
fn complex_components_preserve_axes_frames_units_and_exact_bases() {
    let source = FiniteBasis::new(Id::new(), 2).unwrap();
    let target = FiniteBasis::new(Id::new(), 3).unwrap();
    let unit = DimExponents::from_integers([1, 0, -1, 0, 0, 0, 0]).unwrap();
    let types = |domain| {
        let scalar = ValueType::scalar(domain, unit).unwrap();
        let vector = ValueType::shaped(
            domain,
            unit,
            ValueShape::new([2]).unwrap(),
            ValueFrame::SpatialCartesian,
        )
        .unwrap();
        vec![
            scalar.clone(),
            scalar.array(2).unwrap().array(3).unwrap(),
            vector.clone(),
            vector.array(4).unwrap(),
            ValueType::shaped(
                domain,
                unit,
                ValueShape::new([2, 3]).unwrap(),
                ValueFrame::SpatialCartesian,
            )
            .unwrap(),
            ValueType::coordinates(source.dual(), domain, unit).unwrap(),
            ValueType::linear_map(source, target, domain, unit).unwrap(),
        ]
    };
    for (real, complex) in types(ScalarDomain::Real)
        .into_iter()
        .zip(types(ScalarDomain::Complex))
    {
        let real = ExpressionType::new(real, Some(volume("body")));
        let complex = ExpressionType::new(complex, real.support.clone());
        assert_eq!(real.clone().complex(real.clone()).unwrap(), complex);
        assert_eq!(
            unary_math(UnaryMathFunction::Conj, &complex).unwrap(),
            complex
        );
        for operation in [UnaryMathFunction::Real, UnaryMathFunction::Imag] {
            assert_eq!(unary_math(operation, &complex).unwrap(), real);
            assert_eq!(unary_math(operation, &real).unwrap(), real);
        }
        assert_eq!(unary_math(UnaryMathFunction::Conj, &real).unwrap(), real);
        assert!(complex.clone().complex(real.clone()).is_err());
        let mut foreign_support = real.clone();
        foreign_support.support = Some(volume("foreign"));
        assert!(real.clone().complex(foreign_support).is_err());
        let mut wrong_unit = real.clone();
        wrong_unit.value_type = wrong_unit
            .value_type
            .with_dimension(DimExponents::DIMENSIONLESS)
            .unwrap();
        assert!(real.complex(wrong_unit).is_err());
    }
}

#[test]
fn complex_construction_never_reinterprets_equal_width_components() {
    let unit = DimExponents::DIMENSIONLESS;
    let real = ScalarDomain::Real;
    let left = FiniteBasis::new(Id::new(), 2).unwrap();
    let foreign = FiniteBasis::new(Id::new(), 2).unwrap();
    let ty = |value| ExpressionType::<()>::new(value, None);
    let coordinates = ty(ValueType::coordinates(left, real, unit).unwrap());
    for other in [
        ValueType::coordinates(foreign, real, unit).unwrap(),
        ValueType::coordinates(left.dual(), real, unit).unwrap(),
        ValueType::scalar(real, unit).unwrap().array(2).unwrap(),
        ValueType::shaped(
            real,
            unit,
            ValueShape::new([2]).unwrap(),
            ValueFrame::SpatialCartesian,
        )
        .unwrap(),
    ] {
        assert!(coordinates.clone().complex(ty(other)).is_err());
    }
    let map = ty(ValueType::linear_map(left, foreign, real, unit).unwrap());
    assert!(
        map.complex(ty(ValueType::linear_map(foreign, left, real, unit).unwrap()))
            .is_err()
    );
    for value in [
        ValueType::boolean(),
        ValueType::index(Id::new(), 2).unwrap(),
        ValueType::counts(Id::new(), 2).unwrap(),
    ] {
        assert!(
            value
                .clone()
                .with_scalar_domain(ScalarDomain::Complex)
                .is_none()
        );
        for operation in [
            UnaryMathFunction::Conj,
            UnaryMathFunction::Real,
            UnaryMathFunction::Imag,
        ] {
            assert!(unary_math(operation, &ty(value.clone())).is_err());
        }
    }
}
