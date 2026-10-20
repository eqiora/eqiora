use super::*;
use eqiora_core::{FiniteBasis, Id, ScalarDomain, ValueShape, ValueType};

fn scalar(domain: ScalarDomain) -> ValueType {
    ValueType::scalar(domain, DimExponents::DIMENSIONLESS).unwrap()
}

fn identity(value: &ValueType) -> PureOperatorDefinition {
    let class = PureValueClass::from_value_type(value.clone()).unwrap();
    let mut builder = CalculusBuilder::new([class.clone()], class).unwrap();
    let root = builder
        .push(CalculusNode::FormalComponent {
            formal: 0,
            axes: (0..value.shape().rank())
                .map(|axis| ComponentIndex::Result(axis as u16))
                .collect(),
        })
        .unwrap();
    builder.finish(root).unwrap()
}

#[test]
fn exact_contracts_preserve_axis_roles_frames_and_nominal_bases() {
    let basis = FiniteBasis::new(Id::new(), 2).unwrap();
    let other = FiniteBasis::new(Id::new(), 2).unwrap();
    let domain = ScalarDomain::Complex;
    let unit = DimExponents::DIMENSIONLESS;
    let vector = ValueType::shaped(
        domain,
        unit,
        ValueShape::new([2]).unwrap(),
        ValueFrame::SpatialCartesian,
    )
    .unwrap();
    let values = [
        scalar(domain).array(2).unwrap(),
        vector.clone(),
        scalar(domain).array(2).unwrap().array(2).unwrap(),
        vector.array(2).unwrap(),
        ValueType::coordinates(basis, domain, unit).unwrap(),
        ValueType::coordinates(other, domain, unit).unwrap(),
        ValueType::coordinates(basis.dual(), domain, unit).unwrap(),
        ValueType::linear_map(basis, other, domain, unit).unwrap(),
        ValueType::linear_map(other, basis, domain, unit).unwrap(),
        ValueType::coordinates(FiniteBasis::product(basis, other).unwrap(), domain, unit).unwrap(),
        ValueType::coordinates(FiniteBasis::product(other, basis).unwrap(), domain, unit).unwrap(),
    ];
    let mut digests = std::collections::BTreeSet::new();
    for value in &values {
        let definition = identity(value);
        assert!(digests.insert(definition.digest().to_string()));
        let argument = ExpressionType::<()>::new(value.clone(), None);
        assert_eq!(
            definition
                .instantiate(std::slice::from_ref(&argument))
                .unwrap()
                .result_type(),
            &argument
        );
        for other in &values {
            if other != value {
                assert_eq!(
                    definition
                        .instantiate(&[ExpressionType::<()>::new(other.clone(), None)])
                        .unwrap_err(),
                    PureOperatorError::FormalTypeMismatch,
                );
            }
        }
    }
}

#[test]
fn rectangular_component_maps_use_each_ordered_axis() {
    let input = scalar(ScalarDomain::Real)
        .array(3)
        .unwrap()
        .array(2)
        .unwrap();
    let output = scalar(ScalarDomain::Real)
        .array(2)
        .unwrap()
        .array(3)
        .unwrap();
    let mut builder = CalculusBuilder::new(
        [PureValueClass::from_value_type(input.clone()).unwrap()],
        PureValueClass::from_value_type(output.clone()).unwrap(),
    )
    .unwrap();
    assert_eq!(
        builder.push(CalculusNode::FormalComponent {
            formal: 0,
            axes: Box::new([ComponentIndex::Result(0), ComponentIndex::Result(1)]),
        }),
        Err(PureOperatorError::FormalTypeMismatch)
    );
    assert_eq!(
        builder.push(CalculusNode::FormalComponent {
            formal: 0,
            axes: Box::new([ComponentIndex::Fixed(2), ComponentIndex::Fixed(0)]),
        }),
        Err(PureOperatorError::FormalTypeMismatch)
    );
    let root = builder
        .push(CalculusNode::FormalComponent {
            formal: 0,
            axes: Box::new([ComponentIndex::Result(1), ComponentIndex::Result(0)]),
        })
        .unwrap();
    let definition = builder.finish(root).unwrap();
    assert_eq!(
        definition
            .instantiate(&[ExpressionType::<()>::new(input, None)])
            .unwrap()
            .result_type()
            .value_type,
        output
    );
}

#[test]
fn exact_component_math_retains_shape_and_changes_only_declared_domain() {
    let input = scalar(ScalarDomain::Complex).array(3).unwrap();
    let output = scalar(ScalarDomain::Real).array(3).unwrap();
    let mut builder = CalculusBuilder::new(
        [PureValueClass::from_value_type(input.clone()).unwrap()],
        PureValueClass::from_value_type(output.clone()).unwrap(),
    )
    .unwrap();
    let value = builder
        .push(CalculusNode::FormalComponent {
            formal: 0,
            axes: Box::new([ComponentIndex::Result(0)]),
        })
        .unwrap();
    let real = builder
        .push(CalculusNode::UnaryMath(
            crate::kernel::UnaryMathFunction::Real,
            value,
        ))
        .unwrap();
    let definition = builder.finish(real).unwrap();
    assert_eq!(
        definition
            .instantiate(&[ExpressionType::<()>::new(input, None)])
            .unwrap()
            .result_type()
            .value_type,
        output
    );
}

#[test]
fn fully_constrained_templates_have_the_exact_type_identity() {
    for domain in [ScalarDomain::Real, ScalarDomain::Complex] {
        let exact = PureValueClass::from_value_type(scalar(domain)).unwrap();
        let template = PureValueClass::invariant_scalar()
            .with_dimension(DimExponents::DIMENSIONLESS)
            .with_scalar_domain(domain)
            .unwrap();
        assert_eq!(template, exact);
        let value = ValueType::shaped(
            domain,
            DimExponents::DIMENSIONLESS,
            ValueShape::new([3, 3]).unwrap(),
            ValueFrame::SpatialCartesian,
        )
        .unwrap();
        let template = PureValueClass::spatial_tensor(2)
            .unwrap()
            .with_spatial_extent(3)
            .unwrap()
            .with_scalar_domain(domain)
            .unwrap()
            .with_dimension(DimExponents::DIMENSIONLESS);
        assert_eq!(template, PureValueClass::from_value_type(value).unwrap());
    }
}

#[test]
fn spatial_extent_is_independent_of_outer_array_extent() {
    let vector = ValueType::shaped(
        ScalarDomain::Real,
        DimExponents::DIMENSIONLESS,
        ValueShape::new([2]).unwrap(),
        ValueFrame::SpatialCartesian,
    )
    .unwrap();
    let value = vector.array(7).unwrap();
    let definition = identity(&value);
    for dimensions in [2, 3] {
        let argument = ExpressionType::new(
            value.clone(),
            Some(SpatialSupport::Volume {
                domain: "body",
                dimensions,
            }),
        );
        let result = definition.instantiate(&[argument]);
        assert_eq!(result.is_ok(), dimensions == 2);
    }
}

#[test]
fn fixed_delta_coordinates_still_require_the_instantiated_spatial_extent() {
    let mut builder = CalculusBuilder::new(
        [PureValueClass::spatial_tensor(1).unwrap()],
        PureValueClass::invariant_scalar(),
    )
    .unwrap();
    let root = builder
        .push(CalculusNode::KroneckerDelta(
            ComponentIndex::Fixed(2),
            ComponentIndex::Fixed(2),
        ))
        .unwrap();
    let definition = builder.finish(root).unwrap();
    for extent in [2, 3] {
        let input = ValueType::shaped(
            ScalarDomain::Real,
            DimExponents::DIMENSIONLESS,
            ValueShape::new([extent]).unwrap(),
            ValueFrame::SpatialCartesian,
        )
        .unwrap();
        assert_eq!(
            definition
                .instantiate(&[ExpressionType::<()>::new(input, None)])
                .is_ok(),
            extent == 3
        );
    }
}

#[test]
fn spatial_constraints_reject_unrepresentable_shapes_before_normalization() {
    // (2^32 - 1)^3 exceeds both 32-bit and 64-bit component indexing. The
    // rejection must not depend on when dimension and domain are specified.
    let template = PureValueClass::spatial_tensor(3).unwrap();
    for class in [
        template.clone(),
        template.clone().with_dimension(DimExponents::DIMENSIONLESS),
        template
            .with_dimension(DimExponents::DIMENSIONLESS)
            .with_scalar_domain(ScalarDomain::Real)
            .unwrap(),
    ] {
        assert_eq!(
            class.with_spatial_extent(u32::MAX),
            Err(PureOperatorError::InvalidResultRule)
        );
    }
}
