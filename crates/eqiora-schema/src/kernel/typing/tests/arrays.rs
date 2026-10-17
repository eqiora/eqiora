use super::*;
use eqiora_core::ScalarDomain;
use eqiora_core::ValueType;

#[test]
fn coordinate_partials_preserve_numeric_component_types_and_exact_units() {
    let mass = DimExponents::from_integers([1, 0, 0, 0, 0, 0, 0]).unwrap();
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let density = DimExponents::from_integers([1, -1, 0, 0, 0, 0, 0]).unwrap();
    let selected = ExpressionType::scalar(length, Some(volume("body")));
    for domain in [ScalarDomain::Real, ScalarDomain::Complex] {
        for value in [
            ValueType::scalar(domain, mass).unwrap(),
            ValueType::shaped(
                domain,
                mass,
                ValueShape::new([2]).unwrap(),
                ValueFrame::SpatialCartesian,
            )
            .unwrap(),
            ValueType::shaped(
                domain,
                mass,
                ValueShape::new([2, 2]).unwrap(),
                ValueFrame::SpatialCartesian,
            )
            .unwrap(),
        ] {
            for value in [value.clone(), value.array(3).unwrap()] {
                let ty = ExpressionType::new(value.clone(), Some(volume("body")));
                let derivative = ty.coordinate_partial(&selected).unwrap();
                assert_eq!(
                    derivative.value_type,
                    value.with_dimension(density).unwrap()
                );
                assert_eq!(derivative.support, ty.support);
                assert_eq!(derivative, ty.partial(&selected).unwrap());
                assert!(
                    ty.coordinate_partial(&ExpressionType::scalar(length, Some(volume("foreign"))))
                        .is_err()
                );
                let complex_selector = ExpressionType::new(
                    selected
                        .value_type
                        .clone()
                        .with_scalar_domain(ScalarDomain::Complex)
                        .unwrap(),
                    selected.support.clone(),
                );
                assert!(ty.coordinate_partial(&complex_selector).is_err());
                let shaped_selector = ExpressionType::new(
                    selected.value_type.clone().array(2).unwrap(),
                    selected.support.clone(),
                );
                assert!(ty.coordinate_partial(&shaped_selector).is_err());
            }
        }
    }
}

#[test]
fn arrays_of_vectors_are_not_spatial_tensors_with_the_same_extents() {
    let vector = ValueType::shaped(
        ScalarDomain::Complex,
        DimExponents::DIMENSIONLESS,
        ValueShape::new([2]).unwrap(),
        ValueFrame::SpatialCartesian,
    )
    .unwrap();
    let array = ExpressionType::new(vector.array(2).unwrap(), Some(volume("body")));
    let tensor = ExpressionType::new(
        ValueType::shaped(
            ScalarDomain::Complex,
            DimExponents::DIMENSIONLESS,
            ValueShape::new([2, 2]).unwrap(),
            ValueFrame::SpatialCartesian,
        )
        .unwrap(),
        Some(volume("body")),
    );
    assert_eq!(array.shape(), tensor.shape());
    assert_eq!(array.frame(), tensor.frame());
    assert_ne!(array.value_type, tensor.value_type);
    assert_eq!(array.value_type.array_rank(), 1);
    assert_eq!(tensor.value_type.array_rank(), 0);
    assert!(additive(&array, &tensor).is_err());
    let array_gradient = gradient(&array).unwrap();
    assert_eq!(array_gradient.value_type.array_rank(), 1);
    assert_eq!(array_gradient.shape(), &ValueShape::new([2, 2, 2]).unwrap());
    let array_divergence = divergence(&array).unwrap();
    assert_eq!(array_divergence.value_type.array_rank(), 1);
    assert_eq!(array_divergence.frame(), ValueFrame::Invariant);
    assert_eq!(array_divergence.shape(), &ValueShape::new([2]).unwrap());
    assert!(symmetric_part(&array).is_err());
    assert!(symmetric_part(&tensor).is_ok());

    let scalar = ExpressionType::scalar(DimExponents::DIMENSIONLESS, None);
    for result in [
        additive(&array, &array).unwrap(),
        multiply(&scalar, &array).unwrap(),
        multiply(&array, &scalar).unwrap(),
        divide(&array, &scalar).unwrap(),
    ] {
        assert_eq!(result.value_type, array.value_type);
    }
    let derivative = time_derivative(&array, std::num::NonZeroU32::MIN).unwrap();
    assert_eq!(derivative.value_type.array_rank(), 1);
    assert_eq!(derivative.shape(), array.shape());
    assert_eq!(derivative.value_type.scalar_domain(), ScalarDomain::Complex);
    assert_ne!(derivative.dimension(), array.dimension());
    let boundary = SpatialSupport::Boundary {
        domain: "wall",
        parent: "body",
        dimensions: 2,
    };
    let traced = trace(&array, Some(&boundary)).unwrap();
    assert_eq!(traced.value_type, array.value_type);
    assert_eq!(traced.support, Some(boundary.clone()));
    let contracted = normal(&array, Some(&boundary)).unwrap();
    assert_eq!(contracted.value_type.array_rank(), 1);
    assert_eq!(contracted.frame(), ValueFrame::Invariant);
    assert_eq!(contracted.shape(), &ValueShape::new([2]).unwrap());
}

#[test]
fn nested_channel_arrays_preserve_axis_order_and_reject_zero_extents() {
    let scalar = ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
        .expect("checked scalar type");
    assert!(scalar.clone().array(0).is_err());
    let nested = scalar.array(2).unwrap().array(3).unwrap();
    assert_eq!(nested.array_rank(), 2);
    assert_eq!(nested.shape(), &ValueShape::new([3, 2]).unwrap());
    assert_eq!(nested.frame(), ValueFrame::Invariant);
}

#[test]
fn explicit_channels_promote_domain_merge_static_support_and_retain_element_roles() {
    let scalar = ExpressionType::scalar(DimExponents::DIMENSIONLESS, None);
    let supported = ExpressionType::scalar(DimExponents::DIMENSIONLESS, Some(volume("body")));
    let channel = ExpressionType::array(&[scalar.clone(), supported.clone()]).unwrap();
    assert_eq!(channel.support, supported.support);
    assert_eq!(
        ExpressionType::index(channel.clone(), 1).unwrap(),
        supported
    );
    assert_eq!(
        ExpressionType::index(channel, 2),
        Err(TypeViolation::IndexOutOfBounds)
    );
    assert!(ExpressionType::<&str>::array(&[]).is_err());
    assert!(
        ExpressionType::array(&[
            supported.clone(),
            ExpressionType::scalar(DimExponents::DIMENSIONLESS, Some(volume("other")))
        ])
        .is_err()
    );
    let complex_scalar = ExpressionType::complex(scalar.clone(), supported.clone()).unwrap();
    let channel = ExpressionType::array(&[supported, complex_scalar.clone()]).unwrap();
    assert_eq!(ExpressionType::index(channel, 0).unwrap(), complex_scalar);
    assert!(ExpressionType::complex(complex_scalar, scalar.clone()).is_err());
    assert!(
        ExpressionType::complex(
            scalar.clone(),
            ExpressionType::scalar(
                DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap(),
                None
            )
        )
        .is_err()
    );
    assert!(ExpressionType::index(scalar, 0).is_err());
}

#[test]
fn indexing_removes_only_one_outer_channel_axis() {
    let element = ValueType::shaped(
        ScalarDomain::Real,
        DimExponents::DIMENSIONLESS,
        ValueShape::new([2, 2]).unwrap(),
        ValueFrame::SpatialCartesian,
    )
    .unwrap();
    let value = ExpressionType::new(element.clone(), Some(volume("body")));
    assert_eq!(
        ExpressionType::index(value.clone(), 0),
        Err(TypeViolation::IndexRequiresArray)
    );
    let nested =
        ExpressionType::array(&[ExpressionType::array(&[value.clone(), value.clone()]).unwrap()])
            .unwrap();
    let result = ExpressionType::index(ExpressionType::index(nested, 0).unwrap(), 1).unwrap();
    assert_eq!(result, value);
    assert!(
        ExpressionType::array(&[
            value,
            ExpressionType::scalar(DimExponents::DIMENSIONLESS, None)
        ])
        .is_err()
    );
}

#[test]
fn constructed_nodes_flow_through_dag_typing_and_operand_validation() {
    use super::super::super::ExprDagBuilder;
    let mut builder = ExprDagBuilder::new();
    assert!(builder.array([]).is_err());
    let one = builder
        .constant(eqiora_core::DynQuantity::new(
            1.0,
            DimExponents::DIMENSIONLESS,
        ))
        .unwrap();
    let pair = builder.complex(one, one).unwrap();
    let channels = builder.array([pair, pair]).unwrap();
    let selected = builder.index(channels, 1).unwrap();
    let dag = builder.finish([selected]).unwrap();
    let typed = TypedResidual::<&str>::infer(
        dag,
        None,
        |_| None,
        RootContract::ComponentwiseResidual,
        |_| Err::<ExpressionType<&str>, _>(()),
    )
    .unwrap();
    assert_eq!(
        typed
            .node_type(selected)
            .unwrap()
            .value_type
            .scalar_domain(),
        ScalarDomain::Complex
    );
    assert!(typed.node_type(selected).unwrap().shape().is_scalar());
    let mut empty = ExprDagBuilder::new();
    assert!(empty.array([one]).is_err());
    assert!(empty.index(one, 0).is_err());
    assert!(empty.complex(one, one).is_err());
}

#[test]
fn spatial_rules_preserve_nested_channels_and_only_contract_spatial_axes() {
    for domain in [ScalarDomain::Real, ScalarDomain::Complex] {
        for dimensions in [1, 2, 3] {
            let support = SpatialSupport::Volume {
                domain: "body",
                dimensions,
            };
            let boundary = SpatialSupport::Boundary {
                domain: "wall",
                parent: "body",
                dimensions,
            };
            let channels = ExpressionType::new(
                ValueType::scalar(domain, DimExponents::DIMENSIONLESS)
                    .unwrap()
                    .array(2)
                    .unwrap()
                    .array(3)
                    .unwrap(),
                Some(support.clone()),
            );
            let grad = gradient(&channels).unwrap();
            assert_eq!(
                grad.shape(),
                &ValueShape::new([3, 2, dimensions as u32]).unwrap()
            );
            assert_eq!(grad.value_type.array_rank(), 2);
            assert_eq!(grad.value_type.scalar_domain(), domain);
            assert_eq!(grad.frame(), ValueFrame::SpatialCartesian);
            assert_eq!(grad.support, Some(support));
            let inverse_length = DimExponents::from_integers([0, -1, 0, 0, 0, 0, 0]).unwrap();
            assert_eq!(grad.dimension(), inverse_length);
            let div = divergence(&grad).unwrap();
            assert_eq!(
                div.value_type,
                channels
                    .value_type
                    .clone()
                    .with_dimension(DimExponents::from_integers([0, -2, 0, 0, 0, 0, 0]).unwrap())
                    .unwrap()
            );
            let normal_trace = normal(&grad, Some(&boundary)).unwrap();
            assert_eq!(
                normal_trace.value_type,
                channels
                    .value_type
                    .clone()
                    .with_dimension(inverse_length)
                    .unwrap()
            );
            assert_eq!(normal_trace.support, Some(boundary.clone()));
            let isotropic = isotropic_lift(&channels).unwrap();
            assert_eq!(
                isotropic.shape(),
                &ValueShape::new([3, 2, dimensions as u32, dimensions as u32]).unwrap()
            );
            assert_eq!(isotropic.value_type.array_rank(), 2);
            assert_eq!(symmetric_part(&isotropic).unwrap(), isotropic);
            // Equal-sized channel axes grant no spatial vector/tensor meaning.
            assert!(divergence(&channels).is_err());
            assert!(normal(&channels, Some(&boundary)).is_err());
            assert!(symmetric_part(&channels).is_err());
            let foreign = SpatialSupport::Boundary {
                domain: "foreign",
                parent: "other",
                dimensions,
            };
            assert!(normal(&grad, Some(&foreign)).is_err());
        }
    }
}
