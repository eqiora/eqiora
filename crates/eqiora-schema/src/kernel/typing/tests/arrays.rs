use super::*;
use eqiora_core::ScalarDomain;
use eqiora_core::ValueType;

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
    assert!(gradient(&array).is_err());
    assert!(divergence(&array).is_err());
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
    assert!(normal(&array, Some(&boundary)).is_err());
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
    let typed =
        TypedResidual::<&str>::infer(dag, None, RootContract::ComponentwiseResidual, |_| {
            Err::<ExpressionType<&str>, _>(())
        })
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
