use super::*;
use eqiora_core::Id;
use eqiora_core::entity::kinds;

fn volume(name: &'static str) -> SpatialSupport<&'static str> {
    SpatialSupport::Volume {
        domain: name,
        dimensions: 2,
    }
}

#[test]
fn complex_domain_survives_arithmetic_and_spatial_type_inference() {
    use eqiora_core::ScalarDomain;
    use eqiora_core::ValueType;
    let real = ExpressionType::scalar(DimExponents::DIMENSIONLESS, Some(volume("body")));
    let complex = ExpressionType::new(
        ValueType::scalar(ScalarDomain::Complex, DimExponents::DIMENSIONLESS)
            .expect("checked scalar type"),
        real.support.clone(),
    );
    for result in [
        additive(&real, &complex),
        additive(&complex, &real),
        multiply(&real, &complex),
        divide(&real, &complex),
        power(&complex, 2),
        time_derivative(&complex),
        gradient(&complex),
        isotropic_lift(&complex),
        unary_math(UnaryMathFunction::Sqrt, &complex),
    ] {
        assert_eq!(
            result.unwrap().value_type.scalar_domain(),
            ScalarDomain::Complex
        );
    }
    let gradient = gradient(&complex).unwrap();
    assert_eq!(
        divergence(&gradient).unwrap().value_type.scalar_domain(),
        ScalarDomain::Complex
    );
    let tensor = isotropic_lift(&complex).unwrap();
    assert_eq!(symmetric_part(&tensor).unwrap(), tensor);
    assert!(residual(&complex, complex.support.as_ref()).is_ok());
    assert!(matches!(
        scalar_root(&complex, complex.support.as_ref()),
        Err(TypeViolation::RootRequiresRealScalar)
    ));
    assert!(scalar_root(&real, real.support.as_ref()).is_ok());
}

#[test]
fn pure_definition_preserves_the_complex_argument_domain() {
    use eqiora_core::ScalarDomain;
    use eqiora_core::ValueType;
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
    let definition =
        crate::kernel::pure_operator::PureOperatorDefinition::symmetric_part().unwrap();
    assert_eq!(
        definition
            .instantiate(std::slice::from_ref(&tensor))
            .unwrap()
            .result_type(),
        &tensor,
    );
}

#[test]
fn spatial_rules_are_identity_parametric_and_shape_aware() {
    let scalar = ExpressionType::scalar(DimExponents::DIMENSIONLESS, Some(volume("left")));
    let gradient = gradient(&scalar).expect("gradient");
    assert_eq!(gradient.shape().extents()[0].get(), 2);
    assert!(divergence(&scalar).is_err());

    let other = ExpressionType::scalar(DimExponents::DIMENSIONLESS, Some(volume("right")));
    assert!(matches!(
        additive(&scalar, &other),
        Err(TypeViolation::IncompatibleSupport { .. })
    ));
}

#[test]
fn tensor_structure_comes_only_from_exact_spatial_types() {
    let dimension =
        DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).expect("bounded dimension");
    let tensor = ExpressionType::shaped(
        dimension,
        ValueShape::new([2, 2]).unwrap(),
        ValueFrame::SpatialCartesian,
        Some(volume("body")),
    )
    .unwrap();
    assert_eq!(symmetric_part(&tensor).unwrap(), tensor);

    for shape in [
        ValueShape::new([2]).unwrap(),
        ValueShape::new([2, 3]).unwrap(),
    ] {
        let invalid = ExpressionType::shaped(
            dimension,
            shape,
            ValueFrame::SpatialCartesian,
            Some(volume("body")),
        )
        .unwrap();
        assert!(matches!(
            symmetric_part(&invalid),
            Err(TypeViolation::SymmetricPartRequiresSquareSpatialTensor)
        ));
    }
    let wrong_frame = ExpressionType::shaped(
        dimension,
        ValueShape::new([2, 2]).unwrap(),
        ValueFrame::Invariant,
        Some(volume("body")),
    )
    .unwrap();
    assert!(symmetric_part(&wrong_frame).is_err());

    let scalar = ExpressionType::scalar(dimension, Some(volume("body")));
    let isotropic = isotropic_lift(&scalar).unwrap();
    assert_eq!(isotropic.dimension(), dimension);
    assert_eq!(isotropic.shape(), &ValueShape::new([2, 2]).unwrap());
    assert_eq!(isotropic.frame(), ValueFrame::SpatialCartesian);
    assert_eq!(isotropic.support, scalar.support);

    let global = ExpressionType::<&str>::scalar(dimension, None);
    assert!(matches!(
        isotropic_lift(&global),
        Err(TypeViolation::IsotropicLiftRequiresVolume)
    ));
    assert!(matches!(
        isotropic_lift(&tensor),
        Err(TypeViolation::IsotropicLiftRequiresInvariantScalar)
    ));
}

#[test]
fn tensor_structure_rejects_boundary_support() {
    let boundary = SpatialSupport::Boundary {
        domain: "wall",
        parent: "body",
        dimensions: 2,
    };
    let tensor = ExpressionType::shaped(
        DimExponents::DIMENSIONLESS,
        ValueShape::new([2, 2]).unwrap(),
        ValueFrame::SpatialCartesian,
        Some(boundary.clone()),
    )
    .unwrap();
    let scalar = ExpressionType::scalar(DimExponents::DIMENSIONLESS, Some(boundary));
    assert!(matches!(
        symmetric_part(&tensor),
        Err(TypeViolation::SymmetricPartRequiresVolume)
    ));
    assert!(matches!(
        isotropic_lift(&scalar),
        Err(TypeViolation::IsotropicLiftRequiresVolume)
    ));
}

#[test]
fn typed_residual_separates_componentwise_relations_from_scalar_activations() {
    let port = Id::<kinds::Port>::new();
    let mut builder = super::super::ExprDagBuilder::new();
    let root = builder.symbol(SymbolRef::PortTrace(port)).unwrap();
    let expression = builder.finish([root]).unwrap();
    let vector = ExpressionType::shaped(
        DimExponents::DIMENSIONLESS,
        ValueShape::new([2]).unwrap(),
        ValueFrame::SpatialCartesian,
        None::<SpatialSupport<RawTestId>>,
    )
    .unwrap();

    let typed = TypedResidual::infer(
        expression.clone(),
        None,
        RootContract::ComponentwiseResidual,
        |_| Ok::<_, ()>(vector.clone()),
    )
    .unwrap();
    assert_eq!(typed.node_type(root).unwrap().shape().extents()[0].get(), 2);

    let errors = TypedResidual::infer(expression, None, RootContract::ScalarActivation, |_| {
        Ok::<_, ()>(vector.clone())
    })
    .unwrap_err();
    assert!(matches!(
        errors.as_slice(),
        [TypedResidualError::Type {
            error: TypeViolation::RootRequiresRealScalar,
            ..
        }]
    ));
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RawTestId;

#[test]
fn coordinate_and_boundary_rules_use_relation_support() {
    assert!(matches!(
        coordinate::<&str>(0, None),
        Err(TypeViolation::CoordinateRequiresSpatialScope)
    ));
    assert!(matches!(
        coordinate(2, Some(&volume("body"))),
        Err(TypeViolation::CoordinateAxisOutOfRange { .. })
    ));

    let boundary = SpatialSupport::Boundary {
        domain: "wall",
        parent: "body",
        dimensions: 2,
    };
    let body = ExpressionType::scalar(DimExponents::DIMENSIONLESS, Some(volume("body")));
    assert_eq!(
        trace(&body, Some(&boundary))
            .expect("trace")
            .support
            .as_ref()
            .map(SpatialSupport::domain),
        Some(&"wall")
    );
    assert!(normal(&body, Some(&boundary)).is_err());

    let boundary_coordinate = coordinate(0, Some(&boundary)).unwrap();
    let boundary_scalar =
        ExpressionType::scalar(DimExponents::DIMENSIONLESS, Some(boundary.clone()));
    assert!(matches!(
        additive(&body, &boundary_scalar),
        Err(TypeViolation::IncompatibleSupport { .. })
    ));
    let body_vector = ExpressionType::shaped(
        DimExponents::DIMENSIONLESS,
        ValueShape::new([2]).unwrap(),
        ValueFrame::SpatialCartesian,
        Some(volume("body")),
    )
    .unwrap();
    let restricted_flux = multiply(&boundary_coordinate, &body_vector).unwrap();
    assert_eq!(
        restricted_flux.support.as_ref().map(SpatialSupport::domain),
        Some(&"wall")
    );
    assert!(normal(&restricted_flux, Some(&boundary)).is_ok());
}

#[test]
fn generic_pure_application_derives_shape_support_and_dimension_from_its_table() {
    let left = Id::<kinds::Field>::new();
    let right = Id::<kinds::Field>::new();
    let definition = crate::kernel::pure_operator::PureOperatorDefinition::dyadic_product()
        .expect("standard definition");
    let mut builder = super::super::ExprDagBuilder::new();
    let left_value = builder.symbol(SymbolRef::Field(left)).unwrap();
    let right_value = builder.symbol(SymbolRef::Field(right)).unwrap();
    let product = builder
        .pure_operator(&definition, [left_value, right_value])
        .unwrap();
    let expression = builder.finish([product]).unwrap();
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension");
    let inverse_time =
        DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).expect("bounded dimension");

    let typed = TypedResidual::infer(
        expression,
        Some(volume("body")),
        RootContract::ComponentwiseResidual,
        |symbol| {
            let dimension = match symbol {
                SymbolRef::Field(field) if field == left => length,
                SymbolRef::Field(field) if field == right => inverse_time,
                _ => unreachable!(),
            };
            Ok::<_, ()>(
                ExpressionType::shaped(
                    dimension,
                    ValueShape::new([2]).unwrap(),
                    ValueFrame::SpatialCartesian,
                    Some(volume("body")),
                )
                .unwrap(),
            )
        },
    )
    .unwrap();

    let result = typed.node_type(product).unwrap();
    assert_eq!(result.shape(), &ValueShape::new([2, 2]).unwrap());
    assert_eq!(result.support, Some(volume("body")));
    assert_eq!(
        result.dimension(),
        DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).expect("bounded dimension")
    );
}

#[test]
fn generic_pure_application_rejects_argument_type_and_support_mismatches() {
    let left = Id::<kinds::Field>::new();
    let right = Id::<kinds::Field>::new();
    let definition = crate::kernel::pure_operator::PureOperatorDefinition::dyadic_product()
        .expect("standard definition");
    let expression = {
        let mut builder = super::super::ExprDagBuilder::new();
        let left = builder.symbol(SymbolRef::Field(left)).unwrap();
        let right = builder.symbol(SymbolRef::Field(right)).unwrap();
        let product = builder.pure_operator(&definition, [left, right]).unwrap();
        builder.finish([product]).unwrap()
    };
    let vector = |domain| {
        ExpressionType::shaped(
            DimExponents::DIMENSIONLESS,
            ValueShape::new([2]).unwrap(),
            ValueFrame::SpatialCartesian,
            Some(volume(domain)),
        )
        .unwrap()
    };

    let support_errors = TypedResidual::infer(
        expression.clone(),
        Some(volume("body")),
        RootContract::ComponentwiseResidual,
        |symbol| {
            Ok::<_, ()>(match symbol {
                SymbolRef::Field(field) if field == left => vector("body"),
                SymbolRef::Field(field) if field == right => vector("other"),
                _ => unreachable!(),
            })
        },
    )
    .unwrap_err();
    assert!(matches!(
        support_errors.as_slice(),
        [TypedResidualError::Type {
            error: TypeViolation::PureOperatorApplication(PureOperatorError::CommonSupportMismatch),
            ..
        }]
    ));

    let type_errors = TypedResidual::infer(
        expression,
        Some(volume("body")),
        RootContract::ComponentwiseResidual,
        |symbol| {
            Ok::<_, ()>(match symbol {
                SymbolRef::Field(field) if field == left => vector("body"),
                SymbolRef::Field(field) if field == right => {
                    ExpressionType::scalar(DimExponents::DIMENSIONLESS, Some(volume("body")))
                }
                _ => unreachable!(),
            })
        },
    )
    .unwrap_err();
    assert!(matches!(
        type_errors.as_slice(),
        [TypedResidualError::Type {
            error: TypeViolation::PureOperatorApplication(PureOperatorError::FormalTypeMismatch),
            ..
        }]
    ));
}
mod arrays;
