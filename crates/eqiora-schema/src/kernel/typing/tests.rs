use super::*;
use eqiora_core::Id;
use eqiora_core::entity::kinds;
mod point_evaluation;

fn volume(name: &'static str) -> SpatialSupport<&'static str> {
    SpatialSupport::Volume {
        domain: name,
        dimensions: 2,
    }
}

#[test]
fn time_derivative_checks_the_final_exponent_after_exact_cancellation() {
    let limit = i32::MAX;
    let dimension = DimExponents::from_integers([0, 0, limit, 0, 0, 0, 0]).unwrap();
    let operand = ExpressionType::<()>::scalar(dimension, None);
    let order = std::num::NonZeroU32::new(u32::MAX - 1).unwrap();
    let derivative = time_derivative(&operand, order).unwrap();
    assert_eq!(
        derivative.dimension(),
        DimExponents::from_integers([0, 0, -limit, 0, 0, 0, 0]).unwrap()
    );
    assert!(time_derivative(&operand, std::num::NonZeroU32::MAX).is_err());
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
        time_derivative(&complex, std::num::NonZeroU32::MIN),
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
        |_| None,
        RootContract::ComponentwiseResidual,
        |_| Ok::<_, ()>(vector.clone()),
    )
    .unwrap();
    assert_eq!(typed.node_type(root).unwrap().shape().extents()[0].get(), 2);

    let errors = TypedResidual::infer(
        expression,
        None,
        |_| None,
        RootContract::ScalarActivation,
        |_| Ok::<_, ()>(vector.clone()),
    )
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
        ExpressionType::<&str>::coordinate(&"body", 0, None),
        Err(TypeViolation::CoordinateRequiresSpatialScope)
    ));
    assert!(matches!(
        ExpressionType::coordinate(&"body", 2, Some(&volume("body"))),
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

    let boundary_coordinate = ExpressionType::coordinate(&"body", 0, Some(&boundary)).unwrap();
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
fn physical_interface_traces_join_only_the_exact_adjacent_supports() {
    let interface = SpatialSupport::PhysicalInterface {
        domain: "contact",
        boundaries: Box::new(["left_face", "right_face"]),
        parents: Box::new(["left", "right"]),
        dimensions: 2,
    };
    let left = ExpressionType::scalar(DimExponents::DIMENSIONLESS, Some(volume("left")));
    let right = ExpressionType::scalar(DimExponents::DIMENSIONLESS, Some(volume("right")));
    assert!(additive(&left, &right).is_err());
    let left_trace = trace(&left, Some(&interface)).unwrap();
    let right_trace = trace(&right, Some(&interface)).unwrap();
    assert_eq!(
        additive(&left_trace, &right_trace).unwrap().support,
        Some(interface.clone())
    );
    for support in [
        volume("foreign"),
        SpatialSupport::Volume {
            domain: "left",
            dimensions: 3,
        },
    ] {
        assert!(
            trace(
                &ExpressionType::scalar(DimExponents::DIMENSIONLESS, Some(support)),
                Some(&interface)
            )
            .is_err()
        );
    }
    for shape in [
        ValueShape::new([2]).unwrap(),
        ValueShape::new([2, 2]).unwrap(),
    ] {
        let operand = ExpressionType::shaped(
            DimExponents::DIMENSIONLESS,
            shape.clone(),
            ValueFrame::SpatialCartesian,
            Some(volume("right")),
        )
        .unwrap();
        let traced = trace(&operand, Some(&interface)).unwrap();
        assert_eq!(traced.shape(), &shape);
        let contracted = normal(&operand, Some(&interface)).unwrap();
        assert_eq!(contracted.shape(), &shape.remove_last().unwrap().0);
        assert_eq!(normal(&traced, Some(&interface)).unwrap(), contracted);
        assert_eq!(contracted.support, Some(interface.clone()));
    }
}

#[test]
fn boundary_operators_require_the_complete_parent_support() {
    let boundary = SpatialSupport::Boundary {
        domain: "wall",
        parent: "body",
        dimensions: 2,
    };
    let value = ExpressionType::shaped(
        DimExponents::DIMENSIONLESS,
        ValueShape::new([2]).unwrap(),
        ValueFrame::SpatialCartesian,
        Some(volume("body")),
    )
    .unwrap();
    let traced = trace(&value, Some(&boundary)).unwrap();
    assert_eq!(traced.value_type, value.value_type);
    assert_eq!(traced.support, Some(boundary.clone()));
    let normal_value = normal(&value, Some(&boundary)).unwrap();
    assert!(normal_value.shape().is_scalar());
    assert_eq!(normal_value.support, Some(boundary.clone()));
    assert_eq!(normal(&traced, Some(&boundary)).unwrap(), normal_value);

    // Equal nominal names do not turn another kind or dimension of support
    // into this boundary's exact parent volume.
    for support in [
        volume("foreign"),
        SpatialSupport::Volume {
            domain: "body",
            dimensions: 3,
        },
        SpatialSupport::Coordinates {
            domain: "body",
            factors: vec![("x", DimExponents::DIMENSIONLESS, 2)],
        },
        SpatialSupport::Boundary {
            domain: "body",
            parent: "other",
            dimensions: 2,
        },
        SpatialSupport::Interface {
            connection: "body",
            dimensions: 2,
        },
        SpatialSupport::Boundary {
            domain: "wall",
            parent: "body",
            dimensions: 3,
        },
    ] {
        let operand = ExpressionType::new(value.value_type.clone(), Some(support));
        for result in [
            trace(&operand, Some(&boundary)),
            normal(&operand, Some(&boundary)),
        ] {
            assert!(matches!(
                result,
                Err(TypeViolation::BoundaryOperandSupportMismatch)
            ));
        }
    }
    assert!(matches!(
        trace(&traced, Some(&boundary)),
        Err(TypeViolation::BoundaryOperandSupportMismatch)
    ));
}

#[test]
fn trace_target_is_resolved_from_the_node_without_a_relation_scope() {
    let field = Id::<kinds::Field>::new();
    let on = Id::<kinds::Domain>::new();
    let boundary = SpatialSupport::Boundary {
        domain: "wall",
        parent: "body",
        dimensions: 2,
    };
    let mut builder = super::super::ExprDagBuilder::new();
    let value = builder.symbol(SymbolRef::Field(field)).unwrap();
    let traced = builder.trace(value, on).unwrap();
    let expression = builder.finish([traced]).unwrap();
    let resolve_field = |_| {
        Ok::<_, ()>(ExpressionType::scalar(
            DimExponents::DIMENSIONLESS,
            Some(volume("body")),
        ))
    };
    let typed = TypedResidual::infer(
        expression.clone(),
        None,
        |id| (id == on).then(|| boundary.clone()),
        RootContract::ValueRoots,
        resolve_field,
    )
    .unwrap();
    assert_eq!(
        typed.node_type(traced).unwrap().support,
        Some(boundary.clone())
    );
    // A Relation's scope cannot supply a missing or foreign explicit target.
    assert!(
        TypedResidual::infer(
            expression.clone(),
            Some(boundary.clone()),
            |_| None,
            RootContract::ValueRoots,
            resolve_field,
        )
        .is_err()
    );
    // Resolving an explicit boundary does not waive the owning equation's support.
    assert!(
        TypedResidual::infer(
            expression,
            Some(volume("body")),
            |id| (id == on).then(|| boundary.clone()),
            RootContract::ComponentwiseResidual,
            resolve_field,
        )
        .is_err()
    );
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
        |_| None,
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
        |_| None,
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
        |_| None,
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

#[test]
fn coordinate_factor_projection_keeps_units_identity_and_block_axes() {
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let speed = DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).unwrap();
    let phase = SpatialSupport::Coordinates {
        domain: "phase",
        factors: vec![("position", length, 2), ("velocity", speed, 1)],
    };
    assert_eq!(
        ExpressionType::coordinate(&"position", 1, Some(&phase)).unwrap(),
        ExpressionType::scalar(length, Some(phase.clone()))
    );
    assert_eq!(
        ExpressionType::coordinate(&"velocity", 0, Some(&phase)).unwrap(),
        ExpressionType::scalar(speed, Some(phase.clone()))
    );
    assert!(matches!(
        ExpressionType::coordinate(&"velocity", 1, Some(&phase)),
        Err(TypeViolation::CoordinateAxisOutOfRange { .. })
    ));
    assert!(matches!(
        ExpressionType::coordinate(&"foreign", 0, Some(&phase)),
        Err(TypeViolation::CoordinateFactorMismatch)
    ));
    assert!(matches!(
        ExpressionType::coordinate(&"phase", 0, Some(&phase)),
        Err(TypeViolation::CoordinateFactorMismatch)
    ));
}

#[test]
fn coordinate_partial_requires_exact_selector_and_divides_by_its_unit() {
    let field = Id::<kinds::Field>::new();
    let domain = Id::<kinds::Domain>::new();
    let parameter = Id::<kinds::Parameter>::new();
    let mass = DimExponents::from_integers([1, 0, 0, 0, 0, 0, 0]).unwrap();
    let speed = DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).unwrap();
    let phase = SpatialSupport::Coordinates {
        domain: "phase",
        factors: vec![("velocity", speed, 1)],
    };
    let infer = |exact_selector: bool, foreign_field: bool| {
        let mut builder = super::super::ExprDagBuilder::new();
        let value = builder.symbol(SymbolRef::Field(field)).unwrap();
        let selected = if exact_selector {
            builder.coordinate(domain, domain, 0).unwrap()
        } else {
            builder.symbol(SymbolRef::Parameter(parameter)).unwrap()
        };
        let root = builder.coordinate_partial(value, selected).unwrap();
        let expression = builder.finish([root]).unwrap();
        (
            root,
            TypedResidual::infer(
                expression,
                Some(phase.clone()),
                |_| None,
                RootContract::ComponentwiseResidual,
                |symbol| {
                    Ok::<_, ()>(match symbol {
                        SymbolRef::Field(_) => ExpressionType::scalar(
                            mass,
                            Some(if foreign_field {
                                SpatialSupport::Coordinates {
                                    domain: "foreign",
                                    factors: vec![("velocity", speed, 1)],
                                }
                            } else {
                                phase.clone()
                            }),
                        ),
                        _ => ExpressionType::scalar(speed, Some(phase.clone())),
                    })
                },
            ),
        )
    };
    let (root, positive) = infer(true, false);
    assert_eq!(
        positive.unwrap().node_type(root).unwrap().dimension(),
        DimExponents::from_integers([1, -1, 1, 0, 0, 0, 0]).unwrap()
    );
    let (root, wrong_selector) = infer(false, false);
    assert!(wrong_selector.unwrap_err().iter().any(|error| matches!(error,TypedResidualError::Type {node_index,error:TypeViolation::CoordinatePartialRequiresCoordinate} if *node_index==root.index())));
    let (root, foreign) = infer(true, true);
    assert!(foreign.unwrap_err().iter().any(|error| matches!(error,TypedResidualError::Type {node_index,error:TypeViolation::IncompatibleSupport { .. }} if *node_index==root.index())));
}
