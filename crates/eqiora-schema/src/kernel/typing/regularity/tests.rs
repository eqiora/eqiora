use super::*;
use crate::kernel::ExprDagBuilder;
use eqiora_core::{Id, ValueFrame, entity::kinds};

fn expression(
    dimensions: usize,
    rank: usize,
    build: impl FnOnce(&mut ExprDagBuilder, ExprId, Id<kinds::Domain>) -> ExprId,
) -> TypedResidual<&'static str> {
    let field = Id::<kinds::Field>::new();
    let on = Id::<kinds::Domain>::new();
    let mut builder = ExprDagBuilder::new();
    let value = builder.symbol(SymbolRef::Field(field)).unwrap();
    let root = build(&mut builder, value, on);
    let boundary = SpatialSupport::Boundary {
        domain: "wall",
        parent: "body",
        dimensions,
    };
    TypedResidual::infer(
        builder.finish([root]).unwrap(),
        None,
        |id| (id == on).then(|| boundary.clone()),
        RootContract::ValueRoots,
        |symbol| {
            if matches!(symbol, SymbolRef::Parameter(_)) {
                return Ok(ExpressionType::scalar(DimExponents::DIMENSIONLESS, None));
            }
            Ok::<_, ()>(
                ExpressionType::shaped(
                    DimExponents::DIMENSIONLESS,
                    ValueShape::new(std::iter::repeat_n(dimensions as u32, rank)).unwrap(),
                    if rank == 0 {
                        ValueFrame::Invariant
                    } else {
                        ValueFrame::SpatialCartesian
                    },
                    Some(SpatialSupport::Volume {
                        domain: "body",
                        dimensions,
                    }),
                )
                .unwrap(),
            )
        },
    )
    .unwrap()
}

#[test]
fn missing_and_l2_regularity_do_not_grant_a_trace() {
    for rank in 0..=2 {
        let typed = expression(2, rank, |b, u, on| b.trace(u, on).unwrap());
        for regularity in [
            SpatialRegularity::Unspecified,
            SpatialRegularity::L2,
            SpatialRegularity::HCurl,
            SpatialRegularity::HDiv,
        ] {
            assert!(
                typed.validate_trace_regularity(|_| regularity).is_err(),
                "{regularity:?}"
            );
        }
        for regularity in [SpatialRegularity::H1, SpatialRegularity::Smooth] {
            typed.validate_trace_regularity(|_| regularity).unwrap();
        }
    }
}

#[test]
fn weak_normal_trace_does_not_require_a_full_vector_or_tensor_trace() {
    for rank in 1..=2 {
        let typed = expression(3, rank, |b, u, on| b.normal_component(u, on).unwrap());
        for regularity in [
            SpatialRegularity::HDiv,
            SpatialRegularity::H1,
            SpatialRegularity::Smooth,
        ] {
            typed.validate_trace_regularity(|_| regularity).unwrap();
        }
        for regularity in [
            SpatialRegularity::HCurl,
            SpatialRegularity::L2,
            SpatialRegularity::Unspecified,
        ] {
            assert!(typed.validate_trace_regularity(|_| regularity).is_err());
        }
    }
}

#[test]
fn exact_tangential_lift_transfers_the_curl_trace_theorem() {
    for dimensions in [2, 3] {
        let typed = expression(dimensions, 1, |b, u, on| {
            let lift = PureOperatorDefinition::tangential_lift(dimensions as u32).unwrap();
            let lifted = b.pure_operator(&lift, [u]).unwrap();
            b.normal_component(lifted, on).unwrap()
        });
        for regularity in [
            SpatialRegularity::HCurl,
            SpatialRegularity::H1,
            SpatialRegularity::Smooth,
        ] {
            typed.validate_trace_regularity(|_| regularity).unwrap();
        }
        for regularity in [
            SpatialRegularity::HDiv,
            SpatialRegularity::L2,
            SpatialRegularity::Unspecified,
        ] {
            assert!(typed.validate_trace_regularity(|_| regularity).is_err());
        }
    }
}

#[test]
fn guarded_polynomial_traces_retain_value_and_guard_input_regularity() {
    use crate::kernel::ComparisonOp;
    use crate::kernel::pure_operator::{CalculusBuilder, ExactRational, PureValueClass};

    let class = PureValueClass::invariant_scalar()
        .with_dimension(DimExponents::DIMENSIONLESS)
        .with_scalar_domain(eqiora_core::ScalarDomain::Real)
        .unwrap();
    for squared in [false, true] {
        for branch in [false, true] {
            let mut calculus =
                CalculusBuilder::new([class.clone(), class.clone()], class.clone()).unwrap();
            let u = calculus
                .push(CalculusNode::FormalComponent {
                    formal: 0,
                    axes: Box::new([]),
                })
                .unwrap();
            let guard = calculus
                .push(CalculusNode::FormalComponent {
                    formal: 1,
                    axes: Box::new([]),
                })
                .unwrap();
            let zero = calculus
                .push(CalculusNode::Rational {
                    value: ExactRational::integer(0),
                    dimension: DimExponents::DIMENSIONLESS,
                })
                .unwrap();
            let condition = calculus
                .push(CalculusNode::Compare(ComparisonOp::Less, zero, guard))
                .unwrap();
            let value = if squared {
                calculus.push(CalculusNode::Mul(u, u)).unwrap()
            } else {
                u
            };
            let root = if branch {
                calculus
                    .push(CalculusNode::Select {
                        condition,
                        then_value: value,
                        else_value: zero,
                    })
                    .unwrap()
            } else {
                calculus
                    .push(CalculusNode::Require { condition, value })
                    .unwrap()
            };
            let definition = calculus.finish(root).unwrap();
            let guard_field = Id::<kinds::Field>::new();
            let typed = expression(2, 0, |builder, u, on| {
                let guard = builder.symbol(SymbolRef::Field(guard_field)).unwrap();
                let value = builder.pure_operator(&definition, [u, guard]).unwrap();
                builder.trace(value, on).unwrap()
            });
            for regularity in [
                SpatialRegularity::L2,
                SpatialRegularity::H1,
                SpatialRegularity::Smooth,
            ] {
                let accepted = !branch
                    && (regularity == SpatialRegularity::Smooth
                        || (!squared && regularity == SpatialRegularity::H1));
                assert_eq!(
                    typed.validate_trace_regularity(|_| regularity).is_ok(),
                    accepted
                );
            }
            // A smooth value cannot hide a rough guard-only argument. Neither
            // this type check nor polynomial degree proves that the guard holds.
            assert!(
                typed
                    .validate_trace_regularity(|symbol| {
                        if symbol == SymbolRef::Field(guard_field) {
                            SpatialRegularity::L2
                        } else {
                            SpatialRegularity::Smooth
                        }
                    })
                    .is_err()
            );
        }
    }
}

#[test]
fn h1_does_not_grant_a_trace_of_its_gradient_or_nonlinear_product() {
    for typed in [
        expression(2, 0, |b, u, on| {
            let gradient = b.gradient(u).unwrap();
            b.normal_component(gradient, on).unwrap()
        }),
        expression(2, 0, |b, u, on| {
            let product = b.mul(u, u).unwrap();
            b.trace(product, on).unwrap()
        }),
    ] {
        assert!(
            typed
                .validate_trace_regularity(|_| SpatialRegularity::H1)
                .is_err()
        );
        typed
            .validate_trace_regularity(|_| SpatialRegularity::Smooth)
            .unwrap();
    }
}

#[test]
fn hdiv_cannot_be_laundered_through_a_full_trace_then_contraction() {
    let typed = expression(2, 1, |b, u, on| {
        let full = b.trace(u, on).unwrap();
        b.normal_component(full, on).unwrap()
    });
    assert!(
        typed
            .validate_trace_regularity(|_| SpatialRegularity::HDiv)
            .is_err()
    );
    typed
        .validate_trace_regularity(|_| SpatialRegularity::H1)
        .unwrap();
}

#[test]
fn a_smooth_spatial_multiplier_cannot_be_mistaken_for_a_constant() {
    let typed = expression(2, 0, |b, u, on| {
        let two = b
            .constant(eqiora_core::DynQuantity::new(
                2.0,
                DimExponents::DIMENSIONLESS,
            ))
            .unwrap();
        let denominator = b.mul(u, two).unwrap();
        let ratio = b.div(two, denominator).unwrap();
        b.trace(ratio, on).unwrap()
    });
    // Smooth u may vanish. This profile has not proved a regular reciprocal.
    assert!(
        typed
            .validate_trace_regularity(|_| SpatialRegularity::Smooth)
            .is_err()
    );
}

#[test]
fn an_exact_constant_multiplier_preserves_the_weak_normal_trace() {
    let typed = expression(2, 1, |b, u, on| {
        let two = b
            .constant(eqiora_core::DynQuantity::new(
                2.0,
                DimExponents::DIMENSIONLESS,
            ))
            .unwrap();
        let scaled = b.mul(two, u).unwrap();
        b.normal_component(scaled, on).unwrap()
    });
    typed
        .validate_trace_regularity(|_| SpatialRegularity::HDiv)
        .unwrap();
    assert!(
        typed
            .validate_trace_regularity(|_| SpatialRegularity::HCurl)
            .is_err()
    );
}

#[test]
fn a_compound_global_coefficient_preserves_full_and_weak_traces() {
    for (rank, regularity) in [(0, SpatialRegularity::H1), (1, SpatialRegularity::HDiv)] {
        let typed = expression(2, rank, |b, u, on| {
            let parameter = b.symbol(SymbolRef::Parameter(Id::new())).unwrap();
            let square = b.powi(parameter, 2).unwrap();
            let one = b
                .constant(eqiora_core::DynQuantity::new(
                    1.0,
                    DimExponents::DIMENSIONLESS,
                ))
                .unwrap();
            // 1 + p^2 is a positive spatial constant for every finite real p.
            let denominator = b.add(one, square).unwrap();
            let scaled = b.div(u, denominator).unwrap();
            if rank == 0 {
                b.trace(scaled, on).unwrap()
            } else {
                b.normal_component(scaled, on).unwrap()
            }
        });
        typed.validate_trace_regularity(|_| regularity).unwrap();
        assert!(
            typed
                .validate_trace_regularity(|_| SpatialRegularity::L2)
                .is_err()
        );
    }
}

#[test]
fn finite_matrix_invariants_do_not_inherit_linear_trace_rules() {
    use eqiora_core::{FiniteBasis, ScalarDomain, ValueType};
    let basis = FiniteBasis::new(Id::new(), 2).unwrap();
    for (operation, h1, smooth) in [
        (FiniteUnaryOperation::Transpose, true, true),
        (FiniteUnaryOperation::MatrixTrace, true, true),
        (FiniteUnaryOperation::Determinant, false, true),
        // Smooth entries do not imply a nonsingular matrix on the closure.
        (FiniteUnaryOperation::Inverse, false, false),
    ] {
        let on = Id::new();
        let mut builder = ExprDagBuilder::new();
        let field = builder.symbol(SymbolRef::Field(Id::new())).unwrap();
        let mapped = builder.finite_unary(operation, field).unwrap();
        let root = builder.trace(mapped, on).unwrap();
        let typed = TypedResidual::infer(
            builder.finish([root]).unwrap(),
            None,
            |_| {
                Some(SpatialSupport::Boundary {
                    domain: "wall",
                    parent: "body",
                    dimensions: 2,
                })
            },
            RootContract::ValueRoots,
            |_| {
                Ok::<_, ()>(ExpressionType::new(
                    ValueType::linear_map(
                        basis,
                        basis,
                        ScalarDomain::Real,
                        DimExponents::DIMENSIONLESS,
                    )
                    .unwrap(),
                    Some(SpatialSupport::Volume {
                        domain: "body",
                        dimensions: 2,
                    }),
                ))
            },
        )
        .unwrap();
        assert_eq!(
            typed
                .validate_trace_regularity(|_| SpatialRegularity::H1)
                .is_ok(),
            h1,
            "{operation:?}"
        );
        assert_eq!(
            typed
                .validate_trace_regularity(|_| SpatialRegularity::Smooth)
                .is_ok(),
            smooth,
            "{operation:?}"
        );
    }
}

#[test]
fn weak_normal_values_allow_linear_scaling_but_not_pointwise_products() {
    let product = expression(2, 1, |b, u, on| {
        let normal = b.normal_component(u, on).unwrap();
        b.mul(normal, normal).unwrap()
    });
    let errors = product
        .validate_trace_regularity(|_| SpatialRegularity::HDiv)
        .unwrap_err();
    assert!(errors.iter().any(|error| matches!(
        error,
        TypedResidualError::Type {
            error: TypeViolation::WeakTraceOperationUnsupported,
            ..
        }
    )));
    product
        .validate_trace_regularity(|_| SpatialRegularity::Smooth)
        .unwrap();
    let scaled = expression(2, 1, |b, u, on| {
        let normal = b.normal_component(u, on).unwrap();
        let two = b
            .constant(eqiora_core::DynQuantity::new(
                2.0,
                DimExponents::DIMENSIONLESS,
            ))
            .unwrap();
        b.mul(two, normal).unwrap()
    });
    scaled
        .validate_trace_regularity(|_| SpatialRegularity::HDiv)
        .unwrap();
}
