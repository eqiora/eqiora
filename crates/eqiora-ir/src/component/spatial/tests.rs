use super::*;
use eqiora_core::{DimExponents, Id, ScalarDomain, ValueFrame, ValueShape, entity::kinds};
use eqiora_schema::kernel::{
    ExprDagBuilder,
    typing::{RootContract, SpatialSupport},
};

#[test]
fn field_gradients_keep_channel_component_direction_phase_and_units_separate() {
    let field = SymbolRef::Field(Id::<kinds::Field>::new());
    let support = SpatialSupport::Volume {
        domain: (),
        dimensions: 2,
    };
    for complex in [false, true] {
        for channels in [false, true] {
            let element = ExpressionType::shaped(
                DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap(),
                ValueShape::new([2]).unwrap(),
                ValueFrame::SpatialCartesian,
                Some(support.clone()),
            )
            .unwrap();
            let ty = element
                .value_type
                .with_scalar_domain(if complex {
                    ScalarDomain::Complex
                } else {
                    ScalarDomain::Real
                })
                .unwrap();
            let ty = if channels { ty.array(3).unwrap() } else { ty };
            let mut builder = ExprDagBuilder::new();
            let value = builder.symbol(field).unwrap();
            let gradient = builder.gradient(value).unwrap();
            let hessian = builder.gradient(gradient).unwrap();
            let typed = TypedResidual::infer(
                builder.finish([value, gradient, hessian]).unwrap(),
                Some(support.clone()),
                |_| None,
                RootContract::ComponentwiseResidual,
                |_| Ok::<_, ()>(ExpressionType::new(ty.clone(), Some(support.clone()))),
            )
            .unwrap();
            assert!(ComponentScalarization::lower_selected(&typed, &[gradient]).is_err());
            let lowered = ComponentScalarization::lower_spatial_selected(
                &typed,
                &[value, gradient, hessian],
                None,
            )
            .unwrap();
            for row in lowered.rows() {
                let order = row.root_index();
                let coordinate = &row.symbols()[0];
                assert_eq!(coordinate.symbol(), field);
                let rank = ty.shape().rank();
                assert_eq!(coordinate.component_index(), &row.component_index()[..rank]);
                assert_eq!(
                    coordinate.spatial_derivative_axes(),
                    &row.component_index()[rank..]
                        .iter()
                        .map(|i| *i as usize)
                        .collect::<Vec<_>>()
                );
                assert_eq!(coordinate.spatial_derivative_axes().len(), order);
                assert_eq!(coordinate.is_imaginary(), row.is_imaginary());
                assert_eq!(
                    row.input_slots()[0].dimension(),
                    DimExponents::from_integers([0, 1 - order as i32, 0, 0, 0, 0, 0]).unwrap()
                );
            }
            let values = lowered
                .evaluate(|coordinate| {
                    let component =
                        ty.shape().flat_index(coordinate.component_index()).unwrap() as f64;
                    let direction = coordinate
                        .spatial_derivative_axes()
                        .iter()
                        .fold(0., |value, axis| 3. * value + *axis as f64 + 1.);
                    Some(100. * component + 10. * direction + f64::from(coordinate.is_imaginary()))
                })
                .unwrap();
            let width = ty.shape().component_count().unwrap() * if complex { 2 } else { 1 };
            assert_eq!(values.len(), width * (1 + 2 + 4));
        }
    }
}

#[test]
fn partial_divergence_and_oblique_normal_keep_complex_coordinates_and_axes() {
    let volume = Id::<kinds::Domain>::new();
    let boundary = Id::<kinds::Domain>::new();
    let field = SymbolRef::Field(Id::<kinds::Field>::new());
    let support = SpatialSupport::Volume {
        domain: volume.erase(),
        dimensions: 2,
    };
    let surface = SpatialSupport::Boundary {
        domain: boundary.erase(),
        parent: volume.erase(),
        dimensions: 2,
    };
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let ty = eqiora_core::ValueType::shaped(
        ScalarDomain::Complex,
        length,
        ValueShape::new([2]).unwrap(),
        ValueFrame::SpatialCartesian,
    )
    .unwrap();
    let mut builder = ExprDagBuilder::new();
    let u = builder.symbol(field).unwrap();
    let y = builder
        .symbol(SymbolRef::Coordinate {
            support: volume,
            factor: volume,
            axis: 1,
        })
        .unwrap();
    let partial = builder.coordinate_partial(u, y).unwrap();
    let divergence = builder.divergence(u).unwrap();
    let trace = builder.trace(u, boundary).unwrap();
    let gradient = builder.gradient(u).unwrap();
    let normal = builder.normal_component(gradient, boundary).unwrap();
    let typed = TypedResidual::infer(
        builder
            .finish([partial, divergence, trace, normal])
            .unwrap(),
        None,
        |id| (id == boundary).then(|| surface.clone()),
        RootContract::ValueRoots,
        |symbol| {
            Ok::<_, ()>(if symbol == field {
                ExpressionType::new(ty.clone(), Some(support.clone()))
            } else {
                ExpressionType::scalar(length, Some(support.clone()))
            })
        },
    )
    .unwrap();
    for absent_or_wrong in [
        None,
        Some(&[0.0, 0.0][..]),
        Some(&[1.0][..]),
        Some(&[f64::NAN, 1.0][..]),
    ] {
        assert!(
            ComponentScalarization::lower_spatial_selected(&typed, &[normal], absent_or_wrong)
                .is_err()
        );
    }
    assert!(ComponentScalarization::lower(&typed).is_err());
    let lowered = ComponentScalarization::lower_spatial_selected(
        &typed,
        typed.expression().roots(),
        Some(&[0.6, -0.8]),
    )
    .unwrap();
    let actual = lowered
        .evaluate(|input| {
            assert_eq!(input.symbol(), field);
            let component = input.component_index()[0] as usize;
            let phase = usize::from(input.is_imaginary());
            Some(match input.spatial_derivative_axes() {
                [] => [[2., 3.], [5., -7.]][component][phase],
                [axis] => {
                    [[[11., 13.], [23., 29.]], [[17., 19.], [31., 37.]]][component][phase][*axis]
                }
                _ => panic!("first derivatives only"),
            })
        })
        .unwrap();
    let expected = [
        13., 29., 19., 37., 30., 60., 2., 3., 5., -7., -3.8, -9.4, -5., -11.,
    ];
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn composed_spatial_product_quotient_and_hessian_use_the_same_scalar_differential() {
    let volume = Id::<kinds::Domain>::new();
    let u = SymbolRef::Field(Id::<kinds::Field>::new());
    let v = SymbolRef::Field(Id::<kinds::Field>::new());
    let p = SymbolRef::Parameter(Id::<kinds::Parameter>::new());
    let x = SymbolRef::Coordinate {
        support: volume,
        factor: volume,
        axis: 0,
    };
    let support = SpatialSupport::Volume {
        domain: volume.erase(),
        dimensions: 2,
    };
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let mut builder = ExprDagBuilder::new();
    let ur = builder.symbol(u).unwrap();
    let vr = builder.symbol(v).unwrap();
    let pr = builder.symbol(p).unwrap();
    let xr = builder.symbol(x).unwrap();
    let square = builder.powi(ur, 2).unwrap();
    let product = builder.mul(xr, square).unwrap();
    let product = builder.mul(pr, product).unwrap();
    let quotient = builder.div(product, vr).unwrap();
    let gradient = builder.gradient(quotient).unwrap();
    let hessian = builder.gradient(gradient).unwrap();
    let typed = TypedResidual::infer(
        builder.finish([gradient, hessian]).unwrap(),
        Some(support.clone()),
        |_| None,
        RootContract::ComponentwiseResidual,
        |symbol| {
            Ok::<_, ()>(if symbol == p {
                ExpressionType::scalar(DimExponents::DIMENSIONLESS, None)
            } else {
                ExpressionType::scalar(length, Some(support.clone()))
            })
        },
    )
    .unwrap();
    assert!(ComponentScalarization::lower(&typed).is_err());
    let lowered =
        ComponentScalarization::lower_spatial_selected(&typed, typed.expression().roots(), None)
            .unwrap();
    // At (x,y)=(2,0), p=3, v=2, u=x²+y: f=(3/2)x(x²+y)².
    // Thus grad f=(120,24) and Hess f=[[240,36],[36,6]].
    // These local jets include the dimensional coefficients implicit in u's profile.
    let binding = |coordinate: &ScalarSymbolCoordinate| {
        let symbol = coordinate.symbol();
        let axes = coordinate.spatial_derivative_axes();
        Some(if symbol == p {
            assert!(axes.is_empty());
            3.
        } else if symbol == x {
            assert!(axes.is_empty());
            2.
        } else if symbol == v {
            if axes.is_empty() { 2. } else { 0. }
        } else {
            assert_eq!(symbol, u);
            match axes {
                [] => 4.,
                [0] => 4.,
                [1] => 1.,
                [0, 0] => 2.,
                [_, _] => 0.,
                _ => panic!("unexpected derivative order"),
            }
        })
    };
    assert_eq!(
        lowered.evaluate(binding).unwrap(),
        [120., 24., 240., 36., 36., 6.]
    );
    assert!(
        lowered
            .evaluate(|coordinate| {
                if coordinate.symbol() == v && coordinate.spatial_derivative_axes().is_empty() {
                    Some(0.)
                } else {
                    binding(coordinate)
                }
            })
            .is_err()
    );
}

#[test]
fn composed_complex_quotient_gradient_retains_parameter_and_physical_axes() {
    use eqiora_schema::kernel::UnaryMathFunction;
    let u = SymbolRef::Field(Id::<kinds::Field>::new());
    let p = SymbolRef::Parameter(Id::<kinds::Parameter>::new());
    let support = SpatialSupport::Volume {
        domain: (),
        dimensions: 2,
    };
    let mut builder = ExprDagBuilder::new();
    let value = builder.symbol(u).unwrap();
    let parameter = builder.symbol(p).unwrap();
    let conjugate = builder.unary_math(UnaryMathFunction::Conj, value).unwrap();
    let quotient = builder.div(conjugate, parameter).unwrap();
    let gradient = builder.gradient(quotient).unwrap();
    let typed = TypedResidual::infer(
        builder.finish([gradient]).unwrap(),
        Some(support.clone()),
        |_| None,
        RootContract::ComponentwiseResidual,
        |symbol| {
            Ok::<_, ()>(ExpressionType::new(
                eqiora_core::ValueType::scalar(ScalarDomain::Complex, DimExponents::DIMENSIONLESS)
                    .unwrap(),
                (symbol == u).then(|| support.clone()),
            ))
        },
    )
    .unwrap();
    let lowered =
        ComponentScalarization::lower_spatial_selected(&typed, &[gradient], None).unwrap();
    // u=x+i*y and p=2+3i give d_x(conj(u)/p)=(2-3i)/13,
    // d_y(conj(u)/p)=(-3-2i)/13 independently of x and y.
    let values = lowered
        .evaluate(|coordinate| {
            let part = usize::from(coordinate.is_imaginary());
            Some(if coordinate.symbol() == p {
                [2., 3.][part]
            } else {
                match coordinate.spatial_derivative_axes() {
                    [] => [2., -1.][part],
                    [0] => [1., 0.][part],
                    [1] => [0., 1.][part],
                    _ => panic!("unexpected derivative order"),
                }
            })
        })
        .unwrap();
    for (actual, expected) in values
        .into_iter()
        .zip([2. / 13., -3. / 13., -3. / 13., -2. / 13.])
    {
        assert!((actual - expected).abs() < 16. * f64::EPSILON);
    }
    assert!(
        lowered
            .evaluate(|coordinate| { Some(if coordinate.symbol() == p { 0. } else { 1. }) })
            .is_err()
    );
}

#[test]
fn spatially_constant_math_keeps_primal_domains_without_demanding_parameter_derivatives() {
    use eqiora_schema::kernel::UnaryMathFunction;
    let u = SymbolRef::Field(Id::<kinds::Field>::new());
    let p = SymbolRef::Parameter(Id::<kinds::Parameter>::new());
    let support = SpatialSupport::Volume {
        domain: (),
        dimensions: 2,
    };
    let mut builder = ExprDagBuilder::new();
    let value = builder.symbol(u).unwrap();
    let parameter = builder.symbol(p).unwrap();
    let root = builder
        .unary_math(UnaryMathFunction::Sqrt, parameter)
        .unwrap();
    let sum = builder.add(root, value).unwrap();
    let gradient = builder.gradient(sum).unwrap();
    let typed = TypedResidual::infer(
        builder.finish([gradient]).unwrap(),
        Some(support.clone()),
        |_| None,
        RootContract::ComponentwiseResidual,
        |symbol| {
            Ok::<_, ()>(ExpressionType::scalar(
                DimExponents::DIMENSIONLESS,
                (symbol == u).then(|| support.clone()),
            ))
        },
    )
    .unwrap();
    let lowered =
        ComponentScalarization::lower_spatial_selected(&typed, &[gradient], None).unwrap();
    // sqrt(p) is constant in space, even at p=0 where its p derivative is undefined.
    for parameter in [0., 4.] {
        assert_eq!(
            lowered
                .evaluate(|coordinate| Some(if coordinate.symbol() == p {
                    parameter
                } else {
                    match coordinate.spatial_derivative_axes() {
                        [] => 7.,
                        [0] => 2.,
                        [1] => 3.,
                        _ => unreachable!(),
                    }
                }))
                .unwrap(),
            [2., 3.]
        );
    }
    assert!(
        lowered
            .evaluate(|coordinate| Some(if coordinate.symbol() == p { -1. } else { 0. }))
            .is_err()
    );
}

#[test]
fn elementary_spatial_gradients_and_hessians_share_actions_and_keep_dimensions() {
    use crate::LinearizedRelation;
    use eqiora_schema::kernel::UnaryMathFunction::*;
    let u = SymbolRef::Field(Id::<kinds::Field>::new());
    let support = SpatialSupport::Volume {
        domain: (),
        dimensions: 2,
    };
    // f(u): (u, f', f''). Compose with independent local jets
    // grad u=(2,3), Hess u=[[5,7],[7,11]].
    for (function, point, first, second) in [
        (Sin, 0., 1., 0.),
        (Cos, 0., 0., -1.),
        (Exp, 0., 1., 1.),
        (Log, 1., 1., -1.),
        (Sqrt, 4., 0.25, -1. / 32.),
    ] {
        let mut builder = ExprDagBuilder::new();
        let value = builder.symbol(u).unwrap();
        let value = builder.unary_math(function, value).unwrap();
        let gradient = builder.gradient(value).unwrap();
        let hessian = builder.gradient(gradient).unwrap();
        let typed = TypedResidual::infer(
            builder.finish([gradient, hessian]).unwrap(),
            Some(support.clone()),
            |_| None,
            RootContract::ComponentwiseResidual,
            |_| {
                Ok::<_, ()>(ExpressionType::scalar(
                    DimExponents::DIMENSIONLESS,
                    Some(support.clone()),
                ))
            },
        )
        .unwrap();
        let lowered =
            ComponentScalarization::lower_spatial_selected(&typed, &[gradient, hessian], None)
                .unwrap();
        let binding = |coordinate: &ScalarSymbolCoordinate| {
            Some(match coordinate.spatial_derivative_axes() {
                [] => point,
                [0] => 2.,
                [1] => 3.,
                [0, 0] => 5.,
                [0, 1] | [1, 0] => 7.,
                [1, 1] => 11.,
                _ => unreachable!(),
            })
        };
        let expected = [
            2. * first,
            3. * first,
            5. * first + 4. * second,
            7. * first + 6. * second,
            7. * first + 6. * second,
            11. * first + 9. * second,
        ];
        let actual = lowered.evaluate(binding).unwrap();
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 16. * f64::EPSILON,
                "{function:?}: {actual} != {expected}"
            );
        }
        for (axis, row) in lowered.rows()[..2].iter().enumerate() {
            let inputs = row
                .symbols()
                .iter()
                .map(|c| binding(c).unwrap())
                .collect::<Vec<_>>();
            let directions = row
                .symbols()
                .iter()
                .map(|c| f64::from(c.spatial_derivative_axes().is_empty()))
                .collect::<Vec<_>>();
            let roles = vec![crate::DifferentiationRole::Unknown; inputs.len()];
            let action = row.linearize(&inputs, &roles).unwrap();
            let mut output = [0.];
            action
                .jvp(crate::RelationTangent::Unknown(&directions), &mut output)
                .unwrap();
            assert!((output[0] - [2., 3.][axis] * second).abs() < 16. * f64::EPSILON);
            for slot in row.input_slots() {
                let order = slot.source().spatial_derivative_axes().len();
                assert_eq!(
                    slot.dimension(),
                    DimExponents::from_integers([0, -(order as i32), 0, 0, 0, 0, 0]).unwrap()
                );
            }
        }
        if matches!(function, Log | Sqrt) {
            assert!(lowered.evaluate(|_| Some(0.)).is_err());
            assert!(
                lowered
                    .evaluate(|c| Some(if c.spatial_derivative_axes().is_empty() {
                        -1.
                    } else {
                        0.
                    }))
                    .is_err()
            );
        }
    }
}

#[test]
fn spatial_derivative_retains_frozen_primal_domains_when_other_terms_need_typed_guards() {
    use eqiora_schema::kernel::UnaryMathFunction::Sqrt;
    let u = SymbolRef::Field(Id::<kinds::Field>::new());
    let p = SymbolRef::Parameter(Id::<kinds::Parameter>::new());
    let support = SpatialSupport::Volume {
        domain: (),
        dimensions: 2,
    };
    let mut builder = ExprDagBuilder::new();
    let u_root = builder.symbol(u).unwrap();
    let p_root = builder.symbol(p).unwrap();
    let a = builder.unary_math(Sqrt, u_root).unwrap();
    let b = builder.unary_math(Sqrt, p_root).unwrap();
    let sum = builder.add(a, b).unwrap();
    let gradient = builder.gradient(sum).unwrap();
    let typed = TypedResidual::infer(
        builder.finish([gradient]).unwrap(),
        Some(support.clone()),
        |_| None,
        RootContract::ComponentwiseResidual,
        |symbol| {
            Ok::<_, ()>(ExpressionType::scalar(
                DimExponents::DIMENSIONLESS,
                (symbol == u).then(|| support.clone()),
            ))
        },
    )
    .unwrap();
    let lowered =
        ComponentScalarization::lower_spatial_selected(&typed, &[gradient], None).unwrap();
    for parameter in [0., 4., -1.] {
        let values = lowered.evaluate(|coordinate| {
            Some(if coordinate.symbol() == p {
                parameter
            } else {
                match coordinate.spatial_derivative_axes() {
                    [] => 4.,
                    [0] => 2.,
                    [1] => 3.,
                    _ => unreachable!(),
                }
            })
        });
        if parameter < 0. {
            assert!(values.is_err());
        } else {
            assert_eq!(values.unwrap(), [0.5, 0.75]);
        }
    }
}

#[test]
fn spatial_guard_derivatives_distinguish_active_boundaries_and_frozen_predicates() {
    use eqiora_schema::kernel::ComparisonOp;
    let u = SymbolRef::Field(Id::<kinds::Field>::new());
    let p = SymbolRef::Parameter(Id::<kinds::Parameter>::new());
    let support = SpatialSupport::Volume {
        domain: (),
        dimensions: 2,
    };
    for active in [false, true] {
        let mut builder = ExprDagBuilder::new();
        let value = builder.symbol(u).unwrap();
        let parameter = builder.symbol(p).unwrap();
        let zero = builder
            .constant(DynQuantity::new(0., DimExponents::DIMENSIONLESS))
            .unwrap();
        let condition = builder
            .compare(
                ComparisonOp::GreaterEqual,
                if active { value } else { parameter },
                zero,
            )
            .unwrap();
        let square = builder.powi(value, 2).unwrap();
        let guarded = builder.require(condition, square).unwrap();
        let gradient = builder.gradient(guarded).unwrap();
        let typed = TypedResidual::infer(
            builder.finish([gradient]).unwrap(),
            Some(support.clone()),
            |_| None,
            RootContract::ComponentwiseResidual,
            |symbol| {
                Ok::<_, ()>(ExpressionType::scalar(
                    DimExponents::DIMENSIONLESS,
                    (symbol == u).then(|| support.clone()),
                ))
            },
        )
        .unwrap();
        let lowered =
            ComponentScalarization::lower_spatial_selected(&typed, &[gradient], None).unwrap();
        for point in [-1., 0., 2.] {
            let values = lowered.evaluate(|coordinate| {
                Some(if coordinate.symbol() == p {
                    0.
                } else {
                    match coordinate.spatial_derivative_axes() {
                        [] => point,
                        [0] => 3.,
                        [1] => 5.,
                        _ => unreachable!(),
                    }
                })
            });
            if active && point <= 0. {
                assert!(values.is_err());
            } else {
                assert_eq!(values.unwrap(), [6. * point, 10. * point]);
            }
        }
    }
}

#[test]
fn complex_quotients_compose_with_guarded_real_functions_and_their_jvps() {
    use crate::LinearizedRelation;
    use eqiora_schema::kernel::UnaryMathFunction::{Conj, Real, Sqrt};
    let u = SymbolRef::Field(Id::<kinds::Field>::new());
    let p = SymbolRef::Parameter(Id::<kinds::Parameter>::new());
    let support = SpatialSupport::Volume {
        domain: (),
        dimensions: 2,
    };
    let mut builder = ExprDagBuilder::new();
    let value = builder.symbol(u).unwrap();
    let parameter = builder.symbol(p).unwrap();
    let real = builder.unary_math(Real, value).unwrap();
    let root = builder.unary_math(Sqrt, real).unwrap();
    let conjugate = builder.unary_math(Conj, value).unwrap();
    let quotient = builder.div(conjugate, parameter).unwrap();
    let sum = builder.add(root, quotient).unwrap();
    let gradient = builder.gradient(sum).unwrap();
    let typed = TypedResidual::infer(
        builder.finish([gradient]).unwrap(),
        Some(support.clone()),
        |_| None,
        RootContract::ComponentwiseResidual,
        |symbol| {
            Ok::<_, ()>(ExpressionType::new(
                eqiora_core::ValueType::scalar(ScalarDomain::Complex, DimExponents::DIMENSIONLESS)
                    .unwrap(),
                (symbol == u).then(|| support.clone()),
            ))
        },
    )
    .unwrap();
    let lowered =
        ComponentScalarization::lower_spatial_selected(&typed, &[gradient], None).unwrap();
    let binding = |coordinate: &ScalarSymbolCoordinate| {
        let part = usize::from(coordinate.is_imaginary());
        Some(if coordinate.symbol() == p {
            [2., 3.][part]
        } else {
            match coordinate.spatial_derivative_axes() {
                [] => [4., -1.][part],
                [0] => [1., 0.][part],
                [1] => [0., 1.][part],
                _ => unreachable!(),
            }
        })
    };
    // f=sqrt(x)+(x-i*y)/(2+3i), at x=4; d_x sqrt(x)=1/4.
    for (actual, expected) in lowered.evaluate(binding).unwrap().into_iter().zip([
        0.25 + 2. / 13.,
        -3. / 13.,
        -3. / 13.,
        -2. / 13.,
    ]) {
        assert!((actual - expected).abs() < 16. * f64::EPSILON);
    }
    for (index, row) in lowered.rows().iter().enumerate() {
        let inputs = row
            .symbols()
            .iter()
            .map(|c| binding(c).unwrap())
            .collect::<Vec<_>>();
        let roles = row
            .symbols()
            .iter()
            .map(|c| {
                if c.symbol() == p {
                    crate::DifferentiationRole::Frozen
                } else {
                    crate::DifferentiationRole::Unknown
                }
            })
            .collect::<Vec<_>>();
        let directions = row
            .symbols()
            .iter()
            .filter(|c| c.symbol() == u)
            .map(|c| f64::from(!c.is_imaginary() && c.spatial_derivative_axes().is_empty()))
            .collect::<Vec<_>>();
        let action = row.linearize(&inputs, &roles).unwrap();
        let mut output = [0.];
        action
            .jvp(crate::RelationTangent::Unknown(&directions), &mut output)
            .unwrap();
        assert!((output[0] - if index == 0 { -1. / 32. } else { 0. }).abs() < 16. * f64::EPSILON);
    }
    assert!(
        lowered
            .evaluate(|c| if c.symbol() == p {
                Some(0.)
            } else {
                binding(c)
            })
            .is_err()
    );
    assert!(
        lowered
            .evaluate(|c| {
                if c.symbol() == u && c.spatial_derivative_axes().is_empty() && !c.is_imaginary() {
                    Some(0.)
                } else {
                    binding(c)
                }
            })
            .is_err()
    );
}
