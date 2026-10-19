use super::*;
use crate::{DifferentiationRole, LinearizedRelation, RelationCotangent, RelationTangent};
use eqiora_core::{Id, ScalarDomain, ValueType, entity::kinds};
use eqiora_schema::kernel::{
    ExprDagBuilder,
    typing::{RootContract, SpatialSupport},
};

#[test]
fn pulled_back_factors_differentiate_before_substitution_and_keep_domains() {
    use eqiora_schema::kernel::ComparisonOp;
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    for self_map in [false, true] {
        for nested in [false, true] {
            for reciprocal in [false, true] {
                for factor in [
                    CoordinateMapFactor::SignedJacobian,
                    CoordinateMapFactor::VolumeScale,
                    CoordinateMapFactor::Orientation,
                ] {
                    let source = Id::<kinds::Domain>::new();
                    let middle = if self_map { source } else { Id::new() };
                    let target = Id::new();
                    let outer = if nested { Id::new() } else { source };
                    let p = SymbolRef::Parameter(Id::new());
                    let q = SymbolRef::Parameter(Id::new());
                    let mut b = ExprDagBuilder::new();
                    let mut axis = |support| {
                        b.symbol(SymbolRef::Coordinate {
                            support,
                            factor: support,
                            axis: 0,
                        })
                        .unwrap()
                    };
                    let xi = axis(source);
                    let x = axis(middle);
                    let y = axis(target);
                    let eta = axis(outer);
                    let pr = b.symbol(p).unwrap();
                    let qr = b.symbol(q).unwrap();
                    let unit = b.constant(DynQuantity::new(1., length)).unwrap();
                    let normalized = b.div(x, unit).unwrap();
                    // g(x)=p*x²/L or p*L²/x. The Jacobian is formed in x,
                    // then evaluated at x=q*xi, optionally xi=2*eta.
                    let profile = if reciprocal {
                        b.div(unit, normalized).unwrap()
                    } else {
                        b.mul(x, normalized).unwrap()
                    };
                    let mapped = b.mul(pr, profile).unwrap();
                    let zero = b
                        .constant(DynQuantity::new(0., DimExponents::DIMENSIONLESS))
                        .unwrap();
                    let valid = b.compare(ComparisonOp::Greater, pr, zero).unwrap();
                    let mapped = b.require(valid, mapped).unwrap();
                    let jacobian = b
                        .coordinate_map_factor(factor, vec![x], vec![(y, mapped)])
                        .unwrap();
                    let mapped_x = b.mul(qr, xi).unwrap();
                    let mut root = b.pullback(jacobian, vec![xi], vec![(x, mapped_x)]).unwrap();
                    if nested {
                        let mapped_xi = b.add(eta, eta).unwrap();
                        root = b.pullback(root, vec![eta], vec![(xi, mapped_xi)]).unwrap();
                    }
                    let typed = TypedResidual::infer(
                        b.finish([root]).unwrap(),
                        None,
                        |_| None,
                        RootContract::ValueRoots,
                        |symbol| match symbol {
                            SymbolRef::Parameter(_) => {
                                Ok(ExpressionType::scalar(DimExponents::DIMENSIONLESS, None))
                            }
                            SymbolRef::Coordinate {
                                support,
                                factor,
                                axis,
                            } => ExpressionType::coordinate(
                                &factor.erase(),
                                axis,
                                Some(&SpatialSupport::Volume {
                                    domain: support.erase(),
                                    dimensions: 1,
                                }),
                            )
                            .map_err(|_| ()),
                            _ => Err(()),
                        },
                    )
                    .unwrap();
                    let rows = ComponentScalarization::lower(&typed).unwrap();
                    let row = &rows.rows()[0];
                    assert!(row.symbols().iter().all(|input| match input.symbol() {
                        SymbolRef::Coordinate { support, .. } => support == outer,
                        symbol => symbol == p || symbol == q,
                    }));
                    let close = |a: f64, e: f64| {
                        assert!((a - e).abs() <= 1e-12 * (1. + e.abs()), "{a} != {e}")
                    };
                    for gain in [-3., 3.] {
                        let point = |parameter, position| {
                            row.symbols()
                                .iter()
                                .map(|s| {
                                    if s.symbol() == p {
                                        parameter
                                    } else if s.symbol() == q {
                                        gain
                                    } else {
                                        position
                                    }
                                })
                                .collect::<Vec<_>>()
                        };
                        let values = point(2., 5.);
                        let x: f64 = gain * 5. * if nested { 2. } else { 1. };
                        let signed = if reciprocal { -2. / x.powi(2) } else { 4. * x };
                        let expected = match factor {
                            CoordinateMapFactor::SignedJacobian => signed,
                            CoordinateMapFactor::VolumeScale => signed.abs(),
                            CoordinateMapFactor::Orientation => signed.signum(),
                        };
                        close(row.evaluate(&values).unwrap(), expected);
                        let active = factor != CoordinateMapFactor::Orientation;
                        let dp = if active { expected / 2. } else { 0. };
                        let order = if reciprocal { -2. } else { 1. };
                        let dq = if active { order * expected / gain } else { 0. };
                        let dx = if active { order * expected / 5. } else { 0. };
                        let action = row
                            .linearize(&values, &vec![DifferentiationRole::Unknown; values.len()])
                            .unwrap();
                        let direction = |symbol| {
                            row.symbols()
                                .iter()
                                .map(|s| f64::from(s.symbol() == symbol))
                                .collect::<Vec<_>>()
                        };
                        let mut output = [0.];
                        action
                            .jvp(RelationTangent::Unknown(&direction(p)), &mut output)
                            .unwrap();
                        close(output[0], dp);
                        let mut reverse = vec![0.; values.len()];
                        action
                            .vjp(&[2.], RelationCotangent::Unknown(&mut reverse))
                            .unwrap();
                        for (input, value) in row.symbols().iter().zip(reverse) {
                            close(
                                value,
                                2. * if input.symbol() == p {
                                    dp
                                } else if input.symbol() == q {
                                    dq
                                } else {
                                    dx
                                },
                            );
                        }
                        action
                            .mixed_jvp(
                                RelationTangent::Unknown(&direction(p)),
                                RelationTangent::Unknown(&direction(q)),
                                RelationTangent::Unknown(&vec![0.; values.len()]),
                                &mut output,
                            )
                            .unwrap();
                        close(output[0], dq / 2.);
                        assert!(row.evaluate(&point(-1., 5.)).is_err());
                        if reciprocal || factor != CoordinateMapFactor::SignedJacobian {
                            assert!(row.evaluate(&point(2., 0.)).is_err());
                        } else {
                            assert_eq!(row.evaluate(&point(2., 0.)).unwrap(), 0.);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn differential_factors_share_forward_reverse_and_mixed_parameter_actions() {
    let source = Id::<kinds::Domain>::new();
    let target = Id::<kinds::Domain>::new();
    let p = SymbolRef::Parameter(Id::new());
    for (factor, expected, first, second) in [
        (CoordinateMapFactor::SignedJacobian, -18., -12., -4.),
        (CoordinateMapFactor::VolumeScale, 18., 12., 4.),
        (CoordinateMapFactor::Orientation, -1., 0., 0.),
    ] {
        let mut builder = ExprDagBuilder::new();
        let mut coordinate = |support, axis| {
            builder
                .symbol(SymbolRef::Coordinate {
                    support,
                    factor: support,
                    axis,
                })
                .unwrap()
        };
        let xi = coordinate(source, 0);
        let eta = coordinate(source, 1);
        let x = coordinate(target, 0);
        let y = coordinate(target, 1);
        let parameter = builder.symbol(p).unwrap();
        let px = builder.mul(parameter, xi).unwrap();
        let mapped_x = builder.neg(px).unwrap();
        let py = builder.mul(parameter, eta).unwrap();
        let mapped_y = builder.add(py, py).unwrap();
        // J=diag(-p,2p). The selector order is intentionally reversed.
        let factor = builder
            .coordinate_map_factor(factor, vec![eta, xi], vec![(y, mapped_y), (x, mapped_x)])
            .unwrap();
        let typed = TypedResidual::infer(
            builder.finish([factor]).unwrap(),
            None,
            |_| None,
            RootContract::ValueRoots,
            |symbol| {
                if symbol == p {
                    Ok(ExpressionType::new(
                        ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS).unwrap(),
                        None,
                    ))
                } else if let SymbolRef::Coordinate {
                    support,
                    factor,
                    axis,
                } = symbol
                {
                    ExpressionType::coordinate(
                        &factor.erase(),
                        axis,
                        Some(&SpatialSupport::Volume {
                            domain: support.erase(),
                            dimensions: 2,
                        }),
                    )
                    .map_err(|_| ())
                } else {
                    Err(())
                }
            },
        )
        .unwrap();
        let rows = ComponentScalarization::lower(&typed).unwrap();
        let row = &rows.rows()[0];
        let values = row
            .symbols()
            .iter()
            .map(|c| if c.symbol() == p { 3. } else { 0.25 })
            .collect::<Vec<_>>();
        assert_eq!(row.evaluate(&values).unwrap(), expected);
        let roles = row
            .symbols()
            .iter()
            .map(|c| {
                if c.symbol() == p {
                    DifferentiationRole::Parameter
                } else {
                    DifferentiationRole::Frozen
                }
            })
            .collect::<Vec<_>>();
        let action = row.linearize(&values, &roles).unwrap();
        let mut output = [0.];
        action
            .jvp(RelationTangent::Parameter(&[1.]), &mut output)
            .unwrap();
        assert_eq!(output, [first]);
        action
            .vjp(&[2.], RelationCotangent::Parameter(&mut output))
            .unwrap();
        assert_eq!(output, [2. * first]);
        action
            .mixed_jvp(
                RelationTangent::Parameter(&[1.]),
                RelationTangent::Parameter(&[1.]),
                RelationTangent::Parameter(&[0.]),
                &mut output,
            )
            .unwrap();
        assert_eq!(output, [second]);
        let zero = row
            .symbols()
            .iter()
            .map(|c| if c.symbol() == p { 0. } else { 0.25 })
            .collect::<Vec<_>>();
        if expected == -18. {
            assert_eq!(row.evaluate(&zero).unwrap(), 0.);
        } else {
            assert!(row.evaluate(&zero).is_err());
        }
    }
}
