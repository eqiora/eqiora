use super::*;
use crate::{DifferentiationRole, LinearizedRelation, RelationCotangent, RelationTangent};
use eqiora_core::{Id, ScalarDomain, ValueType, entity::kinds};
use eqiora_schema::kernel::{
    ExprDagBuilder,
    typing::{RootContract, SpatialSupport},
};

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
