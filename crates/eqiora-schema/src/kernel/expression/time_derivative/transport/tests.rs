use super::*;
use crate::kernel::{
    ExprDagBuilder,
    typing::{ExpressionType, RootContract, SpatialSupport},
};
use eqiora_core::DynQuantity;

#[test]
fn a_foreign_parameter_cannot_supply_the_exact_map_rate() {
    let mut builder = ExprDagBuilder::new();
    let reference = Id::new();
    let target = Id::new();
    let field = Id::new();
    let alpha = Id::new();
    let other_alpha = Id::new();
    let velocity_x = Id::new();
    let velocity_y = Id::new();
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let time_dimension = DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).unwrap();
    let rate_dimension = DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).unwrap();
    let velocity_dimension = DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).unwrap();
    let xi = builder.coordinate(reference, reference, 0).unwrap();
    let eta = builder.coordinate(reference, reference, 1).unwrap();
    let x = builder.coordinate(target, target, 0).unwrap();
    let y = builder.coordinate(target, target, 1).unwrap();
    let state = builder.symbol(SymbolRef::Field(field)).unwrap();
    let rate = builder.symbol(SymbolRef::Parameter(alpha)).unwrap();
    let foreign_rate = builder.symbol(SymbolRef::Parameter(other_alpha)).unwrap();
    let vx = builder.symbol(SymbolRef::Parameter(velocity_x)).unwrap();
    let vy = builder.symbol(SymbolRef::Parameter(velocity_y)).unwrap();
    let time = builder.symbol(SymbolRef::Time).unwrap();
    let one = builder
        .constant(DynQuantity::new(1., DimExponents::DIMENSIONLESS))
        .unwrap();
    let elapsed = builder.mul(rate, time).unwrap();
    let scale = builder.add(one, elapsed).unwrap();
    let mapped_x = builder.mul(scale, xi).unwrap();
    let mapped_y = builder.mul(scale, eta).unwrap();
    // Reverse both selector inventories: correspondence must use axis identity.
    let map = builder
        .coordinate_map_factor(
            CoordinateMapFactor::VolumeScale,
            vec![eta, xi],
            vec![(y, mapped_y), (x, mapped_x)],
        )
        .unwrap();
    let capacity = builder
        .constant(DynQuantity::new(
            3.,
            DimExponents::from_integers([0, -2, 1, 0, 0, 0, 0]).unwrap(),
        ))
        .unwrap();
    let density = builder.mul(capacity, state).unwrap();
    let stored = builder.mul(density, map).unwrap();
    let prefactor = builder.mul(density, scale).unwrap();
    // d((1+alpha*t)*xi)/dt=alpha*xi, independently of the evaluator.
    let mesh_x = builder.mul(rate, xi).unwrap();
    let mesh_y = builder.mul(rate, eta).unwrap();
    let wrong_mesh_x = builder.mul(foreign_rate, xi).unwrap();
    let basis_x = builder.push(ExprNode::Gradient(xi)).unwrap();
    let basis_y = builder.push(ExprNode::Gradient(eta)).unwrap();
    let (correct, wrong) = {
        let mut flux = |mesh_x| {
            let relative_x = builder.sub(vx, mesh_x).unwrap();
            let relative_y = builder.sub(vy, mesh_y).unwrap();
            let x = builder.mul(relative_x, basis_x).unwrap();
            let y = builder.mul(relative_y, basis_y).unwrap();
            let velocity = builder.add(x, y).unwrap();
            builder.mul(prefactor, velocity).unwrap()
        };
        (flux(mesh_x), flux(wrong_mesh_x))
    };
    let fixed_map = builder
        .coordinate_map_factor(
            CoordinateMapFactor::VolumeScale,
            vec![xi, eta],
            vec![(x, xi), (y, eta)],
        )
        .unwrap();
    let fixed_stored = builder.mul(density, fixed_map).unwrap();
    let zero_capacity = builder
        .constant(DynQuantity::new(
            0.,
            DimExponents::from_integers([0, -2, 1, 0, 0, 0, 0]).unwrap(),
        ))
        .unwrap();
    let capacity_difference = builder.sub(capacity, zero_capacity).unwrap();
    let spurious_density = builder.mul(capacity_difference, state).unwrap();
    let speed = builder
        .constant(DynQuantity::new(3., velocity_dimension))
        .unwrap();
    let zero_speed = builder
        .constant(DynQuantity::new(0., velocity_dimension))
        .unwrap();
    let relative = builder.sub(speed, zero_speed).unwrap();
    let axes = builder.add(basis_x, basis_y).unwrap();
    let valid_velocity = builder.mul(relative, axes).unwrap();
    let omitted_relative = builder.mul(speed, axes).unwrap();
    let fixed_correct = builder.mul(density, valid_velocity).unwrap();
    let wrong_dimensions = builder.mul(spurious_density, omitted_relative).unwrap();
    let typed = TypedResidual::infer(
        builder
            .finish([
                stored,
                correct,
                wrong,
                fixed_stored,
                fixed_correct,
                wrong_dimensions,
            ])
            .unwrap(),
        None,
        |_| None,
        RootContract::ValueRoots,
        |symbol| {
            let (dimension, support) = match symbol {
                SymbolRef::Field(id) if id == field => {
                    (DimExponents::DIMENSIONLESS, Some(reference))
                }
                SymbolRef::Coordinate { support, .. } => (length, Some(support)),
                SymbolRef::Time => (time_dimension, None),
                SymbolRef::Parameter(id) if id == velocity_x || id == velocity_y => {
                    (velocity_dimension, None)
                }
                SymbolRef::Parameter(id) if id == alpha || id == other_alpha => {
                    (rate_dimension, None)
                }
                _ => panic!("unexpected input"),
            };
            Ok::<_, ()>(ExpressionType::scalar(
                dimension,
                support.map(|domain| SpatialSupport::Volume {
                    domain: domain.erase(),
                    dimensions: 2,
                }),
            ))
        },
    )
    .unwrap();
    assert_eq!(
        typed.verify_uniform_ale_transport(map, stored, correct, field),
        Ok([vx, vy])
    );
    // Even if alpha and other_alpha are later both bound to 1/2, they remain
    // independent inputs. A value sample cannot authorize this replacement.
    assert_eq!(
        typed.verify_uniform_ale_transport(map, stored, wrong, field),
        Err(Error::Mismatch)
    );
    assert_eq!(
        typed.verify_uniform_ale_transport(fixed_map, fixed_stored, fixed_correct, field),
        Ok([speed, speed])
    );
    // Both fluxes normalize to the same numbers. Only the correct expression
    // subtracts a mesh rate with velocity dimensions; (3-0) in capacity units
    // cannot declare a material velocity for the fixed map's zero rate.
    assert_eq!(
        typed.verify_uniform_ale_transport(fixed_map, fixed_stored, wrong_dimensions, field),
        Err(Error::Mismatch)
    );
}
