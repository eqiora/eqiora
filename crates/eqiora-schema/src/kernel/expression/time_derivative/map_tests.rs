use super::*;
use crate::kernel::{CoordinateMapFactor, ExprDagBuilder, SymbolRef};
use eqiora_core::{DimExponents, DynQuantity};

fn constant(builder: &mut ExprDagBuilder, value: f64) -> ExprId {
    builder
        .constant(DynQuantity::new(value, DimExponents::DIMENSIONLESS))
        .unwrap()
}

#[test]
fn storage_map_action_requires_the_exact_motion_and_independent_row_rates() {
    let mut builder = ExprDagBuilder::new();
    let reference = Id::new();
    let target = Id::new();
    let xi = builder.coordinate(reference, reference, 0).unwrap();
    let x = builder.coordinate(target, target, 0).unwrap();
    let time = builder.symbol(SymbolRef::Time).unwrap();
    let parameter = builder.symbol(SymbolRef::Parameter(Id::new())).unwrap();
    let one = constant(&mut builder, 1.0);
    let two = constant(&mut builder, 2.0);
    let zero = constant(&mut builder, 0.0);
    let lambda = builder.add(one, time).unwrap();
    let row = builder.mul(lambda, xi).unwrap();
    let factor = builder
        .coordinate_map_factor(CoordinateMapFactor::VolumeScale, vec![xi], vec![(x, row)])
        .unwrap();
    let storage = builder.mul(two, factor).unwrap();
    // A distinct DAG occurrence of the same exact map must share proof identity.
    let duplicate = builder
        .coordinate_map_factor(CoordinateMapFactor::VolumeScale, vec![xi], vec![(x, row)])
        .unwrap();
    let action = builder
        .coordinate_map_factor_action(duplicate, time, vec![xi])
        .unwrap();
    let correct = builder.mul(two, action).unwrap();
    let twice_xi = builder.mul(two, xi).unwrap();
    let wrong_direction = builder
        .coordinate_map_factor_action(factor, time, vec![twice_xi])
        .unwrap();
    let wrong_parameter = builder
        .coordinate_map_factor_action(factor, parameter, vec![xi])
        .unwrap();
    // Keep the inventory coefficient correct so these probes can fail only on
    // the false direction or parameter correspondence, not on a missing factor 2.
    let wrong_direction = builder.mul(two, wrong_direction).unwrap();
    let wrong_parameter = builder.mul(two, wrong_parameter).unwrap();
    let twice_time = builder.mul(two, time).unwrap();
    let other_lambda = builder.add(one, twice_time).unwrap();
    let other_row = builder.mul(other_lambda, xi).unwrap();
    let other_map = builder
        .coordinate_map_factor(
            CoordinateMapFactor::VolumeScale,
            vec![xi],
            vec![(x, other_row)],
        )
        .unwrap();
    let other_action = builder
        .coordinate_map_factor_action(other_map, time, vec![twice_xi])
        .unwrap();
    let other_inventory_rate = builder.mul(two, other_action).unwrap();
    let dag = builder.finish([correct]).unwrap();
    assert_eq!(dag.verify_time_derivative(storage, correct), Ok(()));
    for wrong in [zero, wrong_direction, wrong_parameter, other_inventory_rate] {
        assert_eq!(
            dag.verify_time_derivative(storage, wrong),
            Err(TimeDerivativeProofError::Mismatch)
        );
    }
}

#[test]
fn explicit_pullback_uses_simultaneous_coordinate_substitution() {
    let mut builder = ExprDagBuilder::new();
    let reference = Id::new();
    let target = Id::new();
    let xi = builder.coordinate(reference, reference, 0).unwrap();
    let x = builder.coordinate(target, target, 0).unwrap();
    let time = builder.symbol(SymbolRef::Time).unwrap();
    let one = constant(&mut builder, 1.0);
    let two = constant(&mut builder, 2.0);
    let lambda = builder.add(one, time).unwrap();
    let row = builder.mul(lambda, xi).unwrap();
    let square = builder.powi(x, 2).unwrap();
    let physical = builder.add(square, time).unwrap();
    let storage = builder
        .pullback(physical, vec![xi], vec![(x, row)])
        .unwrap();
    // ((1+t)*xi)^2+t has rate 2*(1+t)*xi^2+1.
    let xi_squared = builder.powi(xi, 2).unwrap();
    let scale = builder.mul(two, lambda).unwrap();
    let mapped_rate = builder.mul(scale, xi_squared).unwrap();
    let correct = builder.add(mapped_rate, one).unwrap();
    let dag = builder.finish([correct]).unwrap();
    assert_eq!(dag.verify_time_derivative(storage, correct), Ok(()));
    assert_eq!(
        dag.verify_time_derivative(storage, one),
        Err(TimeDerivativeProofError::Mismatch)
    );
}

#[test]
fn mapped_unknown_fields_do_not_gain_admission_by_cancellation() {
    let mut builder = ExprDagBuilder::new();
    let reference = Id::new();
    let target = Id::new();
    let xi = builder.coordinate(reference, reference, 0).unwrap();
    let x = builder.coordinate(target, target, 0).unwrap();
    let field = builder.symbol(SymbolRef::Field(Id::new())).unwrap();
    let canceled = builder.sub(field, field).unwrap();
    let storage = builder.pullback(canceled, vec![xi], vec![(x, xi)]).unwrap();
    let zero = constant(&mut builder, 0.0);
    let dag = builder.finish([zero]).unwrap();
    assert_eq!(
        dag.verify_time_derivative(storage, zero),
        Err(TimeDerivativeProofError::UnsupportedExpression)
    );
}
