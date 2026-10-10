use super::*;
use crate::kernel::{CoordinateMapFactor, ExprDagBuilder, SymbolRef};
use eqiora_core::{DimExponents, DynQuantity};

#[test]
fn mapped_density_retains_both_inventory_and_field_transport_rates() {
    let mut b = ExprDagBuilder::new();
    let reference = Id::new();
    let target = Id::new();
    let xi = b.coordinate(reference, reference, 0).unwrap();
    let x = b.coordinate(target, target, 0).unwrap();
    let eta = b.coordinate(reference, reference, 1).unwrap();
    let y = b.coordinate(target, target, 1).unwrap();
    let time = b.symbol(SymbolRef::Time).unwrap();
    let zero = b
        .constant(DynQuantity::new(0.0, DimExponents::DIMENSIONLESS))
        .unwrap();
    let one = b
        .constant(DynQuantity::new(1.0, DimExponents::DIMENSIONLESS))
        .unwrap();
    let field = Id::new();
    let q = b.symbol(SymbolRef::Field(field)).unwrap();
    let qt = b
        .symbol(SymbolRef::Derivative(field, std::num::NonZeroU32::MIN))
        .unwrap();
    let qx = b.coordinate_partial(q, x).unwrap();
    let lambda = b.add(one, time).unwrap();
    let chi = b.mul(lambda, xi).unwrap();
    let mapped = b
        .pullback(q, vec![xi, eta], vec![(x, chi), (y, eta)])
        .unwrap();
    let mapped_t = b
        .pullback(qt, vec![xi, eta], vec![(x, chi), (y, eta)])
        .unwrap();
    let mapped_x = b
        .pullback(qx, vec![xi, eta], vec![(x, chi), (y, eta)])
        .unwrap();
    let transport = b.mul(mapped_x, xi).unwrap();
    let scalar_rate = b.add(mapped_t, transport).unwrap();
    let jacobian = b
        .coordinate_map_factor(
            CoordinateMapFactor::VolumeScale,
            vec![xi, eta],
            vec![(x, chi), (y, eta)],
        )
        .unwrap();
    let jacobian_rate = b
        .coordinate_map_factor_action(jacobian, time, vec![xi, zero])
        .unwrap();
    let storage = b.mul(mapped, jacobian).unwrap();
    let field_rate = b.mul(scalar_rate, jacobian).unwrap();
    let volume_rate = b.mul(mapped, jacobian_rate).unwrap();
    let correct = b.add(field_rate, volume_rate).unwrap();
    // d(J q(chi,t))/dt = J (q_t(chi,t) + q_x(chi,t) xi) + J_t q(chi,t).
    // Each probe preserves all other terms and changes only one exact role.
    let missing_transport = b.mul(mapped_t, jacobian).unwrap();
    let missing_transport = b.add(missing_transport, volume_rate).unwrap();
    let other_chi = b.add(chi, one).unwrap();
    let wrong_mapped_x = b
        .pullback(qx, vec![xi, eta], vec![(x, other_chi), (y, eta)])
        .unwrap();
    let wrong_transport = b.mul(wrong_mapped_x, xi).unwrap();
    let wrong_rate = b.add(mapped_t, wrong_transport).unwrap();
    let wrong_rate = b.mul(wrong_rate, jacobian).unwrap();
    let wrong_map = b.add(wrong_rate, volume_rate).unwrap();
    let other_field = b
        .symbol(SymbolRef::Derivative(Id::new(), std::num::NonZeroU32::MIN))
        .unwrap();
    let other_t = b
        .pullback(other_field, vec![xi, eta], vec![(x, chi), (y, eta)])
        .unwrap();
    let wrong_rate = b.add(other_t, transport).unwrap();
    let wrong_rate = b.mul(wrong_rate, jacobian).unwrap();
    let wrong_field = b.add(wrong_rate, volume_rate).unwrap();
    let qy = b.coordinate_partial(q, y).unwrap();
    let wrong_y = b
        .pullback(qy, vec![xi, eta], vec![(x, chi), (y, eta)])
        .unwrap();
    let wrong_transport = b.mul(wrong_y, xi).unwrap();
    let wrong_rate = b.add(mapped_t, wrong_transport).unwrap();
    let wrong_rate = b.mul(wrong_rate, jacobian).unwrap();
    let wrong_axis = b.add(wrong_rate, volume_rate).unwrap();
    let missing_time = b.mul(transport, jacobian).unwrap();
    let missing_time = b.add(missing_time, volume_rate).unwrap();
    let canceled = b.sub(mapped, mapped).unwrap();
    let nested_canceled = b
        .pullback(canceled, vec![xi, eta], vec![(xi, xi), (eta, eta)])
        .unwrap();
    let fixed = b
        .pullback(q, vec![xi, eta], vec![(x, xi), (y, eta)])
        .unwrap();
    let fixed_rate = b
        .pullback(qt, vec![xi, eta], vec![(x, xi), (y, eta)])
        .unwrap();
    let dag = b.finish([correct]).unwrap();
    assert_eq!(dag.verify_time_derivative(storage, correct), Ok(()));
    assert_eq!(dag.verify_time_derivative(mapped, scalar_rate), Ok(()));
    assert_eq!(dag.verify_time_derivative(fixed, fixed_rate), Ok(()));
    assert_eq!(
        dag.verify_time_derivative(nested_canceled, zero),
        Err(TimeDerivativeProofError::UnsupportedExpression)
    );
    for wrong in [
        field_rate,
        missing_transport,
        missing_time,
        wrong_map,
        wrong_field,
        wrong_axis,
    ] {
        assert_eq!(
            dag.verify_time_derivative(storage, wrong),
            Err(TimeDerivativeProofError::Mismatch)
        );
    }
}

#[test]
fn nested_explicit_polynomial_pullbacks_keep_their_first_rate() {
    let mut b = ExprDagBuilder::new();
    let reference = Id::new();
    let target = Id::new();
    let xi = b.coordinate(reference, reference, 0).unwrap();
    let x = b.coordinate(target, target, 0).unwrap();
    let time = b.symbol(SymbolRef::Time).unwrap();
    let row = b.add(xi, time).unwrap();
    let square = b.powi(x, 2).unwrap();
    let inner = b.pullback(square, vec![xi], vec![(x, row)]).unwrap();
    let outer = b.pullback(inner, vec![xi], vec![(xi, xi)]).unwrap();
    let two = b
        .constant(DynQuantity::new(2.0, DimExponents::DIMENSIONLESS))
        .unwrap();
    let rate = b.mul(two, row).unwrap();
    let dag = b.finish([rate]).unwrap();
    assert_eq!(dag.verify_time_derivative(outer, rate), Ok(()));
}
