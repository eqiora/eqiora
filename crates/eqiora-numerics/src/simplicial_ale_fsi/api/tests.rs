use eqiora_meshing::VertexId;

use super::*;

#[test]
fn interface_action_exposes_finite_balance_and_power() {
    let action =
        AleFsiInterfaceAction::<2>::new(VertexId::new(3), [2.0, -1.0], [-2.0, 1.0]).unwrap();
    assert_eq!(action.vertex(), VertexId::new(3));
    assert_eq!(action.imbalance(), [0.0, 0.0]);
    assert_eq!(action.imbalance_norm(), 0.0);
    assert_eq!(action.fluid_power([3.0, 4.0]).unwrap(), 2.0);
    assert_eq!(action.solid_power([3.0, 4.0]).unwrap(), -2.0);
    assert_eq!(action.power_imbalance([3.0, 4.0]).unwrap(), 0.0);
    assert!(action.power_imbalance([f64::NAN, 0.0]).is_err());
}

#[test]
fn interface_action_is_dimension_typed_and_fails_closed() {
    let action =
        AleFsiInterfaceAction::<3>::new(VertexId::new(7), [2.0, -1.0, 0.5], [-2.0, 1.0, -0.5])
            .unwrap();
    assert_eq!(action.imbalance(), [0.0; 3]);
    assert_eq!(action.power_imbalance([3.0, 4.0, 2.0]).unwrap(), 0.0);
    assert!(action.fluid_power([0.0, f64::NAN, 0.0]).is_err());
    assert!(
        AleFsiInterfaceAction::<3>::new(VertexId::new(7), [f64::INFINITY, 0.0, 0.0], [0.0; 3],)
            .is_err()
    );
    assert!(AleFsiInterfaceAction::<1>::new(VertexId::new(0), [0.0], [0.0]).is_err());
}

#[test]
fn interface_action_order_is_exact_and_canonical() {
    let ordered = [
        AleFsiInterfaceAction::<3>::new(VertexId::new(2), [0.0; 3], [0.0; 3]).unwrap(),
        AleFsiInterfaceAction::<3>::new(VertexId::new(4), [0.0; 3], [0.0; 3]).unwrap(),
    ];
    assert!(validate_interface_order(&ordered).is_ok());
    assert!(validate_interface_order(&ordered.into_iter().rev().collect::<Vec<_>>()).is_err());
}
