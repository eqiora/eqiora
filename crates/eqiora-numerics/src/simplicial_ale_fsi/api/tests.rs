use eqiora_meshing::VertexId;

use super::*;

#[test]
fn interface_action_norm_retains_finite_dyadic_scales() {
    // The exact 3-4-5 identity applies before and after any binary scaling.
    for exponent in [-1070, -700, 0, 700, 1020] {
        let scale = if exponent < -1022 {
            f64::from_bits(1 << (exponent + 1074))
        } else {
            2.0_f64.powi(exponent)
        };
        let action =
            AleFsiInterfaceAction::<2>::new(VertexId::new(0), [3.0 * scale, 4.0 * scale], [0.0; 2])
                .unwrap();
        assert_eq!(action.imbalance_norm().unwrap(), 5.0 * scale);
        assert_eq!(interface_action_norm(&[action]).unwrap(), 5.0 * scale);
    }
}

#[test]
fn interface_action_exposes_finite_balance_and_power() {
    let action =
        AleFsiInterfaceAction::<2>::new(VertexId::new(3), [2.0, -1.0], [-2.0, 1.0]).unwrap();
    assert_eq!(action.vertex(), VertexId::new(3));
    assert_eq!(action.imbalance(), [0.0, 0.0]);
    assert_eq!(action.imbalance_norm().unwrap(), 0.0);
    assert_eq!(action.fluid_power([3.0, 4.0]).unwrap(), 2.0);
    assert_eq!(action.solid_power([3.0, 4.0]).unwrap(), -2.0);
    assert_eq!(action.power_imbalance([3.0, 4.0]).unwrap(), 0.0);
    assert!(action.power_imbalance([f64::NAN, 0.0]).is_err());
}

#[test]
fn interface_action_norm_rejects_only_unrepresentable_magnitudes() {
    let action =
        |fluid, solid| AleFsiInterfaceAction::<2>::new(VertexId::new(0), fluid, solid).unwrap();
    assert_eq!(
        action([f64::MAX, 0.0], [0.0; 2]).imbalance_norm().unwrap(),
        f64::MAX
    );
    // Finite components can still have a norm or sum outside binary64.
    for invalid in [
        action([f64::MAX; 2], [0.0; 2]),
        action([f64::MAX, 0.0], [f64::MAX, 0.0]),
    ] {
        assert!(invalid.imbalance_norm().is_err());
        assert!(interface_action_norm(&[invalid]).is_err());
    }
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
