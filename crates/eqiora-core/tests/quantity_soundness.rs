//! The soundness boundary of the two-layer unit system is the
//! load-bearing guarantee of `eqiora-core`; these tests are its contract.

use eqiora_core::quantity::aliases::{Length, Pressure, Time, Velocity};
use eqiora_core::quantity::{DynQuantity, dim};
use eqiora_core::{Diagnostic, Dimension};

#[test]
fn promotion_accepts_matching_dimension() {
    let raw = DynQuantity::new(101_325.0, dim::PressureDim::EXPONENTS);
    let p: Pressure = raw.checked_cast().expect("dimensions match");
    assert_eq!(p.value(), 101_325.0);
}

#[test]
fn promotion_rejects_mismatch_with_stable_code() {
    let raw = DynQuantity::new(3.0, dim::LengthDim::EXPONENTS);
    let err: Diagnostic = raw.checked_cast::<dim::PressureDim>().unwrap_err();
    assert_eq!(err.code().0, "EQ0401");
}

#[test]
fn demotion_roundtrip_is_lossless() {
    let p = Pressure::new(2.5);
    let back: Pressure = p.into_dyn().checked_cast().expect("roundtrip");
    assert_eq!(back, p);
}

#[test]
fn dynamic_arithmetic_combines_dimensions() {
    let v = Velocity::new(3.0).into_dyn();
    let t = Time::new(2.0).into_dyn();
    let l: Length = v.try_mul(t).unwrap().checked_cast().expect("L·T⁻¹ × T = L");
    assert_eq!(l.value(), 6.0);
}

#[test]
fn dynamic_addition_requires_equal_dimensions() {
    let v = Velocity::new(1.0).into_dyn();
    let t = Time::new(1.0).into_dyn();
    assert!(v.try_add(t).is_err());
    assert!(v.try_add(v).is_ok());
}

#[test]
fn static_scaling_preserves_dimension() {
    let v = Velocity::new(12.0);
    let scaled = 0.5 * v * 2.0;
    assert_eq!(scaled.value(), 12.0);
}

#[test]
fn complex_quantities_have_arithmetic_without_ordering() {
    use eqiora_core::quantity::{Quantity, dim};
    use num_complex::Complex64;

    let value = Quantity::<Complex64, dim::LengthDim>::from_scalar(Complex64::new(3.0, 4.0));
    let divisor = Complex64::new(1.0, -2.0);
    // (3 + 4i) / (1 - 2i) = (-5 + 10i) / 5.
    assert_eq!((value / divisor).value(), Complex64::new(-1.0, 2.0));
    assert_eq!((value * divisor).value(), Complex64::new(11.0, -2.0));
    assert_eq!((value + value).value(), Complex64::new(6.0, 8.0));
    assert_eq!((-value).value(), Complex64::new(-3.0, -4.0));
    assert_eq!(value, value);
}
