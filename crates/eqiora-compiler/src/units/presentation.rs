//! Presentation of stored coherent quantities through the shared unit catalog.
use super::{UnitCatalog, exact_unit};
use eqiora_core::DynQuantity;
use eqiora_lang::Expr;
use num_rational::BigRational;
use num_traits::ToPrimitive;

impl UnitCatalog {
    /// Rounding policy shared by displayed values, copy text and export metadata.
    pub const PRESENTATION_ROUNDING: &'static str = "nearest-ties-even";

    /// Convert a stored coherent-SI quantity to an explicitly chosen linear unit.
    ///
    /// The stored binary64 is interpreted exactly; conversion by the exact decimal
    /// unit scale rounds once to binary64, to nearest with ties to even. This does
    /// not modify the quantity or infer affine, phase or spectral conventions.
    /// # Errors
    /// Rejects incompatible dimensions, unavailable units, nonfinite values,
    /// overflow and nonzero underflow of the displayed value.
    pub fn display_quantity(value: DynQuantity, unit: &Expr) -> Result<f64, &'static str> {
        let (dimension, power) = exact_unit(unit)?;
        if dimension != value.dim() {
            return Err("display unit is incompatible with the stored quantity dimension");
        }
        let exact = BigRational::from_float(value.value())
            .ok_or("stored presentation quantity must be finite")?;
        if value.value() == 0.0 {
            return Ok(0.0);
        }
        // The ratio between the largest and smallest nonzero finite binary64
        // magnitudes is below 10^632. Greater scale changes must overflow or
        // underflow for every nonzero finite input; do not allocate huge powers.
        if !(-632..=632).contains(&power) {
            return Err("display quantity exceeds the nonzero finite binary64 range");
        }
        let scale = BigRational::from_integer(10u8.into()).pow(power);
        let converted = exact / scale;
        let displayed = converted
            .to_f64()
            .ok_or("display quantity cannot be represented")?;
        if !displayed.is_finite() || displayed == 0.0 {
            return Err("display quantity exceeds the nonzero finite binary64 range");
        }
        Ok(displayed)
    }

    /// Render a chosen unit with explicit fixed decimal precision and C locale.
    ///
    /// Returns the unformatted converted value and copy/accessibility text. Text
    /// uses nearest-even decimal rounding; the stored quantity is unchanged.
    /// # Errors
    /// Rejects unsupported locales, more than 17 decimal places, unit labels over
    /// 4096 bytes, and any failed typed conversion.
    pub fn present_quantity(
        value: DynQuantity,
        unit: &Expr,
        decimal_places: u8,
        locale: &str,
    ) -> Result<(f64, String), &'static str> {
        if locale != "C" {
            return Err("quantity presentation requires the explicit C locale");
        }
        if decimal_places > 17 {
            return Err("presentation precision must be at most 17 decimal places");
        }
        let displayed = Self::display_quantity(value, unit)?;
        let label = unit.to_source();
        if label.len() > 4096 {
            return Err("display unit label exceeds 4096 bytes");
        }
        let precision = usize::from(decimal_places);
        Ok((displayed, format!("{displayed:.precision$} {label}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::tests::unit;

    #[test]
    fn presentation_retains_stored_values_and_explicit_formatting() {
        let pascal = super::super::coherent_dimension("Pa").unwrap();
        let stored = DynQuantity::new(210_000_000_000., pascal);
        let before = stored.value().to_bits();
        assert_eq!(
            UnitCatalog::present_quantity(stored, &unit("GPa"), 3, "C").unwrap(),
            (210., "210.000 GPa".into())
        );
        assert_eq!(stored.value().to_bits(), before);
        for locale in ["", "de_DE", "en_US"] {
            assert!(UnitCatalog::present_quantity(stored, &unit("GPa"), 3, locale).is_err());
        }
        assert!(UnitCatalog::present_quantity(stored, &unit("GPa"), 18, "C").is_err());
        assert!(UnitCatalog::display_quantity(stored, &unit("m")).is_err());
        for (stored, text) in [(2.5, "2 Pa"), (3.5, "4 Pa")] {
            assert_eq!(
                UnitCatalog::present_quantity(
                    DynQuantity::new(stored, pascal),
                    &unit("Pa"),
                    0,
                    "C"
                )
                .unwrap()
                .1,
                text
            );
        }
    }

    #[test]
    fn presentation_scales_the_exact_binary_value_once() {
        let meters = super::super::coherent_dimension("m").unwrap();
        // Exact 1 m = 10^9 nm. Division by the rounded binary64 spelling of
        // 10^-9 instead yields the preceding representable value on this input.
        let stored = DynQuantity::new(1., meters);
        assert_eq!(
            UnitCatalog::display_quantity(stored, &unit("nm"))
                .unwrap()
                .to_bits(),
            0x41cd_cd65_0000_0000
        );
        assert_eq!(
            UnitCatalog::display_quantity(DynQuantity::new(1e-10, meters), &unit("nm"))
                .unwrap()
                .to_bits(),
            0x3fb9_9999_9999_999a
        );
        // 2^-1074 * 10^324, independently rounded from the exact rational,
        // stays finite even though 10^324 cannot itself be stored in binary64.
        let tiny = DynQuantity::new(f64::from_bits(1), meters.pow(36, 1).unwrap());
        assert_eq!(
            UnitCatalog::display_quantity(tiny, &unit("nm ^ 36"))
                .unwrap()
                .to_bits(),
            0x4013_c33b_7256_9c63
        );
        assert!(
            UnitCatalog::display_quantity(DynQuantity::new(f64::MAX, meters), &unit("nm")).is_err()
        );
        assert!(
            UnitCatalog::display_quantity(DynQuantity::new(f64::from_bits(1), meters), &unit("km"))
                .is_err()
        );
        for value in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            assert!(
                UnitCatalog::display_quantity(DynQuantity::new(value, meters), &unit("m")).is_err()
            );
        }
    }
}
