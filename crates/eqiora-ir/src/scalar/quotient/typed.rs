//! Units belong to the retained coefficient slots, including active zeros.
use super::*;
use eqiora_core::{DimExponents, DynQuantity, ValueLiteral};

pub(in crate::scalar) fn evaluate(
    operands: &Operands,
    read: impl FnMut(ValueId) -> Result<ValueLiteral, Diagnostic>,
) -> Result<ValueLiteral, Diagnostic> {
    let (a, b) = operands.values(read)?;
    let real = |values: Vec<Option<ValueLiteral>>| {
        values
            .into_iter()
            .map(|value| {
                value
                    .map(|value| {
                        value.real_scalar_value().ok_or_else(|| {
                            ir_builder_error("quotient coefficients require real scalars")
                        })
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>, _>>()
    };
    let (a, b) = (real(a)?, real(b)?);
    let base = |values: &[Option<DynQuantity>]| {
        values[0].ok_or_else(|| ir_builder_error("quotient coefficient base is missing"))
    };
    let units = |value: Option<DimExponents>| {
        value.ok_or_else(|| ir_builder_error("quotient coefficient dimension overflow"))
    };
    let mut dimension = units(base(&a)?.dim().div(base(&b)?.dim()))?;
    // A coefficient with bit k added has one extra derivative along axis k.
    // Presence, not its sampled value, determines which unit relations exist.
    for bit in 0..a.len().trailing_zeros() {
        let step = 1usize << bit;
        let mut axis = None;
        for values in [&a, &b] {
            for index in 0..values.len() {
                if index & step != 0 {
                    continue;
                }
                if let (Some(low), Some(high)) = (values[index], values[index | step]) {
                    let relative = units(high.dim().div(low.dim()))?;
                    if axis.is_some_and(|prior| prior != relative) {
                        return Err(ir_builder_error(
                            "quotient coefficient derivative units differ",
                        ));
                    }
                    axis = Some(relative);
                }
            }
        }
        dimension =
            units(dimension.mul(
                axis.ok_or_else(|| ir_builder_error("quotient direction has no unit witness"))?,
            ))?;
    }
    let a = a.iter().map(|v| v.map(|v| v.value())).collect::<Vec<_>>();
    let b = b.iter().map(|v| v.map(|v| v.value())).collect::<Vec<_>>();
    ValueLiteral::try_from(DynQuantity::new(jet(&a, &b)?, dimension))
        .map_err(|e| ir_builder_error(e.to_string()))
}
