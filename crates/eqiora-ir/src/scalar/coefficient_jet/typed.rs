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
                            ir_builder_error("solve coefficients require real scalars")
                        })
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>, _>>()
    };
    let (a, b) = (real(a)?, real(b)?);
    let dimension = dimension(
        operands.shape,
        &a.iter().map(|v| v.map(|v| v.dim())).collect::<Vec<_>>(),
        &b.iter().map(|v| v.map(|v| v.dim())).collect::<Vec<_>>(),
    )?;
    let a = a.iter().map(|v| v.map(|v| v.value())).collect::<Vec<_>>();
    let b = b.iter().map(|v| v.map(|v| v.value())).collect::<Vec<_>>();
    ValueLiteral::try_from(DynQuantity::new(operands.shape.jet(&a, &b)?, dimension))
        .map_err(|e| ir_builder_error(e.to_string()))
}

pub(in crate::scalar) fn dimension(
    shape: Shape,
    a: &[Option<DimExponents>],
    b: &[Option<DimExponents>],
) -> Result<DimExponents, Diagnostic> {
    let width = shape.width();
    let dimensions = |values: &[Option<DimExponents>]| {
        values
            .chunks_exact(width)
            .map(|block| {
                let mut dimension = None;
                for value in block.iter().flatten() {
                    if dimension.is_some_and(|prior| prior != *value) {
                        return Err(ir_builder_error(
                            "solve coefficient matrix entry units differ",
                        ));
                    }
                    dimension = Some(*value);
                }
                Ok(dimension)
            })
            .collect::<Result<Vec<_>, Diagnostic>>()
    };
    let (ad, bd) = (dimensions(a)?, dimensions(b)?);
    let base = |values: &[Option<DimExponents>]| {
        values[0].ok_or_else(|| ir_builder_error("solve coefficient base has no unit witness"))
    };
    let units = |value: Option<DimExponents>| {
        value.ok_or_else(|| ir_builder_error("solve coefficient dimension overflow"))
    };
    let mut dimension = match shape {
        Shape::Solve { .. } => units(base(&ad)?.div(base(&bd)?))?,
        Shape::ComplexMath { function, .. } => {
            for (a, b) in ad.iter().zip(&bd) {
                if let (Some(a), Some(b)) = (a, b)
                    && a != b
                {
                    return Err(ir_builder_error(
                        "complex coefficient coordinate units differ",
                    ));
                }
            }
            let ty = eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Complex, base(&ad)?)
                .map_err(|error| ir_builder_error(error.to_string()))?;
            eqiora_schema::kernel::typing::unary_math(
                function,
                &eqiora_schema::kernel::typing::ExpressionType::<()>::new(ty, None),
            )
            .map_err(|error| ir_builder_error(error.to_string()))?
            .dimension()
        }
    };
    // A coefficient with bit k added has one extra derivative along axis k.
    // Presence, not its sampled value, determines which unit relations exist.
    for bit in 0..ad.len().trailing_zeros() {
        let step = 1usize << bit;
        let mut axis = None;
        for values in [&ad, &bd] {
            for index in 0..values.len() {
                if index & step != 0 {
                    continue;
                }
                if let (Some(low), Some(high)) = (values[index], values[index | step]) {
                    let relative = units(high.div(low))?;
                    if axis.is_some_and(|prior| prior != relative) {
                        return Err(ir_builder_error(
                            "solve coefficient derivative units differ",
                        ));
                    }
                    axis = Some(relative);
                }
            }
        }
        dimension = units(
            dimension
                .mul(axis.ok_or_else(|| ir_builder_error("solve direction has no unit witness"))?),
        )?;
    }
    Ok(dimension)
}
