//! Validate closed spatial constants before any live-source or numerical use.
use super::{AuthoredFormExpressionV1 as E, Diagnostic, WireForm, rejection};
use eqiora_core::{DimExponents, ScalarDomain, ValueFrame, ValueLiteral, ValueShape, ValueType};

pub(super) fn literal(
    complex: bool,
    dimension: &[(i32, i32); 7],
    shape: &[u32],
    values: &[(f64, f64)],
) -> Result<ValueLiteral, Diagnostic> {
    if !(1..=2).contains(&shape.len()) || shape.iter().any(|extent| *extent != shape[0]) {
        return Err(rejection(
            "spatial coefficient requires a vector or square matrix",
        ));
    }
    let dimension = DimExponents::from_rationals(*dimension)
        .filter(|value| value.exponents() == *dimension)
        .ok_or_else(|| {
            rejection("spatial coefficient requires canonical rational SI dimensions")
        })?;
    let shape =
        ValueShape::new(shape.iter().copied()).map_err(|error| rejection(&error.to_string()))?;
    let ty = ValueType::shaped(
        if complex {
            ScalarDomain::Complex
        } else {
            ScalarDomain::Real
        },
        dimension,
        shape,
        ValueFrame::SpatialCartesian,
    )
    .map_err(|error| rejection(&error.to_string()))?;
    ValueLiteral::new(ty, values.iter().copied()).map_err(|error| rejection(&error.to_string()))
}

pub(super) fn check(wire: &WireForm) -> Result<(), Diagnostic> {
    fn visit(value: &E, remaining: &mut usize, depth: usize) -> Result<(), Diagnostic> {
        if depth > 96 {
            return Err(rejection(
                "projection expression exceeds the structural bound",
            ));
        }
        *remaining = remaining
            .checked_sub(1)
            .ok_or_else(|| rejection("projection expression exceeds the structural bound"))?;
        if let E::SpatialTensor {
            complex,
            dimension,
            shape,
            values,
        } = value
        {
            *remaining = remaining
                .checked_sub(values.len())
                .ok_or_else(|| rejection("spatial coefficient exceeds the structural bound"))?;
            literal(*complex, dimension, shape, values)?;
        }
        for child in super::field_traces::children(value) {
            visit(child, remaining, depth + 1)?;
        }
        Ok(())
    }
    let mut remaining = 65536;
    for (_, left, right) in &wire.equations {
        visit(left, &mut remaining, 0)?;
        visit(right, &mut remaining, 0)?;
    }
    if let Some(gauge) = &wire.gauge {
        for value in [
            &gauge.reference.0,
            &gauge.reference.1,
            &gauge.compatibility.0,
            &gauge.compatibility.1,
        ] {
            visit(value, &mut remaining, 0)?;
        }
    }
    Ok(())
}
