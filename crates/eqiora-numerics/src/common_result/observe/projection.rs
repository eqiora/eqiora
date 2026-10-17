//! Select mathematical components through the existing typed value owner.
use super::{CommonObservation, Diagnostic, DynQuantity, invalid};
use eqiora_core::{ScalarDomain, ValueLiteral, ValueType};

impl CommonObservation {
    /// Select a declared row-major component without inventing a physical convention.
    /// Phase is the principal argument in radians, undefined exactly at zero.
    /// # Errors
    /// Rejects unknown projections, invalid indices and unsupported numerical ranges.
    pub fn project_component(
        &self,
        projection: &str,
        index: usize,
    ) -> Result<Option<DynQuantity>, Diagnostic> {
        let value = match projection {
            "real" => self.value.component_real(index).map(Some),
            "imaginary" => self.value.component_imaginary(index).map(Some),
            "magnitude" => self.value.component_magnitude(index).map(Some),
            "squared_magnitude" => self.value.component_squared_magnitude(index).map(Some),
            "phase" => self.value.component_phase(index),
            _ => {
                return Err(invalid(
                    "projection must be real, imaginary, magnitude, squared_magnitude or phase",
                ));
            }
        };
        value.map_err(|error| invalid(error.to_string()))
    }
}

fn scalar_width(value_type: &ValueType) -> Result<usize, Diagnostic> {
    match value_type.scalar_domain() {
        ScalarDomain::Real => Ok(1),
        ScalarDomain::Complex => Ok(2),
        _ => Err(invalid(
            "numeric observation requires real or complex coordinates",
        )),
    }
}

pub(super) fn coordinate_index(
    coordinate: &eqiora_ir::ScalarSymbolCoordinate,
    value_type: &ValueType,
) -> Result<usize, Diagnostic> {
    let width = scalar_width(value_type)?;
    let component = value_type
        .shape()
        .flat_index(coordinate.component_index())
        .ok_or_else(|| invalid("Observable component is unavailable"))?;
    if coordinate.is_imaginary() && width == 1 {
        return Err(invalid("a real Observable has no imaginary coordinate"));
    }
    Ok(component * width + usize::from(coordinate.is_imaginary()))
}

pub(super) fn coordinate_count(value_type: &ValueType) -> Result<usize, Diagnostic> {
    value_type
        .shape()
        .component_count()
        .and_then(|count| count.checked_mul(scalar_width(value_type).ok()?))
        .ok_or_else(|| invalid("numeric observation coordinate count is unavailable"))
}

pub(super) fn from_coordinates(
    value_type: &ValueType,
    coordinates: &[f64],
) -> Result<ValueLiteral, Diagnostic> {
    let width = scalar_width(value_type)?;
    if coordinates.len() != coordinate_count(value_type)? {
        return Err(invalid(
            "numeric observation differs from its complete value type",
        ));
    }
    ValueLiteral::new(
        value_type.clone(),
        coordinates
            .chunks_exact(width)
            .map(|part| (part[0], part.get(1).copied().unwrap_or(0.))),
    )
    .map_err(|error| invalid(error.to_string()))
}
