//! Bounded adaptation to the compiler's single static signature binding owner.

use eqiora::compiler::StaticBindingValue;
use eqiora::geometry::{CanonicalGeometryV1, NamedEntitySet};
use eqiora::kernel::ClockDomainDef;
use eqiora::language::Expr;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString, PyTuple};

use crate::clock::PyClockDomain;
use crate::geometry::PyGeometrySelection;

pub(crate) enum OwnedBinding<'g> {
    CoordinateInterval(eqiora::kernel::AxisBounds),
    Expression(Expr),
    Value(eqiora::ValueLiteral),
    Clock(ClockDomainDef),
    Support(String, Option<String>),
    CompleteExterior(Vec<&'g NamedEntitySet>, &'g NamedEntitySet),
}

impl OwnedBinding<'_> {
    pub(crate) fn borrowed<'a>(
        &'a self,
        geometry: Option<&'a CanonicalGeometryV1>,
    ) -> StaticBindingValue<'a> {
        match self {
            Self::CoordinateInterval(bounds) => StaticBindingValue::CoordinateInterval(*bounds),
            Self::Expression(expression) => StaticBindingValue::Expression(expression),
            Self::Value(value) => StaticBindingValue::Value(value),
            Self::Clock(clock) => StaticBindingValue::Clock(clock),
            Self::Support(selection, parent) => {
                let geometry = geometry.expect("support extraction checks exact authority");
                StaticBindingValue::GeometrySupport {
                    geometry,
                    selection: geometry
                        .entity_set(selection)
                        .expect("validated immutable selection"),
                    parent: parent.as_ref().map(|name| {
                        geometry
                            .entity_set(name)
                            .expect("validated immutable parent")
                    }),
                }
            }
            Self::CompleteExterior(members, parent) => StaticBindingValue::CompleteExterior {
                geometry: geometry.expect("support extraction checks exact authority"),
                members,
                parent,
            },
        }
    }
}

fn selection(value: &Bound<'_, PyAny>, geometry: Option<&CanonicalGeometryV1>) -> PyResult<String> {
    let selected = value.extract::<PyRef<'_, PyGeometrySelection>>()?;
    let geometry = geometry
        .ok_or_else(|| PyValueError::new_err("Geometry selections require geometry= authority"))?;
    if selected.bound_source_digest() != crate::geometry::digest_to_hex(&geometry.digest_bytes()) {
        return Err(PyValueError::new_err(
            "Geometry selection belongs to a different exact Geometry revision",
        ));
    }
    geometry
        .entity_set(selected.canonical_name())
        .map(|_| selected.canonical_name().to_owned())
        .ok_or_else(|| PyValueError::new_err("Geometry selection is absent from its authority"))
}

pub(crate) fn extract<'g>(
    bindings: Option<&Bound<'_, PyDict>>,
    geometry: Option<&'g CanonicalGeometryV1>,
) -> PyResult<Vec<(String, OwnedBinding<'g>)>> {
    let Some(bindings) = bindings else {
        return Ok(Vec::new());
    };
    if bindings.len() > 256 {
        return Err(PyValueError::new_err(
            "static bindings exceed the 256-declaration limit",
        ));
    }
    let mut output = Vec::with_capacity(bindings.len());
    for (name, value) in bindings.iter() {
        let name = name
            .cast::<PyString>()
            .map_err(|_| PyTypeError::new_err("static binding names must be strings"))?;
        let name = name.to_str()?;
        if name.is_empty() || name.len() > 1024 {
            return Err(PyValueError::new_err(
                "static binding names must contain 1 to 1024 UTF-8 bytes",
            ));
        }
        let name = name.to_owned();
        let binding = if let Ok(value) =
            value.extract::<PyRef<'_, crate::modeling::enumeration::PyEnumValue>>()
        {
            OwnedBinding::Value(value.value.clone())
        } else if let Ok(interval) =
            value.extract::<PyRef<'_, crate::coordinate_interval::PyCoordinateInterval>>()
        {
            OwnedBinding::CoordinateInterval(interval.bounds)
        } else if let Ok(clock) = value.extract::<PyRef<'_, PyClockDomain>>() {
            OwnedBinding::Clock(clock.value.clone())
        } else if value.is_instance_of::<PyGeometrySelection>() {
            OwnedBinding::Support(selection(&value, geometry)?, None)
        } else if let Ok(pair) = value.cast::<PyTuple>() {
            if pair.len() == 2 && pair.get_item(0)?.is_instance_of::<PyGeometrySelection>() {
                OwnedBinding::Support(
                    selection(&pair.get_item(0)?, geometry)?,
                    Some(selection(&pair.get_item(1)?, geometry)?),
                )
            } else if pair.len() == 2
                && pair.get_item(0)?.is_instance_of::<PyTuple>()
                && pair.get_item(1)?.is_instance_of::<PyGeometrySelection>()
            {
                let member_values = pair.get_item(0)?;
                let member_values = member_values.cast::<PyTuple>()?;
                if member_values.is_empty() || member_values.len() > 256 {
                    return Err(PyValueError::new_err(
                        "complete exterior requires between 1 and 256 exact boundary selections",
                    ));
                }
                let parent = selection(&pair.get_item(1)?, geometry)?;
                let authority = geometry.expect("parent selection checks exact authority");
                let members = member_values
                    .iter()
                    .map(|member| {
                        let name = selection(&member, geometry)?;
                        Ok(authority
                            .entity_set(&name)
                            .expect("validated immutable selection"))
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                OwnedBinding::CompleteExterior(
                    members,
                    authority
                        .entity_set(&parent)
                        .expect("validated immutable parent"),
                )
            } else {
                OwnedBinding::Expression(
                    crate::modeling::value_literal::expression(&value)?
                        .source_ast(|_| None, |_| None)
                        .map_err(|error| PyValueError::new_err(error.to_string()))?,
                )
            }
        } else {
            OwnedBinding::Expression(
                crate::modeling::value_literal::expression(&value)?
                    .source_ast(|_| None, |_| None)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            )
        };
        output.push((name, binding));
    }
    output.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(output)
}
