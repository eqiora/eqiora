//! Exact typed value payloads and nominal references in the structural projection.

use super::{
    ConstructionBudget, Diagnostic, Encoder, RawId, Reference, fingerprint_error, invalid_artifact,
    lookup, push_reference,
};
use eqiora_core::{DimExponents, DynQuantity, ValueFrame, ValueLiteral, ValueShape};
use std::collections::BTreeMap;

pub(super) fn encode_literal(
    encoder: &mut Encoder,
    value: &ValueLiteral,
) -> Result<(), Diagnostic> {
    encode_value_type(encoder, value.value_type())?;
    if let Some(tag) = value.enum_tag() {
        encoder.u8(4)?;
        return encoder.u32(tag);
    }
    if let Some(value) = value.as_bool() {
        encoder.u8(3)?;
        return encoder.bool(value);
    }
    if value.is_zero() {
        return encoder.u8(0);
    }
    if let Some(components) = value.integer_components() {
        encoder.u8(2)?;
        encoder.u64(value.component_count() as u64)?;
        for integer in components {
            encoder.u64(u64::from_be_bytes(integer.to_be_bytes()))?;
        }
        return Ok(());
    }
    encoder.u8(1)?;
    encoder.u64(value.component_count() as u64)?;
    for (real, imag) in value
        .components()
        .ok_or_else(|| fingerprint_error("floating literal payload is absent"))?
    {
        encoder.u64(real.to_bits())?;
        encoder.u64(imag.to_bits())?;
    }
    Ok(())
}

pub(super) fn encode_optional_literal(
    encoder: &mut Encoder,
    value: Option<&ValueLiteral>,
) -> Result<(), Diagnostic> {
    match value {
        Some(value) => {
            encoder.u8(1)?;
            encode_literal(encoder, value)
        }
        None => encoder.u8(0),
    }
}

pub(super) fn encode_quantity(encoder: &mut Encoder, value: DynQuantity) -> Result<(), Diagnostic> {
    if !value.value().is_finite() {
        return Err(fingerprint_error(
            "structural semantic projection requires finite quantities",
        ));
    }
    let scalar = if value.value() == 0.0 {
        0.0
    } else {
        value.value()
    };
    encoder.u64(scalar.to_bits())?;
    encode_dimension(encoder, value.dim())
}

fn encode_dimension(encoder: &mut Encoder, value: DimExponents) -> Result<(), Diagnostic> {
    for (numerator, denominator) in value.exponents() {
        encoder.i32(numerator)?;
        encoder.i32(denominator)?;
    }
    Ok(())
}

fn encode_shape(encoder: &mut Encoder, shape: &ValueShape) -> Result<(), Diagnostic> {
    encoder.len(shape.extents().len())?;
    for extent in shape.extents() {
        encoder.u32(extent.get())?;
    }
    Ok(())
}

pub(super) fn encode_value_type(
    encoder: &mut Encoder,
    value_type: &eqiora_core::ValueType,
) -> Result<(), Diagnostic> {
    encoder.u8(match value_type.scalar_domain() {
        eqiora_core::ScalarDomain::Enum => 4,
        eqiora_core::ScalarDomain::Real => 0,
        eqiora_core::ScalarDomain::Complex => 1,
        eqiora_core::ScalarDomain::Integer => 2,
        eqiora_core::ScalarDomain::Boolean => 3,
    })?;
    encoder.u32(
        u32::try_from(value_type.array_rank())
            .map_err(|_| invalid_artifact("array rank exceeds u32"))?,
    )?;
    encode_dimension(encoder, value_type.dimension())?;
    encode_shape(encoder, value_type.shape())?;
    encode_frame(encoder, value_type.frame())?;
    if let Some(count) = value_type.enum_member_count() {
        encoder.u8(4)?;
        encoder.u32(count)
    } else if let Some(extent) = value_type.index_extent() {
        encoder.u8(3)?;
        encoder.u32(extent)
    } else if let Some((source, target)) = value_type.map_bases() {
        encoder.u8(5)?;
        encode_basis_shape(encoder, source)?;
        encode_basis_shape(encoder, target)
    } else if let Some(basis) = value_type.coordinate_basis() {
        encoder.u8(1)?;
        encode_basis_shape(encoder, basis)
    } else if value_type.is_count() {
        encoder.u8(2)
    } else {
        encoder.u8(0)
    }
}

fn encode_basis_shape(
    encoder: &mut Encoder,
    basis: eqiora_core::FiniteBasis,
) -> Result<(), Diagnostic> {
    encoder.u8(u8::from(basis.is_dual()))?;
    encoder.u8(if basis.factors().is_some() { 2 } else { 1 })?;
    for atom in basis.atoms() {
        encoder.u32(atom.extent())?;
    }
    Ok(())
}

pub(super) fn type_reference(
    value_type: &eqiora_core::ValueType,
    label: Vec<u8>,
    ids: &BTreeMap<RawId, usize>,
    references: &mut Vec<Reference>,
    budget: &mut ConstructionBudget,
) -> Result<(), Diagnostic> {
    if value_type.finite_bases().next().is_some() {
        for (role, basis) in value_type.finite_bases().enumerate() {
            for (factor, atom) in basis.atoms().enumerate() {
                let mut role_label = label.clone();
                role_label.extend([role as u8, factor as u8]);
                push_reference(
                    references,
                    role_label,
                    lookup(
                        ids,
                        atom.space().expect("atomic factor").erase(),
                        "finite basis factor",
                    )?,
                    budget,
                )?;
            }
        }
        return Ok(());
    }
    if let Some(id) = value_type
        .finite_space()
        .map(|id| id.erase())
        .or_else(|| value_type.index_set().map(|id| id.erase()))
        .or_else(|| value_type.enum_definition().map(|id| id.erase()))
    {
        push_reference(
            references,
            label,
            lookup(ids, id, "nominal value basis")?,
            budget,
        )?;
    }
    Ok(())
}

fn encode_frame(encoder: &mut Encoder, frame: ValueFrame) -> Result<(), Diagnostic> {
    match frame {
        ValueFrame::Invariant => encoder.u8(1),
        ValueFrame::SpatialCartesian => encoder.u8(2),
    }
}
