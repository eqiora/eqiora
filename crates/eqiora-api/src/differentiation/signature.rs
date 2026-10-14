//! Exact semantic values and their private real-coordinate storage layout.

use super::*;
use eqiora_core::ValueType;
use eqiora_ir::ScalarSymbolCoordinate;
use eqiora_meshing::MeshEntity;
use eqiora_realization::Space;
use eqiora_schema::kernel::{SymbolRef, typing::ExpressionType};
use std::ops::Range;

/// One selected semantic value, including its exact support and coefficient order.
///
/// Component shape, basis, frame and units stay in the Model's `ValueType`.
/// A spatial Field additionally retains its Plan's coefficient entities and
/// Space. Mapping axes never become intrinsic value or coefficient axes.
#[derive(Debug, Clone, PartialEq)]
pub struct DifferentiableValueSignature {
    entity: RawId,
    mathematical_type: ExpressionType<RawId>,
    shape: Vec<usize>,
    coordinates: Range<usize>,
    space: Option<Space>,
    entities: Vec<MeshEntity>,
}

impl DifferentiableValueSignature {
    /// Exact Parameter, Field or Observable identity in the owning Program's Model.
    #[must_use]
    pub const fn entity(&self) -> RawId {
        self.entity
    }

    /// Original mathematical type and declared support, without batch axes.
    #[must_use]
    pub const fn mathematical_type(&self) -> &ExpressionType<RawId> {
        &self.mathematical_type
    }

    /// Complete native value shape, including any Field coefficient axis.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Coefficient Space for a spatial output; finite values have no Space.
    #[must_use]
    pub const fn space(&self) -> Option<Space> {
        self.space
    }

    /// Exact Mesh associations in coefficient order; empty for finite values.
    #[must_use]
    pub fn coefficient_entities(&self) -> &[MeshEntity] {
        &self.entities
    }

    pub(crate) fn coordinates(&self) -> Range<usize> {
        self.coordinates.clone()
    }

    pub(crate) fn parameter(&self) -> Id<kinds::Parameter> {
        self.entity
            .downcast()
            .expect("input signature contains Parameters")
    }
}

pub(super) fn build(
    plan: &ResolvedCommonPlan,
    inputs: &[Id<kinds::Parameter>],
    outputs: &[RawId],
) -> Result<
    (
        Vec<DifferentiableValueSignature>,
        Vec<DifferentiableValueSignature>,
    ),
    Diagnostic,
> {
    let kernel = match plan {
        ResolvedCommonPlan::Spatial(plan) => plan.kernel(),
        ResolvedCommonPlan::Algebraic(plan) => plan.kernel(),
        _ => {
            return Err(invalid(
                "this Plan does not admit implicit output differentiation",
            ));
        }
    };
    let mut input_offset = 0;
    let inputs = inputs
        .iter()
        .map(|&id| {
            value(
                plan,
                kernel,
                SymbolRef::Parameter(id),
                id.erase(),
                &mut input_offset,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut output_offset = 0;
    let outputs = outputs
        .iter()
        .map(|&output| {
            let symbol = match output.kind() {
                EntityKind::Field => SymbolRef::Field(output.downcast().expect("Field kind")),
                EntityKind::Observable => {
                    SymbolRef::Observable(output.downcast().expect("Observable kind"))
                }
                _ => {
                    return Err(invalid(
                        "differentiable output requires a Field or Observable",
                    ));
                }
            };
            value(plan, kernel, symbol, output, &mut output_offset)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((inputs, outputs))
}

fn value(
    plan: &ResolvedCommonPlan,
    kernel: &eqiora_sem::KernelProgram,
    symbol: SymbolRef,
    entity: RawId,
    offset: &mut usize,
) -> Result<DifferentiableValueSignature, Diagnostic> {
    let mathematical_type = kernel
        .symbol_type(symbol)
        .ok_or_else(|| invalid("differentiable value is outside its exact Model"))?;
    let value_type: &ValueType = &mathematical_type.value_type;
    let components = ScalarSymbolCoordinate::for_value(symbol, value_type)?.len();
    let (space, entities) = match (plan, symbol) {
        (ResolvedCommonPlan::Spatial(plan), SymbolRef::Field(field)) => (
            Some(plan.field_space(field)?),
            plan.field_coefficient_entities(field)?,
        ),
        _ => (None, Vec::new()),
    };
    let mut shape = Vec::new();
    let count = if space.is_some() {
        if entities.is_empty() {
            return Err(invalid("spatial output has no coefficient entities"));
        }
        shape.push(entities.len());
        components.checked_mul(entities.len())
    } else {
        Some(components)
    }
    .ok_or_else(|| invalid("value coordinate count overflows usize"))?;
    shape.extend(
        value_type
            .shape()
            .extents()
            .iter()
            .map(|extent| extent.get() as usize),
    );
    let end = offset
        .checked_add(count)
        .ok_or_else(|| invalid("signature coordinate count overflows usize"))?;
    let coordinates = *offset..end;
    *offset = end;
    Ok(DifferentiableValueSignature {
        entity,
        mathematical_type,
        shape,
        coordinates,
        space,
        entities,
    })
}
