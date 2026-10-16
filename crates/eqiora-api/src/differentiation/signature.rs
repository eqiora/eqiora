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
    model: ModelArtifactReference,
    entity: RawId,
    mathematical_type: ExpressionType<RawId>,
    shape: Vec<usize>,
    coordinate_count: usize,
    space: Option<Space>,
    mesh_digest: Option<String>,
    entities: Vec<MeshEntity>,
}

impl DifferentiableValueSignature {
    /// Exact immutable Model that gives this entity its meaning.
    #[must_use]
    pub const fn model(&self) -> &ModelArtifactReference {
        &self.model
    }

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

    /// Exact Mesh owning the spatial coefficient indices; absent for finite values.
    #[must_use]
    pub fn mesh_digest(&self) -> Option<&str> {
        self.mesh_digest.as_deref()
    }

    /// Exact Mesh associations in coefficient order; empty for finite values.
    #[must_use]
    pub fn coefficient_entities(&self) -> &[MeshEntity] {
        &self.entities
    }

    pub(crate) const fn coordinate_count(&self) -> usize {
        self.coordinate_count
    }

    pub(crate) fn width(&self) -> usize {
        match self.mathematical_type.value_type.scalar_domain() {
            eqiora_core::ScalarDomain::Real => 1,
            eqiora_core::ScalarDomain::Complex => 2,
            _ => unreachable!("admitted differentiable scalar domain"),
        }
    }

    pub(crate) fn parameter(&self) -> Id<kinds::Parameter> {
        self.entity
            .downcast()
            .expect("input signature contains Parameters")
    }
}

pub(super) fn build(
    plan: &ResolvedCommonPlan,
    model: &ModelArtifactReference,
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
    let inputs = inputs
        .iter()
        .map(|&id| {
            value(
                Some(plan),
                kernel,
                SymbolRef::Parameter(id),
                id.erase(),
                model,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
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
            value(Some(plan), kernel, symbol, output, model)
        })
        .collect::<Result<Vec<_>, _>>()?;
    dimension(&inputs)?;
    dimension(&outputs)?;
    Ok((inputs, outputs))
}

pub(crate) fn parameters(
    kernel: &eqiora_sem::KernelProgram,
    model: &ModelArtifactReference,
    inputs: &[Id<kinds::Parameter>],
) -> Result<Vec<DifferentiableValueSignature>, Diagnostic> {
    inputs
        .iter()
        .map(|id| value(None, kernel, SymbolRef::Parameter(*id), id.erase(), model))
        .collect()
}

fn value(
    plan: Option<&ResolvedCommonPlan>,
    kernel: &eqiora_sem::KernelProgram,
    symbol: SymbolRef,
    entity: RawId,
    model: &ModelArtifactReference,
) -> Result<DifferentiableValueSignature, Diagnostic> {
    let mathematical_type = kernel
        .symbol_type(symbol)
        .ok_or_else(|| invalid("differentiable value is outside its exact Model"))?;
    let value_type: &ValueType = &mathematical_type.value_type;
    let components = ScalarSymbolCoordinate::for_value(symbol, value_type)?.len();
    let (space, mesh_digest, entities) = match (plan, symbol) {
        (Some(ResolvedCommonPlan::Spatial(plan)), SymbolRef::Field(field)) => (
            Some(plan.field_space(field)?),
            Some(plan.mesh_digest().to_owned()),
            plan.field_coefficient_entities(field)?,
        ),
        _ => (None, None, Vec::new()),
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
    Ok(DifferentiableValueSignature {
        model: model.clone(),
        entity,
        mathematical_type,
        shape,
        coordinate_count: count,
        space,
        mesh_digest,
        entities,
    })
}

/// Checked total storage extent, never part of a leaf's semantic identity.
pub(crate) fn dimension(values: &[DifferentiableValueSignature]) -> Result<usize, Diagnostic> {
    values.iter().try_fold(0usize, |offset, value| {
        offset
            .checked_add(value.coordinate_count())
            .ok_or_else(|| invalid("signature coordinate count overflows usize"))
    })
}

/// Requires the total extent to have been admitted by Program or Values construction.
pub(crate) fn layout(
    values: &[DifferentiableValueSignature],
) -> impl Iterator<Item = (&DifferentiableValueSignature, Range<usize>)> {
    values.iter().scan(0, |offset, value| {
        let start = *offset;
        *offset += value.coordinate_count();
        Some((value, start..*offset))
    })
}
