//! Typed values and numerical coordinates retain one original Field identity.
use super::*;
use eqiora_core::{ScalarDomain, ValueLiteral};
use eqiora_ir::{ScalarPart, ScalarSymbolCoordinate};
use eqiora_schema::kernel::{
    KernelNode,
    typing::{ExpressionType, RootContract, TypedResidual},
};

pub(super) fn component(
    value: &ValueLiteral,
    coordinate: &ScalarSymbolCoordinate,
) -> Result<f64, Diagnostic> {
    let flat = value
        .value_type()
        .shape()
        .extents()
        .iter()
        .zip(coordinate.component_index())
        .fold(0usize, |flat, (extent, index)| {
            flat * extent.get() as usize + *index as usize
        });
    let pair = value
        .component(flat)
        .ok_or_else(|| invalid("numeric component is unavailable"))?;
    Ok(if coordinate.part() == ScalarPart::Real {
        pair.0
    } else {
        pair.1
    })
}

pub(super) fn typed_expression(
    kernel: &KernelProgram,
    expression: &ExprDag,
) -> Result<TypedResidual<eqiora_core::RawId>, Diagnostic> {
    TypedResidual::infer(
        expression.clone(),
        None,
        RootContract::ComponentwiseResidual,
        |symbol| {
            let ty = match symbol {
                SymbolRef::Field(id) => match kernel.node(id.erase()) {
                    Some(KernelNode::Field(field)) => Some(field.value_type()),
                    _ => None,
                },
                SymbolRef::Parameter(id) => match kernel.node(id.erase()) {
                    Some(KernelNode::Parameter(parameter)) => Some(parameter.value_type()),
                    _ => None,
                },
                _ => None,
            }
            .ok_or_else(|| invalid("finite coordinate references an unavailable Model symbol"))?;
            Ok::<_, Diagnostic>(ExpressionType::new(ty.clone(), None))
        },
    )
    .map_err(|errors| invalid(format!("finite expression typing failed: {errors:?}")))
}

impl FiniteConstraintProblem {
    pub(crate) fn coordinate_count(&self) -> usize {
        self.coordinates.len()
    }

    pub(crate) fn field_values(
        &self,
        values: &[f64],
    ) -> Result<Vec<(Id<kinds::Field>, ValueLiteral)>, Diagnostic> {
        if values.len() != self.coordinates.len() || values.iter().any(|value| !value.is_finite()) {
            return Err(invalid(
                "finite values require the complete numeric coordinate vector",
            ));
        }
        let mut cursor = 0;
        let mut fields = Vec::new();
        for symbol in &self.symbols {
            let SymbolRef::Field(id) = symbol else {
                return Err(invalid("finite unknown is not a Field"));
            };
            let Some(KernelNode::Field(field)) = self.kernel.node(id.erase()) else {
                return Err(invalid("finite Field is absent from its Model"));
            };
            let ty = field.value_type();
            let mut components = Vec::new();
            for _ in 0..ty.shape().component_count().expect("admitted type") {
                let re = values[cursor];
                cursor += 1;
                let im = if ty.scalar_domain() == ScalarDomain::Complex {
                    let im = values[cursor];
                    cursor += 1;
                    im
                } else {
                    0.
                };
                components.push((re, im));
            }
            fields.push((
                *id,
                ValueLiteral::new(ty.clone(), components)
                    .map_err(|error| invalid(error.to_string()))?,
            ));
        }
        Ok(fields)
    }
}
