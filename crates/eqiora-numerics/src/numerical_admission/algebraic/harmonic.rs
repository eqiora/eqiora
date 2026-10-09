use super::*;
use crate::CommonResult;
use eqiora_core::{DynQuantity, ValueLiteral};

impl CommonAlgebraicPlan {
    /// Reconstruct original real Fields at model time under the retained harmonic
    /// response restriction. This does not impose the original initial conditions.
    pub fn reconstruct_harmonic_fields(
        &self,
        result: &CommonResult,
        time: DynQuantity,
    ) -> Result<Vec<(Id<kinds::Field>, ValueLiteral)>, Diagnostic> {
        let reduction = self
            .harmonic
            .as_ref()
            .ok_or_else(|| invalid("Plan has no harmonic reconstruction"))?;
        if result.plan().as_algebraic() != Some(self) {
            return Err(invalid(
                "harmonic reconstruction requires this exact Plan's accepted Result",
            ));
        }
        let values = self
            .field_values(
                result
                    .finite_values()
                    .ok_or_else(|| invalid("harmonic Result has no finite amplitudes"))?,
            )?
            .into_iter()
            .map(|(id, value)| (id.erase(), value))
            .collect::<BTreeMap<_, _>>();
        let original = reduction.original.to_program().map_err(|errors| {
            errors
                .into_iter()
                .next()
                .expect("original Model diagnostic")
        })?;
        reduction
            .amplitudes
            .iter()
            .map(|(_, original_id, amplitude)| {
                let Some(KernelNode::Field(field)) = original.node(original_id.erase()) else {
                    return Err(invalid("harmonic reconstruction lost its original Field"));
                };
                let value = values
                    .get(&amplitude.erase())
                    .ok_or_else(|| invalid("harmonic Result omitted a mapped amplitude"))?;
                let components = reduction.reconstruct(
                    time,
                    (0..value.component_count())
                        .map(|i| value.component(i).expect("in-range amplitude component")),
                )?;
                Ok((
                    *original_id,
                    ValueLiteral::new(
                        field.value_type().clone(),
                        components.into_iter().map(|real| (real, 0.0)),
                    )
                    .map_err(|_| {
                        invalid("harmonic reconstruction produced an invalid real value")
                    })?,
                ))
            })
            .collect()
    }
}
