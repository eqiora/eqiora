use super::*;
use eqiora_core::{DynQuantity, Id, entity::kinds};
use eqiora_meshing::QuadratureRule;

impl CommonResult {
    /// Per-triangle residual estimates of the H1 error seminorm, in canonical
    /// mesh-cell order. Their squared sum defines the global squared estimate.
    /// This is an estimator, not a certified bound or the linear solve residual.
    ///
    /// Supports a steady real scalar continuous P1 Field on one planar Region,
    /// constant positive diffusion and homogeneous essential boundaries.
    /// Source-squared integrals use exactly the supplied simplex quadrature;
    /// integration accuracy remains the caller's responsibility. The output has
    /// the Field's physical dimension in two spatial dimensions.
    ///
    /// # Errors
    /// Rejects unsupported equations, boundaries, mesh/space, foreign Fields,
    /// quadrature mismatch and nonfinite estimates.
    pub fn discretization_error_indicators(
        &self,
        field: Id<kinds::Field>,
        quadrature: &QuadratureRule,
    ) -> Result<Vec<DynQuantity>, Diagnostic> {
        let plan = self
            .plan
            .as_spatial()
            .ok_or_else(|| invalid("residual estimation requires a spatial Result"))?;
        let CommonResultPayload::Static(payload) = &self.payload else {
            return Err(invalid("residual estimation requires a static Result"));
        };
        let stored = payload
            .fields
            .iter()
            .find(|candidate| candidate.field_id == field.ulid().to_string())
            .ok_or_else(|| invalid("residual estimation Field is outside this Result"))?;
        let [block] = stored.blocks.as_slice() else {
            return Err(invalid(
                "residual estimation requires one vertex coefficient block",
            ));
        };
        if block.association != CommonFieldAssociation::Vertex
            || stored.value_type.scalar_domain() != eqiora_core::ScalarDomain::Real
            || !stored.value_type.shape().is_scalar()
        {
            return Err(invalid(
                "residual estimation requires real scalar vertex coefficients",
            ));
        }
        plan.residual_error_indicators(field, &block.values, quadrature)
            .map(|values| {
                values
                    .into_iter()
                    .map(|value| DynQuantity::new(value.sqrt(), stored.value_type.dimension()))
                    .collect()
            })
    }
}
