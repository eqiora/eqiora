//! Demand-driven projections of the exact accepted steady MINI fields.

use eqiora_core::Diagnostic;

use crate::common_result::{
    CommonFieldAssociation, CommonResult, CommonResultPayload, CommonStaticResultPayload,
    StaticAcceptedEvidence, invalid,
};
use crate::{CommonSteadyStokesPlan, ResolvedCommonPlan};

impl CommonResult {
    fn steady_stokes_context(
        &self,
    ) -> Option<(&CommonSteadyStokesPlan, &CommonStaticResultPayload)> {
        match (&self.plan, &self.payload) {
            (ResolvedCommonPlan::SteadyStokes(plan), CommonResultPayload::Static(payload)) => {
                Some((plan, payload))
            }
            _ => None,
        }
    }

    /// Evaluate the pressure range, outward net flux and momentum summary on demand.
    /// # Errors
    /// Rejects inconsistent accepted fields, stale resources or non-finite reconstruction.
    #[allow(clippy::type_complexity)]
    pub fn steady_stokes_observation(
        &self,
    ) -> Result<Option<([f64; 4], [[f64; 2]; 6])>, Diagnostic> {
        let Some((plan, payload)) = self.steady_stokes_context() else {
            return Ok(None);
        };
        let StaticAcceptedEvidence::SteadyStokes(acceptance) = &payload.acceptance else {
            return Err(invalid("steady-Stokes Result omitted acceptance evidence"));
        };
        let pressure = block(
            payload,
            plan.pressure_field_id(),
            CommonFieldAssociation::Vertex,
        )?;
        let minimum = pressure
            .iter()
            .copied()
            .min_by(f64::total_cmp)
            .ok_or_else(|| invalid("steady-Stokes observation has no pressure values"))?;
        let maximum = pressure
            .iter()
            .copied()
            .max_by(f64::total_cmp)
            .expect("nonempty pressure field");
        let bounds = plan.observation_bounds()?;
        let vertices = velocity_block(payload, plan, CommonFieldAssociation::Vertex)?;
        let net_flux = plan.net_boundary_flux(&vertices)?;
        let closure = std::array::from_fn(|axis| {
            acceptance.constrained_reaction[axis]
                + acceptance.integrated_body_force[axis]
                + acceptance.integrated_boundary_traction[axis]
        });
        if closure.iter().any(|value| !value.is_finite()) {
            return Err(invalid("steady-Stokes momentum closure is non-finite"));
        }
        Ok(Some((
            [
                minimum,
                maximum,
                net_flux,
                acceptance.continuity_residual_norm,
            ],
            [
                bounds[0],
                bounds[1],
                acceptance.constrained_reaction,
                acceptance.integrated_body_force,
                acceptance.integrated_boundary_traction,
                closure,
            ],
        )))
    }

    /// Reconstruct force on the fluid for exactly one named constrained surface.
    /// # Errors
    /// Rejects stale resources or inconsistent/non-finite field reconstruction.
    pub fn steady_stokes_boundary_reaction(
        &self,
        name: &str,
    ) -> Result<Option<[f64; 2]>, Diagnostic> {
        let Some((plan, payload)) = self.steady_stokes_context() else {
            return Ok(None);
        };
        let vertices = velocity_block(payload, plan, CommonFieldAssociation::Vertex)?;
        let bubbles = velocity_block(payload, plan, CommonFieldAssociation::CellBubble)?;
        let pressure = block(
            payload,
            plan.pressure_field_id(),
            CommonFieldAssociation::Vertex,
        )?;
        plan.boundary_reaction(&vertices, &bubbles, pressure, name)
    }

    /// Integrate parent-outward volume flux on exactly one named surface.
    /// # Errors
    /// Rejects stale resources or inconsistent/non-finite field reconstruction.
    pub fn steady_stokes_boundary_flux(&self, name: &str) -> Result<Option<f64>, Diagnostic> {
        let Some((plan, payload)) = self.steady_stokes_context() else {
            return Ok(None);
        };
        let vertices = velocity_block(payload, plan, CommonFieldAssociation::Vertex)?;
        plan.boundary_flux(&vertices, name)
    }
}

fn block<'a>(
    payload: &'a CommonStaticResultPayload,
    field: &str,
    association: CommonFieldAssociation,
) -> Result<&'a [f64], Diagnostic> {
    payload
        .fields
        .iter()
        .find(|value| value.field_id == field)
        .and_then(|value| {
            value
                .blocks
                .iter()
                .find(|block| block.association == association)
        })
        .map(|block| block.values.as_slice())
        .ok_or_else(|| {
            invalid("steady-Stokes observation omitted an exact field coefficient block")
        })
}

fn velocity_block(
    payload: &CommonStaticResultPayload,
    plan: &CommonSteadyStokesPlan,
    association: CommonFieldAssociation,
) -> Result<Vec<[f64; 2]>, Diagnostic> {
    let values = block(payload, plan.velocity_field_id(), association)?;
    let (values, remainder) = values.as_chunks::<2>();
    if !remainder.is_empty() {
        return Err(invalid(
            "steady-Stokes velocity block has an incomplete component pair",
        ));
    }
    Ok(values.to_vec())
}
