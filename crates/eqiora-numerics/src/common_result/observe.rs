//! Result-owned derived values and explicitly requested spatial quadrature.

use eqiora_artifact::ModelEnvelope;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, Id, ValueLiteral};
use eqiora_meshing::QuadratureRule;
use std::collections::HashMap;

use super::{CommonResult, CommonResultPayload, StaticObservation, invalid};

mod composite;
mod spatial;
mod tangent;
pub use tangent::CommonObservableStateTangent;

/// Accepted derived value retaining the exact result and numerical integration rule.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonObservation {
    observable: Id<kinds::Observable>,
    result_identity: String,
    value: ValueLiteral,
    quadratures: HashMap<Id<kinds::Domain>, QuadratureRule>,
}

impl CommonObservation {
    /// Exact Model Observable.
    #[must_use]
    pub const fn observable(&self) -> Id<kinds::Observable> {
        self.observable
    }
    /// Complete accepted value and physical type.
    #[must_use]
    pub const fn value(&self) -> &ValueLiteral {
        &self.value
    }
    /// Exact accepted Result, including its Plan, State and field lineage.
    #[must_use]
    pub fn result_identity(&self) -> &str {
        &self.result_identity
    }
    /// Effective quadrature by exact integration Domain; empty for finite values.
    #[must_use]
    pub const fn quadratures(&self) -> &HashMap<Id<kinds::Domain>, QuadratureRule> {
        &self.quadratures
    }
}

impl CommonResult {
    /// Evaluate a typed Model Observable against this exact accepted Result.
    ///
    /// Spatial integrals require explicit quadrature on the measure's reference
    /// cell. The profile admits real scalar Q1 fields and the two-component
    /// displacement of a Cartesian elasticity Result, including traces and normal
    /// gradients. No output cadence controls this operation.
    /// # Errors
    /// Rejects foreign/stale Model meaning, unavailable fields, wrong measures,
    /// unsupported dependence and mismatched quadrature.
    pub fn observe(
        &self,
        model: &ModelEnvelope,
        observable: Id<kinds::Observable>,
        quadratures: &HashMap<Id<kinds::Domain>, QuadratureRule>,
    ) -> Result<CommonObservation, Diagnostic> {
        if model != self.plan().model_artifact() {
            return Err(invalid(
                "Observable Model differs from the exact accepted Result Model",
            ));
        }
        let program = observation_program(self, model)?;
        let (value, _) = composite::evaluate(self, &program, observable, quadratures, None)?;
        Ok(CommonObservation {
            observable,
            result_identity: self.identity().to_owned(),
            value,
            quadratures: quadratures.clone(),
        })
    }
}

fn observation_program(
    result: &CommonResult,
    model: &ModelEnvelope,
) -> Result<eqiora_sem::KernelProgram, Diagnostic> {
    if let Some(plan) = result.plan().as_scalar() {
        Ok(plan.observation_program().clone())
    } else if let Some(plan) = result.plan().as_elasticity() {
        Ok(plan.observation_program().clone())
    } else if let Some(plan) = result.plan().as_algebraic() {
        Ok(plan.kernel().clone())
    } else {
        model.to_program().map_err(|errors| {
            errors
                .into_iter()
                .next()
                .expect("failed replay has diagnostic")
        })
    }
}

impl CommonResult {
    #[must_use]
    #[allow(clippy::type_complexity)]
    pub fn elasticity_observation(
        &self,
    ) -> Option<([f64; 2], [f64; 2], [usize; 2], [[f64; 2]; 2])> {
        match &self.payload {
            CommonResultPayload::Static(payload) => match &payload.observation {
                StaticObservation::Elasticity(value) => Some((
                    value.constrained_reaction,
                    value.integrated_body_force,
                    [
                        payload.assembly.packet_count(),
                        payload.assembly.target_count(),
                    ],
                    value.exact_bounds,
                )),
                StaticObservation::Scalar(_) | StaticObservation::SteadyStokes(_) => None,
            },
            _ => None,
        }
    }
    #[must_use]
    pub fn steady_stokes_observation(&self) -> Option<([f64; 4], [[f64; 2]; 6])> {
        match &self.payload {
            CommonResultPayload::Static(payload) => match &payload.observation {
                StaticObservation::SteadyStokes(value) => Some((value.scalars, value.vectors)),
                StaticObservation::Scalar(_) | StaticObservation::Elasticity(_) => None,
            },
            _ => None,
        }
    }

    #[must_use]
    pub fn steady_stokes_boundary_reaction(&self, name: &str) -> Option<[f64; 2]> {
        let CommonResultPayload::Static(payload) = &self.payload else {
            return None;
        };
        let StaticObservation::SteadyStokes(value) = &payload.observation else {
            return None;
        };
        value
            .reactions
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| *value)
    }

    #[must_use]
    pub fn steady_stokes_boundary_flux(&self, name: &str) -> Option<f64> {
        let CommonResultPayload::Static(payload) = &self.payload else {
            return None;
        };
        let StaticObservation::SteadyStokes(value) = &payload.observation else {
            return None;
        };
        value
            .fluxes
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| *value)
    }
}
