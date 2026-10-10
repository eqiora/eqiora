//! Result-owned derived values and explicitly requested spatial quadrature.

use eqiora_artifact::ModelEnvelope;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, DynQuantity, Id, ValueLiteral};
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
    point: Option<(Id<kinds::Domain>, Vec<DynQuantity>)>,
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
    /// Exact support and dimensioned coordinates for a sampled field-valued output.
    #[must_use]
    pub fn point(&self) -> Option<(Id<kinds::Domain>, &[DynQuantity])> {
        self.point
            .as_ref()
            .map(|(domain, coordinates)| (*domain, coordinates.as_slice()))
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
        let program = self.observation_program()?;
        let (value, _) = composite::evaluate(self, &program, observable, quadratures, None, None)?;
        Ok(CommonObservation {
            observable,
            result_identity: self.identity().to_owned(),
            value,
            point: None,
            quadratures: quadratures.clone(),
        })
    }
    /// Sample a field-valued Observable at dimensioned coordinates on its exact output support.
    /// Quadrature integrates only the declared measure factors; the output coordinates remain fixed.
    /// # Errors
    /// Rejects missing or extra coordinates, wrong units, points outside the support,
    /// stale Models and unavailable density realizations.
    pub fn observe_at(
        &self,
        model: &ModelEnvelope,
        observable: Id<kinds::Observable>,
        coordinates: &[DynQuantity],
        quadratures: &HashMap<Id<kinds::Domain>, QuadratureRule>,
    ) -> Result<CommonObservation, Diagnostic> {
        if model != self.plan().model_artifact() {
            return Err(invalid(
                "Observable Model differs from the exact accepted Result Model",
            ));
        }
        let program = self.observation_program()?;
        let support = program
            .observable_output_support(observable)?
            .ok_or_else(|| invalid("lumped Observable does not have output coordinates"))?;
        let domain = support.domain().downcast().expect("admitted Domain");
        let (value, _) = composite::evaluate(
            self,
            &program,
            observable,
            quadratures,
            None,
            Some(coordinates),
        )?;
        Ok(CommonObservation {
            observable,
            result_identity: self.identity().to_owned(),
            value,
            point: Some((domain, coordinates.to_vec())),
            quadratures: quadratures.clone(),
        })
    }
}

impl CommonResult {
    /// Admitted Model program for this Result's observation context, including Geometry.
    /// Numerical adapters use the same support authority as value and tangent evaluation.
    /// # Errors
    /// Rejects a Model that cannot be admitted in this Result's observation context.
    pub fn observation_program(&self) -> Result<eqiora_sem::KernelProgram, Diagnostic> {
        if let Some(plan) = self.plan().as_linear() {
            Ok(plan.observation_program().clone())
        } else if let Some(plan) = self.plan().as_algebraic() {
            Ok(plan.kernel().clone())
        } else {
            self.plan().model_artifact().to_program().map_err(|errors| {
                errors
                    .into_iter()
                    .next()
                    .expect("failed replay has diagnostic")
            })
        }
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
                StaticObservation::Linear {
                    elasticity: Some(value),
                    ..
                } => Some((
                    value.constrained_reaction,
                    value.integrated_body_force,
                    [
                        payload.assembly.packet_count(),
                        payload.assembly.target_count(),
                    ],
                    value.exact_bounds,
                )),
                StaticObservation::Linear { .. } | StaticObservation::SteadyStokes(_) => None,
            },
            _ => None,
        }
    }
    #[must_use]
    pub fn steady_stokes_observation(&self) -> Option<([f64; 4], [[f64; 2]; 6])> {
        match &self.payload {
            CommonResultPayload::Static(payload) => match &payload.observation {
                StaticObservation::SteadyStokes(value) => Some((value.scalars, value.vectors)),
                StaticObservation::Linear { .. } => None,
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
