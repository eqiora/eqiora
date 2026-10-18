//! Derived values and partials share exact occurrences and explicit spatial quadrature.

use eqiora_artifact::ModelEnvelope;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, DynQuantity, Id, ValueLiteral};
use eqiora_meshing::QuadratureRule;
use std::collections::HashMap;

use super::{CommonResult, CommonResultPayload, invalid};

mod composite;
mod csv;
mod occurrence;
mod partials;
pub(crate) use partials::Covector;
mod projection;
use occurrence::Occurrence;
mod spatial;
mod stokes;
mod tangent;
pub use tangent::CommonObservableStateTangent;

pub(crate) struct SpatialPointOutputs {
    pub(crate) values: Vec<ValueLiteral>,
    pub(crate) partials: Option<Vec<Vec<Covector>>>,
}

/// Evaluate borrowed coefficients at an already validated Parameter point.
/// The caller owns original-equation acceptance; this creates no execution provenance.
pub(crate) fn spatial_point_outputs<'a>(
    plan: &'a crate::CommonSpatialPlan,
    parameters: &'a crate::parameter_point::ParameterPoint,
    fields: impl IntoIterator<Item = (Id<kinds::Field>, &'a [f64])>,
    observables: &[Id<kinds::Observable>],
    quadratures: &HashMap<Id<kinds::Domain>, QuadratureRule>,
    selected: Option<&[eqiora_ir::ScalarSymbolCoordinate]>,
) -> Result<SpatialPointOutputs, Diagnostic> {
    let occurrence = Occurrence::from_point(plan, Some(parameters), fields)?;
    let (values, partials) = composite::evaluate_with_partials(
        &occurrence,
        plan.kernel(),
        observables,
        quadratures,
        None,
        None,
        selected,
    )?;
    Ok(SpatialPointOutputs {
        values: values.into_iter().map(|(value, _)| value).collect(),
        partials,
    })
}

/// Accepted derived value retaining the exact result and numerical integration rule.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonObservation {
    evaluation_kind: &'static str,
    observable: Id<kinds::Observable>,
    result_identity: String,
    value: ValueLiteral,
    output: Option<(usize, f64, String)>,
    point: Option<(Id<kinds::Domain>, Vec<DynQuantity>)>,
    quadratures: HashMap<Id<kinds::Domain>, QuadratureRule>,
}

impl CommonObservation {
    fn from_value(
        result: &CommonResult,
        observable: Id<kinds::Observable>,
        value: ValueLiteral,
        quadratures: &HashMap<Id<kinds::Domain>, QuadratureRule>,
        evaluation_kind: &'static str,
    ) -> Self {
        Self {
            evaluation_kind,
            observable,
            result_identity: result.identity().to_owned(),
            value,
            output: None,
            point: None,
            quadratures: quadratures.clone(),
        }
    }

    /// Whether this is a value, State variation or Parameter derivative.
    #[must_use]
    pub const fn evaluation_kind(&self) -> &'static str {
        self.evaluation_kind
    }

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
    /// Retained output position, SI time, and exact State identity; absent for instantaneous Results.
    #[must_use]
    pub fn output(&self) -> Option<(usize, f64, &str)> {
        self.output
            .as_ref()
            .map(|(index, time, identity)| (*index, *time, identity.as_str()))
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
    pub(crate) fn observe_spatial_state(
        plan: &crate::ResolvedCommonPlan,
        state: &crate::CommonState,
        observable: Id<kinds::Observable>,
        quadratures: &HashMap<Id<kinds::Domain>, QuadratureRule>,
    ) -> Result<ValueLiteral, Diagnostic> {
        let occurrence = Occurrence::from_state(plan, state)?;
        let program = plan
            .as_spatial()
            .expect("admitted spatial occurrence")
            .kernel();
        composite::evaluate(&occurrence, program, observable, quadratures, None, None)
            .map(|(value, _)| value)
    }
}

impl CommonResult {
    /// Evaluate a typed Model Observable against this exact accepted Result.
    ///
    /// Spatial integrals require explicit quadrature on the measure's reference
    /// cell and reuse the admitted real/complex nodal or cell-constant Field reconstruction, including
    /// physical traces and gradients. For an ODE or fixed-geometry spatial Trajectory,
    /// `output_index` selects an exact retained output position; for an instantaneous
    /// Result it must be `None`. Retained outputs do not imply a complete time history.
    /// # Errors
    /// Rejects foreign/stale Model meaning, unavailable fields, wrong measures,
    /// unsupported dependence and mismatched quadrature.
    pub fn observe(
        &self,
        model: &ModelEnvelope,
        observable: Id<kinds::Observable>,
        quadratures: &HashMap<Id<kinds::Domain>, QuadratureRule>,
        output_index: Option<usize>,
    ) -> Result<CommonObservation, Diagnostic> {
        if model != self.plan().model_artifact() {
            return Err(invalid(
                "Observable Model differs from the exact accepted Result Model",
            ));
        }
        if let Some(trajectory) = self
            .trajectory()
            .filter(|trajectory| trajectory.time_request().is_some())
        {
            let index = output_index.ok_or_else(|| {
                invalid("a Trajectory observation requires an explicit output index")
            })?;
            if !quadratures.is_empty() {
                return Err(invalid(
                    "ODE output Observable does not consume spatial quadrature",
                ));
            }
            let (value, state) = trajectory.observe_time_output(model, observable, index)?;
            return Ok(CommonObservation::retained(
                self, observable, value, index, state, "value",
            ));
        }
        let program = self.plan().observation_program()?;
        let occurrence = Occurrence::new(self, output_index)?;
        let (value, _) =
            composite::evaluate(&occurrence, &program, observable, quadratures, None, None)?;

        let mut observation =
            CommonObservation::from_value(self, observable, value, quadratures, "value");
        observation.output = occurrence.output();
        Ok(observation)
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
        output_index: Option<usize>,
    ) -> Result<CommonObservation, Diagnostic> {
        if model != self.plan().model_artifact() {
            return Err(invalid(
                "Observable Model differs from the exact accepted Result Model",
            ));
        }
        let program = self.plan().observation_program()?;
        let support = program
            .observable_output_support(observable)?
            .ok_or_else(|| invalid("lumped Observable does not have output coordinates"))?;
        let domain = support.domain().downcast().expect("admitted Domain");
        let occurrence = Occurrence::new(self, output_index)?;
        let (value, _) = composite::evaluate(
            &occurrence,
            &program,
            observable,
            quadratures,
            None,
            Some(coordinates),
        )?;
        let mut observation =
            CommonObservation::from_value(self, observable, value, quadratures, "value");
        observation.output = occurrence.output();
        observation.point = Some((domain, coordinates.to_vec()));
        Ok(observation)
    }
}

impl CommonResult {
    /// Derive elastic reactions and loads from this exact accepted occurrence.
    /// # Errors
    /// Rejects inconsistent fields, stale resources or non-finite weak-dual recovery.
    #[allow(clippy::type_complexity)]
    pub fn elasticity_observation(
        &self,
    ) -> Result<Option<([f64; 2], [f64; 2], [usize; 2], [[f64; 2]; 2])>, Diagnostic> {
        let (crate::ResolvedCommonPlan::Spatial(plan), CommonResultPayload::Static(payload)) =
            (&self.plan, &self.payload)
        else {
            return Ok(None);
        };
        let values = payload
            .fields
            .iter()
            .flat_map(|field| {
                field
                    .blocks
                    .iter()
                    .flat_map(|block| block.values.iter().copied())
            })
            .collect::<Vec<_>>();
        Ok(plan.elasticity_observation(&values)?.map(|value| {
            (
                value.constrained_reaction(),
                value.integrated_body_force(),
                [
                    payload.assembly.packet_count(),
                    payload.assembly.target_count(),
                ],
                value.exact_bounds(),
            )
        }))
    }
}

impl CommonObservation {
    fn retained(
        result: &CommonResult,
        observable: Id<kinds::Observable>,
        value: ValueLiteral,
        index: usize,
        state: &crate::CommonTimeState,
        evaluation_kind: &'static str,
    ) -> Self {
        let mut observation =
            Self::from_value(result, observable, value, &HashMap::new(), evaluation_kind);
        observation.output = Some((index, state.time_s(), state.identity().to_owned()));
        observation
    }
}

impl CommonResult {
    /// Apply a Parameter direction to a real scalar Observable at a retained ODE output.
    ///
    /// Uses the native output products retained by the sensitivity solve; the
    /// query creates no State or sensitivity interpolant. The observation time
    /// is fixed. Nonzero directions at an event at that exact time require an
    /// observation-side proof not supplied by first-order sensitivity products.
    /// # Errors
    /// Rejects missing products, foreign Model/directions, invalid indices,
    /// wrong direction units and expressions outside the scalar derivative profile.
    pub fn observe_parameter_jvp(
        &self,
        model: &ModelEnvelope,
        observable: Id<kinds::Observable>,
        output_index: usize,
        direction: impl IntoIterator<Item = (Id<kinds::Parameter>, DynQuantity)>,
    ) -> Result<CommonObservation, Diagnostic> {
        if model != self.plan().model_artifact() {
            return Err(invalid(
                "Observable Model differs from the exact accepted Result Model",
            ));
        }
        let trajectory = self
            .trajectory()
            .ok_or_else(|| invalid("Parameter observation requires an accepted Trajectory"))?;
        let sensitivity = self
            .parameter_sensitivity()
            .ok_or_else(|| invalid("Result has no accepted forward Parameter products"))?;
        let (value, state) = trajectory.observe_ode_output_parameter_jvp(
            model,
            observable,
            output_index,
            sensitivity,
            direction,
        )?;
        Ok(CommonObservation::retained(
            self,
            observable,
            value,
            output_index,
            state,
            "parameter-jvp",
        ))
    }
}
