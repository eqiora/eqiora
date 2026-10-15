//! Result-owned derived values and explicitly requested spatial quadrature.

use eqiora_artifact::ModelEnvelope;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, DynQuantity, Id, ValueLiteral};
use eqiora_meshing::QuadratureRule;
use std::collections::HashMap;

use super::{CommonResult, CommonResultPayload, invalid};

mod composite;
mod occurrence;
use occurrence::Occurrence;
mod spatial;
mod stokes;
mod tangent;
pub use tangent::CommonObservableStateTangent;

/// Accepted derived value retaining the exact result and numerical integration rule.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonObservation {
    observable: Id<kinds::Observable>,
    result_identity: String,
    value: ValueLiteral,
    output: Option<(usize, f64, String)>,
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
    /// Select explicit quadrature for every spatial measure consumed by an Observable.
    ///
    /// Coordinate factors use tensor-product Gauss rules. Physical measures use
    /// the exact admitted Mesh family, including simplex Duffy rules. The caller
    /// chooses the points per axis; no scientific accuracy is inferred from this
    /// count. Explicit callers may instead supply their own admitted rules.
    /// # Errors
    /// Rejects missing Observables, finite outputs without an integration measure,
    /// unavailable supports and invalid quadrature orders.
    pub fn observation_quadratures(
        &self,
        observable: Id<kinds::Observable>,
        points_per_axis: usize,
    ) -> Result<HashMap<Id<kinds::Domain>, QuadratureRule>, Diagnostic> {
        use eqiora_schema::kernel::typing::SpatialSupport;
        use eqiora_schema::kernel::{
            ExprNode, KernelNode, ObservableMeasure, ObservableReduction, SymbolRef,
        };
        let program = self.observation_program()?;
        let mut pending = vec![observable];
        let mut visited = std::collections::HashSet::new();
        let mut measures = HashMap::new();
        while let Some(id) = pending.pop() {
            if !visited.insert(id) {
                continue;
            }
            let Some(KernelNode::Observable(definition)) = program.node(id.erase()) else {
                return Err(invalid("Observable is outside this exact Result Model"));
            };
            if let ObservableReduction::SpatialIntegral {
                domain, measure, ..
            } = definition.reduction()
            {
                measures.insert(domain, measure);
            }
            pending.extend(
                definition
                    .expression()
                    .nodes()
                    .iter()
                    .filter_map(|node| match node {
                        ExprNode::Symbol(SymbolRef::Observable(dependency)) => Some(*dependency),
                        _ => None,
                    }),
            );
        }
        if measures.is_empty() {
            return Err(invalid(
                "finite Observable does not accept spatial quadrature",
            ));
        }
        let only_boundaries = measures
            .values()
            .all(|measure| *measure == ObservableMeasure::Boundary);
        measures
            .keys()
            .map(|domain| {
                let support = program
                    .spatial_support(*domain)
                    .ok_or_else(|| invalid("Observable measure Domain is unavailable"))?;
                let dimension = support.intrinsic_dimensions();
                let rule = if dimension == 0 {
                    if points_per_axis != 1 && only_boundaries {
                        return Err(invalid("point measure requires quadrature_points=1"));
                    }
                    QuadratureRule::point()
                } else if !matches!(support, SpatialSupport::Coordinates { .. })
                    && self
                        .plan()
                        .authenticated_mesh()
                        .is_some_and(|mesh| mesh.simplicial_mesh().is_some())
                {
                    eqiora_meshing::simplex_duffy_gauss_legendre(dimension, points_per_axis)?
                } else {
                    QuadratureRule::tensor_product_gauss_legendre(dimension, points_per_axis)?
                };
                Ok((*domain, rule))
            })
            .collect()
    }
    /// Evaluate a typed Model Observable against this exact accepted Result.
    ///
    /// Spatial integrals require explicit quadrature on the measure's reference
    /// cell and reuse the admitted real nodal Field reconstruction, including
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
        let (value, output) = if let Some(trajectory @ crate::CommonTrajectory::Ode { .. }) =
            self.trajectory()
        {
            let index = output_index.ok_or_else(|| {
                invalid("a Trajectory observation requires an explicit output index")
            })?;
            if !quadratures.is_empty() {
                return Err(invalid(
                    "ODE output Observable does not consume spatial quadrature",
                ));
            }
            let (value, state) = trajectory.observe_ode_output(model, observable, index)?;
            (
                value,
                Some((index, state.time_s(), state.identity().to_owned())),
            )
        } else {
            let program = self.observation_program()?;
            let occurrence = Occurrence::new(self, output_index)?;
            let (value, _) =
                composite::evaluate(&occurrence, &program, observable, quadratures, None, None)?;
            (value, occurrence.output())
        };
        Ok(CommonObservation {
            observable,
            result_identity: self.identity().to_owned(),
            output,
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
        output_index: Option<usize>,
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
        let occurrence = Occurrence::new(self, output_index)?;
        let (value, _) = composite::evaluate(
            &occurrence,
            &program,
            observable,
            quadratures,
            None,
            Some(coordinates),
        )?;
        Ok(CommonObservation {
            observable,
            result_identity: self.identity().to_owned(),
            output: occurrence.output(),
            value,
            point: Some((domain, coordinates.to_vec())),
            quadratures: quadratures.clone(),
        })
    }
}

impl CommonResult {
    /// Admitted Model program for this Result's observation context, including Geometry.
    /// Values, tangent evaluation and quadrature selection share this support authority.
    fn observation_program(&self) -> Result<eqiora_sem::KernelProgram, Diagnostic> {
        if let Some(plan) = self.plan().as_spatial() {
            Ok(plan.kernel().clone())
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
