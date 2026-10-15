//! Spatial endpoint observations reuse the accepted-State mathematical owner.
use super::*;
use crate::{
    CommonAcceptedSpatialStep, CommonResult, CommonState, CommonTransientRunRequest,
    ResolvedCommonPlan,
};
use eqiora_meshing::QuadratureRule;

pub(super) struct SpatialObservable<'a> {
    plan: &'a ResolvedCommonPlan,
    observable: Id<kinds::Observable>,
    quadratures: &'a HashMap<Id<kinds::Domain>, QuadratureRule>,
    pub(super) value_type: ValueType,
}

impl<'a> SpatialObservable<'a> {
    pub(super) fn new(
        request: &'a CommonTransientRunRequest,
        model: &ModelEnvelope,
        observable: Id<kinds::Observable>,
        rule: TimeObservationInterpolation,
        quadratures: &'a HashMap<Id<kinds::Domain>, QuadratureRule>,
    ) -> Result<Self, Diagnostic> {
        if rule != TimeObservationInterpolation::EndpointLinear {
            return Err(invalid(
                "spatial time observation requires endpoint linear interpolation; no native midpoint is retained",
            ));
        }
        let plan = request.plan();
        if model != plan.model_artifact() {
            return Err(invalid(
                "time Observable belongs to a foreign or stale Model",
            ));
        }
        let spatial = plan.as_spatial().ok_or_else(|| {
            invalid("time Observable requires the common spatial Field realization")
        })?;
        spatial.require_fixed_observation_geometry()?;
        let program = spatial.kernel();
        if program.observable_output_support(observable)?.is_some() {
            return Err(invalid(
                "time observation requires a finite Observable; reduce or sample its spatial support explicitly",
            ));
        }
        let Some(KernelNode::Observable(definition)) = program.node(observable.erase()) else {
            return Err(invalid("time Observable is outside the exact Model"));
        };
        Ok(Self {
            plan,
            observable,
            quadratures,
            value_type: definition.value_type().clone(),
        })
    }

    pub(super) fn evaluate(&self, state: &CommonState) -> Result<ValueLiteral, Diagnostic> {
        CommonResult::observe_spatial_state(self.plan, state, self.observable, self.quadratures)
    }
}

pub(super) fn integrate(
    request: &CommonTransientRunRequest,
    states: &[(usize, CommonAcceptedSpatialStep)],
    model: &ModelEnvelope,
    observable: Id<kinds::Observable>,
    rule: TimeObservationInterpolation,
    interval: &interval::TimeObservationInterval,
    quadratures: &HashMap<Id<kinds::Domain>, QuadratureRule>,
) -> Result<ValueLiteral, Diagnostic> {
    let evaluator = SpatialObservable::new(request, model, observable, rule, quadratures)?;
    integrate::integrate(
        states,
        interval,
        rule,
        evaluator.value_type.clone(),
        |(_, step)| [step.previous_state().time_s(), step.state().time_s()],
        |(_, step), sample| {
            evaluator.evaluate(match sample {
                0 => step.previous_state(),
                2 => step.state(),
                _ => unreachable!("endpoint rule has no midpoint"),
            })
        },
    )
}
