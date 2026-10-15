//! Spatial endpoint observations reuse the same accepted-State mathematical owner.
use super::*;
use crate::{CommonAcceptedSpatialStep, CommonResult, CommonTransientRunRequest};
use eqiora_meshing::QuadratureRule;

pub(super) fn integrate(
    request: &CommonTransientRunRequest,
    states: &[(usize, CommonAcceptedSpatialStep)],
    model: &ModelEnvelope,
    observable: Id<kinds::Observable>,
    rule: TimeFunctionalQuadrature,
    interval: &interval::TimeIntegralInterval,
    quadratures: &HashMap<Id<kinds::Domain>, QuadratureRule>,
) -> Result<ValueLiteral, Diagnostic> {
    if rule != TimeFunctionalQuadrature::AcceptedStepTrapezoid {
        return Err(invalid(
            "spatial time integral requires accepted endpoint trapezoid quadrature; no native midpoint is retained",
        ));
    }
    let plan = request.plan();
    if model != plan.model_artifact() {
        return Err(invalid(
            "time Observable belongs to a foreign or stale Model",
        ));
    }
    let spatial = plan
        .as_spatial()
        .ok_or_else(|| invalid("time Observable requires the common spatial Field realization"))?;
    spatial.require_fixed_observation_geometry()?;
    let program = spatial.kernel();
    if program.observable_output_support(observable)?.is_some() {
        return Err(invalid(
            "time integral requires a finite Observable; reduce or sample its spatial support explicitly",
        ));
    }
    let Some(KernelNode::Observable(definition)) = program.node(observable.erase()) else {
        return Err(invalid("time Observable is outside the exact Model"));
    };
    integrate::integrate(
        states,
        interval,
        rule,
        definition.value_type().clone(),
        |(_, step)| [step.previous_state().time_s(), step.state().time_s()],
        |(_, step), sample| {
            let state = match sample {
                0 => step.previous_state(),
                2 => step.state(),
                _ => unreachable!("endpoint rule has no midpoint"),
            };
            CommonResult::observe_spatial_state(plan, state, observable, quadratures)
        },
    )
}
