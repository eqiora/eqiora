//! Exact temporal policy payload and conversion for common Plan persistence.
use super::*;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireOdeTolerance {
    pub(super) field_ulid: String,
    pub(super) derivative_order: u32,
    pub(super) component: u64,
    pub(super) imaginary: bool,
    pub(super) value: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum WireOdeMethod {
    Tsitouras45,
    ImplicitMidpoint,
}
impl WireOdeMethod {
    fn encode(method: eqiora_time::TimeMethod) -> Self {
        match method {
            eqiora_time::TimeMethod::Tsitouras45 => Self::Tsitouras45,
            eqiora_time::TimeMethod::ImplicitMidpoint => Self::ImplicitMidpoint,
            _ => unreachable!("validated common ODE policy"),
        }
    }
    pub(super) fn decode(self) -> eqiora_time::TimeMethod {
        match self {
            Self::Tsitouras45 => eqiora_time::TimeMethod::Tsitouras45,
            Self::ImplicitMidpoint => eqiora_time::TimeMethod::ImplicitMidpoint,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum WireTimeCoordinates {
    RealF64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(super) enum WireTemporal {
    BackwardEuler {
        step_s: f64,
    },
    Ode {
        method: WireOdeMethod,
        coordinates: WireTimeCoordinates,
        initial_step_s: f64,
        relative_tolerance: f64,
        absolute_tolerances: Vec<WireOdeTolerance>,
        #[serde(skip_serializing_if = "Option::is_none")]
        events: Option<WireEventPolicy>,
        #[serde(skip_serializing_if = "Option::is_none")]
        forward_sensitivities: Option<WireForwardSensitivity>,
    },
}

pub(super) fn temporal_request(plan: &ResolvedCommonPlan) -> Option<WireTemporal> {
    match plan {
        ResolvedCommonPlan::Ode(plan) => Some(WireTemporal::Ode {
            method: WireOdeMethod::encode(plan.temporal().method()),
            coordinates: WireTimeCoordinates::RealF64,
            events: plan.event_policy().map(WireEventPolicy::from_native),
            forward_sensitivities: plan
                .temporal()
                .forward_sensitivities()
                .map(WireForwardSensitivity::from_native),
            initial_step_s: plan.temporal().initial_step_s(),
            relative_tolerance: plan.temporal().relative_tolerance(),
            absolute_tolerances: plan
                .temporal()
                .absolute_tolerances()
                .iter()
                .map(|entry| WireOdeTolerance {
                    field_ulid: entry.coordinate().field().ulid().to_string(),
                    derivative_order: entry.coordinate().derivative_order(),
                    component: entry.coordinate().component() as u64,
                    imaginary: entry.coordinate().is_imaginary(),
                    value: entry.value(),
                })
                .collect(),
        }),
        ResolvedCommonPlan::Scalar(plan) => {
            plan.admission
                .temporal
                .map(|temporal| WireTemporal::BackwardEuler {
                    step_s: temporal.step().value(),
                })
        }
        ResolvedCommonPlan::TransientFlow(plan) => Some(WireTemporal::BackwardEuler {
            step_s: plan.temporal().step().value(),
        }),
        ResolvedCommonPlan::Fsi(plan) => Some(WireTemporal::BackwardEuler {
            step_s: plan.temporal().step().value(),
        }),
        ResolvedCommonPlan::Eigen(_)
        | ResolvedCommonPlan::Algebraic(_)
        | ResolvedCommonPlan::Elasticity(_)
        | ResolvedCommonPlan::SteadyStokes(_) => None,
    }
}
