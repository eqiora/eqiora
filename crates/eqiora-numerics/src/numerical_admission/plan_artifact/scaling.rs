//! Canonical manual flow scaling retained in common Plan artifacts.
use super::*;
use crate::{ScalingComponent2d, ScalingMode2d};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireScalingRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    length_m: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    velocity_m_per_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pressure_pa: Option<f64>,
}

impl WireScalingRequest {
    pub(super) fn to_native(&self) -> Result<Option<IncompressibleScalingRequest2d>, Diagnostic> {
        if self.length_m.is_none() && self.velocity_m_per_s.is_none() && self.pressure_pa.is_none()
        {
            return Ok(None);
        }
        IncompressibleScalingRequest2d::from_si(
            self.length_m,
            self.velocity_m_per_s,
            self.pressure_pa,
        )
        .map(Some)
    }
}

pub(super) fn scaling_request(plan: &ResolvedCommonPlan) -> Option<WireScalingRequest> {
    let receipt = match plan {
        ResolvedCommonPlan::SteadyStokes(plan) => plan.scaling_receipt(),
        ResolvedCommonPlan::TransientFlow(plan) => plan.scaling_receipt(),
        ResolvedCommonPlan::Algebraic(_)
        | ResolvedCommonPlan::Eigen(_)
        | ResolvedCommonPlan::Ode(_)
        | ResolvedCommonPlan::Linear(_) => return None,
    };
    let manual = |component| {
        let record = receipt.component(component);
        (record.mode() == ScalingMode2d::Manual).then(|| record.value().value())
    };
    Some(WireScalingRequest {
        length_m: manual(ScalingComponent2d::Length),
        velocity_m_per_s: manual(ScalingComponent2d::Velocity),
        pressure_pa: manual(ScalingComponent2d::Pressure),
    })
}
