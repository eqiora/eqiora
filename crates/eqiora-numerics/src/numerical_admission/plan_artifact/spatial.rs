//! Canonical spatial policy encoding, shared by all common Plan families.
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum WireSpatialPolicy {
    TetrahedralEdge,
    TetrahedralFace,
    Q1,
    Q2,
    P1,
    CellCenteredTpfa,
    MiniP1,
    CellCentered,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(super) enum WireSpatialRequest {
    Uniform {
        policy: WireSpatialPolicy,
    },
    Scoped {
        bindings: Vec<WireScopedSpatialPolicy>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireScopedSpatialPolicy {
    pub(super) domain_ulid: String,
    pub(super) policy: WireSpatialPolicy,
}

impl From<CommonSpatialPolicy> for WireSpatialPolicy {
    fn from(value: CommonSpatialPolicy) -> Self {
        match value {
            CommonSpatialPolicy::TetrahedralEdge => Self::TetrahedralEdge,
            CommonSpatialPolicy::TetrahedralFace => Self::TetrahedralFace,
            CommonSpatialPolicy::Q1 => Self::Q1,
            CommonSpatialPolicy::Q2 => Self::Q2,
            CommonSpatialPolicy::P1 => Self::P1,
            CommonSpatialPolicy::CellCenteredTpfa => Self::CellCenteredTpfa,
            CommonSpatialPolicy::MiniP1 => Self::MiniP1,
            CommonSpatialPolicy::CellCentered => Self::CellCentered,
        }
    }
}

impl From<WireSpatialPolicy> for CommonSpatialPolicy {
    fn from(value: WireSpatialPolicy) -> Self {
        match value {
            WireSpatialPolicy::TetrahedralEdge => Self::TetrahedralEdge,
            WireSpatialPolicy::TetrahedralFace => Self::TetrahedralFace,
            WireSpatialPolicy::Q1 => Self::Q1,
            WireSpatialPolicy::Q2 => Self::Q2,
            WireSpatialPolicy::P1 => Self::P1,
            WireSpatialPolicy::CellCenteredTpfa => Self::CellCenteredTpfa,
            WireSpatialPolicy::MiniP1 => Self::MiniP1,
            WireSpatialPolicy::CellCentered => Self::CellCentered,
        }
    }
}

pub(super) fn spatial_request(plan: &ResolvedCommonPlan) -> Option<WireSpatialRequest> {
    let uniform = |policy| WireSpatialRequest::Uniform { policy };
    match plan {
        ResolvedCommonPlan::Eigen(_)
        | ResolvedCommonPlan::Algebraic(_)
        | ResolvedCommonPlan::Time(_) => None,
        ResolvedCommonPlan::Spatial(plan) => Some(match plan.spatial_request() {
            CommonSpatialRequest::Uniform(policy) => uniform(policy.into()),
            CommonSpatialRequest::Scoped(bindings) => WireSpatialRequest::Scoped {
                bindings: bindings
                    .into_iter()
                    .map(|binding| WireScopedSpatialPolicy {
                        domain_ulid: binding.domain().ulid().to_string(),
                        policy: binding.policy().into(),
                    })
                    .collect(),
            },
        }),
        ResolvedCommonPlan::SteadyStokes(_) => Some(uniform(WireSpatialPolicy::MiniP1)),
        ResolvedCommonPlan::TransientFlow(plan) => Some(uniform(match plan.admission.spatial {
            NativeSpatialPolicy::TransientMiniP1(_) => WireSpatialPolicy::MiniP1,
            NativeSpatialPolicy::TransientCellCentered(_) => WireSpatialPolicy::CellCentered,
            _ => unreachable!("transient Plan retains a transient spatial policy"),
        })),
    }
}
