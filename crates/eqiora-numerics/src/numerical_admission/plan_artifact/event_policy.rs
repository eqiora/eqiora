//! Canonical explicit event controls retained by a temporal Plan.
use super::*;
use crate::common_ode::{CommonEventPolicy, CommonGuardTolerance};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireEventPolicy {
    max_events: usize,
    guard_tolerances: Vec<WireGuardTolerance>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireGuardTolerance {
    activation_ulid: String,
    value: f64,
    dimension: [(i32, i32); 7],
}
impl WireEventPolicy {
    pub(super) fn from_native(value: &CommonEventPolicy) -> Self {
        Self {
            max_events: value.max_events(),
            guard_tolerances: value
                .guard_tolerances()
                .iter()
                .map(|entry| WireGuardTolerance {
                    activation_ulid: entry.activation().ulid().to_string(),
                    value: entry.quantity().value(),
                    dimension: entry.quantity().dim().exponents(),
                })
                .collect(),
        }
    }
    pub(super) fn to_native(&self) -> Result<CommonEventPolicy, Diagnostic> {
        let tolerances = self
            .guard_tolerances
            .iter()
            .map(|entry| {
                let activation =
                    parse_id::<kinds::Activation>(&entry.activation_ulid, "Activation")?;
                let dimension = DimExponents::from_rationals(entry.dimension)
                    .ok_or_else(|| invalid("invalid event guard tolerance dimension exponents"))?;
                CommonGuardTolerance::new(activation, DynQuantity::new(entry.value, dimension))
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        CommonEventPolicy::new(self.max_events, tolerances)
    }
}
