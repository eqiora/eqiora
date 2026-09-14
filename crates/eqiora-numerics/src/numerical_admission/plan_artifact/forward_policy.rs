//! Exact dimensioned derivative controls, reconstructed by ordinary Plan admission.
use super::*;
use crate::common_ode::{CommonForwardSensitivity, CommonSensitivityTolerance};
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireForwardSensitivity {
    relative_tolerance: f64,
    absolute_tolerances: Vec<WireSensitivityTolerance>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireSensitivityTolerance {
    field_ulid: String,
    parameter_ulid: String,
    value: f64,
    dimension: [(i32, i32); 7],
}
impl WireForwardSensitivity {
    pub(super) fn from_native(policy: &CommonForwardSensitivity) -> Self {
        Self {
            relative_tolerance: policy.relative_tolerance(),
            absolute_tolerances: policy
                .absolute_tolerances()
                .iter()
                .map(|entry| WireSensitivityTolerance {
                    field_ulid: entry.field().ulid().to_string(),
                    parameter_ulid: entry.parameter().ulid().to_string(),
                    value: entry.quantity().value(),
                    dimension: entry.quantity().dim().exponents(),
                })
                .collect(),
        }
    }
    pub(super) fn to_native(&self) -> Result<CommonForwardSensitivity, Diagnostic> {
        CommonForwardSensitivity::new(
            self.relative_tolerance,
            self.absolute_tolerances
                .iter()
                .map(|entry| {
                    let dimension =
                        DimExponents::from_rationals(entry.dimension).ok_or_else(|| {
                            invalid("invalid sensitivity tolerance dimension exponents")
                        })?;
                    CommonSensitivityTolerance::new(
                        parse_id::<kinds::Field>(&entry.field_ulid, "Field")?,
                        parse_id::<kinds::Parameter>(&entry.parameter_ulid, "Parameter")?,
                        DynQuantity::new(entry.value, dimension),
                    )
                })
                .collect::<Result<Vec<_>, Diagnostic>>()?,
        )
    }
}
