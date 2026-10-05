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
    derivative_order: u32,
    component: u64,
    imaginary: bool,
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
                    field_ulid: entry.coordinate().field().ulid().to_string(),
                    derivative_order: entry.coordinate().derivative_order(),
                    component: entry.coordinate().component() as u64,
                    imaginary: entry.coordinate().is_imaginary(),
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
                        eqiora_core::TimeStateCoordinate::new(
                            parse_id::<kinds::Field>(&entry.field_ulid, "Field")?,
                            entry.derivative_order,
                            usize::try_from(entry.component)
                                .map_err(|_| invalid("time component exceeds address space"))?,
                            entry.imaginary,
                        ),
                        parse_id::<kinds::Parameter>(&entry.parameter_ulid, "Parameter")?,
                        DynQuantity::new(entry.value, dimension),
                    )
                })
                .collect::<Result<Vec<_>, Diagnostic>>()?,
        )
    }
}
