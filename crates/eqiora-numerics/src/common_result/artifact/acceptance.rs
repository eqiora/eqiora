//! Family-specific accepted numerical evidence.
use super::*;

impl WireStaticEvidence {
    pub(super) fn from_acceptance(value: &StaticAcceptedEvidence) -> Result<Self, Diagnostic> {
        Ok(match value {
            StaticAcceptedEvidence::Spatial {
                nullspace,
                initial_state,
                ..
            } => Self::Spatial {
                initial_state_base64: initial_state
                    .as_ref()
                    .map(|state| state.to_bytes().map(|bytes| BASE64_STANDARD.encode(bytes)))
                    .transpose()?,
                nullspace: nullspace
                    .as_ref()
                    .map(crate::nullspace::NullspaceEvidence::to_array),
            },
            StaticAcceptedEvidence::SteadyStokes(value) => Self::SteadyStokes {
                acceptance: value.clone(),
            },
        })
    }

    pub(super) fn replay(
        &self,
        family: WireResultFamily,
        plan: &ResolvedCommonPlan,
    ) -> Result<StaticAcceptedEvidence, Diagnostic> {
        let acceptance = match self {
            Self::Spatial {
                nullspace,
                initial_state_base64,
            } => {
                let initial_state = initial_state_base64
                    .as_ref()
                    .map(|encoded| {
                        let bytes = BASE64_STANDARD
                            .decode(encoded)
                            .map_err(|e| invalid(format!("invalid initial State encoding: {e}")))?;
                        let state = crate::CommonState::from_bytes(&bytes, plan)?;
                        plan.as_spatial()
                            .ok_or_else(|| {
                                invalid("spatial initial State requires a spatial Plan")
                            })?
                            .static_initial_fields(&state)?;
                        Ok(state)
                    })
                    .transpose()?;
                StaticAcceptedEvidence::Spatial {
                    nullspace: nullspace.map(crate::nullspace::NullspaceEvidence::from_array),
                    initial_state,
                    residual: None,
                }
            }
            Self::SteadyStokes { acceptance } => {
                require_finite_nonnegative(
                    &[acceptance.continuity_residual_norm],
                    "steady-Stokes continuity residual",
                )?;
                let stokes = plan.as_steady_stokes().ok_or_else(|| {
                    invalid("steady-Stokes evidence requires a steady-Stokes Plan")
                })?;
                validate_stokes_gauge(
                    acceptance.gauge_multiplier,
                    stokes.gauge() == crate::CommonPressureGauge2d::ZeroIntegral,
                )?;
                StaticAcceptedEvidence::SteadyStokes(acceptance.clone())
            }
        };
        let matches = matches!(
            (family, &acceptance),
            (
                WireResultFamily::Spatial,
                StaticAcceptedEvidence::Spatial { .. }
            ) | (
                WireResultFamily::SteadyStokes,
                StaticAcceptedEvidence::SteadyStokes(_)
            )
        );
        if !matches {
            return Err(invalid(
                "static Result acceptance crossed a different family",
            ));
        }
        Ok(acceptance)
    }
}

fn validate_stokes_gauge(multiplier: Option<f64>, required: bool) -> Result<(), Diagnostic> {
    if multiplier.is_some() != required || multiplier.is_some_and(|value| !value.is_finite()) {
        return Err(invalid(
            "steady-Stokes gauge multiplier must be finite and present exactly when the retained Plan has a pressure gauge",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stokes_gauge_evidence_roundtrips_and_rejects_tampered_presence() {
        for multiplier in [None, Some(-0.25), Some(0.0)] {
            let evidence = SteadyStokesAcceptedEvidence {
                continuity_residual_norm: 0.0,
                gauge_multiplier: multiplier,
            };
            let bytes = serde_json::to_vec(&evidence).unwrap();
            let replay: SteadyStokesAcceptedEvidence = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(replay, evidence);
            assert!(validate_stokes_gauge(replay.gauge_multiplier, multiplier.is_some()).is_ok());
            let mut wire: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            wire["gauge_multiplier"] = if multiplier.is_some() {
                serde_json::Value::Null
            } else {
                serde_json::json!(0.0)
            };
            let tampered: SteadyStokesAcceptedEvidence = serde_json::from_value(wire).unwrap();
            assert!(
                validate_stokes_gauge(tampered.gauge_multiplier, multiplier.is_some()).is_err()
            );
        }
        for multiplier in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(validate_stokes_gauge(Some(multiplier), true).is_err());
        }
    }
}
