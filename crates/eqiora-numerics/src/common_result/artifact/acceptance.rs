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
