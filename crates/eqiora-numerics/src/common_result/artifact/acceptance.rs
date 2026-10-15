//! Family-specific accepted numerical evidence.
use super::*;

impl WireStaticEvidence {
    pub(super) fn from_acceptance(value: &StaticAcceptedEvidence) -> Result<Self, Diagnostic> {
        Ok(match value {
            StaticAcceptedEvidence::Linear { nullspace } => Self::Linear {
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
    ) -> Result<StaticAcceptedEvidence, Diagnostic> {
        let acceptance = match self {
            Self::Linear { nullspace } => StaticAcceptedEvidence::Linear {
                nullspace: nullspace.map(crate::nullspace::NullspaceEvidence::from_array),
            },
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
                StaticAcceptedEvidence::Linear { .. }
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
