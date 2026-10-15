//! Family observations and retained scalar gauge evidence.
use super::*;

impl WireStaticObservation {
    pub(super) fn from_observation(value: &StaticObservation) -> Result<Self, Diagnostic> {
        Ok(match value {
            StaticObservation::Linear {
                nullspace,
                elasticity,
            } => Self::Linear {
                nullspace: nullspace
                    .as_ref()
                    .map(crate::nullspace::NullspaceEvidence::to_array),
                elasticity: elasticity.clone(),
            },
            StaticObservation::SteadyStokes(value) => Self::SteadyStokes {
                acceptance: value.clone(),
            },
        })
    }

    pub(super) fn replay(&self, family: WireResultFamily) -> Result<StaticObservation, Diagnostic> {
        let observation = match self {
            Self::Linear {
                nullspace,
                elasticity,
            } => {
                if let Some(value) = elasticity {
                    let values = value
                        .constrained_reaction
                        .iter()
                        .chain(&value.integrated_body_force)
                        .chain(value.exact_bounds.iter().flatten())
                        .copied()
                        .collect::<Vec<_>>();
                    require_finite(&values, "elasticity Result observation")?;
                }
                StaticObservation::Linear {
                    nullspace: nullspace.map(crate::nullspace::NullspaceEvidence::from_array),
                    elasticity: elasticity.clone(),
                }
            }
            Self::SteadyStokes { acceptance } => {
                let values = acceptance
                    .constrained_reaction
                    .iter()
                    .chain(&acceptance.integrated_body_force)
                    .chain(&acceptance.integrated_boundary_traction)
                    .copied()
                    .collect::<Vec<_>>();
                require_finite(&values, "steady-Stokes accepted forces")?;
                require_finite_nonnegative(
                    &[acceptance.continuity_residual_norm],
                    "steady-Stokes continuity residual",
                )?;
                StaticObservation::SteadyStokes(acceptance.clone())
            }
        };
        let matches = matches!(
            (family, &observation),
            (WireResultFamily::Spatial, StaticObservation::Linear { .. })
                | (
                    WireResultFamily::SteadyStokes,
                    StaticObservation::SteadyStokes(_)
                )
        );
        if !matches {
            return Err(invalid(
                "static Result observation crossed a different family",
            ));
        }
        Ok(observation)
    }
}
