//! Family observations and retained scalar gauge evidence.
use super::*;

impl WireStaticObservation {
    pub(super) fn from_observation(value: &StaticObservation) -> Result<Self, Diagnostic> {
        Ok(match value {
            StaticObservation::Linear(evidence) => Self::Linear {
                nullspace: evidence
                    .as_ref()
                    .map(crate::nullspace::NullspaceEvidence::to_array),
            },
            StaticObservation::Elasticity(value) => Self::Elasticity {
                constrained_reaction: value.constrained_reaction,
                integrated_body_force: value.integrated_body_force,
                exact_bounds: value.exact_bounds,
            },
            StaticObservation::SteadyStokes(value) => Self::SteadyStokes {
                scalars: value.scalars,
                vectors: value.vectors,
                reactions: value.reactions.clone(),
                fluxes: value.fluxes.clone(),
            },
        })
    }

    pub(super) fn replay(&self, family: WireResultFamily) -> Result<StaticObservation, Diagnostic> {
        let observation = match self {
            Self::Linear { nullspace } => StaticObservation::Linear(
                nullspace.map(crate::nullspace::NullspaceEvidence::from_array),
            ),
            Self::Elasticity {
                constrained_reaction,
                integrated_body_force,
                exact_bounds,
            } => {
                let values = constrained_reaction
                    .iter()
                    .chain(integrated_body_force)
                    .chain(exact_bounds.iter().flatten())
                    .copied()
                    .collect::<Vec<_>>();
                require_finite(&values, "elasticity Result observation")?;
                StaticObservation::Elasticity(ElasticityResultObservation {
                    constrained_reaction: *constrained_reaction,
                    integrated_body_force: *integrated_body_force,
                    exact_bounds: *exact_bounds,
                })
            }
            Self::SteadyStokes {
                scalars,
                vectors,
                reactions,
                fluxes,
            } => {
                let values = scalars
                    .iter()
                    .chain(vectors.iter().flatten())
                    .chain(reactions.iter().flat_map(|(_, value)| value))
                    .chain(fluxes.iter().map(|(_, value)| value))
                    .copied()
                    .collect::<Vec<_>>();
                require_finite(&values, "steady-Stokes Result observation")?;
                StaticObservation::SteadyStokes(SteadyStokesResultObservation {
                    scalars: *scalars,
                    vectors: *vectors,
                    reactions: reactions.clone(),
                    fluxes: fluxes.clone(),
                })
            }
        };
        let matches = matches!(
            (family, &observation),
            (WireResultFamily::Linear, StaticObservation::Linear(_))
                | (
                    WireResultFamily::Elasticity,
                    StaticObservation::Elasticity(_)
                )
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
