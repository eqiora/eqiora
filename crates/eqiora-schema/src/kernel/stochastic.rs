//! Process laws and stochastic evolution terms, without a random-number provider.

use super::{ExprDag, RelationMeaning};
use eqiora_core::{Diagnostic, diagnostic::codes};

mod covariance;

/// Mathematical increment law of one nominal finite channel group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseProcess {
    /// Real Wiener increments with an explicit dimensionless covariance per second.
    Wiener,
    /// Conditionally independent counting increments with explicit intensities.
    Poisson,
}

/// Explicit interpretation of a continuous stochastic integral.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StochasticCalculus {
    /// Left-adapted Ito interpretation.
    Ito,
    /// Symmetric Stratonovich interpretation; never converted implicitly.
    Stratonovich,
}

impl NoiseProcess {
    /// Admit an exact finite real dimensionless Wiener covariance, without tolerance.
    /// Returns its channel count; scalar and one-by-one arrays retain distinct shapes.
    /// Exceeding the shared exact-rational proof budget rejects rather than claiming PSD.
    pub fn validate_wiener_covariance(
        value: &eqiora_core::ValueLiteral,
    ) -> Result<usize, Diagnostic> {
        covariance::validate(value)
    }

    /// Realize the exact PSD decomposition as a row-major lower triangular
    /// binary64 factor B for covariance B B^T, subject only to rounding.
    /// Zero pivots retain zero columns; negative pivots are never repaired.
    ///
    /// # Errors
    /// Rejects invalid covariance, exhausted exact proof budget, or a nonzero
    /// factor entry outside binary64 range. Numerical representability does not
    /// restrict semantic admission by `validate_wiener_covariance`.
    pub fn wiener_covariance_factor(
        value: &eqiora_core::ValueLiteral,
    ) -> Result<Vec<f64>, Diagnostic> {
        covariance::factor(value)
    }
}

impl RelationMeaning {
    pub(super) fn validate_process_roots(&self, expression: &ExprDag) -> Result<(), Diagnostic> {
        match self {
            Self::Noise { parameter, .. } => {
                if expression.roots() != [*parameter] || expression.node(*parameter).is_none() {
                    return Err(invalid(
                        "noise Law requires exactly its declared parameter root",
                    ));
                }
            }
            Self::Stochastic {
                calculus,
                drift,
                diffusions,
                jumps,
                ..
            } => {
                if diffusions.is_empty() && jumps.is_empty() {
                    return Err(invalid("stochastic Law requires an explicit noise binding"));
                }
                if diffusions.is_empty() != calculus.is_none()
                    || calculus.is_some() != drift.is_some()
                {
                    return Err(invalid(
                        "continuous stochastic Law requires explicit calculus and drift; a pure jump Law has neither",
                    ));
                }
                let mut seen = std::collections::BTreeSet::new();
                if diffusions
                    .iter()
                    .chain(jumps)
                    .any(|(group, _)| !seen.insert(group.erase()))
                {
                    return Err(invalid(
                        "a stochastic Law must bind each nominal noise group exactly once",
                    ));
                }
                let roots = drift
                    .iter()
                    .copied()
                    .chain(diffusions.iter().chain(jumps).map(|(_, value)| *value));
                if !roots.eq(expression.roots().iter().copied()) {
                    return Err(invalid(
                        "stochastic Law roots differ from its ordered drift and increment coefficients",
                    ));
                }
            }
            _ => unreachable!("process root admission dispatch"),
        }
        Ok(())
    }
}

fn invalid(message: &str) -> Diagnostic {
    Diagnostic::error(codes::INVALID_KERNEL_DEFINITION, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{ExprDagBuilder, RelationDef, SymbolRef};
    use eqiora_core::Id;

    #[test]
    fn stochastic_coefficients_cannot_be_reinterpreted_as_equations() {
        let mut builder = ExprDagBuilder::new();
        let drift = builder.symbol(SymbolRef::Parameter(Id::new())).unwrap();
        let diffusion = builder.symbol(SymbolRef::Parameter(Id::new())).unwrap();
        let channel = Id::new();
        let state = Id::new();
        let expression = builder.finish([drift, diffusion]).unwrap();
        let law = RelationDef::stochastic(
            Id::new(),
            expression.clone(),
            state,
            Some(StochasticCalculus::Ito),
            Some(drift),
            vec![(channel, diffusion)],
            vec![],
        )
        .unwrap();
        assert!(law.equation_sides().is_err());
        for (calculus, drift_root, diffusions, jumps) in [
            (None, Some(drift), vec![(channel, diffusion)], vec![]),
            (
                Some(StochasticCalculus::Ito),
                Some(diffusion),
                vec![(channel, drift)],
                vec![],
            ),
            (
                Some(StochasticCalculus::Ito),
                Some(drift),
                vec![(channel, diffusion)],
                vec![(channel, diffusion)],
            ),
        ] {
            assert!(
                RelationDef::stochastic(
                    Id::new(),
                    expression.clone(),
                    state,
                    calculus,
                    drift_root,
                    diffusions,
                    jumps
                )
                .is_err()
            );
        }
    }

    #[test]
    fn pure_jump_coefficients_have_no_continuous_calculus() {
        let mut builder = ExprDagBuilder::new();
        let increment = builder.symbol(SymbolRef::Parameter(Id::new())).unwrap();
        let law = RelationDef::stochastic(
            Id::new(),
            builder.finish([increment]).unwrap(),
            Id::new(),
            None,
            None,
            vec![],
            vec![(Id::new(), increment)],
        )
        .unwrap();
        assert!(law.equation_sides().is_err());
    }
}
