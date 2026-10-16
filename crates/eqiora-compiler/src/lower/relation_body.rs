//! Exclusive lowered mathematical Relation bodies.

use super::{LoweringEquation, LoweringExpression};

#[derive(Debug, Clone)]
pub(crate) enum LoweringRelationBody {
    Equations(Vec<LoweringEquation>),
    Conservation {
        storage: Option<LoweringExpression>,
        flux: LoweringExpression,
        source: LoweringExpression,
    },
    ProbabilityMass {
        outcomes: eqiora_core::ValueType,
        weights: LoweringExpression,
    },
    ProbabilityDensity {
        domain: String,
        measure: Option<eqiora_schema::kernel::ObservableMeasure>,
        density: LoweringExpression,
    },
    Noise {
        process: eqiora_schema::kernel::NoiseProcess,
        parameter: LoweringExpression,
        independent: bool,
    },
    Stochastic {
        target: String,
        calculus: Option<eqiora_schema::kernel::StochasticCalculus>,
        drift: Option<LoweringExpression>,
        diffusions: Vec<(String, LoweringExpression)>,
        jumps: Vec<(String, LoweringExpression)>,
    },
}

impl From<Vec<LoweringEquation>> for LoweringRelationBody {
    fn from(value: Vec<LoweringEquation>) -> Self {
        Self::Equations(value)
    }
}

impl LoweringRelationBody {
    pub(crate) fn expressions(&self) -> impl Iterator<Item = &LoweringExpression> {
        let roots: Vec<_> = match self {
            Self::Equations(values) => values
                .iter()
                .flat_map(|condition| [&condition.left, &condition.right])
                .collect(),
            Self::Conservation {
                storage,
                flux,
                source,
            } => storage.iter().chain([flux, source]).collect(),
            Self::Noise { parameter, .. } => vec![parameter],
            Self::ProbabilityMass { weights, .. } => vec![weights],
            Self::ProbabilityDensity { density, .. } => vec![density],
            Self::Stochastic {
                drift,
                diffusions,
                jumps,
                ..
            } => drift
                .iter()
                .chain(diffusions.iter().chain(jumps).map(|(_, value)| value))
                .collect(),
        };
        roots.into_iter()
    }
}

impl LoweringRelationBody {
    pub(super) fn lower(
        &self,
        file: &str,
        range: super::TextRange,
        activation: &super::ActivationSyntax,
        domain: Option<&str>,
        initial: bool,
        bindings: &std::collections::BTreeMap<String, super::Binding>,
    ) -> Result<
        (
            super::expression::LoweredRelation,
            eqiora_schema::kernel::RelationMeaning,
        ),
        eqiora_core::Diagnostic,
    > {
        use super::expression::lower_relation;
        use super::{codes, expression, source_error};
        use eqiora_schema::kernel::RelationMeaning;
        match self {
            LoweringRelationBody::Equations(equations) => lower_relation(
                file, range, activation, domain, equations, initial, bindings,
            )
            .map(|lowered| {
                (
                    lowered,
                    RelationMeaning::Conditions(
                        equations.iter().map(|equation| equation.kind).collect(),
                    ),
                )
            }),
            LoweringRelationBody::Conservation {
                storage,
                flux,
                source,
            } => {
                let domain = domain.ok_or_else(|| {
                    source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        file,
                        range,
                        "Law requires volume support",
                    )
                });
                domain
                    .and_then(|domain| {
                        expression::lower_law(
                            file,
                            range,
                            domain,
                            storage.as_ref(),
                            flux,
                            source,
                            bindings,
                        )
                    })
                    .map(|(lowered, terms)| (lowered, RelationMeaning::Conservation(terms)))
            }
            Self::ProbabilityMass { .. } | Self::ProbabilityDensity { .. } => {
                if initial
                    || domain.is_some()
                    || !matches!(activation, super::ActivationSyntax::Continuous)
                {
                    return Err(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        file,
                        range,
                        "probability laws have an intrinsic support and no temporal activation",
                    ));
                }
                expression::lower_coefficients(file, range, self, bindings)
            }
            Self::Noise { .. } | Self::Stochastic { .. } => {
                if initial
                    || domain.is_some()
                    || !matches!(activation, super::ActivationSyntax::Continuous)
                {
                    return Err(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        file,
                        range,
                        "stochastic Laws require global physical-time ownership",
                    ));
                }
                expression::lower_coefficients(file, range, self, bindings)
            }
        }
    }
}
