//! One wire owner for observable reductions and their declared integration measures.

use super::primitive::WireId;
use crate::invalid_artifact;
use eqiora_core::Diagnostic;
use eqiora_schema::kernel::{ExprDag, ObservableMeasure, ObservableReduction, ProbabilityMoment};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum WireMeasure {
    Volume,
    Boundary,
    SphericalVolume,
}

impl WireMeasure {
    pub(crate) const fn encode(measure: ObservableMeasure) -> Self {
        match measure {
            ObservableMeasure::Volume => Self::Volume,
            ObservableMeasure::Boundary => Self::Boundary,
            ObservableMeasure::SphericalVolume => Self::SphericalVolume,
        }
    }
    pub(crate) const fn decode(self) -> ObservableMeasure {
        match self {
            Self::Volume => ObservableMeasure::Volume,
            Self::Boundary => ObservableMeasure::Boundary,
            Self::SphericalVolume => ObservableMeasure::SphericalVolume,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum WireMoment {
    Expectation,
    Variance,
    Covariance,
}

impl WireMoment {
    const fn encode(moment: ProbabilityMoment) -> Self {
        match moment {
            ProbabilityMoment::Expectation => Self::Expectation,
            ProbabilityMoment::Variance => Self::Variance,
            ProbabilityMoment::Covariance => Self::Covariance,
        }
    }
    const fn decode(self) -> ProbabilityMoment {
        match self {
            Self::Expectation => ProbabilityMoment::Expectation,
            Self::Variance => ProbabilityMoment::Variance,
            Self::Covariance => ProbabilityMoment::Covariance,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(crate) enum WireObservableReduction {
    Value,
    SpatialIntegral {
        input: WireId,
        domain: WireId,
        measure: WireMeasure,
        limits: Option<[u32; 2]>,
    },
    RandomVariable {
        distribution: WireId,
    },
    ProbabilityMoment {
        distribution: WireId,
        moment: WireMoment,
    },
}

impl WireObservableReduction {
    pub(crate) fn encode(value: ObservableReduction) -> Self {
        match value {
            ObservableReduction::Value => Self::Value,
            ObservableReduction::SpatialIntegral {
                input,
                domain,
                measure,
                limits,
            } => Self::SpatialIntegral {
                input: WireId::from_raw(input.erase()),
                domain: WireId::from_raw(domain.erase()),
                measure: WireMeasure::encode(measure),
                limits: limits.map(|limits| limits.map(|id| id.index())),
            },
            ObservableReduction::RandomVariable { distribution } => Self::RandomVariable {
                distribution: WireId::from_raw(distribution.erase()),
            },
            ObservableReduction::ProbabilityMoment {
                distribution,
                moment,
            } => Self::ProbabilityMoment {
                distribution: WireId::from_raw(distribution.erase()),
                moment: WireMoment::encode(moment),
            },
        }
    }

    pub(crate) fn decode(&self, expression: &ExprDag) -> Result<ObservableReduction, Diagnostic> {
        Ok(match self {
            Self::Value => ObservableReduction::Value,
            Self::SpatialIntegral {
                input,
                domain,
                measure,
                limits,
            } => ObservableReduction::SpatialIntegral {
                input: input.typed()?,
                domain: domain.typed()?,
                measure: measure.decode(),
                limits: limits
                    .map(|[lower, upper]| {
                        Ok([
                            expression.node_id(lower).ok_or_else(|| {
                                invalid_artifact(
                                    "lower integral limit is outside its expression DAG",
                                )
                            })?,
                            expression.node_id(upper).ok_or_else(|| {
                                invalid_artifact(
                                    "upper integral limit is outside its expression DAG",
                                )
                            })?,
                        ])
                    })
                    .transpose()?,
            },
            Self::RandomVariable { distribution } => ObservableReduction::RandomVariable {
                distribution: distribution.typed()?,
            },
            Self::ProbabilityMoment {
                distribution,
                moment,
            } => ObservableReduction::ProbabilityMoment {
                distribution: distribution.typed()?,
                moment: moment.decode(),
            },
        })
    }

    pub(crate) fn semantic_references(&self) -> Vec<&WireId> {
        match self {
            Self::Value => Vec::new(),
            Self::SpatialIntegral { input, domain, .. } => vec![input, domain],
            Self::RandomVariable { distribution }
            | Self::ProbabilityMoment { distribution, .. } => vec![distribution],
        }
    }
}
